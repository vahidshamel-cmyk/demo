# dashboard.py - Product Recommendation Dashboard for Resellers

import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import torch
import faiss
import pickle
import json
import os
from types import SimpleNamespace
import torch.nn as nn
import random
import sys
from io import StringIO
from contextlib import contextmanager
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==============================================================================
#  PAGE CONFIGURATION & HELPER
# ==============================================================================
st.set_page_config(page_title="Product Recommender for Resellers", page_icon="🛍️", layout="wide")
st.set_option("client.showErrorDetails", False)

@contextmanager
def st_suppress_stdout():
    """A context manager to suppress unwanted prints within a block."""
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = original_stdout

# Load environment variables early so PROCESSED_DATA_DIR is available
load_dotenv()
SHOW_DEBUG = os.getenv("DEBUG", "0").lower() in {"1", "true", "yes"}
BASE_DIR = './data_processed'

# ==============================================================================
#  MODEL ARCHITECTURE (Matching the notebook)
# ==============================================================================
class UserTower(nn.Module):
    """Reseller tower: processes reseller features into embedding."""
    def __init__(self, vocab_sizes, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.user_embedding = nn.Embedding(vocab_sizes.get('userid_encoded', 1000), embedding_dim)
        self.category_embedding = nn.Embedding(vocab_sizes.get('categoryid_encoded', 100), embedding_dim // 2)
        self.month_embedding = nn.Embedding(vocab_sizes.get('month_encoded', 12), embedding_dim // 4)
        
        mlp_input_dim = embedding_dim + embedding_dim // 2 + embedding_dim // 4
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, user_id, category_id, month):
        user_emb = self.user_embedding(user_id)
        category_emb = self.category_embedding(category_id)
        month_emb = self.month_embedding(month)
        concat_emb = torch.cat([user_emb, category_emb, month_emb], dim=-1)
        return self.mlp(concat_emb)


class ProductTower(nn.Module):
    """Product tower: processes product features into embedding."""
    def __init__(self, vocab_sizes, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.product_embedding = nn.Embedding(vocab_sizes.get('productid_encoded', 1000), embedding_dim)
        # These might not exist, provide defaults
        self.product_type_embedding = nn.Embedding(vocab_sizes.get('product_type_encoded', 10), embedding_dim // 2)
        self.variant_embedding = nn.Embedding(vocab_sizes.get('variant_sellable_encoded', 3), embedding_dim // 4)
        
        mlp_input_dim = embedding_dim + embedding_dim // 2 + embedding_dim // 4
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, product_id, product_type=None, variant_sellable=None):
        product_emb = self.product_embedding(product_id)
        if product_type is None:
            product_type = torch.zeros_like(product_id)
        if variant_sellable is None:
            variant_sellable = torch.zeros_like(product_id)
        type_emb = self.product_type_embedding(product_type)
        variant_emb = self.variant_embedding(variant_sellable)
        concat_emb = torch.cat([product_emb, type_emb, variant_emb], dim=-1)
        return self.mlp(concat_emb)


class TwoTowerModel(nn.Module):
    """Two-tower recommendation model combining reseller and product towers."""
    def __init__(self, vocab_sizes, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.user_tower = UserTower(vocab_sizes, embedding_dim, hidden_dim)
        self.product_tower = ProductTower(vocab_sizes, embedding_dim, hidden_dim)

    def forward(self, user_id, category_id, month, product_id, product_type, variant_sellable):
        user_emb = self.user_tower(user_id, category_id, month)
        product_emb = self.product_tower(product_id, product_type, variant_sellable)
        return torch.sum(user_emb * product_emb, dim=-1)

# ==============================================================================
#  ASSET LOADING
# ==============================================================================
@st.cache_resource
def load_all_assets():
    default_processed = os.path.join(BASE_DIR, "data", "processed")
    PROCESSED_DATA_DIR = './data_processed'
    assets = {}

    def _require(path, description):
        if not os.path.exists(path):
            st.error(f"Missing {description}: '{path}'. Ensure preprocessing saved assets and PROCESSED_DATA_DIR is set in .env.")
            raise FileNotFoundError(path)
        return path

    # Load config.json (fallback to defaults if missing)
    default_config = {
        "EMBEDDING_DIM": 64,
        "LEARNING_RATE": 0.001,
        "BATCH_SIZE": 1024,
        "EPOCHS": 10
    }
    config_path = os.path.join(PROCESSED_DATA_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        except Exception as e:
            if SHOW_DEBUG:
                st.warning(f"Failed to read {config_path} ({e}). Using built-in defaults.")
            config_dict = default_config
    else:
        if SHOW_DEBUG:
            st.info(f"{config_path} not found. Using built-in default hyperparameters.")
        config_dict = default_config
    
    assets["config"] = SimpleNamespace(**config_dict)
    assets["config"].DEVICE = "cpu"
    assets["config"].EMBEDDING_DIM = config_dict.get("EMBEDDING_DIM", 64)

    # Load required files
    encoders_path = _require(os.path.join(PROCESSED_DATA_DIR, "encoders.pkl"), "encoders.pkl")
    user_hist_path = _require(os.path.join(PROCESSED_DATA_DIR, "user_history.pkl"), "user_history.pkl")
    products_path = _require(os.path.join(PROCESSED_DATA_DIR, "products_final.parquet"), "products_final.parquet")
    users_path = _require(os.path.join(PROCESSED_DATA_DIR, "users_final.parquet"), "users_final.parquet")
    model_df_path = _require(os.path.join(PROCESSED_DATA_DIR, "model_df_final.parquet"), "model_df_final.parquet")
    title_emb_path = os.path.join(PROCESSED_DATA_DIR, "title_embeddings.pt")  # Optional - dashboard works without it
    faiss_collab_path = _require(os.path.join(PROCESSED_DATA_DIR, "faiss_index.bin"), "faiss_index.bin")
    faiss_semantic_path = os.path.join(PROCESSED_DATA_DIR, "semantic_faiss_index.bin")  # Optional - dashboard works without it

    with open(encoders_path, "rb") as f:
        assets["encoders"] = pickle.load(f)
    with open(user_hist_path, "rb") as f:
        assets["user_history"] = pickle.load(f)
    assets["products"] = pl.read_parquet(products_path)
    assets["users"] = pl.read_parquet(users_path)
    assets["model_df_pd"] = pd.read_parquet(model_df_path)
    
    # Load title embeddings (optional - for semantic search)
    if os.path.exists(title_emb_path):
        assets["title_embeddings"] = torch.load(title_emb_path, map_location="cpu")
    else:
        if SHOW_DEBUG:
            st.warning(f"title_embeddings.pt not found at {title_emb_path}. Semantic search will be limited.")
        assets["title_embeddings"] = None
    
    assets["collaborative_faiss_index"] = faiss.read_index(faiss_collab_path)
    
    # Load semantic index (optional)
    if os.path.exists(faiss_semantic_path):
        assets["semantic_faiss_index"] = faiss.read_index(faiss_semantic_path)
    else:
        if SHOW_DEBUG:
            st.warning(f"semantic_faiss_index.bin not found. Semantic product search will use category-based fallback.")
        assets["semantic_faiss_index"] = None
    
    # Load cold start products (fallback for new resellers)
    cold_start_path = os.path.join(PROCESSED_DATA_DIR, "cold_start_products.parquet")
    if os.path.exists(cold_start_path):
        assets["cold_start_products"] = pl.read_parquet(cold_start_path)
    else:
        # Fallback: use most popular products
        if SHOW_DEBUG:
            st.warning("cold_start_products.parquet not found. Using most popular products as fallback.")
        popular_df = assets["model_df_pd"].groupby('productid')['productid'].count().sort_values(ascending=False).head(20)
        assets["cold_start_products"] = None
        assets["most_popular_product_ids"] = popular_df.index.tolist()
    
    # Load model
    with st_suppress_stdout():
        # Create vocab_sizes: map encoder keys to model's expected keys
        # Model expects keys like 'userid_encoded', but encoders use 'userid'
        vocab_sizes = {}
        for k, v in assets["encoders"].items():
            vocab_sizes[f"{k}_encoded"] = len(v.classes_)
        # Add defaults for optional vocab sizes
        if 'product_type_encoded' not in vocab_sizes:
            vocab_sizes['product_type_encoded'] = 10
        if 'variant_sellable_encoded' not in vocab_sizes:
            vocab_sizes['variant_sellable_encoded'] = 3
        
        model = TwoTowerModel(vocab_sizes, assets["config"].EMBEDDING_DIM, hidden_dim=128)
        model_path = os.path.join(PROCESSED_DATA_DIR, "final_ranking_model.pth")
        if not os.path.exists(model_path):
            st.error(f"Model weights not found at '{model_path}'.")
            raise FileNotFoundError(model_path)
        try:
            raw_state = torch.load(model_path, map_location="cpu")
            model_state = model.state_dict()
            filtered_state = {}
            dropped = []
            for k, v in raw_state.items():
                if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
                    filtered_state[k] = v
                else:
                    dropped.append(k)
            missing, unexpected = model.load_state_dict(filtered_state, strict=False)
            if (dropped or missing or unexpected) and SHOW_DEBUG:
                st.warning(
                    f"Weights partially loaded. Dropped: {len(dropped)}, Missing: {len(missing)}, Unexpected: {len(unexpected)}."
                )
        except Exception as e:
            st.error(f"Failed to load model weights from '{model_path}': {e}")
            raise
        assets["model"] = model.eval()
    
    # Get core reseller IDs (those in the trained model)
    if 'userid_encoded' in assets["model_df_pd"].columns:
        assets["core_user_ids"] = set(assets["model_df_pd"]['userid'].unique())
    else:
        # Fallback: use userid if encoded version doesn't exist
        assets["core_user_ids"] = set(assets["model_df_pd"]['userid'].unique())
    
    # Calculate total inventory per product (aggregate all variant inventories by productid)
    # IMPORTANT: Load variants table and aggregate inventory by productid (not from model_df_pd which has duplicate rows)
    variants_path = os.path.join(os.path.dirname(PROCESSED_DATA_DIR), "fedshi_variants.csv")
    if os.path.exists(variants_path):
        try:
            # Load variants table and aggregate inventory by productid
            variants_df = pl.read_csv(variants_path, separator=',', has_header=True, encoding='utf-8', ignore_errors=True)
            
            # Check for both lowercase and capitalized column names
            product_col = None
            inventory_col = None
            fid_col = None
            
            # Try different possible column name variations
            for col in variants_df.columns:
                col_lower = col.lower()
                if col_lower == 'productid' or col_lower == 'product_id':
                    product_col = col
                elif col_lower == 'inventory':
                    inventory_col = col
                elif col_lower == 'fid':
                    fid_col = col
            
            if product_col and inventory_col:
                # Aggregate: sum inventory by productid (each variant contributes its inventory)
                # Also get FID for each product (first FID if multiple variants)
                product_quantities_df = (
                    variants_df
                    .group_by(product_col)
                    .agg([
                        pl.col(inventory_col).sum().alias('total_inventory'),
                        pl.col(fid_col).first().alias('fid') if fid_col else pl.lit(None).alias('fid')
                    ])
                )
                # Convert to dictionaries with int keys
                product_quantities = {}
                product_fids = {}
                for row in product_quantities_df.iter_rows(named=True):
                    product_id = int(row[product_col])
                    total_inv = int(row['total_inventory']) if row['total_inventory'] is not None else 0
                    product_quantities[product_id] = total_inv
                    if fid_col and row.get('fid'):
                        product_fids[product_id] = row['fid']
                assets["product_quantities"] = product_quantities
                assets["product_fids"] = product_fids
            else:
                assets["product_quantities"] = {}
                assets["product_fids"] = {}
                if SHOW_DEBUG:
                    st.warning(f"Variants table missing required columns. Product col: {product_col}, Inventory col: {inventory_col}. Found columns: {variants_df.columns}")
        except Exception as e:
            if SHOW_DEBUG:
                st.warning(f"Could not load variants from {variants_path}: {e}")
            assets["product_quantities"] = {}
            assets["product_fids"] = {}
    else:
        # Fallback: try to get unique variant-product combinations from model_df_pd
        if 'productid' in assets["model_df_pd"].columns and 'variantid' in assets["model_df_pd"].columns and 'inventory' in assets["model_df_pd"].columns:
            # Get unique variant-productid combinations with inventory, then aggregate by productid
            unique_variants = assets["model_df_pd"][['productid', 'variantid', 'inventory']].drop_duplicates(subset=['productid', 'variantid'])
            product_quantities = unique_variants.groupby('productid')['inventory'].sum().to_dict()
            assets["product_quantities"] = product_quantities
            assets["product_fids"] = {}  # FID not available from model_df
        else:
            assets["product_quantities"] = {}
            assets["product_fids"] = {}
            if SHOW_DEBUG:
                st.warning("Could not calculate product quantities - variants CSV not found and model_df missing required columns")
    
    return assets

# ==============================================================================
#  RECOMMENDATION FUNCTIONS
# ==============================================================================
def get_recommendations(mode, user_id_str, product_name_str, k=5):
    """Generate product recommendations for resellers."""
    assets = load_all_assets()
    products_df = assets['products']
    core_user_ids = assets['core_user_ids']
    encoders = assets['encoders']
    info_text = ""
    product_ids = []
    
    if not mode:
        return "Please select a mode.", []
    
    if mode == "Recommend for Reseller":
        user_id = int(user_id_str) if user_id_str else None
        if not user_id:
            return "Please select a Reseller ID.", []
        
        # Check if user is in encoder vocabulary (safe to encode)
        # Note: encoder keys are 'userid', 'categoryid', 'month' (not '_encoded')
        if 'userid' not in encoders:
            return f"Error: 'userid' encoder not found. Available encoders: {list(encoders.keys())}", []
        
        encoder_classes = set(encoders['userid'].classes_)
        
        if user_id not in encoder_classes:
            # Cold start user: not in training data - show cold start recommendations
            info_text = f"Showing recommended products for Reseller {user_id} (cold start - no purchase history)."
            if assets.get("cold_start_products") is not None:
                cold_start_df = assets["cold_start_products"].sort('cold_start_score', descending=True)
                product_ids = cold_start_df.head(k)['productid'].to_list()
            else:
                # Fallback to most popular products
                if assets.get('most_popular_product_ids'):
                    product_ids = random.sample(assets['most_popular_product_ids'], min(k, len(assets['most_popular_product_ids'])))
                else:
                    return f"Reseller {user_id} not in model and no cold start products available.", []
            # Format and return recommendations
            rec_products = products_df.filter(pl.col('productid').is_in(product_ids) if 'productid' in products_df.columns else pl.col('id').is_in(product_ids))
            product_quantities = assets.get('product_quantities', {})
            product_fids = assets.get('product_fids', {})
            recommendations = []
            for pid in product_ids:
                product_row = rec_products.filter((pl.col('productid') == pid) if 'productid' in rec_products.columns else (pl.col('id') == pid))
                if not product_row.is_empty():
                    row = product_row.head(1).to_dicts()[0]
                    # Get total quantity - aggregate of all variant inventories for this product
                    # Try both int and original pid as keys
                    quantity = 0
                    if product_quantities:
                        quantity = product_quantities.get(pid) or product_quantities.get(int(pid)) or product_quantities.get(str(pid)) or 0
                    if quantity == 0:
                        # Fallback: try to get from product row if aggregation not available
                        quantity = row.get('quantity') or row.get('inventory') or row.get('current_quantity') or 0
                    
                    # CRITICAL: Filter out products with zero quantity - don't recommend out of stock items
                    if quantity <= 0:
                        continue  # Skip this product
                    
                    # Get FID for this product
                    fid = None
                    if product_fids:
                        fid = product_fids.get(pid) or product_fids.get(int(pid)) or product_fids.get(str(pid))
                    recommendations.append({
                        'name': row.get('name', 'Unknown Product'),
                        'category': row.get('categoryid', 'N/A'),
                        'productid': row.get('productid', row.get('id', pid)),
                        'fid': fid or 'N/A',
                        'quantity': int(quantity) if quantity is not None else 0
                    })
            
            # If we filtered out all recommendations, try to get more with quantity > 0
            if len(recommendations) < k and assets.get("cold_start_products") is not None:
                # Get more products that have quantity > 0
                all_cold_start = assets["cold_start_products"].sort('cold_start_score', descending=True)
                for _, row in all_cold_start.iter_rows(named=True):
                    if len(recommendations) >= k:
                        break
                    pid = row.get('productid')
                    if pid in product_ids:  # Already processed
                        continue
                    quantity = product_quantities.get(pid) or product_quantities.get(int(pid)) or product_quantities.get(str(pid)) or 0
                    if quantity > 0:
                        product_row = rec_products.filter((pl.col('productid') == pid) if 'productid' in rec_products.columns else (pl.col('id') == pid))
                        if not product_row.is_empty():
                            prod_row = product_row.head(1).to_dicts()[0]
                            fid = product_fids.get(pid) or product_fids.get(int(pid)) or product_fids.get(str(pid)) if product_fids else None
                            recommendations.append({
                                'name': prod_row.get('name', 'Unknown Product'),
                                'category': prod_row.get('categoryid', 'N/A'),
                                'productid': prod_row.get('productid', prod_row.get('id', pid)),
                                'fid': fid or 'N/A',
                                'quantity': int(quantity)
                            })
            
            return info_text, recommendations
        
        # User is in encoder - use personalized recommendations
        if user_id in core_user_ids:
            # Core reseller with purchase history: use personalized recommendations
            info_text = f"Showing personalized recommendations for Reseller {user_id}."
        else:
            # User in encoder but might have limited history
            info_text = f"Showing recommendations for Reseller {user_id}."
        
        user_tower = assets['model'].user_tower
        model_df_pd = assets['model_df_pd']
        user_history = assets['user_history']
        
        # User is in encoder - proceed with encoding
        try:
            user_id_encoded = encoders['userid'].transform([user_id])[0]
            # Validate encoded ID is within actual embedding layer size (CRITICAL FIX)
            # Get the actual embedding layer size from the model
            actual_embedding_size = user_tower.user_embedding.num_embeddings
            if user_id_encoded < 0 or user_id_encoded >= actual_embedding_size:
                return f"Error: Encoded user ID {user_id_encoded} out of range [0, {actual_embedding_size-1}]. Model embedding size doesn't match encoder. User {user_id} may not be compatible with this model.", []
        except (ValueError, KeyError) as e:
            return f"Error encoding user ID {user_id}: {str(e)}", []
        
        # Get user features (use encoded columns if available, else original)
        if 'userid_encoded' in model_df_pd.columns:
            user_rows = model_df_pd[model_df_pd['userid_encoded'] == user_id_encoded]
            if len(user_rows) == 0:
                return f"User {user_id} (encoded: {user_id_encoded}) not found in model data", []
            user_features_row = user_rows.iloc[0]
            category_id_encoded = user_features_row.get('categoryid_encoded', 0)
            month_encoded = user_features_row.get('month_encoded', 0)
        else:
            # Fallback: get from original columns and encode
            user_rows = model_df_pd[model_df_pd['userid'] == user_id]
            if len(user_rows) == 0:
                return f"User {user_id} not found in model data", []
            user_features_row = user_rows.iloc[0]
            category_id_original = user_features_row.get('categoryid', 0)
            month_original = user_features_row.get('month', 1)
            
            # Encode category and month with validation
            if 'categoryid' in encoders:
                if category_id_original in encoders['categoryid'].classes_:
                    category_id_encoded = encoders['categoryid'].transform([category_id_original])[0]
                else:
                    category_id_encoded = 0  # Default if not in vocabulary
            else:
                category_id_encoded = 0
            
            if 'month' in encoders:
                if month_original in encoders['month'].classes_:
                    month_encoded = encoders['month'].transform([month_original])[0]
                else:
                    month_encoded = 0  # Default if not in vocabulary
            else:
                month_encoded = 0
        
        # Validate all encoded IDs are within their embedding layer sizes
        category_embedding_size = user_tower.category_embedding.num_embeddings
        month_embedding_size = user_tower.month_embedding.num_embeddings
        
        if category_id_encoded < 0 or category_id_encoded >= category_embedding_size:
            category_id_encoded = 0  # Clamp to valid range
        if month_encoded < 0 or month_encoded >= month_embedding_size:
            month_encoded = 0  # Clamp to valid range
        
        # Generate user embedding
        user_id_tensor = torch.tensor([user_id_encoded], dtype=torch.long)
        category_tensor = torch.tensor([category_id_encoded], dtype=torch.long)
        month_tensor = torch.tensor([month_encoded], dtype=torch.long)
        
        with torch.no_grad():
            user_embedding = user_tower(user_id_tensor, category_tensor, month_tensor).cpu().numpy()
        
        faiss.normalize_L2(user_embedding)
        search_k = k + len(user_history.get(user_id, set()))
        distances, indices = assets['collaborative_faiss_index'].search(user_embedding, search_k)
        
        # Map FAISS indices to encoded product IDs
        # FAISS returns index positions [0, 1, 2, ...] which map to encoded product IDs
        faiss_mapping_path = os.path.join(os.getenv("PROCESSED_DATA_DIR", "data/processed"), "faiss_product_mapping.json")
        if os.path.exists(faiss_mapping_path):
            with open(faiss_mapping_path, 'r') as f:
                faiss_to_productid = json.load(f)
                # Mapping: FAISS index position -> encoded productid (keys might be strings)
                recommended_ids_encoded = [faiss_to_productid.get(str(idx), faiss_to_productid.get(int(idx), idx)) for idx in indices[0]]
        else:
            # Fallback: assume FAISS indices are already encoded product IDs
            recommended_ids_encoded = indices[0].tolist()
        
        # Filter out products already purchased by this reseller
        # user_history contains original product IDs, so we need to encode them for comparison
        purchased_products_original = user_history.get(user_id, set())
        if 'productid' in encoders and purchased_products_original:
            # Encode purchased product IDs to compare with FAISS results
            purchased_products_encoded = set(encoders['productid'].transform(list(purchased_products_original)))
            recommended_ids_encoded = [pid for pid in recommended_ids_encoded if pid not in purchased_products_encoded]
        else:
            # If no encoder or no history, skip filtering
            pass
        
        # Convert encoded IDs back to original product IDs
        # Encoder key is 'productid' (not 'productid_encoded')
        if 'productid' in encoders:
            # FAISS indices are already encoded product IDs, just inverse transform
            product_ids_raw = encoders['productid'].inverse_transform(recommended_ids_encoded)
        else:
            # Fallback: assume indices are already original IDs
            product_ids_raw = recommended_ids_encoded
        
        # CRITICAL: Filter out products with zero quantity BEFORE selecting top k
        # This ensures we only recommend products that are in stock
        product_quantities = assets.get('product_quantities', {})
        product_ids = []
        for pid in product_ids_raw:
            quantity = product_quantities.get(pid) or product_quantities.get(int(pid)) or product_quantities.get(str(pid)) or 0
            if quantity > 0:  # Only include products with available inventory
                product_ids.append(pid)
            if len(product_ids) >= k:  # Stop once we have enough
                break
        
        # If we don't have enough products with quantity > 0, search for more
        if len(product_ids) < k and len(recommended_ids_encoded) > len(product_ids_raw):
            # Try more FAISS results
            for pid_encoded in recommended_ids_encoded[len(product_ids_raw):]:
                if len(product_ids) >= k:
                    break
                try:
                    pid = encoders['productid'].inverse_transform([pid_encoded])[0] if 'productid' in encoders else pid_encoded
                    if pid not in product_ids:
                        quantity = product_quantities.get(pid) or product_quantities.get(int(pid)) or product_quantities.get(str(pid)) or 0
                        if quantity > 0:
                            product_ids.append(pid)
                except:
                    continue
    
    elif mode == "Find Similar Products":
        if not product_name_str:
            return "Please select a product name.", []
        
        query_product = products_df.filter(pl.col('name') == product_name_str)
        if query_product.is_empty():
            return f"Product '{product_name_str}' not found.", []
        
        query_product_id = query_product['productid'][0] if 'productid' in query_product.columns else query_product['id'][0]
        info_text = f"Showing products similar to '{product_name_str}'."
        
        # Get encoded product ID
        # Encoder key is 'productid' (not 'productid_encoded')
        if 'productid' in encoders:
            core_product_ids = set(encoders['productid'].classes_)
            if query_product_id in core_product_ids:
                # Try semantic search if embeddings available
                if assets.get('title_embeddings') is not None and assets.get('semantic_faiss_index') is not None:
                    product_id_encoded = encoders['productid'].transform([query_product_id])[0]
                    query_vector = assets['title_embeddings'][product_id_encoded:product_id_encoded+1].cpu().numpy()
                    distances, indices = assets['semantic_faiss_index'].search(query_vector, k * 2)  # Get more to filter
                    similar_ids_encoded = [idx for idx in indices[0] if idx != product_id_encoded]
                    similar_ids_raw = encoders['productid'].inverse_transform(similar_ids_encoded)
                else:
                    # Fallback to category-based similarity
                    similar_ids_raw = []
                
                # Filter out products with zero quantity
                product_quantities = assets.get('product_quantities', {})
                product_ids = []
                for pid in similar_ids_raw:
                    quantity = product_quantities.get(pid) or product_quantities.get(int(pid)) or product_quantities.get(str(pid)) or 0
                    if quantity > 0:  # Only include products with available inventory
                        product_ids.append(pid)
                    if len(product_ids) >= k:  # Stop once we have enough
                        break
            else:
                # Cold start product: find by category
                query_category = None
                if 'categoryid' in query_product.columns:
                    query_category = query_product['categoryid'][0]
                if query_category is not None:
                    category_products = products_df.filter((pl.col('categoryid') == query_category) & (pl.col('productid') != query_product_id))
                    product_ids = category_products.head(k)['productid'].to_list() if 'productid' in category_products.columns else category_products.head(k)['id'].to_list()
                else:
                    product_ids = []
        else:
            # Fallback: category-based similarity
            query_category = None
            if 'categoryid' in query_product.columns:
                query_category = query_product['categoryid'][0]
            if query_category is not None:
                category_products = products_df.filter((pl.col('categoryid') == query_category) & (pl.col('productid') != query_product_id))
                product_ids = category_products.head(k)['productid'].to_list() if 'productid' in category_products.columns else category_products.head(k)['id'].to_list()
            else:
                product_ids = []
    
    # Format recommendations
    if hasattr(product_ids, '__len__') and len(product_ids) > 0:
        if hasattr(product_ids, 'tolist'):
            product_ids = product_ids.tolist()
        final_product_ids = product_ids[:k]
        rec_products = products_df.filter(pl.col('productid').is_in(final_product_ids) if 'productid' in products_df.columns else pl.col('id').is_in(final_product_ids))
        
        recommendations = []
        product_quantities = assets.get('product_quantities', {})
        product_fids = assets.get('product_fids', {})
        for pid in final_product_ids:
            product_row = rec_products.filter((pl.col('productid') == pid) if 'productid' in rec_products.columns else (pl.col('id') == pid))
            if not product_row.is_empty():
                row = product_row.head(1).to_dicts()[0]
                # Get total quantity - aggregate of all variant inventories for this product
                # Try both int and original pid as keys
                quantity = 0
                if product_quantities:
                    quantity = product_quantities.get(pid) or product_quantities.get(int(pid)) or product_quantities.get(str(pid)) or 0
                if quantity == 0:
                    # Fallback: try to get from product row if aggregation not available
                    quantity = row.get('quantity') or row.get('inventory') or row.get('current_quantity') or 0
                
                # CRITICAL: Filter out products with zero quantity - don't recommend out of stock items
                if quantity <= 0:
                    continue  # Skip this product
                
                # Get FID for this product
                fid = None
                if product_fids:
                    fid = product_fids.get(pid) or product_fids.get(int(pid)) or product_fids.get(str(pid))
                recommendations.append({
                    'name': row.get('name', 'Unknown Product'),
                    'category': row.get('categoryid', 'N/A'),
                    'productid': row.get('productid', row.get('id', pid)),
                    'fid': fid or 'N/A',
                    'quantity': int(quantity) if quantity is not None else 0
                })
        
        return info_text, recommendations
    
    return info_text, []

# ==============================================================================
#  ANALYTICS FUNCTIONS
# ==============================================================================
def get_data_overview(assets):
    """Generate data overview statistics."""
    model_df = assets['model_df_pd']
    products_df = assets['products']
    users_df = assets['users']
    
    stats = {
        'total_interactions': len(model_df),
        'unique_resellers': model_df['userid'].nunique() if 'userid' in model_df.columns else 0,
        'unique_products': model_df['productid'].nunique() if 'productid' in model_df.columns else 0,
        'total_products': len(products_df),
        'total_users': len(users_df),
        'avg_interactions_per_user': len(model_df) / model_df['userid'].nunique() if 'userid' in model_df.columns and model_df['userid'].nunique() > 0 else 0,
        'avg_interactions_per_product': len(model_df) / model_df['productid'].nunique() if 'productid' in model_df.columns and model_df['productid'].nunique() > 0 else 0,
    }
    
    # Date range if available
    if 'createdat' in model_df.columns:
        try:
            model_df['createdat'] = pd.to_datetime(model_df['createdat'])
            stats['date_range'] = (model_df['createdat'].min(), model_df['createdat'].max())
        except:
            stats['date_range'] = None
    else:
        stats['date_range'] = None
    
    return stats

def get_product_analytics(assets):
    """Generate product analytics."""
    model_df = assets['model_df_pd']
    products_df = assets['products']
    product_quantities = assets.get('product_quantities', {})
    
    # Top products by interactions
    if 'productid' in model_df.columns:
        product_counts = model_df['productid'].value_counts().head(20)
        top_products = product_counts.to_dict()
    else:
        top_products = {}
    
    # Products with inventory
    products_with_stock = sum(1 for qty in product_quantities.values() if qty > 0)
    products_out_of_stock = len(product_quantities) - products_with_stock
    
    # Category distribution
    if 'categoryid' in model_df.columns:
        category_counts = model_df['categoryid'].value_counts().head(10)
    else:
        category_counts = pd.Series()
    
    return {
        'top_products': top_products,
        'products_with_stock': products_with_stock,
        'products_out_of_stock': products_out_of_stock,
        'category_distribution': category_counts.to_dict()
    }

def get_user_analytics(assets):
    """Generate user/reseller analytics."""
    model_df = assets['model_df_pd']
    
    if 'userid' in model_df.columns:
        user_counts = model_df['userid'].value_counts()
        top_users = user_counts.head(20).to_dict()
        avg_purchases = user_counts.mean()
        median_purchases = user_counts.median()
    else:
        top_users = {}
        avg_purchases = 0
        median_purchases = 0
    
    return {
        'top_users': top_users,
        'avg_purchases_per_user': avg_purchases,
        'median_purchases_per_user': median_purchases
    }

def get_sales_trends(assets):
    """Generate sales trends over time."""
    model_df = assets['model_df_pd']
    
    if 'createdat' in model_df.columns:
        try:
            model_df['createdat'] = pd.to_datetime(model_df['createdat'])
            model_df['date'] = model_df['createdat'].dt.date
            daily_sales = model_df.groupby('date').size()
            return daily_sales.to_dict()
        except:
            return {}
    return {}

# ==============================================================================
#  STREAMLIT UI
# ==============================================================================
st.title("🛍️ Product Recommendation System for Resellers - Dashboard")
assets = load_all_assets()

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "📈 Data Analytics", 
    "🛒 Product Analytics", 
    "👥 Reseller Analytics",
    "🎯 Recommendations",
    "🤖 Model Performance"
])

# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tab1:
    st.header("📊 System Overview")
    
    stats = get_data_overview(assets)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Interactions", f"{stats['total_interactions']:,}")
    
    with col2:
        st.metric("Unique Resellers", f"{stats['unique_resellers']:,}")
    
    with col3:
        st.metric("Unique Products", f"{stats['unique_products']:,}")
    
    with col4:
        st.metric("Total Products", f"{stats['total_products']:,}")
    
    st.divider()
    
    # Additional Stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Statistics")
        st.write(f"**Interactions per Reseller:** {stats['avg_interactions_per_user']:.1f}")
        st.write(f"**Interactions per Product:** {stats['avg_interactions_per_product']:.1f}")
    
    with col2:
        st.subheader("Date Range")
        if stats['date_range']:
            st.write(f"**From:** {stats['date_range'][0]}")
            st.write(f"**To:** {stats['date_range'][1]}")
        else:
            st.write("Date information not available")
    
    # Quick Charts
    st.divider()
    st.subheader("Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Product inventory status
        product_analytics = get_product_analytics(assets)
        inventory_data = {
            'In Stock': product_analytics['products_with_stock'],
            'Out of Stock': product_analytics['products_out_of_stock']
        }
        if sum(inventory_data.values()) > 0:
            fig = px.pie(
                values=list(inventory_data.values()),
                names=list(inventory_data.keys()),
                title="Product Inventory Status"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top categories
        if product_analytics['category_distribution']:
            cat_data = product_analytics['category_distribution']
            fig = px.bar(
                x=list(cat_data.keys()),
                y=list(cat_data.values()),
                title="Top Categories by Interactions",
                labels={'x': 'Category ID', 'y': 'Interactions'}
            )
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 2: DATA ANALYTICS
# ==============================================================================
with tab2:
    st.header("📈 Data Analytics")
    
    model_df = assets['model_df_pd']
    
    # Sales trends over time
    st.subheader("Sales Trends Over Time")
    sales_trends = get_sales_trends(assets)
    
    if sales_trends:
        dates = sorted(sales_trends.keys())
        values = [sales_trends[d] for d in dates]
        fig = px.line(
            x=dates,
            y=values,
            title="Daily Sales Volume",
            labels={'x': 'Date', 'y': 'Number of Transactions'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Date information not available for trend analysis")
    
    st.divider()
    
    # Data distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        if 'sellingprice' in model_df.columns:
            prices = model_df['sellingprice'].dropna()
            if len(prices) > 0:
                fig = px.histogram(
                    x=prices,
                    nbins=50,
                    title="Distribution of Selling Prices",
                    labels={'x': 'Price', 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Quantity Distribution")
        if 'quantity' in model_df.columns:
            quantities = model_df['quantity'].dropna()
            if len(quantities) > 0:
                fig = px.histogram(
                    x=quantities,
                    nbins=30,
                    title="Distribution of Quantities",
                    labels={'x': 'Quantity', 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.divider()
    st.subheader("Sample Data")
    st.dataframe(model_df.head(100), use_container_width=True)

# ==============================================================================
# TAB 3: PRODUCT ANALYTICS
# ==============================================================================
with tab3:
    st.header("🛒 Product Analytics")
    
    product_analytics = get_product_analytics(assets)
    products_df = assets['products']
    model_df = assets['model_df_pd']
    product_quantities = assets.get('product_quantities', {})
    
    # Top products
    st.subheader("Top 20 Products by Interactions")
    if product_analytics['top_products']:
        top_products_df = pd.DataFrame({
            'Product ID': list(product_analytics['top_products'].keys()),
            'Interactions': list(product_analytics['top_products'].values())
        })
        
        # Merge with product names
        if 'id' in products_df.columns or 'productid' in products_df.columns:
            product_id_col = 'productid' if 'productid' in products_df.columns else 'id'
            name_col = 'name' if 'name' in products_df.columns else 'productid'
            products_lookup = products_df.select([product_id_col, name_col]).to_pandas()
            top_products_df = top_products_df.merge(
                products_lookup,
                left_on='Product ID',
                right_on=product_id_col,
                how='left'
            )
            top_products_df['Product Name'] = top_products_df[name_col].fillna('Unknown')
            top_products_df = top_products_df[['Product Name', 'Product ID', 'Interactions']]
        
        # Add inventory information
        top_products_df['Inventory'] = top_products_df['Product ID'].map(
            lambda x: product_quantities.get(x, product_quantities.get(int(x), product_quantities.get(str(x), 0)))
        )
        
        st.dataframe(top_products_df, use_container_width=True)
        
        # Chart
        fig = px.bar(
            top_products_df.head(10),
            x='Product Name' if 'Product Name' in top_products_df.columns else 'Product ID',
            y='Interactions',
            title="Top 10 Products by Interactions"
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Inventory analysis
    st.subheader("Inventory Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Products In Stock", product_analytics['products_with_stock'])
    
    with col2:
        st.metric("Products Out of Stock", product_analytics['products_out_of_stock'])
    
    # Top products by inventory
    if product_quantities:
        inventory_df = pd.DataFrame({
            'Product ID': list(product_quantities.keys()),
            'Inventory': list(product_quantities.values())
        }).sort_values('Inventory', ascending=False).head(20)
        
        # Merge with product names
        if 'id' in products_df.columns or 'productid' in products_df.columns:
            product_id_col = 'productid' if 'productid' in products_df.columns else 'id'
            name_col = 'name' if 'name' in products_df.columns else 'productid'
            products_lookup = products_df.select([product_id_col, name_col]).to_pandas()
            inventory_df = inventory_df.merge(
                products_lookup,
                left_on='Product ID',
                right_on=product_id_col,
                how='left'
            )
            inventory_df['Product Name'] = inventory_df[name_col].fillna('Unknown')
            inventory_df = inventory_df[['Product Name', 'Product ID', 'Inventory']]
        
        st.subheader("Top 20 Products by Inventory")
        st.dataframe(inventory_df, use_container_width=True)

# ==============================================================================
# TAB 4: RESELLER ANALYTICS
# ==============================================================================
with tab4:
    st.header("👥 Reseller Analytics")
    
    user_analytics = get_user_analytics(assets)
    model_df = assets['model_df_pd']
    users_df = assets['users']
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Resellers", user_analytics.get('total_resellers', len(users_df)))
    
    with col2:
        st.metric("Avg Purchases/Reseller", f"{user_analytics['avg_purchases_per_user']:.1f}")
    
    with col3:
        st.metric("Median Purchases/Reseller", f"{user_analytics['median_purchases_per_user']:.1f}")
    
    st.divider()
    
    # Top resellers
    st.subheader("Top 20 Resellers by Activity")
    if user_analytics['top_users']:
        top_users_df = pd.DataFrame({
            'Reseller ID': list(user_analytics['top_users'].keys()),
            'Interactions': list(user_analytics['top_users'].values())
        })
        
        # Merge with user names if available
        if 'id' in users_df.columns or 'userid' in users_df.columns:
            user_id_col = 'userid' if 'userid' in users_df.columns else 'id'
            name_col = 'name' if 'name' in users_df.columns else 'userid'
            users_lookup = users_df.select([user_id_col, name_col]).to_pandas()
            top_users_df = top_users_df.merge(
                users_lookup,
                left_on='Reseller ID',
                right_on=user_id_col,
                how='left'
            )
            top_users_df['Reseller Name'] = top_users_df[name_col].fillna('Unknown')
            top_users_df = top_users_df[['Reseller Name', 'Reseller ID', 'Interactions']]
        
        st.dataframe(top_users_df, use_container_width=True)
        
        # Chart
        fig = px.bar(
            top_users_df.head(10),
            x='Reseller Name' if 'Reseller Name' in top_users_df.columns else 'Reseller ID',
            y='Interactions',
            title="Top 10 Resellers by Activity"
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Reseller activity distribution
    if 'userid' in model_df.columns:
        user_counts = model_df['userid'].value_counts()
        fig = px.histogram(
            x=user_counts.values,
            nbins=30,
            title="Distribution of Interactions per Reseller",
            labels={'x': 'Number of Interactions', 'y': 'Number of Resellers'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 5: RECOMMENDATIONS (Original functionality)
# ==============================================================================
with tab5:
    st.header("🎯 Product Recommendations")
    
    # Get available reseller IDs and product names
    core_user_ids = assets.get('core_user_ids', set())
    
    # Get all users from the users table (sorted, starting from user 1)
    if 'id' in assets["users"].columns:
        user_ids = sorted(assets["users"].select(pl.col('id').drop_nulls().unique())['id'].to_list())
    elif 'userid' in assets["users"].columns:
        user_ids = sorted(assets["users"].select(pl.col('userid').drop_nulls().unique())['userid'].to_list())
    else:
        # Fallback: try to get from model_df or core_user_ids
        if 'userid' in assets["model_df_pd"].columns:
            user_ids = sorted(assets["model_df_pd"]['userid'].unique().tolist())
        elif core_user_ids:
            user_ids = sorted(list(core_user_ids))
        else:
            user_ids = []
    
    product_names = sorted(assets["products"].select(pl.col('name').drop_nulls().unique())['name'].to_list())
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'info_text' not in st.session_state:
        st.session_state.info_text = None
    if 'mode' not in st.session_state:
        st.session_state.mode = "Recommend for Reseller"
    if 'product_name_input' not in st.session_state:
        st.session_state.product_name_input = None
    
    with st.sidebar:
        st.header("Recommendation Controls")
        mode = st.session_state.get("mode", "Recommend for Reseller")
        product_name_prefill = st.session_state.get("product_name_input", None)
        
        mode = st.radio("Select Mode", ["Recommend for Reseller", "Find Similar Products"], 
                        index=0 if mode == "Recommend for Reseller" else 1)
        
        if mode == "Recommend for Reseller":
            user_id_input = st.selectbox("Select Reseller ID", options=user_ids if user_ids else [])
            product_name_input = None
            st.session_state.product_name_input = None
        else:
            if product_name_prefill and product_name_prefill in product_names:
                pre_idx = product_names.index(product_name_prefill)
            else:
                pre_idx = 0
            product_name_input = st.selectbox("Select Product Name", options=product_names, index=pre_idx)
            user_id_input = None
            st.session_state.product_name_input = product_name_input
        
        st.session_state.mode = mode
        
        k = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                st.session_state.info_text, st.session_state.recommendations = get_recommendations(
                    mode, str(user_id_input) if user_id_input else None, product_name_input, k=k
                )
    
    if st.session_state.info_text:
        st.info(st.session_state.info_text)
    
    if st.session_state.recommendations:
        st.subheader(f"Top {len(st.session_state.recommendations)} Recommendations")
        display_cols = st.columns(min(5, len(st.session_state.recommendations)))
        for i, rec in enumerate(st.session_state.recommendations):
            with display_cols[i % len(display_cols)]:
                st.markdown(f"**{rec['name']}**")
                st.markdown(f"Category: {rec.get('category', 'N/A')}")
                st.markdown(f"Product ID: {rec['productid']}")
                st.markdown(f"FID: {rec.get('fid', 'N/A')}")
                st.markdown(f"Quantity: {rec.get('quantity', 0)}")
                # Action to find similar products
                if st.button("Find Similar", key=f"find_sim_{i}"):
                    st.session_state.mode = "Find Similar Products"
                    st.session_state.product_name_input = rec['name']
                    st.rerun()
    elif st.session_state.info_text is not None:
        st.warning("No recommendations could be generated for your selection.")
    else:
        st.write("## Welcome!")
        st.write("Select a mode and an option in the sidebar to begin.")

# ==============================================================================
# TAB 6: MODEL PERFORMANCE
# ==============================================================================
with tab6:
    st.header("🤖 Model Performance")
    
    config = assets['config']
    
    # Model configuration
    st.subheader("Model Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Embedding Dimension", config.EMBEDDING_DIM)
        st.metric("Batch Size", config.BATCH_SIZE)
    
    with col2:
        st.metric("Learning Rate", config.LEARNING_RATE)
        st.metric("Epochs", config.EPOCHS)
    
    with col3:
        st.metric("Loss Type", getattr(config, 'LOSS_TYPE', 'BCE'))
        st.metric("Test Size", f"{getattr(config, 'TEST_SIZE', 0.2)*100:.0f}%")
    
    st.divider()
    
    # Model architecture info
    st.subheader("Model Architecture")
    st.write("**Two-Tower Architecture:**")
    st.write("- **User Tower:** Processes reseller features (ID, category, month) into embeddings")
    st.write("- **Product Tower:** Processes product features (ID, type, variant) into embeddings")
    st.write("- **Scoring:** Dot product of user and product embeddings")
    
    st.divider()
    
    # Vocabulary sizes
    if 'encoders' in assets:
        st.subheader("Model Vocabulary Sizes")
        vocab_sizes = {k: len(v.classes_) for k, v in assets['encoders'].items()}
        vocab_df = pd.DataFrame({
            'Feature': list(vocab_sizes.keys()),
            'Vocabulary Size': list(vocab_sizes.values())
        })
        st.dataframe(vocab_df, use_container_width=True)
    
    st.divider()
    
    # FAISS indexes info
    st.subheader("Search Indexes")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'collaborative_faiss_index' in assets:
            st.write(f"**Collaborative Filtering Index:** {assets['collaborative_faiss_index'].ntotal} vectors")
    
    with col2:
        if 'semantic_faiss_index' in assets:
            st.write(f"**Semantic Search Index:** {assets['semantic_faiss_index'].ntotal} vectors")
    
    st.divider()
    
    # Cold start products
    if 'cold_start_products' in assets and assets['cold_start_products'] is not None:
        st.subheader("Cold Start Products")
        cold_start_df = assets['cold_start_products'].sort('cold_start_score', descending=True).head(20)
        st.dataframe(cold_start_df.to_pandas(), use_container_width=True)
