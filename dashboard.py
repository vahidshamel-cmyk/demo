# dashboard.py (Definitive, Final, and Corrected)

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

# ==============================================================================
#  PAGE CONFIGURATION & HELPER
# ==============================================================================
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")
# Prevent stray object echoes from rendering
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
#  MODEL AND ASSET LOADING
# ==============================================================================
class UserTower(nn.Module):
    def __init__(self, vocab_sizes, config):
        super().__init__();self.config = config; self.user_embedding = nn.Embedding(vocab_sizes['user_id'], config.EMBEDDING_DIM); self.age_embedding = nn.Embedding(vocab_sizes['age_bin'], config.EMBEDDING_DIM // 4); self.country_embedding = nn.Embedding(vocab_sizes['country'], config.EMBEDDING_DIM // 2)
        user_mlp_input_dim = config.EMBEDDING_DIM + (config.EMBEDDING_DIM // 4) + (config.EMBEDDING_DIM // 2)
        user_layers = []
        for i in range(config.N_LAYERS_USER):
            in_dim = user_mlp_input_dim if i == 0 else 256
            user_layers.extend([nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(config.DROPOUT_RATE)])
        user_layers.append(nn.Linear(256, config.EMBEDDING_DIM))
        self.mlp = nn.Sequential(*user_layers)
    def forward(self, features): return self.mlp(torch.cat([self.user_embedding(features['user_id']), self.age_embedding(features['age_bin']), self.country_embedding(features['country'])], dim=1))
class ItemTower(nn.Module):
    def __init__(self, vocab_sizes, title_embedding_dim, config):
        super().__init__();self.config = config; self.work_embedding = nn.Embedding(vocab_sizes['work_id'], config.EMBEDDING_DIM); self.author_embedding = nn.Embedding(vocab_sizes['canonical_author'], config.EMBEDDING_DIM // 2); self.publisher_embedding = nn.Embedding(vocab_sizes['publisher'], config.EMBEDDING_DIM // 2); self.yop_bin_embedding = nn.Embedding(vocab_sizes['yop_bin'], config.EMBEDDING_DIM // 4)
        item_mlp_input_dim = config.EMBEDDING_DIM + (config.EMBEDDING_DIM // 2) * 2 + (config.EMBEDDING_DIM // 4) + title_embedding_dim
        item_layers = []
        for i in range(config.N_LAYERS_ITEM):
            in_dim = item_mlp_input_dim if i == 0 else 256
            item_layers.extend([nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(config.DROPOUT_RATE)])
        item_layers.append(nn.Linear(256, config.EMBEDDING_DIM))
        self.mlp = nn.Sequential(*item_layers)
    def forward(self, features): return self.mlp(torch.cat([self.work_embedding(features['work_id']), self.author_embedding(features['author']), self.publisher_embedding(features['publisher']), self.yop_bin_embedding(features['yop_bin']), features['title_embedding']], dim=1))
class FinalTwoTowerModel(nn.Module):
    def __init__(self, vocab_sizes, title_embedding_dim, config):
        super().__init__(); self.user_tower = UserTower(vocab_sizes, config); self.item_tower = ItemTower(vocab_sizes, title_embedding_dim, config)
    def forward(self, features): return torch.sum(self.user_tower(features) * self.item_tower(features), dim=1)

@st.cache_resource
def load_all_assets():
    default_processed = os.path.join(BASE_DIR, "data", "processed")
    PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", default_processed); assets = {}

    # Validate required files exist in PROCESSED_DATA_DIR
    def _require(path, description):
        if not os.path.exists(path):
            st.error(f"Missing {description}: '{path}'. Ensure preprocessing saved assets and PROCESSED_DATA_DIR is set in .env.")
            raise FileNotFoundError(path)
        return path
    # Load config.json safely (fallback to sensible defaults if missing)
    default_config = {
        "EMBEDDING_DIM": 128,
        "LEARNING_RATE": 0.00037238673471467316,
        "DROPOUT_RATE": 0.20766704720269563,
        "N_LAYERS_USER": 2,
        "N_LAYERS_ITEM": 2,
        "EPOCHS": 30
    }
    config_path = os.path.join(PROCESSED_DATA_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f: config_dict = json.load(f)
        except Exception as e:
            if SHOW_DEBUG:
                st.warning(f"Failed to read {config_path} ({e}). Using built-in defaults.")
            config_dict = default_config
    else:
        if SHOW_DEBUG:
            st.info(f"{config_path} not found. Using built-in default hyperparameters.")
        config_dict = default_config
    assets["config"] = SimpleNamespace(**config_dict); assets["config"].DEVICE = "cpu"
    encoders_path = _require(os.path.join(PROCESSED_DATA_DIR, "encoders.pkl"), "encoders.pkl")
    user_hist_path = _require(os.path.join(PROCESSED_DATA_DIR, "user_history.pkl"), "user_history.pkl")
    books_path = _require(os.path.join(PROCESSED_DATA_DIR, "books_final.parquet"), "books_final.parquet")
    users_path = _require(os.path.join(PROCESSED_DATA_DIR, "users_final.parquet"), "users_final.parquet")
    model_df_path = _require(os.path.join(PROCESSED_DATA_DIR, "model_df_final.parquet"), "model_df_final.parquet")
    title_emb_path = _require(os.path.join(PROCESSED_DATA_DIR, "title_embeddings.pt"), "title_embeddings.pt")
    faiss_collab_path = _require(os.path.join(PROCESSED_DATA_DIR, "faiss_index.bin"), "faiss_index.bin")
    faiss_semantic_path = _require(os.path.join(PROCESSED_DATA_DIR, "semantic_faiss_index.bin"), "semantic_faiss_index.bin")

    with open(encoders_path, "rb") as f: assets["encoders"] = pickle.load(f)
    with open(user_hist_path, "rb") as f: assets["user_history"] = pickle.load(f)
    assets["books"] = pl.read_parquet(books_path)
    assets["users"] = pl.read_parquet(users_path)
    assets["model_df_pd"] = pd.read_parquet(model_df_path)
    assets["title_embeddings"] = torch.load(title_emb_path, map_location="cpu")
    assets["collaborative_faiss_index"] = faiss.read_index(faiss_collab_path)
    assets["semantic_faiss_index"] = faiss.read_index(faiss_semantic_path)
    
    # *** THE DEFINITIVE FIX ***
    # Suppress output ONLY during the model instantiation and loading block.
    with st_suppress_stdout():
        vocab_sizes = {k: len(v.classes_) for k, v in assets["encoders"].items()}
        model = FinalTwoTowerModel(vocab_sizes, assets["title_embeddings"].shape[1], assets["config"])
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
                    f"Weights partially loaded. Dropped: {len(dropped)}, Missing: {len(missing)}, Unexpected: {len(unexpected)}.\n"
                    f"This happens when encoders differ from training time. Ensure encoders.pkl matches the trained model."
                )
        except Exception as e:
            st.error(f"Failed to load model weights from '{model_path}': {e}")
            raise
        assets["model"] = model.eval()
    
    ratings_df = pl.read_parquet(_require(os.path.join(PROCESSED_DATA_DIR, "ratings_final.parquet"), "ratings_final.parquet"))
    popular_df = ratings_df['work_id'].value_counts().sort('count', descending=True).head(20)
    assets["most_popular_ids"] = popular_df['work_id'].to_list()
    assets["core_user_ids"] = set(assets["model_df_pd"]['user_id'].unique())
    return assets

def get_recommendations(mode, user_id_str, book_title_str, k=5):
    # This function is now correct and does not need to change.
    assets = load_all_assets(); books_df = assets['books']; core_user_ids = assets['core_user_ids']; encoders = assets['encoders']
    info_text = ""; work_ids = []
    if not mode: return "Please select a mode.", []
    if mode == "Recommend for User":
        user_id = int(user_id_str) if user_id_str else None
        if not user_id: return "Please select a User ID.", []
        if user_id in core_user_ids:
            info_text = f"Showing personalized recommendations for Core User {user_id}."
            user_tower = assets['model'].user_tower; model_df_pd = assets['model_df_pd']; user_history = assets['user_history']
            user_id_encoded = encoders['user_id'].transform([user_id])[0]
            user_features_row = model_df_pd[model_df_pd['user_id_encoded'] == user_id_encoded].iloc[0]
            features = {'user_id': torch.tensor([user_id_encoded]), 'age_bin': torch.tensor([user_features_row['age_bin_encoded']]), 'country': torch.tensor([user_features_row['country_encoded']])}
            with torch.no_grad(): user_embedding = user_tower(features).cpu().numpy()
            faiss.normalize_L2(user_embedding)
            search_k = k + len(user_history.get(user_id, set()))
            distances, indices = assets['collaborative_faiss_index'].search(user_embedding, search_k)
            recommended_ids_encoded = [idx for idx in indices[0] if idx not in user_history.get(user_id_encoded, set())]
            work_ids = encoders['work_id'].inverse_transform(recommended_ids_encoded)
        else:
            info_text = f"Showing a random selection of popular books for Cold Start User {user_id}."
            work_ids = random.sample(assets['most_popular_ids'], k)
    elif mode == "Find Similar Items":
        if not book_title_str: return "Please select a book title.", []
        query_book = books_df.filter(pl.col('canonical_title') == book_title_str)
        if query_book.is_empty(): return f"Book '{book_title_str}' not found.", []
        query_work_id = query_book['work_id'][0]; query_author = query_book['canonical_author'][0]
        info_text = f"Showing items similar to '{book_title_str}'."
        core_work_ids = set(encoders['work_id'].classes_)
        if query_work_id in core_work_ids:
            work_id_encoded = encoders['work_id'].transform([query_work_id])[0]
            query_vector = assets['title_embeddings'][work_id_encoded:work_id_encoded+1].cpu().numpy()
            distances, indices = assets['semantic_faiss_index'].search(query_vector, k + 1)
            similar_ids_encoded = [idx for idx in indices[0] if idx != work_id_encoded]
            work_ids = encoders['work_id'].inverse_transform(similar_ids_encoded)
        else:
            info_text += f"\n(This is a Cold Start item, trying author fallback...)"
            author_recs = books_df.filter((pl.col('canonical_author') == query_author) & (pl.col('work_id') != query_work_id))
            work_ids = author_recs.head(k)['work_id'].to_list()
            if not work_ids:
                info_text += "\nNo other books by this author found. Showing overall most popular books instead."
                work_ids = random.sample(assets['most_popular_ids'], k)
    if hasattr(work_ids, '__len__') and len(work_ids) > 0:
        if hasattr(work_ids, 'tolist'): work_ids = work_ids.tolist()
        final_work_ids = work_ids[:k]; rec_books_all_editions = books_df.filter(pl.col('work_id').is_in(final_work_ids)); rec_books_unique = rec_books_all_editions.unique(subset=['work_id'], keep='first')
        order_map = {work_id: i for i, work_id in enumerate(final_work_ids)}; rec_books_list = rec_books_unique.to_dicts()
        rec_books_list.sort(key=lambda b: order_map.get(b['work_id'], float('inf')))
        recommendations = [{'title': r['canonical_title'], 'author': r['canonical_author'], 'image_url': r.get('image_m', ''), 'work_id': r['work_id']} for r in rec_books_list]
        return info_text, recommendations
    return info_text, []

# ==============================================================================
#  STREAMLIT UI (Using the clean, state-driven architecture from before)
# ==============================================================================
st.title("📚 Book Recommendation System")
assets = load_all_assets()
user_ids = sorted(assets["users"].select(pl.col('user_id').drop_nulls().unique())['user_id'].to_list())
book_titles = sorted(assets["books"].select(pl.col('canonical_title').drop_nulls().unique())['canonical_title'].to_list())
if 'recommendations' not in st.session_state: st.session_state.recommendations = None
if 'info_text' not in st.session_state: st.session_state.info_text = None
if 'mode' not in st.session_state: st.session_state.mode = "Recommend for User"
if 'book_title_input' not in st.session_state: st.session_state.book_title_input = None

with st.sidebar:
    st.header("Controls")
    # Persisted mode and optional prefilled title
    mode = st.session_state.get("mode", "Recommend for User")
    book_title_prefill = st.session_state.get("book_title_input", None)

    mode = st.radio("Select Mode", ["Recommend for User", "Find Similar Items"], index=0 if mode=="Recommend for User" else 1)

    if mode == "Recommend for User":
        user_id_input = st.selectbox("Select User ID", options=user_ids)
        book_title_input = None
        st.session_state.book_title_input = None
    else:
        if book_title_prefill and book_title_prefill in book_titles:
            pre_idx = book_titles.index(book_title_prefill)
        else:
            pre_idx = 0
        book_title_input = st.selectbox("Select Book Title", options=book_titles, index=pre_idx)
        user_id_input = None
        st.session_state.book_title_input = book_title_input

    st.session_state.mode = mode

    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating..."):
            st.session_state.info_text, st.session_state.recommendations = get_recommendations(
                mode, str(user_id_input) if user_id_input else None, book_title_input
            )

if st.session_state.info_text:
    st.info(st.session_state.info_text)
if st.session_state.recommendations:
    st.subheader(f"Top {len(st.session_state.recommendations)} Recommendations")
    display_cols = st.columns(5)
    for i, rec in enumerate(st.session_state.recommendations):
        with display_cols[i % 5]:
            if rec.get('image_url') and "http" in rec['image_url']:
                st.image(rec['image_url'], use_container_width=True)
            else:
                st.image("https://placehold.co/120x180?text=No+Image", use_container_width=True)
            st.markdown(f"**{rec['title']}**")
            st.markdown(f"by *{rec['author']}*")
            # Action to jump to similar items and prefill selector
            if st.button("Find Similar", key=f"find_sim_{i}"):
                st.session_state.mode = "Find Similar Items"
                st.session_state.book_title_input = rec['title']
                st.rerun()
elif st.session_state.info_text is not None:
    st.warning("No recommendations could be generated for your selection.")
else:
    st.write("## Welcome!")
    st.write("Select a mode and an option in the sidebar to begin.")