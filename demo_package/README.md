# 🛍️ Product Recommendation Dashboard - Databricks Demo

## Quick Setup for Databricks

### 📦 What's Included
- `dashboard.py` - Main dashboard file
- `data_processed/` - All required data files (12 files)
- This README

---

## 🚀 Quick Start (5 minutes)

### Step 1: Upload Files to Databricks

**Upload to Workspace:**
- `dashboard.py` → `/Workspace/Your_Folder/dashboard.py`

**Upload to DBFS:**
- All files from `data_processed/` → `/dbfs/FileStore/dashboard_data/processed/`

### Step 2: Setup in Databricks Notebook

```python
# Install packages
%pip install streamlit plotly python-dotenv torch faiss-cpu polars pandas numpy

# Set environment
import os
os.environ['PROCESSED_DATA_DIR'] = '/dbfs/FileStore/dashboard_data/processed'
```

### Step 3: Access Dashboard

**Method 1: Web Apps (Recommended)**
1. Go to **Apps** → **Web Apps** → **Create Web App**
2. **Entry Point:** `/Workspace/Your_Folder/dashboard.py`
3. **Environment Variable:** `PROCESSED_DATA_DIR=/dbfs/FileStore/dashboard_data/processed`
4. Click **Deploy**
5. Get URL and share!

**Method 2: Cluster IP**
1. Get cluster public IP from Compute settings
2. Run dashboard in notebook
3. Access: `http://CLUSTER_IP:8501`

---

## 📁 File Structure

```
demo_package/
├── dashboard.py              # Main dashboard
├── data_processed/           # All data files (12 files)
│   ├── cold_start_products.parquet
│   ├── config.json
│   ├── encoders.pkl
│   ├── faiss_index.bin
│   ├── faiss_product_mapping.json
│   ├── final_ranking_model.pth
│   ├── model_df_final.parquet
│   ├── products_final.parquet
│   ├── semantic_faiss_index.bin
│   ├── user_history.pkl
│   ├── users_final.parquet
│   └── vocab_sizes.json
└── README.md                 # This file
```

---

## ✅ Checklist

- [ ] Upload `dashboard.py` to Workspace
- [ ] Upload all files from `data_processed/` to DBFS
- [ ] Install packages
- [ ] Set environment variable
- [ ] Create Web App or run dashboard
- [ ] Test dashboard loads
- [ ] Share URL with others

---

## 🎯 That's It!

Total files: 14 (1 dashboard + 12 data + 1 README)

Total size: ~29MB

Ready for demo! 🎉
