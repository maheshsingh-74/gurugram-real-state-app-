# 🏙️ EstateIQ Gurugram — Real Estate Intelligence & Valuation Portal

> **An enterprise-grade, end-to-end proptech platform featuring complete data engineering, exploratory data analysis, ML price prediction with SHAP explainability, multi-angle property recommendations, geospatial market analytics, financial feasibility calculators, and a production FastAPI REST microservice.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-006600?style=for-the-badge)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-FF6F00?style=for-the-badge)](https://shap.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

---

## 🎯 Problem Statement & Overview

Gurugram's residential real estate market encompasses over **104 sectors**, exhibiting dramatic price dispersion driven by infrastructure (Golf Course Road, Dwarka Expressway, SPR), transit hubs, developer pedigree, and luxury amenities. 

EstateIQ provides a transparent, end-to-end machine learning system covering the full data science lifecycle:
1. **Data Cleaning & Missing Value Imputation**
2. **IQR-Based Outlier Treatment**
3. **Advanced Feature Engineering & Multilevel Preprocessing**
4. **Multivariate Exploratory Data Analysis (EDA)**
5. **Hyperparameter Tuned XGBoost Regression ($R^2 = 0.927$)**
6. **SHAP Model Interpretability (Rupees & Percent Impact)**
7. **Multi-Angle Society Recommender**
8. **Interactive Streamlit Portal & FastAPI Microservice**

---

## 🔄 End-to-End Project Workflow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               1. DATA PREPARATION & EDA                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  missing-value-imputation.ipynb       outlier-treatment.ipynb   data-preprocessing-l2  │
│  • Median/Mode imputation              • IQR outlier filtering   • Feature extraction   │
│  • gurgaon_properties_missing_...csv   • gurgaon_properties_...  • Luxury categorization│
│                                                   │                                    │
│                     eda-multivariate-analysis.ipynb                                    │
│                     • Correlation heatmaps & scatter matrices                          │
│                     • Sector-wise price dispersion & BHK distributions                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                               2. MODEL TRAINING & TUNING                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  train_gurgaon_price_model.py  /  finalmodel.py                                         │
│  • TransformedTargetRegressor (log1p → expm1) to normalize skewed target               │
│  • ColumnTransformer (StandardScaler, OrdinalEncoder, OneHotEncoder)                   │
│  • 80-iteration RandomizedSearchCV with 5-Fold Stratified CV                           │
│  • Exported: models/best_model.pkl & models/metadata.json (Holdout R² = 0.927)        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                               3. RECOMMENDATION ENGINE                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  recommender-system.ipynb  /  recommender_system.py                                     │
│  • ~250 Gurugram residential societies                                                │
│  • 5 Distance/Similarity Metrics: Facilities Cosine, Haversine, Price & Config overlap   │
│  • Exported: recommender_artifacts/recommender_artifacts.pkl                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                               4. DEPLOYMENT & CONSUMPTION                              │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│         🖥️ Streamlit Portal (Port 8501)    │         ⚡ FastAPI Service (Port 8000)      │
│   • 1. Valuation + SHAP + EMI & Taxes     │   • POST /api/v1/predict                   │
│   • 2. 5-Angle Recommendations            │   • GET  /api/v1/sectors                   │
│   • 3. Market Analytics & Sector Compare  │   • GET  /api/v1/model-info                │
│   • 4. Audited Model Insights             │   • Interactive Swagger Docs (/docs)       │
│   • 5. About & Project Architecture       │                                            │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

---

## ✨ Key Feature Highlights

### 1. 🏠 Valuation & Pricing Engine
- **Fair-Market Price Range**: Real-time evaluation calibrated with mean absolute error ($\pm \text{MAE}$) bounds.
- **Dual-Mode SHAP Waterfall**: Toggle between **Indian Rupees ($\pm ₹$)** and **Percentage Impact ($\pm\%$)** with clean, unclipped visual styling.
- **₹/Sq.ft. Benchmark**: Direct comparison against sector median pricing.

### 2. 🏦 Home Loan EMI & Haryana Stamp Duty Estimator
- **Interactive Loan Planner**: Adjustable loan amount, tenure (5–30 yrs), and interest rate sliders.
- **Haryana Government Stamp Duty Engine**:
  - **Female Ownership**: 5%
  - **Male Ownership**: 7%
  - **Joint Ownership**: 6%
  - **Registration Charges**: 1% flat
- **Financial Outlay Breakdown**: Interactive Plotly donut chart highlighting Principal, Interest, Down Payment, and Government Taxes.

### 3. 📄 Downloadable Valuation Certificate
- Export client-ready property valuation summaries in structured markdown/text format including property specs, estimated bracket, per-sqft pricing, and disclaimer notice.

### 4. ⚖️ Head-to-Head Sector Comparison
- Compare any two Gurugram sectors side-by-side:
  - Median and average price per sq.ft.
  - Property type breakdown (Flats vs. Independent Houses)
  - Dominant BHK distributions
  - Luxury vs. budget positioning

### 5. 🏘️ 5-Angle Society Recommendation Engine
- Finds matching societies across 5 distinct dimensions:
  1. **Top Overall Similarity** (Blended score)
  2. **Geographical Proximity** (Haversine distance in km)
  3. **Price & Affordability Match**
  4. **Configuration & Unit Size**
  5. **Landmarks & Transit Connectivity**
- Anti-blocking Google Real Estate query integration to view real-world listings without 403 firewall errors.

### 6. 🔬 Model Governance & Audit Insights
- Transparent display of model parameters, 10-fold cross-validation vs. holdout test splits, and global feature importance.

---

## ⚡ FastAPI REST Microservice

EstateIQ includes a standalone **FastAPI** backend for headless integration into mobile apps or external CRM systems.

### Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health and liveness probe |
| `GET` | `/api/v1/model-info` | Model metadata, $R^2$, features, and version |
| `GET` | `/api/v1/sectors` | List of all 104 supported sectors |
| `POST` | `/api/v1/predict` | Predict property price range and ₹/sqft |

### Sample Request (`POST /api/v1/predict`)
```json
{
  "property_type": "flat",
  "sector": "sector 48",
  "bedRoom": 3,
  "bathroom": 3,
  "balcony": "3+",
  "agePossession": "Relatively New",
  "built_up_area": 1850.0,
  "servant_room": 1,
  "store_room": 0,
  "furnishing_type": "semifurnished",
  "luxury_category": "Medium",
  "floor_category": "Mid Floor"
}
```

### Sample Response
```json
{
  "predicted_price_cr": 2.45,
  "price_range_cr": {
    "low": 2.23,
    "high": 2.66
  },
  "price_per_sqft": 13243.24,
  "currency": "INR Crores"
}
```

---

## 📊 Model Performance Metrics

| Metric | Holdout Test (15%) | 10-Fold Cross-Validation |
|---|---|---|
| **$R^2$ Score** | **0.927** | 0.831 |
| **MAE** | **₹0.43 Cr** | ₹0.51 Cr |
| **MAPE** | **20.5%** | 20.5% |

*The test set was partitioned prior to any hyperparameter tuning to ensure zero data leakage.*

---

## 📁 Repository Structure

```
├── 📓 Research, Cleaning & Analysis
│   ├── missing-value-imputation.ipynb               # Missing value treatment
│   ├── gurgaon_properties_missing_value_imputation.csv
│   ├── outlier-treatment.ipynb                      # Boxplot & IQR outlier filtering
│   ├── gurgaon_properties_outlier_treated.csv
│   ├── data-preprocessing-level-2.ipynb             # Feature engineering
│   ├── gurgaon_properties_post_feature_selection_v2.csv
│   ├── eda-multivariate-analysis.ipynb              # Bivariate & multivariate analysis
│   └── recommender-system.ipynb                     # Similarity matrices prototyping
│
├── 🧠 Modeling & Recommender Pipelines
│   ├── train_gurgaon_price_model.py                 # Pipeline training & CV
│   ├── finalmodel.py                                # Hyperparameter tuning (XGBoost)
│   ├── recommender_system.py                        # Production recommender engine
│   ├── models/
│   │   ├── best_model.pkl                           # Exported XGBoost model
│   │   └── metadata.json                            # Audited metrics & feature schemas
│   └── recommender_artifacts/                       # Precomputed matrices
│
├── 🖥️ User Interfaces & Services
│   ├── Price_prediction.py                          # Primary Streamlit app
│   ├── branding.py                                  # Design system & CSS
│   ├── api/
│   │   └── main.py                                  # FastAPI REST backend
│   └── pages/
│       ├── 1_Recommendations.py                     # Multi-angle recommendations UI
│       ├── 2_Analytics.py                           # Maps & Head-to-Head Comparison
│       ├── 3_Model_Insights.py                      # Audited metrics & SHAP insights
│       └── 4_About.py                               # Architecture & project story
│
├── 📦 Supporting Files
│   ├── static/                                      # Hero banner assets
│   ├── appartments.csv                              # Society reference data
│   ├── sector_coordinates.csv                       # Lat/Long for PyDeck mapping
│   ├── requirements.txt                             # Pinned dependencies
│   └── README.md                                    # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up
```bash
git clone https://github.com/maheshsingh-74/gurugram-real-state-app-.git
cd gurugram-real-state-app-
python -m venv .venv
.venv\Scripts\activate      # On Windows
pip install -r requirements.txt
```

### 2. Launch the Streamlit Portal
```bash
streamlit run Price_prediction.py
```
*Access in browser at: `http://localhost:8501`*

### 3. Launch the FastAPI Microservice (Optional)
```bash
uvicorn api.main:app --port 8000 --reload
```
- Interactive Swagger UI: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- Interactive ReDoc: [`http://localhost:8000/redoc`](http://localhost:8000/redoc)

---

## 👤 Author

**Mahesh Singh Beniwal**  
- 📧 **Email**: [maheshbeniwal74@gmail.com](mailto:maheshbeniwal74@gmail.com)  
- 💻 **GitHub**: [@maheshsingh-74](https://github.com/maheshsingh-74)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details. Built for educational and portfolio demonstration using curated 2023–2024 Gurugram residential market data.
