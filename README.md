# 🏙️ EstateIQ Gurugram — Real Estate Intelligence & Valuation Portal

> **An enterprise-grade, ML-powered proptech platform featuring price prediction with SHAP explainability, multi-angle property recommendations, market geospatial analytics, financial feasibility calculators, and a FastAPI REST microservice.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-006600?style=for-the-badge)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-FF6F00?style=for-the-badge)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

---

## 🎯 Problem Statement & Overview

Gurugram's residential real estate market encompasses over **100+ sectors**, exhibiting drastic price variance driven by infrastructure, transit proximity, luxury amenities, and builder reputation. Homebuyers, investors, and developers frequently encounter:

- **Opaque Pricing**: Difficulty gauging whether a property quote is fair or inflated.
- **Explainability Gap**: Understanding *why* a particular home commands a 30% premium.
- **Hidden Acquisition Costs**: Factoring in Haryana Government stamp duties, registration fees, and loan EMIs.
- **Search Fatigue**: Finding comparable properties across subtle dimensions (e.g., matching amenities or transit nodes rather than just identical sectors).

**EstateIQ** solves this with a machine learning engine trained on 4,500+ curated Gurugram properties, combined with explainable AI (SHAP), interactive financial planning, and production-ready REST endpoints.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           EstateIQ Gurugram Platform                            │
├───────────────────────────────────────┬────────────────────────────────────────┤
│     🖥️ Multi-Page Streamlit UI        │       ⚡ FastAPI REST Microservice       │
│                                       │                                        │
│  1. 🏷️ Price Prediction & Valuation   │  • POST /api/v1/predict (Valuation)    │
│     ├─ SHAP Waterfall (₹ & % toggle)  │  • GET  /api/v1/sectors (104 sectors)  │
│     ├─ Loan EMI & Haryana Stamp Duty  │  • GET  /api/v1/model-info (Metadata)  │
│     ├─ Downloadable Valuation Report  │  • GET  /health (Liveness probe)       │
│     └─ Batch CSV Valuation            │  • Interactive Swagger Docs (/docs)    │
│  2. 🏘️ 5-Angle Recommendations       │                                        │
│  3. 📊 Market Analytics & Comparison  │                                        │
│  4. 🔬 Audited Model Insights         │                                        │
│  5. ℹ️ Project Architecture & About   │                                        │
├───────────────────────────────────────┴────────────────────────────────────────┤
│                       🎨 Shared Design System (branding.py)                     │
│         Dark portal styling · Glassmorphism · KPI cards · Responsive layout     │
├────────────────────────────────────────────────────────────────────────────────┤
│                             🧠 Core ML & Data Layer                             │
│                                                                                │
│  ┌─────────────────────────────────┐      ┌─────────────────────────────────┐  │
│  │    Trained XGBoost Pipeline     │      │   Recommendation Vector Space   │  │
│  │  • Target: log1p / expm1 scaled │      │  • ~250 Societies               │  │
│  │  • ColumnTransformer Pipelines  │      │  • Facilities & Amenities OHE   │  │
│  │  • Holdout R² = 0.927           │      │  • Haversine Geographic Distance │  │
│  │  • Test MAE = ₹0.43 Cr          │      │  • Price/Sqft Cosine Similarity │  │
│  └─────────────────────────────────┘      └─────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 1. 🏠 Valuation & Pricing Engine
- **Instant Fair-Value Range**: Computes baseline price along with an MAE-calibrated conservative and upper valuation boundary.
- **Dual-Mode SHAP Waterfall**: Interactive explainability toggle showing positive/negative price drivers in **Indian Rupees ($\pm ₹$)** or **Percentages ($\pm\%$)** with no overlapping text.
- **Rate Benchmark**: Automatically calculates ₹/sq.ft. and benchmarks against sector averages.

### 2. 🏦 Home Loan EMI & Haryana Stamp Duty Estimator
- **Mortgage Calculator**: Dynamic monthly EMI calculation with tenure (5–30 yrs) and interest rate sliders.
- **Haryana Stamp Duty Computation**: Accurately handles state tax rates:
  - **Female Owners**: 5%
  - **Male Owners**: 7%
  - **Joint Ownership**: 6%
  - **Registration Charges**: +1% flat
- **Total Acquisition Outlay**: Donut chart visualizing Down Payment, Loan Principal, Total Interest, and State Taxes.

### 3. 📄 Downloadable Valuation Certificate
- Export client-ready property valuation summaries in structured markdown/text format including property specs, estimated bracket, per-sqft pricing, and disclaimer notice.

### 4. ⚖️ Head-to-Head Sector Comparison
- Compare any two Gurugram sectors side-by-side:
  - Median and average price per sq.ft.
  - Property type breakdown (Flats vs. Independent Houses)
  - Dominant BHK distributions
  - Luxury vs. budget positioning

### 5. 🏘️ 5-Angle Recommendation Engine
- Recommends matching societies across 5 distinct dimensions:
  1. **Top Overall Similarity** (Blended metric)
  2. **Location Proximity** (Haversine distance in km)
  3. **Price & Affordability**
  4. **Configuration & Unit Size**
  5. **Landmarks & Transit Connectivity**
- Anti-blocking Google Real Estate query integration to view real-world listings without 403 firewall errors.

### 6. 🔬 Model Governance & Audit Insights
- Transparent display of model parameters, 10-fold cross-validation vs. holdout test splits, and global feature importance.

---

## ⚡ FastAPI REST Microservice

EstateIQ includes a standalone **FastAPI** backend for headless integration into mobile apps or third-party CRM systems.

### Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status |
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

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/maheshsingh-74/gurugram-real-state-app-.git
cd PythonProject2
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Streamlit Portal
```bash
streamlit run Price_prediction.py
```
*Access the interactive portal at `http://localhost:8501`*

### 4. Run the REST API Backend (Optional)
```bash
uvicorn api.main:app --port 8000 --reload
```
- Interactive Swagger UI: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- Interactive ReDoc: [`http://localhost:8000/redoc`](http://localhost:8000/redoc)

---

## 📁 Repository Structure

```
├── Price_prediction.py          # Primary valuation portal (Tabs: Valuation, Trends, Projects, Batch)
├── branding.py                  # Shared design system, CSS styling, components & cards
├── finalmodel.py                # Model training, preprocessing pipeline & evaluation
├── recommender_system.py        # 5-angle recommendation engine logic
├── api/
│   └── main.py                  # FastAPI REST service & Pydantic validation schemas
├── pages/
│   ├── 1_Recommendations.py     # Multi-angle society matching UI
│   ├── 2_Analytics.py           # Geospatial PyDeck maps & Head-to-Head Sector Comparison
│   ├── 3_Model_Insights.py      # Audited metrics (R²=0.927), SHAP transparency & architecture
│   └── 4_About.py               # Methodology & problem narrative
├── models/
│   ├── best_model.pkl           # Serialized XGBoost pipeline
│   └── metadata.json            # Model parameters, input schemas & metrics
├── recommender_artifacts/       # Precomputed similarity matrices & society coordinates
├── static/                      # Portal hero banners & visual assets
├── .streamlit/
│   └── config.toml              # Streamlit theme configuration
├── .gitignore                   # Ignore rules for virtualenv, cache, and temp files
└── requirements.txt             # Pinned production dependencies
```

---

## 🐙 Pushing to GitHub

To push your local repository changes to GitHub:

```bash
# 1. Initialize git if not already initialized
git init

# 2. Add remote repository (if not already added)
git remote add origin https://github.com/maheshsingh-74/gurugram-real-state-app-.git

# 3. Stage all project files (ignoring items in .gitignore)
git add .

# 4. Commit changes
git commit -m "feat: complete EstateIQ platform with FastAPI, SHAP explainability, EMI calculator & sector comparison"

# 5. Push to GitHub main branch
git branch -M main
git push -u origin main
```

---

## 👤 Author

**Mahesh Singh Beniwal**  
- 📧 **Email**: [maheshbeniwal74@gmail.com](mailto:maheshbeniwal74@gmail.com)  
- 💻 **GitHub**: [@maheshsingh-74](https://github.com/maheshsingh-74)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details. Built for educational and portfolio demonstration using curated 2023–2024 Gurugram residential market data.
