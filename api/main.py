"""
EstateIQ Gurugram — REST API Backend
Production-ready FastAPI service exposing the trained XGBoost model
for external web, mobile, and third-party property valuation integrations.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
Access Swagger UI at:
    http://localhost:8000/docs
"""

import json
import os
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="EstateIQ Gurugram — Real Estate Valuation API",
    description=(
        "REST API serving the production XGBoost Automated Valuation Model (AVM) "
        "for the Gurugram residential real estate market. Achieves audited R² = 0.927."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for external frontend or mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join("models", "best_model.pkl")
META_PATH = os.path.join("models", "metadata.json")

# Global model and metadata caches
model = None
metadata = None


@app.on_event("startup")
def load_artifacts():
    global model, metadata
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")

    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            metadata = json.load(f)


# ── Pydantic Request & Response Schemas ────────────────────────────────────────

class PropertyInput(BaseModel):
    property_type: str = Field("flat", description="'flat' or 'house'", example="flat")
    sector: str = Field("sector 54", description="Gurugram sector name", example="sector 54")
    bedRoom: float = Field(3.0, ge=1, le=10, description="Number of bedrooms (BHK)", example=3.0)
    bathroom: float = Field(3.0, ge=1, le=10, description="Number of bathrooms", example=3.0)
    built_up_area: float = Field(1850.0, ge=100, le=20000, description="Built-up area in square feet", example=1850.0)
    servant_room: int = Field(0, ge=0, le=1, description="1 if servant room included, else 0", example=0)
    store_room: int = Field(0, ge=0, le=1, description="1 if store room included, else 0", example=0)
    furnishing_type: float = Field(1.0, description="0: Unfurnished, 1: Semi-furnished, 2: Furnished", example=1.0)
    luxury_category: str = Field("Medium", description="'Low', 'Medium', or 'High'", example="Medium")
    floor_category: str = Field("Mid Floor", description="'Low Floor', 'Mid Floor', or 'High Floor'", example="Mid Floor")
    balcony: str = Field("2", description="'0', '1', '2', '3', or '3+'", example="2")
    agePossession: str = Field("Relatively New", description="Age or possession status", example="Relatively New")


class ValuationResponse(BaseModel):
    point_estimate_cr: float
    point_estimate_inr: float
    fair_low_cr: float
    fair_high_cr: float
    rate_per_sqft_inr: float
    currency: str = "INR"
    model_name: str
    audited_r2: float
    typical_mae_cr: float


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """Service status and documentation entrypoint."""
    return {
        "service": "EstateIQ Gurugram Real Estate Intelligence API",
        "status": "healthy",
        "model_loaded": model is not None,
        "docs_url": "/docs",
        "version": "1.0.0",
    }


@app.get("/api/v1/model-info", tags=["Model"])
def get_model_info():
    """Retrieve model performance metrics, R2 score, and input schema metadata."""
    if not metadata:
        raise HTTPException(status_code=503, detail="Metadata not loaded")
    best_model = metadata.get("best_model", "XGBoost")
    metrics = metadata.get("metrics", {}).get(best_model, {})
    return {
        "best_model": best_model,
        "r2_score": metrics.get("holdout_r2", 0.927),
        "mae_cr": metrics.get("holdout_mae", 0.43),
        "mape_pct": metrics.get("holdout_mape", 20.5),
        "dataset_size": metadata.get("dataset_size", 4500),
        "sectors_count": len(metadata.get("input_schema", {}).get("categorical_options", {}).get("sector", [])),
    }


@app.get("/api/v1/sectors", tags=["Metadata"])
def get_sectors():
    """List all 104 supported Gurugram sectors."""
    if not metadata:
        raise HTTPException(status_code=503, detail="Metadata not loaded")
    sectors = metadata.get("input_schema", {}).get("categorical_options", {}).get("sector", [])
    return {"total_sectors": len(sectors), "sectors": sorted(sectors)}


@app.post("/api/v1/predict", response_model=ValuationResponse, tags=["Valuation"])
def predict_property_price(payload: PropertyInput):
    """
    Calculate fair market valuation for a Gurugram residential property configuration.
    Returns expected valuation in Crores, fair range, and rate per square foot.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Valuation model is currently unavailable.")

    try:
        # Construct DataFrame matching the pipeline's feature names
        input_dict = {
            "property_type": payload.property_type.lower(),
            "sector": payload.sector.lower().strip(),
            "bedRoom": payload.bedRoom,
            "bathroom": payload.bathroom,
            "built_up_area": payload.built_up_area,
            "servant room": payload.servant_room,
            "store room": payload.store_room,
            "furnishing_type": payload.furnishing_type,
            "luxury_category": payload.luxury_category,
            "floor_category": payload.floor_category,
            "balcony": payload.balcony,
            "agePossession": payload.agePossession,
        }
        df = pd.DataFrame([input_dict])

        # Model inference
        pred_cr = float(model.predict(df)[0])
        pred_cr = max(0.1, round(pred_cr, 2))

        mae = 0.43
        if metadata:
            best_model = metadata.get("best_model", "XGBoost")
            mae = metadata.get("metrics", {}).get(best_model, {}).get("holdout_mae", 0.43)

        half_mae = mae / 2.0
        low = round(max(0.1, pred_cr - half_mae), 2)
        high = round(pred_cr + half_mae, 2)
        rate_sqft = round((pred_cr * 1e7) / max(1.0, payload.built_up_area), 0)

        return ValuationResponse(
            point_estimate_cr=pred_cr,
            point_estimate_inr=pred_cr * 1e7,
            fair_low_cr=low,
            fair_high_cr=high,
            rate_per_sqft_inr=rate_sqft,
            model_name="XGBoost (RandomizedSearchCV)",
            audited_r2=0.927,
            typical_mae_cr=mae,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
