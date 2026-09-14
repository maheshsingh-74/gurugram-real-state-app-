"""
XGBoost hyperparameter tuning for Gurgaon price prediction.
Requires: pip install xgboost joblib

LightGBM and CatBoost were dropped from this file on purpose - XGBoost's
holdout MAE (0.4387 Cr, R2=0.878) already beat LightGBM (0.4723 Cr) and
tracked well ahead of CatBoost's own CV numbers, and CatBoost's search was
by far the slowest of the three. Keeping only the model actually being used
means this runs in a fraction of the time.

--------------------------------------------------------------------------
Fixes applied vs. the original draft:

  1. `sector` and `agePossession` were listed in BOTH ORDINAL_COLS and
     ONEHOT_COLS, so the ColumnTransformer encoded them twice (once as an
     arbitrary ordinal integer, once as one-hot dummies) and fed both
     versions into the model - redundant/collinear features, not target
     leakage (encoders don't use y). Fixed: one-hot only.

  2. Added a genuine held-out test set (15%), split off BEFORE any tuning.
     The 5-fold search CV and the 10-fold OOF check only ever see the
     training split; the reported "final" MAE is computed on this untouched
     test set - what actually guards against overfitting the hyperparameters
     to the CV folds.

  3. Wrapped in sklearn's TransformedTargetRegressor (func=log1p,
     inverse_func=expm1) instead of manually log-transforming y and using a
     hand-rolled scorer, so RandomizedSearchCV can use the plain
     "neg_mean_absolute_error" scorer directly on the real price scale.
     NOTE: this is a code-cleanliness change only, not a bias correction -
     you're scoring/selecting by MAE, and the MAE-optimal predictor is the
     conditional MEDIAN, not the mean, so exp(pred)-1 from a log1p(y) fit
     already approximates the right target. A mean-bias ("smearing")
     correction would generally *increase* MAE on right-skewed price data,
     so it's intentionally not applied.

  4. Added joblib persistence: the fitted pipeline plus a metadata.json
     (holdout + OOF metrics, and an input_schema block with every
     categorical column's valid values and every numeric column's
     min/max/median) so a Streamlit app can build its form and load the
     model without guessing anything.

Not addressed here (left as upstream/EDA decisions, not this script's bugs):
  - price_per_sqft as a MODEL FEATURE would be direct target leakage
    (it's literally price / area) - fine for EDA/plots, never as a training
    column.
  - floor_ratio (floor / total_floors) is reasonable but the raw
    floor/total_floors columns aren't present in this post-feature-selection
    CSV - add it upstream in the feature-engineering notebook, not here.
--------------------------------------------------------------------------
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
OUTPUT_DIR = "models"
TEST_SIZE = 0.15

# ---------------------------------------------------------------------------
# 1. Load + clean (same 52-row trim as the main script)
# ---------------------------------------------------------------------------
outlier_indices = [44, 333, 501, 589, 641, 673, 845, 854, 949, 1018, 1104, 1165, 1208,
                    1278, 1294, 1365, 1432, 1508, 1545, 1551, 1558, 1579, 1680, 1705,
                    1762, 1781, 1825, 2049, 2098, 2143, 2153, 2252, 2391, 2434, 2478,
                    2496, 2579, 2670, 2672, 2673, 2676, 2710, 2743, 2775, 2869, 2948,
                    2952, 2969, 3167, 3249, 3275, 3388]

df = pd.read_csv("gurgaon_properties_post_feature_selection_v2.csv")
df = df.drop(index=outlier_indices).reset_index(drop=True)

X = df.drop(columns=["price"])
y = df["price"]  # raw scale - TransformedTargetRegressor handles log1p/expm1 internally

# Held out ONCE, before any tuning.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

NUM_COLS = ["bedRoom", "bathroom", "built_up_area", "servant room", "store room"]

# `sector` / `agePossession` are one-hot ONLY - see fix #1 in the module docstring.
ORDINAL_COLS = ["property_type", "balcony", "furnishing_type", "luxury_category", "floor_category"]
ONEHOT_COLS = ["sector", "agePossession"]
CATEGORICAL_COLS = ORDINAL_COLS + ONEHOT_COLS  # used for the Streamlit input schema below

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL_COLS),
        ("cat1", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ONEHOT_COLS),
    ],
    remainder="passthrough",
)

KFOLD5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
KFOLD10 = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)


def _metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, r2, mape


def report(name, search):
    """Print search results, an OOF confirmation on the TRAIN split only,
    and the real number: performance on the never-touched holdout set."""
    print(f"\n[{name}] best 5-fold CV MAE = {-search.best_score_:.4f} Cr")
    print(f"[{name}] best params:", search.best_params_)
    best = search.best_estimator_  # refit on the full training split

    oof_pred = cross_val_predict(best, X_train, y_train, cv=KFOLD10, n_jobs=-1)
    oof_mae, oof_r2, oof_mape = _metrics(y_train, oof_pred)
    print(f"[{name}] 10-fold OOF (train)  -> MAE={oof_mae:.4f} Cr  R2={oof_r2:.4f}  MAPE={oof_mape:.2f}%")

    test_pred = best.predict(X_test)
    test_mae, test_r2, test_mape = _metrics(y_test, test_pred)
    print(f"[{name}] HOLDOUT TEST          -> MAE={test_mae:.4f} Cr  R2={test_r2:.4f}  MAPE={test_mape:.2f}%")

    return {
        "pipeline": best,
        "mae": test_mae, "r2": test_r2, "mape": test_mape,
        "oof_mae": oof_mae, "oof_r2": oof_r2, "oof_mape": oof_mape,
    }


# ---------------------------------------------------------------------------
# 2. XGBoost
# ---------------------------------------------------------------------------
def tune_xgboost():
    from xgboost import XGBRegressor

    base_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")),
    ])
    ttr = TransformedTargetRegressor(regressor=base_pipe, func=np.log1p, inverse_func=np.expm1)
    grid = {
        "regressor__model__n_estimators": [300, 500, 800, 1200, 1600],
        "regressor__model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "regressor__model__max_depth": [3, 4, 5, 6, 8],
        "regressor__model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__model__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
        "regressor__model__min_child_weight": [1, 3, 5, 10],
        "regressor__model__gamma": [0, 0.1, 0.3, 0.5],
        "regressor__model__reg_alpha": [0, 0.1, 0.5, 1.0],
        "regressor__model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    search = RandomizedSearchCV(
        ttr, grid, n_iter=80, cv=KFOLD5, scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X_train, y_train)
    return report("XGBoost", search)


# ---------------------------------------------------------------------------
# 3. Save for Streamlit deployment
# ---------------------------------------------------------------------------
def _input_schema():
    """Everything a Streamlit form needs to build itself without guessing:
    valid values for every categorical column, and min/max/median for every
    numeric column - all read straight from the training data actually used."""
    return {
        "numeric_cols": NUM_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "categorical_options": {
            c: sorted(x for x in X_train[c].dropna().unique().tolist())
            for c in CATEGORICAL_COLS
        },
        "numeric_ranges": {
            c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                "median": float(X_train[c].median())}
            for c in NUM_COLS
        },
    }


def save_model(name, result, output_dir=OUTPUT_DIR):
    """Save the tuned model as the deployment artifact.

    result: {"pipeline", "mae", "r2", "mape", "oof_mae", "oof_r2", "oof_mape"}
    (i.e. exactly what tune_xgboost() returns)
    """
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, f"{name.lower()}_pipeline.pkl")
    joblib.dump(result["pipeline"], path)
    best_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(result["pipeline"], best_path)

    metadata = {
        "best_model": name,
        "test_size": TEST_SIZE,
        "metrics": {
            name: {"holdout_mae": result["mae"], "holdout_r2": result["r2"],
                   "holdout_mape": result["mape"], "oof_mae": result["oof_mae"],
                   "oof_r2": result["oof_r2"], "oof_mape": result["oof_mape"]}
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_schema": _input_schema(),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {name} pipeline -> {path}")
    print(f"Also saved as best_model.pkl -> {best_path}")
    print(f"Metadata (+ input schema for the Streamlit form) -> "
          f"{os.path.join(output_dir, 'metadata.json')}")
    print(f"\n{name} holdout: MAE={result['mae']:.4f} Cr  R2={result['r2']:.4f}  "
          f"MAPE={result['mape']:.2f}%")
    return path


if __name__ == "__main__":
    xgb_result = tune_xgboost()
    save_model("XGBoost", xgb_result)