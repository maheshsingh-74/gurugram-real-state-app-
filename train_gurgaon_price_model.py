"""
Gurgaon Property Price Model — v2 (tuned)
==========================================
Fixes vs. the original notebook:
  1. BUG FIX: GridSearchCV found best_params_ for RandomForest, but the pipeline that
     actually got pickled used untuned defaults. The tuned params are now applied.
  2. OrdinalEncoder now uses handle_unknown='use_encoded_value' so a rare sector/category
     that only shows up in a validation fold no longer crashes CV / breaks in production
     when a brand-new sector value is seen.
  3. Outlier trim: ~50 rows have impossible price-per-sqft (e.g. a 9-bedroom flat in a
     60 sqft "built_up_area") — these are data-entry errors, not real signal, and were
     inflating error metrics.
  4. Evaluation is now apples-to-apples: MAE/R2/MAPE are all computed from the SAME
     10-fold out-of-fold predictions on the original price scale (Cr), not a mix of
     CV-R2 + single-split-MAE like before.
  5. Hyperparameters are tuned (RandomizedSearchCV, scored directly on real-scale MAE)
     for RandomForest and two boosting algorithms available in sklearn
     (HistGradientBoosting, GradientBoosting), plus ready-to-run tuning blocks for
     XGBoost / LightGBM / CatBoost if you have them installed locally.

Results (10-fold out-of-fold CV, honest, on original price scale in Cr):
    Old baseline (RF-500, default params)     MAE=0.494  R2=0.830  MAPE=21.6%
    Tuned RandomForest (this script)          MAE=0.487  R2=0.838  MAPE=19.9%
    Tuned HistGradientBoosting                MAE=0.506  R2=0.847  MAPE=21.4%
    Tuned GradientBoosting                    MAE=0.516  R2=0.836  MAPE=21.7%

RandomForest (tuned) wins on MAE; HistGradientBoosting wins on R2 (better on the
expensive-property tail). If you have xgboost/lightgbm/catboost installed, run the
tuning blocks near the bottom — those algorithms usually beat both on this kind of
tabular data, and a stacked ensemble of all of them is very likely to beat any single
model. A basic 3-way stack (RF + HGB + GBR) is included but did NOT clearly beat tuned
RF alone on this dataset (see comments) — worth retrying once XGBoost is in the mix.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "gurgaon_properties_post_feature_selection_v2.csv"


# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop rows with an impossible price-per-sqft (data-entry errors, not real signal).
    # Bounds are generous on purpose — this removes ~1.5% of rows, all clearly broken
    # (e.g. price*1e7/built_up_area < Rs 2,000/sqft or > Rs 60,000/sqft in Gurgaon).
    ppsf = df["price"] * 1e7 / df["built_up_area"]
    before = len(df)
    df = df[(ppsf >= 2000) & (ppsf <= 60000)].reset_index(drop=True)
    print(f"Outlier trim: {before} -> {len(df)} rows ({before - len(df)} removed)")
    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
NUM_COLS = ["bedRoom", "bathroom", "built_up_area", "servant room", "store room"]
ORDINAL_COLS = [
    "property_type",
    "sector",
    "balcony",
    "agePossession",
    "furnishing_type",
    "luxury_category",
    "floor_category",
]
ONEHOT_COLS = ["sector", "agePossession"]  # also onehot'd on top of the ordinal encoding


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            (
                "cat",
                # handle_unknown fix: a category unseen in a training fold (or at
                # inference time, e.g. a brand-new sector) no longer raises an error.
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ORDINAL_COLS,
            ),
            (
                "cat1",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                ONEHOT_COLS,
            ),
        ],
        remainder="passthrough",
    )


# ---------------------------------------------------------------------------
# 3. Real-scale scorer (the model is trained on log1p(price); CV should score
#    on the scale that actually matters — price in Cr, after inverting the log)
# ---------------------------------------------------------------------------
def _mae_on_original_scale(y_true_log, y_pred_log):
    return mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))


MAE_SCORER = make_scorer(_mae_on_original_scale, greater_is_better=False)


def evaluate(name: str, pipe: Pipeline, X: pd.DataFrame, y_log: pd.Series, cv_splits: int = 10):
    """Honest out-of-fold CV eval, scored on the real price scale."""
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_log = cross_val_predict(pipe, X, y_log, cv=kfold, n_jobs=1)
    oof_pred, oof_true = np.expm1(oof_log), np.expm1(y_log)
    mae = mean_absolute_error(oof_true, oof_pred)
    r2 = r2_score(oof_true, oof_pred)
    mape = np.mean(np.abs((oof_true - oof_pred) / oof_true)) * 100
    print(f"{name:40s} MAE={mae:.4f} Cr  R2={r2:.4f}  MAPE={mape:.2f}%")
    return mae, r2, mape


# ---------------------------------------------------------------------------
# 4. Tuned models (hyperparameters found via RandomizedSearchCV, scored on
#    real-scale MAE — see the tune_* blocks at the bottom to reproduce/extend)
# ---------------------------------------------------------------------------
def tuned_random_forest() -> RandomForestRegressor:
    # Best found: n_estimators bumped to 500 for the final model after search
    # picked these regularization-relevant params on a smaller n_estimators grid.
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        max_features=0.6,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def tuned_hist_gb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_leaf_nodes=15,
        max_depth=None,
        min_samples_leaf=10,
        l2_regularization=0,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )


def tuned_gbr() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=5,
        max_features=1.0,
        n_estimators=800,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )


def stacked_ensemble() -> StackingRegressor:
    """Optional: blend of the three tuned models above with a Ridge meta-learner.
    On this dataset it roughly matched (didn't clearly beat) tuned RF alone —
    worth revisiting once XGBoost/LightGBM/CatBoost are added as base learners."""
    return StackingRegressor(
        estimators=[
            ("rf", tuned_random_forest()),
            ("hgb", tuned_hist_gb()),
            ("gbr", tuned_gbr()),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 20)),
        cv=3,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# 5. Main: evaluate candidates, pick the winner, export the FIXED pipeline
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    X = df.drop(columns=["price"])
    y_log = np.log1p(df["price"])
    pre = make_preprocessor()

    print("\n--- Model comparison (10-fold OOF CV, real price scale) ---")
    candidates = {
        "RandomForest (tuned)": tuned_random_forest(),
        "HistGradientBoosting (tuned)": tuned_hist_gb(),
        "GradientBoosting (tuned)": tuned_gbr(),
    }
    results = {}
    for name, model in candidates.items():
        pipe = Pipeline([("preprocessor", pre), ("regressor", model)])
        results[name] = evaluate(name, pipe, X, y_log)

    best_name = min(results, key=lambda k: results[k][0])  # lowest MAE wins
    print(f"\nBest model by MAE: {best_name}")

    # ---- Fit the winner on ALL data and export (bug fix: the tuned params are
    # actually used this time, unlike the original notebook's export step) ----
    final_pipe = Pipeline([("preprocessor", make_preprocessor()), ("regressor", candidates[best_name])])
    final_pipe.fit(X, y_log)

    with open("pipeline.pkl", "wb") as f:
        pickle.dump(final_pipe, f)
    with open("df.pkl", "wb") as f:
        pickle.dump(X, f)
    print("\nSaved pipeline.pkl and df.pkl")

    # Sanity check against the same sample row used in the original notebook
    sample = pd.DataFrame(
        [["house", "sector 102", 4, 3, "3+", "New Property", 2750, 0, 0, "unfurnished", "Low", "Low Floor"]],
        columns=X.columns,
    )
    pred_price = np.expm1(final_pipe.predict(sample))[0]
    print(f"Sanity check prediction for sample row: {pred_price:.2f} Cr")


# ---------------------------------------------------------------------------
# 6. Hyperparameter tuning blocks — reproduce or extend the search.
#    RandomForest / HistGB / GBR ran here in a resource-constrained sandbox
#    (single CPU core), so budgets below are intentionally conservative.
#    Increase n_iter / cv folds if you have more compute.
# ---------------------------------------------------------------------------
def tune_random_forest(X, y_log):
    pipe = Pipeline([("preprocessor", make_preprocessor()), ("regressor", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    grid = {
        "regressor__n_estimators": [200, 300, 500],
        "regressor__max_depth": [None, 20, 25, 30],
        "regressor__max_features": [1.0, "sqrt", 0.6],
        "regressor__min_samples_leaf": [1, 2, 3],
        "regressor__min_samples_split": [2, 5, 10],
    }
    search = RandomizedSearchCV(
        pipe, grid, n_iter=25, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring=MAE_SCORER, n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X, y_log)
    print("Best RF params:", search.best_params_, "| CV MAE:", -search.best_score_)
    return search


def tune_hist_gb(X, y_log):
    pipe = Pipeline([("preprocessor", make_preprocessor()), ("regressor", HistGradientBoostingRegressor(
        random_state=RANDOM_STATE, early_stopping=True, n_iter_no_change=15,
        validation_fraction=0.1, max_iter=1000,
    ))])
    grid = {
        "regressor__learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15],
        "regressor__max_leaf_nodes": [15, 31, 63, 127],
        "regressor__max_depth": [None, 4, 6, 8],
        "regressor__min_samples_leaf": [5, 10, 20, 30],
        "regressor__l2_regularization": [0, 0.1, 0.5, 1.0],
    }
    search = RandomizedSearchCV(
        pipe, grid, n_iter=40, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring=MAE_SCORER, n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X, y_log)
    print("Best HGB params:", search.best_params_, "| CV MAE:", -search.best_score_)
    return search


def tune_gbr(X, y_log):
    pipe = Pipeline([("preprocessor", make_preprocessor()), ("regressor", GradientBoostingRegressor(
        random_state=RANDOM_STATE, n_iter_no_change=15, validation_fraction=0.1, n_estimators=800,
    ))])
    grid = {
        "regressor__learning_rate": [0.02, 0.05, 0.08, 0.1],
        "regressor__max_depth": [2, 3, 4, 5],
        "regressor__subsample": [0.6, 0.8, 1.0],
        "regressor__min_samples_leaf": [1, 5, 10, 20],
        "regressor__max_features": [1.0, "sqrt", 0.5],
    }
    search = RandomizedSearchCV(
        pipe, grid, n_iter=25, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring=MAE_SCORER, n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X, y_log)
    print("Best GBR params:", search.best_params_, "| CV MAE:", -search.best_score_)
    return search


def tune_xgboost(X, y_log):
    """Requires: pip install xgboost. XGBoost usually edges out sklearn's own
    boosting on tabular data like this — worth running if you have it."""
    from xgboost import XGBRegressor

    pipe = Pipeline([("preprocessor", make_preprocessor()), ("regressor", XGBRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    ))])
    grid = {
        "regressor__n_estimators": [300, 500, 800, 1200],
        "regressor__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
        "regressor__max_depth": [3, 4, 5, 6, 8],
        "regressor__subsample": [0.6, 0.7, 0.8, 1.0],
        "regressor__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "regressor__min_child_weight": [1, 3, 5, 10],
        "regressor__reg_alpha": [0, 0.1, 0.5, 1.0],
        "regressor__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    search = RandomizedSearchCV(
        pipe, grid, n_iter=60, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring=MAE_SCORER, n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X, y_log)
    print("Best XGBoost params:", search.best_params_, "| CV MAE:", -search.best_score_)
    return search


def tune_lightgbm(X, y_log):
    """Requires: pip install lightgbm. Handles the high-cardinality 'sector'
    column natively as a categorical feature if you skip one-hot/ordinal
    encoding for it and instead pass it as pandas 'category' dtype."""
    from lightgbm import LGBMRegressor

    pipe = Pipeline([("preprocessor", make_preprocessor()), ("regressor", LGBMRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    ))])
    grid = {
        "regressor__n_estimators": [300, 500, 800, 1200],
        "regressor__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
        "regressor__num_leaves": [15, 31, 63, 127],
        "regressor__max_depth": [-1, 4, 6, 8, 10],
        "regressor__subsample": [0.6, 0.8, 1.0],
        "regressor__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "regressor__min_child_samples": [5, 10, 20, 30],
        "regressor__reg_alpha": [0, 0.1, 0.5, 1.0],
        "regressor__reg_lambda": [0, 0.1, 0.5, 1.0],
    }
    search = RandomizedSearchCV(
        pipe, grid, n_iter=60, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring=MAE_SCORER, n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X, y_log)
    print("Best LightGBM params:", search.best_params_, "| CV MAE:", -search.best_score_)
    return search


def tune_catboost(X, y_log):
    """Requires: pip install catboost. Often the strongest out-of-the-box
    boosting library for mixed numeric/categorical tabular data."""
    from catboost import CatBoostRegressor

    cat_feature_idx = [X.columns.get_loc(c) for c in ORDINAL_COLS]
    pipe = Pipeline([("regressor", CatBoostRegressor(
        random_state=RANDOM_STATE, verbose=0, cat_features=cat_feature_idx,
    ))])  # CatBoost handles categoricals natively -> skip the ColumnTransformer
    grid = {
        "regressor__iterations": [300, 500, 800, 1200],
        "regressor__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
        "regressor__depth": [4, 6, 8, 10],
        "regressor__l2_leaf_reg": [1, 3, 5, 10],
        "regressor__subsample": [0.6, 0.8, 1.0],
    }
    search = RandomizedSearchCV(
        pipe, grid, n_iter=40, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring=MAE_SCORER, n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X, y_log)
    print("Best CatBoost params:", search.best_params_, "| CV MAE:", -search.best_score_)
    return search


if __name__ == "__main__":
    main()

    # Uncomment to re-run tuning searches (slow — see time budget notes above):
    # df = load_data()
    # X, y_log = df.drop(columns=["price"]), np.log1p(df["price"])
    # tune_random_forest(X, y_log)
    # tune_hist_gb(X, y_log)
    # tune_gbr(X, y_log)
    # tune_xgboost(X, y_log)      # needs: pip install xgboost
    # tune_lightgbm(X, y_log)     # needs: pip install lightgbm
    # tune_catboost(X, y_log)     # needs: pip install catboost
