"""
XGBoost / LightGBM / CatBoost hyperparameter tuning for Gurgaon price prediction.
Requires: pip install xgboost lightgbm catboost joblib

Run each block independently (they're not cheap).

--------------------------------------------------------------------------
Fixes applied vs. the original draft:

  1. `sector` and `agePossession` were listed in BOTH ORDINAL_COLS and
     ONEHOT_COLS, so the ColumnTransformer encoded them twice (once as an
     arbitrary ordinal integer, once as one-hot dummies) and fed both
     versions into the model. Note: this was redundant/collinear features,
     not target leakage - encoders don't use y, so no leakage was ever
     possible here. Fixed by making them one-hot only.

  2. CatBoost's `subsample` param requires bootstrap_type in
     {Bernoulli, MVS, Poisson}; CatBoost's default (Bayesian) does NOT
     support it, so the search errored out on the first candidate that
     tried `subsample`. Fixed with bootstrap_type="Bernoulli".

  3. CatBoost's categorical columns are cast to str via a small helper
     (_cast_categoricals) instead of a manually pre-cast DataFrame, and its
     final model (LogTargetCatBoost, see fix #7) exposes the same
     predict(raw_dataframe) interface as the other two - what a Streamlit
     app needs to swap models without special-casing CatBoost. The cast is
     a fixed, row-independent operation (no statistics fitted from data),
     so there's no leakage risk there.

  4. Added a genuine held-out test set (15%), split off BEFORE any tuning.
     RandomizedSearchCV / the 5-fold search CV and the 10-fold OOF check
     only ever see the training split; the reported "final" MAE per model
     is computed on this untouched test set. This is what actually guards
     against overfitting the hyperparameters to the CV folds - the model
     wrapper change in point 5 below does NOT do this by itself.

  5. XGBoost and LightGBM are each wrapped in sklearn's
     TransformedTargetRegressor (func=log1p, inverse_func=expm1) instead of
     manually log-transforming y and using a hand-rolled scorer. This lets
     RandomizedSearchCV use the plain "neg_mean_absolute_error" scorer
     directly on the real price scale. NOTE: this is a code-cleanliness
     change only. It does not correct for Jensen's-inequality bias, and
     deliberately isn't meant to: you're scoring and selecting by MAE, and
     the MAE-optimal predictor is the conditional MEDIAN, not the mean.
     exp(pred)-1 from a model fit on log1p(y) already approximates the
     median. A smearing/mean-bias correction would pull predictions toward
     the mean and would generally *increase* MAE on right-skewed price data
     - so it's intentionally not applied here. (CatBoost does NOT use this
     wrapper - see fix #7.)

  6. Added joblib persistence + a comparison step that prints MAE / R2 /
     MAPE (OOF-on-train AND holdout-test) for all three models, picks the
     best by holdout MAE, and saves it as models/best_model.pkl +
     metadata.json - ready to load straight into a Streamlit app.

  7. CatBoost tuning does NOT use RandomizedSearchCV / cross_val_predict /
     TransformedTargetRegressor. On scikit-learn >= 1.6, sklearn.base.clone()
     added a stricter check that CatBoostRegressor(cat_features=...) fails:
     CatBoost's constructor doesn't store `cat_features` back exactly as
     passed, and every one of those three sklearn tools calls clone()
     internally, so all three crash with:
         RuntimeError: Cannot clone object CatBoostRegressor(...), as the
         constructor either does not set or modifies parameter cat_features
     (XGBoost/LightGBM don't have this problem.) Rather than pin your
     scikit-learn version, tune_catboost() below does its own randomized
     search and K-fold loop by hand, constructing a fresh CatBoostRegressor
     each time instead of asking sklearn to clone one - so it works on any
     scikit-learn version, old or new. See LogTargetCatBoost below.

Not addressed here (left as upstream/EDA decisions, not this script's bugs):
  - price_per_sqft as a MODEL FEATURE would be direct target leakage
    (it's literally price / area) - fine to compute for EDA/plots, never
    as a training column.
  - floor_ratio (floor / total_floors) is a reasonable feature, but the raw
    floor/total_floors columns aren't present in this post-feature-selection
    CSV. Add it upstream in the feature-engineering notebook, not here.
--------------------------------------------------------------------------
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    KFold,
    ParameterSampler,
    RandomizedSearchCV,
    cross_val_predict,
    train_test_split,
)
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

# Held out ONCE, before any tuning. Nothing below this line touches these rows
# again until the final report() call per model.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

NUM_COLS = ["bedRoom", "bathroom", "built_up_area", "servant room", "store room"]

# All 7 nominal/ordinal categorical columns - this full list is what CatBoost
# uses natively (it handles categoricals itself, no need to encode).
CATEGORICAL_COLS = ["property_type", "sector", "balcony", "agePossession",
                    "furnishing_type", "luxury_category", "floor_category"]

# For the sklearn ColumnTransformer (XGBoost / LightGBM path) that list is
# split into ordinal-encoded vs one-hot-encoded columns. `sector` and
# `agePossession` are one-hot ONLY - see fix #1 in the module docstring.
ORDINAL_COLS = ["property_type", "balcony", "furnishing_type", "luxury_category", "floor_category"]
ONEHOT_COLS = ["sector", "agePossession"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL_COLS),
        ("cat1", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ONEHOT_COLS),
    ],
    remainder="passthrough",
)


def _cast_categoricals(X, cols):
    """Module-level (picklable) helper for the CatBoost pipeline: casts the
    categorical columns to str so CatBoost treats them as categorical rather
    than trying (and failing) to read them as numeric. Pure dtype cast, no
    statistics fitted from data -> not a leakage vector."""
    X = X.copy()
    for c in cols:
        X[c] = X[c].astype(str)
    return X


class LogTargetCatBoost(BaseEstimator, RegressorMixin):
    """fit/predict wrapper around CatBoostRegressor: trains on log1p(y),
    predicts back on the real price scale. Used instead of
    TransformedTargetRegressor because that wrapper calls sklearn.base.clone()
    internally, which currently crashes on CatBoostRegressor(cat_features=...)
    - see fix #7 in the module docstring.

    Deliberately builds a fresh CatBoostRegressor inside fit() rather than
    storing one as a constructor attribute, so cloning *this* wrapper (e.g.
    inside StackingRegressor later) only copies plain ints/dicts and never
    touches the problematic CatBoost object - this class is clone-safe even
    though CatBoostRegressor itself currently isn't.
    """

    def __init__(self, cat_feature_idx, params):
        self.cat_feature_idx = cat_feature_idx
        self.params = params

    def fit(self, X, y):
        from catboost import CatBoostRegressor

        X_c = _cast_categoricals(X, CATEGORICAL_COLS)
        self.model_ = CatBoostRegressor(
            random_state=RANDOM_STATE,
            verbose=0,
            cat_features=self.cat_feature_idx,
            bootstrap_type="Bernoulli",  # required for `subsample` to be tunable
            thread_count=-1,  # safe to use all cores: this loop is sequential,
            # no competing sklearn-level parallel jobs
            max_ctr_complexity=2,  # CatBoost's default tries COMBINATIONS of
            # categorical columns to find interaction
            # splits. With 7 categorical columns here -
            # including high-cardinality `sector` - that
            # combinatorial search can silently balloon
            # a single fit from seconds to tens of
            # minutes, especially at high depth/
            # border_count. Capping it at 2 keeps
            # 2-way interactions (still useful) without
            # the runaway cost of 3-way+ combinations.
            **self.params,
        )
        self.model_.fit(X_c, np.log1p(y))
        return self

    def predict(self, X):
        X_c = _cast_categoricals(X, CATEGORICAL_COLS)
        return np.expm1(self.model_.predict(X_c))


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

    # Sanity check on the CV process itself (train data only - the search
    # already used these rows for tuning, so this is NOT a generalization
    # estimate, just a check that the CV folds behave consistently).
    oof_pred = cross_val_predict(best, X_train, y_train, cv=KFOLD10, n_jobs=-1)
    oof_mae, oof_r2, oof_mape = _metrics(y_train, oof_pred)
    print(f"[{name}] 10-fold OOF (train)  -> MAE={oof_mae:.4f} Cr  R2={oof_r2:.4f}  MAPE={oof_mape:.2f}%")

    # The real generalization number: untouched holdout set.
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
# 3. LightGBM
# ---------------------------------------------------------------------------
def tune_lightgbm():
    from lightgbm import LGBMRegressor

    base_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
    ])
    ttr = TransformedTargetRegressor(regressor=base_pipe, func=np.log1p, inverse_func=np.expm1)
    grid = {
        "regressor__model__n_estimators": [300, 500, 800, 1200, 1600],
        "regressor__model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "regressor__model__num_leaves": [15, 31, 63, 127, 255],
        "regressor__model__max_depth": [-1, 4, 6, 8, 10],
        "regressor__model__subsample": [0.6, 0.7, 0.8, 1.0],
        "regressor__model__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
        "regressor__model__min_child_samples": [5, 10, 20, 30, 50],
        "regressor__model__reg_alpha": [0, 0.1, 0.5, 1.0],
        "regressor__model__reg_lambda": [0, 0.1, 0.5, 1.0],
    }
    search = RandomizedSearchCV(
        ttr, grid, n_iter=80, cv=KFOLD5, scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X_train, y_train)
    return report("LightGBM", search)


# ---------------------------------------------------------------------------
# 4. CatBoost
#    CatBoost handles categoricals natively - no need to one-hot/ordinal
#    encode 'sector' etc. Tuned with a hand-rolled randomized search + K-fold
#    loop (NOT RandomizedSearchCV) to avoid the sklearn clone() crash on
#    CatBoostRegressor(cat_features=...) - see fix #7 in the module
#    docstring. Still ends up wrapped in LogTargetCatBoost, which shares the
#    same predict(raw_dataframe) interface as the other two models.
# ---------------------------------------------------------------------------
def tune_catboost(n_iter=60):
    cat_feature_idx = [X_train.columns.get_loc(c) for c in CATEGORICAL_COLS]
    grid = {
        "iterations": [300, 500, 800, 1200, 1600],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 10, 20],
        "subsample": [0.6, 0.8, 1.0],
        "border_count": [32, 64, 128, 254],
    }
    candidates = list(ParameterSampler(grid, n_iter=n_iter, random_state=RANDOM_STATE))

    best_params, best_cv_mae = None, np.inf
    for i, params in enumerate(candidates, 1):
        cand_start = time.time()
        print(f"[CatBoost] candidate {i}/{len(candidates)} starting  params={params}")
        fold_maes = []
        for fold_i, (tr_idx, val_idx) in enumerate(KFOLD5.split(X_train), 1):
            fold_start = time.time()
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model = LogTargetCatBoost(cat_feature_idx, params).fit(X_tr, y_tr)
            fold_maes.append(mean_absolute_error(y_val, model.predict(X_val)))
            print(f"[CatBoost]   candidate {i}/{len(candidates)} fold {fold_i}/5 done "
                  f"in {time.time() - fold_start:.1f}s")
        cv_mae = float(np.mean(fold_maes))
        print(f"[CatBoost] candidate {i}/{len(candidates)} finished "
              f"in {time.time() - cand_start:.1f}s, CV MAE={cv_mae:.4f} Cr")
        if cv_mae < best_cv_mae:
            best_cv_mae, best_params = cv_mae, params

    print(f"\n[CatBoost] best 5-fold CV MAE = {best_cv_mae:.4f} Cr")
    print("[CatBoost] best params:", best_params)

    best = LogTargetCatBoost(cat_feature_idx, best_params).fit(X_train, y_train)

    # 10-fold OOF confirmation on train, computed manually - refits a fresh
    # LogTargetCatBoost per fold instead of cross_val_predict (which would
    # also hit the clone() crash).
    oof_pred = np.empty(len(X_train))
    for tr_idx, val_idx in KFOLD10.split(X_train):
        fold_model = LogTargetCatBoost(cat_feature_idx, best_params).fit(
            X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        )
        oof_pred[val_idx] = fold_model.predict(X_train.iloc[val_idx])
    oof_mae, oof_r2, oof_mape = _metrics(y_train, oof_pred)
    print(f"[CatBoost] 10-fold OOF (train)  -> MAE={oof_mae:.4f} Cr  R2={oof_r2:.4f}  MAPE={oof_mape:.2f}%")

    # Real generalization number: untouched holdout set.
    test_pred = best.predict(X_test)
    test_mae, test_r2, test_mape = _metrics(y_test, test_pred)
    print(f"[CatBoost] HOLDOUT TEST          -> MAE={test_mae:.4f} Cr  R2={test_r2:.4f}  MAPE={test_mape:.2f}%")

    return {
        "pipeline": best,
        "mae": test_mae, "r2": test_r2, "mape": test_mape,
        "oof_mae": oof_mae, "oof_r2": oof_r2, "oof_mape": oof_mape,
    }


# ---------------------------------------------------------------------------
# 5. Compare all models, dump the winner (and every model) for deployment
# ---------------------------------------------------------------------------
def compare_and_select_best(results, output_dir=OUTPUT_DIR):
    """
    results: {name: {"pipeline", "mae", "r2", "mape", "oof_mae", "oof_r2", "oof_mape"}}
    mae/r2/mape here are HOLDOUT-TEST numbers - that's what selection is based on.

    Prints a comparison table, saves every pipeline to disk, and saves the
    best one again as best_model.pkl + metadata.json so a Streamlit app can
    just do `joblib.load('models/best_model.pkl').predict(new_df)`.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"{'Model':<10} {'Holdout MAE':>12} {'Holdout R2':>11} {'Holdout MAPE':>13} "
          f"{'Accuracy*':>10}  {'(OOF MAE)':>10}")
    print("=" * 90)
    for name, r in results.items():
        accuracy = 100 - r["mape"]
        print(f"{name:<10} {r['mae']:>12.4f} {r['r2']:>11.4f} {r['mape']:>12.2f}% "
              f"{accuracy:>9.2f}%  {r['oof_mae']:>10.4f}")
    print("=" * 90)
    print("*There's no classification 'accuracy' for a continuous target - this")
    print(" column is 100 - MAPE, a regression-friendly stand-in. Rank by MAE/R2.")
    print(" (OOF MAE) is the train-only 10-fold number, shown for comparison; it")
    print(" was used for neither final ranking nor selection - holdout MAE was.")

    best_name = min(results, key=lambda n: results[n]["mae"])
    best = results[best_name]
    print(f"\nBest model by HOLDOUT MAE: {best_name}  (MAE={best['mae']:.4f} Cr, "
          f"R2={best['r2']:.4f}, MAPE={best['mape']:.2f}%)")

    for name, r in results.items():
        path = os.path.join(output_dir, f"{name.lower()}_pipeline.pkl")
        joblib.dump(r["pipeline"], path)
        print(f"Saved {name} pipeline -> {path}")

    best_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(best["pipeline"], best_path)

    metadata = {
        "best_model": best_name,
        "test_size": TEST_SIZE,
        "metrics": {
            n: {"holdout_mae": r["mae"], "holdout_r2": r["r2"], "holdout_mape": r["mape"],
                "oof_mae": r["oof_mae"], "oof_r2": r["oof_r2"], "oof_mape": r["oof_mape"]}
            for n, r in results.items()
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest pipeline also saved -> {best_path}")
    print(f"Metadata -> {os.path.join(output_dir, 'metadata.json')}")
    print("\nIn your Streamlit app:")
    print("    import joblib")
    print("    pipe = joblib.load('models/best_model.pkl')")
    print("    pipe.predict(new_data_df)   # new_data_df needs the same raw columns as X,")
    print("                                # price NOT included, and price comes back on")
    print("                                # the real Cr scale (TransformedTargetRegressor")
    print("                                # does the log1p/expm1 round-trip for you).")

    return best_name, best["pipeline"]


if __name__ == "__main__":
    results = {
        "XGBoost": tune_xgboost(),
      "LightGBM": tune_lightgbm(),
       # "CatBoost": tune_catboost(),
    }
    best_name, best_pipeline = compare_and_select_best(results)

    # Optional: stack all three with a Ridge meta-learner once you've got the
    # tuned base models above. All three share the same raw-dataframe
    # interface, so this works without special-casing CatBoost. Fit the stack
    # on X_train/y_train and check it against X_test/y_test same as above.
    #
    # from sklearn.ensemble import StackingRegressor
    # from sklearn.linear_model import RidgeCV
    # stack = StackingRegressor(
    #     estimators=[("xgb", results["XGBoost"]["pipeline"]),
    #                 ("lgbm", results["LightGBM"]["pipeline"]),
    #                 ("cb", results["CatBoost"]["pipeline"])],
    #     final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 20)),
    #     cv=3, n_jobs=-1,
    # )