"""
model_training.py
Trains an XGBoost classifier to predict HR probability per player-game.
Handles class imbalance, cross-validation, calibration, and feature importance.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_engineering import get_feature_columns


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/hr_model.pkl"
FEATURE_IMPORTANCE_PATH = "models/feature_importance.png"

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


# ── Data Prep ─────────────────────────────────────────────────────────────────

def prepare_training_data(features_df: pd.DataFrame):
    """
    Split features and target. Handle missing values.
    Returns X, y, groups (for grouped CV by player).
    """
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in features_df.columns]

    missing = set(feature_cols) - set(available_cols)
    if missing:
        print(f"  Warning: Missing feature columns: {missing}")

    X = features_df[available_cols].copy()

    # Fill remaining NaNs with column medians
    X = X.fillna(X.median())

    y = features_df["hr"].astype(int)
    groups = features_df["player_id"].astype(str)  # for grouped CV

    print(f"  Training samples: {len(X)}")
    print(f"  HR rate (positive class): {y.mean():.3f}")
    print(f"  Features used: {len(available_cols)}")

    return X, y, groups, available_cols


# ── Class Imbalance ───────────────────────────────────────────────────────────

def compute_scale_pos_weight(y: pd.Series) -> float:
    """XGBoost scale_pos_weight to handle HR rarity (~5-8% of games)."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    ratio = neg / pos
    print(f"  scale_pos_weight: {ratio:.2f} ({pos} HRs, {neg} non-HRs)")
    return ratio


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """
    Train XGBoost with grouped cross-validation (player as group).
    Returns calibrated model + CV scores.
    """
    os.makedirs("models", exist_ok=True)

    scale_pw = compute_scale_pos_weight(y)

    base_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pw,
        **XGB_PARAMS
    )

    # Calibrate probabilities using isotonic regression
    # This ensures the predicted 8% HR probability actually means 8% of the time
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv=3
    )

    # Grouped K-Fold: same player stays in same fold to prevent data leakage
    gkf = StratifiedGroupKFold(n_splits=5)

    print("\nRunning 5-fold grouped cross-validation...")
    cv_aucs = []
    cv_briers = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = CalibratedClassifierCV(
            xgb.XGBClassifier(scale_pos_weight=scale_pw, **XGB_PARAMS),
            method="isotonic", cv=3
        )
        fold_model.fit(X_train, y_train)
        preds = fold_model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, preds)
        brier = brier_score_loss(y_val, preds)
        cv_aucs.append(auc)
        cv_briers.append(brier)
        print(f"  Fold {fold+1}: AUC={auc:.4f}, Brier={brier:.4f}")

    print(f"\n  Mean AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    print(f"  Mean Brier: {np.mean(cv_briers):.4f} ± {np.std(cv_briers):.4f}")

    # Final model on all data
    print("\nTraining final model on full dataset...")
    calibrated_model.fit(X, y)

    return calibrated_model, {"cv_auc": cv_aucs, "cv_brier": cv_briers}


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list):
    """Extract and plot feature importances from the XGBoost base estimator."""
    try:
        # Dig through calibrated wrapper to get base XGB model
        base_xgb = model.calibrated_classifiers_[0].base_estimator
        importances = base_xgb.feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=True).tail(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df["feature"], importance_df["importance"], color="#1f77b4")
        ax.set_title("XGBoost Feature Importance (Top 20)", fontsize=14)
        ax.set_xlabel("Importance Score")
        ax.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Feature importance chart saved to {FEATURE_IMPORTANCE_PATH}")

        print("\nTop 10 Features:")
        for _, row in importance_df.tail(10).iloc[::-1].iterrows():
            print(f"  {row['feature']:<35} {row['importance']:.4f}")

    except Exception as e:
        print(f"Could not plot feature importance: {e}")


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model, feature_cols: list):
    """Save model + feature column list to disk."""
    bundle = {"model": model, "feature_cols": feature_cols}
    joblib.dump(bundle, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    """Load model bundle from disk."""
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    features_path = "data/features.csv"

    if not os.path.exists(features_path):
        print("Run feature_engineering.py first to generate features.csv")
    else:
        print("Loading feature matrix...")
        df = pd.read_csv(features_path)

        X, y, groups, feature_cols = prepare_training_data(df)
        model, cv_scores = train_model(X, y, groups)
        plot_feature_importance(model, feature_cols)
        save_model(model, feature_cols)

        print("\n✓ Model training complete.")
        print(f"  Final model AUC: {np.mean(cv_scores['cv_auc']):.4f}")
