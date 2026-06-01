"""
src/models/train.py
====================
Production training script for the Real Estate Price Engine.

This script is the entry point for `make train`. It replicates the training
pipeline defined in the notebooks (02_preprocessing.ipynb, 03_modelling.ipynb)
as runnable Python code — no notebook kernel required.

Usage:
    python -m src.models.train                        # use defaults from config/
    python -m src.models.train --data data/ames_housing.csv
    python -m src.models.train --alpha 0.5

The script:
  1. Loads the raw CSV from data/
  2. Runs the full preprocessing pipeline (drop_identifiers → imputation →
     feature engineering → quality encoding → ColumnTransformer)
  3. Trains a Ridge regression model on log1p(SalePrice)
  4. Evaluates on the held-out test split
  5. Saves best_model.pkl, preprocessor.pkl, feature_names.json, model_metadata.json
  6. Exits non-zero if the model misses the quality gates in config/settings.yaml

A training script that exits with a non-zero exit code on quality-gate
failure integrates naturally with CI/CD pipelines (GitHub Actions, Jenkins, etc.).
The pipeline step "run train.py" fails if the model is not good enough — preventing
a degraded model from being deployed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocessing import (
    add_features,
    apply_quality_encoding,
    drop_identifiers,
    impute_lot_frontage,
    remove_outliers,
    run_imputation,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Project root (resolves correctly regardless of CWD) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Quality gates (fallbacks if YAML not available) ──────────────────────────
DEFAULT_MAX_RMSE = 0.14
DEFAULT_MIN_R2 = 0.85


# =============================================================================
# Helpers
# =============================================================================

def load_config() -> dict:
    """Load config/settings.yaml; return empty dict if not found."""
    try:
        import yaml
        cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# =============================================================================
# Preprocessing pipeline builder
# =============================================================================

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Build and fit a ColumnTransformer on the training set.

    Numeric columns  → StandardScaler (zero mean, unit variance)
    Categorical cols → OneHotEncoder  (handle_unknown="ignore" for robustness)

    The fitted transformer is returned so it can be serialised alongside
    the model. At inference, only .transform() is ever called on it.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(
        "Building ColumnTransformer: %d numeric, %d categorical",
        len(numeric_cols), len(categorical_cols),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    preprocessor.fit(X_train)
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names after fitting the ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())


# =============================================================================
# Full preprocessing pipeline (training mode)
# =============================================================================

def run_training_pipeline(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list[str]]:
    """
    Apply the complete training-time preprocessing sequence and return
    train/val/test splits ready for model.fit().

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names
    """
    # Step 1 — Remove outliers (training only)
    df = remove_outliers(df)
    logger.info("After outlier removal: %d rows", len(df))

    # Step 2 — Extract target (log1p for normality)
    y = np.log1p(df["SalePrice"].values)
    df = df.drop(columns=["SalePrice"])

    # Step 3 — Feature engineering pipeline
    df = drop_identifiers(df)
    df = impute_lot_frontage(df)
    df = run_imputation(df)
    df = add_features(df)
    df = apply_quality_encoding(df)
    logger.info("After preprocessing: shape=%s, nulls=%d", df.shape, int(df.isnull().sum().sum()))

    # Step 4 — Train / val / test split (60 / 20 / 20)
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        df, y, test_size=0.40, random_state=42
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # Step 5 — Fit ColumnTransformer on training set only
    preprocessor = build_preprocessor(X_train_raw)
    feature_names = get_feature_names(preprocessor)

    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    logger.info(
        "Splits — train: %d, val: %d, test: %d | features: %d",
        len(y_train), len(y_val), len(y_test), X_train.shape[1],
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names


# =============================================================================
# Training
# =============================================================================

def train_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 10.0) -> Ridge:
    """Train Ridge regression with the given regularisation strength."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    logger.info("Ridge(alpha=%.1f) fitted on %d samples", alpha, len(y_train))
    return model


# =============================================================================
# Saving artefacts
# =============================================================================

def save_artefacts(
    model: Ridge,
    preprocessor: ColumnTransformer,
    feature_names: list[str],
    metadata: dict,
    cfg: dict,
) -> None:
    """Serialise model, preprocessor, feature names, and metadata to disk."""
    paths_cfg = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})

    models_dir = PROJECT_ROOT / paths_cfg.get("models_dir", "models")
    processed_dir = PROJECT_ROOT / paths_cfg.get("processed_dir", "data/processed")
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / model_cfg.get("model_file", "best_model.pkl")
    preprocessor_path = processed_dir / model_cfg.get("preprocessor_file", "preprocessor.pkl")
    feature_names_path = processed_dir / model_cfg.get("feature_names_file", "feature_names.json")
    metadata_path = models_dir / model_cfg.get("metadata_file", "model_metadata.json")

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model        → %s", model_path)
    logger.info("Saved preprocessor → %s", preprocessor_path)
    logger.info("Saved feature names → %s", feature_names_path)
    logger.info("Saved metadata     → %s", metadata_path)


# =============================================================================
# Quality gate
# =============================================================================

def check_quality_gates(test_rmse: float, test_r2: float, cfg: dict) -> bool:
    """
    Return True if the model meets the deployment thresholds from config.
    Return False (and log the failure) if it does not.
    """
    model_cfg = cfg.get("model", {})
    max_rmse = model_cfg.get("threshold_rmse_log", DEFAULT_MAX_RMSE)
    min_r2 = model_cfg.get("threshold_r2", DEFAULT_MIN_R2)

    passed = True
    if test_rmse > max_rmse:
        logger.error(
            "QUALITY GATE FAILED: test RMSE %.4f > threshold %.4f",
            test_rmse, max_rmse,
        )
        passed = False
    if test_r2 < min_r2:
        logger.error(
            "QUALITY GATE FAILED: test R2 %.4f < threshold %.4f",
            test_r2, min_r2,
        )
        passed = False

    if passed:
        logger.info(
            "Quality gates passed: RMSE=%.4f (<=%.4f), R2=%.4f (>=%.4f)",
            test_rmse, max_rmse, test_r2, min_r2,
        )
    return passed


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Real Estate Price Engine model.")
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to raw CSV. Defaults to the value in config/settings.yaml.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Ridge regularisation strength (default: 10.0).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()

    # Resolve data path
    paths_cfg = cfg.get("paths", {})
    raw_data_rel = paths_cfg.get("raw_data", "data/ames_housing.csv")
    data_path = Path(args.data) if args.data else PROJECT_ROOT / raw_data_rel

    if not data_path.exists():
        logger.error("Raw data not found at %s", data_path)
        return 1

    logger.info("Loading raw data from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows x %d columns", *df.shape)

    # Run full pipeline
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names = (
        run_training_pipeline(df)
    )

    # Train
    model = train_ridge(X_train, y_train, alpha=args.alpha)

    # Evaluate
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_rmse = rmse(y_val, val_preds)
    test_rmse_log = rmse(y_test, test_preds)
    test_r2 = float(r2_score(y_test, test_preds))
    test_mae_dollars = float(mean_absolute_error(np.expm1(y_test), np.expm1(test_preds)))

    logger.info(
        "Results — val RMSE: %.4f | test RMSE: %.4f | test R2: %.4f | test MAE $%.0f",
        val_rmse, test_rmse_log, test_r2, test_mae_dollars,
    )

    metadata = {
        "model_class": type(model).__name__,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse_log,
        "test_r2": test_r2,
        "test_mae_dollars": test_mae_dollars,
        "n_features": len(feature_names),
        "trained on": data_path.name,
    }

    # Check quality gates before saving
    if not check_quality_gates(test_rmse_log, test_r2, cfg):
        logger.error("Model did not meet quality gates. Artefacts NOT saved.")
        return 1

    # Save
    save_artefacts(model, preprocessor, feature_names, metadata, cfg)
    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
