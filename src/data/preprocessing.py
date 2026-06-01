"""
src/data/preprocessing.py
==========================
Production preprocessing module for the Real Estate Price Engine.

This module is the authoritative implementation of every transformation defined
in 02_preprocessing.ipynb. It operates in two contexts:

  Training pipeline (src/models/train.py):
    preprocessor.fit_transform(X_train)

  Serving layer (src/api/app.py):
    preprocess_for_inference(raw_df, preprocessor)

The serving path calls preprocessor.transform() exclusively. Invoking
fit_transform() at inference time recomputes statistics from a single
sample, producing invalid scalings and encodings inconsistent with the
trained model's expectations.

Versioning convention: pair each preprocessor artefact with the model it was
trained alongside (e.g., preprocessor_v2.pkl ↔ model_v2.pkl) to prevent
silent artefact mismatch following a retrain cycle.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Constants — must match PREP_CFG in 02_preprocessing.ipynb EXACTLY
# =============================================================================

#: Hand-crafted ordinal map for quality/condition features.
#: "None" (imputed sentinel for properties lacking the feature) = 0.
#: Do NOT use sklearn OrdinalEncoder — it sorts alphabetically (Ex=0, not 5).
QUALITY_MAP: dict[str, int] = {
    "None": 0,
    "Po":   1,
    "Fa":   2,
    "TA":   3,
    "Gd":   4,
    "Ex":   5,
}

QUALITY_COLS: list[str] = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond",
]

FILL_NONE_COLS: list[str] = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

FILL_ZERO_COLS: list[str] = [
    "MasVnrArea", "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
]

FILL_MODE_COLS: list[str] = [
    "Electrical", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType",
]

DROP_COLS: list[str] = ["Id", "Utilities"]

OUTLIER_INDICES: list[int] = [523, 1298]


# =============================================================================
# Step functions — each function is pure (returns a copy, no in-place mutation)
# =============================================================================

def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that carry no predictive signal: row identifiers and
    near-zero-variance constants.

    errors="ignore" ensures safe operation when columns are absent, as
    occurs when the inference request payload omits the 'Id' column.

    Parameters
    ----------
    df : Raw input DataFrame.

    Returns
    -------
    DataFrame with DROP_COLS removed.
    """
    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df.copy()


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the two domain-confirmed outliers identified in EDA: large properties
    with anomalously low SalePrice consistent with non-market transactions.

    This function is invoked exclusively during training. The index is reset
    after removal to eliminate gaps that cause iloc/loc misalignment in
    downstream pipeline stages.

    Parameters
    ----------
    df : DataFrame prior to target extraction.

    Returns
    -------
    DataFrame with OUTLIER_INDICES removed and a contiguous integer index.
    """
    df = df.drop(index=OUTLIER_INDICES, errors="ignore")
    df = df.reset_index(drop=True)
    return df


def impute_lot_frontage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute LotFrontage using the per-neighbourhood median rather than the
    global median. Properties within the same neighbourhood share similar
    street-frontage characteristics; global median imputation discards this
    spatial signal.

    Strategy:
      1. Fill NaN with the per-Neighborhood group median.
      2. Apply the global median as a fallback for neighbourhoods where all
         LotFrontage values are missing.

    pandas transform() is used in preference to agg() because it returns a
    Series with the same index as the source frame, enabling direct assignment
    without index realignment. The group medians computed here apply only to
    the raw feature frame prior to ColumnTransformer ingestion.

    Parameters
    ----------
    df : DataFrame containing 'Neighborhood' and 'LotFrontage' columns.

    Returns
    -------
    DataFrame with LotFrontage NaNs filled.
    """
    df = df.copy()
    # At inference time the caller may omit LotFrontage entirely (None excluded
    # from the feature dict). Treat an absent column identically to a NaN value
    # so the neighbourhood-median imputation path handles it correctly.
    if "LotFrontage" not in df.columns:
        df["LotFrontage"] = np.nan
    group_median = df.groupby("Neighborhood")["LotFrontage"].transform("median")
    df["LotFrontage"] = df["LotFrontage"].fillna(group_median)
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    return df


def run_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply structurally-driven imputation strategies for columns whose
    missingness is domain-meaningful rather than random:

      - fill_none_cols  → "None"  (property does not possess the feature)
      - fill_zero_cols  → 0       (no area or count; zero is the correct value)
      - fill_mode_cols  → mode    (missing categorical imputed from most frequent class)

    This function must execute before add_features() because the engineered
    features TotalSF and TotalBath depend on columns in fill_zero_cols.

    Parameters
    ----------
    df : DataFrame after drop_identifiers() and impute_lot_frontage().

    Returns
    -------
    DataFrame with all structural null values resolved.
    """
    df = df.copy()
    for col in FILL_NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    for col in FILL_ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in FILL_MODE_COLS:
        if col in df.columns:
            mode_series = df[col].mode()
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series.iloc[0])
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        still_null = df.columns[df.isnull().any()].tolist()
        logger.debug("Nulls remain after domain imputation (ColumnTransformer will resolve): %s", still_null)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive engineered features validated in 01_eda.ipynb Section 7.

    Feature definitions must remain identical to those applied during training.
    Any deviation in formula or column naming introduces train/serve skew —
    the model was calibrated on one feature representation; the serving layer
    must apply exactly the same transformation.

    Derived features:
      TotalSF      : aggregate above-grade and basement square footage
      TotalBath    : weighted bathroom count (full bath = 1.0, half bath = 0.5)
      HouseAge     : elapsed years between original construction and sale date
      RemodAge     : elapsed years since last remodel, clipped to a minimum of 0
      IsRemodeled  : binary indicator; 1 when YearRemodAdd differs from YearBuilt

    Parameters
    ----------
    df : DataFrame after run_imputation().

    Returns
    -------
    DataFrame with 5 additional engineered columns.
    """
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (
        df["FullBath"].fillna(0) + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0) + 0.5 * df["BsmtHalfBath"].fillna(0)
    )
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = np.maximum(0, df["YrSold"] - df["YearRemodAdd"])
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    return df


def apply_quality_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map quality/condition string columns to domain-ordered integers using QUALITY_MAP.

    sklearn OrdinalEncoder is intentionally avoided here. It assigns ranks
    alphabetically (Ex=0, Fa=1, Gd=2, Po=3, TA=4), which inverts the true
    domain ordering. QUALITY_MAP encodes the correct ordinal scale:
    None=0 < Po=1 < Fa=2 < TA=3 < Gd=4 < Ex=5.

    Raises
    ------
    ValueError
        Raised when a column contains a value absent from QUALITY_MAP, which
        signals a new category requiring an explicit QUALITY_MAP extension.
        Silent NaN imputation is deliberately not applied — it would mask an
        upstream data quality issue.

    Parameters
    ----------
    df : DataFrame after add_features().

    Returns
    -------
    DataFrame with QUALITY_COLS replaced by int64 columns.
    """
    df = df.copy()
    for col in QUALITY_COLS:
        if col not in df.columns:
            continue  # column absent at inference - skip safety
        df[col] = df[col].map(QUALITY_MAP)
        if df[col].isnull().any():
            unmapped = df[col][df[col].isnull()].index.tolist()
            raise ValueError(
                f"Column '{col}' has unmapped values at rows {unmapped}."
                f"Extend QUALITY_MAP to include these categories."
            )
    return df.infer_objects()  # coerce object columns to int64 where possible


# =============================================================================
# Inference entry point
# =============================================================================

def preprocess_for_inference(
    raw_df: pd.DataFrame,
    preprocessor,  # sklearn ColumnTransformer, fitted on training data
) -> np.ndarray:
    """
    Execute the full feature engineering and encoding pipeline on a raw input
    DataFrame and return the encoded feature matrix ready for model.predict().

    Replicates the training notebook's preprocessing sequence using the
    fitted preprocessor's .transform() method. fit_transform() must never
    be called here — doing so would refit statistics on the inference sample.

    Pipeline stages:
      1. drop_identifiers       — remove Id, Utilities if present
      2. impute_lot_frontage    — neighbourhood-grouped median fill
      3. run_imputation         — domain-driven None/0/mode fills
      4. add_features           — derive TotalSF, TotalBath, HouseAge, RemodAge, IsRemodeled
      5. apply_quality_encoding — map string quality columns to int64
      6. preprocessor.transform — scale numerics, one-hot encode categoricals

    Per-stage debug logging is emitted at DEBUG level to facilitate diagnosis
    of unexpected null values or column mismatches in production.

    Parameters
    ----------
    raw_df       : DataFrame with original Ames Housing column names.
                   Supports single-row inference requests.
    preprocessor : Fitted sklearn ColumnTransformer loaded from preprocessor.pkl.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features) ready for model.predict().

    Raises
    ------
    ValueError   If quality encoding encounters an unmapped category value.
    RuntimeError If the preprocessor has not been fitted.
    """
    df = drop_identifiers(raw_df)
    logger.debug("After drop_identifiers: shape=%s, nulls=%d", df.shape, df.isnull().sum().sum())

    # Expand partial inference requests to the full 83-column input space.
    # The API schema exposes only the most influential features; absent columns
    # are added here as NaN so that run_imputation() applies domain-specific
    # defaults (FILL_NONE_COLS, FILL_ZERO_COLS, FILL_MODE_COLS) and the
    # ColumnTransformer's internal SimpleImputer resolves any remainder.
    if hasattr(preprocessor, "feature_names_in_"):
        expected = list(preprocessor.feature_names_in_)
        missing = [c for c in expected if c not in df.columns]
        if missing:
            df = df.copy()
            for col in missing:
                df[col] = np.nan
            logger.debug("Padded %d missing columns with NaN for inference", len(missing))

    df = impute_lot_frontage(df)
    logger.debug("After impute_lot_frontage: shape=%s, nulls=%d", df.shape, df.isnull().sum().sum())
    
    df = run_imputation(df)
    logger.debug("After run_imputation: shape=%s, nulls=%d", df.shape, df.isnull().sum().sum())
    
    df = add_features(df)
    logger.debug("After add_features: shape=%s, nulls=%d", df.shape, df.isnull().sum().sum())
    
    df = apply_quality_encoding(df)
    logger.debug("After apply_quality_encoding: shape=%s, nulls=%d", df.shape, df.isnull().sum().sum())
    
    X_enc = preprocessor.transform(df)
    return X_enc
