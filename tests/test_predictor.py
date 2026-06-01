"""
tests/test_predictor.py
========================
Unit and integration tests for HousePricePredictor and the preprocessing pipeline.

Testing approach:
  - Tests that exercise HousePricePredictor load the real model artefacts from
    disk. This validates the full prediction contract against the specific
    trained model, not a stub.
  - Preprocessing tests are pure unit tests with no artefact dependency.
  - Each test function contains a single logical assertion, making failure
    attribution unambiguous.

Execution:
    pytest tests/test_predictor.py -v
    pytest tests/test_predictor.py::test_predictor_loads -v
    pytest tests/test_predictor.py -v -s   # include captured stdout
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ── Path helpers ──────────────────────────────────────────────────────────────
# These paths must point to the artefacts produced by 02_preprocessing.ipynb
# and 03_modelling.ipynb. Tests will be skipped if artefacts are missing.

PROJECT_ROOT        = Path(__file__).resolve().parents[1]
MODEL_PATH          = PROJECT_ROOT / "models"         / "best_model.pkl"
PREPROCESSOR_PATH   = PROJECT_ROOT / "data/processed" / "preprocessor.pkl"
FEATURE_NAMES_PATH  = PROJECT_ROOT / "data/processed" / "feature_names.json"

# ── Skip guard — skip all tests if artefacts not yet produced ─────────────────
ARTEFACTS_EXIST = (
    MODEL_PATH.exists()
    and PREPROCESSOR_PATH.exists()
    and FEATURE_NAMES_PATH.exists()
)
pytestmark = pytest.mark.skipif(
    not ARTEFACTS_EXIST,
    reason="Model artefacts not found — run 02_preprocessing.ipynb and 03_modelling.ipynb first",
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def predictor():
    """
    Load HousePricePredictor once per test module.

    scope="module" amortises the cost of deserialising the model artefact
    across all tests in this file. Without module scope, the artefact would
    be loaded from disk on every test function invocation.
    """
    from src.models.predictor import HousePricePredictor
    return HousePricePredictor.load(
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
    )
    #   )
    from src.models.predictor import HousePricePredictor
    return HousePricePredictor.load(
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
    )

@pytest.fixture
def sample_house() -> dict:
    """
    Representative Ames Housing property expected to predict in the
    $150K–$250K range, consistent with a mid-quality home in the training
    data. Used as the baseline valid input across multiple test cases.
    """
    return {
        "MSSubClass": 20,
        "MSZoning": "RL",
        "LotFrontage": 65.0,
        "LotArea": 8450,
        "Street": "Pave",
        "Alley": "None",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "CollgCr",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "BrkFace",
        "MasVnrArea": 196.0,
        "ExterQual": "Gd",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "Gd",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "GLQ",
        "BsmtFinSF1": 706.0,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0.0,
        "BsmtUnfSF": 150.0,
        "TotalBsmtSF": 856.0,
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "1stFlrSF": 856,
        "2ndFlrSF": 854,
        "LowQualFinSF": 0,
        "GrLivArea": 1710,
        "BsmtFullBath": 1.0,
        "BsmtHalfBath": 0.0,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "KitchenQual": "Gd",
        "TotRmsAbvGrd": 8,
        "Functional": "Typ",
        "Fireplaces": 0,
        "FireplaceQu": "None",
        "GarageType": "Attchd",
        "GarageYrBlt": 2003.0,
        "GarageFinish": "RFn",
        "GarageCars": 2.0,
        "GarageArea": 548.0,
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "WoodDeckSF": 0,
        "OpenPorchSF": 61,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "PoolQC": "None",
        "Fence": "None",
        "MiscFeature": "None",
        "MiscVal": 0,
        "MoSold": 2,
        "YrSold": 2008,
        "SaleType": "WD",
        "SaleCondition": "Normal",
    }


# =============================================================================
# Tests — Predictor loading
# =============================================================================

def test_predictor_loads(predictor):
    """Predictor loads correctly and exposes model_class and n_features."""
    assert predictor is not None
    assert isinstance(predictor.model_class, str) and predictor.model_class
    assert predictor.n_features > 0


def test_predictor_model_class_is_known(predictor):
    """The deployed model must be one of the candidate classes evaluated in the notebook."""
    KNOWN_MODEL_CLASSES = {"Ridge", "Lasso", "ElasticNet", "RandomForestRegressor", "XGBRegressor"}
    assert predictor.model_class in KNOWN_MODEL_CLASSES


def test_feature_names_count_matches_model(predictor):
    """
    The feature name list length must equal the model's expected input
    dimensionality (n_features_in_ for sklearn, n_features for XGBoost).
    """
    if hasattr(predictor.model, "n_features_in_"):
        assert predictor.n_features == predictor.model.n_features_in_
    elif hasattr(predictor.model, "n_features"):
        assert predictor.n_features == predictor.model.n_features


# =============================================================================
# Tests — Single prediction
# =============================================================================

def test_predict_returns_prediction_result(predictor, sample_house):
    """predict() must return a PredictionResult instance."""
    from src.models.predictor import PredictionResult
    result = predictor.predict(sample_house)
    assert isinstance(result, PredictionResult)


def test_predict_price_in_reasonable_range(predictor, sample_house):
    """
    A mid-quality Ames property must predict between $100K and $400K.
    Failure indicates a preprocessing error, incorrect inverse transform,
    or a fundamentally degraded model.
    """
    result = predictor.predict(sample_house)
    assert 100_000 <= result.predicted_price <= 400_000, (
        f"Prediction ${result.predicted_price:,.0f} is outside expected range"
    )


def test_predict_price_is_positive(predictor, sample_house):
    """Predicted price must be strictly positive for any valid input."""
    result = predictor.predict(sample_house)
    assert result.predicted_price > 0, f"Prediction ${result.predicted_price:,.0f} is not positive"


def test_predict_clipped_flag_false_for_normal_input(predictor, sample_house):
    """A mid-quality property must not trigger the price clipping guard."""
    result = predictor.predict(sample_house)
    assert result.clipped is False, "Clipping flag should be False for normal input"


def test_predict_clipped_flag_true_for_absurd_input(predictor, sample_house):
    """
    An extreme high-end input (OverallQual=10, GrLivArea=10,000 sqft) must
    produce a raw prediction above MAX_PRICE, triggering the clipping guard.
    """
    # Ridge regression shrinks toward the mean, so very low-quality inputs
    # may not breach MIN_PRICE. The upper bound is used instead as the
    # reliable clipping trigger.
    sample_house_absurd = sample_house.copy()
    sample_house_absurd["OverallQual"] = 10
    sample_house_absurd["GrLivArea"] = 10_000
    sample_house_absurd["TotalBsmtSF"] = 5_000
    sample_house_absurd["YearBuilt"] = 2010
    result = predictor.predict(sample_house_absurd)
    assert result.clipped is True, "Clipping flag should be True for absurd high-end input"


# =============================================================================
# Tests — Preprocessing module (unit tests without the model)
# =============================================================================

def test_add_features_creates_expected_columns():
    """
    add_features() must produce exactly the five engineered columns defined
    in the preprocessing specification: TotalSF, TotalBath, HouseAge,
    RemodAge, IsRemodeled. No artefacts required.
    """
    import pandas as pd
    from src.data.preprocessing import add_features
    df = pd.DataFrame([{
        "TotalBsmtSF": 800, "1stFlrSF": 1000, "2ndFlrSF": 800,
        "FullBath": 2, "HalfBath": 1, "BsmtFullBath": 1, "BsmtHalfBath": 0,
        "YrSold": 2010, "YearBuilt": 2000, "YearRemodAdd": 2005,
    }])
    result = add_features(df)
    expected_cols = {"TotalSF", "TotalBath", "HouseAge", "RemodAge", "IsRemodeled"}
    assert expected_cols.issubset(result.columns), f"Expected columns {expected_cols} not all found in result columns {set(result.columns)}"    


def test_add_features_does_not_mutate_input():
    """add_features() must satisfy the pure function contract: input DataFrame is unchanged."""
    import pandas as pd
    from src.data.preprocessing import add_features
    df_original = pd.DataFrame([{
        "TotalBsmtSF": 800, "1stFlrSF": 1000, "2ndFlrSF": 800,
        "FullBath": 2, "HalfBath": 1, "BsmtFullBath": 1, "BsmtHalfBath": 0,
        "YrSold": 2010, "YearBuilt": 2000, "YearRemodAdd": 2005,
    }])
    df = df_original.copy()
    _ = add_features(df)
    assert df.equals(df_original), "add_features() mutated its input!"


def test_apply_quality_encoding_maps_correctly():
    """
    apply_quality_encoding() must map "TA" → 3 for all QUALITY_COLS,
    consistent with the QUALITY_MAP ordinal scale.
    """
    import pandas as pd
    from src.data.preprocessing import QUALITY_COLS, apply_quality_encoding
    df = pd.DataFrame([{col: "TA" for col in QUALITY_COLS}])
    result = apply_quality_encoding(df)
    assert result["ExterQual"].iloc[0] == 3
    assert result["KitchenQual"].iloc[0] == 3
    assert result["BsmtQual"].iloc[0] == 3
    assert result["FireplaceQu"].iloc[0] == 3
    assert result["GarageQual"].iloc[0] == 3


def test_apply_quality_encoding_raises_on_unknown_category():
    """
    apply_quality_encoding() must raise ValueError for an unmapped category.
    Silent NaN coercion would obscure upstream data quality issues.
    """
    import pandas as pd
    from src.data.preprocessing import QUALITY_COLS, apply_quality_encoding
    with pytest.raises(ValueError, match="unmapped"):
        df_bad = pd.DataFrame([{col: "BAD_VALUE" for col in QUALITY_COLS}])
        apply_quality_encoding(df_bad)


# =============================================================================
# Tests — Batch prediction
# =============================================================================

def test_predict_batch_returns_correct_count(predictor, sample_house):
    """predict_batch() with N inputs must return exactly N PredictionResults."""
    batch = [sample_house, sample_house, sample_house]
    results = predictor.predict_batch(batch)
    assert len(results) == 3


def test_predict_batch_matches_single_predict(predictor, sample_house):
    """
    predict_batch([house]) must return the same price as predict(house).
    Regression test against divergence between the single and batch code paths
    (e.g., different preprocessing calls or model invocations).
    """
    single = predictor.predict(sample_house)
    batch = predictor.predict_batch([sample_house])
    assert batch[0].predicted_price == pytest.approx(single.predicted_price, rel=1e-5)
