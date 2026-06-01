"""
tests/test_api.py
=================
Integration tests for the FastAPI endpoints.

Tests start the full FastAPI application with the real model loaded and
exercise each endpoint through the ASGI test client. The full request-response
cycle is covered: JSON parsing, Pydantic validation, prediction, and response
serialisation.

Test scope:
  - test_predictor.py covers individual functions in isolation.
  - This file covers the HTTP layer and full pipeline end-to-end.
    Edge cases already verified by unit tests are not duplicated here.

All tests use the synchronous TestClient. The context-manager form
(`with TestClient(app) as c`) triggers the FastAPI lifespan, loading model
artefacts into app.state before any test executes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib import response

import pytest
from fastapi.testclient import TestClient

# ── Skip guard ────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parents[1]
MODEL_PATH        = PROJECT_ROOT / "models"         / "best_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "data/processed" / "preprocessor.pkl"

ARTEFACTS_EXIST = MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()
pytestmark = pytest.mark.skipif(
    not ARTEFACTS_EXIST,
    reason="Model artefacts not found — run the training notebooks first",
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def client():
    """
    Initialise a TestClient for the FastAPI application once per test module.

    scope="module" ensures model artefacts are loaded from disk once and
    shared across all tests in this file. The context-manager form triggers
    the application lifespan, populating app.state.predictor before the
    first test executes.
    """
    from src.api.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    """
    Complete, valid prediction request body representing a mid-quality
    Ames property expected to predict in the $150K–$250K range.
    Mirrors sample_house in test_predictor.py; both fixtures should remain
    synchronised to avoid cross-file test divergence.
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
# Health endpoint tests
# =============================================================================

def test_health_returns_200(client):
    """GET /health must return HTTP 200 when the service is operational."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_model_loaded_true(client):
    """The health response body must confirm model_loaded=True."""
    response = client.get("/health")
    body = response.json()
    assert body["model_loaded"] is True


def test_health_response_has_required_fields(client):
    """The health response body must contain status, model_loaded, and api_version."""
    response = client.get("/health")
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "api_version" in body


# =============================================================================
# /predict endpoint — success cases
# =============================================================================

def test_predict_returns_200_for_valid_payload(client, valid_payload):
    """POST /predict with a valid payload must return HTTP 200."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_predict_response_contains_predicted_price(client, valid_payload):
    """The response body must contain a 'predicted_price' key."""
    response = client.post("/predict", json=valid_payload)
    body = response.json()
    assert "predicted_price" in body


def test_predict_price_is_positive(client, valid_payload):
    """Predicted price must be strictly positive for any valid input."""
    response = client.post("/predict", json=valid_payload)
    body = response.json()
    assert body["predicted_price"] > 0


def test_predict_price_in_reasonable_range(client, valid_payload):
    """A mid-quality property must predict between $100K and $400K."""
    response = client.post("/predict", json=valid_payload)
    body = response.json()
    assert 100_000 <= body["predicted_price"] <= 400_000


def test_predict_response_has_model_class(client, valid_payload):
    """Response must identify the model class that produced the prediction."""
    response = client.post("/predict", json=valid_payload)
    body = response.json()
    assert "model_class" in body
    assert isinstance(body["model_class"], str)


def test_predict_response_has_process_time_header(client, valid_payload):
    """Every response must carry the X-Process-Time header injected by the timing middleware."""
    response = client.post("/predict", json=valid_payload)
    assert "X-Process-Time" in response.headers


# =============================================================================
# /predict endpoint — validation error cases
# =============================================================================

def test_predict_returns_422_for_empty_body(client):
    """
    POST /predict with an empty JSON body must return HTTP 422.

    FastAPI returns 422 (Unprocessable Entity) when Pydantic validation fails,
    distinguishing schema-invalid content from a malformed HTTP request (400).
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_returns_422_for_invalid_overall_qual(client, valid_payload):
    """OverallQual=11 must fail the ge=1, le=10 field constraint with HTTP 422."""
    payload = {**valid_payload, "OverallQual": 11}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_returns_422_for_invalid_quality_string(client, valid_payload):
    """An unrecognised quality string (e.g., ExterQual='PERFECT') must return HTTP 422."""
    payload = {**valid_payload, "ExterQual": "PERFECT"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_error_response_does_not_expose_stack_trace(client):
    """
    Error responses must not include Python stack traces or internal file
    system paths (OWASP A05: Security Misconfiguration). Exposing internals
    reveals application architecture to potential attackers.
    """
    response = client.post("/predict", json={})
    body_str = response.text
    assert "Traceback" not in body_str
    assert "/home/" not in body_str
    assert "src/" not in body_str


# =============================================================================
# /predict/batch endpoint
# =============================================================================

def test_batch_predict_returns_correct_count(client, valid_payload):
    """POST /predict/batch with 3 items must return a response with count=3 and 3 predictions."""
    payload = {"requests": [valid_payload, valid_payload, valid_payload]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3


def test_batch_predict_returns_422_for_empty_list(client):
    """An empty requests list must fail the min_length=1 constraint with HTTP 422."""
    payload = {"requests": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_batch_predict_returns_422_for_oversized_batch(client, valid_payload):
    """A batch of 101 items must fail the max_length=100 constraint with HTTP 422."""
    payload = {"requests": [valid_payload] * 101}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


# =============================================================================
# Edge case — idempotency
# =============================================================================

def test_predict_is_deterministic(client, valid_payload):
    """
    Identical inputs must produce identical predictions across successive
    requests. Non-determinism in inference invalidates A/B test results
    and produces inconsistent quotes for end users.
    """
    r1 = client.post("/predict", json=valid_payload).json()
    r2 = client.post("/predict", json=valid_payload).json()
    assert r1["predicted_price"] == r2["predicted_price"]
