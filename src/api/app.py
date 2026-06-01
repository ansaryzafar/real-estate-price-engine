"""
src/api/app.py
==============
FastAPI application for the Real Estate Price Engine.

Entry point referenced in Makefile:
    uvicorn src.api.app:app --reload --port 8000

ARCHITECTURE NOTES:
  - The model is loaded ONCE at startup via the @asynccontextmanager lifespan
    and stored in app.state.predictor. Endpoint handlers access it from there.
  - Pydantic v2 automatically validates all request/response bodies.
  - The /predict endpoint returns 422 (Unprocessable Entity) automatically for
    any validation error — you do not need to write manual input checking.
  - CORS middleware is enabled for local front-end development.

Mutable shared state is stored in app.state rather than module-level
variables.

TO RUN:
    # from project root, with .venv activated
    uvicorn src.api.app:app --reload --port 8000

    # then test:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"OverallQual": 7, "GrLivArea": 1500, ...}'
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.models.predictor import HousePricePredictor

# ── Logging ───────────────────────────────────────────────────────────────────
# Configured once at the application entry point. All subordinate modules
# obtain a logger via logging.getLogger(__name__) and inherit this handler.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Artefact paths ────────────────────────────────────────────────────────────
# Resolved relative to the project root (two levels above this file).
# The working directory must be the project root when uvicorn is launched.
BASE_DIR = Path(__file__).resolve().parents[2]   # project root
MODEL_PATH          = BASE_DIR / "models"         / "best_model.pkl"
PREPROCESSOR_PATH   = BASE_DIR / "data/processed" / "preprocessor.pkl"
FEATURE_NAMES_PATH  = BASE_DIR / "data/processed" / "feature_names.json"

# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
# The lifespan context manager supersedes the deprecated @app.on_event("startup")
# pattern. It executes before the first request is accepted and tears down after
# the last response is sent. Objects stored in app.state here are available to
# all endpoint handlers throughout the process lifetime.
#
# Heavy objects (models, database connections, HTTP clients) are loaded here
# rather than at module scope. Module-scope loading executes at import time,
# which interferes with test fixtures that mock the model and can trigger
# circular import issues in certain configurations.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load artefacts on startup, release resources on shutdown."""
    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info("Starting up — loading model artefacts...")

    # A startup failure re-raises the exception, terminating the process with
    # a non-zero exit code. Silently continuing with a missing model would
    # serve corrupt or absent predictions without any observable failure signal.
    try:
        app.state.predictor = HousePricePredictor.load(
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
            feature_names_path=FEATURE_NAMES_PATH,
        )
    except Exception as exc:
        logger.critical("Failed to load model: %s", exc)
        raise
    logger.info(
        "Model loaded: %s (%d features)",
        app.state.predictor.model_class,
        app.state.predictor.n_features,
    )
    
    

    yield   # application runs here; everything after yield executes on shutdown

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    # Release any held resources (database connections, HTTP clients, etc.).
    # For this service the model is the only held resource and is released
    # automatically by the process on exit.
    db_connections = getattr(app.state, "db_connections", None)
    if db_connections:
        for conn in db_connections:
            try:
                conn.close()
                logger.info("Closed DB connection: %s", conn)
            except Exception as e:
                logger.warning("Failed to close DB connection %s: %s", conn, e)
    
    logger.info("Shutting down — cleanup complete.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Real Estate Price Prediction API",
    description=(
        "Predicts residential sale price from property features. "
        "Trained on the Ames Housing dataset (2006–2010, Ames, Iowa)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Disable /openapi.json and /docs in production deployments to reduce
    # the attack surface exposed to untrusted networks.
    # openapi_url=None,
    # docs_url=None,
    openapi_url="/openapi.json",
    docs_url="/docs",
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# Required for browser-based frontends that call this API from a different
# origin. In production, restrict allow_origins to the specific frontend
# domain(s) to prevent unintended cross-origin access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Request timing middleware ─────────────────────────────────────────────────
# Appends an X-Process-Time header (milliseconds) to every response.
# Provides a lightweight signal for latency monitoring and SLA baseline
# measurement without requiring a dedicated APM agent.

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Attach X-Process-Time (ms) header to every response."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
    return response


# =============================================================================
# Endpoints
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["ops"],
)
async def health(request: Request) -> HealthResponse:
    """
    Return service health status.

    Consumed by load balancers and container orchestrator liveness/readiness
    probes. The response validates that the model artefact is loaded and the
    predictor is operational, not merely that the process is alive.
    """
    predictor: HousePricePredictor = request.app.state.predictor
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_class=predictor.model_class,
        n_features=predictor.n_features,
        api_version=app.version,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict house sale price",
    tags=["prediction"],
)
async def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    """
    Predict the sale price for a single residential property.

    The request body is deserialised and validated by Pydantic before this
    handler executes. Validation failures are returned as HTTP 422 with
    structured field-level error details; no manual input validation is
    required in the handler body.

    Internal error details (stack traces, file paths) are logged server-side
    and are not propagated to the response body.

    Raises
    ------
    HTTPException(500) on unexpected prediction failure.
    """
    predictor: HousePricePredictor = request.app.state.predictor

    try:
        result = predictor.predict(payload.to_feature_dict())
    except Exception as e:
        logger.exception("Prediction failed for payload: %s", payload.model_dump())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please check your input and try again.",
        )
    return PredictionResponse(
        predicted_price=result.predicted_price,
        model_class=result.model_class,
        n_features=result.n_features,
        clipped=result.clipped,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict sale prices for a batch of properties",
    tags=["prediction"],
)
async def predict_batch(
    request: Request,
    payload: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """
    Predict sale prices for multiple properties in a single request.

    Processing a batch through a single model.predict() call is more efficient
    than N sequential /predict requests. The request schema enforces a
    max_length=100 guard to prevent a single oversized batch from monopolising
    the inference thread and starving concurrent requests.
    """
    predictor: HousePricePredictor = request.app.state.predictor
    feature_dicts = [req.to_feature_dict() for req in payload.requests]
    
    try:
        results = predictor.predict_batch(feature_dicts)
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Batch prediction failed.")
    predictions = [
        PredictionResponse(
            predicted_price=r.predicted_price,
            model_class=r.model_class,
            n_features=r.n_features,
            clipped=r.clipped,
        )
        for r in results
    ]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
