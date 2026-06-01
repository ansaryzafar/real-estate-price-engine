"""
src/models/predictor.py
=======================
Model loading and inference wrapper for the Real Estate Price Engine.

Provides HousePricePredictor — the single interface the serving layer
communicates with. It encapsulates:

  - Artefact loading from disk (model, preprocessor, feature names)
  - The full inference pipeline: raw dict → dollar-denominated prediction
  - Prediction clipping to the valid residential price range
  - Metadata exposure for observability (model class, version, feature count)

The predictor is instantiated once at application startup via the FastAPI
lifespan context and stored in app.state. Model artefacts must not be
deserialised inside request handlers — repeated loading of a multi-megabyte
model artefact per request would saturate I/O and exhaust process memory
under concurrent load.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessing import preprocess_for_inference

logger = logging.getLogger(__name__)

# Dollar bounds for clipping predictions (Ames, Iowa residential market).
# Values outside this range indicate a bad input, not a model failure.
MIN_PRICE = 10_000
MAX_PRICE = 800_000


@dataclass
class PredictionResult:
    """
    Structured prediction output returned by HousePricePredictor.predict().

    A typed dataclass provides a defined contract between the predictor and
    the API layer: fields are IDE-navigable, type-checked, and map directly
    to the response schema. The API response model is derived from this
    object's fields.
    """
    predicted_price: float          # dollar-denominated, already clipped
    predicted_price_log: float      # raw log-scale output from model.predict()
    model_class: str                # e.g. "Ridge", "XGBRegressor"
    n_features: int                 # number of encoded features the model saw
    clipped: bool                   # True if prediction was clipped to min/max bounds


class HousePricePredictor:
    """
    Stateful prediction wrapper that holds the fitted model and preprocessor
    in memory for fast repeated inference.

    Instantiate via the class method HousePricePredictor.load(), which
    validates artefact integrity before returning a ready-to-use predictor.

    Attributes
    ----------
    model         : Fitted sklearn/XGBoost estimator.
    preprocessor  : Fitted sklearn ColumnTransformer.
    feature_names : Encoded feature names for logging and debugging.
    model_class   : Human-readable class name of the deployed model.
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        feature_names: list[str],
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.model_class = type(model).__name__
        self.n_features = len(feature_names)

    # ── Class method constructor (preferred entry point) ──────────────────────

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        preprocessor_path: str | Path,
        feature_names_path: str | Path,
    ) -> "HousePricePredictor":
        """
        Load model artefacts from disk and return an initialised predictor.

        The factory method pattern separates I/O and validation concerns from
        object construction, allowing the constructor to remain a simple
        attribute assignment while loading, integrity checks, and error
        handling are handled here.

        Parameters
        ----------
        model_path          : Path to best_model.pkl
        preprocessor_path   : Path to preprocessor.pkl
        feature_names_path  : Path to feature_names.json

        Returns
        -------
        HousePricePredictor with all artefacts loaded.

        Raises
        ------
        FileNotFoundError   If any artefact file is missing.
        RuntimeError        If the loaded object does not expose a .predict()
                            method, indicating a corrupt or mismatched pickle.
        """
        try:
            model = joblib.load(Path(model_path))
            preprocessor = joblib.load(Path(preprocessor_path))
            with open(feature_names_path) as f:
                feature_names = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load artefact: {e}") from e

        if not hasattr(model, "predict"):
            raise RuntimeError(f"Loaded object is not a model: {type(model)}")

        logger.info("Loaded %s with %d features", type(model).__name__, len(feature_names))
        try:
            return cls(model, preprocessor, feature_names)
        except Exception as e:
            raise RuntimeError(f"Failed to construct predictor: {e}") from e

    # ── Core prediction method ─────────────────────────────────────────────────

    def predict(self, raw_features: dict[str, Any]) -> PredictionResult:
        """
        Execute the full inference pipeline on a single property description.

        Pipeline:
          1. Convert dict to single-row DataFrame
          2. Apply feature engineering and encoding via the preprocessing module
          3. model.predict() → log-scale output
          4. np.expm1() → dollar-denominated prediction
          5. Clip to [MIN_PRICE, MAX_PRICE]
          6. Return PredictionResult

        For high-throughput scenarios, prefer predict_batch() which passes
        all rows through a single model.predict() call, avoiding per-row
        Python overhead.

        Parameters
        ----------
        raw_features : Dict of Ames Housing column names to values, sourced
                       from the validated Pydantic request model.

        Returns
        -------
        PredictionResult with predicted_price in dollars.
        """
        raw_df = pd.DataFrame([raw_features])

        X_enc = preprocess_for_inference(raw_df, self.preprocessor)

        log_pred = float(self.model.predict(X_enc)[0])

        price = float(np.expm1(log_pred))

        clipped = not (MIN_PRICE <= price <= MAX_PRICE)
        price = float(np.clip(price, MIN_PRICE, MAX_PRICE))

        if clipped:
            logger.warning("Prediction clipped: raw=%.0f, clipped=%.0f", np.expm1(log_pred), price)

        return PredictionResult(
            predicted_price=price,
            predicted_price_log=log_pred,
            model_class=self.model_class,
            n_features=self.n_features,
            clipped=clipped,
        )

    def predict_batch(self, raw_features_list: list[dict[str, Any]]) -> list[PredictionResult]:
        """
        Execute inference on a batch of property descriptions.

        Processing a matrix of shape (N, F) through a single model.predict()
        call is significantly more efficient than N individual calls: Python
        dispatch overhead is incurred once regardless of batch size, and
        tree-based models can apply SIMD vectorisation across rows.

        Parameters
        ----------
        raw_features_list : List of raw feature dicts, one per property.

        Returns
        -------
        List of PredictionResult, one per input, in the same order.
        """
        raw_df = pd.DataFrame(raw_features_list)
        X_enc = preprocess_for_inference(raw_df, self.preprocessor)
        log_preds = self.model.predict(X_enc)
        prices = np.expm1(log_preds)
        prices_clipped = np.clip(prices, MIN_PRICE, MAX_PRICE)

        results = []
        for log_p, price_raw, price_clipped in zip(log_preds, prices, prices_clipped):
            results.append(PredictionResult(
                predicted_price=float(price_clipped),
                predicted_price_log=float(log_p),
                model_class=self.model_class,
                n_features=self.n_features,
                clipped=bool(price_raw != price_clipped),
            ))
        return results

    # ── Introspection helpers ──────────────────────────────────────────────────

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Return the top-n most influential features for the loaded model.

        For linear models (Ridge, Lasso, ElasticNet): ranked by |coef_|.
        For tree-based models (RandomForest, XGBRegressor): ranked by
        feature_importances_.

        Exposing feature importance through a dedicated endpoint allows
        ML engineers to diagnose unexpected predictions without requiring
        direct artefact access in production environments.

        Returns
        -------
        List of (feature_name, importance_score) tuples, sorted descending.
        """
        from sklearn.linear_model import ElasticNet, Lasso, Ridge

        if isinstance(self.model, (Ridge, Lasso, ElasticNet)):
            scores = np.abs(self.model.coef_)
        elif hasattr(self.model, "feature_importances_"):
            scores = self.model.feature_importances_
        else:
            return []  # model type does not expose feature-level attribution

        pairs = sorted(zip(self.feature_names, scores), key=lambda x: x[1], reverse=True)
        return pairs[:n]
