"""
src/api/schemas.py
==================
Pydantic v2 request and response schemas for the Real Estate Price Engine API.

Pydantic v2 enforces type coercion and validation at the application boundary.
Invalid requests are rejected with a structured HTTP 422 response before
reaching prediction logic, eliminating the need for manual field validation
in endpoint handlers.

Convention: input models (what the API accepts) are defined separately from
output models (what the API returns). Typed response schemas make breaking
changes visible as type errors and produce accurate OpenAPI documentation.

Pydantic v2 API notes:
  - Use model_config = ConfigDict(...) in place of the inner class Config
  - Validators use @field_validator instead of @validator
  - @field_validator mode="before" replaces pre=True
  - Cross-field validation uses @model_validator
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Module-level constant (must be outside any Pydantic model — Pydantic v2 rejects
# bare unannotated class variables in BaseModel subclasses)
VALID_QUALITY_VALUES: frozenset[str] = frozenset({"None", "Po", "Fa", "TA", "Gd", "Ex"})


# =============================================================================
# Request schema
# =============================================================================

class PredictionRequest(BaseModel):
    """
    Raw property features for a single house price prediction request.

    Column names match the original Ames Housing dataset. Required fields
    are those with the highest feature importance and no universally applicable
    default. Optional fields default to training-data medians or modes,
    reducing the number of fields a caller must supply for a valid request.
    """

    model_config = ConfigDict(
        # Keep extra fields from the request payload so the full Ames Housing
        # feature set passes through to the ColumnTransformer. The schema still
        # validates and provides defaults for the most important columns.
        extra="allow",
        # Validate default values at class definition time, not just at runtime.
        validate_default=True,
        # Auto-generate JSON schema with examples for /docs.
        json_schema_extra={
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1500,
                "GarageCars": 2,
                "TotalBsmtSF": 1000,
                "YearBuilt": 2005,
                "FullBath": 2,
                "LotArea": 8500,
                "Neighborhood": "CollgCr",
                "YrSold": 2010,
                "YearRemodAdd": 2005,
                "ExterQual": "Gd",
                "KitchenQual": "Gd",
            }
        },
    )

    # ── Required fields (no default — must be present in every request) ─────────

    OverallQual: int = Field(
        ...,
        ge=1, le=10,
        description="Overall material and finish quality (1=very poor, 10=excellent)"
    )
    
    GrLivArea: int = Field(
        ...,
        gt=0, le=6000,
        description="Above-grade (ground) living area in square feet"
    )
    TotalBsmtSF: int = Field(
        ...,
        ge=0, le=6000,
        description="Total basement area in square feet"
    )
    YearBuilt: int = Field(
        ...,
        ge=0, le=2024,
        description="Original construction year"
    )
    YrSold: int = Field(
        ...,
        ge=0, le=2024,
        description="Year of sale"
    )
    YearRemodAdd: int = Field(
        ...,
        ge=0, le=2024,
        description="Remodel year (same as YearBuilt if no remodel)"
    )
    Neighborhood: str =  Field(
        ...,
        description="Physical locations within Ames city limits (e.g., 'CollgCr', 'Veenker', 'Crawfor')"
    )
    

    # ── Optional fields (defaults derived from training-data statistics) ────────

    GarageCars: int = Field(
        default=2,
        ge=0, le=5,
        description="Number of cars the garage can hold"
    )
    FullBath: int = Field(
        default=2,
        ge=0, le=4,
        description="Number of full bathrooms above grade"
        
    )
    HalfBath: int = Field(
        default=0,
        ge=0, le=2,
        description="Number of half bathrooms above grade"
    )
    BsmtFullBath: int = Field(
        default=0,
        ge=0, le=3,
        description="Number of full bathroom in the basement"
    )
    BsmtHalfBath: int = Field(
        default=0,
        ge=0, le=2,
        description="Number of half bathroom in the basement"
    )
    LotArea: int = Field(
        default=9000,
        gt=0,
        description="Lot size in square feet"
    )
    LotFrontage: Optional[float] = Field(
        default=None,
        description="Linear feet of street connected to property (None if missing, will be imputed)"
    )
    # Floor areas — used by add_features() to compute TotalSF
    FirstFlrSF: int = Field(
        default=1000,
        alias="1stFlrSF",
        gt=0,
        description="First floor square feet",
    )
    SecondFlrSF: int = Field(
        default=0,
        alias="2ndFlrSF",
        ge=0,
        description="Second floor square feet (0 for single-storey homes)",
    )
    ExterQual: str = Field(
        default="TA",
        description="Exterior material quality (None if missing, will be imputed, allowed values: 'None', 'Po', 'Fa', 'TA', 'Gd', 'Ex')"
    )
    KitchenQual: str = Field(
        default="TA",
        description="Kitchen quality (None if missing, will be imputed, allowed values: 'None', 'Po', 'Fa', 'TA', 'Gd', 'Ex')"
    )
    BsmtQual: str = Field(
        default="TA",
        description="Basement height quality (None if missing, will be imputed, allowed values: 'None', 'Po', 'Fa', 'TA', 'Gd', 'Ex')"
    )
    FireplaceQu: str = Field(
        default="None",
        description="Fireplace quality (None if missing, will be imputed, allowed values: 'None', 'Po', 'Fa', 'TA', 'Gd', 'Ex')"
    )
    GarageQual: str = Field(
        default="TA",
        description="Garage quality (None if missing, will be imputed, allowed values: 'None', 'Po', 'Fa', 'TA', 'Gd', 'Ex')"
    )
    GarageCond: str = Field(
        default="TA",
        description="Garage condition (None if missing, will be imputed, allowed values: 'None', 'Po', 'Fa', 'TA', 'Gd', 'Ex')"
    )
    ExterCond: str = Field(
        default="TA",
        description="Exterior material condition (None if missing, will be imputed, allowed values: 'None', 'Po', 'Fa', 'TA', 'Gd', 'Ex')"
    )
    BsmtCond: str = Field(
        default="TA",
        description="Basement height condition (None if missing, will be imputed, allowed values: 'None', 'Po', 'Fa', 'TA', 'Gd', 'Ex')"
    )
    HeatingQC: str = Field(
        default="Ex",
        description="Heating quality and condition (None if missing, will be imputed, allowed values: 'None', 'Po', 'Fa', 'TA', 'Gd', 'Ex')"
    )
    GarageArea: float =Field(
        default=480.0,
        ge=0,
        description="Size of garage in square feet (None if missing, will be imputed, default=480.0)"
    )
    MasVnrArea: float = Field(
        default=0.0,
        ge=0,
        description="Masonry veneer area in square feet (None if missing, will be imputed, default=0.0)"
    )
    MsZoning: str = Field(
        default="RL",
        description="Identifies the general zoning classicification of the sale (None if missing, will be imputed, default='RL')"
    )
    Street: str = Field(
        default="Pave",
        description="Type of road acess to property (None if missing, will be imputed, default='Pave')"
    )


    # \u2500\u2500 Field validators \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    @field_validator(
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond",
        mode="before",
    )
    @classmethod
    def validate_quality_field(cls, v: str) -> str:
        if v not in VALID_QUALITY_VALUES:
            raise ValueError(
                f"'{v}' is not a valid quality rating."
                f"Allowed values: {sorted(VALID_QUALITY_VALUES)}"
            )
        return v

    @model_validator(mode="after")
    def year_sold_after_year_built(self) -> "PredictionRequest":
        if self.YrSold < self.YearBuilt:
            raise ValueError(
                f"YrSold ({self.YrSold}) cannot be before YearBuilt ({self.YearBuilt})"
            )
        return self

    def to_feature_dict(self) -> dict:
        """
        Serialise the validated model to a plain dict for the predictor.

        by_alias=True ensures aliased fields (e.g., FirstFlrSF → 1stFlrSF)
        are emitted under their original Ames Housing column names.
        exclude_none=True omits fields set to None so that the preprocessing
        module's imputation logic handles them as missing values.
        """
        return self.model_dump(by_alias=True, exclude_none=True)


# =============================================================================
# Response schemas
# =============================================================================

class PredictionResponse(BaseModel):
    """
    Response body for POST /predict.

    Metadata fields accompany the prediction to support observability and
    transparency:
      - model_class identifies the deployed model, which is critical for
        correct result attribution during A/B experiments or canary rollouts.
      - clipped signals that the raw output exceeded the valid price range
        and was bounded, indicating a potentially out-of-distribution input.
    """

    predicted_price: float = Field(
        ...,
        description="Predicted sale price in US dollars",
        examples=[215_000.0],
    )
    model_class: str = Field(
        ...,
        description="Class name of the model that made the prediction",
        examples=["Ridge"],
    )
    n_features: int = Field(
        ...,
        description="Number of encoded features used for this prediction",
        examples=[263],
    )
    clipped: bool = Field(
        ...,
        description=(
            "True if the raw prediction was outside the valid price range "
            "and was clipped. Indicates a potentially out-of-distribution input."
        ),
    )

    # Confidence interval bounds (optional). A ±1 RMSE_dollars interval
    # provides a reasonable first-order approximation when loaded from
    # model_metadata.json at startup.
    predicted_price_range_low: Optional[float] = Field(
        None,
        description="Lower bound of the predicted price confidence interval (optional, may be null)"
    )
    predicted_price_range_high: Optional[float] = Field(
        None,
        description="Upper bound of the predicted price confidence interval (optional, may be null)"
    )
    

class HealthResponse(BaseModel):
    """
    Response body for GET /health.

    Consumed by load balancers and container orchestrator liveness/readiness
    probes. HTTP 200 is returned only when the model is fully loaded and the
    service is ready to accept prediction requests.
    """
    status: str = Field(..., examples=["healthy"])
    model_loaded: bool
    model_class: Optional[str] = None
    n_features: Optional[int] = None
    api_version: str


class BatchPredictionRequest(BaseModel):
    """
    Request body for POST /predict/batch.

    A batch endpoint amortises HTTP round-trip latency for clients that
    process multiple properties simultaneously (e.g., portfolio valuation
    tools). The max_length=100 constraint prevents a single oversized request
    from monopolising the inference thread under concurrent load.
    """
    requests: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,    # upper bound prevents resource exhaustion from oversized payloads
        description="List of property feature dicts, one per house",
    )


class BatchPredictionResponse(BaseModel):
    """Response body for POST /predict/batch."""
    predictions: list[PredictionResponse]
    count: int = Field(..., description="Number of predictions returned")
