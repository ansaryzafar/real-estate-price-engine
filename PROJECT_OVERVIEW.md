# Project Overview -- Real Estate Price Prediction Engine

## 1. Problem Statement and Business Context

**Problem**: Predict the sale price of residential properties based on physical characteristics, location, and condition.

**End users**: Real estate agents, property appraisers, mortgage lenders, homebuyers.

**Business impact**:
- Overestimating prices leads to unsold inventory and lost commission
- Underestimating prices means sellers leave money on the table
- Lenders need accurate valuations for mortgage risk assessment
- Inaccurate automated valuations erode trust in the platform

**Success criteria**:
- R-squared >= 0.85 on held-out test set
- Mean Absolute Error < $20,000 (for a median home price around $180K)
- Prediction latency < 100ms per request via REST API
- Model should generalize across neighborhoods (no extreme bias toward one area)

---

## 2. System Architecture

```
+-------------------+     +--------------------+     +------------------+
|                   |     |                    |     |                  |
|  Raw Data (CSV)   +---->+  Data Pipeline     +---->+  Feature Store   |
|  (Ames Housing)   |     |  (clean, encode,   |     |  (processed      |
|                   |     |   engineer)        |     |   features)      |
+-------------------+     +--------------------+     +--------+---------+
                                                              |
                                                              v
+-------------------+     +--------------------+     +--------+---------+
|                   |     |                    |     |                  |
|  Model Artifacts  +<----+  Training Pipeline +<----+  Train/Val/Test  |
|  (joblib/pickle)  |     |  (train, tune,     |     |  Split           |
|                   |     |   evaluate)        |     |                  |
+--------+----------+     +--------------------+     +------------------+
         |
         v
+--------+----------+     +--------------------+
|                   |     |                    |
|  FastAPI Server   +---->+  JSON Response     |
|  /predict         |     |  {price, model,    |
|  /health          |     |   confidence}      |
+-------------------+     +--------------------+
```

**Components**:
1. Data Pipeline: ingests raw CSV, handles missing values, encodes categoricals, engineers features, outputs processed dataframe
2. Training Pipeline: trains multiple models, cross-validates, tunes hyperparameters, selects best model, saves artifacts
3. Serving Layer: FastAPI app loads saved model, validates input with Pydantic, returns predictions
4. Docker Container: packages everything into a deployable unit

**Communication**: REST API (JSON over HTTP). No message queues needed for this scale.

---

## 3. Data Strategy

**Source**: Ames Housing dataset (Dean De Cock, 2011)
- 2930 observations, 82 features (23 nominal, 23 ordinal, 14 discrete, 20 continuous, 2 identifier)
- Covers individual residential property sales in Ames, Iowa from 2006 to 2010

**Why this dataset over California Housing**:
- Richer feature set (82 vs 8 features) forces real feature engineering decisions
- Mix of numeric and categorical features requires proper preprocessing
- Contains missing values that need strategy (not pre-cleaned)
- More realistic: real-world data is messy, not toy-sized

**Data quality plan**:
- Missing values: identify pattern (MCAR/MAR/MNAR), impute or drop based on missingness percentage
- Outliers: detect via IQR and domain knowledge (e.g., GrLivArea > 4000 sqft with low SalePrice)
- Class distribution: SalePrice is right-skewed, may need log transform
- Feature types: separate numeric vs categorical, ordinal vs nominal

**Feature engineering strategy**:
- Total square footage: combine basement + first floor + second floor
- Total bathrooms: combine full + half baths
- House age: current year - year built
- Remodel age: current year - year remodeled
- Has feature flags: has pool, has garage, has fireplace (binary)
- Quality scores: combine multiple quality/condition ordinal features
- Neighborhood price tiers: group neighborhoods by median price
- Interaction features: overall quality * total sqft

**Data pipeline**: raw CSV -> null analysis -> imputation -> encoding -> feature engineering -> train/val/test split (70/15/15)

---

## 4. Algorithm Design and Model Selection

**Candidate algorithms**:

| Algorithm | Rationale | Strengths | Weaknesses |
|-----------|-----------|-----------|------------|
| Linear Regression | Baseline, interpretable | Fast, interpretable coefficients | Assumes linearity |
| Ridge Regression | Handle multicollinearity | Regularizes correlated features | Still assumes linearity |
| Lasso Regression | Feature selection | Zeros out irrelevant features | Can be unstable with correlated features |
| ElasticNet | Combine Ridge + Lasso | Best of both | Extra hyperparameter |
| Decision Tree | Non-linear baseline | Captures interactions | Overfits easily |
| Random Forest | Ensemble of trees | Robust, feature importance | Slower, less interpretable |
| Gradient Boosting (XGBoost) | State-of-the-art tabular | Top performance on structured data | Requires careful tuning |

**Baseline model**: Linear Regression on top 10 correlated features (no engineering). This establishes the floor.

**Evaluation criteria**:
- Primary: RMSE (penalizes large errors more than MAE -- important for high-value homes)
- Secondary: MAE (interpretable in dollar terms), R-squared (explained variance)
- All metrics computed via 5-fold cross-validation, not single split

**Hyperparameter tuning**:
- Ridge/Lasso/ElasticNet: alpha via GridSearchCV (log scale: 0.001 to 1000)
- Random Forest: n_estimators, max_depth, min_samples_split via RandomizedSearchCV
- XGBoost: learning_rate, max_depth, n_estimators, subsample via RandomizedSearchCV

**Model selection framework**:
- Compare all models on cross-validated RMSE
- If top 2 models are within 2% RMSE, prefer the more interpretable one
- Final model evaluated on held-out test set (never seen during training or tuning)

---

## 5. Project Phases

### Phase 1 -- Research and Design (Day 1)
- Study IBM regression notebooks (Simple LR, Multiple LR, Regression Trees)
- Review Ames Housing dataset documentation
- Write this PROJECT_OVERVIEW.md
- Set up project directory and virtual environment

### Phase 2 -- Data Engineering (Day 2-3)
- Download and load Ames Housing dataset
- Exploratory Data Analysis: distributions, correlations, missing values
- Implement data cleaning pipeline (missing value imputation, outlier removal)
- Feature engineering (create derived features)
- Build sklearn ColumnTransformer for preprocessing
- Create train/validation/test splits

### Phase 3 -- Model Development (Day 3-4)
- Train baseline Linear Regression
- Train regularized models: Ridge, Lasso, ElasticNet
- Train tree-based models: Decision Tree, Random Forest, XGBoost
- Cross-validate all models (5-fold)
- Hyperparameter tuning for top performers
- Select best model

### Phase 4 -- Testing and Validation (Day 4-5)
- Unit tests for data loading and preprocessing
- Unit tests for prediction pipeline
- Final evaluation on held-out test set
- Residual analysis (distribution, heteroscedasticity)
- Error analysis by price range and neighborhood

### Phase 5 -- Deployment and Serving (Day 5-6)
- Build FastAPI app with /predict and /health endpoints
- Pydantic schemas for request/response validation
- Model loading and caching
- Dockerfile and docker-compose.yml
- Integration tests for API

### Phase 6 -- Documentation and Review (Day 6-7)
- Update README with results and architecture
- Write model comparison report
- Review code quality (linting, formatting)
- Clean git history
- Week 1 milestone quiz

---

## 6. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Overfitting to Ames-specific patterns | Model fails on other cities | High | Cross-validation, regularization, note limitation in docs |
| Missing value strategy biases results | Systematic prediction errors | Medium | Compare multiple imputation strategies, validate on subsets |
| Target leakage from features | Inflated performance metrics | High | Review every feature for temporal leakage before training |
| High-value homes poorly predicted | Large errors where it matters most | Medium | Analyze residuals by price range, consider log-transform target |
| API latency exceeds 100ms | Poor user experience | Low | Profile inference, batch preprocessing at startup |

---

## 7. Technical Constraints and Tradeoffs

- **Latency**: Batch training is fine (minutes), inference must be < 100ms
- **Compute**: CPU-only (no GPU needed for tabular regression)
- **Accuracy vs interpretability**: Linear models are interpretable (coefficient analysis), XGBoost is not. We build both and explain the tradeoff.
- **Log-transform target**: Improves RMSE on skewed prices but predictions need inverse transform -- adds complexity
- **Feature count**: More features can improve accuracy but increase preprocessing complexity and API input validation burden

---

## 8. API Contract

### POST /predict
Accepts property features, returns predicted price.

**Request**:
```json
{
  "overall_quality": 7,
  "gr_liv_area": 1500,
  "garage_cars": 2,
  "total_bsmt_sf": 1000,
  "year_built": 2005,
  "full_bath": 2,
  "lot_area": 8500,
  "neighborhood": "CollgCr"
}
```

**Response**:
```json
{
  "predicted_price": 215000.50,
  "model_used": "xgboost",
  "input_features_used": 8
}
```

**Error response**:
```json
{
  "detail": "Invalid value for overall_quality: must be between 1 and 10"
}
```

### GET /health
Returns service health status.
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 9. Testing Strategy

**Unit tests**:
- `test_data.py`: data loading returns expected shape, missing value imputation works, feature engineering produces expected columns
- `test_models.py`: model trains without error, predictions are within reasonable range, saved model loads correctly
- `test_api.py`: /predict returns 200 with valid input, /predict returns 422 with invalid input, /health returns 200

**Integration tests**:
- Full pipeline: raw data -> preprocessing -> prediction matches expected output
- API end-to-end: POST valid payload -> receive price prediction

**Model regression tests**:
- Cross-validated RMSE must not exceed baseline threshold
- R-squared must exceed 0.80

---

## 10. Definition of Done

- [ ] EDA notebook complete with visualizations and insights
- [ ] Data pipeline handles missing values, outliers, encoding, feature engineering
- [ ] At least 6 models trained and compared via cross-validation
- [ ] Best model selected with documented rationale
- [ ] Hyperparameters tuned for top 2 models
- [ ] Residual analysis performed
- [ ] All unit tests pass
- [ ] FastAPI serves predictions correctly
- [ ] Docker container builds and runs
- [ ] README has results table, architecture, and setup instructions
- [ ] Model comparison report written
- [ ] Git history is clean with meaningful commits
