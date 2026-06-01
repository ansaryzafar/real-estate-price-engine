# Real Estate Price Prediction Engine

Week 1 Project -- Regression and Data Science Foundations

## Overview
A complete regression pipeline that predicts real estate prices using multiple ML models, served via a production REST API.

## Key Concepts Applied
- Simple and Multiple Linear Regression
- Regression Trees, Random Forest, XGBoost
- Ridge, Lasso, ElasticNet regularization
- Cross-validation, hyperparameter tuning
- Feature engineering

## Tech Stack
- Python, scikit-learn, XGBoost, Pandas, NumPy
- FastAPI, Pydantic
- Docker

## Quick Start

```bash
make setup      # create venv and install deps
make train      # run training pipeline
make test       # run tests
make serve      # start FastAPI server
make docker     # build and run Docker container
```

## Project Structure
```
src/
  data/         -- data loading and preprocessing
  models/       -- training, evaluation, prediction
  api/          -- FastAPI application
  utils/        -- visualization helpers
```

## Results

All models trained and evaluated on the Ames Housing dataset (1,458 properties).
Target: `log1p(SalePrice)`. RMSE is reported in log space; MAE is in dollars.

| Model | RMSE (log) | MAE ($) | R-squared | Notes |
|-------|-----------|---------|-----------|-------|
| Linear Regression | 0.1399 | 15,171 | 0.885 | Baseline |
| **Ridge (alpha=10)** | **0.1236** | **14,298** | **0.910** | **Selected — deployed** |
| Lasso (alpha=0.001) | 0.1223 | 14,058 | 0.912 | Marginal improvement, less interpretable |
| ElasticNet | 0.1213 | 14,097 | 0.914 | Minimal gain vs Ridge |
| Random Forest | 0.1576 | 17,296 | 0.854 | Overfits; worse on test |
| XGBoost | 0.1430 | 15,766 | 0.880 | Underperforms regularised linear models |

Ridge was selected for deployment: best test RMSE among linear models, fully interpretable
coefficients, and no risk of tree overfitting on the limited (~1,400 row) dataset.

## Status
Complete -- 31/31 tests passing
