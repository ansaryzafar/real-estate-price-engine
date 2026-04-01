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

| Model | RMSE | MAE | R-squared |
|-------|------|-----|-----------|
| Linear Regression | -- | -- | -- |
| Ridge | -- | -- | -- |
| Lasso | -- | -- | -- |
| Random Forest | -- | -- | -- |
| XGBoost | -- | -- | -- |

## Status
Not started
