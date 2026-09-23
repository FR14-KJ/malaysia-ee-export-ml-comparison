# Comparative Forecasting of Malaysia's E&E Exports: Machine Learning vs. ARIMA

Benchmarking tuned Random Forest, SVR, and XGBoost models against a statistical ARIMA baseline for forecasting Malaysia's monthly E&E exports — and making an evidence-based production model choice even when the most accurate model isn't the one deployed.

## Overview

ARIMA forecasts a series from its own history alone. Machine learning models can additionally ingest macroeconomic indicators. This project asks: **does that extra information translate into better forecasts?** It builds a full feature-engineered ML pipeline, tunes three model families with leakage-safe time-series cross-validation, and benchmarks them honestly against ARIMA — reporting the result even when it didn't favor the more sophisticated models.

## Data & feature engineering

- Same base series as the companion ARIMA project: 423 monthly observations (Jan 1990–Mar 2025), reduced to 408 after feature construction
- **13 engineered features** per forecasting horizon: E&E lags (1, 3, 12 months), 3- and 12-month rolling mean/std, growth rate, calendar features (month, quarter), and 1-month-lagged CPI inflation, exchange rate, and U.S. IPI

## Methodology

1. **Chronological 80:20 train-test split** (326 train / 82 test observations) — no random shuffling, to prevent lookahead bias
2. **Hyperparameter tuning** via `GridSearchCV` with 5-fold `TimeSeriesSplit` for Random Forest, SVR, and XGBoost (36–108 parameter combinations per model per horizon)
3. **Feature scaling** — `StandardScaler` fit only on training data, applied to test data (SVR only; tree-based models don't require it)
4. **Evaluation** — MAE, MSE, RMSE, R² on the held-out test set, at 1-, 2-, and 3-month horizons, benchmarked against the ARIMA baseline from the companion project

## Results

| Model | 1M R² | 2M R² | Best ML model? |
|---|---|---|---|
| **ARIMA** | **0.690** | **0.644** | — (baseline) |
| XGBoost (tuned) | −1.344 | −1.668 | ✅ best of the three ML models |
| Random Forest (tuned) | −1.473 | −1.677 | |
| SVR (tuned) | −5.394 | −5.080 | |

**ARIMA outperformed every tuned ML model at every horizon**, and was the only model with positive R² throughout. Among the ML models, XGBoost was consistently the strongest.

**Feature importance:** lagged and rolling E&E export features accounted for **>97%** of XGBoost's predictive weight; the three macroeconomic indicators contributed **<3%** combined — once historical export behavior is captured, exchange rate/inflation/IPI added little further signal.

## The key decision: accuracy vs. deployability

Despite losing to ARIMA on backtested accuracy, **XGBoost was selected for production forecasting** (Apr–Jun 2025), for two reasons:
1. It's multivariate — it can incorporate macroeconomic context ARIMA structurally cannot
2. It produces transparent feature importance — useful for stakeholders who need to understand *why* a forecast moved, not just *what* it is

This trade-off — reporting the statistically superior model (ARIMA) as the benchmark while deploying the operationally more useful model (XGBoost) — is stated explicitly rather than hidden behind a single "winning" metric.

## Tech stack

Python (scikit-learn, XGBoost, pandas), Power BI

## Limitations & future work

- No ML model beat the ARIMA baseline — the current feature set may not fully capture external drivers of E&E exports
- Small effective sample (408 obs / 82 test) limits how confidently close model results (e.g., XGBoost vs. Random Forest at 2-months) can be distinguished
- No prediction intervals — only point forecasts reported for ML models
- Structural breaks (1997–98 crisis, 2008–09 GFC, COVID-19) spanning the 35-year window were examined descriptively but not formally modeled

See the full project report (`/docs`) for complete literature review, EDA, and per-horizon results.

## Related project

The ARIMA baseline referenced here is fully documented in a companion study: [ARIMA-Based Forecasting](#) *(link your first repo here)*
