"""
Forecasting Engine
Supports: ARIMA, Exponential Smoothing (ETS), Linear Regression, Polynomial Regression,
          Holt-Winters, Random Forest (sklearn), Moving Average
"""

import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _make_future_dates(last_date: pd.Timestamp, months: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1), periods=months, freq="MS"
    )


def _confidence_interval(
    forecast: np.ndarray, std: float, z: float = 1.96
) -> Tuple[np.ndarray, np.ndarray]:
    margin = z * std * np.sqrt(np.arange(1, len(forecast) + 1))
    return (forecast - margin).clip(min=0), forecast + margin


def _model_accuracy(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residuals = actual - predicted
    mae  = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    mape = np.mean(np.abs(residuals / (actual + 1e-9))) * 100
    return {
        "mae":  round(float(mae),  3),
        "rmse": round(float(rmse), 3),
        "mape": round(float(mape), 2),
    }


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

def forecast_arima(series: pd.Series, horizon: int) -> dict:
    if not STATSMODELS_AVAILABLE:
        return _fallback_ets(series, horizon, "ARIMA")
    try:
        model   = ARIMA(series.values, order=(2, 1, 2))
        fit     = model.fit()
        pred    = fit.forecast(steps=horizon)
        conf    = fit.get_forecast(steps=horizon).conf_int()
        std     = np.std(fit.resid)
        accuracy = _model_accuracy(series.values[-24:], fit.fittedvalues[-24:])
        return {
            "model":        "ARIMA",
            "forecast":     np.round(np.maximum(pred, 0), 2).tolist(),
            "lower_ci":     np.round(np.maximum(conf.iloc[:, 0].values, 0), 2).tolist(),
            "upper_ci":     np.round(conf.iloc[:, 1].values, 2).tolist(),
            "in_sample":    np.round(fit.fittedvalues, 2).tolist(),
            "accuracy":     accuracy,
            "residual_std": round(float(std), 3),
        }
    except Exception as e:
        return _fallback_ets(series, horizon, f"ARIMA (fallback: {str(e)[:40]})")


def forecast_ets(series: pd.Series, horizon: int) -> dict:
    if not STATSMODELS_AVAILABLE:
        return _fallback_ets(series, horizon, "ETS")
    try:
        model = ExponentialSmoothing(
            series.values, trend="add", seasonal="add", seasonal_periods=12
        )
        fit   = model.fit(optimized=True)
        pred  = fit.forecast(horizon)
        std   = np.std(fit.resid)
        lower, upper = _confidence_interval(pred, std)
        accuracy = _model_accuracy(series.values[-24:], fit.fittedvalues[-24:])
        return {
            "model":        "Holt-Winters ETS",
            "forecast":     np.round(np.maximum(pred, 0), 2).tolist(),
            "lower_ci":     np.round(lower, 2).tolist(),
            "upper_ci":     np.round(upper, 2).tolist(),
            "in_sample":    np.round(fit.fittedvalues, 2).tolist(),
            "accuracy":     accuracy,
            "residual_std": round(float(std), 3),
        }
    except Exception:
        return _fallback_ets(series, horizon, "ETS")


def _fallback_ets(series: pd.Series, horizon: int, label: str) -> dict:
    alpha    = 0.3
    smoothed = [series.iloc[0]]
    for v in series.iloc[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    last      = smoothed[-1]
    trend_est = (series.iloc[-1] - series.iloc[-12]) / 12 if len(series) > 12 else 0
    pred      = np.array([last + trend_est * i for i in range(1, horizon + 1)])
    std       = float(series.std())
    lower, upper = _confidence_interval(pred, std)
    accuracy  = _model_accuracy(series.values[-24:], np.array(smoothed[-24:]))
    return {
        "model":        label,
        "forecast":     np.round(np.maximum(pred, 0), 2).tolist(),
        "lower_ci":     np.round(lower, 2).tolist(),
        "upper_ci":     np.round(upper, 2).tolist(),
        "in_sample":    np.round(smoothed, 2).tolist(),
        "accuracy":     accuracy,
        "residual_std": round(std, 3),
    }


def forecast_linear(series: pd.Series, horizon: int) -> dict:
    t = np.arange(len(series)).reshape(-1, 1)
    y = series.values

    if SKLEARN_AVAILABLE:
        pipe = Pipeline([("poly", PolynomialFeatures(degree=2)), ("reg", Ridge())])
        pipe.fit(t, y)
        in_sample = pipe.predict(t)
        t_future  = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
        pred      = pipe.predict(t_future)
    else:
        coeffs    = np.polyfit(t.ravel(), y, 2)
        in_sample = np.polyval(coeffs, t.ravel())
        t_future  = np.arange(len(series), len(series) + horizon)
        pred      = np.polyval(coeffs, t_future)

    std          = float(np.std(y - in_sample))
    lower, upper = _confidence_interval(pred, std)
    accuracy     = _model_accuracy(y[-24:], in_sample[-24:])
    return {
        "model":        "Polynomial Regression",
        "forecast":     np.round(np.maximum(pred, 0), 2).tolist(),
        "lower_ci":     np.round(lower, 2).tolist(),
        "upper_ci":     np.round(upper, 2).tolist(),
        "in_sample":    np.round(in_sample, 2).tolist(),
        "accuracy":     accuracy,
        "residual_std": round(std, 3),
    }


def forecast_moving_average(series: pd.Series, horizon: int, window: int = 6) -> dict:
    weights     = np.arange(1, window + 1)
    last_window = series.iloc[-window:].values
    wma         = np.dot(weights, last_window) / weights.sum()
    trend       = (series.iloc[-1] - series.iloc[-window]) / window
    pred        = np.array([wma + trend * i for i in range(1, horizon + 1)])

    in_sample   = series.rolling(window).mean().values
    valid       = ~np.isnan(in_sample)
    accuracy    = _model_accuracy(series.values[valid], in_sample[valid]) if valid.sum() > 0 else {}
    std         = float(series.std())
    lower, upper = _confidence_interval(pred, std)
    return {
        "model":        f"Weighted Moving Average (w={window})",
        "forecast":     np.round(np.maximum(pred, 0), 2).tolist(),
        "lower_ci":     np.round(lower, 2).tolist(),
        "upper_ci":     np.round(upper, 2).tolist(),
        "in_sample":    in_sample.tolist(),
        "accuracy":     accuracy,
        "residual_std": round(std, 3),
    }


def forecast_random_forest(series: pd.Series, horizon: int) -> dict:
    if not SKLEARN_AVAILABLE:
        return _fallback_ets(series, horizon, "Random Forest")

    n_lags = min(24, len(series) // 3)
    values = series.values
    X, y   = [], []
    for i in range(n_lags, len(values)):
        X.append(values[i - n_lags:i])
        y.append(values[i])
    X, y = np.array(X), np.array(y)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    in_sample_preds = model.predict(X)
    accuracy        = _model_accuracy(y[-24:], in_sample_preds[-24:])
    std             = float(np.std(y - in_sample_preds))

    history = list(values[-n_lags:])
    pred    = []
    for _ in range(horizon):
        x_next = np.array(history[-n_lags:]).reshape(1, -1)
        p = float(model.predict(x_next)[0])
        pred.append(p)
        history.append(p)

    pred         = np.array(pred)
    lower, upper = _confidence_interval(pred, std)

    in_sample_full            = np.full(len(series), np.nan)
    in_sample_full[n_lags:]   = in_sample_preds

    return {
        "model":               "Random Forest",
        "forecast":            np.round(np.maximum(pred, 0), 2).tolist(),
        "lower_ci":            np.round(lower, 2).tolist(),
        "upper_ci":            np.round(upper, 2).tolist(),
        "in_sample":           np.where(np.isnan(in_sample_full), None, np.round(in_sample_full, 2)).tolist(),
        "accuracy":            accuracy,
        "residual_std":        round(std, 3),
        "feature_importances": model.feature_importances_.tolist(),
    }


def forecast_gradient_boost(series: pd.Series, horizon: int) -> dict:
    if not SKLEARN_AVAILABLE:
        return _fallback_ets(series, horizon, "Gradient Boosting")

    n_lags = min(18, len(series) // 3)
    values = series.values
    X, y   = [], []
    for i in range(n_lags, len(values)):
        X.append(values[i - n_lags:i])
        y.append(values[i])
    X, y = np.array(X), np.array(y)

    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
    model.fit(X, y)

    in_sample_preds = model.predict(X)
    accuracy        = _model_accuracy(y[-24:], in_sample_preds[-24:])
    std             = float(np.std(y - in_sample_preds))

    history = list(values[-n_lags:])
    pred    = []
    for _ in range(horizon):
        x_next = np.array(history[-n_lags:]).reshape(1, -1)
        p = float(model.predict(x_next)[0])
        pred.append(p)
        history.append(p)

    pred         = np.array(pred)
    lower, upper = _confidence_interval(pred, std)

    in_sample_full          = np.full(len(series), np.nan)
    in_sample_full[n_lags:] = in_sample_preds

    return {
        "model":        "Gradient Boosting",
        "forecast":     np.round(np.maximum(pred, 0), 2).tolist(),
        "lower_ci":     np.round(lower, 2).tolist(),
        "upper_ci":     np.round(upper, 2).tolist(),
        "in_sample":    np.where(np.isnan(in_sample_full), None, np.round(in_sample_full, 2)).tolist(),
        "accuracy":     accuracy,
        "residual_std": round(std, 3),
    }


# ─────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────

MODEL_MAP = {
    "ARIMA":            forecast_arima,
    "ETS":              forecast_ets,
    "LinearRegression": forecast_linear,
    "MovingAverage":    forecast_moving_average,
    "RandomForest":     forecast_random_forest,
    "GradientBoosting": forecast_gradient_boost,
}

AVAILABLE_MODELS = list(MODEL_MAP.keys())


def run_forecast(series: pd.Series, model_name: str, horizon: int) -> dict:
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {AVAILABLE_MODELS}")
    return MODEL_MAP[model_name](series, horizon)


def run_ensemble(series: pd.Series, horizon: int) -> dict:
    results = {}
    for name, func in MODEL_MAP.items():
        try:
            results[name] = func(series, horizon)
        except Exception as e:
            results[name] = {"model": name, "error": str(e)}

    valid_forecasts = [
        np.array(r["forecast"])
        for r in results.values()
        if "forecast" in r and len(r["forecast"]) == horizon
    ]

    if valid_forecasts:
        ensemble_pred = np.mean(valid_forecasts, axis=0)
        ensemble_std  = np.std(valid_forecasts, axis=0)
        lower = ensemble_pred - 1.96 * ensemble_std
        upper = ensemble_pred + 1.96 * ensemble_std
    else:
        ensemble_pred = np.zeros(horizon)
        lower = upper  = ensemble_pred

    return {
        "model":             "Ensemble (All Models)",
        "forecast":          np.round(np.maximum(ensemble_pred, 0), 2).tolist(),
        "lower_ci":          np.round(np.maximum(lower, 0), 2).tolist(),
        "upper_ci":          np.round(upper, 2).tolist(),
        "individual_models": results,
        "n_models":          len(valid_forecasts),
    }