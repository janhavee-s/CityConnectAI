"""
/api/forecast routes
GET /api/forecast/{mineral_id}
GET /api/forecast/{mineral_id}/ensemble
GET /api/forecast/{mineral_id}/all-models
GET /api/forecast/combined/all
GET /api/forecast/{mineral_id}/charts
"""

from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np

from data.mineral_data import get_historical_data, get_mineral_metadata, MINERAL_IDS
from models.forecasting import run_forecast, run_ensemble, AVAILABLE_MODELS
from models.visualizations import (
    forecast_line_chart, demand_histogram, annual_bar_chart,
    demand_scatter, seasonal_box_plot, demand_heatmap, demand_violin, yoy_waterfall
)

router = APIRouter()


def _horizon_months(horizon_str: str) -> int:
    return {"6 Months": 6, "9 Months": 9, "12 Months": 12,
            "6": 6, "9": 9, "12": 12}.get(str(horizon_str), 6)


@router.get("/{mineral_id}")
def forecast_mineral(
    mineral_id: str,
    model:   str = Query(default="ARIMA"),
    horizon: str = Query(default="6 Months"),
):
    if mineral_id not in MINERAL_IDS:
        raise HTTPException(404, f"Mineral '{mineral_id}' not found. Available: {MINERAL_IDS}")

    data   = get_historical_data(mineral_id)
    df     = data[mineral_id]
    series = df["demand"]
    months = _horizon_months(horizon)
    result = run_forecast(series, model, months)

    future_dates = pd.date_range(
        start=df["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=months, freq="MS"
    )
    meta       = get_mineral_metadata(mineral_id)
    last_val   = float(series.iloc[-1])
    pct_change = ((result["forecast"][-1] - last_val) / last_val) * 100

    return {
        "mineral_id":   mineral_id,
        "mineral_name": meta["name"],
        "model":        model,
        "horizon_months": months,
        "historical": {
            "dates":  df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "demand": series.tolist(),
        },
        "forecast": {
            "dates":    future_dates.strftime("%Y-%m-%d").tolist(),
            "values":   result["forecast"],
            "lower_ci": result["lower_ci"],
            "upper_ci": result["upper_ci"],
        },
        "in_sample_fit": result.get("in_sample", []),
        "accuracy":      result.get("accuracy", {}),
        "stats": {
            "current_demand":    round(last_val, 2),
            "forecast_peak":     round(max(result["forecast"]), 2),
            "forecast_avg":      round(float(np.mean(result["forecast"])), 2),
            "pct_change":        round(pct_change, 2),
            "current_stock":     meta["stock_current"],
            "recommended_stock": meta["stock_recommended"],
            "alert": "warning" if meta["stock_current"] < meta["stock_recommended"] * 0.8 else "ok",
        },
        "configuration": {
            "resource": meta["name"],
            "model":    model,
            "duration": horizon,
        },
    }


@router.get("/{mineral_id}/ensemble")
def forecast_ensemble(mineral_id: str, horizon: str = Query(default="6 Months")):
    if mineral_id not in MINERAL_IDS:
        raise HTTPException(404, f"Mineral '{mineral_id}' not found.")

    data   = get_historical_data(mineral_id)
    df     = data[mineral_id]
    months = _horizon_months(horizon)
    result = run_ensemble(df["demand"], months)

    future_dates = pd.date_range(
        start=df["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=months, freq="MS"
    )
    model_summary = [
        {"model": name, "mape": res["accuracy"].get("mape"), "rmse": res["accuracy"].get("rmse")}
        for name, res in result.get("individual_models", {}).items()
        if "accuracy" in res
    ]
    return {
        "mineral_id":   mineral_id,
        "model":        "Ensemble",
        "horizon_months": months,
        "historical":   {"dates": df["date"].dt.strftime("%Y-%m-%d").tolist(), "demand": df["demand"].tolist()},
        "forecast": {
            "dates":    future_dates.strftime("%Y-%m-%d").tolist(),
            "values":   result["forecast"],
            "lower_ci": result["lower_ci"],
            "upper_ci": result["upper_ci"],
        },
        "n_models_used":     result.get("n_models", 0),
        "model_performance": sorted(model_summary, key=lambda x: x.get("mape") or 999),
    }


@router.get("/{mineral_id}/all-models")
def forecast_all_models(mineral_id: str, horizon: str = Query(default="6 Months")):
    if mineral_id not in MINERAL_IDS:
        raise HTTPException(404, f"Mineral '{mineral_id}' not found.")

    data   = get_historical_data(mineral_id)
    df     = data[mineral_id]
    months = _horizon_months(horizon)
    meta   = get_mineral_metadata(mineral_id)

    future_dates = pd.date_range(
        start=df["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=months, freq="MS"
    )
    future_str    = future_dates.strftime("%Y-%m-%d").tolist()
    models_output = {}

    for model_name in AVAILABLE_MODELS:
        try:
            res = run_forecast(df["demand"], model_name, months)
            models_output[model_name] = {
                "forecast": res["forecast"],
                "lower_ci": res["lower_ci"],
                "upper_ci": res["upper_ci"],
                "accuracy": res.get("accuracy", {}),
            }
        except Exception as e:
            models_output[model_name] = {"error": str(e)}

    model_colors = ["#3b82f6","#10b981","#f59e0b","#8b5cf6","#ef4444","#06b6d4"]
    chart_traces = [
        {
            "type": "scatter", "name": name,
            "x": future_str, "y": res["forecast"],
            "mode": "lines+markers",
            "line": {"color": model_colors[i % len(model_colors)], "width": 2},
        }
        for i, (name, res) in enumerate(models_output.items())
        if "forecast" in res
    ]

    return {
        "mineral_id":     mineral_id,
        "mineral_name":   meta["name"],
        "horizon_months": months,
        "forecast_dates": future_str,
        "models":         models_output,
        "comparison_chart": {
            "data": chart_traces,
            "layout": {
                "title":        {"text": f"{meta['name']} — Model Comparison"},
                "xaxis":        {"title": "Date"},
                "yaxis":        {"title": "Demand (MT)"},
                "paper_bgcolor":"#f9fafb",
                "plot_bgcolor": "#ffffff",
                "hovermode":    "x unified",
            },
        },
    }


@router.get("/combined/all")
def forecast_combined(
    horizon: str = Query(default="6 Months"),
    model:   str = Query(default="ARIMA"),
):
    months   = _horizon_months(horizon)
    combined = {}

    for mid in MINERAL_IDS:
        data   = get_historical_data(mid)
        df     = data[mid]
        meta   = get_mineral_metadata(mid)
        series = df["demand"]

        try:
            result = run_forecast(series, model, months)
            future_dates = pd.date_range(
                start=df["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=months, freq="MS"
            )
            combined[mid] = {
                "name":           meta["name"],
                "color":          meta["color"],
                "forecast_dates": future_dates.strftime("%Y-%m-%d").tolist(),
                "forecast":       result["forecast"],
                "current_demand": round(float(series.iloc[-1]), 2),
                "forecast_avg":   round(float(np.mean(result["forecast"])), 2),
                "pct_change":     round(
                    ((result["forecast"][-1] - float(series.iloc[-1])) / float(series.iloc[-1])) * 100, 2
                ),
                "accuracy": result.get("accuracy", {}),
            }
        except Exception as e:
            combined[mid] = {"name": meta["name"], "error": str(e)}

    total_current  = sum(v.get("current_demand", 0) for v in combined.values())
    total_forecast = sum(v.get("forecast_avg", 0) for v in combined.values())
    sector_pct     = ((total_forecast - total_current) / total_current * 100) if total_current else 0

    return {
        "model":          model,
        "horizon_months": months,
        "minerals":       combined,
        "sector_index": {
            "total_current_demand":  round(total_current, 2),
            "total_forecast_demand": round(total_forecast, 2),
            "sector_pct_change":     round(sector_pct, 2),
            "trend": "increasing" if sector_pct > 0 else "decreasing",
        },
    }


@router.get("/{mineral_id}/charts")
def get_mineral_charts(mineral_id: str):
    if mineral_id not in MINERAL_IDS:
        raise HTTPException(404, f"Mineral '{mineral_id}' not found.")

    data   = get_historical_data(mineral_id)
    df     = data[mineral_id]
    meta   = get_mineral_metadata(mineral_id)
    name   = meta["name"]
    series = df["demand"]

    forecast_result = run_forecast(series, "ARIMA", 6)
    future_dates    = pd.date_range(
        start=df["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS"
    )

    return {
        "mineral_id":   mineral_id,
        "mineral_name": name,
        "charts": {
            "forecast_line":  forecast_line_chart(
                df, future_dates,
                forecast_result["forecast"], forecast_result["lower_ci"],
                forecast_result["upper_ci"], name, "ARIMA"
            ),
            "histogram":      demand_histogram(df, name, mineral_id),
            "bar_annual":     annual_bar_chart(df, name, mineral_id),
            "scatter":        demand_scatter(df, name, mineral_id),
            "box_seasonal":   seasonal_box_plot(df, name, mineral_id),
            "heatmap":        demand_heatmap(df, name),
            "violin":         demand_violin(df, name, mineral_id),
            "waterfall_yoy":  yoy_waterfall(df, name, mineral_id),
        },
    }