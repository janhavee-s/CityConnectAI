"""Reports route — structured data for export/download generation."""
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
import numpy as np
import pandas as pd

from data.mineral_data import get_historical_data, get_mineral_metadata, MINERAL_IDS
from models.forecasting import run_forecast, run_ensemble, AVAILABLE_MODELS

router = APIRouter()


@router.get("/summary")
def full_report_summary():
    """Full executive summary report for all minerals."""
    all_data = get_historical_data()
    metadata = get_mineral_metadata()

    minerals_report = []
    for meta in metadata:
        mid = meta["id"]
        df = all_data[mid]
        demand = df["demand"]

        forecast_6m  = run_forecast(demand, "ARIMA", 6)
        forecast_12m = run_forecast(demand, "ARIMA", 12)

        df_copy = df.copy()
        df_copy["year"] = df_copy["date"].dt.year
        annual = df_copy.groupby("year")["demand"].agg(["mean", "max", "min"]).round(2)

        minerals_report.append({
            "id": mid,
            "name": meta["name"],
            "category": meta["category"],
            "unit": meta["unit"],
            "sustainability_score": meta["sustainability_score"],
            "carbon_factor": meta["carbon_factor"],
            "historical_stats": {
                "mean_demand": round(float(demand.mean()), 2),
                "peak_demand": round(float(demand.max()), 2),
                "min_demand":  round(float(demand.min()), 2),
                "total_10yr":  round(float(demand.sum()), 2),
                "avg_yoy_pct": round(float(df["yoy_change"].dropna().mean()), 2),
                "volatility_cv": round(float(demand.std() / demand.mean()), 4),
            },
            "forecast_6m": {
                "avg": round(float(np.mean(forecast_6m["forecast"])), 2),
                "peak": round(float(max(forecast_6m["forecast"])), 2),
                "pct_change": round(
                    (forecast_6m["forecast"][-1] - float(demand.iloc[-1])) /
                    float(demand.iloc[-1]) * 100, 2
                ),
                "model_mape": forecast_6m.get("accuracy", {}).get("mape"),
            },
            "forecast_12m": {
                "avg": round(float(np.mean(forecast_12m["forecast"])), 2),
                "peak": round(float(max(forecast_12m["forecast"])), 2),
                "pct_change": round(
                    (forecast_12m["forecast"][-1] - float(demand.iloc[-1])) /
                    float(demand.iloc[-1]) * 100, 2
                ),
            },
            "annual_breakdown": annual.reset_index().rename(
                columns={"mean": "avg", "max": "peak", "min": "trough"}
            ).to_dict(orient="records"),
            "inventory_status": {
                "current_stock": meta["stock_current"],
                "recommended_stock": meta["stock_recommended"],
                "gap": meta["stock_recommended"] - meta["stock_current"],
                "status": (
                    "critical" if meta["stock_current"] < meta["stock_recommended"] * 0.5 else
                    "warning"  if meta["stock_current"] < meta["stock_recommended"] * 0.8 else
                    "ok"
                ),
            },
        })

    # Sector-level KPIs
    total_demand = sum(m["historical_stats"]["mean_demand"] for m in minerals_report)
    avg_sustainability = round(
        float(np.mean([m["sustainability_score"] for m in minerals_report])), 1
    )
    total_carbon_10yr = sum(
        m["historical_stats"]["total_10yr"] * m["carbon_factor"]
        for m in minerals_report
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "report_title": "EcoPredict Mineral Demand Intelligence Report",
        "period_covered": "January 2014 – December 2023 (Historical) + 2024 Forecast",
        "sector_kpis": {
            "total_avg_monthly_demand_all_minerals": round(total_demand, 2),
            "sector_sustainability_index": avg_sustainability,
            "total_estimated_carbon_10yr_tCO2": round(total_carbon_10yr, 2),
            "minerals_with_stock_warnings": sum(
                1 for m in minerals_report if m["inventory_status"]["status"] != "ok"
            ),
        },
        "minerals": minerals_report,
    }


@router.get("/model-comparison")
def model_comparison_report(
    mineral_id: str = Query(default="coal"),
    horizon: int = Query(default=6),
):
    """Compare all forecasting models for a given mineral."""
    if mineral_id not in MINERAL_IDS:
        raise HTTPException(404, f"Mineral '{mineral_id}' not found.")

    all_data = get_historical_data(mineral_id)
    df = all_data[mineral_id]
    meta = get_mineral_metadata(mineral_id)
    series = df["demand"]

    future_dates = pd.date_range(
        start=df["date"].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=horizon, freq="MS"
    )

    comparison = []
    for model_name in AVAILABLE_MODELS:
        try:
            result = run_forecast(series, model_name, horizon)
            comparison.append({
                "model": model_name,
                "forecast_avg": round(float(np.mean(result["forecast"])), 2),
                "forecast_values": result["forecast"],
                "mape": result.get("accuracy", {}).get("mape"),
                "rmse": result.get("accuracy", {}).get("rmse"),
                "mae":  result.get("accuracy", {}).get("mae"),
                "residual_std": result.get("residual_std"),
            })
        except Exception as e:
            comparison.append({"model": model_name, "error": str(e)})

    # Rank by MAPE (lower is better)
    ranked = sorted(
        [m for m in comparison if "mape" in m and m["mape"] is not None],
        key=lambda x: x["mape"]
    )

    return {
        "mineral_id": mineral_id,
        "mineral_name": meta["name"],
        "horizon_months": horizon,
        "forecast_dates": future_dates.strftime("%Y-%m-%d").tolist(),
        "model_comparison": comparison,
        "ranked_by_accuracy": ranked,
        "recommended_model": ranked[0]["model"] if ranked else "ARIMA",
    }


@router.get("/annual-trends")
def annual_trends_report():
    """Year-by-year demand trends for all minerals."""
    all_data = get_historical_data()
    metadata = get_mineral_metadata()
    result = {}

    for meta in metadata:
        mid = meta["id"]
        df = all_data[mid].copy()
        df["year"] = df["date"].dt.year

        annual = df.groupby("year").agg(
            total=("demand", "sum"),
            avg=("demand", "mean"),
            peak=("demand", "max"),
            trough=("demand", "min"),
            std=("demand", "std"),
        ).round(2)

        annual["yoy_growth_pct"] = annual["avg"].pct_change() * 100
        annual["yoy_growth_pct"] = annual["yoy_growth_pct"].round(2)

        result[mid] = {
            "name": meta["name"],
            "annual_data": annual.reset_index().to_dict(orient="records"),
            "best_year": int(annual["avg"].idxmax()),
            "worst_year": int(annual["avg"].idxmin()),
            "avg_annual_growth": round(
                float(annual["yoy_growth_pct"].dropna().mean()), 2
            ),
        }

    return {"annual_trends": result}


@router.get("/sector-dashboard")
def sector_dashboard():
    """
    All data needed to render the combined sector dashboard in one call.
    Useful for initial page load performance.
    """
    all_data = get_historical_data()
    metadata = get_mineral_metadata()

    minerals_summary = []
    chart_data = {}

    for meta in metadata:
        mid = meta["id"]
        df = all_data[mid]
        demand = df["demand"]
        forecast = run_forecast(demand, "ARIMA", 6)

        minerals_summary.append({
            "id": mid,
            "name": meta["name"],
            "color": meta["color"],
            "current_demand": round(float(demand.iloc[-1]), 2),
            "forecast_avg_6m": round(float(np.mean(forecast["forecast"])), 2),
            "trend": "up" if forecast["forecast"][-1] > float(demand.iloc[-1]) else "down",
            "sustainability_score": meta["sustainability_score"],
            "stock_status": (
                "critical" if meta["stock_current"] < meta["stock_recommended"] * 0.5 else
                "warning"  if meta["stock_current"] < meta["stock_recommended"] * 0.8 else
                "ok"
            ),
        })

        chart_data[mid] = {
            "dates": df["date"].dt.strftime("%Y-%m").tolist(),
            "demand": demand.round(2).tolist(),
        }

    # CRDI
    all_demands = [m["current_demand"] for m in minerals_summary]
    normalized = [
        (d - min(all_demands)) / (max(all_demands) - min(all_demands) + 1e-9) * 100
        for d in all_demands
    ]
    crdi = round(float(np.mean(normalized)), 1)

    return {
        "generated_at": datetime.now().isoformat(),
        "crdi": crdi,
        "crdi_level": (
            "critical" if crdi > 80 else
            "high"     if crdi > 65 else
            "moderate" if crdi > 40 else
            "low"
        ),
        "minerals": minerals_summary,
        "chart_data": chart_data,
        "alerts": [
            m for m in minerals_summary if m["stock_status"] in ("critical", "warning")
        ],
    }