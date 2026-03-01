"""
/api/analytics routes
GET /api/analytics/overview
GET /api/analytics/{mineral_id}/stats
GET /api/analytics/correlation
GET /api/analytics/clustering
GET /api/analytics/combined/charts
GET /api/analytics/sector-index
"""

from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np

from data.mineral_data import get_historical_data, get_mineral_metadata, MINERAL_IDS
from models.visualizations import (
    combined_line_chart, correlation_heatmap, mineral_radar,
    demand_clustering_chart, market_share_donut, stacked_area_chart
)

router = APIRouter()


@router.get("/overview")
def analytics_overview():
    all_data = get_historical_data()
    metadata = get_mineral_metadata()
    result   = {}

    for mid, df in all_data.items():
        meta   = next(m for m in metadata if m["id"] == mid)
        demand = df["demand"]
        result[mid] = {
            "name":  meta["name"],
            "color": meta["color"],
            "n_records":  len(df),
            "date_range": {
                "start": str(df["date"].iloc[0])[:10],
                "end":   str(df["date"].iloc[-1])[:10],
            },
            "stats": {
                "mean":   round(float(demand.mean()), 2),
                "median": round(float(demand.median()), 2),
                "std":    round(float(demand.std()), 2),
                "min":    round(float(demand.min()), 2),
                "max":    round(float(demand.max()), 2),
                "cv":     round(float(demand.std() / demand.mean()), 4),
                "q25":    round(float(demand.quantile(0.25)), 2),
                "q75":    round(float(demand.quantile(0.75)), 2),
            },
            "trend": {
                "direction": "up" if df["demand"].iloc[-12:].mean() > df["demand"].iloc[:12].mean() else "down",
                "avg_yoy":   round(float(df["yoy_change"].dropna().mean()), 2),
            },
            "sustainability_score": meta.get("sustainability_score", 50),
        }

    return {"minerals": result, "total_minerals": len(result)}


@router.get("/correlation")
def correlation_analysis():
    all_data = get_historical_data()
    chart    = correlation_heatmap(all_data)
    pivot    = pd.DataFrame({
        mid: df.set_index("date")["demand"]
        for mid, df in all_data.items()
    })
    return {
        "correlation_matrix": pivot.corr().round(4).to_dict(),
        "chart":   chart,
        "insight": (
            "Iron ore and coal show strong positive correlation due to steel production coupling. "
            "Copper demand is driven by electrification trends and diverges from coal."
        ),
    }


@router.get("/clustering")
def clustering_analysis():
    all_data = get_historical_data()
    chart    = demand_clustering_chart(all_data)
    return {
        "chart":       chart,
        "description": "Each point = one mineral × one year. Clusters group similar demand patterns.",
    }


@router.get("/combined/charts")
def combined_charts():
    all_data = get_historical_data()
    metadata = get_mineral_metadata()
    return {
        "charts": {
            "combined_line":      combined_line_chart(all_data, normalize=False),
            "normalized_line":    combined_line_chart(all_data, normalize=True),
            "stacked_area":       stacked_area_chart(all_data),
            "market_share":       market_share_donut(all_data),
            "correlation_heatmap":correlation_heatmap(all_data),
            "radar":              mineral_radar(all_data, metadata),
            "clustering":         demand_clustering_chart(all_data),
        }
    }


@router.get("/sector-index")
def sector_resource_index():
    all_data = get_historical_data()
    metadata = get_mineral_metadata()
    scores   = {}

    for mid, df in all_data.items():
        meta   = next(m for m in metadata if m["id"] == mid)
        demand = df["demand"]
        recent_avg     = float(demand.iloc[-6:].mean())
        historical_max = float(demand.max())
        historical_min = float(demand.min())
        normalized = (recent_avg - historical_min) / (historical_max - historical_min + 1e-9) * 100
        yoy = float(df["yoy_change"].dropna().iloc[-1]) if len(df["yoy_change"].dropna()) > 0 else 0
        scores[mid] = {
            "name":             meta["name"],
            "demand_pressure":  round(normalized, 1),
            "yoy_change":       round(yoy, 2),
            "sustainability":   meta["sustainability_score"],
            "carbon_intensity": meta["carbon_factor"],
        }

    crdi                 = round(float(np.mean([v["demand_pressure"] for v in scores.values()])), 1)
    sustainability_index = round(float(np.mean([v["sustainability"] for v in scores.values()])), 1)

    return {
        "composite_resource_demand_index": crdi,
        "sustainability_index":            sustainability_index,
        "interpretation": (
            "critical" if crdi > 80 else
            "high"     if crdi > 65 else
            "moderate" if crdi > 40 else
            "low"
        ),
        "mineral_scores": scores,
    }


@router.get("/{mineral_id}/stats")
def mineral_stats(mineral_id: str):
    if mineral_id not in MINERAL_IDS:
        raise HTTPException(404, f"Mineral '{mineral_id}' not found.")

    data   = get_historical_data(mineral_id)
    df     = data[mineral_id].copy()
    meta   = get_mineral_metadata(mineral_id)
    demand = df["demand"]

    df["month"] = df["date"].dt.month
    monthly_avg      = df.groupby("month")["demand"].mean()
    seasonal_indices = (monthly_avg / monthly_avg.mean()).round(4)
    df["year"]       = df["date"].dt.year
    annual           = df.groupby("year").agg(
        total=("demand","sum"), avg=("demand","mean"),
        peak=("demand","max"), trough=("demand","min"),
    ).round(2)

    return {
        "mineral_id":   mineral_id,
        "mineral_name": meta["name"],
        "category":     meta["category"],
        "unit":         meta["unit"],
        "sustainability_score": meta["sustainability_score"],
        "carbon_factor":        meta["carbon_factor"],
        "descriptive_stats": {
            "count":    len(demand),
            "mean":     round(float(demand.mean()), 2),
            "std":      round(float(demand.std()), 2),
            "min":      round(float(demand.min()), 2),
            "p10":      round(float(demand.quantile(0.10)), 2),
            "p25":      round(float(demand.quantile(0.25)), 2),
            "median":   round(float(demand.median()), 2),
            "p75":      round(float(demand.quantile(0.75)), 2),
            "p90":      round(float(demand.quantile(0.90)), 2),
            "max":      round(float(demand.max()), 2),
            "skewness": round(float(demand.skew()), 4),
            "kurtosis": round(float(demand.kurt()), 4),
        },
        "time_series_properties": {
            "autocorr_lag1": round(float(demand.autocorr(lag=1)), 4),
            "avg_yoy_pct":   round(float(df["yoy_change"].dropna().mean()), 2),
            "avg_mom_pct":   round(float(df["mom_change"].dropna().mean()), 2),
        },
        "seasonal_indices": dict(zip(
            ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            seasonal_indices.tolist()
        )),
        "annual_summary": annual.reset_index().to_dict(orient="records"),
    }