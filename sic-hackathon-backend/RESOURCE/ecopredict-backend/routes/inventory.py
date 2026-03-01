"""Inventory route"""
from fastapi import APIRouter, HTTPException
import numpy as np
from data.mineral_data import get_historical_data, get_mineral_metadata, MINERAL_IDS
from models.forecasting import run_forecast

router = APIRouter()


@router.get("/overview")
def inventory_overview():
    """Stock status and reorder recommendations for all minerals."""
    metadata = get_mineral_metadata()
    all_data = get_historical_data()
    items = []

    for meta in metadata:
        mid = meta["id"]
        df = all_data[mid]
        forecast = run_forecast(df["demand"], "ARIMA", 6)
        avg_forecast = float(np.mean(forecast["forecast"]))
        peak_forecast = float(max(forecast["forecast"]))

        reorder_point = round(avg_forecast * 1.5, 1)
        safety_stock = round(avg_forecast * 0.5, 1)
        days_of_supply = round(
            meta["stock_current"] / (avg_forecast / 30), 1
        ) if avg_forecast > 0 else None

        status = (
            "critical" if meta["stock_current"] < reorder_point * 0.5 else
            "warning"  if meta["stock_current"] < reorder_point else
            "ok"
        )

        items.append({
            "mineral_id": mid,
            "name": meta["name"],
            "unit": meta["unit"],
            "current_stock": meta["stock_current"],
            "recommended_stock": meta["stock_recommended"],
            "reorder_point": reorder_point,
            "safety_stock": safety_stock,
            "avg_monthly_demand_forecast": round(avg_forecast, 2),
            "peak_monthly_demand_forecast": round(peak_forecast, 2),
            "days_of_supply": days_of_supply,
            "replenishment_needed": max(
                round(meta["stock_recommended"] - meta["stock_current"], 1), 0
            ),
            "status": status,
            "alert_message": {
                "critical": "⛔ Stock critically low — immediate replenishment required.",
                "warning":  "⚠️ Stock below optimal range — monitor closely.",
                "ok":       "✅ Stock levels are adequate.",
            }.get(status),
        })

    total_value_gap = sum(
        max(i["recommended_stock"] - i["current_stock"], 0) for i in items
    )
    critical_count = sum(1 for i in items if i["status"] == "critical")
    warning_count  = sum(1 for i in items if i["status"] == "warning")

    return {
        "inventory": items,
        "summary": {
            "total_minerals": len(items),
            "critical_alerts": critical_count,
            "warning_alerts": warning_count,
            "ok_count": len(items) - critical_count - warning_count,
            "total_replenishment_units_needed": round(total_value_gap, 1),
        },
    }


@router.get("/alerts")
def inventory_alerts():
    """Return only minerals with warning or critical stock status."""
    overview = inventory_overview()
    alerts = [
        item for item in overview["inventory"]
        if item["status"] in ("critical", "warning")
    ]
    return {
        "alerts": sorted(alerts, key=lambda x: (x["status"] == "warning", x["status"] == "critical")),
        "total_alerts": len(alerts),
    }


@router.get("/reorder-schedule")
def reorder_schedule():
    """Generate a 6-month reorder schedule for all minerals."""
    metadata = get_mineral_metadata()
    all_data = get_historical_data()
    schedule = []

    for meta in metadata:
        mid = meta["id"]
        df = all_data[mid]
        forecast = run_forecast(df["demand"], "ARIMA", 6)

        import pandas as pd
        future_dates = pd.date_range(
            start=df["date"].iloc[-1] + pd.offsets.MonthBegin(1),
            periods=6, freq="MS"
        )

        cumulative_demand = 0
        stock = meta["stock_current"]
        monthly_plan = []

        for i, (date, demand) in enumerate(zip(future_dates, forecast["forecast"])):
            cumulative_demand += demand
            stock_after = stock - demand
            reorder_qty = max(meta["stock_recommended"] - stock_after, 0)
            stock = stock_after + reorder_qty

            monthly_plan.append({
                "month": date.strftime("%Y-%m"),
                "forecasted_demand": round(demand, 2),
                "projected_stock": round(stock_after, 2),
                "reorder_quantity": round(reorder_qty, 2),
                "action_required": reorder_qty > 0,
            })

        schedule.append({
            "mineral_id": mid,
            "name": meta["name"],
            "current_stock": meta["stock_current"],
            "monthly_plan": monthly_plan,
            "total_reorder_6m": round(sum(m["reorder_quantity"] for m in monthly_plan), 2),
        })

    return {"reorder_schedule": schedule}


@router.get("/{mineral_id}")
def mineral_inventory(mineral_id: str):
    """Detailed inventory analysis for a single mineral."""
    if mineral_id not in MINERAL_IDS:
        raise HTTPException(404, f"Mineral '{mineral_id}' not found.")

    meta = get_mineral_metadata(mineral_id)
    all_data = get_historical_data(mineral_id)
    df = all_data[mineral_id]

    import pandas as pd
    forecast_6m  = run_forecast(df["demand"], "ARIMA", 6)
    forecast_12m = run_forecast(df["demand"], "ARIMA", 12)

    avg_f6  = float(np.mean(forecast_6m["forecast"]))
    avg_f12 = float(np.mean(forecast_12m["forecast"]))

    reorder_point = round(avg_f6 * 1.5, 1)
    safety_stock  = round(avg_f6 * 0.5, 1)
    eoq = round(
        np.sqrt((2 * avg_f6 * 12 * 50) / (meta["stock_current"] * 0.25 + 1)), 1
    )  # Economic Order Quantity approximation

    future_dates = pd.date_range(
        start=df["date"].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=6, freq="MS"
    )

    return {
        "mineral_id": mineral_id,
        "name": meta["name"],
        "unit": meta["unit"],
        "current_stock": meta["stock_current"],
        "recommended_stock": meta["stock_recommended"],
        "reorder_point": reorder_point,
        "safety_stock": safety_stock,
        "economic_order_quantity": eoq,
        "avg_monthly_forecast_6m":  round(avg_f6, 2),
        "avg_monthly_forecast_12m": round(avg_f12, 2),
        "stock_coverage_months": round(
            meta["stock_current"] / avg_f6, 1
        ) if avg_f6 > 0 else None,
        "replenishment_needed": max(
            round(meta["stock_recommended"] - meta["stock_current"], 1), 0
        ),
        "monthly_forecast_6m": [
            {"month": d.strftime("%Y-%m"), "demand": round(v, 2)}
            for d, v in zip(future_dates, forecast_6m["forecast"])
        ],
        "inventory_chart": _inventory_chart(
            meta, forecast_6m["forecast"], future_dates
        ),
    }


def _inventory_chart(meta: dict, forecast: list, future_dates) -> dict:
    """Bar chart: projected stock vs reorder point vs recommended stock."""
    import pandas as pd
    months = [d.strftime("%Y-%m") for d in future_dates]
    avg_f = float(np.mean(forecast))
    stock = meta["stock_current"]
    projected_stocks = []
    for f in forecast:
        stock = max(stock - f, 0)
        projected_stocks.append(round(stock, 2))

    return {
        "data": [
            {
                "type": "bar",
                "name": "Projected Stock",
                "x": months,
                "y": projected_stocks,
                "marker": {"color": "#3b82f6"},
            },
            {
                "type": "scatter",
                "name": "Reorder Point",
                "x": months,
                "y": [round(avg_f * 1.5, 1)] * len(months),
                "mode": "lines",
                "line": {"color": "#f59e0b", "dash": "dash", "width": 2},
            },
            {
                "type": "scatter",
                "name": "Recommended Stock",
                "x": months,
                "y": [meta["stock_recommended"]] * len(months),
                "mode": "lines",
                "line": {"color": "#10b981", "dash": "dot", "width": 2},
            },
        ],
        "layout": {
            "title": {"text": f"{meta['name']} — 6-Month Stock Projection"},
            "xaxis": {"title": "Month"},
            "yaxis": {"title": f"Stock ({meta['unit']})"},
            "plot_bgcolor": "#ffffff",
            "paper_bgcolor": "#f9fafb",
            "barmode": "group",
            "hovermode": "x unified",
        },
    }