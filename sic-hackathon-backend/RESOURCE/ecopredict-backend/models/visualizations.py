"""
Visualization Engine
Generates Plotly chart configurations (JSON) that the frontend renders.
Supported: Line, Bar, Histogram, Scatter, Box, Heatmap, Pie, Violin, Clustering, Waterfall, Radar
"""

import numpy as np
import pandas as pd
from typing import Optional

# EcoPredict color palette
PALETTE = {
    "coal":       "#374151",
    "iron_ore":   "#b45309",
    "bauxite":    "#dc2626",
    "limestone":  "#6b7280",
    "copper":     "#d97706",
    "historical": "#3b82f6",
    "forecast":   "#10b981",
    "confidence": "rgba(16,185,129,0.15)",
    "warning":    "#f59e0b",
    "danger":     "#ef4444",
    "success":    "#22c55e",
}

MINERAL_COLORS = {
    "coal": "#374151",
    "iron_ore": "#b45309",
    "bauxite": "#dc2626",
    "limestone": "#9ca3af",
    "copper": "#d97706",
}


def _base_layout(title: str, xaxis_title: str = "", yaxis_title: str = "") -> dict:
    return {
        "title": {"text": title, "font": {"size": 18, "color": "#111827"}},
        "plot_bgcolor": "#ffffff",
        "paper_bgcolor": "#f9fafb",
        "font": {"family": "Inter, sans-serif", "size": 13, "color": "#374151"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        "xaxis": {"title": xaxis_title, "gridcolor": "#e5e7eb", "showgrid": True},
        "yaxis": {"title": yaxis_title, "gridcolor": "#e5e7eb", "showgrid": True},
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
        "hovermode": "x unified",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. FORECAST LINE CHART
# ──────────────────────────────────────────────────────────────────────────────

def forecast_line_chart(
    df: pd.DataFrame,
    forecast_dates,
    forecast_values: list,
    lower_ci: list,
    upper_ci: list,
    mineral_name: str,
    model_name: str,
) -> dict:
    dates_str = df["date"].dt.strftime("%Y-%m-%d").tolist()
    future_dates_str = [str(d)[:10] for d in forecast_dates]

    traces = [
        {
            "type": "scatter",
            "name": "Historical",
            "x": dates_str,
            "y": df["demand"].tolist(),
            "mode": "lines+markers",
            "line": {"color": PALETTE["historical"], "width": 2.5},
            "marker": {"size": 4},
        },
        {
            "type": "scatter",
            "name": "Forecast",
            "x": future_dates_str,
            "y": forecast_values,
            "mode": "lines+markers",
            "line": {"color": PALETTE["forecast"], "width": 2.5, "dash": "dot"},
            "marker": {"size": 6, "symbol": "circle-open"},
        },
        {
            "type": "scatter",
            "name": "Upper CI",
            "x": future_dates_str,
            "y": upper_ci,
            "mode": "lines",
            "line": {"width": 0},
            "showlegend": False,
            "hoverinfo": "skip",
        },
        {
            "type": "scatter",
            "name": "Confidence Interval",
            "x": future_dates_str,
            "y": lower_ci,
            "mode": "lines",
            "fill": "tonexty",
            "fillcolor": PALETTE["confidence"],
            "line": {"width": 0},
        },
    ]

    layout = _base_layout(
        f"{mineral_name} — Demand Forecast ({model_name})",
        "Time Period",
        "Resource Demand (MT)"
    )

    return {"data": traces, "layout": layout, "chart_type": "forecast_line"}


# ──────────────────────────────────────────────────────────────────────────────
# 2. HISTOGRAM — Demand distribution
# ──────────────────────────────────────────────────────────────────────────────

def demand_histogram(df: pd.DataFrame, mineral_name: str, mineral_id: str) -> dict:
    color = MINERAL_COLORS.get(mineral_id, "#6b7280")
    traces = [
        {
            "type": "histogram",
            "name": mineral_name,
            "x": df["demand"].tolist(),
            "nbinsx": 30,
            "marker": {"color": color, "opacity": 0.75, "line": {"color": "white", "width": 0.5}},
            "histnorm": "probability density",
        }
    ]
    # KDE overlay (normal approximation)
    mu, sigma = float(df["demand"].mean()), float(df["demand"].std())
    x_range = np.linspace(df["demand"].min(), df["demand"].max(), 200)
    kde = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
    traces.append({
        "type": "scatter",
        "name": "KDE (Normal)",
        "x": x_range.tolist(),
        "y": kde.tolist(),
        "mode": "lines",
        "line": {"color": PALETTE["forecast"], "width": 2.5, "dash": "dash"},
    })

    layout = _base_layout(f"{mineral_name} — Demand Distribution", "Demand (MT)", "Density")
    layout["barmode"] = "overlay"

    return {"data": traces, "layout": layout, "chart_type": "histogram"}


# ──────────────────────────────────────────────────────────────────────────────
# 3. BAR CHART — Monthly / YoY average demand
# ──────────────────────────────────────────────────────────────────────────────

def annual_bar_chart(df: pd.DataFrame, mineral_name: str, mineral_id: str) -> dict:
    color = MINERAL_COLORS.get(mineral_id, "#6b7280")
    yearly = df.groupby(df["date"].dt.year)["demand"].mean().reset_index()
    yearly.columns = ["year", "avg_demand"]

    traces = [{
        "type": "bar",
        "name": "Avg Annual Demand",
        "x": yearly["year"].tolist(),
        "y": yearly["avg_demand"].round(2).tolist(),
        "marker": {
            "color": yearly["avg_demand"].tolist(),
            "colorscale": "Blues",
            "showscale": True,
            "colorbar": {"title": "MT"},
        },
        "text": yearly["avg_demand"].round(1).tolist(),
        "textposition": "outside",
    }]

    layout = _base_layout(f"{mineral_name} — Average Annual Demand", "Year", "Average Demand (MT)")
    return {"data": traces, "layout": layout, "chart_type": "bar"}


# ──────────────────────────────────────────────────────────────────────────────
# 4. SCATTER PLOT — Demand vs Rolling Average
# ──────────────────────────────────────────────────────────────────────────────

def demand_scatter(df: pd.DataFrame, mineral_name: str, mineral_id: str) -> dict:
    color = MINERAL_COLORS.get(mineral_id, "#6b7280")
    df = df.copy()
    df["month"] = df["date"].dt.month

    traces = [
        {
            "type": "scatter",
            "name": "Actual vs 12M MA",
            "x": df["rolling_12m"].dropna().tolist(),
            "y": df["demand"].loc[df["rolling_12m"].notna()].tolist(),
            "mode": "markers",
            "marker": {
                "color": df["date"].dt.year.loc[df["rolling_12m"].notna()].tolist(),
                "colorscale": "Viridis",
                "showscale": True,
                "size": 8,
                "opacity": 0.75,
                "colorbar": {"title": "Year"},
            },
            "text": df["date"].dt.strftime("%b %Y").loc[df["rolling_12m"].notna()].tolist(),
            "hovertemplate": "%{text}<br>12M MA: %{x:.1f}<br>Actual: %{y:.1f}<extra></extra>",
        }
    ]

    # Diagonal reference line
    r_min = float(df["rolling_12m"].min())
    r_max = float(df["rolling_12m"].max())
    traces.append({
        "type": "scatter",
        "name": "Perfect Fit",
        "x": [r_min, r_max],
        "y": [r_min, r_max],
        "mode": "lines",
        "line": {"color": PALETTE["warning"], "dash": "dash", "width": 1.5},
    })

    layout = _base_layout(f"{mineral_name} — Actual vs 12M Moving Average", "12M Moving Average (MT)", "Actual Demand (MT)")
    return {"data": traces, "layout": layout, "chart_type": "scatter"}


# ──────────────────────────────────────────────────────────────────────────────
# 5. BOX PLOT — Monthly seasonality
# ──────────────────────────────────────────────────────────────────────────────

def seasonal_box_plot(df: pd.DataFrame, mineral_name: str, mineral_id: str) -> dict:
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df = df.copy()
    df["month"] = df["date"].dt.month

    traces = []
    color = MINERAL_COLORS.get(mineral_id, "#6b7280")
    for m in range(1, 13):
        vals = df[df["month"] == m]["demand"].tolist()
        traces.append({
            "type": "box",
            "name": month_names[m - 1],
            "y": vals,
            "marker": {"color": color},
            "boxpoints": "outliers",
            "jitter": 0.3,
        })

    layout = _base_layout(f"{mineral_name} — Monthly Seasonality", "Month", "Demand (MT)")
    layout["showlegend"] = False
    return {"data": traces, "layout": layout, "chart_type": "box"}


# ──────────────────────────────────────────────────────────────────────────────
# 6. HEATMAP — Year × Month demand intensity
# ──────────────────────────────────────────────────────────────────────────────

def demand_heatmap(df: pd.DataFrame, mineral_name: str) -> dict:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    pivot = df.pivot_table(index="year", columns="month", values="demand", aggfunc="mean")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    traces = [{
        "type": "heatmap",
        "z": pivot.values.round(1).tolist(),
        "x": month_names,
        "y": pivot.index.tolist(),
        "colorscale": "YlOrRd",
        "showscale": True,
        "colorbar": {"title": "MT"},
        "hoverongaps": False,
    }]

    layout = _base_layout(f"{mineral_name} — Demand Heatmap (Year × Month)", "Month", "Year")
    layout["hovermode"] = "closest"
    return {"data": traces, "layout": layout, "chart_type": "heatmap"}


# ──────────────────────────────────────────────────────────────────────────────
# 7. VIOLIN PLOT — Demand distribution by decade-segment
# ──────────────────────────────────────────────────────────────────────────────

def demand_violin(df: pd.DataFrame, mineral_name: str, mineral_id: str) -> dict:
    df = df.copy()
    df["period"] = pd.cut(
        df["date"].dt.year,
        bins=[2013, 2016, 2019, 2023],
        labels=["2014–2016", "2017–2019", "2020–2023"]
    )

    traces = []
    period_colors = ["#3b82f6", "#10b981", "#f59e0b"]
    for i, period in enumerate(["2014–2016", "2017–2019", "2020–2023"]):
        vals = df[df["period"] == period]["demand"].tolist()
        if vals:
            traces.append({
                "type": "violin",
                "name": str(period),
                "y": vals,
                "box": {"visible": True},
                "meanline": {"visible": True},
                "fillcolor": period_colors[i],
                "opacity": 0.6,
                "line": {"color": period_colors[i]},
            })

    layout = _base_layout(f"{mineral_name} — Demand Distribution by Period", "Period", "Demand (MT)")
    layout["violinmode"] = "overlay"
    return {"data": traces, "layout": layout, "chart_type": "violin"}


# ──────────────────────────────────────────────────────────────────────────────
# 8. MULTI-MINERAL LINE CHART — Combined overview
# ──────────────────────────────────────────────────────────────────────────────

def combined_line_chart(all_data: dict, normalize: bool = False) -> dict:
    traces = []
    for mineral_id, df in all_data.items():
        name = mineral_id.replace("_", " ").title()
        y = df["demand"].values
        if normalize:
            y = (y - y.min()) / (y.max() - y.min() + 1e-9)
        traces.append({
            "type": "scatter",
            "name": name,
            "x": df["date"].dt.strftime("%Y-%m").tolist(),
            "y": np.round(y, 3).tolist(),
            "mode": "lines",
            "line": {"color": MINERAL_COLORS.get(mineral_id, "#6b7280"), "width": 2},
        })

    title = "All Minerals — Normalized Demand Trends" if normalize else "All Minerals — Raw Demand Comparison"
    layout = _base_layout(title, "Month", "Demand (normalized)" if normalize else "Demand (MT)")
    return {"data": traces, "layout": layout, "chart_type": "combined_line"}


# ──────────────────────────────────────────────────────────────────────────────
# 9. RADAR CHART — Mineral profile comparison
# ──────────────────────────────────────────────────────────────────────────────

def mineral_radar(all_data: dict, metadata: list) -> dict:
    categories = ["Avg Demand", "Trend Strength", "Volatility", "Sustainability", "YoY Growth"]
    traces = []

    for m in metadata:
        mid = m["id"]
        df = all_data.get(mid)
        if df is None:
            continue
        avg_d = float(df["demand"].mean())
        trend_strength = abs(float(df["demand"].iloc[-12:].mean() - df["demand"].iloc[:12].mean()))
        vol = float(df["demand"].std() / df["demand"].mean()) * 100
        sust = float(m.get("sustainability_score", 50))
        yoy = float(df["yoy_change"].dropna().mean())

        # Normalize 0–100
        vals = [
            min(avg_d / 300 * 100, 100),
            min(trend_strength / 100 * 100, 100),
            min(vol, 100),
            sust,
            min(max(yoy + 10, 0), 100),
        ]
        vals.append(vals[0])  # close the polygon

        traces.append({
            "type": "scatterpolar",
            "name": m["name"],
            "r": [round(v, 1) for v in vals],
            "theta": categories + [categories[0]],
            "fill": "toself",
            "opacity": 0.5,
            "line": {"color": MINERAL_COLORS.get(mid, "#6b7280")},
        })

    layout = {
        "title": {"text": "Mineral Profile Radar", "font": {"size": 18}},
        "polar": {"radialaxis": {"visible": True, "range": [0, 100]}},
        "showlegend": True,
        "paper_bgcolor": "#f9fafb",
        "font": {"family": "Inter, sans-serif"},
    }
    return {"data": traces, "layout": layout, "chart_type": "radar"}


# ──────────────────────────────────────────────────────────────────────────────
# 10. CLUSTERING CHART — K-Means on annual demand features
# ──────────────────────────────────────────────────────────────────────────────

def demand_clustering_chart(all_data: dict) -> dict:
    """K-Means clustering of year × mineral demand features."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        rows = []
        for mid, df in all_data.items():
            df = df.copy()
            df["year"] = df["date"].dt.year
            for year, grp in df.groupby("year"):
                rows.append({
                    "mineral": mid,
                    "year": year,
                    "mean": grp["demand"].mean(),
                    "std": grp["demand"].std(),
                    "min": grp["demand"].min(),
                    "max": grp["demand"].max(),
                    "yoy": grp["yoy_change"].mean(),
                })

        feat_df = pd.DataFrame(rows)
        features = feat_df[["mean", "std", "yoy"]].fillna(0).values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        k = 4
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features_scaled)
        feat_df["cluster"] = labels

        cluster_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
        traces = []
        for c in range(k):
            mask = feat_df["cluster"] == c
            sub = feat_df[mask]
            traces.append({
                "type": "scatter",
                "name": f"Cluster {c + 1}",
                "x": sub["mean"].round(2).tolist(),
                "y": sub["std"].round(2).tolist(),
                "mode": "markers",
                "marker": {
                    "color": cluster_colors[c],
                    "size": 12,
                    "opacity": 0.8,
                    "line": {"color": "white", "width": 1},
                },
                "text": (sub["mineral"].str.replace("_", " ").str.title() + " " + sub["year"].astype(str)).tolist(),
                "hovertemplate": "%{text}<br>Avg: %{x:.1f}<br>Std: %{y:.1f}<extra></extra>",
            })

        layout = _base_layout(
            "Demand Clustering (K-Means, k=4)", "Average Annual Demand (MT)", "Demand Std Dev (MT)"
        )
        return {"data": traces, "layout": layout, "chart_type": "cluster", "cluster_labels": feat_df["cluster"].tolist()}

    except ImportError:
        return {"error": "sklearn not available for clustering", "chart_type": "cluster"}


# ──────────────────────────────────────────────────────────────────────────────
# 11. CORRELATION HEATMAP — Cross-mineral demand correlations
# ──────────────────────────────────────────────────────────────────────────────

def correlation_heatmap(all_data: dict) -> dict:
    pivot = pd.DataFrame({
        mid.replace("_", " ").title(): df.set_index("date")["demand"]
        for mid, df in all_data.items()
    })
    corr = pivot.corr().round(3)

    traces = [{
        "type": "heatmap",
        "z": corr.values.tolist(),
        "x": corr.columns.tolist(),
        "y": corr.index.tolist(),
        "colorscale": "RdBu",
        "zmid": 0,
        "zmin": -1,
        "zmax": 1,
        "text": corr.values.round(2).tolist(),
        "texttemplate": "%{text}",
        "showscale": True,
        "colorbar": {"title": "r"},
    }]

    layout = _base_layout("Cross-Mineral Demand Correlation Matrix")
    layout["height"] = 450
    return {"data": traces, "layout": layout, "chart_type": "correlation_heatmap"}


# ──────────────────────────────────────────────────────────────────────────────
# 12. WATERFALL — Year-over-Year demand change
# ──────────────────────────────────────────────────────────────────────────────

def yoy_waterfall(df: pd.DataFrame, mineral_name: str, mineral_id: str) -> dict:
    df = df.copy()
    yearly = df.groupby(df["date"].dt.year)["demand"].mean()
    changes = yearly.diff().dropna()
    colors = [PALETTE["success"] if v >= 0 else PALETTE["danger"] for v in changes.values]

    traces = [{
        "type": "waterfall",
        "name": "YoY Change",
        "x": [str(y) for y in changes.index.tolist()],
        "y": changes.round(2).tolist(),
        "connector": {"line": {"color": "rgb(63, 63, 63)"}},
        "decreasing": {"marker": {"color": PALETTE["danger"]}},
        "increasing": {"marker": {"color": PALETTE["success"]}},
        "totals": {"marker": {"color": MINERAL_COLORS.get(mineral_id, "#6b7280")}},
        "text": [f"{v:+.1f}" for v in changes.values],
        "textposition": "outside",
    }]

    layout = _base_layout(f"{mineral_name} — YoY Demand Change", "Year", "Δ Demand (MT)")
    return {"data": traces, "layout": layout, "chart_type": "waterfall"}


# ──────────────────────────────────────────────────────────────────────────────
# 13. PIE/DONUT — Market share by mineral
# ──────────────────────────────────────────────────────────────────────────────

def market_share_donut(all_data: dict) -> dict:
    labels, values, colors = [], [], []
    for mid, df in all_data.items():
        labels.append(mid.replace("_", " ").title())
        values.append(round(float(df["demand"].sum()), 1))
        colors.append(MINERAL_COLORS.get(mid, "#6b7280"))

    traces = [{
        "type": "pie",
        "labels": labels,
        "values": values,
        "hole": 0.45,
        "marker": {"colors": colors, "line": {"color": "white", "width": 2}},
        "textinfo": "label+percent",
        "hovertemplate": "%{label}<br>Total: %{value} MT<br>Share: %{percent}<extra></extra>",
    }]

    layout = {
        "title": {"text": "Total Demand Share by Mineral (10 Years)", "font": {"size": 18}},
        "paper_bgcolor": "#f9fafb",
        "font": {"family": "Inter, sans-serif"},
        "showlegend": True,
        "legend": {"orientation": "h"},
    }
    return {"data": traces, "layout": layout, "chart_type": "donut"}


# ──────────────────────────────────────────────────────────────────────────────
# 14. AREA CHART — Stacked demand over time
# ──────────────────────────────────────────────────────────────────────────────

def stacked_area_chart(all_data: dict) -> dict:
    dates = None
    traces = []
    for mid, df in all_data.items():
        if dates is None:
            dates = df["date"].dt.strftime("%Y-%m").tolist()
        traces.append({
            "type": "scatter",
            "name": mid.replace("_", " ").title(),
            "x": dates,
            "y": df["demand"].round(2).tolist(),
            "mode": "lines",
            "stackgroup": "one",
            "fillcolor": MINERAL_COLORS.get(mid, "#6b7280"),
            "line": {"color": MINERAL_COLORS.get(mid, "#6b7280"), "width": 0.5},
        })

    layout = _base_layout("Stacked Total Demand — All Minerals", "Month", "Cumulative Demand (MT)")
    return {"data": traces, "layout": layout, "chart_type": "stacked_area"}
