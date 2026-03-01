"""
Mineral Data Module
- Metadata for 5 minerals: Coal, Iron Ore, Bauxite, Limestone, Copper
- Realistic 10-year monthly datasets (Jan 2014 - Dec 2023) with trends, seasonality, noise
- All values in metric tonnes (MT) unless noted
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────

MINERAL_METADATA = [
    {
        "id": "coal",
        "name": "Coal",
        "unit": "MT",
        "category": "Energy Mineral",
        "color": "#374151",
        "icon": "⚫",
        "description": "Thermal & coking coal for power generation and steel production.",
        "sustainability_score": 28,
        "carbon_factor": 2.42,
        "stock_current": 150,
        "stock_recommended": 220,
    },
    {
        "id": "iron_ore",
        "name": "Iron Ore",
        "unit": "MT",
        "category": "Metallic Mineral",
        "color": "#b45309",
        "icon": "🔩",
        "description": "Primary feedstock for steel manufacturing and construction.",
        "sustainability_score": 45,
        "carbon_factor": 0.07,
        "stock_current": 320,
        "stock_recommended": 410,
    },
    {
        "id": "bauxite",
        "name": "Bauxite",
        "unit": "MT",
        "category": "Metallic Mineral",
        "color": "#dc2626",
        "icon": "🧱",
        "description": "Primary ore of aluminium used in aerospace, packaging & automotive.",
        "sustainability_score": 52,
        "carbon_factor": 0.04,
        "stock_current": 95,
        "stock_recommended": 140,
    },
    {
        "id": "limestone",
        "name": "Limestone",
        "unit": "MT",
        "category": "Industrial Mineral",
        "color": "#6b7280",
        "icon": "🪨",
        "description": "Key ingredient in cement, glass, and chemical manufacturing.",
        "sustainability_score": 61,
        "carbon_factor": 0.82,
        "stock_current": 210,
        "stock_recommended": 270,
    },
    {
        "id": "copper",
        "name": "Copper",
        "unit": "MT",
        "category": "Metallic Mineral",
        "color": "#d97706",
        "icon": "🟤",
        "description": "Critical for electrical wiring, EVs, and renewable energy systems.",
        "sustainability_score": 69,
        "carbon_factor": 0.03,
        "stock_current": 58,
        "stock_recommended": 82,
    },
]

MINERAL_IDS = [m["id"] for m in MINERAL_METADATA]

STATE_DISTRIBUTION = {
    "coal": {
        "Maharashtra": 0.25,
        "Chhattisgarh": 0.30,
        "Jharkhand": 0.20,
        "Odisha": 0.15,
        "Others": 0.10
    },
    "iron_ore": {
        "Odisha": 0.35,
        "Karnataka": 0.25,
        "Chhattisgarh": 0.20,
        "Jharkhand": 0.15,
        "Others": 0.05
    },
    "bauxite": {
        "Odisha": 0.40,
        "Gujarat": 0.25,
        "Maharashtra": 0.20,
        "Chhattisgarh": 0.10,
        "Others": 0.05
    },
    "limestone": {
        "Rajasthan": 0.30,
        "Andhra Pradesh": 0.25,
        "Gujarat": 0.20,
        "Tamil Nadu": 0.15,
        "Others": 0.10
    },
    "copper": {
        "Rajasthan": 0.45,
        "Madhya Pradesh": 0.20,
        "Jharkhand": 0.15,
        "Karnataka": 0.10,
        "Others": 0.10
    },
}

# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────

def _generate_mineral_series(
    mineral_id: str,
    start: str = "2014-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed + hash(mineral_id) % 1000)

    dates = pd.date_range(start=start, end=end, freq="MS")
    n = len(dates)
    t = np.arange(n)

    params = {
        "coal": {
            "base": 120, "trend": 0.25, "trend_decay": -0.0008,
            "seasonal_amp": 18, "seasonal_phase": 0, "noise_std": 5,
            "shocks": {45: -12, 72: -28, 84: 10},
        },
        "iron_ore": {
            "base": 280, "trend": 1.8, "trend_decay": -0.002,
            "seasonal_amp": 22, "seasonal_phase": 2, "noise_std": 12,
            "shocks": {19: -18, 72: -45, 84: 35, 96: -20},
        },
        "bauxite": {
            "base": 75, "trend": 0.9, "trend_decay": 0.001,
            "seasonal_amp": 8, "seasonal_phase": 3, "noise_std": 4,
            "shocks": {72: -15, 84: 20},
        },
        "limestone": {
            "base": 175, "trend": 1.1, "trend_decay": -0.0005,
            "seasonal_amp": 25, "seasonal_phase": -1, "noise_std": 8,
            "shocks": {72: -30, 84: 18},
        },
        "copper": {
            "base": 42, "trend": 0.6, "trend_decay": 0.003,
            "seasonal_amp": 5, "seasonal_phase": 1, "noise_std": 3,
            "shocks": {72: -10, 84: 15, 96: 8},
        },
    }

    p = params.get(mineral_id, params["coal"])

    trend     = p["base"] + p["trend"] * t + p["trend_decay"] * t**2
    seasonal  = p["seasonal_amp"] * np.sin(2 * np.pi * (t - p["seasonal_phase"]) / 12)
    quarterly = (p["seasonal_amp"] * 0.3) * np.cos(2 * np.pi * t / 3)

    shock = np.zeros(n)
    for idx, magnitude in p["shocks"].items():
        if idx < n:
            shock += magnitude * np.exp(-0.5 * ((t - idx) / 2) ** 2)

    noise  = np.random.normal(0, p["noise_std"], n)
    demand = np.maximum(trend + seasonal + quarterly + shock + noise, 10)

    df = pd.DataFrame({
        "date":     dates,
        "demand":   np.round(demand, 2),
        "trend":    np.round(trend, 2),
        "seasonal": np.round(seasonal + quarterly, 2),
        "shock":    np.round(shock, 2),
        "noise":    np.round(noise, 2),
    })

    df["yoy_change"]  = df["demand"].pct_change(12) * 100
    df["mom_change"]  = df["demand"].pct_change(1) * 100
    df["rolling_3m"]  = df["demand"].rolling(3).mean()
    df["rolling_12m"] = df["demand"].rolling(12).mean()
    df["mineral"]     = mineral_id

    return df


def _expand_to_states(df_national: pd.DataFrame, mineral_id: str) -> pd.DataFrame:
    """
    Expand national-level demand into state-level distribution.
    """
    state_map = STATE_DISTRIBUTION[mineral_id]
    rows = []

    for _, row in df_national.iterrows():
        for state, pct in state_map.items():
            rows.append({
                "date": row["date"],
                "mineral": mineral_id,
                "state": state,
                "demand": round(row["demand"] * pct, 2),
                "trend": round(row["trend"] * pct, 2),
                "seasonal": round(row["seasonal"] * pct, 2),
                "shock": round(row["shock"] * pct, 2),
                "noise": round(row["noise"] * pct, 2),
                "yoy_change": row["yoy_change"],
                "mom_change": row["mom_change"],
                "rolling_3m": row["rolling_3m"],
                "rolling_12m": row["rolling_12m"],
            })

    return pd.DataFrame(rows)

def get_historical_data(mineral_id: str = None) -> dict:
    if mineral_id and mineral_id not in MINERAL_IDS:
        raise ValueError(f"Unknown mineral: {mineral_id}. Choose from {MINERAL_IDS}")

    minerals_to_load = [mineral_id] if mineral_id else MINERAL_IDS
    return {mid: _generate_mineral_series(mid) for mid in minerals_to_load}


def get_combined_data(state_level: bool = False) -> pd.DataFrame:
    dfs = []

    for mid in MINERAL_IDS:
        national_df = _generate_mineral_series(mid)

        if state_level:
            expanded_df = _expand_to_states(national_df, mid)
            dfs.append(expanded_df)
        else:
            dfs.append(national_df)

    return pd.concat(dfs, ignore_index=True)


def get_mineral_metadata(mineral_id: str = None):
    if mineral_id:
        for m in MINERAL_METADATA:
            if m["id"] == mineral_id:
                return m
        raise ValueError(f"Unknown mineral: {mineral_id}")
    return MINERAL_METADATA

