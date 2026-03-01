"""Sustainability route"""
from fastapi import APIRouter, HTTPException
import numpy as np
from data.mineral_data import get_historical_data, get_mineral_metadata, MINERAL_IDS
from models.forecasting import run_forecast

router = APIRouter()

@router.get("/overview")
def sustainability_overview():
    all_data = get_historical_data()
    metadata = get_mineral_metadata()
    result = []
    for meta in metadata:
        mid = meta["id"]
        df = all_data[mid]
        total_demand = float(df["demand"].sum())
        carbon_total = round(total_demand * meta["carbon_factor"], 2)
        result.append({
            "mineral_id": mid,
            "name": meta["name"],
            "sustainability_score": meta["sustainability_score"],
            "carbon_factor": meta["carbon_factor"],
            "total_historical_demand": round(total_demand, 2),
            "estimated_total_carbon_MT": carbon_total,
            "rating": (
                "excellent" if meta["sustainability_score"] > 70 else
                "good" if meta["sustainability_score"] > 50 else
                "poor"
            ),
        })
    return {"minerals": sorted(result, key=lambda x: -x["sustainability_score"])}


@router.get("/carbon-trend")
def carbon_trend():
    """Monthly carbon emission estimates for all minerals."""
    all_data = get_historical_data()
    metadata = get_mineral_metadata()
    result = {}
    for meta in metadata:
        mid = meta["id"]
        df = all_data[mid].copy()
        df["carbon"] = (df["demand"] * meta["carbon_factor"]).round(3)
        result[mid] = {
            "name": meta["name"],
            "carbon_factor": meta["carbon_factor"],
            "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "carbon_emissions": df["carbon"].tolist(),
            "annual_totals": (
                df.groupby(df["date"].dt.year)["carbon"]
                .sum().round(2).to_dict()
            ),
        }
    return result


@router.get("/benchmarks")
def sustainability_benchmarks():
    """Compare minerals against industry sustainability benchmarks."""
    metadata = get_mineral_metadata()
    benchmarks = {
        "coal":      {"industry_avg_score": 22, "best_in_class": 40},
        "iron_ore":  {"industry_avg_score": 42, "best_in_class": 65},
        "bauxite":   {"industry_avg_score": 48, "best_in_class": 70},
        "limestone": {"industry_avg_score": 58, "best_in_class": 75},
        "copper":    {"industry_avg_score": 63, "best_in_class": 82},
    }
    result = []
    for meta in metadata:
        mid = meta["id"]
        bm = benchmarks.get(mid, {})
        score = meta["sustainability_score"]
        result.append({
            "mineral_id": mid,
            "name": meta["name"],
            "our_score": score,
            "industry_avg": bm.get("industry_avg_score"),
            "best_in_class": bm.get("best_in_class"),
            "vs_industry_avg": round(score - bm.get("industry_avg_score", score), 1),
            "vs_best_in_class": round(score - bm.get("best_in_class", score), 1),
            "performance": (
                "above average" if score > bm.get("industry_avg_score", 0) else "below average"
            ),
        })
    return {"benchmarks": result}


@router.get("/{mineral_id}")
def mineral_sustainability(mineral_id: str):
    if mineral_id not in MINERAL_IDS:
        raise HTTPException(404, f"Mineral '{mineral_id}' not found.")
    meta = get_mineral_metadata(mineral_id)
    data = get_historical_data(mineral_id)
    df = data[mineral_id].copy()
    demand = df["demand"]

    carbon_monthly = (demand * meta["carbon_factor"]).round(3)
    df["year"] = df["date"].dt.year
    annual_carbon = (
        df.groupby("year")["demand"].sum() * meta["carbon_factor"]
    ).round(2)

    # Projected carbon savings if sustainability score improved by 10 pts
    improvement_factor = 0.08  # 8% emissions reduction per 10pt score improvement
    projected_saving = round(
        float(demand.mean()) * meta["carbon_factor"] * improvement_factor * 12, 2
    )

    return {
        "mineral_id": mineral_id,
        "name": meta["name"],
        "sustainability_score": meta["sustainability_score"],
        "carbon_factor": meta["carbon_factor"],
        "monthly_carbon": {
            "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "values": carbon_monthly.tolist(),
        },
        "annual_carbon": annual_carbon.to_dict(),
        "total_carbon_10yr": round(float(demand.sum() * meta["carbon_factor"]), 2),
        "avg_monthly_carbon": round(float(demand.mean() * meta["carbon_factor"]), 3),
        "projected_annual_saving_if_improved": projected_saving,
        "recommendations": _get_sustainability_tips(mineral_id),
        "sdg_alignment": _get_sdg_alignment(mineral_id),
    }


def _get_sustainability_tips(mineral_id: str) -> list:
    tips_map = {
        "coal": [
            "Transition to renewable energy sources to replace thermal coal",
            "Invest in carbon capture and storage (CCS) technology",
            "Increase co-firing with biomass to reduce net emissions",
            "Accelerate phaseout timeline aligned with Paris Agreement targets",
        ],
        "iron_ore": [
            "Switch to electric arc furnaces (EAF) powered by renewables",
            "Increase scrap steel recycling rates (saves 1.4t CO2 per tonne)",
            "Optimize blast furnace efficiency with AI-driven process control",
            "Explore green hydrogen-based direct reduced iron (DRI)",
        ],
        "bauxite": [
            "Expand bauxite tailings rehabilitation programs",
            "Increase aluminium recycling — saves 95% of smelting energy",
            "Adopt low-carbon Bayer process modifications",
            "Use renewable energy for high-intensity refining operations",
        ],
        "limestone": [
            "Use supplementary cementitious materials (fly ash, slag) to reduce clinker",
            "Deploy carbon capture at cement and lime plants",
            "Optimize quarrying logistics to reduce transport emissions",
            "Explore geopolymer cement as low-carbon alternative",
        ],
        "copper": [
            "Scale copper recycling infrastructure — secondary copper uses 85% less energy",
            "Electrify mining fleet to eliminate diesel emissions",
            "Use renewable energy for smelting and refining",
            "Improve ore grade management to reduce waste rock processing",
        ],
    }
    return tips_map.get(mineral_id, ["Optimize supply chain", "Reduce waste in extraction"])


def _get_sdg_alignment(mineral_id: str) -> list:
    sdg_map = {
        "coal":      [{"sdg": "SDG 7", "desc": "Affordable & Clean Energy — transitioning away from coal"},
                      {"sdg": "SDG 13", "desc": "Climate Action — reducing carbon emissions"}],
        "iron_ore":  [{"sdg": "SDG 9",  "desc": "Industry & Infrastructure — sustainable steel"},
                      {"sdg": "SDG 12", "desc": "Responsible Consumption — scrap recycling"}],
        "bauxite":   [{"sdg": "SDG 12", "desc": "Responsible Consumption — aluminium recycling"},
                      {"sdg": "SDG 15", "desc": "Life on Land — tailings rehabilitation"}],
        "limestone": [{"sdg": "SDG 11", "desc": "Sustainable Cities — green cement"},
                      {"sdg": "SDG 13", "desc": "Climate Action — low-carbon construction"}],
        "copper":    [{"sdg": "SDG 7",  "desc": "Clean Energy — EV & renewables enabler"},
                      {"sdg": "SDG 9",  "desc": "Industry & Innovation — electrification"}],
    }
    return sdg_map.get(mineral_id, [])