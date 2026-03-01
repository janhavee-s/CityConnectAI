"""
/api/insights — Ollama LLM Integration

Endpoints:
  GET  /api/insights/health
  POST /api/insights/forecast
  POST /api/insights/sector-index
  POST /api/insights/sustainability
  POST /api/insights/anomaly
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import numpy as np
import httpx

from data.mineral_data import get_historical_data, get_mineral_metadata, MINERAL_IDS
from models.forecasting import run_forecast

router = APIRouter()

OLLAMA_URL    = "http://localhost:11434"
DEFAULT_MODEL = "llama3"


# ── Schemas ───────────────────────────────────────────────────────────────────

class ForecastInsightRequest(BaseModel):
    mineral_id:    str
    model:         str = "ARIMA"
    horizon:       int = 6
    extra_context: Optional[str] = None
    ollama_model:  Optional[str] = None


class SectorIndexInsightRequest(BaseModel):
    crdi:                float
    sustainability_index: float
    mineral_scores:       dict
    ollama_model:         Optional[str] = None


class SustainabilityInsightRequest(BaseModel):
    mineral_id:   str
    ollama_model: Optional[str] = None


class AnomalyInsightRequest(BaseModel):
    mineral_id:   str
    ollama_model: Optional[str] = None


# ── Ollama client ─────────────────────────────────────────────────────────────

async def _call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 600},
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except Exception:
        return None


# ── Fallbacks ─────────────────────────────────────────────────────────────────

def _fallback_forecast(name, current, avg, pct, horizon, model):
    direction = "increase" if pct > 0 else "decrease"
    magnitude = "significant" if abs(pct) > 10 else "moderate" if abs(pct) > 5 else "slight"
    return (
        f"📊 **{name} Demand Forecast Summary ({model}, {horizon}-Month Horizon)**\n\n"
        f"The {model} model projects a {magnitude} {direction} in {name} demand over the next "
        f"{horizon} months. Current demand: **{current:.1f} MT**, forecast average: **{avg:.1f} MT** "
        f"({pct:+.1f}%).\n\n"
        f"*Connect Ollama (`ollama serve` + `ollama pull llama3`) for AI-powered detailed analysis.*"
    )


def _fallback_sector(crdi, sustainability_index, mineral_scores):
    level = (
        "critical pressure" if crdi > 80 else
        "elevated demand"   if crdi > 65 else
        "moderate demand"   if crdi > 40 else
        "low demand"
    )
    top = max(mineral_scores.items(), key=lambda x: x[1].get("demand_pressure", 0))
    return (
        f"🌍 **Sector Resource Demand Index (CRDI): {crdi}/100**\n\n"
        f"The sector is experiencing **{level}**. "
        f"Highest pressure: {top[0].replace('_',' ').title()} at {top[1].get('demand_pressure',0):.1f}/100.\n\n"
        f"Sustainability Index: {sustainability_index}/100.\n\n"
        f"*Connect Ollama for AI-powered narrative analysis.*"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
async def ollama_health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp   = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"ollama_available": True, "models": models, "url": OLLAMA_URL}
    except Exception:
        return {
            "ollama_available": False,
            "message": "Run `ollama serve` then `ollama pull llama3`",
            "url": OLLAMA_URL,
        }


@router.post("/forecast")
async def forecast_insight(req: ForecastInsightRequest):
    if req.mineral_id not in MINERAL_IDS:
        raise HTTPException(404, f"Mineral '{req.mineral_id}' not found.")

    data     = get_historical_data(req.mineral_id)
    df       = data[req.mineral_id]
    meta     = get_mineral_metadata(req.mineral_id)
    result   = run_forecast(df["demand"], req.model, req.horizon)
    current  = float(df["demand"].iloc[-1])
    avg      = float(np.mean(result["forecast"]))
    peak     = float(max(result["forecast"]))
    pct      = ((result["forecast"][-1] - current) / current) * 100
    accuracy = result.get("accuracy", {})

    prompt = f"""You are EcoPredict, an expert AI system for mineral supply chain analytics.

Mineral: {meta["name"]} ({meta["category"]})
Forecast Model: {req.model} | Horizon: {req.horizon} months
Current Demand: {current:.1f} MT | Forecast Avg: {avg:.1f} MT | Peak: {peak:.1f} MT
Change: {pct:+.1f}% | MAPE: {accuracy.get("mape","N/A")}%
Stock: {meta["stock_current"]} / Recommended: {meta["stock_recommended"]}
Sustainability Score: {meta["sustainability_score"]}/100 | Carbon Factor: {meta["carbon_factor"]} tCO2/MT
{f"Context: {req.extra_context}" if req.extra_context else ""}

Provide:
1. Forecast Summary (2-3 sentences)
2. Key Drivers (2-3 bullets)
3. Supply Chain Recommendations (2-3 bullets)
4. Risk Factors (2 bullets)
5. Sustainability Note

Professional tone, under 400 words."""

    model_to_use = req.ollama_model or DEFAULT_MODEL
    llm_response = await _call_ollama(prompt, model_to_use)

    return {
        "mineral_id":   req.mineral_id,
        "mineral_name": meta["name"],
        "model_used":   model_to_use if llm_response else "rule-based",
        "source":       "ollama" if llm_response else "fallback",
        "insight":      llm_response or _fallback_forecast(meta["name"], current, avg, pct, req.horizon, req.model),
        "forecast_summary": {"current": current, "forecast_avg": avg, "pct_change": round(pct, 2)},
    }


@router.post("/sector-index")
async def sector_index_insight(req: SectorIndexInsightRequest):
    prompt = f"""You are EcoPredict, an AI expert in mineral market analytics.

CRDI: {req.crdi}/100 | Sustainability Index: {req.sustainability_index}/100

Mineral Demand Pressure:
{json.dumps({k: {"pressure": v.get("demand_pressure"), "yoy": v.get("yoy_change"), "name": v.get("name")} for k, v in req.mineral_scores.items()}, indent=2)}

Provide:
1. Index Interpretation
2. Top 2 Minerals Driving the Index
3. Strategic Outlook (3-6 months)
4. One Key Procurement Recommendation

Under 300 words, professional tone."""

    model_to_use = req.ollama_model or DEFAULT_MODEL
    llm_response = await _call_ollama(prompt, model_to_use)

    return {
        "source":     "ollama" if llm_response else "fallback",
        "model_used": model_to_use if llm_response else "rule-based",
        "insight":    llm_response or _fallback_sector(req.crdi, req.sustainability_index, req.mineral_scores),
    }


@router.post("/sustainability")
async def sustainability_insight(req: SustainabilityInsightRequest):
    if req.mineral_id not in MINERAL_IDS:
        raise HTTPException(404)

    meta         = get_mineral_metadata(req.mineral_id)
    data         = get_historical_data(req.mineral_id)
    df           = data[req.mineral_id]
    total_carbon = round(float(df["demand"].sum()) * meta["carbon_factor"], 2)

    prompt = f"""You are EcoPredict, an AI sustainability advisor.

Mineral: {meta["name"]} | Score: {meta["sustainability_score"]}/100
Carbon Factor: {meta["carbon_factor"]} tCO2/MT | 10-Year Total Carbon: {total_carbon} tCO2
Category: {meta["category"]}

Provide:
1. Environmental impact profile
2. Key decarbonisation opportunities
3. SDG alignment (SDG 7, 9, 12, 13, 15)
4. Circular economy potential

Under 250 words."""

    model_to_use = req.ollama_model or DEFAULT_MODEL
    llm_response = await _call_ollama(prompt, model_to_use)

    return {
        "mineral_id": req.mineral_id,
        "source":     "ollama" if llm_response else "fallback",
        "insight":    llm_response or (
            f"Sustainability score for {meta['name']}: {meta['sustainability_score']}/100. "
            f"Estimated 10-year carbon: {total_carbon} tCO2. "
            f"Focus on circular economy and renewable energy integration."
        ),
    }


@router.post("/anomaly")
async def anomaly_insight(req: AnomalyInsightRequest):
    if req.mineral_id not in MINERAL_IDS:
        raise HTTPException(404)

    meta     = get_mineral_metadata(req.mineral_id)
    data     = get_historical_data(req.mineral_id)
    df       = data[req.mineral_id].copy()
    demand   = df["demand"]
    z_scores = ((demand - demand.mean()) / demand.std()).abs()
    anomalies = df[z_scores > 2.5][["date", "demand"]].copy()
    anomalies["z_score"] = z_scores[z_scores > 2.5].values

    anomaly_list = [
        {"date": str(r.date)[:10], "demand": round(r.demand, 2), "z_score": round(r.z_score, 2)}
        for _, r in anomalies.iterrows()
    ]

    prompt = f"""You are EcoPredict, an AI analyst for mineral demand anomaly detection.

Mineral: {meta["name"]}
Anomalies (Z > 2.5): {json.dumps(anomaly_list, indent=2)}
Stats: Mean={round(float(demand.mean()),1)}, Std={round(float(demand.std()),1)}, Min={round(float(demand.min()),1)}, Max={round(float(demand.max()),1)}

For each anomaly or pattern provide:
1. Likely cause
2. Impact on forecasting reliability
3. Recommended action

Under 250 words."""

    model_to_use = req.ollama_model or DEFAULT_MODEL
    llm_response = await _call_ollama(prompt, model_to_use)

    return {
        "mineral_id":         req.mineral_id,
        "anomalies_detected": len(anomaly_list),
        "anomalies":          anomaly_list,
        "source":             "ollama" if llm_response else "fallback",
        "insight":            llm_response or (
            f"Detected {len(anomaly_list)} anomalies in {meta['name']} demand data. "
            f"Likely caused by economic shocks or supply disruptions. "
            f"Use Ensemble or Random Forest models which are more robust to outliers."
        ),
    }
