"""
EcoPredict Backend - AI-Powered Mineral Resource Demand Forecasting
FastAPI backend with ARIMA/ML forecasting, visualizations, and Ollama LLM integration.
"""
from fastapi.responses import HTMLResponse
from fastapi import Request, Form
from fastapi.templating import Jinja2Templates

from data.mineral_data import (
    MINERAL_METADATA,
    get_historical_data,
    get_combined_data,
    STATE_DISTRIBUTION
)
from models.forecasting import run_forecast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routes import forecast, analytics, sustainability, inventory, reports, llm_insights

app = FastAPI(
    title="EcoPredict API",
    description="AI-Powered Mineral Resource Demand Forecasting Backend",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)
templates = Jinja2Templates(directory="templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(forecast.router,       prefix="/api/forecast",       tags=["Forecast"])
app.include_router(analytics.router,      prefix="/api/analytics",      tags=["Analytics"])
app.include_router(sustainability.router, prefix="/api/sustainability", tags=["Sustainability"])
app.include_router(inventory.router,      prefix="/api/inventory",      tags=["Inventory"])
app.include_router(reports.router,        prefix="/api/reports",        tags=["Reports"])
app.include_router(llm_insights.router,   prefix="/api/insights",       tags=["LLM Insights"])


@app.get("/")
def root():
    return {
        "message": "EcoPredict API v2.0 — Mineral Resource Forecasting",
        "status": "online",
        "docs": "/api/docs"
    }


@app.get("/api/health")
def health():
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/api/minerals")
def get_minerals():
    from data.mineral_data import MINERAL_METADATA
    return {"minerals": MINERAL_METADATA}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

templates = Jinja2Templates(directory="templates")

# ─────────────────────────────────────────────
# DASHBOARD (National + State View)
# ─────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_get(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "minerals": MINERAL_METADATA,
        "selected_model": "ARIMA",
        "selected_horizon": 6,
        "selected_state_view": False,
        "states": []
    })


@app.post("/dashboard", response_class=HTMLResponse)
async def dashboard_post(
    request: Request,
    mineral_id: str = Form(...),
    model: str = Form(...),
    horizon: int = Form(...),
    state_view: str = Form(None),
    state: str = Form(None),
):

    # Limit models
    if model not in ["ARIMA", "RandomForest"]:
        model = "ARIMA"

    # Enforce horizon 1–12
    horizon = min(max(horizon, 1), 12)

    state_view_enabled = state_view is not None

    # ── Load Data ─────────────────────────────
    if state_view_enabled and state:
        df_all = get_combined_data(state_level=True)
        df = df_all[
            (df_all["mineral"] == mineral_id) &
            (df_all["state"] == state)
        ].copy()
        series = df["demand"]
        scope = f"{state} (State Level)"
    else:
        data = get_historical_data(mineral_id)
        df = data[mineral_id]
        series = df["demand"]
        state = None
        scope = "National Level"

    # ── Run Forecast ──────────────────────────
    result = run_forecast(series, model, horizon)

    current = float(series.iloc[-1])
    avg = float(sum(result["forecast"]) / len(result["forecast"]))
    peak = float(max(result["forecast"]))
    pct = ((result["forecast"][-1] - current) / current) * 100
    accuracy = result.get("accuracy", {})

    meta = next(m for m in MINERAL_METADATA if m["id"] == mineral_id)

    # ── Build Detailed LLM Prompt ─────────────
    prompt = f"""
You are EcoPredict, an expert AI system for mineral demand analytics.

Scope: {scope}
Mineral: {meta["name"]} ({meta["category"]})
Forecast Model: {model}
Horizon: {horizon} months

Current Demand: {current:.1f} MT
Forecast Average: {avg:.1f} MT
Forecast Peak: {peak:.1f} MT
Projected Change: {pct:+.2f}%

Model Accuracy (MAPE): {accuracy.get("mape","N/A")}%
Stock: {meta["stock_current"]} MT
Recommended Stock: {meta["stock_recommended"]} MT
Sustainability Score: {meta["sustainability_score"]}/100
Carbon Factor: {meta["carbon_factor"]} tCO2/MT

Provide:
1. Executive Forecast Summary (3 sentences)
2. Key Demand Drivers (3 bullets)
3. Inventory Recommendation
4. Risk Factors (2 bullets)
5. Sustainability Implication

Professional tone. Under 400 words.
"""

    # ── Call Ollama ───────────────────────────
    llm_response = None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                }
            )
            resp.raise_for_status()
            llm_response = resp.json().get("response", "").strip()
    except Exception:
        llm_response = None

    # ── Fallback If Ollama Not Running ───────
    if not llm_response:
        direction = "increase" if pct > 0 else "decline"
        llm_response = (
            f"{scope} forecast for {meta['name']} using {model} model.\n\n"
            f"Demand is projected to {direction} by {pct:+.2f}% over {horizon} months.\n"
            f"Current demand: {current:.1f} MT.\n"
            f"Average forecast: {avg:.1f} MT.\n\n"
            f"Run `ollama serve` for detailed AI explanation."
        )

    # ── State List For Dropdown ──────────────
    states = list(STATE_DISTRIBUTION[mineral_id].keys())

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "minerals": MINERAL_METADATA,
        "insight": llm_response,
        "summary": {
            "current": round(current, 2),
            "forecast_avg": round(avg, 2),
            "pct_change": round(pct, 2)
        },
        "selected_mineral": mineral_id,
        "selected_model": model,
        "selected_horizon": horizon,
        "selected_state_view": state_view_enabled,
        "selected_state": state,
        "states": states
    })