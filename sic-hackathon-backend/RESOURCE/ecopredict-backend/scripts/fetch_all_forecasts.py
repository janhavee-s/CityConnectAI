import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:8000"
MODEL = "ARIMA"
HORIZON = 12

def get_all_minerals():
    """Fetch list of all minerals from backend"""
    response = requests.get(f"{BASE_URL}/api/minerals")
    response.raise_for_status()
    return response.json()["minerals"]

def get_forecast(mineral_id):
    params = {
        "model": MODEL,
        "horizon": HORIZON
    }
    response = requests.get(
        f"{BASE_URL}/api/forecast/{mineral_id}",
        params=params
    )

    if response.status_code != 200:
        print(f"⚠ Skipping {mineral_id}: {response.status_code}")
        return None

    return response.json()

def main():
    # minerals = get_all_minerals()
    minerals = [
    {"id": "coal", "name": "Coal"},
    {"id": "iron_ore", "name": "Iron Ore"},
    {"id": "bauxite", "name": "Bauxite"},
    {"id": "copper", "name": "Copper"},
    {"id": "limestone", "name": "Limestone"},
]
    
    all_rows = []

    print("\n===== FETCHING FORECASTS FOR ALL RESOURCES =====\n")

    for mineral in minerals:
        mineral_id = mineral["id"]
        mineral_name = mineral["name"]

        print(f"Fetching forecast for {mineral_name}...")

        data = get_forecast(mineral_id)

        future_dates = data["forecast"]["dates"]
        future_values = data["forecast"]["values"]
        stats = data["stats"]

        print(f"  Avg Forecast: {stats['forecast_avg']}")
        print(f"  % Change: {stats['pct_change']}%")
        print(f"  Alert: {stats['alert']}")
        print("")

        for date, value in zip(future_dates, future_values):
            all_rows.append({
                "mineral_id": mineral_id,
                "mineral_name": mineral_name,
                "date": date,
                "forecast_value": value,
                "forecast_avg": stats["forecast_avg"],
                "pct_change": stats["pct_change"],
                "alert": stats["alert"]
            })

    # Save all forecasts to CSV
    df = pd.DataFrame(all_rows)
    df.to_csv("all_resource_forecasts.csv", index=False)

    print("All forecasts saved to all_resource_forecasts.csv")

if __name__ == "__main__":
    main()