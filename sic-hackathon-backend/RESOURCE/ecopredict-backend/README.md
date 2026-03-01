🌍 EcoPredict – AI Mineral Demand Forecasting

EcoPredict is a FastAPI-based backend system for forecasting mineral demand at national and state levels.
It combines:
• ARIMA (statistical time-series forecasting)
• Random Forest (machine learning forecasting)
• AI-generated executive insights using Ollama
• Sustainability and carbon impact metrics
The goal: structured forecasting with explainable outputs.

⚙️ Installation Guide
1️⃣ Create a Virtual Environment
Using Conda:
conda create -n ecopredict python=3.11
conda activate ecopredict

Or using venv:
python -m venv ecopredict
source ecopredict/bin/activate   # Mac/Linux
ecopredict\Scripts\activate      # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

🚀 Run the Backend Server
uvicorn main:app --reload --port 8000

Access:
Dashboard → http://127.0.0.1:8000/dashboard
API Documentation → http://127.0.0.1:8000/api/docs

🤖 Enable AI Explanations (Optional)
EcoPredict supports AI-generated executive summaries using Ollama.

Step 1: Install Ollama
Download from:
https://ollama.com/ or curl -fsSL https://ollama.com/install.sh | sh

Step 2: Start Ollama Server
ollama serve

Step 3: Pull Language Model
ollama pull llama3
If Ollama is not running, the system automatically falls back to rule-based explanations.

📊 Dashboard Usage
1. Select Mineral
2. Choose Model (ARIMA or RandomForest)
3. Set Forecast Horizon (1–12 months)
4. (Optional) Enable State-wise View
5. Click Generate

The system produces:
• Forecast values
• Trend insights
• Sustainability metrics
• AI-generated executive explanation (if enabled)

📁 Export State-Level Dataset
Generate dataset:
python -m scripts.export_state_dataset

Output file:
state_level_mineral_dataset_2014_2025.csv

Dataset Columns
• date
• mineral
• state
• demand
• trend
• seasonal
• shock
• yoy_change
• mom_change

📡 Key API Endpoint
Forecast Insight
POST /api/insights/forecast

Example request:
curl -X POST http://127.0.0.1:8000/api/insights/forecast \
  -H "Content-Type: application/json" \
  -d '{"mineral_id":"coal","model":"ARIMA","horizon":6}'

Request Parameters
• mineral_id → e.g., "coal"
• model → "ARIMA" or "RandomForest"
• horizon → Forecast months (1–12)

🧠 System Architecture
Dashboard (Jinja2)
        ↓
FastAPI Routes
        ↓
Forecasting Engine (ARIMA / RF)
        ↓
Data Layer
        ↓
Ollama (LLM-based Interpretation)

🛠 Requirements
Python 3.11 recommended
FastAPI
Uvicorn
scikit-learn
statsmodels
Ollama (optional, for AI explanations)