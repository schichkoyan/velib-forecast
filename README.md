# 🚲 Vélib' ML Forecast — Bike Availability Prediction

**Goal.** Predict the number of available bikes at Paris Vélib’ stations for the next **15 / 30 / 60 minutes**.  
**Stack.** Python · LightGBM (Δ-forecast) · FastAPI API serving · MLflow tracking · (optional) Spark for scalable features.  
**Data.** Real Vélib’ GBFS snapshots (5-minute frequency), ~3.3M rows across ~1468 stations.

---

## 📦 Project Structure

```bash

.
├── src/velib_ml/             # reusable package (data, features, training, inference)
├── scripts/                  # CLI scripts (collect, weather, train)
│   ├── collect_velib_gbfs.py
│   ├── fetch_weather.py
│   └── train.py
├── api/api.py                # FastAPI app serving predictions
├── notebooks/new_velib.ipynb # main notebook (EDA + training + eval)
└── artifacts/v0_2_weather/   # saved models, metrics & sample features

```

---

## ✨ Key Features

- **Feature engineering**  
  - Temporal: hour/day (sin/cos), weekend flag  
  - History: lags (5–60min), rolling means (1–3h), deltas (5–60min)  
  - Station encodings (historical occupancy stats)  
  - Weather: temperature, precipitation, wind, rain indicator  

- **Model**  
  - LightGBM trained on **Δ(occupancy)** then reconstructed to bikes available  
  - Validated with early stopping, evaluated in **MAE (bikes)**  
  - Baselines: last value (naïve), exponential moving average (EWMA)  

- **Serving**  
  - FastAPI endpoint `/predict_batch` returns forecasts for multiple stations  
  - Weather automatically fetched from Open-Meteo API  

---

## 📊 Results

| Horizon | Naïve MAE | LightGBM (Δ, +weather) MAE |
|---------|-----------|-----------------------------|
| 15 min  | 0.749     | **0.749** |
| 30 min  | 1.156     | **1.152** |
| 60 min  | 1.739     | **1.723** |

📝 **Observation:** Weather features bring small but consistent improvements, especially for 30–60 min horizons.

---

## 🚀 Quickstart

### 1. Install
```bash

pip install -e .

2. Collect data

python scripts/collect_velib_gbfs.py --out data/raw/velib_long
python scripts/fetch_weather.py --start 2025-08-20 --end 2025-09-01 --out data/external/weather_hourly.csv

3. Train

python scripts/train.py --input data/raw/velib_timeseries_5min.csv --outdir artifacts/v0_2_weather



Artifacts produced:
	•	lgbm_delta_h15.txt, ..._h30.txt, ..._h60.txt — trained models
	•	metrics.csv — performance summary
	•	feat_cols_delta.json — feature list (used by API)
	•	sample_features.csv — ready-to-use test row

4. Serve API

uvicorn api.api:app --reload
# Open http://127.0.0.1:8000/docs

```

⸻

🌐 API Usage

Example request (Swagger UI or curl):

Request

```bash

curl -X POST "http://127.0.0.1:8000/predict_batch?horizon=30" \
     -H "Content-Type: application/json" \
     -d '{
           "items": [
             {
               "station_id": "1002059045",
               "bikes_available": 7,
               "capacity": 27,
               "ts": "2025-09-02T16:00:00+00:00"
             },
             {
               "station_id": "99950133",
               "bikes_available": 12,
               "capacity": 55,
               "ts": "2025-09-02T16:00:00+00:00"
             }
           ]
         }'

```

Response

```bash

{
  "items": [
    {
      "station_id": "1002059045",
      "ts": "2025-09-02T16:00:00+00:00",
      "predictions": {
        "30": {
          "predicted_bikes": 6.99,
          "delta_model": -0.01
        }
      }
    },
    {
      "station_id": "99950133",
      "ts": "2025-09-02T16:00:00+00:00",
      "predictions": {
        "30": {
          "predicted_bikes": 12.08,
          "delta_model": 0.08
        }
      }
    }
  ]
}

```

⸻

🌍 Relevance
	•	Bike-sharing demand forecasting is essential for rebalancing operations (trucks moving bikes).
	•	Project demonstrates real-time data pipelines, ML training, and serving.
	•	Includes optional Spark batch job for scalable feature engineering.

⸻

📌 Roadmap
	•	Streamlit dashboard with live forecasts & maps
	•	Deploy API/Dashboard on Hugging Face Spaces / Render
	•	Add PyTorch baseline (RNN/TCN) for comparison
	•	Monitoring & drift detection

⸻

📜 License

MIT — free to use, modify, and share.