# 🚲 Velib ML Forecasting Project

## 📌 Project Overview
This project predicts **bike availability at Velib stations in Paris** using machine learning.  
It integrates **real-time station data** with **weather features** and serves predictions through an **API**.

The project demonstrates:
- End-to-end **data pipeline** (collection, processing, training, serving).
- **ML engineering best practices** (LightGBM, feature engineering, evaluation).
- **MLOps components** (MLflow tracking, artifact versioning).
- **Deployment** with FastAPI for real-time predictions.

---

## 🗂 Repository Structure

.
├── api/                   # FastAPI app serving predictions
├── artifacts/v0_2_weather # Saved models + configs + sample features
├── data/                  # Raw & sample data (tiny CSVs only)
├── notebooks/             # Jupyter notebooks (EDA, experiments)
├── scripts/               # Data collection, feature building, training
├── src/velib_ml/          # Reusable Python package (data, features, training)
└── README.md

---

## ⚙️ Pipeline

### 1. Data Collection
- `collect_velib_gbfs.py` → collects **live Velib GBFS snapshots**.  
- `fetch_weather.py` → retrieves **hourly weather** from Open-Meteo API.  

### 2. Feature Engineering
- Time-based features: hour of day, day of week, weekend flag.  
- Station history: lags (5–60min), rolling means, deltas.  
- Weather: temperature, precipitation, wind, rain indicator.  

### 3. Training
- `train.py` trains **LightGBM models** per horizon (15, 30, 60 minutes).  
- Evaluation with **MAE** baseline vs model.  
- Artifacts stored in `artifacts/v0_2_weather/`.

### 4. Serving
- `api/api.py` exposes a **FastAPI endpoint**:  
  - Single or batch predictions.  
  - JSON output with predicted bikes and deltas.  
  - Weather automatically integrated.

---

## 🚀 Quickstart

```bash
# Clone repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run training (optional)
python scripts/train.py --input data/raw/velib_timeseries_5min.csv --outdir artifacts/v0_2_weather

# Launch API
uvicorn api.api:app --reload

```

Test the API (Swagger UI):

http://127.0.0.1:8000/docs

⸻

📊 Results

Example MAE (absolute error in predicted bikes):
	•	Horizon 15 min → ~0.75
	•	Horizon 30 min → ~1.15
	•	Horizon 60 min → ~1.70

⸻

🌍 Relevance
	•	Bike-sharing demand forecasting is critical for rebalancing operations (trucks moving bikes across stations).
	•	Shows ability to handle real-time data, ML models, and deployment pipelines.

⸻

🔧 Tech Stack
	•	Python (pandas, numpy, scikit-learn, lightgbm)
	•	FastAPI for serving
	•	MLflow for tracking experiments
	•	PySpark (optional) for scalable feature engineering
	•	Jupyter for exploration

⸻

📌 Next Steps
	•	Add Streamlit dashboard (station map + live predictions).
	•	Deploy API/Dashboard on Hugging Face Spaces or Render.
	•	Experiment with Transformers (XGBoost, PyTorch).

⸻

👤 Author

Sarkis Chichkoyan
Data Scientist / ML Engineer
LinkedIn • GitHub

---

👉 Tu peux copier-coller ça dans ton `README.md`.  
Ensuite on peut rajouter **captures d’écran** (Swagger UI, notebook, résultats plots) pour le rendre encore plus visuel.  

Veux-tu que je t’ajoute aussi une **section avec des exemples de requêtes API (curl + JSON input/output)** pour que les visiteurs puissent tester direct ?