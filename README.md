# 🚲 Vélib' Forecast — Bike Availability Prediction (15/30/60 min)

**Goal.** Predict the number of available bikes at Paris Vélib’ stations for the next **15 / 30 / 60 minutes**.  
**Stack.** Python · LightGBM (Δ-forecast) · FastAPI · MLflow · (optional) Spark for feature jobs.  
**Data.** Real GBFS snapshots (every 5 minutes), ~3.3M rows, ~1468 stations.

---

## 📦 Project Structure

.
├── src/velib_ml/            # reusable package (data, features, training, inference)
├── scripts/                 # CLI scripts
│   ├── collect_velib_gbfs.py
│   ├── prepare_from_snapshots.py
│   ├── baseline.py
│   └── train.py
├── api/api.py               # FastAPI endpoint
├── notebooks/velib_v01.ipynb
└── artifacts/v0_1/          # light artifacts kept in Git
    ├── metrics.csv
    ├── config.json
    ├── feat_cols_delta.json
    └── sample_features.csv

---

## ✨ What’s inside

- **Feature engineering:** temporal (hour/day sin/cos, weekend), lags (5–60), rolling means (15/30/60/120/180), deltas (5/15/30/60), `occ_now = bikes/capacity`.
- **Model:** LightGBM trained on **Δ(occupancy)** with **gamma calibration** on validation; evaluated in **MAE (bikes)**.
- **Baselines:** Naïve (last value), EWMA.
- **Tracking:** MLflow (params, metrics, artifacts).
- **Serving:** FastAPI endpoint `/predict/{horizon}`.

---

## 📈 Results (MAE, bikes)

| Horizon | Naïve | LightGBM Δ |
|:------:|-----:|-----------:|
| 15 min | 0.750 | 0.754 |
| 30 min | 1.156 | 1.156 |
| 60 min | 1.739 | **1.714** |

Naïve is a strong short-horizon baseline; the model improves at +60 min.

---

## 🚀 Quickstart

### 0) Install
pip install -e .
pip install -r requirements.txt   # or: pip install mlflow "fastapi[standard]" lightgbm pandas scikit-learn

### 1) Collect data (GBFS snapshots → parquet files)
python scripts/collect_velib_gbfs.py --out data/raw/velib_long

### 2) Build a single timeseries CSV (5-min grid)
python scripts/prepare_from_snapshots.py --in data/raw/velib_long --out data/raw/velib_timeseries_5min.csv

### 3) Baselines (Naïve & EWMA)
python scripts/baseline.py --data data/raw/velib_timeseries_5min.csv --out artifacts/baseline_v0

### 4) Train the model (+ MLflow logging)
mlflow ui --port 5000 &   # optional UI at http://127.0.0.1:5000
python scripts/train.py --data data/raw/velib_timeseries_5min.csv --out artifacts/v0_1 --threads 2 --no-ema --no-sta

Outputs:
- artifacts/v0_1/metrics.csv — final MAE.
- artifacts/v0_1/feat_cols_delta.json — required feature columns (order matters).
- artifacts/v0_1/config.json — horizons, splits, gammas.
- artifacts/v0_1/sample_features.csv — one valid feature row for API tests.
- (Models are stored locally; heavy artifacts are not pushed to Git by default.)

### 5) Serve predictions (FastAPI)
uvicorn api.api:app --reload
# → http://127.0.0.1:8000/docs

Create a JSON payload from the sample row and call the API:
python - <<'PY'
import pandas as pd, json
row = pd.read_csv('artifacts/v0_1/sample_features.csv').iloc[0].fillna(0.0).to_dict()
json.dump({"features": row}, open('artifacts/v0_1/sample_payload.json','w'))
print("Wrote artifacts/v0_1/sample_payload.json")
PY

curl -s -X POST http://127.0.0.1:8000/predict/30 -H "Content-Type: application/json" -d @artifacts/v0_1/sample_payload.json

---

## 🔍 MLflow

- Start UI: mlflow ui --port 5000 → http://localhost:5000
- The training script logs: params (splits, flags), metrics per horizon (`mae_*`, `gamma_*`, `best_iter_*`), and artifacts.

---

## 🧪 Repro Tips

- Time-based splits: 70/30 train/test, then 85/15 train/val inside train.
- Evaluate in bikes (not occupancy): MAE(capacity * occ_hat, capacity * occ_true).
- Keep feat_cols_delta.json in sync with training.

---

## 📌 Roadmap

- Weather join (rain/temp) → extra features.
- Spark batch job (window features, Parquet/Delta).
- PyTorch TCN multi-horizon baseline.
- Dockerize API + simple deployment.
- Monitoring & drift checks.

---

## 📜 License

MIT — free to use, modify, and share.
