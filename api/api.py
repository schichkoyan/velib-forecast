from __future__ import annotations
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import requests, json, datetime, time
import lightgbm as lgb

app = FastAPI(title="Velib Forecast API")

# ==== Artifacts ====
ARTIF = Path("artifacts/v0_2_weather")
FEAT_COLS = json.load(open(ARTIF / "feat_cols_delta.json"))
CONFIG = {}
cfg_path = ARTIF / "config.json"
if cfg_path.exists():
    try:
        CONFIG = json.load(open(cfg_path))
    except Exception:
        CONFIG = {}

TARGET_KIND = CONFIG.get("target_kind", "delta_occ")     # "delta_occ" or "delta_bikes"
GAMMAS = {str(k): v for k, v in CONFIG.get("gamma", {"15":1.0,"30":1.0,"60":1.0}).items()}

# Load one model per horizon if available
def _booster(p: Path) -> Optional[lgb.Booster]:
    return lgb.Booster(model_file=str(p)) if p.exists() else None

MODELS: Dict[int, lgb.Booster] = {}
for h in (15, 30, 60):
    m = _booster(ARTIF / f"lgbm_delta_h{h}.txt")
    if m is not None:
        MODELS[h] = m
if not MODELS:
    # fallback for older naming
    m30 = _booster(ARTIF / "model_30.txt")
    if m30:
        MODELS[30] = m30
if not MODELS:
    raise FileNotFoundError("No LightGBM models found in artifacts/v0_2_weather/")

# ==== Weather (cached) ====
_WEATHER_CACHE: Dict[str, float | dict] = {"ts": 0.0, "val": {}}
def fetch_current_weather(ttl_sec: int = 90) -> dict:
    now = time.time()
    if now - float(_WEATHER_CACHE["ts"]) < ttl_sec and _WEATHER_CACHE["val"]:
        return _WEATHER_CACHE["val"]  # cached
    url = "https://api.open-meteo.com/v1/forecast"
    params = dict(latitude=48.8566, longitude=2.3522, timezone="UTC",
                  current="temperature_2m,precipitation,wind_speed_10m")
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        j = r.json().get("current", {})
    except Exception:
        j = {}
    precip = float(j.get("precipitation", 0.0) or 0.0)
    out = {
        "temperature_2m": float(j.get("temperature_2m", 0.0) or 0.0),
        "precipitation": precip,
        "wind_speed_10m": float(j.get("wind_speed_10m", 0.0) or 0.0),
        "is_rain": 1 if precip > 0.1 else 0,
    }
    _WEATHER_CACHE.update({"ts": now, "val": out})
    return out

# ==== Schemas ====
class InputRow(BaseModel):
    station_id: str
    bikes_available: float
    capacity: float
    ts: datetime.datetime
    # Optional 5-min history (oldest→newest), up to last 12 values (60min)
    history_5min: Optional[List[float]] = None

# ==== Feature building ====
def _occ(x: float, cap: float) -> float:
    if not cap or cap <= 0:
        return 0.0
    return float(np.clip(x / cap, 0.0, 1.0))

def _time_feats(ts: datetime.datetime) -> dict:
    ts = pd.Timestamp(ts).tz_localize("UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).tz_convert("UTC")
    h, d = ts.hour, ts.dayofweek
    return {
        "hour_sin": float(np.sin(2*np.pi*h/24)),
        "hour_cos": float(np.cos(2*np.pi*h/24)),
        "dow": int(d),
        "is_weekend": int(d >= 5),
    }

def _history_feats(history: Optional[List[float]], cap: float, occ_now: float) -> dict:
    feats = {}
    hist = list(history or [])[-12:]  # last 60min
    hist_occ = [_occ(v, cap) for v in hist]
    def lag_k(k, default): return float(hist_occ[-k]) if len(hist_occ) >= k else float(default)
    feats["occ_lag_5"]  = lag_k(1, occ_now)
    feats["occ_lag_10"] = lag_k(2, occ_now)
    feats["occ_lag_15"] = lag_k(3, occ_now)
    feats["occ_lag_30"] = lag_k(6, occ_now)
    feats["occ_lag_60"] = lag_k(12, occ_now)
    def roll_mean(n): return float(np.mean(hist_occ[-min(n,len(hist_occ)):])) if hist_occ else float(occ_now)
    feats["occ_roll_60"]  = roll_mean(12)
    feats["occ_roll_120"] = roll_mean(24)
    feats["occ_roll_180"] = roll_mean(36)
    def delta_k(k): return float(occ_now - lag_k(k, occ_now))
    feats["occ_delta_5"]  = delta_k(1)
    feats["occ_delta_15"] = delta_k(3)
    feats["occ_delta_30"] = delta_k(6)
    feats["occ_delta_60"] = delta_k(12)
    return feats

def build_feature_row(inp: InputRow, weather: dict | None = None) -> pd.DataFrame:
    weather = weather or fetch_current_weather()
    occ_now = _occ(inp.bikes_available, inp.capacity)
    base = {
        "station_id": inp.station_id,
        "bikes_available": float(inp.bikes_available),
        "capacity": float(inp.capacity),
        "occ_now": occ_now,
        **_time_feats(inp.ts),
        **weather,
        **_history_feats(inp.history_5min, inp.capacity, occ_now),
    }
    # rough station encodings fallback (replace by real encodings if you export them)
    base.setdefault("sta_mean_occ", occ_now)
    base.setdefault("sta_hdh_occ",  occ_now)
    # align to expected feature order
    row = {c: float(base.get(c, 0.0)) for c in FEAT_COLS}
    return pd.DataFrame([row], columns=FEAT_COLS).astype("float32")

def _predict_for_horizon(h: int, X: pd.DataFrame, bikes_now: float, capacity: float) -> Dict[str, float]:
    mdl = MODELS[h]
    delta = float(mdl.predict(X)[0])
    delta *= float(GAMMAS.get(str(h), 1.0) or 1.0)
    # convert Δocc → Δbikes if needed
    delta_bikes = delta * capacity if TARGET_KIND == "delta_occ" else delta
    y_hat = float(np.clip(bikes_now + delta_bikes, 0, capacity))
    return {"predicted_bikes": round(y_hat, 3), "delta_model": round(delta, 6)}

# ==== Endpoints ====
@app.get("/health")
def health():
    return {
        "models_loaded": sorted(MODELS.keys()),
        "n_features": len(FEAT_COLS),
        "target_kind": TARGET_KIND,
        "weather_cached": bool(_WEATHER_CACHE["val"]),
    }

@app.post("/predict/{horizon}")
def predict(horizon: int, row: InputRow):
    if horizon not in MODELS:
        return {"error": f"Model for horizon {horizon} not available. Have: {sorted(MODELS.keys())}"}
    w = fetch_current_weather()
    X = build_feature_row(row, w)
    out = _predict_for_horizon(horizon, X, row.bikes_available, row.capacity)
    return {"horizon": horizon, **out}

@app.post("/predict_all")
def predict_all(row: InputRow):
    w = fetch_current_weather()
    X = build_feature_row(row, w)  # build once → reuse for all horizons
    res = {}
    for h in sorted(MODELS.keys()):
        res[str(h)] = _predict_for_horizon(h, X, row.bikes_available, row.capacity)
    return {"predictions": res}

class BatchRequest(BaseModel):
    rows: List[InputRow]
    horizons: Optional[List[int]] = None  # default: available models

@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    horizons = req.horizons or sorted(MODELS.keys())
    w = fetch_current_weather()
    results = []
    for r in req.rows:
        X = build_feature_row(r, w)
        pred = {str(h): _predict_for_horizon(h, X, r.bikes_available, r.capacity) for h in horizons if h in MODELS}
        results.append({"station_id": r.station_id, "ts": r.ts, "predictions": pred})
    return {"items": results}