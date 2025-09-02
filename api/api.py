from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import json, lightgbm as lgb, numpy as np

ART = Path("artifacts/v0_1")
feat_cols_delta = json.load(open(ART / "feat_cols_delta.json"))
gammas = json.load(open(ART / "config.json"))["gammas"]

def load_model(h): return lgb.Booster(model_file=str(ART / f"lgbm_delta_h{h}.txt"))

app = FastAPI()
models_cache = {}

class Payload(BaseModel):
    features: dict  # clés = feat_cols_delta

@app.on_event("startup")
def _load_all():
    for h in [15,30,60]:
        models_cache[h] = load_model(h)

@app.post("/predict/{horizon}")
def predict(horizon: int, payload: Payload):
    assert horizon in (15,30,60), "horizon must be 15/30/60"
    row = pd.DataFrame([payload.features])
    row = row.reindex(columns=feat_cols_delta, fill_value=np.nan)
    booster = models_cache[horizon]
    delta = booster.predict(row, num_iteration=booster.best_iteration)[0]
    occ_hat = float(np.clip(row["occ_now"].iloc[0] + gammas[str(horizon)]*delta, 0, 1))
    y_hat = occ_hat * float(row["capacity"].iloc[0])
    return {"y_hat_bikes": round(y_hat, 3), "occ_hat": round(occ_hat, 4)}
