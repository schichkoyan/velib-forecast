import json, numpy as np, lightgbm as lgb
from pathlib import Path

def load_artifacts(dirpath):
    d = Path(dirpath)
    feat_cols = json.load(open(d/"feat_cols_delta.json"))
    cfg = json.load(open(d/"config.json"))
    models = {int(h): lgb.Booster(model_file=str(d/f"lgbm_delta_h{h}.txt")) for h in cfg["horizons"]}
    gammas = cfg["gammas"]
    return models, feat_cols, gammas

def predict_from_features(row_df, booster, feat_cols, gamma):
    delta = booster.predict(row_df[feat_cols])[0]
    occ_hat = float(np.clip(row_df["occ_now"].iloc[0] + gamma*delta, 0, 1))
    return occ_hat * float(row_df["capacity"].iloc[0]), occ_hat
