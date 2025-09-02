import numpy as np, pandas as pd
from .config import FREQ_MIN, HORIZONS

def make_features(df: pd.DataFrame, use_ema=False) -> pd.DataFrame:
    feat = df.copy().sort_values(["station_id","ts"]).reset_index(drop=True)
    feat["hour"] = feat["ts"].dt.hour.astype("uint8")
    feat["dow"]  = feat["ts"].dt.dayofweek.astype("uint8")
    feat["is_weekend"] = (feat["dow"]>=5).astype("uint8")
    feat["hour_sin"] = np.sin(2*np.pi*feat["hour"]/24).astype("float32")
    feat["hour_cos"] = np.cos(2*np.pi*feat["hour"]/24).astype("float32")
    feat["hour_sin2"] = np.sin(4*np.pi*feat["hour"]/24).astype("float32")
    feat["hour_cos2"] = np.cos(4*np.pi*feat["hour"]/24).astype("float32")

    def shift(col,k): return feat.groupby("station_id")[col].shift(k)

    for k in [1,2,3,6,12]:
        feat[f"occ_lag_{k*FREQ_MIN}"] = shift("occ", k).astype("float32")

    feat["occ_roll_15"]  = shift("occ",1).rolling(3).mean().reset_index(level=0, drop=True).astype("float32")
    feat["occ_roll_30"]  = shift("occ",1).rolling(6).mean().reset_index(level=0, drop=True).astype("float32")
    feat["occ_roll_60"]  = shift("occ",1).rolling(12).mean().reset_index(level=0, drop=True).astype("float32")
    feat["occ_roll_120"] = shift("occ",1).rolling(24).mean().reset_index(level=0, drop=True).astype("float32")
    feat["occ_roll_180"] = shift("occ",1).rolling(36).mean().reset_index(level=0, drop=True).astype("float32")

    feat["occ_delta_5"]  = (feat["occ"] - shift("occ",1)).astype("float32")
    feat["occ_delta_15"] = (feat["occ"] - shift("occ",3)).astype("float32")
    feat["occ_delta_30"] = (feat["occ"] - shift("occ",6)).astype("float32")
    feat["occ_delta_60"] = (feat["occ"] - shift("occ",12)).astype("float32")

    if use_ema:
        feat["occ_ema_fast"] = (
            feat.groupby("station_id")["occ"].transform(lambda s: s.shift(1).ewm(alpha=0.5, adjust=False).mean())
        ).astype("float32")
        feat["occ_ema_slow"] = (
            feat.groupby("station_id")["occ"].transform(lambda s: s.shift(1).ewm(alpha=0.1, adjust=False).mean())
        ).astype("float32")
        feat["occ_momentum"] = (feat["occ_ema_fast"] - feat["occ_ema_slow"]).astype("float32")

    for h in HORIZONS:
        sh = h // FREQ_MIN
        feat[f"occ_{h}"] = feat.groupby("station_id")["occ"].shift(-sh).astype("float32")

    return feat

def station_encodings(train: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    sta_mean = (train.groupby("station_id")["occ"].mean().rename("sta_mean_occ")).reset_index()
    sta_hdh  = (train.assign(hour=train["ts"].dt.hour.astype("uint8"),
                             dow=train["ts"].dt.dayofweek.astype("uint8"))
                     .groupby(["station_id","dow","hour"])["occ"].median()
                     .rename("sta_hdh_occ")).reset_index()
    out = frame.merge(sta_mean, on="station_id", how="left")
    out = out.merge(sta_hdh, on=["station_id","dow","hour"], how="left")
    out["sta_hdh_occ"] = out["sta_hdh_occ"].fillna(out["sta_mean_occ"])
    return out

def feature_list(use_ema=False, use_sta=True):
    base = [
        "dow","is_weekend","hour_sin","hour_cos","hour_sin2","hour_cos2",
        "occ_now",
        "occ_lag_5","occ_lag_10","occ_lag_15","occ_lag_30","occ_lag_60",
        "occ_roll_15","occ_roll_30","occ_roll_60","occ_roll_120","occ_roll_180",
        "occ_delta_5","occ_delta_15","occ_delta_30","occ_delta_60",
    ]
    if use_ema:
        base += ["occ_ema_fast","occ_ema_slow","occ_momentum"]
    if use_sta:
        base += ["sta_mean_occ","sta_hdh_occ"]
    base += ["capacity"]
    return base

def make_delta_targets(train: pd.DataFrame, test: pd.DataFrame, freq_min=FREQ_MIN, horizons=HORIZONS):
    tr, te = train.copy(), test.copy()
    for h in horizons:
        sh = h // freq_min
        tr[f"occ_delta_target_{h}"] = (tr.groupby("station_id")["occ"].shift(-sh) - tr["occ"]).astype("float32")
        te[f"occ_delta_target_{h}"]  = (te.groupby("station_id")["occ"].shift(-sh) - te["occ"]).astype("float32")
    return tr, te
