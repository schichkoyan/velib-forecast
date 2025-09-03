import pandas as pd
from typing import Iterable

WEATHER_COLS = ["temperature_2m","precipitation","wind_speed_10m"]

def resample_weather_to_5min(weather_hourly: pd.DataFrame) -> pd.DataFrame:
    w = weather_hourly.copy()
    w = w.set_index("ts").sort_index()
    # forward-fill hourly values to 5-min grid
    w5 = (w
          .reindex(pd.date_range(w.index.min(), w.index.max(), freq="5min", tz="UTC"))
          .ffill())
    w5.index.name = "ts"
    return w5.reset_index()

def add_weather(feat_5min: pd.DataFrame, weather_5min: pd.DataFrame,
                cols: Iterable[str] = WEATHER_COLS) -> pd.DataFrame:
    w = weather_5min[["ts", *cols]].copy()
    out = feat_5min.merge(w, on="ts", how="left")
    # safety fill (rare gaps at edges)
    for c in cols:
        out[c] = out[c].fillna(method="ffill").fillna(method="bfill")
    # a few simple transformations
    out["is_rain"] = (out.get("precipitation", 0) > 0.1).astype("uint8")
    return out