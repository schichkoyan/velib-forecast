#!/usr/bin/env python
from __future__ import annotations
import argparse, time
from pathlib import Path
import requests
import pandas as pd

LAT, LON = 48.8566, 2.3522   # Paris center
HOURLY_VARS = "temperature_2m,precipitation,wind_speed_10m"  # per Open-Meteo docs

ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/era5"
HISTFORECAST  = "https://historical-forecast-api.open-meteo.com/v1/forecast"  # fallback

def _call(url: str, params: dict, attempts: int = 5) -> dict:
    last = None
    for k in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=30)
            last = r
            if r.ok:
                j = r.json()
                if isinstance(j, dict) and j.get("error"):
                    raise RuntimeError(f"{url} → {j.get('reason')}")
                return j
            else:
                # server responded but not ok (400/429/5xx)
                try:
                    j = r.json()
                    reason = j.get("reason")
                except Exception:
                    reason = r.text[:200]
                raise RuntimeError(f"{url} [{r.status_code}] → {reason}")
        except Exception as e:
            if k == attempts - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"HTTP failed after {attempts} attempts; last={getattr(last,'status_code',None)}")

def fetch_hourly(start_date: str, end_date: str) -> pd.DataFrame:
    params = dict(
        latitude=LAT, longitude=LON, timezone="UTC",
        hourly=HOURLY_VARS,
        start_date=start_date, end_date=end_date,
    )
    # 1) primary: ERA5 archive
    try:
        j = _call(ARCHIVE_URL, params)
    except Exception as e:
        print(f"[warn] Archive API failed: {e}\n→ trying Historical Forecast fallback…")
        # 2) fallback: Historical Forecast API (same params)
        j = _call(HISTFORECAST, params)

    hrs = j["hourly"]
    if "time" not in hrs:
        raise RuntimeError("No 'hourly.time' in response")
    df = pd.DataFrame(hrs)
    df["ts"] = pd.to_datetime(df["time"], utc=True)
    df.drop(columns=["time"], inplace=True)
    # ensure expected columns exist (fill missing if model lacks some var)
    for col in ["temperature_2m", "precipitation", "wind_speed_10m"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df[["ts", "temperature_2m", "precipitation", "wind_speed_10m"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--out",   default="data/external/weather_hourly.csv")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_hourly(args.start, args.end)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")

if __name__ == "__main__":
    main()