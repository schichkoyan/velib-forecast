
# -*- coding: utf-8 -*-
"""
baseline_sweep.py
Balaye plusieurs valeurs d'alpha pour l'EWMA et compare à la baseline naïve.
Usage :
    python baseline_sweep.py --input data/raw/velib_timeseries_5min.csv \
                             --output data/processed/baseline_sweep.csv \
                             --alphas 0.2,0.4,0.6,0.8
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def compute_baselines(df, freq_min=5, horizons=[15,30,60], alpha=0.3):
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df = df.sort_values(['station_id','ts'])
    results = []

    # EWMA par station
    for sid, g in df.groupby('station_id'):
        g = g.set_index('ts').asfreq(f'{freq_min}min')
        limit_ffill = max(1, 60 // freq_min)  # ≈ 1h
        g['bikes_available'] = g['bikes_available'].ffill(limit=limit_ffill)
        if 'docks_available' in g.columns:
            g['docks_available'] = g['docks_available'].ffill(limit=limit_ffill)
        if 'capacity' in g.columns:
            g['capacity'] = g['capacity'].ffill().bfill()
        g = g.dropna(subset=['bikes_available'])

        for h in horizons:
            y_true = g['bikes_available'].shift(-h//freq_min)

            # Naïve
            y_pred_naive = g['bikes_available']
            mae_naive = (y_true - y_pred_naive).abs().dropna().mean()
            mape_naive = ((y_true - y_pred_naive).abs() / (y_true.replace(0, np.nan))).dropna().mean()

            # EWMA
            ewma = g['bikes_available'].ewm(alpha=alpha, adjust=False).mean()
            y_pred_ewma = ewma
            mae_ewma = (y_true - y_pred_ewma).abs().dropna().mean()
            mape_ewma = ((y_true - y_pred_ewma).abs() / (y_true.replace(0, np.nan))).dropna().mean()

            results.append({
                'station_id': sid,
                'horizon_min': h,
                'mae_naive': float(mae_naive),
                'mape_naive': float(mape_naive) if pd.notnull(mape_naive) else None,
                'mae_ewma': float(mae_ewma),
                'mape_ewma': float(mape_ewma) if pd.notnull(mape_ewma) else None,
                'alpha': alpha
            })
    return pd.DataFrame(results)

def parse_alphas(s: str):
    try:
        return [float(x.strip()) for x in s.split(',') if x.strip()]
    except:
        raise argparse.ArgumentTypeError("Format attendu pour --alphas : '0.2,0.4,0.6,0.8'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='CSV de la série 5 min (ts, station_id, bikes_available, ...)')
    ap.add_argument('--output', required=True, help='CSV de sortie (comparatif)')
    ap.add_argument('--alphas', type=parse_alphas, default=[0.2,0.4,0.6,0.8], help='Liste d\'alphas séparés par des virgules')
    ap.add_argument('--freq', type=int, default=5, help='Pas temporel (min)')
    ap.add_argument('--horizons', type=str, default='15,30,60', help='Horizons en minutes, ex: "15,30,60"')
    args = ap.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(',') if x.strip()]

    df = pd.read_csv(args.input)

    # Calcul pour chaque alpha puis concat
    frames = []
    for a in args.alphas:
        res = compute_baselines(df, freq_min=args.freq, horizons=horizons, alpha=a)
        frames.append(res)
    out = pd.concat(frames, ignore_index=True)

    # Créer le dossier si besoin
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    # Résumé lisible en console
    summary = (out.groupby(['alpha','horizon_min'])[['mae_naive','mae_ewma']]
                 .mean(numeric_only=True)
                 .rename(columns={'mae_naive':'MAE_naive_avg','mae_ewma':'MAE_EWMA_avg'}))
    print("\n== Moyennes par alpha & horizon ==")
    print(summary.reset_index().to_string(index=False))

    # Meilleur alpha par horizon (sur la moyenne)
    best = (summary.reset_index()
                  .sort_values(['horizon_min','MAE_EWMA_avg'])
                  .groupby('horizon_min', as_index=False)
                  .first()[['horizon_min','alpha','MAE_EWMA_avg']])
    print("\n== Meilleur alpha (EWMA) par horizon (moyenne) ==")
    print(best.to_string(index=False))

if __name__ == '__main__':
    main()
