# ml/baseline.py
# -*- coding: utf-8 -*-
# Calcule les baselines naïve & EWMA sur horizons +15/+30/+60 minutes.
# Usage: python ml/baseline.py --input data/raw/synthetic_velib.csv --output data/processed/baseline_metrics.csv
import argparse
import pandas as pd
import numpy as np

def compute_baselines(df, freq_min=5, horizons=[15,30,60], alpha=0.3):
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df = df.sort_values(['station_id','ts'])
    results = []

    # EWMA par station
    for sid, g in df.groupby('station_id'):
        g = g.set_index('ts').asfreq(f'{freq_min}min')
        # forward-fill pour trous minimes
        g['bikes_available'] = g['bikes_available'].ffill()
        # Naïve: valeur courante -> prédiction t+h
        for h in horizons:
            y_true = g['bikes_available'].shift(-h//freq_min)
            y_pred_naive = g['bikes_available']
            mae_naive = (y_true - y_pred_naive).abs().dropna().mean()
            mape_naive = ( (y_true - y_pred_naive).abs() / (y_true.replace(0,np.nan)) ).dropna().mean()

            # EWMA (prédit valeur lissée à t)
            ewma = g['bikes_available'].ewm(alpha=alpha, adjust=False).mean()
            y_pred_ewma = ewma
            mae_ewma = (y_true - y_pred_ewma).abs().dropna().mean()
            mape_ewma = ( (y_true - y_pred_ewma).abs() / (y_true.replace(0,np.nan)) ).dropna().mean()

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV avec colonnes ts, station_id, bikes_available, docks_available (optionnel: temperature_c, rain_mm)')
    parser.add_argument('--output', required=True, help='Chemin CSV de sortie pour les métriques')
    parser.add_argument('--alpha', type=float, default=0.3, help='Paramètre EWMA (0..1)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    metrics = compute_baselines(df, alpha=args.alpha)
    metrics.to_csv(args.output, index=False)
    print(metrics.groupby('horizon_min')[['mae_naive','mae_ewma']].mean())

if __name__ == '__main__':
    main()
