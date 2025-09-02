# features/prepare_from_snapshots.py
# -*- coding: utf-8 -*-
# Concatène des snapshots GBFS (CSV/Parquet) et produit une série temporelle à pas de 5 minutes par station.
# Usage:
#   python features/prepare_from_snapshots.py --indir data/raw/velib --outcsv data/raw/velib_timeseries_5min.csv --freq 5

import argparse, os, glob
import pandas as pd
import numpy as np

def read_any(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.parquet') or path.endswith('.pq'):
        return pd.read_parquet(path)
    else:
        raise ValueError(f'Format non supporté: {path}')

def round_to_freq(ts, freq='5min'):
    # arrondit au bas de l'intervalle (floor)
    return pd.to_datetime(ts, utc=True).dt.floor(freq)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', required=True, help='Dossier contenant les snapshots velib_snapshot_*.csv|parquet')
    parser.add_argument('--outcsv', required=True, help='Chemin de sortie CSV (timeseries)')
    parser.add_argument('--freq', type=int, default=5, help='Pas en minutes (5 recommandé)')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, 'velib_snapshot_*.*')))
    if not files:
        raise SystemExit(f"Aucun fichier trouvé dans {args.indir} (attendu: velib_snapshot_*.csv|parquet)")

    dfs = []
    for f in files:
        try:
            df = read_any(f)
            dfs.append(df)
        except Exception as e:
            print(f'WARN: {f} ignoré ({e})')

    df = pd.concat(dfs, ignore_index=True)

    # Vérifie colonnes essentielles
    needed = {'station_id','num_bikes_available','num_docks_available','snapshot_ts'}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Colonnes manquantes: {missing}. Vérifie le collecteur.")

    # Normalisation types
    df['snapshot_ts'] = pd.to_datetime(df['snapshot_ts'], utc=True)
    df['station_id'] = df['station_id'].astype(str)
    for col in ['num_bikes_available','num_docks_available','capacity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Déduplique (même station & même snapshot_ts)
    df = df.sort_values(['station_id','snapshot_ts']).drop_duplicates(['station_id','snapshot_ts'], keep='last')

    # Arrondi au pas demandé
    freq_str = f'{args.freq}min'
    df['ts_bin'] = df['snapshot_ts'].dt.floor(freq_str)

    # Agrégation par station & ts_bin: on prend la dernière observation du bin
    agg_cols = {}
    for c in ['num_bikes_available','num_docks_available','capacity','lat','lon']:
        if c in df.columns:
            agg_cols[c] = 'last'
    grouped = df.groupby(['station_id','ts_bin']).agg(agg_cols).reset_index()

    # Renomme colonnes selon schéma S1
    out = grouped.rename(columns={
        'ts_bin':'ts',
        'num_bikes_available':'bikes_available',
        'num_docks_available':'docks_available'
    })

    # Option: borne [0, capacity]
    if 'capacity' in out.columns:
        out['bikes_available'] = out['bikes_available'].clip(lower=0, upper=out['capacity'])
        out['docks_available'] = out['docks_available'].clip(lower=0)

    # Tri & export
    out = out.sort_values(['station_id','ts']).reset_index(drop=True)

    # Colonnes finales recommandées
    final_cols = [c for c in ['ts','station_id','bikes_available','docks_available','capacity','lat','lon'] if c in out.columns]
    out[final_cols].to_csv(args.outcsv, index=False)
    print(f"Ecrit: {args.outcsv}  ({len(out):,} lignes, {out['station_id'].nunique()} stations, pas={freq_str})")

if __name__ == '__main__':
    main()
