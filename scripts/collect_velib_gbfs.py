#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Collecte des données Vélib' (GBFS) en snapshots périodiques et export CSV/Parquet.
# - Récupère station_information.json et station_status.json
# - Merge par station_id + ajoute un horodatage UTC 'snapshot_ts'
# - Ecrit un fichier par snapshot (timestampé) + un fichier 'latest' (option)
# Usage:
#   python scripts/collect_velib_gbfs.py --outdir data/raw/velib --format csv --repeat --interval 60

import argparse, time, sys, os
from datetime import datetime, timezone
import requests
import pandas as pd

VELIB_BASE = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole"
URL_INFO = f"{VELIB_BASE}/station_information.json"
URL_STATUS = f"{VELIB_BASE}/station_status.json"

def fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def one_snapshot() -> pd.DataFrame:
    info = fetch_json(URL_INFO)
    status = fetch_json(URL_STATUS)
    # GBFS structure: {'data': {'stations': [...]}, ...}
    info_df = pd.DataFrame(info['data']['stations'])
    status_df = pd.DataFrame(status['data']['stations'])

    # Harmonisation colonnes (certains champs ont des alias)
    if 'numBikesAvailable' in status_df and 'num_bikes_available' not in status_df:
        status_df['num_bikes_available'] = status_df['numBikesAvailable']
    if 'numDocksAvailable' in status_df and 'num_docks_available' not in status_df:
        status_df['num_docks_available'] = status_df['numDocksAvailable']

    # Merge
    df = pd.merge(status_df, info_df, on='station_id', suffixes=('_status', '_info'), how='inner')

    # Colonnes utiles de base
    keep = [
        'station_id', 'name', 'lat', 'lon', 'capacity',
        'is_installed', 'is_renting', 'is_returning', 'last_reported',
        'num_bikes_available', 'num_docks_available'
    ]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    # Horodatage du snapshot (UTC ISO)
    df['snapshot_ts'] = datetime.now(timezone.utc).isoformat()

    return df

def write_output(df: pd.DataFrame, outdir: str, fmt: str = 'csv', latest: bool = True) -> str:
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base = f'velib_snapshot_{ts}'
    path = os.path.join(outdir, f'{base}.{fmt.lower()}')
    if fmt.lower() == 'csv':
        df.to_csv(path, index=False)
        if latest:
            df.to_csv(os.path.join(outdir, 'latest.csv'), index=False)
    elif fmt.lower() in ('parquet', 'pq'):
        df.to_parquet(path, index=False)
        if latest:
            df.to_parquet(os.path.join(outdir, 'latest.parquet'), index=False)
    else:
        raise ValueError('format non supporté (csv|parquet)')
    return path

def parse_args():
    p = argparse.ArgumentParser(description="Collecte GBFS Vélib' en snapshots")
    p.add_argument('--outdir', default='data/raw/velib', help='Dossier de sortie')
    p.add_argument('--format', default='csv', choices=['csv','parquet'], help='Format de sortie')
    p.add_argument('--repeat', action='store_true', help='Boucle de collecte continue (sinon un seul snapshot)')
    p.add_argument('--interval', type=int, default=60, help='Intervalle en secondes entre snapshots (min ~60s)')
    p.add_argument('--max-snapshots', type=int, default=0, help='Nombre max de snapshots (0 = infini en mode --repeat)')
    return p.parse_args()

def main():
    args = parse_args()

    if not args.repeat:
        df = one_snapshot()
        path = write_output(df, args.outdir, args.format)
        print(f'Snapshot écrit → {path}')
        return

    count = 0
    try:
        while True:
            start = time.time()
            df = one_snapshot()
            path = write_output(df, args.outdir, args.format)
            count += 1
            print(f'[{count}] Snapshot écrit → {path}')
            if args.max_snapshots and count >= args.max_snapshots:
                break
            elapsed = time.time() - start
            to_sleep = max(0, args.interval - elapsed)
            time.sleep(to_sleep)
    except KeyboardInterrupt:
        print('Arrêt demandé (Ctrl+C).')
        sys.exit(0)

if __name__ == '__main__':
    main()
