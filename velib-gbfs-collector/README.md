# Collecteur GBFS — Vélib' Métropole

Ce mini-projet collecte **les flux GBFS officiels Vélib'** et écrit des **snapshots timestampés** (CSV/Parquet) fusionnant **`station_information`** et **`station_status`**.

## Sources officielles
- `gbfs.json` — liste des feeds : https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/gbfs.json
- `station_information.json` : https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json
- `station_status.json` : https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json

Ces flux sont mis à jour ~**chaque minute** et sont publiés sous **Licence Ouverte / Open License** (voir page Open Data Vélib').

## Installation
```bash
pip install requests pandas pyarrow fastparquet
chmod +x scripts/collect_velib_gbfs.py
```

## Utilisation
Un seul snapshot :
```bash
python scripts/collect_velib_gbfs.py --outdir data/raw/velib --format csv
```

Collecte continue toutes les 60s (Ctrl+C pour arrêter) :
```bash
python scripts/collect_velib_gbfs.py --outdir data/raw/velib --format parquet --repeat --interval 60
```

Limiter à 120 snapshots (≈2h) :
```bash
python scripts/collect_velib_gbfs.py --repeat --interval 60 --max-snapshots 120
```

## Intégration avec ton projet S1
1. Lance la collecte continue pour quelques heures afin d'obtenir une **chronologie par station**.
2. Concatène les CSV/Parquet en un seul dataset (ou garde-les partitionnés par horodatage).
3. Alimente `features/spark_ingest_features.py` et `ml/baseline.py` avec ces fichiers (schéma compatible).

## Automatisation (cron – Linux/macOS)
Exécuter un snapshot chaque 5 minutes (CSV) :
```
*/5 * * * * /usr/bin/env python /chemin/collect_velib_gbfs.py --outdir /chemin/data/raw/velib --format csv >> /chemin/logs/collect.log 2>&1
```
> Astuce: utilise un **venv** et un chemin absolu vers Python.

## Remarques
- Si un champ est **absent** ou nommé différemment (ex. `numBikesAvailable`), le script l'harmonise vers `num_bikes_available` / `num_docks_available`.
- `snapshot_ts` est l'heure **UTC** du prélèvement — utile pour aligner la série temporelle par pas de 1–5 minutes.
