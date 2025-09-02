# Vélib' Availability Forecast — S1 (Données & Baseline)

Ce starter couvre la **première étape** : ingestion simple, features de base, et **baselines** (naïve & EWMA) avec **Spark** (optionnel) + **pandas**.

## Objectif S1
1. Préparer un dataset station × temps (granularité 5 minutes) avec colonnes:
   - `ts` (timestamp UTC)
   - `station_id` (str)
   - `bikes_available` (int)
   - `docks_available` (int)
   - `temperature_c` (float, optionnel)
   - `rain_mm` (float, optionnel)
2. Générer des **features** (lags, rolling means).
3. Calculer les **baselines** à horizons +15 / +30 / +60 min.
4. Exporter en `data/processed/` (Parquet).

> Tu peux tester immédiatement avec le **jeu synthétique** fourni (2 stations × 2 jours à 5 min).

## Données réelles (pistes)
- **Vélib'**: l'API temps-réel (GBFS) donne l'état des stations. Pour l'historique, tu peux:
  - Capturer des snapshots périodiques (cron) et les stocker en CSV/Parquet.
  - Ou chercher un **GBFS archive** communautaire (quand disponible).
- À défaut, ce starter inclut `scripts/notes_api.md` (dans ce README) avec des pistes de collecte.

## Lancement (local sans Spark)
```bash
python ml/baseline.py --input data/raw/synthetic_velib.csv --output data/processed/baseline_metrics.csv
```

## Lancement (avec Spark local)
```bash
python features/spark_ingest_features.py --input data/raw/synthetic_velib.csv --outdir data/processed/parquet
```

## Contenu
- `data/raw/synthetic_velib.csv` — données synthétiques de test.
- `features/spark_ingest_features.py` — ingestion & features (Spark).
- `ml/baseline.py` — calcule MAE/MAPE des baselines (pandas).

## Prochaines étapes (S2)
- Intégrer **MLflow Tracking** durant l'entraînement (PyTorch).
- Conserver ces scripts comme **référence** pour la qualité des données et le benchmarking.

## Intégration des snapshots Vélib' (réels)

1) Lance la collecte (ex. 2 heures en Parquet) :
```bash
python scripts/collect_velib_gbfs.py --outdir data/raw/velib --format parquet --repeat --interval 60 --max-snapshots 120
```

2) Prépare la série temporelle 5 minutes :
```bash
python features/prepare_from_snapshots.py --indir data/raw/velib --outcsv data/raw/velib_timeseries_5min.csv --freq 5
```

3) Calcule les baselines sur les vraies données :
```bash
python ml/baseline.py --input data/raw/velib_timeseries_5min.csv --output data/processed/baseline_metrics.csv
```
