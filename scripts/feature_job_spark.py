# features/spark_ingest_features.py
# -*- coding: utf-8 -*-
# Script d'ingestion + features avec Spark (local) pour données station × temps.
# Usage: python features/spark_ingest_features.py --input data/raw/synthetic_velib.csv --outdir data/processed/parquet
import argparse
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Chemin CSV brut (ts, station_id, bikes_available, docks_available, temperature_c, rain_mm)')
    parser.add_argument('--outdir', required=True, help='Dossier de sortie Parquet (partitionné par date, station_id)')
    args = parser.parse_args()

    spark = SparkSession.builder.appName('velib_features').getOrCreate()
    df = spark.read.option('header', True).csv(args.input, inferSchema=True)

    # Convertit ts → timestamp & colonnes de date
    df = df.withColumn('ts', F.to_timestamp('ts'))            .withColumn('date', F.to_date('ts'))            .withColumn('hour', F.hour('ts'))            .withColumn('dow', F.date_format('ts','u').cast('int'))  # 1=lundi..7=dimanche

    # Fenêtre pour calculer des lags/rollings par station
    w = Window.partitionBy('station_id').orderBy(F.col('ts')).rowsBetween(-12, -1)  # 12 * 5min = dernière heure

    # Moyenne mobile sur 1h (excluant t)
    df = df.withColumn('bikes_rolling_1h', F.avg('bikes_available').over(w))

    # Lags explicites
    df = df.withColumn('bikes_lag_5min', F.lag('bikes_available', 1).over(Window.partitionBy('station_id').orderBy('ts')))            .withColumn('bikes_lag_30min', F.lag('bikes_available', 6).over(Window.partitionBy('station_id').orderBy('ts')))            .withColumn('bikes_lag_60min', F.lag('bikes_available', 12).over(Window.partitionBy('station_id').orderBy('ts')))

    # Write parquet partitionné
    (df.write.mode('overwrite')
       .partitionBy('date','station_id')
       .parquet(args.outdir))

    spark.stop()

if __name__ == '__main__':
    main()
