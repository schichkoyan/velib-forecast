# scripts/train.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import mlflow, mlflow.lightgbm

# ==== project imports ====
from velib_ml.config import FREQ_MIN, HORIZONS, SPLIT_TRAINTEST, SPLIT_TRAINVAL
from velib_ml.data import load_timeseries
from velib_ml.features import make_features, station_encodings, feature_list, make_delta_targets
from velib_ml.splits import split_train_test
from velib_ml.training import train_delta_gamma
from velib_ml.io_utils import save_artifacts
from velib_ml.weather import resample_weather_to_5min, add_weather


def naive_mae_bikes(feat: pd.DataFrame, split_q: float) -> dict[int, float]:
    """MAE naïve (prédit 'occ' actuel pour occ_h futur), en vélos, sur TEST uniquement."""
    tmp = feat.copy()
    cut = tmp["ts"].quantile(split_q)
    test = tmp[tmp["ts"] > cut].copy()

    out = {}
    for h in HORIZONS:
        sh = h // FREQ_MIN
        test[f"occ_{h}"] = test.groupby("station_id")["occ"].shift(-sh).astype("float32")
        dfh = test.dropna(subset=[f"occ_{h}", "occ", "capacity"])
        y_true = (dfh[f"occ_{h}"] * dfh["capacity"]).to_numpy()
        y_pred = (dfh["occ"] * dfh["capacity"]).to_numpy()
        out[h] = float((pd.Series(y_true - y_pred)).abs().mean())
    return out


def main(args: argparse.Namespace) -> None:
    # ===== MLflow config =====
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or f"{Path(args.out).name}--ema{args.use_ema}--sta{args.use_sta}"

    with mlflow.start_run(run_name=run_name):
        # ===== Load & features =====
        df = load_timeseries(args.data)
        n_rows, n_sta = len(df), df["station_id"].nunique()
        mlflow.log_params({"n_rows": n_rows, "n_stations": n_sta})

        feat = make_features(df, use_ema=args.use_ema)
        if args.weather:
            w_hourly = pd.read_csv(args.weather, parse_dates=["ts"])
            w_5min   = resample_weather_to_5min(w_hourly)
            feat     = add_weather(feat, w_5min)

        # Split train/test (temps)
        train, test = split_train_test(feat, SPLIT_TRAINTEST)

        # Encodages station (optionnels) – calculés sur TRAIN puis appliqués au TEST
        if args.use_sta:
            train = station_encodings(train, train)
            test  = station_encodings(train, test)

        # Ancre niveau courant
        for d in (train, test):
            d["occ_now"] = d["occ"].astype("float32")

        # Cibles Δ
        train_d, test_d = make_delta_targets(train, test)

        # Liste des features (même ordre que l’entraînement)
        feat_cols = feature_list(use_ema=args.use_ema, use_sta=args.use_sta)
        if args.weather:
            feat_cols = feat_cols + ["temperature_2m","precipitation","wind_speed_10m","is_rain"]

        # ===== Baseline Naïve (sur TEST) =====
        mae_naive = naive_mae_bikes(feat, SPLIT_TRAINTEST)
        for h, v in mae_naive.items():
            mlflow.log_metric(f"mae_naive_{h}", v)

        # ===== Log global params =====
        mlflow.log_params({
            "freq_min": FREQ_MIN,
            "horizons": str(HORIZONS),
            "split_train_test": SPLIT_TRAINTEST,
            "split_train_val": SPLIT_TRAINVAL,
            "threads": args.threads,
            "use_ema": args.use_ema,
            "use_sta": args.use_sta,
        })

        # ===== Train per horizon (Δ + gamma calibration) =====
        results, models = {}, {}
        for h in HORIZONS:
            out = train_delta_gamma(
                train_d, test_d, feat_cols, h, num_threads=args.threads
            )
            results[h] = out
            models[h] = out["model"]

            # MLflow metrics
            mlflow.log_metric(f"mae_{h}", out["mae"])
            mlflow.log_metric(f"best_iter_{h}", out["best_iter"])
            mlflow.log_metric(f"gamma_{h}", out["gamma"])

            # Log model inside MLflow run
            mi = mlflow.lightgbm.log_model(out["model"], artifact_path=f"model_h{h}")

            # Optional: register each horizon model
            if args.register:
                model_name = f"{args.register}_h{h}"
                mlflow.register_model(model_uri=mi.model_uri, name=model_name)

        # ===== Save local artifacts (filesystem) =====
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)

        # Metrics dataframe (filesystem)
        metrics_df = pd.DataFrame({
            "horizon_min": HORIZONS,
            "mae_naive":   [mae_naive[h] for h in HORIZONS],
            "mae_model":   [results[h]["mae"] for h in HORIZONS],
            "best_iter":   [results[h]["best_iter"] for h in HORIZONS],
            "gamma":       [results[h]["gamma"] for h in HORIZONS],
        })
        # Config for reproducibility
        cfg = {
            "freq_min": FREQ_MIN,
            "horizons": HORIZONS,
            "split_train_test": SPLIT_TRAINTEST,
            "split_train_val": SPLIT_TRAINVAL,
            "gammas": {str(h): results[h]["gamma"] for h in HORIZONS},
        }

        # Persist models + feature list + config + metrics (filesystem)
        with open(outdir / "feat_cols_delta.json", "w") as f:
            json.dump(feat_cols, f)
        metrics_df.to_csv(outdir / "metrics.csv", index=False)
        with open(outdir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        # Save boosters to filesystem via helper (also prints path)
        save_artifacts({h: models[h] for h in HORIZONS}, feat_cols, cfg, metrics_df, str(outdir))

        # ===== Log artifacts into MLflow run =====
        mlflow.log_artifact(str(outdir / "metrics.csv"))
        mlflow.log_artifact(str(outdir / "feat_cols_delta.json"))
        mlflow.log_artifact(str(outdir / "config.json"))

        # Also log one sample features row to help API testing later
        try:
            sample = test_d[feat_cols].dropna().head(1)
            sample.to_csv(outdir / "sample_features.csv", index=False)
            mlflow.log_artifact(str(outdir / "sample_features.csv"))
        except Exception as e:
            print("Could not save sample_features.csv:", e)

        # Pretty print
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/raw/velib_timeseries_5min.csv")
    ap.add_argument("--out", type=str,  default="artifacts/v0_1")
    ap.add_argument("--threads", type=int, default=2)

    ap.add_argument("--use-ema", dest="use_ema", action="store_true")
    ap.add_argument("--no-ema",  dest="use_ema", action="store_false")
    ap.set_defaults(use_ema=False)

    ap.add_argument("--use-sta", dest="use_sta", action="store_true")
    ap.add_argument("--no-sta",  dest="use_sta", action="store_false")
    ap.set_defaults(use_sta=False)

    # MLflow options
    ap.add_argument("--experiment", type=str, default="velib-forecast")
    ap.add_argument("--tracking-uri", type=str, default=None)  # e.g. http://127.0.0.1:5000
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--register", type=str, default=None, help="Register models under this base name (one per horizon)")

    # Weather options
    ap.add_argument("--weather", type=str, default=None, help="CSV from fetch_weather.py")

    main(ap.parse_args())
