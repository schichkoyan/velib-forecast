import json, pandas as pd
from pathlib import Path

def save_artifacts(models: dict, feat_cols: list, config: dict, metrics_df: pd.DataFrame, outdir: str):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    import lightgbm as lgb
    for h, mdl in models.items():
        mdl.save_model(str(out / f"lgbm_delta_h{h}.txt"))
    with open(out/"feat_cols_delta.json","w") as f: json.dump(feat_cols, f)
    with open(out/"config.json","w") as f: json.dump(config, f, indent=2)
    metrics_df.to_csv(out/"metrics.csv", index=False)
