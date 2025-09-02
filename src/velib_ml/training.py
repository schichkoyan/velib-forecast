import numpy as np, lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from .config import SPLIT_TRAINVAL

def _clean_for_horizon(df, feat_cols, h):
    # keep only rows fully defined for this horizon
    need = list(dict.fromkeys(feat_cols + [f"occ_{h}", f"occ_delta_target_{h}", "occ_now", "capacity"]))
    return df.dropna(subset=need).copy()

def _time_val_split(df):
    cut = df["ts"].quantile(SPLIT_TRAINVAL)
    return df[df["ts"] <= cut].copy(), df[df["ts"] > cut].copy()

def train_delta_gamma(train_df, test_df, feat_cols, horizon, num_threads=2, gamma_grid=(0.5, 0.7, 0.9, 1.0)):
    y_col = f"occ_delta_target_{horizon}"

    # --- NEW: drop NaNs for this horizon on both splits ---
    tr = _clean_for_horizon(train_df, feat_cols, horizon).sort_values(["station_id","ts"])
    te = _clean_for_horizon(test_df,   feat_cols, horizon).sort_values(["station_id","ts"])

    # temporal train/val split
    tr_tr, tr_val = _time_val_split(tr)
    Xtr, ytr   = tr_tr[feat_cols], tr_tr[y_col]
    Xval, yval = tr_val[feat_cols], tr_val[y_col]
    Xte        = te[feat_cols]

    params = dict(objective="regression_l1", metric="l1",
                  learning_rate=0.08, num_leaves=63, max_depth=-1,
                  min_data_in_leaf=64, feature_fraction=0.85,
                  bagging_fraction=0.8, bagging_freq=1,
                  lambda_l1=0.0, lambda_l2=0.0,
                  seed=42, verbosity=-1, num_threads=num_threads)

    dtrain = lgb.Dataset(Xtr, label=ytr)
    dval   = lgb.Dataset(Xval, label=yval, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=800,
                      valid_sets=[dtrain, dval], valid_names=["train","val"],
                      callbacks=[lgb.early_stopping(stopping_rounds=50)])

    # gamma calibration on val (MAE in bikes)
    delta_val = model.predict(Xval, num_iteration=model.best_iteration)
    cap_val   = Xval["capacity"].to_numpy()
    occ_now_v = tr_val["occ_now"].to_numpy()
    y_true_v  = (tr_val[f"occ_{horizon}"] * cap_val).to_numpy()

    best_g, best_mae = 1.0, 1e9
    for g in gamma_grid:
        y_hat_v = np.clip(occ_now_v + g*delta_val, 0, 1) * cap_val
        mae_v   = mean_absolute_error(y_true_v, y_hat_v)
        if mae_v < best_mae:
            best_mae, best_g = mae_v, g

    # test
    delta_te = model.predict(Xte, num_iteration=model.best_iteration)
    cap_te   = Xte["capacity"].to_numpy()
    occ_now_t= te["occ_now"].to_numpy()
    y_true_t = (te[f"occ_{horizon}"] * cap_te).to_numpy()
    y_hat_t  = np.clip(occ_now_t + best_g*delta_te, 0, 1) * cap_te
    mae_t    = mean_absolute_error(y_true_t, y_hat_t)

    return dict(mae=mae_t, model=model, best_iter=int(model.best_iteration or 0), gamma=float(best_g))
