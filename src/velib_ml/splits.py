def split_train_test(feat, q):
    cut = feat["ts"].quantile(q)
    return feat[feat["ts"] <= cut].copy(), feat[feat["ts"] > cut].copy()
