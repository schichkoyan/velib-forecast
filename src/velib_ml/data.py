import pandas as pd

def load_timeseries(path):
    usecols = ["ts","station_id","bikes_available","capacity"]
    dtypes  = {"station_id":"category","bikes_available":"float32","capacity":"float32"}
    df = (pd.read_csv(path, usecols=usecols, dtype=dtypes, parse_dates=["ts"])
            .sort_values(["station_id","ts"]).reset_index(drop=True))
    df["capacity"] = (df.groupby("station_id")["capacity"]
                        .transform(lambda s: s.ffill().bfill().fillna(s.max()))
                        .astype("float32"))
    df = df[df["capacity"]>0].copy()
    df["occ"] = (df["bikes_available"]/df["capacity"]).clip(0,1).astype("float32")
    return df