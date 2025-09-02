# tests/test_baseline.py
# -*- coding: utf-8 -*-
# Tests très simples pour vérifier la forme des résultats.

import pandas as pd
from ml.baseline import compute_baselines

def test_compute_baselines_shape():
    data = {
        'ts': pd.date_range('2025-07-01', periods=24, freq='5min').astype(str).tolist()*2,
        'station_id': ['S1']*24 + ['S2']*24,
        'bikes_available': [i%10 for i in range(24)] + [5]*(24),
        'docks_available': [10]*48
    }
    df = pd.DataFrame(data)
    res = compute_baselines(df)
    assert {'station_id','horizon_min','mae_naive','mae_ewma'}.issubset(res.columns)
    assert res.shape[0] == 2*3  # 2 stations × 3 horizons
