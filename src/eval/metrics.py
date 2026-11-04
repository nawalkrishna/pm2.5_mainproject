"""Evaluation metrics, computed only on held-out predictions (Stage 5, task 1).

RMSE, MAE, R2 (1 - SSE/SST, about the held-out mean), mean bias (pred - obs), Pearson r.
"""

import numpy as np
import pandas as pd


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    n = len(y)
    if n < 2:
        return {"n": n, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan, "r": np.nan}
    err = p - y
    sst = float(((y - y.mean()) ** 2).sum())
    r = float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else np.nan
    return {"n": n, "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean()),
            "r2": float(1 - (err ** 2).sum() / sst) if sst > 0 else np.nan, "bias": float(err.mean()), "r": r}


def grouped(pred: pd.DataFrame, by: list[str], keys: list[str]) -> pd.DataFrame:
    """Metrics per group of ``by`` within each ``keys`` combination (e.g. split, model, tier)."""
    rows = []
    for k, g in pred.groupby(keys + by, dropna=False, observed=True):
        rows.append({**dict(zip(keys + by, k if isinstance(k, tuple) else (k,))), **metrics(g["y"], g["pred"])})
    return pd.DataFrame(rows)
