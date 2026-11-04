"""Cluster bootstrap confidence intervals for pooled held-out metrics (D043).

Clusters (spatial blocks, D038) are resampled with replacement, keeping every row of a drawn
cluster, so errors correlated within a city are not treated as independent. Metrics are computed
from per-cluster sums, so 1,000 draws cost one matrix product.

Usage:
    from src.eval.bootstrap import paired_ci
    paired_ci(df, cluster="block", y="y", preds=["pred_a", "pred_b"], n=1000, seed=42)
"""

import numpy as np
import pandas as pd


def cluster_sums(df: pd.DataFrame, cluster: str, y: str, preds: list[str]) -> pd.DataFrame:
    """Per-cluster n, sum y, sum y² and sum of squared errors of each prediction column."""
    d = df[[cluster, y, *preds]].dropna()
    s = pd.DataFrame({"n": 1, "sy": d[y], "syy": d[y] ** 2}, index=d.index)
    for p in preds:
        s[f"sse_{p}"] = (d[p] - d[y]) ** 2
    return s.groupby(d[cluster]).sum()


def r2_rmse(s: pd.DataFrame | np.ndarray, cols: list[str], p: str) -> tuple[np.ndarray, np.ndarray]:
    n, sy, syy, sse = (s[..., cols.index(c)] for c in ["n", "sy", "syy", f"sse_{p}"])
    return 1 - sse / (syy - sy ** 2 / n), np.sqrt(sse / n)


def paired_ci(df: pd.DataFrame, cluster: str, y: str, preds: list[str], n: int, seed: int,
              level: float = 0.95) -> dict:
    """Point estimate and percentile CI of R² and RMSE for each prediction column, and of the
    R² difference preds[0] − preds[1] (computed on rows where every column has a value)."""
    s = cluster_sums(df, cluster, y, preds)
    cols = list(s.columns)
    a = s.to_numpy(float)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=n)      # n draws x clusters
    draws = counts @ a
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = {"n_rows": int(a[:, cols.index("n")].sum()), "n_clusters": len(a)}
    r2 = {}
    for p in preds:
        est_r2, est_rmse = r2_rmse(a.sum(0), cols, p)
        r2[p], rmse = r2_rmse(draws, cols, p)
        out[f"r2_{p}"] = float(est_r2)
        out[f"r2_{p}_lo"], out[f"r2_{p}_hi"] = np.percentile(r2[p], [lo, hi])
        out[f"rmse_{p}"] = float(est_rmse)
        out[f"rmse_{p}_lo"], out[f"rmse_{p}_hi"] = np.percentile(rmse, [lo, hi])
    if len(preds) >= 2:
        d = r2[preds[0]] - r2[preds[1]]
        out["d_r2"] = out[f"r2_{preds[0]}"] - out[f"r2_{preds[1]}"]
        out["d_r2_lo"], out["d_r2_hi"] = np.percentile(d, [lo, hi])
        out["p_d_le_0"] = float((d <= 0).mean())
    return out
