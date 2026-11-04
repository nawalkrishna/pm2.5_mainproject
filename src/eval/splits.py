"""Train/test splits without leakage (Stage 5, task 1).

Every split returns a list of ``Fold(name, train_mask, test_mask)`` over the rows of the
dataset. Spatial splits never put a station (or its block) in both train and test.

    temporal   train on train_years, test on test_years (all stations)
    blocked    spatial blocks = stations linked by single linkage within block_km (so a
               city's cluster of stations is one block); blocks are assigned to n_folds
               folds, balancing rows (deterministic)
    loso       one fold per station
    combined   blocked folds AND the temporal split: train = other blocks in train years,
               test = held-out blocks in test years
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.align.spatial import haversine_m


@dataclass
class Fold:
    split: str
    fold: str
    train: np.ndarray
    test: np.ndarray


def spatial_blocks(stations: pd.DataFrame, block_km: float) -> pd.Series:
    """Block id per station: connected components of the 'within block_km' graph."""
    ids = stations["station_id"].to_numpy()
    lat, lon = stations["latitude"].to_numpy(), stations["longitude"].to_numpy()
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ids)):
        d = haversine_m(lat[i], lon[i], lat, lon) / 1000
        for j in np.where(d <= block_km)[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(len(ids))]
    order = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return pd.Series([f"block_{order[r]:02d}" for r in roots], index=ids, name="block")


def assign_folds(sizes: pd.Series, n_folds: int) -> dict:
    """Greedy balanced assignment of groups to folds (largest first, to the lightest fold)."""
    load = [0] * n_folds
    out = {}
    for g, s in sizes.sort_values(ascending=False).items():
        k = int(np.argmin(load))
        out[g] = k
        load[k] += s
    return out


def temporal(df: pd.DataFrame, train_years: list[int], test_years: list[int]) -> list[Fold]:
    return [Fold("temporal", "test_" + "_".join(map(str, test_years)),
                 df["year"].isin(train_years).to_numpy(), df["year"].isin(test_years).to_numpy())]


def blocked(df: pd.DataFrame, blocks: pd.Series, n_folds: int) -> list[Fold]:
    b = df["station_id"].map(blocks)
    folds = assign_folds(b.value_counts(), n_folds)
    f = b.map(folds).to_numpy()
    return [Fold("blocked", f"fold_{k}", f != k, f == k) for k in range(n_folds)]


def loso(df: pd.DataFrame) -> list[Fold]:
    s = df["station_id"].to_numpy()
    return [Fold("loso", sid, s != sid, s == sid) for sid in sorted(df["station_id"].unique())]


def combined(df: pd.DataFrame, blocks: pd.Series, n_folds: int, train_years: list[int],
             test_years: list[int]) -> list[Fold]:
    out = []
    tr_y, te_y = df["year"].isin(train_years).to_numpy(), df["year"].isin(test_years).to_numpy()
    for f in blocked(df, blocks, n_folds):
        out.append(Fold("combined", f.fold, f.train & tr_y, f.test & te_y))
    return out
