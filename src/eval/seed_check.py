"""Seed check: how much do the no-lag Random Forest results move when only the RF seed changes?

Refits Tier A and Tier B without PM2.5 lags (and climatology) on the folds of the chosen splits
with each seed, and reports the satellite gain (Tier A - Tier B on Tier A's rows), the two-tier R²
and its gain over climatology. Seed 42 is the stored main run, so it must reproduce
reports/tables/results_ci.csv. The decision log (D044) records the temporal, blocked and combined
splits; leave-one-station-out was added for the thesis review.

Output (one row per seed x split, appended as each seed finishes):
    reports/tables/seed_check_<splits>.csv

Usage:
    python -m src.eval.seed_check --splits loso --seeds 42 1 2 3 4 5
"""

import argparse
import time

import numpy as np
import pandas as pd

from src.eval import run_experiments as R
from src.features.feature_sets import feature_columns
from src.models.baselines import Climatology, RandomForest
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import REPORTS_TABLES

log = get_logger("seed_check")


def r2(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2))


def run(splits: list[str], seeds: list[int]) -> None:
    cfg = load_config("experiments")
    df = R.load(cfg)
    folds, _ = R.make_folds(df, cfg, splits)
    sat, target, rf = cfg["tier_a_requires"], cfg["target"], cfg["models"]["random_forest"]
    out_path = REPORTS_TABLES / f"seed_check_{'_'.join(splits)}.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["seed", "split"])
    for seed in seeds:
        for split in splits:
            if ((done["seed"] == seed) & (done["split"] == split)).any():
                log.info("seed %d %s already done, skipped", seed, split)
                continue
            t0 = time.time()
            parts = []
            for f in [f for f in folds if f.split == split]:
                tr, te = df[f.train], df[f.test]
                p = te[[target, sat]].copy()
                p["clim"] = Climatology().fit(tr, tr[target]).predict(te)
                p["b"] = RandomForest(feature_columns("tier_b_nolag"), seed, **rf).fit(tr, tr[target]).predict(te)
                tra, tea = tr[tr[sat] == 1], te[te[sat] == 1]
                p["a"] = np.nan
                if len(tea):
                    p.loc[tea.index, "a"] = RandomForest(feature_columns("tier_a_nolag"), seed, **rf).fit(
                        tra, tra[target]).predict(tea)
                parts.append(p)
            p = pd.concat(parts)
            s = p[p[sat] == 1]
            two = np.where(p["a"].notna(), p["a"], p["b"])
            row = {"seed": seed, "split": split, "r2_a": r2(s[target], s["a"]), "r2_b_satrows": r2(s[target], s["b"]),
                   "gain": r2(s[target], s["a"]) - r2(s[target], s["b"]), "two_tier_nolag": r2(p[target], two),
                   "two_tier_minus_clim": r2(p[target], two) - r2(p[target], p["clim"])}
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(out_path, index=False)
            log.info("seed %d %s: gain %+.4f, two-tier %.4f (%.0f s)", seed, split, row["gain"],
                     row["two_tier_nolag"], time.time() - t0)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--splits", nargs="+", default=["loso"])
    a.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4, 5])
    args = a.parse_args()
    run(args.splits, args.seeds)
