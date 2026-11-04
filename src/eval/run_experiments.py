"""Run baselines and the ablation under every split; save held-out predictions (Stage 5, tasks 2-3).

For each split and fold, every model is fitted on the training rows only and predicts the
held-out rows. Tier A (D039) is trained and evaluated only on rows where MAIAC AOD (10 km)
exists; Tier B on all rows. The two-tier prediction (A where available, else B) is
assembled later from these predictions in ``src.eval.report``.

Outputs:
    data/processed/predictions_v<version>.parquet   one row per held-out row x experiment
    reports/tables/spatial_blocks.csv               station -> block -> fold

Usage:
    python -m src.eval.run_experiments [--splits temporal blocked loso combined ablation]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import splits as S
from src.features.feature_sets import feature_columns
from src.models.baselines import Climatology, Linear, Persistence, RandomForest
from src.utils.config import config_hash, load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_PROCESSED, REPORTS_TABLES
from src.utils.run_log import log_run

log = get_logger("run_experiments")

META = ["station_id", "date_ist", "state", "year", "month", "season", "cloud_frac_b10", "sat_aod_b10", "sat_no2_b10"]


def load(cfg: dict, df: pd.DataFrame | None = None, target: str | None = None) -> pd.DataFrame:
    """Rows with a valid target, with station coordinates (for the spatial blocks)."""
    target = target or cfg["target"]
    if df is None:
        df = pd.read_parquet(DATA_PROCESSED / f"pm25_dataset_v{cfg['dataset_version']}.parquet")
    df = df[df[target].notna()].reset_index(drop=True)
    stations = pd.read_csv(DATA_PROCESSED.parent / "interim" / "ground_stations.csv")[["station_id", "latitude", "longitude"]]
    return df.merge(stations, on="station_id", how="left")


def make_folds(df: pd.DataFrame, cfg: dict, names: list[str]) -> tuple[list[S.Fold], pd.Series]:
    sp = cfg["splits"]
    st = df.drop_duplicates("station_id")[["station_id", "latitude", "longitude"]]
    blocks = S.spatial_blocks(st, sp["spatial_blocks"]["block_km"])
    folds = []
    if "temporal" in names:
        folds += S.temporal(df, sp["temporal"]["train_years"], sp["temporal"]["test_years"])
    if "blocked" in names:
        folds += S.blocked(df, blocks, sp["spatial_blocks"]["n_folds"])
    if "loso" in names:
        folds += S.loso(df)
    if "combined" in names:
        folds += S.combined(df, blocks, sp["combined"]["n_folds"], sp["temporal"]["train_years"],
                            sp["temporal"]["test_years"])
    return folds, blocks


def fit_predict(model, train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    if len(train) == 0 or len(test) == 0:
        return np.full(len(test), np.nan)
    return model.fit(train, train[target]).predict(test)


def run_fold(df: pd.DataFrame, fold: S.Fold, cfg: dict, ablation: bool) -> list[pd.DataFrame]:
    target, seed = cfg["target"], cfg["seed"]
    rf_params = cfg["models"]["random_forest"]
    train, test = df[fold.train], df[fold.test]
    out = []

    def record(model_name: str, tier: str, rows: pd.DataFrame, pred: np.ndarray) -> None:
        r = rows[META].copy()
        r["y"], r["pred"] = rows[target].to_numpy(), pred
        r["split"], r["fold"], r["model"], r["tier"] = fold.split, fold.fold, model_name, tier
        out.append(r)

    if ablation:
        groups = cfg["ablation_groups"]
        for name, group_list in cfg["ablation"].items():
            cols = feature_columns(group_list, extra_groups=groups)
            record("random_forest", f"ablation_{name}", test,
                   fit_predict(RandomForest(cols, seed, **rf_params), train, test, target))
        return out

    if fold.split == "temporal":
        record("persistence", "none", test, Persistence().predict(test))
    record("climatology", "none", test, fit_predict(Climatology(), train, test, target))
    sat = cfg["tier_a_requires"]
    for tier in cfg["tiers"]:
        cols = feature_columns(tier)
        tr, te = (train[train[sat] == 1], test[test[sat] == 1]) if tier.startswith("tier_a") else (train, test)
        record("linear", tier, te, fit_predict(Linear(cols, cfg["models"]["linear"]["impute"]), tr, te, target))
        record("random_forest", tier, te, fit_predict(RandomForest(cols, seed, **rf_params), tr, te, target))
    return out


def main(split_names: list[str] | None = None) -> Path:
    cfg = load_config("experiments")
    split_names = split_names or ["temporal", "blocked", "loso", "combined", "ablation"]
    df = load(cfg)

    preds = []
    base_names = [s for s in split_names if s != "ablation"]
    do_ablation = "ablation" in split_names
    fold_splits = sorted(set(base_names) | ({"temporal", "blocked"} if do_ablation else set()))
    folds, blocks = make_folds(df, cfg, fold_splits)
    fold_of = S.assign_folds(df["station_id"].map(blocks).value_counts(), cfg["splits"]["spatial_blocks"]["n_folds"])
    pd.DataFrame({"station_id": blocks.index, "block": blocks.to_numpy(),
                  "fold": [fold_of.get(b) for b in blocks.to_numpy()]}).to_csv(REPORTS_TABLES / "spatial_blocks.csv", index=False)

    for f in folds:
        t0 = time.time()
        if f.split in base_names:
            preds += run_fold(df, f, cfg, ablation=False)
        if do_ablation and f.split in ("temporal", "blocked"):
            preds += run_fold(df, f, cfg, ablation=True)
        log.info("%s %s: train %d, test %d (%.1fs)", f.split, f.fold, int(f.train.sum()), int(f.test.sum()),
                 time.time() - t0)

    out = DATA_PROCESSED / f"predictions_v{cfg['dataset_version']}.parquet"
    pred = pd.concat(preds, ignore_index=True)
    if out.exists() and set(split_names) != {"temporal", "blocked", "loso", "combined", "ablation"}:
        old = pd.read_parquet(out)
        keep_old = ~(old["split"].isin(base_names) | (old["tier"].str.startswith("ablation")
                                                      & ("ablation" in split_names)))
        pred = pd.concat([old[keep_old], pred], ignore_index=True)
    pred.to_parquet(out, index=False)
    log.info("Predictions: %d rows, splits %s", len(pred), sorted(pred["split"].unique()))
    log_run("run_experiments", config_hash(cfg), metrics={"prediction_rows": len(pred)}, notes=",".join(split_names))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="*")
    main(parser.parse_args().splits)
