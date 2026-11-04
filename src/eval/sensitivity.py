"""Sensitivity runs for the Stage 4 choices D028-D031, D035, D036 (Stage 5, D042).

Random Forest only, temporal and blocked splits. Each variant transforms a copy of the dataset
and keeps the column names, so the main feature lists apply unchanged:

    satellite variants   change Tier A's inputs and rows (which rows have AOD); Tier A and
                         Tier A no-lag are refitted, Tier B is taken from the main run
                         (same rows, folds, seed and features, so it is identical)
    target variants      change the target (overpass window, QC S2b); all four tiers are refitted

Output: data/processed/predictions_sensitivity_v<version>.parquet (column ``variant``).

Usage:
    python -m src.eval.sensitivity [--variants aod_best qc_s2b ...]
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from src.align.targets import build_targets
from src.clean.pm25_qc import apply_qc
from src.eval import run_experiments as R
from src.features.build_dataset import add_lags
from src.features.feature_sets import feature_columns
from src.ingest.ground_pm25 import HOURLY_PATH
from src.models.baselines import RandomForest
from src.utils.config import config_hash, load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_PROCESSED
from src.utils.run_log import log_run

log = get_logger("sensitivity")

B10_PIXELS = {"no2_trop": 254, "aod_clear": 366}     # nominal pixels in a 10 km circle (as qc_options)
GAS_COLS = ["no2_trop", "co", "so2", "hcho", "uvai", "cloud_frac", "no2_cloud"]


def mask_satellite(df: pd.DataFrame, g: str, rows: pd.Series, layers: list[str]) -> pd.DataFrame:
    """Set the given satellite layers (and their counts) to missing on ``rows``; recompute the flags."""
    df = df.copy()
    for layer in layers:
        for c in [f"{layer}_{g}", f"{layer}_n_{g}"]:
            df[c] = df[c].astype(float).mask(rows)
    df[f"sat_no2_{g}"] = df[f"no2_trop_{g}"].notna().astype(int)
    df[f"sat_aod_{g}"] = df[f"aod_clear_{g}"].notna().astype(int)
    return df


def qc_s2b_targets(flat_hours: int) -> pd.DataFrame:
    """Daily target and PM2.5 lags rebuilt with a different flat-line rule (D028 sensitivity)."""
    qc = load_config("qc") | {"flat_run_hours": flat_hours}
    h = apply_qc(pd.read_parquet(HOURLY_PATH, columns=["station_id", "timestamp_utc", "pm25"]), qc)
    log.info("QC flat_run >= %d h: %d hours flagged", flat_hours, int((h["qc_flag"] != "ok").sum()))
    t = build_targets(h, load_config("alignment"))[["station_id", "date_ist", "pm25_daily"]]
    return add_lags(t, ["pm25_daily"], load_config("alignment")["lags_days"])


def variant_data(base: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, str, str]:
    """(dataset copy, target column, satellite geometry) for one variant."""
    g = spec.get("geometry", "b10")
    target = spec.get("target", "pm25_daily")
    df = base
    if "aod_variant" in spec:
        v = spec["aod_variant"]
        df = df.assign(**{f"aod_clear_{x}": df[f"{v}_{x}"] for x in ["pt", "b5", "b10"]},
                       **{f"aod_clear_n_{x}": df[f"{v}_n_{x}"] for x in ["pt", "b5", "b10"]})
        df = mask_satellite(df, g, pd.Series(False, index=df.index), [])
    if "max_cloud_fraction" in spec:
        cloudy = df["cloud_frac_b10"] > spec["max_cloud_fraction"]
        df = mask_satellite(df, g, cloudy, GAS_COLS + ["aod_clear"])
    if "min_pixel_share" in spec:
        s = spec["min_pixel_share"]
        df = mask_satellite(df, g, df[f"aod_clear_n_{g}"] < s * B10_PIXELS["aod_clear"], ["aod_clear"])
        df = mask_satellite(df, g, df[f"no2_trop_n_{g}"] < s * B10_PIXELS["no2_trop"], ["no2_trop"])
    if "flat_run_hours" in spec:
        new = qc_s2b_targets(spec["flat_run_hours"])
        cols = ["pm25_daily", "pm25_daily_lag1", "pm25_daily_lag2", "pm25_daily_lag3"]
        df = df.drop(columns=cols).merge(new[["station_id", "date_ist", *cols]], on=["station_id", "date_ist"],
                                         how="left")
    return df, target, g


def run_variant(name: str, spec: dict, base: pd.DataFrame, cfg: dict, refit_b: bool) -> list[pd.DataFrame]:
    df, target, g = variant_data(base, spec)
    df = R.load(cfg, df=df, target=target)
    folds, _ = R.make_folds(df, cfg, cfg["sensitivity"]["splits"])
    feat_target = "overpass" if target == "pm25_overpass" else "daily"
    sat = f"sat_aod_{g}"
    rf = cfg["models"]["random_forest"]
    tiers = ["tier_a", "tier_a_nolag"] + (["tier_b", "tier_b_nolag"] if refit_b else [])
    out = []
    for f in folds:
        t0 = time.time()
        train, test = df[f.train], df[f.test]
        for tier in tiers:
            cols = feature_columns(tier, target=feat_target, geometry=g)
            tr, te = (train[train[sat] == 1], test[test[sat] == 1]) if tier.startswith("tier_a") else (train, test)
            r = te[R.META].copy()
            r["sat_rows"] = te[sat].to_numpy()
            r["y"] = te[target].to_numpy()
            r["pred"] = R.fit_predict(RandomForest(cols, cfg["seed"], **rf), tr, te, target)
            r["split"], r["fold"], r["model"], r["tier"], r["variant"] = f.split, f.fold, "random_forest", tier, name
            out.append(r)
        log.info("%s %s %s: train %d, test %d, Tier A rows %d (%.1fs)", name, f.split, f.fold, int(f.train.sum()),
                 int(f.test.sum()), int((test[sat] == 1).sum()), time.time() - t0)
    return out


def main(names: list[str] | None = None) -> Path:
    cfg = load_config("experiments")
    sc = cfg["sensitivity"]
    variants = {**{k: (v, False) for k, v in sc["satellite_variants"].items()},
                **{k: (v, True) for k, v in sc["target_variants"].items()}}
    names = names or list(variants)
    base = pd.read_parquet(DATA_PROCESSED / f"pm25_dataset_v{cfg['dataset_version']}.parquet")
    preds = []
    for name in names:
        spec, refit_b = variants[name]
        preds += run_variant(name, spec, base, cfg, refit_b)
    out = DATA_PROCESSED / f"predictions_sensitivity_v{cfg['dataset_version']}.parquet"
    pred = pd.concat(preds, ignore_index=True)
    if out.exists():
        old = pd.read_parquet(out)
        pred = pd.concat([old[~old["variant"].isin(names)], pred], ignore_index=True)
    pred.to_parquet(out, index=False)
    log.info("Sensitivity predictions: %d rows, variants %s", len(pred), sorted(pred["variant"].unique()))
    log_run("sensitivity", config_hash(cfg), metrics={"prediction_rows": len(pred)}, notes=",".join(names))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="*")
    main(parser.parse_args().variants)
