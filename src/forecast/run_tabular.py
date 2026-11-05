"""Tabular forecast models: persistence, climatology, linear, Random Forest (Stage 6, D047-D049).

For each reported horizon h, linear and Random Forest models are fitted on issue hours in the
training years (last purge_hours dropped) and predict every issue hour of the test year with a
valid target. Settings (D046): a = past only, b = + reanalysis weather ahead. Tiers (D049):
B on all rows; A (+ satellite) trained and evaluated where the latest satellite day has AOD.

Prediction columns: persistence, climatology, <model>_<setting>_<tier>, e.g. rf_a_B, lin_b_A.

Each horizon is saved to data/processed/forecast_tabular_parts/ as soon as it finishes and is
skipped if already there (resumable); the combined file is written at the end. The Random
Forest's training subsample is drawn per horizon (seed = [seed, h]), so it does not depend on
which horizons run in one call.

Output:
    data/processed/forecast_predictions_tabular_v<version>.parquet

Usage:
    python -m src.forecast.run_tabular [--horizons 1 24]
"""

import argparse
import time

import numpy as np
import pandas as pd

from src.forecast.features import HourlyArrays, base_features, calendar, issue_frame, satellite_features, weather_ahead
from src.forecast.hourly_data import paths
from src.models.baselines import Linear, RandomForest
from src.utils.config import config_hash, load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_PROCESSED
from src.utils.run_log import log_run

log = get_logger("forecast_tabular")
PARTS = DATA_PROCESSED / "forecast_tabular_parts"

META = ["station_id", "state", "issue_idx", "h", "y", "season_code", "month", "sat_cloud_frac", "sat_aod_clear"]


def issue_masks(times: pd.DatetimeIndex, sp: dict) -> dict:
    year = times.year.to_numpy()
    out = {}
    for name in ["train", "val", "test"]:
        m = np.isin(year, sp[f"{name}_years"])
        if name != "test":                                     # purge: no target in the next period
            last = np.nonzero(m)[0].max()
            m[last - sp["purge_hours"] + 1:] = False
        out[name] = m
    return out


def climatology(A: HourlyArrays, mask: np.ndarray, te: pd.DataFrame) -> np.ndarray:
    """Training mean by station x month x hour of day (target hours in the training period);
    a station without training data there -> state x month x hour -> month x hour (as Phase 1)."""
    pm = A.var["pm25"][:, mask]
    t = A.times[mask]
    df = pd.DataFrame({"station_id": np.repeat(A.stations["station_id"].to_numpy(), mask.sum()),
                       "state": np.repeat(A.stations["state"].to_numpy(), mask.sum()),
                       "month": np.tile(t.month.to_numpy(), len(A.stations)),
                       "hour": np.tile(t.hour.to_numpy(), len(A.stations)), "pm25": pm.ravel()})
    out = pd.Series(np.nan, index=te.index)
    for keys in [["station_id", "month", "hour"], ["state", "month", "hour"], ["month", "hour"]]:
        m = df.groupby(keys)["pm25"].mean().rename("c").reset_index()
        out = out.fillna(te[keys].merge(m, on=keys, how="left")["c"].set_axis(te.index))
    return out.to_numpy()


def feature_lists(A, cfg, base, sat, cal_names) -> dict:
    static = [c for c in cfg["static"]]
    common = list(base) + cal_names + static
    ahead = list(weather_ahead(A, cfg, 1))
    sat_cols = list(sat)
    return {("a", "B"): common, ("b", "B"): common + ahead,
            ("a", "A"): common + sat_cols, ("b", "A"): common + ahead + sat_cols}


def main(horizons: list[int] | None = None, version: int = 1) -> None:
    cfg = load_config("forecast")
    seasons = load_config("audit")["seasons"]
    horizons = horizons or cfg["horizons"]
    hourly_path, sat_path = paths(version)
    A = HourlyArrays.load(pd.read_parquet(hourly_path), pd.read_parquet(sat_path), cfg)
    masks = issue_masks(A.times, cfg["splits"])
    base, sat = base_features(A, cfg), satellite_features(A, cfg)
    cal_names = list(calendar(A.times[:1], seasons))
    feats = feature_lists(A, cfg, base, sat, cal_names)
    rf_params = cfg["tabular"]["random_forest"]
    PARTS.mkdir(parents=True, exist_ok=True)
    for h in horizons:
        part = PARTS / f"h{h:02d}.parquet"
        if part.exists():
            log.info("h=%d already saved, skipped", h)
            continue
        t0 = time.time()
        rng = np.random.default_rng([cfg["seed"], h])
        tr = issue_frame(A, cfg, seasons, h, base, sat, masks["train"])
        te = issue_frame(A, cfg, seasons, h, base, sat, masks["test"])
        te["hour"] = A.times[np.minimum(te["issue_idx"] + h, len(A.times) - 1)].hour
        te["climatology"] = climatology(A, masks["train"], te)
        te["persistence"] = te["pm_last24"].fillna(te["climatology"])
        sub = rng.random(len(tr)) < cfg["tabular"]["rf_train_fraction"]
        for (setting, tier), cols in feats.items():
            trm = np.ones(len(tr), bool) if tier == "B" else tr["sat_aod_clear"].notna().to_numpy()
            tem = np.ones(len(te), bool) if tier == "B" else te["sat_aod_clear"].notna().to_numpy()
            lin = Linear(cols).fit(tr.loc[trm], tr.loc[trm, "y"])
            te.loc[tem, f"lin_{setting}_{tier}"] = lin.predict(te.loc[tem])
            t1 = time.time()
            rf = RandomForest(cols, seed=cfg["seed"], **rf_params).fit(tr.loc[trm & sub], tr.loc[trm & sub, "y"])
            te.loc[tem, f"rf_{setting}_{tier}"] = rf.predict(te.loc[tem])
            log.info("h=%d %s/%s: train %d (RF %d), test %d, RF %.0f s", h, setting, tier, int(trm.sum()),
                     int((trm & sub).sum()), int(tem.sum()), time.time() - t1)
        pred_cols = ["persistence", "climatology"] + [f"{m}_{s}_{t}" for m in ["lin", "rf"] for s, t in feats]
        te[META + pred_cols].to_parquet(part, index=False)
        log.info("h=%d done in %.0f s", h, time.time() - t0)
    res = pd.concat([pd.read_parquet(PARTS / f"h{h:02d}.parquet") for h in cfg["horizons"]], ignore_index=True)
    res["time_issue"] = A.times[res["issue_idx"].to_numpy()]
    path = DATA_PROCESSED / f"forecast_predictions_tabular_v{version}.parquet"
    res.to_parquet(path, index=False)
    log_run("forecast_tabular", config_hash(cfg), metrics={"rows": len(res), "horizons": cfg["horizons"]})
    log.info("Saved %d rows to %s", len(res), path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", type=int, nargs="*")
    main(p.parse_args().horizons)
