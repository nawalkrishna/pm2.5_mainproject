"""Analysis-ready station-day dataset (Stage 4 deliverable).

Rows: station-days in the study period (2021-2025, IST) at stations with coordinates where
at least one target (daily or overpass) is valid. Built on the full calendar grid of the
extraction window first, so 1-3 day lags are true calendar lags (a missing day gives a
NaN lag, never the previous available row) and 1 Jan 2021 has lags from December 2020.

Columns:
    keys       station_id, date_ist, state, year, month, season
    targets    pm25_daily, pm25_overpass, n_hours_daily, n_hours_overpass
    satellite  <layer>_<pt|b5|b10>, <layer>_n_<geom> (valid pixels), sat_<no2|aod>_<geom> flags
               layers: no2_trop, no2_cloud, co, so2, hcho, uvai, cloud_frac, aod_clear, aod_best
    reanalysis daily and overpass-window (_op) meteorology, BLH, MERRA-2 PM2.5 / AOD
    temporal   doy_sin, doy_cos, month, season_code
    lags       <var>_lag1..3 for alignment.lag_variables
    static     elevation, land cover shares, population (row's year), distance to coast
Missing satellite values stay NaN with an availability flag (task 7: no imputation).

Outputs:
    data/processed/pm25_dataset_v<version>.parquet
    data/processed/pm25_dataset_v<version>.json     data card: rows, columns, sha256, git, configs
    reports/tables/dataset_reconciliation.csv       hours -> QC -> days -> final rows

Usage:
    python -m src.features.build_dataset [--version 2]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.align.reanalysis import REANALYSIS_PATH
from src.align.targets import TARGETS_PATH
from src.clean.pm25_qc import QC_PATH
from src.eda.build_daily import daily_satellite
from src.features.static import static_features
from src.ingest.ground_pm25 import HOURLY_PATH, STATIONS_PATH
from src.utils.config import config_hash, load_config
from src.utils.logging_setup import get_logger
from src.utils.manifest import sha256_of
from src.utils.paths import DATA_PROCESSED, REPORTS_TABLES
from src.utils.run_log import _git_state, log_run

log = get_logger("build_dataset")

SAT_LAYERS = ["no2_trop", "no2_cloud", "co", "so2", "hcho", "uvai", "cloud_frac", "aod_clear", "aod_best"]
GEOMS = ["pt", "b5", "b10"]
SEASON_CODES = {"winter": 0, "pre_monsoon": 1, "sw_monsoon": 2, "ne_monsoon": 3}


def add_lags(df: pd.DataFrame, variables: list[str], lags: list[int]) -> pd.DataFrame:
    """Calendar-day lags per station: the value on (date - k days), NaN if that day is absent."""
    base = df[["station_id", "date_ist", *variables]]
    dates = pd.to_datetime(base["date_ist"])
    out = df.copy()
    for k in lags:
        shifted = base.assign(date_ist=(dates + pd.Timedelta(days=k)).dt.strftime("%Y-%m-%d"))
        shifted = shifted.rename(columns={v: f"{v}_lag{k}" for v in variables})
        out = out.merge(shifted, on=["station_id", "date_ist"], how="left")
    return out


def temporal_features(df: pd.DataFrame, seasons: dict) -> pd.DataFrame:
    d = pd.to_datetime(df["date_ist"])
    doy = d.dt.day_of_year
    year_len = np.where(d.dt.is_leap_year, 366, 365)
    month_to_season = {m: s for s, months in seasons.items() for m in months}
    return df.assign(year=d.dt.year, month=d.dt.month, doy_sin=np.sin(2 * np.pi * doy / year_len),
                     doy_cos=np.cos(2 * np.pi * doy / year_len),
                     season=d.dt.month.map(month_to_season),
                     season_code=d.dt.month.map(month_to_season).map(SEASON_CODES))


def reconciliation(final: pd.DataFrame, targets: pd.DataFrame, period: dict) -> pd.DataFrame:
    hourly = pd.read_parquet(HOURLY_PATH, columns=["station_id"])
    qc = pd.read_parquet(QC_PATH, columns=["qc_flag"])
    t = targets[targets["date_ist"].between(period["start"], period["end"])]
    rows = [
        ("hourly rows, all sources (2020-12..2025-12)", len(hourly)),
        ("hourly rows passing QC", int((qc["qc_flag"] == "ok").sum())),
        ("station-days with >= 1 QC-passed hour, study period", len(t)),
        ("  of which valid daily target", int(t["pm25_daily"].notna().sum())),
        ("  of which valid overpass target", int(t["pm25_overpass"].notna().sum())),
        ("  of which at least one valid target", int((t["pm25_daily"].notna() | t["pm25_overpass"].notna()).sum())),
        ("  dropped: station without coordinates", int(((t["pm25_daily"].notna() | t["pm25_overpass"].notna())
                                                         & ~t["station_id"].isin(final["station_id"].unique())).sum())),
        ("final dataset rows", len(final)),
    ]
    return pd.DataFrame(rows, columns=["step", "count"])


def main(version: int = 2) -> Path:
    area = load_config("study_area")
    align = load_config("alignment")
    seasons = load_config("audit")["seasons"]
    window, period = area["extraction_period"], area["study_period"]

    stations = pd.read_csv(STATIONS_PATH).dropna(subset=["latitude", "longitude"])
    days = pd.date_range(window["start"], window["end"], freq="D").strftime("%Y-%m-%d")
    df = pd.MultiIndex.from_product([stations["station_id"], days], names=["station_id", "date_ist"]).to_frame(index=False)

    targets = pd.read_parquet(TARGETS_PATH)
    df = df.merge(targets, on=["station_id", "date_ist"], how="left")
    for product in ["tropomi", "maiac"]:
        df = df.merge(daily_satellite(product), on=["station_id", "date_ist"], how="left")
    for g in GEOMS:
        df[f"sat_no2_{g}"] = df[f"no2_trop_{g}"].notna().astype(int)
        df[f"sat_aod_{g}"] = df[f"aod_clear_{g}"].notna().astype(int)
    df = df.merge(pd.read_parquet(REANALYSIS_PATH), on=["station_id", "date_ist"], how="left")
    df = temporal_features(df, seasons)
    df = add_lags(df, align["lag_variables"], align["lags_days"])

    stat = static_features(stations)
    df = df.merge(stat, on="station_id", how="left")
    pop_cols = [c for c in stat.columns if c.startswith("population_b5_")]
    years = {int(c.rsplit("_", 1)[1]): c for c in pop_cols}
    # Population of the row's own year (2021-2025); December 2020 rows use 2021.
    df["population_b5"] = np.select([df["year"] == y for y in years], [df[c] for c in years.values()],
                                    default=df[years[min(years)]])
    df = df.drop(columns=pop_cols).merge(stations[["station_id", "state"]], on="station_id", how="left")

    final = df[df["date_ist"].between(period["start"], period["end"])
               & (df["pm25_daily"].notna() | df["pm25_overpass"].notna())].reset_index(drop=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / f"pm25_dataset_v{version}.parquet"
    final.to_parquet(out, index=False)
    rec = reconciliation(final, targets, period)
    rec.to_csv(REPORTS_TABLES / "dataset_reconciliation.csv", index=False)

    commit, dirty = _git_state()
    card = {"file": out.name, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": len(final), "columns": list(final.columns), "stations": int(final["station_id"].nunique()),
            "period": period, "sha256": sha256_of(out), "git_commit": commit, "git_dirty": dirty,
            "config_hash": config_hash({"area": area, "align": align, "qc": load_config("qc")}),
            "reconciliation": rec.to_dict(orient="records")}
    out.with_suffix(".json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    log.info("Dataset v%d: %d rows, %d stations, %d columns\n%s", version, len(final),
             card["stations"], final.shape[1], rec.to_string(index=False))
    log_run("build_dataset", card["config_hash"], metrics={"rows": len(final), "stations": card["stations"]})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", type=int, default=2)
    main(parser.parse_args().version)
