"""Hourly station table for forecasting (Stage 6, D045-D046).

One row per station x IST hour (hour start) over the extraction window, on the full calendar
grid, so a shift by k rows is a true k-hour lag (a missing hour stays NaN).

Columns:
    keys        station_id, time_ist (hour start, tz-aware IST), state
    target      pm25            QC-passed value (Stage 4 rule, qc_flag == ok), else NaN
    meteorology t2m_c, rh, u10, v10, wind_speed, blh, precip_mm   ERA5-Land, ERA5 where masked
    aerosol     merra2_pm25, merra2_aod
    static      elevation, distance to coast, land cover, population (the hour's year)

ERA5 instants are stamped on the UTC hour, i.e. :30 IST, the middle of an IST hour, so they
map to the hour that starts 30 minutes earlier. MERRA-2 averages and ERA5 accumulations stamped
at :30 IST map the same way; their window may end after that IST hour, which the issue-time
rules of D046 handle (average_lag). Satellite values stay daily (``satellite_daily``).

Outputs:
    data/processed/forecast_hourly_v<version>.parquet
    data/processed/forecast_satellite_daily_v<version>.parquet

Usage:
    python -m src.forecast.hourly_data
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.align.reanalysis import hourly_meteorology, load_hourly, merra2_pm25
from src.align.time_windows import IST
from src.clean.pm25_qc import QC_PATH
from src.eda.build_daily import daily_satellite
from src.features.static import static_features
from src.ingest.ground_pm25 import STATIONS_PATH
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_PROCESSED

log = get_logger("forecast_hourly")


def paths(version: int) -> tuple[Path, Path]:
    return (DATA_PROCESSED / f"forecast_hourly_v{version}.parquet",
            DATA_PROCESSED / f"forecast_satellite_daily_v{version}.parquet")


def ist_hour_start(ts_utc: pd.Series) -> pd.Series:
    """IST hour start of each instant (13:30 IST -> 13:00 IST)."""
    return pd.to_datetime(ts_utc, utc=True).dt.tz_convert(IST).dt.floor("h")


def hourly_pm25(qc: pd.DataFrame) -> pd.DataFrame:
    ok = qc[qc["qc_flag"] == "ok"]
    out = pd.DataFrame({"station_id": ok["station_id"].to_numpy(),
                        "time_ist": ist_hour_start(ok["timestamp_utc"]).to_numpy(),
                        "pm25": ok["pm25"].to_numpy()})
    if out.duplicated(["station_id", "time_ist"]).any():
        raise ValueError("more than one QC-passed value per station-hour")
    return out


def hourly_reanalysis(cfg: dict) -> pd.DataFrame:
    met = hourly_meteorology(load_hourly("era5_land"), load_hourly("era5"))
    met = met[["station_id", "time_utc", *cfg["meteorology"]]]
    m2 = load_hourly("merra2")
    m2 = pd.DataFrame({"station_id": m2["station_id"], "time_utc": m2["time_utc"],
                       "merra2_pm25": merra2_pm25(m2), "merra2_aod": m2["TOTEXTTAU"]})
    out = met.merge(m2, on=["station_id", "time_utc"], how="outer")
    out["time_ist"] = ist_hour_start(out["time_utc"])
    return out.drop(columns="time_utc")


def satellite_daily(cfg: dict) -> pd.DataFrame:
    g = cfg["satellite_geometry"]
    cols = [f"{c}_{g}" for c in cfg["satellite_columns"] + cfg["satellite_counts"]]
    sat = daily_satellite("tropomi").merge(daily_satellite("maiac"), on=["station_id", "date_ist"], how="outer")
    return sat[["station_id", "date_ist", *cols]]


def build(version: int = 1) -> tuple[Path, Path]:
    cfg = load_config("forecast")
    window = load_config("study_area")["extraction_period"]
    qc = pd.read_parquet(QC_PATH)
    pm = hourly_pm25(qc)
    all_stations = pd.read_csv(STATIONS_PATH).dropna(subset=["latitude", "longitude"])
    stations = all_stations[all_stations["station_id"].isin(pm["station_id"].unique())]

    hours = pd.date_range(pd.Timestamp(window["start"], tz=IST), pd.Timestamp(window["end"], tz=IST)
                          + pd.Timedelta(hours=23), freq="h")
    grid = pd.MultiIndex.from_product([stations["station_id"], hours], names=["station_id", "time_ist"]).to_frame(index=False)
    df = grid.merge(pm, on=["station_id", "time_ist"], how="left")
    n_pm_outside = len(pm) - int(df["pm25"].notna().sum())
    df = df.merge(hourly_reanalysis(cfg), on=["station_id", "time_ist"], how="left")

    stat = static_features(all_stations)
    pop = {int(c.rsplit("_", 1)[1]): c for c in stat.columns if c.startswith("population_b5_")}
    df = df.merge(stat, on="station_id", how="left")
    year = df["time_ist"].dt.year
    df["population_b5"] = np.select([year == y for y in pop], [df[c] for c in pop.values()], default=df[pop[min(pop)]])
    df = df.drop(columns=list(pop.values())).merge(stations[["station_id", "state"]], on="station_id", how="left")
    keep = ["station_id", "time_ist", "state", "pm25", *cfg["meteorology"], *cfg["aerosol"], *cfg["static"]]
    df = df[keep]
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype("float32")

    hourly_path, sat_path = paths(version)
    df.to_parquet(hourly_path, index=False)
    satellite_daily(cfg).to_parquet(sat_path, index=False)
    log.info("Hourly table: %d rows, %d stations, %d hours; PM2.5 values %d (QC-passed %d, outside grid %d); "
             "hours without meteorology %d", len(df), df["station_id"].nunique(), len(hours),
             int(df["pm25"].notna().sum()), len(pm), n_pm_outside, int(df["t2m_c"].isna().sum()))
    return hourly_path, sat_path


if __name__ == "__main__":
    build()
