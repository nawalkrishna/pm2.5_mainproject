"""Exploratory station-day table for Stage 3 (not the Stage 4 analysis-ready dataset).

Joins, per station and IST calendar day:
    * ground PM2.5: daily mean, number of hours, valid flag (>= min_hours_per_valid_day)
    * satellite (TROPOMI, MAIAC): mean and valid-pixel count for pt / b5 / b10
    * reanalysis daily means over the IST day: ERA5-Land (falling back to ERA5 where the
      ERA5-Land cell is masked), ERA5 boundary-layer height, MERRA-2 PM2.5 and AOD
    * static: elevation, built-up fraction, population (b5)

No QC is applied here and no rows are dropped: Stage 3 describes the data as received.
Derived quantities:
    RH (%)        from 2 m temperature and dew point (Magnus formula, Alduchov & Eskridge 1996)
    wind speed    sqrt(u^2 + v^2)
    precipitation daily sum, mm (ERA5-Land hourly, m -> mm)
    MERRA-2 PM2.5 DUSMASS25 + OCSMASS + BCSMASS + SSSMASS25 + SO4SMASS * 132.14/96.06, x1e9 -> µg/m³

Output: data/interim/eda_daily.parquet

Usage:
    python -m src.eda.build_daily
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.gee_extract import OUT_DIR as GEE_DIR
from src.ingest.ground_pm25 import HOURLY_PATH, STATIONS_PATH
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_INTERIM

log = get_logger("build_daily")

OUT_PATH: Path = DATA_INTERIM / "eda_daily.parquet"
IST = "Asia/Kolkata"


def relative_humidity(t_k: pd.Series, td_k: pd.Series) -> pd.Series:
    """RH in percent from temperature and dew point (K), Magnus form over water."""
    t, td = t_k - 273.15, td_k - 273.15
    a, b = 17.625, 243.04
    return 100 * np.exp(a * td / (b + td)) / np.exp(a * t / (b + t))


def merra2_pm25(df: pd.DataFrame) -> pd.Series:
    """Reconstructed MERRA-2 surface PM2.5 in µg/m³ (GMAO MERRA-2 FAQ)."""
    kg_m3 = df["DUSMASS25"] + df["OCSMASS"] + df["BCSMASS"] + df["SSSMASS25"] + df["SO4SMASS"] * (132.14 / 96.06)
    return kg_m3 * 1e9


def daily_pm25(min_hours: int) -> pd.DataFrame:
    h = pd.read_parquet(HOURLY_PATH, columns=["station_id", "timestamp_ist", "pm25"])
    h["date_ist"] = h["timestamp_ist"].dt.strftime("%Y-%m-%d")
    d = h.groupby(["station_id", "date_ist"])["pm25"].agg(pm25="mean", pm25_hours="count").reset_index()
    d["pm25_valid"] = d["pm25_hours"] >= min_hours
    return d


def daily_satellite(product: str) -> pd.DataFrame:
    """Wide table: <layer>_<geometry> and <layer>_n_<geometry> per station-day."""
    df = pd.concat([pd.read_csv(f) for f in sorted((GEE_DIR / product).glob("*.csv"))], ignore_index=True)
    layers = [c[:-5] for c in df.columns if c.endswith("_mean")]
    wide = df.pivot_table(index=["station_id", "date_ist"], columns="geometry",
                          values=[f"{l}_mean" for l in layers] + [f"{l}_count" for l in layers], aggfunc="first")
    wide.columns = [f"{v[:-5]}_{g}" if v.endswith("_mean") else f"{v[:-6]}_n_{g}" for v, g in wide.columns]
    return wide.reset_index()


def hourly_product(product: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sorted((GEE_DIR / product).glob("*.csv"))], ignore_index=True)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.drop_duplicates(["station_id", "time_utc"])
    df["date_ist"] = df["time_utc"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d")
    return df


def daily_meteorology() -> pd.DataFrame:
    land = hourly_product("era5_land")
    era5 = hourly_product("era5")
    m = land.merge(era5, on=["station_id", "time_utc", "date_ist"], how="outer", suffixes=("_land", "_era5"))
    out = pd.DataFrame({"station_id": m["station_id"], "date_ist": m["date_ist"]})
    for var in ["temperature_2m", "dewpoint_temperature_2m", "u_component_of_wind_10m", "v_component_of_wind_10m"]:
        out[var] = m[f"{var}_land"].fillna(m[f"{var}_era5"])
    out["met_source_land"] = m["temperature_2m_land"].notna()
    out["precip_mm"] = (m["total_precipitation_hourly"].fillna(m["total_precipitation"]) * 1000)
    out["blh"] = m["boundary_layer_height"]
    out["rh"] = relative_humidity(out["temperature_2m"], out["dewpoint_temperature_2m"])
    out["wind_speed"] = np.hypot(out["u_component_of_wind_10m"], out["v_component_of_wind_10m"])
    out["t2m_c"] = out["temperature_2m"] - 273.15
    g = out.groupby(["station_id", "date_ist"])
    daily = g.agg(t2m_c=("t2m_c", "mean"), rh=("rh", "mean"), wind_speed=("wind_speed", "mean"),
                  u10=("u_component_of_wind_10m", "mean"), v10=("v_component_of_wind_10m", "mean"),
                  precip_mm=("precip_mm", "sum"), blh=("blh", "mean"),
                  met_hours=("t2m_c", "count"), met_land_share=("met_source_land", "mean")).reset_index()
    return daily


def daily_merra2() -> pd.DataFrame:
    m = hourly_product("merra2")
    m["merra2_pm25"] = merra2_pm25(m)
    return m.groupby(["station_id", "date_ist"]).agg(merra2_pm25=("merra2_pm25", "mean"),
                                                     merra2_aod=("TOTEXTTAU", "mean"),
                                                     merra2_hours=("merra2_pm25", "count")).reset_index()


def static_features() -> pd.DataFrame:
    s = pd.read_csv(GEE_DIR / "static" / "static_b5.csv")
    out = s[["station_id", "elevation_mean"]].rename(columns={"elevation_mean": "elevation_m"})
    out["built_frac_b5"] = s.get("worldcover_frac_50", 0)          # WorldCover class 50 = built-up
    out["population_b5"] = s[[c for c in s.columns if c.startswith("population_")]].mean(axis=1)
    return out


def main() -> Path:
    window = load_config("study_area")["extraction_period"]
    audit = load_config("audit")
    seasons = {m: name for name, months in audit["seasons"].items() for m in months}
    stations = pd.read_csv(STATIONS_PATH)[["station_id", "name", "state", "latitude", "longitude"]]

    days = pd.date_range(window["start"], window["end"], freq="D").strftime("%Y-%m-%d")
    grid = pd.MultiIndex.from_product([stations.dropna(subset=["latitude"])["station_id"], days],
                                      names=["station_id", "date_ist"]).to_frame(index=False)
    df = grid.merge(daily_pm25(audit["min_hours_per_valid_day"]), on=["station_id", "date_ist"], how="left")
    df["pm25_valid"] = df["pm25_valid"].fillna(False).astype(bool)
    df["pm25_hours"] = df["pm25_hours"].fillna(0).astype(int)
    for product in ["tropomi", "maiac"]:
        df = df.merge(daily_satellite(product), on=["station_id", "date_ist"], how="left")
    df = df.merge(daily_meteorology(), on=["station_id", "date_ist"], how="left")
    df = df.merge(daily_merra2(), on=["station_id", "date_ist"], how="left")
    df = df.merge(static_features(), on="station_id", how="left").merge(stations, on="station_id", how="left")
    date = pd.to_datetime(df["date_ist"])
    df["year"], df["month"] = date.dt.year, date.dt.month
    df["season"] = df["month"].map(seasons)
    df.to_parquet(OUT_PATH, index=False)
    log.info("EDA daily table: %d rows, %d stations, %d columns; valid PM2.5 days: %d",
             len(df), df["station_id"].nunique(), df.shape[1], int(df["pm25_valid"].sum()))
    return OUT_PATH


if __name__ == "__main__":
    main()
