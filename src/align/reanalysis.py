"""Hourly reanalysis -> daily IST features, plus overpass-window values (Stage 4, task 5).

ERA5-Land is used where its cell has data; ERA5 fills coastal stations whose ERA5-Land cell
is masked as sea (``met_land_share`` records the share of hours from ERA5-Land). MERRA-2
hours are hourly averages stamped at the start of the hour (data dictionary).

Per station and IST day:
    t2m_c, rh, wind_speed, u10, v10, blh         daily mean
    precip_mm                                    daily sum
    t2m_c_op, rh_op, wind_speed_op, blh_op       mean over the overpass window (12-15 IST)
    merra2_pm25, merra2_aod                      daily mean; *_op overpass-window mean
Derived: RH from T and dew point (Magnus); wind speed = mean of hourly sqrt(u^2+v^2);
MERRA-2 PM2.5 = DUSMASS25 + OCSMASS + BCSMASS + SSSMASS25 + SO4SMASS * 132.14/96.06 (x1e9 -> µg/m³).

Output: data/interim/reanalysis_daily.parquet

Usage:
    python -m src.align.reanalysis
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.align.time_windows import in_overpass_window, ist_date
from src.ingest.gee_extract import OUT_DIR as GEE_DIR
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_INTERIM

log = get_logger("reanalysis")

REANALYSIS_PATH: Path = DATA_INTERIM / "reanalysis_daily.parquet"


def relative_humidity(t_k: pd.Series, td_k: pd.Series) -> pd.Series:
    """RH in percent from temperature and dew point (K), Magnus form over water."""
    t, td = t_k - 273.15, td_k - 273.15
    a, b = 17.625, 243.04
    return 100 * np.exp(a * td / (b + td)) / np.exp(a * t / (b + t))


def merra2_pm25(df: pd.DataFrame) -> pd.Series:
    """Reconstructed MERRA-2 surface PM2.5 in µg/m³ (GMAO MERRA-2 FAQ)."""
    kg_m3 = df["DUSMASS25"] + df["OCSMASS"] + df["BCSMASS"] + df["SSSMASS25"] + df["SO4SMASS"] * (132.14 / 96.06)
    return kg_m3 * 1e9


def load_hourly(product: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sorted((GEE_DIR / product).glob("*.csv"))], ignore_index=True)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df.drop_duplicates(["station_id", "time_utc"])


def hourly_meteorology(land: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    m = land.merge(era5, on=["station_id", "time_utc"], how="outer", suffixes=("_land", "_era5"))
    out = pd.DataFrame({"station_id": m["station_id"], "time_utc": m["time_utc"]})
    for var in ["temperature_2m", "dewpoint_temperature_2m", "u_component_of_wind_10m", "v_component_of_wind_10m"]:
        out[var] = m[f"{var}_land"].fillna(m[f"{var}_era5"])
    out["from_land"] = m["temperature_2m_land"].notna()
    out["precip_mm"] = m["total_precipitation_hourly"].fillna(m["total_precipitation"]) * 1000
    out["blh"] = m["boundary_layer_height"]
    out["t2m_c"] = out["temperature_2m"] - 273.15
    out["rh"] = relative_humidity(out["temperature_2m"], out["dewpoint_temperature_2m"])
    out["u10"] = out["u_component_of_wind_10m"]
    out["v10"] = out["v_component_of_wind_10m"]
    out["wind_speed"] = np.hypot(out["u10"], out["v10"])
    return out


def aggregate(df: pd.DataFrame, cols_mean: list[str], cols_sum: list[str], window: dict,
              op_cols: list[str]) -> pd.DataFrame:
    df = df.assign(date_ist=ist_date(df["time_utc"]),
                   in_op=in_overpass_window(df["time_utc"], window["start_hour"], window["end_hour"]))
    g = df.groupby(["station_id", "date_ist"])
    daily = g[cols_mean].mean().join(g[cols_sum].sum(min_count=1)).join(g.size().rename("n_hours"))
    op = df[df["in_op"]].groupby(["station_id", "date_ist"])[op_cols].mean().add_suffix("_op")
    return daily.join(op).reset_index()


def main() -> Path:
    window = load_config("alignment")["overpass_window"]
    met = hourly_meteorology(load_hourly("era5_land"), load_hourly("era5"))
    met_d = aggregate(met, ["t2m_c", "rh", "wind_speed", "u10", "v10", "blh", "from_land"], ["precip_mm"], window,
                      ["t2m_c", "rh", "wind_speed", "blh"]).rename(columns={"from_land": "met_land_share",
                                                                          "n_hours": "met_hours"})
    m2 = load_hourly("merra2")
    m2 = m2.assign(merra2_pm25=merra2_pm25(m2), merra2_aod=m2["TOTEXTTAU"])
    m2_d = aggregate(m2, ["merra2_pm25", "merra2_aod"], [], window, ["merra2_pm25", "merra2_aod"]).rename(
        columns={"n_hours": "merra2_hours"})
    out = met_d.merge(m2_d, on=["station_id", "date_ist"], how="outer")
    out.to_parquet(REANALYSIS_PATH, index=False)
    log.info("Reanalysis daily rows: %d; stations: %d; days with < 24 met hours: %d",
             len(out), out["station_id"].nunique(), int((out["met_hours"] < 24).sum()))
    return REANALYSIS_PATH


if __name__ == "__main__":
    main()
