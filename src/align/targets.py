"""Two PM2.5 target definitions per station and IST day (Stage 4, task 3).

    pm25_daily      mean of QC-passed hours in the IST day; valid if >= daily.min_valid_hours
    pm25_overpass   mean of QC-passed hours starting 12:00-14:00 IST (window 12-15);
                    valid if >= overpass_window.min_valid_hours

Invalid targets are NaN (not dropped), with the hour counts kept, so the reconciliation
from hours to days is visible.

Input:  data/interim/ground_pm25_hourly_qc.parquet (qc_flag == "ok" only)
Output: data/interim/targets_daily.parquet

Usage:
    python -m src.align.targets
"""

from pathlib import Path

import pandas as pd

from src.align.time_windows import in_overpass_window, ist_date
from src.clean.pm25_qc import QC_PATH
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_INTERIM

log = get_logger("targets")

TARGETS_PATH: Path = DATA_INTERIM / "targets_daily.parquet"


def build_targets(h: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Daily and overpass-window targets from QC-passed hourly rows."""
    h = h[h["qc_flag"] == "ok"].copy()
    h["date_ist"] = ist_date(h["timestamp_utc"])
    w = cfg["overpass_window"]
    h["in_window"] = in_overpass_window(h["timestamp_utc"], w["start_hour"], w["end_hour"])
    daily = h.groupby(["station_id", "date_ist"])["pm25"].agg(pm25_daily="mean", n_hours_daily="count")
    win = (h[h["in_window"]].groupby(["station_id", "date_ist"])["pm25"]
           .agg(pm25_overpass="mean", n_hours_overpass="count"))
    out = daily.join(win, how="left").reset_index()
    out["n_hours_overpass"] = out["n_hours_overpass"].fillna(0).astype(int)
    out.loc[out["n_hours_daily"] < cfg["daily"]["min_valid_hours"], "pm25_daily"] = float("nan")
    out.loc[out["n_hours_overpass"] < w["min_valid_hours"], "pm25_overpass"] = float("nan")
    return out


def main() -> Path:
    cfg = load_config("alignment")
    h = pd.read_parquet(QC_PATH, columns=["station_id", "timestamp_utc", "pm25", "qc_flag"])
    t = build_targets(h, cfg)
    t.to_parquet(TARGETS_PATH, index=False)
    log.info("Station-days with any QC-passed hour: %d; valid daily: %d; valid overpass: %d; both: %d",
             len(t), int(t["pm25_daily"].notna().sum()), int(t["pm25_overpass"].notna().sum()),
             int((t["pm25_daily"].notna() & t["pm25_overpass"].notna()).sum()))
    return TARGETS_PATH


if __name__ == "__main__":
    main()
