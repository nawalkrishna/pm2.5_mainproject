"""Count candidate data-quality problems (Stage 3, task 4). Nothing is removed here.

The rules below only *flag* hours/days so their size can be judged; the QC rules that
are actually applied are proposed to the student and decided in Stage 4.

Hourly ground PM2.5 (data/interim/ground_pm25_hourly.parquet, 2021-2025):
    negative        pm25 < 0
    zero            pm25 == 0
    very_high       pm25 > 500
    at_1000         pm25 == 1000 (possible instrument ceiling)
    flat_6h/12h/24h hour belongs to a run of >= N consecutive hours with an identical value
    spike           pm25 > 4 x the centred 25-hour rolling median of the station and pm25 > 100
    gap_7d          station outages: runs of >= 7 days with no valid day
Satellite (eda_daily.parquet):
    negative NO2 / SO2 / HCHO columns, MAIAC AOD > 3

Outputs (reports/tables/):
    eda_qc_summary.csv       one row per rule: hours/days flagged, share, stations affected
    eda_qc_by_station.csv    flagged hours per station and rule
    eda_qc_gaps.csv          one row per outage of >= 7 days

Usage:
    python -m src.eda.data_quality
"""

import numpy as np
import pandas as pd

from src.eda.build_daily import OUT_PATH as DAILY_PATH
from src.ingest.ground_pm25 import HOURLY_PATH
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import REPORTS_TABLES

log = get_logger("data_quality")

FLAT_RUNS = [6, 12, 24]
SPIKE_FACTOR, SPIKE_MIN = 4.0, 100.0
GAP_DAYS = 7


def run_lengths(values: pd.Series, times: pd.Series) -> pd.Series:
    """Length of the run of identical consecutive hourly values each row belongs to.

    A run breaks when the value changes or when the next row is not exactly one hour later.
    """
    new_run = (values != values.shift()) | (times.diff() != pd.Timedelta(hours=1))
    run_id = new_run.cumsum()
    return run_id.map(run_id.value_counts())


def flag_hourly(h: pd.DataFrame) -> pd.DataFrame:
    h = h.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    flags = pd.DataFrame({"station_id": h["station_id"]})
    flags["negative"] = h["pm25"] < 0
    flags["zero"] = h["pm25"] == 0
    flags["very_high"] = h["pm25"] > 500
    flags["at_1000"] = h["pm25"] == 1000
    runs = h.groupby("station_id", group_keys=False).apply(
        lambda g: run_lengths(g["pm25"], g["timestamp_utc"]), include_groups=False)
    for n in FLAT_RUNS:
        flags[f"flat_{n}h"] = runs.to_numpy() >= n
    med = (h.set_index("timestamp_utc").groupby("station_id")["pm25"]
           .rolling("25h", center=True, min_periods=12).median().reset_index(drop=True))
    flags["spike"] = (h["pm25"] > SPIKE_FACTOR * med.to_numpy()) & (h["pm25"] > SPIKE_MIN)
    return flags


def outages(daily: pd.DataFrame) -> pd.DataFrame:
    """Runs of >= GAP_DAYS consecutive days without a valid PM2.5 day, inside each station's active span."""
    rows = []
    for sid, g in daily.groupby("station_id"):
        g = g.sort_values("date_ist")
        valid = g["pm25_valid"].to_numpy()
        if not valid.any():
            continue
        first, last = np.argmax(valid), len(valid) - 1 - np.argmax(valid[::-1])
        seg = g.iloc[first:last + 1]
        miss = ~seg["pm25_valid"]
        run_id = (miss != miss.shift()).cumsum()
        for _, r in seg[miss].groupby(run_id[miss]):
            if len(r) >= GAP_DAYS:
                rows.append({"station_id": sid, "start": r["date_ist"].iloc[0], "end": r["date_ist"].iloc[-1],
                             "days": len(r), "sw_monsoon_days": int(r["month"].between(6, 9).sum())})
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    period = load_config("study_area")["study_period"]
    h = pd.read_parquet(HOURLY_PATH, columns=["station_id", "timestamp_utc", "timestamp_ist", "pm25"])
    h = h[h["timestamp_ist"].dt.strftime("%Y-%m-%d").between(period["start"], period["end"])]
    flags = flag_hourly(h)
    rules = [c for c in flags.columns if c != "station_id"]
    summary = [{"level": "hourly", "rule": r, "flagged": int(flags[r].sum()),
                "share": round(float(flags[r].mean()), 5),
                "stations_affected": int(flags.loc[flags[r], "station_id"].nunique())} for r in rules]
    flags.groupby("station_id")[rules].sum().reset_index().to_csv(REPORTS_TABLES / "eda_qc_by_station.csv", index=False)

    d = pd.read_parquet(DAILY_PATH)
    d = d[d["date_ist"].between(period["start"], period["end"])]
    gaps = outages(d)
    gaps.to_csv(REPORTS_TABLES / "eda_qc_gaps.csv", index=False)
    summary.append({"level": "daily", "rule": f"outage_ge_{GAP_DAYS}d", "flagged": int(gaps["days"].sum()),
                    "share": np.nan, "stations_affected": int(gaps["station_id"].nunique())})
    for col, rule in [("no2_trop_b10", "negative NO2 column (10 km mean)"), ("so2_b10", "negative SO2 column (10 km mean)"),
                      ("hcho_b10", "negative HCHO column (10 km mean)")]:
        x = d[col].dropna()
        summary.append({"level": "satellite", "rule": rule, "flagged": int((x < 0).sum()),
                        "share": round(float((x < 0).mean()), 4), "stations_affected": int(d.loc[d[col] < 0, "station_id"].nunique())})
    x = d["aod_clear_b10"].dropna()
    summary.append({"level": "satellite", "rule": "MAIAC AOD > 3 (10 km mean)", "flagged": int((x > 3).sum()),
                    "share": round(float((x > 3).mean()), 5), "stations_affected": int(d.loc[d["aod_clear_b10"] > 3, "station_id"].nunique())})
    out = pd.DataFrame(summary)
    out.to_csv(REPORTS_TABLES / "eda_qc_summary.csv", index=False)
    log.info("QC candidates:\n%s", out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
