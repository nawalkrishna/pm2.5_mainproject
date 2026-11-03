"""Effect of candidate QC rules and satellite thresholds on data retention (Stage 4, tasks 1-2).

Nothing is applied to the data here: this produces the numbers the student needs to
choose the rules. Study period 2021-2025.

PM2.5 scenarios (cumulative): each removes hours, then valid days (>= 18 h) are recounted.
    S0  as received
    S1  + remove physically invalid: negative, zero, exactly 1000 µg/m³
    S2a + remove flat-line runs >= 24 h           S2b ... >= 12 h           S2c ... >= 6 h
    S3a S2a + remove isolated spikes (spike runs of 1-2 h)
    S3b S2a + remove every spike-flagged hour
Spike = > 4 x centred 25 h rolling median and > 100 µg/m³ (data_quality.py).

Satellite options, on valid PM2.5 days of scenario S0:
    MAIAC QA: any / clear (cloud mask) / best (clear + QA for AOD = best)
    TROPOMI NO2 day-level cloud filter: drop the station-day if the 10 km mean
        cloud fraction exceeds t (t = none, 0.5, 0.3, 0.2)
    Minimum valid-pixel share in the 10 km buffer: >= 0 (any pixel), 10 %, 25 %, 50 %

Outputs (reports/tables/):
    qc_pm25_scenarios.csv     scenario x region: hours removed, valid days kept
    qc_satellite_options.csv  option x region: station-days with a value

Usage:
    python -m src.clean.qc_options
"""

import numpy as np
import pandas as pd

from src.eda.build_daily import OUT_PATH as DAILY_PATH
from src.eda.data_quality import flag_hourly, run_lengths
from src.ingest.ground_pm25 import HOURLY_PATH, STATIONS_PATH
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import REPORTS_TABLES

log = get_logger("qc_options")

B10_PIXELS = {"no2_trop": 254, "aod_clear": 366}    # nominal pixels in a 10 km circle (0.01° / 926 m grids)


def isolated(flag: pd.Series, station: pd.Series, times: pd.Series, max_len: int = 2) -> pd.Series:
    """Flagged hours that belong to a run of at most ``max_len`` consecutive flagged hours."""
    out = pd.Series(False, index=flag.index)
    for _, idx in flag.groupby(station).groups.items():
        f, t = flag.loc[idx], times.loc[idx]
        # run of identical flag values (True runs only matter)
        lengths = run_lengths(f.astype(int), t)
        out.loc[idx] = f & (lengths <= max_len)
    return out


def regions(df: pd.DataFrame) -> dict[str, pd.Series]:
    kerala = df["state"] == "Kerala"
    jjas = df["month"].between(6, 9)
    return {"all": pd.Series(True, index=df.index), "kerala": kerala, "kerala_jjas": kerala & jjas,
            "other_jjas": ~kerala & jjas}


def pm25_scenarios(h: pd.DataFrame, min_hours: int) -> pd.DataFrame:
    flags = flag_hourly(h)
    h = h.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    invalid = flags["negative"] | flags["zero"] | flags["at_1000"]
    iso = isolated(flags["spike"], h["station_id"], h["timestamp_utc"])
    scen = {"S0 as received": pd.Series(False, index=h.index),
            "S1 invalid values": invalid,
            "S2a S1 + flat >= 24 h": invalid | flags["flat_24h"],
            "S2b S1 + flat >= 12 h": invalid | flags["flat_12h"],
            "S2c S1 + flat >= 6 h": invalid | flags["flat_6h"],
            "S3a S2a + isolated spikes": invalid | flags["flat_24h"] | iso,
            "S3b S2a + all spikes": invalid | flags["flat_24h"] | flags["spike"]}
    h["date_ist"] = h["timestamp_ist"].dt.strftime("%Y-%m-%d")
    h["month"] = h["timestamp_ist"].dt.month
    rows = []
    for name, drop in scen.items():
        kept = h[~drop]
        days = kept.groupby(["station_id", "date_ist", "state", "month"]).size().reset_index(name="hours")
        days = days[days["hours"] >= min_hours]
        for reg, mask in regions(days).items():
            hmask = regions(h)[reg]
            rows.append({"scenario": name, "region": reg, "hours_removed": int((drop & hmask).sum()),
                         "valid_days": int(mask.sum())})
    out = pd.DataFrame(rows)
    base = out[out["scenario"] == "S0 as received"].set_index("region")["valid_days"]
    out["valid_days_lost"] = base.reindex(out["region"]).to_numpy() - out["valid_days"]
    out["share_kept"] = (out["valid_days"] / base.reindex(out["region"]).to_numpy()).round(4)
    return out


def satellite_options(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reg = regions(d)

    def add(group, option, has):
        for r, m in reg.items():
            rows.append({"group": group, "option": option, "region": r, "station_days": int(m.sum()),
                         "with_value": int((has & m).sum()), "share": round(float((has & m).sum() / m.sum()), 3)})

    for g in ["pt", "b10"]:
        for q in ["any", "clear", "best"]:
            add(f"MAIAC QA ({g})", q, d[f"aod_{q}_{g}"].notna())
    for g in ["pt", "b10"]:
        for t in [None, 0.5, 0.3, 0.2]:
            has = d[f"no2_trop_{g}"].notna()
            if t is not None:
                has &= d[f"no2_cloud_{g}"] <= t
            add(f"NO2 day-level cloud filter ({g})", "none" if t is None else f"cloud <= {t}", has)
    for col, label in [("no2_trop", "NO2"), ("aod_clear", "MAIAC AOD clear")]:
        for share in [0, 0.10, 0.25, 0.50]:
            has = d[f"{col}_b10"].notna() & (d[f"{col}_n_b10"] >= max(1, share * B10_PIXELS[col]))
            add(f"{label}: min valid-pixel share (b10)", f">= {int(share * 100)} %", has)
    return pd.DataFrame(rows)


def main() -> None:
    period = load_config("study_area")["study_period"]
    min_hours = load_config("audit")["min_hours_per_valid_day"]
    states = pd.read_csv(STATIONS_PATH).set_index("station_id")["state"]
    h = pd.read_parquet(HOURLY_PATH, columns=["station_id", "timestamp_utc", "timestamp_ist", "pm25"])
    h = h[h["timestamp_ist"].dt.strftime("%Y-%m-%d").between(period["start"], period["end"])].copy()
    h = h.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    h["state"] = h["station_id"].map(states)
    h["month"] = h["timestamp_ist"].dt.month
    ps = pm25_scenarios(h, min_hours)
    ps.to_csv(REPORTS_TABLES / "qc_pm25_scenarios.csv", index=False)

    d = pd.read_parquet(DAILY_PATH)
    d = d[d["pm25_valid"] & d["date_ist"].between(period["start"], period["end"])]
    so = satellite_options(d)
    so.to_csv(REPORTS_TABLES / "qc_satellite_options.csv", index=False)
    log.info("PM2.5 scenarios:\n%s", ps.pivot(index="scenario", columns="region", values="valid_days").to_string())
    log.info("Satellite options:\n%s", so.pivot_table(index=["group", "option"], columns="region", values="share").to_string())


if __name__ == "__main__":
    main()
