"""Settle the CPCB hourly timestamp convention by lining CPCB up with OpenAQ (decision D024).

CPCB CCR files label each hour ``YYYY-MM-DD HH:00:00`` with no timezone and no
statement of whether the label is the start or the end of the hour. OpenAQ
republishes the same monitors with explicit UTC start and end times. For a few
stations and two sample weeks in 2025 (when OpenAQ has one record per hour), we
fetch OpenAQ ``/hours`` and compare with CPCB under several hypotheses:

    utc_start = label - 5:30 + k hours,   k in -3..+3

k = 0 means "IST hour-start label"; k = -1 means "IST hour-end label". The
hypothesis with near-identical values is the convention.

Raw output: data/raw/openaq/hours/sensor_<id>_<from>_<to>.json (reused if present)
Output:     reports/tables/cpcb_timestamp_check.csv (one row per station x offset)

Usage:
    python -m src.audit.cpcb_timestamp_check
"""

import json
from pathlib import Path

import pandas as pd

from src.ingest.cpcb_hourly import parse_name
from src.ingest.openaq_daily import DAILY_PATH as OPENAQ_DAILY
from src.ingest.openaq_inventory import OUT_PATH as OPENAQ_SENSORS
from src.ingest.openaq_locations import make_client
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.manifest import record_file
from src.utils.paths import REPORTS_TABLES, resolve

log = get_logger("cpcb_timestamp_check")

WEEKS = [("2025-03-03", "2025-03-10"), ("2025-07-07", "2025-07-14")]
OFFSETS_H = range(-3, 4)
IST = pd.Timedelta(hours=5, minutes=30)


def cpcb_hourly(path: Path) -> pd.Series:
    """CPCB PM2.5 indexed by the naive timestamp label in the file."""
    df = pd.read_csv(path, encoding="utf-8")
    col = [c for c in df.columns if c.startswith("PM2.5")][0]
    return pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(),
                     index=pd.to_datetime(df["Timestamp"]), name="cpcb").dropna()


def openaq_hours(client, sensor_id: int, start: str, end: str, raw_dir: Path) -> pd.Series:
    """OpenAQ hourly PM2.5 indexed by the UTC start of the hour (naive UTC)."""
    path = raw_dir / f"sensor_{sensor_id}_{start}_{end}.json"
    if not path.exists():
        results = list(client.get_all(f"sensors/{sensor_id}/hours",
                                      {"datetime_from": start, "datetime_to": end}))
        path.write_text(json.dumps({"sensor_id": sensor_id, "results": results}), encoding="utf-8")
        record_file(path, source=f"OpenAQ v3 /sensors/{sensor_id}/hours", date_range=f"{start}..{end}",
                    rows=len(results), notes="CPCB timestamp check (D024)")
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    idx = pd.to_datetime([r["period"]["datetimeFrom"]["utc"] for r in results], utc=True).tz_localize(None)
    return pd.Series([r.get("value") for r in results], index=idx, name="openaq").dropna()


def compare(cpcb: pd.Series, openaq: pd.Series, offsets=OFFSETS_H) -> pd.DataFrame:
    """Agreement between CPCB and OpenAQ for each candidate offset k (hours)."""
    rows = []
    for k in offsets:
        shifted = cpcb.copy()
        shifted.index = shifted.index - IST + pd.Timedelta(hours=k)
        both = pd.concat([shifted, openaq], axis=1, join="inner").dropna()
        diff = (both["cpcb"] - both["openaq"]).abs()
        rows.append({"offset_h": k, "pairs": len(both),
                     "share_within_0_5": float((diff <= 0.5).mean()) if len(both) else float("nan"),
                     "median_abs_diff": float(diff.median()) if len(both) else float("nan"),
                     "pearson_r": float(both["cpcb"].corr(both["openaq"])) if len(both) > 2 else float("nan")})
    return pd.DataFrame(rows)


def pick_stations(sites: pd.DataFrame, sensors: pd.DataFrame, daily: pd.DataFrame,
                  n_per_state: int = 1) -> pd.DataFrame:
    """One usable, OpenAQ-matched station per state, using its sensor with most 2025 days.

    Activity is taken from the daily records, not the location-level ``last_utc``,
    because one location can hold an old sensor that stopped in 2022 and a new one.
    """
    days_2025 = daily[daily["date_local"].str[:4] == "2025"].groupby("sensor_id").size()
    rows = []
    for s in sites[sites["usable_study_period"] & sites["openaq_location_ids"].notna()].itertuples():
        loc_ids = [int(i) for i in str(s.openaq_location_ids).split(";")]
        cand = sensors.loc[sensors["location_id"].isin(loc_ids), "sensor_id"]
        counts = days_2025.reindex(cand).dropna()
        if not counts.empty and counts.max() >= 300:
            rows.append({"cpcb_site": s.cpcb_site, "station": s.station, "state": s.state,
                         "sensor_id": int(counts.idxmax())})
    return pd.DataFrame(rows).groupby("state").head(n_per_state)


def main() -> pd.DataFrame:
    src_cfg = load_config("sources")
    raw_dir = resolve(src_cfg["openaq"]["raw_dir"]) / "hours"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(src_cfg["openaq"])
    sites = pd.read_csv(REPORTS_TABLES / "cpcb_site_inventory.csv")
    picks = pick_stations(sites, pd.read_csv(OPENAQ_SENSORS), pd.read_csv(OPENAQ_DAILY))
    files = {parse_name(p.name)["cpcb_site"]: p
             for p in (resolve(src_cfg["cpcb"]["raw_dir"]) / "1Hr" / "2025").glob("Raw_Data_*_1Hr.csv")}

    out = []
    for p in picks.itertuples():
        cpcb = cpcb_hourly(files[p.cpcb_site])
        oa = pd.concat([openaq_hours(client, p.sensor_id, a, b, raw_dir) for a, b in WEEKS])
        res = compare(cpcb, oa)
        res.insert(0, "station", p.station)
        res.insert(1, "sensor_id", p.sensor_id)
        out.append(res)
    table = pd.concat(out, ignore_index=True)
    table.to_csv(REPORTS_TABLES / "cpcb_timestamp_check.csv", index=False)
    ranked = table.dropna(subset=["pearson_r"]).sort_values("pearson_r", ascending=False)
    best = ranked.drop_duplicates("station")[["station", "offset_h", "pearson_r", "median_abs_diff", "pairs"]]
    log.info("Best offset per station:\n%s", best.to_string(index=False))
    return table


if __name__ == "__main__":
    main()
