"""Download OpenAQ daily PM2.5 aggregates for every sensor, one file per sensor-year.

We use ``/v3/sensors/{id}/days`` for the audit because it costs one request per
sensor-year instead of ~9 for hourly data. Each daily record carries a
``coverage.observedCount`` (number of hourly values behind the daily mean),
which is what the completeness rule needs. OpenAQ defines the day as
01:00 to 00:00 local time (https://docs.openaq.org/resources/measurements).

Raw output: ``data/raw/openaq/days/sensor_<id>_<year>.json`` (skipped if present,
so the script can be stopped and resumed).
Interim output: ``data/interim/openaq_daily.csv``

Usage:
    python -m src.ingest.openaq_daily [--max-sensors N]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ingest.openaq_inventory import OUT_PATH as SENSORS_PATH
from src.ingest.openaq_locations import make_client
from src.utils.config import config_hash, load_config
from src.utils.logging_setup import get_logger
from src.utils.manifest import record_file
from src.utils.paths import DATA_INTERIM, resolve
from src.utils.run_log import log_run

log = get_logger("openaq_daily")

DAILY_PATH: Path = DATA_INTERIM / "openaq_daily.csv"


def years_to_fetch(first_utc: str | None, last_utc: str | None, start: str, end: str) -> list[int]:
    """Years in the audit window that overlap the location's reported date span.

    If OpenAQ reports no span, fetch every year in the window rather than guess.
    """
    lo, hi = int(start[:4]), int(end[:4])
    if isinstance(first_utc, str) and first_utc:
        lo = max(lo, int(first_utc[:4]))
    if isinstance(last_utc, str) and last_utc:
        hi = min(hi, int(last_utc[:4]))
    return list(range(lo, hi + 1))


def parse_days(sensor_id: int, results: list[dict]) -> pd.DataFrame:
    """Turn /days results into a tidy table keyed by sensor and local date."""
    rows = []
    for r in results:
        cov = r.get("coverage") or {}
        period_from = ((r.get("period") or {}).get("datetimeFrom") or {})
        rows.append({
            "sensor_id": sensor_id,
            "period_from_local": period_from.get("local"),
            "period_from_utc": period_from.get("utc"),
            "pm25_daily_mean": r.get("value"),
            "observed_count": cov.get("observedCount"),
            "expected_count": cov.get("expectedCount"),
            "percent_complete": cov.get("percentComplete"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # The local timestamp string starts with the local calendar date (YYYY-MM-DD).
    df["date_local"] = df["period_from_local"].str[:10]
    return df


def main(max_sensors: int | None = None) -> Path:
    """Fetch (or reuse) every sensor-year file, then rebuild the interim daily table."""
    src_cfg = load_config("sources")["openaq"]
    window = load_config("study_area")["audit_period"]
    days_dir = resolve(src_cfg["raw_dir"]) / "days"
    days_dir.mkdir(parents=True, exist_ok=True)

    sensors = pd.read_csv(SENSORS_PATH)
    if max_sensors:
        sensors = sensors.head(max_sensors)

    client = None
    for i, s in enumerate(sensors.itertuples(), start=1):
        for year in years_to_fetch(s.first_utc, s.last_utc, window["start"], window["end"]):
            out = days_dir / f"sensor_{s.sensor_id}_{year}.json"
            if out.exists():
                continue
            client = client or make_client(src_cfg)
            # Ask for a slightly wider range and filter by local date later,
            # so no day at the year boundary is lost to timezone offsets.
            params = {"date_from": f"{year}-01-01", "date_to": f"{year + 1}-01-02"}
            results = list(client.get_all(f"sensors/{s.sensor_id}/days", params))
            out.write_text(json.dumps({"sensor_id": s.sensor_id, "year": year,
                                       "query": params, "results": results}), encoding="utf-8")
            record_file(out, source="OpenAQ v3 /sensors/{id}/days", date_range=str(year),
                        rows=len(results))
        log.info("[%d/%d] sensor %s done (%d requests so far)", i, len(sensors), s.sensor_id,
                 client.request_count if client else 0)

    frames = []
    for path in sorted(days_dir.glob("sensor_*_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        df = parse_days(payload["sensor_id"], payload["results"])
        if not df.empty:
            df = df[df["date_local"].str[:4] == str(payload["year"])]
            frames.append(df)
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    n_before = len(daily)
    daily = daily.drop_duplicates(subset=["sensor_id", "date_local"])
    log.info("Daily rows: %d (%d exact sensor-date duplicates dropped)", len(daily), n_before - len(daily))

    daily.to_csv(DAILY_PATH, index=False)
    log_run("openaq_daily", config_hash({"sources": src_cfg, "window": window}),
            metrics={"sensors": len(sensors), "daily_rows": len(daily),
                     "dup_dropped": n_before - len(daily)})
    return DAILY_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sensors", type=int, default=None, help="limit for a quick test run")
    main(parser.parse_args().max_sensors)
