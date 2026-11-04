"""Standardise all ground PM2.5 sources into one hourly table (Stage 2, task 1).

Schema (one row per station and hour with a PM2.5 value):
    station_id     ``cpcb_###`` (CPCB) or ``openaq_<location_id>`` (OpenAQ-only monitors)
    timestamp_utc  start of the hour, UTC (tz-aware)
    timestamp_ist  same instant in Asia/Kolkata (tz-aware)
    pm25           µg/m³, as received
    unit           ``µg/m³``
    source         ``cpcb_ccr_1h`` or ``openaq_hours``
    qc_flag        ``raw_unchecked`` (QC rules are applied in Stage 4)

Sources:
    * CPCB CCR 1-hour files. Labels are IST hour-start times (decision D024).
    * OpenAQ ``/hours`` for reference monitors that are not in CPCB (decision D015):
      the US Consulate monitors in Hyderabad and Chennai (AirNow). OpenAQ gives
      explicit UTC start times. Raw JSON is cached per sensor-month.

Hours with no PM2.5 value are not stored; their count is written to the removal log.

Outputs:
    data/interim/ground_pm25_hourly.parquet
    data/interim/ground_stations.csv        one row per station_id
    reports/tables/ground_removal_log.csv   rows dropped per source and rule

Usage:
    python -m src.ingest.ground_pm25
"""

import json
from pathlib import Path

import pandas as pd

from src.audit.cpcb_audit import link_sites_to_stations
from src.ingest.cpcb_hourly import FILES_PATH, parse_name
from src.ingest.cpcb_stations import OUT_PATH as CPCB_STATIONS
from src.ingest.openaq_inventory import OUT_PATH as OPENAQ_SENSORS
from src.ingest.openaq_locations import make_client
from src.utils.config import config_hash, load_config
from src.utils.logging_setup import get_logger
from src.utils.manifest import record_file
from src.utils.paths import DATA_INTERIM, REPORTS_TABLES, resolve
from src.utils.run_log import log_run

log = get_logger("ground_pm25")

HOURLY_PATH: Path = DATA_INTERIM / "ground_pm25_hourly.parquet"
STATIONS_PATH: Path = DATA_INTERIM / "ground_stations.csv"
REMOVAL_LOG: Path = REPORTS_TABLES / "ground_removal_log.csv"
UNIT = "µg/m³"
IST = "Asia/Kolkata"

# Reference monitors present in OpenAQ but not in CPCB (station_audit.md §6, decision D015).
OPENAQ_ONLY_SENSORS = {24857: "Hyderabad", 5077812: "Chennai"}


def ist_labels_to_utc(labels: pd.Series) -> pd.Series:
    """CPCB naive labels (IST hour start, D024) -> tz-aware UTC hour start."""
    return pd.to_datetime(labels).dt.tz_localize(IST).dt.tz_convert("UTC")


def cpcb_file_rows(path: Path, station_id: str, start_utc: pd.Timestamp,
                   end_utc: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Standardised rows from one CPCB station-year file, plus counts for the removal log."""
    df = pd.read_csv(path, encoding="utf-8")
    col = [c for c in df.columns if c.startswith("PM2.5")][0]
    raw = df[col]
    pm = pd.to_numeric(raw, errors="coerce")
    ts = ist_labels_to_utc(df["Timestamp"])
    in_window = (ts >= start_utc) & (ts < end_utc)
    counts = {"rows_in_window": int(in_window.sum()),
              "missing_value": int((in_window & raw.isna()).sum()),
              "non_numeric_value": int((in_window & raw.notna() & pm.isna()).sum())}
    keep = in_window & pm.notna()
    out = pd.DataFrame({"station_id": station_id, "timestamp_utc": ts[keep], "pm25": pm[keep]})
    return out, counts


def openaq_sensor_rows(client, sensor_id: int, months: list[str], raw_dir: Path,
                       start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    """Hourly OpenAQ values for one sensor, cached as one JSON file per month.

    Monthly requests fit in a single page; deep pagination over a whole year timed out (HTTP 408).
    """
    frames = []
    for month in months:
        path = raw_dir / f"sensor_{sensor_id}_{month}.json"
        if not path.exists():
            first = pd.Timestamp(f"{month}-01")
            nxt = first + pd.offsets.MonthBegin(1)
            # Pad one day each side (UTC vs IST); rows are filtered to the window below.
            results = list(client.get_all(f"sensors/{sensor_id}/hours",
                                          {"datetime_from": (first - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                           "datetime_to": (nxt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")}))
            path.write_text(json.dumps({"sensor_id": sensor_id, "month": month, "results": results}),
                            encoding="utf-8")
            record_file(path, source=f"OpenAQ v3 /sensors/{sensor_id}/hours", date_range=month,
                        rows=len(results), notes="OpenAQ-only reference monitor (D015)")
        results = json.loads(path.read_text(encoding="utf-8"))["results"]
        if not results:
            continue
        frames.append(pd.DataFrame({
            "timestamp_utc": pd.to_datetime([r["period"]["datetimeFrom"]["utc"] for r in results], utc=True),
            "pm25": pd.to_numeric(pd.Series([r.get("value") for r in results]), errors="coerce"),
            "unit": [(r.get("parameter") or {}).get("units") for r in results],
        }))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp_utc", "pm25", "unit"])
    df = df.drop_duplicates("timestamp_utc")          # padding days overlap between months
    return df[(df["timestamp_utc"] >= start_utc) & (df["timestamp_utc"] < end_utc)]


def one_per_ist_hour(df: pd.DataFrame) -> pd.DataFrame:
    """One row per IST clock hour (decision D044).

    Until October 2024 OpenAQ returns many US Consulate hours twice, as one-hour bins starting
    at :00 and at :30 IST (89 % with the same value). Keep the :00 IST bin; an hour with only
    the :30 bin keeps that value, relabelled to its IST hour start (the :30 bin repeats the
    :00 bin of the same hour). Call after rows without a value are dropped.
    """
    ist = df["timestamp_utc"].dt.tz_convert(IST)
    hour = ist.dt.floor("h")
    df = df.assign(_hour=hour, _offset=ist != hour).sort_values(["_hour", "_offset"], kind="stable")
    df = df.drop_duplicates("_hour", keep="first")
    df["timestamp_utc"] = df["_hour"].dt.tz_convert("UTC")
    return df.drop(columns=["_hour", "_offset"]).reset_index(drop=True)


def finalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add the shared columns in schema order."""
    df = df.copy()
    df["timestamp_ist"] = df["timestamp_utc"].dt.tz_convert(IST)
    df["unit"] = UNIT
    df["source"] = source
    df["qc_flag"] = "raw_unchecked"
    return df[["station_id", "timestamp_utc", "timestamp_ist", "pm25", "unit", "source", "qc_flag"]]


def main() -> Path:
    src_cfg = load_config("sources")
    window = load_config("study_area")["extraction_period"]
    start_utc = pd.Timestamp(window["start"], tz=IST).tz_convert("UTC")
    end_utc = (pd.Timestamp(window["end"], tz=IST) + pd.Timedelta(days=1)).tz_convert("UTC")
    years = list(range(start_utc.tz_convert(IST).year, end_utc.tz_convert(IST).year + 1))
    months = [m.strftime("%Y-%m") for m in pd.period_range(window["start"], window["end"], freq="M")]

    # --- CPCB
    files = pd.read_csv(FILES_PATH)
    cpcb_st = pd.read_csv(CPCB_STATIONS)
    sites = link_sites_to_stations(files, cpcb_st)
    site_to_uid = sites.set_index("cpcb_site")["station_uid"]
    raw_dir = resolve(src_cfg["cpcb"]["raw_dir"]) / "1Hr"
    frames, log_rows = [], []
    for path in sorted(raw_dir.rglob("Raw_Data_*_1Hr.csv")):
        meta = parse_name(path.name)
        if meta["year"] not in years:
            continue
        rows, counts = cpcb_file_rows(path, site_to_uid[meta["cpcb_site"]], start_utc, end_utc)
        frames.append(rows)
        log_rows.append({"source": "cpcb_ccr_1h", "file": path.name, **counts, "rows_kept": len(rows)})
    cpcb = finalise(pd.concat(frames, ignore_index=True), "cpcb_ccr_1h")

    # --- OpenAQ-only monitors
    client = make_client(src_cfg["openaq"])
    oa_dir = resolve(src_cfg["openaq"]["raw_dir"]) / "hours"
    oa_dir.mkdir(parents=True, exist_ok=True)
    sensors = pd.read_csv(OPENAQ_SENSORS).set_index("sensor_id")
    oa_frames = []
    for sensor_id in OPENAQ_ONLY_SENSORS:
        df = openaq_sensor_rows(client, sensor_id, months, oa_dir, start_utc, end_utc)
        bad_unit = df["unit"].notna() & (df["unit"] != UNIT)
        if bad_unit.any():
            raise ValueError(f"sensor {sensor_id}: unexpected units {df.loc[bad_unit, 'unit'].unique()}")
        log_rows.append({"source": "openaq_hours", "file": f"sensor_{sensor_id}",
                         "rows_in_window": len(df), "missing_value": int(df["pm25"].isna().sum()),
                         "non_numeric_value": 0, "rows_kept": int(df["pm25"].notna().sum())})
        df = df[df["pm25"].notna()]
        hourly_df = one_per_ist_hour(df)
        log_rows.append({"source": "openaq_hours", "file": f"sensor_{sensor_id} half-hour duplicate (D044)",
                         "rows_in_window": 0, "missing_value": 0, "non_numeric_value": 0,
                         "rows_kept": -(len(df) - len(hourly_df))})
        oa_frames.append(hourly_df.assign(station_id=f"openaq_{int(sensors.loc[sensor_id, 'location_id'])}"))
    openaq = finalise(pd.concat(oa_frames, ignore_index=True), "openaq_hours")

    hourly = pd.concat([cpcb, openaq], ignore_index=True)
    dup = hourly.duplicated(["station_id", "timestamp_utc"], keep="first")
    if dup.any():
        log_rows.append({"source": "all", "file": "duplicate station-hour", "rows_in_window": 0,
                         "missing_value": 0, "non_numeric_value": 0, "rows_kept": -int(dup.sum())})
        hourly = hourly[~dup]
    hourly = hourly.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    hourly.to_parquet(HOURLY_PATH, index=False)

    # --- station table
    cpcb_rows = cpcb_st.assign(station_id=cpcb_st["station_uid"], name=cpcb_st["station"], source="cpcb_ccr_1h")
    cpcb_rows = cpcb_rows.merge(sites[["station_uid", "cpcb_site"]], on="station_uid", how="left")
    oa_rows = pd.DataFrame([{"station_id": f"openaq_{int(sensors.loc[s, 'location_id'])}",
                             "name": f"US Consulate {city} (AirNow via OpenAQ)", "city": city,
                             "latitude": sensors.loc[s, "latitude"], "longitude": sensors.loc[s, "longitude"],
                             "coord_source": "openaq", "source": "openaq_hours", "openaq_sensor_id": s}
                            for s, city in OPENAQ_ONLY_SENSORS.items()])
    stations = pd.concat([cpcb_rows, oa_rows], ignore_index=True)
    stations["state"] = stations["state"].fillna(stations["city"].map({"Hyderabad": "Telangana",
                                                                       "Chennai": "Tamil Nadu"}))
    cols = ["station_id", "name", "state", "city", "latitude", "longitude", "coord_source", "source",
            "cpcb_site", "openaq_sensor_id"]
    stations[cols].to_csv(STATIONS_PATH, index=False)

    removal = pd.DataFrame(log_rows)
    removal.to_csv(REMOVAL_LOG, index=False)
    totals = removal.groupby("source")[["rows_in_window", "missing_value", "non_numeric_value", "rows_kept"]].sum()
    log.info("Ground PM2.5 rows by source:\n%s\nFinal rows: %d, stations: %d",
             totals.to_string(), len(hourly), hourly["station_id"].nunique())
    log_run("ground_pm25", config_hash({"window": window, "sensors": list(OPENAQ_ONLY_SENSORS)}),
            metrics={"rows": len(hourly), "stations": int(hourly["station_id"].nunique())})
    return HOURLY_PATH


if __name__ == "__main__":
    main()
