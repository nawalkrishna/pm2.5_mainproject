"""Check the Earth Engine extraction is complete and summarise availability (Stage 2 deliverable).

Checks per chunk file: expected row count (stations x days, or stations x hours), no
duplicate keys, and at least one non-null value (an all-null chunk means a failed extraction). Summaries: share of station-days (or station-hours) with a value, per
product, layer, geometry and year; and the same for Kerala in June-September.

Outputs (reports/tables/):
    gee_chunk_check.csv          one row per chunk file
    gee_availability.csv         product x layer x geometry x year: share with a value
    gee_availability_kerala_jjas.csv   same, Kerala stations, June-September only

Usage:
    python -m src.audit.gee_inventory
"""

import pandas as pd

from src.ingest.gee_extract import OUT_DIR, station_points
from src.ingest.ground_pm25 import STATIONS_PATH
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import REPORTS_TABLES

log = get_logger("gee_inventory")


def load_product(product: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted((OUT_DIR / product).glob("*.csv"))], ignore_index=True)


def value_columns(df: pd.DataFrame, kind: str) -> list[str]:
    if kind == "daily_mean":
        return [c for c in df.columns if c.endswith("_mean")]
    return [c for c in df.columns if c not in ("station_id", "time_utc", "geometry")]


def chunk_check(product: str, spec: dict, n_stations: int) -> list[dict]:
    rows = []
    for f in sorted((OUT_DIR / product).glob("*.csv")):
        df = pd.read_csv(f)
        if spec["kind"] == "daily_mean":
            expected = n_stations * df["date_ist"].nunique()
            dups = int(df.duplicated(["station_id", "date_ist"]).sum())
            n_days = df["date_ist"].nunique()
            month = f.stem.split("_")[1]
            full_month = n_days == pd.Period(month).days_in_month
        elif spec["kind"] == "hourly":
            expected = n_stations * df["time_utc"].nunique()
            dups = int(df.duplicated(["station_id", "time_utc"]).sum())
            span_h = (pd.to_datetime(f.stem.split("_")[-1]) - pd.to_datetime(f.stem.split("_")[-2])) / pd.Timedelta(hours=1)
            full_month = df["time_utc"].nunique() == span_h
        else:
            expected, dups, full_month = n_stations, int(df.duplicated("station_id").sum()), True
        values = df[value_columns(df, spec["kind"]) if spec["kind"] != "static" else
                    [c for c in df.columns if c not in ("station_id", "geometry")]]
        has_values = bool(values.notna().any().any())
        rows.append({"product": product, "file": f.name, "rows": len(df), "expected_rows": expected,
                     "duplicates": dups, "complete_time_axis": bool(full_month), "has_values": has_values,
                     "ok": len(df) == expected and dups == 0 and full_month and has_values})
    return rows


def availability(df: pd.DataFrame, cols: list[str], by: list[str]) -> pd.DataFrame:
    return (df[by + cols].assign(**{c: df[c].notna() for c in cols})
            .groupby(by)[cols].mean().round(3).reset_index())


def main() -> pd.DataFrame:
    cfg = load_config("gee_products")
    window = load_config("study_area")["extraction_period"]
    stations = station_points()
    states = pd.read_csv(STATIONS_PATH).set_index("station_id")["state"]

    checks, avail, kerala = [], [], []
    for product, spec in cfg["products"].items():
        checks += chunk_check(product, spec, len(stations))
        if spec["kind"] == "static":
            continue
        df = load_product(product)
        cols = value_columns(df, spec["kind"])
        if spec["kind"] == "daily_mean":
            df = df[df["date_ist"].between(window["start"], window["end"])]
            t = pd.to_datetime(df["date_ist"])
            by = ["geometry"]
        else:
            t = pd.to_datetime(df["time_utc"]).dt.tz_convert("Asia/Kolkata")
            keep = t.dt.strftime("%Y-%m-%d").between(window["start"], window["end"])
            df, t = df[keep], t[keep]
            by = []
        df = df.assign(year=t.dt.year.to_numpy(), month=t.dt.month.to_numpy(),
                       state=df["station_id"].map(states).to_numpy())
        a = availability(df, cols, by + ["year"]).melt(id_vars=by + ["year"], var_name="layer", value_name="share_with_value")
        avail.append(a.assign(product=product))
        k = df[(df["state"] == "Kerala") & df["month"].between(6, 9)]
        kk = availability(k, cols, by + ["year"]).melt(id_vars=by + ["year"], var_name="layer", value_name="share_with_value")
        kerala.append(kk.assign(product=product))

    chk = pd.DataFrame(checks)
    chk.to_csv(REPORTS_TABLES / "gee_chunk_check.csv", index=False)
    order = ["product", "layer", "geometry", "year", "share_with_value"]
    pd.concat(avail).reindex(columns=order).to_csv(REPORTS_TABLES / "gee_availability.csv", index=False)
    pd.concat(kerala).reindex(columns=order).to_csv(REPORTS_TABLES / "gee_availability_kerala_jjas.csv", index=False)
    log.info("Chunk files: %d, not ok: %d", len(chk), int((~chk["ok"]).sum()))
    if (~chk["ok"]).any():
        log.warning("Problem chunks:\n%s", chk[~chk["ok"]].to_string(index=False))
    return chk


if __name__ == "__main__":
    main()
