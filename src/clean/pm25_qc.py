"""Apply the approved PM2.5 QC rules to the hourly ground data (Stage 4, task 1; decision D028).

Rules come from ``configs/qc.yaml`` and are applied in this order; an hour removed by an
earlier rule is not counted again by a later one:
    1. invalid_negative   pm25 < 0
    2. invalid_zero       pm25 == 0
    3. invalid_exact      pm25 equal to a listed value (1000: instrument ceiling)
    4. flat_run           run of >= flat_run_hours consecutive identical hourly values
    5. spike_isolated     pm25 > factor x centred rolling median and > min_value, in a
                          run of <= max_isolated_run flagged hours (computed after 1-4)

No row is deleted from the file: every hour keeps its ``qc_flag`` (``ok`` or the rule
name) so the cleaning can be audited and re-run with other settings.

Outputs:
    data/interim/ground_pm25_hourly_qc.parquet   hourly table with qc_flag set
    reports/tables/cleaning_log.csv              rows before / removed / after, per rule
    reports/tables/cleaning_log_by_station.csv   removed hours per station and rule

Usage:
    python -m src.clean.pm25_qc [--flat-hours N]
"""

import argparse
from pathlib import Path

import pandas as pd

from src.eda.data_quality import run_lengths
from src.ingest.ground_pm25 import HOURLY_PATH
from src.utils.config import config_hash, load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_INTERIM, REPORTS_TABLES
from src.utils.run_log import log_run

log = get_logger("pm25_qc")

QC_PATH: Path = DATA_INTERIM / "ground_pm25_hourly_qc.parquet"
RULES = ["invalid_negative", "invalid_zero", "invalid_exact", "flat_run", "spike_isolated"]


def per_station_runs(values: pd.Series, h: pd.DataFrame) -> pd.Series:
    """``run_lengths`` per station, aligned to ``values``' index (``groupby.apply`` returns a
    DataFrame instead of a Series when there is only one station)."""
    parts = [run_lengths(values.loc[idx], h.loc[idx, "timestamp_utc"])
             for idx in h.loc[values.index].groupby("station_id").groups.values()]
    return pd.concat(parts).reindex(values.index)


def spike_flags(h: pd.DataFrame, cfg: dict) -> pd.Series:
    """Isolated spikes among the hours still ``ok``; ``h`` sorted by station and time."""
    ok = h[h["qc_flag"] == "ok"]
    med = (ok.set_index("timestamp_utc").groupby("station_id")["pm25"]
           .rolling(f"{cfg['window_hours']}h", center=True, min_periods=cfg["min_periods"]).median())
    med.index = ok.index
    spike = (ok["pm25"] > cfg["factor"] * med) & (ok["pm25"] > cfg["min_value"])
    run = per_station_runs(spike.astype(int), ok)
    isolated = spike & (run <= cfg["max_isolated_run"])
    return isolated.reindex(h.index, fill_value=False)


def apply_qc(h: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Set ``qc_flag`` on a copy of the hourly table (rules in RULES order)."""
    h = h.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True).copy()
    h["qc_flag"] = "ok"

    def mark(mask: pd.Series, rule: str) -> None:
        h.loc[mask & (h["qc_flag"] == "ok"), "qc_flag"] = rule

    inv = cfg["invalid"]
    if inv.get("negative"):
        mark(h["pm25"] < 0, "invalid_negative")
    if inv.get("zero"):
        mark(h["pm25"] == 0, "invalid_zero")
    mark(h["pm25"].isin(inv.get("exact_values", [])), "invalid_exact")
    mark(per_station_runs(h["pm25"], h) >= cfg["flat_run_hours"], "flat_run")
    mark(spike_flags(h, cfg["spike"]), "spike_isolated")
    return h


def cleaning_log(h: pd.DataFrame) -> pd.DataFrame:
    rows, before = [], len(h)
    for rule in RULES:
        n = int((h["qc_flag"] == rule).sum())
        rows.append({"step": "pm25_qc", "rule": rule, "rows_before": before, "rows_removed": n,
                     "rows_after": before - n})
        before -= n
    return pd.DataFrame(rows)


def main(flat_hours: int | None = None) -> Path:
    cfg = load_config("qc")
    if flat_hours:
        cfg["flat_run_hours"] = flat_hours
    h = pd.read_parquet(HOURLY_PATH)
    q = apply_qc(h, cfg)
    log_df = cleaning_log(q)
    assert log_df["rows_after"].iloc[-1] == int((q["qc_flag"] == "ok").sum())
    q.to_parquet(QC_PATH, index=False)
    log_df.to_csv(REPORTS_TABLES / "cleaning_log.csv", index=False)
    (q[q["qc_flag"] != "ok"].groupby(["station_id", "qc_flag"]).size().unstack(fill_value=0)
     .reset_index().to_csv(REPORTS_TABLES / "cleaning_log_by_station.csv", index=False))
    log.info("PM2.5 QC:\n%s", log_df.to_string(index=False))
    log_run("pm25_qc", config_hash(cfg), metrics=log_df.set_index("rule")["rows_removed"].to_dict())
    return QC_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flat-hours", type=int)
    main(parser.parse_args().flat_hours)
