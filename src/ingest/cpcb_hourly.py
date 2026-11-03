"""Collect CPCB CCR hourly files and summarise them into daily PM2.5 completeness.

Files are downloaded by hand (or by a browser session the student opened, since
the portal has a CAPTCHA) and arrive in the Downloads folder named like
``Raw_Data_2023_site_252_plammoodu_thiruvananthapuram_kerala_pcb_1Hr.csv``.
This script *copies* them, unchanged, into ``data/raw/cpcb/1Hr/<year>/`` and
records each in the data manifest. Files already copied are skipped.

Timestamps: the files run from ``YYYY-01-01 00:00:00`` to ``YYYY-12-31 23:00:00``,
i.e. one label per hour. We treat them as IST hour-start labels for the audit;
this must be confirmed before Stage 2 alignment (see decision log).

Outputs:
    data/interim/cpcb_daily.csv   one row per CPCB site and local date
    data/interim/cpcb_files.csv   one row per file (site, year, rows, PM2.5 hours)

Usage:
    python -m src.ingest.cpcb_hourly [--downloads PATH]
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.manifest import record_file
from src.utils.paths import DATA_INTERIM, resolve

log = get_logger("cpcb_hourly")

FILE_RE = re.compile(r"^Raw_Data_(?P<year>\d{4})_site_(?P<site>\d+)_(?P<slug>.+)_1Hr\.csv$")
DAILY_PATH: Path = DATA_INTERIM / "cpcb_daily.csv"
FILES_PATH: Path = DATA_INTERIM / "cpcb_files.csv"
DEFAULT_DOWNLOADS = Path.home() / "Downloads"


def parse_name(name: str) -> dict | None:
    """Year, CPCB site number and station slug from a CCR file name, or None."""
    m = FILE_RE.match(name)
    if not m:
        return None
    return {"year": int(m["year"]), "cpcb_site": f"site_{m['site']}", "slug": m["slug"]}


def collect(downloads: Path, raw_dir: Path) -> int:
    """Copy new CCR hourly files from ``downloads`` into ``raw_dir``. Returns number copied."""
    copied = 0
    for src in sorted(downloads.glob("Raw_Data_*_1Hr.csv")):
        meta = parse_name(src.name)
        if meta is None:
            continue
        dest = raw_dir / "1Hr" / str(meta["year"]) / src.name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        record_file(dest, source="CPCB CCR Data Repository (1H)", date_range=str(meta["year"]),
                    rows=None, notes=f"{meta['cpcb_site']} {meta['slug']}")
        copied += 1
    return copied


def daily_from_file(path: Path) -> tuple[pd.DataFrame, dict]:
    """Daily PM2.5 mean and number of hours with a PM2.5 value, for one station-year file."""
    meta = parse_name(path.name)
    df = pd.read_csv(path, encoding="utf-8")
    pm_cols = [c for c in df.columns if c.startswith("PM2.5")]
    if len(pm_cols) != 1:
        raise ValueError(f"{path.name}: expected one PM2.5 column, found {pm_cols}")
    pm = pd.to_numeric(df[pm_cols[0]], errors="coerce")
    ts = pd.to_datetime(df["Timestamp"])
    day = ts.dt.strftime("%Y-%m-%d")
    daily = (pd.DataFrame({"date_local": day, "pm25": pm})
             .groupby("date_local")["pm25"].agg(pm25_daily_mean="mean", observed_count="count")
             .reset_index())
    daily.insert(0, "cpcb_site", meta["cpcb_site"])
    info = {**meta, "file": path.name, "rows": len(df), "pm25_hours": int(pm.notna().sum()),
            "pm25_column": pm_cols[0], "first_ts": str(ts.min()), "last_ts": str(ts.max())}
    return daily, info


def main(downloads: Path = DEFAULT_DOWNLOADS) -> Path:
    raw_dir = resolve(load_config("sources")["cpcb"]["raw_dir"])
    n = collect(downloads, raw_dir)
    log.info("Copied %d new files from %s", n, downloads)

    frames, infos = [], []
    for path in sorted((raw_dir / "1Hr").rglob("Raw_Data_*_1Hr.csv")):
        daily, info = daily_from_file(path)
        frames.append(daily)
        infos.append(info)
    files = pd.DataFrame(infos)
    files.to_csv(FILES_PATH, index=False)
    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(DAILY_PATH, index=False)
    log.info("CPCB files: %d (%d sites); daily rows: %d",
             len(files), files["cpcb_site"].nunique(), len(daily))
    return DAILY_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=DEFAULT_DOWNLOADS)
    main(parser.parse_args().downloads)
