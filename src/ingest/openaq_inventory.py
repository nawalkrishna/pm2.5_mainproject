"""Flatten raw OpenAQ location JSON into one row per PM2.5 sensor.

Input:  latest ``data/raw/openaq/locations_pm25_*.json``
Output: ``data/interim/openaq_sensors.csv``

Usage:
    python -m src.ingest.openaq_inventory
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_INTERIM, resolve

log = get_logger("openaq_inventory")

OUT_PATH: Path = DATA_INTERIM / "openaq_sensors.csv"


def latest_locations_file(raw_dir: Path) -> Path:
    """Newest locations snapshot (file names sort by date)."""
    files = sorted(raw_dir.glob("locations_pm25_*.json"))
    if not files:
        raise FileNotFoundError(f"No locations file in {raw_dir}; run src.ingest.openaq_locations first")
    return files[-1]


def flatten_locations(locations: list[dict[str, Any]], pm25_id: int) -> pd.DataFrame:
    """One row per PM2.5 sensor, with the location's metadata copied onto it.

    ``grade`` comes from OpenAQ's ``isMonitor`` flag, which the API documents as
    "Is the location considered a reference monitor?". True -> reference,
    False -> low_cost.
    """
    rows = []
    for loc in locations:
        for sensor in loc.get("sensors", []):
            if sensor.get("parameter", {}).get("id") != pm25_id:
                continue
            rows.append({
                "source": "openaq",
                "location_id": loc["id"],
                "sensor_id": sensor["id"],
                "station_uid": f"openaq_{loc['id']}",
                "name": loc.get("name"),
                "locality": loc.get("locality"),
                "latitude": loc["coordinates"]["latitude"],
                "longitude": loc["coordinates"]["longitude"],
                "provider": (loc.get("provider") or {}).get("name"),
                "owner": (loc.get("owner") or {}).get("name"),
                "instruments": "; ".join(i["name"] for i in loc.get("instruments", [])),
                "is_monitor": loc.get("isMonitor"),
                "is_mobile": loc.get("isMobile"),
                "grade": "reference" if loc.get("isMonitor") else "low_cost",
                "units": sensor["parameter"].get("units"),
                "first_utc": ((loc.get("datetimeFirst") or {}).get("utc")),
                "last_utc": ((loc.get("datetimeLast") or {}).get("utc")),
            })
    return pd.DataFrame(rows)


def main() -> Path:
    """Build the sensor table and write it to data/interim."""
    raw_dir = resolve(load_config("sources")["openaq"]["raw_dir"])
    path = latest_locations_file(raw_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sensors = flatten_locations(payload["results"], payload["parameter"]["id"])

    n_mobile = int(sensors["is_mobile"].fillna(False).sum())
    if n_mobile:
        log.info("Dropping %d sensors on mobile platforms (not fixed stations)", n_mobile)
        sensors = sensors[~sensors["is_mobile"].fillna(False)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sensors.to_csv(OUT_PATH, index=False)
    log.info("Wrote %d PM2.5 sensors at %d locations from %s",
             len(sensors), sensors["location_id"].nunique(), path.name)
    return OUT_PATH


if __name__ == "__main__":
    main()
