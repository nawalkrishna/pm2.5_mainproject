"""Turn the CPCB CCR station list into a table and attach coordinates from OpenAQ.

The CCR Data Repository dropdowns give state, city and station name but no
coordinates. OpenAQ republishes CPCB stations under the same "<site>, <city> - <agency>"
names, so we match on a normalised name. Unmatched stations are listed, not guessed.

Input:  latest data/raw/cpcb/ccr_repository_station_list_*.json, data/interim/openaq_sensors.csv
Output: data/interim/cpcb_stations.csv

Usage:
    python -m src.ingest.cpcb_stations
"""

import json
import re
from pathlib import Path

import pandas as pd

from src.align.spatial import haversine_m
from src.ingest.openaq_inventory import OUT_PATH as OPENAQ_SENSORS_PATH
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.manifest import record_file
from src.utils.paths import DATA_INTERIM, resolve

log = get_logger("cpcb_stations")

OUT_PATH: Path = DATA_INTERIM / "cpcb_stations.csv"


def normalise_name(name: str) -> str:
    """Lower-case, drop punctuation and repeated spaces, so trivial spelling differences match."""
    name = re.sub(r"[^a-z0-9]+", " ", str(name).lower())
    return re.sub(r"\s+", " ", name).strip()


def match_to_openaq(cpcb: pd.DataFrame, openaq_sensors: pd.DataFrame,
                    max_spread_m: float = 500.0) -> pd.DataFrame:
    """Add OpenAQ location IDs, coordinates and a match status to each CPCB station.

    Only exact matches on the normalised name are accepted. OpenAQ often lists one
    site under several location IDs (e.g. an old ID ending 2022 and a new one from
    2025). If all same-named locations lie within ``max_spread_m`` of each other,
    the coordinates are accepted (``openaq_name_match``). If they disagree by more,
    the station is marked ``coord_conflict`` and left without coordinates for
    manual checking.
    """
    locs = (openaq_sensors.drop_duplicates("location_id")
            [["location_id", "name", "latitude", "longitude"]].copy())
    locs["key"] = locs["name"].map(normalise_name)

    groups = {}
    for key, g in locs.groupby("key"):
        spread = float(haversine_m(g["latitude"].iloc[0], g["longitude"].iloc[0],
                                   g["latitude"].to_numpy(), g["longitude"].to_numpy()).max())
        groups[key] = {
            "openaq_location_ids": ";".join(str(i) for i in sorted(g["location_id"])),
            "openaq_coord_spread_m": round(spread),
            "latitude": g["latitude"].iloc[0] if spread <= max_spread_m else float("nan"),
            "longitude": g["longitude"].iloc[0] if spread <= max_spread_m else float("nan"),
            "coord_source": "openaq_name_match" if spread <= max_spread_m else "coord_conflict",
        }

    out = cpcb.copy()
    keys = out["station"].map(normalise_name)
    matched = pd.DataFrame([groups.get(k, {"coord_source": "unmatched"}) for k in keys], index=out.index)
    return pd.concat([out, matched], axis=1)


def main() -> Path:
    raw_dir = resolve(load_config("sources")["cpcb"]["raw_dir"])
    path = sorted(raw_dir.glob("ccr_repository_station_list_*.json"))[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    record_file(path, source=payload["source"], date_range="station list snapshot",
                rows=len(payload["stations"]))

    cpcb = pd.DataFrame(payload["stations"])
    cpcb["agency"] = cpcb["station"].str.rsplit(" - ", n=1).str[-1]
    matched = match_to_openaq(cpcb, pd.read_csv(OPENAQ_SENSORS_PATH))
    matched.insert(0, "station_uid", "cpcb_" + matched.index.astype(str).str.zfill(3))

    matched.to_csv(OUT_PATH, index=False)
    log.info("CPCB stations: %d; match status: %s",
             len(matched), matched["coord_source"].value_counts().to_dict())
    return OUT_PATH


if __name__ == "__main__":
    main()
