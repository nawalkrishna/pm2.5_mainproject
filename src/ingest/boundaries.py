"""Download India state (level-1) boundaries from GADM 4.1.

Why GADM: the geoBoundaries open India layer represents 2011 boundaries, before
Telangana was created (2014), so it cannot assign stations to Telangana.
GADM licence: free for academic use, no redistribution (https://gadm.org/license.html).
Because data/raw is git-ignored, the file is never redistributed from this repo.

Usage:
    python -m src.ingest.boundaries
"""

import zipfile
from pathlib import Path

import requests

from src.utils.logging_setup import get_logger
from src.utils.manifest import record_file
from src.utils.paths import DATA_RAW

log = get_logger("boundaries")

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json.zip"
OUT_DIR: Path = DATA_RAW / "boundaries"
GEOJSON_NAME = "gadm41_IND_1.json"


def main() -> Path:
    """Download and unzip the level-1 GeoJSON once. Returns the GeoJSON path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / Path(GADM_URL).name
    geojson_path = OUT_DIR / GEOJSON_NAME

    if geojson_path.exists():
        log.info("Already present: %s", geojson_path)
        return geojson_path

    log.info("Downloading %s", GADM_URL)
    resp = requests.get(GADM_URL, timeout=120)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path) as zf:
        if GEOJSON_NAME not in zf.namelist():
            raise RuntimeError(f"{GEOJSON_NAME} not in archive: {zf.namelist()}")
        zf.extract(GEOJSON_NAME, OUT_DIR)

    record_file(zip_path, source=GADM_URL, date_range="GADM 4.1", rows=None,
                notes="India level-1 boundaries, academic use only")
    return geojson_path


if __name__ == "__main__":
    main()
