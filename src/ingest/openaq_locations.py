"""Download all OpenAQ locations measuring PM2.5 inside the discovery box.

Output (raw, never edited): ``data/raw/openaq/locations_pm25_<YYYYMMDD>.json``
holding the PM2.5 parameter record and the full list of location objects.
Re-running on the same day reuses the existing file unless ``--refresh`` is given.

Usage:
    python -m src.ingest.openaq_locations [--refresh]
"""

import argparse
import json
from datetime import date
from pathlib import Path

from src.ingest.openaq_client import OpenAQClient
from src.utils.config import config_hash, load_config
from src.utils.logging_setup import get_logger
from src.utils.manifest import record_file
from src.utils.paths import resolve
from src.utils.run_log import log_run

log = get_logger("openaq_locations")


def make_client(cfg: dict) -> OpenAQClient:
    """Build an OpenAQ client from the ``openaq`` section of sources.yaml."""
    return OpenAQClient(
        base_url=cfg["base_url"],
        api_key_env=cfg["api_key_env"],
        min_seconds_between_requests=cfg["min_seconds_between_requests"],
        page_limit=cfg["page_limit"],
    )


def find_parameter(client: OpenAQClient, name: str) -> dict:
    """Look up a parameter (e.g. ``pm25``) by name instead of hard-coding its ID."""
    matches = [p for p in client.get_all("parameters") if p.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one OpenAQ parameter named {name!r}, found {matches}")
    return matches[0]


def main(refresh: bool = False) -> Path:
    """Fetch locations and write the raw JSON file. Returns its path."""
    src_cfg = load_config("sources")["openaq"]
    area = load_config("study_area")
    out_dir = resolve(src_cfg["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"locations_pm25_{date.today():%Y%m%d}.json"

    if out_path.exists() and not refresh:
        log.info("Already downloaded today: %s (use --refresh to redo)", out_path)
        return out_path

    client = make_client(src_cfg)
    param = find_parameter(client, src_cfg["pm25_parameter_name"])
    log.info("PM2.5 parameter: id=%s units=%s", param["id"], param.get("units"))

    bbox = ",".join(f"{v:.4f}" for v in area["discovery_bbox"])
    locations = list(client.get_all("locations", {"bbox": bbox, "parameters_id": param["id"]}))
    log.info("Found %d locations in bbox %s (%d requests)", len(locations), bbox, client.request_count)

    payload = {"query": {"bbox": bbox, "parameters_id": param["id"]},
               "parameter": param, "results": locations}
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    record_file(out_path, source="OpenAQ v3 /locations", date_range="metadata snapshot",
                rows=len(locations), notes=f"bbox={bbox}")
    log_run("openaq_locations", config_hash({"sources": src_cfg, "area": area}),
            metrics={"locations": len(locations)})
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download even if today's file exists")
    main(parser.parse_args().refresh)
