"""Version-stamp raw files: hash, size, row count, retrieval date.

Every file written to data/raw/ gets a row in reports/tables/data_manifest.csv,
so we can later prove which exact inputs produced a result.
"""

import csv
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.utils.paths import PROJECT_ROOT, REPORTS_TABLES

MANIFEST_PATH: Path = REPORTS_TABLES / "data_manifest.csv"
FIELDS = ["file", "source", "date_range", "rows", "bytes", "sha256", "retrieved_utc", "notes"]
# Extraction runs in worker threads; without this lock concurrent read-modify-write
# cycles interleaved and corrupted the manifest (2026-11-03).
_LOCK = threading.Lock()


def sha256_of(path: Path) -> str:
    """SHA-256 hex digest of a file, read in chunks so large files are fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_file(
    path: Path,
    source: str,
    date_range: str,
    rows: int | None,
    notes: str = "",
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    """Add or replace the manifest entry for ``path``.

    If the file is already listed, its old row is replaced, so re-running a
    download leaves exactly one current entry per file.
    """
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()   # outside the project (e.g. test temp dirs)
    entry = {
        "file": rel,
        "source": source,
        "date_range": date_range,
        "rows": "" if rows is None else rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }

    with _LOCK:
        existing: list[dict[str, str]] = []
        if manifest_path.exists():
            with open(manifest_path, newline="", encoding="utf-8") as fh:
                existing = [r for r in csv.DictReader(fh) if r["file"] != rel]

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(existing)
            writer.writerow(entry)
        tmp.replace(manifest_path)
