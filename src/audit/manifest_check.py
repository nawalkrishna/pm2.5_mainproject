"""Check that the data manifest matches data/raw: every file listed once, every hash current.

Reports (and with ``--repair`` fixes):
    * malformed rows (file field not a data/ path)            -> dropped
    * rows whose file no longer exists                          -> dropped
    * files under data/raw with no row                          -> recorded (sources inferred from the path)
    * rows whose SHA-256 no longer matches the file             -> reported only (never silently re-hashed)

Usage:
    python -m src.audit.manifest_check [--repair] [--restore-from GIT_REF]

``--restore-from`` first takes rows for missing files from the manifest committed at
GIT_REF (keeping their original source, date range, retrieval time and notes) when the
file's hash still matches.
"""

import argparse
import csv
import io
import subprocess

import pandas as pd

from src.utils.logging_setup import get_logger
from src.utils.manifest import FIELDS, MANIFEST_PATH, record_file, sha256_of
from src.utils.paths import DATA_RAW, PROJECT_ROOT

log = get_logger("manifest_check")
csv.field_size_limit(1 << 30)          # tolerate corrupted rows long enough to drop them


def read_manifest(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def infer_source(rel: str) -> tuple[str, str, str]:
    """(source, date_range, notes) for a raw file with no manifest row, from its path."""
    parts = rel.split("/")
    name = parts[-1].rsplit(".", 1)[0]
    if parts[2] == "gee":
        return f"Google Earth Engine: {parts[3]}", name.split("_", 1)[1] if "_" in name else "", "re-recorded by manifest_check"
    if parts[2] == "openaq":
        return "OpenAQ v3", name, "re-recorded by manifest_check"
    return parts[2], "", "re-recorded by manifest_check"


def main(repair: bool = False, restore_from: str | None = None) -> dict:
    rows = read_manifest(MANIFEST_PATH.read_text(encoding="utf-8"))
    malformed = [r for r in rows if not str(r.get("file", "")).startswith("data/")]
    rows = [r for r in rows if r not in malformed]
    missing_file = [r for r in rows if not (PROJECT_ROOT / r["file"]).exists()]
    rows = [r for r in rows if r not in missing_file]
    dup = pd.Series([r["file"] for r in rows]).duplicated(keep="last").to_numpy()
    rows = [r for r, d in zip(rows, dup) if not d]
    listed = {r["file"] for r in rows}
    on_disk = sorted(p.relative_to(PROJECT_ROOT).as_posix() for p in DATA_RAW.rglob("*")
                     if p.is_file() and p.name != ".gitkeep" and p.suffix != ".tmp")
    unlisted = [f for f in on_disk if f not in listed]

    restored = []
    if restore_from and unlisted:
        old_text = subprocess.run(["git", "show", f"{restore_from}:reports/tables/data_manifest.csv"],
                                  cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8").stdout
        old = {r["file"]: r for r in read_manifest(old_text)}
        for f in list(unlisted):
            if f in old and old[f]["sha256"] == sha256_of(PROJECT_ROOT / f):
                restored.append(old[f])
                unlisted.remove(f)

    stale = [r["file"] for r in rows if r["sha256"] != sha256_of(PROJECT_ROOT / r["file"])]
    report = {"rows_ok": len(rows), "malformed": len(malformed), "missing_file": len(missing_file),
              "duplicates": int(dup.sum()), "restored_from_git": len(restored), "unlisted": len(unlisted),
              "hash_mismatch": len(stale)}
    log.info("Manifest check: %s", report)
    if stale:
        log.warning("Hash mismatch (file changed since recorded): %s", stale[:20])

    if repair:
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows([{k: r.get(k, "") for k in FIELDS} for r in rows + restored])
        tmp.replace(MANIFEST_PATH)
        for f in unlisted:
            path = PROJECT_ROOT / f
            source, date_range, notes = infer_source(f)
            n_rows = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1 if path.suffix == ".csv" else None
            record_file(path, source=source, date_range=date_range, rows=n_rows, notes=notes)
        log.info("Repaired: %d rows kept, %d restored, %d re-recorded", len(rows), len(restored), len(unlisted))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--restore-from")
    a = parser.parse_args()
    main(a.repair, a.restore_from)
