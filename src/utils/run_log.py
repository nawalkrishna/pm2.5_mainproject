"""Append one row per pipeline or model run to reports/runs.csv.

This is the project's experiment tracker: a plain CSV is easy to read, diff and
back up. Each row records what ran, on which code version and config, and
any headline numbers.
"""

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.paths import PROJECT_ROOT

RUN_LOG_PATH: Path = PROJECT_ROOT / "reports" / "runs.csv"
FIELDS = ["run_utc", "step", "git_commit", "git_dirty", "config_hash", "metrics_json", "notes"]


def _git_state() -> tuple[str, bool]:
    """Return (short commit hash, has uncommitted changes). 'none' if git is unavailable."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip())
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none", True


def log_run(
    step: str,
    config_hash: str,
    metrics: dict[str, Any] | None = None,
    notes: str = "",
    path: Path = RUN_LOG_PATH,
) -> None:
    """Append a run record.

    Args:
        step: Pipeline step name, e.g. ``"audit_openaq"``.
        config_hash: Hash from :func:`src.utils.config.config_hash`.
        metrics: Headline numbers for this run (station counts, RMSE, ...).
        notes: Free text.
        path: CSV file to append to (overridable for tests).
    """
    commit, dirty = _git_state()
    row = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": step,
        "git_commit": commit,
        "git_dirty": dirty,
        "config_hash": config_hash,
        "metrics_json": json.dumps(metrics or {}, sort_keys=True),
        "notes": notes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
