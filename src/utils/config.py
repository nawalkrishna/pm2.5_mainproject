"""Load YAML config files and fingerprint them for the run log."""

import hashlib
import json
from typing import Any

import yaml

from src.utils.paths import CONFIG_DIR


def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` as a dict.

    Args:
        name: Config file name without the ``.yaml`` extension, e.g. ``"audit"``.
    """
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_hash(config: dict[str, Any]) -> str:
    """Short, stable hash of a config dict.

    Keys are sorted before hashing, so the hash changes only when values change,
    not when the YAML is reformatted.
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
