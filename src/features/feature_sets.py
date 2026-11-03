"""Resolve the Tier A / Tier B feature lists from configs/features.yaml.

Usage:
    from src.features.feature_sets import feature_columns
    cols = feature_columns("tier_a", target="daily")
"""

from src.utils.config import load_config


def feature_columns(tier: str, target: str = "daily", geometry: str | None = None,
                    cfg: dict | None = None) -> list[str]:
    """Ordered, de-duplicated column list for a tier and target (``daily`` or ``overpass``)."""
    cfg = cfg or load_config("features")
    g = geometry or cfg["satellite_geometry"]
    cols: list[str] = []
    for group in cfg["tiers"][tier]:
        cols += [c.replace("{g}", g) for c in cfg["groups"][group]]
    if target == "overpass":
        cols = [cfg["overpass_substitutes"].get(c, c) for c in cols]
    return list(dict.fromkeys(cols))
