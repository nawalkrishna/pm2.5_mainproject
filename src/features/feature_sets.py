"""Resolve the Tier A / Tier B feature lists from configs/features.yaml.

Usage:
    from src.features.feature_sets import feature_columns
    cols = feature_columns("tier_a", target="daily")
"""

from src.utils.config import load_config


def feature_columns(tier: str, target: str = "daily", geometry: str | None = None,
                    cfg: dict | None = None, extra_groups: dict | None = None) -> list[str]:
    """Ordered, de-duplicated column list for a tier and target (``daily`` or ``overpass``).

    ``tier`` is a name in ``tiers`` (e.g. ``tier_a``), optionally with the suffix ``_nolag``
    to drop the PM2.5 lags (D037), or a list of group names (ablation). ``extra_groups``
    adds group definitions (e.g. the ablation groups in experiments.yaml).
    """
    cfg = cfg or load_config("features")
    groups = {**cfg["groups"], **(extra_groups or {})}
    g = geometry or cfg["satellite_geometry"]
    if isinstance(tier, str):
        nolag = tier.endswith("_nolag")
        names = [n for n in cfg["tiers"][tier.removesuffix("_nolag")] if not (nolag and n == "lags_pm25")]
    else:
        names = list(tier)
    cols: list[str] = []
    for group in names:
        cols += [c.replace("{g}", g) for c in groups[group]]
    if target == "overpass":
        cols = [cfg["overpass_substitutes"].get(c, c) for c in cols]
    return list(dict.fromkeys(cols))
