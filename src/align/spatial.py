"""Spatial helpers: distances, state assignment, nearby-station pairs."""

import geopandas as gpd
import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in metres between points given in decimal degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def assign_state(points: pd.DataFrame, states: gpd.GeoDataFrame, name_col: str) -> pd.Series:
    """State name for each point (``latitude``/``longitude`` columns), NaN if outside all states.

    Uses a point-in-polygon join. Coastal stations sometimes fall a few metres
    into the sea on simplified boundaries; those are matched to the nearest state
    within 2 km instead of being dropped.
    """
    gpts = gpd.GeoDataFrame(
        points[[]].copy(),
        geometry=gpd.points_from_xy(points["longitude"], points["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gpts, states[[name_col, "geometry"]].to_crs("EPSG:4326"),
                       how="left", predicate="within")
    result = joined[name_col].groupby(level=0).first()

    missing = result.isna()
    if missing.any():
        # Nearest-state fallback, measured in a metric CRS (UTM 43N covers South India).
        metric = "EPSG:32643"
        near = gpd.sjoin_nearest(gpts[missing].to_crs(metric),
                                 states[[name_col, "geometry"]].to_crs(metric),
                                 how="left", max_distance=2000)
        result.loc[missing] = near[name_col].groupby(level=0).first()
    return result.reindex(points.index)


def group_sites(sensors: pd.DataFrame, name_key: pd.Series, max_distance_m: float) -> pd.Series:
    """Assign a site ID to each sensor: same normalised name AND within ``max_distance_m``.

    Sensors are linked pairwise and linked groups are merged (union-find), so a
    chain of old and new location IDs at one site ends up as one site.

    Returns:
        Series of site IDs (``site_0001`` ...) aligned with ``sensors.index``.
    """
    parent = list(range(len(sensors)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lat, lon = sensors["latitude"].to_numpy(), sensors["longitude"].to_numpy()
    keys = name_key.to_numpy()
    for i in range(len(sensors)):
        same = np.nonzero(keys[i + 1:] == keys[i])[0] + i + 1
        if same.size == 0:
            continue
        close = same[haversine_m(lat[i], lon[i], lat[same], lon[same]) <= max_distance_m]
        for j in close:
            parent[find(j)] = find(i)

    roots = [find(i) for i in range(len(sensors))]
    order = {r: n for n, r in enumerate(dict.fromkeys(roots), start=1)}
    return pd.Series([f"site_{order[r]:04d}" for r in roots], index=sensors.index)


def nearby_pairs(stations: pd.DataFrame, key: str, max_distance_m: float) -> pd.DataFrame:
    """All unordered pairs of stations closer than ``max_distance_m``.

    Brute-force O(n^2); fine for a few hundred stations.

    Returns:
        Columns ``<key>_a``, ``<key>_b``, ``distance_m``.
    """
    s = stations.reset_index(drop=True)
    lat, lon = s["latitude"].to_numpy(), s["longitude"].to_numpy()
    rows = []
    for i in range(len(s)):
        dist = haversine_m(lat[i], lon[i], lat[i + 1:], lon[i + 1:])
        for offset in np.nonzero(dist <= max_distance_m)[0]:
            j = i + 1 + offset
            rows.append({f"{key}_a": s.at[i, key], f"{key}_b": s.at[j, key],
                         "distance_m": float(dist[offset])})
    return pd.DataFrame(rows, columns=[f"{key}_a", f"{key}_b", "distance_m"])
