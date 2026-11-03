"""Static station features (Stage 4, task 6).

From the Earth Engine static extraction (5 km buffer, ``static_b5.csv``):
    elevation_m            mean SRTM elevation
    lc_tree, lc_crop, lc_built, lc_water, lc_grass_shrub   ESA WorldCover 2021 class shares
                           (10 tree; 40 crop; 50 built-up; 80 water; 20 shrub + 30 grass)
    population_b5_<year>   WorldPop R2025A people in the buffer, 2021-2025 (matched to the
                           row's year when the dataset is built)
Computed locally:
    dist_to_coast_km       distance to the outer edge of the Indian mainland (GADM level 1). For
                           South Indian stations the nearest outline segment is the coast:
                           the nearest land border is > 1,000 km away.

Usage: imported by src.features.build_dataset
"""

import geopandas as gpd
import pandas as pd

from src.ingest.boundaries import GEOJSON_NAME, OUT_DIR as BOUNDARY_DIR
from src.ingest.gee_extract import OUT_DIR as GEE_DIR

UTM = "EPSG:32643"
LANDCOVER = {"lc_tree": [10], "lc_grass_shrub": [20, 30], "lc_crop": [40], "lc_built": [50], "lc_water": [80]}


def india_outline(states: gpd.GeoDataFrame):
    """Outer edge of the Indian mainland, in UTM metres.

    The union of GADM state polygons leaves thin slivers between neighbouring states, whose
    edges would count as "coast" (first version gave a maximum of 154 km, while Hyderabad
    is ~300 km inland). Closing with a 1 km buffer and keeping only the exterior ring of the
    largest polygon removes them.
    """
    merged = states.to_crs(UTM).union_all().buffer(1000).buffer(-1000)
    polys = list(getattr(merged, "geoms", [merged]))
    return max(polys, key=lambda g: g.area).exterior


def distance_to_coast_km(stations: pd.DataFrame) -> pd.Series:
    outline = india_outline(gpd.read_file(BOUNDARY_DIR / GEOJSON_NAME))
    pts = gpd.GeoSeries(gpd.points_from_xy(stations["longitude"], stations["latitude"]), crs="EPSG:4326").to_crs(UTM)
    return pd.Series(pts.distance(outline).to_numpy() / 1000, index=stations.index).round(2)


def static_features(stations: pd.DataFrame) -> pd.DataFrame:
    s = pd.read_csv(GEE_DIR / "static" / "static_b5.csv")
    out = s[["station_id"]].copy()
    out["elevation_m"] = s["elevation_mean"]
    for name, classes in LANDCOVER.items():
        cols = [f"worldcover_frac_{c}" for c in classes if f"worldcover_frac_{c}" in s.columns]
        out[name] = s[cols].sum(axis=1) if cols else 0.0
    for c in [c for c in s.columns if c.startswith("population_")]:
        out[f"population_b5_{c.split('_')[1]}"] = s[c]
    st = stations.set_index("station_id").loc[out["station_id"]].reset_index()
    out["dist_to_coast_km"] = distance_to_coast_km(st).to_numpy()
    return out
