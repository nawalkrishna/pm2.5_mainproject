import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.align.spatial import assign_state, group_sites, haversine_m, nearby_pairs


def test_group_sites_needs_same_name_and_proximity():
    sensors = pd.DataFrame({
        "latitude": [10.0, 10.0, 10.0, 11.0],
        "longitude": [76.0, 76.0, 76.0, 76.0],
    })
    names = pd.Series(["a", "a", "b", "a"])   # 0+1 same site; 2 different name; 3 too far
    sites = group_sites(sensors, names, 500)
    assert sites[0] == sites[1]
    assert sites[2] != sites[0]
    assert sites[3] != sites[0]
    assert sites.nunique() == 3


def test_haversine_one_degree_latitude_is_about_111_km():
    d = haversine_m(np.array(10.0), np.array(76.0), np.array(11.0), np.array(76.0))
    assert d == pytest.approx(111_195, rel=1e-3)


def test_nearby_pairs_finds_only_close_stations():
    stations = pd.DataFrame({
        "sensor_id": [1, 2, 3],
        "latitude": [10.0, 10.001, 11.0],     # 1 and 2 are ~111 m apart
        "longitude": [76.0, 76.0, 76.0],
    })
    pairs = nearby_pairs(stations, "sensor_id", 500)
    assert len(pairs) == 1
    assert set(pairs.iloc[0][["sensor_id_a", "sensor_id_b"]]) == {1, 2}


def test_assign_state_inside_outside_and_coastal_fallback():
    states = gpd.GeoDataFrame({"NAME_1": ["A", "B"]},
                              geometry=[box(75, 8, 76, 9), box(76, 8, 77, 9)], crs="EPSG:4326")
    points = pd.DataFrame({
        "latitude": [8.5, 8.5, 8.5, 20.0],
        "longitude": [75.5, 76.5, 74.995, 80.0],   # A, B, ~550 m west of A, far away
    }, index=[10, 11, 12, 13])
    result = assign_state(points, states, "NAME_1")
    assert result.loc[10] == "A"
    assert result.loc[11] == "B"
    assert result.loc[12] == "A"          # coastal fallback within 2 km
    assert pd.isna(result.loc[13])
