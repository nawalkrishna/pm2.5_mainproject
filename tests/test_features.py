import numpy as np
import pandas as pd

from src.features.build_dataset import add_lags, temporal_features

SEASONS = {"winter": [1, 2], "pre_monsoon": [3, 4, 5], "sw_monsoon": [6, 7, 8, 9], "ne_monsoon": [10, 11, 12]}


def test_lags_are_calendar_lags_and_do_not_cross_stations():
    df = pd.DataFrame({"station_id": ["a", "a", "a", "b", "b"],
                       "date_ist": ["2021-01-01", "2021-01-02", "2021-01-04", "2021-01-02", "2021-01-03"],
                       "x": [1.0, 2.0, 4.0, 10.0, 11.0]})
    out = add_lags(df, ["x"], [1, 2]).set_index(["station_id", "date_ist"])
    assert out.loc[("a", "2021-01-02"), "x_lag1"] == 1.0
    assert np.isnan(out.loc[("a", "2021-01-04"), "x_lag1"])      # 01-03 missing: no lag from 01-02
    assert out.loc[("a", "2021-01-04"), "x_lag2"] == 2.0          # 01-02 is exactly 2 days back
    assert np.isnan(out.loc[("b", "2021-01-02"), "x_lag1"])      # first day of b: no value from station a


def test_temporal_features():
    df = temporal_features(pd.DataFrame({"date_ist": ["2021-07-01", "2024-12-31"]}), SEASONS)
    assert df["season"].tolist() == ["sw_monsoon", "ne_monsoon"]
    assert df["season_code"].tolist() == [2, 3]
    assert np.isclose(df["doy_sin"].iloc[1], np.sin(2 * np.pi * 366 / 366))   # leap year handled


def test_tier_b_has_no_satellite_columns_and_tier_a_is_superset():
    from src.features.feature_sets import feature_columns

    a, b = feature_columns("tier_a"), feature_columns("tier_b")
    sat_prefixes = ("no2_", "co_", "so2_", "hcho_", "uvai_", "cloud_frac_", "aod_", "sat_")
    assert not [c for c in b if c.startswith(sat_prefixes)]
    assert set(b) < set(a)
    assert "aod_clear_b10" in a and "aod_clear_pt" in feature_columns("tier_a", geometry="pt")
    op = feature_columns("tier_b", target="overpass")
    assert "blh_op" in op and "blh" not in op and "pm25_daily_lag1" in op


def test_distance_to_coast_known_cities():
    import pytest
    from src.ingest.boundaries import GEOJSON_NAME, OUT_DIR

    if not (OUT_DIR / GEOJSON_NAME).exists():
        pytest.skip("GADM boundaries not downloaded")
    from src.features.static import distance_to_coast_km

    c = pd.DataFrame({"latitude": [17.39, 9.98], "longitude": [78.47, 76.28]})      # Hyderabad, Kochi
    km = distance_to_coast_km(c).tolist()
    assert 230 < km[0] < 320 and km[1] < 15


def test_nolag_variant_drops_only_pm25_lags():
    from src.features.feature_sets import feature_columns

    a, a0 = feature_columns("tier_a"), feature_columns("tier_a_nolag")
    assert set(a) - set(a0) == {"pm25_daily_lag1", "pm25_daily_lag2", "pm25_daily_lag3"}
    abl = feature_columns(["meteorology", "aod"], extra_groups={"aod": ["aod_clear_{g}"]})
    assert abl[-1] == "aod_clear_b10" and "t2m_c" in abl
