"""Stage 6: issue-time feature arithmetic and timing rules (D046)."""

import numpy as np
import pandas as pd

from src.forecast.features import HourlyArrays, lag, last_valid, satellite_features, window
from src.forecast.hourly_data import ist_hour_start
from src.forecast.run_tabular import issue_masks

NAN = np.nan


def test_lag_past_and_future():
    a = np.array([[1, 2, 3, 4]], np.float32)
    np.testing.assert_array_equal(lag(a, 1), [[NAN, 1, 2, 3]])
    np.testing.assert_array_equal(lag(a, -2), [[3, 4, NAN, NAN]])


def test_window_mean_count_and_end_lag():
    a = np.array([[1, NAN, 3, 5]], np.float32)
    m, n = window(a, 2)
    np.testing.assert_array_equal(n, [[NAN, 1, 1, 2]])
    np.testing.assert_allclose(m, [[NAN, 1, 3, 4]])
    m1, _ = window(a, 2, end_lag=1)                 # hours t-2 .. t-1
    np.testing.assert_allclose(m1, [[NAN, NAN, 1, 3]])


def test_last_valid_age_limit():
    a = np.array([[5, NAN, NAN, NAN]], np.float32)
    v, age = last_valid(a, 2)
    np.testing.assert_array_equal(v, [[5, 5, 5, NAN]])
    np.testing.assert_array_equal(age, [[0, 1, 2, NAN]])


def test_era5_stamp_maps_to_ist_hour_start():
    # 07:00 UTC = 12:30 IST -> the IST hour starting 12:00
    t = ist_hour_start(pd.Series(pd.to_datetime(["2025-01-01 07:00"], utc=True)))
    assert t.iloc[0] == pd.Timestamp("2025-01-01 12:00", tz="Asia/Kolkata")


def test_satellite_day_used_only_after_15_ist():
    times = pd.date_range("2025-01-01", periods=48, freq="h", tz="Asia/Kolkata")
    A = HourlyArrays(stations=pd.DataFrame({"station_id": ["s"], "state": ["x"]}), times=times, var={},
                     static=pd.DataFrame(), sat={"aod_clear": np.array([[0.1, 0.2]], np.float32)})
    s = satellite_features(A, {"timing": {"satellite_from_hour": 15}})["sat_aod_clear"][0]
    assert np.isnan(s[14])                           # 14:00 day 1: day 0 does not exist
    assert np.isclose(s[15], 0.1)                    # 15:00 day 1: day 1's value
    assert np.isclose(s[24 + 14], 0.1)               # 14:00 day 2: still day 1
    assert np.isclose(s[24 + 15], 0.2)


def test_issue_masks_purge_and_years():
    times = pd.date_range("2023-12-25", "2025-01-02", freq="h", tz="Asia/Kolkata")
    m = issue_masks(times, {"train_years": [2023], "val_years": [2024], "test_years": [2025], "purge_hours": 72})
    last_train = times[m["train"]].max()
    assert last_train == pd.Timestamp("2023-12-28 23:00", tz="Asia/Kolkata")
    assert not (m["train"] & m["test"]).any() and not (m["val"] & m["test"]).any()


def test_issue_frame_uses_no_future_data_in_setting_a():
    from src.forecast.features import base_features, issue_frame
    T = 24 * 6
    times = pd.date_range("2025-01-01", periods=T, freq="h", tz="Asia/Kolkata")
    ramp = np.arange(T, dtype=np.float32)[None, :]
    cfg = {"meteorology": ["t2m_c", "precip_mm"], "instant_met": ["t2m_c", "wind_speed", "blh"], "average_met": ["precip_mm"],
           "aerosol": ["merra2_pm25"], "static": ["elevation_m"],
           "timing": {"instant_lag": 0, "average_lag": 1, "satellite_from_hour": 15},
           "tabular": {"pm_lags": [0, 1], "pm_windows": [6], "met_windows": [24]}}
    var = {v: ramp.copy() for v in ["pm25", "t2m_c", "wind_speed", "blh", "precip_mm", "merra2_pm25", "population_b5"]}
    A = HourlyArrays(stations=pd.DataFrame({"station_id": ["s"], "state": ["x"]}), times=times, var=var,
                     static=pd.DataFrame({"station_id": ["s"], "elevation_m": [1.0], "population_b5": [1.0]}),
                     sat={"aod_clear": np.zeros((1, 6), np.float32)})
    seasons = {"winter": [1, 2], "pre_monsoon": [3, 4, 5], "sw_monsoon": [6, 7, 8, 9], "ne_monsoon": [10, 11, 12]}
    base = base_features(A, cfg)
    f = issue_frame(A, cfg, seasons, 3, base, {}, np.ones(T, bool)).set_index("issue_idx")
    H = 50
    assert f.loc[H, "y"] == H + 3
    assert f.loc[H, "pm_lag0"] == H and f.loc[H, "t2m_c"] == H        # instantaneous: up to H
    assert f.loc[H, "precip_mm"] == H - 1 and f.loc[H, "merra2_pm25"] == H - 1   # averaged: up to H - 1
    assert f.loc[H, "pm_mean6"] == np.mean(np.arange(H - 5, H + 1))
    assert f.loc[H, "precip_mm_sum24"] == np.arange(H - 24, H).sum()  # 24 h ending at H - 1
    # every value-type feature of setting a comes from hour H or earlier (the ramp's value is its hour)
    past_cols = [c for c in f.columns if c.startswith(("pm_lag", "pm_mean", "pm_last")) or c in cfg["instant_met"]
                 or c.endswith("_mean24") or c in cfg["average_met"] + cfg["aerosol"]]
    assert (f.loc[H, past_cols] <= H).all()
