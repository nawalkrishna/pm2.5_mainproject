import pandas as pd
import pytest

from src.clean.completeness import completeness, mark_valid_days

SEASONS = {"winter": [1, 2], "pre": [3, 4, 5], "sw": [6, 7, 8, 9], "ne": [10, 11, 12]}


def _daily(rows):
    return pd.DataFrame(rows, columns=["sensor_id", "date_local", "pm25_daily_mean", "observed_count"])


def test_valid_day_needs_enough_hours_and_a_value():
    d = mark_valid_days(_daily([
        (1, "2021-01-01", 30.0, 18),   # exactly at threshold -> valid
        (1, "2021-01-02", 30.0, 17),   # too few hours
        (1, "2021-01-03", None, 24),   # missing mean
        (1, "2021-01-04", 30.0, None), # unknown hours -> treated as 0
    ]), min_hours=18)
    assert d["valid"].tolist() == [True, False, False, False]


def test_half_hour_offset_records_are_not_double_counted():
    # Sensor 1 in 2020 has 48 records on a full day -> two records per hour.
    # A 30-record day of that sensor is only 15 hours -> invalid.
    # Sensor 2 never exceeds 24 -> counts are hours as-is.
    d = mark_valid_days(_daily([
        (1, "2020-01-01", 30.0, 48),
        (1, "2020-01-02", 30.0, 30),
        (2, "2020-01-01", 30.0, 20),
    ]), min_hours=18)
    assert d["hours_est"].tolist() == [24, 15, 20]
    assert d["valid"].tolist() == [True, False, True]


def test_denominator_is_calendar_days_not_reported_days():
    # One valid day in a 10-day window -> 10 %, even though only one row exists.
    d = mark_valid_days(_daily([(1, "2021-01-05", 20.0, 24)]), 18)
    out = completeness(d, "sensor_id", "2021-01-01", "2021-01-10", SEASONS)["overall"]
    assert out.loc[0, "calendar_days"] == 10
    assert out.loc[0, "completeness"] == pytest.approx(0.1)


def test_days_outside_window_and_duplicates_are_ignored():
    d = mark_valid_days(_daily([
        (1, "2020-12-31", 20.0, 24),   # before window
        (1, "2021-01-01", 20.0, 24),
        (1, "2021-01-01", 21.0, 24),   # duplicate date
    ]), 18)
    out = completeness(d, "sensor_id", "2021-01-01", "2021-01-02", SEASONS)["overall"]
    assert out.loc[0, "valid_days"] == 1


def test_sensor_with_no_valid_days_still_appears_with_zero():
    d = mark_valid_days(_daily([(1, "2021-01-01", 20.0, 24), (2, "2021-01-01", 20.0, 3)]), 18)
    out = completeness(d, "sensor_id", "2021-01-01", "2021-01-01", SEASONS)["overall"].set_index("sensor_id")
    assert out.loc[2, "completeness"] == 0


def test_by_season_uses_season_calendar_days():
    # 2021 has 59 winter days (Jan 31 + Feb 28) and 122 SW-monsoon days (Jun-Sep).
    d = mark_valid_days(_daily([(1, "2021-07-01", 20.0, 24)]), 18)
    out = completeness(d, "sensor_id", "2021-01-01", "2021-12-31", SEASONS)["by_season"].set_index("season")
    assert out.loc["winter", "calendar_days"] == 59
    assert out.loc["sw", "calendar_days"] == 122
    assert out.loc["sw", "valid_days"] == 1
    assert out.loc["winter", "valid_days"] == 0
