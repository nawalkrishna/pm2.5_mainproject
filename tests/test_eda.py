import numpy as np
import pandas as pd

from src.eda.build_daily import merra2_pm25, relative_humidity
from src.eda.data_quality import outages, run_lengths


def test_relative_humidity_is_100_at_saturation_and_drops_with_dewpoint_depression():
    t = pd.Series([300.0, 300.0])
    rh = relative_humidity(t, pd.Series([300.0, 290.0]))
    assert abs(rh.iloc[0] - 100) < 1e-9
    assert 50 < rh.iloc[1] < 60          # 10 K depression at 27 °C is ~55 %


def test_merra2_pm25_formula_and_units():
    df = pd.DataFrame({"DUSMASS25": [1e-9], "OCSMASS": [2e-9], "BCSMASS": [1e-9], "SSSMASS25": [1e-9],
                       "SO4SMASS": [96.06e-11]})
    # 1 + 2 + 1 + 1 µg/m³ + SO4 0.9606 µg/m³ x 132.14/96.06 = 1.3214
    assert np.isclose(merra2_pm25(df).iloc[0], 6.3214)


def test_run_lengths_break_on_value_change_and_time_gap():
    times = pd.Series(pd.to_datetime(["2021-01-01 00:00", "2021-01-01 01:00", "2021-01-01 02:00",
                                      "2021-01-01 05:00", "2021-01-01 06:00"]))
    values = pd.Series([5.0, 5.0, 5.0, 5.0, 7.0])
    assert run_lengths(values, times).tolist() == [3, 3, 3, 1, 1]


def test_outages_only_inside_active_span():
    dates = pd.date_range("2021-01-01", periods=30).strftime("%Y-%m-%d")
    valid = [False] * 5 + [True] * 5 + [False] * 8 + [True] * 12     # leading gap ignored, 8-day gap counted
    d = pd.DataFrame({"station_id": "s1", "date_ist": dates, "pm25_valid": valid, "month": 1})
    g = outages(d)
    assert len(g) == 1 and g.iloc[0]["days"] == 8
