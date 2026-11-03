import pandas as pd
import pytest

from src.align.targets import build_targets
from src.align.time_windows import in_overpass_window, ist_date

CFG = {"overpass_window": {"start_hour": 12, "end_hour": 15, "min_valid_hours": 2}, "daily": {"min_valid_hours": 18}}


def test_ist_date_rolls_over_at_1830_utc():
    ts = pd.Series(pd.to_datetime(["2021-01-01 18:29", "2021-01-01 18:30", "2021-01-01 19:00"], utc=True))
    assert ist_date(ts).tolist() == ["2021-01-01", "2021-01-02", "2021-01-02"]


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError):
        ist_date(pd.Series(pd.to_datetime(["2021-01-01 00:00"])))


def test_overpass_window_uses_ist_hour_start():
    # 06:30 UTC = 12:00 IST (in), 06:00 UTC = 11:30 IST (out), 09:00 UTC = 14:30 IST (in), 09:30 UTC = 15:00 IST (out)
    ts = pd.Series(pd.to_datetime(["2021-01-01 06:30", "2021-01-01 06:00", "2021-01-01 09:00", "2021-01-01 09:30"], utc=True))
    assert in_overpass_window(ts, 12, 15).tolist() == [True, False, True, False]


def test_cpcb_day_maps_to_ist_day_not_utc_day():
    # A full CPCB IST day (labels 00:00-23:00 IST) spans two UTC dates; it must stay one target day.
    ist = pd.date_range("2021-03-01 00:00", periods=24, freq="h", tz="Asia/Kolkata")
    h = pd.DataFrame({"station_id": "s", "timestamp_utc": ist.tz_convert("UTC"), "pm25": range(24), "qc_flag": "ok"})
    t = build_targets(h, CFG)
    assert t["date_ist"].tolist() == ["2021-03-01"] and t["n_hours_daily"].iloc[0] == 24
    assert t["pm25_overpass"].iloc[0] == 13.0            # mean of hours 12, 13, 14
    assert t["n_hours_overpass"].iloc[0] == 3


def test_targets_respect_min_hours_and_qc():
    ist = pd.date_range("2021-03-01 00:00", periods=24, freq="h", tz="Asia/Kolkata")
    flags = ["ok"] * 17 + ["flat_run"] * 7               # only 17 good hours; 12-14 ok, so overpass valid
    h = pd.DataFrame({"station_id": "s", "timestamp_utc": ist.tz_convert("UTC"), "pm25": 10.0, "qc_flag": flags})
    t = build_targets(h, CFG)
    assert pd.isna(t["pm25_daily"].iloc[0]) and t["pm25_overpass"].iloc[0] == 10.0
