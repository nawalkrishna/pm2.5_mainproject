from src.ingest.openaq_daily import parse_days, years_to_fetch
from src.ingest.openaq_inventory import flatten_locations


def _location(loc_id, is_monitor, sensors):
    return {
        "id": loc_id, "name": f"loc{loc_id}", "locality": "Kochi",
        "coordinates": {"latitude": 10.0, "longitude": 76.3},
        "provider": {"name": "CPCB"}, "owner": {"name": "Kerala PCB"},
        "instruments": [{"id": 1, "name": "Government Monitor"}],
        "isMonitor": is_monitor, "isMobile": False,
        "datetimeFirst": {"utc": "2020-01-01T00:00:00Z"}, "datetimeLast": {"utc": "2022-06-01T00:00:00Z"},
        "sensors": sensors,
    }


def test_flatten_keeps_only_pm25_sensors_and_sets_grade():
    pm25 = {"id": 2, "units": "µg/m³"}
    locs = [
        _location(1, True, [{"id": 11, "parameter": pm25}, {"id": 12, "parameter": {"id": 1}}]),
        _location(2, False, [{"id": 21, "parameter": pm25}]),
    ]
    df = flatten_locations(locs, pm25_id=2)
    assert df["sensor_id"].tolist() == [11, 21]
    assert df["grade"].tolist() == ["reference", "low_cost"]
    assert df["station_uid"].tolist() == ["openaq_1", "openaq_2"]


def test_years_to_fetch_clips_to_reported_span():
    assert years_to_fetch("2020-03-01T00:00:00Z", "2022-06-01T00:00:00Z", "2018-01-01", "2025-12-31") == [2020, 2021, 2022]
    assert years_to_fetch(None, None, "2021-01-01", "2022-12-31") == [2021, 2022]


def test_parse_days_uses_local_date():
    # 2021-06-30T19:30Z is already 2021-07-01 01:00 in IST; the local date must win.
    results = [{
        "value": 25.0,
        "period": {"datetimeFrom": {"utc": "2021-06-30T19:30:00Z", "local": "2021-07-01T01:00:00+05:30"}},
        "coverage": {"observedCount": 20, "expectedCount": 24, "percentComplete": 83.3},
    }]
    df = parse_days(99, results)
    assert df.loc[0, "date_local"] == "2021-07-01"
    assert df.loc[0, "observed_count"] == 20


def test_one_per_ist_hour_keeps_hour_start_bin_and_relabels_half_hour_only():
    import pandas as pd
    from src.ingest.ground_pm25 import one_per_ist_hour

    # IST 02:00 and 02:30 bins (same hour, duplicate), 03:00 only, 04:30 only
    ist = pd.to_datetime(["2023-03-20 02:30", "2023-03-20 02:00", "2023-03-20 03:00",
                          "2023-03-20 04:30"]).tz_localize("Asia/Kolkata")
    df = pd.DataFrame({"timestamp_utc": ist.tz_convert("UTC"), "pm25": [99.0, 124.0, 76.0, 58.0]})
    out = one_per_ist_hour(df)
    got = out.assign(ist=out["timestamp_utc"].dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M"))
    assert got[["ist", "pm25"]].values.tolist() == [["02:00", 124.0], ["03:00", 76.0], ["04:00", 58.0]]
    assert str(out["timestamp_utc"].dt.tz) == "UTC"
