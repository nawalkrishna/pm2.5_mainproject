import pandas as pd

from src.ingest.cpcb_hourly import daily_from_file, parse_name


def test_parse_name():
    meta = parse_name("Raw_Data_2023_site_252_plammoodu_thiruvananthapuram_kerala_pcb_1Hr.csv")
    assert meta == {"year": 2023, "cpcb_site": "site_252", "slug": "plammoodu_thiruvananthapuram_kerala_pcb"}
    assert parse_name("something_else.csv") is None


def test_daily_counts_hours_with_pm25(tmp_path):
    ts = pd.date_range("2023-01-01 00:00", "2023-01-02 23:00", freq="h")
    pm = [10.0] * 24 + [20.0] * 10 + [None] * 14         # day 1 full, day 2 has 10 hours
    path = tmp_path / "Raw_Data_2023_site_1_x_1Hr.csv"
    pd.DataFrame({"Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "PM2.5 (µg/m³)": pm,
                  "PM10 (µg/m³)": 1.0}).to_csv(path, index=False, encoding="utf-8")
    daily, info = daily_from_file(path)
    assert daily["observed_count"].tolist() == [24, 10]
    assert daily["pm25_daily_mean"].tolist() == [10.0, 20.0]
    assert info["pm25_hours"] == 34
    assert (daily["cpcb_site"] == "site_1").all()
