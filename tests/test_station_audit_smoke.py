"""End-to-end run of the audit script on tiny synthetic inputs (no network)."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import src.audit.station_audit as sa


def test_audit_runs_end_to_end(tmp_path, monkeypatch):
    # Two fake "states" in GADM spelling, one station in each, plus one far outside.
    bdir = tmp_path / "boundaries"
    bdir.mkdir()
    gpd.GeoDataFrame({"NAME_1": ["Kerala", "TamilNadu"]},
                     geometry=[box(75, 8, 77, 12), box(77, 8, 80, 13)], crs="EPSG:4326"
                     ).to_file(bdir / "b.json", driver="GeoJSON")

    sensors = pd.DataFrame({
        "source": "openaq", "location_id": [1, 2, 3], "sensor_id": [11, 22, 33],
        "station_uid": ["openaq_1", "openaq_2", "openaq_3"], "name": ["a", "b", "c"],
        "locality": "x", "latitude": [10.0, 10.0, 25.0], "longitude": [76.0, 78.0, 76.0],
        "provider": "p", "owner": "o", "instruments": "i", "is_monitor": [True, False, True],
        "is_mobile": False, "grade": ["reference", "low_cost", "reference"], "units": "µg/m³",
        "first_utc": "2021-01-01", "last_utc": "2024-12-31",
    })
    dates = pd.date_range("2021-01-01", "2024-12-31").strftime("%Y-%m-%d")
    daily = pd.concat([
        pd.DataFrame({"sensor_id": 11, "date_local": dates, "pm25_daily_mean": 20.0, "observed_count": 24}),
        pd.DataFrame({"sensor_id": 22, "date_local": dates[::3], "pm25_daily_mean": 20.0, "observed_count": 24}),
    ])
    sensors.to_csv(tmp_path / "sensors.csv", index=False)
    daily.to_csv(tmp_path / "daily.csv", index=False)
    pd.DataFrame({"state": ["Kerala"], "city": ["x"], "station": ["a"]}).to_csv(tmp_path / "cpcb.csv", index=False)
    monkeypatch.setattr(sa, "CPCB_PATH", tmp_path / "cpcb.csv")

    monkeypatch.setattr(sa, "SENSORS_PATH", tmp_path / "sensors.csv")
    monkeypatch.setattr(sa, "DAILY_PATH", tmp_path / "daily.csv")
    monkeypatch.setattr(sa, "BOUNDARY_DIR", bdir)
    monkeypatch.setattr(sa, "GEOJSON_NAME", "b.json")
    monkeypatch.setattr(sa, "REPORTS_TABLES", tmp_path / "tables")
    monkeypatch.setattr(sa, "REPORTS_FIGURES", tmp_path / "figures")
    monkeypatch.setattr(sa, "log_run", lambda *a, **k: None)

    sa.main()

    inv = pd.read_csv(tmp_path / "tables" / "station_inventory.csv").set_index("sensor_id")
    assert set(inv.index) == {11, 22}                       # 33 is outside both states
    assert inv.loc[22, "state"] == "Tamil Nadu"             # GADM name translated
    sites = pd.read_csv(tmp_path / "tables" / "site_inventory.csv").set_index("name")
    assert sites.loc["a", "usable_study_period"]            # 100 % complete
    assert not sites.loc["b", "usable_study_period"]        # ~33 % complete
    summary = pd.read_csv(tmp_path / "tables" / "audit_summary.csv").set_index("state")
    assert summary.loc["ALL", "openaq_usable_sites_study_period"] == 1
    assert summary.loc["Kerala", "cpcb_ccr_stations"] == 1
    by_year = pd.read_csv(tmp_path / "tables" / "usable_sites_by_year.csv", index_col=0)
    assert by_year.loc["Kerala", "2021"] == 1
    assert (tmp_path / "figures" / "station_map.png").stat().st_size > 0
