import pandas as pd

from src.ingest.gee_extract import hourly_blocks, ist_days


def test_ist_days_cover_whole_month():
    days = ist_days("2024-02")
    assert days[0] == "2024-02-01" and days[-1] == "2024-02-29" and len(days) == 29


def test_hourly_blocks_cover_ist_window_without_gaps():
    blocks = hourly_blocks("2020-12-01", "2025-12-31", 8)
    starts = pd.to_datetime([a for a, _ in blocks])
    ends = pd.to_datetime([b for _, b in blocks])
    # Contiguous, and the UTC span covers the IST window (which starts 5:30 earlier in UTC).
    assert (starts[1:] == ends[:-1]).all()
    assert starts[0] <= pd.Timestamp("2020-12-01") - pd.Timedelta(hours=5, minutes=30)
    assert ends[-1] >= pd.Timestamp("2026-01-01")


def test_grid_transform_parses_wkt():
    from src.ingest.gee_extract import grid_transform

    wkt = ('PARAM_MT["Affine", PARAMETER["num_row", 3], PARAMETER["num_col", 3], PARAMETER["elt_0_0", 0.1], '
           'PARAMETER["elt_0_2", -360.05], PARAMETER["elt_1_1", -0.1], PARAMETER["elt_1_2", 90.05]]')
    assert grid_transform(wkt) == [0.1, 0.0, -360.05, 0.0, -0.1, 90.05]
