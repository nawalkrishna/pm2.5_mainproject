"""Explicit UTC <-> IST handling shared by every alignment step (Stage 4, task 5).

All raw timestamps are tz-aware UTC. A "day" is an IST calendar day. The overpass window
selects hours whose *start* (IST) falls in [start_hour, end_hour).
"""

import pandas as pd

IST = "Asia/Kolkata"


def to_utc(ts: pd.Series) -> pd.Series:
    """Parse to tz-aware UTC; refuses naive timestamps, which are ambiguous."""
    ts = pd.to_datetime(ts)
    if ts.dt.tz is None:
        raise ValueError("naive timestamps are ambiguous; localise them first")
    return ts.dt.tz_convert("UTC")


def ist_date(ts_utc: pd.Series) -> pd.Series:
    """IST calendar date (YYYY-MM-DD) of each UTC instant."""
    return to_utc(ts_utc).dt.tz_convert(IST).dt.strftime("%Y-%m-%d")


def in_overpass_window(ts_utc: pd.Series, start_hour: int, end_hour: int) -> pd.Series:
    """True for hours whose IST start time falls in [start_hour, end_hour).

    Hours starting at :30 IST (e.g. US Consulate data, ERA5 hours on the UTC grid) count by
    their IST clock hour: 12:30 IST belongs to the 12-15 window, 11:30 does not.
    """
    hour = to_utc(ts_utc).dt.tz_convert(IST).dt.hour
    return (hour >= start_hour) & (hour < end_hour)
