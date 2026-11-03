"""Daily-completeness calculations for the station audit.

All functions are pure (DataFrame in, DataFrame out), so they are easy to test.
Denominators are always *calendar days in the window*, not days with a record,
so a station that reports nothing for a month is penalised for that month.
"""

import pandas as pd


def estimate_hours(daily: pd.DataFrame) -> pd.Series:
    """Approximate number of distinct clock hours behind each daily record.

    OpenAQ's ``observedCount`` counts hourly *records*. For older CPCB data
    (before 2025) OpenAQ holds two overlapping hourly records per clock hour
    (starting at :00 and :30 IST), so counts reach 48. We detect this per
    sensor-year: if any day in that sensor-year has more than 24 records, the
    sensor-year is treated as two records per hour and counts are halved.
    Verified by inspecting /hours for sample days (decision D006).
    """
    year = daily["date_local"].str[:4]
    doubled = daily.groupby([daily["sensor_id"], year])["observed_count"].transform("max") > 24
    hours = daily["observed_count"].fillna(0) / doubled.map({True: 2, False: 1})
    return hours.clip(upper=24)


def mark_valid_days(daily: pd.DataFrame, min_hours: int) -> pd.DataFrame:
    """Add ``hours_est`` and a boolean ``valid`` column: a day is valid if enough
    hours were observed and the daily mean is not missing.

    Args:
        daily: Needs columns ``sensor_id``, ``date_local``, ``observed_count``, ``pm25_daily_mean``.
        min_hours: Minimum number of hours with data for a valid day.
    """
    out = daily.copy()
    out["hours_est"] = estimate_hours(out)
    out["valid"] = (out["hours_est"] >= min_hours) & out["pm25_daily_mean"].notna()
    return out


def calendar_days(start: str, end: str) -> pd.DataFrame:
    """Every calendar date from ``start`` to ``end`` inclusive, with year and month."""
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": dates, "year": dates.year, "month": dates.month})


def completeness(
    daily: pd.DataFrame,
    key: str,
    start: str,
    end: str,
    seasons: dict[str, list[int]],
) -> dict[str, pd.DataFrame]:
    """Fraction of calendar days that are valid, per station, overall / per year / per season.

    Args:
        daily: Output of :func:`mark_valid_days` with a ``date_local`` column (YYYY-MM-DD)
            and a station key column.
        key: Station identifier column, e.g. ``"sensor_id"``.
        start, end: Inclusive window, ``YYYY-MM-DD``.
        seasons: Season name -> list of calendar months.

    Returns:
        Dict with ``"overall"``, ``"by_year"`` and ``"by_season"`` tables. Each has
        ``valid_days``, ``calendar_days`` and ``completeness`` (0-1) per station.
    """
    month_to_season = {m: name for name, months in seasons.items() for m in months}
    cal = calendar_days(start, end)
    cal["season"] = cal["month"].map(month_to_season)

    d = daily[daily["valid"]].copy()
    d["date"] = pd.to_datetime(d["date_local"])
    d = d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))]
    d = d.drop_duplicates(subset=[key, "date"])
    d["year"] = d["date"].dt.year
    d["season"] = d["date"].dt.month.map(month_to_season)
    stations = pd.Index(daily[key].unique(), name=key)

    def _table(group_col: str | None) -> pd.DataFrame:
        if group_col is None:
            valid = d.groupby(key).size().reindex(stations, fill_value=0).rename("valid_days")
            out = valid.reset_index()
            out["calendar_days"] = len(cal)
        else:
            denom = cal.groupby(group_col).size().rename("calendar_days")
            grid = pd.MultiIndex.from_product([stations, denom.index], names=[key, group_col])
            valid = d.groupby([key, group_col]).size().reindex(grid, fill_value=0).rename("valid_days")
            out = valid.reset_index().merge(denom.reset_index(), on=group_col)
        out["completeness"] = out["valid_days"] / out["calendar_days"]
        return out

    return {"overall": _table(None), "by_year": _table("year"), "by_season": _table("season")}
