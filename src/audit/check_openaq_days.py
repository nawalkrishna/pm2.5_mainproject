"""Spot-check: does ``coverage.observedCount`` on /days equal the number of hourly values?

The audit's valid-day rule (decision D006) assumes it does. For a few sensors and
one month each, we fetch /hours, count hourly values per day, and compare with
the /days observedCount for the same day.

Daily periods in OpenAQ run 01:00 to 00:00 local time, so each hour is assigned
to the day of its *end* time (datetimeTo, local). That matches the /days definition.

Output: reports/tables/openaq_days_vs_hours_check.csv (one row per sensor-day)

Usage:
    python -m src.audit.check_openaq_days
"""

import pandas as pd

from src.ingest.openaq_daily import DAILY_PATH
from src.ingest.openaq_locations import make_client
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import REPORTS_TABLES

log = get_logger("check_openaq_days")

N_SENSORS = 4


def pick_samples(daily: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick ``n`` sensors with many records and, for each, its best-covered month."""
    d = daily.copy()
    d["month"] = d["date_local"].str[:7]
    per_month = d.groupby(["sensor_id", "month"]).size().rename("days").reset_index()
    best = per_month.sort_values("days", ascending=False).drop_duplicates("sensor_id")
    return best.head(n)


def main() -> None:
    daily = pd.read_csv(DAILY_PATH)
    client = make_client(load_config("sources")["openaq"])
    rows = []
    for s in pick_samples(daily, N_SENSORS).itertuples():
        start = pd.Timestamp(f"{s.month}-01")
        end = start + pd.offsets.MonthBegin(1)
        hours = list(client.get_all(f"sensors/{s.sensor_id}/hours",
                                    {"datetime_from": start.strftime("%Y-%m-%d"),
                                     "datetime_to": (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")}))
        h = pd.DataFrame({"end_local": [x["period"]["datetimeTo"]["local"] for x in hours],
                          "value": [x.get("value") for x in hours]})
        # Hour ending at 00:00 belongs to the previous day (01:00-00:00 convention).
        end_ts = pd.to_datetime(h["end_local"].str[:19])
        h["date_local"] = (end_ts - pd.Timedelta(minutes=1)).dt.strftime("%Y-%m-%d")
        counts = h[h["value"].notna()].groupby("date_local").size().rename("hours_counted")

        d = daily[(daily["sensor_id"] == s.sensor_id) & (daily["date_local"].str[:7] == s.month)]
        cmp = d.set_index("date_local")[["observed_count"]].join(counts, how="outer").fillna(0)
        cmp = cmp[cmp.index.str[:7] == s.month]
        cmp["sensor_id"] = s.sensor_id
        rows.append(cmp.reset_index())

    out = pd.concat(rows, ignore_index=True)
    out["match"] = out["observed_count"] == out["hours_counted"]
    out.to_csv(REPORTS_TABLES / "openaq_days_vs_hours_check.csv", index=False)
    log.info("Sensor-days compared: %d; exact matches: %d (%.1f%%); max abs difference: %d",
             len(out), out["match"].sum(), 100 * out["match"].mean(),
             (out["observed_count"] - out["hours_counted"]).abs().max())


if __name__ == "__main__":
    main()
