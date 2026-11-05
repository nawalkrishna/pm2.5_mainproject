"""Issue-time features for the tabular forecast models (Stage 6, D046).

The hourly table is held as station x hour arrays on the full calendar grid, so a shift by k
columns is a true k-hour lag. For an issue hour H and horizon h:

    PM2.5        values at H - k (pm_lags), mean and valid count over the last k hours, last
                 valid value within 24 h (pm_last24) and hours since it
    meteorology  instantaneous fields at H and their mean over the last 24 h (instant_lag 0);
                 precipitation at H - 1 and its sum over the 24 h ending at H - 1 (average_lag 1)
    aerosol      MERRA-2 at H - 1 and its mean over the 24 h ending at H - 1
    calendar     of the target hour H + h: hour and day-of-year sin/cos, month, season, weekday
    static       station features
    setting b    instantaneous fields and precipitation at H + h, precipitation sum and mean
                 wind speed / BLH over H + 1 .. H + h ("perfect weather forecast")
    tier A       the latest satellite day: day of H if H >= 15 IST, else the day before

Usage:
    from src.forecast.features import HourlyArrays, issue_frame
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEASON_CODES = {"winter": 0, "pre_monsoon": 1, "sw_monsoon": 2, "ne_monsoon": 3}


def lag(a: np.ndarray, k: int) -> np.ndarray:
    """Value at column t - k (k > 0: past; k < 0: future), NaN beyond the grid."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:, k:] = a[:, :-k]
    elif k < 0:
        out[:, :k] = a[:, -k:]
    else:
        out[:] = a
    return out


def window(a: np.ndarray, k: int, end_lag: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and valid count over the k hours ending at t - end_lag (NaN mean if none valid)."""
    v = np.nan_to_num(a)
    c = np.isfinite(a).astype(np.float32)
    cs = np.concatenate([np.zeros((a.shape[0], 1), np.float32), np.cumsum(v, 1, dtype=np.float64).astype(np.float32)], 1)
    cc = np.concatenate([np.zeros((a.shape[0], 1), np.float32), np.cumsum(c, 1)], 1)
    n = a.shape[1]
    hi = np.arange(n) - end_lag + 1                    # exclusive end index into cs
    lo = hi - k
    ok = lo >= 0
    s = np.full(a.shape, np.nan, np.float32)
    cnt = np.full(a.shape, np.nan, np.float32)
    s[:, ok] = cs[:, hi[ok]] - cs[:, lo[ok]]
    cnt[:, ok] = cc[:, hi[ok]] - cc[:, lo[ok]]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s / cnt, np.nan)
    return mean.astype(np.float32), cnt


def last_valid(a: np.ndarray, max_age: int) -> tuple[np.ndarray, np.ndarray]:
    """Last valid value at or before t and its age in hours (NaN if older than max_age)."""
    n = a.shape[1]
    idx = np.where(np.isfinite(a), np.arange(n), -1)
    idx = np.maximum.accumulate(idx, axis=1)
    age = (np.arange(n) - idx).astype(np.float32)
    val = np.take_along_axis(a, np.maximum(idx, 0), axis=1)
    bad = (idx < 0) | (age > max_age)
    return np.where(bad, np.nan, val).astype(np.float32), np.where(bad, np.nan, age).astype(np.float32)


@dataclass
class HourlyArrays:
    stations: pd.DataFrame          # station_id, state (row order of the arrays)
    times: pd.DatetimeIndex         # IST hour starts (column order)
    var: dict                       # name -> (n_stations, n_hours) float32
    static: pd.DataFrame            # per station
    sat: dict                       # name -> (n_stations, n_days) float32, day 0 = first grid day

    @classmethod
    def load(cls, hourly: pd.DataFrame, sat_daily: pd.DataFrame, cfg: dict) -> "HourlyArrays":
        hourly = hourly.sort_values(["station_id", "time_ist"])
        st = hourly.drop_duplicates("station_id")[["station_id", "state"]].reset_index(drop=True)
        times = pd.DatetimeIndex(hourly["time_ist"].iloc[: len(hourly) // len(st)])
        shape = (len(st), len(times))
        names = ["pm25", *cfg["meteorology"], *cfg["aerosol"]]
        var = {v: hourly[v].to_numpy(np.float32).reshape(shape) for v in names}
        static = hourly.groupby("station_id", sort=True)[cfg["static"]].first().reset_index()
        # population changes by year: keep the hourly column as an array too
        var["population_b5"] = hourly["population_b5"].to_numpy(np.float32).reshape(shape)
        days = pd.Index(times.normalize().unique().strftime("%Y-%m-%d"))
        g = cfg["satellite_geometry"]
        sat = {}
        s = sat_daily.set_index(["station_id", "date_ist"])
        full = pd.MultiIndex.from_product([st["station_id"], days])
        for c in cfg["satellite_columns"] + cfg["satellite_counts"]:
            sat[c] = s[f"{c}_{g}"].reindex(full).to_numpy(np.float32).reshape(len(st), len(days))
        return cls(st, times, var, static, sat)


def calendar(times: pd.DatetimeIndex, seasons: dict) -> dict:
    month_to_season = {m: SEASON_CODES[s] for s, months in seasons.items() for m in months}
    doy = times.day_of_year.to_numpy()
    year_len = np.where(times.is_leap_year, 366, 365)
    hour = times.hour.to_numpy()
    return {"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * doy / year_len), "doy_cos": np.cos(2 * np.pi * doy / year_len),
            "month": times.month.to_numpy(), "season_code": times.month.map(month_to_season).to_numpy(),
            "weekday": times.day_of_week.to_numpy()}


def base_features(A: HourlyArrays, cfg: dict) -> dict:
    """Features that depend only on the issue hour H (setting a, tier B)."""
    t, lag_avg = cfg["tabular"], cfg["timing"]["average_lag"]
    pm = A.var["pm25"]
    f = {f"pm_lag{k}": lag(pm, k) for k in t["pm_lags"]}
    for k in t["pm_windows"]:
        f[f"pm_mean{k}"], f[f"pm_n{k}"] = window(pm, k)
    f["pm_last24"], f["pm_age"] = last_valid(pm, 24)
    for v in cfg["instant_met"]:
        f[v] = lag(A.var[v], cfg["timing"]["instant_lag"])
        for k in t["met_windows"]:
            f[f"{v}_mean{k}"] = window(A.var[v], k)[0]
    for v in cfg["average_met"]:
        f[v] = lag(A.var[v], lag_avg)
        for k in t["met_windows"]:
            m, n = window(A.var[v], k, end_lag=lag_avg)
            f[f"{v}_sum{k}"] = m * n
    for v in cfg["aerosol"]:
        f[v] = lag(A.var[v], lag_avg)
        f[f"{v}_mean24"] = window(A.var[v], 24, end_lag=lag_avg)[0]
    return f


def weather_ahead(A: HourlyArrays, cfg: dict, h: int) -> dict:
    """Setting b: reanalysis weather at and up to the target hour H + h."""
    f = {f"{v}_target": lag(A.var[v], -h) for v in cfg["instant_met"] + cfg["average_met"]}
    ahead = {v: lag(window(A.var[v], h)[0], -h) for v in ["precip_mm", "wind_speed", "blh"]}   # mean over H+1..H+h
    f["precip_ahead_sum"] = ahead["precip_mm"] * h
    f["wind_speed_ahead_mean"], f["blh_ahead_mean"] = ahead["wind_speed"], ahead["blh"]
    return f


def satellite_features(A: HourlyArrays, cfg: dict) -> dict:
    """Tier A: the latest satellite day at issue hour H, and its age in hours."""
    n_h = len(A.times)
    day = np.arange(n_h) // 24 - (A.times.hour.to_numpy() < cfg["timing"]["satellite_from_hour"])
    ok = day >= 0
    out = {}
    for c, arr in A.sat.items():
        a = np.full((arr.shape[0], n_h), np.nan, np.float32)
        a[:, ok] = arr[:, day[ok]]
        out[f"sat_{c}"] = a
    hour_of_day = np.arange(n_h) % 24
    out["sat_age_h"] = np.broadcast_to(np.where(day == np.arange(n_h) // 24, hour_of_day - 13, hour_of_day + 11),
                                       (arr.shape[0], n_h)).astype(np.float32)
    return out


def issue_frame(A: HourlyArrays, cfg: dict, seasons: dict, h: int, base: dict, sat: dict,
                issue_mask: np.ndarray) -> pd.DataFrame:
    """Rows = (station, issue hour H) with H in issue_mask and a valid target at H + h."""
    y = lag(A.var["pm25"], -h)
    rows = issue_mask[None, :] & np.isfinite(y)
    si, ti = np.nonzero(rows)
    cal = calendar(A.times, seasons)
    cols = {"station_id": A.stations["station_id"].to_numpy()[si], "state": A.stations["state"].to_numpy()[si],
            "issue_idx": ti, "h": np.full(len(ti), h, np.int16), "y": y[si, ti]}
    cols.update({k: v[si, ti] for k, v in base.items()})
    cols.update({k: v[si, ti] for k, v in weather_ahead(A, cfg, h).items()})
    cols.update({k: v[si, ti] for k, v in sat.items()})
    tgt = np.minimum(ti + h, len(A.times) - 1)
    cols.update({k: np.asarray(v)[tgt] for k, v in cal.items()})
    cols["population_b5"] = A.var["population_b5"][si, ti]
    df = pd.DataFrame(cols)
    st = A.static.drop(columns="population_b5")
    return df.merge(st, on="station_id", how="left")
