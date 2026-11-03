"""PM2.5 seasonality, distribution, correlations (Stage 3, task 3).

Uses valid PM2.5 station-days (>= 18 h) in the study period 2021-2025. No QC yet: values are
as received, so extreme values are visible here and listed in data_quality.py.

Tables (reports/tables/):
    eda_pm25_by_station.csv          per station: n days, median, IQR, p95, max
    eda_pm25_monthly_state.csv       state x month: median and IQR of daily PM2.5
    eda_pm25_season_state.csv        state x season: median daily PM2.5
    eda_correlation_spearman.csv     Spearman rho of daily PM2.5 with each predictor, overall,
                                     by season, and for Kerala Jun-Sep (pairwise complete)
    eda_station_pairs.csv            station pairs: distance and Spearman correlation of daily PM2.5
Figures (reports/figures/):
    eda_pm25_monthly_by_state.png, eda_pm25_by_station.png,
    eda_correlation_heatmap.png, eda_station_corr_vs_distance.png

Usage:
    python -m src.eda.pm25_patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.align.spatial import haversine_m
from src.eda import style
from src.eda.build_daily import OUT_PATH as DAILY_PATH
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import REPORTS_FIGURES, REPORTS_TABLES

log = get_logger("pm25_patterns")

PREDICTORS = {
    "no2_trop_b10": "NO2 column (10 km)", "co_b10": "CO column (10 km)", "so2_b10": "SO2 column (10 km)",
    "hcho_b10": "HCHO column (10 km)", "uvai_b10": "UV aerosol index", "cloud_frac_b10": "Cloud fraction",
    "aod_clear_pt": "MAIAC AOD (pixel)", "aod_clear_b10": "MAIAC AOD (10 km)",
    "merra2_pm25": "MERRA-2 PM2.5", "merra2_aod": "MERRA-2 AOD",
    "t2m_c": "Temperature 2 m", "rh": "Relative humidity", "wind_speed": "Wind speed",
    "precip_mm": "Precipitation", "blh": "Boundary-layer height",
    "elevation_m": "Elevation", "population_b5": "Population (5 km)",
}
SEASON_ORDER = ["winter", "pre_monsoon", "sw_monsoon", "ne_monsoon"]
MIN_PAIR_DAYS = 180


def load_valid() -> pd.DataFrame:
    period = load_config("study_area")["study_period"]
    d = pd.read_parquet(DAILY_PATH)
    return d[d["pm25_valid"] & d["date_ist"].between(period["start"], period["end"])].copy()


def by_station(v: pd.DataFrame) -> pd.DataFrame:
    g = v.groupby(["station_id", "name", "state"])["pm25"]
    out = g.agg(n_days="size", median="median", p25=lambda s: s.quantile(0.25),
                p75=lambda s: s.quantile(0.75), p95=lambda s: s.quantile(0.95), max="max").reset_index()
    return out.round(1)


def monthly_state(v: pd.DataFrame) -> pd.DataFrame:
    g = v.groupby(["state", "month"])["pm25"]
    return g.agg(median="median", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75),
                 n="size").round(1).reset_index()


def spearman(v: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho of PM2.5 with each predictor, pairwise complete, for several subsets."""
    subsets = {"all": v, **{s: v[v["season"] == s] for s in SEASON_ORDER},
               "kerala_jjas": v[(v["state"] == "Kerala") & v["month"].between(6, 9)]}
    rows = []
    for col, label in PREDICTORS.items():
        r = {"predictor": col, "label": label}
        for name, sub in subsets.items():
            pair = sub[["pm25", col]].dropna()
            r[f"rho_{name}"] = round(float(pair["pm25"].corr(pair[col], method="spearman")), 3) if len(pair) > 30 else np.nan
            r[f"n_{name}"] = len(pair)
        rows.append(r)
    return pd.DataFrame(rows)


def station_pairs(v: pd.DataFrame) -> pd.DataFrame:
    wide = v.pivot_table(index="date_ist", columns="station_id", values="pm25")
    meta = v.drop_duplicates("station_id").set_index("station_id")[["state", "latitude", "longitude"]]
    ids = list(wide.columns)
    corr = wide.corr(method="spearman", min_periods=MIN_PAIR_DAYS)   # robust to unQCed spikes
    n = wide.notna().astype(int).T @ wide.notna().astype(int)
    rows = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if pd.isna(corr.loc[a, b]):
                continue
            dist = float(haversine_m(meta.loc[a, "latitude"], meta.loc[a, "longitude"],
                                     np.array([meta.loc[b, "latitude"]]), np.array([meta.loc[b, "longitude"]]))[0])
            rows.append({"station_a": a, "station_b": b, "state_a": meta.loc[a, "state"], "state_b": meta.loc[b, "state"],
                         "distance_km": round(dist / 1000, 2), "shared_days": int(n.loc[a, b]),
                         "spearman_rho": round(float(corr.loc[a, b]), 3)})
    return pd.DataFrame(rows)


def plot_monthly(ms: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ends = []
    for state in style.STATE_ORDER:
        s = ms[ms["state"] == state].sort_values("month")
        ax.plot(s["month"], s["median"], color=style.STATE_COLORS[state], linewidth=2.6 if state == "Kerala" else 1.6,
                label=state, marker="o", markersize=4)
        ends.append((state, s["median"].iloc[-1]))
    style.label_line_ends(ax, ends, 12.2, min_gap=2.0)
    ax.axvspan(5.5, 9.5, color=style.GRID, alpha=0.5, linewidth=0)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_xlim(0.8, 13.6)
    ax.set_ylim(0, None)
    ax.set_ylabel("median daily PM2.5 (µg/m³)")
    ax.set_title("PM2.5 by calendar month and state, 2021–2025 (shaded: SW monsoon)")
    ax.legend(loc="lower center", ncol=5)
    style.save(fig, path)


def plot_by_station(bs: pd.DataFrame, path) -> None:
    bs = bs.assign(k=bs["state"].map({s: i for i, s in enumerate(style.STATE_ORDER)})).sort_values(["k", "median"])
    fig, ax = plt.subplots(figsize=(8, 13))
    y = np.arange(len(bs))
    colors = bs["state"].map(style.STATE_COLORS)
    ax.hlines(y, bs["p25"], bs["p75"], color=colors, linewidth=2)
    ax.scatter(bs["median"], y, color=colors, s=16, zorder=3, edgecolor=style.SURFACE, linewidth=0.8)
    ax.set_yticks(y, [n.split(" - ")[0][:34] for n in bs["name"]], fontsize=5.5)
    ax.set_ylim(-1, len(bs))
    ax.set_xlabel("daily PM2.5 (µg/m³): dot = median, bar = 25th–75th percentile")
    for state in style.STATE_ORDER:
        ax.scatter([], [], color=style.STATE_COLORS[state], label=state, s=16)
    ax.legend(loc="lower right")
    ax.set_title("Daily PM2.5 per station, 2021–2025 (valid days)")
    style.save(fig, path)


def plot_correlation(sp: pd.DataFrame, path) -> None:
    cols = ["rho_all"] + [f"rho_{s}" for s in SEASON_ORDER] + ["rho_kerala_jjas"]
    m = sp.set_index("label")[cols]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(m.to_numpy(dtype=float), cmap=style.DIVERGING, vmin=-0.8, vmax=0.8, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(cols)), ["All", "Winter", "Pre-monsoon", "SW monsoon", "NE monsoon", "Kerala\nJun–Sep"])
    ax.set_yticks(range(len(m)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                        color=style.SURFACE if abs(val) > 0.5 else style.TEXT)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Spearman ρ with daily PM2.5")
    ax.set_title("Correlation of daily PM2.5 with candidate predictors (2021–2025)")
    style.save(fig, path)


def plot_pairs(pairs: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    same = pairs["state_a"] == pairs["state_b"]
    ax.scatter(pairs.loc[~same, "distance_km"], pairs.loc[~same, "spearman_rho"], s=5, color="#9cc3f0", alpha=0.6,
               label="pair in different states", linewidths=0)
    ax.scatter(pairs.loc[same, "distance_km"], pairs.loc[same, "spearman_rho"], s=5, color="#eb6834", alpha=0.6,
               label="pair in same state", linewidths=0)
    bins = [0, 10, 25, 50, 100, 200, 400, 700, 1000, 1600]
    b = pairs.groupby(pd.cut(pairs["distance_km"], bins), observed=True)["spearman_rho"].median()
    mids = [(i.left + i.right) / 2 for i in b.index]
    ax.plot(mids, b.to_numpy(), color=style.TEXT, linewidth=2, marker="o", markersize=4, label="median per distance bin")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("distance between stations (km, log scale beyond 10 km)")
    ax.set_ylabel(f"Spearman ρ of daily PM2.5 (≥ {MIN_PAIR_DAYS} shared days)")
    ax.set_ylim(-0.2, 1)
    ax.legend(loc="lower left")
    ax.set_title("Station-to-station correlation of daily PM2.5 falls only slowly with distance")
    style.save(fig, path)


def main() -> None:
    style.apply()
    v = load_valid()
    T, F = REPORTS_TABLES, REPORTS_FIGURES
    bs = by_station(v)
    bs.to_csv(T / "eda_pm25_by_station.csv", index=False)
    ms = monthly_state(v)
    ms.to_csv(T / "eda_pm25_monthly_state.csv", index=False)
    v.groupby(["state", "season"])["pm25"].median().round(1).unstack()[SEASON_ORDER].to_csv(
        T / "eda_pm25_season_state.csv")
    sp = spearman(v)
    sp.to_csv(T / "eda_correlation_spearman.csv", index=False)
    pairs = station_pairs(v)
    pairs.to_csv(T / "eda_station_pairs.csv", index=False)
    plot_monthly(ms, F / "eda_pm25_monthly_by_state.png")
    plot_by_station(bs, F / "eda_pm25_by_station.png")
    plot_correlation(sp, F / "eda_correlation_heatmap.png")
    plot_pairs(pairs, F / "eda_station_corr_vs_distance.png")
    log.info("Valid station-days: %d; station pairs: %d", len(v), len(pairs))


if __name__ == "__main__":
    main()
