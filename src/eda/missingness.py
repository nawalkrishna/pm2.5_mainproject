"""Missingness of ground and satellite data, and how often they coincide (Stage 3, task 2 and 5).

All shares are computed on the full station-day grid 2021-01-01..2025-12-31 (study period,
D011), so a station that had not started yet counts as missing.

Tables (reports/tables/):
    eda_missing_by_variable.csv        share of station-days missing, per variable
    eda_missing_by_station.csv         per station: PM2.5 valid share, NO2/AOD availability (pt, b10)
    eda_sat_availability_month_state.csv   calendar month x state: availability of NO2 / AOD (pt, b10)
    eda_feasibility.csv                state x season: valid PM2.5 days and how many also have
                                       NO2 / AOD / both, per geometry
    eda_aod_vs_cloud.csv               AOD availability by TROPOMI cloud-fraction bin
Figures (reports/figures/):
    eda_pm25_valid_heatmap.png, eda_sat_availability_by_month.png,
    eda_feasibility_by_month.png, eda_aod_vs_cloud.png

Usage:
    python -m src.eda.missingness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.eda import style
from src.eda.build_daily import OUT_PATH as DAILY_PATH
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import REPORTS_FIGURES, REPORTS_TABLES

log = get_logger("missingness")

SAT = {"NO2": "no2_trop", "AOD": "aod_clear"}
GEOMS = ["pt", "b10"]
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def load_study() -> pd.DataFrame:
    period = load_config("study_area")["study_period"]
    d = pd.read_parquet(DAILY_PATH)
    d = d[d["date_ist"].between(period["start"], period["end"])].copy()
    for name, col in SAT.items():
        for g in GEOMS:
            d[f"has_{name}_{g}"] = d[f"{col}_{g}"].notna()
    return d


def missing_by_variable(d: pd.DataFrame) -> pd.DataFrame:
    cols = (["pm25"] + [f"{l}_{g}" for l in ["no2_trop", "co", "so2", "hcho", "uvai", "cloud_frac",
                                              "aod_any", "aod_clear", "aod_best"] for g in ["pt", "b5", "b10"]]
            + ["t2m_c", "rh", "wind_speed", "precip_mm", "blh", "merra2_pm25", "merra2_aod"])
    out = pd.DataFrame({"variable": cols, "share_missing": [float(d[c].isna().mean()) for c in cols]})
    out.loc[0, "share_missing"] = float((~d["pm25_valid"]).mean())     # PM2.5: missing = not a valid day
    out.loc[0, "variable"] = "pm25 (valid day, >= 18 h)"
    return out.round(3)


def by_station(d: pd.DataFrame) -> pd.DataFrame:
    agg = {"pm25_valid": "mean", **{f"has_{n}_{g}": "mean" for n in SAT for g in GEOMS}}
    return d.groupby(["station_id", "name", "state"]).agg(agg).round(3).reset_index()


def month_state(d: pd.DataFrame) -> pd.DataFrame:
    cols = [f"has_{n}_{g}" for n in SAT for g in GEOMS]
    return d.groupby(["state", "month"])[cols].mean().round(3).reset_index()


def feasibility(d: pd.DataFrame) -> pd.DataFrame:
    v = d[d["pm25_valid"]]
    rows = []
    for (state, season), g in list(v.groupby(["state", "season"])) + [(("ALL", "all"), v)] + \
            [((s, "all"), g) for s, g in v.groupby("state")]:
        r = {"state": state, "season": season, "valid_pm25_days": len(g)}
        for geom in GEOMS:
            r[f"with_NO2_{geom}"] = int(g[f"has_NO2_{geom}"].sum())
            r[f"with_AOD_{geom}"] = int(g[f"has_AOD_{geom}"].sum())
            r[f"with_both_{geom}"] = int((g[f"has_NO2_{geom}"] & g[f"has_AOD_{geom}"]).sum())
            r[f"share_with_AOD_{geom}"] = round(r[f"with_AOD_{geom}"] / len(g), 3)
            r[f"share_with_both_{geom}"] = round(r[f"with_both_{geom}"] / len(g), 3)
        rows.append(r)
    return pd.DataFrame(rows)


def aod_vs_cloud(d: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    c = pd.cut(d["cloud_frac_b10"], bins, right=False)
    out = d.groupby(c, observed=True).agg(station_days=("station_id", "size"),
                                          aod_pt=("has_AOD_pt", "mean"), aod_b10=("has_AOD_b10", "mean"),
                                          no2_pt=("has_NO2_pt", "mean"), no2_b10=("has_NO2_b10", "mean")).round(3)
    out.index = [f"{b.left:.1f}-{min(b.right, 1):.1f}" for b in out.index]
    return out.rename_axis("cloud_fraction_b10").reset_index()


def plot_valid_heatmap(d: pd.DataFrame, path) -> None:
    d = d.assign(ym=d["date_ist"].str[:7])
    m = d.pivot_table(index="station_id", columns="ym", values="pm25_valid", aggfunc="mean")
    order = (d.drop_duplicates("station_id").assign(k=lambda x: x["state"].map(
        {s: i for i, s in enumerate(style.STATE_ORDER)})).sort_values(["k", "name"]))
    m = m.reindex(order["station_id"])
    fig, ax = plt.subplots(figsize=(10, 11))
    im = ax.imshow(m.to_numpy(), aspect="auto", cmap=style.SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
    ax.grid(False)
    ticks = [i for i, c in enumerate(m.columns) if c.endswith("-01")]
    ax.set_xticks(ticks, [m.columns[i][:4] for i in ticks])
    states = order["state"].to_numpy()
    starts = [0] + [i for i in range(1, len(states)) if states[i] != states[i - 1]]
    for s in starts[1:]:
        ax.axhline(s - 0.5, color=style.SURFACE, linewidth=2)
    ax.set_yticks([(a + b) / 2 for a, b in zip(starts, starts[1:] + [len(states)])], [states[s] for s in starts])
    ax.set_title("Share of days with a valid PM2.5 daily mean (≥ 18 h), per station and month, 2021–2025")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="share of days valid")
    style.save(fig, path)


def plot_month_state(ms: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for i, name in enumerate(SAT):
        for j, g in enumerate(GEOMS):
            ax = axes[i, j]
            ends = []
            for state in style.STATE_ORDER:
                s = ms[ms["state"] == state].sort_values("month")
                ax.plot(s["month"], s[f"has_{name}_{g}"], color=style.STATE_COLORS[state],
                        linewidth=2.6 if state == "Kerala" else 1.6, label=state)
                ends.append((state, s[f"has_{name}_{g}"].iloc[-1]))
            style.label_line_ends(ax, ends, 12.15)
            ax.axvspan(5.5, 9.5, color=style.GRID, alpha=0.5, linewidth=0)
            ax.set_title(f"{name}, {'station pixel' if g == 'pt' else '10 km buffer'}")
            ax.set_ylim(0, 1)
            ax.set_xlim(1, 13.8)
            ax.set_xticks(range(1, 13), MONTHS)
    axes[0, 0].set_ylabel("share of station-days with a value")
    axes[1, 0].set_ylabel("share of station-days with a value")
    axes[0, 1].legend(loc="lower left", ncol=1)
    fig.suptitle("Satellite availability by calendar month, 2021–2025 (shaded: SW monsoon Jun–Sep)",
                 fontsize=11, color=style.TEXT)
    style.save(fig, path)


def plot_feasibility_month(d: pd.DataFrame, path) -> None:
    v = d[d["pm25_valid"]].assign(region=lambda x: np.where(x["state"] == "Kerala", "Kerala", "Other four states"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, region in zip(axes, ["Kerala", "Other four states"]):
        g = v[v["region"] == region].groupby("month")
        ends = []
        for col, label, color in [("has_NO2_b10", "NO2 (10 km)", "#2a78d6"), ("has_AOD_b10", "AOD (10 km)", "#eb6834"),
                                  ("has_AOD_pt", "AOD (pixel)", "#1baf7a")]:
            s = g[col].mean()
            ax.plot(s.index, s.to_numpy(), color=color, label=label)
            ends.append((label, s.iloc[-1]))
        style.label_line_ends(ax, ends, 12.15)
        ax.axvspan(5.5, 9.5, color=style.GRID, alpha=0.5, linewidth=0)
        ax.set_title(f"{region}: valid PM2.5 days with satellite data")
        ax.set_xticks(range(1, 13), MONTHS)
        ax.set_xlim(1, 14)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share of valid PM2.5 station-days")
    axes[0].legend(loc="lower left")
    style.save(fig, path)


def plot_aod_vs_cloud(t: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(t))
    for col, label, color in [("aod_b10", "AOD, 10 km", "#eb6834"), ("aod_pt", "AOD, pixel", "#1baf7a"),
                              ("no2_b10", "NO2, 10 km", "#2a78d6")]:
        ax.plot(x, t[col], color=color, marker="o", markersize=4, label=label)
    ax.set_xticks(x, t["cloud_fraction_b10"], rotation=45)
    ax.set_xlabel("TROPOMI cloud fraction, 10 km buffer mean")
    ax.set_ylabel("share of station-days with a value")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Satellite availability falls with cloud fraction (2021–2025)")
    style.save(fig, path)


def main() -> dict:
    style.apply()
    d = load_study()
    T, F = REPORTS_TABLES, REPORTS_FIGURES
    missing_by_variable(d).to_csv(T / "eda_missing_by_variable.csv", index=False)
    by_station(d).to_csv(T / "eda_missing_by_station.csv", index=False)
    ms = month_state(d)
    ms.to_csv(T / "eda_sat_availability_month_state.csv", index=False)
    feas = feasibility(d)
    feas.to_csv(T / "eda_feasibility.csv", index=False)
    avc = aod_vs_cloud(d)
    avc.to_csv(T / "eda_aod_vs_cloud.csv", index=False)
    plot_valid_heatmap(d, F / "eda_pm25_valid_heatmap.png")
    plot_month_state(ms, F / "eda_sat_availability_by_month.png")
    plot_feasibility_month(d, F / "eda_feasibility_by_month.png")
    plot_aod_vs_cloud(avc, F / "eda_aod_vs_cloud.png")
    allrow = feas[(feas["state"] == "ALL")].iloc[0].to_dict()
    log.info("Feasibility (all states): %s", allrow)
    return allrow


if __name__ == "__main__":
    main()
