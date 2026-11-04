"""Compare the two target definitions and the three satellite buffers (Stage 4, tasks 3-4).

Inputs: data/processed/pm25_dataset_v<version>.parquet
Tables (reports/tables/):
    target_comparison.csv    per season/region: rows with each target valid, both valid,
                             medians, Spearman rho and median ratio overpass/daily
    buffer_coverage.csv      per geometry, season and region: share of rows with NO2 / AOD,
                             and Spearman rho of each with the daily target
Figures (reports/figures/):
    target_daily_vs_overpass.png, buffer_coverage.png

Usage:
    python -m src.features.compare [--version 2]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda import style
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_PROCESSED, REPORTS_FIGURES, REPORTS_TABLES

log = get_logger("compare")

SEASONS = ["winter", "pre_monsoon", "sw_monsoon", "ne_monsoon"]


def subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"all": df}
    for s in SEASONS:
        out[s] = df[df["season"] == s]
    out["kerala_jjas"] = df[(df["state"] == "Kerala") & df["month"].between(6, 9)]
    return out


def target_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, d in subsets(df).items():
        both = d.dropna(subset=["pm25_daily", "pm25_overpass"])
        rows.append({"subset": name, "rows": len(d), "valid_daily": int(d["pm25_daily"].notna().sum()),
                     "valid_overpass": int(d["pm25_overpass"].notna().sum()), "both_valid": len(both),
                     "median_daily": round(both["pm25_daily"].median(), 1),
                     "median_overpass": round(both["pm25_overpass"].median(), 1),
                     "spearman_rho": round(both["pm25_daily"].corr(both["pm25_overpass"], method="spearman"), 3),
                     "median_ratio_overpass_to_daily": round((both["pm25_overpass"] / both["pm25_daily"]).median(), 3)})
    return pd.DataFrame(rows)


def buffer_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, d in subsets(df).items():
        v = d[d["pm25_daily"].notna()]
        for g in ["pt", "b5", "b10"]:
            r = {"subset": name, "geometry": g, "rows": len(v)}
            for layer, col in [("no2", "no2_trop"), ("aod", "aod_clear")]:
                pair = v[["pm25_daily", f"{col}_{g}"]].dropna()
                r[f"share_{layer}"] = round(len(pair) / len(v), 3) if len(v) else np.nan
                r[f"rho_{layer}"] = round(pair["pm25_daily"].corr(pair[f"{col}_{g}"], method="spearman"), 3) \
                    if len(pair) > 30 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def plot_targets(df: pd.DataFrame, path) -> None:
    both = df.dropna(subset=["pm25_daily", "pm25_overpass"])
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.hexbin(both["pm25_daily"], both["pm25_overpass"], gridsize=60, bins="log", cmap=style.SEQUENTIAL,
              extent=(0, 150, 0, 150), mincnt=1, linewidths=0)
    ax.plot([0, 150], [0, 150], color=style.TEXT_2, linewidth=1, linestyle="--")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 150)
    ax.set_xlabel("daily mean PM2.5 (µg/m³)")
    ax.set_ylabel("12:00–15:00 IST mean PM2.5 (µg/m³)")
    ax.set_title("Overpass-window vs daily target (dashed: 1:1)")
    ax.grid(False)
    style.save(fig, path)


def plot_buffers(bc: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    order = ["all", "winter", "pre_monsoon", "sw_monsoon", "ne_monsoon", "kerala_jjas"]
    labels = ["All", "Winter", "Pre-mons.", "SW mons.", "NE mons.", "Kerala\nJun–Sep"]
    x = np.arange(len(order))
    for ax, layer, title in [(axes[0], "share_no2", "NO2"), (axes[1], "share_aod", "MAIAC AOD (clear)")]:
        for i, (g, color) in enumerate([("pt", "#1baf7a"), ("b5", "#eb6834"), ("b10", "#2a78d6")]):
            s = bc[bc["geometry"] == g].set_index("subset").reindex(order)[layer]
            ax.bar(x + (i - 1) * 0.26, s.to_numpy(), width=0.24, color=color,
                   label={"pt": "station pixel", "b5": "5 km buffer", "b10": "10 km buffer"}[g])
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title}: share of valid PM2.5 days with a value")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right")
    style.save(fig, path)


def main(version: int = 3) -> None:
    style.apply()
    df = pd.read_parquet(DATA_PROCESSED / f"pm25_dataset_v{version}.parquet")
    tc = target_comparison(df)
    tc.to_csv(REPORTS_TABLES / "target_comparison.csv", index=False)
    bc = buffer_coverage(df)
    bc.to_csv(REPORTS_TABLES / "buffer_coverage.csv", index=False)
    plot_targets(df, REPORTS_FIGURES / "target_daily_vs_overpass.png")
    plot_buffers(bc, REPORTS_FIGURES / "buffer_coverage.png")
    log.info("Targets:\n%s\nBuffers:\n%s", tc.to_string(index=False), bc.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", type=int, default=3)
    main(parser.parse_args().version)
