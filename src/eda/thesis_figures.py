"""Two figures made for the thesis: the study-area station map and the pipeline diagram.

    thesis_study_area.png   the 113 stations with valid daily PM2.5 in dataset v3, by state
    thesis_pipeline.png     data sources -> QC -> alignment -> features -> models -> evaluation

Usage:
    python -m src.eda.thesis_figures
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.eda import style
from src.ingest.boundaries import GEOJSON_NAME, OUT_DIR as BOUNDARY_DIR
from src.ingest.ground_pm25 import STATIONS_PATH
from src.utils.config import load_config
from src.utils.paths import DATA_PROCESSED, REPORTS_FIGURES


def study_area() -> None:
    area = load_config("study_area")
    names = {k: v for k, v in area["gadm_state_names"].items()}
    states = gpd.read_file(BOUNDARY_DIR / GEOJSON_NAME)
    states = states[states["NAME_1"].isin(names)].assign(state=lambda g: g["NAME_1"].map(names))
    ds = pd.read_parquet(DATA_PROCESSED / "pm25_dataset_v3.parquet", columns=["station_id", "pm25_daily"])
    used = ds.loc[ds["pm25_daily"].notna(), "station_id"].unique()
    st = pd.read_csv(STATIONS_PATH)
    st = st[st["station_id"].isin(used)]
    fig, ax = plt.subplots(figsize=(6, 6.6))
    states.plot(ax=ax, facecolor="#f3f2ee", edgecolor="#8a8984", linewidth=0.6)
    for s in style.STATE_ORDER:
        g = st[st["state"] == s]
        ax.scatter(g["longitude"], g["latitude"], s=18, color=style.STATE_COLORS[s], edgecolor="white",
                   linewidth=0.4, label=f"{s} ({len(g)})", zorder=3)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", title=f"Stations with valid\ndaily PM2.5 (n = {len(st)})", title_fontsize=8)
    ax.set_title("Study area and ground monitoring stations")
    style.save(fig, REPORTS_FIGURES / "thesis_study_area.png")


def pipeline() -> None:
    steps = [
        ("Ground PM2.5\nCPCB + US Consulates\n(hourly)", "Satellite\nTROPOMI, MAIAC\n(daily, 10 km)",
         "Reanalysis + static\nERA5(-Land), MERRA-2,\nSRTM, WorldCover, WorldPop"),
        ("Quality control\n(rule S3a-12h)",),
        ("Alignment (IST days,\noverpass window, lags)",),
        ("Feature sets\nTier A / Tier B",),
        ("Models\nclimatology, linear,\nRandom Forest,\nLSTM, GC-LSTM",),
        ("Evaluation\ntemporal, blocked,\nLOSO, combined;\nblock bootstrap",),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.4)
    w, h = 1.55, 0.85
    xs = [0.15, 2.25, 4.0, 5.75, 7.5, 9.3]
    centres = []
    for i, (x, group) in enumerate(zip(xs, steps)):
        ys = [2.45, 1.3, 0.15] if len(group) == 3 else [2.45] if i == 1 else [1.3]   # QC beside ground data
        for y, text in zip(ys, group):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", facecolor="#eef4fc",
                                        edgecolor="#2a78d6", linewidth=1))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, color=style.TEXT)
        centres.append((x, [y + h / 2 for y in ys]))
    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, color="#52514e", linewidth=0.9))

    (x_src, y_src), (x_qc, y_qc), (x_al, y_al) = centres[0], centres[1], centres[2]
    arrow((x_src + w + 0.03, y_src[0]), (x_qc - 0.03, y_qc[0]))            # ground PM2.5 -> QC
    for y0 in y_src[1:]:                                                    # satellite, reanalysis -> alignment
        arrow((x_src + w + 0.03, y0), (x_al - 0.03, y_al[0]))
    for (x0, y0s), (x1, y1s) in zip(centres[1:-1], centres[2:]):
        arrow((x0 + w + 0.03, y0s[0]), (x1 - 0.03, y1s[0]))
    style.save(fig, REPORTS_FIGURES / "thesis_pipeline.png")


def main() -> None:
    style.apply()
    study_area()
    pipeline()


if __name__ == "__main__":
    main()
