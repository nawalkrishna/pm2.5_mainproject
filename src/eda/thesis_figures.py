"""Figures for the thesis: the study-area station map, the pipeline diagram, and the export of
every figure the thesis uses into thesis/figures (as RGB PNGs, which compile faster).

    thesis_study_area.png   the 113 stations with valid daily PM2.5 in dataset v3, by state
    thesis_pipeline.png     data sources -> QC -> alignment -> features -> models -> evaluation

Usage:
    python -m src.eda.thesis_figures
"""

import re

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

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
    """Compact layout that stays readable at text width: sources on top, processing below."""
    box = {"ground": (0.1, 3.55, "Ground PM2.5\nCPCB + US Consulates\n(hourly)"),
           "sat": (2.3, 3.55, "Satellite\nTROPOMI, MAIAC\n(daily, 10 km)"),
           "rea": (4.5, 3.55, "Reanalysis + static\nERA5(-Land), MERRA-2,\nSRTM, WorldCover, WorldPop"),
           "qc": (0.1, 2.35, "Quality control\n(rule S3a-12h)"),
           "align": (2.3, 1.15, "Alignment\n(IST days, overpass\nwindow, lags)"),
           "feat": (0.1, -0.05, "Feature sets\nTier A / Tier B"),
           "model": (2.3, -0.05, "Models\nclimatology, linear, RF,\nLSTM, GC-LSTM"),
           "eval": (4.5, -0.05, "Evaluation\ntemporal, blocked, LOSO,\ncombined; bootstrap")}
    w, h = 2.0, 0.85
    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 6.6)
    ax.set_ylim(-0.15, 4.5)
    for x, y, text in box.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", facecolor="#eef4fc",
                                    edgecolor="#2a78d6", linewidth=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=style.TEXT)

    def edge(name, side):
        x, y, _ = box[name]
        return {"top": (x + w / 2, y + h + 0.03), "bottom": (x + w / 2, y - 0.03),
                "left": (x - 0.03, y + h / 2), "right": (x + w + 0.03, y + h / 2)}[side]

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color="#52514e", linewidth=1))

    arrow(edge("ground", "bottom"), edge("qc", "top"))
    arrow(edge("qc", "bottom"), edge("align", "left"))
    arrow(edge("sat", "bottom"), edge("align", "top"))
    arrow(edge("rea", "bottom"), edge("align", "right"))
    arrow(edge("align", "bottom"), edge("feat", "top"))
    arrow(edge("feat", "right"), edge("model", "left"))
    arrow(edge("model", "right"), edge("eval", "left"))
    style.save(fig, REPORTS_FIGURES / "thesis_pipeline.png")


def export_for_thesis() -> None:
    """Copy the figures the thesis uses into thesis/figures as RGB PNGs (no alpha channel).

    pdfTeX has to split and recompress the alpha channel of RGBA PNGs on every compile, which
    doubled the build time and pushed it past Overleaf's free compile limit; RGB PNGs are
    embedded as they are. Figures the thesis does not use are removed from thesis/figures.
    """
    thesis = REPORTS_FIGURES.parents[1] / "thesis"
    out = thesis / "figures"
    used = set()
    for tex in thesis.glob("chapter*.tex"):
        used |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text(encoding="utf-8")))
    used = {u.removesuffix(".png") for u in used}
    for old in out.glob("*.png"):
        if old.stem not in used:
            old.unlink()
    for name in sorted(used):
        im = Image.open(REPORTS_FIGURES / f"{name}.png").convert("RGBA")
        rgb = Image.new("RGB", im.size, (255, 255, 255))
        rgb.paste(im, mask=im.split()[3])
        rgb.save(out / f"{name}.png", optimize=True)
    print(f"{len(used)} figures exported to {out}")


def main() -> None:
    style.apply()
    study_area()
    pipeline()
    export_for_thesis()


if __name__ == "__main__":
    main()
