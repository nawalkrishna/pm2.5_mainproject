"""Results tables and figures from held-out predictions (Stage 5, tasks 1 and 4).

All metrics use pooled held-out predictions of a split (every fold's test rows together).
Derived "tiers":
    two_tier[_nolag]       Tier A where MAIAC AOD (10 km) exists, else Tier B (D039)
    tier_b[_nolag]_satrows Tier B scored only on the rows Tier A covers (fair A vs B comparison)

Tables (reports/tables/):
    results_overall.csv, results_station_sets.csv (D013 subsets), results_persistence_rows.csv,
    results_by_season.csv, results_by_cloud.csv, results_by_station.csv,
    results_kerala_jjas.csv, results_ablation.csv,
    results_ci.csv (block-bootstrap CIs, D043), results_sensitivity.csv (D042, if run)
Figures (reports/figures/):
    results_pred_vs_obs.png, results_residuals_by_season.png, results_station_spread.png,
    results_skill_by_cloud.png, results_ablation.png

Usage:
    python -m src.eval.report
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda import style
from src.eval.bootstrap import paired_ci
from src.eval.metrics import grouped
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_PROCESSED, REPORTS_FIGURES, REPORTS_TABLES

log = get_logger("report")

KEYS = ["split", "model", "tier"]
ROW_KEYS = ["split", "fold", "model", "station_id", "date_ist"]
SPLIT_ORDER = ["temporal", "blocked", "loso", "combined"]
SEASON_ORDER = ["winter", "pre_monsoon", "sw_monsoon", "ne_monsoon"]


def derive_tiers(p: pd.DataFrame) -> pd.DataFrame:
    """Add two_tier and tier_b_satrows predictions for linear and random_forest."""
    base = p[~p["tier"].str.startswith("ablation")]
    extra = []
    for suffix in ["", "_nolag"]:
        a = base[base["tier"] == f"tier_a{suffix}"]
        b = base[base["tier"] == f"tier_b{suffix}"]
        if a.empty or b.empty:
            continue
        merged = b.merge(a[ROW_KEYS + ["pred"]].rename(columns={"pred": "pred_a"}), on=ROW_KEYS, how="left")
        two = merged.assign(pred=merged["pred_a"].fillna(merged["pred"]), tier=f"two_tier{suffix}").drop(columns="pred_a")
        sat = merged[merged["pred_a"].notna()].drop(columns="pred_a").assign(tier=f"tier_b{suffix}_satrows")
        extra += [two, sat]
    return pd.concat([p] + extra, ignore_index=True)


def cloud_bin(p: pd.DataFrame, edges: list[float]) -> pd.Series:
    labels = [f"{a:.1f}-{min(b, 1):.1f}" for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(p["cloud_frac_b10"], edges, right=False, labels=labels).astype(object).fillna("missing").astype(str)


def usable_stations(cfg: dict, threshold: float) -> set[str]:
    """Stations with a valid daily target on >= threshold of study-period days (D013)."""
    period = load_config("study_area")["study_period"]
    n_days = len(pd.date_range(period["start"], period["end"]))
    d = pd.read_parquet(DATA_PROCESSED / f"pm25_dataset_v{cfg['dataset_version']}.parquet",
                        columns=["station_id", cfg["target"]])
    share = d[d[cfg["target"]].notna()].groupby("station_id").size() / n_days
    return set(share[share >= threshold].index)


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    return df.round({"rmse": 2, "mae": 2, "r2": 3, "bias": 2, "r": 3})


def plot_pred_vs_obs(p: pd.DataFrame, path) -> None:
    panels = [("temporal", "two_tier", "Temporal (2025), with PM2.5 lags"),
              ("loso", "two_tier_nolag", "Leave-one-station-out, no PM2.5 lags")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, (split, tier, title) in zip(axes, panels):
        g = p[(p["split"] == split) & (p["model"] == "random_forest") & (p["tier"] == tier)]
        ax.hexbin(g["y"], g["pred"], gridsize=60, bins="log", cmap=style.SEQUENTIAL, extent=(0, 150, 0, 150),
                  mincnt=1, linewidths=0)
        ax.plot([0, 150], [0, 150], color=style.TEXT_2, linewidth=1, linestyle="--")
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 150)
        ax.grid(False)
        ax.set_xlabel("observed daily PM2.5 (µg/m³)")
        ax.set_title(f"Random Forest two-tier\n{title}")
    axes[0].set_ylabel("predicted daily PM2.5 (µg/m³)")
    style.save(fig, path)


def plot_residuals(p: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (split, tier, title) in zip(axes, [("temporal", "two_tier", "Temporal (2025), with lags"),
                                               ("loso", "two_tier_nolag", "LOSO, no lags")]):
        g = p[(p["split"] == split) & (p["model"] == "random_forest") & (p["tier"] == tier)]
        data = [(g.loc[g["season"] == s, "pred"] - g.loc[g["season"] == s, "y"]).dropna() for s in SEASON_ORDER]
        ax.boxplot(data, showfliers=False, widths=0.5, medianprops={"color": "#eb6834", "linewidth": 2},
                   boxprops={"color": style.TEXT_2}, whiskerprops={"color": style.TEXT_2}, capprops={"color": style.TEXT_2})
        ax.axhline(0, color=style.TEXT_2, linewidth=1)
        ax.set_xticks(range(1, 5), ["Winter", "Pre-monsoon", "SW monsoon", "NE monsoon"])
        ax.set_title(f"Residuals (predicted − observed): {title}")
    axes[0].set_ylabel("µg/m³ (box: IQR, whiskers 1.5 IQR)")
    style.save(fig, path)


def plot_station_spread(st: pd.DataFrame, path) -> None:
    g = st[(st["split"] == "loso") & (st["model"] == "random_forest")]
    tiers = ["tier_b_nolag", "two_tier_nolag"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, tier in enumerate(tiers):
        v = g[g["tier"] == tier]
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
        ax.scatter(np.full(len(v), i) + jitter, v["r2"], s=14,
                   c=v["state"].map(style.STATE_COLORS).fillna(style.TEXT_2), alpha=0.85, linewidths=0)
        ax.hlines(v["r2"].median(), i - 0.25, i + 0.25, color=style.TEXT, linewidth=2)
        clipped = int((v["r2"] < -1.5).sum())
        if clipped:
            ax.annotate(f"{clipped} below −1.5", (i, -1.5), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=style.TEXT_2)
    for s in style.STATE_ORDER:
        ax.scatter([], [], color=style.STATE_COLORS[s], label=s, s=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.axhline(0, color=style.TEXT_2, linewidth=1)
    ax.set_xticks(range(len(tiers)), ["Tier B, no lags", "Two-tier, no lags"])
    ax.set_ylabel("R² at the held-out station")
    ax.set_ylim(-1.5, 1)
    ax.set_title("Per-station skill, leave-one-station-out (bar = median)")
    style.save(fig, path)


def plot_skill_by_cloud(cl: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, split, title in [(axes[0], "temporal", "Temporal (2025), no lags"), (axes[1], "loso", "LOSO, no lags")]:
        g = cl[(cl["split"] == split) & (cl["model"] == "random_forest")]
        for tier, label, color in [("tier_a_nolag", "Tier A (satellite)", "#2a78d6"),
                                   ("tier_b_nolag_satrows", "Tier B, same rows", "#eb6834"),
                                   ("tier_b_nolag", "Tier B, all rows", "#1baf7a")]:
            s = g[(g["tier"] == tier) & (g["cloud_bin"] != "missing")].sort_values("cloud_bin")
            ax.plot(s["cloud_bin"], s["rmse"], marker="o", markersize=4, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("TROPOMI cloud fraction (10 km)")
    axes[0].set_ylabel("RMSE (µg/m³)")
    axes[0].legend()
    style.save(fig, path)


def plot_ablation(ab: pd.DataFrame, path) -> None:
    order = ["ablation_met_only", "ablation_met_gases", "ablation_met_gases_aod", "ablation_all_no_lag"]
    labels = ["Met only", "+ gases", "+ AOD", "+ MERRA-2"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, split in zip(axes, ["temporal", "blocked"]):
        x = np.arange(len(order))
        for i, (subset, color, label) in enumerate([("all_rows", "#2a78d6", "all rows"),
                                                    ("aod_rows", "#eb6834", "rows with AOD")]):
            s = ab[(ab["split"] == split) & (ab["subset"] == subset)].set_index("tier").reindex(order)
            ax.bar(x + (i - 0.5) * 0.36, s["r2"], width=0.34, color=color, label=label)
        ax.set_xticks(x, labels)
        ax.set_title(f"Random Forest ablation, {split} split")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("R² (held-out)")
    axes[0].legend(loc="upper left")
    style.save(fig, path)


ROW_ID = ["split", "fold", "station_id", "date_ist"]
CI_COMPARISONS = [   # (label, first, second): CI of each R² and of R²(first) − R²(second), same rows
    ("Tier A − Tier B, same rows, lags", ("random_forest", "tier_a"), ("random_forest", "tier_b_satrows")),
    ("Tier A − Tier B, same rows, no lags", ("random_forest", "tier_a_nolag"), ("random_forest", "tier_b_nolag_satrows")),
    ("Linear: Tier A − Tier B, same rows, no lags", ("linear", "tier_a_nolag"), ("linear", "tier_b_nolag_satrows")),
    ("Two-tier no lags − climatology", ("random_forest", "two_tier_nolag"), ("climatology", "none")),
    ("Two-tier with lags − persistence", ("random_forest", "two_tier"), ("persistence", "none")),
    ("RF − linear, two-tier no lags", ("random_forest", "two_tier_nolag"), ("linear", "two_tier_nolag")),
]


def paired_rows(p: pd.DataFrame, first: tuple, second: tuple) -> pd.DataFrame:
    """Rows scored by both (model, tier) pairs: columns y, a, b plus the row keys."""
    def one(m, t, name):
        return p[(p["model"] == m) & (p["tier"] == t)].set_index(ROW_ID)[["y", "pred", "state", "season"]].rename(
            columns={"pred": name})
    a = one(*first, "a")
    return a.join(one(*second, "b")[["b"]], how="inner").dropna(subset=["y", "a", "b"]).reset_index()


def ci_rows(w: pd.DataFrame, blocks: pd.Series, cfg: dict) -> dict:
    return paired_ci(w.assign(block=w["station_id"].map(blocks)), "block", "y", ["a", "b"],
                     cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"])


def ci_table(base: pd.DataFrame, blocks: pd.Series, cfg: dict) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        s = base[base["split"] == split]
        for label, first, second in CI_COMPARISONS:
            w = paired_rows(s, first, second)
            for subset, ww in [("all", w), ("sw_monsoon", w[w["season"] == "sw_monsoon"])]:
                if len(ww):
                    rows.append({"split": split, "comparison": label, "subset": subset, **ci_rows(ww, blocks, cfg)})
    return pd.DataFrame(rows)


def sensitivity_table(main_rf: pd.DataFrame, sens: pd.DataFrame, blocks: pd.Series, cfg: dict) -> pd.DataFrame:
    """Tier A vs Tier B on each variant's Tier A rows, and the two-tier R², with CIs."""
    variants = [("main", main_rf)] + [(v, sens[sens["variant"] == v]) for v in sens["variant"].unique()]
    rows = []
    for name, v in variants:
        b_src = v if (v["tier"] == "tier_b").any() else main_rf      # satellite variants reuse Tier B
        for split in cfg["sensitivity"]["splits"]:
            for suffix, lags in [("_nolag", "no"), ("", "yes")]:
                a = v[(v["split"] == split) & (v["tier"] == f"tier_a{suffix}")]
                b = b_src[(b_src["split"] == split) & (b_src["tier"] == f"tier_b{suffix}")]
                both = pd.concat([a, b])
                same = paired_rows(both, ("random_forest", f"tier_a{suffix}"), ("random_forest", f"tier_b{suffix}"))
                two = b.set_index(ROW_ID)[["y", "pred"]].join(a.set_index(ROW_ID)["pred"].rename("pred_a"), how="left")
                two["a"] = two["pred_a"].fillna(two["pred"])
                two = two.rename(columns={"pred": "b"}).reset_index()
                ci_same, ci_two = ci_rows(same, blocks, cfg), ci_rows(two, blocks, cfg)
                rows.append({"variant": name, "split": split, "pm25_lags": lags,
                             "tier_a_rows": ci_same["n_rows"], "r2_tier_a": ci_same["r2_a"],
                             "r2_tier_b_same_rows": ci_same["r2_b"], "d_r2": ci_same["d_r2"],
                             "d_r2_lo": ci_same["d_r2_lo"], "d_r2_hi": ci_same["d_r2_hi"],
                             "rows": ci_two["n_rows"], "r2_two_tier": ci_two["r2_a"],
                             "r2_two_tier_lo": ci_two["r2_a_lo"], "r2_two_tier_hi": ci_two["r2_a_hi"],
                             "r2_tier_b_all_rows": ci_two["r2_b"]})
    return pd.DataFrame(rows)


def main() -> None:
    style.apply()
    cfg = load_config("experiments")
    p = pd.read_parquet(DATA_PROCESSED / f"predictions_v{cfg['dataset_version']}.parquet")
    p = derive_tiers(p)
    p["cloud_bin"] = cloud_bin(p, cfg["cloud_bins"])
    base = p[~p["tier"].str.startswith("ablation")]
    T, F = REPORTS_TABLES, REPORTS_FIGURES

    overall = fmt(grouped(base, [], KEYS))
    overall.to_csv(T / "results_overall.csv", index=False)
    audit = load_config("audit")
    sets = [overall.assign(station_set="all")]
    for name, thr in [("usable70", audit["min_daily_completeness"]), ("usable50", audit["sensitivity_daily_completeness"])]:
        keep = usable_stations(cfg, thr)
        sets.append(fmt(grouped(base[base["station_id"].isin(keep)], [], KEYS)).assign(station_set=name,
                                                                                         n_stations=len(keep)))
    pd.concat(sets).to_csv(T / "results_station_sets.csv", index=False)
    tp = base[base["split"] == "temporal"]
    per_rows = tp.loc[(tp["model"] == "persistence") & tp["pred"].notna(), ["station_id", "date_ist"]]
    fmt(grouped(tp.merge(per_rows, on=["station_id", "date_ist"]), [], KEYS)).to_csv(
        T / "results_persistence_rows.csv", index=False)       # every model on the rows persistence can score
    fmt(grouped(base, ["season"], KEYS)).to_csv(T / "results_by_season.csv", index=False)
    cl = fmt(grouped(base, ["cloud_bin"], KEYS))
    cl.to_csv(T / "results_by_cloud.csv", index=False)
    st = fmt(grouped(base, ["station_id", "state"], KEYS))
    st.to_csv(T / "results_by_station.csv", index=False)
    kj = base[(base["state"] == "Kerala") & base["month"].between(6, 9)]
    fmt(grouped(kj, [], KEYS)).to_csv(T / "results_kerala_jjas.csv", index=False)

    ab = p[p["tier"].str.startswith("ablation")]
    ab = pd.concat([grouped(ab, [], KEYS).assign(subset="all_rows"),
                    grouped(ab[ab["sat_aod_b10"] == 1], [], KEYS).assign(subset="aod_rows"),
                    grouped(ab[ab["season"] == "sw_monsoon"], [], KEYS).assign(subset="sw_monsoon")])
    ab = fmt(ab)
    ab.to_csv(T / "results_ablation.csv", index=False)

    blocks = pd.read_csv(T / "spatial_blocks.csv").set_index("station_id")["block"]
    ci = ci_table(base, blocks, cfg)
    ci.round(4).to_csv(T / "results_ci.csv", index=False)
    sens_path = DATA_PROCESSED / f"predictions_sensitivity_v{cfg['dataset_version']}.parquet"
    if sens_path.exists():
        main_rf = base[(base["model"] == "random_forest") & base["split"].isin(cfg["sensitivity"]["splits"])]
        sensitivity_table(main_rf, pd.read_parquet(sens_path), blocks, cfg).round(4).to_csv(
            T / "results_sensitivity.csv", index=False)

    plot_pred_vs_obs(base, F / "results_pred_vs_obs.png")
    plot_residuals(base, F / "results_residuals_by_season.png")
    plot_station_spread(st, F / "results_station_spread.png")
    plot_skill_by_cloud(cl, F / "results_skill_by_cloud.png")
    plot_ablation(ab, F / "results_ablation.png")
    show = overall[overall["model"].isin(["persistence", "climatology", "linear", "random_forest"])]
    log.info("Overall:\n%s", show.to_string(index=False))


if __name__ == "__main__":
    main()
