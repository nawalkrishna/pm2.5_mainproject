"""Stage 6 results: skill by horizon, satellite effect, cloud and season, bootstrap intervals.

Neural-network rows: seed 42 predictions for the bootstrap and breakdowns; the seed range
(min-max over seeds) is reported beside them. Intervals: paired cluster bootstrap over spatial
block x calendar month of the test year (D050).

Outputs (reports/tables, reports/figures):
    forecast_skill.csv              R², RMSE, MAE by model x setting x horizon (Tier B, all rows)
    forecast_skill_ci.csv           paired bootstrap of key differences by horizon
    forecast_satellite.csv          Tier A vs Tier B on Tier A rows, by horizon (+ CI)
    forecast_satellite_groups.csv   the same at 24 h by cloud bin and season
    forecast_satellite_seeds.csv    neural networks: Tier A - Tier B R² for every seed
    forecast_aod_coverage.csv       test rows with a satellite-day AOD value, by cloud bin
    forecast_seed_range.csv         neural networks: R² range over seeds
    forecast_r2_by_horizon.png      R² against horizon, settings a and b

Usage:
    python -m src.forecast.report
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval import splits as S
from src.eval.bootstrap import paired_ci
from src.eval.metrics import metrics
from src.ingest.ground_pm25 import STATIONS_PATH
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import DATA_PROCESSED, REPORTS_FIGURES, REPORTS_TABLES

log = get_logger("forecast_report")
KEY = ["station_id", "issue_idx", "h"]
SEASONS = {0: "winter", 1: "pre_monsoon", 2: "sw_monsoon", 3: "ne_monsoon"}


def load(version: int = 1) -> tuple[pd.DataFrame, list[int]]:
    tab = pd.read_parquet(DATA_PROCESSED / f"forecast_predictions_tabular_v{version}.parquet")
    nets_path = DATA_PROCESSED / f"forecast_predictions_nets_v{version}.parquet"
    if nets_path.exists():
        nets = pd.read_parquet(nets_path).drop(columns=["y", "time_issue"])
        tab = tab.merge(nets, on=KEY, how="left")
    cfg = load_config("experiments")["splits"]["spatial_blocks"]
    st = pd.read_csv(STATIONS_PATH)
    st = st[st["station_id"].isin(tab["station_id"].unique())][["station_id", "latitude", "longitude"]]
    blocks = S.spatial_blocks(st, cfg["block_km"])
    tab["cluster"] = tab["station_id"].map(blocks) + "_" + pd.to_datetime(tab["time_issue"]).dt.month.astype(str)
    seeds = sorted({int(c.rsplit("_s", 1)[1]) for c in tab.columns if c.startswith(("lstm_", "gcn_"))})
    return tab, seeds


def cloud_bin(frac: pd.Series, bins: list) -> pd.Series:
    """Cloud-fraction bin label (D041), "missing" where the satellite day has no cloud fraction."""
    return pd.cut(frac, bins, right=False).astype(str).where(frac.notna(), "missing")


def model_cols(df: pd.DataFrame, tier: str, seed: int = 42) -> dict:
    """Display name -> column, for one tier."""
    out = {"persistence": "persistence", "climatology": "climatology"}
    for s in ["a", "b"]:
        for m in ["lin", "rf", "lstm", "gcn"]:
            c = f"{m}_{s}_{tier}" if m in ["lin", "rf"] else f"{m}_{s}_{tier}_s{seed}"
            if c in df:
                out[f"{m}_{s}"] = c
    return out


def skill_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in model_cols(df, "B").items():
        for h, g in df.groupby("h"):
            rows.append({"model": name, "h": h, **metrics(g["y"], g[col])})
    return pd.DataFrame(rows)


def main(version: int = 1) -> None:
    cfg = load_config("forecast")
    bcfg = cfg["bootstrap"]
    df, seeds = load(version)
    REPORTS_TABLES.mkdir(parents=True, exist_ok=True)

    skill = skill_table(df)
    skill.to_csv(REPORTS_TABLES / "forecast_skill.csv", index=False)
    B = model_cols(df, "B")
    pairs = {(B["rf_a"], "persistence"): "RF (a) - persistence",
             (B["rf_b"], B["rf_a"]): "RF: setting b - a"}
    for m in ["lstm", "gcn"]:
        if f"{m}_a" in B:
            pairs[(B[f"{m}_a"], B["rf_a"])] = f"{m.upper()} (a) - RF (a)"
            pairs[(B[f"{m}_b"], B[f"{m}_a"])] = f"{m.upper()}: setting b - a"
    if "gcn_a" in B and "lstm_a" in B:
        pairs[(B["gcn_a"], B["lstm_a"])] = "GCN (a) - LSTM (a)"
    ci = []
    for (a, b), label in pairs.items():
        for h, g in df.groupby("h"):
            r = paired_ci(g, "cluster", "y", [a, b], bcfg["n"], bcfg["seed"])
            ci.append({"comparison": label, "h": h, "n_rows": r["n_rows"], "n_clusters": r["n_clusters"],
                       "r2_first": r[f"r2_{a}"], "r2_second": r[f"r2_{b}"], "d_r2": r["d_r2"],
                       "d_r2_lo": r["d_r2_lo"], "d_r2_hi": r["d_r2_hi"]})
    pd.DataFrame(ci).to_csv(REPORTS_TABLES / "forecast_skill_ci.csv", index=False)

    # Satellite effect: Tier A vs Tier B on Tier A rows (D049)
    A = model_cols(df, "A")
    sat, groups = [], []
    rows_a = df[df["sat_aod_clear"].notna()]
    for name in ["rf_a", "rf_b", "lstm_a", "lstm_b", "gcn_a", "gcn_b"]:
        if name not in A or name not in B:
            continue
        a, b = A[name], B[name]
        for h, g in rows_a.groupby("h"):
            r = paired_ci(g, "cluster", "y", [a, b], bcfg["n"], bcfg["seed"])
            sat.append({"model": name, "h": h, "n_rows": r["n_rows"], "r2_tier_a": r[f"r2_{a}"],
                        "r2_tier_b": r[f"r2_{b}"], "d_r2": r["d_r2"], "d_r2_lo": r["d_r2_lo"], "d_r2_hi": r["d_r2_hi"],
                        "rmse_tier_a": r[f"rmse_{a}"], "rmse_tier_b": r[f"rmse_{b}"]})
        g24 = rows_a[rows_a["h"] == 24].copy()
        g24["cloud_bin"] = cloud_bin(g24["sat_cloud_frac"], cfg["cloud_bins"])
        g24["season"] = g24["season_code"].map(SEASONS)
        for by in ["cloud_bin", "season"]:
            for k, g in g24.groupby(by):
                r = paired_ci(g, "cluster", "y", [a, b], bcfg["n"], bcfg["seed"])
                groups.append({"model": name, "group": by, "value": k, "n_rows": r["n_rows"],
                               "r2_tier_a": r[f"r2_{a}"], "r2_tier_b": r[f"r2_{b}"], "d_r2": r["d_r2"],
                               "d_r2_lo": r["d_r2_lo"], "d_r2_hi": r["d_r2_hi"]})
    pd.DataFrame(sat).to_csv(REPORTS_TABLES / "forecast_satellite.csv", index=False)
    # Neural networks: the Tier A - Tier B difference for every seed (seed variation is not in the bootstrap)
    per_seed = []
    for m in ["lstm", "gcn"]:
        for s_ in ["a", "b"]:
            for seed in seeds:
                a, b = f"{m}_{s_}_A_s{seed}", f"{m}_{s_}_B_s{seed}"
                if a not in df or b not in df:
                    continue
                for h, g in rows_a.groupby("h"):
                    per_seed.append({"model": f"{m}_{s_}", "seed": seed, "h": h,
                                     "d_r2": metrics(g["y"], g[a])["r2"] - metrics(g["y"], g[b])["r2"]})
    if per_seed:
        pd.DataFrame(per_seed).to_csv(REPORTS_TABLES / "forecast_satellite_seeds.csv", index=False)
    pd.DataFrame(groups).to_csv(REPORTS_TABLES / "forecast_satellite_groups.csv", index=False)

    # Share of all test rows that have a satellite day with AOD, by cloud bin (context for Tier A)
    d24 = df[df["h"] == 24]
    cov = d24.assign(cloud_bin=cloud_bin(d24["sat_cloud_frac"], cfg["cloud_bins"])) \
        .groupby("cloud_bin").agg(rows=("y", "size"), with_aod=("sat_aod_clear", lambda s: int(s.notna().sum())))
    cov.to_csv(REPORTS_TABLES / "forecast_aod_coverage.csv")

    # Seed range for the neural networks
    sr = []
    for c in [c for c in df.columns if c.startswith(("lstm_", "gcn_"))]:
        base = c.rsplit("_s", 1)[0]
        for h, g in df.groupby("h"):
            sr.append({"config": base, "seed": int(c.rsplit("_s", 1)[1]), "h": h, "r2": metrics(g["y"], g[c])["r2"]})
    if sr:
        sr = pd.DataFrame(sr).groupby(["config", "h"])["r2"].agg(["mean", "min", "max", "count"]).reset_index()
        sr.to_csv(REPORTS_TABLES / "forecast_seed_range.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, s in zip(axes, ["a", "b"]):
        for name in ["persistence", "climatology", f"lin_{s}", f"rf_{s}", f"lstm_{s}", f"gcn_{s}"]:
            g = skill[skill["model"] == name]
            if len(g):
                ax.plot(g["h"], g["r2"], marker="o", label=name.split("_")[0])
        ax.set_title({"a": "(a) past only", "b": "(b) + reanalysis weather ahead"}[s])
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_xticks(cfg["horizons"])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("R² (test year 2025, Tier B)")
    axes[0].legend()
    fig.tight_layout()
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_FIGURES / "forecast_r2_by_horizon.png", dpi=150)
    log.info("Skill (R²) by horizon:\n%s", skill.pivot(index="model", columns="h", values="r2").round(3).to_string())


if __name__ == "__main__":
    main()
