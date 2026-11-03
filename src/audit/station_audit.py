"""Stage 1 station audit: inventory, completeness, duplicates, map and summary numbers.

Inputs (produced by src.ingest.*):
    data/interim/openaq_sensors.csv, data/interim/openaq_daily.csv,
    data/interim/cpcb_stations.csv, data/raw/boundaries/gadm41_IND_1.json

Completeness is reported for each OpenAQ *site*: sensors with the same
normalised name within ``duplicate_distance_m`` are one site (OpenAQ often
lists one monitor under an old and a new location ID). A site-day is valid if
any of its sensors has a valid day.

Outputs:
    reports/tables/station_inventory.csv        one row per OpenAQ sensor (with site_id)
    reports/tables/site_inventory.csv           one row per site, with completeness
    reports/tables/completeness_by_year.csv     per site and year (audit window)
    reports/tables/completeness_by_season.csv   per site and season (study period)
    reports/tables/usable_sites_by_year.csv     per state and year: sites meeting the threshold
    reports/tables/duplicate_candidates.csv     nearby sensor pairs with overlap statistics
    reports/tables/cpcb_station_list.csv        CPCB CCR stations with OpenAQ match status
    reports/tables/audit_summary.csv            headline counts quoted in docs/station_audit.md
    reports/figures/station_map.png

Usage:
    python -m src.audit.station_audit
"""

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")   # write PNGs without needing a display
import matplotlib.pyplot as plt
import pandas as pd

from src.align.spatial import assign_state, group_sites, nearby_pairs
from src.clean.completeness import completeness, mark_valid_days
from src.ingest.boundaries import OUT_DIR as BOUNDARY_DIR, GEOJSON_NAME
from src.ingest.cpcb_stations import OUT_PATH as CPCB_PATH, normalise_name
from src.ingest.openaq_daily import DAILY_PATH
from src.ingest.openaq_inventory import OUT_PATH as SENSORS_PATH
from src.utils.config import config_hash, load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import REPORTS_FIGURES, REPORTS_TABLES
from src.utils.run_log import log_run

log = get_logger("station_audit")

STATE_COL = "NAME_1"   # GADM level-1 name field


def overlap_stats(daily: pd.DataFrame, a: int, b: int) -> dict[str, float]:
    """Compare two sensors on the days both have valid data."""
    da = daily[(daily["sensor_id"] == a) & daily["valid"]].set_index("date_local")["pm25_daily_mean"]
    db = daily[(daily["sensor_id"] == b) & daily["valid"]].set_index("date_local")["pm25_daily_mean"]
    both = pd.concat([da, db], axis=1, join="inner").dropna()
    if len(both) < 2:
        return {"overlap_days": len(both), "pearson_r": float("nan"), "median_abs_diff": float("nan")}
    return {
        "overlap_days": len(both),
        "pearson_r": float(both.iloc[:, 0].corr(both.iloc[:, 1])),
        "median_abs_diff": float((both.iloc[:, 0] - both.iloc[:, 1]).abs().median()),
    }


def plot_map(sites: pd.DataFrame, states: gpd.GeoDataFrame, focus: list[str], path) -> None:
    """Site locations over the study states; filled marker = usable over the study period."""
    fig, ax = plt.subplots(figsize=(8, 9))
    states[states[STATE_COL].isin(focus)].boundary.plot(ax=ax, color="0.4", linewidth=0.6)
    styles = {"reference": ("o", "tab:blue"), "low_cost": ("^", "tab:orange")}
    for grade, (marker, colour) in styles.items():
        for usable in (True, False):
            sub = sites[(sites["grade"] == grade) & (sites["usable_study_period"] == usable)]
            if sub.empty:
                continue
            ax.scatter(sub["longitude"], sub["latitude"], marker=marker, s=28,
                       facecolors=colour if usable else "none", edgecolors=colour, linewidths=0.9,
                       label=f"{grade}, {'usable' if usable else 'not usable'} (n={len(sub)})")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("OpenAQ PM2.5 sites, filled = meets completeness threshold in study period")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    area = load_config("study_area")
    audit = load_config("audit")
    REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
    threshold = audit["min_daily_completeness"]

    sensors = pd.read_csv(SENSORS_PATH)
    daily = mark_valid_days(pd.read_csv(DAILY_PATH), audit["min_hours_per_valid_day"])
    states = gpd.read_file(BOUNDARY_DIR / GEOJSON_NAME)
    # GADM spells some names without spaces; translate to the project's names.
    states[STATE_COL] = states[STATE_COL].replace(area["gadm_state_names"])

    # --- State assignment: keep only the study states -------------------------------
    sensors["state"] = assign_state(sensors, states, STATE_COL)
    n_all = len(sensors)
    sensors = sensors[sensors["state"].isin(area["states"])].reset_index(drop=True)
    log.info("%d of %d sensors fall in the study states", len(sensors), n_all)

    # --- Sites: one monitor listed under several OpenAQ IDs counts once -------------
    sensors["site_id"] = group_sites(sensors, sensors["name"].map(normalise_name),
                                     audit["duplicate_distance_m"])
    daily = daily[daily["sensor_id"].isin(sensors["sensor_id"])]
    daily = daily.merge(sensors[["sensor_id", "site_id"]], on="sensor_id")

    windows = {"audit": area["audit_period"], "study": area["study_period"]}
    comp = {name: completeness(daily, "site_id", w["start"], w["end"], audit["seasons"])
            for name, w in windows.items()}

    sites = (sensors.sort_values("last_utc", na_position="first")
             .groupby("site_id")
             .agg(name=("name", "last"), state=("state", "first"), locality=("locality", "last"),
                  latitude=("latitude", "last"), longitude=("longitude", "last"),
                  provider=("provider", "last"),
                  grade=("grade", lambda g: "reference" if (g == "reference").any() else "low_cost"),
                  n_sensors=("sensor_id", "nunique"),
                  sensor_ids=("sensor_id", lambda s: ";".join(map(str, sorted(s)))))
             .reset_index())
    valid = daily[daily["valid"]]
    first_last = valid.groupby("site_id")["date_local"].agg(first_valid_day="min", last_valid_day="max")
    sites = sites.merge(first_last, on="site_id", how="left")
    for name in windows:
        overall = comp[name]["overall"].rename(columns={"valid_days": f"valid_days_{name}",
                                                        "completeness": f"completeness_{name}"})
        sites = sites.merge(overall.drop(columns="calendar_days"), on="site_id", how="left")
    for col in [c for c in sites.columns if c.startswith(("valid_days_", "completeness_"))]:
        sites[col] = sites[col].fillna(0)
    sites["usable_study_period"] = sites["completeness_study"] >= threshold

    # --- Year-by-year usable sites (shows gaps such as 2023-2024) ---------------------
    meta = sites[["site_id", "state", "name", "grade"]]
    by_year = comp["audit"]["by_year"].merge(meta, on="site_id")
    by_year["usable"] = by_year["completeness"] >= threshold
    usable_by_year = (by_year[by_year["usable"]].groupby(["state", "year"]).size()
                      .unstack(fill_value=0).reindex(area["states"], fill_value=0))
    usable_by_year = usable_by_year.reindex(columns=sorted(by_year["year"].unique()), fill_value=0)
    usable_by_year.loc["ALL"] = usable_by_year.sum()

    # --- Duplicate candidates: nearby sensors, with value agreement ------------------
    pairs = nearby_pairs(sensors, "sensor_id", audit["duplicate_distance_m"])
    if not pairs.empty:
        stats = pd.DataFrame([overlap_stats(daily, a, b)
                              for a, b in zip(pairs["sensor_id_a"], pairs["sensor_id_b"])])
        pairs = pd.concat([pairs, stats], axis=1)

    # --- CPCB list ---------------------------------------------------------------------
    cpcb = pd.read_csv(CPCB_PATH)

    # --- Write tables ----------------------------------------------------------------
    sensors.to_csv(REPORTS_TABLES / "station_inventory.csv", index=False)
    sites.to_csv(REPORTS_TABLES / "site_inventory.csv", index=False)
    by_year.to_csv(REPORTS_TABLES / "completeness_by_year.csv", index=False)
    comp["study"]["by_season"].merge(meta, on="site_id").to_csv(
        REPORTS_TABLES / "completeness_by_season.csv", index=False)
    usable_by_year.to_csv(REPORTS_TABLES / "usable_sites_by_year.csv")
    pairs.to_csv(REPORTS_TABLES / "duplicate_candidates.csv", index=False)
    cpcb.to_csv(REPORTS_TABLES / "cpcb_station_list.csv", index=False)

    rows = []
    for state in area["states"]:
        s = sites[sites["state"] == state]
        rows.append({
            "state": state,
            "cpcb_ccr_stations": int((cpcb["state"] == state).sum()),
            "openaq_sensors": int((sensors["state"] == state).sum()),
            "openaq_sites": len(s),
            "openaq_reference_sites": int((s["grade"] == "reference").sum()),
            "openaq_low_cost_sites": int((s["grade"] == "low_cost").sum()),
            "openaq_usable_sites_study_period": int(s["usable_study_period"].sum()),
            "openaq_usable_reference_sites_study_period":
                int((s["usable_study_period"] & (s["grade"] == "reference")).sum()),
        })
    summary = pd.DataFrame(rows)
    summary.loc[len(summary)] = {"state": "ALL", **summary.drop(columns="state").sum().to_dict()}
    summary.to_csv(REPORTS_TABLES / "audit_summary.csv", index=False)

    plot_map(sites, states, area["states"], REPORTS_FIGURES / "station_map.png")

    log.info("\n%s", summary.to_string(index=False))
    log.info("Usable sites by year (completeness >= %.0f%%):\n%s", threshold * 100, usable_by_year.to_string())
    log_run("station_audit", config_hash({"area": area, "audit": audit}),
            metrics=summary.set_index("state")["openaq_usable_sites_study_period"].to_dict())


if __name__ == "__main__":
    main()
