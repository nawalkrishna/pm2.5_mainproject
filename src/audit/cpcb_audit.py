"""Completeness of CPCB CCR hourly files per station: the decisive Stage 1 numbers.

Inputs:  data/interim/cpcb_daily.csv, data/interim/cpcb_files.csv, data/interim/cpcb_stations.csv
Outputs (reports/tables/):
    cpcb_site_inventory.csv         one row per CPCB station with completeness
    cpcb_completeness_by_year.csv   per station and year
    cpcb_completeness_by_season.csv per station and season (study period)
    cpcb_usable_by_year.csv         per state and year: stations meeting the threshold
    cpcb_usable_by_threshold.csv    usable stations per state for several thresholds
    cpcb_usable_by_period.csv       usable stations per state for candidate study periods
    cpcb_vs_openaq_agreement.csv    CPCB vs OpenAQ daily values for name-matched stations
Figure: reports/figures/cpcb_station_map.png

Usage:
    python -m src.audit.cpcb_audit
"""

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.clean.completeness import completeness
from src.ingest.boundaries import GEOJSON_NAME, OUT_DIR as BOUNDARY_DIR
from src.ingest.cpcb_hourly import DAILY_PATH, FILES_PATH
from src.ingest.cpcb_stations import OUT_PATH as STATIONS_PATH, normalise_name
from src.utils.config import config_hash, load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import REPORTS_FIGURES, REPORTS_TABLES
from src.utils.run_log import log_run

log = get_logger("cpcb_audit")

THRESHOLD_OPTIONS = [0.5, 0.6, 0.7, 0.75]


def link_sites_to_stations(files: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Map CPCB site numbers (from file names) to the station list via the name slug.

    Slugs drop some separators (``RVCE-Mailasandra`` -> ``rvcemailasandra``), so both
    sides are compared with all spaces removed.
    """
    sites = files.drop_duplicates("cpcb_site")[["cpcb_site", "slug"]].copy()
    sites["key"] = sites["slug"].map(lambda s: normalise_name(s).replace(" ", ""))
    st = stations.copy()
    st["key"] = st["station"].map(lambda s: normalise_name(s).replace(" ", ""))
    out = sites.merge(st, on="key", how="left").drop(columns="key")
    unmatched = out[out["station"].isna()]
    if not unmatched.empty:
        log.warning("CPCB files with no station-list match: %s", unmatched["slug"].tolist())
    return out


def usable_by_period(daily: pd.DataFrame, sites: pd.DataFrame, periods: list[tuple[str, str]],
                     seasons: dict, threshold: float) -> pd.DataFrame:
    """Usable stations per state for several candidate study periods."""
    rows = []
    for start, end in periods:
        overall = completeness(daily, "cpcb_site", start, end, seasons)["overall"]
        overall = overall.merge(sites[["cpcb_site", "state"]], on="cpcb_site")
        usable = overall[overall["completeness"] >= threshold].groupby("state").size()
        rows.append({"period": f"{start[:4]}-{end[:4]}", "all": int(usable.sum()), **usable.to_dict()})
    return pd.DataFrame(rows).fillna(0)


def agreement_with_openaq(daily: pd.DataFrame, sites: pd.DataFrame, min_hours: int,
                          min_shared_days: int = 30) -> pd.DataFrame:
    """Compare CPCB daily means with OpenAQ daily means for name-matched stations.

    Only days valid in CPCB (>= ``min_hours``) are used. OpenAQ values for a site are the
    mean over all its location IDs on that day.
    """
    from src.ingest.openaq_daily import DAILY_PATH as OPENAQ_DAILY
    from src.ingest.openaq_inventory import OUT_PATH as OPENAQ_SENSORS

    od = pd.read_csv(OPENAQ_DAILY)
    sensors = pd.read_csv(OPENAQ_SENSORS)
    cd = daily[daily["observed_count"] >= min_hours]
    rows = []
    for r in sites.dropna(subset=["openaq_location_ids"]).itertuples():
        loc_ids = [int(i) for i in str(r.openaq_location_ids).split(";")]
        sensor_ids = sensors.loc[sensors["location_id"].isin(loc_ids), "sensor_id"]
        o = od[od["sensor_id"].isin(sensor_ids)].groupby("date_local")["pm25_daily_mean"].mean()
        c = cd[cd["cpcb_site"] == r.cpcb_site].set_index("date_local")["pm25_daily_mean"]
        both = pd.concat([c.rename("cpcb"), o.rename("openaq")], axis=1, join="inner").dropna()
        if len(both) < min_shared_days:
            continue
        diff = (both["cpcb"] - both["openaq"]).abs()
        rows.append({"cpcb_site": r.cpcb_site, "station": r.station, "shared_days": len(both),
                     "pearson_r": both["cpcb"].corr(both["openaq"]), "median_abs_diff": diff.median(),
                     "share_within_1": (diff <= 1).mean(), "share_diff_gt_10": (diff > 10).mean()})
    return pd.DataFrame(rows)


def plot_map(sites: pd.DataFrame, area: dict, path) -> None:
    """CPCB stations; filled = usable over the study period; grey x = no coordinates known."""
    states = gpd.read_file(BOUNDARY_DIR / GEOJSON_NAME)
    states["NAME_1"] = states["NAME_1"].replace(area["gadm_state_names"])
    fig, ax = plt.subplots(figsize=(8, 9))
    states[states["NAME_1"].isin(area["states"])].boundary.plot(ax=ax, color="0.4", linewidth=0.6)
    has_xy = sites["latitude"].notna()
    for usable, face in ((True, "tab:blue"), (False, "none")):
        sub = sites[has_xy & (sites["usable_study_period"] == usable)]
        ax.scatter(sub["longitude"], sub["latitude"], s=30, facecolors=face, edgecolors="tab:blue",
                   linewidths=0.9, label=f"{'usable' if usable else 'not usable'} (n={len(sub)})")
    n_noxy = int((~has_xy).sum())
    ax.set_title(f"CPCB CAAQMS stations, filled = usable in study period\n"
                 f"({n_noxy} stations without coordinates not shown)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> pd.DataFrame:
    """Build the CPCB completeness tables. Returns the per-station table."""
    area = load_config("study_area")
    audit = load_config("audit")
    threshold = audit["min_daily_completeness"]

    daily = pd.read_csv(DAILY_PATH)
    daily["valid"] = (daily["observed_count"] >= audit["min_hours_per_valid_day"]) & daily["pm25_daily_mean"].notna()
    sites = link_sites_to_stations(pd.read_csv(FILES_PATH), pd.read_csv(STATIONS_PATH))

    windows = {"audit": area["audit_period"], "study": area["study_period"]}
    comp = {n: completeness(daily, "cpcb_site", w["start"], w["end"], audit["seasons"]) for n, w in windows.items()}
    for name in windows:
        overall = comp[name]["overall"].rename(columns={"valid_days": f"valid_days_{name}",
                                                        "completeness": f"completeness_{name}"})
        sites = sites.merge(overall.drop(columns="calendar_days"), on="cpcb_site", how="left")
    first_last = daily[daily["valid"]].groupby("cpcb_site")["date_local"].agg(first_valid_day="min",
                                                                              last_valid_day="max")
    sites = sites.merge(first_last, on="cpcb_site", how="left")
    sites["usable_study_period"] = sites["completeness_study"] >= threshold
    sites["usable_sensitivity"] = sites["completeness_study"] >= audit["sensitivity_daily_completeness"]

    meta = sites[["cpcb_site", "state", "city", "station"]]
    by_year = comp["audit"]["by_year"].merge(meta, on="cpcb_site")
    by_year["usable"] = by_year["completeness"] >= threshold
    usable_by_year = (by_year[by_year["usable"]].groupby(["state", "year"]).size()
                      .unstack(fill_value=0).reindex(area["states"], fill_value=0))
    usable_by_year = usable_by_year.reindex(columns=sorted(by_year["year"].unique()), fill_value=0)
    usable_by_year.loc["ALL"] = usable_by_year.sum()

    by_threshold = pd.DataFrame({
        f"usable_at_{int(t * 100)}pct": sites[sites["completeness_study"] >= t].groupby("state").size()
        for t in THRESHOLD_OPTIONS}).reindex(area["states"]).fillna(0).astype(int)
    by_threshold.loc["ALL"] = by_threshold.sum()

    sites.to_csv(REPORTS_TABLES / "cpcb_site_inventory.csv", index=False)
    by_year.to_csv(REPORTS_TABLES / "cpcb_completeness_by_year.csv", index=False)
    comp["study"]["by_season"].merge(meta, on="cpcb_site").to_csv(
        REPORTS_TABLES / "cpcb_completeness_by_season.csv", index=False)
    usable_by_year.to_csv(REPORTS_TABLES / "cpcb_usable_by_year.csv")
    by_threshold.to_csv(REPORTS_TABLES / "cpcb_usable_by_threshold.csv")
    plot_map(sites, area, REPORTS_FIGURES / "cpcb_station_map.png")

    periods = [("2019-01-01", "2022-12-31"), ("2020-01-01", "2023-12-31"), ("2021-01-01", "2024-12-31"),
               ("2021-01-01", "2025-12-31"), ("2022-01-01", "2025-12-31"), ("2023-01-01", "2025-12-31")]
    usable_by_period(daily, sites, periods, audit["seasons"], threshold).to_csv(
        REPORTS_TABLES / "cpcb_usable_by_period.csv", index=False)
    agreement_with_openaq(daily, sites, audit["min_hours_per_valid_day"]).to_csv(
        REPORTS_TABLES / "cpcb_vs_openaq_agreement.csv", index=False)

    log.info("CPCB usable stations by year (>= %.0f%%):\n%s", threshold * 100, usable_by_year.to_string())
    log.info("CPCB usable stations over study period, by threshold:\n%s", by_threshold.to_string())
    log_run("cpcb_audit", config_hash({"area": area, "audit": audit}),
            metrics=by_threshold[f"usable_at_{int(threshold * 100)}pct"].to_dict())
    return sites


if __name__ == "__main__":
    main()
