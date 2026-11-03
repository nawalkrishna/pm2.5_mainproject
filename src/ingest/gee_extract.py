"""Extract Earth Engine products at the ground stations, in monthly chunks (Stage 2, task 2-4).

Products and bands are listed in ``configs/gee_products.yaml``. Every chunk is saved
as its own CSV under ``data/raw/gee/<product>/`` and skipped if it already exists, so
the run can be stopped and resumed. Each file is recorded in the data manifest.

Chunks:
    daily_mean  one file per product, month and geometry (pt / b5 / b10). Days are IST
                calendar days: images whose start time falls in [00:00, 24:00) IST.
                Columns: station_id, date_ist, <layer>_mean, <layer>_count.
    hourly      one file per product and block of ``hourly_chunk_days`` days. Long format:
                station_id, time_utc, <band>... (value of the grid cell containing the station).
    static      one file per geometry.

Usage:
    python -m src.ingest.gee_extract [product ...] [--months 2021-01 2021-02] [--workers N]
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import pandas as pd

from src.ingest.ground_pm25 import STATIONS_PATH
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.manifest import record_file
from src.utils.paths import DATA_RAW

log = get_logger("gee_extract")

OUT_DIR: Path = DATA_RAW / "gee"
IST_OFFSET_H = 5.5


def init_ee() -> None:
    ee.Initialize(project=load_config("sources")["gee"]["project"])


def station_points(path: Path = STATIONS_PATH) -> pd.DataFrame:
    """Stations with coordinates (decision D021)."""
    st = pd.read_csv(path)
    return st.dropna(subset=["latitude", "longitude"])[["station_id", "latitude", "longitude"]]


def station_fc(stations: pd.DataFrame, radius_m: float) -> ee.FeatureCollection:
    feats = []
    for r in stations.itertuples():
        geom = ee.Geometry.Point([float(r.longitude), float(r.latitude)])
        if radius_m > 0:
            geom = geom.buffer(radius_m)
        feats.append(ee.Feature(geom, {"station_id": r.station_id}))
    return ee.FeatureCollection(feats)


def ist_days(month: str) -> list[str]:
    """All IST calendar dates in a month, as YYYY-MM-DD."""
    start = pd.Timestamp(f"{month}-01")
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, start + pd.offsets.MonthEnd(0), freq="D")]


def ist_day_window(date: str) -> tuple[ee.Date, ee.Date]:
    """UTC start and end of an IST calendar day."""
    start = ee.Date(date, "Asia/Kolkata")
    return start, start.advance(1, "day")


def empty_band(name: str) -> ee.Image:
    """A fully masked single-band image, used when a day has no observations."""
    return ee.Image.constant(0).rename(name).updateMask(ee.Image.constant(0))


def mean_or_empty(coll: ee.ImageCollection, name: str) -> ee.Image:
    """Per-pixel mean of a (possibly empty) collection as one band called ``name``.

    A fully masked placeholder is merged in so an empty day still yields the band
    (masked everywhere) without a server-side ``size()`` check per day, which made
    requests fail with "Too many concurrent aggregations".
    """
    renamed = coll.map(lambda im: im.rename(name))
    return renamed.merge(ee.ImageCollection([empty_band(name)])).mean().rename(name)


def tropomi_day(spec: dict, date: str) -> ee.Image:
    start, end = ist_day_window(date)
    bands = [mean_or_empty(ee.ImageCollection(l["collection"]).filterDate(start, end).select(l["band"]), name)
             for name, l in spec["layers"].items()]
    return ee.Image.cat(bands)


def maiac_mask(img: ee.Image, variant: str) -> ee.Image:
    """Apply one of the QA variants defined in gee_products.yaml to a MAIAC granule."""
    qa = img.select("AOD_QA")
    cloud_clear = qa.bitwiseAnd(0b111).eq(1)
    best = qa.rightShift(8).bitwiseAnd(0b1111).eq(0)
    aod = img.select("Optical_Depth_055").multiply(0.001)
    if variant == "aod_any":
        return aod
    if variant == "aod_clear":
        return aod.updateMask(cloud_clear)
    if variant == "aod_best":
        return aod.updateMask(cloud_clear.And(best))
    raise ValueError(variant)


def maiac_day(spec: dict, date: str) -> ee.Image:
    start, end = ist_day_window(date)
    coll = ee.ImageCollection(spec["collection"]).filterDate(start, end)
    return ee.Image.cat([mean_or_empty(coll.map(lambda im, v=v: maiac_mask(im, v)), v) for v in spec["variants"]])


DAY_BUILDERS = {"tropomi": tropomi_day, "maiac": maiac_day}


def native_projection(collection_id: str) -> tuple[str, list[float]]:
    """CRS and affine transform of the first image's first band (the product's native grid)."""
    proj = ee.ImageCollection(collection_id).first().select(0).projection().getInfo()
    return proj["crs"], proj["transform"]


def feature_rows(fc: ee.FeatureCollection) -> list[dict]:
    return [f["properties"] for f in fc.getInfo()["features"]]


def extract_daily(product: str, spec: dict, month: str, geom: str, radius: float,
                  stations: pd.DataFrame, proj: tuple[str, list[float]]) -> pd.DataFrame:
    """Mean and valid-pixel count per station and IST day for one month and geometry.

    All days of the month are stacked as bands (``<yyyymmdd>__<layer>``) so the whole
    month is one ``reduceRegions`` call returning one feature per station.
    """
    days = ist_days(month)
    stacked = ee.Image.cat([DAY_BUILDERS[product](spec, d).regexpRename("^", d.replace("-", "") + "__")
                            for d in days])
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
    crs, transform = proj
    rows = feature_rows(stacked.reduceRegions(collection=station_fc(stations, radius), reducer=reducer,
                                              crs=crs, crsTransform=transform, tileScale=4))
    layers = list(spec.get("layers") or spec["variants"])
    long = []
    for row in rows:
        for d in days:
            key = d.replace("-", "")
            rec = {"geometry": geom, "station_id": row["station_id"], "date_ist": d}
            for layer in layers:
                rec[f"{layer}_mean"] = row.get(f"{key}__{layer}_mean")
                rec[f"{layer}_count"] = row.get(f"{key}__{layer}_count")
            long.append(rec)
    return pd.DataFrame(long)


def grid_transform(transform_wkt: str) -> list[float]:
    """Affine transform [a, b, c, d, e, f] from an Earth Engine projection transform in WKT form."""
    vals = dict(re.findall(r'"elt_(\d_\d)",\s*(-?[\d.eE+-]+)', transform_wkt))
    get = lambda k, default: float(vals.get(k, default))
    return [get("0_0", 1), get("0_1", 0), get("0_2", 0), get("1_0", 0), get("1_1", 1), get("1_2", 0)]


def fix_lon_origin(img: ee.Image, transform: list[float]) -> ee.Image:
    """Re-label a grid stored with x origin < -180 so it lines up with -180..180 longitudes.

    ERA5-Land images in Earth Engine from mid-January 2024 carry the grid origin -360.05
    instead of -180.05. ``reduceRegion`` on a single point still finds the pixel, but
    ``reduceRegions`` over the station collection returns nulls. Shifting the origin by
    +360 degrees (same longitudes) gives values identical to ``reduceRegion``
    (checked at three stations, 2026-11-03).
    """
    if transform[2] >= -180.1:
        return img
    proj = img.select(0).projection()
    shifted = transform[:2] + [transform[2] + 360] + transform[3:]
    return img.changeProj(proj, ee.Projection(proj.crs(), shifted))


def extract_hourly(spec: dict, start: str, end: str, stations: pd.DataFrame,
                   proj: tuple[str, list[float]]) -> pd.DataFrame:
    """Value of each band at each station for every hour in [start, end) UTC.

    The hours are stacked as bands (``toBands``) so one request returns one feature per
    station, which stays under Earth Engine's 5000-element limit for ``getInfo``.
    """
    coll = ee.ImageCollection(spec["collection"]).filterDate(start, end).select(spec["bands"])
    coll = coll.map(lambda im: im.set("x0", im.select(0).projection().transform()))
    meta = ee.Dictionary({"index": coll.aggregate_array("system:index"),
                          "time": coll.aggregate_array("system:time_start"),
                          "x0": coll.aggregate_array("x0")}).getInfo()
    if not meta["index"]:
        return pd.DataFrame(columns=["station_id", "time_utc", *spec["bands"]])
    crs, transform = proj
    images = coll.toList(len(meta["index"]))
    parts = []
    for i, (idx, wkt) in enumerate(zip(meta["index"], meta["x0"])):
        im = fix_lon_origin(ee.Image(images.get(i)), grid_transform(wkt))
        parts.append(im.rename([f"{idx}_{b}" for b in spec["bands"]]))
    stacked = ee.Image.cat(parts)
    rows = feature_rows(stacked.reduceRegions(collection=station_fc(stations, 0), reducer=ee.Reducer.first(),
                                              crs=crs, crsTransform=transform))
    times = dict(zip(meta["index"], pd.to_datetime(meta["time"], unit="ms", utc=True)))
    long = []
    for row in rows:
        for idx, t in times.items():
            long.append({"station_id": row["station_id"], "time_utc": t,
                         **{b: row.get(f"{idx}_{b}") for b in spec["bands"]}})
    return pd.DataFrame(long)


def extract_static(spec: dict, geom: str, radius: float, stations: pd.DataFrame) -> pd.DataFrame:
    fc = station_fc(stations, radius)
    out = pd.DataFrame({"station_id": stations["station_id"].to_numpy()})
    for name, layer in spec["layers"].items():
        if layer["reducer"] == "mean":
            img = ee.Image(layer["image"]).select(layer["band"]).rename(name)
            red = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
            res = pd.DataFrame(feature_rows(img.reduceRegions(fc, red, scale=layer["scale"], tileScale=4)))
            out = out.merge(res[["station_id", "mean", "count"]].rename(
                columns={"mean": f"{name}_mean", "count": f"{name}_count"}), on="station_id")
        elif layer["reducer"] == "histogram":
            img = ee.ImageCollection(layer["collection"]).first().select(layer["band"]).rename(name)
            res = pd.DataFrame(feature_rows(img.reduceRegions(fc, ee.Reducer.frequencyHistogram(),
                                                              scale=layer["scale"], tileScale=4)))
            hist = pd.json_normalize(res["histogram"]).fillna(0)
            frac = hist.div(hist.sum(axis=1), axis=0).add_prefix(f"{name}_frac_")
            frac.insert(0, "station_id", res["station_id"])
            out = out.merge(frac, on="station_id")
        elif layer["reducer"] == "sum":
            coll = ee.ImageCollection(layer["collection"]).filterBounds(fc.geometry())
            for year in layer["years"]:
                img = coll.filter(ee.Filter.eq("year", year)).mosaic().select(layer["band"]).rename(name)
                res = pd.DataFrame(feature_rows(img.reduceRegions(fc, ee.Reducer.sum(), scale=layer["scale"],
                                                                  tileScale=4)))
                out = out.merge(res[["station_id", "sum"]].rename(columns={"sum": f"{name}_{year}"}),
                                on="station_id")
        else:
            raise ValueError(layer["reducer"])
    out.insert(1, "geometry", geom)
    return out


def hourly_blocks(start: str, end: str, days: int) -> list[tuple[str, str]]:
    """[start, end) UTC blocks covering the IST extraction window, padded by one day each side."""
    t0 = pd.Timestamp(start) - pd.Timedelta(days=1)
    t1 = pd.Timestamp(end) + pd.Timedelta(days=2)
    edges = list(pd.date_range(t0, t1, freq=f"{days}D")) + [t1]
    return [(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")) for a, b in zip(edges[:-1], edges[1:]) if a < b]


def save(df: pd.DataFrame, path: Path, product: str, date_range: str, notes: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)                       # atomic: a half-written file never looks finished
    record_file(path, source=f"Google Earth Engine: {product}", date_range=date_range, rows=len(df), notes=notes)


def plan_tasks(products: list[str], cfg: dict, months: list[str], window: dict) -> list[tuple]:
    """(product, kind, key, output path) for every chunk not yet on disk."""
    tasks = []
    for product in products:
        spec = cfg["products"][product]
        pdir = OUT_DIR / product
        if spec["kind"] == "daily_mean":
            for month in months:
                for geom in spec["geometries"]:
                    tasks.append((product, "daily_mean", (month, geom), pdir / f"{product}_{month}_{geom}.csv"))
        elif spec["kind"] == "hourly":
            for a, b in hourly_blocks(window["start"], window["end"], cfg["hourly_chunk_days"]):
                tasks.append((product, "hourly", (a, b), pdir / f"{product}_{a}_{b}.csv"))
        elif spec["kind"] == "static":
            for geom in spec["geometries"]:
                tasks.append((product, "static", (geom,), pdir / f"{product}_{geom}.csv"))
    return [t for t in tasks if not t[3].exists()]


def run_task(task: tuple, cfg: dict, stations: pd.DataFrame, projections: dict) -> str:
    product, kind, key, path = task
    spec = cfg["products"][product]
    if kind == "daily_mean":
        month, geom = key
        df = extract_daily(product, spec, month, geom, cfg["geometries"][geom], stations, projections[product])
        notes = "; ".join(f"{k}: {v.get('prefilter', '')}" for k, v in spec.get("layers", {}).items()) \
            or "; ".join(f"{k}: {v}" for k, v in spec["variants"].items())
        save(df, path, product, month, f"{geom}; {notes}")
    elif kind == "hourly":
        a, b = key
        df = extract_hourly(spec, a, b, stations, projections[product])
        save(df, path, product, f"{a}..{b} UTC", "bands: " + ", ".join(spec["bands"]))
    else:
        (geom,) = key
        df = extract_static(spec, geom, cfg["geometries"][geom], stations)
        save(df, path, product, "static", geom)
    return f"{path.name}: {len(df)} rows"


def main(products: list[str] | None = None, months: list[str] | None = None, workers: int | None = None) -> int:
    init_ee()
    cfg = load_config("gee_products")
    window = load_config("study_area")["extraction_period"]
    products = products or list(cfg["products"])
    months = months or [m.strftime("%Y-%m") for m in pd.period_range(window["start"], window["end"], freq="M")]
    stations = station_points()
    log.info("Stations with coordinates: %d", len(stations))

    projections = {}
    for p in products:
        spec = cfg["products"][p]
        if spec["kind"] != "static":
            projections[p] = native_projection(spec.get("projection_from") or spec["collection"])
            log.info("%s native grid: %s %s", p, *projections[p])

    tasks = plan_tasks(products, cfg, months, window)
    log.info("Chunks to extract: %d", len(tasks))
    failed = 0
    with ThreadPoolExecutor(max_workers=workers or cfg["max_workers"]) as pool:
        futures = {pool.submit(run_task, t, cfg, stations, projections): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            task = futures[fut]
            try:
                log.info("[%d/%d] %s", i, len(tasks), fut.result())
            except Exception as exc:                       # keep going; rerun picks up failures
                failed += 1
                log.error("[%d/%d] FAILED %s: %s", i, len(tasks), task[3].name, str(exc)[:300])
    log.info("Done. Failed chunks: %d (rerun to retry)", failed)
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("products", nargs="*")
    parser.add_argument("--months", nargs="*")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    main(args.products or None, args.months or None, args.workers)
