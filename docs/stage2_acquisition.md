# Stage 2: data acquisition

**Status: COMPLETE, awaiting sign-off.** Station-point extraction only (task 3a); the grid (3b)
waits for approval (D026). Every number below comes from a file in `reports/tables/`.
Regenerate with `python -m src.run stage2`, then `python -m src.audit.gee_inventory` and
`python -m src.audit.manifest_check`.

## 1. What was acquired

| Data | Script | Raw files | Notes |
|---|---|---|---|
| CPCB CCR hourly PM2.5 | `src/ingest/cpcb_hourly.py` (Stage 1) | 833 station-year CSVs | Manual browser download (CAPTCHA); expected names `Raw_Data_<year>_site_<id>_<station>_1Hr.csv`, see D008 |
| OpenAQ hourly, US Consulate Hyderabad and Chennai | `src/ingest/ground_pm25.py` | 122 sensor-month JSON | Reference monitors not in CPCB (D015) |
| OpenAQ hourly, timestamp check | `src/audit/cpcb_timestamp_check.py` | 10 JSON | D024 |
| Sentinel-5P L3 (NO2, CO, SO2, HCHO, AER_AI, CLOUD) | `src/ingest/gee_extract.py` | 183 (61 months × 3 geometries) | Daily IST mean, pt / 5 km / 10 km, with valid-pixel counts |
| MODIS MAIAC AOD 550 nm | same | 183 | Three QA variants: any / clear / best |
| ERA5-Land hourly | same | 233 (8-day blocks) | T2m, Td2m, u10, v10, surface pressure, hourly precipitation |
| ERA5 hourly | same | 233 | Boundary-layer height (+ fallback meteorology) |
| MERRA-2 aerosol hourly | same | 233 | Species for reconstructed PM2.5 and total AOD |
| Static: SRTM, ESA WorldCover 2021, WorldPop 2021–2025 | same | 3 (pt / 5 km / 10 km) | |

Extraction points: 117 = 115 CPCB stations with coordinates + 2 US Consulate monitors (D021).
Window: 2020-12-01 to 2025-12-31 IST (D023). Raw Earth Engine output: 2.3 GB, 1,068 chunk files.
The manifest (`reports/tables/data_manifest.csv`, 3,081 files) records source, date range,
rows, bytes, SHA-256 and retrieval time for every raw file; `manifest_check` confirms it
matches `data/raw` exactly.

## 2. Ground PM2.5 in one schema (task 1)

`data/interim/ground_pm25_hourly.parquet`: `station_id, timestamp_utc, timestamp_ist, pm25,
unit, source, qc_flag`. Timestamps are the start of the hour. Row reconciliation
(`reports/tables/ground_removal_log.csv`):

| Source | Hours in window | Empty (no value) | Non-numeric | Kept |
|---|---|---|---|---|
| CPCB CCR 1 h | 4,697,112 | 1,780,868 | 0 | 2,916,244 |
| OpenAQ (US Consulates) | 92,441 | 13,095 | 0 | 79,346 |
| **Total** | | | | **2,995,590** (113 stations) |

**CPCB timestamp convention (D024):** labels are IST hour-start. For 5 stations (one per
state) and two weeks of 2025, aligning CPCB with OpenAQ's explicit UTC times gives the best
match at offset 0 at every station: median r 0.986, against 0.944 for hour-end labels
(`cpcb_timestamp_check.csv`). A unit test checks the conversion
(`2025-01-01 00:00 IST` → `2024-12-31 18:30 UTC`).

## 3. Satellite availability (task 4)

Share of station-days with a value, 2021–2025 (`gee_availability.csv`,
`gee_availability_kerala_jjas.csv`). pt = station pixel, b5/b10 = 5/10 km buffer.

| Layer | All stations, pt | b5 | b10 | Kerala Jun–Sep, pt | b5 | b10 |
|---|---|---|---|---|---|---|
| NO2 (tropospheric) | 0.53 | 0.63 | 0.69 | 0.37 | 0.47 | 0.54 |
| CO | 0.60 | 0.72 | 0.78 | 0.42 | 0.52 | 0.58 |
| SO2 | 0.52 | 0.61 | 0.67 | 0.40 | 0.49 | 0.55 |
| HCHO | 0.62 | 0.71 | 0.76 | 0.51 | 0.60 | 0.66 |
| UV aerosol index | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| MAIAC AOD, clear | 0.31 | 0.48 | 0.54 | 0.11 | 0.28 | 0.36 |
| MAIAC AOD, best quality | 0.25 | 0.41 | 0.47 | 0.06 | 0.19 | 0.24 |

Reanalysis: ERA5, MERRA-2 ≈ 100 % of station-hours (MERRA-2 except 6 missing days); ERA5-Land
94 % (7 coastal stations have no land cell; ERA5 covers them).

Reading: during the Kerala SW monsoon, MAIAC AOD exists at the station pixel on about 1 day in
9. A 10 km buffer triples that but is still about 1 day in 3. TROPOMI gases fare better. The
UV aerosol index is retrieved under cloud and is always present. "Any" and "clear" AOD are
almost identical: MAIAC retrieves AOD only for clear pixels.

## 4. Problems found and fixed

1. **Manifest corruption.** Parallel extraction workers rewrote the manifest at the same time.
   Fixed with a lock and atomic replace in `src/utils/manifest.py`; repaired and verified with
   the new `src/audit/manifest_check.py`. Rows that had to be re-created say so in `notes`.
2. **ERA5-Land nulls from mid-January 2024.** Earth Engine images from then on carry the grid
   origin −360.05° instead of −180.05°, so `reduceRegions` returned nulls. Fixed by shifting the
   origin by +360° (`fix_lon_origin`); values identical to `reduceRegion` at three stations.
   The affected 90 chunks were re-extracted. Test added for the transform parser.
3. **OpenAQ deep pagination timed out** (HTTP 408). Now requested per month.
4. **Earth Engine "too many concurrent aggregations".** Days are now stacked as bands, one
   `reduceRegions` call per month and geometry.

## 5. Known issues carried to Stage 3/4

- **TROPOMI QA (D027, needs decision):** Earth Engine L3 is pre-filtered (NO2 qa ≥ 0.75,
  AER_AI ≥ 0.8, others ≥ 0.5) and has no `qa_value` band. Stage 4 can only tighten with cloud fraction.
- **MERRA-2 gap:** 6 days missing in Earth Engine (2025-09-04, 05, 20, 23, 26, 28).
- **PM2.5 values:** CPCB maximum is exactly 1000 µg/m³ (2 hours, possibly an instrument ceiling); 17 negative values (US Consulates); 58 zeros.
- **Hour alignment:** CPCB hours start at :30 UTC (00:00 IST), US Consulate hours at :00 UTC.
- **Bollaram coordinates:** CPCB dashboard and OpenAQ differ by 3.6 km (D017).
- **Not yet extracted:** road density (GRIP4, needs download approval, D025); distance to coast
  (computed locally in Stage 4); the regular grid (D026).
