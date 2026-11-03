# Station audit (Stage 1)

**Status: COMPLETE, signed off 2026-11-03** (decisions D010–D018 in `docs/decision_log.md`). Every number below is copied from a file in
`reports/tables/` produced on 2026-11-03 by `python -m src.run stage1` (OpenAQ, CPCB station
list) plus the CPCB hourly files (see §2). Approved thresholds (`configs/audit.yaml`): valid day = at least 18 hours with PM2.5; usable
station = at least 70 % of calendar days valid (50 % for the sensitivity set). **Study period: 2021–2025** (D011).

## 1. Headline findings

1. **CPCB CCR hourly files are the only usable ground source for 2021–2025.** Over 2021–2025,
   **42 CPCB stations** meet the 70 % threshold (56 at 50 %) (`cpcb_usable_by_threshold.csv`). On OpenAQ,
   **0 stations** do (`audit_summary.csv`).
2. **Kerala has 9 CPCB stations, of which 6 are usable over 2021–2025.** That is below the 15–20 needed for
   a Kerala-only study, so **the study area should be South India (five states), with Kerala as the
   monsoon focus region.**
3. **Monitoring is uneven.** Karnataka has 19 of the 42 usable stations; Andhra Pradesh has 4. All
   Telangana stations are in Hyderabad.
4. **Monsoon months are the weakest for several Kerala stations.** For example, Plammoodu has 64 % of
   SW-monsoon days valid against 90 % in winter (`cpcb_completeness_by_season.csv`).

## 2. Sources and how they were obtained

| Source | How | Files | Notes |
|---|---|---|---|
| CPCB CCR Data Repository, hourly (1H) | Browser session opened by the student (portal requires a CAPTCHA); downloads triggered from the page itself, 3–5 s apart | 833 station-year CSVs, 117 stations, 2018–2025, in `data/raw/cpcb/1Hr/<year>/` | Download log: `data/raw/cpcb/ccr_bulk_download_log_20261103.txt`. All 833 requested files received (7 retried). Each file is in `reports/tables/data_manifest.csv` with SHA-256. |
| CPCB station list | Read from the portal's State/City/Station menus | 122 stations | `data/raw/cpcb/ccr_repository_station_list_20261103.json`. 5 APPCB stations (Kakinada, Kurnool, Nellore, Ongole, Srikakulam) have files only for 2026. |
| OpenAQ v3 | API | 195 locations, 266 PM2.5 sensors; 1,042 sensor-year daily files | 214 sensors / 140 sites inside the five states |

## 3. CPCB results

Usable CPCB stations over the study period 2021–2025, by completeness threshold (`cpcb_usable_by_threshold.csv`):

| State | ≥ 50 % (sensitivity) | ≥ 60 % | ≥ 70 % (headline) | ≥ 75 % |
|---|---|---|---|---|
| Kerala | 6 | 6 | 6 | 4 |
| Karnataka | 24 | 21 | 19 | 18 |
| Tamil Nadu | 9 | 8 | 8 | 7 |
| Andhra Pradesh | 6 | 4 | 4 | 3 |
| Telangana | 11 | 6 | 5 | 5 |
| **All** | **56** | **45** | **42** | **37** |

For 2021–2024 the same table gave 54 / 46 / 40 / 35 (the period first planned).

Usable CPCB stations (≥ 70 %) per single year (`cpcb_usable_by_year.csv`):

| State | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| Kerala | 1 | 1 | 6 | 8 | 7 | 6 | 3 | 5 |
| Karnataka | 3 | 10 | 14 | 20 | 21 | 24 | 30 | 20 |
| Tamil Nadu | 2 | 3 | 4 | 7 | 7 | 11 | 10 | 20 |
| Andhra Pradesh | 4 | 4 | 4 | 4 | 4 | 7 | 6 | 9 |
| Telangana | 6 | 6 | 6 | 5 | 6 | 8 | 8 | 11 |
| **All** | **16** | **24** | **34** | **44** | **45** | **56** | **57** | **65** |

Alternative study periods (usable at ≥ 70 % over the whole period; `cpcb_usable_by_period.csv`):

| Period | All | Kerala |
|---|---|---|
| 2019–2022 | 27 | 2 |
| 2020–2023 | 37 | 6 |
| 2021–2024 | 40 | 6 |
| **2021–2025 (chosen)** | **42** | **6** |
| 2022–2025 | 43 | 5 |
| 2023–2025 | 54 | 3 |

### Kerala seasonal completeness, 2021–2025 (share of days valid)

| Station | Winter (JF) | Pre-monsoon (MAM) | SW monsoon (JJAS) | NE monsoon (OND) |
|---|---|---|---|---|
| Kariavattom, Thiruvananthapuram | 0.92 | 0.88 | 0.80 | 0.83 |
| Corporation Ground, Thrissur | 0.75 | 0.77 | 0.79 | 0.71 |
| Polayathode, Kollam | 0.76 | 0.75 | 0.71 | 0.71 |
| Udyogamandal, Eloor | 0.74 | 0.78 | 0.71 | 0.68 |
| Thavakkara, Kannur | 0.93 | 0.74 | 0.68 | 0.80 |
| Plammoodu, Thiruvananthapuram | 0.90 | 0.85 | 0.64 | 0.78 |
| Vyttila, Kochi | 0.57 | 0.42 | 0.47 | 0.45 |
| Palayam, Kozhikode | 0.51 | 0.39 | 0.35 | 0.28 |
| Kacheripady, Ernakulam | 0.30 | 0.30 | 0.28 | 0.26 |

Across all five states, median station completeness by season over 2021–2025 is 0.48 (winter),
0.46 (pre-monsoon), 0.47 (SW monsoon) and 0.58 (NE monsoon). This median includes stations that
started part-way through the period.

## 4. OpenAQ results (for the record)

OpenAQ is a partial mirror of the same CPCB monitors. Usable OpenAQ sites (≥ 70 %) per year,
all states (`usable_sites_by_year.csv`): 0 in each year 2018–2022, 1 in 2023, 1 in 2024 and 24 in 2025.
Sensors with any record: 54 (2021), 75 (2022), 2 (2023), 3 (2024), 113 (2025). Most CPCB locations
in OpenAQ stop in October 2022 and restart under new IDs in February 2025.

## 5. Station coordinates

The CCR portal gives no coordinates, so CPCB stations are matched to OpenAQ by name (`cpcb_station_list.csv`):
- **111** matched, where all same-named OpenAQ entries agree within 500 m.
- **3** have conflicting OpenAQ coordinates: Manali Village, Chennai (entries 363 km apart); ICRISAT Patancheru (27 km); Sanathnagar (5.7 km).
- **8** are unmatched: City Railway Station and Sanegurava Halli (Bengaluru); Gandhi Nagar_Ennore (Chennai); Meelavittan (Thoothukudi); Jakrama Nagar (Ongole); Anand Kala Kshetram (Rajamahendravaram); Toll Gate (Tirumala); Vaikuntapuram (Tirupati).

**Four usable stations (2021–2025) lack coordinates:** Anand Kala Kshetram (Rajamahendravaram), ICRISAT Patancheru,
Sanathnagar and Toll Gate (Tirumala). They are not on the map (`cpcb_station_map.png` shows 38 of the 42). Coordinates must
come from an authoritative list before Stage 2. Per D017: CPCB's official station list first, state boards as fallback.

## 6. Duplicates: CPCB vs OpenAQ (rule D015)

For the 104 name-matched stations with at least 30 shared valid days (`cpcb_vs_openaq_agreement.csv`):
- On the median station, **66 % of shared days agree within 1 µg/m³** (median absolute difference 0.45 µg/m³).
- On the median station, **5 % of days differ by more than 10 µg/m³**.
- Pearson r is only 0.91 at the median station and near zero at a few stations, driven by those outlier days.

Interpretation: these are the **same monitors**. The differences most likely come from different
day definitions (OpenAQ 01:00–00:00 vs CPCB 00:00–23:00 labels) and OpenAQ's overlapping half-hour
records. A correlation-based duplicate rule (originally r ≥ 0.95) would wrongly call many of them
different stations.

**Revised rule:** a CPCB station and an OpenAQ location are the same site if their names match
(normalised) or they lie within 500 m. **Keep CPCB, drop the OpenAQ copy.** Use OpenAQ only for
monitors that are not in CPCB at all (e.g. the US Consulate monitors "Hyderabad"/"Chennai", and
low-cost AirGradient/Clarity sensors). Even then, the low-cost ones should be excluded from the
reference target by default.

## 7. Data-quality issues already seen (for Stage 3/4)
- **Overlapping OpenAQ records:** pre-2025 OpenAQ data hold two overlapping hourly records per clock hour (:00 and :30 IST). This is handled in the OpenAQ completeness count (`estimate_hours`) and is moot if CPCB is primary.
- **Flatlined value:** OpenAQ sensor 40619 holds a constant 985 µg/m³ for consecutive hours on 2021-01-15.
- **Spike:** Plammoodu 2023 (CPCB) has a maximum of 541 µg/m³ against a median of 14 µg/m³. This is a candidate spike.
- **Timestamp convention:** CPCB timestamps run 00:00–23:00 per day. Whether a label marks the start or the end of its hour, and confirmation that it is IST, must be settled before Stage 2 alignment.
- **Station-years with no PM2.5:** some station-year files exist but contain no PM2.5 values (e.g. Eloor 2018 and Kannur 2018 have 8,760 rows and 0 PM2.5 hours, `cpcb_files.csv`).

## 8. Outputs
`reports/tables/`: `cpcb_site_inventory.csv`, `cpcb_completeness_by_year.csv`, `cpcb_completeness_by_season.csv`,
`cpcb_usable_by_year.csv`, `cpcb_usable_by_threshold.csv`, `cpcb_usable_by_period.csv`, `cpcb_station_list.csv`, `cpcb_vs_openaq_agreement.csv`,
`station_inventory.csv`, `site_inventory.csv`, `completeness_by_year.csv`, `completeness_by_season.csv`,
`usable_sites_by_year.csv`, `duplicate_candidates.csv`, `audit_summary.csv`, `openaq_days_vs_hours_check.csv`,
`data_manifest.csv`.
`reports/figures/`: `cpcb_station_map.png`, `station_map.png` (OpenAQ).
