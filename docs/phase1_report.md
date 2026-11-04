# Phase 1 report: satellite-based ground-level PM2.5 in South India

**Status: Stages 1–5 complete; Stage 5 awaiting sign-off (D038–D043 proposed; QC rule S3a vs S2b open).**
Every number below comes from a file in `reports/tables/` or the dataset's data card, and can be
regenerated with `python -m src.run stage5` (after `stage2` and `stage4`). Stage details are in
`station_audit.md`, `stage2_acquisition.md`, `stage3_exploration.md`, `feasibility_summary.md`
and `stage4_cleaning_features.md`; every decision is in `decision_log.md`.

## 1. Question and design

How much do cloud cover and the monsoon degrade satellite-based daily PM2.5 estimation in
South India, and how much does a two-tier model recover?

- **Tier A:** satellite gases (TROPOMI) and AOD (MAIAC), plus reanalysis aerosol (MERRA-2),
  meteorology (ERA5-Land / ERA5), static and temporal features. 43 features.
- **Tier B:** the same without satellite inputs, usable on cloudy days. 31 features.
- **Two-tier:** Tier A where MAIAC AOD exists within 10 km, otherwise Tier B (D039).

Each tier is run with and without PM2.5 lags (D037). The lag versions need yesterday's
measurement at the same station, so they describe gap filling at a monitored site. **The no-lag
versions are the honest measure for unmonitored locations**, which is what a satellite map is for.

## 2. Scope decisions

| Decision | Choice |
|---|---|
| Study area (D010) | Kerala, Karnataka, Tamil Nadu, Andhra Pradesh, Telangana; Kerala reported separately as the monsoon focus |
| Period (D011) | 2021-01-01 to 2025-12-31 (extraction from 2020-12-01 for lags, D023) |
| Valid day (D012) | ≥ 18 of 24 hourly values |
| Headline station set (D013) | stations with valid days on ≥ 70 % of the period (43 stations; ≥ 50 %: 56) |
| Seasons (D014) | IMD: winter Jan–Feb, pre-monsoon Mar–May, SW monsoon Jun–Sep, NE monsoon Oct–Dec |
| Ground sources (D008, D009, D015) | CPCB CCR hourly (manual download); OpenAQ only for the two US Consulate monitors |
| Time convention (D024) | CPCB labels are IST hour-start (checked against OpenAQ UTC, median r 0.986) |
| Satellite products (D022, D027, D029, D030) | TROPOMI OFFL L3 (Earth Engine, pre-filtered); MAIAC "clear" AOD; no extra cloud filter |
| Reanalysis (D019, D020) | MERRA-2 reconstructed PM2.5 and AOD; ERA5-Land with ERA5 fallback; ERA5 BLH |
| Static (D025, D033) | SRTM, ESA WorldCover, WorldPop, distance to coast; road density dropped |
| Sampling (D026, D031, D035) | station points only (grid pending); 10 km buffer mean; no minimum pixel count |
| Target (D034, D036) | daily IST mean is primary; 12–15 IST overpass mean is in the dataset as secondary |

## 3. Data summary

- **Ground:** 2,995,590 hourly values (CPCB 2,916,244; US Consulates 79,346) at 120 located
  stations, 2020-12 to 2025-12.
- **Satellite and reanalysis:** Earth Engine extracts for 120 points (TROPOMI, MAIAC, ERA5-Land, ERA5, MERRA-2, static), every raw file hashed in
  `reports/tables/data_manifest.csv`.
- **Coverage on valid PM2.5 days** (Stage 3, `eda_feasibility.csv`): MAIAC AOD at the station
  pixel 32.5 %, within 10 km 54.9 %; NO2 within 10 km 68.9 %. Kerala SW monsoon: AOD at the
  pixel 11.8 %, within 10 km 36.9 %. Gaps follow cloud: AOD at the pixel exists on 71 % of
  clear days and 1 % of overcast days.

## 4. Cleaning outcomes

QC rules S3a (D028), from `cleaning_log.csv`. No row is deleted; each hour carries a flag.

| Rule | Hours removed |
|---|---|
| Negative / zero / exactly 1000 | 17 / 58 / 2 |
| Flat run ≥ 24 h | 8,964 |
| Isolated spike (1–2 h, > 4 × centred 25 h median and > 100 µg/m³) | 3,926 |
| **Total** | **12,967 of 2,995,590 (0.43 %)** |

Dataset `pm25_dataset_v1.parquet`: 127,043 station-days at 113 stations, 113,727 with a valid
daily target and 124,342 with a valid overpass target (`dataset_reconciliation.csv`).
The two targets agree closely (Spearman ρ 0.884).

## 5. Features

| Group | Columns | Tier |
|---|---|---|
| Satellite | TROPOMI NO2, CO, SO2, HCHO, UV aerosol index, cloud fraction; MAIAC AOD; valid-pixel counts and availability flags (10 km) | A |
| Reanalysis aerosol | MERRA-2 PM2.5 and AOD | A, B |
| Meteorology | T2m, RH, u10, v10, wind speed, precipitation, BLH | A, B |
| Lags | PM2.5 1–3 days (dropped in `_nolag`); meteorology 1 day; precipitation 2–3 days | A, B |
| Temporal | day-of-year sin/cos, month, season | A, B |
| Static | elevation, distance to coast, land-cover shares, population | A, B |

Missing satellite values stay missing (Random Forest handles NaN natively; linear regression
uses medians fitted on the training rows).

## 6. Evaluation framework

`src/eval/` (configured in `configs/experiments.yaml`):

| Split | Train | Test |
|---|---|---|
| Temporal | 2021–2024 | 2025 |
| Spatially blocked (D038) | 4 of 5 folds of 46 blocks (stations linked within 50 km) | the 5th fold |
| Leave-one-station-out (LOSO) | all other stations | one station |
| Combined | other blocks, 2021–2024 | held-out blocks, 2025 |

- **Metrics:** RMSE, MAE, R², bias and Pearson r, on pooled held-out rows. They are reported
  overall, by season, by cloud-fraction bin (D041), by station and for Kerala Jun–Sep.
- **Baselines:** persistence (yesterday's value; temporal split only), climatology (station ×
  month, falling back to state × month), linear regression, and Random Forest (fixed settings, D040).
- **Leakage controls:** tests check that no station, block or year appears on both sides. All
  preprocessing is fitted on training rows, and lags are calendar lags.

## 7. Baseline results

### 7.1 Overall (`results_overall.csv`, `results_station_sets.csv`)

Random Forest, held-out R² (RMSE in µg/m³). All 113 stations; the D013 headline set
(43 stations) is in brackets.

| Model | Temporal | Blocked | LOSO | Combined |
|---|---|---|---|---|
| Climatology | 0.24 (0.30) | 0.16 (0.19) | 0.21 (0.25) | 0.12 (0.15) |
| Two-tier, with lags | 0.58 (0.61), 14.1 | 0.69 (0.71), 11.2 | 0.70 (0.72), 11.0 | 0.57 (0.59), 14.2 |
| Two-tier, no lags | 0.34 (0.41), 17.6 | 0.29 (0.33), 16.9 | 0.29 (0.35), 16.9 | 0.17 (0.23), 19.8 |
| Linear two-tier, no lags | 0.20 (0.30) | 0.31 (0.37) | 0.32 (0.39) | 0.18 (0.28) |

- **PM2.5 lags carry most of the skill.** Persistence alone reaches R² 0.56 in 2025. On the
  same 24,690 rows, the two-tier RF with lags reaches 0.64, RMSE 12.8 against 14.0
  (`results_persistence_rows.csv`).
- **Without lags, skill at an unseen station is low:** LOSO R² 0.29, against 0.21 for
  climatology. This matches the Stage 3 finding that stations within 10 km correlate only at
  median ρ 0.49.
- **Linear regression and the RF are indistinguishable in the spatial no-lag splits.** LOSO R²
  is 0.32 against 0.29, but the difference is not significant: RF − linear −0.026, 95 % CI
  [−0.092, +0.036] (`results_ci.csv`). In the temporal split the RF is clearly better
  (+0.137, CI [+0.059, +0.239]).
- **Per station (LOSO, no lags):** median R² 0.19. It is below 0 at 34 of 113 stations and below
  −1.5 at 10. Medians by state: Tamil Nadu 0.36, Karnataka 0.29, Andhra Pradesh 0.11, Kerala 0.03,
  Telangana −0.19.

![Predicted vs observed](../reports/figures/results_pred_vs_obs.png)
![Per-station skill](../reports/figures/results_station_spread.png)

### 7.2 What the satellite adds (Tier A against Tier B on the same rows)

RF R² on the 62,544 rows where MAIAC AOD exists within 10 km (15,419 in the 2025 test set):

| Split | Tier A, no lags | Tier B, no lags, same rows | Tier A, lags | Tier B, lags, same rows |
|---|---|---|---|---|
| Temporal | 0.330 | 0.336 | 0.579 | 0.581 |
| Blocked | 0.252 | 0.230 | 0.688 | 0.689 |
| LOSO | 0.248 | 0.226 | 0.698 | 0.699 |
| Combined | 0.154 | 0.142 | 0.568 | 0.569 |

- **With MERRA-2 and meteorology already in the model, the satellite inputs add little, but the
  gain is real where it matters.** At an unseen location (no lags) Tier A beats Tier B on the same
  rows by +0.022 R² in both the blocked split (95 % CI [+0.009, +0.035]) and LOSO
  ([+0.009, +0.038]). In the SW monsoon the gain is larger: LOSO +0.047 [+0.015, +0.086] and
  blocked +0.036 [+0.003, +0.060]. There is no gain in the temporal split (−0.007
  [−0.015, −0.001]) or with PM2.5 lags (|Δ| ≤ 0.002 everywhere).
- **The ablation explains why** (`results_ablation.csv`; RF, no lags, blocked split, rows with
  AOD). R² rises from 0.12 (met only) to 0.17 (+ gases), 0.19 (+ AOD) and 0.24 (+ MERRA-2).
  The satellite inputs do carry signal, about +0.07 R² over meteorology alone, but MERRA-2
  overlaps with most of it and adds more on its own.

![Ablation](../reports/figures/results_ablation.png)

### 7.3 Cloud and the monsoon

**By cloud fraction** (`results_by_cloud.csv`, RF, no lags, LOSO):

| Cloud fraction | Rows | RMSE, Tier B (all rows) | Rows with AOD | RMSE, Tier A | RMSE, Tier B (same rows) |
|---|---|---|---|---|---|
| 0.0–0.2 | 41,296 | 18.5 | 36,704 | 18.5 | 18.8 |
| 0.2–0.5 | 27,575 | 16.5 | 15,214 | 17.3 | 17.5 |
| 0.5–0.8 | 16,423 | 15.3 | 4,651 | 16.2 | 16.6 |
| 0.8–1.0 | 24,058 | 15.0 | 2,107 | 18.4 | 18.5 |

- **AOD almost disappears under cloud.** Above cloud fraction 0.8, Tier A covers 2,107 of
  24,058 rows (9 %). The two-tier system is therefore almost entirely Tier B on cloudy days.
- **Absolute RMSE falls with cloud**, because cloudy days are mostly monsoon days with low
  PM2.5. Where Tier A exists, its advantage over Tier B stays at 0.2–0.4 µg/m³ RMSE in every bin.
  So cloud removes the satellite's coverage more than it degrades the satellite's skill.
- **The temporal split's 0.8–1.0 bin with AOD is small** (581 rows) and noisy (RMSE 26).

**By season** (`results_by_season.csv`, temporal 2025, RF two-tier no lags): R² is 0.29 in
winter, 0.08 pre-monsoon, 0.26 SW monsoon and 0.36 NE monsoon. With lags it is 0.51, 0.35, 0.74
and 0.59. In the SW monsoon, RMSE is lowest (10.0 µg/m³ without lags) because concentrations
are low.

![Residuals by season](../reports/figures/results_residuals_by_season.png)
![Skill by cloud](../reports/figures/results_skill_by_cloud.png)

**Kerala, June–September** (`results_kerala_jjas.csv`; 3,306 station-days, 1,220 with AOD):

| Model (LOSO) | R² | RMSE |
|---|---|---|
| Climatology | −0.04 | 13.6 |
| Two-tier RF, no lags | −0.07 | 13.8 |
| Tier A RF, no lags (AOD rows) | −0.03 | 13.5 |
| Two-tier RF, with lags | 0.52 | 9.2 |

Without the station's own recent measurements, no model beats a monthly mean in the Kerala
monsoon. This confirms the Stage 3 finding that no predictor correlates with PM2.5 there
at |ρ| > 0.1.

### 7.4 Confidence intervals (D043, `results_ci.csv`)

Intervals come from 1,000 resamples of the 46 spatial blocks. Model differences are
bootstrapped on the same rows.

- **Absolute R² is uncertain because blocks differ a lot.** Two-tier without lags: LOSO 0.29
  [0.22, 0.36], temporal 0.34 [0.19, 0.53]. Tier A with lags (AOD rows): LOSO 0.70 [0.58, 0.78].
- **Differences between models on the same rows are much tighter.** The comparisons in 7.1–7.2
  are reported with these paired intervals.
- **Against the simple baselines:** the RF beats climatology without lags (LOSO +0.086
  [+0.008, +0.160]) and persistence with lags (temporal +0.072 [+0.027, +0.169]).
- **In the SW monsoon the no-lag model is no better than climatology** (blocked −0.005
  [−0.151, +0.064], LOSO +0.012 [−0.061, +0.089]).

### 7.5 Sensitivity to the Stage 4 choices (D042, `results_sensitivity.csv`)

Random Forest, no PM2.5 lags. "Δ" is Tier A − Tier B on the variant's Tier A rows, with a 95 %
block-bootstrap CI.

| Variant | Tier A rows (blocked) | Δ R², blocked | Δ R², temporal | Two-tier R², blocked / temporal |
|---|---|---|---|---|
| Main (clear AOD, 10 km, no filters) | 62,544 | +0.022 [+0.009, +0.035] | −0.007 [−0.015, −0.001] | 0.290 / 0.341 |
| MAIAC "best" QA (D029) | 54,427 | +0.019 [+0.009, +0.033] | −0.007 [−0.019, +0.001] | 0.287 / 0.341 |
| Station pixel (D035) | 36,867 | +0.008 [−0.012, +0.026] | −0.009 [−0.034, +0.005] | 0.279 / 0.342 |
| 5 km buffer (D035) | 55,188 | +0.020 [+0.009, +0.029] | −0.011 [−0.032, +0.002] | 0.287 / 0.339 |
| No satellite when cloud > 0.3 (D030) | 47,714 | +0.017 [+0.004, +0.035] | −0.002 [−0.018, +0.010] | 0.284 / 0.344 |
| ≥ 10 % valid pixels (D031) | 51,287 | +0.016 [+0.004, +0.031] | −0.009 [−0.026, +0.002] | 0.285 / 0.341 |
| ≥ 25 % valid pixels (D031) | 45,306 | +0.018 [+0.005, +0.032] | −0.008 [−0.027, +0.004] | 0.284 / 0.341 |
| Overpass target 12–15 IST (D036) | 67,849 | +0.018 [+0.011, +0.030] | +0.005 [−0.002, +0.016] | 0.181 / 0.190 |
| QC S2b, flat-line ≥ 12 h (D028) | 62,342 | +0.011 [−0.014, +0.030] | −0.008 [−0.017, −0.000] | 0.314 / 0.410 |

- **The satellite choices do not change the conclusion.** Every buffer, QA or filter variant
  gives the same small spatial gain (+0.016 to +0.022) and no temporal gain. The main choices
  (clear AOD, 10 km, no filters) are at least as good as the alternatives. The station pixel
  is weakest: it has the fewest rows and no significant gain.
- **The overpass target is much harder to predict** than the daily mean: two-tier R² 0.18–0.19
  without lags and 0.35–0.41 with lags, because a 3-hour mean is noisier. The satellite gain is
  similar, so it does not favour the satellite-matched target.
- **QC S2b raises skill substantially:** temporal R² 0.34 → 0.41 without lags and 0.58 → 0.65
  with lags.
  - It flags 20,318 hours instead of 12,967 and removes 432 of 113,727 days (0.4 %). Those
    days carry 11–30 % of all squared error, including one at 770 µg/m³.
  - The extra hours are runs of 12–23 identical hourly values (7,370 hours). Repeated round
    values such as 45.0 occur 379 times, and values above 100 µg/m³ are 3.5 times more common
    than elsewhere.
  - This points to stuck instruments, an argument for the stricter rule on QC grounds rather
    than because it improves the score. **Whether to adopt S2b as the main rule is an open
    decision (D028 revision).** If adopted, the main runs are repeated.

### 7.6 Headline answer for Phase 1

1. Cloud and the monsoon act mainly by **removing** satellite coverage: AOD exists on 9 % of
   days with cloud fraction > 0.8 and on 37 % of Kerala monsoon days within 10 km.
2. Where satellite data exist, they add a small but significant amount of skill at unmonitored
   locations once MERRA-2 and meteorology are present: +0.022 R² [+0.009, +0.035] in the spatial
   splits, and about +0.04 in the SW monsoon. They add none for gap filling at a monitored
   station. This holds for every buffer, QA and filter variant tested.
3. **Tier B therefore recovers essentially all of the two-tier skill.** For an unmonitored
   location, though, that skill is low: LOSO R² 0.29 (0.35 on the 43 headline stations), and
   no skill in the Kerala monsoon.

## 8. Limitations

- **Point extraction only.** No 1 km grid yet (D026). The 10 km buffer smooths the satellite
  signal, which may understate what a pixel-level model could do.
- **Untuned baselines** (D040). Tuning, gradient boosting and spatial features such as
  neighbour-station values or distance-weighted PM2.5 are Phase 2 work.
- **Sensitivity coverage.** The sensitivity runs (7.5) use the Random Forest on the temporal
  and blocked splits only. LOSO and linear regression were not rerun. The QC choice (S3a vs S2b)
  changes absolute skill noticeably and is still open.
- **Bootstrap clusters.** The bootstrap resamples spatial blocks only. In the temporal split,
  errors are also correlated across days, so those intervals may still be somewhat narrow.
- **Uneven station coverage.** Kerala has 9 stations (6 at ≥ 70 %), so its per-state numbers rest
  on few sites. Several Tamil Nadu and Andhra Pradesh stations start in 2024, so they have little
  training data in the temporal split.
- **Pooled metrics.** R² is computed on pooled held-out rows, so it mixes between-station and
  within-station variance. Per-station metrics are in `results_by_station.csv`.
- **Data caveats.** CPCB may still revise 2025 data. Earth Engine MERRA-2 is missing 6 days in
  September 2025. TROPOMI L3 uses Google's fixed pre-filter (D027).

## 9. Proposed for Phase 2

1. Settle the QC rule (S3a or S2b) and, if it changes, repeat the main runs.
2. Add spatial predictors for unmonitored sites (inverse-distance PM2.5 from other stations on
   the same day; this needs care to avoid leakage in LOSO).
3. Tune models with nested validation, and add gradient boosting.
4. Extract the 1 km grid (D026) and produce maps.
5. Develop the Kerala monsoon case separately, for example with longer-range transport,
   rainfall-washout features or a different target scale.
