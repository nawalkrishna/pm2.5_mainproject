# Stage 6 (Phase 1 add-on): hourly forecasting at existing stations

**Status: COMPLETE, awaiting sign-off.** Decisions D045–D051 in `docs/decision_log.md`
(D045, D046, D048 decided by the student; D047, D049–D051 proposed). Phase 1 (Stages 1–5) is
unchanged and stays at tag `phase1-final`. Every number below comes from a file in
`reports/tables/` (`forecast_*.csv`). Regenerate with `python -m src.run stage6`.

## 1. Why this stage

The papers in the literature review (Qi et al. 2019; Chen et al. 2025; Ni et al. 2023; Shi et
al. 2026; Shaban 2026) forecast PM2.5 hours to days ahead at stations that already have a
monitor, using the stations' own recent PM2.5. Phase 1's main task is different: estimating
PM2.5 where there is no monitor. This stage adds the forecasting task on the same data, so the
thesis can (a) compare with those papers on their own task and (b) ask whether satellite data
and cloud matter for forecasting as they do for estimation.

Phase 1's "with lags" models are not forecasts: they use yesterday's PM2.5 but *today's*
weather and satellite data. Here every input is restricted to what is known when the forecast
is issued.

## 2. Task (D045)

- A forecast is issued at the end of each IST hour H, after that hour's PM2.5 is observed.
- Target: the QC-passed hourly PM2.5 (Stage 4 rule S3a-12h) at hour H + h, h = 1…72. Results
  are reported at h = 1, 3, 6, 12, 24, 48, 72 h, as in Qi et al. and Chen et al.
- Stations: all 113 with QC-passed hourly data; 110 have test-year data.
- Hourly table (`forecast_hourly_v1.parquet`): 113 stations × 44,568 hours = 5,036,184 rows,
  2,945,207 PM2.5 values (the Stage 4 QC-passed count), meteorology for every hour.

## 3. What each forecast may use (D046)

| Input | Setting (a): past only | Setting (b): + perfect weather |
|---|---|---|
| PM2.5 at the station (and neighbours, graph model) | hours ≤ H | same |
| ERA5 / ERA5-Land instantaneous fields (T, dew point, wind, BLH) | hours ≤ H | + target hours |
| Precipitation, MERRA-2 aerosol (hourly averages / accumulations) | hours ≤ H − 1 | + precipitation at target hours; MERRA-2 never |
| Satellite (TROPOMI, MAIAC; daily) | day d if H ≥ 15 IST, else day d − 1 | same |
| Calendar (target hour, day of year, weekday), static station features | yes | yes |

Setting (b) is an upper bound: reanalysis weather stands in for a perfect weather forecast.
Reanalysis is not available in real time; in (a) it stands in for real-time weather analyses.
A unit test (`tests/test_forecast.py`) checks on a synthetic series that no setting-(a) feature
uses data after hour H, and that averaged fields stop at H − 1.

## 4. Splits (D047)

Train on issue hours in 2021–2023, validate on 2024 (early stopping and the small tuning of the
neural networks only), test on 2025. The last 72 h of the training and validation periods are
purged so no target falls in the next period. Test rows: 4,935,056 station × issue-hour ×
horizon combinations with an observed target (about 705,000 per horizon).

## 5. Models (D048)

| Model | Description |
|---|---|
| Persistence | Last valid PM2.5 within 24 h; else climatology |
| Climatology | Training mean by station × month × hour of day; else state × month × hour; else month × hour |
| Linear | OLS on median-imputed, standardised features, one model per horizon |
| Random Forest | Phase 1 settings (D040), one model per horizon, native NaN handling; trained on a random third of training rows (about 506,000; cost) |
| LSTM | Encoder over the past 72 h at each station + multi-horizon head (1–72 h at once), shared across stations; the head predicts the change from the latest PM2.5 value |
| Graph network (GC-LSTM) | As the LSTM, with a graph convolution over the station network at every time step (Qi et al. 2019): stations within 200 km linked (median 19 neighbours), Gaussian distance weights |

Neural networks: PyTorch 2.7.1 + CUDA 12.6 on an RTX 3050 (4 GB); 64 hidden units, Adam 1e-3,
batches of 16 issue hours (each with all stations), each epoch a random third of the training
issue hours, early stopping on 2024 (patience 8). Three seeds (42, 1, 2) for every configuration,
24 runs, 4–12 min each. See the D048 addendum for the residual-head change and the
validation-year tuning (`forecast_nets_tuning.csv`).

## 6. Results

### 6.1 Skill by horizon (Tier B, all test rows; neural networks seed 42)

R² (`forecast_skill.csv`):

| Model | 1 h | 3 h | 6 h | 12 h | 24 h | 48 h | 72 h |
|---|---|---|---|---|---|---|---|
| Persistence | 0.780 | 0.432 | 0.148 | −0.040 | 0.060 | −0.096 | −0.232 |
| Climatology | 0.105 | 0.105 | 0.105 | 0.105 | 0.105 | 0.105 | 0.106 |
| Linear (a) | 0.807 | 0.576 | 0.462 | 0.410 | 0.378 | 0.309 | 0.284 |
| Random Forest (a) | **0.813** | **0.576** | **0.496** | **0.450** | **0.430** | **0.361** | **0.321** |
| LSTM (a) | 0.744 | 0.548 | 0.466 | 0.433 | 0.418 | 0.341 | 0.299 |
| Graph network (a) | 0.750 | 0.563 | 0.478 | 0.438 | 0.419 | 0.339 | 0.299 |
| Random Forest (b) | 0.814 | 0.577 | 0.498 | 0.455 | 0.438 | 0.376 | 0.345 |
| LSTM (b) | 0.773 | 0.552 | 0.473 | 0.440 | 0.426 | 0.360 | 0.338 |
| Graph network (b) | 0.780 | 0.558 | 0.483 | 0.442 | 0.424 | 0.365 | 0.340 |

RMSE (µg/m³): Random Forest (a) 10.8 at 1 h, 18.8 at 24 h, 20.5 at 72 h; persistence 11.7, 24.1,
27.6; climatology 23.5. Figure: `reports/figures/forecast_r2_by_horizon.png`.

Seed range (min–max over seeds 42, 1, 2; `forecast_seed_range.csv`), setting a, Tier B: LSTM
0.744–0.774 at 1 h, 0.413–0.421 at 24 h, 0.299–0.306 at 72 h; graph network 0.750–0.788,
0.419–0.423, 0.299–0.307. Every seed of both networks is below the Random Forest at 1, 24 and
72 h.

Paired bootstrap, block × month clusters (about 500; `forecast_skill_ci.csv`):

| Comparison | 1 h | 24 h | 72 h |
|---|---|---|---|
| Random Forest (a) − persistence | +0.033 [+0.017, +0.048] | +0.371 [+0.270, +0.477] | +0.553 [+0.406, +0.722] |
| LSTM (a) − Random Forest (a) | −0.069 [−0.101, −0.033] | −0.013 [−0.018, −0.006] | −0.022 [−0.036, −0.009] |
| Graph network (a) − Random Forest (a) | −0.063 [−0.094, −0.030] | −0.012 [−0.017, −0.005] | −0.022 [−0.035, −0.011] |
| Graph network (a) − LSTM (a) | +0.006 [−0.001, +0.013] | +0.001 [−0.003, +0.005] | +0.000 [−0.009, +0.011] |
| Random Forest: setting b − a | +0.001 [−0.000, +0.002] | +0.007 [+0.004, +0.011] | +0.024 [+0.014, +0.036] |
| LSTM: setting b − a | +0.028 [+0.012, +0.043] | +0.008 [−0.000, +0.017] | +0.039 [+0.020, +0.060] |
| Graph network: setting b − a | +0.030 [+0.013, +0.046] | +0.005 [−0.005, +0.016] | +0.041 [+0.023, +0.064] |

Reading:
- Every trained model beats persistence beyond 1 h by a wide margin; persistence is below
  climatology from 12 h on. At 1 h the Random Forest beats persistence by +0.033.
- **The Random Forest is the best model at every horizon.** Both neural networks are
  significantly below it (−0.01 to −0.07), and the graph network is no better than the LSTM
  except at 3–6 h (+0.011 to +0.015). Linking neighbouring stations adds little here: South
  Indian stations are sparse and only loosely correlated (Phase 1: correlation about 0.5 within
  10 km).
- A perfect weather forecast (setting b) helps most at long horizons (+0.02 to +0.04 at 72 h).
- Skill falls from about 0.81 at 1 h to 0.43 at 24 h and 0.32 at 72 h. Qi et al. (2019) report
  R² 0.72 at 72 h for Beijing–Tianjin–Hebei, where pollution episodes last for days; South Indian
  PM2.5 is lower and changes faster, so persistence-like information decays sooner.

### 6.2 Does satellite data help forecasting? (D049)

Tier A (+ satellite) against Tier B on the same rows (test rows whose latest satellite day has
MAIAC AOD, about 395,000 per horizon; `forecast_satellite.csv`):

| Model | 1 h | 24 h | 72 h |
|---|---|---|---|
| Random Forest (a) | −0.002 [−0.005, +0.001] | −0.004 [−0.007, −0.002] | −0.001 [−0.005, +0.003] |
| Random Forest (b) | −0.003 [−0.005, −0.001] | −0.005 [−0.006, −0.003] | −0.007 [−0.012, −0.002] |

Neural networks, Tier A − Tier B, mean [min, max] over the three seeds
(`forecast_satellite_seeds.csv`): LSTM (a) −0.001 [−0.015, +0.011] at 1 h, −0.002 [−0.006,
+0.002] at 24 h, +0.004 [−0.002, +0.011] at 72 h; graph network (a) −0.005 [−0.016, +0.005],
−0.010 [−0.020, −0.003], −0.007 [−0.015, −0.001].

**Satellite data add nothing to forecasting.** The Tier A − Tier B difference is between −0.007
and +0.001 for the Random Forest at every horizon, sometimes significantly below zero (extra
noisy inputs), and within seed-to-seed variation for the networks. Once the station's own
recent PM2.5 is known, a once-a-day column measurement carries no extra information.

By cloud and season (Random Forest (a), 24 h; `forecast_satellite_groups.csv`): the difference
is −0.003 to −0.013 in every cloud bin and season (SW monsoon −0.009 [−0.017, −0.004]).
Satellite coverage collapses under cloud as in Phase 1: the latest satellite day has AOD for
224,446 of 252,063 rows with cloud fraction < 0.2 (89 %) but 15,687 of 161,239 with ≥ 0.8 (10 %)
(`forecast_aod_coverage.csv`).

## 7. Problems found and fixed

1. **LSTM below persistence at short horizons.** The first LSTM run scored 1 h R² 0.63 against
   persistence's 0.78 on the test year. The head now predicts the change from the latest
   PM2.5 value (1 h 0.74). This change was prompted by a test-year comparison (disclosed in the
   D048 addendum); all later choices used the validation year only.
2. **Climatology missing for 15 % of test rows** (stations with no training data for that month
   and hour). Fallback to state × month × hour, then month × hour, as in Phase 1.
3. **GPU memory.** Evaluation batches of 64 issue hours used more than the 4 GB card; now 16.
4. **Session crash (5 November, about 08:47).** Both runs stopped. The 5 finished network runs
   were checked (row counts and R² recomputed from the saved predictions) and kept; the tabular
   run saved only at the end and was lost. Both scripts now save each horizon / run as soon as
   it finishes and skip finished parts, and the reruns ran as detached processes. All 7
   horizons and 24 network runs were then completed and checked: 4,935,056 rows in both files,
   identical keys and targets, no missing predictions.

## 8. Limitations

- Reanalysis stands in for real-time weather data, so even setting (a) is optimistic compared
  with an operational forecast; setting (b) assumes a perfect weather forecast.
- Satellite values are treated as available at the end of the overpass window; real products
  arrive hours (TROPOMI) to days (MAIAC) later. This favours Tier A, which still gains nothing.
- One test year (2025). Random Forest and linear models: one seed. Neural networks: small
  architectures, one round of tuning on the validation year; larger models might do better.
- The bootstrap does not include seed variation; for the networks it is reported separately.

## 9. Files

| File | Content |
|---|---|
| `configs/forecast.yaml` | All Stage 6 settings |
| `src/forecast/hourly_data.py` | Hourly station table |
| `src/forecast/features.py` | Issue-time features for the tabular models (timing rules of Section 3) |
| `src/forecast/run_tabular.py` | Persistence, climatology, linear, Random Forest |
| `src/forecast/nets.py`, `run_nets.py` | LSTM and graph network |
| `src/forecast/tune_nets.py` | Validation-year tuning of the LSTM |
| `src/forecast/report.py` | Metrics by horizon, bootstrap, satellite, cloud and season tables, figure |
| `tests/test_forecast.py` | Lags, windows, timing rules, purge, no future data |
| `data/processed/forecast_hourly_v1.parquet` | Hourly station table (not committed) |
| `data/processed/forecast_predictions_{tabular,nets}_v1.parquet` | Test-year predictions (not committed) |
