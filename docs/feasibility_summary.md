# Feasibility summary (Stage 3)

*One page. Numbers from `reports/tables/eda_feasibility.csv`, 2021–2025, valid PM2.5 station-days (≥ 18 hours).*

**Ground truth is adequate.** 112,234 valid PM2.5 station-days at 110 stations in five states
(Kerala 10,436; Karnataka 45,910; Tamil Nadu 25,812; Telangana 18,231; Andhra Pradesh 11,845).

**Satellite data cover only part of those days, and the monsoon is the gap.**

| Valid PM2.5 days that also have… | All states, all year | Kerala, all year | Kerala, SW monsoon (Jun–Sep) |
|---|---|---|---|
| MAIAC AOD at the station pixel | 32.5 % (36,509) | 30.9 % | **11.8 % (390 of 3,315)** |
| MAIAC AOD within 10 km | 54.9 % | 57.9 % | 36.9 % (1,223) |
| NO2 within 10 km | 68.9 % (77,281) | 72.6 % | 56.1 % (1,860) |
| Both NO2 and AOD within 10 km | 50.5 % (56,710) | 53.7 % | 32.4 % (1,073) |

In the SW monsoon the gap is worst in Karnataka (AOD at the pixel on 3.0 % of valid days,
15.9 % within 10 km) and Telangana (6.5 %, 21.0 %), not in Kerala.

**What this means for the two-tier design**

1. **Tier A (with satellite) is trainable** on about 56,700 station-days with NO2 + AOD at
   10 km, but in the monsoon it would cover only about a third of Kerala's days. The 10 km
   buffer is necessary: the station pixel alone loses two thirds of the monsoon days that remain.
2. **Tier B (no satellite) is essential, not a fallback detail**: it is the only model
   available for roughly two thirds of Kerala monsoon days and most Karnataka/Telangana monsoon days.
3. **The monsoon signal is weak in Kerala.** In June–September no satellite or reanalysis
   aerosol variable correlates with daily PM2.5 at |ρ| > 0.10, while PM2.5 itself stays near
   24 µg/m³. Expect low skill there whichever tier is used; the thesis should measure and report it,
   not promise to fix it.
4. **Spatial generalisation will be hard.** Daily PM2.5 at stations within 10 km of each
   other correlates only at median ρ 0.49. Leave-one-station-out results will be much lower
   than random-split results, which is why the evaluation uses spatial splits (brief, Stage 5).
5. **Missingness is not random.** Satellite gaps follow cloud (AOD at the pixel: 71 % on clear days,
   1 % on overcast days), and station outages are slightly monsoon-biased. Models trained only on
   satellite-available days learn a clear-sky world; Stage 5 must report skill per cloud-fraction bin.

**Verdict:** the study is feasible as designed. The headline result will likely be a large
skill drop for Tier A in the monsoon, partly recovered by Tier B, with Kerala's monsoon
the hardest case because its PM2.5 varies least.
