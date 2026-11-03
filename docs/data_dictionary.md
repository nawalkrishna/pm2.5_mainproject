# Data dictionary

Every dataset ID, band name, scale factor and unit is checked against the
official documentation before use; the link and check date are given here.
Stage 3 completes this file.

| Dataset | Variable / field | Unit | Resolution | Coverage | Source / docs (checked) | Known issues |
|---|---|---|---|---|---|---|
| OpenAQ v3 locations | location metadata, `isMonitor` (reference monitor flag) | — | point | snapshot | https://docs.openaq.org/resources/locations, https://api.openaq.org/openapi.json (2026-11-03) | Provider/owner labels are as reported by OpenAQ |
| OpenAQ v3 sensor days | `value` daily mean PM2.5; `coverage.observedCount` | µg/m³; count | point, daily (01:00–00:00 local) | per sensor | https://docs.openaq.org/resources/measurements (2026-11-03) | `observedCount` meaning to be spot-checked against `/hours` |
| CPCB CCR Data Repository, 1H | `PM2.5 (µg/m³)` hourly; also PM10, NO, NO2, NOx, NH3, SO2, CO, O3, BTEX, AT, RH, WS, WD, RF, SR, BP | µg/m³ (PM2.5) | station, hourly | 2018–2025 downloaded (some stations from 2009) | https://airquality.cpcb.gov.in/ccr/ (Data Repository; CAPTCHA), files `Raw_Data_<year>_site_<id>_<station>_1Hr.csv` (2026-11-03) | Timestamps 00:00–23:00 per day; whether labels mark hour start or end, and the timezone (assumed IST), are unconfirmed. Some station-years contain no PM2.5. |
| GADM 4.1 level 1 | `NAME_1` state name | — | polygon | — | https://gadm.org (licence: https://gadm.org/license.html, 2026-11-03) | Names without spaces (e.g. `TamilNadu`); mapped in `configs/study_area.yaml` |
| CPCB CAAQMS Stations List All India (PDF) | station name, city | — | — | snapshot 2026-11-03 | https://airquality.cpcb.gov.in/caaqms-common/api/reports/caaqm-stations/pdf (2026-11-03) | No coordinates |
| Manual station coordinates | `latitude`, `longitude` | degrees WGS84 (assumed) | point | 4 stations | CPCB All India AQI Dashboard map centre, https://airquality.cpcb.gov.in/ccr/all-india-aqi-portal (2026-11-03); `configs/station_coordinates_manual.csv` | Toll Gate only 2 decimals; Bollaram dashboard vs OpenAQ differ 3.6 km |
