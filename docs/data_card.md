# Data card

## Source

| Item | Value |
|---|---|
| Provider | NASA POWER Project (NASA Langley Research Center), hourly point API, community "RE" (renewable energy) |
| Location | 23.23°N, 56.52°E, near Ibri, Ad Dhahirah, Oman (area of the Ibri II solar plant) |
| Period | 1 Jan 2015 00:00 to 31 Dec 2025 23:00, hourly, local solar time (LST) |
| Downloaded | 1 November 2026, one request for all 11 years (exact link in `data/manifest.json`) |
| File used | `data/ibri_hourly.csv.gz` (committed; 96,432 rows); checksum in `data/manifest.json` |
| Terms | NASA POWER data are free to use; NASA asks for the acknowledgement given in `NOTICE` |

The data file is committed because NASA can revise recent values. `solpredict download` re-fetches it from the source (one request per year).

## What the data are (and are not)

These are **satellite and weather-model estimates for a grid cell, not measurements from a pyranometer at a plant**. Radiation comes from CERES SYN1deg (about 1° resolution, roughly 100 km); weather from MERRA-2 (0.5° × 0.625°). Local effects such as a dust plume over the plant itself or panel soiling are not visible at this resolution.

| Column | Meaning | Unit |
|---|---|---|
| `ALLSKY_SFC_SW_DWN` | Global horizontal irradiance (GHI): sunlight reaching the ground. **The target.** | W/m² (hourly mean) |
| `CLRSKY_SFC_SW_DWN` | GHI if there were no clouds; still reduced by dust, haze and water vapour | W/m² |
| `TOA_SW_DWN` | Sunlight at the top of the atmosphere: pure astronomy | W/m² |
| `T2M`, `RH2M`, `WS10M` | Temperature at 2 m, relative humidity at 2 m, wind speed at 10 m | °C, %, m/s |
| `PRECTOTCORR` | Precipitation | mm/hour |

## Checks (`solpredict check`, saved in `results/data_checks.json`)

| Check | Result |
|---|---|
| Rows vs hours in the period | 96,432 of 96,432; no gaps, no duplicates |
| Missing values (NASA's −999) | 0 in every column |
| Sunlight at night (TOA = 0 but GHI > 1 W/m²) | 0 hours |
| GHI above top-of-atmosphere | 0 hours |
| GHI above clear-sky by more than 50 W/m² | 0 hours |
| Maximum GHI | 1,104 W/m² (physically plausible) |
| Hour with the highest average GHI | 12:00 LST, so the time standard is as stated |

## Cross-checks against other sources

See `docs/verification.md`, layer 4.

| Check | Result |
|---|---|
| Yearly sunlight on a 25° south-facing panel vs PVGIS-SARAH (2,548 kWh/m², 2005–2016) | NASA 2,362 for the shared years 2015–2016 (−7.3%); 2,311 on average for 2015–2025 (−9.3%) |
| Yearly horizontal sunlight, NASA | 2,244 (2015), 2,228, 2,213, 2,201, 2,175, 2,226, 2,234, 2,168, 2,142, **2,105 (2024)**, 2,169 (2025) kWh/m² |

The 2024 low matters for the evaluation: the test years were dimmer than the training years, so the seasonal baseline over-forecasts them.

## Preprocessing

1. Read NASA's CSV, turn −999 into missing, index by timestamp (LST).
2. Build one row per (forecast day, hour); see `src/solpredict/features.py`.
3. Night hours (top-of-atmosphere sunlight ≤ 20 W/m²) are not used for training or scoring; the forecast for them is 0.

## Split

By the day being forecast; never shuffled.

| Part | Forecast days | Daylight hours scored |
|---|---|---:|
| Train | 2 Jan 2015 – 31 Dec 2022 | — |
| Validation (model choice) | 2023 | 4,516 |
| Test (used once) | 2024 – 2025 | 9,043 |

Consecutive days are correlated, so the last training day (31 Dec 2022) and the first validation day touch. The forecast for 1 Jan 2023 uses 31 Dec 2022 as its input, which is exactly what happens in real use, so no gap is needed.
