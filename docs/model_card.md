# Model card: SolPredict (Ibri)

## Summary

| Item | Value |
|---|---|
| Task | Forecast, at the end of each day, the sunlight reaching the ground (GHI) for every hour of the next day at Ibri, Oman |
| Model | Ridge regression (linear) predicting tomorrow's clearness; forecast GHI = clearness × known top-of-atmosphere sunlight |
| Chosen over | Gradient boosting (HistGradientBoostingRegressor), on 2023 validation data |
| Inputs | Today's sunlight, clearness, haze, temperature, humidity, wind and rain; tomorrow's sun position |
| Output | 24 hourly GHI values (W/m²) and the matching output of an example 500 MW plant |
| Training data | NASA POWER hourly data 2015–2023 (see `docs/data_card.md`) |
| Artefacts | `models/model.joblib` (2 KB) with `model_meta.json` (sha256) and `toa_table.csv` (sun position by date and hour) |

## Intended use

* **User:** an energy planner at the grid operator (Nama) or a solar plant operator who needs tomorrow's expected solar output to plan the rest of the generation mix.
* **Use:** a first-guess day-ahead forecast and a demonstration of the method. It should be compared with, and later combined with, a weather-forecast-based product.

## Out of scope

* Plant dispatch or trading decisions on its own: it cannot foresee storms (see failure cases).
* Other sites without retraining: the "typical" patterns are learnt for one grid cell near Ibri.
* Plant output in MW is a rough conversion (rated output at 1,000 W/m² of horizontal sunlight, linear below), not a plant model: it ignores tilt, tracking, panel temperature, inverter limits and soiling. It implies a 25% capacity factor against about 36.5% expected for Ibri II, so it understates a real plant by about a third (docs/verification.md, layer 4).

## Evaluation

Scores are over daylight hours. RMSE = root mean square error. Skill = share of the "tomorrow = today" error removed.

| Forecast | Validation 2023 RMSE | Test 2024–2025 RMSE | Test skill | Test MAE | Test MAE, 500 MW plant | Test daily energy error |
|---|---:|---:|---:|---:|---:|---:|
| Tomorrow = today | 72.2 | 77.9 | 0% | 40.4 W/m² | 20.1 MW | 0.43 kWh/m² |
| Typical day for the season | 68.0 | 76.6 | +2% | 42.5 W/m² | 21.1 MW | 0.49 kWh/m² |
| **Ridge (chosen)** | **61.7** | **67.0** | **+14%** | **37.0 W/m²** | **18.4 MW** | **0.40 kWh/m²** |
| Gradient boosting | 62.3 | 66.1 | +15% | 36.6 W/m² | 18.2 MW | 0.40 kWh/m² |

Ridge was chosen on 2023 because its validation error was lower. On the test years gradient boosting is 0.8 W/m² better; the two are effectively tied, and the simpler model was kept rather than switching after seeing test results.

### What the model learnt

Standardised weights of the ridge model (target: tomorrow's clearness):

| Feature | Weight | Reading |
|---|---:|---|
| Clearness yesterday at the same hour | +0.090 | Weather persists; the strongest signal |
| Sun height (top-of-atmosphere sunlight) | +0.070 | A higher sun passes through less air, so more of it gets through |
| Temperature yesterday | −0.028 | Warmer days tend to be followed by slightly less clear skies |
| Haze yesterday (clear-sky ÷ top-of-atmosphere) | +0.013 | A clean sky yesterday means a clearer sky tomorrow |
| Season (sine) | −0.012 | Seasonal pattern |
| Others (daily clearness, humidity, hour, wind, rain, day before) | < 0.01 each | Little extra information |

## Failure cases

1. **Storms are invisible to the model.** The 10 worst test days were all clouds or rain, and 9 of them were over-forecasts: a sunny forecast for a stormy day. The single worst, 2 May 2024, received 2.8 kWh/m² against 6.9 forecast. Of the 50 worst days, 49 were clouds or rain and one dust or haze; the worst 10% of days account for 40% of all daily error. The model only sees today's conditions, so it cannot know a storm is coming. This is the main limitation.
2. **November: worse than "tomorrow = today".** November skies are steady (daily clearness only ranged from 0.60 to 0.74), so yesterday is an excellent guide. The model forecast about 15 W/m² too little at almost every hour, pulling towards the long-term average (RMSE 24.7 vs 19.7 W/m² for persistence).
3. **Late winter and spring are hardest.** February and March have the largest errors (model RMSE 106 and 120 W/m²), when frontal systems bring cloud and rain to northern Oman.

## Limitations

* Satellite estimates for a ~100 km cell, not plant measurements (data card).
* The "likely cause" labels for bad days are a rule of thumb from the data (rain, clear-sky vs actual, haze vs the monthly norm), not observed weather reports.
* One location, one model, one fixed forecast time (end of day).
* Plant MW is an approximate conversion that understates a real plant by about a third.
* NASA's data are about 7% below PVGIS (another satellite product) for the same years, and NASA's annual sunlight falls from 2,244 kWh/m² (2015) to 2,105 (2024); absolute energy figures are likely conservative, and the seasonal baseline over-forecasts the test years.

## Next improvements

1. **Add weather forecasts** (for example from Open-Meteo) as inputs; they are what makes a day-ahead forecast see storms coming.
2. **Train on plant measurements** if an operator shares them, so the forecast covers soiling, temperature and tracking effects.
3. **Give a range, not a single number** (for example 10th–90th percentile), so planners can hold the right amount of reserve.

## Resource needs

All steps run on a laptop: the full run (features, both models, evaluation, figures) took about 10 seconds on 2 CPUs; the model is 2 KB; a forecast takes well under a second.
