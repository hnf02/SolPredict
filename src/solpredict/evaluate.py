from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from . import config, plots
from .data import check
from .features import baselines, build_examples, toa_table
from .models import fit, predict_ghi

MODELS = ["ridge", "boosting"]
BASELINES = ["persistence", "seasonal"]


def plant_mw(ghi: np.ndarray) -> np.ndarray:
    return config.PLANT_MW * np.clip(np.asarray(ghi) / config.PLANT_FULL_OUTPUT_GHI, 0, 1)


def scores(actual, forecast, dates, daylight) -> dict:
    a, f = np.asarray(actual)[daylight], np.asarray(forecast)[daylight]
    err = f - a
    per_day = pd.DataFrame({"date": np.asarray(dates)[daylight], "a": a, "f": f}).groupby("date").sum()
    return {
        "mae_wm2": round(float(np.mean(np.abs(err))), 2),
        "rmse_wm2": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "bias_wm2": round(float(np.mean(err)), 2),
        "mae_mw": round(float(np.mean(np.abs(plant_mw(f) - plant_mw(a)))), 2),
        "daily_energy_mae_kwh_m2": round(float(np.mean(np.abs(per_day["f"] - per_day["a"]))) / 1000, 3),
        "hours": int(daylight.sum()),
        "days": int(len(per_day)),
    }


def skill(rmse: float, reference_rmse: float) -> float:
    return round(1 - rmse / reference_rmse, 4)


def split(ex: pd.DataFrame) -> dict[str, pd.Series]:
    d = ex["date"]
    return {"train": d <= config.TRAIN_END,
            "valid": (d > config.TRAIN_END) & (d <= config.VALID_END),
            "test": d > config.VALID_END}


def _score_all(ex, forecasts: dict, mask) -> dict:
    sub = ex[mask]
    ok = sub["ghi"].notna() & sub["daylight"]
    out = {name: scores(sub["ghi"][ok], f[mask][ok], sub["date"][ok], np.ones(ok.sum(), bool))
           for name, f in forecasts.items()}
    ref = out["persistence"]["rmse_wm2"]
    for name in out:
        out[name]["skill_vs_persistence"] = skill(out[name]["rmse_wm2"], ref)
    return out


def run(df: pd.DataFrame, verbose: bool = True) -> dict:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    checks = check(df)
    (config.RESULTS_DIR / "data_checks.json").write_text(json.dumps(checks, indent=2) + "\n")

    table = toa_table(df[df.index <= config.TRAIN_END])
    ex = build_examples(df, table)
    s = split(ex)
    if not s["valid"].any() or not s["test"].any():
        raise ValueError("data must cover the validation (2023) and test (2024 onwards) periods")
    base = baselines(ex, s["train"])

  
    forecasts = {b: base[b].to_numpy() for b in BASELINES}
    for m in MODELS:
        forecasts[m] = predict_ghi(fit(m, ex[s["train"]]), ex)
    valid = _score_all(ex, forecasts, s["valid"])
    chosen = min(MODELS, key=lambda m: valid[m]["rmse_wm2"])


    trainval = s["train"] | s["valid"]
    final = {}
    for m in MODELS:
        final[m] = fit(m, ex[trainval])
        forecasts[m] = predict_ghi(final[m], ex)
    test = _score_all(ex, forecasts, s["test"])


    model_path = config.MODELS_DIR / "model.joblib"
    joblib.dump(final[chosen], model_path)
    table.to_csv(config.MODELS_DIR / "toa_table.csv", index=False)
    meta = {"model": chosen, "trained_on": f"forecast days {ex['date'].min():%Y-%m-%d} to {config.VALID_END}",
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "site": config.SITE,
            "plant_mw": config.PLANT_MW}
    (config.MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    analysis = error_analysis(df, ex, forecasts[chosen], s["test"])
    out = {"chosen_model": chosen, "validation_2023": valid, "test_2024_2025": test,
           "rows": {k: int(v.sum()) for k, v in s.items()}, "data_checks": checks}
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    summary = pd.DataFrame([{"period": p, "forecast": n, **v} for p, res in
                            (("validation_2023", valid), ("test_2024_2025", test)) for n, v in res.items()])
    summary.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    plots.skill_bars(test, config.RESULTS_DIR / "test_scores.png")
    plots.by_month(analysis["by_month"], chosen, config.RESULTS_DIR / "error_by_month.png")
    plots.example_days(ex, forecasts[chosen], base["persistence"].to_numpy(), analysis["worst_days"],
                       s["test"], chosen, config.RESULTS_DIR / "example_days.png")
    if verbose:
        print(f"chosen on validation: {chosen}")
        print(summary.to_string(index=False))
    return out


def error_analysis(df, ex, forecast, test_mask) -> dict:
    sub = ex[test_mask & ex["daylight"] & ex["ghi"].notna()].copy()
    sub["f"] = np.asarray(forecast)[sub.index]
    sub["persist"] = sub["ghi_yesterday"]
    sub["month"] = sub["date"].dt.month
    by_month = sub.groupby("month").apply(lambda g: pd.Series({
        "model_rmse": np.sqrt(np.mean((g["f"] - g["ghi"]) ** 2)),
        "persistence_rmse": np.sqrt(np.mean((g["persist"] - g["ghi"]) ** 2)),
    }), include_groups=False).round(1)
    by_month.to_csv(config.RESULTS_DIR / "error_by_month.csv")


    day = sub.groupby("date").agg(actual=("ghi", "sum"), forecast=("f", "sum"))
    obs = df.groupby(df.index.normalize()).agg(
        sun=("TOA_SW_DWN", "sum"), ghi=("ALLSKY_SFC_SW_DWN", "sum"), clear=("CLRSKY_SFC_SW_DWN", "sum"),
        rain=("PRECTOTCORR", "sum"), humidity=("RH2M", "mean"), wind=("WS10M", "mean"))
    obs["clearness"] = obs["ghi"] / obs["sun"]
    obs["haze"] = obs["clear"] / obs["sun"]
    typical_haze = obs["haze"].groupby(obs.index.month).median()
    day = day.join(obs[["clearness", "haze", "rain", "humidity", "wind"]])
    day["error_kwh_m2"] = (day["forecast"] - day["actual"]) / 1000
    cloud_loss = day["haze"] - day["clearness"]
    dust_loss = day.index.month.map(typical_haze).to_numpy() - day["haze"]
    day["likely_cause"] = np.where(day["rain"] > 0.5, "rain / clouds",
                                   np.where(cloud_loss > dust_loss, "clouds", "dust / haze"))
    worst = day.reindex(day["error_kwh_m2"].abs().sort_values(ascending=False).index).head(10).round(3)
    worst.to_csv(config.RESULTS_DIR / "worst_days.csv")
    return {"by_month": by_month, "worst_days": worst}
