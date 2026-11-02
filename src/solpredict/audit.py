from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import config, plots
from .data import MANIFEST, check
from .evaluate import plant_mw, split
from .features import FEATURES, baselines, build_examples, toa_table
from .models import fit, predict_ghi

# Independent references (see docs/verification.md for links)
PVGIS_IBRI_TILTED_25 = 2547.62     # kWh/m2/year, fixed panel, 25 deg tilt, facing south (PVGIS-SARAH 2005-2016)
IBRI_II_EXPECTED_GWH = 1600.0      # GWh/year expected from the 500 MW Ibri II plant (Oman Observer)


def layer1(df: pd.DataFrame) -> dict:
    c = check(df)
    lst_noon_offset_min = round((60.0 - config.SITE["longitude"]) * 4, 1)   # Oman clock time is UTC+4 (60 deg E)
    plots.data_weeks(df, config.RESULTS_DIR / "audit_weeks.png")
    return {**c, "time_standard": "LST (local solar time) per file header",
            "note_on_clock_time": f"Ibri is {60 - config.SITE['longitude']:.2f} deg west of Oman's 60 deg E time meridian, "
                                  f"so solar noon is about {lst_noon_offset_min:.0f} min after 12:00 Oman time "
                                  "(plus up to +/-16 min over the year)"}


def _rmse(f, a):
    return float(np.sqrt(np.mean((np.asarray(f) - np.asarray(a)) ** 2)))


def layer2(df: pd.DataFrame) -> dict:
    table = toa_table(df[df.index <= config.TRAIN_END])
    ex = build_examples(df, table)
    s = split(ex)


    rng = np.random.default_rng(config.SEED)
    days = pd.DatetimeIndex(rng.choice(ex["date"].unique()[2:-2], size=20, replace=False))
    changed = df.copy()
    for d in days:
        changed.loc[d:d + pd.Timedelta(hours=23)] *= 3.0
    ex2 = build_examples(changed, table)
    rows = ex["date"].isin(days)
    feats_same = bool(np.allclose(ex.loc[rows, FEATURES].to_numpy(dtype=float),
                                  ex2.loc[rows, FEATURES].to_numpy(dtype=float), equal_nan=True))
    target_changed = bool(not np.allclose(ex.loc[rows, "ghi"], ex2.loc[rows, "ghi"]))


    dates = {k: (str(ex.loc[v, "date"].min().date()), str(ex.loc[v, "date"].max().date())) for k, v in s.items()}
    ordered = ex.loc[s["train"], "date"].max() < ex.loc[s["valid"], "date"].min() and \
        ex.loc[s["valid"], "date"].max() < ex.loc[s["test"], "date"].min()

  
    tr = ex[s["train"]].copy()
    mask = tr["daylight"] & tr["k"].notna()
    tr.loc[mask, "k"] = rng.permutation(tr.loc[mask, "k"].to_numpy())
    va = s["valid"] & ex["daylight"] & ex["ghi"].notna()
    real = _rmse(predict_ghi(fit("ridge", ex[s["train"]]), ex)[va], ex.loc[va, "ghi"])
    shuffled = _rmse(predict_ghi(fit("ridge", tr), ex)[va], ex.loc[va, "ghi"])
    persist = _rmse(ex.loc[va, "ghi_yesterday"], ex.loc[va, "ghi"])
    seasonal = _rmse(baselines(ex, s["train"])["seasonal"][va], ex.loc[va, "ghi"])
    return {
        "perturbed_forecast_days": len(days),
        "features_unchanged_when_forecast_day_changes": feats_same,
        "target_did_change": target_changed,
        "split_dates": dates,
        "split_is_chronological": bool(ordered),
        "validation_rmse": {"real_model": round(real, 2), "model_trained_on_shuffled_targets": round(shuffled, 2),
                            "persistence": round(persist, 2), "seasonal": round(seasonal, 2)},
        "shuffled_model_no_better_than_seasonal": bool(shuffled >= seasonal - 1.0),
    }


def _tilted(df: pd.DataFrame, tilt_deg: float = 25.0, albedo: float = 0.25) -> pd.Series:
    t = df.index
    doy = t.dayofyear.to_numpy()
    ghi, toa = df["ALLSKY_SFC_SW_DWN"].to_numpy(), df["TOA_SW_DWN"].to_numpy()
    e0 = 1361.0 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))
    cos_z = np.clip(toa / e0, 0, 1)                                   
    with np.errstate(invalid="ignore", divide="ignore"):
        kt = np.where(toa > 1, ghi / toa, 0)
    fd = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.80, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                           0.165))
    dhi = fd * ghi
    bh = ghi - dhi
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365)
    omega = np.radians(15 * (t.hour.to_numpy() + 0.5 - 12))
    phi, beta = np.radians(config.SITE["latitude"]), np.radians(tilt_deg)
    cos_theta = (np.sin(decl) * np.sin(phi - beta) + np.cos(decl) * np.cos(phi - beta) * np.cos(omega))
    rb = np.where(cos_z > 0.087, np.clip(cos_theta, 0, None) / np.maximum(cos_z, 0.087), 0)   
    poa = bh * rb + dhi * (1 + np.cos(beta)) / 2 + ghi * albedo * (1 - np.cos(beta)) / 2
    return pd.Series(np.clip(poa, 0, None), index=t)


def layer4(df: pd.DataFrame) -> dict:
    metrics = json.loads((config.RESULTS_DIR / "metrics.json").read_text())
    by_month = pd.read_csv(config.RESULTS_DIR / "error_by_month.csv", index_col=0)
    table = toa_table(df[df.index <= config.TRAIN_END])
    ex = build_examples(df, table)
    s = split(ex)
    model = fit(metrics["chosen_model"], ex[s["train"] | s["valid"]])
    f = predict_ghi(model, ex)
    sub = ex[s["test"] & ex["daylight"]].assign(f=f[(s["test"] & ex["daylight"]).to_numpy()])
    day = sub.groupby("date").agg(actual=("ghi", "sum"), forecast=("f", "sum"))
    day["abs_err"] = (day["forecast"] - day["actual"]).abs() / 1000
    obs = df.groupby(df.index.normalize()).agg(rain=("PRECTOTCORR", "sum"), humidity=("RH2M", "mean"),
                                               clear=("CLRSKY_SFC_SW_DWN", "sum"), sun=("TOA_SW_DWN", "sum"),
                                               ghi=("ALLSKY_SFC_SW_DWN", "sum"))
    day = day.join((obs.assign(clearness=obs["ghi"] / obs["sun"], haze=obs["clear"] / obs["sun"])
                    [["clearness", "haze", "rain", "humidity"]]))
    best, worst = day.nsmallest(3, "abs_err"), day.nlargest(3, "abs_err")
    plots.best_worst(sub, best.index, worst.index, config.RESULTS_DIR / "audit_best_worst.png")

    # Independent cross-checks
    years = df[df.index.year <= 2025]
    ghi_year = years["ALLSKY_SFC_SW_DWN"].groupby(years.index.year).sum() / 1000
    tilted_year = _tilted(years).groupby(years.index.year).sum() / 1000
    sarah = tilted_year.loc[2015:2016].mean()      # the only years shared with PVGIS-SARAH (2005-2016)
    cf_simple = float(plant_mw(years["ALLSKY_SFC_SW_DWN"]).mean() / config.PLANT_MW)
    cf_ibri2 = IBRI_II_EXPECTED_GWH * 1000 / (config.PLANT_MW * 8760)
    month_ok = bool((by_month["model_rmse"] < by_month["persistence_rmse"]).sum() >= 10)
    return {
        "months_where_model_beats_persistence": int((by_month["model_rmse"] < by_month["persistence_rmse"]).sum()),
        "worst_month": int(by_month["model_rmse"].idxmax()), "best_month": int(by_month["model_rmse"].idxmin()),
        "model_beats_persistence_in_at_least_10_months": month_ok,
        "best_days": best.round(3).reset_index().astype({"date": str}).to_dict("records"),
        "worst_days": worst.round(3).reset_index().astype({"date": str}).to_dict("records"),
        "annual_ghi_kwh_m2": {int(k): round(float(v), 0) for k, v in ghi_year.items()},
        "annual_tilted_25deg_kwh_m2_nasa": {int(k): round(float(v), 0) for k, v in tilted_year.items()},
        "tilted_25deg_2015_2016_mean_nasa": round(float(sarah), 0),
        "tilted_25deg_2015_2025_mean_nasa": round(float(tilted_year.mean()), 0),
        "tilted_25deg_pvgis_sarah_2005_2016": PVGIS_IBRI_TILTED_25,
        "nasa_vs_pvgis_percent_overlap_years": round(100 * (sarah / PVGIS_IBRI_TILTED_25 - 1), 1),
        "nasa_vs_pvgis_percent_all_years": round(100 * (tilted_year.mean() / PVGIS_IBRI_TILTED_25 - 1), 1),
        "capacity_factor_simple_conversion": round(cf_simple, 3),
        "capacity_factor_ibri_ii_expected": round(cf_ibri2, 3),
    }


def layer5(df: pd.DataFrame) -> dict:
    from .evaluate import run
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [config.RESULTS_DIR / "metrics.json", config.RESULTS_DIR / "summary.csv",
                        config.MODELS_DIR / "model.joblib"]}
    run(df, verbose=False)
    after = {name: hashlib.sha256((config.RESULTS_DIR if name != "model.joblib" else config.MODELS_DIR)
                                  .joinpath(name).read_bytes()).hexdigest() for name in before}
    manifest = json.loads(MANIFEST.read_text())
    return {"second_run_identical": before == after, "files_compared": list(before),
            "seed": config.SEED, "data_sha256": manifest["sha256_uncompressed_csv"],
            "data_downloaded": manifest.get("downloaded"), "data_url": manifest.get("source_url")}


def run_audit(df: pd.DataFrame) -> dict:
    out = {"layer1_data": layer1(df), "layer2_leakage": layer2(df), "layer4_results": layer4(df),
           "layer5_reproducibility": layer5(df)}
    (config.RESULTS_DIR / "audit.json").write_text(json.dumps(out, indent=2) + "\n")
    return out
