from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import read_file
from .evaluate import plant_mw
from .features import next_day_examples
from .models import predict_ghi


def load_model(models_dir: Path = config.MODELS_DIR):
    meta = json.loads((models_dir / "model_meta.json").read_text())
    path = models_dir / "model.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError(f"{path} does not match the checksum in model_meta.json; refusing to use it")
    table = pd.read_csv(models_dir / "toa_table.csv")
    return joblib.load(path), table, meta


def forecast(df: pd.DataFrame, models_dir: Path = config.MODELS_DIR) -> pd.DataFrame:
    model, table, meta = load_model(models_dir)
    ex = next_day_examples(df, table)
    ghi = predict_ghi(model, ex)
    return pd.DataFrame({
        "date": ex["date"].dt.strftime("%Y-%m-%d"),
        "hour_lst": ex["hour"],
        "ghi_forecast_wm2": np.round(ghi, 1),
        "plant_mw_forecast": np.round(plant_mw(ghi), 1),
    })


def forecast_file(path: str | Path, output: str | Path | None = None) -> pd.DataFrame:
    df, standard = read_file(path)
    if standard != "LST":
        raise ValueError(f"{path} uses time standard '{standard}'; the model was trained on LST "
                         "(local solar time). Download the NASA POWER file with time standard LST.")
    out = forecast(df)
    if output:
        out.to_csv(output, index=False)
    return out
