from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURES


def make(name: str):
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if name == "boosting":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=50, random_state=config.SEED)
    raise ValueError(name)


def fit(name: str, ex: pd.DataFrame):
    rows = ex["daylight"] & ex["k"].notna()
    return make(name).fit(ex.loc[rows, FEATURES], ex.loc[rows, "k"])


def predict_ghi(model, ex: pd.DataFrame) -> np.ndarray:
    k = np.clip(model.predict(ex[FEATURES]), 0.0, 1.0)
    return np.where(ex["daylight"], k * ex["toa"], 0.0)
