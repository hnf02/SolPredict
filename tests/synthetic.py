import numpy as np
import pandas as pd

HEADER = """-BEGIN HEADER-
NASA/POWER Source Native Resolution Hourly Data
Dates (month/day/year): {start} through {end} in LST
Location: latitude  23.23   longitude 56.52
The value for missing source data that cannot be computed or is outside of the sources availability range: -999
Parameter(s):
ALLSKY_SFC_SW_DWN     CERES SYN1deg All Sky Surface Shortwave Downward Irradiance (Wh/m^2)
-END HEADER-
"""


def toa(day_of_year: np.ndarray, hour: np.ndarray, lat: float = 23.23) -> np.ndarray:
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    hour_angle = np.radians(15 * (hour + 0.5 - 12))
    phi = np.radians(lat)
    cos_z = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    return np.clip(1361 * (1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)) * cos_z, 0, None)


def make_text(days: int = 60, start: str = "2023-01-01", seed: int = 0, missing: int = 3) -> str:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    doy, hour = idx.dayofyear.to_numpy(), idx.hour.to_numpy()
    top = toa(doy, hour)
    k_day = np.empty(days)
    k_day[0] = 0.7
    for i in range(1, days):                               
        k_day[i] = np.clip(0.7 + 0.6 * (k_day[i - 1] - 0.7) + rng.normal(0, 0.08), 0.15, 0.85)
    k = np.repeat(k_day, 24) * (1 + rng.normal(0, 0.03, len(idx)))
    haze = np.repeat(np.clip(rng.normal(0.78, 0.04, days), 0.6, 0.9), 24)
    df = pd.DataFrame({
        "YEAR": idx.year, "MO": idx.month, "DY": idx.day, "HR": idx.hour,
        "ALLSKY_SFC_SW_DWN": np.round(top * np.minimum(k, haze), 2),
        "CLRSKY_SFC_SW_DWN": np.round(top * haze, 2),
        "TOA_SW_DWN": np.round(top, 2),
        "T2M": np.round(25 + 10 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1, len(idx)), 2),
        "RH2M": np.round(np.clip(40 + rng.normal(0, 10, len(idx)), 5, 100), 2),
        "WS10M": np.round(np.abs(rng.normal(3, 1, len(idx))), 2),
        "PRECTOTCORR": np.round(np.where(rng.random(len(idx)) < 0.01, rng.random(len(idx)), 0.0), 2),
    })
    for _ in range(missing):                                   
        df.iloc[rng.integers(len(df)), rng.integers(4, df.shape[1])] = -999
    head = HEADER.format(start=idx[0].strftime("%m/%d/%Y"), end=idx[-1].strftime("%m/%d/%Y"))
    return head + df.to_csv(index=False)
