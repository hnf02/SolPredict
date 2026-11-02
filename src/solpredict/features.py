from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

FEATURES = ["hour", "season_sin", "season_cos", "toa", "k_same_hour", "k_day", "k_day_before",
            "haze_day", "temp", "humidity", "wind", "rain"]


def toa_table(df: pd.DataFrame) -> pd.DataFrame:
    toa = df["TOA_SW_DWN"]
    t = toa.groupby([toa.index.month, toa.index.day, toa.index.hour]).mean()
    t.index.names = ["month", "day", "hour"]
    t = t.rename("toa").reset_index().sort_values(["month", "day", "hour"])
    t["toa"] = t.groupby("hour")["toa"].transform(lambda s: s.interpolate(limit_direction="both"))
    return t.reset_index(drop=True)


def lookup_toa(dates: pd.DatetimeIndex, table: pd.DataFrame) -> np.ndarray:
    by_day = {(m, d): g.set_index("hour")["toa"].reindex(range(24)).to_numpy()
              for (m, d), g in table.groupby(["month", "day"])}
    out = np.zeros((len(dates), 24))
    for i, d in enumerate(dates):
        out[i] = by_day.get((d.month, d.day), by_day.get((2, 28)))     
    return out


def _daily(df: pd.DataFrame) -> pd.DataFrame:
    day = df.index.normalize()
    g = df.groupby(day)
    sun = g["TOA_SW_DWN"].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({
            "k_day": g["ALLSKY_SFC_SW_DWN"].sum(min_count=20) / sun,
            "haze_day": g["CLRSKY_SFC_SW_DWN"].sum(min_count=20) / sun,
            "temp": g["T2M"].mean(),
            "humidity": g["RH2M"].mean(),
            "wind": g["WS10M"].mean(),
            "rain": g["PRECTOTCORR"].sum(min_count=20),
        })
    out.index.name = "date"
    return out


def _hourly_matrix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col]
    m = s.groupby([s.index.normalize(), s.index.hour]).mean().unstack()
    return m.reindex(columns=range(24))


def _rows_for(d: pd.Timestamp, daily, ghi, toa_obs, toa_next: np.ndarray) -> list[dict]:
    t = d + pd.Timedelta(days=1)
    prev = d - pd.Timedelta(days=1)
    doy = t.dayofyear / 365.25 * 2 * np.pi
    with np.errstate(invalid="ignore", divide="ignore"):
        k_same = np.where(toa_obs.loc[d] > config.DAYLIGHT_TOA, ghi.loc[d] / toa_obs.loc[d], np.nan)
    base = {
        "date": t, "season_sin": np.sin(doy), "season_cos": np.cos(doy),
        "k_day": daily.at[d, "k_day"],
        "k_day_before": daily.at[prev, "k_day"] if prev in daily.index else np.nan,
        "haze_day": daily.at[d, "haze_day"], "temp": daily.at[d, "temp"],
        "humidity": daily.at[d, "humidity"], "wind": daily.at[d, "wind"], "rain": daily.at[d, "rain"],
    }
    return [{**base, "hour": h, "toa": toa_next[h], "k_same_hour": k_same[h], "ghi_yesterday": ghi.at[d, h]}
            for h in range(24)]


def _finish(ex: pd.DataFrame) -> pd.DataFrame:
    ex["daylight"] = ex["toa"] > config.DAYLIGHT_TOA
    if "ghi" in ex:
        with np.errstate(invalid="ignore", divide="ignore"):
            ex["k"] = np.where(ex["daylight"], ex["ghi"] / ex["toa"], np.nan)
    return ex


def build_examples(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    daily = _daily(df)
    ghi = _hourly_matrix(df, "ALLSKY_SFC_SW_DWN")
    toa_obs = _hourly_matrix(df, "TOA_SW_DWN")
    days = ghi.index
    toa_next = lookup_toa(days + pd.Timedelta(days=1), table)
    rows = []
    for i, d in enumerate(days):
        t = d + pd.Timedelta(days=1)
        if t not in ghi.index:
            continue
        for r in _rows_for(d, daily, ghi, toa_obs, toa_next[i]):
            r["ghi"] = ghi.at[t, r["hour"]]
            rows.append(r)
    return _finish(pd.DataFrame(rows))


def next_day_examples(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    daily = _daily(df)
    ghi = _hourly_matrix(df, "ALLSKY_SFC_SW_DWN")
    toa_obs = _hourly_matrix(df, "TOA_SW_DWN")
    complete = ghi.index[ghi.notna().sum(axis=1) >= 20]
    if len(complete) == 0:
        raise ValueError("input needs at least one complete day (20+ hourly values)")
    d = complete[-1]
    toa_next = lookup_toa(pd.DatetimeIndex([d + pd.Timedelta(days=1)]), table)[0]
    return _finish(pd.DataFrame(_rows_for(d, daily, ghi, toa_obs, toa_next)))


def baselines(ex: pd.DataFrame, train: pd.Series) -> pd.DataFrame:
    month = ex["date"].dt.month
    typical = ex[train & ex["daylight"]].groupby([month[train], "hour"])["k"].mean()
    k = pd.Series([typical.get((m, h), np.nan) for m, h in zip(month, ex["hour"])], index=ex.index)
    return pd.DataFrame({
        "persistence": ex["ghi_yesterday"].fillna(0.0),
        "seasonal": np.where(ex["daylight"], (k * ex["toa"]).fillna(0.0), 0.0),
    }, index=ex.index)
