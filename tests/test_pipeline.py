import numpy as np
import pandas as pd
import pytest

import synthetic
from solpredict import config
from solpredict.data import read_power_csv
from solpredict.evaluate import scores, skill, split
from solpredict.features import FEATURES, baselines, build_examples, next_day_examples, toa_table


@pytest.fixture(scope="module")
def df():
    return read_power_csv(synthetic.make_text(days=40, start="2023-03-01"))[0]


@pytest.fixture(scope="module")
def ex(df):
    return build_examples(df, toa_table(df))


def test_reader_parses_header_time_standard_and_missing_values():
    df, standard = read_power_csv(synthetic.make_text(days=3, missing=2))
    assert standard == "LST"
    assert len(df) == 72 and df.index[0] == pd.Timestamp("2023-01-01 00:00")
    assert df.isna().sum().sum() == 2                    
    assert (df.min(numeric_only=True).fillna(0) > -999).all()


def test_persistence_is_yesterday_same_hour(df, ex):
    row = ex[(ex["date"] == pd.Timestamp("2023-03-10")) & (ex["hour"] == 12)].iloc[0]
    assert baselines(ex, ex["date"].notna()).loc[row.name, "persistence"] == df.loc["2023-03-09 12:00", "ALLSKY_SFC_SW_DWN"]


def test_features_do_not_use_the_forecast_day(df):
    table = toa_table(df)
    t = pd.Timestamp("2023-03-20")
    changed = df.copy()
    changed.loc[t:t + pd.Timedelta(hours=23)] *= 3
    a = build_examples(df, table)
    b = build_examples(changed, table)
    rows_a, rows_b = a[a["date"] == t], b[b["date"] == t]
    pd.testing.assert_frame_equal(rows_a[FEATURES].reset_index(drop=True), rows_b[FEATURES].reset_index(drop=True))
    assert not np.allclose(rows_a["ghi"], rows_b["ghi"])  


def test_live_forecast_rows_match_training_rows(df, ex):
    table = toa_table(df)
    upto = df.loc[:"2023-03-14 23:00"]
    live = next_day_examples(upto, table)
    train = ex[ex["date"] == pd.Timestamp("2023-03-15")]
    pd.testing.assert_frame_equal(live[FEATURES].reset_index(drop=True), train[FEATURES].reset_index(drop=True))


def test_skill_score_edges():
    assert skill(50.0, 50.0) == 0
    assert skill(0.0, 50.0) == 1
    assert skill(40.0, 50.0) == pytest.approx(0.2)


def test_scores_on_a_known_example():
    actual = np.array([100.0, 200.0, 300.0])
    forecast = np.array([110.0, 190.0, 300.0])
    s = scores(actual, forecast, ["d1", "d1", "d2"], np.array([True, True, True]))
    assert s["mae_wm2"] == pytest.approx(6.67, abs=0.01)
    assert s["bias_wm2"] == 0
    assert s["daily_energy_mae_kwh_m2"] == 0               


def test_split_is_chronological():
    ex = pd.DataFrame({"date": pd.date_range("2022-12-25", "2024-01-05", freq="D")})
    s = split(ex)
    assert ex["date"][s["train"]].max() < ex["date"][s["valid"]].min()
    assert ex["date"][s["valid"]].max() < ex["date"][s["test"]].min()
    assert (s["train"] | s["valid"] | s["test"]).all()


def test_night_hours_are_not_scored(ex):
    night = ex[~ex["daylight"]]
    assert (night["toa"] <= config.DAYLIGHT_TOA).all()
    assert night["k"].isna().all()


def test_seasonal_baseline_uses_only_training_days(ex):
    train = ex["date"] < pd.Timestamp("2023-03-20")
    a = baselines(ex, train)["seasonal"]
    changed = ex.copy()
    changed.loc[~train, "k"] = 0.0                       
    b = baselines(changed, train)["seasonal"]
    assert np.allclose(a, b)
