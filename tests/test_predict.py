import json
import shutil

import pandas as pd
import pytest

from solpredict import config
from solpredict.data import read_power_csv
from solpredict.predict import forecast, forecast_file, load_model

SAMPLE_IN = config.ROOT / "samples" / "sample_input.csv"
SAMPLE_OUT = config.ROOT / "samples" / "expected_output.csv"


def test_sample_forecast_matches_expected():
    got = forecast_file(SAMPLE_IN)
    want = pd.read_csv(SAMPLE_OUT)
    assert list(got.columns) == list(want.columns)
    assert (got["date"] == want["date"]).all()
    assert (got["ghi_forecast_wm2"] - want["ghi_forecast_wm2"]).abs().max() <= 0.1


def test_forecast_is_for_the_next_day_and_zero_at_night():
    got = forecast_file(SAMPLE_IN)
    assert len(got) == 24 and (got["date"] == "2026-01-01").all()
    assert (got.loc[got["hour_lst"].isin([0, 1, 2, 3, 22, 23]), "ghi_forecast_wm2"] == 0).all()
    assert got["plant_mw_forecast"].max() <= config.PLANT_MW


def test_tampered_model_is_refused(tmp_path):
    for f in ("model.joblib", "model_meta.json", "toa_table.csv"):
        shutil.copy(config.MODELS_DIR / f, tmp_path / f)
    meta = json.loads((tmp_path / "model_meta.json").read_text())
    meta["sha256"] = "0" * 64
    (tmp_path / "model_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path)


def test_input_without_a_complete_day_is_rejected():
    text = SAMPLE_IN.read_text()
    head, _, rows = text.partition("YEAR,")
    partial = head + "YEAR," + "\n".join(rows.splitlines()[:11])     
    df, _ = read_power_csv(partial)
    with pytest.raises(ValueError, match="complete day"):
        forecast(df)


def test_input_not_in_local_solar_time_is_rejected(tmp_path):
    for label in ("UTC", None):
        text = SAMPLE_IN.read_text()
        text = text.replace(" in LST", " in UTC") if label else text.replace(" in LST", "")
        path = tmp_path / "input.csv"
        path.write_text(text)
        with pytest.raises(ValueError, match="LST"):
            forecast_file(path)
