"""NSE market inputs for the model (no network)."""
import numpy as np
import pandas as pd

from capstone import features, nse_market


def _daily():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-23"]),
        "nifty_open": [np.nan, 101.0, 99.0, 103.0], "nifty_close": [np.nan, 100.0, 102.0, 101.0],
        "vix_close": [np.nan, 15.0, 16.5, 15.0],
        "advances": [1000, 1500, 500, 999], "declines": [1000, 500, 1500, 999]})


def test_features_use_only_that_days_close():
    f = nse_market.features(_daily()).set_index("date")
    d = pd.Timestamp("2026-11-20")
    assert abs(f.loc[d, "nifty_ret_1"] - (102 / 100 - 1)) < 1e-12
    assert abs(f.loc[d, "nifty_gap_today"] - (99 / 100 - 1)) < 1e-12  # that morning's open vs prior close
    assert abs(f.loc[d, "vix_chg_1"] - 0.1) < 1e-12 and f.loc[d, "vix_level"] == 16.5
    assert f.loc[d, "breadth_log_ad"] < 0 < f.loc[pd.Timestamp("2026-11-19"), "breadth_log_ad"]
    assert f.loc[pd.Timestamp("2026-11-23"), "breadth_log_ad"] == 0.0
    # Before NSE's index history starts, the index inputs are missing (not zero); breadth still counts.
    first = f.loc[pd.Timestamp("2026-11-18")]
    assert np.isnan(first["nifty_ret_1"]) and np.isnan(first["vix_level"]) and first["breadth_log_ad"] == 0.0
    assert np.isnan(f.loc[pd.Timestamp("2026-11-19"), "nifty_ret_1"])  # no earlier close to compare with


def test_empty_table_gives_empty_features():
    assert nse_market.features(pd.DataFrame(columns=nse_market.COLUMNS)).empty


def test_attach_nse_without_saved_file(monkeypatch):
    monkeypatch.setattr(nse_market, "features", lambda daily=None: pd.DataFrame(columns=["date", *nse_market.FEATURES]))
    df = features.attach_nse(pd.DataFrame({"date": pd.to_datetime(["2026-11-20"]), "x": [1]}))
    assert list(df.columns[-len(nse_market.FEATURES):]) == nse_market.FEATURES and df[nse_market.FEATURES].isna().all().all()


def test_attach_nse_joins_by_date(monkeypatch):
    monkeypatch.setattr(nse_market, "load", _daily)
    df = features.attach_nse(pd.DataFrame({"date": pd.to_datetime(["2026-11-20", "2026-11-24"]), "x": [1, 2]}))
    assert abs(df.loc[0, "nifty_ret_1"] - 0.02) < 1e-12 and df.loc[1, nse_market.FEATURES].isna().all()
    assert set(nse_market.FEATURES) <= set(features.FEATURES)


def test_nse_inputs_kept_only_if_they_help(tmp_path, monkeypatch):
    import json

    from capstone import train
    monkeypatch.setattr(train, "NSE_CHOICE_FILE", tmp_path / "nse.json")
    monkeypatch.setattr(train, "MODELS", tmp_path)
    cols = ["ret_1", *nse_market.FEATURES]
    worse = pd.DataFrame({"mape_pct": [0.55, 0.54]}, index=["model", "model_without_nse"])
    assert train.choose_nse_inputs(worse, cols) == ["ret_1"]
    assert json.loads((tmp_path / "nse.json").read_text())["used"] is False
    better = pd.DataFrame({"mape_pct": [0.50, 0.54]}, index=["model", "model_without_nse"])
    assert train.choose_nse_inputs(better, cols) == cols
    assert train.choose_nse_inputs(better.drop("model_without_nse"), cols) == cols
