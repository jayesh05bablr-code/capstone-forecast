"""Metrics definitions, leakage-free baselines and walk-forward folds."""
import numpy as np
import pandas as pd
import pytest

from capstone import metrics, model_eval, train


def test_directional_accuracy_zero_rules():
    actual = [0.01, -0.01, 0.0, 0.02, 0.0]
    pred = [0.02, 0.01, 0.0, 0.0, 0.01]
    # correct: row 0 (up/up), row 2 (zero/zero); wrong: up vs down, zero forecast on a move, up on zero
    assert metrics.directional_accuracy(actual, pred) == pytest.approx(40.0)
    assert metrics.directional_accuracy([0.01, -0.02], [0, 0]) == 0.0  # naive is never right on a move


def test_error_metrics():
    s = metrics.forecast_scores(np.array([0.0, 0.02]), np.array([0.01, 0.0]))
    assert s["mae_pct"] == pytest.approx(1.5)
    assert s["rmse_pct"] == pytest.approx(np.sqrt((0.01**2 + 0.02**2) / 2) * 100)
    assert s["bias_pct"] == pytest.approx(-0.5)


def test_max_drawdown_and_sharpe():
    assert metrics.max_drawdown([100, 120, 90, 130, 117]) == pytest.approx(90 / 120 - 1)
    assert metrics.max_drawdown([100, 110, 120]) == 0.0
    assert np.isnan(metrics.sharpe_ratio([0.01] * 3))


def _panel(n=400, symbols=("AAA", "BBB")):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n)
    rows = []
    for s in symbols:
        rows.append(pd.DataFrame({"date": dates, "symbol": s, "target_gap": rng.normal(0.001, 0.01, n)}))
    return pd.concat(rows, ignore_index=True)


def test_historical_average_uses_only_past_gaps():
    df = _panel()
    hist = model_eval.historical_average_gap(df)
    a = df[df["symbol"] == "AAA"].sort_values("date")
    h = hist[a.index]
    assert np.isnan(h.iloc[0])  # nothing known before the first forecast
    assert h.iloc[10] == pytest.approx(a["target_gap"].iloc[:10].mean())  # excludes its own target
    # Changing a future gap must not change earlier baseline values.
    df2 = df.copy()
    df2.loc[a.index[200], "target_gap"] = 5.0
    assert np.allclose(model_eval.historical_average_gap(df2)[a.index[:201]], h.iloc[:201], equal_nan=True)


def test_folds_are_ordered_and_cover_the_end():
    df = _panel()
    seen = []
    for k, tr, te in train.folds(df, n_folds=3, min_train=100):
        assert tr["date"].max() < te["date"].min()  # training strictly before testing
        seen.append(te)
    allte = pd.concat(seen)
    assert not allte.duplicated(["date", "symbol"]).any()
    assert allte["date"].max() == df["date"].max()  # last fold reaches the final date


def test_diebold_mariano_detects_a_better_model():
    rng = np.random.default_rng(1)
    dates = np.repeat(pd.bdate_range("2021-01-01", periods=300), 2)
    good, bad = rng.uniform(0, 1, 600), rng.uniform(0.5, 1.5, 600)
    assert metrics.diebold_mariano(good, bad, dates)["t_stat"] < -1.96
