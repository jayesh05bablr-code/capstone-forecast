"""Today-page guidance, and the data-hole and corporate-action fixes in the feature table."""
from datetime import datetime

import numpy as np
import pandas as pd

from capstone import features, guidance, planner


def _oos(n=200, hit=0.7, size=0.003, seed=0):
    rng = np.random.default_rng(seed)
    actual = rng.choice([-1, 1], n) * 0.004
    right = rng.random(n) < hit
    pred = np.where(right, np.sign(actual), -np.sign(actual)) * size
    return pd.DataFrame({"model": "forecast", "pred": pred, "actual": actual})


def test_confidence_uses_past_hit_rate_for_same_size_forecasts():
    table = guidance.confidence_table(_oos(hit=0.7, size=0.003))
    c = guidance.confidence_for(0.30, table)  # 0.30% falls in the 0.25-0.50% bucket
    assert c.bucket == "0.25-0.50%" and c.cases == 200
    assert 60 < c.hit_rate_pct < 80 and c.level in ("High", "Medium")
    assert "times out of 100" in c.sentence
    # A bucket with no past cases is "Unknown", never a made-up number
    assert guidance.confidence_for(1.5, table).level == "Unknown"


def test_coin_toss_record_is_low_confidence():
    table = guidance.confidence_table(_oos(hit=0.5, size=0.003, n=2000))
    assert guidance.confidence_for(0.3, table).level == "Low"


def test_advice_says_no_edge_when_move_is_below_costs():
    conf = guidance.Confidence("High", 70.0, 500, "0.10-0.25%")
    a = guidance.advice("TCS", 0.12, 100.12, 99.4, 100.8, conf, cost_pct=0.18)
    assert a.verdict == "No clear edge"
    a = guidance.advice("TCS", 0.60, 100.6, 99.9, 101.3, guidance.Confidence("Low", 51.0, 500, "x"), 0.18)
    assert a.verdict == "Unclear direction"
    a = guidance.advice("TCS", -0.60, 99.4, 98.7, 100.1, guidance.Confidence("High", 70.0, 500, "x"), 0.18)
    assert a.verdict == "Likely lower open" and "panic-sell" in a.action


def test_freshness_flags_old_and_preliminary_forecasts():
    now = datetime(2026, 11, 23, 9, 0)
    assert guidance.freshness("2026-11-20", True, now)[0] == "old"
    assert guidance.freshness("2026-11-23", False, now)[0] == "preliminary"
    assert guidance.freshness("2026-11-23", True, now)[0] == "ok"


def test_live_track_record_matches_forecast_to_next_open():
    log = pd.DataFrame({"symbol": ["A", "A"], "as_of_close": ["2026-01-01", "2026-01-02"], "close": [100.0, 101.0],
                        "pred_gap_pct": [0.5, -0.5]})
    prices = {"A": pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]),
                                 "open": [99.0, 101.0, 100.0], "close": [100.0, 101.0, 100.5]})}
    rec = guidance.live_track_record(log, prices)
    # 01-01 forecast +0.5%, next open 101 vs close 100 (+1%): right; 01-02 forecast -0.5%, open 100 vs 101: right
    assert rec["cases"] == 2 and rec["hits"] == 2


def _prices(dates, closes):
    return pd.DataFrame({"date": pd.to_datetime(dates), "open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": 1000})


def test_rows_across_a_data_hole_have_no_target_or_return_features():
    days = list(pd.bdate_range("2022-01-03", periods=30)) + list(pd.bdate_range("2023-06-01", periods=30))
    g = _prices(days, np.linspace(100, 160, 60)).assign(symbol="X")
    out = features._per_symbol(g).reset_index(drop=True)
    assert np.isnan(out.loc[29, "target_gap"])  # last row before the hole: next "open" is months later
    assert out.loc[30:50, "ret_20"].isna().all()  # 20-day features after the hole span it
    assert out.loc[28, "target_gap"] == out.loc[28, "target_gap"]  # normal rows keep their target


def test_planner_backtest_subtracts_costs():
    p = _prices(pd.bdate_range("2024-01-01", periods=80), np.linspace(100, 120, 80))
    gross = planner.backtest(p, 1.5, 2.0, 1)
    net = planner.backtest(p, 1.5, 2.0, 1, cost=0.004)
    assert np.isclose(gross["avg_ret"] - net["avg_ret"], 0.004)
