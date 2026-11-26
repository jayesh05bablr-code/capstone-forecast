"""Beginner safety rules: sizing, the trade check, paper trading and the after-charges track record."""
from datetime import date, datetime

import numpy as np
import pandas as pd

from capstone import guidance, safety


def _prices(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def test_position_size_keeps_loss_at_stop_within_the_limit():
    rules = safety.Rules(capital=50_000, risk_per_trade_pct=1.0, stop_loss_pct=2.0)
    s = safety.position_size(1_000.0, rules)
    assert s.qty > 0 and s.limited_by == "risk"
    assert s.loss_at_stop <= rules.risk_per_trade + 1e-6  # ₹500, charges included
    # one more share would break the limit
    bigger = safety._loss_at_stop(1_000.0, s.qty + 1, 0.02, safety.INTRADAY)
    assert bigger > rules.risk_per_trade


def test_position_size_never_uses_more_than_the_capital_cap():
    rules = safety.Rules(capital=10_000, risk_per_trade_pct=5.0, stop_loss_pct=0.3, max_position_pct=50)
    s = safety.position_size(100.0, rules)
    assert s.value <= 5_000 and s.limited_by == "capital"
    assert safety.position_size(100_000.0, rules).qty == 0  # one share is more than allowed


def test_rules_warn_when_limits_are_loose():
    assert safety.Rules(risk_per_trade_pct=3).problems()
    assert safety.Rules(daily_loss_limit_pct=8).problems()
    assert not safety.Rules().problems()


def _conf(level="High", approx=False):
    return guidance.Confidence(level, 70.0, 500, "0.50-1.00%", approximate=approx)


GOOD_RECORD = {"trades": 100, "wins": 60, "avg_net_pct": 0.1}
READY = {"ready": True, "settled": 25, "net": 1200.0}
NEW = {"ready": False, "settled": 0, "net": 0.0}


def test_check_says_no_when_move_does_not_clearly_beat_costs():
    v = safety.should_i_trade(0.30, _conf(), 0.18, GOOD_RECORD, READY, 0, safety.Rules())
    assert v.answer == "No trade today" and not v.checks[0].ok  # 0.30 < 2 x 0.18


def test_check_never_trusts_approximate_or_weak_confidence_or_losing_record():
    r = safety.Rules()
    assert safety.should_i_trade(0.8, _conf(approx=True), 0.18, GOOD_RECORD, READY, 0, r).tone == "stop"
    assert safety.should_i_trade(0.8, _conf("Low"), 0.18, GOOD_RECORD, READY, 0, r).tone == "stop"
    losing = {"trades": 300, "wins": 120, "avg_net_pct": -0.15}
    assert safety.should_i_trade(0.8, _conf(), 0.18, losing, READY, 0, r).tone == "stop"
    few = {"trades": 5, "wins": 5, "avg_net_pct": 0.5}
    assert safety.should_i_trade(0.8, _conf(), 0.18, few, READY, 0, r).tone == "stop"


def test_check_sends_beginners_to_paper_first_and_stops_at_the_daily_maximum():
    r = safety.Rules(max_trades_per_day=2)
    assert safety.should_i_trade(0.8, _conf(), 0.18, GOOD_RECORD, NEW, 0, r).answer == "Practise on paper only"
    assert safety.should_i_trade(0.8, _conf(), 0.18, GOOD_RECORD, READY, 0, r).tone == "go"
    assert safety.should_i_trade(0.8, _conf(), 0.18, GOOD_RECORD, READY, 2, r).answer == "Stop for today"


def test_signal_trades_enter_at_next_open_and_pay_costs():
    prices = {"A": _prices([("2026-01-01", 100, 101, 99, 100), ("2026-01-02", 101, 103, 100, 102),
                            ("2026-01-05", 102, 102, 99, 100)])}
    sig = pd.DataFrame({"symbol": ["A", "A"], "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                        "pred_gap_pct": [0.5, -0.4]})
    t = safety.signal_trades(sig, prices, cost_pct=0.2)
    # 01-01 forecast up -> buy 01-02 open 101, close 102; 01-02 forecast down -> short 01-05 open 102, close 100
    assert list(t["side"]) == ["BUY", "SELL"]
    assert np.isclose(t["gross_pct"].iloc[0], (102 / 101 - 1) * 100)
    assert np.isclose(t["net_pct"].iloc[1], (1 - 100 / 102) * 100 - 0.2)
    assert len(safety.signal_trades(sig, prices, 0.2, min_move_pct=0.45)) == 1
    s = safety.track_summary(t, 10_000)
    assert s["trades"] == 2 and s["wins"] == 2


def test_signal_trades_skip_holes_in_the_price_history():
    prices = {"A": _prices([("2023-01-02", 100, 101, 99, 100), ("2023-03-01", 120, 121, 119, 121)])}
    sig = pd.DataFrame({"symbol": ["A"], "date": pd.to_datetime(["2023-01-02"]), "pred_gap_pct": [0.5]})
    assert safety.signal_trades(sig, prices, 0.2).empty


def test_paper_orders_fill_only_at_an_open_still_ahead():
    # Wed 25 Nov 2026 before 09:15 -> that day's open; after 09:15 -> the next trading day's
    assert safety.paper_trade_date(datetime(2026, 11, 25, 9, 0)) == date(2026, 11, 25)
    assert safety.paper_trade_date(datetime(2026, 11, 25, 11, 0)) > date(2026, 11, 25)


def test_paper_trade_settles_at_close_or_stop_with_charges():
    rules = safety.Rules(capital=100_000, risk_per_trade_pct=2, daily_loss_limit_pct=5, stop_loss_pct=1)
    j = safety.empty_journal()
    now = datetime(2026, 1, 1, 20, 0)
    j = safety.new_trade(j, now, date(2026, 1, 2), "A", "BUY", 10, 1.0, 100.0)
    j = safety.new_trade(j, now, date(2026, 1, 2), "B", "BUY", 10, 1.0, 100.0)
    prices = {"A": _prices([("2026-01-02", 100, 103, 99.5, 102)]),    # no stop: exit at close 102
              "B": _prices([("2026-01-02", 100, 100.5, 98, 99.8)])}   # low 98 hits stop at 99
    s = safety.settle(j, prices)
    a, b = s.iloc[0], s.iloc[1]
    assert a["status"] == "settled" and a["exit"] == 102 and np.isclose(a["gross"], 20)
    assert b["exit_reason"] == "stop-loss hit" and np.isclose(b["exit"], 99) and np.isclose(b["gross"], -10)
    assert a["charges"] > 0 and np.isclose(a["net"], a["gross"] - a["charges"])
    summ = safety.paper_summary(s)
    assert summ["settled"] == 2 and not summ["ready"]
    assert np.isclose(summ["balance"], safety.PAPER_START + a["net"] + b["net"])


def test_paper_limits_block_overtrading_and_oversized_orders():
    rules = safety.Rules(capital=50_000, risk_per_trade_pct=1, daily_loss_limit_pct=2, stop_loss_pct=1,
                         max_trades_per_day=2)
    d = date(2026, 1, 2)
    j = safety.empty_journal()
    ok, _ = safety.can_place(j, rules, d, "A", 20, 1_000.0, 100_000)
    assert ok
    ok, why = safety.can_place(j, rules, d, "A", 200, 1_000.0, 1_000_000)  # ₹2 lakh, over the 50% cap
    assert not ok
    j = safety.new_trade(j, datetime(2026, 1, 1, 20), d, "A", "BUY", 20, 1.0, 1_000.0)
    assert not safety.can_place(j, rules, d, "A", 10, 1_000.0, 100_000)[0]  # same stock twice
    j = safety.new_trade(j, datetime(2026, 1, 1, 20), d, "B", "BUY", 20, 1.0, 1_000.0)
    ok, why = safety.can_place(j, rules, d, "C", 10, 1_000.0, 100_000)
    assert not ok and "daily maximum" in why


def test_confidence_falls_back_to_all_forecasts_and_says_approximate():
    rng = np.random.default_rng(1)
    n = 400
    actual = rng.choice([-1, 1], n) * 0.004
    pred = np.where(rng.random(n) < 0.55, np.sign(actual), -np.sign(actual)) * 0.0005  # all in "under 0.10%"
    table = guidance.confidence_table(pd.DataFrame({"model": "forecast", "pred": pred, "actual": actual}))
    c = guidance.confidence_for(1.26, table)  # no past forecasts over 1%
    assert c.approximate and c.level != "Unknown" and c.bucket_cases == 0
    assert "Approximate" in c.sentence and "(approximate)" in c.label
    a = guidance.advice("INFY", 1.26, 101.26, 100, 102, c, 0.18)
    assert a.verdict == "Unclear direction"  # an approximate rate never backs a direction call
    exact = guidance.confidence_for(0.05, table)
    assert not exact.approximate and exact.cases == n
