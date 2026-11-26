"""Scalping lab: signals, fills, costs, walk-forward and paper scalps (synthetic minutes, no network)."""
import numpy as np
import pandas as pd

from capstone import costs, safety, scalp_lab as sl

RULES = safety.Rules(capital=100_000, risk_per_trade_pct=1.0, max_position_pct=100)


def _day(day: str, path) -> pd.DataFrame:
    t = pd.date_range(f"{day} 09:15", periods=len(path), freq="min")
    c = np.asarray(path, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"datetime": t, "open": o, "high": np.maximum(o, c) + 0.05, "low": np.minimum(o, c) - 0.05,
                         "close": c, "volume": 1000})


def _trend_day(day: str, drift: float, n: int = 375, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _day(day, 1000 + np.cumsum(drift + rng.normal(0, 0.3, n)))


def test_resample_anchors_at_open():
    m = _trend_day("2026-11-02", 0.0)
    b = sl.resample(m, 5)
    assert b["datetime"].iloc[0] == pd.Timestamp("2026-11-02 09:15")
    assert b["datetime"].iloc[1] == pd.Timestamp("2026-11-02 09:20")
    assert b["open"].iloc[0] == m["open"].iloc[0] and b["close"].iloc[0] == m["close"].iloc[4]
    assert len(b) == 75


def test_entry_is_next_bar_open_and_short_mirrors_long():
    b = sl.add_indicators(sl.resample(_trend_day("2026-11-02", 0.0), 1))
    sig = np.zeros(len(b), int)
    sig[30], sig[200] = 1, -1
    p = sl.Params("orb", stop_atr=1.0, rr=1.0, max_hold=10)
    t = sl.simulate(b, sig, p, RULES, symbol="X")
    assert list(t["side"]) == ["BUY", "SELL"]
    assert t["entry"].iloc[0] == b["open"].iloc[31]          # never the signal bar's own close
    assert t["entry_time"].iloc[1] == b["datetime"].iloc[201]
    assert t["stop"].iloc[0] < t["entry"].iloc[0] < t["target"].iloc[0]
    assert t["target"].iloc[1] < t["entry"].iloc[1] < t["stop"].iloc[1]
    assert (t["net"] < t["gross"]).all()                       # charges always paid


def test_stop_counts_before_target_in_one_bar():
    path = [100.0] * 30 + [100.0] * 3
    m = _day("2026-11-02", path)
    m.loc[31, ["high", "low"]] = [130.0, 70.0]  # the bar after entry spans both stop and target
    b = sl.add_indicators(sl.resample(m, 1))
    b["atr"] = 1.0
    sig = np.zeros(len(b), int)
    sig[29] = 1
    t = sl.simulate(b, sig, sl.Params("orb", stop_atr=1.0, rr=1.0), RULES)
    assert t["exit_by"].iloc[0] == "stop" and t["exit"].iloc[0] == t["stop"].iloc[0]


def test_square_off_and_no_late_entries():
    b = sl.add_indicators(sl.resample(_trend_day("2026-11-02", 0.0), 1))
    sig = sl.signals(b, sl.Params("random", (("p", 0.5),)))
    assert not sig[(b["tmin"] >= sl.LAST_ENTRY).to_numpy()].any()
    t = sl.simulate(b, sig, sl.Params("random", (("p", 0.5),), max_hold=500), RULES)
    assert (pd.to_datetime(t["exit_time"]).dt.hour * 60 + pd.to_datetime(t["exit_time"]).dt.minute <= sl.SQUARE_OFF).all()


def test_size_keeps_loss_at_stop_within_limit():
    for entry, stop in ((2500.0, 2490.0), (150.0, 149.2), (1000.0, 1003.0)):
        q = sl.size_for(entry, stop, RULES)
        assert q > 0
        loss = q * abs(entry - stop) + costs.round_trip(q * entry, sl.INTRADAY, exit_value=q * stop)["total"]
        assert loss <= RULES.risk_per_trade + 1e-6
        assert q * entry <= RULES.capital * RULES.max_position_pct / 100


def test_orb_catches_a_trend_day_in_both_directions():
    for drift, side in ((0.4, "BUY"), (-0.4, "SELL")):
        b = sl.add_indicators(sl.resample(_trend_day("2026-11-02", drift), 5))
        t = sl.simulate(b, sl.signals(b, sl.Params("orb", (("or_minutes", 15),))), sl.Params("orb", rr=2.0), RULES)
        assert t["side"].iloc[0] == side and t["gross"].iloc[0] > 0


def test_walk_forward_only_counts_unseen_days():
    days = pd.bdate_range("2026-10-01", periods=14)
    m = pd.concat([_trend_day(d.strftime("%Y-%m-%d"), 0.0, seed=k) for k, d in enumerate(days)], ignore_index=True)
    bars = {"X": sl.add_indicators(sl.resample(m, 5))}
    wf = sl.walk_forward(bars, "orb", RULES, train_days=10)
    assert wf.test_days == 4 and len(wf.chosen) == 4
    assert (pd.to_datetime(wf.trades["day"]) >= days[10]).all() if len(wf.trades) else True
    s = sl.summary(wf.trades)
    assert "verdict" in s and s.get("edge") in (False, True)


def test_summary_needs_enough_trades_and_a_positive_range():
    few = pd.DataFrame({"net": [100.0] * 5, "gross": [120.0] * 5, "net_pct": [0.2] * 5, "gross_pct": [0.24] * 5,
                        "charges": [20.0] * 5, "value": [50_000.0] * 5, "side": ["BUY"] * 5, "day": range(5)})
    assert sl.summary(few)["edge"] is False
    rng = np.random.default_rng(1)
    pct = rng.normal(-0.15, 0.3, 200)
    losing = pd.DataFrame({"net": pct * 500, "gross": (pct + 0.18) * 500, "net_pct": pct, "gross_pct": pct + 0.18,
                           "charges": 90.0, "value": 50_000.0, "side": "SELL", "day": np.arange(200) // 5})
    s = sl.summary(losing)
    assert s["edge"] is False and "no edge" in s["verdict"].lower()


def test_paper_scalp_fills_after_placing_and_respects_limits():
    m = _trend_day("2026-11-02", 0.3)
    j = sl.empty_scalp_journal()
    entry = float(m["close"].iloc[29])
    setup = {"symbol": "X", "side": "BUY", "qty": 10, "ref_entry": entry, "stop": entry - 2, "target": entry + 3}
    j = sl.place_scalp(j, pd.Timestamp("2026-11-02 09:44:30"), setup)
    j = sl.settle_scalps(j, {"X": m})
    r = j.iloc[0]
    assert r["entry_time"] == "2026-11-02 09:45"            # first minute starting after the order
    assert r["entry"] == round(float(m.loc[m["datetime"] == pd.Timestamp("2026-11-02 09:45"), "open"].iloc[0]), 2)
    assert r["status"] == "closed" and r["net"] < r["gross"]
    rules = safety.Rules(capital=100_000, max_trades_per_day=1)
    ok, why = sl.can_place_scalp(j, rules, pd.Timestamp("2026-11-02").date(), "Y", 10, 100.0, 99.0)
    assert not ok and "daily maximum" in why
    ok, _ = sl.can_place_scalp(sl.empty_scalp_journal(), rules, pd.Timestamp("2026-11-02").date(), "Y", 10_000, 100.0, 99.0)
    assert not ok  # over the per-trade loss limit


def test_gate_never_says_go_without_edge():
    paper = {"ready": True}
    assert sl.gate(False, paper, 0, RULES)[0] == "paper"
    assert sl.gate(True, {"ready": False}, 0, RULES)[0] == "paper"
    assert sl.gate(True, paper, RULES.max_trades_per_day, RULES)[0] == "stop"
    assert sl.gate(True, paper, 0, RULES)[0] == "go"
