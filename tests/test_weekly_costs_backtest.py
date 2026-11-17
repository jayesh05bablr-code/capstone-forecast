"""Weekly returns and holiday weeks, trading costs, backtest accounting, Monte Carlo."""
import numpy as np
import pandas as pd
import pytest

from capstone import backtest, costs, montecarlo, weekly, weekly_model


def bars(dates, opens, closes):
    return pd.DataFrame({"date": pd.to_datetime(dates), "open": opens, "close": closes})


# --- weekly returns and the trading calendar --------------------------------------------
def test_weekly_return_uses_first_open_and_last_close():
    p = bars(["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"],
             [100, 101, 102, 103, 104], [101, 102, 103, 104, 110])
    w = weekly.weekly_returns(p)
    assert len(w) == 1 and w["valid"].iloc[0]
    assert w["weekly_return"].iloc[0] == pytest.approx(110 / 100 - 1)
    assert w["sessions"].iloc[0] == 5 and not w["holiday_shortened"].iloc[0]


def test_holiday_shortened_week_monday_and_friday_holidays():
    # Monday 9 Nov and Friday 13 Nov are holidays: entry Tuesday open, exit Thursday close.
    p = bars(["2026-11-06", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-16"],
             [90, 200, 205, 210, 300], [95, 204, 209, 220, 301])
    w = weekly.weekly_returns(p, data_end=pd.Timestamp("2026-11-20")).set_index("week_start")
    wk = w.loc[pd.Timestamp("2026-11-09")]
    assert wk["entry_date"] == pd.Timestamp("2026-11-10") and wk["exit_date"] == pd.Timestamp("2026-11-12")
    assert wk["weekly_return"] == pytest.approx(220 / 200 - 1) and wk["holiday_shortened"]


def test_week_in_progress_and_bad_prices_and_corporate_action_excluded():
    p = bars(["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-09", "2026-11-10"],
             [100, 50, 51, 100, 101], [101, 52, 53, 101, 102])
    w = weekly.weekly_returns(p)  # data ends Tuesday 10 Nov: that week is unfinished
    reasons = dict(zip(w["week_start"].dt.strftime("%Y-%m-%d"), w["exclude_reason"]))
    assert reasons["2026-11-02"].startswith("corporate action")  # -50% overnight inside the week
    assert reasons["2026-11-09"] == "week still in progress"
    p.loc[0, "open"] = np.nan
    assert weekly.weekly_returns(p)["exclude_reason"].iloc[0] == "missing entry or exit price"


def test_known_ex_date_excludes_small_bonus():
    p = bars(["2026-11-02", "2026-11-03"], [100, 91], [100, 92])
    assert weekly.weekly_returns(p, ex_dates={pd.Timestamp("2026-11-03")})["valid"].iloc[0] == False  # noqa: E712


def test_trading_calendar_is_union_of_sessions():
    a = bars(["2026-11-02", "2026-11-03"], [1, 1], [1, 1])
    b = bars(["2026-11-03", "2026-11-07"], [1, 1], [1, 1])  # a special Saturday session
    cal = weekly.trading_calendar({"A": a, "B": b})
    assert list(cal.strftime("%Y-%m-%d")) == ["2026-11-02", "2026-11-03", "2026-11-07"]


def test_tradable_stats():
    w = pd.DataFrame({"symbol": "X", "week": ["w1", "w2", "w3", "w4"], "weekly_return": [0.01, 0.002, -0.01, 0.005],
                      "valid": True})
    s = weekly.tradable_stats(w, 0.004).iloc[0]
    assert s["valid_weeks"] == 4 and s["tradable_weeks"] == 2 and s["tradable_pct"] == 50


def test_weekly_features_use_only_data_before_entry():
    dates = pd.bdate_range("2025-01-01", periods=120)
    p = pd.DataFrame({"date": dates, "open": 100.0, "close": np.linspace(100, 160, 120)})
    weeks = weekly.weekly_returns(p)
    f1 = weekly_model.stock_features(p, "TEST", weeks)
    p2 = p.copy()
    p2.loc[p2["date"] >= dates[80], "close"] *= 3  # change the future only
    f2 = weekly_model.stock_features(p2, "TEST", weekly.weekly_returns(p2))
    early = f1["entry_date"] <= dates[80]  # entry on dates[80] only sees closes before it
    cols = ["ret_1w", "ret_4w", "vol_20d", "dist_sma_50", "last_day_ret"]
    assert np.allclose(f1.loc[early, cols], f2.loc[early, cols], equal_nan=True)


# --- costs --------------------------------------------------------------------------------
def test_delivery_cost_components_add_up():
    a = costs.CostAssumptions(brokerage_pct=0.0, brokerage_cap=0, brokerage_flat=20, dp_charge_per_sell=0,
                              half_spread=0.001, slippage=0.0005)
    p = costs.round_trip(100_000, a)
    assert p["brokerage"] == pytest.approx(40)
    assert p["stt"] == pytest.approx(0.001 * 200_000)  # delivery STT on both legs
    assert p["stamp_duty"] == pytest.approx(0.00015 * 100_000)  # buy side only
    assert p["gst"] == pytest.approx(0.18 * (p["brokerage"] + p["exchange"] + p["sebi"]))
    assert p["spread"] == pytest.approx(200) and p["slippage"] == pytest.approx(100)
    assert p["total"] == pytest.approx(p["explicit_total"] + p["spread"] + p["slippage"])
    assert p["total_pct"] == pytest.approx(p["total"] / 100_000)


def test_intraday_has_no_buy_stt_and_brokerage_cap():
    a = costs.CostAssumptions(instrument="equity_intraday")
    p = costs.explicit_charges(1_000_000, 1_000_000, a)
    assert p["stt"] == pytest.approx(0.00025 * 1_000_000) and p["brokerage"] == pytest.approx(40)


# --- backtest -----------------------------------------------------------------------------
def signals():
    weeks = pd.to_datetime(["2026-01-05", "2026-01-12"])
    rows = []
    for w, (pa, pb), (ra, rb) in zip(weeks, [(0.05, 0.001), (-0.01, 0.02)], [(0.10, -0.05), (0.03, 0.01)]):
        for sym, pred, ret in (("AAA", pa, ra), ("BBB", pb, rb)):
            rows.append({"symbol": sym, "week_start": w, "entry_date": w, "exit_date": w + pd.offsets.Day(4),
                         "entry_open": 100.0, "exit_close": 100.0 * (1 + ret), "pred": pred})
    return pd.DataFrame(rows)


ZERO = costs.CostAssumptions(brokerage_pct=0, brokerage_cap=0, exchange_txn=0, sebi_fee=0, gst=0, stt_buy=0,
                             stt_sell=0, stamp_buy=0, dp_charge_per_sell=0, half_spread=0, slippage=0)


def test_backtest_position_sizing_and_compounding_without_costs():
    cfg = backtest.BacktestConfig(initial_capital=10_000, assumptions=ZERO)
    trades, weeks = backtest.run(signals(), "A", cfg)
    # Week 1: both forecasts > 0, 5,000 each -> 50 shares each: +500 and -250.
    w1 = trades[trades["week_start"] == pd.Timestamp("2026-01-05")]
    assert list(w1["qty"]) == [50, 50] and w1["net_pnl"].sum() == pytest.approx(250)
    # Week 2 sizes from the new equity 10,250 -> 5,125 per trade -> 51 shares; only BBB qualifies.
    w2 = trades[trades["week_start"] == pd.Timestamp("2026-01-12")]
    assert list(w2["symbol"]) == ["BBB"] and w2["qty"].iloc[0] == 51
    assert weeks["equity"].iloc[-1] == pytest.approx(10_250 + 51 * 1.0)
    assert not trades.duplicated(["symbol", "week_start"]).any()


def test_cost_aware_filter_and_costs_deducted():
    a = costs.CostAssumptions(brokerage_pct=0, brokerage_cap=0, brokerage_flat=10, exchange_txn=0, sebi_fee=0,
                              gst=0, stt_buy=0, stt_sell=0, stamp_buy=0, dp_charge_per_sell=0, half_spread=0,
                              slippage=0)
    cfg = backtest.BacktestConfig(initial_capital=10_000, assumptions=a)
    # Position 5,000: cost 20 = 0.4%, so B needs a forecast > 0.8%.
    tb, wb = backtest.run(signals(), "B", cfg)
    assert set(zip(tb["symbol"], tb["week_start"].dt.day)) == {("AAA", 5), ("BBB", 12)}
    assert (tb["gross_pnl"] - tb["net_pnl"]).round(6).eq(20).all()
    s = backtest.summary(tb, wb, cfg)
    assert s["total_costs"] == pytest.approx(40) and s["trades"] == 2
    assert s["net_pnl"] == pytest.approx(s["gross_pnl"] - s["total_costs"])
    res = backtest.compare(signals(), cfg)
    assert (res["summary"]["weeks"] == 2).all()  # same dates for every strategy


# --- Monte Carlo --------------------------------------------------------------------------
def test_monte_carlo_reproducible_and_consistent():
    r = np.array([0.01, -0.02, 0.005, 0.0, 0.03, -0.01, 0.002, -0.004])
    a = montecarlo.simulate(r, 100_000, n_sims=500, block=2, seed=7)
    b = montecarlo.simulate(r, 100_000, n_sims=500, block=2, seed=7)
    assert a.equals(b) and len(a) == 500
    assert (a["max_drawdown_pct"] <= 0).all()
    assert np.allclose(a["cumulative_return_pct"], (a["ending_capital"] / 100_000 - 1) * 100)
    s = montecarlo.summarise(a, 100_000)
    assert s["prob_loss_pct"] == pytest.approx((a["ending_capital"] < 100_000).mean() * 100)


def test_block_bootstrap_keeps_consecutive_weeks():
    r = np.arange(20, dtype=float)
    paths = montecarlo.block_bootstrap_paths(r, n_sims=50, horizon=8, block=4, seed=1)
    assert paths.shape == (50, 8)
    assert np.all(np.diff(paths[:, :4], axis=1) == 1)  # each block is a run of consecutive weeks
