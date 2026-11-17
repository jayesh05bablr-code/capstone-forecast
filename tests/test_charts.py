"""Price-chart indicators and figure builders."""
import numpy as np
import pandas as pd
import pytest

from capstone import charts, charts_eval, indicators as ind


def _prices(n=300, bonus_at=None):
    d = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.sin(np.arange(n) / 7))
    df = pd.DataFrame({"date": d, "open": close - 0.5, "high": close + 1, "low": close - 1, "close": close,
                       "volume": 1000 + np.arange(n)})
    if bonus_at is not None:  # unadjusted 1:1 bonus: prices before the ex-date are double
        before = df.index < bonus_at
        df.loc[before, ["open", "high", "low", "close"]] *= 2
    return df


def test_adjust_for_actions_removes_bonus_drop():
    raw = _prices(bonus_at=150)
    ca = pd.DataFrame({"ex_date": [raw["date"][150]], "price_factor": [0.5], "action": ["bonus 1:1"]})
    adj = ind.adjust_for_actions(raw, ca)
    jump = adj["close"].pct_change().abs().max()
    assert jump < 0.05                                      # no fake -50% day left
    assert adj["close"].iloc[150:].equals(raw["close"].iloc[150:])   # recent prices unchanged
    assert adj["volume"].iloc[0] == pytest.approx(raw["volume"].iloc[0] * 2)


def test_rsi_bounds_and_trend():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up).dropna().iloc[-1] == pytest.approx(100)
    r = ind.rsi(_prices()["close"]).dropna()
    assert r.between(0, 100).all()


def test_indicators_use_only_the_past():
    df = _prices()
    base = ind.sma(df["close"], 20).iloc[100], ind.ema(df["close"], 21).iloc[100], ind.rsi(df["close"]).iloc[100]
    df.loc[150:, "close"] *= 3                               # change the future
    after = ind.sma(df["close"], 20).iloc[100], ind.ema(df["close"], 21).iloc[100], ind.rsi(df["close"]).iloc[100]
    assert base == pytest.approx(after)


def test_heikin_ashi_and_pivots():
    df = _prices(30)
    ha = ind.heikin_ashi(df)
    assert (ha["high"] >= ha[["open", "close"]].max(axis=1)).all()
    p = ind.pivots(df.iloc[-1])
    assert p["S2"] < p["S1"] < p["Pivot"] < p["R1"] < p["R2"]


@pytest.mark.parametrize("style", charts.STYLES)
def test_price_chart_fits_axis_to_visible_window(style):
    df = _prices(bonus_at=None)
    df.loc[:100, ["open", "high", "low", "close"]] *= 3    # old prices far higher
    last = df["close"].iloc[-1]
    fc = {"date": df["date"].iloc[-1] + pd.Timedelta(days=3), "open": last, "low": last - 1, "high": last + 1}
    fig = charts.price_chart(df, "t", period="1M", chart_style=style, overlays=charts.OVERLAYS,
                             panels=charts.PANELS, forecast=fc)
    lo, hi = fig.layout.yaxis.range
    assert hi < df["high"].iloc[:100].min()                 # old, higher prices do not stretch the axis
    assert lo < df["low"].tail(21).min() and hi > df["high"].tail(21).max()
    assert len(fig.data[0].x) == 21


def test_breaks_hide_holidays():
    d = pd.Series(pd.to_datetime(["2026-11-09", "2026-11-11", "2026-11-12"]))   # 10 Nov missing
    br = charts._breaks(d)
    assert "2026-11-10" in br[1]["values"]


def test_equity_and_drawdown_share_axis():
    w = pd.DataFrame({"week_start": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
                      "equity": [100, 110, 99, 105, 120]})
    fig = charts_eval.equity_and_drawdown({"A": w}, {"A": "A"}, 100)
    dd = fig.data[1].y
    assert min(dd) == pytest.approx((99 / 110 - 1) * 100)


def test_breaks_keep_special_weekend_sessions():
    d = pd.Series(pd.to_datetime(["2026-11-06", "2026-11-08", "2026-11-09"]))   # Sunday 8 Nov traded
    br = charts._breaks(d)
    assert all("bounds" not in b for b in br)
    assert "2026-11-08" not in br[0]["values"] and "2026-11-07" in br[0]["values"]


def _minutes(days=("2026-11-12", "2026-11-13")):
    rows = []
    for d in days:
        t = pd.date_range(f"{d} 09:15", periods=60, freq="min")
        base = 100 + np.r_[np.zeros(15), np.linspace(0, 3, 45)]   # flat range, then a breakout up
        rows.append(pd.DataFrame({"datetime": t, "open": base, "high": base + 0.2, "low": base - 0.2,
                                  "close": base + 0.1, "volume": 1000}))
    return pd.concat(rows, ignore_index=True)


def test_scalping_backtest_reports_trade_levels():
    from capstone import scalping
    bt = scalping.backtest(_minutes(), 15, 60, 2.0, 0.0, "Both directions", 1000, 20)
    t = bt[bt["side"] != "No trade"].iloc[0]
    assert t["or_low"] < t["or_high"] <= t["entry"] and t["stop"] == t["or_low"]
    assert t["entry_time"] <= t["exit_time"]


def test_intraday_chart_and_single_trade_pnl_visible():
    from capstone import scalping
    m = _minutes(("2026-11-13",))
    bt = scalping.backtest(m, 15, 60, 2.0, 0.0, "Both directions", 1000, 20)
    r = bt.iloc[0].to_dict()
    r["or_end"] = pd.Timestamp("2026-11-13 09:30")
    fig = charts.intraday_chart(m, "t", r)
    names = [d.name for d in fig.data]
    assert "Entry" in names and "Exit" in names and "VWAP" in names
    lo, hi = fig.layout.yaxis.range
    assert lo <= r["stop"] and hi >= max(r["target"], m["high"].max()) - 1e-9


def test_apply_axes_units_and_manual_range():
    fig = charts.price_chart(_prices(), "t", period="3M")
    base = float(_prices()["close"].iloc[-63])
    charts.apply_axes(fig, {"y_units": "% change", "y_step": 1.0, "x_step": "Week"}, base=base)
    assert all(t.endswith("%") for t in fig.layout.yaxis.ticktext)
    assert fig.layout.xaxis.dtick == 7 * 86_400_000
    charts.apply_axes(fig, {"y_mode": "Manual", "y_min": 90, "y_max": 120})
    assert tuple(fig.layout.yaxis.range) == (90, 120)
