"""Market flow filters: period views from daily history, metrics, filters, relative volume, risk measures.

No network: the daily rows are made up here.
"""
import numpy as np
import pandas as pd
import pytest

from capstone import flow_history as fh
from capstone import market_flow as mf
from capstone import stock_metrics as sm
from capstone.ui import flow_anim

UNI = pd.DataFrame({"symbol": ["AAA", "BBB"], "name": ["a", "b"], "sector": ["Banks", "IT"], "index": "NIFTY 50"})


def _hist(days=30):
    dates = pd.bdate_range("2026-10-01", periods=days)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "symbol": "AAA", "open": 100 + i, "high": 102 + i, "low": 99 + i, "close": 102 + i,
                     "prev_close": 100 + i, "volume": 1000, "turnover": 1e7 * 10})  # closes at the high: all buying
        rows.append({"date": d, "symbol": "BBB", "open": 50, "high": 51, "low": 49, "close": 49, "prev_close": 50,
                     "volume": 2000 if i < days - 1 else 6000, "turnover": 1e7 * 20})  # at the low: all selling
    return pd.DataFrame(rows)


def test_period_snapshot_adds_up_published_days():
    h = _hist()
    s = fh.period_snapshot(h, UNI, 5).set_index("symbol")
    assert s.loc["AAA", "turnover_cr"] == pytest.approx(50) and s.loc["BBB", "turnover_cr"] == pytest.approx(100)
    assert s.loc["AAA", "buy_cr"] == pytest.approx(50) and s.loc["BBB", "sell_cr"] == pytest.approx(100)
    assert s.loc["AAA", "volume"] == 5000
    # % change is the last close vs the close before the window
    assert s.loc["AAA", "pct_change"] == pytest.approx(100 * ((102 + 29) / (100 + 25) - 1))
    assert s["to_date"].max() == h["date"].max()


def test_period_snapshot_skips_a_stock_missing_a_day():
    h = _hist()
    h = h[~((h["symbol"] == "BBB") & (h["date"] == h["date"].max()))]
    assert list(fh.period_snapshot(h, UNI, 5)["symbol"]) == ["AAA"]


def test_relative_volume_vs_20_session_average():
    h = _hist()
    s = fh.period_snapshot(h, UNI, 1)
    rv = fh.relative_volume(s, h, intraday=False)
    assert rv["AAA"] == pytest.approx(1.0) and rv["BBB"] == pytest.approx(3.0)


def test_metrics_and_filters():
    s = fh.period_snapshot(_hist(), UNI, 1)
    vol = mf.with_metric(s, "volume").set_index("symbol")
    assert vol.loc["BBB", "turnover_cr"] == pytest.approx(6000 / 1e5) and vol.loc["BBB", "value_cr"] == 20
    pr = mf.with_metric(s, "pressure").set_index("symbol")
    assert pr.loc["AAA", "buy_cr"] == pytest.approx(10) and pr.loc["BBB", "sell_cr"] == pytest.approx(20)
    assert list(mf.filter_stocks(s, direction="Losers")["symbol"]) == ["BBB"]
    assert list(mf.filter_stocks(s, sectors=["Banks"])["symbol"]) == ["AAA"]
    s["rvol"] = [1.0, 3.0]
    assert list(mf.filter_stocks(s, min_rvol=2)["symbol"]) == ["BBB"]


def test_charts_take_any_metric():
    s = fh.period_snapshot(_hist(), UNI, 5)
    for metric in ("value", "volume", "pressure"):
        snap = mf.with_metric(s, metric)
        cap = flow_anim.volume_flow_data(snap, unit=metric)
        assert cap["estimate"] is (metric == "pressure")
        ctl = flow_anim.sector_flow_data(snap, mf.sector_table(snap), unit=metric)
        assert ctl["estimate"] is True
    assert flow_anim.fmt(12.345, "volume") == "12.3 L sh" and flow_anim.fmt(-1500, "value", True) == "−₹1,500 cr"


def test_universes_from_the_official_lists():
    sizes = {k: len(mf.universe(k)) for k in mf.UNIVERSES}
    assert sizes == {"NIFTY 50": 50, "NIFTY Next 50": 50, "NIFTY 100": 100, "NIFTY 500": 500}
    assert mf.universe()["sector"].notna().all()


def test_risk_measures():
    h = _hist(70)
    a = h[h["symbol"] == "AAA"].reset_index(drop=True)
    ix = a[["date"]].assign(close=a["close"] / 2)  # index moving exactly in step: beta 1
    m = sm.summary(a, ix)
    assert m["beta"] == pytest.approx(1.0)
    assert m["max_dd"] == pytest.approx(0.0) and m["ret_1w"] > 0 and m["atr_pct"] > 0
    assert sm.summary(a.head(5), ix)["beta"] is None  # not enough data: no number
    bars = pd.DataFrame({"high": [11, 12], "low": [9, 10], "close": [10, 11], "volume": [100, 300]})
    assert sm.intraday_vwap(bars).iloc[-1] == pytest.approx((10 * 100 + 11 * 300) / 400)
