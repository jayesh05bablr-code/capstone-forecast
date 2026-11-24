"""Sector intelligence and breadth calculations on small hand-made data (no network)."""
import numpy as np
import pandas as pd

from capstone import breadth, sector_intel


def _index_frame(n=40):
    dates = pd.bdate_range("2026-01-01", periods=n)
    rows = []
    # NIFTY 50 flat; IT rising faster and faster (RS up and growing); Bank falling faster and faster
    for name, k in [("NIFTY 50", 0.0), ("NIFTY IT", 0.0005), ("NIFTY BANK", -0.0005)]:
        for i, d in enumerate(dates):
            rows.append({"index": name, "date": d, "close": 100 * np.exp(k * i * i), "volume": 1000 + 10 * i})
    return pd.DataFrame(rows)


def test_leaderboard_rs_and_quadrants():
    lb = sector_intel.leaderboard(_index_frame()).set_index("index")
    assert set(lb.index) == {"NIFTY IT", "NIFTY BANK"}
    it, bank = lb.loc["NIFTY IT"], lb.loc["NIFTY BANK"]
    assert abs(it["ret_20d"] - 100 * (np.exp(0.0005 * (39 ** 2 - 19 ** 2)) - 1)) < 1e-9
    assert abs(it["rs_20d"] - it["ret_20d"]) < 1e-9  # benchmark is flat
    assert it["rs_20d"] > 0 and bank["rs_20d"] < 0
    assert it["quadrant"] == "Leading" and bank["quadrant"] == "Lagging"


def test_leaderboard_needs_benchmark():
    ix = _index_frame()
    assert sector_intel.leaderboard(ix[ix["index"] != "NIFTY 50"]).empty


def test_rotation_trails_shape():
    t = sector_intel.rotation_trails(_index_frame(), ["NIFTY IT"], points=4)
    assert len(t) == 4 and list(t["step"]) == [0, 1, 2, 3]


def _hist(n=30):
    dates = pd.bdate_range("2026-01-01", periods=n)
    rows = []
    for sym, drift in [("AAA", 1.01), ("BBB", 0.99), ("CCC", 1.0)]:
        for i, d in enumerate(dates):
            c = 100 * drift ** i
            rows.append({"date": d, "symbol": sym, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "prev_close": c / drift, "volume": 1000 * (3 if (sym == "AAA" and i == n - 1) else 1),
                         "turnover": c * 1000 * 1e5})
    return pd.DataFrame(rows)


UNI = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "name": ["A", "B", "C"], "sector": ["X", "X", "Y"]})


def test_constituent_contributions_add_up():
    cs = sector_intel.constituent_stats(_hist(), UNI)
    avg = np.mean([100 * 0.01, 100 * -0.01, 0.0])
    assert abs(cs["contrib_1d"].sum() - avg) < 1e-9
    assert cs.set_index("sector").loc["X", "stocks"] == 2


def test_stock_stats_and_breadth():
    s = breadth.stock_stats(_hist(), UNI).set_index("symbol")
    assert bool(s.loc["AAA", "unusual_volume"]) and not bool(s.loc["CCC", "unusual_volume"])
    assert s.loc["AAA", "above_20"] == 1 and s.loc["BBB", "above_20"] == 0
    assert np.isnan(s.loc["AAA", "dma50"])  # only 30 sessions: no 50-day average, never a guess
    ub = breadth.universe_breadth(s.reset_index())
    assert ub["advancing"] == 1 and ub["declining"] == 1 and ub["above_50"] is None and ub["above_200"] is None


def test_liquidity_bands():
    assert [breadth.liquidity(v) for v in (900, 150, 25, 5)] == ["Very high", "High", "Medium", "Low"]
    assert breadth.liquidity(float("nan")) is None


def test_divergence_alert_narrow_rally():
    dates = pd.bdate_range("2026-01-01", periods=15)
    bh = pd.DataFrame({"date": dates, "advancing": 10, "declining": 30, "above_20": 50.0})
    bh["ad_line"] = (bh["advancing"] - bh["declining"]).cumsum()
    ix = pd.DataFrame({"date": dates, "close": np.linspace(100, 105, 15)})
    alerts = breadth.divergences(ix, bh)
    assert alerts and "narrow rally" in alerts[0]
