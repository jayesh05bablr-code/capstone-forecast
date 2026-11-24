"""Market terminal helpers (no network)."""
import numpy as np
import pandas as pd

from capstone.ui import tab_terminal, terminal


def _snap():
    return pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC", "DDD"], "name": ["A Ltd", "B Ltd", "C Ltd", "D Ltd"],
        "sector": ["Banks", "Banks", "IT", "IT"],
        "ltp": [100.0, 50.0, 200.0, 10.0], "pct_change": [1.5, -2.0, 0.0, -0.5],
        "vwap": [99.0, 51.0, 200.0, np.nan], "turnover_cr": [10.0, 30.0, 5.0, 1.0],
        "high_52w": [101.0, 80.0, 300.0, 20.0], "low_52w": [60.0, 49.5, 150.0, 9.9]})


def test_pulse_counts_from_feed():
    p = tab_terminal.pulse(_snap())
    assert (p["up"], p["down"], p["flat"], p["n"]) == (1, 2, 1, 4)
    assert p["above_vwap"] == 1 and p["vwap_n"] == 3
    assert p["value_cr"] == 46.0
    assert list(p["near_hi"]["symbol"]) == ["AAA"]  # within 2% of its 52-week high
    assert set(p["near_lo"]["symbol"]) == {"BBB", "DDD"}


def test_heat_map_sizes_and_colours():
    s = _snap()
    t = tab_terminal.heat_map(s, "Traded value", dark=True).data[0]
    vals = dict(zip(t.ids, t.values))
    assert vals["Banks"] == 40.0 and vals["Banks/BBB"] == 30.0  # sector box = sum of its stocks
    # sector colour = traded-value-weighted % change: (1.5*10 - 2*30) / 40
    assert abs(dict(zip(t.ids, t.marker.colors))["Banks"] - (-1.125)) < 1e-9
    eq = tab_terminal.heat_map(s, "Equal", dark=False).data[0]
    assert dict(zip(eq.ids, eq.values))["IT"] == 2.0


def test_ribbon_indices_then_stocks_alphabetical():
    q = pd.DataFrame({"symbol": ["ZED", "ABC", "OUT"], "ltp": [10.0, 20.0, 5.0], "pct_change": [1.0, -1.0, 0.0]})
    uni = pd.DataFrame({"symbol": ["ABC", "ZED"]})
    items = terminal.ribbon_items({"NIFTY 50": {"last": 25000.0, "percChange": 0.4}}, q, uni)
    assert len(items) == 3 and "NIFTY 50" in items[0]
    assert "ABC" in items[1] and "ZED" in items[2]  # stocks outside the universe are left out
    assert "▲" in items[0] and "▼" in items[1]
    assert terminal.ribbon_items({}, None, uni) == []
