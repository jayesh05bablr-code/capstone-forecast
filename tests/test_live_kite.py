"""Checks for the live buyer/seller split, live trade status, Kite fields and update times."""
from datetime import datetime

import pandas as pd

from capstone import autorun, kite, live, scalping


def _bars(closes, start="2026-11-03 09:15", vol=100):
    t = pd.date_range(start, periods=len(closes), freq="1min")
    c = pd.Series(closes, dtype=float)
    return pd.DataFrame({"datetime": t, "open": c.shift(1).fillna(c.iloc[0]), "high": c + 0.5,
                         "low": c - 0.5, "close": c, "volume": vol})


def test_tick_rule_split():
    df = live.buy_sell_split(_bars([100, 101, 100, 100]), prev_close=99)
    assert list(df["buy_vol"]) == [100, 100, 0, 50]  # up, up, down, unchanged closing mid-range
    s = live.volume_summary(_bars([100, 101, 100, 100]), 99)
    assert s["buy_pct"] == 62.5 and s["buy_vol"] + s["sell_vol"] == s["volume"]


def test_trade_status_long_target():
    plan = {"side": "Long", "trigger": 102.0, "stop": 99.0, "target": 106.0}
    closes = [100, 101, 100, 101, 100] + [101, 103, 104, 106, 107]  # 5-min range then breakout
    status, _ = live.trade_status(plan, _bars(closes), or_minutes=5, window=20)
    assert status == "Target hit"
    status, _ = live.trade_status(plan, _bars(closes[:7]), 5, 20)
    assert status == "In trade"
    status, _ = live.trade_status(plan, _bars(closes[:3]), 5, 20)
    assert status == "Range forming"


def test_kite_fields_match_screens():
    a = kite.alert_ato("HDFCBANK", "Long", 725.52, 718.03, 736.77, 10, ltp=725.5)
    assert a["Alert: is"] == "Greater than or equal to (>=)"
    assert a["Alert: than (₹)"] == 725.55  # rounded up to the 0.05 tick
    assert a["ATO product"].startswith("MIS") and a["ATO price (₹)"] >= a["Alert: than (₹)"]
    g = kite.gtt_entry("HDFCBANK", 761.75, 1, ltp=725.5)
    assert (g["Type"], g["Trigger type"], g["Product"], g["Order"]) == ("Buy", "Single", "CNC", "LIMIT")
    assert g["% of LTP"] == 5.0
    o = kite.gtt_exit_oco("HDFCBANK", 700, 760, 1, ltp=725.5)
    assert o["Trigger type"] == "OCO" and o["Stop-loss price (₹)"] < o["Stop-loss trigger (₹)"]
    s = kite.alert_ato("TCS", "Short", 3000.03, 3020, 2970, 5)
    assert s["Alert: is"].startswith("Less") and s["ATO order"] == "SELL" and s["Alert: than (₹)"] == 3000.0


def test_plan_to_kite_round_trip():
    p = scalping.plan_row("INFY", 1500, 1505, 1495, 1, 100000, 0.005, 1.5, 1, 0.0005, 20)["Long"]
    a = kite.alert_ato(p["symbol"], p["side"], p["trigger"], p["stop"], p["target"], p["qty"])
    assert a["ATO qty"] == p["qty"] > 0


def test_update_slots():
    assert autorun.last_slot(datetime(2026, 11, 3, 9, 0)) == datetime(2026, 11, 3, 8, 45)
    assert autorun.last_slot(datetime(2026, 11, 3, 17, 0)) == datetime(2026, 11, 3, 16, 15)
    assert autorun.last_slot(datetime(2026, 11, 3, 7, 0)) == datetime(2026, 11, 2, 16, 15)
