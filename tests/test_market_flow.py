"""Market flow page: flow estimate, sector totals, Sankey links, heat map and the shared quote cache.

No network: quotes are made up here.
"""
import time

import pandas as pd
import pytest

from capstone import market_flow as mf
from capstone.ui import flow_anim, tab_flow

UNI = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC", "DDD"], "name": list("abcd"),
                    "sector": ["Banks", "Banks", "IT", "IT"], "index": "NIFTY 50"})


def _quotes():
    return pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC", "DDD"],
        "ltp": [110, 90, 100, 50], "open": [100, 100, 100, 50], "high": [110, 100, 104, 50],
        "low": [100, 90, 96, 50], "prev_close": [100, 100, 99, 49], "pct_change": [10, -10, 1.01, 2.04],
        "volume": [1, 1, 1, 1], "turnover_cr": [100, 300, 50, 10],
        "as_of": pd.Timestamp("2026-11-23 12:00")})


def test_flow_estimate_uses_position_in_days_range():
    s = mf.snapshot(_quotes(), UNI).set_index("symbol")
    assert s.loc["AAA", "position"] == 1 and s.loc["AAA", "net_flow_cr"] == 100   # at the high
    assert s.loc["BBB", "position"] == -1 and s.loc["BBB", "net_flow_cr"] == -300  # at the low
    assert s.loc["CCC", "position"] == 0                                           # mid-range
    assert s.loc["DDD", "position"] == 1  # no range yet: direction of the day's change


def test_sector_table_splits_buying_and_selling():
    sec = mf.sector_table(mf.snapshot(_quotes(), UNI)).set_index("sector")
    assert sec.loc["Banks", "turnover_cr"] == 400 and sec.loc["Banks", "net_flow_cr"] == -200
    assert sec.loc["Banks", "buy_cr"] == 100 and sec.loc["Banks", "sell_cr"] == 300
    assert sec.loc["Banks", "pct_change"] == pytest.approx((10 * 100 - 10 * 300) / 400)
    assert sec.loc["IT", "net_flow_cr"] == 10


def _balanced(data):
    """Every middle node passes on what it receives, so the bands line up."""
    for i, n in enumerate(data["nodes"]):
        if n["col"] == 1:
            inflow = sum(l["value"] for l in data["links"] if l["target"] == i)
            outflow = sum(l["value"] for l in data["links"] if l["source"] == i)
            assert inflow == pytest.approx(outflow)


def test_animated_sector_flow_data():
    snap = mf.snapshot(_quotes(), UNI)
    data = flow_anim.sector_flow_data(snap, mf.sector_table(snap), top=1)
    labels = [n["label"] for n in data["nodes"]]
    assert {"Buying pressure", "Selling pressure", "BBB", "AAA"} <= set(labels)
    assert any(lab.startswith("Other") for lab in labels)  # DDD grouped with top=1
    assert data["estimate"] is True
    _balanced(data)


def test_animated_volume_flow_data_adds_up():
    snap = mf.snapshot(_quotes(), UNI)
    data = flow_anim.volume_flow_data(snap, top=2)
    from_hub = sum(l["value"] for l in data["links"] if l["source"] == 0)
    assert from_hub == pytest.approx(snap["turnover_cr"].sum())
    assert data["estimate"] is False
    _balanced(data)


def test_minute_pace_and_surge_glow():
    snap = mf.snapshot(_quotes(), UNI)
    prev = _quotes().assign(turnover_cr=[99, 299, 40, 10], as_of=pd.Timestamp("2026-11-23 11:59"))
    # 165 minutes since 09:15: AAA's average is 100/165 per minute and its last minute 1, about 1.65x
    pace = mf.pace(snap, prev)
    assert pace["AAA"] == pytest.approx(1 / (100 / 165))
    assert pace["CCC"] > 2.5 and pace["DDD"] == 0
    data = flow_anim.volume_flow_data(snap, pace, top=4)
    assert {n["label"] for n in data["nodes"] if n.get("surge")} == {"CCC"}


def test_heat_map_and_headline():
    snap = mf.snapshot(_quotes(), UNI)
    assert tab_flow.heat_map(snap).data[0].type == "treemap"
    text = mf.headline(snap, mf.sector_table(snap))
    assert "3 of 4" in text and "Banks" in text


def test_universe_file_is_clean():
    uni = mf.universe()
    assert len(uni) >= 100 and uni["symbol"].is_unique and uni["sector"].notna().all()


def test_latest_quotes_shares_one_download(monkeypatch):
    calls = []

    def fake(symbols):
        calls.append(1)
        return _quotes()
    monkeypatch.setattr(mf, "fetch_quotes", fake)
    monkeypatch.setattr(mf, "_latest", {"quotes": None, "at": 0.0, "busy": False})
    q1, _ = mf.latest_quotes(max_age=60)
    q2, _ = mf.latest_quotes(max_age=60)
    assert len(calls) == 1 and len(q1) == len(q2) == 4
    mf._latest["at"] = time.time() - 120  # stale: served at once, refreshed in the background
    q3, _ = mf.latest_quotes(max_age=60)
    assert q3 is not None
    for _ in range(50):
        if len(calls) == 2 and not mf._latest["busy"]:
            break
        time.sleep(0.02)
    assert len(calls) == 2
