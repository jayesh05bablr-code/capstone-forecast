"""Market flow page: flow estimate, sector totals, Sankey links, heat map and the shared quote cache.

No network: quotes are made up here.
"""
import time

import pandas as pd
import pytest

from capstone import market_flow as mf
from capstone.ui import tab_flow

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


def test_sector_table_and_rotation():
    sec = mf.sector_table(mf.snapshot(_quotes(), UNI)).set_index("sector")
    assert sec.loc["Banks", "turnover_cr"] == 400 and sec.loc["Banks", "net_flow_cr"] == -200
    assert sec.loc["Banks", "pct_change"] == pytest.approx((10 * 100 - 10 * 300) / 400)
    assert sec.loc["IT", "net_flow_cr"] == 10
    links = mf.rotation_links(sec.reset_index().sort_values("net_flow_cr", ascending=False), hub="All")
    assert links["side"] == ["out", "hub", "in"]
    assert links["value"] == [200, 10] and links["sold_cr"] == 200 and links["bought_cr"] == 10


def test_volume_links_add_up_to_turnover():
    snap = mf.snapshot(_quotes(), UNI)
    links = mf.volume_links(snap, top=2, hub="All")
    from_hub = sum(v for s, v in zip(links["source"], links["value"]) if s == 0)
    assert from_hub == pytest.approx(snap["turnover_cr"].sum())
    assert {"BBB", "AAA"} <= set(links["labels"]) and any(lab.startswith("Other IT") for lab in links["labels"])
    for i, kind in enumerate(links["kind"]):  # each sector passes on exactly what it receives
        if kind == "sector":
            inflow = sum(v for t, v in zip(links["target"], links["value"]) if t == i)
            outflow = sum(v for s, v in zip(links["source"], links["value"]) if s == i)
            assert inflow == pytest.approx(outflow)


def test_charts_build_and_headline():
    snap = mf.snapshot(_quotes(), UNI)
    sec = mf.sector_table(snap)
    assert tab_flow.heat_map(snap).data[0].type == "treemap"
    assert tab_flow.sector_flow(mf.rotation_links(sec)).data[0].type == "sankey"
    assert tab_flow.volume_flow(mf.volume_links(snap)).data[0].type == "sankey"
    text = mf.headline(snap, sec)
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
