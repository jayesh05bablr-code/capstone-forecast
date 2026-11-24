"""Morning brief helpers (no network)."""
import pandas as pd

from capstone import brief

MOOD = {"date": "2026-11-23", "india_vix": {"level": 17.7, "change_1d_pct": 2.2, "vs_1w_ago": {"change_pct": 12.9, "base_level": 15.68},
                                            "range": {"percentile": 80.5}},
        "stock_breadth": {"up": 813, "down": 2143, "up_pct": 27.2, "reading": "broad decline"}}


def test_risks_only_from_published_numbers():
    r = brief.risks(MOOD, ["divergence x"], {"overnight_complete": False}, {"pe": {"current": 24.0, "percentile": 90}})
    text = " ".join(r)
    assert "17.7" in text and "80%" in text and "13%" in text and "27%" in text and "divergence x" in r
    assert "not in the data yet" in text and "P/E 24.0" in text


def test_risks_quiet_market():
    calm = {"india_vix": {"level": 11, "range": {"percentile": 20}, "vs_1w_ago": {"change_pct": -3}},
            "stock_breadth": {"up_pct": 55, "reading": "mixed"}}
    assert brief.risks(calm, [], {"overnight_complete": True}, {"pe": {"current": 20, "percentile": 40}}) == []
    assert brief.risks(None, [], None, None) == []


def test_forecast_highlights_split_up_down():
    p = pd.DataFrame({"symbol": list("ABCD"), "pred_gap_pct": [0.5, -0.2, 1.1, -0.9], "pred_open": [1, 2, 3, 4]})
    h = brief.forecast_highlights(p, n=1)
    assert [r["symbol"] for r in h["up"]] == ["C"] and [r["symbol"] for r in h["down"]] == ["D"]
    assert brief.forecast_highlights(pd.DataFrame()) == {"up": [], "down": []}


def test_parse_yahoo_daily():
    body = {"chart": {"result": [{"meta": {"regularMarketPrice": 110.0, "regularMarketTime": 1795500000},
                                  "timestamp": [1795300000, 1795400000, 1795500000],
                                  "indicators": {"quote": [{"close": [95.0, 100.0, 110.0]}]}}]}}
    p = brief._parse_daily(body)
    assert abs(p["pct"] - 10.0) < 1e-9 and p["close"] == 110.0
    assert brief._parse_daily({"chart": {"result": []}}) is None
