"""TradingView-style chart payload (no browser)."""
import json

import pandas as pd

from capstone.ui import tv_chart


def _bars():
    t = pd.date_range("2026-11-23 09:15", periods=3, freq="min")
    return pd.DataFrame({"datetime": t, "open": [10, 11, 12.0], "high": [11, 12, 13.0], "low": [9, 10, 11.0],
                         "close": [10.5, 11.5, 12.5], "volume": [100, None, 300]})


def test_bars_read_as_ist_wall_clock():
    b = tv_chart.bars_json(_bars())
    assert [x["t"] for x in b] == [int(pd.Timestamp("2026-11-23 09:15").timestamp()) + 60 * i for i in range(3)]
    assert b[1]["v"] == 0.0 and b[2]["c"] == 12.5
    aware = _bars().assign(datetime=lambda d: d["datetime"].dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC"))
    assert tv_chart.bars_json(aware) == b  # a UTC feed is shown at the same IST times


def test_html_embeds_library_and_config():
    h = tv_chart.html("TCS", _bars(), prev_close=10.0, pred_open=None, opening_range={"high": 12, "low": 9})
    assert "LightweightCharts" in h and "__CFG__" not in h and "__LIB__" not in h
    cfg = json.loads(h.split("const C=", 1)[1].split(", T=C.theme", 1)[0])
    assert cfg["symbol"] == "TCS" and cfg["prev"] == 10.0 and cfg["fcst"] is None and len(cfg["bars"]) == 3
