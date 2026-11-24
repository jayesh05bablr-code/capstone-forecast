"""Parsers and summaries for NSE institutional files (fixtures in NSE's published formats; no network)."""
import pandas as pd
import pytest

from capstone import institutional as inst

FII_JSON = [{"category": "DII **", "date": "23-Nov-2026", "buyValue": "14896.63", "sellValue": "17918.07", "netValue": "-3021.44"},
            {"category": "FII/FPI *", "date": "23-Nov-2026", "buyValue": "22768.87", "sellValue": "17311.59", "netValue": "5457.28"}]
FII_CSV = ('"CATEGORY","DATE","BUY VALUE(₹ Crores)","SELL VALUE (₹ Crores)","NET VALUE (₹ Crores)"\n'
           '"DII","23-Nov-2026","14,896.63","17,918.07","-3,021.44"\n"FII/FPI","23-Nov-2026","22,768.87","17,311.59","5,457.28"\n')
BULK = ("Date,Symbol,Security Name,Client Name,Buy/Sell,Quantity Traded,Trade Price / Wght. Avg. Price,Remarks\n"
        "23-NOV-2026,ABC,ABC Ltd,SOME FUND,BUY,1000000,250.50,-\n23-NOV-2026,ABC,ABC Ltd,OTHER LLP,SELL,400000,251.00,-\n")
DELIV = ("SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, "
         "TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
         "TCS, EQ, 23-Nov-2026, 3000, 3010, 3050, 2990, 3020, 3025, 3021, 1000000, 30210.5, 50000, 600000, 60.00\n"
         "TCS, N1, 23-Nov-2026, 100, 100, 100, 100, 100, 100, 100, 10, 0.01, 1, 10, 100.00\n"
         "INFY, EQ, 23-Nov-2026, 1500, 1500, 1520, 1490, 1510, 1505, 1503, 2000000, 30060, 80000, 900000, 45.00\n")


def test_fii_json_and_csv_agree():
    a, b = inst.parse_fii_dii(FII_JSON), inst.parse_fii_dii(FII_CSV)
    for df in (a, b):
        d = df.set_index("category")
        assert d.loc["FII/FPI", "net_cr"] == pytest.approx(5457.28) and d.loc["DII", "buy_cr"] == pytest.approx(14896.63)
        assert d["date"].iloc[0] == pd.Timestamp("2026-11-23")


def test_wrong_file_is_refused():
    with pytest.raises(ValueError):
        inst.parse_fii_dii("a,b\n1,2\n")
    with pytest.raises(ValueError):
        inst.parse_delivery("a,b\n1,2\n")


def test_deals_value_and_side():
    d = inst.parse_deals(BULK, "bulk")
    assert list(d["side"]) == ["Buy", "Sell"]
    assert d["value_cr"].iloc[0] == pytest.approx(1_000_000 * 250.5 / 1e7)


def test_delivery_keeps_equity_series_only():
    d = inst.parse_delivery(DELIV)
    assert sorted(d["symbol"]) == ["INFY", "TCS"]
    assert d.set_index("symbol").loc["TCS", "deliv_pct"] == 60.0
    assert d.set_index("symbol").loc["TCS", "value_cr"] == pytest.approx(302.105)


def test_fii_summary_needs_enough_days():
    s = inst.fii_summary(inst.parse_fii_dii(FII_JSON))
    assert s["FII/FPI"]["net"] == pytest.approx(5457.28) and s["FII/FPI"]["sum5"] is None


def test_store_and_dedupe(tmp_path, monkeypatch):
    monkeypatch.setitem(inst.FILES, "fii_dii", tmp_path / "f.csv")
    inst.save("fii_dii", inst.parse_fii_dii(FII_JSON), inst.KEYS["fii_dii"])
    both = inst.save("fii_dii", inst.parse_fii_dii(FII_CSV), inst.KEYS["fii_dii"])
    assert len(both) == 2


def test_delivery_view_average():
    uni = pd.DataFrame({"symbol": ["TCS"], "name": ["TCS"], "sector": ["IT"]})
    rows = [{"symbol": "TCS", "date": pd.Timestamp("2026-11-01") + pd.Timedelta(days=i), "deliv_pct": 40.0,
             "close": 1, "volume": 1, "deliv_qty": 1, "deliv_value_cr": 0} for i in range(6)]
    rows.append({**rows[0], "date": pd.Timestamp("2026-11-20"), "deliv_pct": 70.0})
    v = inst.delivery_view(pd.DataFrame(rows), uni)
    assert v["deliv_pct_avg"].iloc[0] == 40.0 and v["deliv_vs_avg"].iloc[0] == 30.0


def test_refresh_reports_block_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setitem(inst.FILES, "deals", tmp_path / "d.csv")
    def blocked():
        raise inst.NseBlocked("nsearchives.nseindia.com did not answer (403)")
    monkeypatch.setitem(inst.FETCHERS, "deals", blocked)
    inst._state.pop("deals", None)
    r = inst.refresh("deals")
    assert not r["ok"] and "403" in r["error"]
