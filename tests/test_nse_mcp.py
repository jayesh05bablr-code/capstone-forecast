"""NSE MCP client, the NSE-first price merge, NSE quotes in the Live tab and the Ask chat loop.

No network: a tiny local HTTP server stands in for mcp.nseindia.in, and the language model is faked.
"""
import json
import threading
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from capstone import ask_market, fetch_data, live, nse_mcp


def _history(end: date, n: int = 3):
    days = [end - timedelta(days=i) for i in range(n)][::-1]
    return {"symbol": "TCS", "from_date": days[0].isoformat(), "to_date": days[-1].isoformat(),
            "data": [{"symbol": "TCS", "series": "EQ", "date": d.isoformat(), "open": 10, "high": 11, "low": 9,
                      "close": 10.5, "prevClose": 10, "volume": 100, "totalTradedValue": 1050} for d in days]}


class FakeMcp(BaseHTTPRequestHandler):
    calls: list = []
    gap_before = None  # end dates before this return "Symbol not found" (like NSE's 2023 gap)

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200 if "id" in body else 202)
        self.send_header("Content-Type", "application/json")
        self.send_header("Mcp-Session-Id", "s1")
        self.end_headers()
        if "id" not in body:
            return
        method, params = body["method"], body.get("params", {})
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "serverInfo": {"name": "fake"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "cm_get_stock_quote", "description": "Quote. Disclaimer: long text",
                                 "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}}}}]}
        else:
            FakeMcp.calls.append(params)
            args = params["arguments"]
            if params["name"] == "cm_get_stock_quote":
                data = {"stock": {"symbol": args["symbol"], "lastTradedPrice": 101.0, "openPrice": 100.5,
                                  "preClosePrice": 100.0, "perChange": 1.0, "highPrice": 102, "lowPrice": 99,
                                  "volume": 5, "latestTimestamp": "2026-11-03 10:00:00"}}
            else:
                end = date.fromisoformat(args["endDate"])
                data = ({"error": "Symbol not found: TCS"} if FakeMcp.gap_before and end < FakeMcp.gap_before
                        and end > FakeMcp.gap_before - timedelta(days=100) else _history(end))
            result = {"content": [{"type": "text", "text": json.dumps(data)}], "isError": False}
        self.wfile.write(json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode())


@pytest.fixture
def fake_server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), FakeMcp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/mcp"
    monkeypatch.setitem(nse_mcp.SERVERS, "market", url)
    monkeypatch.setitem(nse_mcp.SERVERS, "bhavcopy", url)
    nse_mcp._sessions.clear()
    nse_mcp._tools_cache.clear()
    FakeMcp.calls, FakeMcp.gap_before = [], None
    yield
    srv.shutdown()
    nse_mcp._sessions.clear()
    nse_mcp._tools_cache.clear()


def test_quote_and_tools(fake_server):
    q = nse_mcp.quote("INFY")
    assert q["ltp"] == 101.0 and q["prev_close"] == 100.0 and q["as_of"] == datetime(2026, 11, 3, 10, 0)
    assert nse_mcp.list_tools("market")[0]["name"] == "cm_get_stock_quote"


def test_history_walks_back_and_skips_gaps(fake_server):
    start = date.today() - timedelta(days=400)
    FakeMcp.gap_before = date.today() - timedelta(days=150)
    rows = nse_mcp.stock_history("TCS", start.isoformat())
    assert rows and rows == sorted(rows, key=lambda r: r["date"])
    assert set(rows[0]) == set(nse_mcp.COLUMNS) and rows[0]["prev_close"] == 10
    ends = [date.fromisoformat(c["arguments"]["endDate"]) for c in FakeMcp.calls]
    assert ends == sorted(ends, reverse=True) and min(ends) < FakeMcp.gap_before - timedelta(days=100)


def test_nse_rows_win_over_other_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data, "RAW", tmp_path)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=30)
    other = pd.DataFrame({"date": days, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1})
    nse = other.iloc[-10:].assign(close=2.0)  # NSE covers only the last 10 days
    monkeypatch.setattr(fetch_data, "nse_daily", lambda s, start: nse.copy())
    monkeypatch.setattr(fetch_data, "other_daily", lambda s, start: other.copy())
    n, used = fetch_data.update_symbol("TCS", days[0].date().isoformat())
    saved = pd.read_csv(tmp_path / "TCS.csv", parse_dates=["date"])
    assert n == 30 and used == "NSE bhavcopy + TejHQ/Yahoo"
    assert (saved["close"].iloc[-10:] == 2.0).all() and (saved["close"].iloc[:-10] == 1.0).all()


def test_live_rows_prefer_nse_quote():
    t = pd.date_range("2026-11-03 09:15", periods=30, freq="1min")
    bars = pd.DataFrame({"datetime": t, "open": 99.0, "high": 99.5, "low": 98.5, "close": 99.0, "volume": 10})
    preds = pd.DataFrame({"symbol": ["TCS", "INFY"], "close": [100.0, 50.0], "pred_open": [100.5, 50.0],
                          "pred_gap_pct": [0.5, 0.0]})
    quotes = {s: {"ltp": 101.0, "open": 100.5, "prev_close": 100.0, "pct_change": 1.0,
                  "as_of": datetime(2026, 11, 3, 9, 50)} for s in ("TCS", "INFY")}

    class Cfg:
        or_min, bias_src, capital, risk, reward, leverage, buffer_pct, brokerage, window = (
            15, "Both directions", 100000, 1.0, 2.0, 5, 0.05, 20, 60)
    table, _ = live.live_rows(preds, {"TCS": bars}, Cfg, quotes)
    tcs = table[table["Stock"] == "TCS"].iloc[0]
    assert tcs["LTP (₹)"] == 101.0 and tcs["Source"] == "NSE" and tcs["Forecast direction right?"] == "Yes"
    infy = table[table["Stock"] == "INFY"].iloc[0]
    assert infy["Status"] == "No minute candles" and infy["LTP (₹)"] == 101.0
    qt = live.quote_table(preds, quotes)
    assert list(qt["Stock"]) == ["TCS", "INFY"]


def test_ask_runs_tools_then_answers(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(nse_mcp, "list_tools", lambda server: [
        {"name": n, "description": "d", "inputSchema": {"type": "object"}} for n in ask_market.TOOLS[server]])
    monkeypatch.setattr(nse_mcp, "call", lambda server, name, args: {"stock": {"lastTradedPrice": 1368.1}})
    seen = []

    def fake_chat(p, messages, tools):
        seen.append(messages[-1])
        if messages[-1]["role"] == "user":
            return {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "cm_get_stock_quote", "arguments": '{"symbol": "RELIANCE"}'}}]}}]}
        return {"choices": [{"message": {"content": "RELIANCE is at ₹1,368.10."}}]}

    answer, used = ask_market.ask("Price of Reliance?", chat=fake_chat)
    assert answer == "RELIANCE is at ₹1,368.10."
    assert used == [{"tool": "cm_get_stock_quote", "args": {"symbol": "RELIANCE"}}]
    assert seen[-1]["role"] == "tool" and "1368.1" in seen[-1]["content"]
    assert ask_market.provider()["id"] == "groq"


def test_no_key_means_no_provider(monkeypatch):
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    assert ask_market.provider() is None


def test_retired_model_is_replaced():
    groq = ["whisper-large-v3", "llama-3.1-8b-instant", "openai/gpt-oss-120b", "meta-llama/llama-guard-4-12b"]
    assert ask_market.pick_model("groq", groq) == "openai/gpt-oss-120b"
    assert ask_market.pick_model("groq", ["whisper-large-v3", "llama-3.1-8b-instant"]) == "llama-3.1-8b-instant"
    assert ask_market.pick_model("gemini", ["text-embedding-004", "gemini-2.5-flash"]) == "gemini-2.5-flash"
