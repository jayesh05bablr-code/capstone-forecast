"""Client for NSE's official public MCP servers (no key, no IP whitelist).

Two servers, both free and open:

* Bhavcopy (end-of-day, official NSE bhavcopy files, last ~5 years):
  https://mcp.nseindia.in/bhavcopy/cm/mcp
* Cash market (near-real-time quotes, refreshed every 1-3 minutes):
  https://mcp.nseindia.in/cmmkt/mcp

MCP over "streamable HTTP" is plain JSON-RPC sent with POST, so this module needs only the
standard library: no extra package to install on Streamlit Community Cloud. Each call opens a
short session (initialize, then the tool call) and reuses it for later calls in the same process.

Every tool returns its data as JSON text inside the MCP reply; `call()` parses it for you.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta

BHAVCOPY_URL = os.environ.get("NSE_MCP_BHAVCOPY_URL", "https://mcp.nseindia.in/bhavcopy/cm/mcp")
MARKET_URL = os.environ.get("NSE_MCP_MARKET_URL", "https://mcp.nseindia.in/cmmkt/mcp")
SERVERS = {"bhavcopy": BHAVCOPY_URL, "market": MARKET_URL}
PROTOCOL = "2025-03-26"
RETRYABLE = {429, 500, 502, 503, 504}
HISTORY_YEARS = 5  # the bhavcopy server keeps about five years of daily data


class NseMcpError(RuntimeError):
    network = False  # True when mcp.nseindia.in could not be reached at all


class _Session:
    def __init__(self, url: str, timeout: float = 30):
        self.url, self.timeout = url, timeout
        self.sid: str | None = None
        self.next_id = 0
        self.lock = threading.Lock()

    def _post(self, body: dict) -> dict | None:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL, "User-Agent": "nse-capstone/0.3"}
        if self.sid:
            headers["Mcp-Session-Id"] = self.sid
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            self.sid = resp.headers.get("Mcp-Session-Id") or self.sid
            text = resp.read().decode("utf-8")
        if "id" not in body or not text.strip():
            return None  # a notification: no answer expected
        if text.lstrip().startswith("{"):
            return json.loads(text)
        for line in text.splitlines():  # an event-stream answer: the JSON is on a data: line
            if line.startswith("data:"):
                msg = json.loads(line[5:])
                if msg.get("id") == body["id"]:
                    return msg
        raise NseMcpError("NSE MCP sent an answer this app cannot read")

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        body = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}
        msg = self._post(body)
        if msg and "error" in msg:
            raise NseMcpError(f"NSE MCP {method}: {msg['error'].get('message', msg['error'])}")
        return (msg or {}).get("result", {})

    def _open(self) -> None:
        self.sid = None
        self._rpc("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                 "clientInfo": {"name": "nse-capstone", "version": "0.3"}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def request(self, method: str, params: dict | None = None, retries: int = 3) -> dict:
        with self.lock:
            for attempt in range(retries + 1):
                try:
                    if self.sid is None:
                        self._open()
                    return self._rpc(method, params)
                except urllib.error.HTTPError as exc:
                    self.sid = None  # the server may have dropped the session: start a new one
                    if (exc.code in RETRYABLE or exc.code in (400, 404)) and attempt < retries:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise NseMcpError(f"NSE MCP failed with HTTP {exc.code}") from None
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    self.sid = None
                    if attempt < retries:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    err = NseMcpError(f"Could not reach mcp.nseindia.in ({getattr(exc, 'reason', exc)}). "
                                      "Is the site blocked by your network?")
                    err.network = True
                    raise err from None
        raise NseMcpError("NSE MCP failed")


_sessions: dict[str, _Session] = {}
_tools_cache: dict[str, list[dict]] = {}


def _session(server: str) -> _Session:
    if server not in _sessions:
        _sessions[server] = _Session(SERVERS[server])
    return _sessions[server]


def list_tools(server: str) -> list[dict]:
    """The tools a server offers: name, description, inputSchema (cached per process)."""
    if server not in _tools_cache:
        _tools_cache[server] = _session(server).request("tools/list").get("tools", [])
    return _tools_cache[server]


def call(server: str, tool: str, args: dict | None = None):
    """Run one tool and return its parsed JSON answer (or the raw text if it is not JSON)."""
    result = _session(server).request("tools/call", {"name": tool, "arguments": args or {}})
    text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
    if result.get("isError"):
        raise NseMcpError(f"{tool}: {text[:300]}")
    try:
        return json.loads(text)
    except ValueError:
        return text


# ---------- End-of-day prices (bhavcopy) ----------

COLUMNS = ["date", "open", "high", "low", "close", "prev_close", "volume", "turnover"]


def _rows(payload: dict) -> list[dict]:
    out = []
    for r in payload.get("data") or []:
        if str(r.get("series", "EQ")).upper() not in ("EQ", "BE", "BZ"):
            continue
        out.append({"date": r.get("date"), "open": r.get("open"), "high": r.get("high"), "low": r.get("low"),
                    "close": r.get("close"), "prev_close": r.get("prevClose"), "volume": r.get("volume"),
                    "turnover": r.get("totalTradedValue")})
    return out


def stock_history(symbol: str, start: str, end: str | None = None) -> list[dict]:
    """Daily OHLCV for one stock between start and end (yyyy-mm-dd), oldest first.

    The server returns at most 3 months per call, so this walks back in 3-month chunks until it
    passes `start` or runs out of history (about 5 years). Prices are as NSE published them
    (not adjusted for splits or bonuses), like TejHQ's.
    """
    start_d = date.fromisoformat(start)
    oldest = date.today() - timedelta(days=365 * HISTORY_YEARS + 30)
    start_d = max(start_d, oldest)
    end_d = date.fromisoformat(end) if end else date.today()
    rows: dict[str, dict] = {}
    misses = 0
    while end_d >= start_d and misses < 6:
        span = (end_d - start_d).days
        months = 3 if span > 62 else (2 if span > 31 else 1)
        payload = call("bhavcopy", "get_stock_history", {"symbol": symbol.upper(), "months": months,
                                                          "endDate": end_d.isoformat()})
        if not isinstance(payload, dict) or payload.get("error") or not payload.get("data"):
            # The server has a few gaps (a quarter missing for some stocks): skip back and go on.
            misses += 1
            end_d -= timedelta(days=92)
            continue
        misses = 0
        for r in _rows(payload):
            rows[r["date"]] = r
        end_d = date.fromisoformat(payload["from_date"]) - timedelta(days=1)
    if not rows:
        raise NseMcpError(f"NSE bhavcopy has no daily prices for {symbol}")
    return [rows[d] for d in sorted(rows) if start_d.isoformat() <= d <= (end or "9999")]


def bulk_quote(symbols: list[str]) -> dict[str, dict]:
    """Latest finished day's close for several stocks in one call: {symbol: {close, date, ...}}."""
    payload = call("bhavcopy", "get_bulk_quote", {"symbols": [s.upper() for s in symbols]})
    return {q["symbol"]: q for q in (payload or {}).get("quotes", [])} if isinstance(payload, dict) else {}


# ---------- Live quotes (cash market) ----------

def quote(symbol: str) -> dict:
    """Near-real-time quote: ltp, open, high, low, prev_close, change, pct_change, volume, as_of."""
    payload = call("market", "cm_get_stock_quote", {"symbol": symbol.upper()})
    s = (payload or {}).get("stock") if isinstance(payload, dict) else None
    if not s:
        raise NseMcpError(f"No live quote for {symbol}")
    as_of = s.get("latestTimestamp")
    return {"symbol": s.get("symbol", symbol), "ltp": s.get("lastTradedPrice"), "open": s.get("openPrice"),
            "high": s.get("highPrice"), "low": s.get("lowPrice"), "prev_close": s.get("preClosePrice"),
            "change": s.get("change"), "pct_change": s.get("perChange"), "volume": s.get("volume"),
            "value_cr": s.get("value"), "high_52w": s.get("fiftyTwoWeekHigh"),
            "low_52w": s.get("fiftyTwoWeekLow"),
            "as_of": datetime.strptime(as_of, "%Y-%m-%d %H:%M:%S") if as_of else None}


def quotes(symbols: list[str]) -> dict[str, dict]:
    """Live quotes for several stocks; a stock that fails is left out (never raises)."""
    out = {}
    for s in symbols:
        try:
            out[s] = quote(s)
        except NseMcpError:
            continue
    return out


def index_quote(name: str = "NIFTY 50") -> dict:
    """Live index value, e.g. 'NIFTY 50', 'NIFTY BANK', 'INDIA VIX'."""
    payload = call("market", "cm_get_index_quote", {"indexName": name})
    return payload if isinstance(payload, dict) else {}


def reachable() -> bool:
    try:
        list_tools("market")
        return True
    except NseMcpError:
        return False
