"""Morning brief: one page that gathers what the app already knows before the 09:15 IST open.

Every line comes from data the app fetched, with its source and date; a section with no data says so
instead of guessing. Nothing here is a recommendation.

Sources
-------
US overnight, USD/INR, forecast   : the app's own forecast files (overnight_story.json, predictions.csv)
Asia                              : Yahoo Finance daily closes (Nikkei 225, Hang Seng, KOSPI), best effort
NIFTY, VIX, breadth, valuation    : NSE bhavcopy MCP (get_market_mood, get_index_valuation), end of day
NIFTY right now                   : NSE cash-market MCP cm_get_index_quote (last traded / pre-open value)
Sector outlook                    : sector_intel leaderboard (official NSE sector indices)
Corporate actions                 : NSE bhavcopy MCP get_corporate_actions for NIFTY 50 stocks, ex-dates in
                                    the next 7 days (dividends, splits, bonuses)
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

from capstone import nse_mcp
from capstone.paths import PROCESSED

ASIA = {"^N225": "Nikkei 225 (Japan)", "^HSI": "Hang Seng (Hong Kong)", "^KS11": "KOSPI (Korea)"}
ACTIONS_FILE = PROCESSED / "upcoming_corporate_actions.csv"
YAHOO_DAILY = "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&range=10d"

_cache: dict = {}
_lock = threading.Lock()
_actions_busy = {"busy": False}


def _cached(key: str, max_age: float, fn):
    """Shared, time-limited cache for small network results (one fetch serves every visitor)."""
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < max_age:
            return hit[1]
    try:
        val = fn()
    except Exception:
        val = None
    if val is not None:
        with _lock:
            _cache[key] = (time.time(), val)
    elif hit:
        return hit[1]  # keep the last good copy if a refresh fails
    return val


# ---------- NSE ----------

def market_mood() -> dict | None:
    return _cached("mood", 1800, lambda: nse_mcp.call("bhavcopy", "get_market_mood", {}))


def valuation() -> dict | None:
    return _cached("val", 6 * 3600, lambda: nse_mcp.call("bhavcopy", "get_index_valuation",
                                                         {"indexName": "Nifty 50", "months": 24}))


def nifty_live() -> dict | None:
    """NIFTY 50 from NSE's live index feed (last, change %, time) or None."""
    def get():
        res = nse_mcp.call("market", "cm_get_index_quote", {"indexName": "NIFTY 50"}) or {}
        ix = res.get("index") or {}
        if ix.get("last") is None:
            return None
        return {"last": ix["last"], "pct": ix.get("percChange"), "prev": ix.get("previousClose"),
                "time": ix.get("timeVal") or res.get("nseTimestamp"),
                "chg_1m": ((ix.get("comparisons") or {}).get("oneMonthAgo") or {}).get("changePct")}
    return _cached("nifty_live", 60, get)


# ---------- Asia (best effort) ----------

def _parse_daily(body: dict) -> dict | None:
    r = ((body.get("chart") or {}).get("result") or [None])[0]
    if not r:
        return None
    closes = [c for c in ((r.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []]
    stamps = r.get("timestamp") or []
    pts = [(t, c) for t, c in zip(stamps, closes) if c is not None]
    if len(pts) < 2:
        return None
    meta = r.get("meta") or {}
    last_t, last = pts[-1]
    price = meta.get("regularMarketPrice") or last
    prev = pts[-2][1] if price == last else last
    when = pd.Timestamp(meta.get("regularMarketTime") or last_t, unit="s", tz="UTC").tz_convert("Asia/Kolkata")
    return {"close": float(price), "pct": 100 * (float(price) / float(prev) - 1), "as_of": when.tz_localize(None)}


def asia() -> list[dict]:
    from capstone.http_json import get_json
    from capstone.intraday import BROWSER

    def one(t):
        try:
            body = get_json(YAHOO_DAILY.format(urllib.parse.quote(t)), f"Yahoo {t}", BROWSER, retries=1)
            p = _parse_daily(body)
            return None if p is None else {"ticker": t, "name": ASIA[t], **p}
        except Exception:
            return None

    def get():
        with ThreadPoolExecutor(3) as ex:
            rows = [r for r in ex.map(one, ASIA) if r]
        return rows or None
    return _cached("asia", 900, get) or []


# ---------- corporate actions ----------

def load_actions() -> pd.DataFrame:
    if not ACTIONS_FILE.exists():
        return pd.DataFrame(columns=["symbol", "ex_date", "purpose", "fetched"])
    return pd.read_csv(ACTIONS_FILE)


def refresh_actions(symbols: list[str], days: int = 7) -> None:
    """NIFTY 50 corporate actions with an ex-date in the next `days` days, once a day, in the background."""
    have = load_actions()
    if len(have) and str(have["fetched"].iloc[0]) == date.today().isoformat():
        return
    with _lock:
        if _actions_busy["busy"]:
            return
        _actions_busy["busy"] = True
    frm, to = date.today().isoformat(), (date.today() + timedelta(days=days)).isoformat()
    local = threading.local()

    def one(sym):
        try:
            if not hasattr(local, "s"):
                local.s = nse_mcp._Session(nse_mcp.BHAVCOPY_URL, timeout=30)
            res = local.s.request("tools/call", {"name": "get_corporate_actions",
                                                 "arguments": {"symbol": sym, "fromDate": frm, "toDate": to}}, retries=1)
            text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
            acts = (json.loads(text) or {}).get("actions") or []
        except Exception:
            return None
        return [{"symbol": sym, "ex_date": a.get("exDate") or a.get("ex_date") or a.get("date"),
                 "purpose": a.get("purpose") or a.get("subject") or a.get("type") or a.get("action")} for a in acts]

    def run():
        try:
            with ThreadPoolExecutor(6) as ex:
                parts = list(ex.map(one, symbols))
            if all(p is None for p in parts):
                return  # NSE did not answer: keep yesterday's file, try again later
            rows = [r for p in parts if p for r in p]
            df = pd.DataFrame(rows, columns=["symbol", "ex_date", "purpose"])
            df["fetched"] = date.today().isoformat()
            if df.empty:  # remember that there are none, so we do not ask again today
                df = pd.DataFrame([{"symbol": "", "ex_date": "", "purpose": "", "fetched": date.today().isoformat()}])
            ACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = ACTIONS_FILE.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(ACTIONS_FILE)
        finally:
            _actions_busy["busy"] = False

    threading.Thread(target=run, daemon=True).start()


def upcoming_actions() -> pd.DataFrame:
    a = load_actions()
    return a[a["symbol"].fillna("").astype(str).str.len() > 0].sort_values("ex_date").reset_index(drop=True)


# ---------- risks and highlights (pure, testable) ----------

def risks(mood: dict | None, alerts: list[str], story: dict | None, val: dict | None) -> list[str]:
    """Plain risk notes, each tied to a published number."""
    out = []
    vix = (mood or {}).get("india_vix") or {}
    pct = (vix.get("range") or {}).get("percentile")
    if vix.get("level") is not None and pct is not None and pct >= 75:
        out.append(f"India VIX is {vix['level']:.1f}, higher than on {pct:.0f}% of days in the past year: "
                   "expect wider swings than usual.")
    wk = vix.get("vs_1w_ago") or {}
    if wk.get("change_pct") is not None and wk["change_pct"] >= 10:
        out.append(f"India VIX rose {wk['change_pct']:.0f}% in a week (from {wk.get('base_level')}).")
    sb = (mood or {}).get("stock_breadth") or {}
    if sb.get("reading") == "broad decline":
        out.append(f"Last session was a broad decline: only {sb.get('up_pct', 0):.0f}% of NSE stocks rose.")
    out += alerts
    if story is not None and not story.get("overnight_complete", True):
        out.append("The overnight US session is not in the data yet, so the forecast is missing part of its signal.")
    pe = (val or {}).get("pe") or {}
    if pe.get("percentile") is not None and pe["percentile"] >= 85:
        out.append(f"NIFTY 50 P/E {pe['current']:.1f} is near the top of its 2-year range ({pe['percentile']:.0f}th "
                   "percentile): less cushion if earnings disappoint.")
    return out


def forecast_highlights(preds: pd.DataFrame, n: int = 3) -> dict:
    """Biggest expected gaps up and down from the model's forecast."""
    if preds is None or preds.empty or "pred_gap_pct" not in preds:
        return {"up": [], "down": []}
    p = preds.sort_values("pred_gap_pct")
    cols = [c for c in ("symbol", "pred_gap_pct", "pred_open", "close", "reason") if c in p]
    up = p[p["pred_gap_pct"] > 0].tail(n).iloc[::-1][cols].to_dict("records")
    down = p[p["pred_gap_pct"] < 0].head(n)[cols].to_dict("records")
    return {"up": up, "down": down}
