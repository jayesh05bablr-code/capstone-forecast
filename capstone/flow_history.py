"""Daily history for the Market flow page: 1 Day / 5 Days / 1 Month views and relative volume.

Source: NSE's official end-of-day bhavcopy (mcp.nseindia.in/bhavcopy, `get_stock_history`), one call per
stock for the last ~3 months. Nothing here is made up: the 5-day and 1-month views add up the daily
traded value and volume NSE published, and the price change is close-to-close. Intraday history is NOT
reconstructed (NSE's free feeds give no stored intraday snapshots), so "Intraday" always means the live
snapshot.

The history is downloaded in the background, at most once per trading day per stock, saved to
data/processed/flow_history.csv and shared by every visitor. About 1.5 s per stock with 4 calls in
parallel: ~40 s for the Nifty 100 and ~3 minutes the first time the Nifty 500 is asked for.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from capstone import market_calendar, nse_mcp
from capstone.paths import PROCESSED

HISTORY_FILE = PROCESSED / "flow_history.csv"
COLS = ["date", "symbol", "open", "high", "low", "close", "prev_close", "volume", "turnover"]
PERIODS = {"1 Day": 1, "5 Days": 5, "1 Month": 21}  # trading sessions
WORKERS = 6  # parallel bhavcopy calls, each on its own MCP session (one shared session would serialise them)
AVG_DAYS = 20  # relative volume = volume / average of the previous 20 sessions

_lock = threading.Lock()
_state = {"busy": False, "done": 0, "total": 0, "error": "", "data": None, "mtime": 0.0}


def load() -> pd.DataFrame:
    """All saved daily rows (cached in memory until the file changes)."""
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=COLS)
    m = HISTORY_FILE.stat().st_mtime
    if _state["data"] is None or m != _state["mtime"]:
        df = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
        _state["data"], _state["mtime"] = df, m
    return _state["data"]


def expected_last_session(now=None) -> date:
    """The newest session whose bhavcopy should be out: today after ~18:30 IST, else the one before."""
    now = now or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    today = now.date()
    if market_calendar.is_trading_day(today) and now.hour * 60 + now.minute >= 18 * 60 + 30:
        return today
    return market_calendar.previous_trading_day(today)


def missing(symbols: list[str], hist: pd.DataFrame | None = None, now=None) -> list[str]:
    """Stocks whose saved history is absent or older than the last published session."""
    hist = load() if hist is None else hist
    checked = _state.get("checked", {})
    symbols = [s for s in symbols if checked.get(s) != date.today()]  # asked today already (e.g. REITs: no rows)
    if hist.empty:
        return list(symbols)
    last = hist.groupby("symbol")["date"].max()
    want = pd.Timestamp(expected_last_session(now))
    # A stock already checked today but still behind (bhavcopy late) is not asked again until tomorrow.
    return [s for s in symbols if s not in last.index or last[s] < want - pd.Timedelta(days=6)
            or (last[s] < want and _state.get("checked", {}).get(s) != date.today())]


_local = threading.local()


def _one(symbol: str) -> list[dict]:
    import json
    if not hasattr(_local, "session"):
        _local.session = nse_mcp._Session(nse_mcp.BHAVCOPY_URL, timeout=30)
    try:
        res = _local.session.request("tools/call", {"name": "get_stock_history",
                                                     "arguments": {"symbol": symbol, "months": 3}}, retries=2)
        text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        payload = json.loads(text) if not res.get("isError") else {}
    except (nse_mcp.NseMcpError, ValueError, OSError):
        return []
    rows = nse_mcp._rows(payload) if isinstance(payload, dict) else []
    for r in rows:
        r["symbol"] = symbol
    return rows


def refresh(symbols: list[str], block: bool = False) -> None:
    """Download the missing stocks' history in a background thread (one at a time for the server)."""
    todo = missing(symbols)
    with _lock:
        if _state["busy"] or not todo:
            return
        _state.update(busy=True, done=0, total=len(todo), error="")

    def run():
        try:
            got = []
            with ThreadPoolExecutor(WORKERS) as ex:
                for rows in ex.map(_one, todo):
                    got += rows
                    _state["done"] += 1
            checked = _state.setdefault("checked", {})
            for s in todo:
                checked[s] = date.today()
            if not got:
                _state["error"] = "NSE's bhavcopy server returned no history"
                return
            new = pd.DataFrame(got)
            new["date"] = pd.to_datetime(new["date"])
            old = load()
            both = pd.concat([old, new[COLS]], ignore_index=True) if len(old) else new[COLS]
            both = both.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
            cutoff = both["date"].max() - pd.Timedelta(days=150)  # keep ~5 months
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = HISTORY_FILE.with_suffix(".tmp")
            both[both["date"] >= cutoff].to_csv(tmp, index=False)
            tmp.replace(HISTORY_FILE)
        except Exception as e:  # never break the page over history
            _state["error"] = str(e)
        finally:
            _state["busy"] = False

    if block:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()


def progress() -> dict:
    return {k: _state[k] for k in ("busy", "done", "total", "error")}


def period_snapshot(hist: pd.DataFrame, uni: pd.DataFrame, sessions: int) -> pd.DataFrame:
    """A snapshot-shaped table (same columns as market_flow.snapshot) over the last N published sessions.

    traded value and volume = sums over the sessions; price change = last close vs the close before the
    window; high/low = the window's range; the buyer/seller split is the sum of each day's estimate
    (that day's value x where the close sat in that day's range), so it is still an ESTIMATE.
    """
    if hist.empty:
        return pd.DataFrame()
    h = hist[hist["symbol"].isin(uni["symbol"])].sort_values(["symbol", "date"]).copy()
    for c in ("open", "high", "low", "close", "prev_close", "volume", "turnover"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    dates = np.sort(h["date"].unique())
    if len(dates) == 0:
        return pd.DataFrame()
    window = dates[-sessions:]
    w = h[h["date"].isin(window)].copy()
    span = (w["high"] - w["low"]).where(lambda s: s > 0)
    w["pos"] = (((w["close"] - w["low"]) - (w["high"] - w["close"])) / span).fillna(
        np.sign(w["close"] - w["prev_close"]).fillna(0)).clip(-1, 1)
    w["tv_cr"] = w["turnover"] / 1e7
    w["buy"] = w["tv_cr"] * (1 + w["pos"]) / 2
    g = w.groupby("symbol")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "ltp": g["close"].last(),
        "prev_close": g["prev_close"].first(), "volume": g["volume"].sum(), "turnover_cr": g["tv_cr"].sum(),
        "buy_cr": g["buy"].sum(), "days": g["date"].nunique(), "as_of": g["date"].max()}).reset_index()
    out = out[out["days"] == len(window)]  # a stock missing a day in the window would mislead
    out["pct_change"] = 100 * (out["ltp"] / out["prev_close"] - 1)
    out = uni.merge(out, on="symbol", how="inner")
    out = out[out["turnover_cr"] > 0].copy()
    out["sell_cr"] = out["turnover_cr"] - out["buy_cr"]
    out["net_flow_cr"] = out["buy_cr"] - out["sell_cr"]
    out["position"] = (out["net_flow_cr"] / out["turnover_cr"]).clip(-1, 1)
    out["vwap"] = (out["turnover_cr"] * 1e7 / out["volume"].where(out["volume"] > 0)).round(2)
    out["from_date"], out["to_date"] = pd.Timestamp(window[0]), pd.Timestamp(window[-1])
    return out.drop(columns="days").reset_index(drop=True)


def avg_volume(hist: pd.DataFrame, before=None, days: int = AVG_DAYS) -> pd.Series:
    """Average daily volume per stock over the `days` sessions before `before` (default: all saved)."""
    if hist.empty:
        return pd.Series(dtype=float)
    h = hist if before is None else hist[hist["date"] < pd.Timestamp(before)]
    h = h.sort_values("date").groupby("symbol").tail(days)
    n = h.groupby("symbol")["volume"].count()
    return h.groupby("symbol")["volume"].mean()[n >= min(days, 10)]


def session_fraction(now=None) -> float:
    """Share of the 375-minute session that has passed (1 outside market hours)."""
    now = now or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    if not market_calendar.market_open(now.to_pydatetime() if hasattr(now, "to_pydatetime") else now):
        return 1.0
    mins = now.hour * 60 + now.minute - (9 * 60 + 15)
    return min(max(mins, 5) / 375, 1.0)


def relative_volume(snap: pd.DataFrame, hist: pd.DataFrame, intraday: bool, now=None) -> pd.Series:
    """Volume vs the 20-session average, per symbol. During market hours the live volume is scaled up by
    the share of the session gone (volume so far / fraction of the day), so 1.0 means a normal pace."""
    if snap.empty or hist.empty:
        return pd.Series(dtype=float)
    as_of = snap["as_of"].max() if intraday else snap["from_date"].min()
    avg = avg_volume(hist, before=pd.Timestamp(as_of).normalize())
    sessions = 1 if intraday else max(1, len(np.unique(hist[(hist["date"] >= snap["from_date"].min()) &
                                                              (hist["date"] <= snap["to_date"].max())]["date"])))
    frac = session_fraction(now) if intraday else 1.0
    vol = snap.set_index("symbol")["volume"].astype(float)
    return (vol / frac / sessions / avg.reindex(vol.index)).replace([np.inf, -np.inf], np.nan).dropna()
