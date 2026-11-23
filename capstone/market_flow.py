"""Market flow: which sectors and stocks are drawing today's trading, and in which direction.

Data: NSE's own cash-market feed (mcp.nseindia.in/cmmkt, the `cm_get_equity_stocks` tool), about
1 minute behind the exchange. The tool searches by symbol prefix and returns up to 500 stocks per
call, so one call per starting letter (about 20 calls, run 4 at a time, roughly 5 seconds) brings
the whole universe. The dashboard caches the result for everyone for one minute.

Universe: Nifty 50 and Nifty Next 50 stocks with a plain sector label (data/nifty100_sectors.csv).
NSE's feed has no sector or index-membership tool, so the list is kept by hand; edit the CSV when
NSE rebalances the indices (every March and September).

What is measured and what is estimated
--------------------------------------
* Traded value (turnover, in ₹ crore) and % change are reported by NSE. They are facts.
* "Net flow" is an ESTIMATE. Every share bought is a share sold, so money does not literally flow
  into a stock. What can be seen is who was more eager: when a stock trades near its high of the
  day, buyers kept paying up; near its low, sellers kept accepting less. The estimate scales the
  day's traded value by where the price sits in the day's range (the "close location value" used
  in the Chaikin money-flow indicator):

      position = ((price - low) - (high - price)) / (high - low)      from -1 (at the low) to +1 (at the high)
      net flow = traded value x position

  A stock with ₹1,000 crore traded and the price three quarters of the way up its range gets
  +₹500 crore. Added up by sector, it shows which sectors had buying or selling pressure.

  The sector-flow chart splits the same traded value into a buyer-driven part, value x (1 + position) / 2,
  and a seller-driven part, the rest (₹750 crore and ₹250 crore in the example). Their difference is the
  net flow above. Showing both keeps the chart readable on a day when nearly every stock fell.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from capstone import nse_mcp
from capstone.paths import DATA

UNIVERSE_FILE = DATA / "nifty100_sectors.csv"
WORKERS = 4  # parallel NSE calls; each one is mostly waiting on the network, not CPU
FIELDS = {"lastTradedPrice": "ltp", "openPrice": "open", "highPrice": "high", "lowPrice": "low",
          "preClosePrice": "prev_close", "perChange": "pct_change", "volume": "volume", "value": "turnover_cr",
          "latestTimestamp": "as_of"}


def universe() -> pd.DataFrame:
    """symbol, name, sector, index for the stocks shown on the page."""
    return pd.read_csv(UNIVERSE_FILE).drop_duplicates("symbol")


# A few long-lived MCP sessions, one per worker, so each call does not repeat the handshake.
_pool: list[nse_mcp._Session] = []
_pool_lock = threading.Lock()


def _sessions() -> list[nse_mcp._Session]:
    with _pool_lock:
        while len(_pool) < WORKERS:
            _pool.append(nse_mcp._Session(nse_mcp.MARKET_URL, timeout=20))
    return _pool


def _by_prefix(session: nse_mcp._Session, prefixes: list[str]) -> list[dict]:
    import json
    rows = []
    for p in prefixes:
        try:
            res = session.request("tools/call", {"name": "cm_get_equity_stocks",
                                                 "arguments": {"limit": 500, "symbolFilter": p}}, retries=1)
            text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
            rows += json.loads(text).get("stocks", []) if not res.get("isError") else []
        except (nse_mcp.NseMcpError, ValueError):
            continue  # one letter failing leaves those stocks out; the page says how many are missing
    return rows


def fetch_quotes(symbols: list[str]) -> pd.DataFrame:
    """Live quotes for the given NSE symbols, one row each (stocks NSE did not return are left out)."""
    prefixes = sorted({s[0] for s in symbols})
    sessions = _sessions()
    groups = [prefixes[i::WORKERS] for i in range(WORKERS)]
    with ThreadPoolExecutor(WORKERS) as ex:
        parts = list(ex.map(_by_prefix, sessions, groups))
    wanted = set(symbols)
    rows = [r for part in parts for r in part if r.get("symbol") in wanted and r.get("series", "EQ") == "EQ"]
    if not rows:
        return pd.DataFrame(columns=["symbol", *FIELDS.values()])
    df = pd.DataFrame(rows).drop_duplicates("symbol")[["symbol", *FIELDS]].rename(columns=FIELDS)
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    return df


_latest: dict = {"quotes": None, "at": 0.0, "busy": False, "prev": None, "prev_at": 0.0}
_latest_lock = threading.Lock()


def latest_quotes(max_age: float = 60) -> tuple[pd.DataFrame | None, float]:
    """Quotes for the universe, shared by every visitor in this server process: (quotes, fetched at).

    The first call waits for the download (~5 seconds). After that, a visitor always gets the last copy
    at once; if it is older than max_age seconds, one background thread fetches a fresh copy for the
    next visitor. So at most one download runs at a time, however many people have the page open.
    """
    import time
    with _latest_lock:
        q, at = _latest["quotes"], _latest["at"]
        due = (q is None or time.time() - at > max_age) and not _latest["busy"]
        if due:
            _latest["busy"] = True

    def refresh():
        try:
            fresh = fetch_quotes(list(universe()["symbol"]))
            with _latest_lock:
                if len(fresh) or _latest["quotes"] is None:
                    _latest["prev"], _latest["prev_at"] = _latest["quotes"], _latest["at"]
                    _latest["quotes"], _latest["at"] = fresh, time.time()
        finally:
            _latest["busy"] = False

    if due and q is None:
        refresh()  # nothing to show yet: wait for the first download
    elif due:
        threading.Thread(target=refresh, daemon=True).start()
    elif q is None:  # another visitor's first download is running: wait for it
        for _ in range(60):
            time.sleep(0.5)
            if _latest["quotes"] is not None:
                break
    return _latest["quotes"], _latest["at"]


def snapshot(quotes: pd.DataFrame, uni: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join quotes to sectors and add the range position and the net-flow estimate (see module doc)."""
    uni = universe() if uni is None else uni
    df = uni.merge(quotes, on="symbol", how="inner")
    for c in ("ltp", "open", "high", "low", "prev_close", "pct_change", "volume", "turnover_cr"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ltp", "turnover_cr"])
    df = df[df["turnover_cr"] > 0].copy()
    span = df["high"] - df["low"]
    pos = ((df["ltp"] - df["low"]) - (df["high"] - df["ltp"])) / span.where(span > 0)
    # No range yet (one trade, or a frozen stock): fall back to the direction of the day's change.
    df["position"] = pos.fillna(np.sign(df["pct_change"].fillna(0))).clip(-1, 1)
    df["net_flow_cr"] = df["turnover_cr"] * df["position"]
    # The same estimate split two ways: the buyer-driven and seller-driven parts of the traded value.
    # At the day's high all of it counts as buying, at the low all as selling, in the middle half each.
    df["buy_cr"] = df["turnover_cr"] * (1 + df["position"]) / 2
    df["sell_cr"] = df["turnover_cr"] - df["buy_cr"]
    return df.reset_index(drop=True)


def pace(snap: pd.DataFrame, prev: pd.DataFrame | None = None, now_minutes: float | None = None,
         seconds: float | None = None) -> pd.Series:
    """How fast each stock traded in the last download interval compared with its average pace today.

    2.0 means it traded twice as many rupees per minute as its average minute so far today. Uses the
    previous one-minute download, so it is empty until the page has been open for two downloads.
    Both numbers are reported by NSE (no estimate involved).
    """
    if prev is None:
        prev, seconds = _latest.get("prev"), _latest.get("at", 0) - _latest.get("prev_at", 0)
    if prev is None or prev.empty or snap.empty:
        return pd.Series(dtype=float)
    if "as_of" in prev and prev["as_of"].notna().any() and snap["as_of"].notna().any():
        seconds = (snap["as_of"].max() - prev["as_of"].max()).total_seconds()  # NSE's own clock
    if not seconds or seconds < 30:
        return pd.Series(dtype=float)
    if now_minutes is None:
        t = snap["as_of"].max()
        now_minutes = (t.hour * 60 + t.minute) - (9 * 60 + 15) if pd.notna(t) else 0
    if now_minutes < 5:
        return pd.Series(dtype=float)
    before = prev.set_index("symbol")["turnover_cr"]
    now = snap.set_index("symbol")["turnover_cr"]
    delta = (now - before.reindex(now.index)).clip(lower=0)
    per_min_now = delta / (seconds / 60)
    per_min_avg = now / now_minutes
    return (per_min_now / per_min_avg.where(per_min_avg > 0)).dropna()


def sector_table(snap: pd.DataFrame) -> pd.DataFrame:
    """Per sector: traded value, estimated net flow, turnover-weighted % change, stocks up/down."""
    if snap.empty:
        return pd.DataFrame(columns=["sector", "turnover_cr", "net_flow_cr", "buy_cr", "sell_cr", "pct_change", "stocks", "up", "down"])
    g = snap.assign(w=snap["pct_change"] * snap["turnover_cr"], buy=snap["buy_cr"], sell=snap["sell_cr"],
                    up=(snap["pct_change"] > 0).astype(int), down=(snap["pct_change"] < 0).astype(int))
    out = g.groupby("sector").agg(turnover_cr=("turnover_cr", "sum"), net_flow_cr=("net_flow_cr", "sum"),
                                  buy_cr=("buy", "sum"), sell_cr=("sell", "sum"), w=("w", "sum"), stocks=("symbol", "count"), up=("up", "sum"),
                                  down=("down", "sum")).reset_index()
    out["pct_change"] = out["w"] / out["turnover_cr"]
    out["flow_share"] = out["net_flow_cr"] / out["turnover_cr"]  # -1..+1: how one-sided the sector was
    return out.drop(columns="w").sort_values("net_flow_cr", ascending=False).reset_index(drop=True)


def headline(snap: pd.DataFrame, sectors: pd.DataFrame) -> str:
    """One plain sentence for a beginner: breadth, the busiest sector, strongest buying and selling."""
    if snap.empty:
        return ""
    up, down = int((snap["pct_change"] > 0).sum()), int((snap["pct_change"] < 0).sum())
    busiest = sectors.sort_values("turnover_cr", ascending=False).iloc[0]
    best, worst = sectors.iloc[0], sectors.iloc[-1]
    mood = "more stocks are rising than falling" if up > down else (
        "more stocks are falling than rising" if down > up else "rising and falling stocks are balanced")
    parts = [f"{up} of {len(snap)} large stocks are up and {down} down, so {mood}.",
             f"The most money is changing hands in {busiest['sector']} (₹{busiest['turnover_cr']:,.0f} cr)."]
    if best["net_flow_cr"] > 0:
        parts.append(f"Buyers look strongest in {best['sector']}")
    if worst["net_flow_cr"] < 0:
        parts.append(("and sellers" if best["net_flow_cr"] > 0 else "Sellers look strongest") +
                     f" in {worst['sector']}")
    return " ".join(parts[:2]) + (" " + " ".join(parts[2:]) + "." if len(parts) > 2 else "")
