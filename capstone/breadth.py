"""Market breadth and liquidity/risk screens, computed only from published NSE data.

Sources
-------
* Whole-market advances/declines per session: NSE bhavcopy `get_market_breadth` (every NSE equity).
* Universe breadth, % above 20/50-day average, unusual volume, ADTV, volatility, ATR, drawdown: the saved
  daily bhavcopy history of the chosen universe (flow_history).
* % above the 200-day average: NSE bhavcopy `moving_average(symbol, 200)` per stock, fetched in the
  background once a day (the saved history covers only ~5 months, too short for a 200-day average).
* New 52-week highs/lows: the 52-week high/low in NSE's cash-market feed (the live snapshot).

Bhavcopy prices are not adjusted for splits or bonuses, so a stock that split recently can look far
below its 200-day average; the page says so.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from capstone import market_calendar, nse_mcp
from capstone.paths import PROCESSED

MARKET_FILE = PROCESSED / "market_breadth.csv"
DMA_FILE = PROCESSED / "dma200.csv"
SESSIONS = 40  # whole-market A/D history kept
LIQUIDITY = [(500, "Very high"), (100, "High"), (20, "Medium"), (0, "Low")]  # ADTV ₹ cr thresholds (ours)
UNUSUAL = 2.0  # latest volume >= 2x its 20-session average

_state = {"market_busy": False, "dma_busy": False, "error": ""}
_lock = threading.Lock()
_local = threading.local()


def _call(tool: str, args: dict):
    if not hasattr(_local, "s"):
        _local.s = nse_mcp._Session(nse_mcp.BHAVCOPY_URL, timeout=30)
    res = _local.s.request("tools/call", {"name": tool, "arguments": args}, retries=2)
    if res.get("isError"):
        return None
    text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
    try:
        return json.loads(text)
    except ValueError:
        return None


def _safe_call(tool, args):
    try:
        return _call(tool, args)
    except Exception:
        return None


def _write(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _sessions(n: int, until: date) -> list[date]:
    out, d = [], until
    while len(out) < n:
        if market_calendar.is_trading_day(d):
            out.append(d)
        d = d - pd.Timedelta(days=1)
        d = d.date() if hasattr(d, "date") else d
    return out[::-1]


# ---------- whole-market advances / declines ----------

def load_market() -> pd.DataFrame:
    if not MARKET_FILE.exists():
        return pd.DataFrame(columns=["date", "advances", "declines", "unchanged", "total_stocks", "ad_ratio"])
    return pd.read_csv(MARKET_FILE, parse_dates=["date"])


def refresh_market(block: bool = False) -> None:
    from capstone.flow_history import expected_last_session
    have = load_market()
    want = _sessions(SESSIONS, expected_last_session())
    got = set(have["date"].dt.date) if len(have) else set()
    todo = [d for d in want if d not in got]
    with _lock:
        if _state["market_busy"] or not todo:
            return
        _state["market_busy"] = True

    def run():
        try:
            with ThreadPoolExecutor(4) as ex:
                rows = [r for r in ex.map(lambda d: _safe_call("get_market_breadth", {"date": d.isoformat()}), todo)
                        if isinstance(r, dict) and r.get("date")]
            if not rows:
                return
            new = pd.DataFrame(rows)[["date", "advances", "declines", "unchanged", "total_stocks", "ad_ratio"]]
            new["date"] = pd.to_datetime(new["date"])
            both = pd.concat([have, new]) if len(have) else new
            both = both.drop_duplicates("date", keep="last").sort_values("date").tail(SESSIONS + 20)
            _write(both, MARKET_FILE)
        except Exception as e:
            _state["error"] = str(e)
        finally:
            _state["market_busy"] = False

    threading.Thread(target=run, daemon=True).start() if not block else run()


# ---------- 200-day averages ----------

def load_dma() -> pd.DataFrame:
    if not DMA_FILE.exists():
        return pd.DataFrame(columns=["symbol", "dma200", "fetched"])
    return pd.read_csv(DMA_FILE)


def refresh_dma(symbols: list[str], block: bool = False) -> None:
    """Fetch each stock's 200-day average once a day (in the background)."""
    have = load_dma()
    today = date.today().isoformat()
    fresh = set(have.loc[have["fetched"] == today, "symbol"]) if len(have) else set()
    todo = [s for s in symbols if s not in fresh]
    with _lock:
        if _state["dma_busy"] or not todo:
            return
        _state["dma_busy"] = True

    def run():
        try:
            with ThreadPoolExecutor(6) as ex:
                vals = list(ex.map(lambda s: _safe_call("moving_average", {"symbol": s, "days": 200}), todo))
            rows = [{"symbol": s, "dma200": float(v), "fetched": today} for s, v in zip(todo, vals)
                    if isinstance(v, (int, float)) and v > 0]
            new = pd.DataFrame(rows, columns=["symbol", "dma200", "fetched"])
            both = pd.concat([have, new]).drop_duplicates("symbol", keep="last") if len(have) and len(new) else (new if len(new) else have)
            _write(both, DMA_FILE)
        except Exception as e:
            _state["error"] = str(e)
        finally:
            _state["dma_busy"] = False

    threading.Thread(target=run, daemon=True).start() if not block else run()


def status() -> dict:
    return dict(_state)


# ---------- universe breadth and risk from daily history ----------

def stock_stats(hist: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """One row per stock with a published close on the latest session: returns, averages, ADTV, risk."""
    if hist.empty:
        return pd.DataFrame()
    h = hist[hist["symbol"].isin(uni["symbol"])].sort_values(["symbol", "date"])
    last_day = h["date"].max()
    rows = []
    for sym, g in h.groupby("symbol"):
        if len(g) < 6 or g["date"].iloc[-1] != last_day:
            continue
        c, v = g["close"].astype(float).to_numpy(), g["volume"].astype(float).to_numpy()
        hi, lo = g["high"].astype(float).to_numpy(), g["low"].astype(float).to_numpy()
        tv = g["turnover"].astype(float).to_numpy() / 1e7
        prev = np.r_[np.nan, c[:-1]]
        tr = np.nanmax(np.c_[hi - lo, np.abs(hi - prev), np.abs(lo - prev)], axis=1)
        lr = np.diff(np.log(c))
        avg_v = v[-21:-1].mean() if len(v) >= 21 else np.nan
        rows.append({
            "symbol": sym, "close": c[-1], "ret_1d": 100 * (c[-1] / c[-2] - 1), "ret_5d": 100 * (c[-1] / c[-6] - 1),
            "dma20": c[-20:].mean() if len(c) >= 20 else np.nan, "dma50": c[-50:].mean() if len(c) >= 50 else np.nan,
            "adtv_cr": tv[-20:].mean() if len(tv) >= 10 else np.nan,
            "vol_20d": 100 * lr[-20:].std(ddof=1) * np.sqrt(252) if len(lr) >= 20 else np.nan,
            "atr_pct": 100 * tr[-14:].mean() / c[-1] if len(c) >= 15 else np.nan,
            "max_dd": 100 * (c / np.maximum.accumulate(c) - 1).min() if len(c) >= 20 else np.nan,
            "rvol": v[-1] / avg_v if avg_v and avg_v > 0 else np.nan, "sessions": len(c),
            "from": g["date"].iloc[0], "to": last_day})
    s = pd.DataFrame(rows)
    if s.empty:
        return s
    s = uni[["symbol", "name", "sector"]].merge(s, on="symbol")
    s["above_20"] = np.where(s["dma20"].notna(), s["close"] > s["dma20"], np.nan)
    s["above_50"] = np.where(s["dma50"].notna(), s["close"] > s["dma50"], np.nan)
    s["liquidity"] = s["adtv_cr"].apply(liquidity)
    s["unusual_volume"] = s["rvol"] >= UNUSUAL
    return s


def liquidity(adtv_cr: float) -> str | None:
    if adtv_cr is None or pd.isna(adtv_cr):
        return None
    return next(label for th, label in LIQUIDITY if adtv_cr >= th)


def _pct(x: pd.Series) -> float | None:
    x = x.dropna()
    return float(100 * x.astype(float).mean()) if len(x) else None


def universe_breadth(stats: pd.DataFrame, dma: pd.DataFrame | None = None, snap: pd.DataFrame | None = None) -> dict:
    """Breadth of the universe on the latest session. Each figure is None when the data is not there."""
    if stats.empty:
        return {}
    out = {"stocks": len(stats), "advancing": int((stats["ret_1d"] > 0).sum()),
           "declining": int((stats["ret_1d"] < 0).sum()), "unchanged": int((stats["ret_1d"] == 0).sum()),
           "above_20": _pct(stats["above_20"]), "above_50": _pct(stats["above_50"]), "above_200": None,
           "dma200_n": 0, "new_high": None, "new_low": None, "as_of": stats["to"].max()}
    out["ad_ratio"] = out["advancing"] / out["declining"] if out["declining"] else None
    if dma is not None and len(dma):
        m = stats.merge(dma[["symbol", "dma200"]], on="symbol")
        if len(m) >= 0.8 * len(stats):  # only when most stocks have one
            out["above_200"], out["dma200_n"] = _pct(m["close"] > m["dma200"]), len(m)
    if snap is not None and len(snap) and {"high_52w", "low_52w"} <= set(snap.columns):
        s = snap.dropna(subset=["high_52w", "low_52w"])
        if len(s):
            # within 0.5% of the 52-week extreme counts as a new high/low (NSE's figure lags by a session)
            out["new_high"] = int((s["high"] >= 0.995 * s["high_52w"]).sum())
            out["new_low"] = int((s["low"] <= 1.005 * s["low_52w"]).sum())
            out["hl_n"] = len(s)
    return out


def breadth_history(hist: pd.DataFrame, uni: pd.DataFrame, sessions: int = 40) -> pd.DataFrame:
    """Per session: advancing, declining, cumulative A/D line, % above 20-day average (universe)."""
    h = hist[hist["symbol"].isin(uni["symbol"])].sort_values(["symbol", "date"]).copy()
    if h.empty:
        return pd.DataFrame()
    h["chg"] = h.groupby("symbol")["close"].diff()
    h["ma20"] = h.groupby("symbol")["close"].transform(lambda c: c.rolling(20).mean())
    d = h.dropna(subset=["chg"]).groupby("date").agg(
        advancing=("chg", lambda x: int((x > 0).sum())), declining=("chg", lambda x: int((x < 0).sum())),
        above_20=("ma20", lambda x: np.nan), n=("chg", "count")).reset_index()
    a20 = h.dropna(subset=["ma20"]).assign(up=lambda x: x["close"] > x["ma20"]).groupby("date")["up"].mean() * 100
    d["above_20"] = d["date"].map(a20)
    d = d.tail(sessions).reset_index(drop=True)
    d["ad_line"] = (d["advancing"] - d["declining"]).cumsum()
    return d


def divergences(index: pd.DataFrame, bh: pd.DataFrame, lookback: int = 10) -> list[str]:
    """Plain-language alerts when the index and the breadth of the universe disagree over `lookback` sessions."""
    if index is None or index.empty or bh.empty or len(bh) < lookback + 1:
        return []
    ix = index.set_index("date")["close"].reindex(bh["date"]).dropna()
    if len(ix) < lookback + 1:
        return []
    ix_ret = 100 * (ix.iloc[-1] / ix.iloc[-1 - lookback] - 1)
    ad_chg = bh["ad_line"].iloc[-1] - bh["ad_line"].iloc[-1 - lookback]
    a20_chg = bh["above_20"].iloc[-1] - bh["above_20"].iloc[-1 - lookback]
    alerts = []
    if ix_ret > 1 and ad_chg < 0:
        alerts.append(f"NIFTY 50 is up {ix_ret:.1f}% over {lookback} sessions but more stocks fell than rose "
                      f"(A/D line {ad_chg:+.0f}): a narrow rally led by few stocks.")
    if ix_ret < -1 and ad_chg > 0:
        alerts.append(f"NIFTY 50 is down {ix_ret:.1f}% over {lookback} sessions but more stocks rose than fell "
                      f"(A/D line {ad_chg:+.0f}): selling concentrated in the large index stocks.")
    if pd.notna(a20_chg) and ix_ret > 1 and a20_chg < -10:
        alerts.append(f"NIFTY 50 is up {ix_ret:.1f}% but the share of stocks above their 20-day average fell "
                      f"{-a20_chg:.0f} points: participation is thinning.")
    if pd.notna(a20_chg) and ix_ret < -1 and a20_chg > 10:
        alerts.append(f"NIFTY 50 is down {ix_ret:.1f}% but the share of stocks above their 20-day average rose "
                      f"{a20_chg:.0f} points: participation is improving under the surface.")
    return alerts
