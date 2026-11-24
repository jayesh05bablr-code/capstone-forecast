"""Risk and performance measures from daily prices (NSE bhavcopy), for the stock panel and risk pages.

All are standard definitions computed only when there is enough data; otherwise None, never a guess.

* ATR %          : 14-session average true range / last close
* volatility     : stdev of daily log returns over 20 sessions x sqrt(252), in %
* beta           : cov(stock, NIFTY 50) / var(NIFTY 50) on daily returns, aligned by date (>= 40 days)
* max drawdown   : worst fall from a running peak of the close, in %
* returns        : close-to-close over 5, 21 and 60 sessions
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd

MIN_DAYS = {"atr": 15, "vol": 21, "beta": 40, "dd": 20}

_index_cache: dict = {"at": 0.0, "df": None}
_index_lock = threading.Lock()


def nifty_history(max_age: float = 6 * 3600) -> pd.DataFrame:
    """NIFTY 50 daily closes for the last ~12 months (date, close), shared and refreshed a few times a day."""
    with _index_lock:
        if _index_cache["df"] is not None and time.time() - _index_cache["at"] < max_age:
            return _index_cache["df"]
        try:
            from capstone import nse_mcp
            r = nse_mcp.call("bhavcopy", "get_index_history", {"indexName": "NIFTY 50", "months": 12})
            df = pd.DataFrame(r.get("data") or [])[["date", "open", "high", "low", "close"]]
            df["date"] = pd.to_datetime(df["date"])
            _index_cache.update(at=time.time(), df=df.sort_values("date").reset_index(drop=True))
        except Exception:
            if _index_cache["df"] is None:
                return pd.DataFrame(columns=["date", "close"])
        return _index_cache["df"]


def atr_pct(df: pd.DataFrame, n: int = 14) -> float | None:
    d = df.sort_values("date")
    if len(d) < MIN_DAYS["atr"]:
        return None
    prev = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
    return float(100 * tr.tail(n).mean() / d["close"].iloc[-1])


def volatility(df: pd.DataFrame, n: int = 20) -> float | None:
    d = df.sort_values("date")
    if len(d) < MIN_DAYS["vol"]:
        return None
    r = np.log(d["close"] / d["close"].shift(1)).dropna().tail(n)
    return float(100 * r.std() * np.sqrt(252))


def beta(df: pd.DataFrame, index: pd.DataFrame, n: int = 120) -> float | None:
    if index is None or index.empty:
        return None
    a = df[["date", "close"]].merge(index[["date", "close"]], on="date", suffixes=("", "_ix")).sort_values("date")
    r = a[["close", "close_ix"]].pct_change().dropna().tail(n)
    if len(r) < MIN_DAYS["beta"] or r["close_ix"].var() == 0:
        return None
    return float(r["close"].cov(r["close_ix"]) / r["close_ix"].var())


def max_drawdown(df: pd.DataFrame) -> float | None:
    d = df.sort_values("date")
    if len(d) < MIN_DAYS["dd"]:
        return None
    c = d["close"]
    return float(100 * (c / c.cummax() - 1).min())


def ret(df: pd.DataFrame, sessions: int) -> float | None:
    d = df.sort_values("date")
    if len(d) <= sessions:
        return None
    return float(100 * (d["close"].iloc[-1] / d["close"].iloc[-1 - sessions] - 1))


def summary(df: pd.DataFrame, index: pd.DataFrame | None = None) -> dict:
    """Every measure above for one stock's daily rows (date, open, high, low, close, volume, turnover)."""
    return {"atr_pct": atr_pct(df), "vol_20d": volatility(df), "beta": beta(df, index) if index is not None else None,
            "max_dd": max_drawdown(df), "ret_1w": ret(df, 5), "ret_1m": ret(df, 21), "ret_3m": ret(df, 60),
            "avg_value_cr": float(df.sort_values("date")["turnover"].tail(20).mean() / 1e7)
            if "turnover" in df and len(df) >= 5 else None,
            "sessions": int(len(df)), "from": df["date"].min(), "to": df["date"].max()}


def intraday_vwap(bars: pd.DataFrame) -> pd.Series:
    """Running VWAP of one session's 1-minute bars: cumulative (typical price x volume) / cumulative volume."""
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    vol = bars["volume"].fillna(0)
    return (tp * vol).cumsum() / vol.cumsum().where(vol.cumsum() > 0)
