"""Technical indicators and price adjustments for the price charts.

Pure pandas functions (data in, series out), each using only past and current rows, so the
values on a chart are the ones a trader would have seen on that day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from capstone.paths import DATA

CORPORATE_ACTIONS = DATA / "corporate_actions.csv"
PERIODS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "All": None}


def corporate_actions(symbol: str) -> pd.DataFrame:
    """Known bonus/split ex-dates for one stock (ex_date, price_factor)."""
    if not CORPORATE_ACTIONS.exists():
        return pd.DataFrame(columns=["ex_date", "price_factor", "action"])
    ca = pd.read_csv(CORPORATE_ACTIONS, parse_dates=["ex_date"])
    return ca[ca["symbol"] == symbol][["ex_date", "price_factor", "action"]].sort_values("ex_date")


def adjust_for_actions(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust OHLC (and volume) before each bonus/split so the history is continuous.

    Prices before an ex-date are multiplied by its price factor (0.5 for a 1:1 bonus) and volume
    divided by it, the way charting platforms show adjusted history. Recent prices are unchanged.
    """
    out = prices.sort_values("date").copy()
    price_cols = [c for c in ("open", "high", "low", "close", "prev_close") if c in out]
    for _, a in actions.iterrows():
        before = out["date"] < a["ex_date"]
        out.loc[before, price_cols] = out.loc[before, price_cols] * a["price_factor"]
        if "volume" in out:
            out.loc[before, "volume"] = out.loc[before, "volume"] / a["price_factor"]
    if "prev_close" in out:  # the ex-date's previous close is the pre-bonus close: re-derive it
        out["prev_close"] = out["close"].shift(1).fillna(out["prev_close"])
    return out


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (0-100)."""
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi candles (smoothed trend view)."""
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = np.empty(len(df))
    if len(df):
        ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
        hc = ha_close.to_numpy()
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i - 1] + hc[i - 1]) / 2
    ha_open = pd.Series(ha_open, index=df.index)
    return df.assign(open=ha_open, close=ha_close,
                     high=pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1),
                     low=pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1))


def pivots(last: pd.Series) -> dict[str, float]:
    """Classic floor pivots for the next session from the last session's high, low and close."""
    p = (last["high"] + last["low"] + last["close"]) / 3
    return {"R2": p + (last["high"] - last["low"]), "R1": 2 * p - last["low"], "Pivot": p,
            "S1": 2 * p - last["high"], "S2": p - (last["high"] - last["low"])}


def window(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """The last N sessions for a period label (1M = 21 sessions ... All)."""
    n = PERIODS.get(period)
    return df if n is None else df.tail(n)


def snapshot(df: pd.DataFrame) -> dict:
    """Key numbers for the info strip above the price chart (df: adjusted daily OHLCV, sorted)."""
    last, c = df.iloc[-1], df["close"]
    prev = c.iloc[-2] if len(c) > 1 else np.nan
    year = df.tail(252)
    vol20 = df["volume"].tail(21).iloc[:-1].mean() if "volume" in df else np.nan
    out = {
        "date": last["date"], "close": last["close"], "change": last["close"] - prev,
        "change_pct": (last["close"] / prev - 1) * 100 if prev == prev else np.nan,
        "day_low": last["low"], "day_high": last["high"],
        "wk52_low": year["low"].min(), "wk52_high": year["high"].max(),
        "from_52w_high_pct": (last["close"] / year["high"].max() - 1) * 100,
        "volume": last.get("volume", np.nan),
        "volume_vs_avg": last.get("volume", np.nan) / vol20 if vol20 and vol20 == vol20 else np.nan,
        "rsi": rsi(c).iloc[-1],
        "atr_pct": atr(df).iloc[-1] / last["close"] * 100,
    }
    for n in (20, 50, 200):
        m = sma(c, n).iloc[-1]
        out[f"sma{n}"] = m
        out[f"above_sma{n}"] = bool(last["close"] > m) if m == m else None
    return out


def trend_label(snap: dict) -> str:
    a50, a200 = snap.get("above_sma50"), snap.get("above_sma200")
    if a50 is None or a200 is None:
        return "Not enough history"
    if a50 and a200:
        return "Uptrend (above 50 & 200-day averages)"
    if not a50 and not a200:
        return "Downtrend (below 50 & 200-day averages)"
    return "Mixed (between the 50 and 200-day averages)"
