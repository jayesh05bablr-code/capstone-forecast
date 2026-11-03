"""Download or top up daily NSE OHLCV for every symbol in data/symbols.txt.

    python -m capstone.fetch_data --start 2015-01-01

Re-running only fetches dates after the last row already saved. Source: TejHQ; if TejHQ
fails or has nothing new, Yahoo Finance daily prices (SYMBOL.NS, no key) are used instead.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from . import tej_client
from .paths import RAW, load_symbols

COLUMNS = ["date", "open", "high", "low", "close", "prev_close", "volume", "turnover", "trades"]


def update_symbol(symbol: str, start: str) -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{symbol}.csv"
    old = pd.read_csv(path, parse_dates=["date"]) if path.exists() else None
    if old is not None and len(old):
        start = (old["date"].max().date() + timedelta(days=1)).isoformat()
    if start > date.today().isoformat():
        return 0
    try:
        rows = tej_client.ohlcv(symbol, start=start, end=date.today().isoformat())
    except tej_client.TejError:
        rows = []
    new = pd.DataFrame(rows) if rows else yahoo_daily(symbol, start)
    if new.empty:
        return 0
    new = new[[c for c in COLUMNS if c in new.columns]]
    new["date"] = pd.to_datetime(new["date"])
    df = new if old is None else pd.concat([old, new])
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    df.to_csv(path, index=False)
    return len(new)


def yahoo_daily(symbol: str, start: str) -> pd.DataFrame:
    """Daily NSE prices from Yahoo Finance (unadjusted, like the exchange's own figures)."""
    try:
        import yfinance as yf

        h = yf.Ticker(f"{symbol}.NS").history(start=start, interval="1d", auto_adjust=False, actions=False)
    except Exception:
        return pd.DataFrame()
    if h is None or h.empty:
        return pd.DataFrame()
    h = h.rename(columns=str.lower)
    idx = pd.DatetimeIndex(h.index)
    out = pd.DataFrame({"date": (idx.tz_localize(None) if idx.tz is not None else idx).normalize(),
                        **{c: h[c].to_numpy() for c in ("open", "high", "low", "close", "volume")}})
    out = out.dropna(subset=["open", "high", "low", "close"])
    # Today's row is a part-day bar while the market is open; keep only finished days.
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    if now.hour * 60 + now.minute < 15 * 60 + 45:
        out = out[out["date"] < now.normalize()]
    out["prev_close"] = out["close"].shift(1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01", help="first date for a new symbol")
    args = parser.parse_args()
    for symbol in load_symbols():
        try:
            n = update_symbol(symbol, args.start)
            print(f"{symbol}: +{n} rows")
        except tej_client.TejError as exc:
            print(f"{symbol}: skipped ({exc})")


if __name__ == "__main__":
    main()
