"""Download or top up daily NSE OHLCV for every symbol in data/symbols.txt.

    python -m capstone.fetch_data --start 2015-01-01

Re-running only fetches dates after the last row already saved (plus the last week again, so
NSE's official figures replace any earlier estimate). Sources, best first:

1. NSE's official bhavcopy via NSE's public MCP server (mcp.nseindia.in, no key). It keeps about
   five years and has a few gaps (e.g. most of 2023), so it is merged on top of the others:
   for any date NSE has, NSE's row wins.
2. TejHQ (optional TEJ_API_KEY), for older history and NSE's gaps.
3. Yahoo Finance daily prices (SYMBOL.NS, no key) when both of the above fail.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from . import nse_mcp, tej_client
from .paths import RAW, load_symbols

COLUMNS = ["date", "open", "high", "low", "close", "prev_close", "volume", "turnover", "trades"]


def nse_daily(symbol: str, start: str) -> pd.DataFrame:
    """Official NSE bhavcopy rows from `start` (empty if the NSE server can't be reached)."""
    try:
        rows = nse_mcp.stock_history(symbol, start)
    except nse_mcp.NseMcpError:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
        # Today's row is not final until NSE publishes the evening bhavcopy.
        now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
        if now.hour * 60 + now.minute < 15 * 60 + 45:
            df = df[df["date"] < now.normalize()]
    return df


def other_daily(symbol: str, start: str) -> pd.DataFrame:
    """TejHQ rows from `start`, topped up from Yahoo; or Yahoo alone if TejHQ fails."""
    try:
        new = pd.DataFrame(tej_client.ohlcv(symbol, start=start, end=date.today().isoformat()))
    except tej_client.TejError:
        new = pd.DataFrame()
    if len(new):
        new["date"] = pd.to_datetime(new["date"])
        # TejHQ can publish a day late: fill any newer finished days from Yahoo.
        after = (new["date"].max().date() + timedelta(days=1)).isoformat()
        extra = yahoo_daily(symbol, after) if after <= date.today().isoformat() else pd.DataFrame()
        return pd.concat([new, extra], ignore_index=True) if len(extra) else new
    return yahoo_daily(symbol, start)


def update_symbol(symbol: str, start: str) -> tuple[int, str]:
    """Top up data/raw/SYMBOL.csv. Returns (new rows, which sources were used)."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{symbol}.csv"
    old = pd.read_csv(path, parse_dates=["date"]) if path.exists() else None
    if old is not None and len(old):
        start = (old["date"].max().date() + timedelta(days=1)).isoformat()
    if start > date.today().isoformat():
        return 0, ""
    # Re-read the last week from NSE too, so its official figures replace any earlier estimate.
    nse_start = (date.fromisoformat(start) - timedelta(days=7)).isoformat() if old is not None else start
    nse = nse_daily(symbol, nse_start)
    first_needed = pd.Timestamp(start)
    covered_from = nse["date"].min() if len(nse) else None
    # Other sources are needed before NSE's history starts, or when NSE has nothing new.
    need_other = (len(nse) == 0 or covered_from > first_needed + pd.Timedelta(days=7)
                  or nse["date"].max() < pd.Timestamp(date.today()) - pd.Timedelta(days=4)
                  or _has_gap(nse))
    other = other_daily(symbol, start) if need_other else pd.DataFrame()
    used = [name for name, df in (("NSE bhavcopy", nse), ("TejHQ/Yahoo", other)) if len(df)]
    new = pd.concat([d for d in (other, nse) if len(d)], ignore_index=True) if used else pd.DataFrame()
    if new.empty:
        return 0, ""
    new = new[[c for c in COLUMNS if c in new.columns]]
    new["date"] = pd.to_datetime(new["date"])
    before = 0 if old is None else len(old)
    df = new if old is None else pd.concat([old, new])
    df = df.drop_duplicates("date", keep="last").sort_values("date")  # NSE rows come last, so they win
    df.to_csv(path, index=False)
    return len(df) - before, " + ".join(used)


def _has_gap(df: pd.DataFrame, days: int = 10) -> bool:
    """True if the NSE rows skip more than `days` calendar days (the server misses a few months)."""
    return len(df) > 1 and bool((df["date"].sort_values().diff().dt.days > days).any())


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
            n, used = update_symbol(symbol, args.start)
            print(f"{symbol}: +{n} rows" + (f" ({used})" if used else ""))
        except tej_client.TejError as exc:
            print(f"{symbol}: skipped ({exc})")


if __name__ == "__main__":
    main()
