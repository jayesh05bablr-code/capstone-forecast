"""Live NSE prices from Yahoo Finance, a buyer/seller volume estimate, and live trade status.

Data: 1-minute candles for today from Yahoo Finance (the yfinance library, with Yahoo's chart
endpoint as a fallback). No key is needed. Yahoo's NSE prices can lag the exchange by a minute
or more, so treat them as near-real-time, not tick-by-tick.

Buyers vs sellers: every trade has one buyer and one seller, so "buy volume" means volume that
traded on an uptick (buyers were paying up) and "sell volume" volume on a downtick. Yahoo does not
publish the order book (pending buy/sell quantities you see in Kite's market depth) or individual
trades, so this module estimates it from 1-minute candles with the tick rule:
- a candle that closes above the previous close counts as buying,
- one that closes below counts as selling,
- an unchanged candle is split by where it closed inside its high-low range.
"""
from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd

from . import intraday, scalping

IST = "Asia/Kolkata"
OPEN, CLOSE = time(9, 15), time(15, 30)


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST).tz_localize(None)


def market_open(at: datetime | None = None) -> bool:
    """True between 09:15 and 15:30 IST on an NSE trading day (weekends and NSE holidays excluded)."""
    from . import market_calendar

    return market_calendar.market_open(at or now_ist())


def _from_yfinance(symbols: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    tickers = [f"{s}.NS" for s in symbols]
    raw = yf.download(tickers, period="2d", interval="1m", group_by="ticker", auto_adjust=False,
                      progress=False, threads=True, prepost=False)
    out = {}
    for sym, tk in zip(symbols, tickers):
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df.rename(columns=str.lower).dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            continue
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_convert(IST).tz_localize(None) if idx.tz is not None else idx
        out[sym] = pd.DataFrame({"datetime": idx, **{c: df[c].to_numpy() for c in intraday.COLS[1:]}})
    return out


def minute_bars(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], str]:
    """Recent 1-minute candles per symbol (last few sessions), and the source used."""
    bars, source = {}, "Yahoo Finance (yfinance)"
    try:
        bars = _from_yfinance(symbols)
    except Exception:  # library missing, Yahoo changed something, or the network is down
        bars = {}
    missing = [s for s in symbols if s not in bars]
    for sym in missing:  # fallback: Yahoo's chart endpoint directly
        try:
            df = intraday.fetch_yahoo(sym, days=5)
            if len(df):
                bars[sym] = df
                source = "Yahoo Finance"
        except Exception:
            pass
    return bars, source


def session(bars: pd.DataFrame, day=None) -> pd.DataFrame:
    """Candles of one trading day (the latest one in the data if day is None), market hours only."""
    if bars is None or bars.empty:
        return pd.DataFrame(columns=intraday.COLS)
    d = bars["datetime"].dt.date
    day = day or d.max()
    g = bars[d == day]
    t = g["datetime"].dt.hour * 60 + g["datetime"].dt.minute
    return g[(t >= 9 * 60 + 15) & (t < 15 * 60 + 30)].reset_index(drop=True)


def buy_sell_split(bars: pd.DataFrame, prev_close: float | None = None) -> pd.DataFrame:
    """Add buy_vol and sell_vol columns to 1-minute candles with the tick rule (see module doc)."""
    df = bars.copy()
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    prev = df["close"].shift(1)
    if prev_close is not None and len(df):
        prev.iloc[0] = prev_close
    else:
        prev.iloc[:1] = df["open"].iloc[:1]
    span = (df["high"] - df["low"]).replace(0, np.nan)
    in_range = ((df["close"] - df["low"]) / span).fillna(0.5)
    share = np.where(df["close"] > prev, 1.0, np.where(df["close"] < prev, 0.0, in_range))
    df["buy_vol"] = vol * share
    df["sell_vol"] = vol - df["buy_vol"]
    return df


def volume_summary(bars: pd.DataFrame, prev_close: float | None = None, last_minutes: int = 15) -> dict:
    """Buy vs sell volume for the day and for the last few minutes."""
    if bars is None or bars.empty:
        return {}
    df = buy_sell_split(bars, prev_close)
    recent = df.tail(last_minutes)

    def pct(part, whole):
        return round(100 * part / whole, 1) if whole else None
    day_b, day_s = df["buy_vol"].sum(), df["sell_vol"].sum()
    rec_b, rec_s = recent["buy_vol"].sum(), recent["sell_vol"].sum()
    return {"volume": int(day_b + day_s), "buy_vol": int(day_b), "sell_vol": int(day_s),
            "buy_pct": pct(day_b, day_b + day_s), "sell_pct": pct(day_s, day_b + day_s),
            "recent_buy_pct": pct(rec_b, rec_b + rec_s), "recent_minutes": len(recent)}


def pressure_word(buy_pct: float | None) -> str:
    if buy_pct is None:
        return "No volume yet"
    if buy_pct >= 60:
        return "Buyers in control"
    if buy_pct >= 53:
        return "Buyers slightly ahead"
    if buy_pct > 47:
        return "Balanced"
    if buy_pct > 40:
        return "Sellers slightly ahead"
    return "Sellers in control"


def trade_status(plan: dict, today: pd.DataFrame, or_minutes: int, window: int) -> tuple[str, str]:
    """Where a planned breakout trade stands right now, from today's candles after the range.

    Returns (status, suggestion in plain words). Same rule as scalping.backtest: the stop counts
    first when one candle touches both levels.
    """
    if today.empty:
        return "Waiting", "No candles today yet."
    start = pd.Timestamp(today["datetime"].dt.date.iloc[0]) + pd.Timedelta(hours=9, minutes=15)
    range_end, stop_time = start + pd.Timedelta(minutes=or_minutes), start + pd.Timedelta(minutes=window)
    after = today[(today["datetime"] >= range_end) & (today["datetime"] < stop_time)]
    last_time = today["datetime"].max()
    long = plan["side"] == "Long"
    if last_time < range_end - pd.Timedelta(minutes=1):
        return "Range forming", f"Wait: the opening range is set at {range_end:%H:%M}."
    entered = None
    for bar in after.itertuples():
        if entered is None:
            if (long and bar.high >= plan["trigger"]) or (not long and bar.low <= plan["trigger"]):
                entered = bar.datetime
            else:
                continue
        hit_stop = bar.low <= plan["stop"] if long else bar.high >= plan["stop"]
        hit_target = bar.high >= plan["target"] if long else bar.low <= plan["target"]
        if hit_stop:
            return "Stopped out", f"Entered {entered:%H:%M}, stop hit at {bar.datetime:%H:%M}. No new trade today."
        if hit_target:
            return "Target hit", f"Entered {entered:%H:%M}, target hit at {bar.datetime:%H:%M}. Done for today."
    over = last_time >= stop_time - pd.Timedelta(minutes=1)
    if entered is not None:
        if over:
            return "Window over", f"Entered {entered:%H:%M}; window ended at {stop_time:%H:%M}: exit at market."
        return "In trade", (f"Entered {entered:%H:%M}. Keep the stop at ₹{plan['stop']:,.2f}, "
                            f"target ₹{plan['target']:,.2f}, exit by {stop_time:%H:%M}.")
    if over:
        return "No trigger", f"Price never crossed ₹{plan['trigger']:,.2f} before {stop_time:%H:%M}. Skip."
    word = "above" if long else "below"
    return "Armed", f"{'Buy' if long else 'Short'} if price goes {word} ₹{plan['trigger']:,.2f} before {stop_time:%H:%M}."


def _forecast_right(open_: float, prev_close: float, pred_gap) -> str:
    if abs(open_ / prev_close - 1) < 0.0005:
        return "Flat"
    return "Yes" if (open_ - prev_close) * pred_gap > 0 else "No"


def nse_quotes(symbols: list[str]) -> dict[str, dict]:
    """Official near-real-time NSE quotes (1-3 minute delay) from NSE's public MCP server.

    Returns {} if mcp.nseindia.in can't be reached, so the Live tab falls back to Yahoo alone.
    """
    from . import nse_mcp

    try:
        return nse_mcp.quotes(symbols)
    except Exception:
        return {}


def _same_day(as_of, target) -> bool:
    """True when a quote's time falls on the forecast's target day (or the target is unknown)."""
    if target is None or (isinstance(target, float) and np.isnan(target)) or as_of is None:
        return True
    return pd.Timestamp(as_of).date() == pd.Timestamp(target).date()


def quote_table(preds: pd.DataFrame, quotes: dict[str, dict]) -> pd.DataFrame:
    """One row per stock from the official NSE quotes, with the morning forecast beside it."""
    rows = []
    for r in preds.itertuples():
        q = quotes.get(r.symbol)
        if not q or not q.get("ltp"):
            continue
        open_, prev = q.get("open") or 0, q.get("prev_close") or 0
        if not _same_day(q.get("as_of"), getattr(r, "target_date", None)):
            open_ = 0  # an older session's open: the forecast is for a later day, nothing to check yet
        rows.append({"Stock": r.symbol, "LTP (₹)": q["ltp"], "Change %": q.get("pct_change"),
                     "Open (₹)": q.get("open") or None, "High (₹)": q.get("high"), "Low (₹)": q.get("low"),
                     "Prev close (₹)": prev or None, "Volume": q.get("volume"),
                     "Forecast open (₹)": round(r.pred_open, 2),
                     "Forecast direction right?": _forecast_right(open_, prev, r.pred_gap_pct)
                     if open_ and prev else "",
                     "As of (IST)": q["as_of"].strftime("%d %b %H:%M") if q.get("as_of") else ""})
    return pd.DataFrame(rows)


def live_rows(preds: pd.DataFrame, bars: dict[str, pd.DataFrame], cfg,
              quotes: dict[str, dict] | None = None, now: datetime | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """One row per stock: LTP, move, buyer/seller split, forecast check and live setup status.

    cfg is a tab_scalping.Settings. `quotes` are official NSE quotes (nse_quotes); when a stock has
    one, its LTP, open and previous close come from NSE and the minute candles only feed the
    buyer/seller split and the setups. Returns the table and the plan rows (for the Kite fields).
    """
    rows, plans = [], []
    quotes = quotes or {}
    for r in preds.itertuples():
        b = bars.get(r.symbol)
        today = session(b)
        q = quotes.get(r.symbol) or {}
        if today.empty:
            if q.get("ltp"):
                rows.append({"Stock": r.symbol, "LTP (₹)": q["ltp"], "Change %": q.get("pct_change"),
                             "As of (IST)": q["as_of"].strftime("%d %b %H:%M") if q.get("as_of") else "",
                             "Source": "NSE", "Status": "No minute candles",
                             "Suggestion": "NSE price shown; setups need Yahoo's 1-minute candles."})
            else:
                rows.append({"Stock": r.symbol, "LTP (₹)": None, "Status": "No live data",
                             "Suggestion": "Neither NSE nor Yahoo returned a price for this stock."})
            continue
        day = today["datetime"].dt.date.iloc[0]
        older = b[b["datetime"].dt.date < day]
        prev_close = float(older["close"].iloc[-1]) if len(older) else float(r.close)
        ltp = float(today["close"].iloc[-1])
        open_ = float(today["open"].iloc[0])
        as_of = today["datetime"].max()
        source = "Yahoo"
        q_day = q["as_of"].date() if q.get("as_of") else None
        if q.get("ltp") and q_day == day:  # same session: prefer NSE's own figures
            ltp, source = float(q["ltp"]), "NSE"
            prev_close = float(q.get("prev_close") or prev_close)
            open_ = float(q.get("open") or open_)
            as_of = max(as_of, pd.Timestamp(q["as_of"]))
        vs = volume_summary(today, prev_close)
        check = _same_day(day, getattr(r, "target_date", None))
        row = {"Stock": r.symbol, "LTP (₹)": round(ltp, 2), "Change %": round((ltp / prev_close - 1) * 100, 2),
               "As of (IST)": as_of.strftime("%d %b %H:%M"), "Source": source,
               "Forecast open (₹)": round(r.pred_open, 2), "Actual open (₹)": round(open_, 2),
               "Forecast direction right?": _forecast_right(open_, prev_close, r.pred_gap_pct) if check else "",
               "Buy volume %": vs.get("buy_pct"), "Sell volume %": vs.get("sell_pct"),
               f"Buy % last {vs.get('recent_minutes', 0)} min": vs.get("recent_buy_pct"),
               "Pressure": pressure_word(vs.get("recent_buy_pct"))}
        if day < (now or now_ist()).date():
            rows.append({**row, "Status": f"Session of {day:%a %d %b} is over",
                         "Suggestion": "Market closed now. Setups come back on the next trading day."})
            continue
        orr = intraday.opening_range(today, day, cfg.or_min)
        if orr is None:
            rows.append({**row, "Status": "Range forming",
                         "Suggestion": f"Wait for the first {cfg.or_min} minutes after 09:15."})
            continue
        direction = {"Model's expected gap": r.pred_gap_pct, "Actual opening gap": open_ - prev_close,
                     "Both directions": 0}[cfg.bias_src]
        setups = scalping.plan_row(r.symbol, orr["open"], orr["high"], orr["low"], direction, cfg.capital,
                                   cfg.risk, cfg.reward, cfg.leverage, cfg.buffer_pct, cfg.brokerage)
        if not setups:
            rows.append({**row, "Status": "No setup", "Suggestion": "Range too narrow or no direction."})
            continue
        for p in setups.values():
            status, says = trade_status(p, today, cfg.or_min, cfg.window)
            rows.append({**row, "Side": p["side"], "Enter at (₹)": round(p["trigger"], 2),
                         "Stop (₹)": round(p["stop"], 2), "Target (₹)": round(p["target"], 2),
                         "Qty": p["qty"], "Status": status, "Suggestion": says})
            plans.append({**p, "ltp": ltp, "status": status})
    return pd.DataFrame(rows), plans
