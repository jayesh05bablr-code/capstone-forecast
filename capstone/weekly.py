"""Weekly trading analysis: buy at the week's first session open, sell at its last session close.

    Weekly delta = (exit close - entry open) / entry open

Trading calendar
    The NSE calendar is taken from the price history itself: a date is a trading session if NSE
    published a bar for it. This needs no hand-typed holiday list and automatically includes
    holidays and special sessions (e.g. a Saturday budget-day or Muhurat session).

Week definition and holiday handling
    Weeks run Monday to Sunday (calendar weeks).
    - Entry session = the first trading session of the week (Tuesday when Monday is a holiday).
    - Exit session  = the last trading session of the same week (Thursday when Friday is a holiday).
    - Holiday-shortened weeks are kept; `sessions` records how many sessions the week had.
    - Excluded weeks (with `exclude_reason`):
        * missing or non-positive entry open / exit close;
        * a corporate action (bonus/split) inside the holding period: an ex-date listed in
          data/corporate_actions.csv, or any overnight gap larger than CORPORATE_ACTION_GAP between
          two sessions of the week. Prices are unadjusted, so such a week would show a fake loss
          (-50% for a 1:1 bonus). An ex-date on the entry session itself does not distort the
          week, because the entry open is already ex-bonus;
        * the week still in progress at the end of the data (its exit is not known yet).
    The entry price is the open of the entry session, so it never uses information from later in
    the week; any signal for the week must be computed from data before that open (weekly_model.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import costs as cost_model
from .paths import DATA, RAW, load_symbols

CORPORATE_ACTIONS_FILE = DATA / "corporate_actions.csv"

CORPORATE_ACTION_GAP = 0.20


def trading_calendar(prices: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """All dates on which any stock in the universe traded (the observed NSE calendar)."""
    dates = pd.DatetimeIndex([])
    for df in prices.values():
        dates = dates.union(pd.DatetimeIndex(pd.to_datetime(df["date"])))
    return dates.sort_values()


def corporate_action_dates(symbol: str | None = None) -> set[pd.Timestamp]:
    """Known ex-dates for a stock from data/corporate_actions.csv (empty set if none)."""
    if symbol is None or not CORPORATE_ACTIONS_FILE.exists():
        return set()
    ca = pd.read_csv(CORPORATE_ACTIONS_FILE, parse_dates=["ex_date"])
    return set(ca.loc[ca["symbol"].str.upper() == symbol.upper(), "ex_date"])


def weekly_returns(prices: pd.DataFrame, data_end: pd.Timestamp | None = None,
                   ex_dates: set[pd.Timestamp] | None = None) -> pd.DataFrame:
    """One row per calendar week for one stock's daily OHLC (date, open, close required).

    data_end: last date the data could contain (defaults to the last price date). The final week
    is marked in progress when data_end falls before that week's Friday.
    ex_dates: known corporate-action ex-dates for this stock.
    """
    ex_dates = ex_dates or set()
    p = prices.sort_values("date").copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.drop_duplicates("date", keep="last")
    p["overnight_gap"] = p["open"] / p["close"].shift(1) - 1
    p["week"] = p["date"].dt.to_period("W-SUN")
    data_end = pd.Timestamp(data_end) if data_end is not None else p["date"].max()

    rows = []
    for week, g in p.groupby("week", sort=True):
        first, last = g.iloc[0], g.iloc[-1]
        in_week_gaps = g["overnight_gap"].iloc[1:]  # gaps between sessions inside the holding period
        reason = ""
        if not (np.isfinite(first["open"]) and first["open"] > 0 and np.isfinite(last["close"]) and last["close"] > 0):
            reason = "missing entry or exit price"
        elif (in_week_gaps.abs() > CORPORATE_ACTION_GAP).any() or any(d in ex_dates for d in g["date"].iloc[1:]):
            reason = "corporate action (split/bonus) inside the week"
        elif week.end_time.normalize() - pd.Timedelta(days=2) > data_end and last["date"] == data_end:
            reason = "week still in progress"
        rows.append({
            "week": str(week), "week_start": week.start_time.normalize(),
            "entry_date": first["date"], "exit_date": last["date"], "sessions": len(g),
            "entry_open": float(first["open"]), "exit_close": float(last["close"]),
            "weekly_return": float(last["close"] / first["open"] - 1) if not reason.startswith("missing") else np.nan,
            "holiday_shortened": len(g) < 5, "valid": reason == "", "exclude_reason": reason,
        })
    return pd.DataFrame(rows)


def load_prices(symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols or load_symbols():
        path = RAW / f"{s}.csv"
        if path.exists():
            out[s] = pd.read_csv(path, parse_dates=["date"])
    return out


def all_weeks(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """weekly_returns for every stock, stacked with a `symbol` column."""
    end = max(df["date"].max() for df in prices.values()) if prices else None
    frames = [weekly_returns(df, end, corporate_action_dates(s)).assign(symbol=s) for s, df in prices.items()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def tradable_stats(weeks: pd.DataFrame, cost_pct: float) -> pd.DataFrame:
    """Per-stock summary of valid weeks against a round-trip cost (fraction of position).

    Tradable Weeks (%) = valid weeks whose realised return > round-trip cost / valid weeks x 100.
    """
    rows = []
    for sym, g in weeks[weeks["valid"]].groupby("symbol"):
        r = g["weekly_return"]
        tradable = int((r > cost_pct).sum())
        best, worst = g.loc[r.idxmax()], g.loc[r.idxmin()]
        rows.append({"symbol": sym, "valid_weeks": len(g), "tradable_weeks": tradable,
                     "tradable_pct": tradable / len(g) * 100 if len(g) else np.nan,
                     "avg_weekly_return_pct": r.mean() * 100, "median_weekly_return_pct": r.median() * 100,
                     "best_week": best["week"], "best_week_return_pct": best["weekly_return"] * 100,
                     "worst_week": worst["week"], "worst_week_return_pct": worst["weekly_return"] * 100,
                     "round_trip_cost_pct": cost_pct * 100,
                     "excluded_weeks": int((weeks[weeks["symbol"] == sym]["valid"] == False).sum())})  # noqa: E712
    return pd.DataFrame(rows)


def cost_for_position(capital: float, allocation: float, assumptions: cost_model.CostAssumptions) -> float:
    """Round-trip cost (fraction) for one position of capital x allocation rupees."""
    return cost_model.round_trip_pct(capital * allocation, assumptions)
