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
    p["week_start"] = p["date"] - pd.to_timedelta(p["date"].dt.dayofweek, unit="D")  # Monday
    data_end = pd.Timestamp(data_end) if data_end is not None else p["date"].max()

    p["after_entry"] = p.groupby("week_start").cumcount() > 0  # sessions after the entry session
    p["ca_gap"] = p["after_entry"] & ((p["overnight_gap"].abs() > CORPORATE_ACTION_GAP) | p["date"].isin(list(ex_dates)))
    g = p.groupby("week_start", sort=True)
    w = pd.DataFrame({"entry_date": g["date"].first(), "exit_date": g["date"].last(), "sessions": g.size(),
                      "entry_open": g["open"].first().astype(float), "exit_close": g["close"].last().astype(float),
                      "corporate_action": g["ca_gap"].any()}).reset_index()
    # first()/last() skip NaN, so take the true first open and last close for the validity check
    w["entry_open"] = p.loc[g["date"].idxmin(), "open"].to_numpy(dtype=float)
    w["exit_close"] = p.loc[g["date"].idxmax(), "close"].to_numpy(dtype=float)
    bad_price = ~(np.isfinite(w["entry_open"]) & (w["entry_open"] > 0) & np.isfinite(w["exit_close"]) & (w["exit_close"] > 0))
    in_progress = (w["week_start"] + pd.Timedelta(days=4) > data_end) & (w["exit_date"] == data_end)
    w["exclude_reason"] = np.select([bad_price, w["corporate_action"], in_progress],
                                    ["missing entry or exit price", "corporate action (split/bonus) inside the week",
                                     "week still in progress"], "")
    w["weekly_return"] = np.where(bad_price, np.nan, w["exit_close"] / w["entry_open"] - 1)
    w["week"] = w["week_start"].dt.strftime("%Y-%m-%d") + "/" + (w["week_start"] + pd.Timedelta(days=6)).dt.strftime("%Y-%m-%d")
    w["holiday_shortened"] = w["sessions"] < 5
    w["valid"] = w["exclude_reason"] == ""
    return w[["week", "week_start", "entry_date", "exit_date", "sessions", "entry_open", "exit_close", "weekly_return",
              "holiday_shortened", "valid", "exclude_reason"]]


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
