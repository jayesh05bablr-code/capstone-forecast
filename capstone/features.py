"""Turn raw OHLCV + fundamentals into one model-ready table.

Every price feature for day t uses only data known at day t's close; the
overnight features (overnight.py) add what the US market and the rupee did
before the next open. The target is the overnight gap: next day's open
divided by today's close, minus one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fundamentals, overnight
from .indicators import adjust_for_actions, corporate_actions
from .paths import PROCESSED, RAW, load_symbols

# Overnight moves beyond this are almost always splits, bonuses or data errors.
MAX_ABS_GAP = 0.20
# Consecutive rows further apart than this (calendar days) are a hole in the data, not a weekend
# or a holiday: NSE's longest closures are 4-5 days. A return measured across a hole is not an
# overnight move, so neither the target nor the features of that row are used.
MAX_SESSION_GAP_DAYS = 7
# Bump when the feature table changes meaning, so a saved model built from the old table is retrained
# once (refresh.stale() compares it like an input file).
VERSION = "2026-11-23 corporate-action adjusted, data holes removed"

PRICE_FEATURES = [
    "ret_1", "ret_5", "ret_20", "intraday_ret", "gap_today", "range_pct",
    "close_pos", "vol_ratio_20", "volatility_20", "dist_sma_20", "dow",
    "mkt_ret_1", "mkt_gap_today",
]
FUNDAMENTAL_FEATURES = ["log_mcap", "pe", "pb", "eps_ttm", "roe_pct", "debt_to_equity",
                        "promoter_pct", "fii_pct", "dii_pct"]
OVERNIGHT_FEATURES = overnight.OVERNIGHT_FEATURES
FEATURES = PRICE_FEATURES + OVERNIGHT_FEATURES + FUNDAMENTAL_FEATURES


def load_prices(symbols: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for symbol in symbols or load_symbols():
        path = RAW / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            # Bonus issues and splits halve the price overnight. Without this, 5- and 20-day returns,
            # the moving average and volatility look like a crash for weeks after each ex-date,
            # and a bonus below the 20% filter (ICICIBANK 1:10) is learnt as a real -9% gap.
            df = adjust_for_actions(df, corporate_actions(symbol))
            df["symbol"] = symbol
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No price files in {RAW}. Run python -m capstone.fetch_data first.")
    return pd.concat(frames, ignore_index=True)


def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    prev_close = g["close"].shift(1)
    g["ret_1"] = g["close"] / prev_close - 1
    g["ret_5"] = g["close"] / g["close"].shift(5) - 1
    g["ret_20"] = g["close"] / g["close"].shift(20) - 1
    g["intraday_ret"] = g["close"] / g["open"] - 1
    g["gap_today"] = g["open"] / prev_close - 1
    g["range_pct"] = (g["high"] - g["low"]) / g["close"]
    span = (g["high"] - g["low"]).replace(0, np.nan)
    g["close_pos"] = (g["close"] - g["low"]) / span
    g["vol_ratio_20"] = g["volume"] / g["volume"].rolling(20).mean()
    g["volatility_20"] = g["ret_1"].rolling(20).std()
    g["dist_sma_20"] = g["close"] / g["close"].rolling(20).mean() - 1
    g["next_date"] = g["date"].shift(-1)
    g["next_open"] = g["open"].shift(-1)
    g["target_gap"] = g["next_open"] / g["close"] - 1
    # Rows next to a hole in the data (e.g. NSE's bhavcopy server misses most of 2023 when no
    # other source fills it) would measure months of movement as one night.
    hole_after = (g["next_date"] - g["date"]).dt.days > MAX_SESSION_GAP_DAYS
    g.loc[hole_after, ["target_gap", "next_open"]] = np.nan
    days_since = (g["date"] - g["date"].shift(1)).dt.days
    hole_within = days_since.gt(MAX_SESSION_GAP_DAYS).astype(int).rolling(21, min_periods=1).max().astype(bool)
    g.loc[hole_within, ["ret_1", "ret_5", "ret_20", "gap_today", "vol_ratio_20", "volatility_20",
                        "dist_sma_20"]] = np.nan
    return g


def build(symbols: list[str] | None = None, save: bool = True) -> pd.DataFrame:
    prices = load_prices(symbols)
    df = pd.concat([_per_symbol(g) for _, g in prices.groupby("symbol")], ignore_index=True)
    df["dow"] = df["date"].dt.dayofweek
    # Market-wide mood: the average move across all symbols that day.
    daily = df.groupby("date")[["ret_1", "gap_today"]].mean()
    daily.columns = ["mkt_ret_1", "mkt_gap_today"]
    df = df.merge(daily, left_on="date", right_index=True, how="left")
    df = overnight.attach(df)
    df = fundamentals.attach(df, fundamentals.load())
    df["log_mcap"] = np.log(df["market_cap_cr"])
    # Drop corporate-action days so splits do not look like crashes.
    bad = (df["target_gap"].abs() > MAX_ABS_GAP) | (df["gap_today"].abs() > MAX_ABS_GAP)
    df = df[~bad].replace([np.inf, -np.inf], np.nan)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED / "features.csv", index=False)
    return df
