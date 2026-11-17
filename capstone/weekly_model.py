"""Weekly return forecast: expected return from the week's first session open to its last close.

    python -m capstone.weekly_model

This is a separate model from the daily opening-gap model. The gap model forecasts one overnight
move; the weekly strategy holds for a whole week, so its signal must forecast the whole week.

Information set (everything known before the entry session's 09:15 IST open):
- the stock's daily bars dated before the entry session (closes back-adjusted for the bonus/split
  ex-dates in data/corporate_actions.csv, which changes no return inside an unaffected window);
- the equal-weight average of the same features across the universe (market mood);
- US sessions dated before the entry session (the US Friday session closes about 02:30 IST on
  Saturday, well before the Monday open), and USD/INR closes dated before the entry session;
- the number of sessions in the week (NSE publishes its holiday calendar in advance).

Target: weekly.weekly_return for valid weeks.

Walk-forward: the first MIN_TRAIN_WEEKS weeks are training only; then the models are refitted
every REFIT_WEEKS weeks on all weeks whose exit came before the block's first entry, and predict
the next block. Models: zero, each stock's historical mean weekly return, Linear Regression and
HistGradientBoosting (same settings as the daily model). The backtest uses HistGradientBoosting,
chosen before seeing results to match the daily model; the dashboard can switch to the others.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics, overnight, weekly
from .model_eval import make_linear_model
from .paths import MODELS

MIN_TRAIN_WEEKS = 156   # three years
REFIT_WEEKS = 13        # refit every quarter
WEEKLY_OOS_FILE = MODELS / "weekly_oos_predictions.csv"
WEEKLY_COMPARISON_FILE = MODELS / "weekly_model_comparison.csv"
SIGNAL_MODEL = "hist_gradient_boosting"
WEEKLY_LABELS = {"zero": "Zero (no edge)", "historical_average": "Historical average week",
                 "linear_regression": "Linear Regression", "hist_gradient_boosting": "HistGradientBoosting"}

STOCK_FEATURES = ["ret_1w", "ret_4w", "ret_13w", "vol_20d", "dist_sma_50", "last_day_ret", "prev_week_ret"]
MARKET_FEATURES = [f"mkt_{c}" for c in ("ret_1w", "ret_4w", "vol_20d")]
US_FEATURES = ["spy_last_ret", "spy_1w", "qqq_1w", "sector_1w", "usdinr_1w"]
CAL_FEATURES = ["sessions"]
FEATURES = STOCK_FEATURES + MARKET_FEATURES + US_FEATURES + CAL_FEATURES


def adjusted_close(prices: pd.DataFrame, symbol: str) -> pd.Series:
    """Daily closes indexed by date, back-adjusted for known bonus/split ex-dates."""
    p = prices.sort_values("date").set_index("date")["close"].astype(float)
    if weekly.CORPORATE_ACTIONS_FILE.exists():
        ca = pd.read_csv(weekly.CORPORATE_ACTIONS_FILE, parse_dates=["ex_date"])
        for r in ca[ca["symbol"].str.upper() == symbol.upper()].itertuples():
            p[p.index < r.ex_date] *= r.price_factor
    return p


def _before(series: pd.Series, when: pd.Series, lag: int) -> np.ndarray:
    """Value `lag` observations before the last observation dated strictly before each `when`."""
    idx = np.searchsorted(series.index.values, pd.to_datetime(when).values, side="left") - 1 - lag
    vals = series.to_numpy(float)
    return np.where(idx >= 0, vals[np.clip(idx, 0, None)], np.nan)


def _ret(series: pd.Series, when: pd.Series, n: int) -> np.ndarray:
    return _before(series, when, 0) / _before(series, when, n) - 1


def stock_features(prices: pd.DataFrame, symbol: str, weeks: pd.DataFrame) -> pd.DataFrame:
    """Features for each week of one stock, from data dated before its entry session."""
    close = adjusted_close(prices, symbol)
    daily_ret = close.pct_change()
    vol = daily_ret.rolling(20).std()
    sma50 = close.rolling(50).mean()
    e = weeks["entry_date"]
    out = weeks.copy()
    out["ret_1w"] = _ret(close, e, 5)
    out["ret_4w"] = _ret(close, e, 20)
    out["ret_13w"] = _ret(close, e, 65)
    out["vol_20d"] = _before(vol, e, 0)
    out["dist_sma_50"] = _before(close, e, 0) / _before(sma50, e, 0) - 1
    out["last_day_ret"] = _ret(close, e, 1)
    # Previous calendar week's open-to-close return (fully known before this week's entry).
    out["prev_week_ret"] = out["weekly_return"].where(out["valid"]).shift(1)
    return out


def us_features(df: pd.DataFrame, stock_map: pd.DataFrame) -> pd.DataFrame:
    e = df["entry_date"]
    spy, qqq, fx = overnight.load_close(overnight.MARKET_ETF), overnight.load_close(overnight.TECH_ETF), overnight.load_fx()
    df["spy_last_ret"] = _ret(spy, e, 1) if len(spy) else np.nan
    df["spy_1w"] = _ret(spy, e, 5) if len(spy) else np.nan
    df["qqq_1w"] = _ret(qqq, e, 5) if len(qqq) else np.nan
    df["usdinr_1w"] = _ret(fx, e, 5) if len(fx) else np.nan
    df["sector_1w"] = np.nan
    etf = df["symbol"].map(stock_map.set_index("symbol")["us_etf"])
    for t in etf.dropna().unique():
        s = overnight.load_close(t)
        rows = etf == t
        if len(s):
            df.loc[rows, "sector_1w"] = _ret(s, e[rows], 5)
    return df


def build_dataset(prices: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per (stock, week) with FEATURES and the realised weekly_return."""
    prices = prices or weekly.load_prices()
    weeks = weekly.all_weeks(prices)
    frames = [stock_features(prices[s], s, weeks[weeks["symbol"] == s].reset_index(drop=True)) for s in prices]
    df = pd.concat(frames, ignore_index=True)
    mkt = df.groupby("week")[["ret_1w", "ret_4w", "vol_20d"]].mean().add_prefix("mkt_")
    df = df.merge(mkt, left_on="week", right_index=True, how="left")
    try:
        smap = overnight.stock_map()
    except FileNotFoundError:
        smap = pd.DataFrame({"symbol": [], "us_etf": []})
    return us_features(df, smap).sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def walk_forward(df: pd.DataFrame, min_train: int = MIN_TRAIN_WEEKS, refit: int = REFIT_WEEKS) -> pd.DataFrame:
    """Out-of-sample weekly forecasts for valid weeks from every model."""
    from . import train

    data = df[df["valid"]].copy()
    week_starts = np.sort(data["week_start"].unique())
    if len(week_starts) <= min_train + refit:
        raise ValueError(f"Only {len(week_starts)} weeks; need more than {min_train + refit}.")
    feats = [f for f in FEATURES if f in data.columns]
    out = []
    for b in range(min_train, len(week_starts), refit):
        block = week_starts[b:b + refit]
        test = data[data["week_start"].isin(block)]
        first_entry = test["entry_date"].min()
        tr = data[data["exit_date"] < first_entry]  # only weeks already finished
        cols = [f for f in feats if tr[f].nunique(dropna=True) >= 2]
        hist_mean = tr.groupby("symbol")["weekly_return"].mean()
        preds = {
            "zero": np.zeros(len(test)),
            "historical_average": test["symbol"].map(hist_mean).fillna(tr["weekly_return"].mean()).to_numpy(),
            "linear_regression": make_linear_model().fit(tr[cols], tr["weekly_return"]).predict(test[cols]),
            "hist_gradient_boosting": train.make_model().fit(tr[cols], tr["weekly_return"]).predict(test[cols]),
        }
        base = test[["symbol", "week", "week_start", "entry_date", "exit_date", "sessions", "entry_open",
                     "exit_close", "weekly_return"]].assign(train_weeks=tr["week_start"].nunique())
        for name, p in preds.items():
            out.append(base.assign(model=name, pred=p))
    return pd.concat(out, ignore_index=True).rename(columns={"weekly_return": "actual"})


def evaluate(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    oos = walk_forward(build_dataset())
    table = metrics.score_table(oos, ["model"])
    table.insert(1, "model_label", table["model"].map(WEEKLY_LABELS))
    by_stock = metrics.score_table(oos, ["model", "symbol"])
    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        oos.to_csv(WEEKLY_OOS_FILE, index=False)
        table.to_csv(WEEKLY_COMPARISON_FILE, index=False)
        by_stock.to_csv(MODELS / "weekly_model_comparison_by_stock.csv", index=False)
    return oos, table


def load_oos() -> pd.DataFrame | None:
    if not WEEKLY_OOS_FILE.exists():
        return None
    return pd.read_csv(WEEKLY_OOS_FILE, parse_dates=["week_start", "entry_date", "exit_date"])


def main() -> None:
    oos, table = evaluate()
    print(f"Weekly walk-forward: {oos['week_start'].nunique()} test weeks, "
          f"{oos['week_start'].min():%Y-%m-%d} to {oos['week_start'].max():%Y-%m-%d}")
    print(table.round(4).to_string(index=False))
    wide = oos.pivot_table(index=["week_start", "symbol"], columns="model", values="pred").join(
        oos.drop_duplicates(["week_start", "symbol"]).set_index(["week_start", "symbol"])["actual"])
    for m in [k for k in WEEKLY_LABELS if k != "zero"]:
        print(f"corr({m}, actual) = {wide[m].corr(wide['actual']):.3f}")


if __name__ == "__main__":
    main()
