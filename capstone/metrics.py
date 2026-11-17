"""Forecast-accuracy and risk metrics shared by the daily and weekly models and the backtests.

All return-type inputs are fractions (0.01 = 1%). Functions take array-likes and ignore nothing
silently: NaNs must be removed by the caller, so every metric is computed on the same rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def directional_accuracy(actual, predicted) -> float:
    """Share of forecasts whose direction matches the actual move, in percent.

    Definition (used everywhere in this project):
    - a positive forecast is correct only if the actual move is positive;
    - a negative forecast is correct only if the actual move is negative;
    - a zero forecast is correct only if the actual move is exactly zero.
    Every row counts, including days when the actual move was zero.
    """
    a, p = np.sign(np.asarray(actual, float)), np.sign(np.asarray(predicted, float))
    if a.size == 0:
        return float("nan")
    return float((a == p).mean() * 100)


def mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.asarray(predicted, float) - np.asarray(actual, float))))


def rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean((np.asarray(predicted, float) - np.asarray(actual, float)) ** 2)))


def bias(actual, predicted) -> float:
    """Mean of (forecast - actual): positive means the forecasts were too high on average."""
    return float(np.mean(np.asarray(predicted, float) - np.asarray(actual, float)))


def forecast_scores(actual, predicted) -> dict[str, float]:
    """MAE, RMSE and bias in percentage points of the gap, plus directional accuracy and N."""
    return {"n": int(len(actual)),
            "mae_pct": mae(actual, predicted) * 100,
            "rmse_pct": rmse(actual, predicted) * 100,
            "bias_pct": bias(actual, predicted) * 100,
            "direction_acc_pct": directional_accuracy(actual, predicted)}


def score_table(oos: pd.DataFrame, by: list[str], actual: str = "actual", pred: str = "pred") -> pd.DataFrame:
    """forecast_scores for every group of an out-of-sample prediction table."""
    rows = []
    for key, g in oos.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **forecast_scores(g[actual], g[pred])})
    return pd.DataFrame(rows)


def max_drawdown(equity) -> float:
    """Largest peak-to-trough fall of an equity curve, as a negative fraction (0 = never fell).

    The starting value counts as the first peak, so a curve that falls from the start is measured
    from its starting capital.
    """
    eq = np.asarray(equity, float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1))


def sharpe_ratio(period_returns, periods_per_year: int = 52) -> float:
    """Annualised Sharpe ratio of per-period returns, risk-free rate taken as 0.

    Returns NaN with fewer than 8 periods or zero volatility, where the ratio is not meaningful.
    """
    r = np.asarray(period_returns, float)
    if r.size < 8 or np.std(r, ddof=1) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(periods_per_year))


def rolling_scores(df: pd.DataFrame, window: int, actual: str = "actual", pred: str = "pred") -> pd.DataFrame:
    """Rolling MAE (pct points) and rolling directional accuracy (%) over `window` rows, in date order."""
    d = df.sort_values("date")
    abs_err = (d[pred] - d[actual]).abs() * 100
    hit = (np.sign(d[pred]) == np.sign(d[actual])).astype(float) * 100
    return pd.DataFrame({"date": d["date"].to_numpy(),
                         "rolling_mae_pct": abs_err.rolling(window, min_periods=window // 2).mean().to_numpy(),
                         "rolling_direction_acc_pct": hit.rolling(window, min_periods=window // 2).mean().to_numpy()})


def diebold_mariano(loss_a, loss_b, dates, max_lag: int = 5) -> dict[str, float]:
    """Is model A's average loss different from model B's? (Diebold-Mariano test, abs-error loss)

    Losses are first averaged across stocks on each date, because stocks move together and are not
    independent observations; the daily differences then get a Newey-West (HAC) standard error.
    Negative mean_diff means A had the smaller error. |t| > 1.96 is about 5% significance.
    """
    d = pd.DataFrame({"date": dates, "diff": np.asarray(loss_a, float) - np.asarray(loss_b, float)})
    x = d.groupby("date")["diff"].mean().to_numpy()
    n = len(x)
    if n < 30:
        return {"mean_diff": float("nan"), "t_stat": float("nan"), "n_days": n}
    u = x - x.mean()
    var = np.mean(u * u)
    for lag in range(1, max_lag + 1):
        var += 2 * (1 - lag / (max_lag + 1)) * np.mean(u[lag:] * u[:-lag])
    return {"mean_diff": float(x.mean()), "t_stat": float(x.mean() / np.sqrt(var / n)), "n_days": n}
