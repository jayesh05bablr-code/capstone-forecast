"""Monte Carlo simulation of the cost-aware strategy by bootstrap resampling of its history.

Unit of resampling: the WEEK, not the trade. All trades of one week share the same market moves
(the five stocks are correlated), so drawing trades one by one would treat them as independent
and understate risk. Each simulated path is built from the strategy's realised weekly net returns
(weeks with no trade count as 0%, because the capital sat in cash).

Moving-block bootstrap: weeks are drawn in consecutive blocks of `block_weeks` (default 4) so that
short-run dependence such as volatile months is kept together. Paths have the same length as the
backtest unless `horizon_weeks` is given. Each path compounds from the starting capital.

Assumptions and limitations
- The future is assumed to look like a reshuffled version of the backtest period: regimes that
  never happened in it (a new crash, a change in costs or market structure) cannot appear.
- Inputs are themselves out-of-sample backtest returns with costs, but they share the backtest's
  assumptions (cost rates, fills at the open/close, no liquidity limits).
- Results are hypothetical and are not a forecast or a guarantee of future performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics

N_SIMULATIONS = 5000
SEED = 42
PERCENTILES = (5, 25, 50, 75, 95)


def block_bootstrap_paths(weekly_returns, n_sims: int = N_SIMULATIONS, horizon: int | None = None,
                          block: int = 4, seed: int = SEED) -> np.ndarray:
    """Matrix (n_sims x horizon) of resampled weekly returns using moving blocks."""
    r = np.asarray(weekly_returns, float)
    n = len(r)
    if n == 0:
        raise ValueError("No weekly returns to resample.")
    horizon = horizon or n
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n - block + 1, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :horizon]
    return r[idx]


def simulate(weekly_returns, initial_capital: float, n_sims: int = N_SIMULATIONS, horizon: int | None = None,
             block: int = 4, seed: int = SEED) -> pd.DataFrame:
    """One row per simulated path: ending capital, cumulative return and maximum drawdown."""
    paths = block_bootstrap_paths(weekly_returns, n_sims, horizon, block, seed)
    equity = initial_capital * np.cumprod(1 + paths, axis=1)
    full = np.hstack([np.full((n_sims, 1), initial_capital), equity])
    peak = np.maximum.accumulate(full, axis=1)
    mdd = (full / peak - 1).min(axis=1)
    return pd.DataFrame({"sim": np.arange(n_sims), "ending_capital": equity[:, -1],
                         "cumulative_return_pct": (equity[:, -1] / initial_capital - 1) * 100,
                         "max_drawdown_pct": mdd * 100})


def summarise(sims: pd.DataFrame, initial_capital: float) -> dict:
    out = {"simulations": len(sims), "prob_loss_pct": float((sims["ending_capital"] < initial_capital).mean() * 100)}
    for col in ("ending_capital", "cumulative_return_pct", "max_drawdown_pct"):
        for p in PERCENTILES:
            out[f"{col}_p{p}"] = float(np.percentile(sims[col], p))
    return out


def percentile_table(sims: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"{p}th": [np.percentile(sims[c], p) for c in ("ending_capital", "cumulative_return_pct",
                                                                         "max_drawdown_pct")] for p in PERCENTILES},
                        index=["Ending capital (₹)", "Cumulative return %", "Max drawdown %"])


def historical_vs_simulated(weeks: pd.DataFrame, sims: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Backtest outcome next to the simulated median and 5-95% range."""
    eq = np.r_[initial_capital, weeks["equity"].to_numpy()]
    hist = {"Ending capital (₹)": eq[-1], "Cumulative return %": (eq[-1] / initial_capital - 1) * 100,
            "Max drawdown %": metrics.max_drawdown(eq) * 100}
    rows = []
    for (label, h), col in zip(hist.items(), ["ending_capital", "cumulative_return_pct", "max_drawdown_pct"]):
        rows.append({"Measure": label, "Historical backtest": h, "Simulated median": np.median(sims[col]),
                     "Simulated 5th pct": np.percentile(sims[col], 5), "Simulated 95th pct": np.percentile(sims[col], 95)})
    return pd.DataFrame(rows)
