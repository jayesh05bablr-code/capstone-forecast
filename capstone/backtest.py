"""Backtest the weekly signal: unfiltered vs cost-aware, on identical weeks, capital and costs.

Rules (both strategies)
- Universe: the selected stocks; signals are the walk-forward weekly forecasts
  (weekly_model.py), so every signal uses only data from before that week's entry open.
- Entry at the open of the week's first session, exit at the close of its last session.
- Long only: a cash-market (delivery) short cannot be carried for a week in India.
- Position sizing: at the start of each week every trade gets `allocation` x the equity at that
  moment (default 1 / number of stocks, so all positions together never exceed the equity).
  Whole shares only. Positions are opened and closed inside the same week, so trades never
  overlap across weeks and capital is always back in cash by the next entry.
- Costs: costs.round_trip on the actual buy and sell values (explicit + spread + slippage).

Strategy A (unfiltered):  trade every stock whose forecast weekly return is positive.
Strategy B (cost-aware):  trade only when the forecast exceeds `cost_multiple` (default 2) x the
                          estimated round-trip cost of that position (known before entry).
A reference "hold every week" line (all stocks, every valid week) shows what the signal adds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import costs as cost_model
from . import metrics

STRATEGIES = {"A": "A. Unfiltered (forecast > 0)", "B": "B. Cost-aware (forecast > 2x cost)",
              "H": "Reference: hold every stock every week"}


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    allocation: float | None = None          # fraction of equity per trade; None = 1 / n stocks
    cost_multiple: float = 2.0
    assumptions: cost_model.CostAssumptions = cost_model.CostAssumptions()
    symbols: tuple[str, ...] | None = None   # None = every stock in the signal table

    def alloc(self, n_stocks: int) -> float:
        a = self.allocation if self.allocation else 1.0 / max(n_stocks, 1)
        return min(a, 1.0 / max(n_stocks, 1)) if a * n_stocks > 1 else a


def _select(signals: pd.DataFrame, kind: str, est_cost: pd.Series, cfg: BacktestConfig) -> pd.Series:
    if kind == "A":
        return signals["pred"] > 0
    if kind == "B":
        return signals["pred"] > cfg.cost_multiple * est_cost
    return pd.Series(True, index=signals.index)


def run(signals: pd.DataFrame, kind: str, cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one strategy. signals: one row per (symbol, week) with pred, entry_open, exit_close.

    Returns (trades, weeks): every trade with gross/cost/net P&L, and the weekly equity curve
    (weeks with no trade are kept with a 0 return, so all strategies share the same dates).
    """
    sig = signals.copy()
    if cfg.symbols:
        sig = sig[sig["symbol"].isin(cfg.symbols)]
    sig = sig.sort_values(["entry_date", "symbol"])
    n_stocks = sig["symbol"].nunique()
    alloc = cfg.alloc(n_stocks)
    equity = cfg.initial_capital
    trades, weeks = [], []
    for week_start, g in sig.groupby("week_start", sort=True):
        start_equity = equity
        position_value = start_equity * alloc
        est_cost = pd.Series(cost_model.round_trip_pct(position_value, cfg.assumptions), index=g.index)
        chosen = g[_select(g, kind, est_cost, cfg)]
        week_net = week_gross = week_cost = 0.0
        n = 0
        for r in chosen.itertuples():
            qty = int(position_value // r.entry_open)
            if qty <= 0:
                continue
            buy, sell = qty * r.entry_open, qty * r.exit_close
            c = cost_model.round_trip(buy, cfg.assumptions, sell)
            gross = sell - buy
            net = gross - c["total"]
            trades.append({"week_start": week_start, "entry_date": r.entry_date, "exit_date": r.exit_date,
                           "symbol": r.symbol, "pred_pct": r.pred * 100, "est_cost_pct": est_cost[r.Index] * 100,
                           "qty": qty, "entry_price": r.entry_open, "exit_price": r.exit_close,
                           "buy_value": buy, "gross_pnl": gross, "explicit_cost": c["explicit_total"],
                           "spread_cost": c["spread"], "slippage_cost": c["slippage"], "total_cost": c["total"],
                           "net_pnl": net, "net_return_pct": net / buy * 100})
            week_net += net
            week_gross += gross
            week_cost += c["total"]
            n += 1
        equity = start_equity + week_net
        weeks.append({"week_start": week_start, "trades": n, "start_equity": start_equity, "gross_pnl": week_gross,
                      "costs": week_cost, "net_pnl": week_net, "net_return": week_net / start_equity,
                      "equity": equity})
    return pd.DataFrame(trades), pd.DataFrame(weeks)


def summary(trades: pd.DataFrame, weeks: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """Headline statistics of one backtest."""
    n = len(trades)
    wins = int((trades["net_pnl"] > 0).sum()) if n else 0
    eq = np.r_[cfg.initial_capital, weeks["equity"].to_numpy()] if len(weeks) else np.array([cfg.initial_capital])
    return {
        "weeks": len(weeks), "trades": n, "winning_trades": wins, "losing_trades": n - wins,
        "win_rate_pct": wins / n * 100 if n else np.nan,
        "gross_pnl": float(trades["gross_pnl"].sum()) if n else 0.0,
        "total_costs": float(trades["total_cost"].sum()) if n else 0.0,
        "net_pnl": float(trades["net_pnl"].sum()) if n else 0.0,
        "ending_capital": float(eq[-1]),
        "cumulative_return_pct": (eq[-1] / cfg.initial_capital - 1) * 100,
        "max_drawdown_pct": metrics.max_drawdown(eq) * 100,
        "sharpe_annualised": metrics.sharpe_ratio(weeks["net_return"], 52) if len(weeks) else np.nan,
        "avg_net_return_per_trade_pct": float(trades["net_return_pct"].mean()) if n else np.nan,
    }


def compare(signals: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """Run A, B and the reference on the same weeks. Returns summaries, trades and equity curves."""
    out = {"summary": [], "trades": {}, "weeks": {}}
    for kind in ("A", "B", "H"):
        t, w = run(signals, kind, cfg)
        out["trades"][kind], out["weeks"][kind] = t, w
        out["summary"].append({"strategy": STRATEGIES[kind], **summary(t, w, cfg)})
    out["summary"] = pd.DataFrame(out["summary"])
    return out


def main() -> None:
    from . import weekly_model

    oos = weekly_model.load_oos()
    if oos is None:
        _, _ = weekly_model.evaluate()
        oos = weekly_model.load_oos()
    sig = oos[oos["model"] == weekly_model.SIGNAL_MODEL]
    res = compare(sig, BacktestConfig())
    pd.set_option("display.width", 200)
    print(res["summary"].round(2).T.to_string())


if __name__ == "__main__":
    main()
