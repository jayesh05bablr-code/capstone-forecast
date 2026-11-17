"""Weekly analysis, backtests and Monte Carlo with the default settings, saved for the dashboard.

    python -m capstone.analysis          (also run at the end of python -m capstone.train)

Writes models/weekly_*.csv, models/backtest_*.csv and models/montecarlo_*.csv, and, when Supabase
is configured, persists the same results plus the model-evaluation metrics (best effort: a
database problem is reported but never stops the pipeline). The dashboard recomputes the
backtest and simulation live when you change the assumptions; these files are the defaults.
"""
from __future__ import annotations

import json

import pandas as pd

from . import backtest, costs, db, model_eval, montecarlo, overnight, weekly, weekly_model
from .env import redact
from .paths import MODELS

DEFAULT_CAPITAL = 1_000_000.0


def default_config() -> backtest.BacktestConfig:
    return backtest.BacktestConfig(initial_capital=DEFAULT_CAPITAL)


def config_dict(cfg: backtest.BacktestConfig, signal_model: str, symbols: list[str]) -> dict:
    return {"initial_capital": cfg.initial_capital, "allocation": cfg.alloc(len(symbols)),
            "cost_multiple": cfg.cost_multiple, "symbols": sorted(symbols), "signal_model": signal_model,
            "costs": cfg.assumptions.to_dict()}


def run(save_db: bool = True) -> dict:
    prices = weekly.load_prices()
    weeks = weekly.all_weeks(prices)
    oos, comparison = weekly_model.evaluate()
    cfg = default_config()
    signals = oos[oos["model"] == weekly_model.SIGNAL_MODEL]
    res = backtest.compare(signals, cfg)
    sims = montecarlo.simulate(res["weeks"]["B"]["net_return"], cfg.initial_capital)
    mc_summary = montecarlo.summarise(sims, cfg.initial_capital)

    MODELS.mkdir(parents=True, exist_ok=True)
    weeks.to_csv(MODELS / "weekly_returns.csv", index=False)
    res["summary"].to_csv(MODELS / "backtest_summary.csv", index=False)
    for k in ("A", "B", "H"):
        res["trades"][k].to_csv(MODELS / f"backtest_trades_{k}.csv", index=False)
        res["weeks"][k].to_csv(MODELS / f"backtest_weeks_{k}.csv", index=False)
    montecarlo.percentile_table(sims).to_csv(MODELS / "montecarlo_percentiles.csv")
    (MODELS / "montecarlo_summary.json").write_text(json.dumps(mc_summary, indent=2))

    symbols = sorted(signals["symbol"].unique())
    out = {"weeks": weeks, "weekly_comparison": comparison, "backtest": res, "mc_summary": mc_summary}
    if save_db:
        out["db"] = persist(weeks, res, cfg, symbols, mc_summary)
    return out


def persist(weeks, res, cfg, symbols, mc_summary) -> str:
    """Best-effort write of everything to Supabase. Returns a one-line status."""
    try:
        store = db.get_store()
    except db.SupabaseError as exc:
        return f"Supabase not used: {exc}"
    if store is None:
        return "Supabase not configured: results kept in local files only."
    try:
        evaluation = None
        if model_eval.COMPARISON_FILE.exists() and model_eval.META_FILE.exists():
            evaluation = ("daily_gap", pd.read_csv(model_eval.COMPARISON_FILE), pd.read_csv(model_eval.BY_STOCK_FILE),
                          json.loads(model_eval.META_FILE.read_text()))
        counts = db.save_analysis(
            store, stock_map=overnight.stock_map(), weeks=weeks[weeks["symbol"].isin(symbols)], evaluation=evaluation,
            config=config_dict(cfg, weekly_model.SIGNAL_MODEL, symbols),
            backtests=[(backtest.STRATEGIES[k], weekly_model.SIGNAL_MODEL, res["summary"].iloc[i].to_dict(),
                        res["trades"][k], res["weeks"][k]) for i, k in enumerate(("A", "B", "H"))],
            montecarlo={"strategy": backtest.STRATEGIES["B"], "summary": mc_summary, "seed": montecarlo.SEED,
                        "block": 4, "horizon": len(res["weeks"]["B"]), "initial_capital": cfg.initial_capital})
        return "Saved to Supabase: " + ", ".join(f"{k} {v}" for k, v in counts.items())
    except (db.SupabaseError, ValueError) as exc:
        return redact(f"Supabase save failed (local files are still up to date): {exc}")


def main() -> None:
    out = run()
    pd.set_option("display.width", 200)
    print("\nWeekly forecast models (walk-forward, % points):")
    print(out["weekly_comparison"].round(3).to_string(index=False))
    print("\nBacktests (default assumptions):")
    print(out["backtest"]["summary"].round(2).T.to_string())
    print(f"\nMonte Carlo ({out['mc_summary']['simulations']} runs, cost-aware strategy): "
          f"probability of ending below starting capital {out['mc_summary']['prob_loss_pct']:.1f}%")
    print(out.get("db", ""))


if __name__ == "__main__":
    main()
