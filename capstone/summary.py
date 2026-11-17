"""Executive summary built from the measured result files (never from hard-coded numbers).

    python -m capstone.summary    (prints the Markdown used in EXECUTIVE_SUMMARY.md)
"""
from __future__ import annotations

import json

import pandas as pd

from . import backtest, model_eval, weekly_model
from .paths import MODELS


def measured() -> dict:
    """Headline numbers from models/*.csv; missing files give an empty dict section."""
    out: dict = {}
    if model_eval.COMPARISON_FILE.exists():
        t = pd.read_csv(model_eval.COMPARISON_FILE).set_index("model")
        out["daily"] = t.to_dict("index")
        out["daily_meta"] = json.loads(model_eval.META_FILE.read_text()) if model_eval.META_FILE.exists() else {}
    if model_eval.TESTS_FILE.exists():
        out["tests"] = pd.read_csv(model_eval.TESTS_FILE).to_dict("records")
    if model_eval.COVERAGE_FILE.exists():
        cov = pd.read_csv(model_eval.COVERAGE_FILE)
        out["coverage"] = cov[cov["model"] == "hist_gradient_boosting"]["coverage_pct"].tolist()
    if weekly_model.WEEKLY_COMPARISON_FILE.exists():
        out["weekly"] = pd.read_csv(weekly_model.WEEKLY_COMPARISON_FILE).set_index("model").to_dict("index")
    if (MODELS / "backtest_summary.csv").exists():
        out["backtest"] = pd.read_csv(MODELS / "backtest_summary.csv").set_index("strategy").to_dict("index")
    if (MODELS / "montecarlo_summary.json").exists():
        out["mc"] = json.loads((MODELS / "montecarlo_summary.json").read_text())
    if (MODELS / "weekly_returns.csv").exists():
        w = pd.read_csv(MODELS / "weekly_returns.csv")
        out["weeks_valid"] = int(w["valid"].sum())
    return out


def markdown(m: dict | None = None) -> str:
    m = measured() if m is None else m
    parts = ["""**Business problem.** Before 09:15 IST an investor has to digest what happened overnight (the US close,
the rupee, ADRs) and how each stock traded the day before, and decide whether to act at the open. Doing this by
hand for every stock each morning is slow and inconsistent.

**Target user and morning decision.** An investor or market analyst following selected NSE large caps who wants a
structured, data-driven view at 08:45 IST: which stocks are likely to open higher or lower, by how much, how
confident to be, and whether a short-term trade is worth its costs.

**Proposed solution.** A Streamlit dashboard that refreshes itself after the US close and after the NSE close:
it forecasts each stock's opening gap with a 10th-90th percentile range and explains the drivers, tracks its own
historical accuracy against simple baselines, analyses weekly trading opportunities net of realistic costs,
backtests a weekly signal, stress-tests it with Monte Carlo simulation, and flags disagreements with company
fundamentals.

**Data sources.** NSE daily OHLCV (TejHQ API, Yahoo Finance fallback); US S&P 500, Nasdaq 100, 11 sector ETFs and
Indian ADRs, and USD/INR history (Twelve Data, Alpha Vantage fallback); live USD/INR (exchangerate.dev);
quarterly results and shareholding (Tracxn exports / company data workbooks). Optional Supabase stores the outputs.

**Model architecture.** Features known at the close plus the overnight US/rupee window up to the next open;
target = next open / close - 1. HistGradientBoosting (existing model) compared with Linear Regression, a naive
"open = last close" baseline and a leakage-free historical-average baseline on identical expanding walk-forward
folds. Intervals from earlier folds' out-of-sample residuals. Driver attribution by what-if neutralisation."""]

    d = m.get("daily")
    if d:
        meta = m.get("daily_meta", {})
        hgb, lr, nv, ha = (d.get(k, {}) for k in ("hist_gradient_boosting", "linear_regression", "naive",
                                                   "historical_average"))
        improvement = (1 - hgb["mae_pct"] / nv["mae_pct"]) * 100
        verdict = next((t["verdict"] for t in m.get("tests", []) if "Linear" in t["model_b"]), "not tested")
        cov = m.get("coverage", [])
        parts.append(
            f"**Model performance vs naive baseline** ({meta.get('test_rows_per_model', hgb.get('n', 0)):,} stock-days, "
            f"{meta.get('test_start', '')} to {meta.get('test_end', '')}). HistGradientBoosting: MAE "
            f"{hgb['mae_pct']:.3f} % points, directional accuracy {hgb['direction_acc_pct']:.1f}%. Naive baseline: MAE "
            f"{nv['mae_pct']:.3f}, directional accuracy {nv['direction_acc_pct']:.1f}% (a zero forecast is only right "
            f"on a zero gap). Historical average: MAE {ha['mae_pct']:.3f}, {ha['direction_acc_pct']:.1f}%. The model cuts "
            f"the naive error by {improvement:.1f}%. Linear Regression: MAE {lr['mae_pct']:.3f}, "
            f"{lr['direction_acc_pct']:.1f}%; HistGradientBoosting vs Linear Regression: {verdict.lower()}, so the "
            f"simpler model is about as good on average."
            + (f" The 10-90% interval covered {min(cov):.0f}-{max(cov):.0f}% of actual gaps per test period "
               f"(target 80%)." if cov else ""))
    w, bt, mc = m.get("weekly"), m.get("backtest"), m.get("mc")
    if w and bt:
        hgb_w, zero_w = w["hist_gradient_boosting"], w["zero"]
        a, b, h = (bt[backtest.STRATEGIES[k]] for k in ("A", "B", "H"))
        edge = ("has no measurable edge" if hgb_w["mae_pct"] >= zero_w["mae_pct"] or hgb_w["direction_acc_pct"] <= 52
                else "has a small edge")
        from . import analysis, costs
        cfg = analysis.default_config()
        n_sym = len(m.get("daily_meta", {}).get("symbols", [])) or 5
        pos = cfg.initial_capital * cfg.alloc(n_sym)
        cost = costs.round_trip_pct(pos, cfg.assumptions) * 100
        ending = ("weekly round trips in these stocks did not cover their costs" if h["net_pnl"] < 0
                  else "holding every week stayed profitable after costs")
        parts.append(
            f"**Weekly trading analysis.** {m.get('weeks_valid', 0):,} valid stock-weeks (first-session open to "
            f"last-session close, holiday weeks kept, bonus/split weeks excluded). The weekly forecast {edge}: MAE {hgb_w['mae_pct']:.2f} % points vs {zero_w['mae_pct']:.2f} for a zero forecast, "
            f"directional accuracy {hgb_w['direction_acc_pct']:.1f}%. With default costs ({cost:.2f}% per round trip "
            f"for a ₹{pos:,.0f} delivery position) on ₹{cfg.initial_capital:,.0f}: unfiltered strategy {a['trades']:.0f} trades, net "
            f"{a['cumulative_return_pct']:+.1f}%, max drawdown {a['max_drawdown_pct']:.1f}%; cost-aware strategy "
            f"{b['trades']:.0f} trades, net {b['cumulative_return_pct']:+.1f}%, max drawdown "
            f"{b['max_drawdown_pct']:.1f}%; holding every stock every week {h['cumulative_return_pct']:+.1f}% "
            f"(gross P&L ₹{h['gross_pnl']:,.0f}, costs ₹{h['total_costs']:,.0f}). The cost-aware filter traded "
            f"{b['trades'] / max(a['trades'], 1) * 100:.0f}% as often as the unfiltered strategy; {ending}."
            + (f" Monte Carlo ({mc['simulations']:,} block-bootstrap paths of the cost-aware strategy): "
               f"{mc['prob_loss_pct']:.0f}% probability of ending below the starting capital; median return "
               f"{mc['cumulative_return_pct_p50']:+.1f}%, 5th-95th percentile {mc['cumulative_return_pct_p5']:+.0f}% "
               f"to {mc['cumulative_return_pct_p95']:+.0f}%." if mc else ""))
    parts.append("""**Business value.** One screen replaces a manual overnight scan: a consistent, explainable opening
view with an honest error range, evidence on when the model beats simple rules, and a cost-aware check that
stops the user from chasing short-term trades whose expected edge is smaller than their costs.

**Limitations.** Decision support only, not a guaranteed profit system or investment advice. Earnings, corporate
announcements, geopolitical shocks, liquidity and regime changes can break any forecast; historical and simulated
results do not guarantee future performance; the five stocks are survivors (survivorship bias); cost rates are
assumptions to verify with a broker.""")
    return "\n\n".join(parts)


if __name__ == "__main__":
    print(markdown())
