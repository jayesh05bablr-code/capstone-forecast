"""Tab 4, Backtesting: unfiltered vs cost-aware weekly strategy on the same weeks and capital."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import backtest, charts_eval, weekly_model
from capstone.ui import data_cache
from capstone.charts import CONFIG as CHART_CONFIG

RULES = """
**Rules.** Signals are the weekly model's walk-forward forecasts, made only from data before each week's entry
open. Long only (a cash-market short cannot be held for a week). Every trade gets the allocation % of the equity
at the start of the week, in whole shares; all trades close at the week's last session, so trades never overlap
across weeks. Every trade pays the full round-trip cost from the sidebar (explicit charges + spread + slippage).
Strategy A takes every positive forecast; Strategy B only forecasts above N x the estimated cost. The grey
reference holds every stock every week, to show what the signal adds or removes. Sharpe ratio uses weekly returns
(cash weeks count as 0), annualised by sqrt(52), risk-free rate 0; with weak, noisy returns treat it as indicative.
"""


DEFAULT_MODEL_INDEX = 2


def result(cfg) -> dict | None:
    """The backtest for the signal model last picked on the Backtesting page (cached), for Monte Carlo."""
    if data_cache.weekly_oos() is None:
        return None
    options = list(weekly_model.WEEKLY_LABELS)[1:]
    return data_cache.backtests(st.session_state.get("bt_model", options[DEFAULT_MODEL_INDEX]), cfg)


def render(cfg, symbols: list[str]) -> dict | None:
    if data_cache.weekly_oos() is None:
        st.warning("No weekly forecasts yet: run `python -m capstone.train`.")
        return None
    model = st.selectbox("Signal model", list(weekly_model.WEEKLY_LABELS)[1:], index=DEFAULT_MODEL_INDEX,
                         format_func=weekly_model.WEEKLY_LABELS.get, key="bt_model",
                         help="HistGradientBoosting was fixed in advance as the signal; the others are for comparison.")
    res = data_cache.backtests(model, cfg)
    comp = data_cache.models_csv("weekly_model_comparison.csv")
    if comp is not None:
        with st.expander("How good is the weekly forecast itself?", expanded=True):
            st.dataframe(comp.round(3), hide_index=True)
            st.caption("Same walk-forward test weeks for every model. If no model beats 'Zero' on MAE and the "
                       "directional accuracy is near 50%, the signal has no measurable edge and the backtest "
                       "mostly measures costs and market drift.")

    s = res["summary"].set_index("strategy")
    names = {k: v for k, v in backtest.STRATEGIES.items()}
    c = st.columns(3)
    for col, k in zip(c, ["A", "B", "H"]):
        r = s.loc[names[k]]
        col.metric(names[k], f"₹{r['net_pnl']:,.0f} net", f"{r['cumulative_return_pct']:+.1f}%")
    table = s.T.rename(index={
        "weeks": "Weeks", "trades": "Trades", "winning_trades": "Winning trades", "losing_trades": "Losing trades",
        "win_rate_pct": "Win rate %", "gross_pnl": "Gross P&L (₹)", "total_costs": "Total transaction costs (₹)",
        "net_pnl": "Net P&L (₹)", "ending_capital": "Ending capital (₹)", "cumulative_return_pct": "Cumulative return %",
        "max_drawdown_pct": "Maximum drawdown %", "sharpe_annualised": "Sharpe ratio (annualised)",
        "avg_net_return_per_trade_pct": "Average net return per trade %"})
    st.dataframe(table.astype(float).round(2), use_container_width=True)
    st.plotly_chart(charts_eval.equity_and_drawdown(res["weeks"], names, cfg.initial_capital),
                    use_container_width=True, config=CHART_CONFIG)
    st.caption("Top: capital after all costs. Bottom: how far each strategy is below its previous peak. "
               "Click a legend entry to hide a strategy; drag to zoom into a period.")
    kind = st.radio("Trades of", ["A", "B"], format_func=names.get, horizontal=True, key="bt_trades")
    trades = res["trades"][kind]
    st.dataframe(trades.round(2), hide_index=True, use_container_width=True, height=300)
    st.download_button("Download these trades (CSV)", trades.to_csv(index=False).encode(), f"trades_{kind}.csv",
                       "text/csv")
    st.markdown(RULES)
    return res
