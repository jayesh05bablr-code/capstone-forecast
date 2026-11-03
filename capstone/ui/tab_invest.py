"""Invest planner tab: split an amount across the stocks with ATR stops and targets."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import planner
from capstone.ui import kite_view
from capstone.ui.common import raw_prices


def plan_table(pl: pd.DataFrame, capital: float) -> pd.DataFrame:
    """The plan with readable column names and rounding."""
    return pd.DataFrame({
        "Stock": pl["symbol"], "Sector": pl["sector"],
        "Split %": (pl["invest"] / capital * 100).round(1),
        "Shares": pl["shares"], "Amount (₹)": pl["invest"].round(0),
        "Buy near (₹)": pl["entry"].round(2), "Stop loss (₹)": pl["stop"].round(2),
        "Stop %": (pl["stop_pct"] * 100).round(2), "Target (₹)": pl["target"].round(2),
        "Max loss (₹)": pl["loss_at_stop"].round(0), "Profit at target (₹)": pl["profit_at_target"].round(0),
        "Backtest win rate %": (pl["win_rate"] * 100).round(1),
        "Avg return per trade %": (pl["avg_ret"] * 100).round(2),
        "Expected profit (₹)": pl["expected_profit"].round(0),
        "Forecast gap %": pl["pred_gap_pct"].round(2),
    })


def render(preds: pd.DataFrame) -> None:
    st.warning("Educational tool for the capstone, not investment advice. The forecast is for the "
               "opening price, which is already priced in once the market opens, so this plan "
               "assumes you buy at the expected open and focuses on how much to put in each stock "
               "and where to cut losses. Past backtest results do not guarantee future returns.")
    c1, c2, c3 = st.columns(3)
    capital = c1.number_input("Amount to invest (₹)", min_value=1000, value=100000, step=5000)
    strategy = c2.selectbox("How to split", list(planner.STRATEGIES))
    hold = c3.slider("Holding period (trading days)", 1, 20, 5)
    c4, c5, c6, c7 = st.columns(4)
    stop_mult = c4.slider("Stop loss distance (× ATR)", 0.5, 4.0, 1.5, 0.25,
                          help="ATR is the stock's average daily range over 14 days.")
    reward = c5.slider("Target (× the stop distance)", 1.0, 4.0, 2.0, 0.5)
    risk_pct = c6.slider("Max loss per stock (% of amount)", 0.5, 5.0, 1.0, 0.5) / 100
    max_w = c7.slider("Max in one stock (%)", 20, 100, 40, 5) / 100
    st.caption(planner.STRATEGIES[strategy])

    prices = {sym: h for sym in preds["symbol"] if (h := raw_prices(sym)) is not None}
    with st.spinner("Backtesting the stop/target plan on each stock's history..."):
        pl = planner.plan(preds, prices, capital, strategy, risk_pct, stop_mult, reward, hold, max_w)
    if pl.empty:
        st.info("No price history found for these stocks yet.")
        return

    invested = pl["invest"].sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Invested", f"₹{invested:,.0f}", f"₹{capital - invested:,.0f} kept as cash", delta_color="off")
    m2.metric("Expected profit (backtest)", f"₹{pl['expected_profit'].sum():,.0f}",
              f"{pl['expected_profit'].sum() / capital * 100:+.2f}% of amount", delta_color="off")
    m3.metric("If every target is hit", f"₹{pl['profit_at_target'].sum():,.0f}")
    m4.metric("If every stop is hit", f"₹{pl['loss_at_stop'].sum():,.0f}")
    st.dataframe(plan_table(pl, capital), use_container_width=True, hide_index=True)
    notes = planner.allocation_notes(pl, strategy, risk_pct)
    if notes:
        st.markdown("\n".join(f"- {n}" for n in notes))
    st.caption(f"Backtest: last ~500 trading days per stock, buying at each day's open, selling at "
               f"the stop, the target or the close after {hold} days, whichever comes first "
               "(a day that touches both counts as a stop). Expected profit = amount × average "
               "return per trade. Brokerage and taxes are not included.")
    kite_view.render_swing(pl)
