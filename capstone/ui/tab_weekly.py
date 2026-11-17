"""Tab 3, Weekly market analysis: first-session open to last-session close, against trading cost."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import charts_eval, costs, weekly
from capstone.ui import assumptions, data_cache

METHOD = """
**Method.** Weekly delta = (exit close - entry open) / entry open, for calendar weeks (Monday to Sunday).
The NSE trading calendar is the set of dates with an NSE price bar, so holidays and special sessions come
from the data. Entry = first trading session of the week (Tuesday if Monday is a holiday); exit = last session
of the same week (Thursday if Friday is a holiday). Holiday-shortened weeks are kept. Weeks are excluded when an
entry or exit price is missing, when a bonus or split falls inside the holding period (prices are unadjusted, so
the week would show a fake loss), or when the week is still in progress. No future price is used to decide entry:
this tab only describes realised weeks; the forecast signal is on the Backtesting tab.

**Tradable weeks (%)** = valid weeks whose realised return beats the round-trip cost / all valid weeks x 100.
The cost line uses the sidebar assumptions for one position of (capital x allocation).
"""


def render(cfg, symbols: list[str]) -> None:
    weeks = data_cache.weeks()
    if weeks.empty:
        st.warning("No price history yet.")
        return
    weeks = weeks[weeks["symbol"].isin(symbols)]
    pos = assumptions.position_value(cfg, symbols)
    cost_pct = costs.round_trip_pct(pos, cfg.assumptions)

    lo, hi = weeks["week_start"].min().date(), weeks["week_start"].max().date()
    rng = st.date_input("Weeks from / to", (lo, hi), min_value=lo, max_value=hi, key="wk_dates")
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        weeks = weeks[(weeks["week_start"] >= pd.Timestamp(rng[0])) & (weeks["week_start"] <= pd.Timestamp(rng[1]))]

    c = st.columns(4)
    c[0].metric("Position size", f"₹{pos:,.0f}")
    c[1].metric("Total round-trip cost", f"{cost_pct * 100:.3f}%", f"₹{cost_pct * pos:,.0f} per trade", delta_color="off")
    valid = weeks[weeks["valid"]]
    c[2].metric("Valid stock-weeks", f"{len(valid):,}", f"{(~weeks['valid']).sum()} excluded", delta_color="off")
    c[3].metric("Tradable (all stocks)", f"{(valid['weekly_return'] > cost_pct).mean() * 100:.1f}%")

    stats = weekly.tradable_stats(weeks, cost_pct)
    show = stats.rename(columns={
        "symbol": "Stock", "valid_weeks": "Valid weeks", "tradable_weeks": "Tradable weeks",
        "tradable_pct": "Tradable weeks %", "avg_weekly_return_pct": "Average %", "median_weekly_return_pct": "Median %",
        "best_week": "Best week", "best_week_return_pct": "Best %", "worst_week": "Worst week",
        "worst_week_return_pct": "Worst %", "round_trip_cost_pct": "Round-trip cost %", "excluded_weeks": "Excluded"})
    st.dataframe(show.round(2), hide_index=True, use_container_width=True)

    for sym in symbols:
        w = weeks[(weeks["symbol"] == sym) & weeks["valid"]]
        if len(w):
            st.plotly_chart(charts_eval.weekly_delta(w, sym, cost_pct), use_container_width=True)
    st.caption("Green = the week's realised return beat the round-trip cost (dashed line); grey = it did not.")

    with st.expander("Cost breakdown for one position"):
        st.dataframe(costs.breakdown_table(pos, cfg.assumptions), hide_index=True)
        st.caption(costs.__doc__.split("IMPORTANT:")[1].strip())
    with st.expander("Excluded weeks"):
        st.dataframe(weeks[~weeks["valid"]][["symbol", "week", "entry_date", "exit_date", "weekly_return",
                                             "exclude_reason"]], hide_index=True)
    st.markdown(METHOD)
