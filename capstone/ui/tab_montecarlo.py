"""Tab 5, Monte Carlo: 5,000 resampled futures of the cost-aware strategy."""
from __future__ import annotations

import streamlit as st

from capstone import backtest, charts_eval, montecarlo
from capstone.ui import data_cache

NOTE = """
**How it works.** Each simulated path draws blocks of consecutive weeks (with replacement) from Strategy B's
realised weekly net returns in the backtest, including weeks in cash, and compounds them from the starting
capital. The week, not the single trade, is the unit of resampling, because trades in the same week share the
same market move; blocks keep runs of good or bad weeks together. Seed fixed for reproducibility.

**Limitations.** Only reshuffles the past: a crash, regime change or cost change that is not in the backtest
cannot appear. It inherits every backtest assumption (fills at the open/close, the cost rates, no liquidity
limits). **These are hypothetical outcomes, not a forecast and not a guarantee of future performance.**
"""


def render(cfg, bt: dict | None) -> None:
    if bt is None:
        st.warning("Run the backtest first (Backtesting tab needs weekly forecasts).")
        return
    weeks = bt["weeks"]["B"]
    if weeks.empty or weeks["trades"].sum() == 0:
        st.info("The cost-aware strategy made no trades with these assumptions, so there is nothing to simulate.")
        return
    c = st.columns(4)
    n_sims = c[0].number_input("Simulations", 1000, 20000, montecarlo.N_SIMULATIONS, 1000, key="mc_n")
    horizon_choice = c[1].selectbox("Horizon", ["Same length as backtest", "52 weeks", "156 weeks"], key="mc_h")
    block = c[2].number_input("Block length (weeks)", 1, 26, 4, key="mc_block")
    seed = c[3].number_input("Random seed", 0, 10_000, montecarlo.SEED, key="mc_seed")
    horizon = {"52 weeks": 52, "156 weeks": 156}.get(horizon_choice, len(weeks))
    sims = data_cache.simulate(tuple(weeks["net_return"]), cfg.initial_capital, int(n_sims), int(horizon), int(block),
                               int(seed))
    summ = montecarlo.summarise(sims, cfg.initial_capital)

    c = st.columns(4)
    c[0].metric("Probability of capital loss", f"{summ['prob_loss_pct']:.1f}%")
    c[1].metric("Median ending capital", f"₹{summ['ending_capital_p50']:,.0f}")
    c[2].metric("Median max drawdown", f"{summ['max_drawdown_pct_p50']:.1f}%")
    c[3].metric("5th-95th pct return", f"{summ['cumulative_return_pct_p5']:.0f}% to {summ['cumulative_return_pct_p95']:.0f}%")
    st.markdown(f"**Percentile outcomes** ({int(n_sims):,} paths of {horizon} weeks, strategy: "
                f"{backtest.STRATEGIES['B']})")
    st.dataframe(montecarlo.percentile_table(sims).round(1), use_container_width=True)
    if horizon == len(weeks):
        st.markdown("**Historical backtest vs simulated outcomes**")
        st.dataframe(montecarlo.historical_vs_simulated(weeks, sims, cfg.initial_capital).round(1), hide_index=True)
    else:
        st.caption("Historical vs simulated comparison is shown when the horizon equals the backtest length.")

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts_eval.histogram(sims["ending_capital"], "Ending capital", "₹", "#2a78d6",
                                          {"Start": cfg.initial_capital}), use_container_width=True)
    c2.plotly_chart(charts_eval.histogram(sims["max_drawdown_pct"], "Maximum drawdown", "%", "#eb6834"),
                    use_container_width=True)
    st.plotly_chart(charts_eval.histogram(sims["cumulative_return_pct"], "Cumulative return", "%", "#1baf7a",
                                          {"Break-even": 0.0}), use_container_width=True)
    st.markdown(NOTE)
