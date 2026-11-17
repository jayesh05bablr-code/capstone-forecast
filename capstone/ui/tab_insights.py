"""Tab 6, Business insights: forecast vs fundamentals, executive summary and limitations."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import fundamental_check
from capstone.ui import summary

LIMITS = """
**Important limitations**

- This is a decision-support tool, not a profit-generating system and not investment advice.
- Unexpected announcements, earnings, block deals, geopolitical events, liquidity and regime changes can move
  the open far from the forecast; the 10-90% range is a statistical estimate, not a boundary.
- Historical performance, backtests and simulations do not guarantee future results.
- The stock list is today's large caps (survivorship bias): past returns of companies that later shrank or
  were delisted are not in the sample.
- Cost rates are configurable assumptions; real costs depend on the broker and change over time.
- Fundamentals are quarterly and slow-moving; they are compared with the forecast here, not used to predict
  tomorrow's open.
"""


def render(preds: pd.DataFrame, story: dict) -> None:
    st.markdown("### Forecast direction vs company fundamentals")
    table = fundamental_check.compare(preds, pd.Timestamp(story["next_open"]))
    show = table.rename(columns={"symbol": "Stock", "forecast_gap_pct": "Forecast gap %",
                                 "forecast_direction": "Forecast direction", "trend": "Fundamental trend",
                                 "quarter": "Latest quarter", "revenue_yoy_pct": "Revenue YoY %",
                                 "profit_yoy_pct": "Net profit YoY %", "momentum": "Growth momentum",
                                 "assessment": "Assessment"})
    cols = ["Stock", "Forecast gap %", "Forecast direction", "Fundamental trend", "Latest quarter", "Revenue YoY %",
            "Net profit YoY %", "Growth momentum", "Assessment"]
    st.dataframe(show[cols].round(2), hide_index=True, use_container_width=True)
    flagged = table[table["assessment"].str.startswith("Disagree")]
    for r in flagged.itertuples():
        st.warning(f"**{r.symbol}**: {r.assessment}")
    st.caption(fundamental_check.__doc__.split("Fundamental trend")[0].strip())
    st.markdown("### Executive summary")
    summary.render(compact=False)
    st.markdown(LIMITS)
