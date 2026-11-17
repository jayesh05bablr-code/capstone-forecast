"""Stocks tab: the forecast table and a breakdown of one stock."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import charts
from capstone.explain import DRIVERS, OTHER
from capstone.ui.common import raw_prices, signed_bars

TABLE_COLUMNS = {"symbol": "Stock", "india_sector": "Sector", "close": "Last close (₹)",
                 "pred_open": "Expected open (₹)", "pred_gap_pct": "Gap %",
                 "pred_open_low": "Low (₹)", "pred_open_high": "High (₹)"}


def driver_effects(row: pd.Series) -> pd.DataFrame:
    """How many percentage points each driver added to one stock's expected gap."""
    names = [*DRIVERS, OTHER]
    return pd.DataFrame({"driver": names, "effect": [row.get(f"contrib_{d}", 0.0) for d in names]})


def render(preds: pd.DataFrame, story: dict) -> str:
    """Draw the tab and return the stock the user picked (other tabs follow it)."""
    table = preds[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS)
    st.dataframe(table.sort_values("Gap %", ascending=False), use_container_width=True, hide_index=True)
    st.caption("Low to high is the 10th to 90th percentile of recent out-of-sample errors.")

    symbol = st.selectbox("Explain one stock", preds["symbol"])
    row = preds.set_index("symbol").loc[symbol]
    st.info(row["reason"])
    left, right = st.columns(2)
    with left:
        st.markdown("**What moved the forecast** (percentage points of the gap)")
        signed_bars(driver_effects(row), "driver", "effect", "Effect on expected gap (%)")
        st.caption("What-if estimate: the forecast with that driver set to no move, compared with "
                   "the real forecast. It shows what the model reacted to, not proven cause.")
    with right:
        st.markdown("**The forecast in numbers**")
        st.dataframe(pd.DataFrame({
            "": ["Last close", "Expected open", "Likely low", "Likely high", "Expected gap"],
            symbol: [f"₹{row['close']:,.2f}", f"₹{row['pred_open']:,.2f}", f"₹{row['pred_open_low']:,.2f}",
                     f"₹{row['pred_open_high']:,.2f}", f"{row['pred_gap_pct']:+.2f}%"]}), hide_index=True)
    hist = raw_prices(symbol)
    if hist is not None:
        fc = {"date": pd.Timestamp(story["next_open"]), "open": row["pred_open"],
              "low": row["pred_open_low"], "high": row["pred_open_high"]}
        st.plotly_chart(charts.candles(hist, f"{symbol}: daily candles with the expected open (◆)",
                                       forecast=fc), use_container_width=True)
        st.caption("Green candle = closed above its open, red = below. Lines are 20- and 50-day "
                   "moving averages; bars underneath are volume. Use the 1M/3M/6M/1Y buttons or drag to zoom.")
    return symbol
