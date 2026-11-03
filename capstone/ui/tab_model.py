"""Model accuracy tab: walk-forward validation and past forecasts vs actual opens."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import charts, evaluation
from capstone.paths import MODELS, PROCESSED
from capstone.ui.common import raw_prices


def render(symbol: str) -> None:
    """symbol: the stock whose past forecasts are charted (picked on the Stocks tab)."""
    report = MODELS / "walk_forward_report.csv"
    if report.exists():
        summary = evaluation.accuracy_summary(pd.read_csv(report))
        st.markdown("**Walk-forward validation** (average over folds, lower error is better)")
        st.dataframe(summary)
        gain = evaluation.overnight_gain(summary)
        if gain is not None:
            st.caption(f"Adding the overnight US and rupee features {'lowers' if gain > 0 else 'raises'} "
                       f"the average error by ₹{abs(gain):.2f} per share.")

    log_path = PROCESSED / "predictions_log.csv"
    hist = raw_prices(symbol)
    if log_path.exists() and hist is not None:
        joined = evaluation.predictions_vs_actual(pd.read_csv(log_path), hist, symbol)
        if len(joined):
            st.plotly_chart(charts.line(joined, "as_of_close", {"pred_open": "Predicted open",
                                                                "actual_open": "Actual open"},
                                        f"{symbol}: past predictions vs actual open", "₹"),
                            use_container_width=True)
