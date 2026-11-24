"""Tab 1, Morning market forecast: today's gap forecast per stock with its 10-90% interval."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone.paths import PROCESSED
from capstone.ui import data_cache, tab_stocks

MODEL_USED = "HistGradientBoosting (gradient-boosted trees, median loss), trained on all history"


def last_updated(preds: pd.DataFrame) -> str:
    """When this forecast was made, in IST."""
    if "predicted_at_utc" in preds.columns and preds["predicted_at_utc"].notna().any():
        t = pd.Timestamp(preds["predicted_at_utc"].dropna().iloc[0], tz="UTC")
    else:
        t = pd.Timestamp((PROCESSED / "predictions.csv").stat().st_mtime, unit="s", tz="UTC")
    return t.tz_convert("Asia/Kolkata").strftime("%d %b %Y, %H:%M IST")


def direction(gap_pct: float, flat: float = 0.05) -> str:
    return "Up (gap-up open)" if gap_pct > flat else "Down (gap-down open)" if gap_pct < -flat else "Flat"


def render(preds: pd.DataFrame, story: dict) -> str:
    symbol = st.selectbox("Stock", preds["symbol"], key="forecast_symbol")
    row = preds.set_index("symbol").loc[symbol]
    gap = row["pred_gap_pct"]
    q10 = row["q10_gap_pct"] if "q10_gap_pct" in row and pd.notna(row.get("q10_gap_pct")) else \
        (row["pred_open_low"] / row["close"] - 1) * 100
    q90 = row["q90_gap_pct"] if "q90_gap_pct" in row and pd.notna(row.get("q90_gap_pct")) else \
        (row["pred_open_high"] / row["close"] - 1) * 100

    st.markdown(f"#### {symbol}: forecast for the NSE open on {pd.Timestamp(story['next_open']):%a %d %b %Y}")
    c = st.columns(4)
    c[0].metric("Predicted opening gap", f"{gap:+.2f}%")
    c[1].metric("Predicted opening price", f"₹{row['pred_open']:,.2f}", f"from close ₹{row['close']:,.2f}",
                delta_color="off")
    c[2].metric("10th percentile", f"₹{row['pred_open_low']:,.2f}", f"{q10:+.2f}%", delta_color="off")
    c[3].metric("90th percentile", f"₹{row['pred_open_high']:,.2f}", f"{q90:+.2f}%", delta_color="off")
    c = st.columns(3)
    c[0].metric("Expected direction", direction(gap))
    c[1].markdown(f"**Model used**  \n{MODEL_USED}")
    c[2].markdown(f"**Last updated**  \n{last_updated(preds)}")

    cov = data_cache.models_csv("interval_coverage.csv")
    note = ""
    if cov is not None and len(cov):
        hgb = cov[cov["model"] == ("forecast" if (cov["model"] == "forecast").any() else "hist_gradient_boosting")]
        note = (f" In walk-forward testing the actual gap fell inside such a band "
                f"{hgb['coverage_pct'].min():.0f}% to {hgb['coverage_pct'].max():.0f}% of the time per test period "
                f"(target 80%).")
    st.info("**About the 10th-90th percentile range.** It comes from the model's own out-of-sample errors "
            "(residuals on data it was not trained on): about 8 opens in 10 should land inside it." + note +
            " It is a statistical estimate, not a guaranteed price boundary: earnings, announcements, global "
            "shocks and thin liquidity can push the open well outside it.")

    st.markdown("**All stocks and the drivers of this forecast**")
    return tab_stocks.render(preds, story, symbol=symbol)
