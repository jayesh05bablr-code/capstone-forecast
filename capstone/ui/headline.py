"""Top of the page: retrain results, the four headline numbers and the reasons."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def show_update_messages() -> None:
    """Show the outcome of a retrain started by an upload, then any saved-data messages."""
    result = st.session_state.pop("update_result", None)
    if result:
        ok, log = result
        if ok:
            st.success("Model retrained and predictions updated with the new company data.")
        else:
            st.error("Saved your data, but retraining failed. Details below (also in the terminal).")
        with st.expander("Training and prediction output"):
            st.code(log[-6000:] or "(no output)")
    for msg in st.session_state.pop("saved_msgs", []):
        st.success(msg)


def rupee_delta(fx_chg: float | None) -> str | None:
    """'Rupee 0.42% stronger' style label for a USD/INR change, or None if unknown."""
    if fx_chg is None or pd.isna(fx_chg):
        return None
    word = "weaker" if fx_chg > 0 else "stronger"
    return f"Rupee {abs(fx_chg) * 100:.2f}% {word}"


def render(preds: pd.DataFrame, story: dict, moves: pd.DataFrame) -> None:
    st.caption(f"Forecast for the NSE open on **{story['next_open']}**, using closes from "
               f"{story['as_of_close']} and the US session of {story['us_session'] or 'n/a'}. "
               "Run predict after about 02:30 IST so the US close is included.")
    if not story["overnight_complete"]:
        st.warning("The US session after India's last close is not in the data yet, so the overnight "
                   "signal is missing. Run fetch_overnight after the US close, then predict again.")

    avg_gap = preds["pred_gap_pct"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected opening", story["mood"], f"{avg_gap:+.2f}% average gap")
    c2.metric("S&P 500 overnight", f"{moves.loc['SPY', 'ret'] * 100:+.2f}%")
    c3.metric("Nasdaq 100 overnight", f"{moves.loc['QQQ', 'ret'] * 100:+.2f}%")
    if story.get("usdinr"):
        c4.metric("USD/INR", f"₹{story['usdinr']:.2f}", rupee_delta(story.get("usdinr_chg")),
                  delta_color="off")

    st.subheader("Why India should open this way")
    st.markdown("\n".join(f"- {line}" for line in story["lines"]))
