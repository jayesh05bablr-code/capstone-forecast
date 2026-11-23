"""How it works: one diagram of the whole platform, for first-time visitors and presentations."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

DIAGRAM = Path(__file__).resolve().parents[2] / "docs" / "pipeline_diagram.png"

STEPS = [
    ("1. Data comes in", "Official NSE prices (daily Bhavcopy and near-live quotes), last night's US market and "
                         "the rupee (Twelve Data, Alpha Vantage as backup), and company fundamentals from Excel files."),
    ("2. The model forecasts", "Python turns that data into features and a scikit-learn model predicts each stock's "
                               "next-day opening price with a likely low-high range. It retrains when the data changes."),
    ("3. You see the result", "This app shows the forecast on the Today page, plus live prices, market flow, "
                              "an AI chat about the market, and backtests. Supabase keeps every forecast next to the "
                              "actual open, so accuracy is checked honestly."),
    ("4. How the app is built", "Code is written in VS Code, reviewed and tested on GitHub, and every merge "
                                "redeploys this app on Streamlit Cloud. API keys stay in Streamlit Secrets."),
]


def render() -> None:
    st.subheader("How this app works")
    st.caption("From raw market data to the forecast you see, and how the app itself is built and kept online.")
    if DIAGRAM.exists():
        st.image(str(DIAGRAM), width="stretch")
    cols = st.columns(len(STEPS))
    for col, (title, text) in zip(cols, STEPS):
        with col:
            st.markdown(f"**{title}**")
            st.caption(text)
    st.caption("Snowflake (dotted box) is future scope only: a data warehouse for covering all NSE stocks. "
               "It is not used today.")
