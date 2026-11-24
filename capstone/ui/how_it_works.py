"""How it works: one diagram of the whole platform, for first-time visitors and presentations."""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

DIAGRAM = Path(__file__).resolve().parents[2] / "docs" / "pipeline_diagram.png"

STEPS = [
    ("1. Data comes in", "Official NSE prices (daily Bhavcopy and near-live quotes) with Yahoo Finance as a backup, "
                         "last night's US market and the rupee (Twelve Data, Alpha Vantage as backup), and company "
                         "fundamentals from Excel files."),
    ("2. The model forecasts", "Python turns that data into features and a scikit-learn model predicts each stock's "
                               "next-day opening price with a likely low-high range. It retrains when the data changes."),
    ("3. You see the result", "This app shows the forecast on the Today page, plus live prices, market flow, "
                              "an AI chat about the market (Groq) that looks up NSE data, and backtests. Supabase keeps every forecast next to the "
                              "actual open, so accuracy is checked honestly."),
    ("4. How the app is built", "Code is written in VS Code, reviewed and tested on GitHub, and every merge "
                                "redeploys this app on Streamlit Cloud. API keys stay in Streamlit Secrets."),
]


@lru_cache(maxsize=2)
def _diagram_html(large: bool = False) -> str:
    """The diagram as an inline <img>, sent byte for byte.

    st.image shrinks anything wider than ~1460 px and re-saves it as JPEG, which smeared the small text.
    The file is 3840 px wide, so it stays sharp even on high-DPI screens and when zoomed.
    """
    data = base64.b64encode(DIAGRAM.read_bytes()).decode()
    size = "width:1700px;max-width:none" if large else "width:100%"
    return (f'<div style="overflow-x:auto"><img src="data:image/png;base64,{data}" alt="How the platform works: data sources feed the model, '
            'the model feeds this app, and code flows from VS Code through GitHub to Streamlit Cloud" '
            f'style="{size};height:auto;border-radius:8px;"></div>')


def render() -> None:
    st.subheader("How this app works")
    st.caption("From raw market data to the forecast you see, and how the app itself is built and kept online.")
    if DIAGRAM.exists():
        large = st.toggle("Larger view (scroll sideways)", key="diagram_large",
                          help="Shows the diagram bigger so the small text is easier to read on a laptop.")
        st.markdown(_diagram_html(large), unsafe_allow_html=True)
        st.download_button("Download full-size diagram (PNG)", DIAGRAM.read_bytes(),
                           file_name="pipeline_diagram.png", mime="image/png")
    cols = st.columns(len(STEPS))
    for col, (title, text) in zip(cols, STEPS):
        with col:
            st.markdown(f"**{title}**")
            st.caption(text)
    st.caption("Snowflake (dotted box) is future scope only: a data warehouse for covering all NSE stocks. "
               "It is not used today.")
