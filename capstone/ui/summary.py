"""Executive summary block (landing page expander and Business insights tab)."""
from __future__ import annotations

import streamlit as st

from capstone import summary


def render(compact: bool = True) -> None:
    text = summary.markdown()
    if compact:
        with st.expander("Executive summary: what this dashboard is for and how well it works", expanded=False):
            st.markdown(text)
    else:
        st.markdown(text)
