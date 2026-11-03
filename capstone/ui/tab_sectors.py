"""US sectors tab: overnight US sector moves and how they map to Indian sectors."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone.explain import move
from capstone.paths import SECTOR_LINKS_FILE
from capstone.ui.common import signed_bars


def sector_map(links: pd.DataFrame, moves: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """One row per US-India sector link with last night's US move and the user's stocks in it."""
    held = preds.groupby("india_sector")["symbol"].apply(", ".join)
    m = links.assign(us_move=links["us_etf"].map(moves["ret"]))
    m["Your stocks"] = m["india_sector"].map(held).fillna("")
    m["What happened"] = m["us_move"].map(move)
    m = m.sort_values("Your stocks", ascending=False, key=lambda s: s.str.len())
    return (m[["india_sector", "us_sector", "us_etf", "What happened", "Your stocks", "why_us"]]
            .rename(columns={"india_sector": "Indian sector", "us_sector": "US sector",
                             "us_etf": "ETF", "why_us": "Why it matters"}))


def render(preds: pd.DataFrame, moves: pd.DataFrame) -> None:
    sectors = moves[moves["is_sector"]].reset_index()
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**US sector moves overnight**")
        signed_bars(sectors, "name", "ret", "Move (%)")
    with right:
        st.markdown("**How each US sector maps to India**")
        st.dataframe(sector_map(pd.read_csv(SECTOR_LINKS_FILE), moves, preds),
                     use_container_width=True, hide_index=True)
    adrs = preds.dropna(subset=["adr"])
    if len(adrs):
        st.markdown("**Indian companies listed in the US (ADRs)**")
        st.caption("An ADR trades in New York while India is closed, so it is often the clearest "
                   "overnight hint for that stock.")
        st.dataframe(adrs[["symbol", "adr", "adr_ret"]].assign(adr_ret=adrs["adr_ret"].map(move))
                     .rename(columns={"symbol": "NSE stock", "adr": "US ticker", "adr_ret": "Overnight"}),
                     hide_index=True)
