"""Rupee tab: USD/INR history and each sector's exposure to the rupee."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import charts, overnight
from capstone.paths import SECTOR_LINKS_FILE
from capstone.charts import CONFIG as CHART_CONFIG
from capstone.indicators import PERIODS


def fx_effect(exposure: float) -> str:
    """What a weaker rupee means for a sector with this exposure (+ exporter, - importer)."""
    return "Good" if exposure > 0 else "Bad" if exposure < 0 else "Neutral"


def render() -> None:
    fx = overnight.load_fx().rename("USD/INR").reset_index()
    if fx.empty:
        st.info("No USD/INR data yet. Run fetch_overnight.")
    else:
        period = st.segmented_control("Period", list(PERIODS), default="6M", key="fx_period") or "6M"
        st.plotly_chart(charts.line(fx, "date", {"USD/INR": "USD/INR"},
                                    "Rupees per US dollar (up means the rupee weakened)", "₹ per $",
                                    period=period, prefix="₹"), use_container_width=True, config=CHART_CONFIG)
    st.markdown(
        "**How the rupee affects the open.** When the rupee weakens, exporters that earn dollars "
        "(IT, pharma, metals) earn more in rupees, while importers (oil, autos, FMCG inputs) pay "
        "more. A falling rupee can also push foreign investors to sell Indian shares. A stronger "
        "rupee does the opposite. The model multiplies the rupee move by each sector's exposure "
        "below to get the rupee effect for each stock.")
    exp = pd.read_csv(SECTOR_LINKS_FILE)[["india_sector", "fx_exposure", "why_fx"]]
    exp["Weaker rupee is"] = exp["fx_exposure"].map(fx_effect)
    st.dataframe(exp.rename(columns={"india_sector": "Indian sector", "fx_exposure": "Exposure",
                                     "why_fx": "Why"}), use_container_width=True, hide_index=True)
    st.caption("Exposures are editable in data/sector_links.csv (+1 strong exporter, -1 strong importer).")
