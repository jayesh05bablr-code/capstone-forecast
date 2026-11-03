"""Helpers shared by several dashboard tabs."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from capstone import refresh
from capstone.paths import RAW

UP, DOWN = "#2a78d6", "#e34948"  # diverging pair: blue = up, red = down
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def auto_update(why: str) -> None:
    """Retrain + re-predict in the background process, then reload the page with fresh numbers."""
    with st.spinner(f"{why}: retraining the model and updating predictions (this can take a minute)..."):
        ok, log = refresh.retrain_and_predict()
    st.cache_data.clear()
    st.session_state["update_result"] = (ok, log)
    st.rerun()


@st.cache_data
def load_raw(sym: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{sym}.csv", parse_dates=["date"])


def raw_prices(sym: str) -> pd.DataFrame | None:
    """Daily prices for one stock, cached until its CSV changes; None if never fetched."""
    path = RAW / f"{sym}.csv"
    return load_raw(sym, path.stat().st_mtime) if path.exists() else None


def signed_bars(df: pd.DataFrame, label: str, value: str, title: str, height: int | None = None):
    """Horizontal bars from zero, blue for up and red for down, with the % written on each."""
    df = df.assign(pct=df[value] * 100, direction=(df[value] >= 0).map({True: "Up", False: "Down"}))
    base = alt.Chart(df).encode(
        y=alt.Y(f"{label}:N", sort=list(df.sort_values("pct", ascending=False)[label]), title=None,
                axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip("pct:Q", title="Move %", format="+.2f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=14).encode(
        x=alt.X("pct:Q", title=title),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Up", "Down"], range=[UP, DOWN]),
                        legend=None),
    )
    text = alt.Text("pct:Q", format="+.2f")
    right = base.mark_text(align="left", dx=4).encode(x="pct:Q", text=text).transform_filter("datum.pct >= 0")
    left = base.mark_text(align="right", dx=-4).encode(x="pct:Q", text=text).transform_filter("datum.pct < 0")
    chart = alt.layer(bars, right, left).properties(height=height or 34 * len(df) + 30)
    st.altair_chart(chart, use_container_width=True)
