"""Institutional activity page: reported FII/DII flows, delivery %, bulk and block deals.

These are figures NSE reports after the session. They are kept apart from the Market flow page's
estimated buyer/seller pressure, and each block names its source and date. If NSE refuses the server's
request, the block says so and offers an upload of the CSV that NSE's website lets anyone download.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from capstone import charts, institutional as inst, market_flow
from capstone.charts import CONFIG
from capstone.ui.common import DOWN, UP

HOW_TO = {
    "fii_dii": "Open nseindia.com/reports/fii-dii and click **Download (.csv)** under *FII/FPI & DII trading activity "
               "on NSE in Capital Market Segment*.",
    "deals": "Open nseindia.com → Market Data → Large Deals, and download the **bulk** and **block** deals CSV "
             "(file names containing 'bulk' or 'block').",
    "delivery": "Download nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv for the day "
                "(e.g. sec_bhavdata_full_23112026.csv).",
}


def _status(name: str, label: str) -> pd.DataFrame:
    """Try NSE (at most every 30 min after a failure), then show source/date or the unavailable note + upload."""
    with st.spinner(f"Checking NSE for {label}..."):
        res = inst.refresh(name)
    df = inst.load(name)
    if df.empty or not res["ok"]:
        msg = (f"Could not get {label} from NSE just now ({res['error'] or 'no data yet'}). NSE's website often "
               "refuses requests from cloud servers.")
        (st.warning if df.empty else st.caption)(msg + (" Showing the last saved copy." if len(df) else ""))
        with st.expander(f"Upload NSE's {label} CSV instead", expanded=df.empty):
            st.markdown(HOW_TO[name])
            files = st.file_uploader("NSE CSV", type=["csv"], key=f"up_{name}",
                                     accept_multiple_files=(name == "deals"))
            for f in ([files] if files and not isinstance(files, list) else files or []):
                try:
                    df = inst.store_upload(name, f)
                    st.success(f"Saved {f.name}.")
                except ValueError as e:
                    st.error(str(e))
    if len(df):
        st.caption(f"Source: {inst.SOURCES[name]} · latest session {pd.Timestamp(df['date'].max()):%d %b %Y} · "
                   f"{df['date'].nunique()} session(s) saved. Reported by NSE, not estimated.")
    return df


def _fii_chart(df: pd.DataFrame) -> go.Figure:
    w = df.pivot_table(index="date", columns="category", values="net_cr", aggfunc="last").sort_index().tail(30)
    fig = go.Figure()
    for c, col in (("FII/FPI", "#6a5cff"), ("DII", "#f2b84b")):
        if c in w:
            fig.add_trace(go.Bar(x=w.index.strftime("%d %b"), y=w[c], name=f"{c} net", marker_color=col,
                                 hovertemplate=f"{c} net ₹%{{y:,.0f}} cr<extra></extra>"))
    charts.style(fig, height=340)
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    fig.update_yaxes(title="Net buy (+) / sell (−), ₹ crore")
    return fig


def _fii(df: pd.DataFrame) -> None:
    s = inst.fii_summary(df)
    if not s:
        return
    c = st.columns(4)
    for i, cat in enumerate(("FII/FPI", "DII")):
        if cat in s:
            v = s[cat]
            c[2 * i].metric(f"{cat} net, {s['as_of']:%d %b}", f"₹{v['net']:+,.0f} cr",
                            help="Buy value minus sell value in NSE's cash market (provisional).")
            c[2 * i + 1].metric(f"{cat} net, last 5 sessions", f"₹{v['sum5']:+,.0f} cr" if v["sum5"] is not None else "n/a",
                                help="Needs 5 saved sessions; history builds up a day at a time.")
    if s["days"] > 1:
        st.plotly_chart(_fii_chart(df), use_container_width=True, config=CONFIG)
    last = df[df["date"] == df["date"].max()]
    st.dataframe(last[["category", "buy_cr", "sell_cr", "net_cr"]], hide_index=True, use_container_width=True,
                 column_config={"category": "Category", "buy_cr": st.column_config.NumberColumn("Buy ₹ cr", format="%.2f"),
                                "sell_cr": st.column_config.NumberColumn("Sell ₹ cr", format="%.2f"),
                                "net_cr": st.column_config.NumberColumn("Net ₹ cr", format="%+.2f")})


def _deals(df: pd.DataFrame, uni: pd.DataFrame) -> None:
    last = df[df["date"] == df["date"].max()].copy()
    only = st.toggle("Only stocks in the chosen universe", value=False, key="deal_uni")
    if only:
        last = last[last["symbol"].isin(uni["symbol"])]
    by = last.groupby(["kind", "side"])["value_cr"].sum()
    c = st.columns(4)
    for i, (k, s) in enumerate([("bulk", "Buy"), ("bulk", "Sell"), ("block", "Buy"), ("block", "Sell")]):
        c[i].metric(f"{k.title()} deals, {s.lower()}", f"₹{by.get((k, s), 0):,.0f} cr")
    net = last.assign(sgn=last["side"].map({"Buy": 1, "Sell": -1}) * last["value_cr"]).groupby("symbol")["sgn"].sum()
    top = net.reindex(net.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    if len(top):
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=[UP if v >= 0 else DOWN for v in top.values],
                               hovertemplate="%{y}: net ₹%{x:,.1f} cr<extra></extra>"))
        charts.style(fig, height=max(300, 24 * len(top) + 60), title="Net deal value by stock (buy − sell)", hover="closest")
        st.plotly_chart(fig, use_container_width=True, config=CONFIG)
    st.dataframe(last.sort_values("value_cr", ascending=False)[["kind", "symbol", "name", "client", "side", "qty", "price",
                                                                "value_cr"]],
                 hide_index=True, use_container_width=True, column_config={
                     "kind": "Type", "symbol": "Symbol", "name": "Security", "client": "Client", "side": "Buy/Sell",
                     "qty": st.column_config.NumberColumn("Quantity", format="%d"),
                     "price": st.column_config.NumberColumn("Price ₹", format="%.2f"),
                     "value_cr": st.column_config.NumberColumn("Value ₹ cr", format="%.2f")})
    st.caption("Bulk deal: one client trades more than 0.5% of a company's shares in a day. Block deal: a single "
               "trade of at least ₹10 crore in NSE's block-deal window. A buyer and a seller appear for each.")


def _delivery(df: pd.DataFrame, uni: pd.DataFrame, uni_name: str) -> None:
    v = inst.delivery_view(df, uni)
    if v.empty:
        st.info(f"No delivery data for {uni_name} stocks in the saved files.")
        return
    c = st.columns(3)
    c[0].metric("Median delivery %", f"{v['deliv_pct'].median():.1f}%",
                help="Share of traded shares taken into demat (not squared off the same day).")
    c[1].metric("Delivery value", f"₹{v['deliv_value_cr'].sum():,.0f} cr", help="Delivered quantity × close.")
    hi = v[v["deliv_vs_avg"] >= 15]
    c[2].metric("Delivery spikes", f"{len(hi)}" if v["deliv_pct_avg"].notna().any() else "n/a",
                help="Stocks whose delivery % is 15+ points above their own average (needs 5+ saved sessions).")
    st.dataframe(v.sort_values("deliv_pct", ascending=False)[["symbol", "name", "sector", "close", "volume", "deliv_qty",
                                                              "deliv_pct", "deliv_pct_avg", "deliv_value_cr"]],
                 hide_index=True, use_container_width=True, column_config={
                     "symbol": "Symbol", "name": "Company", "sector": "Sector",
                     "close": st.column_config.NumberColumn("Close ₹", format="%.2f"),
                     "volume": st.column_config.NumberColumn("Volume", format="%d"),
                     "deliv_qty": st.column_config.NumberColumn("Delivery volume", format="%d"),
                     "deliv_pct": st.column_config.ProgressColumn("Delivery %", format="%.1f%%", min_value=0, max_value=100),
                     "deliv_pct_avg": st.column_config.NumberColumn("Own avg %", format="%.1f%%"),
                     "deliv_value_cr": st.column_config.NumberColumn("Delivery ₹ cr", format="%.1f")})


def render() -> None:
    st.caption("Reported by NSE after each session: who bought and sold (FII/FPI vs domestic institutions), large "
               "deals, and how much traded volume was actually delivered. These are not the buyer/seller **estimates** "
               "on Market flow. NSE publishes them in the evening (about 17:00–19:00 IST).")
    uni_name = st.selectbox("Universe", list(market_flow.UNIVERSES), index=2, key="inst_uni")
    uni = market_flow.universe(uni_name)
    t1, t2, t3 = st.tabs(["FII / DII flows", "Bulk and block deals", "Delivery %"])
    with t1:
        df = _status("fii_dii", "FII/DII flows")
        if len(df):
            _fii(df)
    with t2:
        df = _status("deals", "bulk and block deals")
        if len(df):
            _deals(df, uni)
    with t3:
        df = _status("delivery", "delivery data")
        if len(df):
            _delivery(df, uni, uni_name)
    st.caption("Promoter transactions (insider trading disclosures) and quarterly shareholding changes are not "
               "included yet: NSE publishes them only through its website search, with no free feed this app can rely on.")
