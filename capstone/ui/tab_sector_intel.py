"""Sector intelligence page: sector leaderboard, rotation map, return heat map and sector breadth.

Official numbers (returns, relative strength, momentum, index volume) come from NSE's published sector
index values. Breadth, relative volume and contribution are our own equal-weighted sector groups of the
chosen universe, from the saved daily bhavcopy; the page labels which is which.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from capstone import charts, flow_history, market_flow, sector_intel
from capstone.charts import CONFIG
from capstone.ui.common import DOWN, UP

QUAD_COLOR = {"Leading": "#2a9d6f", "Weakening": "#e0a030", "Lagging": "#e34948", "Improving": "#2a78d6"}


def _short(name: str) -> str:
    """'NIFTY PSU BANK' -> 'PSU Bank'."""
    keep = {"IT", "PSU", "FMCG", "PSE", "&"}
    words = name.replace("NIFTY ", "", 1).replace(" INDEX", "").split()
    return " ".join(w if w in keep else w.title() for w in words)


@st.cache_data(ttl=600, show_spinner=False)
def _load(mtime: float) -> pd.DataFrame:
    return sector_intel.load_indices()


def _indices() -> pd.DataFrame:
    sector_intel.refresh_indices()
    f = sector_intel.INDEX_FILE
    if not f.exists():
        return pd.DataFrame()
    return _load(f.stat().st_mtime)


def rotation_chart(trails: pd.DataFrame, lb: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trails.empty:
        return fig
    span_x = max(3.0, float(trails["rs"].abs().max()) * 1.15)
    span_y = max(2.0, float(trails["mom"].abs().max()) * 1.15)
    for (x0, x1, y0, y1, q) in [(0, span_x, 0, span_y, "Leading"), (0, span_x, -span_y, 0, "Weakening"),
                                (-span_x, 0, -span_y, 0, "Lagging"), (-span_x, 0, 0, span_y, "Improving")]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=QUAD_COLOR[q], opacity=0.07, line_width=0,
                      layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=y1 * 0.94 if y1 > 0 else y0 * 0.94, text=f"<b>{q}</b>", showarrow=False,
                           font=dict(color=QUAD_COLOR[q], size=13), opacity=0.85)
    quad = lb.set_index("index")["quadrant"].to_dict()
    for name, t in trails.groupby("index"):
        col = QUAD_COLOR.get(quad.get(name), "#8b93a3")
        fig.add_trace(go.Scatter(x=t["rs"], y=t["mom"], mode="lines", line=dict(color=col, width=1.4), opacity=0.45,
                                 hoverinfo="skip", showlegend=False))
        last = t.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last["rs"]], y=[last["mom"]], mode="markers+text", text=[_short(name)], textposition="top center",
            marker=dict(size=11, color=col, line=dict(color="white", width=1)), showlegend=False,
            hovertemplate=f"<b>{name}</b><br>Relative strength (20d): %{{x:+.2f}} pts<br>"
                          f"RS momentum (5d): %{{y:+.2f}} pts<br>{quad.get(name, '')}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#8b93a3", width=1))
    fig.add_vline(x=0, line=dict(color="#8b93a3", width=1))
    charts.style(fig, height=560, hover="closest")
    fig.update_xaxes(title="Relative strength vs NIFTY 50, last 20 sessions (pts)", range=[-span_x, span_x],
                     zeroline=False, showspikes=False)
    fig.update_yaxes(title="Change in relative strength, last 5 sessions (pts)", range=[-span_y, span_y],
                     showspikes=False)
    return fig


def returns_heatmap(lb: pd.DataFrame) -> go.Figure:
    cols = {"ret_1d": "1 day", "ret_5d": "5 days", "ret_20d": "20 days", "ret_60d": "60 days", "rs_20d": "RS vs NIFTY 50 (20d)"}
    d = lb.set_index("index")[list(cols)].iloc[::-1]
    z = d.to_numpy(dtype=float)
    lim = float(np.nanpercentile(np.abs(z), 90)) or 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=list(cols.values()), y=[_short(i) for i in d.index], zmin=-lim, zmax=lim, zmid=0,
        colorscale=[[0, DOWN], [0.5, "#f0efe9"], [1, UP]], text=[[f"{v:+.1f}%" if pd.notna(v) else "" for v in r] for r in z],
        texttemplate="%{text}", hovertemplate="%{y}, %{x}: %{z:+.2f}<extra></extra>",
        colorbar=dict(title="%", thickness=10)))
    charts.style(fig, height=max(380, 26 * len(d) + 80), hover="closest")
    fig.update_xaxes(side="top", showspikes=False, showgrid=False)
    fig.update_yaxes(showspikes=False, showgrid=False)
    return fig


def _pct_cfg(label, help_=None):
    return st.column_config.NumberColumn(label, format="%+.2f%%", help=help_)


def _leaderboard(lb: pd.DataFrame) -> None:
    show = lb.assign(sector=lb["index"].map(_short))[
        ["sector", "close", "ret_1d", "ret_5d", "ret_20d", "rs_20d", "rs_mom", "quadrant", "trend", "rvol"]]
    st.dataframe(show, hide_index=True, use_container_width=True, column_config={
        "sector": st.column_config.TextColumn("Sector index"),
        "close": st.column_config.NumberColumn("Close", format="%.2f"),
        "ret_1d": _pct_cfg("1D"), "ret_5d": _pct_cfg("5D momentum"), "ret_20d": _pct_cfg("20D momentum"),
        "rs_20d": st.column_config.NumberColumn("RS vs NIFTY 50", format="%+.2f pts",
                                                help="20-session return minus the NIFTY 50's 20-session return."),
        "rs_mom": st.column_config.NumberColumn("RS change (5d)", format="%+.2f pts",
                                                help="Relative strength now minus 5 sessions ago."),
        "quadrant": st.column_config.TextColumn("Rotation"),
        "trend": st.column_config.TextColumn("Relative strength is"),
        "rvol": st.column_config.NumberColumn("Rel. volume", format="%.2f×",
                                              help="Index volume, last 5 sessions vs the 20 before them."),
    })


def _ours(uni_name: str) -> None:
    uni = market_flow.universe(uni_name)
    hist = flow_history.load()
    flow_history.refresh(list(uni["symbol"]))
    p = flow_history.progress()
    if p["busy"]:
        st.info(f"Downloading daily history from NSE's bhavcopy: {p['done']} of {p['total']} stocks.")
    cs = sector_intel.constituent_stats(hist, uni)
    if cs.empty:
        st.info("No daily history saved yet for this universe; it is downloading in the background.")
        return
    avg_1d = cs["contrib_1d"].sum()
    st.caption(f"{int(cs['stocks'].sum())} of {len(uni)} {uni_name} stocks, session of {cs['as_of'].iloc[0]:%d %b %Y}. "
               f"Equal-weighted: the average stock moved {avg_1d:+.2f}% and the bars split that move by sector "
               "(free NSE feeds do not publish index weights, so this is not the index's own attribution).")
    c1, c2 = st.columns(2)
    for col, k, lab in [(c1, "contrib_1d", "Contribution to the average 1-day move"),
                        (c2, "contrib_5d", "Contribution to the average 5-day move")]:
        d = cs.sort_values(k)
        fig = go.Figure(go.Bar(x=d[k], y=d["sector"], orientation="h", marker_color=[UP if v >= 0 else DOWN for v in d[k]],
                               text=[f"{v:+.2f}" for v in d[k]], textposition="outside",
                               hovertemplate="%{y}: %{x:+.3f} pts<extra></extra>", cliponaxis=False))
        charts.style(fig, height=max(320, 24 * len(d) + 60), title=lab, hover="closest")
        lo, hi = float(min(d[k].min(), 0)), float(max(d[k].max(), 0))
        pad = 0.3 * (hi - lo or 1)
        fig.update_xaxes(range=[lo - pad, hi + pad])
        col.plotly_chart(fig, use_container_width=True, config=CONFIG)
    st.dataframe(cs.drop(columns=["as_of"]), hide_index=True, use_container_width=True, column_config={
        "sector": "Sector (our grouping)", "stocks": "Stocks",
        "up_pct": st.column_config.ProgressColumn("Stocks up today", format="%.0f%%", min_value=0, max_value=100),
        "ret_1d": _pct_cfg("Avg 1D"), "ret_5d": _pct_cfg("Avg 5D"),
        "above_20_pct": st.column_config.ProgressColumn("Above 20-day avg", format="%.0f%%", min_value=0, max_value=100),
        "above_50_pct": st.column_config.ProgressColumn("Above 50-day avg", format="%.0f%%", min_value=0, max_value=100),
        "rvol": st.column_config.NumberColumn("Median rel. volume", format="%.2f×",
                                              help="Each stock's last-5-session volume vs the 20 sessions before."),
        "contrib_1d": st.column_config.NumberColumn("Contribution 1D", format="%+.3f pts"),
        "contrib_5d": st.column_config.NumberColumn("Contribution 5D", format="%+.3f pts")})


def render() -> None:
    ix = _indices()
    s = sector_intel.status()
    if ix.empty:
        if s["busy"]:
            st.info("Downloading NSE sector index history (about 10 seconds). This page fills in when it is done.")
        else:
            st.warning("NSE's sector index history is not available right now"
                       + (f" ({s['error']})." if s["error"] else ".") + " Try again in a few minutes.")
        if st.button("Check again"):
            st.rerun()
        return
    lb = sector_intel.leaderboard(ix)
    if lb.empty:
        st.warning("Not enough sector index history yet to compare sectors.")
        return
    as_of = pd.Timestamp(lb["as_of"].max())
    bench = ix[ix["index"] == sector_intel.BENCH].sort_values("date")["close"]
    n20 = 100 * (bench.iloc[-1] / bench.iloc[-21] - 1) if len(bench) > 21 else None
    c = st.columns(5)
    c[0].metric("Leading sectors", f"{(lb['quadrant'] == 'Leading').sum()} of {len(lb)}",
                help="Stronger than the NIFTY 50 over 20 sessions and still gaining strength.")
    c[1].metric("Strongest vs NIFTY 50", _short(lb.iloc[0]["index"]), f"{lb.iloc[0]['rs_20d']:+.2f} pts")
    c[2].metric("Weakest vs NIFTY 50", _short(lb.iloc[-1]["index"]), f"{lb.iloc[-1]['rs_20d']:+.2f} pts")
    g = lb.sort_values("rs_mom")
    c[3].metric("Gaining strength fastest", _short(g.iloc[-1]["index"]), f"{g.iloc[-1]['rs_mom']:+.2f} pts (5d)")
    c[4].metric("NIFTY 50, 20 sessions", f"{n20:+.2f}%" if n20 is not None else "n/a")
    st.caption(f"Official NSE sector indices, closing values to {as_of:%d %b %Y} (NSE bhavcopy, mcp.nseindia.in). "
               "Refreshed a few times a day; sector index values are published after the close.")

    t1, t2, t3, t4 = st.tabs(["Leaderboard", "Rotation map", "Return heat map", "Sector breadth (our groups)"])
    with t1:
        _leaderboard(lb)
        gain = lb[lb["rs_mom"] > 0].sort_values("rs_mom", ascending=False)
        lose = lb[lb["rs_mom"] < 0].sort_values("rs_mom")
        a, b = st.columns(2)
        a.markdown("**Gaining relative strength:** " + (", ".join(f"{_short(i)} ({v:+.1f})" for i, v in
                                                                 zip(gain["index"], gain["rs_mom"])) or "none"))
        b.markdown("**Losing relative strength:** " + (", ".join(f"{_short(i)} ({v:+.1f})" for i, v in
                                                                zip(lose["index"], lose["rs_mom"])) or "none"))
    with t2:
        pts = st.slider("Trail length (sessions)", 1, 15, 4, key="rot_trail")
        picks = st.multiselect("Sectors", list(lb["index"]), default=list(lb["index"]), format_func=_short,
                               key="rot_pick")
        st.plotly_chart(rotation_chart(sector_intel.rotation_trails(ix, picks, pts), lb), use_container_width=True,
                        config=CONFIG)
        st.caption("Right of centre = stronger than the NIFTY 50 over 20 sessions; above centre = that edge grew over "
                   "the last 5 sessions. Sectors usually rotate clockwise: Improving → Leading → Weakening → Lagging. "
                   "Trails show where each sector was on earlier sessions.")
    with t3:
        st.plotly_chart(returns_heatmap(lb), use_container_width=True, config=CONFIG)
    with t4:
        uni_name = st.selectbox("Universe", list(market_flow.UNIVERSES), index=2, key="si_uni")
        _ours(uni_name)
