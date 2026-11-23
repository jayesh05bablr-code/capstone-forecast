"""Market flow page: heat map, sector flow and stock volume flow for the Nifty 50 and Next 50 stocks.

Kept light for Streamlit Community Cloud: at most one NSE download per minute for the whole server
(about 20 small calls made together, ~5 seconds, in the background, so visitors do not wait), only
the chosen chart is drawn, and auto-refresh is off unless the visitor turns it on (market hours
only, pauses after 30 minutes).
"""
from __future__ import annotations

import time
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from capstone import charts, live, market_flow
from capstone.charts import CONFIG as CHART_CONFIG
from capstone.ui.common import DOWN, UP

HUB = "Nifty 100 stocks"
NEUTRAL = "#b4b2a9"
SCALE = [[0, DOWN], [0.5, "#ecebe6"], [1, UP]]  # red = down, light grey = flat, blue = up
PCT_RANGE = 3.0  # colour saturates at ±3%, so one big mover does not wash out the rest


def _bucket() -> int:
    """Cache key: a new one every minute while NSE is open, every 30 minutes when it is closed."""
    return int(time.time() // (60 if live.market_open() else 1800))


@st.cache_data(ttl=1800, max_entries=2, show_spinner=False)
def _sector_indices(bucket: int) -> pd.DataFrame:
    """NSE's official sector indices (one call), for comparison with our own sector grouping."""
    try:
        from capstone import nse_mcp
        res = nse_mcp.call("market", "cm_get_live_indices", {"group": "sectoral", "nameFilter": ""}) or {}
        rows = [i for g in (res.get("indices") or {}).values() for i in g]
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty or "percChange" not in df:
        return pd.DataFrame()
    return df[["indexName", "last", "percChange"]].dropna().sort_values("percChange")


def _hex_rgba(color: str, alpha: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{alpha})"


def _pct_color(pct: float | None, alpha: float = 1.0) -> str:
    if pct is None or pd.isna(pct) or abs(pct) < 0.05:
        return _hex_rgba(NEUTRAL, alpha)
    return _hex_rgba(UP if pct > 0 else DOWN, alpha * (0.45 + 0.55 * min(abs(pct) / PCT_RANGE, 1)))


def heat_map(snap: pd.DataFrame) -> go.Figure:
    """Treemap: one box per stock inside its sector, size = traded value, colour = % change today.

    A sector's colour is the traded-value-weighted % change of its stocks, and so is the top box's.
    """
    def weighted(df):
        return float((df["pct_change"] * df["turnover_cr"]).sum() / df["turnover_cr"].sum())
    ids, labels, parents, values, pct, names = [HUB], [HUB], [""], [float(snap["turnover_cr"].sum())], \
        [weighted(snap)], [f"{len(snap)} stocks"]
    for sec, part in snap.groupby("sector"):
        ids.append(sec), labels.append(sec), parents.append(HUB), values.append(float(part["turnover_cr"].sum()))
        pct.append(weighted(part)), names.append(f"{len(part)} stocks")
        for _, r in part.iterrows():
            ids.append(f"{sec}/{r['symbol']}"), labels.append(r["symbol"]), parents.append(sec)
            values.append(float(r["turnover_cr"])), pct.append(float(r["pct_change"])), names.append(r["name"])
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        customdata=list(zip(pct, names)),
        marker=dict(colors=pct, colorscale=SCALE, cmin=-PCT_RANGE, cmax=PCT_RANGE, cmid=0,
                    line=dict(width=1, color="rgba(255,255,255,0.7)"),
                    colorbar=dict(title="% today", ticksuffix="%", thickness=12, len=0.6)),
        texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%", textposition="middle center",
        hovertemplate="<b>%{label}</b> (%{customdata[1]})<br>Change today: %{customdata[0]:+.2f}%<br>"
                      "Traded value: ₹%{value:,.0f} cr<extra></extra>",
        tiling=dict(pad=2), pathbar=dict(visible=True), maxdepth=3))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=30, b=0), font=dict(family=charts.FONT))
    return fig


def _node_values(links: dict) -> list[float]:
    """Total width of each node (what flows through it)."""
    vals = [0.0] * len(links["labels"])
    for s_, t, v in zip(links["source"], links["target"], links["value"]):
        vals[t] += v
        if links["side"][s_] == "out":
            vals[s_] += v
    return vals


def _sankey_layout(fig: go.Figure, height: int, top: int) -> None:
    fig.update_traces(textfont=dict(family=charts.FONT, size=12, shadow="none"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), font=dict(family=charts.FONT, size=12),
                      hoverlabel=dict(font=dict(family=charts.FONT, size=12)))


def sector_flow(links: dict) -> go.Figure:
    """Sankey: sectors with net selling (left, red) -> market -> sectors with net buying (right, blue)."""
    colors = {"out": DOWN, "hub": NEUTRAL, "in": UP}
    node_colors = [colors[s] for s in links["side"]]
    labels = [lab if side == "hub" else f"{lab.strip()}  ₹{v:,.0f} cr"
              for lab, side, v in zip(links["labels"], links["side"], _node_values(links))]
    link_colors = [_hex_rgba(DOWN, 0.35) if links["side"][s] == "out" else _hex_rgba(UP, 0.35)
                   for s in links["source"]]
    fig = go.Figure(go.Sankey(
        arrangement="snap", valueformat=",.0f", valuesuffix=" cr",
        node=dict(label=labels, color=node_colors, pad=14, thickness=16,
                  line=dict(width=0), hovertemplate="%{label}: ₹%{value:,.0f} cr<extra></extra>"),
        link=dict(source=links["source"], target=links["target"], value=links["value"], color=link_colors,
                  hovertemplate="%{source.label} → %{target.label}<br>₹%{value:,.0f} cr<extra></extra>")))
    _sankey_layout(fig, max(420, 34 * len(links["labels"])), top=50)
    for x, text in ((0.0, "← Net selling pressure (estimate)"), (1.0, "Net buying pressure (estimate) →")):
        fig.add_annotation(x=x, y=1.06, xref="paper", yref="paper", text=f"<b>{text}</b>", showarrow=False,
                           xanchor="left" if x == 0 else "right", font=dict(size=13))
    return fig


def volume_flow(links: dict) -> go.Figure:
    """Sankey: market -> sector -> most traded stocks, width = traded value, colour = % change today."""
    node_colors = [NEUTRAL if k == "hub" else _pct_color(p) for k, p in zip(links["kind"], links["pct"])]
    link_colors = [_pct_color(links["pct"][t], 0.45) for t in links["target"]]
    labels = [f"{lab}  {p:+.1f}%" if p is not None and k != "hub" else lab
              for lab, p, k in zip(links["labels"], links["pct"], links["kind"])]
    fig = go.Figure(go.Sankey(
        arrangement="snap", valueformat=",.0f", valuesuffix=" cr",
        node=dict(label=labels, color=node_colors, pad=6, thickness=14, line=dict(width=0),
                  hovertemplate="%{label}<br>Traded: ₹%{value:,.0f} cr<extra></extra>"),
        link=dict(source=links["source"], target=links["target"], value=links["value"], color=link_colors,
                  hovertemplate="%{source.label} → %{target.label}<br>₹%{value:,.0f} cr<extra></extra>")))
    stocks = sum(k != "sector" and k != "hub" for k in links["kind"])
    _sankey_layout(fig, max(520, 22 * stocks), top=20)
    return fig


def _sector_index_bars(ix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=ix["percChange"], y=ix["indexName"], orientation="h",
                           marker_color=[UP if v >= 0 else DOWN for v in ix["percChange"]],
                           text=[f"{v:+.2f}%" for v in ix["percChange"]], textposition="outside",
                           hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
    charts.style(fig, height=max(300, 22 * len(ix)), hover="closest")
    fig.update_xaxes(ticksuffix="%", zeroline=True, zerolinecolor="rgba(128,128,128,0.5)")
    return fig


HOW = """
**What the charts show**

* **Heat map**: every box is one stock, grouped by sector. A bigger box means more money traded in it
  today (traded value, also called turnover). Blue means the price is up today, red down; the deeper the
  colour, the bigger the move. Sector boxes take the traded-value-weighted average of their stocks.
* **Stock volume flow**: where today's trading is concentrated, from the whole group to each sector to its
  busiest stocks. Band width is traded value in ₹ crore, which NSE reports, so this one is **measured**.
* **Sector flow**: which sectors had more eager buyers or more eager sellers today. This one is an
  **estimate**, explained below.

**Why "money flow" can only be estimated**

Every share someone buys, someone else sells, so money does not really "flow into" a stock. What we can
see is who was more eager. If a stock trades near its high of the day, buyers kept paying up; near its
low, sellers kept accepting less. So for each stock we take today's traded value and multiply it by where
the price sits in today's range: +1 at the high, 0 in the middle, −1 at the low. ₹1,000 crore traded with
the price three quarters of the way up the range counts as +₹500 crore of buying pressure. Adding these
up by sector gives the sector flow. Traders know this as the "close location value" behind the Chaikin
money-flow indicator.

This can disagree with the day's % change. A sector that opened with a jump and has since slipped back
from its high is up on the day but shows **selling** pressure, because sellers have had the upper hand
since the jump. That is information, not a contradiction.

**Limits**: the estimate does not see the order book, large block deals or who the buyers are (funds,
foreign investors, retail). The middle box in the sector flow is only a meeting point; the chart does not
claim that money sold in one sector was spent in another. Prices are NSE's own feed, about a minute
behind the exchange. Use it to understand the day, not as a trading signal.
"""


def _body() -> None:
    st.session_state["flow_seen"] = time.time()
    bucket = _bucket()
    with st.spinner("Getting live prices for about 100 stocks from NSE (shared by everyone)..."):
        quotes, fetched_at = market_flow.latest_quotes(max_age=60 if live.market_open() else 1800)
    uni = market_flow.universe()
    snap = market_flow.snapshot(quotes, uni) if quotes is not None else pd.DataFrame()
    fetched = pd.Timestamp(fetched_at, unit="s", tz="UTC").tz_convert(live.IST).strftime("%H:%M:%S")
    if snap.empty:
        st.warning("NSE's live feed (mcp.nseindia.in) did not answer just now. Try again in a minute.")
        return
    sectors = market_flow.sector_table(snap)
    as_of = snap["as_of"].max()
    is_open = live.market_open()
    missing = len(uni) - len(snap)
    st.caption(f"NSE prices as of {as_of:%d %b %H:%M} IST (checked {fetched}). Market "
               f"{'open' if is_open else 'closed: showing the last session'}. {len(snap)} stocks"
               + (f"; {missing} not returned by NSE right now." if missing > 0 else "."))
    st.success(market_flow.headline(snap, sectors))

    up, down = int((snap["pct_change"] > 0).sum()), int((snap["pct_change"] < 0).sum())
    buy = float(snap.loc[snap["net_flow_cr"] > 0, "net_flow_cr"].sum())
    sell = float(-snap.loc[snap["net_flow_cr"] < 0, "net_flow_cr"].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks up / down", f"{up} / {down}")
    m2.metric("Traded value today", f"₹{snap['turnover_cr'].sum():,.0f} cr")
    m3.metric("Buying pressure (est.)", f"₹{buy:,.0f} cr")
    m4.metric("Selling pressure (est.)", f"₹{sell:,.0f} cr")

    view = st.radio("Chart", ["Heat map", "Sector flow (estimate)", "Stock volume flow"], horizontal=True,
                    key="flow_view", label_visibility="collapsed")
    if view == "Heat map":
        st.markdown("**Heat map**: box size = money traded today, colour = % change today. Click a sector to "
                    "zoom in; click the top bar to zoom out.")
        st.plotly_chart(heat_map(snap), use_container_width=True, config=CHART_CONFIG)
        ix = _sector_indices(bucket)
        if len(ix):
            with st.expander("NSE's official sector indices, % change today"):
                st.caption("These are NSE's own sector indices (weighted by company size), so they can differ "
                           "a little from the heat map, which uses only the stocks on this page.")
                st.plotly_chart(_sector_index_bars(ix), use_container_width=True, config=CHART_CONFIG)
    elif view == "Sector flow (estimate)":
        st.markdown("**Sector flow**: left, sectors where sellers had the upper hand today; right, sectors where "
                    "buyers did. Band width = estimated buying or selling pressure in ₹ crore. "
                    "*Estimate, see 'How this works' below.*")
        st.plotly_chart(sector_flow(market_flow.rotation_links(sectors, HUB)), use_container_width=True,
                        config=CHART_CONFIG)
        table = sectors.assign(flow_share=(sectors["flow_share"] * 100).round(0)).rename(columns={
            "sector": "Sector", "turnover_cr": "Traded ₹ cr", "net_flow_cr": "Net pressure ₹ cr (est.)",
            "flow_share": "How one-sided (%)", "pct_change": "% change (weighted)", "stocks": "Stocks",
            "up": "Up", "down": "Down"})
        st.dataframe(table.round(2), hide_index=True, use_container_width=True)
    else:
        top = st.slider("Stocks to show by name (the rest are grouped as 'Other')", 10, 50, 20, step=5,
                        key="flow_top")
        st.markdown("**Stock volume flow**: how today's trading splits into sectors and then into the busiest "
                    "stocks. Band width = traded value in ₹ crore (measured by NSE). Colour = % change today.")
        st.plotly_chart(volume_flow(market_flow.volume_links(snap, top, HUB)), use_container_width=True,
                        config=CHART_CONFIG)
    with st.expander("How this works, in plain words"):
        st.markdown(HOW)
        st.caption("Stock list and sectors: data/nifty100_sectors.csv (Nifty 50 and Nifty Next 50, edit it "
                   "when NSE changes the indices).")


AUTO_REFRESH_MINUTES = 30


def render() -> None:
    st.warning("Educational tool, not trading advice. 'Flow' here is an estimate from prices; see "
               "'How this works' at the bottom.")
    c1, c2 = st.columns([1, 3])
    auto = c1.toggle("Auto-refresh every minute", value=False, key="flow_auto",
                     help="Off by default to keep the server light. Works only while NSE is open.")
    if time.time() - st.session_state.get("flow_seen", 0) > 300:
        st.session_state["flow_started"] = time.time()
    if auto and time.time() - st.session_state.get("flow_started", time.time()) > AUTO_REFRESH_MINUTES * 60:
        c2.info(f"Auto-refresh paused after {AUTO_REFRESH_MINUTES} minutes to save the server's CPU.")
        if c2.button("Resume", key="flow_resume"):
            st.session_state["flow_started"] = time.time()
            st.rerun()
        auto = False
    run_every = timedelta(seconds=60) if auto and live.market_open() else None
    st.fragment(run_every=run_every)(_body)()
