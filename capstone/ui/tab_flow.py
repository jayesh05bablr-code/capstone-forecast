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
from capstone.ui import flow_anim, stock_panel
from capstone.ui.common import DOWN, UP

HUB = "Nifty 100 stocks"
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


def _sector_index_bars(ix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=ix["percChange"], y=ix["indexName"], orientation="h",
                           marker_color=[UP if v >= 0 else DOWN for v in ix["percChange"]],
                           text=[f"{v:+.2f}%" for v in ix["percChange"]], textposition="outside",
                           hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
    charts.style(fig, height=max(300, 22 * len(ix)), hover="closest")
    fig.update_xaxes(ticksuffix="%", zeroline=True, zerolinecolor="rgba(128,128,128,0.5)")
    return fig


def _forecast_symbols() -> list[str]:
    """Stocks the opening-price model forecasts (the panel links to their forecast page)."""
    from capstone.paths import PROCESSED
    try:
        return list(pd.read_csv(PROCESSED / "predictions.csv", usecols=["symbol"])["symbol"])
    except (OSError, ValueError):
        return []


HOW = """
**What the charts show**

* **Heat map**: every box is one stock, grouped by sector. A bigger box means more money traded in it
  today (traded value, also called turnover). Blue means the price is up today, red down; the deeper the
  colour, the bigger the move. Sector boxes take the traded-value-weighted average of their stocks.
* **Capital distribution** (stock volume flow): where today's trading is concentrated, from the whole group to each sector to its
  busiest stocks. Band width is traded value in ₹ crore, which NSE reports, so this one is **measured**.
* **Buyer vs seller control** (sector flow): how much of each sector's and each big stock's trading was driven by buyers and how
  much by sellers today. This one is an **estimate**, explained below.

**What the moving dots mean**

The two flow charts move in your browser (the server does no extra work). Thicker bands and more dots mean
more rupees. Faster dots mean a stronger move: a bigger % change on the volume chart, a more one-sided
stock (price right at its high or low) on the sector chart. An amber glow means the stock traded at least
2.5 times its usual pace in the last minute, by comparing NSE's traded value now with a minute ago. It
appears once the page has had two downloads. Use "Pause" to stop the animation and the
speed buttons (0.5× to 4×) to slow it down or speed it up; the chart stops drawing by itself while it is
off screen. Click a sector to see all of its stocks, or a stock to open its panel below the chart.

**Why "money flow" can only be estimated**

Every share someone buys, someone else sells, so money does not really "flow into" a stock. What we can
see is who was more eager. If a stock trades near its high of the day, buyers kept paying up; near its
low, sellers kept accepting less. So for each stock we take today's traded value and multiply it by where
the price sits in today's range: +1 at the high, 0 in the middle, −1 at the low. ₹1,000 crore traded with
the price three quarters of the way up the range counts as +₹500 crore of net buying pressure. Traders know
this as the "close location value" behind the Chaikin money-flow indicator.

The sector flow chart shows the same idea as a split: that stock's ₹1,000 crore becomes about ₹750 crore
"buyers in control" and ₹250 crore "sellers in control" (at the day's high it is all buyers, at the low all
sellers, in the middle half and half). On a day when nearly everything falls, you still see which sectors
and stocks held up better.

This can disagree with the day's % change. A sector that opened with a jump and has since slipped back
from its high is up on the day but shows **selling** pressure, because sellers have had the upper hand
since the jump. That is information, not a contradiction.

**Limits**: the estimate does not see the order book, large block deals or who the buyers are (funds,
foreign investors, retail). The two middle boxes are totals, not a claim that money sold in one sector
was spent in another. Prices are NSE's own feed, about a minute
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
    from capstone import market_calendar
    mkt = market_calendar.status()
    st.caption(f"NSE prices as of {as_of:%d %b %H:%M} IST (checked {fetched}). {mkt.headline}"
               f"{'' if is_open else ': showing the last session'}. {len(snap)} stocks"
               + (f"; {missing} not returned by NSE right now." if missing > 0 else "."))
    st.success(market_flow.headline(snap, sectors))

    up, down = int((snap["pct_change"] > 0).sum()), int((snap["pct_change"] < 0).sum())
    buy, sell = float(snap["buy_cr"].sum()), float(snap["sell_cr"].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks up / down", f"{up} / {down}")
    m2.metric("Traded value today", f"₹{snap['turnover_cr'].sum():,.0f} cr")
    m3.metric("Buyer-driven trading (est.)", f"₹{buy:,.0f} cr", f"{100 * buy / (buy + sell):.0f}% of all", delta_color="off")
    m4.metric("Seller-driven trading (est.)", f"₹{sell:,.0f} cr", f"{100 * sell / (buy + sell):.0f}% of all", delta_color="off")

    focus = st.session_state.get("flow_focus")
    if focus and focus not in set(snap["sector"]):
        focus = st.session_state["flow_focus"] = None
    views = ["Buyer vs seller control", "Capital distribution", "Heat map"]
    if st.session_state.get("flow_view") not in views:
        st.session_state["flow_view"] = views[0]  # old saved choice from before the rename
    view = st.radio("Chart", views, horizontal=True, key="flow_view", label_visibility="collapsed")
    click = None
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
    elif view == "Buyer vs seller control":
        st.markdown("**Buyer vs seller control**: sectors → buyers or sellers in control → stocks. Blue = estimated "
                    "buyer-driven, red = estimated seller-driven trading. *Estimate from prices, not reported "
                    "institutional orders; see 'How this works' below.*")
        click = flow_anim.render(flow_anim.sector_flow_data(snap, sectors, market_flow.pace(snap), focus=focus),
                                 key="flow_control")
        table = sectors.assign(flow_share=(sectors["flow_share"] * 100).round(0))[
            ["sector", "turnover_cr", "buy_cr", "sell_cr", "net_flow_cr", "flow_share", "pct_change", "stocks", "up",
             "down"]].rename(columns={
                "sector": "Sector", "turnover_cr": "Traded ₹ cr", "buy_cr": "Buying ₹ cr (est.)",
                "sell_cr": "Selling ₹ cr (est.)", "net_flow_cr": "Net ₹ cr (est.)", "flow_share": "How one-sided (%)",
                "pct_change": "% change (weighted)", "stocks": "Stocks", "up": "Up", "down": "Down"})
        with st.expander("Sector numbers as a table"):
            st.dataframe(table.round(2), hide_index=True, use_container_width=True)
    else:
        top = st.slider("Stocks to show by name (the rest are grouped as 'other')", 10, 40, 20, step=5,
                        key="flow_top", disabled=bool(focus))
        st.markdown("**Capital distribution**: today's traded value (₹ crore, reported by NSE) from all stocks to "
                    "sectors to the busiest stocks. Blue = up, red = down, grey = flat. *Measured, no estimate.*")
        click = flow_anim.render(flow_anim.volume_flow_data(snap, market_flow.pace(snap), top, focus=focus),
                                 key="flow_capital")
    if click:
        if click["kind"] == "sector":
            st.session_state["flow_focus"] = click["id"]
        elif click["kind"] == "reset":
            st.session_state["flow_focus"] = None
        elif click["kind"] == "stock":
            st.session_state["flow_stock"] = click["id"]
        st.rerun(scope="fragment")
    if focus:
        c1, c2 = st.columns([4, 1])
        c1.caption(f"Showing only **{focus}**. Click '◀ All sectors' in the chart or the button to go back.")
        if c2.button("Show all sectors", key="flow_unfocus", use_container_width=True):
            st.session_state["flow_focus"] = None
            st.rerun(scope="fragment")
    picked = st.session_state.get("flow_stock")
    if picked:
        stock_panel.render(snap, picked, _forecast_symbols(), key="flow_sp")
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
