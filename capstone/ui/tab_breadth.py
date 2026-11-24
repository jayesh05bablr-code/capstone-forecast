"""Breadth and risk page: how many stocks take part in the move, and how liquid and risky they are.

Whole-market advances/declines are NSE's own count over every listed equity. Everything else is
computed from the chosen universe's saved daily bhavcopy (and NSE's 200-day average per stock), and a
figure is shown only when the data behind it is there.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from capstone import breadth, charts, flow_history, market_flow, stock_metrics
from capstone.charts import CONFIG
from capstone.ui.common import DOWN, UP


def _na(v, fmt):
    return fmt.format(v) if v is not None and pd.notna(v) else "n/a"


def market_chart(mk: pd.DataFrame) -> go.Figure:
    mk = mk.assign(date=mk["date"].dt.strftime("%d %b"))  # sessions only, no weekend gaps
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=mk["date"], y=mk["advances"], name="Advances", marker_color=UP))
    fig.add_trace(go.Bar(x=mk["date"], y=-mk["declines"], name="Declines", marker_color=DOWN,
                         customdata=mk["declines"], hovertemplate="%{customdata:,}"))
    fig.add_trace(go.Scatter(x=mk["date"], y=mk["ad_ratio"], name="A/D ratio", line=dict(color="#f2b84b", width=2)),
                  secondary_y=True)
    charts.style(fig, height=380)
    fig.update_layout(barmode="relative")
    fig.update_xaxes(type="category", nticks=12)
    fig.update_yaxes(title="Stocks", secondary_y=False)
    fig.update_yaxes(title="A/D ratio", secondary_y=True, showgrid=False)
    return fig


def participation_chart(bh: pd.DataFrame, ix: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.06,
                        specs=[[{"secondary_y": True}], [{}]])
    nifty = ix.set_index("date")["close"].reindex(bh["date"])
    fig.add_trace(go.Scatter(x=bh["date"], y=bh["ad_line"], name="A/D line (universe)", line=dict(color=UP, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=bh["date"], y=nifty, name="NIFTY 50", line=dict(color="#8b93a3", width=1.6, dash="dot")),
                  1, 1, secondary_y=True)
    fig.add_trace(go.Scatter(x=bh["date"], y=bh["above_20"], name="% above 20-day avg", fill="tozeroy",
                             line=dict(color="#6a5cff", width=1.8)), 2, 1)
    charts.style(fig, height=520)
    fig.update_yaxes(title="Cumulative adv − dec", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title="NIFTY 50", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title="%", range=[0, 100], row=2, col=1)
    return fig


@st.cache_data(ttl=300, show_spinner=False)
def _stats(uni_name: str, mtime: float) -> pd.DataFrame:
    return breadth.stock_stats(flow_history.load(), market_flow.universe(uni_name))


def _snapshot(uni: pd.DataFrame) -> pd.DataFrame | None:
    """Live NSE snapshot (shared with the Market flow page) for 52-week highs/lows; None if unavailable."""
    try:
        quotes, _ = market_flow.latest_quotes()
        return market_flow.snapshot(quotes, uni) if quotes is not None and len(quotes) else None
    except Exception:
        return None


def render() -> None:
    uni_name = st.selectbox("Universe", list(market_flow.UNIVERSES), index=2, key="br_uni",
                            help="Breadth and risk of these stocks. The whole-market count above covers every NSE equity.")
    uni = market_flow.universe(uni_name)
    breadth.refresh_market()
    flow_history.refresh(list(uni["symbol"]))
    breadth.refresh_dma(list(uni["symbol"]))

    # ---- whole market ----
    mk = breadth.load_market()
    st.markdown("#### Whole NSE market")
    if mk.empty:
        st.info("Downloading NSE's daily advance/decline counts (about 15 seconds)."
                if breadth.status()["market_busy"] else "NSE's market breadth is not available right now.")
    else:
        last = mk.iloc[-1]
        c = st.columns(4)
        c[0].metric("Advances", f"{int(last['advances']):,}")
        c[1].metric("Declines", f"{int(last['declines']):,}")
        c[2].metric("A/D ratio", f"{last['ad_ratio']:.2f}",
                    help="Advances ÷ declines. Above 1.5 is a broad up day, below 0.7 a broad down day (NSE's guide).")
        up_days = int((mk.tail(10)["ad_ratio"] > 1).sum())
        c[3].metric("Up-breadth sessions, last 10", f"{up_days} of 10")
        st.plotly_chart(market_chart(mk.tail(breadth.SESSIONS)), use_container_width=True, config=CONFIG)
        st.caption(f"Every NSE-listed equity ({int(last['total_stocks']):,} on {last['date']:%d %b %Y}). "
                   "Source: NSE bhavcopy market breadth (mcp.nseindia.in), end of day.")

    # ---- universe breadth ----
    hist = flow_history.load()
    p = flow_history.progress()
    if p["busy"]:
        st.info(f"Downloading daily history from NSE's bhavcopy: {p['done']} of {p['total']} stocks.")
    mtime = flow_history.HISTORY_FILE.stat().st_mtime if flow_history.HISTORY_FILE.exists() else 0.0
    stats = _stats(uni_name, mtime)
    st.markdown(f"#### {uni_name} breadth")
    if stats.empty:
        st.info("No daily history saved for these stocks yet; it is downloading in the background.")
        return
    dma = breadth.load_dma()
    ub = breadth.universe_breadth(stats, dma, _snapshot(uni))
    c = st.columns(6)
    c[0].metric("Advancing / declining", f"{ub['advancing']} / {ub['declining']}",
                help=f"Close vs previous close on {ub['as_of']:%d %b %Y}.")
    c[1].metric("A/D ratio", _na(ub["ad_ratio"], "{:.2f}"))
    c[2].metric("Above 20-day avg", _na(ub["above_20"], "{:.0f}%"))
    c[3].metric("Above 50-day avg", _na(ub["above_50"], "{:.0f}%"))
    c[4].metric("Above 200-day avg", _na(ub["above_200"], "{:.0f}%"),
                help="NSE's 200-day average per stock (unadjusted for splits/bonuses). "
                     + (f"{ub['dma200_n']} stocks." if ub["above_200"] is not None else "Still downloading."))
    c[5].metric("At 52-week high / low", f"{ub['new_high']} / {ub['new_low']}" if ub["new_high"] is not None else "n/a",
                help="Stocks whose day's high (low) is within 0.5% of the 52-week high (low) in NSE's live feed.")
    st.caption(f"{ub['stocks']} of {len(uni)} stocks with a close on {ub['as_of']:%d %b %Y}. Participation = share of "
               "stocks above their own moving average. Prices are NSE bhavcopy closes, not adjusted for splits.")

    bh = breadth.breadth_history(hist, uni)
    ix = stock_metrics.nifty_history()
    for a in breadth.divergences(ix, bh):
        st.warning("**Divergence:** " + a)
    if len(bh):
        st.plotly_chart(participation_chart(bh, ix), use_container_width=True, config=CONFIG)

    # ---- liquidity and risk ----
    st.markdown(f"#### {uni_name} liquidity and risk")
    unusual = stats[stats["unusual_volume"]].sort_values("rvol", ascending=False)
    if len(unusual):
        st.markdown("**Unusual volume** (last session ≥ 2× the 20-session average): " + ", ".join(
            f"{r.symbol} {r.rvol:.1f}× ({r.ret_1d:+.1f}%)" for r in unusual.head(12).itertuples()))
    c = st.columns([1.4, 1.4, 1])
    liq = c[0].multiselect("Liquidity", ["Very high", "High", "Medium", "Low"], key="br_liq", placeholder="All")
    secs = c[1].multiselect("Sectors", sorted(stats["sector"].unique()), key="br_sec", placeholder="All sectors")
    sort = c[2].selectbox("Sort by", ["ADTV", "Volatility", "ATR %", "Max drawdown", "Relative volume"], key="br_sort")
    t = stats.copy()
    if liq:
        t = t[t["liquidity"].isin(liq)]
    if secs:
        t = t[t["sector"].isin(secs)]
    key = {"ADTV": ("adtv_cr", False), "Volatility": ("vol_20d", False), "ATR %": ("atr_pct", False),
           "Max drawdown": ("max_dd", True), "Relative volume": ("rvol", False)}[sort]
    t = t.sort_values(key[0], ascending=key[1])
    st.dataframe(t[["symbol", "name", "sector", "close", "ret_1d", "adtv_cr", "liquidity", "vol_20d", "atr_pct",
                    "max_dd", "rvol"]], hide_index=True, use_container_width=True, column_config={
        "symbol": "Symbol", "name": "Company", "sector": "Sector",
        "close": st.column_config.NumberColumn("Close ₹", format="%.2f"),
        "ret_1d": st.column_config.NumberColumn("1D", format="%+.2f%%"),
        "adtv_cr": st.column_config.NumberColumn("ADTV ₹ cr", format="%.0f",
                                                 help="Average daily traded value over 20 sessions."),
        "liquidity": st.column_config.TextColumn("Liquidity", help="Our bands on ADTV: ≥ ₹500 cr Very high, "
                                                 "≥ ₹100 cr High, ≥ ₹20 cr Medium, else Low."),
        "vol_20d": st.column_config.NumberColumn("Volatility 20D", format="%.1f%%",
                                                 help="Annualised stdev of daily log returns, 20 sessions."),
        "atr_pct": st.column_config.NumberColumn("ATR %", format="%.2f%%", help="14-session average true range ÷ close."),
        "max_dd": st.column_config.NumberColumn("Max drawdown", format="%.1f%%",
                                                help="Worst fall from a high over the saved history (~3-5 months)."),
        "rvol": st.column_config.NumberColumn("Rel. volume", format="%.2f×",
                                              help="Last session's volume ÷ the 20 sessions before it."),
    })
    st.caption(f"{len(t)} stocks. Daily NSE bhavcopy {stats['from'].min():%d %b} – {stats['to'].max():%d %b %Y}; "
               "measures need 15-20 sessions and show blank when a stock has fewer.")
