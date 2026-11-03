"""Live tab: prices from Yahoo Finance that refresh by themselves, buyer vs seller volume,
and where each opening-range setup stands right now, with the Kite settings to act on it."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from capstone import live
from capstone.ui import kite_view
from capstone.ui.common import DOWN, UP
from capstone.ui.tab_scalping import settings_from_state


@st.cache_data(ttl=20, show_spinner=False)
def _bars(symbols: tuple[str, ...]):
    """Shared by every visitor for 20 seconds, so many open browsers don't hammer Yahoo."""
    return live.minute_bars(list(symbols))


def _pressure_chart(today: pd.DataFrame, prev_close: float, sym: str):
    df = live.buy_sell_split(today, prev_close)
    fig = go.Figure()
    fig.add_bar(x=df["datetime"], y=df["buy_vol"], name="Buy volume (uptick)", marker_color=UP)
    fig.add_bar(x=df["datetime"], y=-df["sell_vol"], name="Sell volume (downtick)", marker_color=DOWN)
    fig.add_scatter(x=df["datetime"], y=(df["buy_vol"] - df["sell_vol"]).cumsum(), name="Running buy − sell",
                    yaxis="y2", line=dict(color="#555", width=2))
    fig.update_layout(barmode="relative", height=340, margin=dict(l=10, r=10, t=40, b=10),
                      title=f"{sym}: buyers vs sellers, minute by minute",
                      yaxis=dict(title="Shares per minute"),
                      yaxis2=dict(title="Running total", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def _body(preds: pd.DataFrame) -> None:
    cfg = settings_from_state()
    bars, source = _bars(tuple(preds["symbol"]))
    now = live.now_ist()
    is_open = live.market_open(now)
    st.caption(f"Updated {now:%d %b %H:%M:%S} IST from {source}. Market "
               f"{'open' if is_open else 'closed: showing the last session'}. "
               f"Settings (capital, risk, range, window) come from the Intraday scalping tab.")
    if not bars:
        st.warning("No live data from Yahoo Finance right now. If this keeps happening, Yahoo may be "
                   "blocking the server; try again in a few minutes.")
        return
    table, plans = live.live_rows(preds, bars, cfg)
    st.dataframe(table, hide_index=True, use_container_width=True)

    sym = st.selectbox("Buyers vs sellers for", [s for s in preds["symbol"] if s in bars], key="live_sym")
    b = bars[sym]
    today = live.session(b)
    older = b[b["datetime"].dt.date < today["datetime"].dt.date.iloc[0]] if len(today) else b
    prev_close = float(older["close"].iloc[-1]) if len(older) else float(today["open"].iloc[0])
    vs = live.volume_summary(today, prev_close)
    if vs:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Volume today", f"{vs['volume']:,}")
        m2.metric("Bought on upticks", f"{vs['buy_vol']:,}", f"{vs['buy_pct']}%", delta_color="off")
        m3.metric("Sold on downticks", f"{vs['sell_vol']:,}", f"{vs['sell_pct']}%", delta_color="off")
        m4.metric(f"Last {vs['recent_minutes']} min", live.pressure_word(vs["recent_buy_pct"]),
                  f"{vs['recent_buy_pct']}% buying", delta_color="off")
        st.plotly_chart(_pressure_chart(today, prev_close, sym), use_container_width=True)
    st.caption("How this is measured: every trade has a buyer and a seller, so the split shows who was "
               "more eager. Volume in a minute that closed higher counts as buying, lower as selling, "
               "unchanged is split by where it closed in its range. Yahoo Finance does not give the "
               "order book (the pending buy and sell quantities in Kite's market depth), so check "
               "Kite's depth for that.")

    active = [p for p in plans if p["status"] in ("Armed", "In trade") and p["qty"] > 0]
    if active:
        kite_view.render_scalps(active, key="live_kite", ltps={p["symbol"]: p["ltp"] for p in active})
    elif plans:
        st.info("No setup is waiting for a trigger right now, so there is nothing to set on Kite.")


def render(preds: pd.DataFrame) -> None:
    st.warning("Educational tool, not trading advice. Nothing here places orders.")
    c1, c2 = st.columns([1, 2])
    auto = c1.toggle("Auto-refresh", value=True, key="live_auto")
    every = c2.select_slider("Refresh every", [15, 30, 60, 120], value=30, key="live_every",
                             format_func=lambda s: f"{s} seconds")
    # Outside market hours nothing moves, so check only every 2 minutes (to catch 09:15).
    run_every = (timedelta(seconds=every if live.market_open() else 120)) if auto else None
    st.fragment(run_every=run_every)(_body)(preds)
