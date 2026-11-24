"""Live tab: index and stock prices from NSE's own feed, today's 1-minute candles (Yahoo Finance),
buyer vs seller volume, and where each opening-range setup stands, with the Kite settings to act on it.
Refreshes by itself during market hours; every download is shared by all visitors for 55 seconds."""
from __future__ import annotations

import time
from datetime import timedelta

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

from capstone import charts, live, market_calendar
from capstone.charts import CONFIG as CHART_CONFIG
from capstone.ui import kite_view
from capstone.ui.common import DOWN, UP
from capstone.ui.tab_scalping import settings_from_state


@st.cache_data(ttl=55, show_spinner=False)
def _bars(symbols: tuple[str, ...]):
    """Shared by every visitor for 55 seconds (1-minute candles change once a minute anyway)."""
    return live.minute_bars(list(symbols))


INDICES = ["NIFTY 50", "NIFTY BANK", "NIFTY IT", "NIFTY FIN SERVICE", "NIFTY MIDCAP 100", "INDIA VIX"]


@st.cache_data(ttl=55, show_spinner=False)
def _nse(symbols: tuple[str, ...]):
    """Live quotes for the stocks and the main indices, shared by every visitor for 55 seconds.

    NSE refreshes them every 1-3 minutes, so sharing loses nothing. The indices come from one call.
    """
    quotes = live.nse_quotes(list(symbols))
    indices = {}
    try:
        from capstone import nse_mcp
        groups = (nse_mcp.call("market", "cm_get_live_indices", {"group": "", "nameFilter": ""}) or {})
        found = {i["indexName"]: i for g in groups.get("indices", {}).values() for i in g}
        indices = {name: found[name] for name in INDICES if name in found}
    except Exception:
        pass
    return quotes, indices


def _strip(items: list[tuple[str, float | None, float | None, bool]], per_row: int = 6) -> None:
    """A row of price cards: (label, value, % change, rising is bad)."""
    items = [i for i in items if i[1] is not None]
    for start in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (label, value, pct, inverse) in zip(cols, items[start:start + per_row]):
            col.metric(label, f"{value:,.2f}", f"{pct:+.2f}%" if pct is not None else None,
                       delta_color="inverse" if inverse else "normal")


def _nse_block(preds: pd.DataFrame, quotes: dict, indices: dict) -> None:
    st.markdown("**Indices** (NSE's own feed, 1-3 minutes behind the exchange)")
    _strip([(name, ix.get("last"), ix.get("percChange"), name == "INDIA VIX") for name, ix in indices.items()])
    st.markdown("**Your 5 stocks**" + ("" if live.market_open() else
                " (last session's close; % change is versus the session before it)"))
    _strip([(f"{s} (₹)", (quotes.get(s) or {}).get("ltp"), (quotes.get(s) or {}).get("pct_change"), False)
            for s in preds["symbol"]], per_row=len(preds) or 1)
    table = live.quote_table(preds, quotes)
    if len(table):
        with st.expander("Open, high, low, volume and forecast check for each stock"):
            st.dataframe(table, hide_index=True, use_container_width=True)


def _candles(sym: str, today: pd.DataFrame, prev_close: float, pred_open: float | None, cfg) -> None:
    """Today's 1-minute candles for one stock, with the opening range, previous close and forecast open."""
    if today.empty:
        st.info(f"No 1-minute candles for {sym} yet today.")
        return
    from capstone import intraday
    day = today["datetime"].dt.date.iloc[0]
    orr = intraday.opening_range(today, day, cfg.or_min)
    trade = None
    if orr is not None:
        start = pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=15)
        trade = {"or_high": orr["high"], "or_low": orr["low"], "or_end": start + pd.Timedelta(minutes=cfg.or_min)}
    fig = charts.intraday_chart(today, f"{sym}: 1-minute candles, {day:%a %d %b}", trade, price_height=460)
    for label, value, color in (("Prev close", prev_close, "#888"), ("Forecast open", pred_open, "#6250d6")):
        if value:
            fig.add_hline(y=value, line=dict(color=color, width=1, dash="dot"), row=1, col=1,
                          annotation=dict(text=f"{label} {value:,.2f}", font=dict(size=11, color=color)),
                          annotation_position="bottom left")
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def _pressure_chart(today: pd.DataFrame, prev_close: float, sym: str):
    """Buy/sell volume per minute (top) and the running net (bottom) on a shared time axis."""
    df = live.buy_sell_split(today, prev_close)
    net = (df["buy_vol"] - df["sell_vol"]).cumsum()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.05)
    fig.add_bar(x=df["datetime"], y=df["buy_vol"], name="Buy volume (uptick)", marker_color=UP,
                marker_line_width=0, row=1, col=1, hovertemplate="Bought %{y:,.0f}<extra></extra>")
    fig.add_bar(x=df["datetime"], y=-df["sell_vol"], name="Sell volume (downtick)", marker_color=DOWN,
                marker_line_width=0, row=1, col=1, hovertemplate="Sold %{customdata:,.0f}<extra></extra>",
                customdata=df["sell_vol"])
    fig.add_scatter(x=df["datetime"], y=net, name="Running buy − sell", mode="lines", fill="tozeroy",
                    line=dict(color="#2a78d6", width=2), fillcolor="rgba(42,120,214,0.12)", row=2, col=1,
                    hovertemplate="Net %{y:,.0f}<extra></extra>")
    charts.style(fig, height=460, title=f"{sym}: buyers vs sellers, minute by minute")
    fig.update_layout(barmode="relative", bargap=0.05)
    fig.update_yaxes(title_text="Shares / minute", tickformat=".2s", row=1, col=1)
    fig.update_yaxes(title_text="Running net", tickformat=".2s", row=2, col=1)
    return fig


def _body(preds: pd.DataFrame) -> None:
    st.session_state["live_seen"] = time.time()
    cfg = settings_from_state()
    bars, source = _bars(tuple(preds["symbol"]))
    quotes, indices = _nse(tuple(preds["symbol"]))
    now = live.now_ist()
    mkt = market_calendar.status(now)
    sources = " and ".join((["NSE"] if quotes else []) + ([source] if bars else [])) or "nowhere"
    if mkt.is_open:
        st.success(f"**{mkt.headline}.** {mkt.detail}")
    elif mkt.state not in ("holiday", "weekend"):  # those already have the banner at the top of the page
        st.info(f"**{mkt.headline}.** {mkt.detail}")
    st.caption(f"Updated {now:%d %b %H:%M:%S} IST from {sources}. "
               f"Settings (capital, risk, range, window) come from the Intraday scalping tab.")
    if quotes:
        _nse_block(preds, quotes, indices)
    else:
        st.info("NSE's live feed (mcp.nseindia.in) did not answer just now, so prices below are from "
                "Yahoo Finance only.")
    if not bars:
        st.warning("No 1-minute candles from Yahoo Finance right now, so the buyer/seller split and "
                   "the breakout setups can't be shown. Yahoo may be blocking the server; try again "
                   "in a few minutes.")
        return
    syms = [s for s in preds["symbol"] if s in bars]
    sym = st.radio("1-minute chart for", syms, horizontal=True, key="live_sym")
    b = bars[sym]
    today = live.session(b)
    older = b[b["datetime"].dt.date < today["datetime"].dt.date.iloc[0]] if len(today) else b
    prev_close = float(older["close"].iloc[-1]) if len(older) else float(today["open"].iloc[0])
    q = quotes.get(sym) or {}
    if q.get("prev_close") and q.get("as_of") and len(today) and q["as_of"].date() == today["datetime"].dt.date.iloc[0]:
        prev_close = float(q["prev_close"])  # NSE's official previous close
    row = preds.loc[preds["symbol"] == sym]
    pred = float(row["pred_open"].iloc[0]) if len(row) else None
    target = row["target_date"].iloc[0] if len(row) and "target_date" in row else None
    if pred and len(today) and not live._same_day(today["datetime"].iloc[0], target):
        st.caption(f"The forecast open (₹{pred:,.2f}) is for the next trading day, {pd.Timestamp(target):%a %d %b}, "
                   "so it is not drawn on this older session.")
        pred = None
    _candles(sym, today, prev_close, pred, cfg)

    vs = live.volume_summary(today, prev_close)
    if vs:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Volume today", f"{vs['volume']:,}")
        m2.metric("Bought on upticks", f"{vs['buy_vol']:,}", f"{vs['buy_pct']}%", delta_color="off")
        m3.metric("Sold on downticks", f"{vs['sell_vol']:,}", f"{vs['sell_pct']}%", delta_color="off")
        m4.metric(f"Last {vs['recent_minutes']} min", live.pressure_word(vs["recent_buy_pct"]),
                  f"{vs['recent_buy_pct']}% buying", delta_color="off")
        with st.expander("Buyers vs sellers, minute by minute"):
            st.plotly_chart(_pressure_chart(today, prev_close, sym), use_container_width=True, config=CHART_CONFIG)
    st.caption("How this is measured: every trade has a buyer and a seller, so the split shows who was "
               "more eager. Volume in a minute that closed higher counts as buying, lower as selling, "
               "unchanged is split by where it closed in its range. Yahoo Finance does not give the "
               "order book (the pending buy and sell quantities in Kite's market depth), so check "
               "Kite's depth for that.")

    st.markdown("**Breakout setups** (settings from the Intraday scalping tool)")
    table, plans = live.live_rows(preds, bars, cfg, quotes, now)
    st.dataframe(table, hide_index=True, use_container_width=True)
    active = [p for p in plans if p["status"] in ("Armed", "In trade") and p["qty"] > 0]
    if active:
        kite_view.render_scalps(active, key="live_kite", ltps={p["symbol"]: p["ltp"] for p in active})
    elif plans:
        st.info("No setup is waiting for a trigger right now, so there is nothing to set on Kite.")


AUTO_REFRESH_MINUTES = 30  # an auto-refresh left running pauses after this, to save server CPU


def render(preds: pd.DataFrame) -> None:
    st.warning("Educational tool, not trading advice. Nothing here places orders.")
    c1, c2 = st.columns([1, 2])
    auto = c1.toggle("Auto-refresh", value=True, key="live_auto")
    every = c2.select_slider("Refresh every", [30, 60, 120], value=60, key="live_every",
                             format_func=lambda s: f"{s} seconds")
    if time.time() - st.session_state.get("live_seen", 0) > 300:  # (re)opened after a break: start the clock
        st.session_state["live_started"] = time.time()
    started = st.session_state["live_started"]
    if auto and time.time() - started > AUTO_REFRESH_MINUTES * 60:
        st.info(f"Auto-refresh paused after {AUTO_REFRESH_MINUTES} minutes to save the server's CPU.")
        if st.button("Resume auto-refresh", key="live_resume"):
            st.session_state["live_started"] = time.time()
            st.rerun()
        auto = False
    # Outside market hours nothing moves: show the last session once, no auto-refresh.
    run_every = timedelta(seconds=every) if auto and live.market_open() else None
    st.fragment(run_every=run_every)(_body)(preds)
