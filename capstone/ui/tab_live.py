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
from capstone.ui.tab_overview import CSS, _e, _pct, range_bar, sparkline, src
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


SHORT = {"NIFTY FIN SERVICE": "NIFTY FIN SERVICES", "NIFTY MIDCAP 100": "MIDCAP 100"}


def _day_pos(last, low, high):
    return (last - low) / (high - low) if last is not None and low and high and high > low else None


def _tile(label, value, pct, inverse=False, graphic="", note="", title="") -> str:
    if pct is None or pct == 0:
        color = "inherit"
    else:
        color = UP if (pct > 0) != inverse else DOWN  # a rising VIX is shown red
    return (f'<div class="p8-tile" title="{_e(title)}"><div><div class="lab">{_e(label)}</div>'
            f'<div class="val">{_e(value)}</div><div class="chg" style="color:{color}">{_e(_pct(pct))}</div></div>'
            f'<div>{graphic}<div class="note">{_e(note)}</div></div></div>')


def index_tiles(indices: dict) -> str:
    out = []
    for name in INDICES:
        ix = indices.get(name)
        if not ix or ix.get("last") is None:
            continue
        pos = _day_pos(ix["last"], ix.get("low"), ix.get("high"))
        vix = name == "INDIA VIX"
        out.append(_tile(SHORT.get(name, name), f"{ix['last']:,.2f}", ix.get("percChange"), inverse=vix,
                         graphic=range_bar(pos, "#e0a030" if vix else ("#2a78d6" if (ix.get("percChange") or 0) >= 0 else DOWN)),
                         note=f"Day {ix.get('low', 0):,.0f} – {ix.get('high', 0):,.0f}" if ix.get("low") else "",
                         title=f"NSE live index, {ix.get('timeVal', '')}"))
    return '<div class="p8-grid">' + "".join(out) + "</div>"


def stock_tiles(preds: pd.DataFrame, quotes: dict, bars: dict) -> str:
    out = []
    for r in preds.itertuples():
        q = quotes.get(r.symbol) or {}
        b = bars.get(r.symbol)
        today = live.session(b) if b is not None and len(b) else pd.DataFrame()
        ltp = q.get("ltp") or (float(today["close"].iloc[-1]) if len(today) else None)
        if ltp is None:
            continue
        spark = sparkline(today["close"].iloc[::3]) if len(today) > 5 else range_bar(
            _day_pos(ltp, q.get("low"), q.get("high")), "#2a78d6")
        fc = getattr(r, "pred_open", None)
        note = f"Forecast open ₹{fc:,.2f}" if fc and pd.notna(fc) else ""
        if q.get("open") and fc and live._same_day(q.get("as_of"), getattr(r, "target_date", None)):
            note = f"Open ₹{q['open']:,.2f} · forecast ₹{fc:,.2f}"
        out.append(_tile(r.symbol, f"₹{ltp:,.2f}", q.get("pct_change"), graphic=spark, note=note,
                         title=f"Day {q.get('low', 0):,.2f} – {q.get('high', 0):,.2f}" if q else ""))
    cols = max(1, min(6, len(out)))
    return (f'<style>.p8-grid.p8-n{{grid-template-columns:repeat({cols},minmax(0,1fr))}}'
            '@media(max-width:1100px){.p8-grid.p8-n{grid-template-columns:repeat(3,minmax(0,1fr))}}'
            '@media(max-width:640px){.p8-grid.p8-n{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>'
            '<div class="p8-grid p8-n">' + "".join(out)
            + "</div>") if out else ""


def _candles(sym: str, today: pd.DataFrame, prev_close: float, pred_open: float | None, cfg, source: str = "") -> None:
    """Today's candles for one stock in a TradingView-style chart: volume, VWAP/EMA, opening range, levels."""
    if today.empty:
        st.info(f"No 1-minute candles for {sym} yet today.")
        return
    from capstone import intraday
    from capstone.ui import tv_chart
    day = today["datetime"].dt.date.iloc[0]
    orr = intraday.opening_range(today, day, cfg.or_min)
    rng = {"high": float(orr["high"]), "low": float(orr["low"])} if orr is not None else None
    tv_chart.render(sym, today, prev_close=float(prev_close) if prev_close else None,
                    pred_open=float(pred_open) if pred_open else None, opening_range=rng, dark=True, height=540,
                    source=source)


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


def _header(mkt, now, sources) -> None:
    c = {"open": "#2a9d6f", "pre_open": "#e0a030", "before_open": "#e0a030"}.get(mkt.state, "#8b93a3")
    st.markdown(CSS + f'<div class="p8-strip"><div><h2>Live markets</h2><div class="sub">{now:%A, %d %B %Y} · '
                f'updated {now:%H:%M:%S} IST · {_e(sources)}</div></div><div style="display:flex;gap:8px;flex-wrap:wrap">'
                f'<span class="p8-pill"><span class="p8-dot" style="background:{c}"></span>{_e(mkt.headline)}</span>'
                '<span class="p8-pill" style="opacity:.8">Educational · places no orders</span></div></div>',
                unsafe_allow_html=True)
    if mkt.detail:
        st.caption(mkt.detail)


def _body(preds: pd.DataFrame) -> None:
    st.session_state["live_seen"] = time.time()
    cfg = settings_from_state()
    bars, source = _bars(tuple(preds["symbol"]))
    quotes, indices = _nse(tuple(preds["symbol"]))
    now = live.now_ist()
    mkt = market_calendar.status(now)
    sources = " and ".join((["NSE"] if quotes else []) + ([source] if bars else [])) or "no live source answered"
    _header(mkt, now, f"sources: {sources}")

    st.markdown('<div class="p8-h">Indices</div><div class="p8-cap">NSE\'s own feed, 1–3 minutes behind the '
                'exchange; the bar shows where the index sits in today\'s range</div>', unsafe_allow_html=True)
    if indices:
        st.markdown(index_tiles(indices), unsafe_allow_html=True)
    else:
        st.caption("NSE's index feed did not answer just now.")
    st.markdown('<div class="p8-h" style="margin-top:10px">Your stocks</div><div class="p8-cap">'
                + ("Live price, change and today\'s 1-minute path" if live.market_open() else
                   "Last session\'s close; % change is versus the session before it") + "</div>", unsafe_allow_html=True)
    tiles = stock_tiles(preds, quotes, bars)
    if tiles:
        st.markdown(tiles, unsafe_allow_html=True)
    if not quotes:
        st.info("NSE's live feed (mcp.nseindia.in) did not answer just now, so prices are from Yahoo Finance only.")
    table = live.quote_table(preds, quotes)
    if len(table):
        with st.expander("Open, high, low, volume and forecast check for each stock"):
            st.dataframe(table, hide_index=True, use_container_width=True)

    if not bars:
        with st.container(border=True):
            st.markdown('<div class="p8-h">Intraday charts</div>', unsafe_allow_html=True)
            st.warning("No 1-minute candles from Yahoo Finance right now, so the candle chart, the buyer/seller "
                       "split and the breakout setups can\'t be shown. Yahoo may be blocking the server; try again "
                       "in a few minutes.")
        return
    syms = [s for s in preds["symbol"] if s in bars]
    sym = st.segmented_control("Chart for", syms, default=syms[0], key="live_sym") or syms[0]
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
    note = ""
    if pred and len(today) and not live._same_day(today["datetime"].iloc[0], target):
        note = (f"The forecast open (₹{pred:,.2f}) is for {pd.Timestamp(target):%a %d %b}, so it is not drawn on "
                "this older session.")
        pred = None

    left, right = st.columns([2.1, 1], gap="medium")
    with left, st.container(border=True):
        st.markdown(f'<div class="p8-h">{_e(sym)} · intraday chart, {today["datetime"].iloc[0]:%a %d %b}</div><div class="p8-cap">1, 3, 5 or 15-minute candles with volume, VWAP and EMAs; opening range, previous '
                    'close and forecast open marked</div>', unsafe_allow_html=True)
        _candles(sym, today, prev_close, pred, cfg)
        if note:
            st.caption(note)
        st.markdown(src(f"1-minute candles: {source}", today["datetime"].iloc[-1] if len(today) else None, live=True),
                    unsafe_allow_html=True)
    vs = live.volume_summary(today, prev_close)
    with right, st.container(border=True):
        st.markdown('<div class="p8-h">Buyers vs sellers</div><div class="p8-cap">Volume on upticks vs downticks '
                    '(an estimate, not the order book)</div>', unsafe_allow_html=True)
        if vs:
            bp = vs["buy_pct"] or 0
            rec = vs["recent_buy_pct"]
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:.8rem;opacity:.75">'
                f'<span>Bought {bp:.0f}%</span><span>Sold {100 - bp:.0f}%</span></div>'
                f'<div class="p8-bar" style="height:10px;margin:4px 0 12px"><span style="width:{bp:.1f}%;background:{UP}"></span>'
                f'<span style="left:{bp:.1f}%;right:0;width:auto;background:{DOWN}"></span></div>'
                '<table class="p8-t">'
                f'<tr><td>Volume today</td><td class="n">{vs["volume"]:,}</td></tr>'
                f'<tr><td>Bought on upticks</td><td class="n" style="color:{UP}">{vs["buy_vol"]:,}</td></tr>'
                f'<tr><td>Sold on downticks</td><td class="n" style="color:{DOWN}">{vs["sell_vol"]:,}</td></tr>'
                f'<tr><td>Last {vs["recent_minutes"]} min</td><td class="n">{_e(live.pressure_word(rec))}'
                + (f" ({rec:.0f}% buying)" if rec is not None else "") + "</td></tr></table>",
                unsafe_allow_html=True)
            st.plotly_chart(_net_chart(today, prev_close), use_container_width=True,
                            config={**CHART_CONFIG, "displayModeBar": False}, key="live_net")
        else:
            st.caption("No volume yet today.")
    with st.expander("Buyers vs sellers, minute by minute (full chart)"):
        st.plotly_chart(_pressure_chart(today, prev_close, sym), use_container_width=True, config=CHART_CONFIG)
        st.caption("Every trade has a buyer and a seller, so the split shows who was more eager. Volume in a minute "
                   "that closed higher counts as buying, lower as selling, unchanged is split by where it closed in "
                   "its range. Yahoo Finance does not give the order book; check Kite\'s market depth for that.")

    with st.container(border=True):
        st.markdown('<div class="p8-h">Breakout setups</div><div class="p8-cap">Opening-range breakouts with the '
                    'capital, risk and window set on the Intraday scalping page</div>', unsafe_allow_html=True)
        table, plans = live.live_rows(preds, bars, cfg, quotes, now)
        st.dataframe(table, hide_index=True, use_container_width=True)
        active = [p for p in plans if p["status"] in ("Armed", "In trade") and p["qty"] > 0]
        if active:
            kite_view.render_scalps(active, key="live_kite", ltps={p["symbol"]: p["ltp"] for p in active})
        elif plans:
            st.caption("No setup is waiting for a trigger right now, so there is nothing to set on Kite.")


def _net_chart(today: pd.DataFrame, prev_close: float):
    """Compact running buy − sell volume through the session."""
    import plotly.graph_objects as go
    df = live.buy_sell_split(today, prev_close)
    net = (df["buy_vol"] - df["sell_vol"]).cumsum()
    c = UP if len(net) and net.iloc[-1] >= 0 else DOWN
    fig = go.Figure(go.Scatter(x=df["datetime"], y=net, mode="lines", fill="tozeroy", line=dict(color=c, width=1.8),
                               fillcolor="rgba(42,120,214,.12)" if c == UP else "rgba(227,73,72,.12)",
                               hovertemplate="%{x|%H:%M} net %{y:,.0f}<extra></extra>"))
    charts.style(fig, height=190, hover="closest", legend_top=False)
    fig.update_layout(margin=dict(l=4, r=6, t=22, b=4), showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      title=dict(text="Running buy − sell volume", font=dict(size=12), x=0, y=0.98))
    fig.update_yaxes(tickformat=".2s", zeroline=True, zerolinecolor="rgba(128,128,128,.5)", showspikes=False)
    fig.update_xaxes(tickformat="%H:%M", showspikes=False, showgrid=False)
    return fig


AUTO_REFRESH_MINUTES = 30  # an auto-refresh left running pauses after this, to save server CPU


def render(preds: pd.DataFrame) -> None:
    c1, c2, _ = st.columns([1, 1.4, 2.6])
    auto = c1.toggle("Auto-refresh", value=True, key="live_auto", help="Market hours only; pauses after 30 minutes.")
    every = c2.segmented_control("Every", [30, 60, 120], default=60, key="live_every",
                                 format_func=lambda s: f"{s}s", label_visibility="collapsed") or 60
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
