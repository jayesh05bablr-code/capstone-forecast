"""Stock panel: opens when a stock is clicked in a flow chart.

Shows what NSE reports for the stock right now (price, previous close, change, traded value, volume,
day's range, VWAP, 52-week range) next to the estimated buyer/seller split, clearly labelled as an
estimate, and how the stock did against its own sector. Everything comes from the same once-a-minute
snapshot the charts use, so opening the panel costs no extra NSE call.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def _go_to_forecast(symbol: str) -> None:
    # app.py applies "nav_to" before it draws the page menu (the menu cannot be changed after it is drawn).
    st.session_state["forecast_symbol"] = symbol
    st.session_state["nav_to"] = "Stock details"


def facts(snap: pd.DataFrame, symbol: str) -> dict | None:
    """Numbers for one stock (None if it is not in the snapshot). Pure, so it can be tested."""
    row = snap[snap["symbol"] == symbol]
    if row.empty:
        return None
    r = row.iloc[0]
    sec = snap[snap["sector"] == r["sector"]]
    sec_pct = float((sec["pct_change"] * sec["turnover_cr"]).sum() / sec["turnover_cr"].sum())
    span = r["high"] - r["low"]
    out = {
        "symbol": r["symbol"], "name": r.get("name", ""), "sector": r["sector"], "ltp": float(r["ltp"]),
        "prev_close": float(r["prev_close"]), "pct_change": float(r["pct_change"]),
        "turnover_cr": float(r["value_cr"] if "value_cr" in r else r["turnover_cr"]), "volume": float(r["volume"]),
        "high": float(r["high"]), "low": float(r["low"]), "open": float(r["open"]),
        "vwap": float(r["vwap"]) if pd.notna(r.get("vwap")) else None,
        "buy_share": float(r["buy_cr"] / r["turnover_cr"]) if r["turnover_cr"] else None,
        "sector_pct": sec_pct, "vs_sector": float(r["pct_change"]) - sec_pct,
        "range_pos": float((r["ltp"] - r["low"]) / span) if span > 0 else None,
        "high_52w": float(r["high_52w"]) if pd.notna(r.get("high_52w")) else None,
        "low_52w": float(r["low_52w"]) if pd.notna(r.get("low_52w")) else None,
        "as_of": r.get("as_of"),
    }
    out["vs_vwap"] = 100 * (out["ltp"] / out["vwap"] - 1) if out["vwap"] else None
    rank = snap["turnover_cr"].rank(ascending=False)
    out["value_rank"] = int(rank[row.index[0]])
    out["rvol"] = float(r["rvol"]) if "rvol" in r and pd.notna(r.get("rvol")) else None
    return out


def _rolling_vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """20-session VWAP from daily rows: sum(traded value) / sum(volume) over the window."""
    return df["turnover"].rolling(n).sum() / df["volume"].rolling(n).sum()


@st.cache_data(ttl=60, max_entries=8, show_spinner=False)
def _minute_bars(symbol: str, bucket: int) -> pd.DataFrame:
    from capstone import live
    bars, _ = live.minute_bars([symbol])
    df = bars.get(symbol)
    return live.session(df) if df is not None and len(df) else pd.DataFrame()


def _intraday_chart(bars: pd.DataFrame, prev_close: float | None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from capstone import charts, stock_metrics
    vw = stock_metrics.intraday_vwap(bars)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=bars["datetime"], open=bars["open"], high=bars["high"], low=bars["low"],
                                 close=bars["close"], name="1-min", showlegend=False,
                                 increasing=dict(line=dict(color=charts.UP, width=1), fillcolor=charts.UP),
                                 decreasing=dict(line=dict(color=charts.DOWN, width=1), fillcolor=charts.DOWN)), 1, 1)
    fig.add_trace(go.Scatter(x=bars["datetime"], y=vw, name="VWAP", line=dict(color="#f2b84b", width=1.6)), 1, 1)
    if prev_close:
        fig.add_hline(y=prev_close, line=dict(color="#8b93a3", width=1, dash="dot"), row=1, col=1,
                      annotation_text="prev close", annotation_position="top left")
    fig.add_trace(go.Bar(x=bars["datetime"], y=bars["volume"], name="Volume", showlegend=False,
                         marker_color="#5d6b85"), 2, 1)
    charts.style(fig, height=460)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def render_history(symbol: str, hist: pd.DataFrame, live_row: dict | None, key: str) -> None:
    """Risk measures and charts from the saved daily history (and today's 1-minute bars if asked)."""
    import plotly.graph_objects as go
    from capstone import charts, stock_metrics
    df = hist[hist["symbol"] == symbol].sort_values("date").reset_index(drop=True) if len(hist) else hist
    if df is None or len(df) < 5:
        st.caption("Daily history for this stock is still downloading from NSE's bhavcopy; the risk measures "
                   "and chart appear once it is in (usually within a minute).")
        return
    ix = stock_metrics.nifty_history()
    m = stock_metrics.summary(df, ix)
    na = "n/a"
    c = st.columns(5)
    rv = live_row.get("rvol") if live_row else None
    c[0].metric("Relative volume", f"{rv:.2f}×" if rv else na, help="Volume vs its 20-session average. "
                "During market hours today's volume is scaled to a full day.")
    c[1].metric("ATR %", f"{m['atr_pct']:.2f}%" if m["atr_pct"] is not None else na,
                help="14-session average true range as % of the last close: a typical day's swing.")
    c[2].metric("Volatility (20d, yearly)", f"{m['vol_20d']:.1f}%" if m["vol_20d"] is not None else na,
                help="Standard deviation of daily returns over 20 sessions, annualised (×√252).")
    c[3].metric("Beta vs NIFTY 50", f"{m['beta']:.2f}" if m["beta"] is not None else na,
                help="How much the stock has moved per 1% move of the NIFTY 50 (daily returns, up to 120 sessions).")
    c[4].metric("Max drawdown", f"{m['max_dd']:.1f}%" if m["max_dd"] is not None else na,
                help=f"Worst fall from a high, {m['from']:%d %b} to {m['to']:%d %b %Y}.")
    c = st.columns(5)
    for col, (lab, k) in zip(c, [("1 week", "ret_1w"), ("1 month", "ret_1m"), ("~3 months (60 sessions)", "ret_3m")]):
        col.metric(f"Return {lab}", f"{m[k]:+.2f}%" if m[k] is not None else na)
    nifty_1m = stock_metrics.ret(ix, 21) if len(ix) else None
    c[3].metric("NIFTY 50, 1 month", f"{nifty_1m:+.2f}%" if nifty_1m is not None else na)
    c[4].metric("Avg daily value (20d)", f"₹{m['avg_value_cr']:,.0f} cr" if m["avg_value_cr"] else na)

    tab_d, tab_i = st.tabs(["Daily candles", "Today, 1-minute + VWAP"])
    with tab_d:
        fig = charts.price_chart(df, f"{symbol} daily", period="3M", overlays=("SMA 20",), panels=("Volume",),
                                 price_height=360)
        shown = df.tail(63)
        fig.add_trace(go.Scatter(x=shown["date"], y=_rolling_vwap(df).tail(63), name="VWAP 20d",
                                 line=dict(color="#f2b84b", width=1.4, dash="dash")), row=1, col=1)
        st.plotly_chart(fig, use_container_width=True, config=charts.CONFIG, key=f"{key}_daily")
        st.caption(f"NSE bhavcopy daily prices {m['from']:%d %b} – {m['to']:%d %b %Y} (not adjusted for splits or "
                   "bonuses). VWAP 20d = traded value ÷ volume over 20 sessions.")
    with tab_i:
        if st.toggle("Load today's 1-minute candles", key=f"{key}_intra",
                     help="Fetched only when you ask (Yahoo Finance 1-minute data, cached for a minute)."):
            import time as _t
            bars = _minute_bars(symbol, int(_t.time() // 60))
            if bars.empty:
                st.info("No 1-minute candles available right now (market closed for a while, or the source "
                        "did not answer).")
            else:
                st.plotly_chart(_intraday_chart(bars, live_row.get("prev_close") if live_row else None),
                                use_container_width=True, config=charts.CONFIG, key=f"{key}_intra_chart")
                st.caption(f"Session of {bars['datetime'].iloc[0]:%d %b %Y}. Source: Yahoo Finance 1-minute bars; "
                           "VWAP computed from them (typical price × volume).")


def render(snap: pd.DataFrame, symbol: str, forecast_symbols: list[str] | None = None, key: str = "sp",
           hist: pd.DataFrame | None = None, period: str = "Intraday") -> None:
    f = facts(snap, symbol)
    if f is None:
        st.info(f"{symbol} is not in the current NSE snapshot.")
        return
    with st.container(border=True):
        head, btns = st.columns([3, 2])
        head.markdown(f"#### {f['symbol']} · {f['name']}")
        head.caption(f"NSE: {f['symbol']} · Sector: {f['sector']} · #{f['value_rank']} by traded value in this view"
                     + (f" · as of {pd.Timestamp(f['as_of']):%d %b %H:%M} IST" if pd.notna(f['as_of']) else ""))
        if forecast_symbols and f["symbol"] in forecast_symbols:
            if btns.button("Open its opening-price forecast →", key=f"{key}_fc", use_container_width=True):
                _go_to_forecast(f["symbol"])
                st.rerun(scope="app")  # the panel runs inside a fragment; the whole page must change
        if btns.button("Close panel", key=f"{key}_close", use_container_width=True):
            st.session_state.pop("flow_stock", None)
            st.rerun(scope="fragment")

        c = st.columns(4)
        c[0].metric("Price (LTP)", f"₹{f['ltp']:,.2f}", f"{f['pct_change']:+.2f}%")
        c[1].metric("Previous close" if period in ("Intraday", "1 Day") else "Close before the period",
                    f"₹{f['prev_close']:,.2f}")
        c[2].metric("Traded value", f"₹{f['turnover_cr']:,.0f} cr")
        c[3].metric("Volume", f"{f['volume']:,.0f}")
        c = st.columns(4)
        c[0].metric("VWAP", f"₹{f['vwap']:,.2f}" if f["vwap"] else "n/a",
                    f"{f['vs_vwap']:+.2f}% price vs VWAP" if f["vs_vwap"] is not None else None,
                    help="Volume-weighted average price of the session: traded value ÷ volume.")
        day = "Day" if period in ("Intraday", "1 Day") else f"{period} period"
        c[1].metric(f"{day} high", f"₹{f['high']:,.2f}")
        c[2].metric(f"{day} low", f"₹{f['low']:,.2f}")
        c[3].metric("Vs its sector" + (" today" if period == "Intraday" else f", {period.lower()}"), f"{f['vs_sector']:+.2f} pts",
                    help=f"Stock's % change minus its sector's value-weighted % change ({f['sector_pct']:+.2f}%).")
        bits = []
        if f["buy_share"] is not None:
            bits.append(f"**Estimated buyer-driven share: {100 * f['buy_share']:.0f}%** (seller-driven "
                        f"{100 - 100 * f['buy_share']:.0f}%), from where the price sits in the day's range"
                        + (f" ({100 * f['range_pos']:.0f}% of the way up)" if f["range_pos"] is not None else "")
                        + ". An estimate, not reported order flow.")
        st.markdown(" ".join(bits))
        if hist is not None:
            row = snap[snap["symbol"] == symbol].iloc[0].to_dict()
            render_history(symbol, hist, row, key)
        if f["high_52w"] and f["low_52w"]:
            st.caption(f"52-week range ₹{f['low_52w']:,.2f} – ₹{f['high_52w']:,.2f} "
                       f"(price is {100 * (f['ltp'] / f['high_52w'] - 1):+.1f}% from the 52-week high). "
                       "Source: NSE cash-market feed (mcp.nseindia.in), about a minute delayed.")
