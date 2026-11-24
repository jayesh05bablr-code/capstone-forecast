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
        "turnover_cr": float(r["turnover_cr"]), "volume": float(r["volume"]),
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
    return out


def render(snap: pd.DataFrame, symbol: str, forecast_symbols: list[str] | None = None, key: str = "sp") -> None:
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
        c[1].metric("Previous close", f"₹{f['prev_close']:,.2f}")
        c[2].metric("Traded value", f"₹{f['turnover_cr']:,.0f} cr")
        c[3].metric("Volume", f"{f['volume']:,.0f}")
        c = st.columns(4)
        c[0].metric("VWAP", f"₹{f['vwap']:,.2f}" if f["vwap"] else "n/a",
                    f"{f['vs_vwap']:+.2f}% price vs VWAP" if f["vs_vwap"] is not None else None,
                    help="Volume-weighted average price of the session: traded value ÷ volume.")
        c[1].metric("Day high", f"₹{f['high']:,.2f}")
        c[2].metric("Day low", f"₹{f['low']:,.2f}")
        c[3].metric("Vs its sector today", f"{f['vs_sector']:+.2f} pts",
                    help=f"Stock's % change minus its sector's value-weighted % change ({f['sector_pct']:+.2f}%).")
        bits = []
        if f["buy_share"] is not None:
            bits.append(f"**Estimated buyer-driven share: {100 * f['buy_share']:.0f}%** (seller-driven "
                        f"{100 - 100 * f['buy_share']:.0f}%), from where the price sits in the day's range"
                        + (f" ({100 * f['range_pos']:.0f}% of the way up)" if f["range_pos"] is not None else "")
                        + ". An estimate, not reported order flow.")
        st.markdown(" ".join(bits))
        if f["high_52w"] and f["low_52w"]:
            st.caption(f"52-week range ₹{f['low_52w']:,.2f} – ₹{f['high_52w']:,.2f} "
                       f"(price is {100 * (f['ltp'] / f['high_52w'] - 1):+.1f}% from the 52-week high). "
                       "Source: NSE cash-market feed (mcp.nseindia.in), about a minute delayed.")
