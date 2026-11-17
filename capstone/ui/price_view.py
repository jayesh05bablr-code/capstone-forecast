"""Interactive price chart with controls and a key-numbers strip (used on the Morning forecast tab)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import charts, indicators as ind

TIPS = ("Scroll to zoom, drag to pan, double-click to reset. Use the pen buttons in the chart's top-right "
        "toolbar to draw trend lines or boxes (the eraser removes them) and the camera to save a PNG.")


CARD = ("<div style='border:1px solid rgba(128,128,128,0.25);border-radius:8px;padding:8px 12px'>"
        "<div style='font-size:12px;opacity:0.7'>{label}</div>"
        "<div style='font-size:19px;font-weight:600;white-space:nowrap'>{value}</div>"
        "<div style='font-size:12px;color:{color}'>{note}</div></div>")
GOOD, BAD, NEUTRAL = "#089981", "#f23645", "rgba(128,128,128,0.9)"


def _cards(snap: dict) -> list[dict]:
    ch, vr, r = snap["change_pct"], snap["volume_vs_avg"], snap["rsi"]
    ok = lambda v: v == v  # noqa: E731  (not NaN)
    return [
        dict(label="Last close", value=f"₹{snap['close']:,.2f}",
             note=f"{snap['change']:+,.2f} ({ch:+.2f}%) on the day" if ok(ch) else "",
             color=GOOD if ok(ch) and ch >= 0 else BAD),
        dict(label="Day range", value=f"{snap['day_low']:,.1f} – {snap['day_high']:,.1f}",
             note=f"spread {(snap['day_high'] / snap['day_low'] - 1) * 100:.2f}%", color=NEUTRAL),
        dict(label="52-week range", value=f"{snap['wk52_low']:,.0f} – {snap['wk52_high']:,.0f}",
             note=f"{snap['from_52w_high_pct']:+.1f}% from the 52-week high", color=NEUTRAL),
        dict(label="Volume vs 20-day avg", value=f"{vr:.2f}×" if ok(vr) else "–",
             note=("heavier than usual" if vr > 1.2 else "lighter than usual" if vr < 0.8 else "normal")
             if ok(vr) else "", color=NEUTRAL),
        dict(label="RSI 14", value=f"{r:.0f}" if ok(r) else "–",
             note=("overbought (above 70)" if r > 70 else "oversold (below 30)" if r < 30 else "neutral (30–70)")
             if ok(r) else "", color=BAD if ok(r) and r > 70 else GOOD if ok(r) and r < 30 else NEUTRAL),
        dict(label="Typical daily move (ATR 14)", value=f"{snap['atr_pct']:.2f}%",
             note=f"≈ ₹{snap['atr_pct'] / 100 * snap['close']:,.1f} a day", color=NEUTRAL),
    ]


def _metric_strip(snap: dict) -> None:
    cards = "".join(CARD.format(**c) for c in _cards(snap))
    st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;"
                f"margin:4px 0 8px'>{cards}</div>", unsafe_allow_html=True)
    st.caption(f"**Trend:** {ind.trend_label(snap)} · 20-day avg ₹{snap['sma20']:,.2f} · 50-day ₹{snap['sma50']:,.2f}"
               f" · 200-day ₹{snap['sma200']:,.2f} · data to {pd.Timestamp(snap['date']):%a %d %b %Y}")


def render(symbol: str, hist: pd.DataFrame, forecast: dict | None = None, key: str = "pv") -> None:
    actions = ind.corporate_actions(symbol)
    top = st.columns([4, 2, 2, 2])
    period = top[0].segmented_control("Period", list(ind.PERIODS), default="6M", key=f"{key}_period") or "6M"
    chart_style = top[1].selectbox("Chart type", charts.STYLES, key=f"{key}_style")
    log = top[2].toggle("Log scale", key=f"{key}_log", help="Equal % moves look equal; useful for 3Y-All.")
    adjust = top[3].toggle("Adjust for bonus/split", value=True, key=f"{key}_adj",
                           help="Scales prices before each bonus/split ex-date so the history has no fake drop "
                                f"({len(actions)} known for {symbol}).")
    mid = st.columns([3, 2])
    overlays = mid[0].multiselect("Overlays", charts.OVERLAYS, default=["SMA 20", "SMA 50"], key=f"{key}_ovl")
    panels = mid[1].multiselect("Lower panels", charts.PANELS, default=["Volume", "RSI 14"], key=f"{key}_pan")

    df = ind.adjust_for_actions(hist, actions) if adjust else hist.sort_values("date")
    _metric_strip(ind.snapshot(df))
    fig = charts.price_chart(df, f"{symbol} · daily", period=period, chart_style=chart_style, overlays=overlays,
                             panels=panels, log=log, forecast=forecast)
    st.plotly_chart(fig, use_container_width=True, config=charts.CONFIG, key=f"{key}_chart")
    notes = [TIPS]
    if forecast:
        notes.insert(0, "◆ = expected open; the vertical bar is the 10th–90th percentile range.")
    if adjust and len(actions):
        acts = ", ".join(f"{a.action} on {a.ex_date:%d %b %Y}" for a in actions.itertuples())
        notes.append(f"Adjusted for {acts}.")
    st.caption(" ".join(notes))
