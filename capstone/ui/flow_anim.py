"""Animated flow charts drawn in the visitor's browser (canvas), so the motion costs the server nothing.

Python only builds a small JSON description (nodes, links, colours, tooltips) from the once-a-minute
NSE data; flow_component/index.html lays it out and animates it, and sends a click on a sector or a
stock back to Python for the drill-down. What the motion means:

* band thickness = rupees (traded value, or estimated buyer/seller-driven value on the control chart)
* number of moving dots = rupees too (more money, more dots)
* dot speed = how strong the move is (% change, or how one-sided buying/selling was)
* amber glow = the stock traded at 2.5x or more its usual pace in the last minute (measured by NSE)

Two charts:

* Buyer vs seller control (ESTIMATE): sectors -> buyers in control / sellers in control -> stocks
* Capital distribution (MEASURED): all stocks -> sectors -> stocks, band = traded value in ₹ crore
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from capstone.ui.common import DOWN, UP

COMPONENT_DIR = Path(__file__).with_name("flow_component")
NEUTRAL = "#8b93a3"
HUB = "#6a5cff"
UP_TEXT, DOWN_TEXT = "#5aa9ff", "#ff6b6a"  # lighter tints of UP/DOWN that stay readable on the dark panel
SURGE_AT = 2.5  # last minute's pace vs the day's average pace that counts as a surge
PCT_FULL = 3.0  # % move at which colour and speed reach full strength
FLAT = 0.05  # |% change| below this counts as unchanged (neutral colour)

_component = components.declare_component("pulse8_flow", path=str(COMPONENT_DIR))


def _mix(a: str, b: str, t: float) -> str:
    """Blend two #rrggbb colours: t=0 gives a, t=1 gives b."""
    a_, b_ = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a_, b_))


def pct_color(pct: float | None) -> str:
    """Blue when up, red when down, grey when flat; deeper colour for a bigger move."""
    if pct is None or pd.isna(pct) or abs(pct) < FLAT:
        return NEUTRAL
    return _mix(NEUTRAL, UP if pct > 0 else DOWN, 0.45 + 0.55 * min(abs(pct) / PCT_FULL, 1))


def buy_color(buy_pct: float) -> str:
    """Blue when buyers drove most of the trading, red when sellers did, a blend near 50/50."""
    if 40 < buy_pct < 60:
        return _mix(DOWN, UP, buy_pct / 100)
    return UP if buy_pct >= 50 else DOWN


def _cr(v: float, sign: bool = False) -> str:
    return f"{'+' if sign and v > 0 else '−' if sign and v < 0 else ''}₹{abs(v):,.0f} cr"


# How band sizes are written for each flow metric (see market_flow.with_metric).
UNITS = {"value": "₹ crore traded", "volume": "lakh shares traded", "pressure": "₹ crore of net pressure (est.)"}


def fmt(v: float, unit: str = "value", sign: bool = False) -> str:
    """A band size in the chart's unit: ₹ crore, or lakh shares for the volume metric."""
    if unit == "volume":
        s = '+' if sign and v > 0 else '−' if sign and v < 0 else ''
        return f"{s}{abs(v):,.1f} L sh" if abs(v) < 100 else f"{s}{abs(v):,.0f} L sh"
    return _cr(v, sign)


def _e(text) -> str:
    return html.escape(str(text))


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def _surges(snap: pd.DataFrame, pace: pd.Series | None) -> set[str]:
    if pace is None or pace.empty:
        return set()
    return set(pace[pace >= SURGE_AT].index) & set(snap["symbol"])


def _weighted_pct(df: pd.DataFrame) -> float:
    """% change weighted by traded value (the real ₹ value, whichever metric sizes the bands)."""
    w = df["value_cr"] if "value_cr" in df else df["turnover_cr"]
    tv = float(w.sum())
    return float((df["pct_change"] * w).sum() / tv) if tv else 0.0


def stock_tip(r, pace: pd.Series | None = None, hot: bool = False) -> str:
    """Hover text for one stock: price, change, traded value, VWAP, range and the buy/sell split."""
    value = r["value_cr"] if "value_cr" in r and pd.notna(r.get("value_cr")) else r["turnover_cr"]
    pos = r.get("position", 0) if pd.notna(r.get("position", 0)) else 0
    bp = 50 * (1 + pos)
    lines = [f"<b>{_e(r['symbol'])}</b> · {_e(r.get('name', ''))} · {_e(r.get('sector', ''))}",
             f"Price ₹{r['ltp']:,.2f} ({r['pct_change']:+.2f}% today)"]
    if _num(r.get("prev_close")):
        lines.append(f"Previous close ₹{r['prev_close']:,.2f} · Day range ₹{r['low']:,.2f} – ₹{r['high']:,.2f}")
    lines.append(f"Traded value {_cr(value)} · Volume {r['volume']:,.0f}")
    if _num(r.get("rvol")):
        lines.append(f"Relative volume {r['rvol']:.2f}× its 20-session average")
    if _num(r.get("vwap")):
        lines.append(f"VWAP ₹{r['vwap']:,.2f} (price {100 * (r['ltp'] / r['vwap'] - 1):+.2f}% vs VWAP)")
    lines.append(f"Estimated split: {bp:.0f}% buyer-driven, {100 - bp:.0f}% seller-driven")
    if hot and pace is not None:
        lines.append(f"<b>Trading {pace[r['symbol']]:.1f}× its usual pace this minute</b>")
    return "<br>".join(lines)


def sector_flow_data(snap: pd.DataFrame, sectors: pd.DataFrame, pace: pd.Series | None = None,
                     top: int = 14, focus: str | None = None, unit: str = "value", rank_by: str = "turnover_cr",
                     title_note: str = "", period_word: str = "today") -> dict:
    """Sectors (left) -> buyers / sellers in control (middle) -> the most traded stocks (right). ESTIMATE.

    Every stock's traded value is split into an estimated buyer-driven and seller-driven part (see
    market_flow.snapshot), so both sides always have weight, even on a day when nearly everything fell.
    With focus=<sector>, only that sector is drawn and every one of its stocks is named.
    """
    def cr(v, sign=False):
        return fmt(v, unit, sign)
    if focus:
        snap = snap[snap["sector"] == focus]
        sectors = sectors[sectors["sector"] == focus]
        top = len(snap)
    surge = _surges(snap, pace)
    secs = sectors.assign(act=sectors["buy_cr"] + sectors["sell_cr"]).sort_values("act", ascending=False)
    secs = secs[secs["act"] > 0]
    nodes, links = [], []
    for _, r in secs.iterrows():
        bp = 100 * r["buy_cr"] / r["act"]
        nodes.append({"col": 0, "label": r["sector"], "sub": f"{bp:.0f}% buy", "chg": float(r["pct_change"]),
                      "color": buy_color(bp), "kind": "sector", "id": r["sector"],
                      "tip": f"<b>{_e(r['sector'])}</b> ({int(r['stocks'])} stocks: {int(r['up'])} up, "
                             f"{int(r['down'])} down)<br>Traded value {cr(r['turnover_cr'])} · "
                             f"price change {r['pct_change']:+.2f}% (value-weighted)<br>"
                             f"Buyer-driven (est.) {cr(r['buy_cr'])} ({bp:.0f}%)<br>"
                             f"Seller-driven (est.) {cr(r['sell_cr'])} ({100 - bp:.0f}%)"})
    buy_total, sell_total = float(secs["buy_cr"].sum()), float(secs["sell_cr"].sum())
    all_ = buy_total + sell_total or 1
    hub_buy, hub_sell = len(nodes), len(nodes) + 1
    nodes.append({"col": 1, "label": "Buyers in control", "sub": f"{cr(buy_total)} · {100 * buy_total / all_:.0f}%",
                  "color": UP, "big": True, "subColor": UP_TEXT,
                  "tip": f"<b>Buyer-driven trading (estimate)</b><br>{cr(buy_total)}, "
                         f"{100 * buy_total / all_:.0f}% of all traded value"})
    nodes.append({"col": 1, "label": "Sellers in control", "sub": f"{cr(sell_total)} · {100 * sell_total / all_:.0f}%",
                  "color": DOWN, "big": True, "subColor": DOWN_TEXT,
                  "tip": f"<b>Seller-driven trading (estimate)</b><br>{cr(sell_total)}, "
                         f"{100 * sell_total / all_:.0f}% of all traded value"})
    for i, (_, r) in enumerate(secs.iterrows()):
        one_sided = abs(r["buy_cr"] - r["sell_cr"]) / r["act"]
        for hub, val, color, side in ((hub_buy, r["buy_cr"], UP, "buyers"), (hub_sell, r["sell_cr"], DOWN, "sellers")):
            if val > 0:
                links.append({"source": i, "target": hub, "value": float(val), "color": color, "speed": one_sided,
                              "tip": f"<b>{_e(r['sector'])}</b> → {side} in control<br>{cr(val)} "
                                     f"({100 * val / r['act']:.0f}% of the sector, estimate)"})
    shown = snap.nlargest(top, rank_by if rank_by in snap else "turnover_cr")
    for _, r in shown.iterrows():
        idx = len(nodes)
        hot = r["symbol"] in surge
        bp = 100 * r["buy_cr"] / r["turnover_cr"]
        tip = stock_tip(r, pace, hot)
        sub = f"net {cr(r['buy_cr'] - r['sell_cr'], True)}" if unit == "pressure" else f"{bp:.0f}% buy"
        nodes.append({"col": 2, "label": r["symbol"], "sub": sub, "chg": float(r["pct_change"]),
                      "chgDp": 1, "color": buy_color(bp), "surge": hot, "tip": tip, "kind": "stock", "id": r["symbol"]})
        for hub, val, color, side in ((hub_buy, r["buy_cr"], UP, "buyers"), (hub_sell, r["sell_cr"], DOWN, "sellers")):
            if val > 0:
                links.append({"source": hub, "target": idx, "value": float(val), "color": color,
                              "speed": float(abs(r["position"])), "surge": hot,
                              "tip": f"<b>{_e(r['symbol'])}</b>: {cr(val)} {side}-driven (estimate)<br>"
                                     f"Price {r['pct_change']:+.2f}% today"})
    rest = snap[~snap["symbol"].isin(shown["symbol"])]
    if len(rest):
        idx = len(nodes)
        v = float(rest["turnover_cr"].sum())
        bp = 100 * float(rest["buy_cr"].sum()) / v
        tip = (f"<b>{len(rest)} other stocks</b><br>Traded {cr(v)}: about {bp:.0f}% buyer-driven<br>"
               f"Price change {_weighted_pct(rest):+.2f}% (value-weighted)")
        nodes.append({"col": 2, "label": f"Other {len(rest)} stocks", "sub": f"{bp:.0f}% buy",
                      "chg": _weighted_pct(rest), "chgDp": 1, "color": NEUTRAL, "tip": tip})
        for hub, val, color in ((hub_buy, rest["buy_cr"].sum(), UP), (hub_sell, rest["sell_cr"].sum(), DOWN)):
            if val > 0:
                links.append({"source": hub, "target": idx, "value": float(val), "color": color,
                              "speed": float(rest["position"].abs().mean()), "tip": tip})
    legend = [{"color": UP, "text": "Buyer-driven (est.)"}, {"color": DOWN, "text": "Seller-driven (est.)"},
              {"text": f"Thickness = {UNITS[unit]}"}, {"text": "Dot speed = how one-sided"}]
    if pace is not None and len(pace):
        legend.append({"color": "#f2b84b", "text": "Glow = 2.5×+ usual pace this minute"})
    rows = max(len(secs), min(top, len(snap)) + 1)
    return {"title": "BUYER VS SELLER CONTROL" + (f" · {focus.upper()}" if focus else "") + title_note,
            "nodes": nodes, "links": links, "legend": legend, "estimate": True, "focus": focus,
            "height": max(460, 31 * rows), "padLeft": 230, "padRight": 230, "gap": 7,
            "upText": UP_TEXT, "downText": DOWN_TEXT,
            "note": "Each stock's traded value is split into a buyer-driven and a seller-driven part by where its "
                    "price sits in the day's range" + ("" if period_word == "today" else ", day by day, added up") +
                    " (an <b>estimate</b> from prices, not reported institutional orders). "
                    f"% next to each name is the price change {period_word}. Click a sector to see all its stocks, or a "
                    "stock for its panel."}


def volume_flow_data(snap: pd.DataFrame, pace: pd.Series | None = None, top: int = 20,
                     focus: str | None = None, unit: str = "value", rank_by: str = "turnover_cr",
                     title_note: str = "", period_word: str = "today") -> dict:
    """All stocks (left) -> sectors (middle) -> the most traded stocks (right). MEASURED traded value.

    With focus=<sector>, the chart becomes that sector (left) -> every one of its stocks (right).
    """
    def cr(v, sign=False):
        return fmt(v, unit, sign)
    surge = _surges(snap, pace)
    if focus:
        snap = snap[snap["sector"] == focus]
        top = len(snap)
    total = float(snap["turnover_cr"].sum())
    keep = set(snap.nlargest(top, rank_by if rank_by in snap else "turnover_cr")["symbol"])
    nodes, links, stock_nodes = [], [], []
    if not focus:
        nodes.append({"col": 0, "label": "All stocks", "sub": cr(total), "chg": _weighted_pct(snap),
                      "color": HUB, "big": True, "align": "right",
                      "tip": f"<b>{len(snap)} stocks in this view</b><br>Traded value {cr(total)}<br>"
                             f"Price change {_weighted_pct(snap):+.2f}% (value-weighted)"})
    order = snap.groupby("sector")["turnover_cr"].sum().sort_values(ascending=False)
    col_sec = 0 if focus else 1
    for sec, tv in order.items():
        part = snap[snap["sector"] == sec]
        pct = _weighted_pct(part)
        up, down = int((part["pct_change"] > 0).sum()), int((part["pct_change"] < 0).sum())
        si = len(nodes)
        sc = buy_color(100 * part["buy_cr"].sum() / tv) if unit == "pressure" else pct_color(pct)
        nodes.append({"col": col_sec, "label": sec, "sub": cr(tv), "chg": pct, "color": sc,
                      "kind": None if focus else "sector", "id": sec, "big": bool(focus),
                      "tip": f"<b>{_e(sec)}</b> ({len(part)} stocks: {up} up, {down} down)<br>Traded value "
                             f"{cr(tv)} ({100 * tv / total:.1f}% of this view)<br>"
                             f"Price change {pct:+.2f}% (value-weighted)"})
        if not focus:
            links.append({"source": 0, "target": si, "value": float(tv), "color": sc, "colorFrom": HUB,
                          "speed": min(abs(pct) / PCT_FULL, 1),
                          "tip": f"<b>{_e(sec)}</b>: {cr(tv)} traded, {100 * tv / total:.1f}% of all<br>"
                                 f"{pct:+.2f}% today"})
        shown = part[part["symbol"].isin(keep)].sort_values("turnover_cr", ascending=False)
        for _, r in shown.iterrows():
            stock_nodes.append((si, r))
        rest = part[~part["symbol"].isin(keep)]
        if len(rest):
            v = float(rest["turnover_cr"].sum())
            stock_nodes.append((si, {"symbol": f"{len(rest)} other", "turnover_cr": v,
                                     "buy_cr": float(rest["buy_cr"].sum()),
                                     "pct_change": _weighted_pct(rest), "other": True, "names": list(rest["symbol"])}))
    for si, r in stock_nodes:
        idx = len(nodes)
        other = bool(r.get("other"))
        hot = not other and r["symbol"] in surge
        c = buy_color(100 * r["buy_cr"] / r["turnover_cr"]) if unit == "pressure" else pct_color(r["pct_change"])
        if other:
            names = ", ".join(r["names"][:12]) + ("…" if len(r["names"]) > 12 else "")
            tip = (f"<b>{_e(r['symbol'])} stocks</b> in {_e(nodes[si]['label'])}<br>{_e(names)}<br>"
                   f"Traded {cr(r['turnover_cr'])} · {r['pct_change']:+.2f}% (value-weighted)")
        else:
            tip = stock_tip(r, pace, hot)
        nodes.append({"col": col_sec + 1, "label": r["symbol"], "sub": cr(r["turnover_cr"]), "chg": float(r["pct_change"]),
                      "chgDp": 1, "color": NEUTRAL if other else c, "surge": hot, "tip": tip,
                      "kind": "sector" if other else "stock", "id": nodes[si]["label"] if other else r["symbol"]})
        links.append({"source": si, "target": idx, "value": float(r["turnover_cr"]), "color": c,
                      "speed": min(abs(r["pct_change"]) / PCT_FULL, 1), "surge": hot, "tip": tip})
    legend = ([{"color": UP, "text": "Net buying (est.)"}, {"color": DOWN, "text": "Net selling (est.)"}]
              if unit == "pressure" else
              [{"color": UP, "text": "Price up"}, {"color": DOWN, "text": "Price down"}, {"color": NEUTRAL, "text": "Flat"}]) + [{"text": f"Thickness = {UNITS[unit]}"},
              {"text": "Dot speed = size of % move"}]
    if pace is not None and len(pace):
        legend.append({"color": "#f2b84b", "text": "Glow = 2.5×+ usual pace this minute"})
    rows = sum(n["col"] == col_sec + 1 for n in nodes)
    return {"title": "CAPITAL DISTRIBUTION" + (f" · {focus.upper()}" if focus else "") + title_note,
            "nodes": nodes, "links": links, "legend": legend, "estimate": unit == "pressure", "focus": focus,
            "height": max(500, 23 * rows), "padLeft": 120 if not focus else 200, "padRight": 230, "gap": 5,
            "upText": UP_TEXT, "downText": DOWN_TEXT,
            "note": {"value": "Traded value (turnover reported by NSE)", "volume": "Traded volume (shares, reported by NSE)",
                     "pressure": "Estimated net buying or selling pressure (an <b>estimate</b> from prices)"}[unit]
                    + " splitting into sectors, then into the busiest stocks. "
                    f"Labels show the band size and the price change {period_word}. Click a sector (or an 'other' group) to "
                    "see all its stocks, or a stock for its panel."}


def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (bool, str)) or v is None:
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return None if f != f or f in (float("inf"), float("-inf")) else (int(f) if isinstance(v, int) else f)


def render(data: dict, key: str, dark: bool | None = None) -> dict | None:
    """Draw one animated chart and return a new click ({kind, id}) once, else None.

    The browser keeps animating between reruns; it only re-lays out when the chart data changes.
    """
    if dark is None:
        dark = st.session_state.get("terminal_theme", "Dark") == "Dark"
    data = {**data, "dark": dark}
    data = _clean(data)  # plain JSON types only: numpy numbers become floats, NaN becomes null
    value = _component(data=data, key=key, default=None)
    if not value or value.get("nonce") == st.session_state.get(f"{key}_nonce"):
        return None
    st.session_state[f"{key}_nonce"] = value.get("nonce")
    return {"kind": value.get("kind"), "id": value.get("id")}
