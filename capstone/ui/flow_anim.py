"""Animated flow charts drawn in the visitor's browser (canvas), so the motion costs the server nothing.

Python only builds a small JSON description (nodes, links, colours, tooltips) from the once-a-minute
NSE data; flow_canvas.html lays it out and animates it. What the motion means:

* band thickness = rupees (traded value, or estimated pressure on the sector-flow chart)
* number of moving dots = rupees too (more money, more dots)
* dot speed = how strong the move is (% change, or how one-sided buying/selling was)
* amber glow = the stock traded at 2.5x or more its usual pace in the last minute (measured by NSE)
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from capstone.ui.common import DOWN, UP

TEMPLATE = Path(__file__).with_name("flow_canvas.html")
NEUTRAL = "#9a988f"
HUB = "#6250d6"
SURGE_AT = 2.5  # last minute's pace vs the day's average pace that counts as a surge
PCT_FULL = 3.0  # % move at which colour and speed reach full strength


def _mix(a: str, b: str, t: float) -> str:
    """Blend two #rrggbb colours: t=0 gives a, t=1 gives b."""
    a_, b_ = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a_, b_))


def pct_color(pct: float | None) -> str:
    if pct is None or pd.isna(pct) or abs(pct) < 0.05:
        return NEUTRAL
    return _mix(NEUTRAL, UP if pct > 0 else DOWN, 0.45 + 0.55 * min(abs(pct) / PCT_FULL, 1))


def _cr(v: float, sign: bool = False) -> str:
    return f"{'+' if sign and v > 0 else '−' if sign and v < 0 else ''}₹{abs(v):,.0f} cr"


def _e(text) -> str:
    return html.escape(str(text))


def _surges(snap: pd.DataFrame, pace: pd.Series | None) -> set[str]:
    if pace is None or pace.empty:
        return set()
    return set(pace[pace >= SURGE_AT].index) & set(snap["symbol"])


def sector_flow_data(snap: pd.DataFrame, sectors: pd.DataFrame, pace: pd.Series | None = None,
                     top: int = 14) -> dict:
    """Sectors (left) -> buyer-driven / seller-driven trading (middle) -> the most traded stocks (right).

    Every stock's traded value is split into an estimated buyer-driven and seller-driven part (see
    market_flow.snapshot), so both sides always have weight, even on a day when nearly everything fell.
    """
    surge = _surges(snap, pace)
    secs = sectors.assign(act=sectors["buy_cr"] + sectors["sell_cr"]).sort_values("act", ascending=False)
    secs = secs[secs["act"] > 0]
    nodes, links = [], []
    for _, r in secs.iterrows():
        bp = 100 * r["buy_cr"] / r["act"]
        nodes.append({"col": 0, "label": r["sector"], "color": _mix(DOWN, UP, bp / 100) if 40 < bp < 60
                      else (UP if bp >= 50 else DOWN),
                      "sub": f"{bp:.0f}% buy",
                      "tip": f"<b>{_e(r['sector'])}</b> ({int(r['stocks'])} stocks, {int(r['up'])} up, "
                             f"{int(r['down'])} down)<br>Traded today: {_cr(r['turnover_cr'])}<br>"
                             f"Buyer-driven (est.): {_cr(r['buy_cr'])} ({bp:.0f}%)<br>"
                             f"Seller-driven (est.): {_cr(r['sell_cr'])} ({100 - bp:.0f}%)"})
    buy_total, sell_total = float(secs["buy_cr"].sum()), float(secs["sell_cr"].sum())
    all_ = buy_total + sell_total or 1
    hub_buy, hub_sell = len(nodes), len(nodes) + 1
    nodes.append({"col": 1, "label": "Buyers in control", "sub": f"{_cr(buy_total)} · {100 * buy_total / all_:.0f}%",
                  "color": UP, "big": True, "subColor": UP,
                  "tip": f"<b>Buyer-driven trading (estimate)</b><br>{_cr(buy_total)}, "
                         f"{100 * buy_total / all_:.0f}% of all traded value"})
    nodes.append({"col": 1, "label": "Sellers in control", "sub": f"{_cr(sell_total)} · {100 * sell_total / all_:.0f}%",
                  "color": DOWN, "big": True, "subColor": DOWN,
                  "tip": f"<b>Seller-driven trading (estimate)</b><br>{_cr(sell_total)}, "
                         f"{100 * sell_total / all_:.0f}% of all traded value"})
    for i, (_, r) in enumerate(secs.iterrows()):
        one_sided = abs(r["buy_cr"] - r["sell_cr"]) / r["act"]
        for hub, val, color, side in ((hub_buy, r["buy_cr"], UP, "buyers"), (hub_sell, r["sell_cr"], DOWN, "sellers")):
            if val > 0:
                links.append({"source": i, "target": hub, "value": float(val), "color": color, "speed": one_sided,
                              "tip": f"<b>{_e(r['sector'])}</b>: {_cr(val)} driven by {side} (estimate)"})
    shown = snap.nlargest(top, "turnover_cr")
    for _, r in shown.iterrows():
        idx = len(nodes)
        hot = r["symbol"] in surge
        bp = 100 * r["buy_cr"] / r["turnover_cr"]
        tip = (f"<b>{_e(r['symbol'])}</b> {_e(r['name'])}<br>Price ₹{r['ltp']:,.2f} ({r['pct_change']:+.2f}% today)"
               f"<br>Traded {_cr(r['turnover_cr'])}: about {bp:.0f}% buyer-driven, {100 - bp:.0f}% seller-driven"
               + (f"<br><b>Trading {pace[r['symbol']]:.1f}× its usual pace this minute</b>" if hot else ""))
        nodes.append({"col": 2, "label": r["symbol"], "sub": f"{bp:.0f}% buy · {r['pct_change']:+.1f}%",
                      "color": UP if bp >= 50 else DOWN, "surge": hot, "tip": tip})
        for hub, val, color in ((hub_buy, r["buy_cr"], UP), (hub_sell, r["sell_cr"], DOWN)):
            if val > 0:
                links.append({"source": hub, "target": idx, "value": float(val), "color": color,
                              "speed": float(abs(r["position"])), "surge": hot, "tip": tip})
    rest = snap[~snap["symbol"].isin(shown["symbol"])]
    if len(rest):
        idx = len(nodes)
        v = float(rest["turnover_cr"].sum())
        bp = 100 * float(rest["buy_cr"].sum()) / v
        tip = f"<b>{len(rest)} other stocks</b><br>Traded {_cr(v)}: about {bp:.0f}% buyer-driven"
        nodes.append({"col": 2, "label": f"Other {len(rest)} stocks", "sub": f"{bp:.0f}% buy", "color": NEUTRAL,
                      "tip": tip})
        for hub, val, color in ((hub_buy, rest["buy_cr"].sum(), UP), (hub_sell, rest["sell_cr"].sum(), DOWN)):
            if val > 0:
                links.append({"source": hub, "target": idx, "value": float(val), "color": color,
                              "speed": float(rest["position"].abs().mean()), "tip": tip})
    legend = [{"color": UP, "text": "Buyers in control"}, {"color": DOWN, "text": "Sellers in control"},
              {"text": "Thickness and dots = ₹ (estimate)"}, {"text": "Faster dots = more one-sided"}]
    if pace is not None and len(pace):
        legend.append({"color": "#eda100", "text": "Glow = trading 2.5×+ its usual pace this minute"})
    rows = max(len(secs), top + 1)
    return {"nodes": nodes, "links": links, "legend": legend, "estimate": True,
            "height": max(480, 32 * rows), "padLeft": 220, "padRight": 210, "gap": 8,
            "note": "Each stock's traded money is split into a buyer-driven and a seller-driven part, depending on "
                    "where its price sits in today's range. Left: sectors. Middle: the two totals. Right: the most "
                    "traded stocks. Hover anything for details."}


def volume_flow_data(snap: pd.DataFrame, pace: pd.Series | None = None, top: int = 20) -> dict:
    """All stocks (left) -> sectors (middle) -> the most traded stocks (right). Measured traded value."""
    surge = _surges(snap, pace)
    total = float(snap["turnover_cr"].sum())
    keep = set(snap.nlargest(top, "turnover_cr")["symbol"])
    nodes = [{"col": 0, "label": "All stocks", "sub": _cr(total), "color": HUB, "big": True, "align": "right",
              "tip": f"<b>{len(snap)} Nifty 50 and Next 50 stocks</b><br>Traded today: {_cr(total)}"}]
    links = []
    order = snap.groupby("sector")["turnover_cr"].sum().sort_values(ascending=False)
    stock_nodes = []
    for sec, tv in order.items():
        part = snap[snap["sector"] == sec]
        pct = float((part["pct_change"] * part["turnover_cr"]).sum() / tv)
        si = len(nodes)
        nodes.append({"col": 1, "label": sec, "sub": f"{_cr(tv)} · {pct:+.2f}%", "color": pct_color(pct),
                      "tip": f"<b>{_e(sec)}</b><br>Traded {_cr(tv)} ({100 * tv / total:.0f}% of all)<br>"
                             f"% change (weighted): {pct:+.2f}%"})
        links.append({"source": 0, "target": si, "value": float(tv), "color": pct_color(pct), "colorFrom": HUB,
                      "speed": min(abs(pct) / PCT_FULL, 1), "tip": f"<b>{_e(sec)}</b>: {_cr(tv)} traded"})
        shown = part[part["symbol"].isin(keep)].sort_values("turnover_cr", ascending=False)
        for _, r in shown.iterrows():
            stock_nodes.append((si, r))
        rest = part[~part["symbol"].isin(keep)]
        if len(rest):
            v = float(rest["turnover_cr"].sum())
            rp = float((rest["pct_change"] * rest["turnover_cr"]).sum() / v)
            stock_nodes.append((si, {"symbol": f"{len(rest)} other", "turnover_cr": v, "pct_change": rp,
                                     "other": True}))
    for si, r in stock_nodes:
        idx = len(nodes)
        hot = not r.get("other") and r["symbol"] in surge
        c = pct_color(r["pct_change"])
        tip = (f"<b>{_e(r['symbol'])}</b>" + ("" if r.get("other") else f" {_e(r['name'])}<br>Price ₹{r['ltp']:,.2f}")
               + f"<br>{r['pct_change']:+.2f}% today · traded {_cr(r['turnover_cr'])}"
               + (f"<br><b>Trading {pace[r['symbol']]:.1f}× its usual pace this minute</b>" if hot else ""))
        nodes.append({"col": 2, "label": r["symbol"], "sub": f"{_cr(r['turnover_cr'])} · {r['pct_change']:+.1f}%",
                      "color": NEUTRAL if r.get("other") else c, "surge": hot, "tip": tip})
        links.append({"source": si, "target": idx, "value": float(r["turnover_cr"]), "color": c,
                      "speed": min(abs(r["pct_change"]) / PCT_FULL, 1), "surge": hot, "tip": tip})
    legend = [{"color": UP, "text": "Up today"}, {"color": DOWN, "text": "Down today"},
              {"text": "Thickness and dots = ₹ traded (measured)"}, {"text": "Faster dots = bigger % move"}]
    if pace is not None and len(pace):
        legend.append({"color": "#eda100", "text": "Glow = trading 2.5×+ its usual pace this minute"})
    rows = sum(n["col"] == 2 for n in nodes)
    return {"nodes": nodes, "links": links, "legend": legend, "estimate": False,
            "height": max(520, 22 * rows), "padLeft": 110, "padRight": 250, "gap": 5,
            "note": "Today's traded money splitting into sectors, then into the busiest stocks. "
                    "Hover anything for details."}


def render(data: dict) -> None:
    """Draw one animated chart. The iframe gets the data once; all motion runs in the browser."""
    try:
        data["dark"] = st.context.theme.type == "dark"
    except Exception:
        data["dark"] = False
    page = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    components.html(page, height=data["height"] + 90, scrolling=False)
