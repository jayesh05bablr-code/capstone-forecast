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
                     top: int = 8) -> dict:
    """Sectors (left) -> buying / selling pressure (middle) -> the stocks with the most of each (right)."""
    surge = _surges(snap, pace)
    secs = sectors.assign(act=sectors["buy_cr"] + sectors["sell_cr"]).sort_values("act", ascending=False)
    secs = secs[secs["act"] > 0]
    nodes, links = [], []
    for _, r in secs.iterrows():
        share = (r["buy_cr"] - r["sell_cr"]) / r["act"]
        nodes.append({"col": 0, "label": r["sector"], "color": UP if share >= 0 else DOWN,
                      "sub": f"+₹{r['buy_cr']:,.0f} / −₹{r['sell_cr']:,.0f} cr",
                      "tip": f"<b>{_e(r['sector'])}</b> ({int(r['stocks'])} stocks, {int(r['up'])} up, "
                             f"{int(r['down'])} down)<br>Buying pressure (est.): {_cr(r['buy_cr'])}<br>"
                             f"Selling pressure (est.): {_cr(r['sell_cr'])}<br>Traded today: "
                             f"{_cr(r['turnover_cr'])}"})
    buy_total, sell_total = float(secs["buy_cr"].sum()), float(secs["sell_cr"].sum())
    hub_buy, hub_sell = len(nodes), len(nodes) + 1
    nodes.append({"col": 1, "label": "Buying pressure", "sub": _cr(buy_total), "color": UP, "big": True,
                  "subColor": UP, "tip": f"<b>Estimated buying pressure</b><br>{_cr(buy_total)} across "
                                         f"{int((snap['net_flow_cr'] > 0).sum())} stocks"})
    nodes.append({"col": 1, "label": "Selling pressure", "sub": _cr(sell_total), "color": DOWN, "big": True,
                  "subColor": DOWN, "tip": f"<b>Estimated selling pressure</b><br>{_cr(sell_total)} across "
                                           f"{int((snap['net_flow_cr'] < 0).sum())} stocks"})
    for i, (_, r) in enumerate(secs.iterrows()):
        part = snap[snap["sector"] == r["sector"]]
        for hub, val, color, side in ((hub_buy, r["buy_cr"], UP, "buying"), (hub_sell, r["sell_cr"], DOWN, "selling")):
            if val <= 0:
                continue
            mask = part["net_flow_cr"] > 0 if side == "buying" else part["net_flow_cr"] < 0
            strength = float(part.loc[mask, "position"].abs().mean()) if mask.any() else 0.0
            links.append({"source": i, "target": hub, "value": float(val), "color": color, "speed": strength,
                          "tip": f"<b>{_e(r['sector'])} → {side}</b><br>{_cr(val)} (estimate)"})
    for hub, side, color in ((hub_buy, "buy", UP), (hub_sell, "sell", DOWN)):
        pool = snap[snap["net_flow_cr"] > 0] if side == "buy" else snap[snap["net_flow_cr"] < 0]
        pool = pool.reindex(pool["net_flow_cr"].abs().sort_values(ascending=False).index)
        for _, r in pool.head(top).iterrows():
            idx = len(nodes)
            hot = r["symbol"] in surge
            nodes.append({"col": 2, "label": r["symbol"], "sub": f"{_cr(r['net_flow_cr'], True)} · {r['pct_change']:+.2f}%",
                          "color": color, "surge": hot,
                          "tip": f"<b>{_e(r['symbol'])}</b> {_e(r['name'])}<br>Price ₹{r['ltp']:,.2f} "
                                 f"({r['pct_change']:+.2f}% today)<br>Traded {_cr(r['turnover_cr'])}, estimated "
                                 f"{'buying' if side == 'buy' else 'selling'} {_cr(abs(r['net_flow_cr']))}"
                                 + (f"<br><b>Trading {pace[r['symbol']]:.1f}× its usual pace this minute</b>"
                                    if hot else "")})
            links.append({"source": hub, "target": idx, "value": float(abs(r["net_flow_cr"])), "color": color,
                          "speed": float(abs(r["position"])), "surge": hot,
                          "tip": f"<b>{_e(r['symbol'])}</b>: {_cr(abs(r['net_flow_cr']))} {side}ing pressure (est.)"})
        rest = pool.iloc[top:]
        if len(rest):
            idx = len(nodes)
            v = float(rest["net_flow_cr"].abs().sum())
            nodes.append({"col": 2, "label": f"Other {len(rest)} stocks", "sub": _cr(v), "color": NEUTRAL,
                          "tip": f"<b>{len(rest)} other stocks</b> with net {side}ing pressure: {_cr(v)}"})
            links.append({"source": hub, "target": idx, "value": v, "color": color,
                          "speed": float(rest["position"].abs().mean()),
                          "tip": f"Other {len(rest)} stocks: {_cr(v)} (estimate)"})
    legend = [{"color": UP, "text": "Buying pressure"}, {"color": DOWN, "text": "Selling pressure"},
              {"text": "Thickness and dots = ₹ (estimate)"}, {"text": "Faster dots = more one-sided"}]
    if surge or pace is not None and len(pace):
        legend.append({"color": "#eda100", "text": "Glow = trading 2.5×+ its usual pace this minute"})
    rows = max(len(secs), sum(n["col"] == 2 for n in nodes))
    return {"nodes": nodes, "links": links, "legend": legend, "estimate": True,
            "height": max(460, 30 * rows), "padLeft": 250, "padRight": 250, "gap": 8,
            "note": "Left: each sector's estimated buying (+) and selling (−) in ₹ crore. Middle: the totals. "
                    "Right: the stocks with the most of each. Hover anything for details."}


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
