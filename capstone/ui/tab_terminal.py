"""Market terminal: the page the app opens on. Everything about the market in one screen.

Layout:
  index tiles     NIFTY 50, BANK, IT, FIN SERVICES, MIDCAP 100, INDIA VIX (NSE live feed, day-range bars)
  row 1           Heat map (sector > stock; size = traded value or equal; colour = % change) | Market pulse:
                  breadth, VWAP and 52-week counts, top gainers / losers, most active by value
  row 2           NSE sector indices | Next-open forecast (Pulse8 model) | 52-week watch + latest FII/DII
  row 3           Market watch: every stock of the chosen universe, sortable

Prices are NSE's own feed (1-3 minutes behind the exchange); outside market hours they are the last
session's close and each card says so. Breadth and VWAP figures are counts from the same feed. The
FII/DII line is NSE's reported figure from the Institutional activity page's saved copy, never estimated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from capstone import charts, institutional as inst, market_flow
from capstone.charts import CONFIG
from capstone.ui import tab_live
from capstone.ui.common import DOWN, UP
from capstone.ui.tab_overview import CSS, _e, diverging_bars, split_bar, src
from capstone.ui.terminal import live_indices

STATIC = {**CONFIG, "displayModeBar": False}
PCT_RANGE = 3.0  # heat-map colour saturates at ±3%
NEAR = 2.0  # "near a 52-week high/low" = within 2%

CSS_T = """
<style>
.p8-kv{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:8px 0 4px}
.p8-kv>div{border:1px solid rgba(128,128,128,.18);border-radius:8px;padding:6px 8px}
.p8-kv .k{font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;opacity:.62;font-weight:600}
.p8-kv .v{font-size:.94rem;font-weight:700;white-space:nowrap;font-variant-numeric:tabular-nums}
.p8-two{display:grid;grid-template-columns:1fr 1fr;gap:10px}
table.p8-m{width:100%;border-collapse:collapse;font-size:.8rem;font-variant-numeric:tabular-nums}
table.p8-m td,table.p8-m th{padding:4px 4px;border-bottom:1px solid rgba(128,128,128,.13);white-space:nowrap}
table.p8-m th{font-size:.64rem;text-transform:uppercase;letter-spacing:.06em;opacity:.58;font-weight:600;text-align:left}
table.p8-m td.n,table.p8-m th.n{text-align:right}
.p8-sub{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;opacity:.7;margin:10px 0 2px}
.p8-chip{display:inline-block;padding:2px 8px;margin:0 4px 4px 0;border-radius:6px;font-size:.76rem;font-weight:600;
  border:1px solid rgba(128,128,128,.22);font-variant-numeric:tabular-nums}
</style>
"""


def _c(v) -> str:
    return UP if v is not None and pd.notna(v) and v > 0 else DOWN if v is not None and pd.notna(v) and v < 0 else "inherit"


def _p(v, d=2) -> str:
    return f"{v:+.{d}f}%" if v is not None and pd.notna(v) else "—"


@st.cache_data(ttl=55, show_spinner=False)
def _snap(at: float, uni_name: str) -> pd.DataFrame:
    """Snapshot of the chosen universe from the shared quote copy fetched at `at` (cache key)."""
    q, _ = market_flow.latest_quotes()
    if q is None or q.empty:
        return pd.DataFrame()
    return market_flow.snapshot(q, market_flow.universe(uni_name))


def pulse(snap: pd.DataFrame) -> dict:
    """Counts from NSE's feed: advancing/declining, above VWAP, near 52-week highs/lows, value traded."""
    ch = snap["pct_change"]
    hi = snap["high_52w"].where(snap["high_52w"] > 0)
    lo = snap["low_52w"].where(snap["low_52w"] > 0)
    near_hi = snap[(snap["ltp"] >= hi * (1 - NEAR / 100))]
    near_lo = snap[(snap["ltp"] <= lo * (1 + NEAR / 100))]
    return {"n": len(snap), "up": int((ch > 0).sum()), "down": int((ch < 0).sum()), "flat": int((ch == 0).sum()),
            "above_vwap": int((snap["ltp"] > snap["vwap"]).sum()), "vwap_n": int(snap["vwap"].notna().sum()),
            "value_cr": float(snap["turnover_cr"].sum()), "near_hi": near_hi, "near_lo": near_lo,
            "median": float(ch.median()) if len(ch) else None}


def heat_map(snap: pd.DataFrame, size_by: str, dark: bool, height: int = 560) -> go.Figure:
    """Treemap of sectors and their stocks: size = traded value (or equal), colour = % change."""
    mid = "#262d3a" if dark else "#e9e8e3"
    scale = [[0, DOWN], [0.5, mid], [1, UP]]
    size = snap["turnover_cr"] if size_by == "Traded value" else pd.Series(1.0, index=snap.index)
    df = snap.assign(sz=size.astype(float))

    def weighted(d):
        return float((d["pct_change"] * d["turnover_cr"]).sum() / d["turnover_cr"].sum()) if d["turnover_cr"].sum() else 0.0

    root = "All"
    ids, labels, parents, values, pct, info = [root], [""], [""], [float(df["sz"].sum())], [0.0], [("", 0.0, 0.0)]
    for sec, part in df.groupby("sector"):
        ids.append(sec), labels.append(sec), parents.append(root), values.append(float(part["sz"].sum()))
        pct.append(weighted(part)), info.append((f"{len(part)} stocks", float(part["turnover_cr"].sum()), 0.0))
        for r in part.itertuples():
            ids.append(f"{sec}/{r.symbol}"), labels.append(r.symbol), parents.append(sec), values.append(float(r.sz))
            pct.append(float(r.pct_change)), info.append((r.name, float(r.turnover_cr), float(r.ltp)))
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        customdata=[(p, *i) for p, i in zip(pct, info)],
        marker=dict(colors=pct, colorscale=scale, cmin=-PCT_RANGE, cmax=PCT_RANGE, cmid=0,
                    line=dict(width=1, color="rgba(11,15,23,.9)" if dark else "rgba(255,255,255,.9)"),
                    showscale=False),
        texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%", textposition="middle center",
        textfont=dict(color="#f4f6fa"), insidetextfont=dict(color="#f4f6fa"),
        hovertemplate="<b>%{label}</b> · %{customdata[1]}<br>Change: %{customdata[0]:+.2f}%<br>"
                      "Traded value: ₹%{customdata[2]:,.0f} cr<extra></extra>",
        tiling=dict(pad=1.5), pathbar=dict(visible=False), maxdepth=3,
        outsidetextfont=dict(size=12)))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), font=dict(family=charts.FONT, size=12),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def legend_bar() -> str:
    stops = ", ".join(f"{c} {p}%" for c, p in ((DOWN, 0), ("#262d3a", 50), (UP, 100)))
    return (f'<div style="display:flex;align-items:center;gap:8px;font-size:.7rem;opacity:.75;margin-top:6px">'
            f'<span>−{PCT_RANGE:.0f}%</span><span style="flex:1;height:6px;border-radius:3px;'
            f'background:linear-gradient(90deg,{stops})"></span><span>+{PCT_RANGE:.0f}%</span></div>')


def _mini(rows: pd.DataFrame, value: str = "pct") -> str:
    body = ""
    for r in rows.itertuples():
        last = f'<td class="n">{r.ltp:,.2f}</td>' if value != "pct" else ""
        right = (f'<td class="n" style="color:{_c(r.pct_change)}">{_p(r.pct_change)}</td>' if value == "pct" else
                 f'<td class="n">₹{r.turnover_cr:,.0f} cr</td><td class="n" style="color:{_c(r.pct_change)}">{_p(r.pct_change)}</td>')
        body += f'<tr><td title="LTP {r.ltp:,.2f}"><b>{_e(r.symbol)}</b></td>{last}{right}</tr>'
    head = ('<tr><th>Stock</th><th class="n">Chg</th></tr>' if value == "pct" else
            '<tr><th>Stock</th><th class="n">LTP</th><th class="n">Value</th><th class="n">Chg</th></tr>')
    return f'<table class="p8-m">{head}{body}</table>'


def _pulse_card(snap: pd.DataFrame, pu: dict, uni_name: str, as_of) -> None:
    with st.container(border=True):
        st.markdown(f'<div class="p8-h">Market pulse · {_e(uni_name)}</div><div class="p8-cap">Counts from NSE\'s feed '
                    'for these stocks</div>', unsafe_allow_html=True)
        adv = (f'<div style="display:flex;justify-content:space-between;font-size:.78rem;font-weight:600">'
               f'<span style="color:{UP}">▲ {pu["up"]} advancing</span><span style="opacity:.6">{pu["flat"]} flat</span>'
               f'<span style="color:{DOWN}">{pu["down"]} declining ▼</span></div>' + split_bar(pu["up"], pu["down"]))
        kv = (f'<div class="p8-kv"><div><div class="k">Above VWAP</div><div class="v">{pu["above_vwap"]}/{pu["vwap_n"]}</div></div>'
              f'<div><div class="k">Value traded</div><div class="v">₹{pu["value_cr"]:,.0f} cr</div></div>'
              f'<div><div class="k">Median move</div><div class="v" style="color:{_c(pu["median"])}">{_p(pu["median"])}</div></div></div>')
        top = snap.sort_values("pct_change", ascending=False)
        two = (f'<div class="p8-two"><div><div class="p8-sub" style="color:{UP}">Top gainers</div>{_mini(top.head(6))}</div>'
               f'<div><div class="p8-sub" style="color:{DOWN}">Top losers</div>{_mini(top.tail(6).iloc[::-1])}</div></div>')
        act = (f'<div class="p8-sub">Most active by value</div>'
               f'{_mini(snap.sort_values("turnover_cr", ascending=False).head(6), value="value")}')
        st.markdown(CSS_T + adv + kv + two + act + src("NSE live feed", as_of, live=True), unsafe_allow_html=True)


def _sectors(indices: dict) -> None:
    with st.container(border=True):
        st.markdown('<div class="p8-h">NSE sector indices</div><div class="p8-cap">Today\'s % change, official '
                    'indices</div>', unsafe_allow_html=True)
        rows = [(n.replace("NIFTY ", ""), i.get("percChange")) for n, i in indices.items()
                if i.get("percChange") is not None and n in SECTORS]
        if not rows:
            st.caption("NSE's index feed is not available right now.")
            return
        df = pd.DataFrame(rows, columns=["name", "pct"]).sort_values("pct", ascending=False)
        st.plotly_chart(diverging_bars(df["name"], df["pct"], height=max(220, 17 * len(df) + 40)),
                        use_container_width=True, config=STATIC, key="tm_sec")
        tv = next((i.get("timeVal") for i in indices.values() if i.get("timeVal")), None)
        st.markdown(src("NSE indices", pd.to_datetime(tv, errors="coerce", dayfirst=True) if tv else None, live=True),
                    unsafe_allow_html=True)


SECTORS = ["NIFTY BANK", "NIFTY IT", "NIFTY AUTO", "NIFTY PHARMA", "NIFTY FMCG", "NIFTY METAL", "NIFTY REALTY",
           "NIFTY ENERGY", "NIFTY MEDIA", "NIFTY PSU BANK", "NIFTY PRIVATE BANK", "NIFTY FIN SERVICE",
           "NIFTY OIL & GAS", "NIFTY CONSUMER DURABLES", "NIFTY HEALTHCARE INDEX", "NIFTY INFRA", "NIFTY COMMODITIES",
           "NIFTY CPSE", "NIFTY PSE", "NIFTY MNC", "NIFTY SERV SECTOR", "NIFTY CONSUMPTION"]


def _forecast(preds: pd.DataFrame, story) -> None:
    with st.container(border=True):
        st.markdown('<div class="p8-h">Next-open forecast</div><div class="p8-cap">Pulse8 model\'s expected gap at '
                    '09:15 IST (see Model accuracy before relying on it)</div>', unsafe_allow_html=True)
        if preds is None or preds.empty or "pred_gap_pct" not in preds:
            st.caption("No forecast available.")
            return
        p = preds.sort_values("pred_gap_pct", ascending=False)
        m = max(float(p["pred_gap_pct"].abs().max()), 0.1)
        body = ""
        for r in p.itertuples():
            w = 50 * abs(r.pred_gap_pct) / m
            bar = (f'<div style="position:relative;height:8px;background:rgba(128,128,128,.14);border-radius:4px">'
                   f'<span style="position:absolute;top:0;bottom:0;{"left:50%" if r.pred_gap_pct >= 0 else f"left:{50 - w:.1f}%"};'
                   f'width:{w:.1f}%;background:{_c(r.pred_gap_pct)};border-radius:4px"></span>'
                   f'<span style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:rgba(128,128,128,.6)"></span></div>')
            close = getattr(r, "close", np.nan)
            body += (f'<tr><td><b>{_e(r.symbol)}</b></td><td class="n">{close:,.2f}</td><td class="n">{r.pred_open:,.2f}</td>'
                     f'<td class="n" style="color:{_c(r.pred_gap_pct)}">{_p(r.pred_gap_pct)}</td>'
                     f'<td style="width:34%">{bar}</td></tr>')
        st.markdown(CSS_T + '<table class="p8-m"><tr><th>Stock</th><th class="n">Close</th><th class="n">Fcst open</th>'
                    f'<th class="n">Gap</th><th></th></tr>{body}</table>'
                    + src("Pulse8 model", pd.Timestamp(story["as_of_close"]) if story and story.get("as_of_close") else None),
                    unsafe_allow_html=True)


def _watch(pu: dict) -> None:
    with st.container(border=True):
        st.markdown(f'<div class="p8-h">52-week watch · institutions</div><div class="p8-cap">Stocks within {NEAR:.0f}% '
                    'of a 52-week high or low; latest NSE-reported FII/DII flow</div>', unsafe_allow_html=True)

        def chips(df, col):
            if df.empty:
                return '<span style="opacity:.6;font-size:.8rem">None</span>'
            return "".join(f'<span class="p8-chip" style="color:{col}">{_e(r.symbol)} {_p(r.pct_change, 1)}</span>'
                           for r in df.sort_values("pct_change", ascending=(col == DOWN)).head(14).itertuples())
        html_ = (CSS_T + f'<div class="p8-sub">Near 52-week high · {len(pu["near_hi"])}</div>{chips(pu["near_hi"], UP)}'
                 f'<div class="p8-sub">Near 52-week low · {len(pu["near_lo"])}</div>{chips(pu["near_lo"], DOWN)}')
        fii = inst.fii_summary(inst.load("fii_dii")) if inst.FILES["fii_dii"].exists() else None
        if fii:
            cells = "".join(f'<div><div class="k">{_e(k)} net</div><div class="v" style="color:{_c(fii[k]["net"])}">'
                            f'₹{fii[k]["net"]:+,.0f} cr</div></div>' for k in ("FII/FPI", "DII") if k in fii)
            html_ += (f'<div class="p8-sub">FII / DII cash, {fii["as_of"]:%d %b} (reported by NSE)</div>'
                      f'<div class="p8-kv" style="grid-template-columns:1fr 1fr">{cells}</div>')
        else:
            html_ += ('<div class="p8-sub">FII / DII cash</div><div style="font-size:.8rem;opacity:.7">Not saved yet: '
                      'open Institutional activity to fetch or upload NSE\'s report.</div>')
        st.markdown(html_, unsafe_allow_html=True)


def _market_watch(snap: pd.DataFrame, uni_name: str) -> None:
    st.markdown(f"#### Market watch · {uni_name}")
    t = snap.copy()
    span = (t["high_52w"] - t["low_52w"]).where(lambda s: s > 0)
    t["pos52"] = ((t["ltp"] - t["low_52w"]) / span * 100).clip(0, 100)
    t["vs_vwap"] = (t["ltp"] / t["vwap"] - 1) * 100
    t = t.sort_values("turnover_cr", ascending=False)
    st.dataframe(t[["symbol", "name", "sector", "ltp", "pct_change", "open", "high", "low", "prev_close", "vwap",
                    "vs_vwap", "volume", "turnover_cr", "pos52", "pct_30d"]], hide_index=True, use_container_width=True,
                 height=440, column_config={
                     "symbol": st.column_config.TextColumn("Symbol", pinned=True), "name": "Company", "sector": "Sector",
                     "ltp": st.column_config.NumberColumn("LTP ₹", format="%.2f"),
                     "pct_change": st.column_config.NumberColumn("Chg %", format="%+.2f%%"),
                     "open": st.column_config.NumberColumn("Open", format="%.2f"),
                     "high": st.column_config.NumberColumn("High", format="%.2f"),
                     "low": st.column_config.NumberColumn("Low", format="%.2f"),
                     "prev_close": st.column_config.NumberColumn("Prev close", format="%.2f"),
                     "vwap": st.column_config.NumberColumn("VWAP", format="%.2f", help="Traded value ÷ traded quantity today."),
                     "vs_vwap": st.column_config.NumberColumn("vs VWAP", format="%+.2f%%"),
                     "volume": st.column_config.NumberColumn("Volume", format="%d"),
                     "turnover_cr": st.column_config.NumberColumn("Value ₹ cr", format="%.1f"),
                     "pos52": st.column_config.ProgressColumn("52-week range", format="%.0f%%", min_value=0, max_value=100,
                                                              help="Where the price sits between the 52-week low (0%) and high (100%)."),
                     "pct_30d": st.column_config.NumberColumn("30D %", format="%+.1f%%")})


def render(preds: pd.DataFrame, story: dict | None) -> None:
    indices = live_indices()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(tab_live.index_tiles({k: v for k, v in indices.items() if k in tab_live.INDICES}), unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.7, 1.4, 2.4])
    uni_name = c1.segmented_control("Universe", ["NIFTY 50", "NIFTY 100", "NIFTY 500"], default="NIFTY 100",
                                    key="tm_uni") or "NIFTY 100"
    size_by = c2.segmented_control("Box size", ["Traded value", "Equal"], default="Traded value", key="tm_size") or "Traded value"
    with st.spinner("Loading NSE prices (about 10 seconds the first time)..."):
        _, at = market_flow.latest_quotes()
    snap = _snap(at, uni_name)
    if snap.empty:
        st.warning("NSE's price feed is not available right now, so the heat map and market pulse cannot be drawn. "
                   "The other pages still work; try again in a minute.")
        _forecast(preds, story)
        return
    as_of = snap["as_of"].max()
    c3.markdown(f'<div style="text-align:right;font-size:.76rem;opacity:.7;padding-top:30px">{len(snap)} stocks · '
                f'NSE feed as of {pd.Timestamp(as_of):%d %b %Y %H:%M} IST · hover a box for details</div>',
                unsafe_allow_html=True)
    pu = pulse(snap)
    dark = True  # terminal-style dark tiles in both themes, so the white labels always read
    left, right = st.columns([2.15, 1], gap="medium")
    with left, st.container(border=True):
        st.markdown(f'<div class="p8-h">Heat map · {_e(uni_name)}</div><div class="p8-cap">Sector › stock · box size = '
                    f'{"traded value today" if size_by == "Traded value" else "equal"} · colour = % change</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(heat_map(snap, size_by, dark, height=600 if uni_name != "NIFTY 50" else 520),
                        use_container_width=True, config=STATIC, key="tm_heat")
        st.markdown(legend_bar() + src("NSE live feed", as_of, live=True), unsafe_allow_html=True)
    with right:
        _pulse_card(snap, pu, uni_name, as_of)

    a, b, c = st.columns(3, gap="medium")
    with a:
        _sectors(indices)
    with b:
        _forecast(preds, story)
    with c:
        _watch(pu)
    _market_watch(snap, uni_name)
    st.caption("Prices: NSE's live feed, 1-3 minutes behind the exchange; outside market hours, the last session's "
               "close. Decision support for a student project, not investment advice.")
