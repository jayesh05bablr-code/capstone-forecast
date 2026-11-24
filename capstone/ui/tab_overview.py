"""Market overview: the morning brief as a one-screen terminal dashboard.

Layout (aligned on a 12-column grid, every block the same card style):
  header strip   date, NSE session state, model's opening call
  KPI tiles      NIFTY 50 (+ 3-month sparkline), India VIX (+ 1-year percentile bar), NSE breadth
                 (advance/decline bar), USD/INR, NIFTY 50 P/E (+ 2-year range bar), opening call
  row 2          Global cues: one diverging bar chart (US indices, US sectors, Asia) | NIFTY 50 trend
  row 3          Sector strength vs NIFTY 50 (diverging bars by rotation quadrant) | Risks to watch
  row 4          Forecast highlights (gap bars) | Corporate actions in the next 7 days

Every number comes from data the app fetched; each card names its source and date with a freshness
dot (green = latest published session). A card with no data says so. Nothing is a recommendation.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from capstone import breadth, brief, charts, flow_history, market_flow, sector_intel, stock_metrics
from capstone.charts import CONFIG
from capstone.ui.common import DOWN, UP

ACCENT = "#5aa9ff"
QUAD = {"Leading": "#2a9d6f", "Improving": "#2a78d6", "Weakening": "#e0a030", "Lagging": "#e34948"}
STATIC = {**CONFIG, "displayModeBar": False, "staticPlot": False}

CSS = """
<style>
.p8-strip{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;justify-content:space-between;
  padding:14px 18px;border:1px solid rgba(128,128,128,.22);border-radius:12px;
  background:linear-gradient(90deg,rgba(90,169,255,.10),rgba(106,92,255,.06) 60%,rgba(0,0,0,0));margin-bottom:14px}
.p8-strip h2{margin:0;font-size:1.35rem;font-weight:700;letter-spacing:-.01em}
.p8-strip .sub{opacity:.68;font-size:.86rem;margin-top:2px}
.p8-pill{display:inline-flex;align-items:center;gap:7px;padding:5px 12px;border-radius:999px;font-size:.82rem;
  font-weight:600;border:1px solid rgba(128,128,128,.28)}
.p8-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.p8-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin-bottom:6px}
@media(max-width:1100px){.p8-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:640px){.p8-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
.p8-tile{border:1px solid rgba(128,128,128,.22);border-radius:12px;padding:12px 14px 10px;min-height:132px;
  display:flex;flex-direction:column;justify-content:space-between;background:rgba(128,128,128,.045)}
.p8-tile .lab{font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;opacity:.65;font-weight:600}
.p8-tile .val{font-size:1.55rem;font-weight:700;line-height:1.15;margin-top:4px;font-variant-numeric:tabular-nums;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.p8-tile .chg{font-size:.84rem;font-weight:600;font-variant-numeric:tabular-nums}
.p8-tile .note{font-size:.72rem;opacity:.6;margin-top:4px}
.p8-bar{height:6px;border-radius:3px;background:rgba(128,128,128,.22);position:relative;margin-top:8px;overflow:hidden}
.p8-bar>span{position:absolute;top:0;bottom:0;left:0;border-radius:3px}
.p8-mark{position:absolute;top:-3px;width:3px;height:12px;border-radius:2px;background:currentColor}
.p8-h{font-size:.95rem;font-weight:700;margin:0 0 2px}
.p8-cap{font-size:.74rem;opacity:.62;margin:-2px 0 6px}
.p8-risk{border-left:3px solid #e0a030;padding:7px 10px;margin:0 0 8px;border-radius:0 8px 8px 0;
  background:rgba(224,160,48,.08);font-size:.88rem;line-height:1.35}
.p8-risk.hi{border-left-color:#e34948;background:rgba(227,73,72,.08)}
.p8-ok{border-left:3px solid #2a9d6f;padding:7px 10px;border-radius:0 8px 8px 0;background:rgba(42,157,111,.08);font-size:.88rem}
.p8-src{font-size:.72rem;opacity:.6;margin-top:6px}
table.p8-t{width:100%;border-collapse:collapse;font-size:.86rem}
table.p8-t td,table.p8-t th{padding:6px 6px;border-bottom:1px solid rgba(128,128,128,.16);text-align:left}
table.p8-t th{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;opacity:.6;font-weight:600}
table.p8-t td.n{text-align:right;font-variant-numeric:tabular-nums}
</style>
"""


# ---------- small building blocks ----------

def _e(s) -> str:
    return html.escape(str(s))


def _col(v) -> str:
    return UP if v is not None and v > 0 else DOWN if v is not None and v < 0 else "inherit"


def _pct(v, d=2) -> str:
    return f"{v:+.{d}f}%" if v is not None and pd.notna(v) else "—"


def fresh_dot(as_of, live: bool = False) -> tuple[str, str]:
    """(colour, words) for a freshness dot: green = latest published session (or < 15 min for live data)."""
    if as_of is None or pd.isna(as_of):
        return "#8b93a3", "no date"
    ts = pd.Timestamp(as_of)
    if live:
        ok = (pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None) - ts).total_seconds() <= 900
    else:
        ok = ts.normalize() >= pd.Timestamp(flow_history.expected_last_session())
    when = f"{ts:%d %b %Y %H:%M} IST" if (ts.hour or ts.minute) else f"{ts:%d %b %Y}"
    return ("#2a9d6f" if ok else "#e0a030"), when + ("" if ok else " · not the latest session")


def src(source: str, as_of, live: bool = False) -> str:
    c, w = fresh_dot(as_of, live)
    return f'<div class="p8-src"><span class="p8-dot" style="background:{c}"></span>&nbsp;{_e(source)} · {_e(w)}</div>'


def sparkline(values, w: int = 150, h: int = 34) -> str:
    """Inline SVG line with a soft area; colour by direction over the window."""
    v = [float(x) for x in values if x is not None and pd.notna(x)]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pts = [(i * w / (len(v) - 1), h - 3 - (x - lo) / span * (h - 6)) for i, x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    c = UP if v[-1] >= v[0] else DOWN
    area = f"0,{h} " + line + f" {w},{h}"
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none" style="margin-top:6px">'
            f'<polygon points="{area}" fill="{c}" opacity=".13"/>'
            f'<polyline points="{line}" fill="none" stroke="{c}" stroke-width="1.8" vector-effect="non-scaling-stroke"/>'
            f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="2.6" fill="{c}"/></svg>')


def range_bar(pos: float | None, color: str = ACCENT) -> str:
    """Where today sits in a range, 0..1, as a thin bar with a marker."""
    if pos is None or pd.isna(pos):
        return ""
    p = 100 * min(max(pos, 0), 1)
    return (f'<div class="p8-bar"><span style="width:{p:.0f}%;background:{color};opacity:.35"></span>'
            f'<i class="p8-mark" style="left:calc({p:.0f}% - 1px);color:{color}"></i></div>')


def split_bar(up: int, down: int) -> str:
    tot = (up or 0) + (down or 0)
    if not tot:
        return ""
    a = 100 * up / tot
    return (f'<div class="p8-bar"><span style="width:{a:.1f}%;background:{UP}"></span>'
            f'<span style="left:{a:.1f}%;right:0;width:auto;background:{DOWN}"></span></div>')


def tile(label: str, value: str, change: str = "", change_color: str = "inherit", graphic: str = "",
         note: str = "", title: str = "") -> str:
    return (f'<div class="p8-tile" title="{_e(title)}"><div><div class="lab">{_e(label)}</div>'
            f'<div class="val">{_e(value)}</div>'
            f'<div class="chg" style="color:{change_color}">{_e(change)}</div></div>'
            f'<div>{graphic}<div class="note">{_e(note)}</div></div></div>')


def _fig_base(fig: go.Figure, height: int) -> go.Figure:
    charts.style(fig, height=height, hover="closest", legend_top=False)
    fig.update_layout(margin=dict(l=6, r=10, t=6, b=6), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showspikes=False)
    fig.update_yaxes(showspikes=False)
    return fig


def diverging_bars(labels, values, colors=None, height=300, suffix="%", groups=None, text=True) -> go.Figure:
    """Horizontal bars about zero, labelled with their value, sorted as given (top = first)."""
    labels, values = list(labels)[::-1], [float(v) for v in values][::-1]
    colors = (list(colors)[::-1] if colors is not None else [UP if v >= 0 else DOWN for v in values])
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors, line_width=0),
                           text=[f"{v:+.2f}{suffix}" for v in values] if text else None, textposition="outside", cliponaxis=False,
                           textfont=dict(size=11), hovertemplate="%{y}: %{x:+.2f}" + suffix + "<extra></extra>"))
    m = max([abs(v) for v in values] + [0.1])
    _fig_base(fig, height)
    lo = -m * 1.6 if min(values) < 0 else -m * 0.15
    hi = m * 1.6 if max(values) > 0 else m * 0.15
    fig.update_xaxes(range=[lo, hi], zeroline=True, zerolinecolor="rgba(128,128,128,.55)",
                     zerolinewidth=1, ticksuffix=suffix, showgrid=True)
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(size=11.5))
    fig.update_layout(bargap=0.32)
    if groups:  # thin separators between groups (e.g. US | sectors | Asia)
        n, acc = len(labels), 0
        for g in groups[:-1]:  # groups are counted from the top; category k sits at y = k from the bottom
            acc += g
            fig.add_hline(y=n - acc - 0.5, line=dict(color="rgba(128,128,128,.35)", width=1, dash="dot"))
    return fig


# ---------- sections ----------

def _header(story, mood) -> None:
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    from capstone import market_calendar
    open_now = market_calendar.market_open(now.to_pydatetime())
    trading = market_calendar.is_trading_day(now.date())
    if open_now:
        state, sc = "NSE open", "#2a9d6f"
    elif trading and now.hour * 60 + now.minute < 9 * 60 + 15:
        state, sc = "Before the open", "#e0a030"
    elif trading:
        state, sc = "NSE closed for the day", "#8b93a3"
    else:
        state, sc = "NSE holiday / weekend", "#8b93a3"
    call = (story or {}).get("mood") or "No forecast yet"
    nxt = (story or {}).get("next_open")
    call_c = DOWN if "down" in call.lower() or "weak" in call.lower() else UP if ("up" in call.lower() or "strong" in call.lower()) else ACCENT
    st.markdown(
        f'<div class="p8-strip"><div><h2>Morning brief</h2><div class="sub">{now:%A, %d %B %Y} · {now:%H:%M} IST'
        + (f" · forecast for the open of {pd.Timestamp(nxt):%a %d %b}" if nxt else "") + '</div></div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap"><span class="p8-pill"><span class="p8-dot" style="background:{sc}"></span>{_e(state)}</span>'
        f'<span class="p8-pill" style="color:{call_c};border-color:{call_c}">Opening call: {_e(call)}</span></div></div>',
        unsafe_allow_html=True)


def _tiles(mood, val, live, story, nifty: pd.DataFrame) -> None:
    t = []
    spark = sparkline(nifty["close"].tail(63)) if len(nifty) else ""
    if live:
        t.append(tile("NIFTY 50", f"{live['last']:,.2f}", _pct(live.get("pct")), _col(live.get("pct")), spark,
                      f"3 months · 1M {_pct(live.get('chg_1m'))}", f"NSE index feed, {live.get('time')}"))
    else:
        t.append(tile("NIFTY 50", "—", "", graphic=spark, note="NSE feed unavailable"))
    vix = (mood or {}).get("india_vix") or {}
    pctl = (vix.get("range") or {}).get("percentile")
    vix_c = DOWN if pctl and pctl >= 75 else "#e0a030" if pctl and pctl >= 50 else "#2a9d6f"
    t.append(tile("India VIX", f"{vix['level']:.2f}" if vix.get("level") else "—", _pct(vix.get("change_1d_pct")),
                  _col(-(vix.get("change_1d_pct") or 0)) if vix.get("change_1d_pct") else "inherit",
                  range_bar(pctl / 100 if pctl is not None else None, vix_c),
                  f"{pctl:.0f}th percentile, 1 year" if pctl is not None else "", "Expected volatility (NSE)"))
    sb = (mood or {}).get("stock_breadth") or {}
    t.append(tile("NSE breadth", f"{sb['up_pct']:.0f}% up" if sb.get("up_pct") is not None else "—",
                  (sb.get("reading") or "").capitalize(), _col((sb.get("up_pct") or 50) - 50),
                  split_bar(sb.get("up", 0), sb.get("down", 0)),
                  f"{sb.get('up', 0):,} up · {sb.get('down', 0):,} down" if sb else "", "Mainboard stocks, last session"))
    fx, fxc = (story or {}).get("usdinr"), (story or {}).get("usdinr_chg")
    t.append(tile("USD / INR", f"₹{fx:.2f}" if fx else "—",
                  (f"Rupee {abs(fxc) * 100:.2f}% {'weaker' if fxc > 0 else 'stronger'}" if fxc is not None and pd.notna(fxc) else ""),
                  DOWN if fxc and fxc > 0 else UP if fxc else "inherit", "", "Latest rate in the forecast data"))
    pe = (val or {}).get("pe") or {}
    if pe.get("current"):
        pos = (pe["current"] - pe["min"]) / ((pe["max"] - pe["min"]) or 1)
        t.append(tile("NIFTY 50 P/E", f"{pe['current']:.1f}×", f"median {pe['median']:.1f}×", "inherit",
                      range_bar(pos, "#6a5cff"), f"2-yr range {pe['min']:.1f}–{pe['max']:.1f}", "NSE published ratios"))
    else:
        t.append(tile("NIFTY 50 P/E", "—", note="NSE valuation unavailable"))
    preds_gap = st.session_state.get("_brief_avg_gap")
    t.append(tile("Model's average gap", _pct(preds_gap) if preds_gap is not None else "—", "expected at 09:15",
                  _col(preds_gap), "", (story or {}).get("mood") or "", "Average expected gap of the forecast stocks"))
    st.markdown(CSS + '<div class="p8-grid">' + "".join(t) + "</div>", unsafe_allow_html=True)
    if live or mood:
        a = live and pd.to_datetime(live.get("time"), format="%d-%b-%Y %H:%M", errors="coerce")
        st.markdown(src("NSE (mcp.nseindia.in): index feed, market mood, valuation", a if a is not None and pd.notna(a)
                        else (mood or {}).get("date")), unsafe_allow_html=True)


def _global(story, moves) -> None:
    with st.container(border=True):
        st.markdown('<div class="p8-h">Global cues overnight</div><div class="p8-cap">US index and sector moves in the '
                    'last US session, and Asia this morning</div>', unsafe_allow_html=True)
        labels, vals, groups = [], [], []
        if moves is not None and len(moves):
            m = moves.reset_index() if "ticker" not in moves.columns else moves
            idx = m[~m["is_sector"]].dropna(subset=["ret"])
            sec = m[m["is_sector"]].dropna(subset=["ret"]).sort_values("ret", ascending=False)
            sec = pd.concat([sec.head(3), sec.tail(3)]).drop_duplicates("ticker")
            labels += [f"US · {n}" for n in idx["name"]] + [f"US sector · {n}" for n in sec["name"]]
            vals += list(100 * idx["ret"]) + list(100 * sec["ret"])
            groups += [len(idx), len(sec)]
        asia = brief.asia()
        if asia:
            labels += [f"Asia · {r['name'].split(' (')[0]}" for r in asia]
            vals += [r["pct"] for r in asia]
            groups.append(len(asia))
        if not labels:
            st.caption("No overnight data yet.")
            return
        st.plotly_chart(diverging_bars(labels, vals, height=max(330, 27 * len(labels) + 30), groups=groups),
                        use_container_width=True, config=STATIC, key="ov_global")
        notes = []
        if story and not story.get("overnight_complete", True):
            notes.append("latest US session not in the data yet")
        if not asia:
            notes.append("Asia unavailable (Yahoo Finance did not answer)")
        st.markdown(src("US ETFs (Twelve Data / Alpha Vantage)" + (", Asia (Yahoo Finance)" if asia else "")
                        + (" · " + "; ".join(notes) if notes else ""),
                        pd.Timestamp(story["as_of_close"]) if story and story.get("as_of_close") else None),
                    unsafe_allow_html=True)


def _nifty(nifty: pd.DataFrame, live) -> None:
    with st.container(border=True):
        st.markdown('<div class="p8-h">NIFTY 50 trend</div><div class="p8-cap">Daily close, last 6 months, with '
                    '20- and 50-day averages</div>', unsafe_allow_html=True)
        if nifty.empty:
            st.caption("NIFTY 50 history unavailable.")
            return
        d = nifty.sort_values("date").copy()
        d["ma20"], d["ma50"] = d["close"].rolling(20).mean(), d["close"].rolling(50).mean()
        d = d.tail(126)
        up = d["close"].iloc[-1] >= d["close"].iloc[0]
        c = UP if up else DOWN
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["date"], y=d["close"], name="NIFTY 50", line=dict(color=c, width=2.2),
                                 fill="tozeroy", fillcolor=("rgba(42,120,214,.10)" if up else "rgba(227,73,72,.10)"),
                                 hovertemplate="%{x|%d %b %Y}: %{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=d["date"], y=d["ma20"], name="20-day avg", line=dict(color="#f2b84b", width=1.3),
                                 hovertemplate="20-day avg %{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=d["date"], y=d["ma50"], name="50-day avg", line=dict(color="#8b93a3", width=1.3, dash="dot"),
                                 hovertemplate="50-day avg %{y:,.0f}<extra></extra>"))
        _fig_base(fig, 352)
        lo, hi = float(d[["close", "ma20", "ma50"]].min().min()), float(d[["close", "ma20", "ma50"]].max().max())
        fig.update_yaxes(range=[lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.12], tickformat=",.0f", showgrid=True)
        fig.update_xaxes(showgrid=False, tickformat="%b")
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom", font=dict(size=11)),
                          margin=dict(l=6, r=10, t=24, b=6))
        last = d.iloc[-1]
        fig.add_annotation(x=last["date"], y=last["close"], text=f"<b>{last['close']:,.0f}</b>", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=c, size=12))
        st.plotly_chart(fig, use_container_width=True, config=STATIC, key="ov_nifty")
        st.markdown(src("NSE bhavcopy index history", last["date"]), unsafe_allow_html=True)


def _sectors() -> None:
    with st.container(border=True):
        st.markdown('<div class="p8-h">Sector strength vs NIFTY 50</div><div class="p8-cap">20-session return minus '
                    "the NIFTY 50's, official NSE sector indices; colour = rotation quadrant</div>", unsafe_allow_html=True)
        sector_intel.refresh_indices()
        ix = sector_intel.load_indices()
        lb = sector_intel.leaderboard(ix) if len(ix) else pd.DataFrame()
        if lb.empty:
            st.caption("Sector index history is downloading (about 10 seconds).")
            return
        from capstone.ui.tab_sector_intel import _short
        fig = diverging_bars([_short(i) for i in lb["index"]], lb["rs_20d"], [QUAD[q] for q in lb["quadrant"]],
                             height=max(300, 22 * len(lb) + 30), suffix=" pts")
        st.plotly_chart(fig, use_container_width=True, config=STATIC, key="ov_sec")
        st.markdown(" ".join(f'<span class="p8-pill" style="font-size:.72rem;padding:2px 9px;margin-right:4px">'
                             f'<span class="p8-dot" style="background:{c}"></span>{q}</span>' for q, c in QUAD.items()),
                    unsafe_allow_html=True)
        st.markdown(src("NSE sector indices (bhavcopy) · a description of recent strength, not a forecast",
                        lb["as_of"].max()), unsafe_allow_html=True)


def _risks(mood, val, story) -> None:
    with st.container(border=True):
        st.markdown('<div class="p8-h">Risks to watch</div><div class="p8-cap">Each tied to a published number</div>',
                    unsafe_allow_html=True)
        uni = market_flow.universe("NIFTY 100")
        bh = breadth.breadth_history(flow_history.load(), uni)
        alerts = breadth.divergences(stock_metrics.nifty_history(), bh) if len(bh) else []
        notes = brief.risks(mood, alerts, story, val)
        if not notes:
            st.markdown('<div class="p8-ok">No elevated-risk readings in the data shown here.</div>', unsafe_allow_html=True)
        else:
            hi_words = ("VIX is", "broad decline")
            st.markdown("".join(f'<div class="p8-risk{" hi" if any(w in n for w in hi_words) else ""}">{_e(n)}</div>'
                                for n in notes), unsafe_allow_html=True)
        st.markdown(src("NSE market mood, valuation; NIFTY 100 breadth", (mood or {}).get("date")),
                    unsafe_allow_html=True)


def _forecast(preds: pd.DataFrame, story) -> None:
    with st.container(border=True):
        st.markdown('<div class="p8-h">Forecast highlights</div><div class="p8-cap">Expected gap at the 09:15 open, '
                    'from the trained model (check Model accuracy before relying on it)</div>', unsafe_allow_html=True)
        if preds is None or preds.empty or "pred_gap_pct" not in preds:
            st.caption("No forecast available.")
            return
        p = preds.sort_values("pred_gap_pct", ascending=False)
        lo = p.get("q10_gap_pct")
        labels = [f"<b>{r.symbol}</b>  {r.pred_gap_pct:+.2f}%" for r in p.itertuples()]
        p = p.assign(symbol=labels)
        fig = diverging_bars(p["symbol"], p["pred_gap_pct"], height=max(200, 34 * len(p) + 30), text=False)
        if lo is not None and "q90_gap_pct" in p:  # likely range as a whisker over each bar
            fig.add_trace(go.Scatter(x=list(p["pred_gap_pct"])[::-1], y=list(p["symbol"])[::-1], mode="markers",
                                     marker=dict(size=1, color="rgba(0,0,0,0)"),
                                     error_x=dict(type="data", symmetric=False, thickness=1.2, width=4, color="rgba(128,128,128,.8)",
                                                  array=list(p["q90_gap_pct"] - p["pred_gap_pct"])[::-1],
                                                  arrayminus=list(p["pred_gap_pct"] - p["q10_gap_pct"])[::-1]),
                                     hoverinfo="skip"))
            m = float(np.nanmax(np.abs(np.r_[p["q10_gap_pct"], p["q90_gap_pct"]]))) * 1.25
            fig.update_xaxes(range=[-m, m])
        st.plotly_chart(fig, use_container_width=True, config=STATIC, key="ov_fc")
        st.markdown(src("Pulse8 model" + (" · whiskers = likely range (10th–90th percentile)" if lo is not None else ""),
                        pd.Timestamp(story["as_of_close"]) if story and story.get("as_of_close") else None),
                    unsafe_allow_html=True)


def _actions(uni50: pd.DataFrame) -> None:
    with st.container(border=True):
        st.markdown('<div class="p8-h">Corporate actions, next 7 days</div><div class="p8-cap">NIFTY 50 dividend, '
                    'split and bonus ex-dates</div>', unsafe_allow_html=True)
        brief.refresh_actions(list(uni50["symbol"]))
        raw, a = brief.load_actions(), brief.upcoming_actions()
        if raw.empty:
            st.caption("Checking NSE for upcoming ex-dates (about 15 seconds, once a day).")
            return
        if a.empty:
            st.markdown('<div class="p8-ok">No NIFTY 50 ex-dates in the next 7 days.</div>', unsafe_allow_html=True)
        else:
            rows = "".join(f"<tr><td><b>{_e(r.symbol)}</b></td><td>{pd.Timestamp(r.ex_date):%a %d %b}</td>"
                           f"<td>{_e(r.purpose)}</td></tr>" for r in a.itertuples())
            st.markdown(f'<table class="p8-t"><tr><th>Stock</th><th>Ex-date</th><th>Action</th></tr>{rows}</table>',
                        unsafe_allow_html=True)
        st.markdown(src("NSE corporate actions", pd.Timestamp(raw["fetched"].iloc[0])), unsafe_allow_html=True)


def render(preds: pd.DataFrame, story: dict | None, moves: pd.DataFrame | None) -> None:
    mood, val, live = brief.market_mood(), brief.valuation(), brief.nifty_live()
    nifty = stock_metrics.nifty_history()
    if preds is not None and "pred_gap_pct" in preds and len(preds):
        st.session_state["_brief_avg_gap"] = float(preds["pred_gap_pct"].mean())
    _header(story, mood)
    _tiles(mood, val, live, story, nifty)
    a, b = st.columns([1, 1], gap="medium")
    with a:
        _global(story, moves)
    with b:
        _nifty(nifty, live)
    a, b = st.columns([1, 1], gap="medium")
    with a:
        _sectors()
    with b:
        _risks(mood, val, story)
        _actions(market_flow.universe("NIFTY 50"))
    _forecast(preds, story)
    st.caption("Every figure is read from the named source; nothing here is a recommendation to buy or sell.")
