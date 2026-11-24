"""App-wide terminal shell: styling, the brand bar and the scrolling NSE ticker ribbon.

Shown above every page. The ribbon uses the same shared NSE quote download as the Market flow page
(one download a minute for the whole server) and never makes a page wait for it: when no copy is in
memory yet, it shows the indices only and starts the download in the background.
"""
from __future__ import annotations

import html
import threading

import pandas as pd
import streamlit as st

from capstone import market_flow
from capstone.ui.common import DOWN, UP

AMBER = "#f2a93b"
RIBBON_INDICES = ["NIFTY 50", "NIFTY NEXT 50", "NIFTY BANK", "NIFTY IT", "NIFTY FIN SERVICE", "NIFTY MIDCAP 100",
                  "NIFTY SMLCAP 100", "NIFTY AUTO", "NIFTY PHARMA", "NIFTY FMCG", "NIFTY METAL", "INDIA VIX"]

GLOBAL_CSS = f"""
<style>
:root{{--p8-amber:{AMBER};--p8-line:rgba(128,128,128,.22);--p8-soft:rgba(128,128,128,.05)}}
.block-container{{padding-top:3.3rem!important;padding-bottom:3rem!important;max-width:1680px}}
[data-testid="stAppDeployButton"]{{display:none}}
header[data-testid="stHeader"]{{background:transparent;height:2.6rem}}
h1,h2,h3{{letter-spacing:-.01em}}
.block-container h4{{font-size:.78rem!important;text-transform:uppercase;letter-spacing:.09em;font-weight:700;
  border-left:3px solid var(--p8-amber);padding:2px 0 2px 9px!important;margin:18px 0 10px!important;opacity:.92}}
[data-testid="stMetric"]{{border:1px solid var(--p8-line);border-radius:10px;padding:10px 12px 8px;background:var(--p8-soft)}}
[data-testid="stMetricLabel"] p{{font-size:.72rem!important;text-transform:uppercase;letter-spacing:.06em;opacity:.7}}
[data-testid="stMetricValue"]{{font-variant-numeric:tabular-nums;font-size:1.45rem!important}}
[data-testid="stDataFrame"]{{border:1px solid var(--p8-line);border-radius:10px;overflow:hidden}}
[data-testid="stExpander"] details{{border-color:var(--p8-line)!important;border-radius:10px!important}}
[data-testid="stTabs"] button p{{font-size:.84rem;font-weight:600;letter-spacing:.01em}}
[data-testid="stVerticalBlockBorderWrapper"]{{border-radius:12px}}
[data-testid="stNavSectionHeader"]{{font-size:.68rem!important;text-transform:uppercase;letter-spacing:.1em;
  color:var(--p8-amber)!important;font-weight:700}}
[data-testid="stSidebarNavLink"] span{{font-size:.86rem}}
.p8-brand{{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:6px 16px;
  padding:8px 14px;border:1px solid var(--p8-line);border-radius:10px 10px 0 0;border-bottom:none;
  background:linear-gradient(90deg,rgba(242,169,59,.13),rgba(90,169,255,.05) 55%,transparent)}}
.p8-brand .logo{{font-weight:800;letter-spacing:.16em;font-size:1.02rem;color:var(--p8-amber)}}
.p8-brand .tag{{font-size:.78rem;opacity:.68;margin-left:10px}}
.p8-brand .meta{{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;font-size:.76rem;opacity:.9;
  font-variant-numeric:tabular-nums}}
.p8-brand .meta b{{font-weight:700}}
.p8-st{{display:inline-flex;align-items:center;gap:6px;padding:2px 10px;border-radius:999px;border:1px solid var(--p8-line);
  font-weight:600}}
.p8-st i{{width:7px;height:7px;border-radius:50%;display:inline-block}}
.p8-rib{{overflow:hidden;white-space:nowrap;border:1px solid var(--p8-line);border-radius:0 0 10px 10px;
  background:rgba(0,0,0,.18);margin-bottom:14px;position:relative;font-variant-numeric:tabular-nums;
  -webkit-mask-image:linear-gradient(90deg,transparent,#000 3%,#000 97%,transparent);
  mask-image:linear-gradient(90deg,transparent,#000 3%,#000 97%,transparent)}}
.p8-rib .trk{{display:inline-block;padding:7px 0;animation:p8scroll var(--dur,240s) linear infinite}}
.p8-rib:hover .trk{{animation-play-state:paused}}
.p8-rib .tk{{display:inline-block;padding:0 16px;font-size:.8rem;border-right:1px solid rgba(128,128,128,.18)}}
.p8-rib .tk b{{font-weight:700;margin-right:6px}}
.p8-rib .tk.ix b{{color:var(--p8-amber)}}
.p8-rib .tk em{{font-style:normal;font-weight:600;margin-left:6px}}
@keyframes p8scroll{{from{{transform:translateX(0)}}to{{transform:translateX(-50%)}}}}
@media (prefers-reduced-motion:reduce){{.p8-rib .trk{{animation:none}}}}
.p8-plain{{font-size:.84rem;line-height:1.4;padding:7px 12px;margin:0 0 12px;border-left:3px solid #5aa9ff;
  background:rgba(90,169,255,.07);border-radius:0 8px 8px 0}}
@media(max-width:640px){{.p8-brand .tag{{display:none}}.block-container{{padding-top:3.4rem!important}}}}
</style>
"""


def _e(s) -> str:
    return html.escape(str(s))


@st.cache_data(ttl=55, show_spinner=False)
def live_indices() -> dict:
    """NSE's live index rows by name (one call), shared by every visitor for 55 seconds. {} if NSE is down."""
    try:
        from capstone import nse_mcp
        groups = nse_mcp.call("market", "cm_get_live_indices", {"group": "", "nameFilter": ""}) or {}
        return {i["indexName"]: i for g in (groups.get("indices") or {}).values() for i in g if i.get("indexName")}
    except Exception:
        return {}


def peek_quotes() -> tuple[pd.DataFrame | None, float]:
    """The shared NSE quote copy without waiting: starts a background download when there is none yet."""
    q, at = market_flow._latest.get("quotes"), market_flow._latest.get("at", 0.0)
    if q is None and not market_flow._latest.get("busy"):
        threading.Thread(target=market_flow.latest_quotes, daemon=True).start()
    elif q is not None:
        market_flow.latest_quotes()  # returns at once; refreshes in the background when over a minute old
    return q, at


def _chg(p) -> str:
    if p is None or pd.isna(p):
        return '<em style="opacity:.6">—</em>'
    col = UP if p > 0 else DOWN if p < 0 else "inherit"
    arrow = "▲" if p > 0 else "▼" if p < 0 else "■"
    return f'<em style="color:{col}">{arrow} {abs(p):.2f}%</em>'


def ribbon_items(indices: dict, quotes: pd.DataFrame | None, universe: pd.DataFrame) -> list[str]:
    """One span per index, then per stock of the universe in NSE's alphabetical order."""
    out = []
    for name in RIBBON_INDICES:
        ix = indices.get(name)
        if ix and ix.get("last") is not None:
            out.append(f'<span class="tk ix"><b>{_e(name)}</b>{ix["last"]:,.2f}{_chg(ix.get("percChange"))}</span>')
    if quotes is not None and len(quotes):
        q = quotes[quotes["symbol"].isin(universe["symbol"])].sort_values("symbol")
        for r in q.itertuples():
            if pd.notna(r.ltp):
                out.append(f'<span class="tk"><b>{_e(r.symbol)}</b>{float(r.ltp):,.2f}{_chg(r.pct_change)}</span>')
    return out


def header() -> None:
    """Brand bar + ticker ribbon, at the top of every page."""
    from capstone import market_calendar
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    s = market_calendar.status()
    dot = "#2a9d6f" if s.is_open else AMBER if s.state == "before_open" else "#8b93a3"
    state = "NSE OPEN" if s.is_open else s.headline
    indices = live_indices()
    quotes, at = peek_quotes()
    uni = market_flow.universe("NIFTY 100")
    stamp = ""
    if quotes is not None and len(quotes) and quotes["as_of"].notna().any():
        stamp = f'Quotes as of <b>{pd.Timestamp(quotes["as_of"].max()):%d %b %H:%M}</b>'
    st.markdown(GLOBAL_CSS + f"""
<div class="p8-brand"><div><span class="logo">◆ PULSE8</span><span class="tag">NSE market terminal · India opening
forecast for the 09:15 IST open</span></div><div class="meta"><span class="p8-st"><i style="background:{dot}"></i>
{_e(state)}</span><span>{now:%a %d %b %Y} · <b>{now:%H:%M}</b> IST</span><span>{stamp}</span></div></div>""",
                unsafe_allow_html=True)
    items = ribbon_items(indices, quotes, uni)
    if not items:
        st.markdown('<div class="p8-rib"><span class="trk" style="animation:none;padding-left:14px;opacity:.7">'
                    'NSE prices are loading…</span></div>', unsafe_allow_html=True)
        return
    track = "".join(items)
    dur = max(60, int(len(items) * 3.2))  # about the same reading speed however many stocks there are
    st.markdown(f'<div class="p8-rib" title="NSE live feed (1-3 minutes behind the exchange). Hover to pause.">'
                f'<div class="trk" style="--dur:{dur}s">{track}{track}</div></div>', unsafe_allow_html=True)


def plain(text: str) -> None:
    """The one-line 'In plain words' note at the top of an analysis page."""
    st.markdown(f'<div class="p8-plain"><b>In plain words:</b> {_e(text)}</div>', unsafe_allow_html=True)
