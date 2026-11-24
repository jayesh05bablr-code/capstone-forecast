"""Market overview: the morning brief, built only from data the app has fetched.

Each block names its source and the date of its data; a block with no data says so. The page never
writes a forecast or opinion that is not already in the model's output.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import breadth, brief, flow_history, market_flow, sector_intel, stock_metrics
from capstone.ui.headline import rupee_delta


def freshness(source: str, as_of, live: bool = False) -> None:
    """Grey caption naming the source and when the data is from, with a dot.

    End-of-day data is green when it is from the latest session NSE has published (weekends and NSE
    holidays do not make it stale), amber when older. Live data (`live=True`) is green within 15 minutes.
    """
    if as_of is None or pd.isna(as_of):
        st.caption(f"⚪ {source} · no date")
        return
    ts = pd.Timestamp(as_of)
    if live:
        now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
        fresh = (now - ts).total_seconds() <= 900
    else:
        fresh = ts.normalize() >= pd.Timestamp(flow_history.expected_last_session())
    when = f"{ts:%d %b %Y %H:%M} IST" if (ts.hour or ts.minute) else f"{ts:%d %b %Y}"
    st.caption(f"{'🟢' if fresh else '🟠'} {source} · as of {when}" + ("" if fresh else " (not the latest session)"))


def _pct(v, digits=2):
    return f"{v:+.{digits}f}%" if v is not None and pd.notna(v) else "n/a"


def _kpis(mood, val, live, story) -> None:
    c = st.columns(6)
    if live:
        c[0].metric("NIFTY 50", f"{live['last']:,.2f}", _pct(live.get("pct")),
                    help=f"NSE live index feed, {live.get('time')}. Before 09:15 this is the last close "
                         "(or the pre-open value once NSE publishes it).")
    else:
        c[0].metric("NIFTY 50", "n/a")
    vix = (mood or {}).get("india_vix") or {}
    pctl = (vix.get("range") or {}).get("percentile")
    c[1].metric("India VIX", f"{vix['level']:.2f}" if vix.get("level") else "n/a", _pct(vix.get("change_1d_pct")),
                delta_color="inverse",
                help=f"Expected volatility. Higher than on {pctl:.0f}% of days in the past year." if pctl else None)
    sb = (mood or {}).get("stock_breadth") or {}
    c[2].metric("NSE stocks up", f"{sb['up_pct']:.0f}%" if sb.get("up_pct") is not None else "n/a",
                help=f"{sb.get('up', '?')} rose, {sb.get('down', '?')} fell last session ({sb.get('reading', '')}).")
    if story and story.get("usdinr"):
        c[3].metric("USD/INR", f"₹{story['usdinr']:.2f}", rupee_delta(story.get("usdinr_chg")), delta_color="off")
    else:
        c[3].metric("USD/INR", "n/a")
    pe = (val or {}).get("pe") or {}
    c[4].metric("NIFTY 50 P/E", f"{pe['current']:.1f}" if pe.get("current") else "n/a",
                help=f"{pe.get('percentile', 0):.0f}th percentile of the last 2 years (NSE)." if pe.get("current") else None)
    c[5].metric("Model's opening call", story.get("mood", "n/a") if story else "n/a",
                help="Average expected gap of the forecast stocks (see Stock forecasting).")


def _us(story, moves: pd.DataFrame) -> None:
    st.markdown("##### 🇺🇸 US overnight")
    if story is None or moves is None or moves.empty:
        st.caption("No overnight US data yet.")
        return
    m = moves.reset_index() if "ticker" not in moves.columns else moves
    idx = m[~m["is_sector"]].dropna(subset=["ret"])
    sec = m[m["is_sector"]].dropna(subset=["ret"]).sort_values("ret", ascending=False)
    lines = [f"**{r['name']}** {100 * r['ret']:+.2f}%" for _, r in idx.iterrows()]
    if len(sec):
        lines.append(f"Best sector: {sec.iloc[0]['name']} {100 * sec.iloc[0]['ret']:+.2f}%; weakest: "
                     f"{sec.iloc[-1]['name']} {100 * sec.iloc[-1]['ret']:+.2f}%")
    st.markdown("  \n".join(lines))
    if not story.get("overnight_complete", True):
        st.caption("⚠️ The latest US session is not in the data yet.")
    freshness(f"US ETF closes (Twelve Data / Alpha Vantage), session {story.get('us_session') or 'n/a'}",
              pd.Timestamp(story["as_of_close"]) if story.get("as_of_close") else None)


def _asia() -> None:
    st.markdown("##### 🌏 Asia")
    rows = brief.asia()
    if not rows:
        st.caption("Asian index prices are not available right now (Yahoo Finance did not answer).")
        return
    st.markdown("  \n".join(f"**{r['name']}** {r['close']:,.0f} ({r['pct']:+.2f}%)" for r in rows))
    freshness("Yahoo Finance, last traded / close", max(r["as_of"] for r in rows))


def _india(mood, live, story) -> None:
    st.markdown("##### 🇮🇳 India before the open")
    bits = []
    if live:
        bits.append(f"**NIFTY 50** {live['last']:,.2f} ({_pct(live.get('pct'))} vs previous close {live.get('prev'):,.2f})"
                    if live.get("prev") else f"**NIFTY 50** {live['last']:,.2f}")
        if live.get("chg_1m") is not None:
            bits.append(f"One month: {live['chg_1m']:+.2f}%")
    bm = (mood or {}).get("benchmarks") or {}
    others = [f"{k} {v['change_pct']:+.2f}%" for k, v in bm.items() if k != "Nifty 50" and v.get("change_pct") is not None]
    if others:
        bits.append("Last session: " + ", ".join(others))
    if story and story.get("lines"):
        bits.append("Model: " + story["lines"][0])
    st.markdown("  \n".join(bits) if bits else "No NSE data right now.")
    freshness("NSE live index feed and bhavcopy (mcp.nseindia.in)",
              pd.to_datetime(live["time"], format="%d-%b-%Y %H:%M", errors="coerce") if live and live.get("time") else
              (pd.Timestamp(mood["date"]) if mood and mood.get("date") else None))


def _sectors() -> None:
    st.markdown("##### 🧭 Sector outlook (relative strength)")
    sector_intel.refresh_indices()
    ix = sector_intel.load_indices()
    lb = sector_intel.leaderboard(ix) if len(ix) else pd.DataFrame()
    if lb.empty:
        st.caption("Sector index history is downloading.")
        return
    from capstone.ui.tab_sector_intel import _short
    lead = lb[lb["quadrant"] == "Leading"].head(4)
    lag = lb[lb["quadrant"] == "Lagging"].tail(4)
    imp = lb[lb["quadrant"] == "Improving"].sort_values("rs_mom", ascending=False).head(3)
    lines = []
    if len(lead):
        lines.append("**Leading:** " + ", ".join(f"{_short(i)} ({v:+.1f})" for i, v in zip(lead["index"], lead["rs_20d"])))
    if len(imp):
        lines.append("**Improving:** " + ", ".join(_short(i) for i in imp["index"]))
    if len(lag):
        lines.append("**Lagging:** " + ", ".join(f"{_short(i)} ({v:+.1f})" for i, v in zip(lag["index"], lag["rs_20d"])))
    st.markdown("  \n".join(lines))
    st.caption("Points = 20-session return minus the NIFTY 50's. A description of recent strength, not a forecast.")
    freshness("NSE sector indices (bhavcopy)", lb["as_of"].max())


def _actions(uni50: pd.DataFrame) -> None:
    st.markdown("##### 📅 Corporate actions, next 7 days (NIFTY 50)")
    brief.refresh_actions(list(uni50["symbol"]))
    a = brief.upcoming_actions()
    raw = brief.load_actions()
    if raw.empty:
        st.caption("Checking NSE for upcoming ex-dates (about 15 seconds, once a day).")
        return
    if a.empty:
        st.markdown("No dividend, split or bonus ex-dates for NIFTY 50 stocks in the next 7 days.")
    else:
        st.markdown("  \n".join(f"**{r.symbol}** · ex-date {pd.Timestamp(r.ex_date):%a %d %b} · {r.purpose}"
                                for r in a.itertuples()))
    freshness("NSE corporate actions (bhavcopy MCP)", pd.Timestamp(raw["fetched"].iloc[0]))


def _risks(mood, val, story) -> None:
    st.markdown("##### ⚠️ Risks to watch")
    uni = market_flow.universe("NIFTY 100")
    bh = breadth.breadth_history(flow_history.load(), uni)
    alerts = breadth.divergences(stock_metrics.nifty_history(), bh) if len(bh) else []
    notes = brief.risks(mood, alerts, story, val)
    st.markdown("\n".join(f"- {n}" for n in notes) if notes else "No elevated-risk readings in the data shown here.")
    freshness("NSE market mood, valuation and NIFTY 100 breadth", pd.Timestamp(mood["date"]) if mood else None)


def _forecast(preds: pd.DataFrame, story) -> None:
    st.markdown("##### 🎯 Forecast highlights (model output)")
    h = brief.forecast_highlights(preds)
    if not h["up"] and not h["down"]:
        st.caption("No forecast available.")
        return
    lines = [f"**{r['symbol']}** expected gap {r['pred_gap_pct']:+.2f}% (open ≈ ₹{r['pred_open']:,.2f})"
             for r in h["up"] + h["down"]]
    st.markdown("  \n".join(lines))
    st.caption("Forecasts of the 09:15 open from the trained model; check Model accuracy before relying on them.")
    if story:
        freshness(f"Pulse8 model, for the open of {story.get('next_open')}", pd.Timestamp(story["as_of_close"]))


def render(preds: pd.DataFrame, story: dict | None, moves: pd.DataFrame | None) -> None:
    mood, val, live = brief.market_mood(), brief.valuation(), brief.nifty_live()
    st.subheader(f"Morning brief · {pd.Timestamp.now(tz='Asia/Kolkata'):%A %d %B %Y}")
    _kpis(mood, val, live, story)
    st.divider()
    a, b, c = st.columns(3)
    with a:
        _us(story, moves)
    with b:
        _asia()
    with c:
        _india(mood, live, story)
    st.divider()
    a, b = st.columns(2)
    with a:
        _sectors()
        _actions(market_flow.universe("NIFTY 50"))
    with b:
        _risks(mood, val, story)
        _forecast(preds, story)
    st.caption("Every line above is read from the named source; nothing is a recommendation to buy or sell.")
