"""Intraday scalping tab: opening-range breakout plan for the first minutes after 09:15."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from capstone import charts, intraday, scalping
from capstone.ui import kite_view
from capstone.ui.common import raw_prices


@dataclass
class Settings:
    capital: float
    risk: float          # fraction of capital risked per trade
    reward: float        # target as a multiple of the risk
    leverage: float
    or_min: int          # opening-range length in minutes
    window: int          # minutes after 09:15 to stay in a trade
    buffer_pct: float    # fraction above/below the range to confirm a breakout
    brokerage: float     # ₹ per order
    bias_src: str        # one of scalping.BIAS_SOURCES


def settings_from_state() -> Settings:
    """The settings chosen on this tab (or the defaults before the tab has drawn), for other tabs."""
    g = st.session_state.get
    return Settings(g("s_cap", 100000), g("s_risk", 0.5) / 100, g("s_rew", 1.5), g("s_lev", 1.0), g("s_or", 5),
                    g("s_win", 20), g("s_buf", 0.05) / 100, g("s_brk", 20.0), g("s_bias", scalping.BIAS_SOURCES[0]))


def _settings() -> Settings:
    c1, c2, c3, c4 = st.columns(4)
    capital = c1.number_input("Trading capital (₹)", min_value=1000, value=100000, step=5000, key="s_cap")
    risk = c2.slider("Risk per trade (% of capital)", 0.1, 2.0, 0.5, 0.1, key="s_risk") / 100
    reward = c3.slider("Target (× risk)", 1.0, 3.0, 1.5, 0.25, key="s_rew")
    leverage = c4.slider("Leverage (MIS margin)", 1.0, 5.0, 1.0, 0.5, key="s_lev",
                         help="Exposure allowed per trade as a multiple of capital. 1 = no borrowing.")
    c5, c6, c7, c8 = st.columns(4)
    or_min = c5.selectbox("Opening range (minutes)", [5, 10, 15], index=0, key="s_or")
    window = c6.slider("Trade window after 09:15 (minutes)", 10, 60, 20, 5, key="s_win")
    buffer_pct = c7.number_input("Breakout buffer (%)", 0.0, 0.5, 0.05, 0.01, key="s_buf") / 100
    brokerage = c8.number_input("Brokerage per order (₹)", 0.0, 50.0, 20.0, 1.0, key="s_brk")
    bias_src = st.radio("Trade direction", scalping.BIAS_SOURCES, horizontal=True, key="s_bias")
    return Settings(capital, risk, reward, leverage, or_min, window, buffer_pct, brokerage, bias_src)


def _status_row(sym: str, source: str, minutes: int, latest) -> dict:
    return {"Stock": sym, "Source": source, "Minutes stored": minutes, "Latest candle (IST)": latest}


def _live_data(preds: pd.DataFrame) -> None:
    """Fetch, upload or download 1-minute candles."""
    st.markdown("**Live 1-minute data.** After 09:15 + the opening range, click to pull today's "
                "1-minute candles for your stocks from the internet (Yahoo Finance, no key needed; "
                "Twelve Data as a backup if your plan covers NSE). The real opening range then "
                "replaces the estimate below, and the candles are kept for backtesting in Step 3.")
    if st.button("Fetch 1-minute data from the internet", key="fetch_min"):
        status = []
        with st.spinner("Fetching 1-minute candles..."):
            for sym in preds["symbol"]:
                src, n = intraday.update(sym)
                m = intraday.load(sym)
                status.append(_status_row(sym, src, n, m["datetime"].max() if len(m) else None))
        st.session_state["min_status"] = status
    if st.session_state.get("min_status"):
        st.dataframe(pd.DataFrame(st.session_state["min_status"]), hide_index=True, use_container_width=True)
        st.caption("Free 1-minute data can lag the market by a minute or two. Click again to refresh.")

    with st.expander("Or upload today's opening data yourself (CSV or Excel)"):
        st.markdown("Two kinds of file work here:\n"
                    "- **NSE pre-open file** (nseindia.com, Market Data, Pre-Open Market, Download CSV, "
                    "after 09:08): the real opening price of every stock. The stock picker is ignored.\n"
                    "- **1-minute candles** for one stock (`datetime, open, high, low, close`, or `date, time, ...`), "
                    "e.g. from your broker. The candles from 09:15 set the real opening range and are kept for Step 3.")
        u1, u2 = st.columns([1, 2])
        o_sym = u1.selectbox("Stock in this file", preds["symbol"], key="open_sym")
        o_up = u2.file_uploader("Opening minutes", type=["csv", "xlsx", "xls"], key="open_upload")
        if o_up is not None and st.session_state.get("open_saved") != (o_up.file_id, o_sym):
            _save_opening_upload(preds, o_sym, o_up)

    stored = [s for s in preds["symbol"] if len(intraday.load(s))]
    if stored:
        d1, d2 = st.columns([1, 2])
        dl_sym = d1.selectbox("Download stored 1-minute data", stored, key="dl_sym")
        d2.download_button(f"Download {dl_sym} 1-minute candles (CSV)",
                           intraday.load(dl_sym).to_csv(index=False).encode(),
                           file_name=f"{dl_sym}_1min.csv", mime="text/csv")


def _save_opening_upload(preds: pd.DataFrame, sym: str, up) -> None:
    """Store an uploaded NSE pre-open file or one stock's 1-minute candles, then rerun."""
    pre = scalping.read_preopen(up.name, up.getvalue())
    if pre is not None:
        intraday.save_preopen(pre)
        st.session_state["open_saved"] = (up.file_id, sym)
        mine = pre[pre["symbol"].isin(preds["symbol"])]
        st.session_state["min_status"] = [_status_row(r.symbol, f"NSE pre-open {r.date}", 0, None)
                                          for r in mine.itertuples()]
        st.rerun()
    try:
        got = scalping.read_minutes(up.name, up.getvalue())
        kept = intraday.save(sym, got.assign(volume=got.get("volume")))
        st.session_state["open_saved"] = (up.file_id, sym)
        st.session_state["min_status"] = [_status_row(sym, f"Upload: {up.name}", len(kept), kept["datetime"].max())]
        st.rerun()
    except (ValueError, KeyError) as exc:
        st.error(f"Could not use this file: {exc}")


def _opening_ranges(preds: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Step 1: estimated or real opening range per stock, editable by the user."""
    today_ist = pd.Timestamp.now(tz="Asia/Kolkata").date()
    st.markdown(f"**Step 1. Opening range.** Before 09:{15 + cfg.or_min:02d} the table shows estimates from "
                f"each stock's volatility. After that, fetch live data (above) or type in the real open, "
                f"high and low of the first {cfg.or_min} minutes, and the plan updates.")
    daily = {s: raw_prices(s) for s in preds["symbol"]}
    live = {s: intraday.opening_range(intraday.load(s), today_ist, cfg.or_min)
            for s, h in daily.items() if h is not None and len(h) >= 30}
    est = scalping.range_rows(preds, daily, live, cfg.or_min)
    n_live = sum(e["Source"] != "Estimate" for e in est)
    pre_today = intraday.preopen(today_ist).set_index("symbol")
    if len(pre_today):
        n_live += scalping.apply_preopen(est, pre_today)
        chk = scalping.forecast_vs_open(preds, pre_today)
        if len(chk):
            st.markdown(f"**Today's forecast vs the real open** (NSE pre-open, {today_ist})")
            st.dataframe(chk, hide_index=True, use_container_width=True)
            st.caption("Uses the last close the model saw. If NSE's previous close differs (e.g. predict was run "
                       "on older data), the gaps will be off by that difference.")
    return st.data_editor(pd.DataFrame(est), disabled=["Stock", "Last close", "Model gap %", "Source"],
                          hide_index=True, use_container_width=True, key=f"or_{cfg.or_min}_{today_ist}_{n_live}")


def _plan(ranges: pd.DataFrame, cfg: Settings) -> None:
    """Step 2: entry, stop, target and size for each breakout."""
    rows = []
    for r in ranges.to_dict("records"):
        plan = scalping.plan_row(r["Stock"], r["Open"], r["Range high"], r["Range low"],
                                 scalping.bias(r, cfg.bias_src), cfg.capital, cfg.risk, cfg.reward,
                                 cfg.leverage, cfg.buffer_pct, cfg.brokerage)
        rows += list(plan.values())
    st.markdown("**Step 2. The plan**")
    if not rows:
        st.info("No setups: check the range high is above the range low.")
        return
    sp = pd.DataFrame(rows)
    st.dataframe(pd.DataFrame({
        "Stock": sp["symbol"], "Side": sp["side"],
        "Enter if price crosses (₹)": sp["trigger"].round(2), "Stop (₹)": sp["stop"].round(2),
        "Target (₹)": sp["target"].round(2), "Qty": sp["qty"], "Exposure (₹)": sp["exposure"].round(0),
        "Net profit at target (₹)": sp["net_profit_at_target"].round(0),
        "Net loss at stop (₹)": sp["net_loss_at_stop"].round(0),
        "Charges (₹)": sp["charges"].round(0), "Move to break even (₹/share)": sp["breakeven_move"].round(2),
    }), use_container_width=True, hide_index=True)
    st.caption("Charges are approximate NSE intraday costs (brokerage, STT 0.025% on the sell side, "
               f"exchange, SEBI, stamp duty, GST). Exit everything by {scalping.exit_clock(cfg.window)}.")
    kite_view.render_scalps([p for p in rows if p["qty"] > 0], key="scalp_kite")


def _backtest(preds: pd.DataFrame, cfg: Settings) -> None:
    """Step 3: run the rule over stored or uploaded 1-minute candles."""
    st.markdown("**Step 3. Test the rule on real minute data (optional)**")
    st.markdown("Upload 1-minute candles for one stock (CSV or Excel with `datetime, open, high, low, close`, "
                "or `date, time, ...`), e.g. exported from your broker, or use the data fetched above. "
                "Free sources keep only about 7 days of 1-minute bars, so fetch daily to build history.")
    m_up = st.file_uploader("1-minute candles", type=["csv", "xlsx", "xls"], key="minute_upload")
    fetched = [s for s in preds["symbol"] if len(intraday.load(s))]
    bt_sym = None
    if m_up is None and fetched:
        bt_sym = st.selectbox("Or test on the 1-minute data fetched from the internet", fetched, key="bt_sym")
    if m_up is None and not bt_sym:
        return
    try:
        mins = scalping.read_minutes(m_up.name, m_up.getvalue()) if m_up is not None \
            else intraday.load(bt_sym)[["datetime", "open", "high", "low", "close"]]
        mode = "Both directions" if cfg.bias_src == "Both directions" else "Gap direction only"
        bt = scalping.backtest(mins, cfg.or_min, cfg.window, cfg.reward, cfg.buffer_pct, mode,
                               cfg.risk * cfg.capital, cfg.brokerage)
    except (ValueError, KeyError) as exc:
        st.error(f"Could not use this file: {exc}")
        bt = pd.DataFrame()
    trades = bt[bt["side"] != "No trade"] if len(bt) else bt
    if not len(trades):
        st.info("No trades triggered in this data with these settings.")
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Days tested", len(bt))
    k2.metric("Trades", len(trades))
    k3.metric("Win rate", f"{(trades['net_pnl'] > 0).mean() * 100:.1f}%")
    k4.metric("Net P&L after charges", f"₹{trades['net_pnl'].sum():,.0f}")
    curve = trades.assign(date=pd.to_datetime(trades["date"]), equity=trades["net_pnl"].cumsum())
    st.plotly_chart(charts.line(curve, "date", {"equity": "Cumulative net P&L"},
                                "Backtest: cumulative profit after charges", "₹"),
                    use_container_width=True)
    st.dataframe(bt.round(2), use_container_width=True, hide_index=True)
    if "Model" in cfg.bias_src:
        st.caption("Past model forecasts aren't stored per day, so the backtest uses the actual "
                   "opening gap as the direction filter.")


def render(preds: pd.DataFrame) -> None:
    st.warning("Educational tool, not trading advice. Scalping the first minutes is the riskiest "
               "part of the day: spreads are wide and prices jump. SEBI's 2024 study found about 7 in "
               "10 individual intraday traders in the cash segment lost money in FY23. Paper-trade "
               "first and always use the stop.")
    st.markdown("**The rule (opening-range breakout).** Wait for the first few minutes after 09:15 to "
                "set a high and a low. Buy if price breaks above the high, or short if it breaks below "
                "the low, only in the direction of the bias. Stop at the other side of the range, "
                "target a multiple of the risk, and close whatever is open when the window ends.")
    cfg = _settings()
    _live_data(preds)
    ranges = _opening_ranges(preds, cfg)
    _plan(ranges, cfg)
    _backtest(preds, cfg)
