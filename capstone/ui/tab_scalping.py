"""Intraday scalping, Pulse8's main page: long and short setups on 1- to 5-minute bars, each one tested
walk-forward on real minute data with full NSE intraday charges, a practice (paper) account for scalps,
and the user's own loss limits. Logic lives in capstone/scalp_lab.py. The older opening-range planner is
kept below as one of the page's views."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import streamlit as st

from capstone import charts, intraday, safety, scalping
from capstone.ui import kite_view, price_view
from capstone.ui.common import raw_prices
from capstone.charts import CONFIG as CHART_CONFIG


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
    if "scalp_settings" in st.session_state:  # saved by this tab, kept while another tool is shown
        return st.session_state["scalp_settings"]
    g = st.session_state.get
    return Settings(g("s_cap", 100000), g("s_risk", 0.5) / 100, g("s_rew", 1.5), g("s_lev", 1.0), g("s_or", 5),
                    g("s_win", 20), g("s_buf", 0.05) / 100, g("s_brk", 20.0), g("s_bias", scalping.BIAS_SOURCES[0]))


def _restore_widgets() -> None:
    """Put back the values chosen earlier (or the defaults): Streamlit forgets a widget's value while
    it is hidden. The widgets below then take their values from here, so they pass no value=."""
    cfg = st.session_state.get("scalp_settings") or Settings(100000, 0.005, 1.5, 1.0, 5, 20, 0.0005, 20.0,
                                                               scalping.BIAS_SOURCES[0])
    saved = {"s_cap": cfg.capital, "s_risk": round(cfg.risk * 100, 4), "s_rew": cfg.reward, "s_lev": cfg.leverage,
             "s_or": cfg.or_min, "s_win": cfg.window, "s_buf": round(cfg.buffer_pct * 100, 6),
             "s_brk": cfg.brokerage, "s_bias": cfg.bias_src}
    for key, value in saved.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _settings() -> Settings:
    _restore_widgets()
    c1, c2, c3, c4 = st.columns(4)
    capital = c1.number_input("Trading capital (₹)", min_value=1000, step=5000, key="s_cap")
    risk = c2.slider("Risk per trade (% of capital)", 0.1, 2.0, step=0.1, key="s_risk") / 100
    reward = c3.slider("Target (× risk)", 1.0, 3.0, step=0.25, key="s_rew")
    leverage = c4.slider("Leverage (MIS margin)", 1.0, 5.0, step=0.5, key="s_lev",
                         help="Exposure allowed per trade as a multiple of capital. 1 = no borrowing.")
    c5, c6, c7, c8 = st.columns(4)
    or_min = c5.selectbox("Opening range (minutes)", [5, 10, 15], key="s_or")
    window = c6.slider("Trade window after 09:15 (minutes)", 10, 60, step=5, key="s_win")
    buffer_pct = c7.number_input("Breakout buffer (%)", 0.0, 0.5, step=0.01, key="s_buf") / 100
    brokerage = c8.number_input("Brokerage per order (₹)", 0.0, 50.0, step=1.0, key="s_brk")
    bias_src = st.radio("Trade direction", scalping.BIAS_SOURCES, horizontal=True, key="s_bias")
    cfg = Settings(capital, risk, reward, leverage, or_min, window, buffer_pct, brokerage, bias_src)
    st.session_state["scalp_settings"] = cfg
    return cfg


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


def _pnl_chart(trades: pd.DataFrame):
    """Net P&L of each trade (bars) and the running total (line with markers, visible even for one trade)."""
    t = trades.assign(date=pd.to_datetime(trades["date"])).sort_values("date")
    t = t.assign(equity=t["net_pnl"].cumsum(), label=t["date"].dt.strftime("%d %b %Y"))
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_bar(x=t["label"], y=t["net_pnl"], name="Trade P&L", marker_line_width=0,
                marker_color=np.where(t["net_pnl"] >= 0, charts.UP, charts.DOWN),
                customdata=np.c_[t["side"], t["exit_by"].fillna("")],
                hovertemplate="%{x}: %{customdata[0]}, exit by %{customdata[1]}<br>Net ₹%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=t["label"], y=t["equity"], name="Running total", mode="lines+markers",
                    line=dict(color="#2a78d6", width=2), marker=dict(size=8),
                    hovertemplate="Running total ₹%{y:,.0f}<extra></extra>")
    fig.add_hline(y=0, line=dict(color=charts.MUTED, width=1))
    charts.style(fig, height=420, title="Backtest: profit after charges, trade by trade")
    fig.update_xaxes(type="category", title_text="Trading day")
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f", title_text="₹")
    return fig


def _day_chart(mins: pd.DataFrame, bt: pd.DataFrame, cfg: Settings, name: str) -> None:
    """1-minute candles for one tested day with the opening range, entry, stop, target and exit drawn."""
    days = sorted(mins["datetime"].dt.date.unique(), reverse=True)
    if not days:
        return
    traded = set(pd.to_datetime(bt.loc[bt["side"] != "No trade", "date"]).dt.date) if len(bt) else set()
    c1, c2 = st.columns([2, 3])
    day = c1.selectbox("Show the minute chart for", days, key="scalp_day",
                       format_func=lambda d: f"{d:%a %d %b %Y}" + ("  · traded" if d in traded else ""))
    whole = c2.toggle("Show the whole session (not only the trading window)", key="scalp_whole")
    ax = price_view.axis_controls("scalp_ax", intraday=True, default_height=480)
    m = mins[mins["datetime"].dt.date == day]
    start = pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=15)
    if not whole:
        m = m[(m["datetime"] >= start) & (m["datetime"] < start + pd.Timedelta(minutes=cfg.window + 15))]
    if ax["bars"]:
        m = m.tail(int(ax["bars"]))
    if m.empty:
        st.info("No minute candles for this day in the trading window.")
        return
    row = bt[pd.to_datetime(bt["date"]).dt.date == day] if len(bt) else bt
    trade = None
    if len(row):
        r = row.iloc[0].to_dict()
        trade = {k: r.get(k) for k in ("or_high", "or_low", "entry", "stop", "target", "exit", "entry_time",
                                       "exit_time")}
        trade["or_end"] = start + pd.Timedelta(minutes=cfg.or_min)
        if r.get("side") == "No trade":
            trade.update(entry=None, stop=None, target=None, exit=None)
    fig = charts.intraday_chart(m, f"{name}: 1-minute candles, {day:%a %d %b %Y}", trade,
                                price_height=ax["price_height"])
    charts.apply_axes(fig, ax, base=float(m["close"].iloc[0]))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, key="scalp_day_chart")
    st.caption("Shaded box = opening range. Dashed lines = entry, stop and target; ▲ = entry, ✕ = exit. "
               "Orange line = VWAP (when volume is available).")


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
    mins = pd.DataFrame()
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
    name = bt_sym or (m_up.name if m_up is not None else "")
    if not len(trades):
        st.info("No trades triggered in this data with these settings. The minute chart below still shows "
                "each day's candles and opening range.")
        if len(mins):
            _day_chart(intraday.load(bt_sym) if bt_sym else mins, bt, cfg, name)
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Days tested", len(bt))
    k2.metric("Trades", len(trades))
    k3.metric("Win rate", f"{(trades['net_pnl'] > 0).mean() * 100:.1f}%")
    k4.metric("Net P&L after charges", f"₹{trades['net_pnl'].sum():,.0f}")
    st.plotly_chart(_pnl_chart(trades), use_container_width=True, config=CHART_CONFIG)
    _day_chart(intraday.load(bt_sym) if bt_sym else mins, bt, cfg, name)  # full columns: volume for VWAP
    st.dataframe(bt.round(2), use_container_width=True, hide_index=True)
    if "Model" in cfg.bias_src:
        st.caption("Past model forecasts aren't stored per day, so the backtest uses the actual "
                   "opening gap as the direction filter.")


def _classic(preds: pd.DataFrame) -> None:
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


# --- Main scalping workflow ---------------------------------------------------------------------------

import io  # noqa: E402

from capstone import scalp_lab  # noqa: E402

SCALP_JOURNAL = "scalp_journal"
VIEWS = ["Live setups", "Strategy test (walk-forward)", "Paper scalping", "Opening-range planner"]


def _stamp(syms: tuple) -> tuple:
    def mt(p):
        return p.stat().st_mtime if p.exists() else 0.0
    return tuple((s, mt(intraday.INTRADAY_DIR / f"{s}.csv"), mt(intraday.FIVE_DIR / f"{s}.csv")) for s in syms)


def _bars(syms: tuple, bar_min: int, history: str) -> dict[str, pd.DataFrame]:
    """Bars with indicators per stock. history: '1m' (about 30 days) or '5m' (about 60 days, 5-min bars only)."""
    out = {}
    for s in syms:
        m = intraday.load(s, five=(history == "5m"))
        if len(m) == 0:
            continue
        m = m.dropna(subset=["open", "high", "low", "close"])
        out[s] = scalp_lab.add_indicators(scalp_lab.resample(m, bar_min))
    return out


def _completed_days(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Leave out today's unfinished session from the test (its trades could still be open)."""
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    now_min = pd.Timestamp.now(tz="Asia/Kolkata")
    if now_min.hour * 60 + now_min.minute >= 15 * 60 + 30:
        return bars
    return {s: b[b["day"] != today] for s, b in bars.items()}


@st.cache_data(show_spinner=False, max_entries=6)
def _scoreboard(syms: tuple, bar_min: int, history: str, train_days: int, rules_key: tuple, stamp: tuple):
    rules = safety.Rules(*rules_key)
    bars = _completed_days(_bars(syms, bar_min, history))
    table, runs = scalp_lab.scoreboard(bars, rules, train_days)
    return table, {k: (v.trades, v.chosen, v.in_sample_best) for k, v in runs.items()}


def _rules_key(r) -> tuple:
    return (r.capital, r.risk_per_trade_pct, r.daily_loss_limit_pct, r.stop_loss_pct, r.max_trades_per_day,
            r.max_position_pct)


def _scalp_journal() -> pd.DataFrame:
    if SCALP_JOURNAL not in st.session_state:
        st.session_state[SCALP_JOURNAL] = scalp_lab.empty_scalp_journal()
    return st.session_state[SCALP_JOURNAL]


def _rs(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}₹{abs(x):,.0f}"


def _header() -> None:
    st.markdown("## Intraday scalping")
    st.warning(
        "**Read this first.** Scalping means many small, fast trades, and every trade pays brokerage, STT, exchange "
        "fees, GST, stamp duty, the bid-ask spread and slippage, about **0.18% of the trade value** for each "
        "buy-and-sell here. Most scalps aim for moves of 0.1% to 0.5%, so costs eat most of them. SEBI found that "
        "about 7 in 10 individual intraday traders lost money. **Do not trade money you need to live on.** This page "
        "tests every setup honestly and says plainly when none of them makes money after costs.")


def _data_bar(preds: pd.DataFrame) -> tuple[tuple, int, str, int]:
    syms_all = sorted(preds["symbol"])
    c = st.columns([3, 1.2, 1.4, 1.2])
    syms = tuple(c[0].multiselect("Stocks", syms_all, default=syms_all, key="sl_syms"))
    bar_min = c[1].selectbox("Bar size", [1, 3, 5], index=2, key="sl_bar", format_func=lambda m: f"{m}-minute")
    hist_opts = ["1m", "5m"] if bar_min == 5 else ["1m"]
    history = c[2].selectbox("History", hist_opts, key="sl_hist",
                             format_func=lambda h: "1-minute (about 30 days)" if h == "1m" else "5-minute (about 60 days)",
                             help="Free sources keep about 30 days of 1-minute bars and 60 days of 5-minute bars. "
                                  "Fetching every day makes the stored history grow.")
    train_days = int(c[3].number_input("Training days", 5, 30, 10, 1, key="sl_train",
                                       help="Each test day uses the settings that did best over this many days before it."))
    b1, b2 = st.columns([1, 3])
    if b1.button("Fetch latest minute data", key="sl_fetch", type="primary"):
        status = []
        with st.spinner("Fetching 1-minute and 5-minute candles..."):
            for s in syms:
                src, n = intraday.update(s)
                status.append({"Stock": s, "Source": src, "1-minute bars stored": n,
                               "5-minute bars stored": len(intraday.load(s, five=True))})
        st.session_state["sl_status"] = status
        st.cache_data.clear()
    stored = {s: len(intraday.load(s)) for s in syms}
    b2.caption("Stored now: " + ", ".join(f"{s} {n:,} min" for s, n in stored.items()) if stored else "")
    if st.session_state.get("sl_status"):
        st.dataframe(pd.DataFrame(st.session_state["sl_status"]), hide_index=True, use_container_width=True)
    return syms, bar_min, history, train_days


def _equity_chart(runs: dict):
    fig = make_subplots()
    for s, (trades, _chosen, _ib) in runs.items():
        if not len(trades):
            continue
        t = trades.sort_values("entry_time")
        fig.add_scatter(x=t["entry_time"], y=t["net"].cumsum(), mode="lines", name=scalp_lab.STRATEGIES[s],
                        line=dict(width=2, dash="dot" if s == "random" else "solid"),
                        hovertemplate="%{x|%d %b %H:%M}<br>Running total ₹%{y:,.0f}<extra></extra>")
    fig.add_hline(y=0, line=dict(color=charts.MUTED, width=1))
    charts.style(fig, height=380, title="Out-of-sample running total after all costs (₹)")
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    return fig


def _test_view(syms, bar_min, history, train_days, rules) -> tuple[pd.DataFrame, dict]:
    st.markdown("### Strategy test: would these setups have made money after costs?")
    st.caption(f"Walk-forward: for every day tested, each setup uses the settings that made the most after charges "
               f"over the {train_days} trading days before it, and only that unseen day's trades count. Entries fill at "
               "the bar after the signal; stops are assumed hit before targets in the same bar; all positions close "
               "by 15:15. Charges: brokerage, STT, exchange, SEBI, GST, stamp duty, plus spread and slippage on both legs.")
    with st.spinner("Testing every setup on real minute data..."):
        table, runs = _scoreboard(syms, bar_min, history, train_days, _rules_key(rules), _stamp(syms))
    if table.empty or table["Trades"].sum() == 0:
        st.info(f"Not enough minute data yet: the test needs more than {train_days} stored trading days. Click "
                "'Fetch latest minute data' above.")
        return table, runs
    winners = table[table["edge"]]
    if winners.empty:
        st.error("**No setup made money after costs in this test.** Every setup, long and short, lost money once "
                 "charges were paid, and none clearly beat the coin-flip baseline. The honest answer is: do not scalp "
                 "these stocks with these rules. Practise on paper if you want to learn, with virtual money only.")
    else:
        st.success(f"{len(winners)} setup(s) were profitable after costs in this test: "
                   + ", ".join(winners["Setup"]) + ". Past results are not a promise. Practise on paper first.")
    show = table.drop(columns=["strategy", "edge"])
    st.dataframe(show.style.format({"Win rate %": "{:.1f}", "Avg before costs %": "{:+.3f}", "Avg costs %": "{:.3f}",
                                    "Avg after costs %": "{:+.3f}", "Net ₹ after costs": "{:+,.0f}",
                                    "Worst drop ₹": "{:,.0f}"}),
                 hide_index=True, use_container_width=True)
    g = table[table["Trades"] > 0]
    st.caption(f"Read 'Avg before costs %' first: it is the setup's raw edge per trade. Across these setups it is "
               f"between {g['Avg before costs %'].min():+.3f}% and {g['Avg before costs %'].max():+.3f}%, while a round "
               f"trip costs about {g['Avg costs %'].mean():.2f}%. A setup needs its raw edge clearly above its costs.")
    st.plotly_chart(_equity_chart(runs), use_container_width=True, config=CHART_CONFIG, key="sl_equity")
    pick = st.selectbox("See the trades of", list(runs), format_func=lambda s: scalp_lab.STRATEGIES[s], key="sl_pick")
    trades, chosen, ib = runs[pick]
    st.caption(scalp_lab.PLAIN[pick])
    st.caption(f"For contrast, the single best settings picked with hindsight over all days ({ib['settings']}) made "
               f"{_rs(ib['net'])} over {ib['trades']} trades. Hindsight always looks better than real use.")
    if len(trades):
        t = trades.copy()
        st.dataframe(t[["symbol", "entry_time", "side", "entry", "stop", "target", "exit", "exit_by", "qty", "gross",
                        "charges", "net"]].round(2).sort_values("entry_time", ascending=False),
                     hide_index=True, use_container_width=True, height=320)
        st.download_button("Download these test trades (CSV)", t.to_csv(index=False).encode(),
                           f"pulse8_scalp_test_{pick}.csv", "text/csv", key="sl_dl_trades")
    with st.expander("Settings chosen each test day"):
        st.dataframe(chosen, hide_index=True, use_container_width=True)
    return table, runs


def _live_view(syms, bar_min, history, train_days, rules, table) -> None:
    st.markdown("### Live setups on today's bars")
    st.caption("A setup appears when a bar closes. The plan uses that close as a reference: entry fills at the next "
               "bar, stop and target come from the stock's recent bar range (ATR), and the size keeps the loss at the "
               "stop, charges included, inside your per-trade limit.")
    bars = _bars(syms, bar_min, "1m")
    if not bars:
        st.info("No minute data stored yet. Click 'Fetch latest minute data' above (after 09:15 on a trading day).")
        return
    edge = dict(zip(table["strategy"], table["edge"])) if len(table) else {}
    journal = _scalp_journal()
    paper = scalp_lab.scalp_paper_summary(journal)
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    used = int((pd.to_datetime(journal["placed_at"]).dt.date == today).sum()) if len(journal) else 0
    hist = _completed_days(_bars(syms, bar_min, history))
    rows = []
    for strat in [s for s in scalp_lab.STRATEGIES if s != "random"]:
        p = scalp_lab.best_recent(hist, strat, rules, train_days)
        for sym, b in bars.items():
            rows += scalp_lab.live_setups(b, p, rules, sym)
    latest = max(b["datetime"].max() for b in bars.values())
    st.caption(f"Latest stored bar: {latest:%a %d %b %H:%M} IST. Free data can lag a minute or two; fetch again to refresh.")
    if not rows:
        late = latest.hour * 60 + latest.minute >= scalp_lab.LAST_ENTRY
        st.info("No new trades after 14:45: there is too little time left before the 15:15 square-off." if late else
                "No setup fired on the last few bars. No setup means no trade: waiting costs nothing.")
    else:
        df = (pd.DataFrame(rows).sort_values("time", ascending=False)
              .drop_duplicates(["symbol", "strategy"]).reset_index(drop=True))  # latest setup per stock and setup
        losers = df[df["profit_at_target"] <= 0]
        df = df[df["profit_at_target"] > 0].reset_index(drop=True)
        if len(losers):
            st.warning(f"{len(losers)} setup(s) hidden: even hitting the target would lose money after charges at the "
                       "size your limits allow (" + ", ".join(losers["symbol"] + " " + losers["side"]) + ").")
        df["need_win"] = df["loss_at_stop"] / (df["loss_at_stop"] + df["profit_at_target"].clip(lower=0)) * 100
    if rows and len(df):
        st.dataframe(pd.DataFrame({
            "Time": df["time"].dt.strftime("%H:%M"), "Stock": df["symbol"], "Side": df["side"],
            "Setup": df["strategy"].map(scalp_lab.STRATEGIES), "Ref entry ₹": df["ref_entry"].round(2),
            "Stop ₹": df["stop"].round(2), "Target ₹": df["target"].round(2), "Qty": df["qty"],
            "Charges ₹": df["charges"].round(0), "Loss at stop ₹": df["loss_at_stop"].round(0),
            "Profit at target ₹": df["profit_at_target"].round(0),
            "Move just to cover costs %": df["move_to_breakeven_pct"].round(3),
            "Must win % of the time to break even": df["need_win"].round(0),
            "Tested edge?": df["strategy"].map(lambda s: "Yes" if edge.get(s) else "No")}),
            hide_index=True, use_container_width=True)
        st.caption("'Must win % of the time' = loss at stop ÷ (loss at stop + profit at target). Charges shrink the "
                   "profit and grow the loss, so a 1:1.5 plan can need far more than 40% winners. The test above shows "
                   "how often each setup actually won.")
        for strat in df["strategy"].unique():
            tone, why = scalp_lab.gate(bool(edge.get(strat)), paper, used, rules)
            {"stop": st.error, "paper": st.warning, "go": st.success}[tone](f"**{scalp_lab.STRATEGIES[strat]}:** {why}")
        opts = list(range(len(df)))
        k = st.selectbox("Practise one of these on paper", opts, key="sl_live_pick",
                         format_func=lambda i: f"{df.iloc[i]['time']:%H:%M} {df.iloc[i]['side']} {df.iloc[i]['qty']} "
                                               f"{df.iloc[i]['symbol']} ({scalp_lab.STRATEGIES[df.iloc[i]['strategy']]})")
        if st.button("Place practice scalp (virtual money)", key="sl_live_place"):
            setup = df.iloc[k].to_dict()
            ok, why = scalp_lab.can_place_scalp(journal, rules, today, setup["symbol"], int(setup["qty"]),
                                                float(setup["ref_entry"]), float(setup["stop"]))
            if ok:
                st.session_state[SCALP_JOURNAL] = scalp_lab.place_scalp(journal, _ist_now(), setup)
                st.success("Practice scalp saved. It fills at the next minute's open; see 'Paper scalping'.")
            else:
                st.error(why)
    sym = st.selectbox("Chart", list(bars), key="sl_chart_sym")
    m = intraday.load(sym)
    day = m["datetime"].dt.date.max() if len(m) else None
    if day is not None:
        from capstone.ui import tv_chart
        tv_chart.render(sym, m[m["datetime"].dt.date == day], dark=True, height=480,
                        source="Yahoo Finance / Twelve Data minute bars")


def _ist_now():
    return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)


def _paper_view(syms, rules) -> None:
    st.markdown("### Paper scalping: practise with virtual money")
    st.caption(f"You start with ₹{safety.PAPER_START:,.0f} of virtual money. A practice scalp fills at the open of the "
               "first minute after you place it and closes at its stop, its target, after 60 minutes or at 15:15, "
               "using real 1-minute candles and full intraday charges. Your limits apply: trades per day, loss per "
               f"trade and the daily loss limit. Practise until at least {safety.PAPER_TRADES_TO_GRADUATE} scalps "
               "show a profit after charges.")
    journal = scalp_lab.settle_scalps(_scalp_journal(), {s: intraday.load(s) for s in syms})
    st.session_state[SCALP_JOURNAL] = journal
    sm = scalp_lab.scalp_paper_summary(journal)
    c = st.columns(4)
    c[0].metric("Virtual balance", f"₹{sm['balance']:,.0f}", f"{_rs(sm['net'])} after charges",
                delta_color="normal" if sm["closed"] else "off")
    c[1].metric("Scalps closed", f"{sm['closed']} of {safety.PAPER_TRADES_TO_GRADUATE}", f"{sm['open']} open", delta_color="off")
    c[2].metric("Made money after charges", f"{sm['wins']} of {sm['closed']}" if sm["closed"] else "–")
    c[3].metric("Charges paid", f"₹{sm['charges']:,.0f}")
    with st.form("sl_manual", border=True):
        st.markdown("**Manual practice scalp** (fills at the next minute's open)")
        f = st.columns(5)
        sym = f[0].selectbox("Stock", list(syms), key="sl_m_sym")
        side = f[1].selectbox("Direction", ["BUY", "SELL"], key="sl_m_side",
                              help="SELL = sell first, buy back before 15:15 (intraday short).")
        m = intraday.load(sym)
        ref = float(m["close"].iloc[-1]) if len(m) else 0.0
        stop_pct = f[2].number_input("Stop (% from entry)", 0.1, 3.0, 0.4, 0.05, key="sl_m_stop")
        rr = f[3].number_input("Target (× risk)", 0.5, 4.0, 1.5, 0.25, key="sl_m_rr")
        sgn = 1 if side == "BUY" else -1
        stop = ref * (1 - sgn * stop_pct / 100)
        target = ref * (1 + sgn * rr * stop_pct / 100)
        qty = scalp_lab.size_for(ref, stop, rules) if ref else 0
        f[4].markdown(f"Last ₹{ref:,.2f}  \nQty {qty}")
        if st.form_submit_button("Place practice scalp"):
            setup = {"symbol": sym, "side": side, "qty": qty, "ref_entry": ref, "stop": stop, "target": target,
                     "strategy": "manual"}
            ok, why = scalp_lab.can_place_scalp(journal, rules, _ist_now().date(), sym, qty, ref, stop)
            if ok:
                st.session_state[SCALP_JOURNAL] = journal = scalp_lab.place_scalp(journal, _ist_now(), setup)
                st.success(f"Saved: {side} {qty} {sym}, stop ₹{stop:,.2f}, target ₹{target:,.2f}.")
            else:
                st.error(why)
    if len(journal):
        st.dataframe(journal.drop(columns=["planned_risk"], errors="ignore").sort_values("id", ascending=False),
                     hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    buf = io.StringIO()
    journal.to_csv(buf, index=False)
    c1.download_button("Save my scalp journal (CSV)", buf.getvalue(), "pulse8_scalp_journal.csv", "text/csv",
                       help="Practice scalps live only in this browser tab. Save them and load the file next time.")
    up = c2.file_uploader("Load a saved scalp journal", type="csv", key="sl_upload", label_visibility="collapsed")
    if up is not None and st.session_state.get("sl_loaded") != up.file_id:
        loaded = pd.read_csv(up)
        missing = [k for k in scalp_lab.SCALP_COLS if k not in loaded.columns]
        if missing:
            st.error(f"That file is not a Pulse8 scalp journal (missing {', '.join(missing)}).")
        else:
            st.session_state[SCALP_JOURNAL] = loaded
            st.session_state["sl_loaded"] = up.file_id
            st.rerun()


def render(preds: pd.DataFrame) -> None:
    from capstone.ui import tab_safety
    _header()
    with st.expander("Your loss limits (shared with the safety page)", expanded=False):
        rules = tab_safety._rules()
    st.caption(f"Limits: at most ₹{rules.risk_per_trade:,.0f} lost per trade, ₹{rules.daily_loss_limit:,.0f} per day, "
               f"{rules.max_trades_per_day} trades a day, no leverage.")
    syms, bar_min, history, train_days = _data_bar(preds)
    if not syms:
        st.info("Pick at least one stock.")
        return
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], key="sl_view", label_visibility="collapsed") or VIEWS[0]
    if view == "Opening-range planner":
        _classic(preds)
        return
    if view == "Paper scalping":
        _paper_view(syms, rules)
        return
    if view == "Strategy test (walk-forward)":
        _test_view(syms, bar_min, history, train_days, rules)
        return
    table, _runs = _scoreboard(syms, bar_min, history, train_days, _rules_key(rules), _stamp(syms))
    if len(table) and table["Trades"].sum() and not table["edge"].any():
        st.error("In the walk-forward test (see 'Strategy test'), no setup made money after costs. The setups below "
                 "are for practice on paper only.")
    _live_view(syms, bar_min, history, train_days, rules, table)
