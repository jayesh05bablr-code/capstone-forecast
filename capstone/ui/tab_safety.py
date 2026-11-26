"""Trade safely: "Should I trade today?", loss limits and position size, paper trading, and the honest
after-charges record of the app's own signals. Logic lives in capstone/safety.py.

Practice trades are kept in this browser session only (Streamlit Cloud has no per-user storage), so
the page offers a download of the journal and loads it back from the file.
"""
from __future__ import annotations

import io

import altair as alt
import pandas as pd
import streamlit as st

from capstone import guidance, safety
from capstone.paths import MODELS, PROCESSED, RAW
from capstone.ui import data_cache
from capstone.ui.common import raw_prices

TONE = {"stop": st.error, "paper": st.warning, "go": st.success}
JOURNAL_KEY = "paper_journal"


def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _raw_stamp() -> tuple:
    return tuple(sorted((p.name, p.stat().st_mtime) for p in RAW.glob("*.csv")))


@st.cache_data(show_spinner=False, max_entries=4)
def _signal_trades(source: str, cost_pct: float, min_move: float, stamp: tuple) -> pd.DataFrame:
    if source == "oos":
        sig = safety.oos_signals(data_cache.oos())
    else:
        path = PROCESSED / "predictions_log.csv"
        sig = safety.log_signals(pd.read_csv(path) if path.exists() else None)
    syms = sorted(sig["symbol"].unique()) if len(sig) else []
    return safety.signal_trades(sig, {s: raw_prices(s) for s in syms}, cost_pct, min_move)


@st.cache_data(show_spinner=False, max_entries=2)
def _confidence_table(stamp: float) -> pd.DataFrame:
    return guidance.confidence_table(data_cache.oos())


def _rs(x: float, sign: bool = True) -> str:
    """Rupees with the sign in front: +₹1,200 / -₹93,580."""
    lead = ("+" if x >= 0 else "-") if sign else ("-" if x < 0 else "")
    return f"{lead}₹{abs(x):,.0f}"


def _now():
    return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).to_pydatetime()


def _journal() -> pd.DataFrame:
    if JOURNAL_KEY not in st.session_state:
        st.session_state[JOURNAL_KEY] = safety.empty_journal()
    return st.session_state[JOURNAL_KEY]


# --- Sections ------------------------------------------------------------------------------------

def _warning() -> None:
    st.warning(
        "**Read this first.** SEBI's study of individual intraday traders in India found that about 7 in 10 lost "
        "money, and beginners lose most often. This app's own tests show its forecast is only slightly better than "
        "guessing 'no change', which is not enough to beat trading costs on most days. **Do not trade money you "
        "need for rent, food, fees or loan payments.** The rules on this page are built to keep you out of bad "
        "trades and to keep losses small; they cannot make trading safe or promise any income.")


def _rules() -> safety.Rules:
    st.markdown("### 1. Your limits")
    st.caption("Set these once, before looking at any forecast. They decide how much you can lose, not how much "
               "you hope to make.")
    c = st.columns(5)
    capital = c[0].number_input("Trading capital (₹)", 1_000, 10_000_000, 50_000, 1_000, key="safe_capital",
                                help="Only money you could lose completely without it hurting your daily life.")
    risk = c[1].number_input("Max loss per trade (% of capital)", 0.1, 5.0, 1.0, 0.1, key="safe_risk",
                             help="Professionals usually risk 0.5% to 1% per trade. 1% of ₹50,000 is ₹500.")
    daily = c[2].number_input("Daily loss limit (% of capital)", 0.5, 10.0, 2.0, 0.5, key="safe_daily",
                              help="Once losses could reach this, stop for the day. No 'winning it back'.")
    stop = c[3].number_input("Stop-loss (% from entry)", 0.2, 5.0, 1.0, 0.1, key="safe_stop",
                             help="Where you exit if the price goes against you. Decide it before you enter.")
    max_trades = c[4].number_input("Max trades per day", 1, 3, 2, 1, key="safe_max",
                                   help="Capped at 3 here: every extra trade pays costs again.")
    rules = safety.Rules(capital=float(capital), risk_per_trade_pct=float(risk), daily_loss_limit_pct=float(daily),
                         stop_loss_pct=float(stop), max_trades_per_day=int(max_trades))
    for p in rules.problems():
        st.warning(p)
    st.caption(f"With these limits: at most **₹{rules.risk_per_trade:,.0f}** lost on one trade and **₹{rules.daily_loss_limit:,.0f}** "
               f"in a day. A run of 10 losing trades in a row would cost about ₹{rules.risk_per_trade * 10:,.0f} "
               f"({rules.risk_per_trade_pct * 10:.0f}% of your capital), which you could recover from.")
    return rules


def _check(preds: pd.DataFrame, rules: safety.Rules, cost_pct: float, record: dict, paper: dict) -> None:
    st.markdown("### 2. Should I trade today?")
    table = _confidence_table(_mtime(MODELS / "oos_predictions.csv"))
    trades_today = st.number_input("Real trades you have already made today", 0, 10, 0, 1, key="safe_done",
                                   help="Counts towards your daily maximum.")
    verdicts = []
    for _, r in preds.iterrows():
        conf = guidance.confidence_for(r["pred_gap_pct"], table)
        verdicts.append((r, conf, safety.should_i_trade(r["pred_gap_pct"], conf, cost_pct, record, paper,
                                                         int(trades_today), rules)))
    top = safety.day_answer([v for *_, v in verdicts])
    if top is None:
        st.info("No forecast to check yet.")
        return
    if top.tone == "stop":
        TONE["stop"](f"**{top.answer}.** {top.summary}")
    else:
        TONE[top.tone](f"**{top.answer}** for at least one stock. Check the cards below.")
    st.caption(f"A buy-and-sell costs about {cost_pct:.2f}% for an intraday trade. A trade is only considered when "
               f"the expected move is at least {safety.clear_edge_pct(cost_pct):.2f}% ({safety.CLEAR_EDGE_MULTIPLE:.0f} times that) and every other check passes.")
    cols = st.columns(len(verdicts) if len(verdicts) <= 3 else 3)
    for i, (r, conf, v) in enumerate(verdicts):
        with cols[i % len(cols)].container(border=True):
            mark = {"stop": "red", "paper": "orange", "go": "green"}[v.tone]
            st.markdown(f"**{r['symbol']}** :{mark}-background[{v.answer}]")
            for c in v.checks:
                st.markdown(f"{'✅' if c.ok else '❌'} {c.title}")
                st.caption(c.detail)
            if v.tone != "stop":
                size = safety.position_size(float(r["close"]), rules)
                st.markdown(f"Size if you trade: **{size.qty} shares** (about ₹{size.value:,.0f}), stop-loss "
                            f"{rules.stop_loss_pct:.1f}% away. Worst planned loss ₹{size.loss_at_stop:,.0f}.")


def _sizing(preds: pd.DataFrame, rules: safety.Rules) -> None:
    st.markdown("### 3. How many shares, at most")
    st.caption("Size comes from the loss you accept if the stop-loss is hit, charges included. "
               "Prices are yesterday's close; recheck at the open.")
    rows = []
    for _, r in preds.iterrows():
        s = safety.position_size(float(r["close"]), rules)
        rows.append({"Stock": r["symbol"], "Price (₹)": round(float(r["close"]), 2), "Max shares": s.qty,
                     "Order value (₹)": round(s.value), "Charges for the round trip (₹)": round(s.charges, 2),
                     "Loss if stop-loss hit (₹)": round(s.loss_at_stop),
                     "Limited by": {"risk": "your loss limit", "capital": f"{rules.max_position_pct:.0f}% of capital cap",
                                    "price": "one share costs more than your loss limit allows"}[s.limited_by]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("No leverage and no borrowed margin: losses with margin can be bigger than the money you put in.")


def _paper(preds: pd.DataFrame, rules: safety.Rules) -> dict:
    st.markdown("### 4. Practise with virtual money (paper trading)")
    st.caption(f"You start with ₹{safety.PAPER_START:,.0f} of virtual money. Orders fill at the next 09:15 open "
               "(never at a price you have already seen) and close at 15:30, or earlier at your stop-loss, using the "
               "day's real prices with full intraday charges. Practise until the record shows a profit after "
               f"charges over at least {safety.PAPER_TRADES_TO_GRADUATE} trades.")
    syms = sorted(preds["symbol"])
    journal = safety.settle(_journal(), {s: raw_prices(s) for s in syms})
    st.session_state[JOURNAL_KEY] = journal
    summ = safety.paper_summary(journal)
    c = st.columns(4)
    c[0].metric("Virtual balance", f"₹{summ['balance']:,.0f}", f"{_rs(summ['net'])} after charges",
                delta_color="normal" if summ["settled"] else "off")
    c[1].metric("Practice trades settled", f"{summ['settled']} of {safety.PAPER_TRADES_TO_GRADUATE}",
                f"{summ['waiting']} waiting for the open", delta_color="off")
    c[2].metric("Made money after charges", f"{summ['wins']} of {summ['settled']}" if summ["settled"] else "–")
    c[3].metric("Charges paid", f"₹{summ['charges']:,.0f}", help="What the same trades would have cost in real money.")
    if summ["ready"]:
        st.success("Your practice record is positive after charges. That is a good sign of discipline, not proof you "
                   "will make money with real funds. If you start, start with a small part of your capital.")

    trade_date = safety.paper_trade_date(_now())
    with st.form("paper_order", clear_on_submit=False, border=True):
        st.markdown(f"**New practice order** for the open on **{trade_date:%a %d %b %Y}**")
        f = st.columns(4)
        sym = f[0].selectbox("Stock", syms, key="paper_sym")
        side = f[1].selectbox("Direction", ["BUY", "SELL"], key="paper_side",
                              help="SELL means sell first and buy back before the close (intraday short).")
        price = float(preds.set_index("symbol").loc[sym, "close"])
        suggested = max(safety.position_size(price, rules).qty, 1)
        qty = f[2].number_input("Shares", 1, 100_000, suggested, 1, key="paper_qty",
                                help=f"The sizing table's maximum for {sym} is {suggested}.")
        f[3].markdown(f"Stop-loss {rules.stop_loss_pct:.1f}%  \nPrice ref ₹{price:,.2f}")
        if st.form_submit_button("Place practice order"):
            ok, why = safety.can_place(journal, rules, trade_date, sym, int(qty), price, summ["balance"])
            if ok:
                st.session_state[JOURNAL_KEY] = safety.new_trade(journal, _now(), trade_date, sym, side, int(qty),
                                                                 rules.stop_loss_pct, price)
                st.success(f"Practice order saved: {side} {qty} {sym} at the {trade_date:%d %b} open.")
                journal = st.session_state[JOURNAL_KEY]
            else:
                st.error(why)

    if len(journal):
        show = journal.drop(columns=["planned_risk"], errors="ignore").sort_values("id", ascending=False)
        st.dataframe(show, hide_index=True, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    buf = io.StringIO()
    journal.to_csv(buf, index=False)
    c1.download_button("Save my practice journal (CSV)", buf.getvalue(), "pulse8_paper_journal.csv", "text/csv",
                       help="Practice trades live only in this browser tab. Save them, and load the file next time.")
    up = c2.file_uploader("Load a saved journal", type="csv", key="paper_upload", label_visibility="collapsed")
    if up is not None and st.session_state.get("paper_loaded") != up.file_id:
        loaded = pd.read_csv(up)
        missing = [k for k in safety.JOURNAL_COLS if k not in loaded.columns and k != "planned_risk"]
        if missing:
            st.error(f"That file is not a Pulse8 practice journal (missing {', '.join(missing)}).")
        else:
            st.session_state[JOURNAL_KEY] = loaded
            st.session_state["paper_loaded"] = up.file_id
            st.rerun()
    if c3.button("Cancel orders still waiting", disabled=not summ["waiting"]):
        j = journal.copy()
        j.loc[j["status"] == "waiting", "status"] = "cancelled"
        st.session_state[JOURNAL_KEY] = j
        st.rerun()
    return summ


def _track(cost_pct: float, amount: float) -> dict:
    st.markdown("### 5. Honest track record: what following the app's signals would have done")
    st.caption("Each past forecast becomes one intraday trade: buy at the 09:15 open if the forecast was up (sell "
               "first if down) and exit at the 15:30 close, minus the full round-trip cost. The forecasts were made on "
               "days the model had not seen in training. The overnight gap itself cannot be captured, so it is not counted.")
    need = safety.clear_edge_pct(cost_pct)
    stamp = (_mtime(MODELS / "oos_predictions.csv"), _raw_stamp())
    every = _signal_trades("oos", cost_pct, 0.0, stamp)
    clear = _signal_trades("oos", cost_pct, need, stamp)
    live = _signal_trades("log", cost_pct, 0.0, (_mtime(PROCESSED / "predictions_log.csv"), _raw_stamp()))
    rows = []
    for name, t in [("Every day the model leaned up or down", every), (f"Only 'clear edge' days (move ≥ {need:.2f}%)", clear),
                    ("Live: forecasts this app actually logged", live)]:
        s = safety.track_summary(t, amount)
        if not s["trades"]:
            rows.append({"Rule": name, "Trades": 0, "Made money after charges": "Building up: needs logged "
                         "forecasts whose day has closed"})
            continue
        rows.append({"Rule": name, "Trades": s["trades"], "Made money after charges": f"{s['win_rate_pct']:.0f}%",
                     "Average per trade before costs": f"{s['avg_gross_pct']:+.2f}%",
                     "Average per trade after costs": f"{s['avg_net_pct']:+.2f}%",
                     f"Total on ₹{amount:,.0f} a trade": _rs(s["total_rupees"]),
                     "Worst losing run": _rs(s["worst_run"], sign=False),
                     "Period": f"{s['first']:%b %Y} to {s['last']:%b %Y}"})
    st.dataframe(pd.DataFrame(rows).fillna(""), hide_index=True, use_container_width=True)
    s_every = safety.track_summary(every, amount)
    if s_every["trades"]:
        verdict = ("lost money" if s_every["total_rupees"] < 0 else "made money")
        st.markdown(f"Following every signal **{verdict}** after charges: {_rs(s_every['total_rupees'])} over "
                    f"{s_every['trades']:,} trades of ₹{amount:,.0f}. Costs alone took about "
                    f"₹{cost_pct / 100 * amount * s_every['trades']:,.0f}.")
        chart_df = every.assign(rupees=every["net_pct"] / 100 * amount).groupby("trade_date", as_index=False)["rupees"].sum()
        chart_df["Running total (₹)"] = chart_df["rupees"].cumsum()
        line = alt.Chart(chart_df).mark_line(color="#2a78d6").encode(
            x=alt.X("trade_date:T", title=None), y=alt.Y("Running total (₹):Q"),
            tooltip=[alt.Tooltip("trade_date:T", title="Day"), alt.Tooltip("Running total (₹):Q", format=",.0f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart((line + zero).properties(height=260, title="Every signal, after charges"),
                        use_container_width=True)
    else:
        st.info("The track record appears once the model has been evaluated (Model accuracy page).")
    return safety.track_summary(clear, amount)


def render(preds: pd.DataFrame) -> None:
    _warning()
    rules = _rules()
    cost_pct = guidance.round_trip_cost_pct("equity_intraday")
    # Sections 4 and 5 feed the check in section 2, so compute them first and draw them below it.
    top = st.container()
    st.divider()
    sizing = st.container()
    st.divider()
    paper_box = st.container()
    st.divider()
    track_box = st.container()
    with track_box:
        record = _track(cost_pct, amount=10_000.0)
    with paper_box:
        paper = _paper(preds, rules)
    with top:
        _check(preds, rules, cost_pct, record, paper)
    with sizing:
        _sizing(preds, rules)
    st.caption("Decision support for a student project, not investment advice. No rule or tool can guarantee profit.")
