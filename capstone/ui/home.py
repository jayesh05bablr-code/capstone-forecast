"""Today page: what a first-time visitor needs at a glance, in plain language.

Reads only files that already exist (predictions, the out-of-sample table, the prediction log), with
every computation cached until those files change, so this page stays light.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import guidance, summary
from capstone.paths import MODELS, PROCESSED, RAW
from capstone.ui import data_cache, glossary
from capstone.ui.common import raw_prices

TONE_COLOR = {"up": "blue", "down": "red", "caution": "orange", "neutral": "gray"}
CONF_COLOR = {"High": "green", "Medium": "blue", "Low": "orange", "Unknown": "gray"}
COST_TYPES = {"Same-day trade (intraday)": "equity_intraday", "Hold overnight or longer (delivery)": "equity_delivery"}


def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False, max_entries=2)
def _confidence_table(stamp: float) -> pd.DataFrame:
    return guidance.confidence_table(data_cache.oos())


@st.cache_data(show_spinner=False, max_entries=2)
def _live_record(stamp: tuple) -> dict:
    log_path = PROCESSED / "predictions_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() else None
    syms = [] if log is None else sorted(log["symbol"].unique())
    return guidance.live_track_record(log, {s: raw_prices(s) for s in syms})


@st.cache_data(show_spinner=False, max_entries=2)
def _measured(stamp: tuple) -> dict:
    return summary.measured()


def _stamp_models() -> tuple:
    return tuple(sorted((p.name, p.stat().st_mtime) for p in MODELS.glob("*.*"))) if MODELS.exists() else ()


def _welcome() -> None:
    st.info("**How to read this page:** 1) the market outlook, 2) each stock card (expected open, likely range, "
            "how much to trust it, what a careful investor might do), 3) *How good is this model* at the bottom. "
            "It is decision support for planning around the 09:15 open, not a buy or sell signal and not a "
            "promise of profit.")
    with st.expander("New here? What this app does and cannot do", expanded=False):
        st.markdown(
            "**What it does.** Every morning it estimates where five large NSE stocks (Reliance, TCS, Infosys, "
            "HDFC Bank, ICICI Bank) will **open** at 09:15, using yesterday's prices, what US markets did "
            "overnight and the rupee. A trained machine-learning model (gradient-boosted trees) makes the "
            "forecast; it is retrained every week and tested only on data it never saw.\n\n"
            "**What it cannot do.** It cannot make you money by itself. The gap from yesterday's close to today's "
            "open happens overnight, and nobody can buy at yesterday's close after seeing last night's news. So "
            "use the forecast to plan around the open (a sensible limit price, not chasing a jump, not panic-"
            "selling a dip). Opening gaps are mostly noise: even a good model is right about the direction only "
            "a little more often than a coin toss on small moves, and a trade has to beat its costs too.\n\n"
            "**New to the words?** See the glossary at the bottom of this page.")


def _outlook(preds: pd.DataFrame, story: dict, moves: pd.DataFrame) -> None:
    title, meaning = guidance.market_mood(preds["pred_gap_pct"].mean())
    st.markdown(f"## {title}")
    st.markdown(meaning)
    c = st.columns(4)
    c[0].metric("Average expected gap", f"{preds['pred_gap_pct'].mean():+.2f}%",
                help="Average of the five stocks' expected opening gaps versus yesterday's close.")
    if "SPY" in moves.index:
        c[1].metric("US market overnight (S&P 500)", _pct(moves.loc["SPY", "ret"]),
                    help="The US market trades while India sleeps. Its move is the biggest overnight clue.")
    if "QQQ" in moves.index:
        c[2].metric("US tech overnight (Nasdaq 100)", _pct(moves.loc["QQQ", "ret"]),
                    help="Matters most for TCS and Infosys.")
    if story.get("usdinr"):
        chg = story.get("usdinr_chg")
        word = "" if chg is None or pd.isna(chg) else (" (rupee weaker)" if chg > 0 else " (rupee stronger)")
        c[3].metric("Rupee per US dollar", f"₹{story['usdinr']:.2f}", help="A weaker rupee tends to help exporters "
                    "such as TCS and Infosys." + word)
    if story.get("lines"):
        with st.expander("Why the model expects this"):
            st.markdown("\n".join(f"- {line}" for line in story["lines"]))


def _pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:+.2f}%"


def _cards(preds: pd.DataFrame, table: pd.DataFrame, cost_pct: float, amount: float) -> None:
    order = preds.assign(size=preds["pred_gap_pct"].abs()).sort_values("size", ascending=False)
    cols = st.columns(3)
    for i, (_, r) in enumerate(order.iterrows()):
        conf = guidance.confidence_for(r["pred_gap_pct"], table)
        adv = guidance.advice(r["symbol"], r["pred_gap_pct"], r["pred_open"], r["pred_open_low"],
                              r["pred_open_high"], conf, cost_pct)
        lo, hi = guidance.risk_on(amount, r["pred_open_low"], r["pred_open_high"], r["close"])
        with cols[i % 3].container(border=True):
            st.markdown(f"### {r['symbol']}")
            st.markdown(f":{TONE_COLOR[adv.tone]}-background[{adv.verdict}] "
                        f":{CONF_COLOR[conf.level]}[Confidence: {conf.level}]")
            st.metric("Expected open", f"₹{r['pred_open']:,.2f}", f"{r['pred_gap_pct']:+.2f}% vs close "
                      f"₹{r['close']:,.2f}", delta_color="normal" if abs(r["pred_gap_pct"]) >= 0.05 else "off")
            st.caption(f"Likely range ₹{r['pred_open_low']:,.2f} to ₹{r['pred_open_high']:,.2f} (8 in 10 opens). "
                       f"On ₹{amount:,.0f} held overnight: {_rs(lo)} to {_rs(hi)} at the open.")
            st.markdown(adv.action)
            st.caption(conf.sentence)


def _rs(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}₹{abs(x):,.0f}"


def _report_card(cost_pct: float) -> None:
    st.markdown("### How good is this model? (read before acting)")
    m = _measured(_stamp_models())
    rec = _live_record((_mtime(PROCESSED / "predictions_log.csv"),
                        tuple(sorted((p.name, p.stat().st_mtime) for p in RAW.glob("*.csv")))))
    c = st.columns(3)
    d = m.get("daily", {})
    hgb, naive = d.get("forecast") or d.get("hist_gradient_boosting"), d.get("naive")
    if hgb and naive:
        better = (1 - hgb["mae_pct"] / naive["mae_pct"]) * 100
        c[0].metric("Direction right (past tests)", f"{hgb['direction_acc_pct']:.0f}%",
                    help="Share of past test days where the model called up or down correctly, on data it had "
                         "not seen. 50% is a coin toss.")
        vs = (f"{better:.0f}% better than guessing 'no gap'" if better >= 0.5 else
              f"-{abs(better):.0f}%: no better than guessing 'no gap'")
        c[1].metric("Average miss", f"{hgb['mae_pct']:.2f}%", vs,
                    help="Average size of the error in the opening gap, compared with the simple guess that today "
                         "opens at yesterday's close.")
    if rec.get("cases"):
        c[2].metric("Live record (app's own forecasts)", f"{rec['hits']} of {rec['cases']} right",
                    f"last {rec['days']} trading days", delta_color="off",
                    help="Forecasts this app actually logged before each open, checked against the real open.")
    else:
        c[2].metric("Live record (app's own forecasts)", "Building up",
                    help="Appears once the app has logged forecasts and the following opens are known.")
    bt = m.get("backtest", {})
    lines = [f"A buy-and-sell costs about **{cost_pct:.2f}%** with typical discount-broker charges (sidebar pages "
             "let you change them). Most expected gaps are smaller than that."]
    b = next((v for k, v in bt.items() if "ost-aware" in k), None)
    if b is not None and "cumulative_return_pct" in b:
        lines.append(f"Trading the weekly signal with costs in a backtest returned **{b['cumulative_return_pct']:+.1f}%** "
                     "over the test period (see Backtesting).")
    mc = m.get("mc")
    if mc and "prob_loss_pct" in mc:
        lines.append(f"In {mc.get('n_sims', 5000):,} simulated futures of that strategy, **{mc['prob_loss_pct']:.0f}%** "
                     "ended with a loss (see Monte Carlo).")
    lines.append("Bottom line: use this as one input for planning around the open, never as a promise of profit. "
                 "This is a student project, not investment advice.")
    st.markdown("\n".join(f"- {x}" for x in lines))


def render(preds: pd.DataFrame, story: dict, moves: pd.DataFrame) -> None:
    status, msg = guidance.freshness(story["next_open"], bool(story.get("overnight_complete")),
                                     pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).to_pydatetime())
    {"ok": st.success, "preliminary": st.warning, "old": st.error}[status](msg)
    _welcome()
    _outlook(preds, story, moves)

    st.markdown("### Stock by stock")
    c1, c2 = st.columns(2)
    kind = c1.selectbox("How would you trade?", list(COST_TYPES), key="home_cost_kind",
                        help="Sets the trading cost the expected move is compared with.")
    amount = c2.number_input("Amount to think about (₹)", 1000, 10_000_000, 10_000, 1000, key="home_amount",
                             help="Only used to turn % into rupees on the cards.")
    cost_pct = guidance.round_trip_cost_pct(COST_TYPES[kind])
    table = _confidence_table(_mtime(MODELS / "oos_predictions.csv"))
    _cards(preds, table, cost_pct, amount)
    _report_card(cost_pct)
    glossary.render(compact=True)
