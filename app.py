"""Dashboard: streamlit run app.py

This file only loads the forecast and lays out the page. Each tab lives in its own
module under capstone/ui/, and the calculations they show live in the capstone package.
"""
import json
import time

import pandas as pd
import streamlit as st

import codeversion

codeversion.reload_if_changed()  # pick up a new deploy's code without rebooting the app (see codeversion.py)

from capstone import autorun, env, refresh  # noqa: E402
from capstone.paths import PROCESSED, SYNTHETIC_FLAG
from capstone.ui import (assumptions, common, glossary, headline, home, how_it_works, market_status, tab_accuracy,
                         tab_ask, tab_backtest, tab_company, tab_download, tab_flow, tab_forecast, tab_insights, tab_invest, tab_live, tab_montecarlo,
                         tab_rupee, tab_scalping, tab_sectors, tab_uploads, tab_weekly)

st.set_page_config(page_title="Pulse8 · India opening forecast", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
env.load_streamlit_secrets()  # keys from the Secrets box when hosted on Streamlit Community Cloud
st.title("Pulse8")
st.caption("The market's pulse before the bell: India market opening forecast for the 09:15 IST open.")
market_status.banner()  # "Market closed today: <holiday>" on NSE holidays and weekends

# Keep the forecast current without a terminal: fetch, (re)train and predict when an update is due.
pred_path = PROCESSED / "predictions.csv"
if not pred_path.exists() and not SYNTHETIC_FLAG.exists():
    with st.status("First start: downloading prices, training the model and making the forecast. "
                   "This takes a few minutes, once.", expanded=True) as box:
        autorun.ensure_fresh(block=True)
        while autorun.status()["running"]:  # another visitor started it first: wait for theirs
            time.sleep(3)
        info = autorun.status()
        box.update(label="Done" if info.get("ok") else "First update failed", state="complete" if info.get("ok") else "error")
        if not info.get("ok"):
            st.code(info.get("log", "")[-4000:])
            st.stop()
    st.cache_data.clear()
elif not SYNTHETIC_FLAG.exists():
    autorun.ensure_fresh()

with st.sidebar:
    info = autorun.status()
    st.markdown("**Automatic updates**")
    if info["running"]:
        st.info(f"Updating now ({info['step']}). New numbers appear when you reload.")
    elif info.get("finished"):
        (st.success if info.get("ok") else st.error)(
            f"Last update {info['finished']} IST: {'OK' if info.get('ok') else 'failed'}")
    st.caption("Updates by itself at 08:45 IST (after the US close) and 16:15 IST (after NSE closes) "
               "whenever someone opens the page.")
    if st.button("Update now", disabled=info["running"]):
        with st.spinner("Fetching, training if due, and predicting..."):
            autorun.update_now()
        st.cache_data.clear()
        st.rerun()
    if info.get("log"):
        with st.expander("Last update log"):
            st.code(info["log"][-4000:])

# Data saved since the last training (e.g. an upload whose retrain was interrupted): update now.
if refresh.stale() and not st.session_state.get("auto_tried"):
    st.session_state["auto_tried"] = True  # at most once per browser session if training keeps failing
    common.auto_update("Company data changed since the model was trained")

headline.show_update_messages()

story_path = PROCESSED / "overnight_story.json"
if not pred_path.exists():
    st.warning("No predictions yet. Run fetch_data, fetch_overnight, train and predict first.")
    st.stop()
if SYNTHETIC_FLAG.exists():
    st.error("SAMPLE DATA: these prices, US moves and rupee rates are synthetic, made to test the "
             "pipeline. They are not market data. Delete data/raw, data/us, data/fx and "
             "data/SYNTHETIC_DATA, then fetch real data.")

preds = pd.read_csv(pred_path)
story = json.loads(story_path.read_text()) if story_path.exists() else None
if not SYNTHETIC_FLAG.exists() and market_status.redo_forecast_if_wrong(story, autorun):
    st.info("The saved forecast was for a day NSE is closed. A new one for the next trading day is being made "
            "now; reload the page in a few minutes.")
if "reason" not in preds.columns or story is None:
    st.info("These predictions come from the older model. Run fetch_overnight, train and predict "
            "to get overnight US and rupee reasons.")
    st.dataframe(preds, use_container_width=True, hide_index=True)
    st.stop()

moves = pd.DataFrame(story["moves"]).set_index("ticker")

# Only the chosen page runs. st.tabs would run every tab's code (backtests, simulations, charts) on
# every click, for every visitor, which is what made the app slow and CPU-throttled.
# One plain-language line at the top of each analysis page, for readers new to markets.
PLAIN = {
    "Stock details": "The forecast for one stock, what pushed it up or down, and its price history.",
    "Model accuracy": "Every forecast here was made on days the model had not seen. If the trained model's "
                      "average miss is not clearly below the naive guess (open = yesterday's close), it has no edge.",
    "Market flow": "Which sectors and stocks are drawing today's trading, and whether buyers or sellers had "
                   "the upper hand. Box and band sizes are real traded money; the buying/selling split is an estimate.",
    "Sector intelligence": "Which sectors are beating the NIFTY 50 and which are gaining or losing that edge, from "
                           "NSE's official sector indices.",
    "Institutional activity": "What foreign (FII/FPI) and domestic (DII) institutions bought and sold, the day's large "
                              "deals and how much trading ended in delivery, as reported by NSE.",
    "Breadth and risk": "Whether a market move is broad (most stocks join in) or narrow (a few big stocks), and how "
                        "liquid and volatile each stock has been.",
    "Weekly analysis": "Would buying on Monday's open and selling on Friday's close have beaten the trading costs? "
                       "Bars below the cost line lost money after charges.",
    "Backtesting": "A replay of trading the weekly forecast with every charge paid. A negative result means the "
                   "rule lost money in the past; it is shown so you are not tempted to trade it blindly.",
    "Monte Carlo": "Thousands of reshuffled versions of that backtest. 'Probability of capital loss' is the share of "
                   "simulated futures that ended below the starting money.",
}
symbol = st.session_state.get("forecast_symbol", preds["symbol"].iloc[0])


def _strategy(which: str) -> None:
    cfg, chosen = assumptions.render(sorted(preds["symbol"]))
    if which == "Weekly analysis":
        tab_weekly.render(cfg, chosen)
    elif which == "Backtesting":
        tab_backtest.render(cfg, chosen)
    else:
        tab_montecarlo.render(cfg, tab_backtest.result(cfg))


def _lazy(module: str):
    def run():
        import importlib
        importlib.import_module(f"capstone.ui.{module}").render()
    return run


def _stock_details() -> None:
    headline.render(preds, story, moves)
    tab_forecast.render(preds, story)


# key (used by links such as the stock panel's "nav_to") -> (menu section, menu title, icon, url, page body)
ROUTES = {
    "Today (start here)": ("Start", "Today (start here)", ":material/today:", "today", lambda: home.render(preds, story, moves)),
    "Market overview": ("Start", "Market overview · morning brief", ":material/dashboard:", "overview",
                        lambda: __import__("capstone.ui.tab_overview", fromlist=["render"]).render(preds, story, moves)),
    "How it works": ("Start", "How it works", ":material/schema:", "how-it-works", how_it_works.render),
    "Live market": ("Markets", "Live markets", ":material/monitoring:", "live", lambda: tab_live.render(preds)),
    "Market flow": ("Markets", "Market flow intelligence", ":material/account_tree:", "flow", tab_flow.render),
    "Sector intelligence": ("Markets", "Sector intelligence", ":material/donut_large:", "sectors", _lazy("tab_sector_intel")),
    "Institutional activity": ("Markets", "Institutional activity", ":material/account_balance:", "institutional",
                               _lazy("tab_institutional")),
    "Stock details": ("Forecasting", "Stock forecasting", ":material/insights:", "forecast", _stock_details),
    "Model accuracy": ("Forecasting", "Model accuracy", ":material/rule:", "accuracy", lambda: tab_accuracy.render(symbol)),
    "Weekly analysis": ("Forecasting", "Weekly analysis", ":material/date_range:", "weekly", lambda: _strategy("Weekly analysis")),
    "Backtesting": ("Forecasting", "Backtesting", ":material/history:", "backtest", lambda: _strategy("Backtesting")),
    "Monte Carlo": ("Forecasting", "Monte Carlo", ":material/casino:", "monte-carlo", lambda: _strategy("Monte Carlo")),
    "Company profile": ("Equity research", "Company profile", ":material/business:", "company", lambda: tab_company.render(preds)),
    "Business insights": ("Equity research", "Business insights", ":material/lightbulb:", "insights",
                          lambda: tab_insights.render(preds, story)),
    "US sectors and India": ("Equity research", "US sectors and India", ":material/public:", "us-sectors",
                             lambda: tab_sectors.render(preds, moves)),
    "Rupee": ("Equity research", "Rupee", ":material/currency_rupee:", "rupee", tab_rupee.render),
    "Breadth and risk": ("Portfolio and risk", "Breadth, liquidity and risk", ":material/stacked_bar_chart:", "risk",
                         _lazy("tab_breadth")),
    "Invest planner": ("Portfolio and risk", "Invest planner", ":material/savings:", "planner", lambda: tab_invest.render(preds)),
    "Intraday scalping": ("Portfolio and risk", "Intraday scalping", ":material/speed:", "scalping",
                          lambda: tab_scalping.render(preds)),
    "Ask the market": ("Assistant and data", "AI market assistant", ":material/forum:", "ask", tab_ask.render),
    "Upload company data": ("Assistant and data", "Upload company data", ":material/upload_file:", "upload",
                            lambda: tab_uploads.render(preds, story)),
    "Download data": ("Assistant and data", "Download data", ":material/download:", "download", tab_download.render),
    "Glossary": ("Assistant and data", "Glossary", ":material/menu_book:", "glossary", glossary.render),
}


def _page(key: str):
    _, title, icon, url, body = ROUTES[key]

    def run():
        if key in PLAIN:
            st.info(f"**In plain words:** {PLAIN[key]}")
        body()
    run.__name__ = url.replace("-", "_")
    return st.Page(run, title=title, icon=icon, url_path=url, default=(key == "Today (start here)"))


PAGE_OBJS = {k: _page(k) for k in ROUTES}
sections: dict[str, list] = {}
for k, (sec, *_rest) in ROUTES.items():
    sections.setdefault(sec, []).append(PAGE_OBJS[k])
nav = st.navigation(sections, expanded=True)
st.session_state["_pulse8_pages"] = PAGE_OBJS  # for links from inside pages (stock panel -> forecast)
if st.session_state.get("nav_to") in PAGE_OBJS:  # a link from another page (e.g. the stock panel)
    st.switch_page(PAGE_OBJS[st.session_state.pop("nav_to")])
with st.sidebar:
    st.caption("Theme: dark by default. Switch to light under ⋮ (top right) → Settings.")
st.sidebar.caption("Decision support for a student project, not investment advice. No tool can guarantee profit.")
nav.run()
