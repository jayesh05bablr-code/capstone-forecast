"""Dashboard: streamlit run app.py

This file only loads the forecast and lays out the page. Each tab lives in its own
module under capstone/ui/, and the calculations they show live in the capstone package.
"""
import json
import time

import pandas as pd
import streamlit as st

from capstone import autorun, env, refresh
from capstone.paths import PROCESSED, SYNTHETIC_FLAG
from capstone.ui import (assumptions, common, glossary, headline, home, tab_accuracy, tab_ask, tab_backtest,
                         tab_company, tab_download, tab_flow, tab_forecast, tab_insights, tab_invest, tab_live, tab_montecarlo,
                         tab_rupee, tab_scalping, tab_sectors, tab_uploads, tab_weekly)

st.set_page_config(page_title="India Opening Forecast", layout="wide", initial_sidebar_state="expanded")
env.load_streamlit_secrets()  # keys from the Secrets box when hosted on Streamlit Community Cloud
st.title("India market opening forecast")

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

nav_box = st.sidebar.container()  # page menu first, so it is the first thing in the sidebar
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
if "reason" not in preds.columns or story is None:
    st.info("These predictions come from the older model. Run fetch_overnight, train and predict "
            "to get overnight US and rupee reasons.")
    st.dataframe(preds, use_container_width=True, hide_index=True)
    st.stop()

moves = pd.DataFrame(story["moves"]).set_index("ticker")

# Only the chosen page runs. st.tabs would run every tab's code (backtests, simulations, charts) on
# every click, for every visitor, which is what made the app slow and CPU-throttled.
PAGES = {
    "Today (start here)": "Plain-language outlook, confidence and what to do",
    "Stock details": "Forecast, drivers and price chart for one stock",
    "Model accuracy": "How the model did on data it never saw",
    "Live market": "Index and stock prices during market hours",
    "Market flow": "Heat map, sector flow and stock volume flow",
    "Weekly analysis": "Weekly moves versus trading costs",
    "Backtesting": "What trading the weekly signal would have done",
    "Monte Carlo": "Range of possible outcomes of that strategy",
    "Business insights": "Executive summary, fundamentals and limitations",
    "Ask the market": "Ask questions in plain English",
    "More tools": "Sectors, rupee, planner, scalping, company data, downloads",
    "Glossary": "Market words explained",
}
with nav_box:
    page = st.radio("Go to", list(PAGES), key="page", captions=list(PAGES.values()))
    st.divider()
st.sidebar.caption("Decision support for a student project, not investment advice. No tool can guarantee profit.")

# One plain-language line at the top of each analysis page, for readers new to markets.
PLAIN = {
    "Stock details": "The forecast for one stock, what pushed it up or down, and its price history.",
    "Model accuracy": "Every forecast here was made on days the model had not seen. If the trained model's "
                      "average miss is not clearly below the naive guess (open = yesterday's close), it has no edge.",
    "Market flow": "Which sectors and stocks are drawing today's trading, and whether buyers or sellers had "
                   "the upper hand. Box and band sizes are real traded money; the buying/selling split is an estimate.",
    "Weekly analysis": "Would buying on Monday's open and selling on Friday's close have beaten the trading costs? "
                       "Bars below the cost line lost money after charges.",
    "Backtesting": "A replay of trading the weekly forecast with every charge paid. A negative result means the "
                   "rule lost money in the past; it is shown so you are not tempted to trade it blindly.",
    "Monte Carlo": "Thousands of reshuffled versions of that backtest. 'Probability of capital loss' is the share of "
                   "simulated futures that ended below the starting money.",
}
if page in PLAIN:
    st.info(f"**In plain words:** {PLAIN[page]}")

symbol = st.session_state.get("forecast_symbol", preds["symbol"].iloc[0])
if page == "Today (start here)":
    home.render(preds, story, moves)
elif page == "Stock details":
    headline.render(preds, story, moves)
    tab_forecast.render(preds, story)
elif page == "Model accuracy":
    tab_accuracy.render(symbol)  # follows the stock picked on Stock details
elif page == "Live market":
    tab_live.render(preds)
elif page == "Market flow":
    tab_flow.render()
elif page in ("Weekly analysis", "Backtesting", "Monte Carlo"):
    cfg, chosen = assumptions.render(sorted(preds["symbol"]))
    if page == "Weekly analysis":
        tab_weekly.render(cfg, chosen)
    elif page == "Backtesting":
        tab_backtest.render(cfg, chosen)
    else:
        tab_montecarlo.render(cfg, tab_backtest.result(cfg))
elif page == "Business insights":
    tab_insights.render(preds, story)
elif page == "Ask the market":
    tab_ask.render()
elif page == "Glossary":
    glossary.render()
else:
    # Only the chosen tool runs (the Live view would keep refreshing in the background otherwise).
    tools = {"US sectors and India": lambda: tab_sectors.render(preds, moves), "Rupee": tab_rupee.render,
             "Invest planner": lambda: tab_invest.render(preds),
             "Intraday scalping": lambda: tab_scalping.render(preds),
             "Company profile": lambda: tab_company.render(preds),
             "Upload company data": lambda: tab_uploads.render(preds, story), "Download data": tab_download.render}
    pick = st.radio("Tool", list(tools), horizontal=True, key="more_tool", label_visibility="collapsed")
    tools[pick]()
