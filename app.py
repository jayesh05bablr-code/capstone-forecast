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
from capstone.ui import (assumptions, common, headline, summary, tab_accuracy, tab_ask, tab_backtest, tab_company,
                         tab_download, tab_forecast, tab_insights, tab_invest, tab_live, tab_montecarlo, tab_rupee,
                         tab_scalping, tab_sectors, tab_uploads, tab_weekly)

st.set_page_config(page_title="India Opening Forecast", layout="wide")
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
headline.render(preds, story, moves)
summary.render(compact=True)
cfg, chosen = assumptions.render(sorted(preds["symbol"]))

(t_forecast, t_accuracy, t_weekly, t_backtest, t_mc, t_insights, t_ask, t_more) = st.tabs([
    "Morning forecast", "Model accuracy", "Weekly analysis", "Backtesting", "Monte Carlo", "Business insights",
    "Ask the market", "More tools"])

with t_forecast:
    symbol = tab_forecast.render(preds, story)
with t_accuracy:
    tab_accuracy.render(symbol)  # follows the stock picked on the Morning forecast tab
with t_weekly:
    tab_weekly.render(cfg, chosen)
with t_backtest:
    bt = tab_backtest.render(cfg, chosen)
with t_mc:
    tab_montecarlo.render(cfg, bt)
with t_insights:
    tab_insights.render(preds, story)
with t_ask:
    tab_ask.render()
with t_more:
    # Only the chosen tool runs (st.tabs would run all eight, and the Live view would keep
    # refreshing in the background for every open browser, which uses a lot of CPU).
    tools = {"US sectors and India": lambda: tab_sectors.render(preds, moves), "Rupee": tab_rupee.render,
             "Live": lambda: tab_live.render(preds), "Invest planner": lambda: tab_invest.render(preds),
             "Intraday scalping": lambda: tab_scalping.render(preds),
             "Company profile": lambda: tab_company.render(preds),
             "Upload company data": lambda: tab_uploads.render(preds, story), "Download data": tab_download.render}
    pick = st.radio("Tool", list(tools), horizontal=True, key="more_tool", label_visibility="collapsed")
    tools[pick]()
