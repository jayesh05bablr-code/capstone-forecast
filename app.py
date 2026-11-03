"""Dashboard: streamlit run app.py

This file only loads the forecast and lays out the page. Each tab lives in its own
module under capstone/ui/, and the calculations they show live in the capstone package.
"""
import json

import pandas as pd
import streamlit as st

from capstone import autorun, env, refresh
from capstone.paths import PROCESSED, SYNTHETIC_FLAG
from capstone.ui import (common, headline, tab_company, tab_download, tab_invest, tab_live, tab_model,
                         tab_rupee, tab_scalping, tab_sectors, tab_stocks, tab_uploads)

st.set_page_config(page_title="India Opening Forecast", layout="wide")
env.load_streamlit_secrets()  # keys from the Secrets box when hosted on Streamlit Community Cloud
st.title("India market opening forecast")

# Keep the forecast current without a terminal: fetch, (re)train and predict when an update is due.
pred_path = PROCESSED / "predictions.csv"
if not pred_path.exists() and not SYNTHETIC_FLAG.exists():
    with st.status("First start: downloading prices, training the model and making the forecast. "
                   "This takes a few minutes, once.", expanded=True) as box:
        autorun.ensure_fresh(block=True)
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

(t_live, t_stocks, t_sectors, t_rupee, t_invest, t_scalp, t_company, t_uploads, t_download,
 t_model) = st.tabs(["Live", "Stocks", "US sectors and India", "Rupee", "Invest planner", "Intraday scalping",
                     "Company profile", "Upload company data", "Download data", "Model accuracy"])

with t_stocks:
    symbol = tab_stocks.render(preds, story)
with t_sectors:
    tab_sectors.render(preds, moves)
with t_rupee:
    tab_rupee.render()
with t_model:
    tab_model.render(symbol)  # follows the stock picked on the Stocks tab
with t_invest:
    tab_invest.render(preds)
with t_uploads:
    tab_uploads.render(preds, story)
with t_download:
    tab_download.render()
with t_scalp:
    tab_scalping.render(preds)
with t_company:
    tab_company.render(preds)
with t_live:  # drawn last so it reads the settings chosen on the Intraday scalping tab
    tab_live.render(preds)
