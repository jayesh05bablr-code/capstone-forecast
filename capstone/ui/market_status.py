"""One line at the top of every page saying whether NSE trades today, in plain words.

Also re-makes a saved forecast that aims at a day NSE is closed (for example one made on the morning of
a holiday before the app knew the holiday list), so no page shows a forecast "for today" on a holiday.
"""
from __future__ import annotations

import threading
import time

import streamlit as st

from capstone import market_calendar

RETRY_SECONDS = 30 * 60
_last_try = {"t": 0.0}


def banner() -> market_calendar.Status:
    s = market_calendar.status()
    if s.state in ("holiday", "weekend"):
        st.info(f"**{s.headline}.** {s.detail}", icon="🗓️")
    if not market_calendar.covered(s.next_session):
        st.caption(f"NSE has not published its {s.next_session.year} holiday list yet (it usually does in "
                   "December), so for that year only weekends are known to be closed.")
    return s


def redo_forecast_if_wrong(story: dict | None, autorun) -> bool:
    """Start a fresh forecast in the background if the saved one targets a closed day. True if started."""
    if not market_calendar.forecast_target_wrong(story):
        return False
    if autorun.status()["running"] or time.time() - _last_try["t"] < RETRY_SECONDS:
        return False
    _last_try["t"] = time.time()
    threading.Thread(target=autorun.update_now, daemon=True, name="holiday-redo").start()
    return True
