"""Ask the market tab: chat with NSE's official data through a free hosted language model."""
from __future__ import annotations

import streamlit as st

from capstone import ask_market

EXAMPLES = ["What is RELIANCE trading at right now?", "Top 5 gainers today?",
            "How has TCS done against the Nifty IT index over 6 months?",
            "What does the model forecast for INFY's open, and what is it based on?"]


def _setup_help() -> None:
    g, m = ask_market.PROVIDERS["groq"], ask_market.PROVIDERS["gemini"]
    st.info("To turn this chat on, add a free key for one language model:\n\n"
            f"1. Get a free key from **Groq** ({g['signup']}) or **Google Gemini** ({m['signup']}).\n"
            "2. On share.streamlit.io open this app's **⋮ > Settings > Secrets** (on your laptop: the "
            "`.env` file) and add one line, for example `GROQ_API_KEY = \"gsk_...\"` or "
            "`GEMINI_API_KEY = \"AIza...\"`.\n"
            "3. Save. The app restarts and this chat works.")


def render() -> None:
    st.markdown("### Ask the market")
    p = ask_market.provider()
    st.caption("Answers come from NSE's official data (live quotes and daily bhavcopy via NSE's public "
               "MCP servers) and this app's own forecast. The language model only fetches and explains; "
               "it does not forecast. Educational, not investment advice.")
    if p is None:
        _setup_help()
        return
    history = st.session_state.setdefault("ask_history", [])
    top = st.columns([3, 1])
    top[0].caption(f"Model: {p['name']} ({p['model']})")
    if top[1].button("Clear chat", disabled=not history):
        history.clear()
    box = st.container(height=520)
    with box:
        if not history:
            st.markdown("Try: " + " · ".join(f"*{e}*" for e in EXAMPLES))
        for turn in history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn.get("used"):
                    st.caption("NSE data used: " + ", ".join(u["tool"] for u in turn["used"]))
    question = st.chat_input("Ask about NSE stocks, indices or the forecast", key="ask_input")
    if not question:
        return
    with box:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Fetching NSE data..."):
                try:
                    answer, used = ask_market.ask(
                        question, [{"role": t["role"], "content": t["content"]} for t in history])
                except (ask_market.AskError, Exception) as exc:  # show the reason, never a stack trace
                    st.error(str(exc) or "Something went wrong; try again.")
                    return
            st.markdown(answer)
            if used:
                st.caption("NSE data used: " + ", ".join(u["tool"] for u in used))
    history += [{"role": "user", "content": question},
                {"role": "assistant", "content": answer, "used": used}]
