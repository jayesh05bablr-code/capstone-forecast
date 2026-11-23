"""Plain-language glossary, shown on the Today page and as its own page."""
from __future__ import annotations

import streamlit as st

TERMS = {
    "Close": "The last traded price of the day at 15:30 IST. The forecast starts from yesterday's close.",
    "Open": "The first price of the day at 09:15 IST, set in the 09:00-09:08 pre-open auction.",
    "Gap": "How far today's open is from yesterday's close, in %. A gap up means the stock opens higher; a gap "
           "down means it opens lower. Most gaps in these large stocks are under 1%.",
    "Likely range (10th-90th percentile)": "In past tests on data the model had not seen, about 8 opens in 10 "
                                           "landed inside a range this wide. 1 in 10 opened below it and 1 in 10 "
                                           "above it. It is not a hard limit.",
    "Confidence": "How often the model called the direction (up or down) right in the past when it expected a "
                  "move of the same size. 50% is a coin toss.",
    "Direction accuracy": "The share of days on which the forecast and the actual gap had the same sign (both up "
                          "or both down).",
    "MAE (mean absolute error)": "The average size of the miss, ignoring whether it was too high or too low. "
                                 "Lower is better. 0.50% on a ₹1,000 stock is ₹5.",
    "Naive baseline": "The simplest guess: today opens exactly at yesterday's close. A model is only useful if it "
                      "beats this.",
    "Walk-forward test": "The model is trained on older data and tested on the following months it never saw, "
                         "again and again, the way it would have been used in real time.",
    "Round-trip cost": "Everything you pay to buy and then sell: brokerage, STT, exchange and SEBI fees, GST, stamp "
                       "duty, plus the bid-ask spread and slippage. A trade only makes money if the move is bigger.",
    "Intraday (MIS) vs delivery (CNC)": "Intraday means buying and selling the same day (cheaper STT). Delivery "
                                        "means holding overnight or longer (STT on both sides, a DP charge on sale).",
    "Limit order": "An order that only fills at your price or better. Safer than a market order at 09:15, when "
                   "prices jump around.",
    "Stop loss": "A price at which you exit to cap a loss. Prices can gap through it, so the real exit can be "
                 "worse.",
    "Opening range": "The high and low of the first 15 minutes. Many traders wait for it before acting.",
    "ADR": "American Depositary Receipt: an Indian company's shares traded in the US (INFY, IBN for ICICI, HDB "
           "for HDFC Bank). Its overnight move often hints at the Indian open.",
    "USD/INR": "Rupees per US dollar. Up means a weaker rupee, which tends to help exporters such as TCS and "
               "Infosys.",
    "S&P 500 / Nasdaq 100": "The main US stock indices. They trade while India sleeps, so they carry most of the "
                            "overnight news.",
    "Backtest": "Replaying a trading rule on past data, with costs, to see what would have happened. Past results "
                "do not guarantee future ones.",
    "Monte Carlo": "Thousands of reshuffled versions of the backtest's weeks, to show the spread of possible "
                   "outcomes rather than one lucky or unlucky path.",
    "Drawdown": "The fall from the highest value so far to a later low. A 20% drawdown means ₹1,00,000 became "
                "₹80,000 at some point.",
    "Sharpe ratio": "Return per unit of risk. Above 1 is good; around 0 or negative means the risk was not paid "
                    "for.",
}


def render(compact: bool = False) -> None:
    if compact:
        with st.expander("Glossary: what the words on this page mean"):
            for term, meaning in TERMS.items():
                st.markdown(f"**{term}.** {meaning}")
        return
    st.subheader("Glossary")
    st.caption("Market words used in this app, in plain language.")
    for term, meaning in TERMS.items():
        st.markdown(f"**{term}.** {meaning}")
