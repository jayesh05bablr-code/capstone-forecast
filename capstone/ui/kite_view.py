"""Zerodha Kite order settings (alert + ATO, and GTT) for a list of planned trades."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import kite


def _controls(key: str) -> tuple[float, float]:
    c1, c2 = st.columns(2)
    slip = c1.number_input("Limit price beyond the trigger (%)", 0.0, 1.0, 0.1, 0.05, key=f"{key}_slip",
                           help="A LIMIT order a little past the trigger fills on a fast move but never "
                                "pays more than this.")
    tick = c2.selectbox("Tick size (₹)", [0.05, 0.01, 0.10], key=f"{key}_tick",
                        help="Kite rejects prices that are not a multiple of the stock's tick size. "
                             "If Kite complains, try 0.10 (higher-priced stocks) or 0.01 (low-priced).")
    return slip, tick


def render_scalps(plans: list[dict], key: str, ltps: dict[str, float] | None = None) -> None:
    """Kite settings for opening-range breakout plans (scalping.plan_row dicts)."""
    if not plans:
        return
    ltps = ltps or {}
    st.markdown("**On Zerodha Kite.** Nothing is placed from here: copy these into Kite yourself.")
    slip, tick = _controls(key)
    alerts = [kite.alert_ato(p["symbol"], p["side"], p["trigger"], p["stop"], p["target"], p["qty"],
                             ltps.get(p["symbol"], p.get("ltp")), slip, tick) for p in plans]
    st.markdown("*Alert + ATO (best for intraday, works for long and short).* Create alert: "
                "If Last price of the stock is >= / <= the value, and the alert triggers this MIS order.")
    st.dataframe(pd.DataFrame(alerts), hide_index=True, use_container_width=True)
    longs = [p for p in plans if p["side"] == "Long"]
    if longs:
        st.markdown("*GTT version (long setups only).* Kite's equity GTT uses CNC (delivery), as in your "
                    "screen: if you don't sell the same day the shares stay in your demat, and a short "
                    "can't be done with a GTT. Use it when you are happy to carry the stock.")
        entries = [kite.gtt_entry(p["symbol"], p["trigger"], p["qty"], ltps.get(p["symbol"], p.get("ltp")),
                                  slip, tick) for p in longs]
        exits = [kite.gtt_exit_oco(p["symbol"], p["stop"], p["target"], p["qty"],
                                   ltps.get(p["symbol"], p.get("ltp")), slip, tick) for p in longs]
        st.dataframe(pd.DataFrame(entries), hide_index=True, use_container_width=True)
        st.caption("Then, after the buy fills, a GTT OCO sell to exit at the stop or the target:")
        st.dataframe(pd.DataFrame(exits), hide_index=True, use_container_width=True)
    labels = [f"{a['Stock']} {a['Side']}" for a in alerts]
    pick = st.selectbox("Step-by-step Kite instructions for", labels, key=f"{key}_pick")
    a = alerts[labels.index(pick)]
    st.markdown("**Alert + ATO**\n\n" + kite.walkthrough_alert(a))
    if a["Side"] == "Long":
        p = plans[labels.index(pick)]
        ltp = ltps.get(p["symbol"], p.get("ltp"))
        e = kite.gtt_entry(p["symbol"], p["trigger"], p["qty"], ltp, slip, tick)
        x = kite.gtt_exit_oco(p["symbol"], p["stop"], p["target"], p["qty"], ltp, slip, tick)
        with st.expander("Or as a GTT (CNC)"):
            st.markdown(kite.walkthrough_gtt(e, x))


def render_swing(pl: pd.DataFrame, key: str = "inv") -> None:
    """GTT settings for the Invest planner (multi-day CNC trades): a buy and an OCO exit per stock."""
    if pl.empty:
        return
    st.markdown("**On Zerodha Kite (GTT).** Buy with a limit at the planned price, then protect it with a "
                "GTT OCO: one leg at the stop loss, one at the target. Nothing is placed from here.")
    slip, tick = _controls(key)
    rows = []
    for r in pl.itertuples():
        if r.shares <= 0:
            continue
        x = kite.gtt_exit_oco(r.symbol, r.stop, r.target, int(r.shares), r.entry, slip, tick)
        rows.append({"Stock": r.symbol, "Buy": f"LIMIT, CNC, qty {int(r.shares)}, price "
                                               f"₹{kite.to_tick(r.entry, tick):,.2f} (or a GTT Buy Single with "
                                               f"that trigger)", **{k: v for k, v in x.items() if k != "Stock"}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("% of LTP here is measured from the planned buy price; Kite shows it from the live price.")
