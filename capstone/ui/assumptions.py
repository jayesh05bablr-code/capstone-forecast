"""Sidebar inputs shared by the Weekly analysis, Backtesting and Monte Carlo tabs.

One set of assumptions drives all three tabs, so the cost line on the weekly charts, the
cost-aware filter and the simulated strategy are always consistent.
"""
from __future__ import annotations

import streamlit as st

from capstone import backtest, costs


def render(all_symbols: list[str]) -> tuple[backtest.BacktestConfig, list[str]]:
    with st.sidebar.expander("Trading assumptions (weekly analysis, backtest, Monte Carlo)", expanded=False):
        symbols = st.multiselect("Stocks", all_symbols, default=all_symbols, key="bt_symbols") or all_symbols
        capital = st.number_input("Initial capital (₹)", 10_000.0, 1e9, 1_000_000.0, 50_000.0, key="bt_capital")
        default_alloc = round(100 / len(symbols), 1)
        alloc = st.number_input("Position allocation per stock (% of capital)", 1.0, 100.0, default_alloc, 1.0,
                                key=f"bt_alloc_{len(symbols)}",
                                help="Each trade gets this share of the equity at the start of the week. "
                                     "Capped so all positions together never exceed 100%.") / 100
        mult = st.number_input("Cost-aware filter: forecast must exceed N x cost", 0.0, 10.0, 2.0, 0.5, key="bt_mult")
        st.markdown("**Charges** (assumptions: check your broker's current rates)")
        instrument = st.selectbox("Instrument", ["equity_delivery", "equity_intraday"], key="bt_inst",
                                  help="A Monday-to-Friday hold is a delivery (CNC) trade.")
        d = costs.INSTRUMENT_DEFAULTS[instrument]
        brokerage_pct = st.number_input("Brokerage % of order", 0.0, 1.0, 0.03, 0.01, format="%.3f", key="bt_brk") / 100
        brokerage_cap = st.number_input("Brokerage cap per order (₹, 0 = none)", 0.0, 1000.0, 20.0, 5.0, key="bt_cap")
        stt_buy = st.number_input("STT on buy %", 0.0, 1.0, d["stt_buy"] * 100, 0.005, format="%.4f", key=f"bt_sttb_{instrument}") / 100
        stt_sell = st.number_input("STT on sell %", 0.0, 1.0, d["stt_sell"] * 100, 0.005, format="%.4f", key=f"bt_stts_{instrument}") / 100
        stamp = st.number_input("Stamp duty on buy %", 0.0, 1.0, d["stamp_buy"] * 100, 0.001, format="%.4f", key=f"bt_stamp_{instrument}") / 100
        exch = st.number_input("Exchange transaction charge %", 0.0, 0.1, 0.00297, 0.0001, format="%.5f", key="bt_exch") / 100
        dp = st.number_input("DP charge per sell (₹, before GST)", 0.0, 100.0, float(d["dp_charge_per_sell"]), 0.5, key=f"bt_dp_{instrument}")
        spread = st.number_input("Half bid-ask spread per leg %", 0.0, 1.0, 0.02, 0.005, format="%.3f", key="bt_spread") / 100
        slip = st.number_input("Slippage per leg %", 0.0, 1.0, 0.03, 0.005, format="%.3f", key="bt_slip") / 100
    a = costs.CostAssumptions(instrument=instrument, brokerage_pct=brokerage_pct, brokerage_cap=brokerage_cap,
                              exchange_txn=exch, stt_buy=stt_buy, stt_sell=stt_sell, stamp_buy=stamp,
                              dp_charge_per_sell=dp, half_spread=spread, slippage=slip)
    cfg = backtest.BacktestConfig(initial_capital=capital, allocation=alloc, cost_multiple=mult, assumptions=a,
                                  symbols=tuple(symbols))
    return cfg, symbols


def position_value(cfg: backtest.BacktestConfig, symbols: list[str]) -> float:
    return cfg.initial_capital * cfg.alloc(len(symbols))
