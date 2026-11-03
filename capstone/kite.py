"""Turn a trade plan into the exact fields you fill in on Zerodha Kite.

Nothing here places an order. It only works out what to type in:

- **Alert + ATO** (Alert Triggers Order): "If Last price of SYM (NSE) is >= X", with an order
  that Kite places when the alert fires. This is the way to automate an intraday (MIS) scalp,
  long or short.
- **GTT** (Good Till Triggered): Buy/Sell, Single/OCO, CNC, trigger price, limit price and
  quantity. On Kite, equity GTTs use the CNC (delivery) product, so a GTT buy that you do not
  sell the same day stays in your demat as delivery and needs the full amount in cash. GTT
  fits the Invest planner (multi-day) better than a 20-minute scalp, and it cannot open a short.

Prices are rounded to the tick size (₹0.05 by default; Kite rejects a price that is not a
multiple of the stock's tick size).
"""
from __future__ import annotations

import math

DEFAULT_TICK = 0.05


def to_tick(price: float, tick: float = DEFAULT_TICK, up: bool | None = None) -> float:
    """Round a price to the tick size: up, down, or (None) to the nearest tick."""
    steps = price / tick
    steps = math.ceil(steps - 1e-9) if up else math.floor(steps + 1e-9) if up is False else round(steps)
    return round(steps * tick, 2)


def pct_of(price: float, ltp: float | None) -> float | None:
    """How far a price is from the last traded price, in % (what Kite shows as '% of LTP')."""
    if not ltp:
        return None
    return round((price / ltp - 1) * 100, 2)


def alert_ato(symbol: str, side: str, trigger: float, stop: float, target: float, qty: int,
              ltp: float | None = None, slippage_pct: float = 0.1, tick: float = DEFAULT_TICK) -> dict:
    """Kite alert with an ATO basket for an intraday breakout (side 'Long' or 'Short').

    The entry is a LIMIT order a little beyond the trigger (slippage_pct), so it fills on a fast
    breakout without paying any price. Exit orders are placed by hand right after the entry fills.
    """
    long = side == "Long"
    trig = to_tick(trigger, tick, up=long)
    limit = to_tick(trigger * (1 + slippage_pct / 100 * (1 if long else -1)), tick, up=long)
    stop_t = to_tick(stop, tick, up=not long)
    target_t = to_tick(target, tick, up=not long)
    exit_side = "SELL" if long else "BUY"
    return {
        "Stock": symbol,
        "Side": side,
        "Alert: If": "Last price",
        "Alert: of": f"{symbol} (NSE)",
        "Alert: is": "Greater than or equal to (>=)" if long else "Less than or equal to (<=)",
        "Alert: than (₹)": trig,
        "Alert: % of Last price": pct_of(trig, ltp),
        "ATO order": "BUY" if long else "SELL",
        "ATO product": "MIS (intraday)",
        "ATO order type": "LIMIT",
        "ATO qty": qty,
        "ATO price (₹)": limit,
        "Exit 1: stop-loss": f"{exit_side} SL-M, MIS, trigger ₹{stop_t:,.2f}, qty {qty}",
        "Exit 2: target": f"{exit_side} LIMIT, MIS, price ₹{target_t:,.2f}, qty {qty}",
    }


def gtt_entry(symbol: str, trigger: float, qty: int, ltp: float | None = None,
              slippage_pct: float = 0.1, tick: float = DEFAULT_TICK) -> dict:
    """GTT to buy on a breakout above `trigger` (Type Buy, Trigger type Single, CNC, LIMIT)."""
    trig = to_tick(trigger, tick, up=True)
    return {
        "Stock": symbol,
        "Type": "Buy",
        "Trigger type": "Single",
        "Product": "CNC",
        "Order": "LIMIT",
        "Trigger price (₹)": trig,
        "% of LTP": pct_of(trig, ltp),
        "Quantity": qty,
        "Price (₹)": to_tick(trigger * (1 + slippage_pct / 100), tick, up=True),
    }


def gtt_exit_oco(symbol: str, stop: float, target: float, qty: int, ltp: float | None = None,
                 slippage_pct: float = 0.1, tick: float = DEFAULT_TICK) -> dict:
    """GTT OCO to sell shares you hold: one leg at the stop, one at the target; the first to
    trigger cancels the other. The stop leg's limit sits below its trigger so it fills on a fall."""
    stop_t = to_tick(stop, tick, up=False)
    target_t = to_tick(target, tick, up=False)
    return {
        "Stock": symbol,
        "Type": "Sell",
        "Trigger type": "OCO",
        "Product": "CNC",
        "Order": "LIMIT",
        "Stop-loss trigger (₹)": stop_t,
        "Stop-loss % of LTP": pct_of(stop_t, ltp),
        "Stop-loss price (₹)": to_tick(stop * (1 - slippage_pct / 100), tick, up=False),
        "Target trigger (₹)": target_t,
        "Target % of LTP": pct_of(target_t, ltp),
        "Target price (₹)": target_t,
        "Quantity": qty,
    }


def walkthrough_alert(row: dict) -> str:
    """Step-by-step Kite instructions for one alert_ato() row, in plain words."""
    pct = row["Alert: % of Last price"]
    pct_txt = f" (Kite will show about {pct:+.2f}% of Last price)" if pct is not None else ""
    return "\n".join([
        f"1. In Kite, open **{row['Stock']}**, tap **Create alert**.",
        f"2. If **Last price** of **{row['Alert: of']}** is **{row['Alert: is']}** than "
        f"**{row['Alert: than (₹)']:,.2f}**{pct_txt}.",
        f"3. Turn on **Alert Triggers Order (ATO)**, add {row['Stock']} and set: **{row['ATO order']}**, "
        f"**{row['ATO product']}**, **{row['ATO order type']}**, qty **{row['ATO qty']}**, "
        f"price **{row['ATO price (₹)']:,.2f}**. Swipe **Create**.",
        f"4. As soon as the entry fills, place the exits: {row['Exit 1: stop-loss']}; and "
        f"{row['Exit 2: target']}. When one fills, cancel the other.",
        "5. Kite squares off open MIS positions near 15:20 automatically, but close the trade yourself "
        "when your trade window ends. Delete the alert if the setup is gone.",
    ])


def walkthrough_gtt(entry: dict, exit_: dict) -> str:
    """Step-by-step Kite GTT instructions for a gtt_entry() and gtt_exit_oco() pair."""
    return "\n".join([
        f"1. In Kite, open **{entry['Stock']}**, tap **GTT** (Create GTT).",
        f"2. Type **Buy**, Trigger type **Single**, Product **CNC**, Order **LIMIT**.",
        f"3. Trigger price **{entry['Trigger price (₹)']:,.2f}**, Quantity **{entry['Quantity']}**, "
        f"Price **{entry['Price (₹)']:,.2f}**. Agree to the terms and swipe **Create GTT**.",
        f"4. After the buy fills, create a second GTT: Type **Sell**, Trigger type **OCO**, Product **CNC**. "
        f"Stop-loss trigger **{exit_['Stop-loss trigger (₹)']:,.2f}**, price **{exit_['Stop-loss price (₹)']:,.2f}**; "
        f"Target trigger **{exit_['Target trigger (₹)']:,.2f}**, price **{exit_['Target price (₹)']:,.2f}**; "
        f"Quantity **{exit_['Quantity']}**.",
        "5. CNC is delivery: you need the full amount in cash, and shares you do not sell stay in your demat.",
    ])
