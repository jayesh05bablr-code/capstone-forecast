"""One trading-cost model for every strategy in the project.

A round trip (buy, then sell) costs three different things, kept separate so their size is visible:

1. Explicit charges, paid to the broker, the exchange and the government:
   brokerage, exchange transaction charges, SEBI turnover fee, GST (on brokerage + exchange +
   SEBI fees), Securities Transaction Tax (STT) and stamp duty (buy side only), and for delivery
   trades an optional depository (DP) charge on the sell.
2. Spread: you buy at the ask and sell at the bid, so each leg loses half the bid-ask spread.
3. Slippage: the fill is worse than the quoted price (market impact, latency), per leg.

Which charges apply depends on the instrument and execution method:

| Charge          | Equity delivery (CNC, held overnight) | Equity intraday (MIS, same day) |
| --------------- | ------------------------------------- | ------------------------------- |
| STT             | buy and sell                          | sell only                       |
| Stamp duty      | buy                                   | buy                             |
| DP charge       | sell (flat per stock)                 | none                            |
| Exchange, SEBI  | both legs                             | both legs                       |
| GST             | on brokerage + exchange + SEBI        | same                            |

The weekly strategy holds from Monday's open to Friday's close, so it is a delivery trade.

IMPORTANT: the default rates below are ASSUMPTIONS taken from published Indian discount-broker
charge sheets, not verified current statutory rates. Rates change (STT, stamp duty and exchange
charges have all been revised in recent years) and differ by broker and exchange. Check them
against your broker's current charge list or a recent contract note and override them in the
dashboard or in code. Nothing in this module fetches live rates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

# Per-instrument statutory-style rates, as fractions of traded value. ASSUMPTIONS: see docstring.
INSTRUMENT_DEFAULTS = {
    "equity_delivery": {"stt_buy": 0.001, "stt_sell": 0.001, "stamp_buy": 0.00015, "dp_charge_per_sell": 15.93},
    "equity_intraday": {"stt_buy": 0.0, "stt_sell": 0.00025, "stamp_buy": 0.00003, "dp_charge_per_sell": 0.0},
}


@dataclass(frozen=True, eq=False)
class CostAssumptions:
    """Every number that drives the cost of a trade. All rates are fractions (0.001 = 0.1%)."""

    instrument: str = "equity_delivery"
    brokerage_pct: float = 0.0003          # brokerage as % of order value ...
    brokerage_cap: float = 20.0            # ... capped at this many rupees per order (0 = no cap)
    brokerage_flat: float = 0.0            # or a flat fee per order, added on top
    exchange_txn: float = 0.0000297        # NSE transaction charge, both legs
    sebi_fee: float = 0.000001             # SEBI turnover fee (Rs 10 per crore), both legs
    gst: float = 0.18                      # GST on brokerage + exchange + SEBI fees
    stt_buy: float | None = None           # None = take the instrument default
    stt_sell: float | None = None
    stamp_buy: float | None = None
    dp_charge_per_sell: float | None = None
    half_spread: float = 0.0002            # half the bid-ask spread, paid on each leg (2 bps)
    slippage: float = 0.0003               # adverse fill per leg (3 bps)
    extra: dict = field(default_factory=dict, compare=False)

    def resolved(self) -> "CostAssumptions":
        """Copy with instrument defaults filled in for any rate left as None."""
        d = INSTRUMENT_DEFAULTS[self.instrument]
        missing = {k: v for k, v in d.items() if getattr(self, k) is None}
        return replace(self, **missing) if missing else self

    def to_dict(self) -> dict:
        out = asdict(self.resolved())
        out.pop("extra", None)
        return out


def _brokerage(value: float, a: CostAssumptions) -> float:
    pct = a.brokerage_pct * value
    if a.brokerage_cap > 0:
        pct = min(pct, a.brokerage_cap)
    return pct + a.brokerage_flat if value > 0 else 0.0


def explicit_charges(buy_value: float, sell_value: float, a: CostAssumptions) -> dict[str, float]:
    """Brokerage, exchange, SEBI, GST, STT, stamp duty and DP charges for one round trip, in rupees."""
    a = a.resolved()
    brokerage = _brokerage(buy_value, a) + _brokerage(sell_value, a)
    exchange = a.exchange_txn * (buy_value + sell_value)
    sebi = a.sebi_fee * (buy_value + sell_value)
    gst = a.gst * (brokerage + exchange + sebi)
    stt = a.stt_buy * buy_value + a.stt_sell * sell_value
    stamp = a.stamp_buy * buy_value
    dp = a.dp_charge_per_sell * (1 + a.gst) if sell_value > 0 and a.dp_charge_per_sell else 0.0
    parts = {"brokerage": brokerage, "exchange": exchange, "sebi": sebi, "gst": gst, "stt": stt,
             "stamp_duty": stamp, "dp_charge": dp}
    parts["explicit_total"] = sum(parts.values())
    return parts


def round_trip(position_value: float, a: CostAssumptions, exit_value: float | None = None) -> dict[str, float]:
    """Full cost of buying `position_value` and selling it (at `exit_value`, default the same).

    Returns rupee amounts for explicit charges, spread and slippage, their total, and the total
    as a fraction of the position value (`total_pct`), which is what the weekly chart compares
    returns against.
    """
    exit_value = position_value if exit_value is None else exit_value
    parts = explicit_charges(position_value, exit_value, a)
    parts["spread"] = a.half_spread * (position_value + exit_value)
    parts["slippage"] = a.slippage * (position_value + exit_value)
    parts["total"] = parts["explicit_total"] + parts["spread"] + parts["slippage"]
    parts["total_pct"] = parts["total"] / position_value if position_value > 0 else float("nan")
    for k in ("explicit_total", "spread", "slippage"):
        parts[f"{k}_pct"] = parts[k] / position_value if position_value > 0 else float("nan")
    return parts


def round_trip_pct(position_value: float, a: CostAssumptions) -> float:
    """Total round-trip cost as a fraction of the position (e.g. 0.0035 = 0.35%)."""
    return round_trip(position_value, a)["total_pct"]


def breakdown_table(position_value: float, a: CostAssumptions):
    """Rows for the dashboard: each cost line in rupees and as % of the position."""
    import pandas as pd

    p = round_trip(position_value, a)
    names = [("brokerage", "Brokerage (both legs)"), ("exchange", "Exchange transaction charges"),
             ("sebi", "SEBI turnover fee"), ("gst", "GST"), ("stt", "STT"), ("stamp_duty", "Stamp duty (buy)"),
             ("dp_charge", "DP charge incl. GST (sell)"), ("explicit_total", "Explicit charges subtotal"),
             ("spread", "Bid-ask spread (half on each leg)"), ("slippage", "Slippage (each leg)"),
             ("total", "Total round trip")]
    return pd.DataFrame([{"Cost": label, "Rupees": round(p[k], 2), "% of position": round(p[k] / position_value * 100, 4)}
                         for k, label in names])
