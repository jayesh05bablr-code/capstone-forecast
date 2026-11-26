"""Beginner safety: the "Should I trade today?" check, loss limits, position sizing, paper trading and
an honest after-charges track record of the app's own signals.

Pure pandas (no Streamlit), tested in tests/test_safety.py. The page is capstone/ui/tab_safety.py.

Why this module exists. Pulse8 is meant for people with little market experience. SEBI's study of
FY2023 equity intraday traders found that about 7 in 10 lost money, and the app's own tests show its
opening forecast barely beats "no change". So every rule here leans towards NOT trading:

* The check says "No trade today" unless the expected move is clearly bigger than the costs, the
  model's past record on forecasts that size is decent, the app's own past signals actually made money
  after charges, and the user has a profitable practice record on paper.
* Position size comes from how much the user is willing to lose, not from how much they have.
* A daily loss limit and a cap of two trades a day stop one bad day from becoming a disaster.

What a "signal" trades. The forecast is for the overnight gap (yesterday's close to today's open).
Nobody can buy at yesterday's close after seeing last night's news, so the gap itself cannot be
collected. What a beginner CAN do is enter at the open in the forecast's direction and exit by the
close. That open-to-close trade, after charges, is what the track record and paper trades measure.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time

import numpy as np
import pandas as pd

from . import costs, market_calendar

CLEAR_EDGE_MULTIPLE = 2.0   # "clearly beats costs": expected move at least this many times the round trip
MIN_SIGNALS = 30            # past signals needed before the track record counts as evidence
PAPER_TRADES_TO_GRADUATE = 20  # settled practice trades needed before the check stops saying "paper only"
MAX_SESSION_GAP_DAYS = 7    # same rule as features.py: skip trades across holes in the price history
OPEN_TIME = time(9, 15)
PAPER_START = 100_000.0     # virtual money for practice
INTRADAY = costs.CostAssumptions(instrument="equity_intraday")


def clear_edge_pct(cost_pct: float) -> float:
    """Smallest expected move that "clearly beats costs", rounded up to the 0.01% shown on screen."""
    return math.ceil(CLEAR_EDGE_MULTIPLE * cost_pct * 100 - 1e-9) / 100


# --- Rules and position sizing -------------------------------------------------------------------

@dataclass
class Rules:
    """The user's own limits. Defaults are deliberately cautious."""

    capital: float = 50_000.0          # money set aside for trading (never money needed to live on)
    risk_per_trade_pct: float = 1.0    # most you accept losing on one trade, % of capital
    daily_loss_limit_pct: float = 2.0  # stop trading for the day once losses could reach this, % of capital
    stop_loss_pct: float = 1.0         # exit if the price moves this much against you, % of entry
    max_trades_per_day: int = 2
    max_position_pct: float = 50.0     # never put more than this share of capital in one trade (no leverage)

    @property
    def risk_per_trade(self) -> float:
        return self.capital * self.risk_per_trade_pct / 100

    @property
    def daily_loss_limit(self) -> float:
        return self.capital * self.daily_loss_limit_pct / 100

    def problems(self) -> list[str]:
        """Plain warnings when a limit is set looser than a beginner should use."""
        out = []
        if self.risk_per_trade_pct > 2:
            out.append("Risking more than 2% of your capital on one trade means a short losing streak does real damage.")
        if self.daily_loss_limit_pct > 5:
            out.append("A daily loss limit above 5% can wipe out a month of careful trading in one bad day.")
        if self.daily_loss_limit_pct < self.risk_per_trade_pct:
            out.append("Your daily loss limit is smaller than the risk on one trade, so no trade fits inside it.")
        if self.stop_loss_pct < 0.3:
            out.append("A stop-loss under 0.3% is inside normal price noise and will be hit on most trades.")
        return out


@dataclass
class Size:
    qty: int
    value: float          # rupees committed at the entry price
    loss_at_stop: float   # rupees lost if the stop-loss is hit, charges included
    charges: float        # round-trip charges, spread and slippage for this size
    limited_by: str       # "risk", "capital" or "price" (one share costs more than allowed)


def position_size(price: float, rules: Rules, cost: costs.CostAssumptions = INTRADAY) -> Size:
    """How many shares to buy so that hitting the stop-loss loses at most rules.risk_per_trade.

    The loss at the stop includes the round-trip charges, because a beginner pays them win or lose.
    The position is also capped at rules.max_position_pct of capital: no borrowed money.
    """
    if price <= 0 or rules.capital <= 0:
        return Size(0, 0.0, 0.0, 0.0, "price")
    stop = rules.stop_loss_pct / 100
    per_share = price * stop + price * costs.round_trip_pct(price * 100, cost)  # rough cost share per share
    by_risk = math.floor(rules.risk_per_trade / per_share) if per_share > 0 else 0
    by_capital = math.floor(rules.capital * rules.max_position_pct / 100 / price)
    qty, why = (by_risk, "risk") if by_risk <= by_capital else (by_capital, "capital")
    # Exact charges for the chosen size; shrink while the true loss at the stop is over the limit.
    while qty > 0 and _loss_at_stop(price, qty, stop, cost) > rules.risk_per_trade + 1e-9:
        qty -= 1
    if qty <= 0:
        return Size(0, 0.0, 0.0, 0.0, "price")
    value = qty * price
    return Size(qty, value, _loss_at_stop(price, qty, stop, cost), costs.round_trip(value, cost)["total"], why)


def _loss_at_stop(price: float, qty: int, stop: float, cost: costs.CostAssumptions) -> float:
    value = qty * price
    return value * stop + costs.round_trip(value, cost, exit_value=value * (1 - stop))["total"]


# --- Honest track record of past signals ----------------------------------------------------------

def signal_trades(signals: pd.DataFrame, prices: dict[str, pd.DataFrame], cost_pct: float,
                  min_move_pct: float = 0.0) -> pd.DataFrame:
    """What trading each past forecast would have done: enter at the next open, exit at that close.

    signals: one row per forecast with symbol, date (the close the forecast was made after) and
    pred_gap_pct (% points). prices: symbol -> daily prices (date, open, close). Only forecasts with
    |pred_gap_pct| >= min_move_pct count as signals. Direction follows the forecast (up = buy, down =
    sell short intraday). net_pct = gross_pct - cost_pct, all in % points of the position.
    """
    cols = ["symbol", "date", "trade_date", "pred_gap_pct", "side", "gross_pct", "net_pct"]
    s = signals.dropna(subset=["pred_gap_pct"])
    s = s[(s["pred_gap_pct"].abs() >= min_move_pct) & (s["pred_gap_pct"] != 0)]
    out = []
    for sym, g in s.groupby("symbol"):
        p = prices.get(sym)
        if p is None or p.empty:
            continue
        p = p.sort_values("date")
        nxt = pd.DataFrame({"date": p["date"].to_numpy(), "trade_date": p["date"].shift(-1).to_numpy(),
                            "t_open": p["open"].shift(-1).to_numpy(), "t_close": p["close"].shift(-1).to_numpy()})
        nxt = nxt[(nxt["trade_date"] - nxt["date"]).dt.days <= MAX_SESSION_GAP_DAYS]
        j = g.assign(date=pd.to_datetime(g["date"])).merge(nxt, on="date", how="inner").dropna(subset=["t_open", "t_close"])
        j = j[j["t_open"] > 0]
        if j.empty:
            continue
        direction = np.sign(j["pred_gap_pct"])
        gross = direction * (j["t_close"] / j["t_open"] - 1) * 100
        out.append(j.assign(side=np.where(direction > 0, "BUY", "SELL"), gross_pct=gross, net_pct=gross - cost_pct)[cols])
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True).sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def track_summary(trades: pd.DataFrame, amount: float = 10_000.0) -> dict:
    """Plain numbers for a set of signal trades: count, wins after charges, average and total in rupees."""
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    net = trades["net_pct"]
    per_trade = net / 100 * amount
    return {"trades": n, "wins": int((net > 0).sum()), "win_rate_pct": float((net > 0).mean() * 100),
            "avg_net_pct": float(net.mean()), "avg_gross_pct": float(trades["gross_pct"].mean()),
            "total_rupees": float(per_trade.sum()), "amount": amount,
            "worst_rupees": float(per_trade.min()), "best_rupees": float(per_trade.max()),
            "first": pd.Timestamp(trades["trade_date"].min()).date(), "last": pd.Timestamp(trades["trade_date"].max()).date(),
            "worst_run": _worst_run(per_trade)}


def _worst_run(per_trade: pd.Series) -> float:
    """Largest peak-to-trough fall of the running rupee total (0 if it never fell)."""
    equity = per_trade.cumsum()
    return float((equity - equity.cummax().clip(lower=0)).min()) if len(equity) else 0.0


def oos_signals(oos: pd.DataFrame | None, model: str = "forecast") -> pd.DataFrame:
    """Past out-of-sample forecasts (models/oos_predictions.csv) as signals for signal_trades()."""
    if oos is None or oos.empty:
        return pd.DataFrame(columns=["symbol", "date", "pred_gap_pct"])
    d = oos[oos["model"] == model]
    return pd.DataFrame({"symbol": d["symbol"], "date": pd.to_datetime(d["date"]), "pred_gap_pct": d["pred"] * 100})


def log_signals(log: pd.DataFrame | None) -> pd.DataFrame:
    """Forecasts the app actually logged before each open (data/processed/predictions_log.csv)."""
    if log is None or log.empty:
        return pd.DataFrame(columns=["symbol", "date", "pred_gap_pct"])
    d = log.drop_duplicates(["symbol", "as_of_close"], keep="last")
    return pd.DataFrame({"symbol": d["symbol"], "date": pd.to_datetime(d["as_of_close"]), "pred_gap_pct": d["pred_gap_pct"]})


def record_supports_trading(summary: dict) -> bool:
    """The app's own past signals only count in favour of trading if there are enough and they made money."""
    return summary.get("trades", 0) >= MIN_SIGNALS and summary.get("avg_net_pct", -1) > 0


# --- Paper trading ---------------------------------------------------------------------------------

JOURNAL_COLS = ["id", "placed_at", "trade_date", "symbol", "side", "qty", "stop_pct", "status", "entry", "exit",
                "exit_reason", "gross", "charges", "net", "planned_risk"]


def empty_journal() -> pd.DataFrame:
    return pd.DataFrame(columns=JOURNAL_COLS)


def paper_trade_date(now: datetime) -> date:
    """Practice orders fill at an open that is still ahead: today's before 09:15, else the next session's.

    This keeps practice honest: nobody can place an order after seeing how the day went.
    """
    today = now.date()
    if market_calendar.is_trading_day(today) and now.time() < OPEN_TIME:
        return today
    return market_calendar.next_trading_day(today)


def can_place(journal: pd.DataFrame, rules: Rules, trade_date: date, symbol: str, qty: int, ref_price: float,
              balance: float) -> tuple[bool, str]:
    """Check a new practice trade against the user's own limits before it is accepted."""
    if qty <= 0:
        return False, "Quantity must be at least one share."
    day = journal[(pd.to_datetime(journal["trade_date"]).dt.date == trade_date) & (journal["status"] != "cancelled")] \
        if len(journal) else journal
    if len(day) >= rules.max_trades_per_day:
        return False, (f"You already have {len(day)} practice trades for {trade_date:%a %d %b}, your daily maximum. "
                       "Fewer, better trades beat many small ones once costs are paid.")
    if len(day) and (day["symbol"] == symbol).any():
        return False, f"You already have a practice trade in {symbol} for that day."
    value = qty * ref_price
    if value > balance:
        return False, f"That needs about ₹{value:,.0f}, more than your virtual balance of ₹{balance:,.0f}. No borrowing."
    if value > rules.capital * rules.max_position_pct / 100 + 1e-6:
        return False, (f"That is ₹{value:,.0f}, over your {rules.max_position_pct:.0f}% of capital limit per trade "
                       f"(₹{rules.capital * rules.max_position_pct / 100:,.0f}).")
    stop = rules.stop_loss_pct / 100
    new_risk = _loss_at_stop(ref_price, qty, stop, INTRADAY)
    if new_risk > rules.risk_per_trade + 1:
        return False, (f"If the stop-loss is hit you would lose about ₹{new_risk:,.0f}, over your limit of "
                       f"₹{rules.risk_per_trade:,.0f} per trade. Use fewer shares (the sizing box shows how many).")
    open_risk = float(day["planned_risk"].fillna(0).sum()) if len(day) and "planned_risk" in day else 0.0
    if open_risk + new_risk > rules.daily_loss_limit + 1:
        return False, (f"Together with your other trades that day you could lose ₹{open_risk + new_risk:,.0f}, over your "
                       f"daily loss limit of ₹{rules.daily_loss_limit:,.0f}. That's the limit doing its job.")
    return True, "Within your limits."


def new_trade(journal: pd.DataFrame, now: datetime, trade_date: date, symbol: str, side: str, qty: int,
              stop_pct: float, ref_price: float) -> pd.DataFrame:
    """Journal with one more practice order, waiting to fill at trade_date's open."""
    row = {"id": int(journal["id"].max()) + 1 if len(journal) else 1, "placed_at": now.strftime("%Y-%m-%d %H:%M"),
           "trade_date": trade_date.isoformat(), "symbol": symbol, "side": side, "qty": int(qty),
           "stop_pct": float(stop_pct), "status": "waiting", "entry": np.nan, "exit": np.nan, "exit_reason": "",
           "gross": np.nan, "charges": np.nan, "net": np.nan,
           "planned_risk": _loss_at_stop(ref_price, int(qty), stop_pct / 100, INTRADAY)}
    return pd.concat([journal, pd.DataFrame([row])], ignore_index=True) if len(journal) else pd.DataFrame([row])


def settle(journal: pd.DataFrame, prices: dict[str, pd.DataFrame], cost: costs.CostAssumptions = INTRADAY) -> pd.DataFrame:
    """Fill waiting practice trades from the day's real prices once that day's candle exists.

    Entry is that day's open. Exit is the stop-loss price if the day's low (for a buy) or high (for a
    short) reached it, otherwise the close. When the stop was touched we assume it filled exactly at
    the stop; real fills are often a little worse. Charges are the intraday round trip.
    """
    if journal is None or journal.empty:
        return empty_journal()
    j = journal.copy()
    for k in ("entry", "exit", "gross", "charges", "net"):
        j[k] = pd.to_numeric(j[k], errors="coerce").astype(float)
    j["exit_reason"] = j["exit_reason"].fillna("").astype(object)
    for i, r in j[j["status"] == "waiting"].iterrows():
        p = prices.get(r["symbol"])
        if p is None or p.empty:
            continue
        day = p[pd.to_datetime(p["date"]).dt.date == pd.Timestamp(r["trade_date"]).date()]
        if day.empty:
            continue
        d = day.iloc[-1]
        entry, qty, stop = float(d["open"]), int(r["qty"]), float(r["stop_pct"]) / 100
        if r["side"] == "BUY":
            stop_px = entry * (1 - stop)
            hit = float(d.get("low", np.nan)) <= stop_px
            exit_px = stop_px if hit else float(d["close"])
            gross = (exit_px - entry) * qty
            buy_value, sell_value = entry * qty, exit_px * qty
        else:
            stop_px = entry * (1 + stop)
            hit = float(d.get("high", np.nan)) >= stop_px
            exit_px = stop_px if hit else float(d["close"])
            gross = (entry - exit_px) * qty
            buy_value, sell_value = exit_px * qty, entry * qty
        charges = costs.round_trip(buy_value, cost, exit_value=sell_value)["total"]
        filled = {"status": "settled", "entry": round(entry, 2), "exit": round(exit_px, 2),
                  "exit_reason": "stop-loss hit" if hit else "closed at 15:30", "gross": round(gross, 2),
                  "charges": round(charges, 2), "net": round(gross - charges, 2)}
        for k, v in filled.items():
            j.at[i, k] = v
    return j


def paper_summary(journal: pd.DataFrame, start: float = PAPER_START) -> dict:
    """Balance and record of the practice account. 'ready' only after enough trades AND a profit after charges."""
    done = journal[journal["status"] == "settled"] if len(journal) else journal
    net = pd.to_numeric(done["net"], errors="coerce").fillna(0) if len(done) else pd.Series(dtype=float)
    n = len(done)
    total = float(net.sum()) if n else 0.0
    return {"start": start, "balance": start + total, "settled": n, "waiting": int((journal["status"] == "waiting").sum()) if len(journal) else 0,
            "wins": int((net > 0).sum()) if n else 0, "net": total,
            "charges": float(pd.to_numeric(done["charges"], errors="coerce").fillna(0).sum()) if n else 0.0,
            "worst_run": _worst_run(net.reset_index(drop=True)) if n else 0.0,
            "ready": n >= PAPER_TRADES_TO_GRADUATE and total > 0}


# --- The "Should I trade today?" check ---------------------------------------------------------------

@dataclass
class Check:
    ok: bool
    title: str
    detail: str


@dataclass
class Verdict:
    answer: str      # "No trade today", "Practise on paper only", "Stop for today", "Rules allow one small trade"
    tone: str        # "stop", "paper", "go"
    summary: str
    checks: list[Check]

    def to_dict(self) -> dict:
        return asdict(self)


def should_i_trade(gap_pct: float, conf, cost_pct: float, record: dict, paper: dict, trades_today: int,
                   rules: Rules) -> Verdict:
    """One stock's answer to "Should I trade today?". Every check must pass for anything but 'no'.

    conf: guidance.Confidence. record: track_summary() of past signals the same size or bigger.
    paper: paper_summary(). Approximate confidence never counts as decent.
    """
    need = clear_edge_pct(cost_pct)
    checks = [
        Check(abs(gap_pct) >= need, "The expected move clearly beats the costs",
              f"Expected move {gap_pct:+.2f}%; it needs to be at least {need:.2f}% ({CLEAR_EDGE_MULTIPLE:.0f} times the "
              f"{cost_pct:.2f}% a buy-and-sell costs) to leave room for being wrong."),
        Check(conf.level in ("High", "Medium") and not conf.approximate, "The model has a decent record on moves this size",
              f"Confidence: {conf.label}. " + (conf.sentence if conf.level != "Unknown" else "No usable record yet.")),
        Check(record_supports_trading(record), "Past signals like this made money after charges",
              _record_line(record)),
        Check(paper.get("ready", False), "Your own practice record is profitable",
              f"{paper.get('settled', 0)} practice trades settled (need {PAPER_TRADES_TO_GRADUATE}), "
              f"net {'+' if paper.get('net', 0) >= 0 else '-'}₹{abs(paper.get('net', 0)):,.0f} after charges."),
        Check(trades_today < rules.max_trades_per_day, "You are within today's limits",
              f"{trades_today} of {rules.max_trades_per_day} trades used today; daily loss limit ₹{rules.daily_loss_limit:,.0f}."),
    ]
    edge_ok = all(c.ok for c in checks[:3])
    if not checks[4].ok:
        return Verdict("Stop for today", "stop", "You have used today's trades. Stopping is part of the plan.", checks)
    if not edge_ok:
        return Verdict("No trade today", "stop", "The numbers do not show an edge after costs. Not trading is the "
                       "right call, and it costs nothing.", checks)
    if not checks[3].ok:
        return Verdict("Practise on paper only", "paper", "The signal passes, but you have not yet shown a profit "
                       "after charges on paper. Practise this one with virtual money.", checks)
    return Verdict("Rules allow one small trade", "go", "Every check passes. If you trade, use the size and stop-loss "
                   "below and nothing bigger. This is still not a promise of profit.", checks)


def _record_line(record: dict) -> str:
    n = record.get("trades", 0)
    if n == 0:
        return "No past signals of this size to check."
    words = (f"{n} past signals this size or bigger: {record['wins']} made money after charges, average "
             f"{record['avg_net_pct']:+.2f}% per trade")
    if n < MIN_SIGNALS:
        return words + f". Too few to count (need {MIN_SIGNALS})."
    return words + "."


def day_answer(verdicts: list[Verdict]) -> Verdict | None:
    """The page's headline: the most permissive single-stock answer (usually still 'no')."""
    if not verdicts:
        return None
    rank = {"go": 2, "paper": 1, "stop": 0}
    return max(verdicts, key=lambda v: rank[v.tone])
