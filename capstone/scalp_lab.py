"""Scalping lab: long and short intraday setups on 1- to 5-minute bars, tested walk-forward with every charge.

Pure pandas/numpy (no Streamlit), tested in tests/test_scalp_lab.py. The page is capstone/ui/tab_scalping.py.

How a trade is simulated (the same rules for the backtest, the live signal and paper trading):

* A setup fires when a bar CLOSES. The trade enters at the NEXT bar's open, never at a price that was
  already in the past when the signal appeared.
* Stop-loss = entry -/+ `stop_atr` x ATR (average true range of recent bars). Target = entry +/- `rr` x
  that risk. Short trades are the mirror image of long ones (intraday MIS shorting).
* Each later bar: if it OPENS beyond the stop, the exit is that open (a gap through the stop is worse
  than the stop). If both the stop and the target sit inside one bar, the stop is assumed first.
* A trade still open after `max_hold` bars exits at that bar's close. Everything is closed by 15:15
  (brokers square MIS positions off around 15:20), and no new trade starts after 14:45.
* At most one open trade per stock, and at most `max_trades` per stock per day.
* Size comes from the user's loss limit (safety.Rules): the loss at the stop, charges included, stays
  within the risk per trade, and the position never exceeds max_position_pct of capital (no leverage).
* Costs: costs.round_trip with the equity-intraday profile, i.e. brokerage, STT on the sell, exchange,
  SEBI, GST, stamp duty, plus half the bid-ask spread and slippage on BOTH legs.

Walk-forward. For each strategy there is a small grid of settings. For every test day, the settings
that made the most money after charges over the previous `train_days` trading days are chosen, and
only that day's trades with those settings are counted. So every number in the record is from a day
the settings had not seen. Nothing here can promise a profit; the verdict says "no proven edge" unless
the out-of-sample record is positive and large enough to be more than luck.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import costs, safety

INTRADAY = costs.CostAssumptions(instrument="equity_intraday")
LAST_ENTRY = 14 * 60 + 45   # minutes after midnight: no new trades after 14:45
SQUARE_OFF = 15 * 60 + 15   # flat by 15:15
OPEN_MIN = 9 * 60 + 15
MIN_TRADES_FOR_VERDICT = 30

STRATEGIES = {
    "orb": "Opening-range breakout",
    "vwap_trend": "VWAP trend (EMA 9/21 cross)",
    "vwap_revert": "VWAP stretch reversal",
    "random": "Coin-flip entries (baseline)",
}
PLAIN = {
    "orb": "Buy when a bar closes above the high of the first minutes of the day; sell short when one closes "
           "below the low.",
    "vwap_trend": "Buy when the fast average (EMA 9) crosses above the slow one (EMA 21) while price is above "
                  "VWAP; sell short on the opposite cross below VWAP.",
    "vwap_revert": "When price stretches far below VWAP and a green bar appears, buy for a bounce; when it "
                   "stretches far above and a red bar appears, sell short for a pullback.",
    "random": "Enters long or short at random times with the same stops, targets and charges. Any real "
              "setup should beat this; if it doesn't, its 'edge' is luck.",
}
# Setting grids searched walk-forward. Signal settings first, then exit settings.
SIGNAL_GRID = {
    "orb": {"or_minutes": [15, 30]},
    "vwap_trend": {"fast": [9], "slow": [21]},
    "vwap_revert": {"stretch_atr": [2.0, 3.0]},
    "random": {"p": [0.02]},
}
EXIT_GRID = {"stop_atr": [1.0, 1.5, 2.0], "rr": [1.0, 1.5, 2.0]}


@dataclass(frozen=True)
class Params:
    strategy: str
    signal: tuple = ()          # sorted (name, value) pairs from SIGNAL_GRID
    stop_atr: float = 1.5
    rr: float = 1.5
    max_hold: int = 12          # bars
    max_trades: int = 3         # per stock per day

    @property
    def sig(self) -> dict:
        return dict(self.signal)

    def label(self) -> str:
        bits = [f"{k.replace('_', ' ')} {v:g}" for k, v in self.signal if k != "p"]
        return ", ".join(bits + [f"stop {self.stop_atr:g}×ATR", f"target {self.rr:g}×risk"])


def grid(strategy: str) -> list[Params]:
    sg = SIGNAL_GRID[strategy]
    out = []
    for sv in itertools.product(*sg.values()):
        sig = tuple(sorted(zip(sg.keys(), sv)))
        for s, r in itertools.product(EXIT_GRID["stop_atr"], EXIT_GRID["rr"]):
            out.append(Params(strategy, sig, s, r))
    return out


# --- Bars and indicators ---------------------------------------------------------------------------

def resample(minutes: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """1-minute candles to N-minute candles, bars anchored at 09:15 each day."""
    m = minutes.copy()
    m["datetime"] = pd.to_datetime(m["datetime"])
    if "volume" not in m:
        m["volume"] = 0.0
    m["volume"] = pd.to_numeric(m["volume"], errors="coerce").fillna(0.0)
    if bar_minutes <= 1:
        return m[["datetime", "open", "high", "low", "close", "volume"]].sort_values("datetime").reset_index(drop=True)
    day = m["datetime"].dt.normalize()
    mins = (m["datetime"] - day).dt.total_seconds() // 60 - OPEN_MIN
    m["bucket"] = day + pd.to_timedelta(OPEN_MIN + (mins // bar_minutes) * bar_minutes, unit="m")
    g = m.sort_values("datetime").groupby("bucket")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    return out.rename_axis("datetime").reset_index()


def add_indicators(bars: pd.DataFrame, atr_n: int = 14) -> pd.DataFrame:
    """Per-day VWAP, EMA 9 and 21 (reset each morning), ATR (carried across days), minute of day."""
    b = bars.sort_values("datetime").reset_index(drop=True).copy()
    b["day"] = b["datetime"].dt.date
    b["tmin"] = b["datetime"].dt.hour * 60 + b["datetime"].dt.minute
    typical = (b["high"] + b["low"] + b["close"]) / 3
    vol = b["volume"].where(b["volume"] > 0, 1.0)  # no volume (e.g. some uploads): equal weights
    b["vwap"] = (typical * vol).groupby(b["day"]).cumsum() / vol.groupby(b["day"]).cumsum()
    for n in (9, 21):
        b[f"ema{n}"] = b.groupby("day")["close"].transform(lambda s, n=n: s.ewm(span=n, adjust=False).mean())
    prev = b["close"].shift()
    tr = pd.concat([b["high"] - b["low"], (b["high"] - prev).abs(), (b["low"] - prev).abs()], axis=1).max(axis=1)
    b["atr"] = tr.rolling(atr_n, min_periods=3).mean()
    b["bar_no"] = b.groupby("day").cumcount()
    return b


def signals(b: pd.DataFrame, p: Params, seed: int = 7) -> np.ndarray:
    """+1 (buy), -1 (sell short) or 0 for each bar, decided on that bar's close only."""
    s = np.zeros(len(b), dtype=int)
    k = p.sig
    if p.strategy == "orb":
        orm = k.get("or_minutes", 15)
        in_or = b["tmin"] < OPEN_MIN + orm
        hi = b["high"].where(in_or).groupby(b["day"]).transform("max")
        lo = b["low"].where(in_or).groupby(b["day"]).transform("min")
        after = ~in_or
        up = after & (b["close"] > hi)
        dn = after & (b["close"] < lo)
        # first close beyond the range, each direction, each day
        first_up = up & (up.groupby(b["day"]).cumsum() == 1)
        first_dn = dn & (dn.groupby(b["day"]).cumsum() == 1)
        s[first_up.to_numpy()] = 1
        s[first_dn.to_numpy()] = -1
    elif p.strategy == "vwap_trend":
        f, sl = b[f"ema{k.get('fast', 9)}"], b[f"ema{k.get('slow', 21)}"]
        diff = f - sl
        prev = diff.groupby(b["day"]).shift()
        cross_up = (prev <= 0) & (diff > 0) & (b["close"] > b["vwap"])
        cross_dn = (prev >= 0) & (diff < 0) & (b["close"] < b["vwap"])
        warm = b["bar_no"] >= 3
        s[(cross_up & warm).to_numpy()] = 1
        s[(cross_dn & warm).to_numpy()] = -1
    elif p.strategy == "vwap_revert":
        st_ = k.get("stretch_atr", 2.0) * b["atr"]
        long_ = (b["close"] < b["vwap"] - st_) & (b["close"] > b["open"])
        short = (b["close"] > b["vwap"] + st_) & (b["close"] < b["open"])
        warm = b["bar_no"] >= 3
        s[(long_ & warm).to_numpy()] = 1
        s[(short & warm).to_numpy()] = -1
    elif p.strategy == "random":
        rng = np.random.default_rng(seed)
        fire = rng.random(len(b)) < k.get("p", 0.02)
        s[fire] = rng.choice([-1, 1], size=int(fire.sum()))
    s[(b["tmin"] >= LAST_ENTRY).to_numpy()] = 0
    s[b["atr"].isna().to_numpy()] = 0
    return s


# --- Trade simulation --------------------------------------------------------------------------------

def size_for(entry: float, stop: float, rules: safety.Rules, cost: costs.CostAssumptions = INTRADAY) -> int:
    """Shares so that the loss at the stop, charges included, stays within rules.risk_per_trade."""
    risk = abs(entry - stop)
    if entry <= 0 or risk <= 0:
        return 0
    by_cap = math.floor(rules.capital * rules.max_position_pct / 100 / entry)
    q = min(by_cap, math.floor(rules.risk_per_trade / risk))
    while q > 0:
        v = q * entry
        loss = q * risk + costs.round_trip(v, cost, exit_value=q * stop)["total"]
        if loss <= rules.risk_per_trade + 1e-9:
            break
        q = max(q - max(1, q // 50), 0)
    return int(q)


def simulate(b: pd.DataFrame, sig: np.ndarray, p: Params, rules: safety.Rules,
             cost: costs.CostAssumptions = INTRADAY, symbol: str = "") -> pd.DataFrame:
    """Trades for one stock's bars (with indicators) and a signal array. One row per trade."""
    o, h, l, c = (b[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    atr, tmin = b["atr"].to_numpy(float), b["tmin"].to_numpy()
    days, times = b["day"].to_numpy(), b["datetime"].to_numpy()
    n, i, rows = len(b), 0, []
    per_day: dict = {}
    while i < n - 1:
        side = sig[i]
        if side == 0 or days[i + 1] != days[i] or per_day.get(days[i], 0) >= p.max_trades:
            i += 1
            continue
        e_i = i + 1
        entry = o[e_i]
        risk = p.stop_atr * atr[i]
        if not (risk > 0) or tmin[e_i] >= SQUARE_OFF:
            i += 1
            continue
        stop = entry - side * risk
        target = entry + side * p.rr * risk
        qty = size_for(entry, stop, rules, cost)
        if qty <= 0:
            i += 1
            continue
        exit_px, how, x_i = None, "time", e_i
        for j in range(e_i, n):
            if days[j] != days[e_i]:
                x_i = j - 1
                exit_px, how = c[x_i], "square-off"
                break
            if j > e_i and side * (o[j] - stop) <= 0:  # opened beyond the stop
                exit_px, how, x_i = o[j], "stop (gap)", j
                break
            if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
                exit_px, how, x_i = stop, "stop", j
                break
            if (side > 0 and h[j] >= target) or (side < 0 and l[j] <= target):
                exit_px, how, x_i = target, "target", j
                break
            if tmin[j] >= SQUARE_OFF:
                exit_px, how, x_i = c[j], "square-off", j
                break
            if j - e_i + 1 >= p.max_hold:
                exit_px, how, x_i = c[j], "time", j
                break
        if exit_px is None:
            x_i = n - 1
            exit_px, how = c[x_i], "end of data"
        gross = side * (exit_px - entry) * qty
        buy_v, sell_v = (entry * qty, exit_px * qty) if side > 0 else (exit_px * qty, entry * qty)
        rt = costs.round_trip(buy_v, cost, exit_value=sell_v)
        rows.append({"symbol": symbol, "day": days[i], "signal_time": times[i], "entry_time": times[e_i],
                     "exit_time": times[x_i], "side": "BUY" if side > 0 else "SELL", "entry": entry, "stop": stop,
                     "target": target, "exit": exit_px, "exit_by": how, "qty": qty, "value": entry * qty,
                     "gross": gross, "charges": rt["total"], "net": gross - rt["total"],
                     "gross_pct": gross / (entry * qty) * 100, "net_pct": (gross - rt["total"]) / (entry * qty) * 100})
        per_day[days[i]] = per_day.get(days[i], 0) + 1
        i = x_i + 1  # one open trade at a time
    cols = ["symbol", "day", "signal_time", "entry_time", "exit_time", "side", "entry", "stop", "target", "exit",
            "exit_by", "qty", "value", "gross", "charges", "net", "gross_pct", "net_pct"]
    return pd.DataFrame(rows, columns=cols)


def run(bars: dict[str, pd.DataFrame], p: Params, rules: safety.Rules,
        cost: costs.CostAssumptions = INTRADAY) -> pd.DataFrame:
    """Trades for every stock with one set of settings. bars: symbol -> add_indicators() output."""
    parts = [simulate(b, signals(b, p, seed=7 + k), p, rules, cost, sym)
             for k, (sym, b) in enumerate(sorted(bars.items())) if len(b)]
    parts = [x for x in parts if len(x)]
    return pd.concat(parts, ignore_index=True) if parts else simulate(pd.DataFrame(
        columns=["open", "high", "low", "close", "atr", "tmin", "day", "datetime"]), np.zeros(0, int), p, rules)


# --- Walk-forward -----------------------------------------------------------------------------------

@dataclass
class WalkForward:
    strategy: str
    trades: pd.DataFrame                 # out-of-sample trades only
    chosen: pd.DataFrame                 # per test day: the settings picked and their training score
    test_days: int
    train_days: int
    in_sample_best: dict = field(default_factory=dict)  # best settings over ALL days (for contrast only)


def walk_forward(bars: dict[str, pd.DataFrame], strategy: str, rules: safety.Rules, train_days: int = 10,
                 cost: costs.CostAssumptions = INTRADAY) -> WalkForward:
    """Pick settings on the previous `train_days` days, trade the next day, step forward one day."""
    combos = grid(strategy)
    all_days = sorted({d for b in bars.values() for d in b["day"].unique()})
    by_combo = {pp: run(bars, pp, rules, cost) for pp in combos}
    daily = {pp: (t.groupby("day")["net"].sum() if len(t) else pd.Series(dtype=float)) for pp, t in by_combo.items()}
    picked, oos = [], []
    for k in range(train_days, len(all_days)):
        train, test = all_days[k - train_days:k], all_days[k]
        score = {pp: float(daily[pp].reindex(train).fillna(0).sum()) for pp in combos}
        best = max(combos, key=lambda pp: (score[pp], -combos.index(pp)))
        t = by_combo[best]
        oos.append(t[t["day"] == test])
        picked.append({"day": test, "settings": best.label(), "train_net": score[best]})
    trades = pd.concat([x for x in oos if len(x)], ignore_index=True) if any(len(x) for x in oos) \
        else by_combo[combos[0]].iloc[0:0]
    totals = {pp: float(daily[pp].sum()) for pp in combos}
    ib = max(combos, key=lambda pp: totals[pp])
    return WalkForward(strategy, trades, pd.DataFrame(picked), max(len(all_days) - train_days, 0), train_days,
                       {"settings": ib.label(), "net": totals[ib], "trades": len(by_combo[ib])})


def summary(trades: pd.DataFrame, seed: int = 11) -> dict:
    """Plain numbers plus a bootstrap 90% range for the average net % per trade."""
    n = len(trades)
    if n == 0:
        return {"trades": 0, "verdict": "No trades", "edge": False}
    net, gross = trades["net"].to_numpy(float), trades["gross"].to_numpy(float)
    pct = trades["net_pct"].to_numpy(float)
    eq = np.cumsum(net)
    dd = float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min())
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    rng = np.random.default_rng(seed)
    boots = rng.choice(pct, size=(2000, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [5, 95])
    edge = n >= MIN_TRADES_FOR_VERDICT and lo > 0
    if n < MIN_TRADES_FOR_VERDICT:
        verdict = f"Too few trades to judge (need {MIN_TRADES_FOR_VERDICT})"
    elif edge:
        verdict = "Profitable after charges in this test (still not a promise)"
    elif pct.mean() > 0:
        verdict = "Slightly positive, but within luck: no proven edge"
    else:
        verdict = "Lost money after charges: no edge"
    return {"trades": n, "longs": int((trades["side"] == "BUY").sum()), "shorts": int((trades["side"] == "SELL").sum()),
            "win_rate_pct": float((net > 0).mean() * 100), "net": float(net.sum()), "gross": float(gross.sum()),
            "charges": float(trades["charges"].sum()), "avg_net_pct": float(pct.mean()),
            "avg_gross_pct": float(trades["gross_pct"].mean()),
            "avg_cost_pct": float((trades["charges"] / trades["value"] * 100).mean()),
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            "max_drawdown": dd, "ci_low_pct": float(lo), "ci_high_pct": float(hi),
            "days": int(trades["day"].nunique()), "edge": bool(edge), "verdict": verdict}


def scoreboard(bars: dict[str, pd.DataFrame], rules: safety.Rules, train_days: int = 10,
               strategies: list[str] | None = None, cost: costs.CostAssumptions = INTRADAY) -> tuple[pd.DataFrame, dict]:
    """Walk-forward result for each strategy, best first. Returns (table, {strategy: WalkForward})."""
    runs, rows = {}, []
    for s in strategies or list(STRATEGIES):
        wf = walk_forward(bars, s, rules, train_days, cost)
        runs[s] = wf
        sm = summary(wf.trades)
        rows.append({"strategy": s, "Setup": STRATEGIES[s], "Test days": wf.test_days, "Trades": sm["trades"],
                     "Buys": sm.get("longs", 0), "Short sells": sm.get("shorts", 0),
                     "Win rate %": sm.get("win_rate_pct", np.nan), "Avg before costs %": sm.get("avg_gross_pct", np.nan),
                     "Avg costs %": sm.get("avg_cost_pct", np.nan), "Avg after costs %": sm.get("avg_net_pct", np.nan),
                     "Net ₹ after costs": sm.get("net", 0.0), "Worst drop ₹": sm.get("max_drawdown", 0.0),
                     "Verdict": sm["verdict"], "edge": sm.get("edge", False)})
    t = pd.DataFrame(rows).sort_values("Net ₹ after costs", ascending=False).reset_index(drop=True)
    return t, runs


def cost_only_needed(price: float, rules: safety.Rules, cost: costs.CostAssumptions = INTRADAY) -> float:
    """The move (in %) a trade must make just to pay its round trip at the usual size for this price."""
    qty = max(1, math.floor(rules.capital * rules.max_position_pct / 100 / price))
    return costs.round_trip(qty * price, cost)["total_pct"] * 100


# --- Live signal --------------------------------------------------------------------------------------

def live_setups(b: pd.DataFrame, p: Params, rules: safety.Rules, symbol: str, lookback_bars: int = 3,
                cost: costs.CostAssumptions = INTRADAY) -> list[dict]:
    """Setups that fired on the last `lookback_bars` closed bars of the latest day, with a plan each.

    The plan's entry is the last close as a reference; in paper trading the fill is the next bar's open.
    """
    if b.empty:
        return []
    last_day = b["day"].iloc[-1]
    today = b[b["day"] == last_day]
    sig = signals(b, p)[-len(today):]
    out = []
    for k in range(max(0, len(today) - lookback_bars), len(today)):
        if sig[k] == 0:
            continue
        r = today.iloc[k]
        side = int(sig[k])
        entry = float(r["close"])
        risk = p.stop_atr * float(r["atr"])
        stop, target = entry - side * risk, entry + side * p.rr * risk
        qty = size_for(entry, stop, rules, cost)
        rt = costs.round_trip(qty * entry, cost) if qty else {"total": 0.0}
        out.append({"symbol": symbol, "time": r["datetime"], "side": "BUY" if side > 0 else "SELL",
                    "strategy": p.strategy, "settings": p.label(), "ref_entry": entry, "stop": stop, "target": target,
                    "qty": qty, "value": qty * entry, "charges": rt["total"],
                    "loss_at_stop": qty * risk + rt["total"], "profit_at_target": qty * p.rr * risk - rt["total"],
                    "move_to_breakeven_pct": rt["total"] / (qty * entry) * 100 if qty else np.nan})
    return out


def best_recent(bars: dict[str, pd.DataFrame], strategy: str, rules: safety.Rules, train_days: int = 10,
                cost: costs.CostAssumptions = INTRADAY) -> Params:
    """The settings the walk-forward would use for the NEXT day (best over the last `train_days` days)."""
    combos = grid(strategy)
    all_days = sorted({d for b in bars.values() for d in b["day"].unique()})
    recent = set(all_days[-train_days:])
    score = {}
    for pp in combos:
        t = run(bars, pp, rules, cost)
        score[pp] = float(t.loc[t["day"].isin(recent), "net"].sum()) if len(t) else 0.0
    return max(combos, key=lambda pp: (score[pp], -combos.index(pp)))


# --- Scalp paper trading -----------------------------------------------------------------------------

SCALP_COLS = ["id", "placed_at", "symbol", "side", "strategy", "qty", "stop", "target", "status", "entry_time",
              "entry", "exit_time", "exit", "exit_by", "gross", "charges", "net", "planned_risk"]


def empty_scalp_journal() -> pd.DataFrame:
    return pd.DataFrame(columns=SCALP_COLS)


def can_place_scalp(journal: pd.DataFrame, rules: safety.Rules, day, symbol: str, qty: int, entry: float,
                    stop: float, cost: costs.CostAssumptions = INTRADAY) -> tuple[bool, str]:
    """The same limits as the daily paper trades: trades per day, position cap, risk per trade, daily loss."""
    if qty <= 0:
        return False, "Quantity must be at least one share."
    j = journal if len(journal) else empty_scalp_journal()
    today = j[pd.to_datetime(j["placed_at"]).dt.date == day] if len(j) else j
    if len(today) >= rules.max_trades_per_day:
        return False, (f"You already have {len(today)} practice scalps today, your daily maximum. Most scalpers "
                       "who lose do it by trading too often; costs are paid on every trade.")
    if len(today) and ((today["symbol"] == symbol) & (today["status"] == "open")).any():
        return False, f"You already have an open practice scalp in {symbol}."
    value = qty * entry
    if value > rules.capital * rules.max_position_pct / 100 + 1e-6:
        return False, f"₹{value:,.0f} is over your {rules.max_position_pct:.0f}% of capital limit per trade. No leverage."
    risk = qty * abs(entry - stop) + costs.round_trip(value, cost, exit_value=qty * stop)["total"]
    if risk > rules.risk_per_trade + 1:
        return False, f"At the stop you would lose about ₹{risk:,.0f}, over your ₹{rules.risk_per_trade:,.0f} limit."
    lost = float(pd.to_numeric(today.loc[today["status"] == "closed", "net"], errors="coerce").clip(upper=0).sum()) \
        if len(today) else 0.0
    at_risk = float(pd.to_numeric(today.loc[today["status"] == "open", "planned_risk"], errors="coerce").sum()) \
        if len(today) else 0.0
    if -lost + at_risk + risk > rules.daily_loss_limit + 1:
        return False, (f"Today's losses (₹{-lost:,.0f}) plus open risk could pass your daily loss limit of "
                       f"₹{rules.daily_loss_limit:,.0f}. Stop for today: that is the limit working.")
    return True, "Within your limits."


def place_scalp(journal: pd.DataFrame, now, setup: dict, cost: costs.CostAssumptions = INTRADAY) -> pd.DataFrame:
    """Add a practice scalp. It fills at the first bar that OPENS after `now` (never at a past price)."""
    q, e, s = int(setup["qty"]), float(setup["ref_entry"]), float(setup["stop"])
    row = {"id": int(pd.to_numeric(journal["id"]).max()) + 1 if len(journal) else 1,
           "placed_at": pd.Timestamp(now).strftime("%Y-%m-%d %H:%M:%S"), "symbol": setup["symbol"],
           "side": setup["side"], "strategy": setup.get("strategy", ""), "qty": q, "stop": s,
           "target": float(setup["target"]), "status": "open", "entry_time": None, "entry": np.nan,
           "exit_time": None, "exit": np.nan, "exit_by": "", "gross": np.nan, "charges": np.nan, "net": np.nan,
           "planned_risk": q * abs(e - s) + costs.round_trip(q * e, cost, exit_value=q * s)["total"]}
    return pd.concat([journal, pd.DataFrame([row])], ignore_index=True) if len(journal) else pd.DataFrame([row])


def settle_scalps(journal: pd.DataFrame, minutes: dict[str, pd.DataFrame], max_hold_min: int = 60,
                  cost: costs.CostAssumptions = INTRADAY) -> pd.DataFrame:
    """Fill and close open practice scalps from real 1-minute candles that printed after they were placed.

    Entry: open of the first minute starting after the order. Exit: stop (or the open if a minute opened
    beyond it), target, `max_hold_min` minutes, or 15:15. A scalp stays open until such a candle exists.
    """
    if journal is None or journal.empty:
        return empty_scalp_journal()
    j = journal.copy()
    for k in ("entry", "exit", "gross", "charges", "net", "stop", "target"):
        j[k] = pd.to_numeric(j[k], errors="coerce").astype(float)
    for col in ("entry_time", "exit_time", "exit_by"):
        j[col] = j[col].astype(object)
    for i, r in j[j["status"] == "open"].iterrows():
        m = minutes.get(r["symbol"])
        if m is None or m.empty:
            continue
        placed = pd.Timestamp(r["placed_at"])
        m = m[pd.to_datetime(m["datetime"]).dt.date == placed.date()].sort_values("datetime")
        m = m[pd.to_datetime(m["datetime"]) > placed]
        if m.empty:
            continue
        side = 1 if r["side"] == "BUY" else -1
        stop, target, qty = float(r["stop"]), float(r["target"]), int(r["qty"])
        e = m.iloc[0]
        entry, e_t = float(e["open"]), pd.Timestamp(e["datetime"])
        exit_px = exit_t = None
        how = ""
        for k, (_, bar) in enumerate(m.iterrows()):
            t = pd.Timestamp(bar["datetime"])
            tm = t.hour * 60 + t.minute
            if k > 0 and side * (float(bar["open"]) - stop) <= 0:
                exit_px, how = float(bar["open"]), "stop (gap)"
            elif (side > 0 and bar["low"] <= stop) or (side < 0 and bar["high"] >= stop):
                exit_px, how = stop, "stop"
            elif (side > 0 and bar["high"] >= target) or (side < 0 and bar["low"] <= target):
                exit_px, how = target, "target"
            elif tm >= SQUARE_OFF:
                exit_px, how = float(bar["close"]), "square-off 15:15"
            elif (t - e_t).total_seconds() / 60 >= max_hold_min - 1:
                exit_px, how = float(bar["close"]), f"time ({max_hold_min} min)"
            if exit_px is not None:
                exit_t = t
                break
        j.at[i, "entry"], j.at[i, "entry_time"] = round(entry, 2), e_t.strftime("%Y-%m-%d %H:%M")
        if exit_px is None:
            continue
        gross = side * (exit_px - entry) * qty
        buy_v, sell_v = (entry * qty, exit_px * qty) if side > 0 else (exit_px * qty, entry * qty)
        ch = costs.round_trip(buy_v, cost, exit_value=sell_v)["total"]
        for k, v in {"status": "closed", "exit": round(exit_px, 2), "exit_time": exit_t.strftime("%Y-%m-%d %H:%M"),
                     "exit_by": how, "gross": round(gross, 2), "charges": round(ch, 2),
                     "net": round(gross - ch, 2)}.items():
            j.at[i, k] = v
    return j


def scalp_paper_summary(journal: pd.DataFrame, start: float = safety.PAPER_START) -> dict:
    done = journal[journal["status"] == "closed"] if len(journal) else journal
    net = pd.to_numeric(done["net"], errors="coerce").fillna(0) if len(done) else pd.Series(dtype=float)
    n = len(done)
    total = float(net.sum()) if n else 0.0
    return {"start": start, "balance": start + total, "closed": n,
            "open": int((journal["status"] == "open").sum()) if len(journal) else 0,
            "wins": int((net > 0).sum()) if n else 0, "net": total,
            "charges": float(pd.to_numeric(done["charges"], errors="coerce").fillna(0).sum()) if n else 0.0,
            "ready": n >= safety.PAPER_TRADES_TO_GRADUATE and total > 0}


def gate(record_edge: bool, paper: dict, trades_today: int, rules: safety.Rules) -> tuple[str, str]:
    """What the page lets a live setup be used for: ('stop'|'paper'|'go', reason)."""
    if trades_today >= rules.max_trades_per_day:
        return "stop", "You have used today's trades. Stopping is part of the plan."
    if not record_edge:
        return "paper", ("This setup has no proven edge after charges in the walk-forward test. Practise it on "
                         "paper only; do not put real money on it.")
    if not paper.get("ready"):
        return "paper", (f"The test passes, but your own practice record is not yet profitable over "
                         f"{safety.PAPER_TRADES_TO_GRADUATE}+ scalps. Keep practising on paper.")
    return "go", "The test and your practice record both pass. If you trade, use the size and stop shown. Not a promise."
