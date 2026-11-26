"""Plain-language guidance for the Today page: confidence, risk, costs and what to do.

Everything here is pure pandas (no Streamlit), so it is tested in tests/test_guidance.py.

Two facts shape every message on the page:

1. The forecast is for the overnight gap, from yesterday's close to today's open. Nobody can buy at
   yesterday's close after seeing last night's US session, so the gap itself is not profit anyone
   can collect. The forecast helps plan around the open (a sensible limit price, not chasing a gap,
   not panic-selling one); it is not a buy or sell signal.
2. Confidence comes from the model's own out-of-sample record, not from its forecast size:
   "when the model expected a move this big, how often was it right about the direction?"
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from . import costs

MODEL = "forecast"  # model_eval.FORECAST: the model the app forecasts with
# Buckets of |forecast gap| in % points. Bigger forecasts are rarer but usually more reliable.
BUCKETS = [0.0, 0.10, 0.25, 0.50, 1.00, np.inf]
BUCKET_LABELS = ["under 0.10%", "0.10-0.25%", "0.25-0.50%", "0.50-1.00%", "over 1.00%"]
MIN_CASES = 30  # fewer past cases than this and the hit rate is too noisy to quote
REFERENCE_POSITION = 100_000.0  # cost % is quoted for a ₹1 lakh order


def round_trip_cost_pct(instrument: str = "equity_intraday", position: float = REFERENCE_POSITION) -> float:
    """Round-trip cost in % points with the project's default (assumed) broker charges."""
    return costs.round_trip_pct(position, costs.CostAssumptions(instrument=instrument)) * 100


def confidence_table(oos: pd.DataFrame | None, model: str = MODEL) -> pd.DataFrame:
    """Direction hit rate of past out-of-sample forecasts, by forecast size.

    oos: models/oos_predictions.csv (columns model, pred, actual as fractions). Days with a zero
    actual gap are left out (there was no direction to get right).
    """
    cols = ["bucket", "cases", "hit_rate_pct", "avg_abs_actual_pct"]
    if oos is None or oos.empty:
        return pd.DataFrame(columns=cols)
    d = oos[(oos["model"] == model) & (oos["actual"] != 0)].dropna(subset=["pred", "actual"])
    size = d["pred"].abs() * 100
    d = d.assign(bucket=pd.cut(size, BUCKETS, labels=BUCKET_LABELS, right=False),
                 hit=np.sign(d["pred"]) == np.sign(d["actual"]))
    t = d.groupby("bucket", observed=False).agg(cases=("hit", "size"), hit_rate_pct=("hit", "mean"),
                                                avg_abs_actual_pct=("actual", lambda s: s.abs().mean()))
    t["hit_rate_pct"] *= 100
    t["avg_abs_actual_pct"] *= 100
    return t.reset_index()[cols]


@dataclass
class Confidence:
    level: str          # "High", "Medium", "Low" or "Unknown"
    hit_rate_pct: float  # NaN when unknown
    cases: int
    bucket: str
    approximate: bool = False  # True when the rate is from ALL past forecasts, not ones of this size
    bucket_cases: int = 0      # past forecasts of this size (shown when approximate)

    @property
    def label(self) -> str:
        """Short label for a card, e.g. "Low (approximate)"."""
        return f"{self.level} (approximate)" if self.approximate else self.level

    @property
    def sentence(self) -> str:
        if self.level == "Unknown":
            return "Not enough past forecasts yet to say how reliable it is."
        right = round(self.hit_rate_pct)
        if self.approximate:
            return (f"Approximate: only {self.bucket_cases} past forecasts were of this size ({self.bucket}), too few "
                    f"to score on their own. Across all {self.cases:,} past forecasts it had not seen in training, "
                    f"the model got the direction right {right} times out of 100.")
        return (f"When the model expected a move of {self.bucket}, it got the direction right "
                f"{right} times out of 100 ({self.cases:,} past cases it had not seen in training).")


def _level(hit: float) -> str:
    return "High" if hit >= 65 else "Medium" if hit >= 57 else "Low"


def confidence_for(gap_pct: float, table: pd.DataFrame) -> Confidence:
    """Confidence for one forecast (gap in % points) from confidence_table().

    Uses the record of past forecasts of the same size. When there are fewer than MIN_CASES of those,
    it falls back to the record of all past forecasts and marks the result approximate. Only when
    even that is too thin is it "Unknown".
    """
    if table.empty or pd.isna(gap_pct):
        return Confidence("Unknown", float("nan"), 0, "")
    idx = int(np.searchsorted(BUCKETS, abs(gap_pct), side="right") - 1)
    idx = min(max(idx, 0), len(BUCKET_LABELS) - 1)
    row = table.iloc[idx]
    cases, hit = int(row["cases"]), float(row["hit_rate_pct"])
    if cases >= MIN_CASES and not np.isnan(hit):
        return Confidence(_level(hit), hit, cases, BUCKET_LABELS[idx])
    n = table["cases"].fillna(0).astype(int)
    total = int(n.sum())
    if total < MIN_CASES:
        return Confidence("Unknown", float("nan"), cases, BUCKET_LABELS[idx], bucket_cases=cases)
    pooled = float((table["hit_rate_pct"].fillna(0) * n).sum() / total)
    return Confidence(_level(pooled), pooled, total, BUCKET_LABELS[idx], approximate=True, bucket_cases=cases)


@dataclass
class Advice:
    verdict: str   # short label for the card
    tone: str      # "neutral", "up", "down", "caution"
    action: str    # one or two plain sentences


def advice(symbol: str, gap_pct: float, pred_open: float, low: float, high: float, conf: Confidence,
           cost_pct: float) -> Advice:
    """What a careful investor might do around today's open, in plain words. Never a promise."""
    if abs(gap_pct) < cost_pct:
        return Advice("No clear edge", "neutral",
                      f"The expected move ({gap_pct:+.2f}%) is smaller than what a buy-and-sell costs "
                      f"(about {cost_pct:.2f}%). Nothing to act on because of this forecast. If you already "
                      "hold it, there is no reason here to change anything.")
    if conf.level in ("Low", "Unknown") or conf.approximate:
        why = ("there are too few past forecasts this size to trust it" if conf.approximate and conf.level != "Low"
               else "its record on forecasts this size is weak")
        return Advice("Unclear direction", "caution",
                      f"The model leans {'up' if gap_pct > 0 else 'down'} ({gap_pct:+.2f}%), but {why}. "
                      "Treat it as noise and let the first 15 minutes of trading "
                      "show the direction.")
    if gap_pct > 0:
        return Advice("Likely higher open", "up",
                      f"Expect {symbol} to open near ₹{pred_open:,.2f}. If you plan to buy, avoid a market order "
                      f"at 09:15: set a limit price and do not chase above ₹{high:,.2f}. If you hold it, a higher "
                      "open can be a chance to book part of a profit you already have.")
    return Advice("Likely lower open", "down",
                  f"Expect {symbol} to open near ₹{pred_open:,.2f}. If you hold it, a gap down alone is not a "
                  f"reason to panic-sell at 09:15. If you plan to buy, a limit order near ₹{low:,.2f} may fill "
                  "cheaper than chasing.")


def range_crosses_close(gap_pct: float, low: float, high: float, pred_open: float) -> bool:
    """True when the likely range includes both a higher and a lower open than yesterday's close."""
    close = pred_open / (1 + gap_pct / 100)
    return low < close < high


def risk_on(amount: float, low: float, high: float, close: float) -> tuple[float, float]:
    """Rupee change at the open, for `amount` invested at yesterday's close, at the band's edges."""
    return amount * (low / close - 1), amount * (high / close - 1)


def market_mood(avg_gap_pct: float) -> tuple[str, str]:
    """Headline and one-line meaning of the average expected gap across the stocks."""
    a = avg_gap_pct
    if a >= 0.5:
        return "Strong start expected", "Most stocks are expected to open clearly higher than yesterday's close."
    if a >= 0.15:
        return "Mildly positive start expected", "Stocks lean towards opening a little higher than yesterday's close."
    if a > -0.15:
        return "Flat start expected", "No strong push either way: opens close to yesterday's prices are likely."
    if a > -0.5:
        return "Mildly negative start expected", "Stocks lean towards opening a little lower than yesterday's close."
    return "Weak start expected", "Most stocks are expected to open clearly lower than yesterday's close."


def freshness(next_open: str, overnight_complete: bool, now_ist: datetime) -> tuple[str, str]:
    """('ok' | 'preliminary' | 'old', message) for the forecast's target date."""
    from . import market_calendar

    target = pd.Timestamp(next_open).date()
    today = now_ist.date()
    closed = market_calendar.holiday_name(target)
    if closed:
        nxt = market_calendar.next_trading_day(target)
        return "old", (f"This forecast was made for {target:%a %d %b}, but NSE is closed that day ({closed}). "
                       f"A new forecast for the next trading day, {nxt:%a %d %b}, is being made; reload in a "
                       "few minutes or use Update now in the sidebar.")
    if target < today:
        return "old", (f"This forecast was for {target:%a %d %b}, which has passed. The app updates itself at "
                       "08:45 and 16:15 IST; use Update now in the sidebar if it has not.")
    closed_today = market_calendar.holiday_name(today)
    lead = (f"NSE is closed today ({today:%a %d %b}: {closed_today}). " if closed_today else "")
    if not overnight_complete:
        night = "the US session before that open" if closed_today else "last night's US session"
        return "preliminary", (f"{lead}Preliminary forecast for the open on {target:%a %d %b %Y}: {night} is "
                               "not in yet, so the biggest overnight driver is missing. The 08:45 IST update "
                               "adds it.")
    if closed_today:
        return "ok", f"{lead}This forecast is for the next trading day's open, {target:%a %d %b %Y}."
    return "ok", f"Forecast for the open on {target:%a %d %b %Y}, with last night's US and rupee moves included."


def live_track_record(log: pd.DataFrame | None, prices: dict[str, pd.DataFrame], last_n_days: int = 20) -> dict:
    """How the app's own logged forecasts did against the opens that followed (true live record).

    log: predictions_log.csv. prices: symbol -> daily prices (date, open). Returns cases, direction
    hits (skipping zero gaps), average absolute error in % points, and the date span.
    """
    if log is None or log.empty:
        return {"cases": 0}
    rows = []
    for sym, p in prices.items():
        if p is None or p.empty:
            continue
        lg = log[log["symbol"] == sym].drop_duplicates("as_of_close", keep="last")
        p = p.sort_values("date")
        nxt = pd.DataFrame({"as_of_close": p["date"].dt.strftime("%Y-%m-%d"), "actual_open": p["open"].shift(-1),
                            "close_actual": p["close"]})
        rows.append(lg.merge(nxt, on="as_of_close", how="inner").dropna(subset=["actual_open"]))
    if not rows:
        return {"cases": 0}
    j = pd.concat(rows, ignore_index=True)
    if j.empty:
        return {"cases": 0}
    days = sorted(j["as_of_close"].unique())[-last_n_days:]
    j = j[j["as_of_close"].isin(days)]
    actual = (j["actual_open"] / j["close"] - 1) * 100
    pred = j["pred_gap_pct"]
    moved = actual.round(6) != 0
    hits = int((np.sign(pred[moved]) == np.sign(actual[moved])).sum())
    return {"cases": int(moved.sum()), "hits": hits, "mae_pct": float((pred - actual).abs().mean()),
            "first": days[0], "last": days[-1], "days": len(days)}
