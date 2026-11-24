"""Predict the next NSE trading day's opening price for every symbol from its latest close.

The target day skips weekends and NSE holidays (see market_calendar).

    python -m capstone.predict

Best run in the morning after the US close (about 02:30 IST) and after
fetch_data and fetch_overnight, so the overnight US and rupee moves are in.
Writes data/processed/predictions.csv (with reasons), overnight_story.json,
and appends to predictions_log.csv.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from . import explain, features, market_calendar, overnight
from .paths import FX_DIR, MODELS, PROCESSED

LOG_COLUMNS = ["symbol", "as_of_close", "close", "pred_gap_pct", "pred_open", "pred_open_low", "pred_open_high"]
# Added for the accuracy dashboard and Supabase; older logs without them are upgraded in place.
EXTRA_LOG_COLUMNS = ["target_date", "q10_gap_pct", "q90_gap_pct", "model_name", "predicted_at_utc"]
MODEL_NAME = "hist_gradient_boosting"


def append_log(out: pd.DataFrame, log) -> None:
    """Append today's rows to the prediction log, rewriting it once if its header is older."""
    cols = LOG_COLUMNS + EXTRA_LOG_COLUMNS
    if log.exists():
        old = pd.read_csv(log)
        if list(old.columns) != cols:
            pd.concat([old, out[cols]], ignore_index=True).reindex(columns=cols).to_csv(log, index=False)
            return
    out[cols].to_csv(log, mode="a", header=not log.exists(), index=False)


def save_to_supabase(out: pd.DataFrame) -> str:
    """Best effort: daily prediction log and stock mapping to Supabase when it is configured."""
    from . import db
    from .env import redact

    try:
        store = db.get_store()
        if store is None:
            return "Supabase not configured: prediction logged locally only."
        store.upsert("stocks", db.stock_rows(overnight.stock_map()))
        n = store.upsert("daily_predictions", db.prediction_rows(out, MODEL_NAME))
        return f"Saved {n} predictions to Supabase."
    except Exception as exc:  # persistence must never break the forecast
        return redact(f"Supabase save failed (local log is fine): {exc}")


def _live_fx() -> float | None:
    path = FX_DIR / "USDINR_live.json"
    return json.loads(path.read_text())["rate"] if path.exists() else None


def main() -> None:
    print(market_calendar.update_from_nse())  # best effort: NSE's latest holiday list
    bundle = joblib.load(MODELS / "gap_model.joblib")
    cols = bundle["features"]
    df = features.build(save=False)
    latest = df.sort_values("date").groupby("symbol").tail(1).copy()
    gap = bundle["model"].predict(latest[cols])
    latest["pred_gap_pct"] = gap * 100
    latest["pred_open"] = latest["close"] * (1 + gap)
    latest["pred_open_low"] = latest["close"] * (1 + gap + bundle["band"]["q10"])
    latest["pred_open_high"] = latest["close"] * (1 + gap + bundle["band"]["q90"])
    contrib = explain.contributions(bundle["model"], latest, cols)
    for name in contrib.columns:
        latest[f"contrib_{name}"] = contrib[name]

    as_of = latest["date"].max()
    # The open being forecast: the next day NSE trades (skips weekends and NSE holidays).
    end = pd.Timestamp(market_calendar.next_trading_day(as_of))
    us_last = overnight.latest_session()
    us_needed = pd.Timestamp(market_calendar.last_us_session_before(end))
    complete = us_last is not None and us_last >= us_needed
    if not complete:
        print(f"Note: the US session of {us_needed:%Y-%m-%d} is not in data/us yet. Run fetch_overnight after "
              "the US close (about 02:30 IST) for the full overnight signal.")
    smap = overnight.stock_map().set_index("symbol")
    for c in ["us_etf", "us_sector", "adr", "fx_exposure"]:
        latest[c] = latest["symbol"].map(smap[c])
    latest["india_sector"] = latest["india_sector"].fillna("Unmapped")

    out = latest.rename(columns={"date": "as_of_close"})
    out["reason"] = out.apply(explain.stock_reason, axis=1)
    out["as_of_close"] = out["as_of_close"].dt.strftime("%Y-%m-%d")
    keep = LOG_COLUMNS + ["india_sector", "us_etf", "us_sector", "adr", "fx_exposure",
                          *[c for c in features.OVERNIGHT_FEATURES if c in out.columns],
                          *[f"contrib_{n}" for n in contrib.columns], "reason"]
    out["target_date"] = f"{end:%Y-%m-%d}"
    out["q10_gap_pct"] = (gap + bundle["band"]["q10"]) * 100
    out["q90_gap_pct"] = (gap + bundle["band"]["q90"]) * 100
    out["model_name"] = MODEL_NAME
    out["predicted_at_utc"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")
    keep += EXTRA_LOG_COLUMNS
    out = out[keep]
    out[LOG_COLUMNS[2:]] = out[LOG_COLUMNS[2:]].round(2)
    out[["q10_gap_pct", "q90_gap_pct"]] = out[["q10_gap_pct", "q90_gap_pct"]].round(3)

    moves = overnight.sector_moves(as_of, end)
    fx_chg = float(latest["usdinr_chg"].iloc[0]) if latest["usdinr_chg"].notna().any() else float("nan")
    fx_level = _live_fx()
    if fx_level is None and not overnight.load_fx().empty:
        fx_level = float(overnight.load_fx().iloc[-1])
    links = pd.read_csv(overnight.SECTOR_LINKS_FILE)
    story = {
        "as_of_close": f"{as_of:%Y-%m-%d}",
        "next_open": f"{end:%Y-%m-%d}",
        "us_session": None if us_last is None else f"{us_last:%a %d %b %Y}",
        "overnight_complete": bool(complete),
        "usdinr": fx_level,
        "usdinr_chg": fx_chg,
        "mood": explain.mood(out["pred_gap_pct"].mean() / 100),
        "lines": explain.market_story(moves, fx_chg, fx_level, out, links,
                                      None if us_last is None else f"{us_last:%a %d %b}"),
        "moves": moves.to_dict(orient="records"),
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED / "predictions.csv", index=False)
    (PROCESSED / "overnight_story.json").write_text(json.dumps(story, indent=2, default=float))
    append_log(out, PROCESSED / "predictions_log.csv")
    print(save_to_supabase(out))
    print(f"Expected opening: {story['mood']}")
    print("\n".join(f"- {line}" for line in story["lines"]))
    print()
    print(out[LOG_COLUMNS].to_string(index=False))


if __name__ == "__main__":
    main()
