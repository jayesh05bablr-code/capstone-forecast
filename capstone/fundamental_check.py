"""Compare the short-term forecast with each company's fundamental trend, and flag disagreements.

This is a consistency check for the analyst, not a prediction: company fundamentals do not
directly predict the next day's opening gap (they move slowly, the gap reflects overnight news).
A disagreement is a prompt to investigate, e.g. "the model expects a gap up, but profit growth
has turned negative".

Fundamental trend, from quarterly results (period_end, revenue_cr, net_profit_cr):
- Point in time: a quarter counts only once its results would have been published, taken as
  RESULTS_LAG_DAYS after the quarter end (Indian listed companies must publish within 45 days).
- Revenue and net-profit growth are year on year (the same quarter a year earlier), which removes
  seasonality.
- Trend label from the average of the two growth rates: "Strong growth" above +10%,
  "Growing" above 0, "Declining" at or below 0. "Accelerating"/"Slowing" compares the latest
  average growth with the previous quarter's.

Sources, in order: uploaded results (data/financials.csv), then the bundled company_data/*.xlsx
"financials" sheets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import uploads
from .paths import ROOT

RESULTS_LAG_DAYS = 45
COMPANY_DATA_DIR = ROOT / "company_data"


def load_financials() -> pd.DataFrame:
    fin = uploads.load("financials")
    frames = [fin] if len(fin) else []
    for path in sorted(COMPANY_DATA_DIR.glob("*_company_data.xlsx")):
        try:
            frames.append(pd.read_excel(path, sheet_name="financials"))
        except (ValueError, KeyError):
            continue
    if not frames:
        return pd.DataFrame(columns=["symbol", "period_end", "revenue_cr", "net_profit_cr"])
    df = pd.concat(frames, ignore_index=True)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["period_end"] = pd.to_datetime(df["period_end"])
    for c in ("revenue_cr", "net_profit_cr"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df.drop_duplicates(["symbol", "period_end"], keep="first").sort_values(["symbol", "period_end"])


def growth_table(fin: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year growth per company and quarter, with the date the results became public."""
    out = []
    for sym, g in fin.groupby("symbol"):
        g = g.set_index("period_end").sort_index()
        for col, name in (("revenue_cr", "revenue_yoy"), ("net_profit_cr", "profit_yoy")):
            prior = g[col].reindex(g.index - pd.DateOffset(years=1)).to_numpy()
            # Year-on-year needs a positive base; a loss-making base quarter gives no growth figure.
            g[name] = np.where(prior > 0, g[col].to_numpy() / prior - 1, np.nan)
        g["avg_growth"] = g[["revenue_yoy", "profit_yoy"]].mean(axis=1)
        g["published"] = g.index + pd.Timedelta(days=RESULTS_LAG_DAYS)
        out.append(g.reset_index().assign(symbol=sym))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def trend_label(growth: float) -> str:
    if pd.isna(growth):
        return "Unknown"
    return "Strong growth" if growth > 0.10 else "Growing" if growth > 0 else "Declining"


def latest_trend(growth: pd.DataFrame, symbol: str, as_of: pd.Timestamp) -> dict:
    """The trend known on `as_of` for one company (only quarters published by then)."""
    g = growth[(growth["symbol"] == symbol) & (growth["published"] <= as_of)].dropna(subset=["avg_growth"])
    if g.empty:
        return {"symbol": symbol, "trend": "Unknown", "quarter": None, "revenue_yoy_pct": np.nan,
                "profit_yoy_pct": np.nan, "momentum": "Unknown"}
    last = g.iloc[-1]
    prev = g.iloc[-2] if len(g) > 1 else None
    momentum = ("Unknown" if prev is None else "Accelerating" if last["avg_growth"] > prev["avg_growth"]
                else "Slowing")
    return {"symbol": symbol, "trend": trend_label(last["avg_growth"]), "quarter": f"{last['period_end']:%b %Y}",
            "revenue_yoy_pct": last["revenue_yoy"] * 100, "profit_yoy_pct": last["profit_yoy"] * 100,
            "momentum": momentum}


def compare(preds: pd.DataFrame, as_of: pd.Timestamp | None = None, flat_band_pct: float = 0.05) -> pd.DataFrame:
    """Forecast direction next to the fundamental trend, with a flag where they disagree.

    preds: predictions with symbol and pred_gap_pct. Gaps within +/- flat_band_pct count as flat.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    growth = growth_table(load_financials())
    rows = []
    for r in preds.itertuples():
        t = latest_trend(growth, r.symbol, as_of) if len(growth) else {"symbol": r.symbol, "trend": "Unknown"}
        direction = "Up" if r.pred_gap_pct > flat_band_pct else "Down" if r.pred_gap_pct < -flat_band_pct else "Flat"
        if t["trend"] == "Unknown" or direction == "Flat":
            flag = "No comparison"
        elif direction == "Up" and t["trend"] == "Declining":
            flag = "Disagree: gap-up forecast but fundamentals are deteriorating. Investigate."
        elif direction == "Down" and t["trend"] == "Strong growth":
            flag = "Disagree: gap-down forecast despite strong growth. Check for news."
        else:
            flag = "Consistent"
        rows.append({**t, "forecast_gap_pct": r.pred_gap_pct, "forecast_direction": direction, "assessment": flag})
    return pd.DataFrame(rows)
