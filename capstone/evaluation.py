"""Accuracy tables for the dashboard: walk-forward summary and past forecasts vs actual opens."""
from __future__ import annotations

import pandas as pd

MODEL = "Model with overnight US + rupee"
WITHOUT_OVERNIGHT = "Same model without overnight data"
METHOD_NAMES = {"model": MODEL,
                "model_without_overnight": WITHOUT_OVERNIGHT,
                "baseline_open_eq_close": "Baseline: open = last close",
                "baseline_mean_gap": "Baseline: average gap"}
METRIC_NAMES = {"mae_rupees": "MAE (₹)", "mape_pct": "MAPE %", "direction_acc_pct": "Direction accuracy %"}


def accuracy_summary(report: pd.DataFrame) -> pd.DataFrame:
    """Average each method's error metrics over the walk-forward folds, with readable names."""
    return (report.groupby("method")[list(METRIC_NAMES)].mean().round(3)
            .rename(columns=METRIC_NAMES).rename(index=METHOD_NAMES))


def overnight_gain(summary: pd.DataFrame) -> float | None:
    """Rupees per share the overnight features cut from the average error (negative = they hurt)."""
    if WITHOUT_OVERNIGHT not in summary.index:
        return None
    return summary.loc[WITHOUT_OVERNIGHT, "MAE (₹)"] - summary.loc[MODEL, "MAE (₹)"]


def predictions_vs_actual(log: pd.DataFrame, hist: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Each logged forecast for one stock next to the open that actually followed it.

    log: predictions_log.csv rows (symbol, as_of_close, pred_open, ...).
    hist: that stock's daily prices (date, open, ...). The open on day t answers the
    forecast made from the close on day t-1.
    """
    log = log[log["symbol"] == symbol].drop_duplicates("as_of_close", keep="last")
    hist = hist.assign(as_of_close=hist["date"].shift(1).dt.strftime("%Y-%m-%d"))
    joined = log.merge(hist[["as_of_close", "open"]].rename(columns={"open": "actual_open"}),
                       on="as_of_close", how="inner")
    joined["as_of_close"] = pd.to_datetime(joined["as_of_close"])
    return joined
