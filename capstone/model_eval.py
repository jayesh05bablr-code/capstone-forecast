"""Compare the opening-gap forecasters on identical walk-forward test dates.

    python -m capstone.model_eval        (also run automatically by python -m capstone.train)

Models (all predict target_gap = next open / close - 1 for each stock and day):

A. naive               yesterday's close is today's open, so the predicted gap is 0.
B. historical_average  each stock's average gap over all sessions whose gap was already known
                       before the forecast (expanding mean, shifted one session): no look-ahead.
C. linear_regression   ordinary least squares on the same features as the existing model
                       (median imputation fitted on the training window, standardised).
D. hist_gradient_boosting  the previous model, unchanged (train.make_model).
E. forecast            the model the app forecasts with: the winner of train.tournament (model_zoo).

Every model is fitted only on rows dated before its fold's first test date and scored on the same
rows. The output is one row per (model, stock, test day) in models/oos_predictions.csv, so every
metric, chart and backtest downstream is computed from the same out-of-sample forecasts.

Prediction interval: for fold k the 10th/90th percentiles of the HistGradientBoosting residuals
from folds 1..k-1 (all out-of-sample and all earlier) are added to the forecast. Its empirical
coverage on fold k is reported, so the claim "80% interval" can be checked.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features, metrics
from .paths import MODELS

MODEL_LABELS = {
    "naive": "A. Naive (gap = 0)",
    "historical_average": "B. Historical average gap",
    "linear_regression": "C. Linear Regression",
    "hist_gradient_boosting": "D. HistGradientBoosting (previous model)",
    "forecast": "E. Pulse8 forecast (best in walk-forward tournament)",
}
FORECAST = "forecast"  # the model behind the app's forecasts, confidence table and interval checks
OOS_FILE = MODELS / "oos_predictions.csv"
COMPARISON_FILE = MODELS / "model_comparison.csv"
BY_STOCK_FILE = MODELS / "model_comparison_by_stock.csv"
COVERAGE_FILE = MODELS / "interval_coverage.csv"
META_FILE = MODELS / "model_eval_meta.json"
TESTS_FILE = MODELS / "model_significance.csv"
PAIRS = [(FORECAST, "naive"), (FORECAST, "hist_gradient_boosting"), ("hist_gradient_boosting", "naive"), ("hist_gradient_boosting", "historical_average"),
         ("hist_gradient_boosting", "linear_regression"), ("linear_regression", "naive")]


def make_linear_model():
    """OLS with train-window median imputation (plus missing-value flags) and scaling."""
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LinearRegression())


def historical_average_gap(df: pd.DataFrame) -> pd.Series:
    """Per-stock expanding mean of gaps already realised before each forecast.

    The row dated d forecasts the gap into the next session. The gap of the row dated d-1 is
    realised at the open of d, before the close of d when the forecast is made, so shifting the
    expanding mean by one row uses only past information. The first row of each stock is NaN.
    """
    d = df.sort_values(["symbol", "date"])
    hist = d.groupby("symbol")["target_gap"].transform(lambda s: s.expanding().mean().shift(1))
    return hist.reindex(df.index)


def evaluate(df: pd.DataFrame, cols: list[str], kind: str | None = None, save: bool = True) -> dict[str, pd.DataFrame]:
    """Run all five models on the shared walk-forward folds. Returns the result tables.

    kind: the model_zoo candidate the app forecasts with (default: the last tournament's winner).
    """
    from . import model_zoo, train  # local import: train imports this module

    kind = kind or chosen_kind()

    df = df.dropna(subset=["target_gap"]).copy()
    df["hist_avg_gap"] = historical_average_gap(df)
    frames = []
    for k, train_df, test in train.folds(df):
        use = train.usable_features(train_df, cols)
        preds = {
            "naive": np.zeros(len(test)),
            "historical_average": test["hist_avg_gap"].fillna(0.0).to_numpy(),
            "linear_regression": make_linear_model().fit(train_df[use], train_df["target_gap"]).predict(test[use]),
            "hist_gradient_boosting": train.make_model().fit(train_df[use], train_df["target_gap"]).predict(test[use]),
            FORECAST: model_zoo.fit_candidate(kind, train_df[use], train_df["target_gap"], train_df["date"]).predict(test[use]),
        }
        base = test[["date", "symbol", "close", "next_open", "target_gap"]].assign(fold=k)
        for name, p in preds.items():
            frames.append(base.assign(model=name, pred=p))
    oos = pd.concat(frames, ignore_index=True).rename(columns={"target_gap": "actual"})
    oos["error"] = oos["pred"] - oos["actual"]
    oos["pred_open"] = oos["close"] * (1 + oos["pred"])

    oos, coverage = add_intervals(oos)
    overall = metrics.score_table(oos, ["model"])
    overall["mae_rupees"] = oos.groupby("model").apply(
        lambda g: float((g["pred_open"] - g["next_open"]).abs().mean()), include_groups=False).reindex(overall["model"]).to_numpy()
    by_stock = metrics.score_table(oos, ["model", "symbol"])
    by_fold = metrics.score_table(oos, ["model", "fold"])
    for t in (overall, by_stock, by_fold):
        t.insert(1, "model_label", t["model"].map(MODEL_LABELS))
    tests = significance(oos)
    tables = {"oos": oos, "overall": overall, "by_stock": by_stock, "by_fold": by_fold, "coverage": coverage,
              "tests": tests}
    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        oos.to_csv(OOS_FILE, index=False)
        overall.to_csv(COMPARISON_FILE, index=False)
        by_stock.to_csv(BY_STOCK_FILE, index=False)
        by_fold.to_csv(MODELS / "model_comparison_by_fold.csv", index=False)
        coverage.to_csv(COVERAGE_FILE, index=False)
        tests.to_csv(TESTS_FILE, index=False)
        META_FILE.write_text(json.dumps({
            "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "test_start": str(oos["date"].min())[:10], "test_end": str(oos["date"].max())[:10],
            "test_rows_per_model": int(len(oos) // len(MODEL_LABELS)),
            "symbols": sorted(oos["symbol"].unique().tolist()), "features": cols,
            "forecast_model": kind, "forecast_model_label": model_zoo.label(kind)}, indent=2))
    return tables


def chosen_kind() -> str:
    """The tournament winner saved by the last training run (gradient boosting before the first one)."""
    from . import model_zoo
    try:
        return json.loads((MODELS / "model_choice.json").read_text())["chosen"]
    except (OSError, ValueError, KeyError):
        return model_zoo.DEFAULT


def add_intervals(oos: pd.DataFrame, lo: float = 0.10, hi: float = 0.90) -> tuple[pd.DataFrame, pd.DataFrame]:
    """10/90 band per fold from earlier folds' out-of-sample residuals of the same model."""
    oos = oos.copy()
    oos["pred_q10"], oos["pred_q90"] = np.nan, np.nan
    rows = []
    for model, g in oos.groupby("model"):
        for k in sorted(g["fold"].unique()):
            past = g[g["fold"] < k]
            if past.empty:
                continue
            resid = past["actual"] - past["pred"]
            q_lo, q_hi = resid.quantile(lo), resid.quantile(hi)
            idx = g.index[g["fold"] == k]
            oos.loc[idx, "pred_q10"] = oos.loc[idx, "pred"] + q_lo
            oos.loc[idx, "pred_q90"] = oos.loc[idx, "pred"] + q_hi
            inside = oos.loc[idx, "actual"].between(oos.loc[idx, "pred_q10"], oos.loc[idx, "pred_q90"])
            rows.append({"model": model, "fold": int(k), "q10_resid_pct": q_lo * 100, "q90_resid_pct": q_hi * 100,
                         "coverage_pct": float(inside.mean() * 100), "n": int(len(idx))})
    return oos, pd.DataFrame(rows)


def significance(oos: pd.DataFrame) -> pd.DataFrame:
    """Diebold-Mariano comparison of absolute errors for the key model pairs."""
    wide = oos.pivot_table(index=["date", "symbol"], columns="model", values="error").abs().reset_index()
    rows = []
    for a, b in PAIRS:
        res = metrics.diebold_mariano(wide[a], wide[b], wide["date"])
        rows.append({"model_a": MODEL_LABELS[a], "model_b": MODEL_LABELS[b],
                     "mean_abs_error_diff_pct": res["mean_diff"] * 100, "t_stat": res["t_stat"],
                     "n_days": res["n_days"],
                     "verdict": ("A significantly better" if res["t_stat"] < -1.96 else
                                 "B significantly better" if res["t_stat"] > 1.96 else
                                 "No significant difference (|t| <= 1.96)")})
    return pd.DataFrame(rows)


def load_oos() -> pd.DataFrame | None:
    """The saved out-of-sample table (None before the first training run)."""
    if not OOS_FILE.exists():
        return None
    return pd.read_csv(OOS_FILE, parse_dates=["date"])


def main() -> None:
    from . import train

    df = features.build(save=False).dropna(subset=["target_gap"])
    tables = evaluate(df, train.usable_features(df))
    print_report(tables)


def print_report(tables: dict[str, pd.DataFrame]) -> None:
    pd.set_option("display.width", 140)
    cols = ["model_label", "n", "mae_pct", "rmse_pct", "bias_pct", "direction_acc_pct", "mae_rupees"]
    print("\nModel comparison on identical walk-forward test rows (gap in % points):")
    print(tables["overall"][cols].round(4).to_string(index=False))
    cov = tables["coverage"]
    if len(cov):
        print("\n10-90% interval coverage on later folds (target 80%):")
        print(cov.round(2).to_string(index=False))
    print("\nDiebold-Mariano tests on absolute error (negative t = model A more accurate):")
    print(tables["tests"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
