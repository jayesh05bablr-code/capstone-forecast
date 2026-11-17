"""Tab 2, Historical model accuracy: four forecasters on identical walk-forward test days."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from capstone import charts, charts_eval, evaluation, metrics
from capstone.model_eval import MODEL_LABELS
from capstone.paths import MODELS, PROCESSED
from capstone.ui import data_cache
from capstone.ui.common import raw_prices

METRIC_COLS = {"model_label": "Model", "n": "Test days x stocks", "mae_pct": "MAE (% pts)", "rmse_pct": "RMSE (% pts)",
               "bias_pct": "Bias (% pts)", "direction_acc_pct": "Directional accuracy %"}

WHY_DIFFERENT = """
**Why Linear Regression and HistGradientBoosting can differ**

- *Interpretability.* Linear Regression gives one coefficient per feature (for example, how much of
  last night's S&P 500 move passes into the Indian open), which is easy to explain. Gradient-boosted
  trees combine hundreds of shallow trees; they are explained here with what-if driver attribution.
- *Non-linear relationships.* Trees can learn thresholds and interactions (a large US move matters more
  than a small one, the ADR matters only for stocks that have one). A linear model treats every effect
  as straight-line and additive.
- *Overfitting risk.* Trees can fit noise; the existing model limits this with depth 3, at least 50
  rows per leaf and L2 regularisation. Linear Regression has few parameters and is hard to overfit.
- *Dataset size.* About 2,900 sessions x 5 stocks is small for a flexible model, and opening gaps are
  mostly noise, so the extra flexibility of trees has little to work with.
- *Loss function.* The existing model minimises absolute error (it forecasts the median gap); Linear
  Regression minimises squared error (the mean). This shows up in bias and RMSE.
"""


def _filters(oos: pd.DataFrame):
    c1, c2, c3 = st.columns([1, 2, 2])
    stocks = ["All stocks"] + sorted(oos["symbol"].unique())
    stock = c1.selectbox("Stock", stocks, key="acc_stock")
    lo, hi = oos["date"].min().date(), oos["date"].max().date()
    default_lo = max(lo, (pd.Timestamp(hi) - pd.DateOffset(years=1)).date())
    rng = c2.date_input("Date range", (lo, hi), min_value=lo, max_value=hi, key="acc_dates")
    models = c3.multiselect("Models", list(MODEL_LABELS), default=list(MODEL_LABELS),
                            format_func=MODEL_LABELS.get, key="acc_models") or list(MODEL_LABELS)
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    else:
        start, end = pd.Timestamp(lo), pd.Timestamp(hi)
    sel = oos[(oos["date"] >= start) & (oos["date"] <= end) & oos["model"].isin(models)]
    if stock != "All stocks":
        sel = sel[sel["symbol"] == stock]
    return sel, stock, models, default_lo


def render(symbol: str) -> None:
    oos = data_cache.oos()
    if oos is None:
        st.warning("No model comparison yet: run `python -m capstone.train` (the dashboard does this on its weekly "
                   "retrain).")
        return
    meta = data_cache.meta()
    st.markdown(f"Every model was trained only on data before each test period and scored on the **same** "
                f"{meta.get('test_rows_per_model', len(oos) // 4):,} stock-days from {meta.get('test_start', '')} "
                f"to {meta.get('test_end', '')} (5 expanding walk-forward folds). Gap = next open / last close - 1. "
                "Directional accuracy counts a zero forecast as correct only when the actual gap was exactly zero, "
                "so the naive baseline scores close to 0% by design.")
    sel, stock, models, recent = _filters(oos)
    if sel.empty:
        st.info("No test days in this selection.")
        return

    table = metrics.score_table(sel, ["model"]).assign(model_label=lambda t: t["model"].map(MODEL_LABELS))
    st.dataframe(table[list(METRIC_COLS)].rename(columns=METRIC_COLS).round(3), hide_index=True,
                 use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts_eval.metric_bars(table, "mae_pct", "MAE (% points, lower is better)", MODEL_LABELS),
                    use_container_width=True)
    c2.plotly_chart(charts_eval.metric_bars(table, "direction_acc_pct", "Directional accuracy (%, higher is better)",
                                            MODEL_LABELS), use_container_width=True)

    tests = data_cache.models_csv("model_significance.csv")
    if tests is not None:
        st.markdown("**Is the difference real?** Diebold-Mariano test on absolute errors over all test days "
                    "(errors averaged across stocks per day, Newey-West standard errors).")
        st.dataframe(tests.round(4), hide_index=True, use_container_width=True)

    one_stock = stock if stock != "All stocks" else symbol
    st.markdown(f"**Day by day: {one_stock}**" + (" (pick a stock above to change)" if stock == "All stocks" else ""))
    day = sel[sel["symbol"] == one_stock]
    if stock == "All stocks":
        day = day[day["date"] >= pd.Timestamp(recent)]
        st.caption("Showing the last 12 months of the selection for readability.")
    st.plotly_chart(charts_eval.predicted_vs_actual(day, MODEL_LABELS), use_container_width=True)
    focus = st.selectbox("Model for the scatter and error charts", models, format_func=MODEL_LABELS.get,
                         index=models.index("hist_gradient_boosting") if "hist_gradient_boosting" in models else 0,
                         key="acc_focus")
    f = day[day["model"] == focus]
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts_eval.scatter_actual_vs_pred(f, MODEL_LABELS[focus], focus), use_container_width=True)
    c2.plotly_chart(charts_eval.daily_error(f, MODEL_LABELS[focus], focus), use_container_width=True)

    window = st.slider("Rolling window (trading days)", 20, 250, 60, 10, key="acc_window")
    roll = {}
    for m, g in sel.groupby("model"):
        g = g.sort_values("date")
        r = metrics.rolling_scores(g, window * g["symbol"].nunique())  # window days x stocks in view
        roll[m] = r.groupby("date", as_index=False).last()
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts_eval.rolling_lines(roll, "rolling_mae_pct", f"Rolling MAE ({window} days)", "MAE (% points)",
                                              MODEL_LABELS), use_container_width=True)
    c2.plotly_chart(charts_eval.rolling_lines(roll, "rolling_direction_acc_pct", f"Rolling directional accuracy "
                                              f"({window} days)", "Directional accuracy (%)", MODEL_LABELS),
                    use_container_width=True)

    st.markdown("**Stock-wise accuracy** (selected dates and models)")
    by_stock = metrics.score_table(sel, ["symbol", "model"]).assign(model=lambda t: t["model"].map(MODEL_LABELS))
    st.dataframe(by_stock.pivot(index="symbol", columns="model", values="mae_pct").round(3), use_container_width=True)
    st.caption("MAE in percentage points of the gap, per stock (lower is better).")
    st.dataframe(by_stock.pivot(index="symbol", columns="model", values="direction_acc_pct").round(1),
                 use_container_width=True)
    st.caption("Directional accuracy %, per stock.")

    cov = data_cache.models_csv("interval_coverage.csv")
    if cov is not None:
        with st.expander("Does the 10-90% interval hold? (coverage per test period)"):
            st.dataframe(cov.assign(model=cov["model"].map(MODEL_LABELS)).round(2), hide_index=True)
            st.caption("Band for each period = 10th and 90th percentiles of the same model's errors in earlier test "
                       "periods. Coverage near 80% means the band is honest; it varies with market volatility.")
    st.markdown(WHY_DIFFERENT)

    with st.expander("Earlier report (rupee errors, kept for continuity)"):
        report = MODELS / "walk_forward_report.csv"
        if report.exists():
            st.dataframe(evaluation.accuracy_summary(pd.read_csv(report)))
            st.caption("Original metrics: MAE in ₹ pooled across stocks, and a direction accuracy that skips days "
                       "with a zero actual gap. The table above replaces it.")
    log_path = PROCESSED / "predictions_log.csv"
    hist = raw_prices(symbol)
    if log_path.exists() and hist is not None:
        joined = evaluation.predictions_vs_actual(pd.read_csv(log_path), hist, symbol)
        if len(joined):
            st.markdown("**Live forecasts logged by the app vs what happened**")
            st.plotly_chart(charts.line(joined, "as_of_close", {"pred_open": "Predicted open",
                                                                "actual_open": "Actual open"},
                                        f"{symbol}: logged predictions vs actual open", "₹"),
                            use_container_width=True)
