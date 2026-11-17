"""Build one Excel workbook with every dataset behind the model, for EDA."""
from __future__ import annotations

import io

import pandas as pd

from . import overnight
from .paths import DATA, FUNDAMENTALS_FILE, FX_DIR, MODELS, PROCESSED, RAW, SECTOR_LINKS_FILE, STOCK_MAP_FILE, US_DIR

README = [
    ("NSE_prices", "Daily OHLCV for each NSE stock (TejHQ)."),
    ("US_closes", "Daily closes of SPY, QQQ, the 11 sector ETFs and ADRs, one column each."),
    ("USDINR", "Rupees per US dollar, daily close."),
    ("Model_features", "One row per stock per day: every feature the model sees and the target "
                       "(target_gap = next open / close - 1). Overnight columns are explained in overnight.py."),
    ("Predictions", "Latest forecast per stock with reasons and driver contributions."),
    ("Prediction_log", "Every forecast made so far, to compare with actual opens."),
    ("Walk_forward", "Out-of-sample accuracy per fold for the model and baselines."),
    ("Fundamentals", "Company fundamentals used by the model (from CSV / Tracxn uploads)."),
    ("Financials", "Uploaded quarterly/annual results (₹ crore)."),
    ("Investors", "Uploaded major holders with type (FII/FPI, DII, promoter...)."),
    ("Company_profiles", "Uploaded company details (founded, HQ, employees...)."),
    ("Tracxn_profiles", "Fields extracted from uploaded Tracxn PDFs."),
    ("Stock_map", "NSE symbol -> Indian sector and US ADR."),
    ("Model_comparison", "Naive, historical average, Linear Regression and HistGradientBoosting on identical "
                         "walk-forward test days: MAE, RMSE, bias (% points), directional accuracy."),
    ("Model_by_stock", "The same metrics per stock."),
    ("Model_significance", "Diebold-Mariano tests of the absolute-error differences."),
    ("Interval_coverage", "Share of actual gaps inside the 10-90% band, per test period."),
    ("OOS_predictions", "Every out-of-sample forecast (model, stock, day): actual, forecast, error, band."),
    ("Weekly_returns", "Each stock-week: entry/exit session, sessions, return, valid flag and exclusion reason."),
    ("Weekly_model", "Weekly return forecasters compared on identical walk-forward weeks."),
    ("Backtest_summary", "Unfiltered vs cost-aware strategy vs hold-all, default assumptions."),
    ("Backtest_trades_A", "Trades of the unfiltered strategy."),
    ("Backtest_trades_B", "Trades of the cost-aware strategy."),
    ("Backtest_equity", "Weekly equity of each strategy."),
    ("MonteCarlo_percentiles", "5th-95th percentile outcomes of 5,000 simulations of the cost-aware strategy."),
    ("Sector_links", "Indian sector -> US sector ETF and rupee exposure."),
]


def _read(path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


def sheets() -> dict[str, pd.DataFrame]:
    out = {}
    prices = [pd.read_csv(p).assign(symbol=p.stem) for p in sorted(RAW.glob("*.csv"))]
    if prices:
        out["NSE_prices"] = pd.concat(prices, ignore_index=True)
    us = {p.stem: overnight.load_close(p.stem) for p in sorted(US_DIR.glob("*.csv"))}
    us = {k: v for k, v in us.items() if not v.empty}
    if us:
        out["US_closes"] = pd.DataFrame(us).rename_axis("date").reset_index()
    fx = overnight.load_fx()
    if not fx.empty:
        out["USDINR"] = fx.rename("usdinr").rename_axis("date").reset_index()
    for name, path in [("Model_features", PROCESSED / "features.csv"),
                       ("Predictions", PROCESSED / "predictions.csv"),
                       ("Prediction_log", PROCESSED / "predictions_log.csv"),
                       ("Walk_forward", MODELS / "walk_forward_report.csv"),
                       ("Fundamentals", FUNDAMENTALS_FILE),
                       ("Financials", DATA / "financials.csv"),
                       ("Investors", DATA / "investors.csv"),
                       ("Company_profiles", DATA / "company_profiles.csv"),
                       ("Tracxn_profiles", DATA / "tracxn_profiles.csv"),
                       ("Stock_map", STOCK_MAP_FILE), ("Sector_links", SECTOR_LINKS_FILE),
                       ("Model_comparison", MODELS / "model_comparison.csv"),
                       ("Model_by_stock", MODELS / "model_comparison_by_stock.csv"),
                       ("Model_significance", MODELS / "model_significance.csv"),
                       ("Interval_coverage", MODELS / "interval_coverage.csv"),
                       ("OOS_predictions", MODELS / "oos_predictions.csv"),
                       ("Weekly_returns", MODELS / "weekly_returns.csv"),
                       ("Weekly_model", MODELS / "weekly_model_comparison.csv"),
                       ("Backtest_summary", MODELS / "backtest_summary.csv"),
                       ("Backtest_trades_A", MODELS / "backtest_trades_A.csv"),
                       ("Backtest_trades_B", MODELS / "backtest_trades_B.csv"),
                       ("MonteCarlo_percentiles", MODELS / "montecarlo_percentiles.csv")]:
        df = _read(path)
        if df is not None:
            out[name] = df
    eq = [_read(MODELS / f"backtest_weeks_{k}.csv") for k in ("A", "B", "H")]
    if all(e is not None for e in eq):
        out["Backtest_equity"] = pd.DataFrame({"week_start": eq[0]["week_start"], "equity_A_unfiltered": eq[0]["equity"],
                                               "equity_B_cost_aware": eq[1]["equity"], "equity_hold_all": eq[2]["equity"]})
    return out


def workbook() -> bytes:
    data = sheets()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        pd.DataFrame([(n, d, len(data[n]) if n in data else 0) for n, d in README],
                     columns=["Sheet", "What it holds", "Rows"]).to_excel(xl, sheet_name="README", index=False)
        for name, df in data.items():
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = df[c].dt.date
            df.to_excel(xl, sheet_name=name[:31], index=False)
    return buf.getvalue()
