"""Unit tests for the calculations that used to live inside app.py.

    python -m pytest tests
"""
import numpy as np
import pandas as pd

from capstone import company, evaluation, planner, scalping

PREDS = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "close": [100.0, 200.0, 50.0],
                      "pred_open": [101.0, 198.0, 50.0], "pred_gap_pct": [1.0, -1.0, 0.0]})


def daily(n=40, start=100.0):
    close = start * np.exp(np.cumsum(np.r_[0, np.tile([0.01, -0.01], n)[: n - 1]]))
    return pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n), "close": close})


# --- scalping ------------------------------------------------------------------------
def test_range_rows_live_estimate_and_skip():
    rows = scalping.range_rows(PREDS, {"AAA": daily(), "BBB": daily(), "CCC": daily(10)},
                               {"AAA": {"open": 101.2, "high": 102.0, "low": 100.5}, "BBB": None}, 5)
    assert [r["Stock"] for r in rows] == ["AAA", "BBB"]  # CCC has too little history
    assert rows[0]["Source"] == "Live, first 5 min today" and rows[0]["Range high"] == 102.0
    est = rows[1]
    assert est["Source"] == "Estimate"
    assert est["Range low"] < est["Open"] == 198.0 < est["Range high"]
    assert abs((est["Range high"] + est["Range low"]) / 2 - 198.0) < 0.01  # centred on the forecast


def test_apply_preopen_recentres_only_estimates():
    rows = [{"Stock": "AAA", "Open": 101.0, "Range high": 102.0, "Range low": 100.0, "Source": "Estimate"},
            {"Stock": "BBB", "Open": 198.0, "Range high": 199.0, "Range low": 197.0, "Source": "Live"}]
    pre = pd.DataFrame({"symbol": ["AAA", "BBB"], "open": [105.0, 190.0]}).set_index("symbol")
    assert scalping.apply_preopen(rows, pre) == 1
    assert rows[0] == {"Stock": "AAA", "Open": 105.0, "Range high": 106.0, "Range low": 104.0,
                       "Source": "NSE pre-open open, estimated range"}
    assert rows[1]["Open"] == 198.0


def test_forecast_vs_open_direction():
    pre = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "open": [102.0, 201.0, 50.01]}).set_index("symbol")
    chk = scalping.forecast_vs_open(PREDS, pre).set_index("Stock")
    assert chk.loc["AAA", "Direction right?"] == "Yes"   # forecast up, opened up
    assert chk.loc["BBB", "Direction right?"] == "No"    # forecast down, opened up
    assert chk.loc["CCC", "Direction right?"] == "Flat"  # moved less than 0.05%
    assert chk.loc["AAA", "Actual gap %"] == 2.0


def test_bias_sources():
    row = {"Model gap %": -0.4, "Open": 101.0, "Last close": 100.0}
    assert scalping.bias(row, "Model's expected gap") == -0.4
    assert scalping.bias(row, "Actual opening gap") == 1.0
    assert scalping.bias(row, "Both directions") == 0



def test_exit_clock():
    assert scalping.exit_clock(20) == "09:35"
    assert scalping.exit_clock(45) == "10:00"
    assert scalping.exit_clock(50) == "10:05"


# --- planner -------------------------------------------------------------------------
def test_allocation_notes():
    pl = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "shares": [0, 5, 10], "avg_ret": [-0.01, 0.01, 0.02],
                       "limited_by_risk": [False, True, False], "pred_gap_pct": [0.2, 0.1, -0.8]})
    notes = planner.allocation_notes(pl, "Favour historical edge", 0.01)
    assert notes == [
        "**AAA**: nothing allocated (its plan lost money in the backtest).",
        "**BBB**: amount reduced so a stop-out loses at most 1.0% of your money.",
        "**CCC**: the model expects a -0.80% gap down; consider waiting for the open to settle before buying.",
    ]
    assert "costs more" in planner.allocation_notes(pl, "Equal amount", 0.01)[0]


# --- evaluation ----------------------------------------------------------------------
def test_accuracy_summary_and_gain():
    report = pd.DataFrame({"method": ["model", "model", "model_without_overnight"],
                           "mae_rupees": [2.0, 4.0, 5.0], "mape_pct": [0.1, 0.3, 0.4],
                           "direction_acc_pct": [80, 60, 50]})
    s = evaluation.accuracy_summary(report)
    assert s.loc[evaluation.MODEL, "MAE (₹)"] == 3.0
    assert evaluation.overnight_gain(s) == 2.0
    assert evaluation.overnight_gain(s.drop(evaluation.WITHOUT_OVERNIGHT)) is None


def test_predictions_vs_actual_matches_next_open():
    hist = pd.DataFrame({"date": pd.to_datetime(["2026-10-20", "2026-10-21", "2026-10-22"]),
                         "open": [10.0, 11.0, 12.0]})
    log = pd.DataFrame({"symbol": ["AAA", "AAA", "AAA", "BBB"],
                        "as_of_close": ["2026-10-20", "2026-10-21", "2026-10-21", "2026-10-21"],
                        "pred_open": [10.5, 11.1, 11.2, 99.0]})
    out = evaluation.predictions_vs_actual(log, hist, "AAA")
    assert list(out["actual_open"]) == [11.0, 12.0]  # the open after each close
    assert list(out["pred_open"]) == [10.5, 11.2]     # last forecast per day wins


# --- company -------------------------------------------------------------------------
def test_classify_investors():
    inv = pd.DataFrame({"symbol": ["X"] * 5,
                        "investor": ["LIC", "Vanguard", "SBI MF", "Some Fund", "Norway"],
                        "investor_type": ["Insurance", "", "Mutual Fund", "", ""],
                        "country": ["India", "", "", "", "Norway"]})
    foreign, domestic, unknown = company.classify_investors(inv)
    assert list(inv["investor"][foreign]) == ["Norway"]
    assert list(inv["investor"][domestic]) == ["LIC", "SBI MF"]
    assert list(inv["investor"][unknown]) == ["Vanguard", "Some Fund"]


def test_shareholding_helpers():
    fund = pd.DataFrame({"symbol": ["x", "X"], "as_of": ["2026-06-30", "2026-09-30"],
                         "promoter_pct": [50.0, 49.0], "fii_pct": [20.0, 21.5], "dii_pct": [15.0, np.nan],
                         "public_pct": [15.0, 14.5]})
    assert len(company.for_symbol(fund, "X")) == 2  # case-insensitive
    sh = company.shareholding(fund)
    last, prev = sh.iloc[-1], sh.iloc[-2]
    assert company.change_vs_previous(last, prev, "fii_pct") == "+1.50 pp vs previous filing"
    assert company.change_vs_previous(last, prev, "dii_pct") is None
    assert company.domestic_total(prev) == 80.0
    assert company.tracxn_ownership(fund).empty
