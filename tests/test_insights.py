"""Fundamentals vs forecast check, executive summary and chart builders."""
import numpy as np
import pandas as pd

from capstone import charts_eval, fundamental_check as fc, summary


def fin():
    q = pd.to_datetime(["2025-03-31", "2025-06-30", "2026-03-31", "2026-06-30"])
    return pd.DataFrame({"symbol": "AAA", "period_end": q, "revenue_cr": [100, 100, 120, 80],
                         "net_profit_cr": [10, 10, 12, 8]})


def test_growth_is_year_on_year_and_point_in_time():
    g = fc.growth_table(fin())
    row = g[g["period_end"] == pd.Timestamp("2026-06-30")].iloc[0]
    assert np.isclose(row["revenue_yoy"], -0.2) and np.isclose(row["profit_yoy"], -0.2)
    # On 1 Aug 2026 the June quarter is not public yet (45-day lag): the March quarter is the latest.
    early = fc.latest_trend(g, "AAA", pd.Timestamp("2026-08-01"))
    late = fc.latest_trend(g, "AAA", pd.Timestamp("2026-08-20"))
    assert early["quarter"] == "Mar 2026" and early["trend"] == "Strong growth"
    assert late["quarter"] == "Jun 2026" and late["trend"] == "Declining" and late["momentum"] == "Slowing"


def test_disagreement_flag(monkeypatch):
    monkeypatch.setattr(fc, "load_financials", fin)
    preds = pd.DataFrame({"symbol": ["AAA", "AAA", "ZZZ"], "pred_gap_pct": [0.4, 0.01, 0.4]})
    out = fc.compare(preds, pd.Timestamp("2026-09-01"))
    assert out["assessment"].iloc[0].startswith("Disagree")       # gap-up vs declining fundamentals
    assert out["assessment"].iloc[1] == "No comparison"            # flat forecast
    assert out["trend"].iloc[2] == "Unknown"                       # no data for this company


def test_summary_without_results_still_renders():
    text = summary.markdown({})
    assert "Business problem" in text and "Limitations" in text


def test_weekly_chart_colours_by_cost():
    w = pd.DataFrame({"week_start": pd.to_datetime(["2026-01-05", "2026-01-12"]),
                      "entry_date": pd.to_datetime(["2026-01-05", "2026-01-12"]),
                      "exit_date": pd.to_datetime(["2026-01-09", "2026-01-16"]), "sessions": [5, 5],
                      "weekly_return": [0.01, 0.001]})
    fig = charts_eval.weekly_delta(w, "AAA", 0.004)
    assert list(fig.data[0].marker.color) == [charts_eval.TRADABLE, charts_eval.NOT_TRADABLE]
