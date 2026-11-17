# Model evaluation: NSE opening-gap forecast

All numbers below were produced by `python -m capstone.train` on 17 Nov 2026 from real data: NSE daily prices for
RELIANCE, TCS, HDFCBANK, INFY and ICICIBANK (1 Jan 2015 to 13 Nov 2026), US closes (SPY, QQQ, 11 sector ETFs,
ADRs INFY/HDB/IBN) and USD/INR history. Re-running the command on the same data reproduces them exactly (fixed
seeds, deterministic folds). Files: `models/model_comparison*.csv`, `models/model_significance.csv`,
`models/interval_coverage.csv`, `models/oos_predictions.csv`.

## 1. What is forecast

`target_gap = next session's open / today's close - 1`, per stock and trading day. Features use only information
available before the next open: price features from today's close and earlier (`features.py`), the overnight US
and rupee moves from US sessions dated from today up to the day before the next NSE session (`overnight.py`,
tested in `tests/test_overnight.py`), and fundamentals joined as-of their publication date (`fundamentals.py`).
Days with a gap beyond ±20% (bonus/split ex-dates in unadjusted prices) are excluded from training and testing.

## 2. Models (all on identical rows)

| Model | Definition | Leakage control |
| --- | --- | --- |
| A. Naive | predicted gap = 0 (yesterday's close = today's open) | none needed |
| B. Historical average | each stock's mean gap over all sessions whose gap was already realised before the forecast (expanding mean shifted one session) | test: changing a future gap leaves earlier baseline values unchanged (`tests/test_model_eval.py`) |
| C. Linear Regression | OLS on the existing feature set; median imputation + missing flags + standardisation fitted on the training window only | pipeline refitted per fold |
| D. HistGradientBoosting | existing model, unchanged: absolute-error loss, depth 3, 300 iterations, learning rate 0.05, min 50 rows per leaf, L2 = 1, seed 0 | refitted per fold |

**Walk-forward validation.** Five expanding folds (`train.folds`). The first 250 sessions are training only; each
fold trains on all rows dated before its first test date and tests on the next 538 sessions. Every model sees the
same training and test rows. Test periods:

| Fold | First test day | Last test day | Sessions |
| --- | --- | --- | --- |
| 1 | 2016-01-05 | 2018-03-07 | 538 |
| 2 | 2018-03-08 | 2020-05-15 | 538 |
| 3 | 2020-05-18 | 2022-07-11 | 538 |
| 4 | 2022-07-12 | 2024-09-11 | 538 |
| 5 | 2024-09-12 | 2026-11-12 | 538 |

Change from the earlier report: the last fold now includes the final session (it previously stopped one day
early). The HistGradientBoosting model itself was not changed.

## 3. Metric definitions

- **MAE / RMSE** in percentage points of the gap (0.50 = half a percent), so stocks at ₹350 and ₹2,600 are
  comparable. MAE in rupees per share is also reported for continuity.
- **Bias** = mean(forecast - actual); positive = forecasts too high.
- **Directional accuracy**: a positive forecast is right only when the actual gap is positive, a negative one only
  when it is negative, and a zero forecast only when the actual gap is exactly zero. Every test row counts. The
  naive model therefore scores about 3% (the share of days with an exactly unchanged open).
- **Diebold-Mariano test** on absolute errors, with errors first averaged across the five stocks each day (they are
  correlated) and Newey-West standard errors (5 lags). |t| > 1.96 ≈ significant at 5%.

## 4. Results: all stocks pooled (13,438 stock-days per model)

| Model | MAE (% pts) | RMSE (% pts) | Bias (% pts) | Directional accuracy % | MAE (₹/share) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A. Naive (gap = 0) | 0.560 | 0.926 | -0.076 | 2.8 | 8.32 |
| B. Historical average gap | 0.554 | 0.924 | 0.028 | 56.9 | 8.25 |
| C. Linear Regression | 0.488 | **0.810** | 0.027 | 65.9 | 7.47 |
| D. HistGradientBoosting (existing) | **0.486** | 0.824 | 0.034 | **66.2** | **7.30** |

**Significance (Diebold-Mariano, 2,690 test days):**

| Comparison | Mean difference in absolute error (% pts) | t | Verdict |
| --- | ---: | ---: | --- |
| HistGradientBoosting vs Naive | -0.0745 | -15.8 | HGB significantly better |
| HistGradientBoosting vs Historical average | -0.0687 | -15.1 | HGB significantly better |
| HistGradientBoosting vs Linear Regression | -0.0025 | -1.0 | **No significant difference** |
| Linear Regression vs Naive | -0.0720 | -12.7 | LR significantly better |

**Conclusion.** Both learned models beat both baselines clearly and consistently (every fold, every stock). The
existing HistGradientBoosting model reduces the naive MAE by 13.3% and is right about the direction on 66% of
stock-days. It is **not** demonstrably better than Linear Regression: its MAE is 0.002 points lower, its RMSE is
higher, and the difference is not statistically significant. The project keeps HistGradientBoosting as the
production model (unchanged behaviour, native missing-value handling, slightly better MAE and direction), and the
report says plainly that a linear model performs about as well.

## 5. By stock

MAE (% points):

| Stock | A. Naive | B. Hist. average | C. Linear Regression | D. HistGradientBoosting |
| --- | ---: | ---: | ---: | ---: |
| HDFCBANK | 0.530 | 0.526 | **0.453** | 0.457 |
| ICICIBANK | 0.645 | 0.643 | **0.541** | 0.554 |
| INFY | 0.639 | 0.634 | **0.494** | 0.512 |
| RELIANCE | 0.509 | 0.495 | 0.485 | **0.461** |
| TCS | 0.478 | 0.475 | 0.468 | **0.444** |

Directional accuracy %:

| Stock | A. Naive | B. Hist. average | C. Linear Regression | D. HistGradientBoosting |
| --- | ---: | ---: | ---: | ---: |
| HDFCBANK | 2.4 | 55.1 | **67.3** | 65.7 |
| ICICIBANK | 2.9 | 54.7 | **68.7** | 67.5 |
| INFY | 2.8 | 56.2 | **71.5** | 71.1 |
| RELIANCE | 2.6 | 60.8 | 61.0 | **63.4** |
| TCS | 3.1 | 57.4 | 60.9 | **63.4** |

The three stocks with a US-listed ADR (HDFCBANK, ICICIBANK, INFY) are where the overnight signal is strongest and
Linear Regression does best: the ADR's overnight move passes almost linearly into the Indian open. RELIANCE and
TCS, without ADRs, depend on weaker, less linear signals, and the trees do better there.

## 6. By fold (MAE, % points)

| Fold | A. Naive | B. Hist. average | C. Linear Regression | D. HistGradientBoosting |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.462 | 0.454 | 0.418 | 0.415 |
| 2 | 0.635 | 0.622 | 0.563 | 0.563 |
| 3 | 0.657 | 0.641 | 0.546 | 0.543 |
| 4 | 0.486 | 0.485 | 0.396 | 0.392 |
| 5 | 0.561 | 0.571 | 0.517 | 0.515 |

The edge over the naive forecast is smallest in fold 5 (Sep 2024 to Nov 2026): directional accuracy fell to 59.6%
for HistGradientBoosting and 61.3% for Linear Regression. Worth monitoring on the dashboard's rolling charts.

## 7. Prediction interval (10th to 90th percentile)

Method: for each test fold, the 10th and 90th percentiles of the same model's out-of-sample residuals in all
earlier folds are added to the forecast (no residual from the fold itself is used). Coverage of the
HistGradientBoosting band (target 80%):

| Fold | 10th pct residual | 90th pct residual | Coverage |
| --- | ---: | ---: | ---: |
| 2 | -0.60% | +0.53% | 73.2% |
| 3 | -0.66% | +0.62% | 70.3% |
| 4 | -0.71% | +0.70% | 87.0% |
| 5 | -0.68% | +0.65% | 76.3% |

The band is roughly right on average (77% across folds 2-5) but too narrow in volatile periods (2018-2022) and too
wide in calm ones. The live dashboard band uses the existing method (last fold's out-of-sample residuals) and
states that it is a statistical estimate, not a guaranteed price boundary.

## 8. Why Linear Regression and HistGradientBoosting can differ

- **Interpretability.** Linear Regression has one coefficient per feature (e.g. how much of the ADR's overnight
  move passes into the open). The trees are explained with what-if driver attribution (`explain.py`).
- **Non-linear relationships.** Trees learn thresholds and interactions (a large US move matters more than a small
  one; ADR features matter only for stocks that have an ADR). OLS is additive and straight-line.
- **Overfitting.** Trees can fit noise; the existing settings (depth 3, ≥50 rows per leaf, L2) limit this. OLS has
  few parameters and is hard to overfit, which helps on noisy targets.
- **Dataset size.** About 2,900 sessions × 5 stocks, with a target that is mostly noise, gives a flexible model
  little to learn beyond the strong linear ADR/US-market link.
- **Loss function.** HistGradientBoosting minimises absolute error (forecasts the median gap) and so wins on MAE;
  OLS minimises squared error (the mean) and so wins on RMSE. Both are visible in the table above.

## 9. Leakage and bias checks

| Risk | Check |
| --- | --- |
| Look-ahead in training | each fold trains on rows dated strictly before its first test date (`test_folds_are_ordered_and_cover_the_end`) |
| Look-ahead in the average baseline | expanding mean shifted one session; future changes do not alter it (`test_historical_average_uses_only_past_gaps`) |
| Overnight window | US sessions dated d..n-1 only; tests in `tests/test_overnight.py` |
| Fundamentals | merge-as-of on the publication date, backward only |
| Interval | built from earlier folds' residuals only |
| Date alignment | the actual next open is joined by the next trading date in the stock's own history |
| Corporate actions | ±20% gap days dropped; known ex-dates listed in `data/corporate_actions.csv` (5- and 20-day features after an ex-date remain slightly distorted; documented, not silently changed) |
| Survivorship bias | the 5 stocks are today's large caps; results flatter what a 2015 investor could have chosen |
