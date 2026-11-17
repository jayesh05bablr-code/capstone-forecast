# Implementation audit

Audit date: 17 Nov 2026. Written before any code was changed (Phase 1 of the reviewer upgrade).

## 1. What was inspected

| Source | What it is | Notes |
| --- | --- | --- |
| GitHub `jayesh05bablr-code/capstone-forecast` (public) | The deployed code (Streamlit Community Cloud, https://pri-capstone.streamlit.app) | 14 commits, all made on 4 Nov 2026 through the GitHub web uploader ("Add files via upload"), plus two edits to `keep-awake.yml`. Cloned read-only; this session has no GitHub write access. |
| Project folder `capstone_online/` | The latest working copy | Differs from GitHub in three files: `capstone/live.py` (Yahoo `period="2d"` vs `"5d"`), `capstone/ui/tab_download.py` (cached sheet sizes) and `.github/workflows/keep-awake.yml`; it also has `.streamlit/` (the web uploader skips dot-folders). The upgrade starts from this copy. |
| Local laptop copy `Downloads\capstone-forecast-main` | GitHub zip plus real data | Has real NSE prices (5 stocks, 1 Jan 2015 to 13 Nov 2026, 2,941 sessions each), US closes (SPY, QQQ, 11 sector ETFs, ADRs INFY/HDB/IBN from 2006), USD/INR history, a trained model, the walk-forward report and a 5-row prediction log. Also holds a local `.env` (not read, not copied). |
| Older copies `capstone/`, `capstone_refactored/`, zips | Earlier versions | Superseded; kept as is. |

No Jupyter notebooks exist anywhere (repo, project folder or laptop copy). No Supabase code, configuration or dependency exists.

Baseline check before changes: the existing 25 unit tests pass (`pytest tests`, 25 passed), and re-running `python -m capstone.train` on the real data reproduces the laptop's `walk_forward_report.csv` exactly.

## 2. Existing architecture

```
fetch_data.py ── TejHQ (key optional) ─┐ fallback: Yahoo daily (yfinance)
                                        ├─> data/raw/{SYMBOL}.csv  (NSE OHLCV, unadjusted)
fetch_overnight.py ── Twelve Data ─────┤ fallback: Alpha Vantage; exchangerate.dev for live USD/INR
                                        └─> data/us/*.csv, data/fx/USDINR.csv
features.py  price features (t close only) + overnight.py (US/FX window d..n-1) + fundamentals.py (as-of join)
             target_gap = next_open / close - 1;  rows with |gap| > 20% dropped (splits/bonuses)
train.py     HistGradientBoostingRegressor(loss=absolute_error, depth 3, 300 iters, seed 0)
             5-fold expanding-window walk-forward (first 250 sessions = minimum train)
             baselines: open = last close, train-mean gap; ablation without overnight features
             10/90 band = quantiles of last-fold out-of-sample residuals
predict.py   latest row per stock -> gap, open, low/high band, driver contributions, story
             -> data/processed/predictions.csv, overnight_story.json, appends predictions_log.csv
explain.py   what-if driver attribution (US market, US sector, ADR, rupee, Indian factors)
autorun.py   self-refresh at 08:45 and 16:15 IST when the page is opened; weekly retrain
app.py + capstone/ui/tab_*.py   Streamlit dashboard (10 tabs)
export.py    Excel workbook of every dataset (Download data tab)
```

Dashboard tabs today: Live, Stocks, US sectors and India, Rupee, Invest planner, Intraday scalping, Company profile, Upload company data, Download data, Model accuracy.

## 3. Existing features mapped to the reviewer's requirements

| Requirement | Status | Where |
| --- | --- | --- |
| Data collection via API | Done | `tej_client.py`, `twelvedata_client.py`, `alphavantage_client.py`, `exchangerate_client.py`, `fetch_*.py` |
| Twelve Data integration | Done, key sent in the `Authorization` header and masked in errors | `twelvedata_client.py` |
| Preprocessing / feature engineering | Done | `features.py`, `overnight.py`, `fundamentals.py` |
| Overnight timezone handling | Done and tested | `overnight.py`, `tests/test_overnight.py` |
| Trading-holiday handling (overnight window) | Done: the window runs to the actual next NSE session, so weekends and Indian/US holidays are handled from the data | `overnight.window_return` |
| Walk-forward validation | Done (5 expanding folds) | `train.walk_forward` |
| Naive baseline (gap = 0) | Done | `baseline_open_eq_close` |
| Historical-average baseline | Done, training-window mean only (no leakage) | `baseline_mean_gap` |
| Linear Regression | **Missing** | |
| HistGradientBoosting | Done | `train.make_model` |
| MAE / RMSE / bias / per-stock metrics | **Partial**: MAE in rupees, MAPE and direction accuracy only, pooled across stocks | `train.scores` |
| Directional accuracy, reviewer definition | **Not compliant** (see bug B1) | |
| 10th–90th percentile interval | Done (from out-of-sample residuals), but shown only inside the reason text; no coverage check | `train.main`, `predict.py` |
| Driver attribution | Done | `explain.contributions` |
| Daily prediction log | Done but thin (see B6) | `predict.py` -> `predictions_log.csv` |
| Stock mapping | Done | `data/stock_map.csv`, `data/sector_links.csv` |
| Historical accuracy dashboard | **Partial**: fold-average table plus one predicted-vs-actual line from the short live log | `ui/tab_model.py` |
| Weekly Monday-open to Friday-close analysis | **Missing** | |
| Trading cost model | **Partial**: intraday (MIS) charges only, rates hard-coded, no spread/slippage split | `scalping.charges` |
| Weekly forecast model | **Missing** | |
| Backtesting (unfiltered vs cost-aware) | **Missing** (only a stop/target backtest in the Invest planner) | `planner.backtest` |
| Monte Carlo | **Missing** | |
| Supabase | **Missing** | |
| Excel export | Done; needs weekly, backtest and model-comparison sheets | `export.py` |
| Fundamentals vs forecast comparison | **Partial**: quarterly financials and shareholding are loaded and shown, but not compared with the forecast | `company_data/*.xlsx`, `uploads.py`, `ui/tab_company.py` |
| Executive summary, model evaluation, methodology docs, demo script | **Missing** | |
| Deployment files | Done (`requirements.txt`, `.streamlit/config.toml`, `secrets.toml.example`, Dockerfile, keep-awake workflow) | |

## 4. Measured results of the current model (real data, unchanged code)

Average over the 5 walk-forward folds (test periods from 5 Jan 2016 to 13 Nov 2026), as produced by the existing `train.py`:

| Method | MAE (₹/share) | MAPE % | Direction accuracy % (old definition) |
| --- | --- | --- | --- |
| HistGradientBoosting with overnight data | 7.305 | 0.487 | 68.07 |
| Same model without overnight data | 8.349 | 0.562 | 56.62 |
| Baseline: open = last close | 8.322 | 0.561 | 0.00 |
| Baseline: training-mean gap | 8.270 | 0.556 | 58.44 |

These numbers are pooled in rupees across stocks priced from ₹350 to ₹2,600, so they will be restated in gap percentage points in the new evaluation.

## 5. Bugs and technical debt

| ID | Issue | Effect | Fix planned |
| --- | --- | --- | --- |
| B1 | Direction accuracy drops days whose actual gap is exactly 0, and the open = close baseline scores 0% by construction | Not the reviewer's definition; slightly flatters every model | New `metrics.directional_accuracy`: all days count; a zero forecast is correct only if the actual gap is zero |
| B2 | Errors are in rupees pooled across stocks | High-priced stocks dominate; not comparable between stocks | Report MAE/RMSE/bias in gap % (and keep ₹ MAE) overall and per stock |
| B3 | Corporate actions in unadjusted prices: 7 bonus/split days found (HDFCBANK 2019-09-19 and 2025-08-26, INFY 2015-06-15 and 2018-09-04, RELIANCE 2017-09-07 and 2024-10-28, TCS 2018-05-31, each a -50% gap). The target row is dropped, but 5- and 20-day features (`ret_5`, `ret_20`, `dist_sma_20`, `volatility_20`, `vol_ratio_20`) stay distorted for up to 20 sessions afterwards; ICICIBANK's 1:10 bonus (June 2017, about -9%) is under the 20% filter | Noisy features on about 140 rows; a weekly return across an ex-date would read as a -50% week | Weekly module excludes weeks that contain a detected corporate-action gap. The daily feature fix is a model-behaviour change, so it is reported but not applied silently |
| B4 | The 10/90 band is one pair of quantiles for all stocks, from the last fold only, with no coverage check | Users cannot tell whether "80% interval" holds | Report empirical coverage of the band on every fold; show the band prominently with a "not a guaranteed boundary" note |
| B5 | Last walk-forward fold stops one session early (`test < stop` with `stop = last date`) | One test day lost | Fixed in the shared fold generator; reported |
| B6 | `predictions_log.csv` has no model name, timestamp or band in gap terms; on Streamlit Cloud the disk is wiped on restart, so the log is lost | No durable record of live forecasts | Add columns; persist to Supabase when configured |
| B7 | `predict.py` sets the next session as `date + 1 business day`, ignoring NSE holidays (label only; the features use the actual next date in history) | The headline can name a holiday as the forecast date | Documented; low priority |
| B8 | HistGradientBoosting uses absolute-error loss (predicts the median) while Linear Regression will use squared error (the mean) | Bias comparisons must say so | Documented in MODEL_EVALUATION.md |
| B9 | Survivorship bias: the 5 stocks are today's large caps | Back-tested returns are flattering | Documented as a limitation |
| B10 | `env.py` is a hand-written `.env` parser; the reviewer asks for `python-dotenv` | Requirement gap only | Switch to `python-dotenv` with the same precedence, keep the fallback |
| B11 | Minor: pandas/NumPy deprecation warning in `live.py` (timedelta arithmetic) | Future break | Leave unless touched |

## 6. Security review

**Hard-coded credentials: none found.** Every client reads its key from an environment variable (`TWELVEDATA_API_KEY`, `EXCHANGERATE_API_KEY`, `TEJ_API_KEY`, `ALPHAVANTAGE_API_KEY`); Twelve Data and exchangerate.dev send it in a header; Alpha Vantage puts it in the URL but `http_json.py` never prints URLs, and `check_keys.py` prints only whether a key is set and its length.

**Git history: no key found.** All blobs of all 14 commits in the public repository were scanned for API-key patterns (named key assignments, 32-hex strings, long tokens near "key"/"token"/"secret"). No `.env` or `secrets.toml` file was ever committed. The project zips and the D1 charter (.docx/.pdf/.html) were scanned the same way, also clean.

**Where the reviewer's "exposed Twelve Data key" most likely came from** (cannot be confirmed from here):
1. The key was pasted into chat on 23 Oct 2026 (project notes record advice to regenerate it then).
2. The laptop folder `Downloads\capstone-forecast-main` contains a real `.env`. If that folder was zipped and submitted, the key went with it.
3. A different repository or fork than `capstone-forecast`.

**Consequences:** history rewriting of `capstone-forecast` is not needed as far as can be seen, because there is nothing to remove; if the reviewer names another location, use the `git filter-repo` procedure in DEPLOYMENT_GUIDE.md. The key must still be treated as exposed and **rotated by Pri** in the Twelve Data dashboard (no account access here), then the new key placed only in `.env` (laptop) and Streamlit Secrets (cloud).

Other findings: `.gitignore` already excludes `.env` and `.streamlit/secrets.toml`; the public repo means anything committed is world-readable; the Dockerfile disables XSRF protection (`--server.enableXsrfProtection=false`), acceptable for a read-only demo but worth noting.

## 7. Files requiring modification

| File | Change |
| --- | --- |
| `capstone/env.py`, `requirements.txt` | `python-dotenv`; Supabase optional dependency |
| `capstone/train.py` | Shared fold generator; Linear Regression; per-row out-of-sample predictions saved; HGB unchanged |
| `capstone/predict.py` | Richer log (model, timestamp, q10/q90 gap), optional Supabase write |
| `capstone/export.py` | New sheets: model comparison, out-of-sample predictions, weekly analysis, backtests, Monte Carlo |
| `app.py`, `capstone/ui/*` | Reorganised into the six reviewer tabs while keeping the existing tabs |
| `.env.example`, `.streamlit/secrets.toml.example`, README | Supabase variables, docs |

New modules: `metrics.py`, `model_eval.py`, `costs.py`, `weekly.py`, `weekly_model.py`, `backtest.py`, `montecarlo.py`, `fundamental_check.py`, `db.py`; `migrations/001_init.sql`; tests for each.

## 8. Implementation sequence

1. Security: `python-dotenv`, examples, docs, rotation instructions.
2. Evaluation: one fold generator, four models on identical dates, metrics module, per-stock tables, coverage of the band.
3. Weekly analysis and the cost model.
4. Weekly forecast model (walk-forward) and the two backtests.
5. Monte Carlo (block bootstrap by week).
6. Supabase schema, migration, connection module (no-op without credentials).
7. Dashboard tabs and Excel export.
8. Full test run; documentation; demo script.

## 9. Items that need Pri

- Rotate the Twelve Data key (Twelve Data dashboard > API keys) and update `.env` and Streamlit Secrets.
- Say where the reviewer saw the exposed key, if not one of the three places above.
- Create a Supabase project and provide `SUPABASE_URL` and a key through Streamlit Secrets / `.env` (never in chat), then run `migrations/001_init.sql` in the Supabase SQL editor.
- Push: this session cannot write to GitHub. Either connect GitHub to Claude so a branch can be pushed, or upload the delivered files as before.
