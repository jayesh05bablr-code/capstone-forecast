---
title: India Opening Forecast
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# NSE morning forecast and trading decision support

A Streamlit dashboard for investors and analysts who want a structured, data-driven view of selected NSE stocks
before the 09:15 IST open: each stock's expected opening gap with a 10th-90th percentile range and its drivers,
the model's measured accuracy against simple baselines, a weekly trading analysis net of realistic costs,
unfiltered vs cost-aware backtests, and a Monte Carlo risk view. **It is a decision-support tool, not a
profit-generating system or investment advice.**

Live app: https://pri-capstone.streamlit.app

## Executive summary

**Business problem.** Before 09:15 IST an investor has to digest what happened overnight (the US close,
the rupee, ADRs) and how each stock traded the day before, and decide whether to act at the open. Doing this by
hand for every stock each morning is slow and inconsistent.

**Target user and morning decision.** An investor or market analyst following selected NSE large caps who wants a
structured, data-driven view at 08:45 IST: which stocks are likely to open higher or lower, by how much, how
confident to be, and whether a short-term trade is worth its costs.

**Proposed solution.** A Streamlit dashboard that refreshes itself after the US close and after the NSE close:
it forecasts each stock's opening gap with a 10th-90th percentile range and explains the drivers, tracks its own
historical accuracy against simple baselines, analyses weekly trading opportunities net of realistic costs,
backtests a weekly signal, stress-tests it with Monte Carlo simulation, and flags disagreements with company
fundamentals.

**Data sources.** NSE daily OHLCV (TejHQ API, Yahoo Finance fallback); US S&P 500, Nasdaq 100, 11 sector ETFs and
Indian ADRs, and USD/INR history (Twelve Data, Alpha Vantage fallback); live USD/INR (exchangerate.dev);
quarterly results and shareholding (Tracxn exports / company data workbooks). Optional Supabase stores the outputs.

**Model architecture.** Features known at the close plus the overnight US/rupee window up to the next open;
target = next open / close - 1. HistGradientBoosting (existing model) compared with Linear Regression, a naive
"open = last close" baseline and a leakage-free historical-average baseline on identical expanding walk-forward
folds. Intervals from earlier folds' out-of-sample residuals. Driver attribution by what-if neutralisation.

**Model performance vs naive baseline** (13,438 stock-days, 2016-01-05 to 2026-11-12). HistGradientBoosting: MAE 0.486 % points, directional accuracy 66.2%. Naive baseline: MAE 0.560, directional accuracy 2.8% (a zero forecast is only right on a zero gap). Historical average: MAE 0.554, 56.9%. The model cuts the naive error by 13.3%. Linear Regression: MAE 0.488, 65.9%; HistGradientBoosting vs Linear Regression: no significant difference (|t| <= 1.96), so the simpler model is about as good on average. The 10-90% interval covered 70-87% of actual gaps per test period (target 80%).

**Weekly trading analysis.** 3,094 valid stock-weeks (first-session open to last-session close, holiday weeks kept, bonus/split weeks excluded). The weekly forecast has no measurable edge: MAE 2.61 % points vs 2.52 for a zero forecast, directional accuracy 49.3%. With default costs (0.36% per round trip for a ₹200,000 delivery position) on ₹1,000,000: unfiltered strategy 1467 trades, net -50.3%, max drawdown -54.2%; cost-aware strategy 534 trades, net -21.3%, max drawdown -28.1%; holding every stock every week -58.0% (gross P&L ₹790,947, costs ₹1,370,550). The cost-aware filter traded 36% as often as the unfiltered strategy; weekly round trips in these stocks did not cover their costs. Monte Carlo (5,000 block-bootstrap paths of the cost-aware strategy): 84% probability of ending below the starting capital; median return -20.4%, 5th-95th percentile -47% to +16%.

**Business value.** One screen replaces a manual overnight scan: a consistent, explainable opening
view with an honest error range, evidence on when the model beats simple rules, and a cost-aware check that
stops the user from chasing short-term trades whose expected edge is smaller than their costs.

**Limitations.** Decision support only, not a guaranteed profit system or investment advice. Earnings, corporate
announcements, geopolitical shocks, liquidity and regime changes can break any forecast; historical and simulated
results do not guarantee future performance; the five stocks are survivors (survivorship bias); cost rates are
assumptions to verify with a broker.

## Capstone documents

| Document | Contents |
| --- | --- |
| [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) | One-page business case with measured results |
| [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) | Audit before the upgrade: architecture, gaps, bugs, security review |
| [MODEL_EVALUATION.md](MODEL_EVALUATION.md) | Naive, historical average, Linear Regression, HistGradientBoosting on identical walk-forward days |
| [BACKTESTING_METHODOLOGY.md](BACKTESTING_METHODOLOGY.md) | Weekly delta, NSE calendar and holidays, cost model, weekly model, backtests, Monte Carlo |
| [DATABASE_SCHEMA.sql](DATABASE_SCHEMA.sql) | Supabase/PostgreSQL schema (same as `migrations/001_init.sql`) |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Local setup on Windows, secrets, Streamlit Cloud, Supabase, key rotation, git history |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | Three-minute demo |

## Dashboard tabs

| Tab | What it shows |
| --- | --- |
| Morning forecast | Stock selector, predicted gap and opening price, 10th/90th percentile, direction, drivers, model, last update |
| Model accuracy | MAE, RMSE, bias, directional accuracy; baselines vs Linear Regression vs HistGradientBoosting; significance tests; predicted vs actual; scatter; daily error; rolling MAE and accuracy; stock-wise tables; interval coverage. Filters: stock, dates, models |
| Weekly analysis | Weekly delta chart per stock with the round-trip cost line (green above, grey below), tradable weeks %, weekly statistics, cost breakdown |
| Backtesting | Unfiltered vs cost-aware strategy vs hold-all: trades, win rate, gross/net P&L, costs, return, drawdown, Sharpe, equity and drawdown curves, trade lists |
| Monte Carlo | 5,000 block-bootstrap paths: ending capital, drawdown and return distributions, probability of loss, percentiles, backtest vs simulation |
| Business insights | Forecast direction vs fundamental trend with disagreement flags, executive summary, limitations |
| Ask the market | Chat that answers with NSE's official data (live quotes, daily history, movers, breadth) and the app's own forecast, through a free Groq or Gemini key |
| More tools | The earlier tabs: Live, US sectors and India, Rupee, Invest planner, Intraday scalping, Company profile, Upload company data, Download data (Excel) |

Trading assumptions (capital, allocation, number of stocks, brokerage, STT, stamp duty, exchange and DP charges,
spread, slippage, cost-filter multiple) are in the sidebar and drive the last three analysis tabs together.

## Quick start

```powershell
py -3 -m venv .venv; .\.venv\Scripts\Activate.ps1      # Mac/Linux: python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
copy .env.example .env                                  # then paste your keys into .env
python -m capstone.train                                # evaluation, weekly analysis, backtests, Monte Carlo
python -m capstone.predict
python -m pytest -q tests
streamlit run app.py
```

Full steps, Streamlit Cloud and Supabase: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## New modules (reviewer upgrade)

| Path | What it does |
| --- | --- |
| `capstone/metrics.py` | MAE, RMSE, bias, directional accuracy, drawdown, Sharpe, rolling scores, Diebold-Mariano test |
| `capstone/model_eval.py` | Four forecasters on the shared walk-forward folds; out-of-sample table; interval coverage |
| `capstone/costs.py` | Configurable trading costs: explicit charges, spread and slippage, delivery vs intraday |
| `capstone/weekly.py` | First-open to last-close weekly returns on the observed NSE calendar, exclusions, tradable weeks |
| `capstone/weekly_model.py` | Weekly return forecast from data before the entry open, quarterly-refit walk-forward |
| `capstone/backtest.py` | Unfiltered vs cost-aware weekly backtests with equity-based sizing |
| `capstone/montecarlo.py` | 5,000-path moving-block bootstrap of weekly net returns |
| `capstone/fundamental_check.py` | Point-in-time revenue/profit trend vs forecast direction |
| `capstone/db.py`, `migrations/` | Optional Supabase persistence (REST, idempotent upserts) |
| `capstone/analysis.py`, `capstone/summary.py` | Default analysis run after training; executive summary from measured files |
| `capstone/charts_eval.py`, `capstone/ui/tab_*.py` | Charts and the new dashboard tabs |

---

## Original project notes: India market opening forecast

Predicts each NSE stock's next opening price from:

- the previous day's OHLCV (TejHQ API) and company fundamentals (CSV now; Bloomberg/Tracxn later), and
- **what happened overnight before India opens**: the US close (S&P 500, Nasdaq 100, the 11 US
  sector ETFs, and US-listed Indian ADRs, from Alpha Vantage) and the rupee (USD/INR, from
  Alpha Vantage history plus exchangerate.dev for the latest days).

The dashboard explains each forecast in plain words: which US sectors moved, how they map to
Indian sectors, how the rupee move helps or hurts each company, and how much of the forecast
comes from each factor.

## Use it online, 24x7 (no VS Code, no terminal)

Host the dashboard free on **Streamlit Community Cloud** from a private GitHub repository, then
open it from a bookmark on your phone or laptop. The app keeps itself up to date: when someone
opens it and an update is due (08:45 IST after the US close, 16:15 IST after NSE closes) it
fetches prices, retrains weekly, and predicts in the background. Keys go in the app's **Secrets**
box (format in `.streamlit/secrets.toml.example`), never in GitHub. Step-by-step guide: see the
"Put the forecast online" guide shared with this project.

Free-plan limits to know: an app nobody opens for about 12 hours goes to sleep and wakes with one
click (about a minute); the server's disk is wiped when it restarts, so the first visit after a
restart rebuilds the data (a few minutes).

## Live tab, buyers vs sellers, and Zerodha Kite settings

- **Live** tab: 1-minute prices from Yahoo Finance (`yfinance`, no key) that refresh every 15 to
  120 seconds while NSE is open; the live status of each opening-range setup (range forming,
  armed, in trade, target hit, stopped out, no trigger) with a plain-words suggestion.
- **Buyers vs sellers**: volume that traded on upticks vs downticks, per minute and for the day.
  Yahoo does not publish the order book, so this is an estimate (tick rule), not Kite's market depth.
- **Kite settings** (`capstone/kite.py`), on the Live, Intraday scalping and Invest planner tabs:
  - *Alert + ATO* for intraday: "If Last price of SYM (NSE) is >= X", then a BUY (or SELL) MIS LIMIT
    order, plus the SL-M stop and LIMIT target to place once the entry fills.
  - *GTT* for longs: Buy, Single, CNC, LIMIT, trigger, price, quantity; then a Sell OCO GTT with the
    stop and target legs. Kite's equity GTT is CNC (delivery), so it suits the multi-day Invest
    planner more than a 20-minute scalp, and it cannot short.
  Nothing is ever sent to Zerodha: you copy the values into Kite yourself.

## Keys

Copy `.env.example` to `.env` in the project folder and paste your keys after the `=` signs
(`run_all.bat` does the copy and opens Notepad for you). The `.env` file stays on your computer:
it is git-ignored, never printed, and must not be shared. Values in `.env` override
environment variables of the same name.

| Variable | Used for |
| --- | --- |
| `TWELVEDATA_API_KEY` | US ETFs, ADRs and USD/INR history (full history on a free key) |
| `EXCHANGERATE_API_KEY` | Latest USD/INR |
| `TEJ_API_KEY` | NSE prices (optional; NSE's own MCP server is used first and needs no key) |
| `GROQ_API_KEY` or `GEMINI_API_KEY` | Optional: turns on the "Ask the market" chat (free keys at console.groq.com/keys or aistudio.google.com/apikey). `LLM_PROVIDER` (groq/gemini) and `LLM_MODEL` can pick a different model |
| `ALPHAVANTAGE_API_KEY` | Optional fallback when no Twelve Data key is set |

Check them with `python -m capstone.check_keys` (shows only whether each is set and works).
If a network blocks outbound calls, allow `mcp.nseindia.in`, `api.tejhq.dev`, `api.twelvedata.com` and
`api.exchangerate.dev` (plus `api.groq.com` or `generativelanguage.googleapis.com` for the chat).

## NSE's official MCP servers (no key, no IP whitelist)

`capstone/nse_mcp.py` talks to NSE's public MCP servers with the standard library only:

| Server | URL | Used for |
| --- | --- | --- |
| Bhavcopy | `https://mcp.nseindia.in/bhavcopy/cm/mcp` | Daily OHLCV for training and the previous close (official bhavcopy, about 5 years, unadjusted) |
| Cash market | `https://mcp.nseindia.in/cmmkt/mcp` | Live tab quotes and NIFTY 50 / NIFTY BANK / INDIA VIX (1-3 minute delay) |

Price order in `fetch_data`: NSE bhavcopy first; TejHQ, then Yahoo Finance fill anything NSE lacks (older than
5 years, and a gap NSE's server has for most of 2023). For any date NSE has, NSE's row wins. The Live tab shows
NSE's quotes on top and keeps Yahoo's 1-minute candles for the buyer/seller split and breakout setups. The
chat's language model only fetches and explains; the forecast always comes from the trained model.

## Quick start (original step-by-step commands)

One command: `./run_all.sh` (Mac/Linux) or double-click `run_all.bat` (Windows). It installs the
packages and opens the dashboard, which fetches, trains and predicts by itself (first start: a few
minutes). The manual steps below still work:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env                               # Mac: cp; then paste your keys into .env
python -m capstone.check_keys

python -m capstone.fetch_data --start 2015-01-01   # 1. NSE prices
python -m capstone.fetch_overnight                  # 2. US close + USD/INR
cp data/fundamentals_template.csv data/fundamentals.csv  # 3. fundamentals (optional)
python -m capstone.train                            # 4. walk-forward validation + save model
python -m capstone.predict                          # 5. next open + reasons
streamlit run app.py                                # 6. dashboard
```

**Daily routine:** after 4 pm IST run `fetch_data`. Next morning between about 02:30 IST (US
close) and 09:00 IST run `fetch_overnight`, then `predict`, then open the dashboard. Retrain weekly.

**Twelve Data free limits:** about 8 calls a minute and 800 a day; each call returns up to 5000
days, so the first run fetches full history (about 19 calls, 3 minutes).

**Alpha Vantage free limits** (only if you use it instead): about 25 calls a day, 5 a minute, and only the latest 100 days
per US series. `fetch_overnight` makes about 18 calls, waits between them, caches everything in
`data/us/` and `data/fx/`, and adds new days on each run, so history grows over time. For years of
US history at once, either use a premium key or drop CSVs with `date,close` columns into
`data/us/{TICKER}.csv`.

## Test without internet (synthetic sample)

```bash
python -m tests.make_synthetic_overnight   # fake prices, US moves and USD/INR
python -m capstone.train && python -m capstone.predict && streamlit run app.py
python -m pytest tests                     # checks the overnight window never peeks ahead
```

The sample has the Indian gaps built from the fake US and rupee moves, so the model will look
very accurate on it. That is by design and says nothing about real accuracy. The dashboard shows
a red banner while sample data is present. Before real data, delete `data/raw`, `data/us`,
`data/fx` and `data/SYNTHETIC_DATA`.

## How the overnight window works

NSE closes at 15:30 IST. The US session dated the same day closes at 01:30 or 02:30 IST, before
NSE opens at 09:15. For a close on day d and next open on day n, the model uses every US session
dated d to n-1 (several after a weekend or an Indian holiday, none on a US holiday). Moves are
measured from the last close before d to the last close before n, so nothing after the open leaks
in. `tests/test_overnight.py` checks this.

## Mappings you can edit

- `data/stock_map.csv`: NSE symbol, Indian sector, US ADR ticker (if any).
- `data/sector_links.csv`: Indian sector, matching US sector ETF, rupee exposure (+1 exporter,
  -1 importer) and the plain-language reasons shown on the dashboard.

## Layout

| Path | What it does |
| --- | --- |
| `capstone/tej_client.py` | NSE OHLCV from TejHQ; key from `TEJ_API_KEY` |
| `capstone/twelvedata_client.py` | US daily bars and USD/INR; key from `TWELVEDATA_API_KEY` (header, never in URLs) |
| `capstone/alphavantage_client.py` | US daily bars and FX_DAILY; key from `ALPHAVANTAGE_API_KEY` |
| `capstone/exchangerate_client.py` | Latest and recent USD/INR; key from `EXCHANGERATE_API_KEY` |
| `capstone/fetch_data.py` | Saves/tops up `data/raw/{SYMBOL}.csv` |
| `capstone/fetch_overnight.py` | Saves/tops up `data/us/*.csv` and `data/fx/USDINR.csv` |
| `capstone/overnight.py` | Lines up US and rupee moves with each India close-to-open window |
| `capstone/fundamentals.py` | CSV loader + point-in-time join |
| `capstone/features.py` | Price, overnight and fundamental features, and the target (overnight gap) |
| `capstone/train.py` | Walk-forward validation vs baselines and vs the same model without overnight data |
| `capstone/explain.py` | What-if attribution per driver and the plain-language reasons |
| `capstone/predict.py` | Next-open predictions, 10-90% band, reasons, market story |
| `capstone/planner.py` | Invest planner: split, ATR stop loss, target, backtested expected profit |
| `capstone/tracxn.py` | Reads Tracxn profile PDFs and saves checked numbers to `data/fundamentals.csv` |
| `capstone/uploads.py` | CSV/Excel upload of fundamentals, shareholding (FII/DII), financials, investors, profiles |
| `capstone/autorun.py` | Keeps the forecast fresh: runs fetch, train (weekly) and predict when an update is due |
| `capstone/live.py` | Yahoo Finance 1-minute data, buyer/seller volume estimate, live setup status |
| `capstone/kite.py` | Zerodha Kite alert + ATO and GTT fields for a plan (never places orders) |
| `capstone/scalping.py` | Intraday opening-range planner, NSE charges, 1-minute backtest |
| `capstone/charts.py` | Candlestick and line charts (Plotly) |
| `capstone/export.py` | Builds the Excel workbook on the Download data tab |
| `capstone/evaluation.py` | Walk-forward accuracy summary and past forecasts vs actual opens |
| `capstone/company.py` | Company profile facts, shareholding and investor classification |
| `app.py` | Streamlit dashboard entry point: loads the forecast and lays out the tabs |
| `capstone/ui/` | One module per dashboard tab (`tab_*.py`), the headline (`headline.py`) and shared helpers (`common.py`). UI code only; the calculations live in the modules above |
| `tests/` | Synthetic data generators, the overnight-window tests and unit tests for the dashboard calculations |
