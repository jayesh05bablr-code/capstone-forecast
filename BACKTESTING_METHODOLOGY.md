# Weekly analysis, trading costs, backtesting and Monte Carlo: methodology and results

All results come from `python -m capstone.train` (which runs `capstone.analysis`) on 17 Nov 2026, using real NSE
prices for RELIANCE, TCS, HDFCBANK, INFY and ICICIBANK from 1 Jan 2015 to 13 Nov 2026 and the default
assumptions below. The dashboard recomputes everything live when you change the assumptions in the sidebar.

## 1. Weekly delta (`capstone/weekly.py`)

`Weekly delta = (exit close - entry open) / entry open`

- **Trading calendar.** A date is an NSE session if the price file has a bar for it. This reflects every exchange
  holiday and special session (e.g. Saturday budget or Muhurat sessions) without a hand-typed holiday list.
- **Weeks** are calendar weeks, Monday to Sunday.
- **Entry** = open of the first session of the week (Tuesday if Monday is a holiday). **Exit** = close of the last
  session of the same week (Thursday if Friday is a holiday).
- **Holiday-shortened weeks are kept** and flagged: 755 of 3,100 stock-weeks had fewer than 5 sessions
  (sessions per week: 2 → 5 weeks, 3 → 80, 4 → 670, 5 → 2,295, 6 → 50).
- **Excluded weeks:** missing or non-positive entry/exit price; a bonus or split ex-date after the entry session
  (prices are unadjusted, so the week would show a fake loss); the week still in progress at the end of the data.
  Six stock-weeks were excluded, all corporate actions:

  | Stock | Week | Raw return | Reason |
  | --- | --- | ---: | --- |
  | RELIANCE | 2017-09-04 | -49.4% | bonus 1:1 |
  | TCS | 2018-05-28 | -52.1% | bonus 1:1 |
  | INFY | 2018-09-03 | -49.4% | bonus 1:1 |
  | HDFCBANK | 2019-09-16 | -46.7% | split 2:1 |
  | HDFCBANK | 2025-08-25 | -51.5% | bonus 1:1 |
  | ICICIBANK | 2017-06-19 | -8.2% | bonus 1:10 (listed in `data/corporate_actions.csv`, below the 20% automatic filter) |

  Ex-dates that fall on the entry session itself (INFY 2015-06-15, RELIANCE 2024-10-28) do not distort the week,
  because the entry open is already ex-bonus.
- **No future prices** are used to decide an entry. This tab only describes realised weeks; the trading signal
  comes from the weekly model (section 3).

**Tradable Weeks (%)** = valid weeks whose realised return exceeds the round-trip cost ÷ all valid weeks × 100.

Results with the default round-trip cost of 0.355% (one ₹2,00,000 delivery position):

| Stock | Valid weeks | Tradable weeks | Tradable % | Average | Median | Best week | Worst week |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HDFCBANK | 618 | 292 | 47.3% | +0.17% | +0.21% | 2020-03-23: +13.8% | 2020-03-16: -14.7% |
| ICICIBANK | 619 | 282 | 45.6% | +0.22% | +0.14% | 2016-02-29: +19.3% | 2020-03-16: -18.5% |
| INFY | 619 | 300 | 48.5% | +0.22% | +0.24% | 2020-03-23: +20.9% | 2026-04-20: -12.5% |
| RELIANCE | 619 | 299 | 48.3% | +0.27% | +0.28% | 2020-03-23: +16.3% | 2018-10-01: -16.6% |
| TCS | 619 | 283 | 45.7% | +0.10% | +0.09% | 2026-10-12: +12.8% | 2020-03-09: -13.0% |

Fewer than half of all weeks beat the cost of a round trip, even though the average week is positive.

## 2. Trading cost model (`capstone/costs.py`)

One model is used for every strategy (the intraday scalping tool now uses the same code with the intraday profile,
with identical results to before).

| Component | Delivery (CNC, Monday-Friday hold) | Intraday (MIS) | Default assumption |
| --- | --- | --- | --- |
| Brokerage | both legs | both legs | 0.03% of order value, capped at ₹20 per order |
| Exchange transaction charge | both legs | both legs | 0.00297% |
| SEBI turnover fee | both legs | both legs | 0.0001% (₹10 per crore) |
| GST | on brokerage + exchange + SEBI | same | 18% |
| STT | buy and sell | sell only | 0.1% each side (delivery); 0.025% sell (intraday) |
| Stamp duty | buy | buy | 0.015% (delivery); 0.003% (intraday) |
| DP charge | per sell | none | ₹15.93 + GST |
| Bid-ask spread | half spread per leg | same | 0.02% per leg |
| Slippage | per leg | same | 0.03% per leg |

**These rates are assumptions taken from published discount-broker charge sheets, not verified current statutory
rates.** They change over time and by broker; check them against a recent contract note and override them in the
dashboard sidebar. Nothing fetches live rates.

The three cost types are reported separately. Example, one ₹2,00,000 round trip:

| Cost | ₹ | % of position |
| --- | ---: | ---: |
| Brokerage (both legs) | 40.00 | 0.020 |
| Exchange transaction charges | 11.88 | 0.006 |
| SEBI turnover fee | 0.40 | 0.000 |
| GST | 9.41 | 0.005 |
| STT | 400.00 | 0.200 |
| Stamp duty (buy) | 30.00 | 0.015 |
| DP charge incl. GST (sell) | 18.80 | 0.009 |
| **Explicit charges** | **510.49** | **0.255** |
| Bid-ask spread | 80.00 | 0.040 |
| Slippage | 120.00 | 0.060 |
| **Total round trip** | **710.49** | **0.355** |

STT on both legs is more than half of the cost of a delivery round trip.

User-adjustable: initial capital, allocation per stock, brokerage %, brokerage cap, STT buy/sell, stamp duty,
exchange charge, DP charge, spread, slippage, the stocks in the universe (number of stocks) and the cost-filter
multiple.

## 3. Weekly forecast signal (`capstone/weekly_model.py`)

The daily model forecasts one overnight gap; it is **not** reused as a weekly forecast. A separate model forecasts
the weekly delta from information available before the entry session's 09:15 open:

| Group | Features |
| --- | --- |
| Stock | 1-, 4- and 13-week return, 20-day volatility, distance from 50-day average, last day's return, previous week's open-to-close return (closes back-adjusted for the listed bonus/split ex-dates) |
| Market | equal-weight average of the stock features across the universe |
| US / FX | S&P 500 last session and 1-week return, Nasdaq 100 1-week return, the stock's US sector ETF 1-week return, USD/INR 1-week change (all from sessions dated before the entry date) |
| Calendar | sessions in the week (NSE publishes its holiday calendar in advance) |

**Walk-forward:** the first 156 weeks (three years) are training only; the models are refitted every 13 weeks on
all weeks whose exit came before the block's first entry, then forecast the next 13 weeks. 464 test weeks
(25 Dec 2017 to 9 Nov 2026), 2,316 stock-weeks. HistGradientBoosting (same settings as the daily model) was fixed
in advance as the backtest signal; the dashboard can switch to the others.

| Model | MAE (% pts) | RMSE (% pts) | Bias | Directional accuracy |
| --- | ---: | ---: | ---: | ---: |
| Zero (no edge) | 2.520 | 3.430 | -0.203 | 0.1% |
| Historical average week | 2.523 | 3.431 | +0.035 | 52.1% |
| Linear Regression | 2.577 | 3.526 | -0.056 | 51.6% |
| HistGradientBoosting | 2.611 | 3.533 | +0.036 | 49.3% |

**Finding: the weekly signal has no measurable edge.** No model beats the zero forecast on MAE, and directional
accuracy is at coin-flip level (correlation of forecasts with outcomes between -0.05 and -0.02). The data and
features are sufficient to *test* a weekly model honestly; they are not sufficient to *find* one. This is
reported, not hidden: the backtests below therefore measure mostly costs and market drift. Possible extensions
(not implemented, so no claims are made): longer history, more stocks, sector-relative or earnings-calendar
features, and options-implied volatility.

## 4. Backtests (`capstone/backtest.py`)

Shared rules for both strategies:

- Same 464 test weeks, same ₹10,00,000 starting capital, same 5 stocks, same cost assumptions.
- Long only (a cash-market short cannot be carried for a week in India).
- Entry at the week's first-session open, exit at its last-session close. Positions open and close inside the
  same week, so trades never overlap across weeks and capital is back in cash before the next entry.
- **Position sizing:** each trade gets allocation × equity at the start of the week (default 20% = 1/5), whole
  shares only; total exposure never exceeds 100% of equity (the allocation is capped at 1 / number of stocks if a
  larger value is entered). Equity compounds week to week.
- Costs are charged on the actual buy and sell values of every trade.
- Duplicate trades are impossible: at most one trade per stock per week (tested).

**Strategy A, unfiltered:** trade every stock with a positive weekly forecast.
**Strategy B, cost-aware:** trade only when the forecast exceeds 2 × the estimated round-trip cost of that
position (known before entry).
**Reference:** hold every stock every valid week (shows what the signal adds).

| | A. Unfiltered | B. Cost-aware | Reference: hold all |
| --- | ---: | ---: | ---: |
| Trades | 1,467 | 534 | 2,316 |
| Winning / losing trades | 659 / 808 | 246 / 288 | 1,073 / 1,243 |
| Win rate (after costs) | 44.9% | 46.1% | 46.3% |
| Gross P&L | ₹3,49,838 | ₹1,25,070 | ₹7,90,947 |
| Total transaction costs | ₹8,52,669 | ₹3,37,726 | ₹13,70,550 |
| Net P&L | -₹5,02,831 | -₹2,12,656 | -₹5,79,603 |
| Ending capital | ₹4,97,169 | ₹7,87,344 | ₹4,20,397 |
| Cumulative return | -50.3% | -21.3% | -58.0% |
| Maximum drawdown | -54.2% | -28.1% | -66.4% |
| Sharpe ratio (annualised, weekly) | -0.54 | -0.29 | -0.50 |

**Interpretation.** Every strategy was profitable before costs and lost money after them: weekly round trips cost
about 0.36% each, while the average week returned 0.10-0.27%. The cost-aware filter cut the number of trades by
64% and the loss by more than half, which is the filter doing its job. It cannot create an edge the forecast does
not have. The business conclusion is that a weekly in-and-out strategy in these large caps is not worthwhile at
retail delivery costs; the cost-aware rule protects the user from over-trading.

Sharpe ratio uses weekly returns including cash weeks, annualised by √52, risk-free rate 0. With returns this weak
and noisy it is indicative only.

## 5. Monte Carlo (`capstone/montecarlo.py`)

- **What is resampled:** Strategy B's 464 realised **weekly** net portfolio returns, cash weeks (0%) included.
  The week is the unit because the stocks traded in the same week share the same market moves; resampling single
  trades would treat them as independent and understate risk.
- **How:** moving-block bootstrap, blocks of 4 consecutive weeks (keeps short runs of good or bad weeks together),
  5,000 paths, each 464 weeks long (also 52 or 156 weeks in the dashboard), compounded from ₹10,00,000, seed 42.
- **Per path:** ending capital, cumulative return, maximum drawdown; across paths: probability of ending below the
  starting capital and the 5th/25th/50th/75th/95th percentiles.

| Outcome | 5th | 25th | 50th | 75th | 95th |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ending capital | ₹5,30,051 | ₹6,77,208 | ₹7,96,284 | ₹9,31,468 | ₹11,58,341 |
| Cumulative return | -47.0% | -32.3% | -20.4% | -6.9% | +15.8% |
| Maximum drawdown | -51.1% | -39.3% | -30.4% | -22.6% | -14.0% |

Probability of ending below starting capital: **84.2%**. The historical backtest (-21.3%, drawdown -28.1%) sits
close to the simulated median, as it should when the paths are built from its own weeks.

**Limitations.** The simulation only reshuffles the backtest period: crashes, regime changes or cost changes that
did not happen in it cannot appear. It inherits every backtest assumption (fills exactly at the open and close,
the cost rates, no liquidity limits). **The results are hypothetical; they are not a forecast and not a guarantee
of future performance.**

## 6. Bias and error checks

| Risk | Control | Test |
| --- | --- | --- |
| Look-ahead in signals | features from bars and US/FX sessions dated before the entry date; training weeks must have exited before the test block | `test_weekly_features_use_only_data_before_entry` |
| Incorrect date alignment | entry/exit taken from each stock's own sessions | `test_weekly_return_uses_first_open_and_last_close`, `test_holiday_shortened_week_monday_and_friday_holidays` |
| Corporate actions | ex-date weeks excluded; features back-adjusted | `test_week_in_progress_and_bad_prices_and_corporate_action_excluded`, `test_known_ex_date_excludes_small_bonus` |
| Wrong costs | each component tested; buy-only stamp duty, delivery vs intraday STT | `test_delivery_cost_components_add_up`, `test_intraday_has_no_buy_stt_and_brokerage_cap` |
| Position sizing / compounding | whole shares from start-of-week equity | `test_backtest_position_sizing_and_compounding_without_costs` |
| Duplicate trades | one per stock per week; Supabase primary key (run, stock, week) | same test; `test_save_analysis_is_idempotent` |
| Same dates for both strategies | every strategy reports all test weeks | `test_cost_aware_filter_and_costs_deducted` |
| Monte Carlo reproducibility | fixed seed; block structure | `test_monte_carlo_reproducible_and_consistent`, `test_block_bootstrap_keeps_consecutive_weeks` |
| Survivorship bias | not controlled: today's large caps only | documented limitation |
