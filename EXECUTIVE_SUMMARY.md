# Executive summary

_Stock market forecasting and trading decision support for selected NSE stocks. Every figure below is measured by the project's own walk-forward evaluation and backtests (generated with `python -m capstone.summary` on 17 Nov 2026 from real data, 1 Jan 2015 to 13 Nov 2026). Details: MODEL_EVALUATION.md and BACKTESTING_METHODOLOGY.md._

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
