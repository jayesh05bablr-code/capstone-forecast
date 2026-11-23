# Three-minute demo script

Open https://pulse8.streamlit.app (or `streamlit run app.py`) before you start, wait for it to finish
loading, and leave the sidebar collapsed. The numbers below are from the 17 Nov 2026 run; say the numbers shown on
screen if they have changed.

## 0:00-0:30 Business problem and target user

*Screen: page title, headline metrics and "Why India should open this way".*

> "Every morning before 9:15, an investor following NSE large caps has to make sense of what happened overnight:
> Wall Street, the rupee, the ADRs of Indian banks and IT companies. Doing that by hand for every stock is slow and
> inconsistent. Our dashboard is built for that investor or analyst. It updates itself at 8:45 and turns the
> overnight news into a structured opening view for each stock. It supports a decision; it does not make one."

## 0:30-1:10 Morning forecast, interval and drivers

*Screen: **Morning forecast** tab, pick ICICIBANK (or the stock with the largest gap).*

> "For each stock we forecast the opening gap: today ICICIBANK is expected to open about 0.2% lower, around
> ₹1,580. The two boxes next to it are the 10th and 90th percentile: eight times out of ten the open should land
> between them. The range comes from the model's own out-of-sample errors, so it is an estimate, not a guaranteed
> boundary: news can push the open outside it. Below, the driver chart shows what moved the forecast: here mostly
> the stock's own recent trading, plus a weak US close, partly offset by its US-listed ADR. We also show which
> model made the forecast and when it was last updated."

*(Drivers change every day: read the largest bars on screen.)*

## 1:10-1:40 Historical accuracy vs the naive baseline

*Screen: **Model accuracy** tab: metrics table and the two bar charts, then scroll to the predicted vs actual chart.*

> "Is it any better than guessing? We tested four forecasters on exactly the same 13,438 stock-days from 2016 to
> 2026, always training only on the past. The naive rule, open equals yesterday's close, has an average error of
> 0.56 percentage points. Our gradient-boosting model cuts that to 0.49, a 13% improvement, and gets the direction
> right 66% of the time; the difference is statistically significant. An honest finding: plain Linear Regression
> does about as well. The trees are not significantly better, and we say so."

## 1:40-2:20 Weekly analysis, costs and backtesting

*Screen: **Weekly analysis** (one chart with the red cost line), then **Backtesting** (equity curves).*

> "We also asked a trading question: buy at Monday's open, sell at Friday's close. Each bar is one week, green
> when it beat the full round-trip cost of 0.36%: brokerage, STT, stamp duty, exchange fees, GST, spread and
> slippage. Fewer than half the weeks clear it. We built a separate weekly forecast and backtested two
> strategies on the same 464 weeks and ₹10 lakh: trade every positive forecast, or only when the forecast is at
> least twice the cost. Both were profitable before costs and lost money after them. The cost-aware filter traded
> a third as often and lost less than half as much. The weekly forecast itself has no measurable edge, and the
> dashboard shows that instead of hiding it."

## 2:20-2:45 Monte Carlo and risk

*Screen: **Monte Carlo** tab: probability of loss and the ending-capital histogram.*

> "To see the range of outcomes we resampled the cost-aware strategy's weekly results 5,000 times, in blocks of
> weeks so that correlated trades stay together. In 84% of simulated paths the capital ends below where it
> started; the median drawdown is about 30%. These paths are hypothetical, not a forecast."

## 2:45-3:00 Business value, limitations, next steps

*Screen: **Business insights** tab: fundamentals table with the disagreement flag.*

> "The value is a fast, consistent and explainable morning view, honest about its accuracy, and a cost check that
> stops users from over-trading. It also flags when the forecast disagrees with company fundamentals, like
> ICICIBANK here. Limitations: surprise news, earnings and regime changes, and past results do not guarantee the
> future. Next, we would add more stocks, an earnings calendar and live Supabase history. Thank you."
