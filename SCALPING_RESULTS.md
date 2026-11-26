# Intraday scalping: walk-forward results on real NSE minute data

Pulse8's main page is now **Intraday scalping**. This file records what its strategy test found on real data,
so the numbers in the capstone report can be checked. Code: `capstone/scalp_lab.py`; tests: `tests/test_scalp_lab.py`.

## Data

* Yahoo Finance chart data for RELIANCE, TCS, HDFCBANK, INFY and ICICIBANK, fetched on 26 Nov 2026.
* 1-minute bars: 30 Oct to 25 Nov 2026 (17 full sessions). 5-minute bars: 31 Aug to 25 Nov 2026 (58 sessions).
* The first, partial day and the unfinished session of 26 Nov are left out.

## Method

* Setups, each trading both **long (buy)** and **short (sell first, buy back the same day)**:
  opening-range breakout, VWAP trend (EMA 9/21 cross on the right side of VWAP), VWAP stretch reversal, and a
  coin-flip baseline with identical stops, targets and charges.
* A setup fires on a bar's close; the trade enters at the **next bar's open**. Stop = 1 to 2 × ATR, target = 1 to 2 ×
  risk. Stop is assumed hit first when one bar touches both; a bar opening beyond the stop exits at that open.
  No new trades after 14:45, everything closed by 15:15, at most 3 trades per stock per day.
* Size from the beginner limits on the safety page (₹50,000 capital, 1% risk per trade, at most 50% of capital
  in one trade, no leverage).
* Costs per round trip (`capstone/costs.py`, intraday profile): brokerage 0.03% capped at ₹20 per order, STT 0.025%
  on the sell, NSE transaction charges, SEBI fee, GST, stamp duty 0.003% on the buy, plus half the bid-ask spread
  (0.02%) and slippage (0.03%) on each leg. About **0.21%** of the trade value at this size.
* **Walk-forward:** each test day uses the settings that made the most after costs over the previous 10 sessions.
  Only those unseen days count.
* Verdict "profitable" needs at least 30 trades **and** the lower end of a bootstrap 90% range of the average net
  return per trade above zero.

## Results (out-of-sample only)

**1-minute bars, 17 trading days of 1-minute history (7 test days)**

| Setup                        |   Trades |   Buys |   Short sells |   Win rate % |   Avg before costs % |   Avg costs % |   Avg after costs % |   Net ₹ after costs |
|:-----------------------------|---------:|-------:|--------------:|-------------:|---------------------:|--------------:|--------------------:|--------------------:|
| Opening-range breakout       |       41 |     20 |            21 |       14.634 |               -0.027 |         0.206 |              -0.233 |            -2331.89 |
| Coin-flip entries (baseline) |      101 |     54 |            47 |       18.812 |                0.037 |         0.206 |              -0.169 |            -4150.74 |
| VWAP stretch reversal        |      105 |     54 |            51 |       14.286 |                0.013 |         0.206 |              -0.193 |            -4924.65 |
| VWAP trend (EMA 9/21 cross)  |      104 |     53 |            51 |       21.154 |               -0.003 |         0.206 |              -0.209 |            -5273.14 |

**5-minute bars, 17 trading days of 1-minute history (7 test days)**

| Setup                        |   Trades |   Buys |   Short sells |   Win rate % |   Avg before costs % |   Avg costs % |   Avg after costs % |   Net ₹ after costs |
|:-----------------------------|---------:|-------:|--------------:|-------------:|---------------------:|--------------:|--------------------:|--------------------:|
| Opening-range breakout       |       37 |     20 |            17 |       35.135 |                0.021 |         0.206 |              -0.185 |            -1670.47 |
| Coin-flip entries (baseline) |       46 |     26 |            20 |       36.957 |                0.018 |         0.206 |              -0.188 |            -2095.73 |
| VWAP trend (EMA 9/21 cross)  |       55 |     29 |            26 |       25.455 |               -0.03  |         0.206 |              -0.237 |            -3136.83 |
| VWAP stretch reversal        |       64 |     40 |            24 |       18.75  |               -0.007 |         0.206 |              -0.213 |            -3315.15 |

**5-minute bars, 58 trading days of 5-minute history (48 test days)**

| Setup                        |   Trades |   Buys |   Short sells |   Win rate % |   Avg before costs % |   Avg costs % |   Avg after costs % |   Net ₹ after costs |
|:-----------------------------|---------:|-------:|--------------:|-------------:|---------------------:|--------------:|--------------------:|--------------------:|
| Opening-range breakout       |      268 |    134 |           134 |       30.224 |               -0.005 |         0.206 |              -0.211 |            -13728.6 |
| Coin-flip entries (baseline) |      269 |    125 |           144 |       21.19  |               -0.03  |         0.206 |              -0.236 |            -15435.2 |
| VWAP trend (EMA 9/21 cross)  |      382 |    181 |           201 |       28.272 |               -0.007 |         0.206 |              -0.213 |            -19808.9 |
| VWAP stretch reversal        |      364 |    234 |           130 |       22.802 |               -0.044 |         0.206 |              -0.25  |            -22158.3 |

## What this means

* **No setup made money after costs, long or short, at any bar size.** None clearly beat the coin-flip baseline.
* Look at "Avg before costs %": the raw edge of every setup is between about -0.04% and +0.04% per trade, i.e. close to
  zero. A round trip costs about 0.21%. A scalping rule needs a raw edge several times larger than that, and these
  simple, widely known rules do not have one on these large, liquid stocks.
* Even the best settings picked with hindsight over all days lost money (shown on the page for contrast).
* Win rates of 15% to 35% are not a bug: with costs of about 0.2%, many trades that move the right way still end
  below zero after charges.
* With only 17 sessions of 1-minute data the 1-minute tables are thin; the 5-minute table (48 test days, about 1,300
  trades) is the main evidence. Fetching data every day lets the page's test grow.

The page says this plainly, marks every live setup "practise on paper only" unless the test finds an edge, hides
setups whose target would still lose money after charges, and never promises a profit.
