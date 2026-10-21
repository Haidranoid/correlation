# One week, one month, two months: what each horizon is good for

The same test at three horizons: from the close of a trading day, can a model name the 10 stocks that will rise the most over the next week (5 trading days), month (21) or two months (42)? Same data (survivor-only Stooq), same 14 test years (June 2010 to June 2024), same models, rules and grading. Details: `test-results-1w.md`, `test-results-30d.md`, `test-results-2m.md`.

## LightGBM side by side

| | 1 week | 1 month | 2 months |
|---|---|---|---|
| Graded periods | 700 | 162 | 78 |
| Picks' extra return per period over the average stock (t) | +0.32 points (2.4) | +2.18 points (2.8) | +0.91 points (0.7) |
| ... without the 5 best periods (t) | +0.19 (1.6) | +1.07 (1.75) | −1.28 (−1.2) |
| Over comparable stocks, same volatility and past-year fall (t) | +0.21 (2.0) | +2.05 (3.6) | −0.22 (−0.3) |
| ... without the 5 best periods (t) | +0.14 (1.4) | +1.36 (2.9) | −0.97 (−1.3) |
| Whole list ranked: rank correlation with the outcome (t) | +0.025 (4.9) | +0.022 (2.9) | +0.024 (2.0) |
| Picks that had collapsed (down more than half in a year) | 29% | 33% | 45% |
| Without collapsed stocks, over comparable stocks (t); the control model | +0.17 (2.0); −0.03 | +0.77 (1.9); +0.57 | −0.69 (−0.9); −0.82 |
| 500 largest stocks, over comparable stocks (t); the control model | +0.25 (3.1); +0.00 | +0.79 (2.5); −0.06 | +1.13 (1.5); +0.79 |
| Trading cost a year at 0.20% per trade | about 20 points | about 5 points | about 2 points |
| $10,000 after 14 years, before costs | $214,869 | $399,700 | $41,133 |
| ... at 0.20% per trade | $19,052 | $230,343 | $31,176 |
| ... at 0.50% per trade | $495 | $100,417 | $20,536 |
| SPY, buy and hold | $65,036 | $65,036 | $65,036 |

The accounts are the grader's, the median of five random tie-breaks. "Points" are percentage points of return per period.

## stockGPT side by side

| | 1 week | 1 month | 2 months |
|---|---|---|---|
| Picks' extra return per period (t) | +0.03 points (0.4) | −0.35 points (−1.7) | −0.06 points (−0.1) |
| Whole list: rank correlation (t) | +0.012 (2.2) | +0.012 (1.2) | +0.006 (0.3) |
| $10,000 after 14 years at 0.20% per trade | $4,818 | $15,796 | $33,914 |

At every horizon stockGPT mostly learned to pick calm, low-volatility stocks, which rarely make the biggest gainers: at two months its picks never once landed among a period's 10 biggest gainers.

## What each horizon is good for

- **One week: the most trustworthy evidence, but no money.** With 700 periods it has the most data, and its signal survives the checks that matter: among the 500 largest stocks and without collapsed stocks the picks still beat comparable stocks, while the control model shows nothing. That makes it the least likely to be a survivor-only artifact. But the edge is small, about 0.3 points a week, and replacing the list every week costs about 20 points a year at 0.20% per trade, more than the edge. It would pay only with far less trading (keeping picks until they fall well out of the top) or much cheaper trades.
- **One month: the only horizon that made money, and the least trustworthy.** After costs the account ended at $230,343 against SPY's $65,036, and still $100,417 at 0.50% per trade, because trading costs only about 5 points a year. But a third of the picks are collapsed stocks, which survivor-only data flatters, and without them the edge mostly fades. It is the one to retest on Sharadar, which decides whether the money was real.
- **Two months: the cheapest to trade, and nothing to trade.** Costs drop to about 2 points a year, but the picks don't beat comparable stocks, and the account trails SPY. Nearly half the picks are collapsed stocks with wildly varying outcomes. The model still ranks the whole list about as well as at one month, but its top 10 don't benefit. With 78 periods it is also the noisiest test.

**In short:** the ranking signal is about the same at every horizon. What changes is how much of it survives in the top 10 and how much trading eats. At one week trading eats it; at two months the top 10 don't hold it; one month sits in between and is the only candidate, pending the survivorship-free rerun.
