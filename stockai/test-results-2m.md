# stockai-2m: can a model pick the biggest gainers two months out?

The same test as stockai-30d (`test-results-30d.md`) and stockai-1w (`test-results-1w.md`), with a two-month horizon: from the close of a trading day, using only what is known by then, can a model name the 10 stocks that will rise the most over the next 42 trading days? 78 graded two-month periods over 14 test years (June 2010 to June 2024), each year predicted by a model trained only on earlier data, on the same survivor-only Stooq data. Results are in `results/2m/`; run it with `scripts/30d.sh --horizon 42` (4 minutes). The three horizons side by side: `test-results-horizons.md`.

## The answer, in plain words

**No edge at two months.** LightGBM's top 10 earned a little more than the average stock (0.9 percentage points per two-month period, t = 0.7), but not more than comparable stocks, and a $10,000 account ended at about $31,000 after 14 years at 0.20% per trade, against SPY's $65,036.

- **Hits:** 0.46 of its 10 picks per period were among the period's 10 biggest gainers, against 0.40 for equally volatile random picks (t = +0.8). Only 47% of its picks in the two tails were gainers.
- **Returns:** +2.99% per period against +2.09% for the average stock, t = 0.7. The 5 best of the 78 periods carry more than all of that: without them the picks trail the average stock by 1.3 points a period (t = −1.2).
- **Against comparable stocks** (the 20 closest in volatility and past-year fall): −0.22 points a period (t = −0.3).
- **The whole list is still ranked a little better than chance:** rank correlation +0.024 (t = 2.0), about what it is at one week and one month. The top 10 don't capture it: 45% of them are stocks that had fallen more than half in the past year, whose outcomes swing widely.
- **Survivorship checks:** without collapsed stocks, −0.37 points a period (t = −0.4). Among the 500 largest, +1.8 points (t = 1.7), but the control model gets +1.1 there and neither survives dropping the 5 best periods.
- **Costs are low:** trading every two months costs about 2 points a year at 20 bps (5 at one month, 20 at one week). There is just no edge to keep.

## Results

Over 78 graded two-month periods, 2010–2023, all liquid stocks:

| Scores | Hits on the 10 biggest gainers per period | Chance: random | Chance: same volatility | Hits vs same-volatility chance, period-level t | Share of tail hits that were gainers | Picks' period return vs all (t) | Periods ahead | Rank correlation (t) | Extra over comparable stocks (t) |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.46 | 0.08 | 0.40 | +0.8 | 47% | +2.99% vs +2.09% (0.7) | 40 of 78 | +0.024 (2.0) | −0.22 points (−0.3) |
| Sign-flip null | 0.29 | 0.08 | 0.31 | −0.9 | 38% | +1.15% vs +2.09% (−0.9) | 35 | −0.004 (−0.2) | −0.76 points (−1.2) |
| 12-1 momentum | 0.50 | 0.08 | 0.31 | +2.6 | 49% | +2.41% vs +2.09% (0.2) | 32 | +0.011 (0.6) | +0.06 points (0.1) |
| 42-day reversal | 0.71 | 0.08 | 0.55 | +1.8 | 52% | +3.35% vs +2.09% (0.7) | 37 | +0.007 (0.4) | +0.73 points (0.6) |
| Most volatile | 0.73 | 0.08 | 0.75 | −0.3 | 41% | +1.34% vs +2.09% (−0.5) | 37 | −0.028 (−1.1) | −1.98 points (−1.9) |
| Oracle (cheats on purpose) | 0.32 | 0.08 | 0.09 | +3.9 | 100% | +9.92% vs +2.09% (10.2) | 76 | +0.21 (64) | +7.27 points (9.8) |

The 5 best periods, from the day the picks were made to the day the period ended: 2020-10-26 to 2020-12-24, 2022-12-23 to 2023-02-27, 2018-12-27 to 2019-02-28, 2011-12-23 to 2012-02-27 and 2020-12-24 to 2021-02-26. In them the picks beat the average stock by 48, 33, 32, 26 and 25 points. By test year the picks were ahead of the average stock in 9 of 14 years, behind in 5; only 2017 and 2019 reach t = 2, and 2021 is as far behind (t = −2.3).

## stockGPT

stockGPT with a two-month target (`scripts/gpt_horizon.sh --horizon 42`, scores in `runs/gpt-42d/`; built as in `test-results-30d.md`, "stockGPT on the same test"), graded like LightGBM:

| Scores | Hits on the 10 biggest gainers per period | Chance: same volatility | Picks' period return vs all (t) | Rank correlation (t) | Extra over comparable stocks (t) | $10,000 after 14 years, 20 bps |
|---|---|---|---|---|---|---|
| stockGPT (3 seeds) | 0.00 | 0.02 | +2.03% vs +2.09% (−0.1) | +0.006 (0.3) | +0.47 points (1.3) | $33,914 |
| stockGPT's sign-flip null | 0.59 | 0.37 | +3.84% vs +2.09% (0.8) | −0.014 (−0.7) | +1.06 points (0.7) | $78,366 |

As at one week and one month, it picks calm stocks: none of its picks ever landed among a period's 10 biggest gainers, and they earned what the average stock did. The null's account is luck: it bought collapsed stocks (24% of its picks) in a run where they did well.

## The $10,000 account

| Account, test years 2010–2023 | Average year, 20 bps | $10,000 after 14 years: 0 bps | 20 bps | 50 bps | Years beating 90% of random accounts (same volatility), 20 bps | Years worse than 90% of them |
|---|---|---|---|---|---|---|
| LightGBM (median of 5 random tie-breaks) | +17.0% | $41,133 (10.6% a year) | $31,176 (8.5% a year) | $20,536 | 1 to 2 of 14 | 1 to 3 |
| Sign-flip null (median of 5) | +5.5% | $10,892 | $8,087 | $5,163 | 2 to 3 (uniform) | 5 to 7 |
| 42-day reversal | +18.8% | | $43,125 | | 6 of 14 (uniform) | 4 |
| SPY, buy and hold | +15.0% | $65,036 (14.3% a year) | | | | |

The average year looks better than SPY's, but the years swing so widely (the picks lost 11% a period in 2021) that the account compounds to far less.

## What was tested

Everything as in the monthly test, except the horizon: each stock's return over the next 42 trading days; one graded decision every 42 trading days, 78 in all (7 rebalance days left ungraded because their period would overlap the next year's first, and 1 because its period reaches the locked year); every gap between training, validation and test outcomes at least 42 trading days; 42-day reversal as the reversal rule. LightGBM's features and settings are unchanged.

**No look-ahead:** refitting the 2016 and 2021 models on real data, with every close, v, dollar volume and market value after the last validation outcome scrambled, gave bit-identical training and validation (`runs/2m/verification/lookahead_42.log`).

## Caveats

The same as the monthly test: survivor-only data (here 45% of the picks are collapsed stocks), the reverse-split route into the $5 universe, Stooq's missing dividends after late 2022, and test years used before. With only 78 periods this is the noisiest of the three tests.
