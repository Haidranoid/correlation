# stockai-1w: can a model pick the week's biggest gainers?

The same test as stockai-30d (`test-results-30d.md`), with a one-week horizon: from the close of a trading day, using only what is known by then, can a model name the 10 stocks that will rise the most over the next 5 trading days? 700 graded weeks over 14 test years (June 2010 to June 2024), each year predicted by a model trained only on earlier data, on the same survivor-only Stooq data. Results are in `results/1w/`; run it with `scripts/30d.sh --horizon 5` (5 minutes).

## The answer, in plain words

**In this data, the weekly picks rank stocks better than chance. Less of that depends on a few lucky weeks or on collapsed stocks than in the monthly test, and it holds among the 500 largest stocks. But the edge is small, about 0.3 percentage points a week, and trading every week costs more than that. At 0.20% per trade a $10,000 account ends at about $19,000 after 14 years, against $65,036 for SPY.**

- **What it does:** each week the model names its top 10. On average 0.34 of them are among the week's 10 biggest gainers.
  - Ten random picks would manage 0.08, and ten random picks as volatile as the model's 0.24 (week-level t = +4.5).
  - **That yardstick is too lenient at one week:** the control model that can't learn anything real also beats it (t = +3.6), and so do all the simple rules. Over a week, stocks that just moved a lot swing more than their 3-month volatility says. Returns, the gainer share and the comparable-stock yardstick are the better guides here.
  - 58% of its picks that land among the week's 10 biggest gainers or losers are gainers; 52% for the control model.
- **What its picks earned:** 0.59% a week, against 0.26% for the average stock: 0.32 points a week extra (t = 2.4). The rank correlation with the outcome is +0.025 (t = 4.9); the control model's is −0.002. Below, "points" means percentage points of weekly return.
  - **Against comparable stocks** (the 20 closest in volatility and past-year fall): 0.21 points a week more (t = 2.0), 0.14 without the 5 best weeks (t = 1.4).
  - **A few weeks matter less than a few months did:** the 5 best of 700 weeks carry 42% of the extra return, against 53% in the monthly test.
  - **The simple 5-day reversal rule** (buy last week's losers) earns about as much: 0.34 points a week (t = 1.9), and 0.33 against comparable stocks (t = 2.3). LightGBM ranks the whole list better (rank correlation 0.025 against 0.013), but its top 10 are not more profitable.
- **Survivorship checks:**
  - **Collapsed stocks get less weight:** 29% of the picks had fallen more than half in the past year, against 33% in the monthly test.
  - **Without collapsed stocks** the picks still earn 0.19 points a week more than the rest (t = 1.9), 0.17 against comparable stocks (t = 2.0). The control model earns nothing there.
  - **Among the 500 largest stocks** they earn 0.33 points a week more (t = 3.2), 0.22 without the 5 best weeks (t = 2.4), and 0.25 against comparable stocks (t = 3.1). The control model earns nothing there either.
- **Money:**
  - **Before costs** the grader's account would have reached $214,869 (median of five random tie-breaks), 24.5% a year.
  - **At 20 bps per trade** (0.20% per buy or sell) it ends at $19,052, 4.7% a year, against SPY's $65,036 (14.3% a year).
  - **At 50 bps** it ends at $495.
  - The picks change almost completely every week, which costs about 20 points a year at 20 bps: more than the edge.
- **Verdict:** a small ranking signal that holds up better than the monthly one under these checks, but not a way to make money at weekly turnover, and still measured on survivor-only data. Two things could change that: the Sharadar rerun for the signal, and holding picks longer (trading only when a pick falls well out of the top) for the costs.

## What was tested

Everything as in the monthly test (`test-results-30d.md`, "What was tested"), except:
- **The target:** each stock's return over the next 5 trading days. If the close 5 days later is missing, the last close up to 4 days before it stands in, never the decision day's own close.
- **Graded days:** one every 5 trading days, the days the grader's weekly account trades: 50 a year, 700 in all.
  - **Not graded:** 12 rebalance days whose week would overlap the next year's first graded week, and 1 whose week reaches the locked year.
- **The model:** the same LightGBM, features and settings, trained on each day's ranking of the next week's returns. Every gap between training, validation and test outcomes is at least 5 trading days.
- **The rules:** 5-day reversal in place of 21-day reversal; 12-1 momentum, the most and least volatile stocks and the oracle as before.
- **Costs:** the grader charges 0, 5, 20 and 50 bps per buy or sell.

## Results

Over 700 graded weeks, 2010–2023, all liquid stocks:

| Scores | Hits on the 10 biggest gainers per week | Chance: random | Chance: random, same volatility | Hits vs same-volatility chance, week-level t | Share of tail hits that were gainers | Picks' average week return vs all stocks (t) | Weeks ahead of the average stock | Rank correlation with the outcome (t) | Extra return over comparable stocks (t) |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.34 | 0.08 | 0.24 | +4.5 | 58% | +0.59% vs +0.26% (2.4) | 355 of 700 | +0.025 (4.9) | +0.21 points (2.0) |
| Sign-flip null | 0.46 | 0.08 | 0.37 | +3.6 | 52% | +0.43% vs +0.26% (1.1) | 350 | −0.002 (−0.3) | +0.13 points (1.1) |
| 12-1 momentum | 0.40 | 0.08 | 0.28 | +5.1 | 47% | +0.32% vs +0.26% (0.3) | 344 | +0.017 (2.3) | −0.03 points (−0.3) |
| 5-day reversal | 0.47 | 0.08 | 0.35 | +4.9 | 54% | +0.60% vs +0.26% (1.9) | 346 | +0.013 (2.6) | +0.33 points (2.3) |
| Most volatile | 0.79 | 0.08 | 0.66 | +5.8 | 51% | +0.62% vs +0.26% (1.7) | 339 | −0.013 (−1.5) | +0.24 points (1.5) |
| Oracle (cheats on purpose) | 0.29 | 0.08 | 0.10 | +10.5 | 94% | +2.68% vs +0.26% (27.7) | 682 | +0.20 (183) | +2.24 points (26.3) |

The model by test year (50 graded weeks each; hits summed over the year; tied picks count fractionally):

| Test year | Hits | Chance: same volatility | Loser hits | Picks' week return vs all (t) | Extra over comparable stocks (t) |
|---|---|---|---|---|---|
| 2010 | 8.0 | 6.67 | 6.0 | +0.64% vs +0.46% (0.4) | +0.05 (0.1) |
| 2011 | 20.0 | 12.37 | 13.0 | +0.77% vs +0.06% (1.7) | +0.51 (1.5) |
| 2012 | 14.0 | 12.15 | 10.0 | +0.92% vs +0.53% (1.6) | +0.34 (1.6) |
| 2013 | 18.9 | 14.45 | 11.6 | +0.73% vs +0.46% (0.7) | +0.17 (0.7) |
| 2014 | 14.0 | 14.83 | 15.0 | −0.29% vs +0.14% (−1.0) | −0.24 (−0.7) |
| 2015 | 21.0 | 14.21 | 19.0 | +0.03% vs +0.03% (0.0) | −0.01 (0.0) |
| 2016 | 16.0 | 10.40 | 5.7 | +0.92% vs +0.51% (1.7) | +0.19 (0.7) |
| 2017 | 13.4 | 8.38 | 8.2 | +0.51% vs +0.29% (1.0) | +0.11 (0.5) |
| 2018 | 12.6 | 10.06 | 9.4 | +0.33% vs +0.06% (1.1) | +0.23 (1.1) |
| 2019 | 20.0 | 10.17 | 11.0 | +1.55% vs +0.06% (1.9) | +0.92 (1.7) |
| 2020 | 5.0 | 6.39 | 7.0 | +1.34% vs +1.10% (0.5) | −0.30 (−1.2) |
| 2021 | 18.5 | 10.35 | 11.2 | +0.32% vs −0.37% (0.9) | +0.97 (2.1) |
| 2022 | 27.0 | 16.82 | 18.0 | +0.17% vs +0.16% (0.0) | +0.02 (0.0) |
| 2023 | 26.6 | 23.73 | 23.5 | +0.26% vs +0.19% (0.1) | −0.02 (0.0) |

The picks were ahead of the average stock in 12 of the 14 years, but no single year is significant at t = 2. The 5 best weeks, from the day the picks were made to the day the week ended: 2020-06-03 to 2020-06-10, 2022-03-14 to 2022-03-21, 2022-08-02 to 2022-08-09, 2010-08-16 to 2010-08-23 and 2022-06-15 to 2022-06-23. In them the picks beat the average stock by 23, 20, 18, 18 and 16 points.

## The $10,000 account

The grader's account, as in the monthly test, but rebalanced every week. LightGBM's scores tie at 10th place in 232 of the 700 weeks (single-tree years), so its account is the median of five random tie-breaks; the grader's own ticker-order tie-break is in `results/1w/summary.md`.

| Account, test years 2010–2023 | Average year, 0 bps | $10,000 after 14 years, 0 bps | 20 bps | 50 bps | Years beating 90% of random accounts (same volatility), 20 bps | Years worse than 90% of them |
|---|---|---|---|---|---|---|
| LightGBM (median of 5 tie-breaks) | +28.0% | $214,869 (24.5% a year) | $19,052 (4.7% a year) | $495 | 5 to 6 of 14 | 0 |
| Sign-flip null (median of 5) | +20.1% | $71,683 | $6,589 | $181 | 2 to 6 (against uniform random accounts) | 1 to 3 |
| 5-day reversal | +33.6% | | $17,798 | | 7 of 14 | 1 |
| Most volatile | +32.0% | | $113,953 | | 6 of 14 (uniform) | 3 |
| SPY, buy and hold | +15.0% | $65,036 (14.3% a year) | | | | |

- **Costs decide it:** buying and selling nearly every pick every week costs about 0.4% a week at 20 bps, about 20 points a year. The model's edge before costs is about 16 points a year.
- **The most volatile stocks** end at $113,953 at 20 bps, because they change little from week to week, so trading costs stay low. Yet they show no skill at picking direction (51% gainer share, 0.24 points against comparable stocks, t = 1.5). That is a measure of how much survivor-only data flatters volatile stocks: the ones that collapsed for good are missing.

## Survivorship checks

| Restricted to | Picks' week return vs all (t) | Extra return without the 5 best weeks (t) | Extra over comparable stocks (t) | Without the 5 best weeks (t) | Sign-flip null over comparable stocks (t) | $10,000 at 20 bps (simple account) | Owning every eligible stock equally |
|---|---|---|---|---|---|---|---|
| All liquid stocks | +0.59% vs +0.26% (2.4) | +0.19 points (1.6) | +0.21 points (2.0) | +0.14 points (1.4) | +0.13 points (1.1) | $22,480 | $48,389 |
| No stock down more than 50% over the past year | +0.44% vs +0.25% (1.9) | +0.10 points (1.1) | +0.17 points (2.0) | +0.10 points (1.3) | −0.03 points (−0.4) | $9,481 | $44,217 |
| The 500 largest, by average trading value over 63 days | +0.61% vs +0.28% (3.2) | +0.22 points (2.4) | +0.25 points (3.1) | +0.18 points (2.5) | +0.00 points (0.0) | $36,954 | $56,575 |

- **Collapsed picks:** 29% of the pick weight; they earned 1.03% a week (pooled), the other picks 0.41%, every eligible stock 0.26%.
- **Among the 500 largest stocks** the signal is clearest and the control model shows nothing. Large companies rarely vanish without warning, so survivor-only data distorts them least. Even there, weekly costs at 20 bps leave the simple account below owning every stock equally without costs.

## stockGPT on the same test

The POC's stockGPT with a one-week target, built the same way as the monthly one (`test-results-30d.md`, "stockGPT on the same test"; `scripts/gpt_horizon.sh --horizon 5`, scores in `runs/gpt-5d/`), graded like LightGBM (`results/1w/summary.md`):

| Scores | Hits on the 10 biggest gainers per week | Chance: same volatility | Hits vs that chance, week-level t | Share of tail hits that were gainers | Picks' week return vs all (t) | Rank correlation with the outcome (t) | Extra over comparable stocks (t) | $10,000 after 14 years: 0 bps; 20 bps |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.34 | 0.24 | +4.5 | 58% | +0.59% vs +0.26% (2.4) | +0.025 (4.9) | +0.21 points (2.0) | $214,869; $19,052 |
| stockGPT (3 seeds) | 0.02 | 0.03 | −2.6 | 31% | +0.29% vs +0.26% (0.4) | +0.012 (2.2) | +0.03 points (0.6) | $57,576; $4,818 |
| stockGPT's sign-flip null | 0.25 | 0.20 | +2.8 | 50% | +0.16% vs +0.26% (−0.9) | −0.003 (−0.5) | −0.08 points (−1.0) | $14,374; $2,003 |

- **The same lean towards calm stocks** as at one month: its scores correlate −0.52 with volatility on validation days on average, so its top 10 rarely make the week's biggest gainers.
- **A faint ranking signal over the whole list:** the rank correlation is +0.012 (t = 2.2), and with the volatility tilt removed its validation rank correlation is positive in 13 of 14 years, against 9 of 14 for the null. It doesn't reach the top 10, which earn no more than the average stock.
- **The null's account** sat in cash on 258 days with no positive score, so its money figures are understated.
- **Verdict:** no usable edge at one week either.

## Other things checked

- **No look-ahead.** Refitting the 2016 and 2021 models on real data, with every close, v, dollar volume and market value after the last validation outcome scrambled, gave bit-identical training and validation (same best iteration, validation L2 and rank correlation, same rows). The code also asserts on every year that no outcome window crosses into a later range.
- **No locked year:** as in the monthly test, nothing is trained, scored or graded on or after 2024-06-28.
- **The code is the monthly code:** `scripts/30d.sh --horizon 5`. The changes made for this test (the shorter fill window and the horizon passed through every step) were checked against the monthly results, which came out unchanged.

## Caveats

- **Survivor-only data:** the same as the monthly test. The 500-largest result is the least affected; the rest needs the Sharadar rerun.
- **The test years aren't fresh:** the same 14 years were used before. Only the locked year is untouched.
- **The same-volatility yardstick is too lenient at one week** (above), so the hit counts mean less here than in the monthly test.
- **Stooq's missing dividends after late 2022** (`test-results-30d.md`, Caveats) affect the last test years here too, in the same small way.
- **The weeks aren't fully independent:** consecutive weeks share market conditions, so t-statistics that count each week as one observation are somewhat optimistic.

## Next steps

1. **The same test on Sharadar,** with the monthly one (`test-results-30d.md`, "Next steps"): `scripts/30d.sh --horizon 5` with the Sharadar data.
2. **Turnover limits:** hold a pick until it falls well out of the top (for example out of the top 50) instead of replacing the whole list each week. That cuts trading costs, the binding problem here. The rule has to be fixed before looking at its results.
3. **stockGPT:** done at one week and one month, with no edge (above). A changed design, for example early stopping on a measure with the volatility tilt removed, would be a new experiment.
