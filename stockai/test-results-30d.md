# stockai-30d: can a model pick the month's biggest gainers?

The question: from the close of a trading day, using only what is known by then, can a model name the stocks that will rise the most over the next calendar month (21 trading days)? This file reports the test over 14 years (June 2010 to June 2024), each year predicted by a model trained only on earlier data. Results are in `results/30d/`; run it with `scripts/30d.sh`.

## The answer, in plain words

**In this data, the model's monthly picks beat chance, the average stock, and comparable stocks. But when it may not pick stocks that had already collapsed, little of the edge is left, and collapsed stocks are exactly where this data is known to flatter results.** Treat it as a lead worth retesting on better data, not as a finding.

- **What it does:** once a month the model names its top 10. On average 0.55 of them turn out to be among the 10 stocks that actually rose the most that month.
  - Ten random picks would manage 0.08.
  - Ten random picks that are just as jumpy (volatile) as the model's would manage 0.31. That is the fair yardstick, because a month's biggest gainers are mostly volatile stocks.
  - Against it the model is clearly ahead: t = +4.9, counting each month as one observation.
  - 58% of the picks that land among the month's 10 biggest gainers or 10 biggest losers land among the gainers. A pure volatility bet lands about evenly.
- **What its picks earned:** 3.2% a month on average, against 1.0% for the average stock. That is 2.2 percentage points a month of extra return (t = 2.8). Below, "points" always means percentage points of monthly return.
  - **A few months carry the raw return:** the 5 best of 162 months account for half of the extra return. Without them the extra return is 1.1 points a month (t = 1.75).
  - **Against comparable stocks it doesn't hang on those months:** compared with the 20 stocks closest to each pick in volatility and past-year fall, the picks earned 2.1 points a month more (t = 3.6), and 1.4 points more without the 5 best months (t = 2.9). The sign-flip null, a model that can't learn anything real, gets 0.2 points (t = 0.5).
  - **Most years are within noise:** the picks were ahead in 13 of the 14 years, but only 2016 is significant on its own.
- **Money:** a $10,000 account that buys the 10 picks every month and pays 0.20% per trade ends at about $230,000 after 14 years. That is about 25% a year compounded; holding SPY ended at $65,036, 14% a year.
- **The data flaw: survivor-only prices.** The price data contains only companies still listed when it was downloaded, in 2025. So none of the stocks the model could pick in 2010–2023 went on to disappear before then. That makes buying collapsed stocks look far safer than it was.
  - **A third of the picks had collapsed:** they had fallen more than 50% in the past year.
  - **Leave collapsed stocks out and the edge mostly goes:** picking only among the rest, the picks earn 1.6% a month against 1.0%, an extra 0.65 points (t = 1.4). Without the 5 best months the extra is 0.2 points (t = 0.4). Against comparable stocks it is 0.8 points (t = 1.9), barely above the null's 0.6.
  - **Among the 500 largest stocks it holds up better,** though a sixth of its picks there had collapsed too: 2.35% a month against 1.08%, an extra 1.3 points (t = 3.0). Without the 5 best months the extra is 0.8 points (t = 2.0).
- **What would settle it:** the same test on prices that include the companies that later disappeared, with their final losses counted. The code for that is built and checked; it needs the data (see Next steps). Sharadar ($39 for a month, one of the open decisions in PLAN.md) has them. After that, the locked year (June 2024 to June 2025), once.

## What was tested

- **The target:** each stock's return over the next 21 trading days (a calendar month): the close 21 days later divided by today's close, minus 1. Prices are adjusted for splits and, until late 2022 at least, for dividends (see Caveats).
- **The question:** on each graded day, how many of a score's 10 highest-ranked stocks are among the 10 stocks whose return turned out highest. Also reported: how many are among the 10 lowest (the biggest losers), and the picks' average return.
- **Who is eligible:** the grader's universe for that year, about 900 to 2,000 stocks.
  - A census value on 90% of the previous three years.
  - A median price of at least $5 and a median trading value of at least $5 million a day over the previous year.
  - My copy of this rule picks exactly the grader's stocks in every year 2007–2023, checked by running the grader.
- **Graded days:** one every 21 trading days, the same days the grader's monthly account trades, 162 months in all.
  - **Each test year holds 11 or 12 graded months.** The test years start on fixed dates, so a year's last rebalance day can fall fewer than 21 trading days before the next year's first. Its month would then overlap the next one: by a day or two in five years, almost entirely in two (2010 and 2011, whose last rebalance fell one day before the next year's first). Those 7 months are not graded, so no two graded months overlap.
  - **The last month of the 2023 test year is not graded.** It would end on 2024-06-27, the day before the locked year, and the development data stops a day short of that.
- **The model:** LightGBM, chosen because it was the strongest model in the first test.
  - It uses the same 21 price features as the first test.
  - It is trained on each day's ranking of the next month's returns, scaled to a bell curve, so it learns which stocks do better than others rather than where the market as a whole goes.
  - **Training and tuning windows:** each test year gets its own model, trained on data from 2005 until about 16 months before the test year. Training stops when it no longer improves on the next 252 trading days, which end about 2 months before the test year.
  - **Gaps:** every gap is at least 21 trading days, so no outcome window crosses from one range into the next. The code checks this on every year.
- **Compared with:**
  - **Random picks:** exact chance levels, both uniform and volatility-matched. Volatility-matched means a random stock from the pick's own fiftieth of the day's stocks, ranked by 63-day volatility.
  - **Simple rules:** 12-1 momentum (last year's winners), 21-day reversal (last month's losers), the most volatile stocks.
  - **A "sign-flip null":** the same model trained on targets whose sign is flipped at random each day, so it can't learn anything real.
  - **An "oracle":** a deliberately cheating score that peeks at the actual outcome through heavy noise, to show the test can see real skill.

## Results

Over 162 graded months, 2010–2023, all liquid stocks:

| Scores | Hits on the 10 biggest gainers per month | Chance: random | Chance: random, same volatility | Hits vs same-volatility chance, month-level t | Share of tail hits that were gainers | Picks' average month return vs all stocks (t) | Months ahead of the average stock | Rank correlation with the outcome (t) |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.55 | 0.08 | 0.31 | +4.9 | 58% | +3.21% vs +1.03% (2.8) | 93 of 162 | +0.022 (2.9) |
| Sign-flip null | 0.42 | 0.08 | 0.36 | +1.6 | 51% | +1.49% vs +1.03% (0.8) | 82 | +0.003 (0.3) |
| 12-1 momentum | 0.43 | 0.08 | 0.30 | +2.5 | 47% | +1.27% vs +1.03% (0.4) | 74 | +0.019 (1.4) |
| 21-day reversal | 0.56 | 0.08 | 0.50 | +1.2 | 49% | +2.52% vs +1.03% (1.5) | 81 | +0.013 (1.3) |
| Most volatile | 0.75 | 0.08 | 0.72 | +0.6 | 45% | +2.01% vs +1.03% (1.0) | 86 | −0.023 (−1.4) |
| Oracle (cheats on purpose) | 0.29 | 0.08 | 0.10 | +5.1 | 100% | +6.54% vs +1.03% (17.6) | 156 | +0.20 (102) |

How to read it:
- **Volatility alone produces a lot of "hits":** the most volatile stocks land among the month's 10 biggest gainers about ten times as often as random picks. They land among the 10 biggest losers even more often. Against the same-volatility chance level they score no better than any other equally volatile stocks.
- **The null shows what luck looks like:** a model that can't know direction scores t = +1.6 against that chance level, and its picks' extra return is not significant (t = 0.8).
- **The oracle shows the test works:** a cheat whose scores correlate only 0.2 with the outcome is easy to tell from chance, and every one of its tail hits is a gainer.

The model by test year (hits summed over the year's graded months; tied picks count fractionally):

| Test year | Months | Hits | Chance: random | Chance: same volatility | Loser hits | Picks' month return vs all (t) |
|---|---|---|---|---|---|---|
| 2010 | 12 | 2.5 | 1.31 | 0.60 | 1.1 | +4.01% vs +1.81% (1.7) |
| 2011 | 12 | 8.0 | 1.21 | 4.17 | 7.0 | +1.12% vs +0.09% (0.6) |
| 2012 | 11 | 6.0 | 1.06 | 3.40 | 5.0 | +3.62% vs +2.49% (0.7) |
| 2013 | 12 | 6.2 | 1.16 | 3.63 | 3.0 | +4.28% vs +1.96% (1.8) |
| 2014 | 11 | 2.0 | 0.92 | 4.32 | 5.0 | −0.65% vs +0.56% (−0.7) |
| 2015 | 12 | 9.0 | 0.95 | 5.47 | 6.0 | +0.77% vs −0.56% (0.4) |
| 2016 | 12 | 12.0 | 0.93 | 5.23 | 5.0 | +7.14% vs +2.11% (3.0) |
| 2017 | 12 | 5.4 | 0.86 | 2.03 | 4.4 | +1.56% vs +1.15% (0.3) |
| 2018 | 11 | 6.0 | 0.72 | 3.89 | 6.0 | +2.45% vs −0.01% (1.0) |
| 2019 | 12 | 9.0 | 0.74 | 4.83 | 3.0 | +7.70% vs −0.10% (1.3) |
| 2020 | 11 | 1.7 | 0.66 | 0.80 | 1.3 | +5.40% vs +5.04% (0.2) |
| 2021 | 12 | 6.0 | 0.65 | 3.09 | 4.0 | +2.08% vs −1.43% (0.7) |
| 2022 | 11 | 10.4 | 0.56 | 3.99 | 7.2 | +3.80% vs +0.53% (1.0) |
| 2023 | 11 | 5.5 | 0.56 | 4.80 | 7.6 | +1.37% vs +1.12% (0.1) |

The 5 best months, from the day the picks were made to the day the month ended: 2020-04-28 to 2020-05-28, 2022-05-25 to 2022-06-27, 2020-03-27 to 2020-04-28, 2022-07-27 to 2022-08-25, and 2016-03-29 to 2016-04-27. In them the picks beat the average stock by 55, 45, 35, 28 and 22 points. Together they account for 53% of the model's total extra return.

## The $10,000 account

The grader runs the account. Each month it buys the 10 highest positive scores in equal amounts and holds them until the next month. It pays 0, 5 or 20 bps (hundredths of a percent) per buy or sell. The account is compared with SPY and with 1,000 random accounts that hold as many names and trade as often, either uniformly random or matched on volatility. Each test year is a separate account.

| Account, test years 2010–2023 | Average year, 20 bps | $10,000 after 14 years, 20 bps (compounded per year) | Years beating 90% of random accounts (same volatility), 20 bps | Years worse than 90% of them, 20 bps |
|---|---|---|---|---|
| LightGBM, ties broken at random (median of 5 draws; range) | +29.8% (+28.1% to +32.3%) | $230,343 (25.1% a year); $187,074 to $305,318 | 6 to 8 of 14 | 0 to 1 |
| Sign-flip null, same | +13.1% | $24,965 | 5 to 6 (against uniform random accounts) | 3 to 6 |
| SPY, buy and hold | +15.0% | $65,036 (14.3% a year) | | |

- **The null's account is understated:** on 2020-03-27 it had no positive score, so the grader held cash for that month. The null's picks would have gained 11.9% that month.
- **Costs barely matter at this pace:** monthly trading costs the model's account about 5 points a year at 20 bps, against 45 points for the daily account in the first test.
- **Ties:** the grader breaks tied scores by ticker order. The table uses five random tie-breaks instead (see the disclosed changes below). `results/30d/summary.md` lists each one, and the ticker-order account, which ended at $275,586.

## Is it the data? The survivorship checks

### How survivor-only data favours collapsed stocks

Stooq's history contains only companies still listed when it was downloaded, in 2025. A stock that collapsed in 2015 and was delisted in 2016 isn't in the data at all. So every collapsed stock the model could buy is one that, as it happens, survived. A model trained on this data can learn that collapsed stocks bounce. The walk-forward test can't catch this, because the test years share the same bias.

### The model leans into collapsed stocks

| | All liquid stocks | 500 largest stocks |
|---|---|---|
| Model's pick weight on stocks down more than 50% over the past year | 32.7% | 16.0% |
| Those picks' next-month return: pooled; averaged month by month | +5.54%; +2.65% | +4.15%; +5.35% |
| The model's other picks | +2.08% | +2.01% |
| Every eligible stock | +1.03% | +1.08% |
| The null's picks: collapsed / other (pooled) | +1.81% / +1.36% | +1.76% / +0.95% |

The independent verification split the model's 2.18 points a month of extra return into two parts (`runs/30d/verification/recheck162.log`):
- **Collapsed stocks rising as a group:** about 0.6 points, 27% to 30% of the total. Most of it comes from stocks down more than 80% over the past year (about 5 of the eligible stocks each month), which gained 8.0% a month on average in this data. That is the clearest sign of survivorship bias.
- **Which stocks the model chose,** compared with stocks that had fallen by a similar amount: the other 1.59 points. The choosing helps in every group. Where it helps most depends on how you count:
  - **Per pick,** it works best among the stocks that had fallen furthest. Picks down more than 80% beat their group by 6.7 points a month, against 0.7 to 2.2 points in the other groups. That is where survivor-only data could mislead a model most, by teaching it which collapsed stocks look like survivors.
  - **In total,** those picks carry only 6% of the weight and add 0.40 points. Picks that had not collapsed add 0.86 points, and the largest single part, 0.43 points, comes from stocks that had risen more than 25% over the past year.

Nothing here tests whether that choosing is skill or survivorship.

### Against comparable stocks

A tighter yardstick compares each pick with the 20 eligible stocks it didn't pick that are closest to it that day in volatility and in past-year fall (by percentile rank), so collapsed and volatile stocks rising as a group cancel out. The 5 best months are each run's 5 months with the highest raw extra return. The verification computed it first (`knn162.log`, `knn_subsets.log`); `evaluate.py` now computes it for every run (the "Against comparable stocks" table in each `summary.md`), with the same numbers.

| Picking among | Model: extra return over its comparable stocks (t) | Without the 5 best months (t) | Sign-flip null (t) |
|---|---|---|---|
| All liquid stocks | +2.05 points (3.6) | +1.36 points (2.9) | +0.22 points (0.5) |
| No stock down more than 50% over the past year | +0.77 points (1.9) | +0.35 points (1.0) | +0.57 points (1.6) |
| The 500 largest | +0.79 points (2.5) | +0.56 points (1.85) | −0.06 points (−0.15) |

- **On all liquid stocks the edge doesn't depend on a few months** by this yardstick. The raw extra return does, because in those months collapsed and volatile stocks rose as a group.
- **Without collapsed stocks the model is barely ahead of the null,** which itself gets +0.57 points there.
- **Among the 500 largest the edge is smaller, but clear of the null.**

### Restricting to safer stocks

The same test, restricted to stocks known at the decision day to be safer bets:

| Restricted to | Picks' month return vs all (t) | Extra return without the 5 best months (t) | Share of tail hits that were gainers | Hits vs same-volatility chance, month-level t | $10,000 after 14 years at 20 bps | Owning every eligible stock equally | The null at 20 bps |
|---|---|---|---|---|---|---|---|
| No stock down more than 50% over the past year | +1.63% vs +0.98% (1.4) | +0.17 points (0.4) | 54% | +3.3 | $41,454 | $38,689 | $19,942 |
| The 500 largest, by average trading value over 63 days | +2.35% vs +1.08% (3.0) | +0.76 points (2.0) | 62% | +2.8 | $149,345 | $46,684 | $18,289 |

The accounts in this table come from the evaluation's own simple account, not the grader. It holds each month's picks for the month, with tied picks sharing their slots, and is compared with owning every eligible stock equally (no costs), not with SPY. The grader can't restrict its random accounts to a subset.

### A second survivorship route: reverse splits

The $5 price test uses prices adjusted for later splits. A company that later did a reverse split (for example, 1 new share for 20 old) has its earlier prices multiplied up. So stocks that really traded under $5 can pass the test. Doing a reverse split later means they survived, which is the same bias again. The model picked some of them. Each one's median price as actually traded, over the year that set its test year's universe, was under $5:
- **NBR**, about $3.93 (June 2018 to June 2019): bought on 2020-04-28, it rose 294% over the next month.
- **TXMD**, about $2.73 (June 2020 to June 2021): bought on 2022-02-24 and 2022-05-25, it rose 96% and 355% over those months.
- **GSAT**, about $2.75 (June 2014 to June 2015): bought on 2016-03-29, it rose 41% over the next month.

Removing the stocks flagged this way from the universe cuts the model's extra return from 2.18 to between 1.88 and 2.06 points a month, depending on how strictly they are flagged. That is a rough estimate from the verification. Forward splits work the other way: they drop a few future winners such as NVDA from early years. Those are large names, not the kind of stock this model picks.

## stockGPT on the same test

Built after the LightGBM results above (Next steps, step 6): the POC's stockGPT with a one-month target (`scripts/gpt_horizon.sh --horizon 21`, `stockai/gpt/`, scores in `runs/gpt-21d/`).
- **How:** the same transformer and inputs as in the POC. Instead of the next day's census token, each day's target is the stock's rank among that day's stocks by its next-month return, scaled to a bell curve as LightGBM's is, in 101 bins; the score is the expected rank. Three seeds averaged, and a sign-flip null, each test year trained only on earlier data with the same gaps. At a one-day horizon the new code reproduces the POC's runs byte for byte.
- **Graded exactly like LightGBM** (`results/30d/summary.md`):

| Scores | Hits on the 10 biggest gainers per month | Chance: same volatility | Hits vs that chance, month-level t | Share of tail hits that were gainers | Picks' month return vs all (t) | Rank correlation with the outcome (t) | Extra over comparable stocks (t) | $10,000 after 14 years, 20 bps |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.55 | 0.31 | +4.9 | 58% | +3.21% vs +1.03% (2.8) | +0.022 (2.9) | +2.05 points (3.6) | $230,343 |
| stockGPT (3 seeds) | 0.01 | 0.03 | −3.1 | 14% | +0.68% vs +1.03% (−1.7) | +0.012 (1.2) | −0.37 points (−1.9) | $15,796 |
| stockGPT's sign-flip null | 0.32 | 0.19 | +2.8 | 56% | +1.67% vs +1.03% (1.1) | −0.005 (−0.4) | +0.79 points (1.7) | $66,455 |

- **What it learned is mostly "pick calm stocks."** On validation days its scores correlate −0.43 with volatility on average, and −0.96 in 2022. Calm stocks almost never make the month's biggest gainers (0.01 hits a month), and its top 10 earned less than the average stock.
- **Its validation numbers were no evidence of skill either:** the sign-flip null reached the same validation rank correlation (+0.028 against the seeds' +0.021) through the same volatility tilt. 16 of the 42 models kept their first checkpoint.
- **Verdict: no edge at one month.** Early stopping on a measure with the volatility tilt removed might train it differently. That would be a new experiment, to be decided before looking at its results.

## Changes made after seeing results (disclosed)

- **Volatility matching:** volatility-matched chance was first computed in tenths. After the quick run showed that the most volatile stocks still beat it in both tails, it was changed to fiftieths. This made the yardstick stricter.
- **Tie handling, after the full run:** in 5 of the 14 years validation stopped the model after one or a few trees, so its scores took few values. Days had 20 to 46 distinct values in 2010, 2017, 2020 and 2023, and 134 to 236 in 2022, so 10th place was usually a tie among many stocks. The grader breaks ties by ticker order.
  - **In the hit counts and the simple accounts:** tied stocks now share the remaining slots equally, the average over every possible tie-break.
  - **In the grader's accounts:** ties were broken by tiny random nudges, 5 times (`stockai/h30/jitter.py`).
  - **Effect, on the 162 graded months:** against same-volatility chance, the model's month-level t moved from +4.71 to +4.85, and its picks' average return from +3.30% to +3.21% a month. The null's t moved from +1.80 to +1.64. In the grader, ticker order gave the model $275,586 at 20 bps, inside the range of the five random tie-breaks.
- **Overlapping months, after verification:** the first version graded 169 months, 7 of which overlapped the next. It now grades 162 non-overlapping months. The extra return moved from +2.10 to +2.18 points a month, and the simple accounts no longer count two months twice.
- **Subsets:** the survivorship checks above were added after the first results were seen. They are diagnostics, not a second test.

## Other things checked

- **No look-ahead.** The independent verification ran these on real data:
  - It refitted the 2016 and 2021 models after garbling every close, v, dollar volume and market value past the last validation outcome. It got the same models and bit-identical predictions for the whole test year.
  - It garbled everything after five sampled decision days. The scores on those days were bit-identical.
  - The code also asserts, for every year, that training and validation outcome windows end before the next range starts.
- **No locked year:** nothing is trained, scored or graded on or after 2024-06-28. The last graded month runs from 2024-04-26 to 2024-05-28. No score file has a finite value from 2024-06-27 on.
- **Stocks without an outcome:** only 6 universe stock-months had no month-end price; none would have been picked.
- **Two models that didn't learn:** the 2010 and 2023 models stopped after one tree and did worse on validation than predicting nothing. The 2010 model even ranked its validation stocks backwards (rank correlation −0.034). Both were used anyway, as the rules said. The 2023 model contributes little, 0.8% of the extra return. The 2010 model's picks still beat the average stock (+4.01% against +1.81% a month) and supply 7.5% of the extra return, about their share of the months.
- **Validation numbers are optimistic:** the training log's validation t-stats come from overlapping monthly windows (one a day) and are inflated about 4×. All results here use the non-overlapping graded months.

## Caveats

- **Survivorship bias** (above) is the main reason not to trust the size of the edge.
- **The sample is small:** 162 months, with half of the raw extra return in 5 of them.
- **The test years aren't fresh:** the same 14 years were used in the first test, where LightGBM was chosen. Only the locked year is untouched.
- **Two picks are graded differently by the grader.** The grader chains daily returns and counts a day as zero when either of its closes is missing, so a jump across a missing close is lost.
  - **DVAX**, picked on 2017-07-27, has no close on 2017-07-28. Across that gap it jumped from $9.25 to $15.85. The evaluation counts its month as +81.6%, the grader as +6.0%.
  - **PCRX**, picked on 2018-01-26, lost the same way a 16% drop across a missing close on 2018-02-15: −22.4% here, −7.1% in the grader.
  - **Effect:** grading every stock the grader's way moves the extra return from 2.18 to 2.13 points a month.
- **Stooq stopped adjusting some stocks for dividends in late 2022.** Compared with Sharadar's free sample, the "adjusted" closes of IBM, KO, JNJ, PG and MMM equal their as-traded prices from about November 2022 on, so their returns after that miss their dividends. AAPL, MSFT and NVDA are still adjusted. Spin-offs aren't adjusted either: on 2021-11-04, the Kyndryl spin-off day, IBM shows −4.9% where it should be −0.5%. This probably matters little here: it understates dividend payers' returns in the last 17 graded months (from 2022-11-23), and the model's picks are rarely dividend payers, so if anything it flatters them slightly. How many stocks it affects is a day-one check on the Sharadar data.
- **Costs:** 20 bps per trade is fair for large stocks but low for small, beaten-down ones, where real costs can be far higher. At 50 bps the grader's account still ends at $100,417 (median of the five random tie-breaks; $81,431 to $132,840), 17.9% a year compounded, against SPY's $65,036. The null ends at $10,181.
- **stockGPT came later:** its training code assumed a next-day target, and LightGBM was the stronger model, so LightGBM went first. The one-month stockGPT, built since, has no edge ("stockGPT on the same test").

## Next steps, in order

Status on 2026-10-19: steps 2 and 3 are built and checked, steps 4 and 5 wait for the data, step 6 is done on the Stooq data (no edge), and step 7 waits for step 5.

1. **Get survivorship-free data (your decision):** Sharadar "Prices", full history, $39 for one month (PLAN.md §5.3, decision 1). It has the companies that were later delisted, prices as actually traded, and a corporate-action table. The day-one checks (`scripts/sharadar_check.sh`) look for SIVB, BBBY, FRC and TWTR, check why each company was delisted, and compare every shared stock's daily returns with Stooq's.
2. **Import it beside the Stooq data (built):** `stockai-judge import-sharadar` and `export` (`docs/sharadar.md`) write the same store and export the pipeline reads, in separate folders. correlation2 and the Stooq results stay as they are. `scripts/sharadar_smoke.py` runs the whole chain on fake Sharadar-shaped data, and passes.
3. **Change the test where it would hide the failures, fixed before any Sharadar data was seen (built):**
   - **Count delisted stocks.** A stock with no price at the end of its month used to be dropped, both from what the model learns from (`targets.forward_returns`, used by `model.py`) and from the grades (`metrics.grade_day`). On Stooq that happened only 6 times in the graded months. On Sharadar it would have silently dropped exactly the failures this test is about.
     - **The rule (main):** a stock whose prices stop before its month ends counts at its last price if it was bought out or merged, and as a total loss if it failed (bankruptcy, or removed by the exchange) or the reason is unknown. The number of unknowns is reported.
     - **Two bounds alongside:** every such stock at its last price, and every one as a total loss.
     - **The grader** applies the same rule, to its random accounts too, and they can't buy a stock after its last close.
   - **The $5 test on prices as actually traded,** which closes the reverse-split route.
   - **The comparable-stocks yardstick** is in every evaluation (`evaluate.py`).
   - **Costs:** also graded at 50 bps.
   - **What counts as holding up, fixed now:** under the main delisting rule, the picks beat comparable stocks with t ≥ 2, the sign-flip null doesn't, and the grader's account beats SPY at 20 bps (the median of the random tie-breaks). Every `summary.md` prints these checks.
   - **Checked:** on the Stooq data none of this changes anything reported before (`runs/regress30d/README.txt`).
4. **Check the data swap (waits for the data):** the test on Sharadar restricted to companies still listed in mid-2025, which mimics Stooq. It should roughly reproduce the Stooq result. If it doesn't, the data itself differs, and that has to be understood before step 5.
5. **The real test (waits for the data):** the same model, features, settings (training from 2005), years and yardsticks on every Sharadar company, monthly and weekly. A third run puts the $5 test back on adjusted prices, which separates its effect from the delistings'. `scripts/sharadar_runs.sh` runs steps 4 and 5. The gap between steps 4 and 5 measures survivorship directly. Also watch whether the model still leans into collapsed stocks.
6. **stockGPT on monthly targets:** built and run on the Stooq data, monthly and weekly: no edge at either horizon ("stockGPT on the same test" above; `test-results-1w.md`). On Sharadar only if a changed design is decided first.
7. **The locked year:** once, at the very end, after step 5.

## Files

- **Code:** `stockai/h30/`, run by `scripts/30d.sh` (full: about 5 minutes on an idle machine; 33 minutes measured while other jobs used the CPU). `--horizon 5` runs the weekly test (`test-results-1w.md`). The script writes everything in `results/30d/`, including the grader's random tie-breaks (step `jitter`) and the two subsets (step `subsets`). The split by past-year fall and the reverse-split estimate come from the verification scripts; the comparable-stock yardstick is in every `summary.md`. The Sharadar path: `docs/sharadar.md`, `scripts/sharadar_download.sh`, `sharadar_check.sh`, `sharadar_smoke.py` and `sharadar_runs.sh`.
  - `targets.py`: forward returns (with the delisting rule), the grader's universe (with the as-traded $5 test) and schedule.
  - `metrics.py`: hits, chance levels, month-level statistics and the simple account.
  - `model.py`: LightGBM.
  - `rules.py`: the simple rules and the oracle.
  - `evaluate.py`: the tables, including `--subset large500|no_collapse`, the collapsed-picks split, the comparable-stock yardstick, the delisting rule's bounds and the checks fixed in advance.
  - `jitter.py`: random tie-breaking for the grader.
- **Tests:** `tests/test_h30.py`, 18 tests.
- **Results:**
  - `results/30d/`: `summary.md` and `grades.json` (every graded month).
  - `results/30d/large500/` and `results/30d/no_collapse/`: the subsets.
  - `results/30d/judge/`: the grader's accounts, including the random tie-breaks (`jitter_*`).
- **Not in git:**
  - `runs/30d/`: scores and the models' logs.
  - `runs/30d/verification/`: the independent verification's scripts and logs. `NOTES.md` there lists them.
