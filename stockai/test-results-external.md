# Does correlation2's outside data improve the predictions?

correlation2's data now holds 1,753 outside time series besides the stock prices: weather, politics, news, rates, currencies, commodities, world stock indexes, search trends, prediction markets, crypto, migration, regional prosperity and more (`~/.local/share/correlation2/external/`). This test gives LightGBM those series as inputs alongside the stock data, at all three horizons (one week, one month, two months), and checks whether its picks get better.

## What correlation2 already found (2026-10-19, `external/results/ext-all-2/summary.txt`)

correlation2 tested every outside series against every stock for lead-lag ("the series moves, the stock follows days later"), with its own sign-flip null copies and 220 pure-noise series as controls. Its verdict: "nothing in 1,753 external series leads US stocks in a way that survives into the next correlation period, apart from known market-plumbing and calendar effects that carry no outside information." At predictive offsets the strong correlations were 1.04 times chance, the same as for pure noise; relationships picked in one five-year period kept their sign in the next 50.7% of the time, against 50.1% for null picks.

So a stock-by-stock lead-lag model (each stock's exposure to a series times the series' latest move) would rebuild what that scan already covered, and isn't tried again here. What the scan can't see is whether outside conditions change *which kinds* of stocks do well: for example, whether last month's losers bounce more when volatility is high, or calm stocks do better when rates rise. A model that ranks stocks can use that through interactions between outside conditions and stock features. That is what this test gives LightGBM a chance to learn.

## The plan, fixed before any run (2026-10-19)

- **The outside inputs:** each series as correlation2 builds it for its scan (`stockai-judge export-external`, a verbatim port of correlation2's loader at commit 310ea71: each observation counts from its publication lag `known_after_days`, the move is the change since the last known value, standardized by its trailing 252-day mean and spread through the day before, clipped at ±5). Series flagged annual, single-contract or noise by correlation2's rules are left out.
- **The features:** for each of the 23 categories, the mean of that day's standardized moves over the category's live series (none when fewer than 3 are live), summed over the last 1, 5 and 21 trading days and divided by the square root of the window, missing days counting as 0 (no value when every day of the window is missing). That is 69 inputs, the same for every stock on a day, added to the 21 stock inputs. Because LightGBM is trained on each day's ranking of stocks, these can only matter through how they change which stocks rank high.
- **The models**, each the monthly test's LightGBM with the same settings, trained walk-forward per test year as before:
  - `lgbm`: the stock-only model already graded (`results/1w`, `results/30d`, `results/2m`);
  - `lgbm_ext`: stock plus outside inputs;
  - `lgbm_ext_null`: the same with the sign-flipped target (the negative control the checks table needs);
  - `lgbm_ext_shifted`: the same outside inputs, all series shifted together by 1,000 trading days (about four years, circularly within the development years) before the features are built. The inputs keep their structure but no longer line up with the stocks. If the outside data only helps the model tell periods apart (fitting the training years better without knowing anything), this copy gains just as much.
- **Pass rule:** the outside data improves the predictions at a horizon only if `lgbm_ext` beats both `lgbm` and `lgbm_ext_shifted` on two counts: the excess over comparable stocks (its t-statistic) and the grader's $10,000 account at 0.20% per trade.
- **If the pass rule fires**, the run is repeated without the categories whose official figures get revised after release (macro, prosperity, migration, population): the series hold today's revised values, which a decision at the time could not have known.
- **Reported either way:** how much of each model's splitting gain went to outside inputs (`gain_share` in the train logs).

## Known limits, written down before the runs

- **Revisions:** macro, prosperity, migration and population series are the latest vintage, so they may contain revisions that weren't known at the time. The publication lag can't fix that (see the pass rule).
- **Same-day values:** a series dated day d with a publication lag of 0 is taken as known by the 4 pm close of d, as correlation2's catalog notes say.
- **Google Trends** is normalized over its download window; standardizing removes a single overall rescaling, but not one that differs between stitched windows.
- **Short series:** prediction markets and many crypto series start between 2014 and 2021, so they're missing in early years (LightGBM handles missing values; they just can't help there).
- **Same survivor-only stock data** as every test so far: whatever the outside data adds or doesn't, the survivorship caveats of `test-results-30d.md` still apply.

## Results

**The outside data doesn't improve the predictions.** By the rule fixed in advance it helped at one week only, and that gain disappeared when the revised official statistics were left out. At one month and two months it made the picks worse.

Each horizon, graded exactly as before (`results/ext-1w`, `results/ext-30d`, `results/ext-2m`; accounts are the grader's, the median of five random tie-breaks; SPY turned $10,000 into $65,036):

| Horizon | Model | Extra over comparable stocks per period (t) | Extra over the average stock (t) | Rank correlation (t) | $10,000 after 14 years: before costs | at 0.20% per trade |
|---|---|---|---|---|---|---|
| 1 week | stock only (`lgbm`) | +0.21 points (2.0) | +0.32 points (2.4) | +0.025 (4.9) | $214,869 | $19,052 |
| | with outside data (`lgbm_ext`) | +0.25 points (2.5) | +0.46 points (3.3) | +0.018 (3.7) | $620,847 | $52,079 |
| | outside data shifted 4 years (control) | +0.15 points (1.3) | +0.08 points (0.5) | +0.010 (2.0) | $40,474 | $3,640 |
| | with outside data, revised statistics left out | +0.18 points (1.6) | +0.35 points (2.4) | +0.020 (4.1) | $247,970 | $21,504 |
| 1 month | stock only | +2.05 points (3.6) | +2.18 points (2.8) | +0.022 (2.9) | $399,700 | $230,343 |
| | with outside data | +0.79 points (1.2) | +1.03 points (1.2) | +0.022 (2.4) | $112,865 | $62,851 |
| | outside data shifted 4 years (control) | +0.05 points (0.1) | +0.32 points (0.6) | +0.014 (1.4) | $83,704 | $45,844 |
| 2 months | stock only | −0.22 points (−0.3) | +0.91 points (0.7) | +0.024 (2.0) | $41,133 | $31,176 |
| | with outside data | −0.59 points (−0.8) | −0.03 points (0.0) | +0.008 (0.6) | $21,372 | $15,990 |
| | outside data shifted 4 years (control) | +1.91 points (1.8) | +2.64 points (1.7) | +0.026 (1.7) | $154,690 | $115,205 |

- **One week:** the version with outside data passed the pass rule (comparable-stock t 2.5 against 2.0 for stock only and 1.3 for the shifted control; $52,079 against $19,052 and $3,640). It still ends below SPY after costs.
  - **The required check undid it.** Leaving out macro, prosperity, migration and population, whose figures are revised after release, drops it to t 1.6, below the stock-only model, and $21,504. The gain rested on the series most likely to contain hindsight, which is no basis for trusting it.
- **One month:** the outside data made the picks worse: t 1.2 against 3.6, and $62,851 against $230,343.
- **Two months:** worse again, and here the shifted control, whose outside inputs mean nothing, did best of all ($115,205). That shows how far these accounts can swing on noise.
- **The sign-flip controls** with outside data showed nothing at any horizon (comparable-stock t 0.2, −0.1 and 0.2).

### What the model did with the outside data

At every horizon the outside inputs took 62–70% of LightGBM's splitting gain, but the shifted and sign-flip controls used them just as heavily (63–85%). The categories it leaned on most were slow, calendar-like series (daylight hours, health, climate indexes, geophysical), not ones with an obvious market story. That is the pattern of a model using time-only inputs to tell periods apart and fit each training period's quirks, which then don't carry into the test year. The shifted control was built to catch exactly this, and it gained as much as the real data did, or more.

### Verdict

This agrees with correlation2's own scan: nothing in its 1,753 outside series improves the stock predictions, at one week, one month or two months. The stock-only model's one-month result remains the only lead, and it still waits for the survivorship-free rerun.

Checked: the outside features never read past the decision day (`tests/test_h30_external.py`); on real data, scrambling every stock array and every outside series after the last validation outcome left training and validation bit-identical (`runs/ext-verification/lookahead_ext_21.log`). Run with `scripts/external_runs.sh` (about 7 minutes per horizon, plus grading), after `stockai-judge export-external` (docs/external.md).
