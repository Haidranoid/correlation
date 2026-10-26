# Hypothesis A of the market-geometry plan: is an aggregate more forecastable than its stocks?

**TL;DR:** Luca's claim A is that an aggregate is more forecastable than its stocks, the way temperature is and molecules aren't. On the plan's ladder (single stocks, random baskets of 5, 25 and 100, and the market), the claim fails for direction. Over 2008–2019, five simple forecasters (12-month trend, 5-day reversal and last-period return, at 5 and 21 days) beat no level's own historical mean: out-of-sample R² is at or below 0 on every rung but one. Forecastability doesn't rise with basket size: the best slope is +0.08 R² points per unit of ln size, with p = 0.38 against the bar of 0.0013. Co-moving groups did a little better than matched random baskets for all five signals (best p = 0.017, or 0.084 after correcting for the five), but neither is forecastable, and the same gap appears when there is no real relation at all. Volatility, the plan's control, is forecastable at every level (market Clark-West t = 3.2). Yet its slope failed the bar too (p = 0.25), so the ladder test as written has little power. Verdict: A1 fails; A2 is not passed.

| | |
|---|---|
| Part of | `experiments/-3.1-luca-market-geometry.md`, hypothesis A (tests A1 and A2). The plan file itself is unedited |
| Run | 2026-10-23, numpy only, about 3½ minutes of compute in all |
| Test window | 144 decision months, 2008-01 to 2019-12 (old export). Reported: 2020-01 to 2025-06 (65 months) and the unseen months (15, 2025-07 to 2026-09) |
| A1 (primary) | **Fails** both of its conditions for all 5 distinct forecasters |
| **A1's raw one-sided p** | **0.382** (AR(1), 21 days): the larger of its slope's bootstrap p (0.382) and its market Clark-West p (0.239). It is the smallest such p over the 5 distinct forecasters, because the rule is "for at least one"; × 5 (Bonferroni) gives 1.0. This goes into the merge's Holm correction across A1, B2 direction, B2 risk and B5 |
| A2 (secondary) | **Not passed:** smallest raw p 0.017 (trend, 21 days), 0.084 after Holm across the 5 forecasters, so above 0.05 before the correction among all secondaries, which can only raise it |
| The control | **Failed sanity check:** the volatility forecaster's slope p is 0.25 (5 days) and 0.35 (21 days), though its market Clark-West passes (t 3.17 and 3.29) |
| Money test | Not run: A1 didn't pass (and a pass would have needed a rule written into the plan file first) |
| Forward test | Pending: no commit and no tag yet; sha256 of the frozen scripts below |

## Results (run 2026-10-23)

**Definitions used throughout:**
- **R²** is out-of-sample R² against the level's historical mean, in percentage points: 1 − Σ(outcome − forecast)² ÷ Σ(outcome − historical mean)², pooled over a level's members and the period's months. Above 0 means the forecaster beat the historical mean.
  - **Outcome:** the log return from the close of t₀+1 to the close of t₀+1+h, for h = 5 or 21 days, divided by the member's trailing 63-day volatility.
  - **Forecast:** a + b × signal, fitted by least squares on all the level's earlier pairs.
- **The slope** is the least-squares slope of the five levels' R² (single stocks, baskets of 5, 25 and 100, and the export's equal-weight market of about 1,290 stocks) on ln(size), in R² points per unit of ln size. Going from one stock to the market is 7.16 units, so a slope of +0.10 would mean the market's R² is about 0.7 points above a single stock's.
- **Bootstrap p:** a moving-block bootstrap of decision months, blocks of 6, 10,000 draws, one-sided: the share of draws with a slope ≤ 0.

### The tests, in order (the verdict stops at the first failure; every number is still computed)

1. **A1, the slope, 2008–2019: fails.** No forecaster's slope is positive with p < 0.0013. Four of the five slopes are negative. The only positive one, AR(1) at 21 days, is +0.078 (90% interval −0.38 to +0.49), with p = 0.38.
2. **A1, the market above zero by Clark-West: fails.** The best is AR(1) at 21 days, t = +0.71 (needed ≥ 3). No forecaster would pass even at 1.645.
3. **The control, the volatility forecaster: fails its slope.** Its R² is positive at every level, from +4.1% for single stocks to +7.4% for 100-baskets at 5 days. But the market's +6.6% sits below the 100-baskets, and the slope's p is 0.25 (21 days: 0.35). Its market Clark-West passes (t 3.17 and 3.29). So the ladder as the plan specifies it failed to detect even the effect the plan expected it to detect. That is a failed sanity check, and it limits how much A1's failure can say about a small effect. The return R²s aren't small and positive, though: they are below 0.
4. **A2, groups above matched baskets: not passed.** Groups beat their matched random baskets for all five forecasters, by +0.09 to +1.31 R² points. The best is trend at 21 days, p = 0.017 raw and 0.084 after Holm across the five, but both groups' and baskets' R² are negative.

**The verdict:** A1 fails at test 1, and its other condition fails too. A2, a separate conclusion, is not passed.

### A1: the ladder

**R² against the historical mean (%), 2008-01..2019-12, and the slope:**

| Forecaster | Stocks | 5 | 25 | 100 | Market | Slope [90% interval] | Bootstrap p | NW t | Market Clark-West t (p) | Both conditions |
|---|---|---|---|---|---|---|---|---|---|---|
| Trend, 5 days | −0.17 | −0.55 | −1.19 | −1.49 | −3.35 | −0.433 [−0.987, +0.060] | 0.919 | −1.49 | −1.09 (0.86) | fail |
| Trend, 21 days | −0.16 | −0.77 | −1.63 | −1.98 | −2.20 | −0.296 [−0.574, −0.070] | 0.989 | −1.86 | −1.32 (0.91) | fail |
| Reversal, 5 days (= AR(1), 5 days) | −0.27 | −0.85 | −1.42 | −1.75 | −2.28 | −0.281 [−0.879, +0.382] | 0.763 | −0.73 | −0.13 (0.55) | fail |
| Reversal, 21 days | −0.03 | −0.25 | −0.54 | −0.70 | −0.41 | −0.063 [−0.633, +0.453] | 0.564 | −0.19 | +0.62 (0.27) | fail |
| AR(1), 21 days | −0.05 | −0.17 | −0.20 | −0.20 | +0.55 | +0.078 [−0.376, +0.493] | 0.382 | +0.30 | +0.71 (0.24) | fail |

- **No rung is forecastable.** 24 of the 25 R²s are at or below 0. The exception is AR(1) at 21 days on the market, +0.55%, whose 90% interval runs from −2.73 to +3.55.
- **R² falls up the ladder** from single stocks to 100-baskets for four forecasters, and is flat from 25 to 100 for AR(1) at 21 days. The calibrated forecasts get worse as baskets grow. The next point is why.
- **NW t:** the slope over its delta-method Newey-West standard error (2 lags). It agrees with the bootstrap everywhere.
- **The market's trend forecaster** had 48–49 training months at 2008-01 (the others 56–57; see the decisions). At its first scored month, the trend slope b for the market was −0.34 at 5 days. It was fitted on 2003–2007, when trends reversed, and went into 2008 betting on reversal.

**Reading aids (reported, not pass/fail; both decided before any result was read):**

| Forecaster | Sign-flip null slope: mean (sd) | Share of null draws ≥ the real slope | Four-level slope (no market) [90%] | Its bootstrap p | Null R² at stocks / 5 / 25 / 100 / market |
|---|---|---|---|---|---|
| Trend, 5 days | −0.182 (0.200) | 0.88 | −0.298 [−0.652, +0.033] | 0.929 | −0.15 / −0.51 / −0.97 / −1.18 / −1.43 |
| Trend, 21 days | −0.085 (0.179) | 0.93 | −0.410 [−0.743, −0.123] | 0.999 | −0.17 / −0.42 / −0.70 / −0.80 / −0.75 |
| Reversal, 5 days | −0.159 (0.233) | 0.78 | −0.326 [−0.976, +0.377] | 0.784 | −0.23 / −0.65 / −1.11 / −1.25 / −1.35 |
| Reversal, 21 days | −0.121 (0.184) | 0.26 | −0.149 [−0.751, +0.414] | 0.664 | −0.13 / −0.43 / −0.78 / −0.91 / −0.97 |
| AR(1), 21 days | −0.061 (0.185) | 0.15 | −0.030 [−0.343, +0.272] | 0.572 | −0.10 / −0.25 / −0.43 / −0.51 / −0.51 |

- **The fitting penalty climbs the ladder.** With every real relation destroyed (200 draws, each flipping all signals of a month by one random sign), R² still falls from about −0.15% at single stocks to −0.5% to −1.4% at the top. The slope's own null centre is −0.06 to −0.18, not 0. The reason: the baskets in a month all have nearly the same signal. The spread of the scaled signal across members within a month is 0.83 for stocks, 0.34 for 25-baskets and 0 for the market, so a basket level's slope is fitted from few effective observations.
- **Against that null the real slopes are still ordinary.** 15% to 93% of null draws are at least as large, so even judged against "no predictability, same structure", nothing shows.
- **Without the market rung** (the four pooled levels only), the slopes are negative for all five forecasters.

### The control: volatility

**R² (%) of the two-term HAR, 2008-01..2019-12:**

| Forecaster | Stocks | 5 | 25 | 100 | Market | Slope [90%] | Bootstrap p | Four-level slope (p) | Market Clark-West t |
|---|---|---|---|---|---|---|---|---|---|
| HAR, 5 days (the control: outcome divided by trailing volatility) | +4.09 | +5.28 | +6.68 | +7.35 | +6.59 | +0.379 [−0.583, +1.214] | 0.248 | +0.727 (0.193) | +3.17 |
| HAR, 21 days (the control) | +7.21 | +9.30 | +9.15 | +9.24 | +9.86 | +0.297 [−0.954, +1.378] | 0.347 | +0.393 (0.358) | +3.29 |
| HAR, 5 days, undivided (reported) | +43.17 | +37.79 | +37.11 | +36.95 | +39.19 | −0.444 [−2.144, +0.610] | 0.739 | −1.273 (0.868) | +3.32 |
| HAR, 21 days, undivided (reported) | +57.48 | +53.03 | +51.40 | +50.70 | +52.67 | −0.624 [−2.804, +0.382] | 0.823 | −1.437 (0.912) | +3.28 |

- **Volatility is forecastable at every level,** the "thermodynamics" half the plan already expected. The market's Clark-West t is above 3.
- **It rises with aggregation only from stocks to small baskets.** The market rung is one noisy series, the 90% interval is wide, and in 2008–2009 the control's R² was negative at the top of the ladder (−0.7% for 100-baskets, −0.1% for the market at 5 days). Calibrated on calm 2003–2007, it couldn't foresee the crisis volatility.
- **Undivided,** most of the R² is the cross-sectional difference in volatility levels between stocks (43–57%), which is why single stocks score highest. That isn't time-varying predictability, which is why the divided version is the control.

### A2: co-moving groups against matched random baskets

**R² (%), 2008-01..2019-12** (groups: universe stocks sharing a twin fund, about 32 a month, median 9 stocks; matched: 20 replicates of one beta- and volatility-matched random basket per group):

| Forecaster | Groups | Matched baskets | Groups − matched [90%] | Bootstrap p | NW t | Holm across the 5 | Twin funds' own prices | SPY |
|---|---|---|---|---|---|---|---|---|
| Trend, 5 days | −0.90 | −1.32 | +0.42 [−0.55, +1.64] | 0.266 | +0.67 | 0.533 | −1.14 | −1.31 |
| Trend, 21 days | −1.23 | −2.54 | +1.31 [+0.23, +2.63] | **0.017** | +1.75 | 0.084 | −1.55 | −1.49 |
| Reversal, 5 days | −1.48 | −2.40 | +0.93 [−0.11, +1.97] | 0.071 | +1.61 | 0.283 | −1.89 | −2.14 |
| Reversal, 21 days | −0.47 | −1.34 | +0.87 [−0.53, +2.34] | 0.166 | +1.03 | 0.499 | −0.46 | −0.37 |
| AR(1), 21 days | −0.34 | −0.43 | +0.09 [−0.26, +0.45] | 0.324 | +0.40 | 0.533 | −0.20 | +0.13 |

- **Groups lose less than matched baskets in all five,** but neither is forecastable, since every R² is negative.
- **Most of the gap is a fitting effect, not structure** (a reading aid added after results; it changes no rule). Groups' signals are about 28% more spread out within a month than matched baskets' (sd 0.71 against 0.56 for trend, 0.64 against 0.50 for reversal). The sign-flip null keeps that structure and destroys every real relation (200 draws, one flip per month shared by the groups and all 20 replicates), and under it the gap is already positive.

**The A2 gap against its sign-flip null (R² points, 2008–2019):**

| Forecaster | Observed groups − matched | Sign-flip null: mean (sd) | Share of null draws ≥ observed |
|---|---|---|---|
| Trend, 5 days | +0.42 | +0.27 (0.33) | 0.27 |
| Trend, 21 days | +1.31 | +0.55 (0.49) | 0.07 |
| Reversal, 5 days | +0.93 | +0.70 (0.36) | 0.22 |
| Reversal, 21 days | +0.87 | +0.52 (0.40) | 0.13 |
| AR(1), 21 days | +0.09 | +0.29 (0.27) | 0.85 |

- **Against that null no forecaster's gap is unusual:** the closest is trend at 21 days, where 7% of null draws are as large. The pass rule is unchanged, since A2's bootstrap p decides, and this only says how to read the lean.
- **What "structure" means here:** about 76% of group-months have a sector or industry fund as their twin, 21% a broad-market or style fund and 3% a country fund.
- The twin funds on their own prices and SPY are no more forecastable than the groups.

### Against zero instead of the historical mean (the drift the plan warned about)

**R² against zero (%), with R² against the historical mean in brackets:**

| Forecaster | Stocks | 100-baskets | Market | SPY |
|---|---|---|---|---|
| Trend, 5 days | −0.54 (−0.17) | −3.04 (−1.49) | −5.77 (−3.35) | −1.21 (−1.31) |
| Trend, 21 days | +0.07 (−0.16) | −2.05 (−1.98) | −2.58 (−2.20) | +2.66 (−1.49) |
| Reversal, 5 days | −0.57 (−0.27) | −3.30 (−1.75) | −3.94 (−2.28) | −2.04 (−2.14) |
| Reversal, 21 days | +0.19 (−0.03) | −0.76 (−0.70) | −1.12 (−0.41) | +3.74 (−0.37) |
| AR(1), 21 days | +0.16 (−0.05) | −0.27 (−0.20) | −0.15 (+0.55) | +4.21 (+0.13) |

- **SPY scores +2.7% to +4.2% against zero at 21 days** but −1.5% to +0.1% against its historical mean. That is SPY's upward drift being "forecast", not predictability. GKX's bottom-up S&P 500 R², measured against zero, has the same issue.

### Direction (reported)

**The signal's sign against the outcome minus its historical mean** (hit rate, Newey-West t of the monthly hit rate minus 50%), and the **rank correlation** between signal and outcome across a level's members each month (mean, Newey-West t):

| Forecaster | Hit: stocks | 100-baskets | Market | SPY | Rank: stocks | 100-baskets | Groups | Twin funds |
|---|---|---|---|---|---|---|---|---|
| Trend, 5 days | 52.3% (+1.7) | 56.4% (+1.6) | 54.9% (+1.1) | 57.6% (+1.7) | +0.021 (+1.3) | +0.013 (+0.8) | +0.044 (+1.7) | +0.089 (+3.0) |
| Trend, 21 days | 52.3% (+2.1) | 53.8% (+1.1) | 51.4% (+0.3) | 57.6% (+2.0) | +0.029 (+1.9) | +0.010 (+0.6) | +0.049 (+1.8) | +0.097 (+3.5) |
| Reversal, 5 days | 52.2% (+1.7) | 46.6% (−0.9) | 44.4% (−1.3) | 47.2% (−0.7) | +0.016 (+1.3) | +0.015 (+1.2) | +0.024 (+1.1) | +0.017 (+0.7) |
| Reversal, 21 days | 51.8% (+1.6) | 50.9% (+0.3) | 52.8% (+0.7) | 51.4% (+0.4) | +0.021 (+1.8) | +0.008 (+0.7) | +0.037 (+1.6) | +0.035 (+1.4) |
| AR(1), 21 days | 49.1% (−1.1) | 49.3% (−0.2) | 50.0% (+0.0) | 48.6% (−0.3) | −0.016 (−1.4) | +0.006 (+0.5) | −0.005 (−0.2) | −0.005 (−0.2) |

- **No hit rate beats 50% at t ≥ 3.** The trend's sign is right 51–58% of the time on aggregates, mostly because both the trend and the market tended upward. Its calibrated forecast's sign (not tabled; in `a_results_old.json`) is right 42–52% of the time on the aggregates.
- **Twin funds ranked by trend:** the twin funds' 12-month trend ranks their next month at +0.09 to +0.10 (t 3.0 and 3.5), about 32 funds a month. That is the known industry momentum (Moskowitz & Grinblatt 1999), a cross-sectional result, not the time-series forecastability A1 asks about. It is one of about 90 reported direction numbers and isn't a test here.

### The upper bound: no skipped day (reported)

**R² (%) with outcomes from the close of t₀:**

| Forecaster | Stocks | 100-baskets | Market | Slope | Bootstrap p | Slope with the skipped day |
|---|---|---|---|---|---|---|
| Trend, 5 days | −0.20 | −0.88 | −1.99 | −0.240 | 0.915 | −0.433 |
| Trend, 21 days | −0.12 | −1.94 | −1.89 | −0.266 | 0.980 | −0.296 |
| Reversal, 5 days | −0.51 | −2.04 | −2.30 | −0.245 | 0.979 | −0.281 |
| Reversal, 21 days | −0.21 | −2.44 | −2.85 | −0.386 | 0.884 | −0.063 |
| AR(1), 21 days | −0.15 | −0.80 | −0.44 | −0.046 | 0.606 | +0.078 |

- **The skipped day doesn't hide an effect.** Without it the ladder is no better. Stale prices in thin stocks (Lo & MacKinlay) don't make these 500 liquid stocks' baskets look predictable.

### By period

**Slope (bootstrap p)** per forecaster, the control's slopes, and A2's best difference (trend, 21 days):

| Period | Months | Trend 5 | Trend 21 | Reversal 5 | Reversal 21 | AR(1) 21 | Vol 5 (control) | Vol 21 (control) | A2: trend 21, groups − matched (p) |
|---|---|---|---|---|---|---|---|---|---|
| 2008-01..2019-12 (the test) | 144 | −0.433 (0.92) | −0.296 (0.99) | −0.281 (0.76) | −0.063 (0.56) | +0.078 (0.38) | +0.379 (0.25) | +0.297 (0.35) | +1.31 (0.02) |
| 2008–2009 | 24 | −0.892 (0.75) | −1.255 (1.00) | −0.783 (0.79) | −0.119 (0.56) | −0.221 (0.81) | −0.886 (0.70) | −1.348 (0.77) | +5.00 (0.01) |
| 2010–2014 | 60 | −0.574 (0.87) | −0.086 (0.84) | +0.166 (0.40) | +0.277 (0.22) | +0.475 (0.05) | +0.531 (0.24) | +0.159 (0.48) | +0.36 (0.27) |
| 2015–2019 | 60 | −0.127 (0.90) | −0.016 (0.57) | −0.401 (0.75) | −0.430 (0.72) | −0.223 (0.64) | +0.611 (0.22) | +0.903 (0.17) | +0.12 (0.36) |
| 2020-01..2025-06 (reported) | 65 | +0.086 (0.17) | −0.036 (1.00) | −0.274 (0.96) | −0.102 (0.66) | −0.197 (0.93) | +0.416 (0.27) | +1.618 (0.02) | +0.15 (0.03) |
| The unseen months (reported) | 15 | −0.137 (0.70) | −0.022 (0.89) | −0.036 (0.73) | −0.161 (0.59) | −0.081 (0.56) | −1.219 (0.79) | −2.131 (0.98) | −0.01 (0.56) |

**The market's R² (%) and Clark-West t,** per forecaster:

| Period | Trend 5 | Trend 21 | Reversal 5 | Reversal 21 | AR(1) 21 |
|---|---|---|---|---|---|
| 2008-01..2019-12 | −3.35 (−1.1) | −2.20 (−1.3) | −2.28 (−0.1) | −0.41 (+0.6) | +0.55 (+0.7) |
| 2008–2009 | −7.38 (−0.3) | −9.13 (−1.3) | −7.04 (−1.2) | −0.83 (+0.2) | −1.65 (−0.8) |
| 2010–2014 | −4.25 (−1.2) | −0.71 (−0.6) | +1.06 (+0.8) | +1.86 (+1.2) | +3.29 (+1.5) |
| 2015–2019 | −0.82 (−0.9) | −0.07 (+0.1) | −2.72 (−0.2) | −2.97 (−0.1) | −1.53 (+0.0) |
| 2020-01..2025-06 | +0.53 (+0.9) | −0.30 (−1.8) | −2.59 (−1.1) | −1.03 (−0.1) | −1.55 (−1.1) |
| The unseen months | −1.12 (−0.4) | −0.09 (−1.2) | −0.31 (−0.7) | −1.76 (−0.3) | −0.97 (−0.1) |

- **2010–2014 is the only stretch that leans Luca's way:** the market's AR(1) R² is +3.3% and its slope +0.48 (p 0.05), with reversal also positive. It doesn't carry over to 2015–2019, 2020–2025 or the unseen months.
- **2008–2009 is the worst period for every return forecaster** and also for the control. The data hold only surviving companies, so crisis-period results may be biased (the plan's "Survivor-only data").
- **A2's gap is largest in 2008–2009** (+5.0 points, p 0.01) and small afterwards (+0.1 to +0.4).

### By year

**Slope per calendar year of decision** (12 months each, no bootstrap):

| Year | Trend 5 | Trend 21 | Reversal 5 | Reversal 21 | AR(1) 21 | Vol 5 | Vol 21 |
|---|---|---|---|---|---|---|---|
| 2008 | −2.91 | −1.02 | −1.88 | −0.94 | −0.47 | −6.42 | −3.53 |
| 2009 | +1.74 | −1.62 | +0.93 | +1.42 | +0.24 | +1.26 | +4.27 |
| 2010 | −2.92 | −0.27 | +1.20 | −0.86 | +0.17 | +0.46 | −1.30 |
| 2011 | +0.49 | −0.16 | +0.09 | +0.88 | +0.00 | +1.45 | +0.50 |
| 2012 | +0.25 | −0.01 | +1.89 | +0.37 | +0.44 | +2.23 | +5.21 |
| 2013 | −0.82 | +0.24 | −4.68 | −0.89 | −0.76 | +0.35 | −4.63 |
| 2014 | +0.02 | −0.03 | −0.40 | +0.62 | +1.37 | −2.02 | −4.25 |
| 2015 | −0.33 | +0.28 | +0.39 | −2.09 | −1.34 | −1.36 | +0.39 |
| 2016 | +0.07 | −0.20 | +0.12 | −0.14 | +0.33 | −2.42 | −0.30 |
| 2017 | −0.60 | +0.49 | +0.06 | +0.06 | −0.56 | −3.10 | −7.68 |
| 2018 | −0.05 | −0.23 | −1.46 | −0.07 | −0.34 | +1.88 | +2.49 |
| 2019 | +0.08 | −0.13 | −1.00 | −0.34 | +0.71 | +1.14 | +1.61 |
| 2020 | −0.03 | −0.02 | −0.57 | +0.10 | −0.21 | +0.34 | +1.40 |
| 2021 | +0.50 | +0.06 | +0.00 | −0.09 | −0.31 | −0.92 | −1.16 |
| 2022 | −0.21 | −0.11 | −0.11 | −0.76 | −0.09 | −0.75 | −2.24 |
| 2023 | +0.40 | −0.03 | +0.04 | −1.16 | −0.21 | +3.17 | +5.53 |
| 2024 | +0.19 | −0.03 | −0.07 | +0.91 | +0.27 | +0.27 | +0.92 |

- **Positive years, 2008–2019 (of 12):** trend 5 days 6, trend 21 days 3, reversal 5 days 7, reversal 21 days 5, AR(1) 21 days 7, about what chance gives. The control is positive in 7 (5 days) and 6 (21 days).
- **One year is 12 market observations,** so a year's slope is mostly the market rung's noise.

### The unseen months (run once, at the end, with the frozen code)

- **15 decision months, 2025-07-16 to 2026-09-16** (the new export, rebuilt from 2002-12, with every forecaster refitted on it). They have been read by about 65 earlier experiments, so they are reported, never confirming.
- **A1:** every slope is negative (−0.02 to −0.16). The smallest A1 p is 0.56 (AR(1), 21 days), and the market's best Clark-West t is −0.1.
- **A2:** groups − matched is −0.6 to +0.05 points. The smallest p is 0.29.
- **The control:** its slope is negative too (5 days −1.22, 21 days −2.13), because the market's volatility R² was −7.6% and −16%. Fifteen months of one series are too few to read.
- **The calendar continues:** the new export's first 5,911 days equal the old export's. Recomputed on the new export, 2008–2019 gives the same picture: AR(1) at 21 days has slope +0.091 with p 0.37 (old: +0.078, p 0.38), and A2's trend at 21 days has p 0.007 (old: 0.017). The new export's market averages slightly fewer stocks over 2008–2019, 1,212 against 1,290.

### What the results can and can't show

The plan's own lines, applied to this result:
- **A1, "time-varying predictability rises with aggregation, against the historical mean and not only against zero": not shown.** Against the historical mean, no level was forecastable by these signals, so there was nothing for aggregation to raise.
- **It can't show that aggregates are unforecastable,** only that five plain price signals (12-month trend, 5-day reversal, last-month return), each calibrated by one least-squares line per level, didn't forecast them. Many-characteristic forecasts like GKX's weren't tested.
- **The test as written has little power, and the failed control proves it.** The control's volatility R² rises from 4.1% at single stocks to 7.4% at 100-baskets, and the market's Clark-West t is 3.2, yet its slope p is 0.25 against a bar of 0.0013. Three things weaken the slope:
  - the market rung is one series, 144 months;
  - the fitting penalty grows up the ladder, since the sign-flip null's slope is centred at −0.06 to −0.18;
  - the bar is t ≥ 3.
  - So a small real aggregation effect in direction could hide here. What this run does show is that the levels' direction R²s are negative, not small and positive.
- **Against zero, aggregates look forecastable when they aren't.** SPY's 21-day R² against zero is +2.7% to +4.2% while it is −1.5% to +0.1% against its historical mean. That is the plan's drift warning, measured.
- **A2, "economically linked groups add to that beyond matched random baskets": not shown.** Groups lose less than matched baskets for all five signals, but both are below zero. Under the sign-flip null, which has no real relation, the gap is already +0.3 to +0.7 points, so most of it comes from groups' more spread-out signals making a fitted line less noisy, not from structure.
- **Crisis years:** 2008–2009 is the worst stretch for every forecaster and the best for A2's gap. The data hold only survivors, so those years are flagged as possibly biased.
- **The unseen months agree** with the test window, but they are 15 months already read by about 65 experiments, so they confirm nothing.
- **Nothing here bears on the curvature claim (B),** which is tested separately.
- **No pass, so no money test.** And no result here shows any law, relativistic or otherwise, governing markets.

### Frozen for forward confirmation (tag pending commit)

The forward test (forecasts logged from each new Stooq download, read at 12 and 36 months) is **pending**: Brandon hasn't asked for a commit, so there is no git tag yet. The final scripts, as run:

```
8d7e31dd47195cf3222841a4b69cb2759f773da07bd422b63b2298947a3afc8a  research/market_geometry/ladder.py
f94de50c3ac6094a2b1807433eef09c582d75df9dae57a2a04181a450f5f5097  research/market_geometry/unseen_a.py
ba66c08e0fb093a96452c89825eb1f3ab05463f6c8c97958414dd689bee214d2  research/market_geometry/a_check_pipeline.py
```

A1 failed, so a forward reading can't change this file's verdict. It would only matter if Brandon decides to keep the volatility ladder or A2's lean under watch.

### Files, rerun and runtime

**Code** (`research/market_geometry/`, system `python3` with numpy 2.3.5, no virtual environment):
- `ladder.py old`: builds the ladder on the old export (59 s), fits and scores every forecaster, runs the statistics and the sign-flip null (15 s), and writes `a_results_old.json`. `ladder.py new --build-only` builds the new export's ladder.
- `unseen_a.py`: rebuilds on the new export (105 s), scores the unseen months and writes `a_results_new.json`.
- `a_check_pipeline.py` writes `a_check_pipeline.txt`: the synthetic-signal checks (5 s).
- `a_check_a2_null.py` writes `a_check_a2_null.txt`: A2's sign-flip null, added after results (2 s).

**Outputs:**
- `results_A.md`: this file;
- `artifact_data_A.json`: the data for the explainer page;
- `a_results_old.json`: every statistic, per period, per year, per level and per forecaster, including the variants;
- `a_results_new.json`: the unseen months and the new export's test window.

**Large files** (`~/.local/share/correlation2/experiments/market_geometry/a/`):
- `rows_old.npz`, `rows_new.npz`: one row per level member per decision date;
- `build_old.json`, `build_new.json`: universe and group sizes, matching statistics, the market and SPY checks;
- the logs and `frozen_sha256.txt`.

**Rerun:**

```
cd ~/programming/correlation/research/market_geometry
CAP="systemd-run --user --scope -q -p MemoryMax=4G -- env OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3"
$CAP python3 -I -B ladder.py old            # ~75 s: build + scoring + report
$CAP python3 -I -B a_check_pipeline.py      # ~5 s
$CAP python3 -I -B a_check_a2_null.py       # ~2 s, after results
$CAP python3 -I -B unseen_a.py              # ~2 min, once, at the end
```

**Runtime:** about 3½ minutes of compute in all, on a machine shared with other jobs (load about 43 on 24 cores), within the 4 GB memory cap.

## Novelty check (2026-10-23, run before any test; it changed no rules)

**Searched for:** the same forecaster compared across levels of aggregation (stocks, portfolios, the index); forecastability against basket size; industry groups against random or matched baskets; volatility forecastability by level. Seven web searches plus reading GKX's text.

**Found:**
- **Gu, Kelly & Xiu 2020** (*RFS*, "Empirical Asset Pricing via Machine Learning"), the closest prior and essentially claim A for machine-learning forecasts. Monthly out-of-sample R² is 0.33–0.40% for individual stocks (neural networks) and 1.08–1.80% for the bottom-up S&P 500. The authors credit aggregation: it "averages out much of the unpredictable stock-level noise". **But their R² is measured against a forecast of zero**, which the plan warns mixes in drift. They note that against the historical mean the stock-level R² changes a lot (3.74% for their OLS-3), because a per-stock historical mean is a noisy forecaster. This plan uses one pooled historical mean per level, which avoids that problem, and it reports R² against zero beside it.
- **Jung & Shiller 2005** (*Economic Inquiry* 43(2)), "Samuelson's Dictum and the Stock Market": the market is "micro efficient, macro inefficient". This is the framing, on valuation ratios and dividend growth rather than short-horizon price signals.
- **Lo & MacKinlay 1990** (*J. Econometrics*): portfolio returns are much more autocorrelated than single stocks' (a weekly first-order 46% for a small-firm portfolio). Nonsynchronous trading explains only part of it, and cross-autocorrelation between stocks explains much of the rest. This is the reason for the plan's skipped day.
- **Lewellen 2002** (*RFS*): size, book-to-market and industry portfolios show momentum about as strong as single stocks, driven by cross-serial correlation. The nearest test of "groups against other portfolios", though with no matched random baskets.
- **Haddad, Kozak & Santosh 2020** (*RFS*, "Factor Timing"): market-neutral factor portfolios are strongly predictable, more than the aggregate market. This is aggregation along a different axis.
- **Goulet Coulombe et al. 2023** (arXiv 2306.05568, "Maximally Machine-Learnable Portfolios"): builds the most predictable portfolios, and per the search summary also reports random 20- and 100-stock portfolios. Not verified past the abstract.
- **Volatility by level:** no direct comparison of HAR forecastability across aggregation levels was found. A thesis (Rivard, HEC 2024) finds HAR extensions help the S&P 500 more consistently than single stocks.

**Not found:** a ladder of random baskets of increasing size against single stocks and the market, using one forecaster and scored against the historical mean; or industry groups against random baskets matched on beta and volatility. So A1 and A2 as designed look new, while claim A itself has prior support (GKX) measured against zero.

**Changed no rules.**

## Decisions

Every choice the plan leaves open, and every deviation from it. All were made **before results were seen** (before any real forecaster's R², slope, Clark-West statistic, hit rate or rank correlation was computed or read) unless marked otherwise. The only data looked at first were universe and group counts, the matching statistics, the market-series and SPY checks, and synthetic-signal checks of the code.

**Data and calendar**
- **Calendar:** the twins' fixed calendar, export day 252 + 21k (`tw['rebal']`), so `c2eval.excluded(E, tw)[k]` and the twin of decision k line up by construction. There are 270 decision dates on the old export, from 2002-12-31 to 2025-06-13. *Before results.*
- **Universe at t₀:** exactly the rule of `states()` in `research/market_timing/common.py`, copied rather than imported: drop `excluded`; require a median dollar volume above 0 and a daily log return on at least 90% of the last 63 days; take the 500 with the highest median dollar volume. Before 2005-02 the export holds only about 316 eligible stocks, and the rule is applied to what exists, as `-3.2` did. That affects only training months. *Before results.*
- **Prices:** the export's closes, forward-filled over gaps of at most 5 days, as daily log returns. The export's bad-print repair looks at close[d+1], so a signal's last close can carry one day of look-ahead in a repaired print; the skipped day keeps that day out of every outcome. *Before results.*
- **Sums** (signals, outcomes, volatilities) need a return on at least 90% of their days, or the pair is dropped. Outcomes are not winsorized, because the plan names no winsorizing. *Before results.*

**Levels**
- **Baskets** are the equal-weight daily mean of their members' daily log returns, the definition of the export's `market`. A basket has no return on a day when fewer than half its members have one. Members are fixed at t₀, and their own past returns form the basket's history for its signals and trailing volatility. *Before results.*
- **Random baskets:** 1,000 each of 5, 25 and 100 stocks per decision date, drawn uniformly from that date's universe with no repeat inside a basket, freshly drawn each date (seeded). *Before results.*
- **Groups:** universe stocks with the same twin at decision k (`tw['fund_row'][k]`), at least 5 of them. In 2008–2019 there are 27 to 41 groups a month (median size 9, mean 12, largest 101), covering about 393 of the 500 stocks. **What the groups are:** of the 4,626 group-months, about 21% (25% of member-months) have a broad-market or style fund as their twin (SPY, MDY, IWM, QQQ, DIA, IWF and the like), 3% a country fund, and about 76% a sector or industry fund (IYT, XLY, XLV, XLE, XLI, XRT, XLU, SMH, XLP, XLF and others; a hand classification, approximate). So A2's "structure" is mostly industry. Before 2005 almost every stock shares one twin (one group of about 307), which affects only training. *Before results; the counts were seen before any result.*
- **Matched baskets, 20 replicates:** each replicate is the plan's literal design, one matched basket per group with one non-member per member. Each replicate also has its **own forecaster, fitted only on its own pairs**, and the matched level's R² pools the 20 replicates' losses. *Before results.* The reasons:
  - one draw is noisy;
  - fitting one forecaster on all 20 replicates' pairs would give the matched level 20 times the groups' training data. Out-of-sample R² rises with training size (the pure-noise check below), so that would bias A2 against the groups.
  - The matched level has 99.8% of 20 × the groups' pairs (92,372 against 92,520 for trend). Small baskets lose a pair when fewer than half their members have a return on some day of the window.
- **The 5 × 5 grid:**
  - beta to the export's equal-weight `market` and volatility (sd of daily log returns), both over days t₀−251 … t₀, with at least 126 common days;
  - independent quintile cuts over that date's universe. Non-members are universe stocks outside the group with a valid cell;
  - no repeats inside a basket while the cell has unused non-members, then repeats are allowed;
  - a member whose cell holds no non-member is matched from the nearest cell that does (Chebyshev, then Manhattan distance);
  - in 2002–2003, when the export's market doesn't yet have the history for beta, a uniform draw from non-members is used, and a group with no non-members at all gets no matched basket that month (40 group-months, all in 2003–2004).
  - In the scored months 2008–2019, 0.4% of members used the nearest-cell fallback and 0.5% were repeats. In 2003–2007 the figures were 31% and 34%, training only.
  - *Before results; the counts were seen before any result.*
- **Twin funds:** the groups' own funds, from the store's unrepaired closes, forward-filled over gaps of at most 5 days, with daily log returns beyond ±50% treated as missing (`c2eval`'s rule). *Before results.*
- **The market:** the export's `market` series. It starts on 2002-12-31, because it needs a year of liquidity history, so its 12-month trend first exists on 2003-12-31. The market's trend forecaster therefore has **48–49 monthly training pairs at 2008-01, not the plan's 60.** The other forecasters have 56–57, because the market's 63-day volatility first exists in 2003-03. SPY (Yahoo) has 59–60. The plan's scoring window, 2008-01 to 2019-12, is kept for every level so the slope compares the same months. *Before results.*
- **The market's size in the slope:** the mean number of stocks averaged into the export's `market` at the scored decision dates, 1,290 over 2008–2019 (range 880–1,805). The count was rebuilt from the export's closes and dollar volume with the rule in `crates/c2-census/src/variable.rs` (liquid when, over the previous 252 rows, there are at least 126 closes, a median close of at least $5 and a median dollar volume of at least $5M, re-evaluated every 21 rows). The rebuilt equal-weight series matches the export's `market` at correlation 0.999999, with a largest daily difference of 0.00045. So the slope's five points sit at ln 1, ln 5, ln 25, ln 100 and ln 1,290. *Before results.*
- **SPY from Yahoo's adjusted close** (`pubdata.etf_adjclose`), **not from the store** as the plan's Data row says. The reason: the store's SPY begins on 2005-02-25, which would leave its 12-month trend about 23 monthly training pairs at 2008-01. Over the 5,116 overlapping days, Yahoo's and the store's daily returns correlate at 0.9989, and Yahoo's run 0.24 bps a day higher. SPY is reported only; it isn't in the slope. *Before results.*

**Signals, outcomes and fitting**
- **Signals:**
  - trend: the log return from the close of t₀−252 to the close of t₀−21;
  - reversal: minus the log return over the last 5 days;
  - AR(1): the last h-day log return;
  - volatility: the log of the root-mean-square daily log return over the last 21 and 63 days.
  - *Before results.*
- **Scaling:** every return signal is divided by the member's trailing 63-day root-mean-square volatility as of t₀, times the square root of the signal's length in days. Every outcome, the log return from the close of t₀+1 to the close of t₀+1+h, is divided by the same volatility times √h. The square roots are constants within a level, so they change no R². *Before results.*
- **AR(1) at h = 5 is minus the reversal signal,** so its calibrated forecast is identical to reversal at h = 5. It is tabled, but counted once: there are **5 distinct return forecasters** (trend h5 and h21, reversal h5 and h21, AR(1) h21). *Before results.*
- **The volatility control:** the outcome is the log realized volatility (root mean square) over days t₀+2 … t₀+1+h minus the log of the member's trailing 63-day volatility. That is the plan's "divided by its own trailing 63-day volatility". The two HAR terms, log vol over 21 and 63 days, stay as free regressors. Its historical-mean benchmark is therefore "volatility stays at its 63-day level, times the level's average drift". The undivided version (raw log realized volatility against its pooled historical mean, where cross-sectional differences in volatility levels count as predictability) is reported beside it. *Before results.*
- **Fits:** least squares on all the level's pooled pairs from decision dates whose outcome ended at or before t₀. For h = 5 that is decision k−1 and earlier; for h = 21, k−2 and earlier, because the outcome of k−1 ends at t₀+1. Fits expand from 2002-12. A fit needs at least 3 pairs (the HAR, 10). The historical mean is one pooled number per level per date, over the same pairs. *Before results.*
- **Out-of-sample R²** is Campbell and Thompson's, pooled over members and months: 1 − Σ(y − ŷ)² ÷ Σ(y − ȳ_hist)² over the period. R² against zero uses Σy² as the denominator. *Before results.*
- **Months:** a decision month is scored only when its 21-day outcome fits in the export, so both horizons use the same months. Periods go by decision date. The 2008-01 … 2019-12 test window has 144 decision dates. *Before results.*

**Statistics and the pass rules**
- **The slope:** the least-squares slope of the five R²s on ln size. It is tested by a circular moving-block bootstrap of decision dates (blocks of 6, 10,000 draws). The same draws are used for every level and forecaster, so the comparisons are paired. p = (1 + draws with a slope ≤ 0) ÷ 10,001, the convention of `-3.2`. A Newey-West t (2 lags) by the delta method is reported beside it. **The pass uses the plan's explicit A1 rule,** a bootstrap one-sided p below 0.0013 with a positive slope. *Before results.*
- **Clark-West at the market:** the monthly f = (y − ȳ)² − [(y − ŷ)² − (ȳ − ŷ)²], with a Newey-West t (2 lags) and a one-sided normal p. **The bar is t ≥ 3, the plan's bar for a primary test.** Whether it would pass at the conventional 1.645 is also said. *Before results.*
- **A1 per forecaster** needs both conditions, for the same forecaster. Its p is the larger of the two one-sided p's (an intersection-union test). **A1's raw p** is the smallest such p over the 5 distinct forecasters, because the plan's rule is "for at least one". A Bonferroni × 5 version is given beside it for information. *Before results.*
- **A2 per forecaster:** R²(groups) − R²(matched) over 2008–2019, with the same bootstrap's one-sided p and a delta-method Newey-West t beside it. The raw p's are reported, plus a Holm correction within A2 across the 5 distinct forecasters. The plan's Holm correction among all secondaries belongs to the merge. *Before results.*
- **Direction, reported:**
  - the hit rate of the signal's sign against the sign of (outcome − historical mean), pooled, and the same for the calibrated forecast's sign;
  - the Spearman rank correlation between signal and outcome across a level's members each month, its mean and Newey-West t. There is none for the market or SPY, which have one member.
  - *Before results.*
- **Two reading aids for A1, reported only; neither changes the pass rule.** Added after the code checks, before any real result was read. The reason: the pure-noise check draws an independent signal for every row. Real basket signals are nearly identical across the 1,000 baskets of a month (every 100-stock basket's trend is close to the market's), so a basket level's effective sample for fitting b is closer to the number of months than to the number of pairs. The fitting penalty may therefore climb the ladder, rather than hitting only the market rung, and bias the slope.
  - **A sign-flip null:** 200 draws, each flipping every member's signal in month k by one random s_k = ±1, with the same flip for every level, so the levels stay paired. Every forecaster is then refitted and the five slope levels rescored. This keeps each month's cross-sectional structure and removes any real relation. The real slope is reported beside the null's mean and spread, and each level's R² beside its null mean.
  - **A four-level slope** over the pooled levels only (single stocks and the 5-, 25- and 100-baskets, without the market), with the same bootstrap.
- **The upper bound:** the same ladder with outcomes from the close of t₀ (no skipped day), reported only. *Before results.*
- **The money test** runs only if A1 passes. The plan wants that rule "written into this file before running", and this run may not edit the plan, so a pass would leave it not run. *Before results.*
- **The unseen months:** rebuilt on the new export, with its own twins, universe and baskets and Yahoo SPY on its days. Every forecaster is refitted from 2002-12 on that export, and only decision dates from 2025-06-30 whose 21-day outcome ends by 2026-10-19 are scored. Before that, a check that the new export's first 5,911 days equal the old export's. Reported, never confirming. *Before results.*
- **Software:** the system Python 3.14.4 with numpy 2.3.5 only. No virtual environment was needed: Spearman, Newey-West and Clark-West are a few lines each. *Before results.*

**Checks of the code** (`a_check_pipeline.py` → `a_check_pipeline.txt`; synthetic signals only, before any result):
- **Perfect signal** (x = the outcome): R² = 1.000000 at every level tested, and b = 1.
- **Pure noise:** mean out-of-sample R² −0.79% at the market (sd 1.06%, 200 draws, 17% of draws above 0), and −0.002% and −0.000% for single stocks and 100-baskets. **So the market's R² carries a fitting penalty of about −0.8 points that the pooled levels don't,** which pulls the plan's slope down. That is the cost of forecasting one series from 48–204 monthly pairs. It is part of the test as written, and it is listed under what the results can't show.
- **Leakage:** asserted for every horizon that each fit's newest training outcome ends at or before t₀.
- **A bug found and fixed before results were read:** the first run of the report computed the delta-method Newey-West t's of the slope and of A2's difference from influence terms that sum to zero by construction, so those t's were about 0. Now each t is the point estimate over the Newey-West standard error of the influence terms. That first run's output was not read (only its pair counts and training sizes were checked), and it was overwritten.


**Made after results were seen** (none changes a rule or a number in the tests):
- **The signal-spread check behind A2's reading:** the mean within-month sd of each level's scaled signal. It is descriptive, added to interpret A2's gap, and nothing was rerun because of it.
- **The sign-flip null for A2's gap** (`a_check_a2_null.py` → `a_check_a2_null.txt`): the same flip as A1's reading aid, applied to the groups and the 20 matched replicates. It is a reading aid only, and A2's pass rule and verdict are unchanged. It imports the frozen `ladder.py` without changing it, and with no flip it reproduces the real gaps exactly.
- **Compact JSON:** `ladder.py`'s output switched from indented to compact JSON, to shrink `a_results_old.json` from 1.4 MB to 0.9 MB. It was rerun, and the results were identical, compared field by field. This was done before the sha256s above were taken and before the unseen months were run.
