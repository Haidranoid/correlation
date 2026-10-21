# stockai-30d results (horizon 21 trading days, top 10)

Test years 2022-2023, one graded decision every 21 trading days (22 in all). Rebalance days left ungraded: 1 whose month would overlap the next graded month, 1 whose month reaches the locked year.

## Top-10 hits on the month's 10 biggest gainers

| Scores | Graded days | Hits per day | Chance (uniform) | Chance (same volatility) | z vs uniform | z vs same volatility | Days tied at the cut | Hits among the 10 biggest losers per day | Chance of a loser hit (same volatility) | Tail hits that were gainers | Picks in the top / bottom decile | Picks' mean month return vs all | Rank IC (t; days > 0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 22 | 0.422 | 0.051 | 0.298 | +7.76 | +1.10 | 22 | 0.373 | 0.285 | 53% | 20.8% / 26.7% | +1.34% vs +0.82% (t 0.3) | +0.0095 (0.4; 11) |
| lgbm_null | 22 | 0.239 | 0.051 | 0.213 | +3.93 | +0.27 | 21 | 0.214 | 0.187 | 53% | 12.2% / 23.8% | -0.63% vs +0.82% (t -0.9) | -0.0089 (-0.3; 9) |
| momentum | 22 | 0.364 | 0.051 | 0.167 | +6.54 | +2.33 | 0 | 0.182 | 0.182 | 67% | 29.1% / 15.0% | +4.73% vs +0.82% (t 1.7) | +0.0490 (1.3; 18) |
| reversal | 22 | 0.409 | 0.051 | 0.483 | +7.49 | -0.52 | 0 | 1.000 | 0.445 | 29% | 23.2% / 36.8% | -0.48% vs +0.82% (t -0.4) | -0.0002 (-0.0; 13) |
| high_vol | 22 | 0.636 | 0.051 | 0.606 | +12.24 | +0.19 | 0 | 1.182 | 0.676 | 35% | 25.0% / 34.1% | +0.61% vs +0.82% (t -0.1) | -0.0430 (-0.8; 9) |
| low_vol | 22 | 0.000 | 0.051 | 0.000 | -1.06 | n/a | 0 | 0.000 | 0.023 | n/a | 3.2% / 3.6% | -0.31% vs +0.82% (t -1.1) | +0.0430 (0.8; 13) |
| oracle | 22 | 0.227 | 0.051 | 0.063 | +3.69 | +3.18 | 0 | 0.000 | 0.058 | 100% | 19.1% / 1.4% | +7.45% vs +0.82% (t 6.4) | +0.1913 (35.0; 22) |

Chance (uniform): hits expected from 10 random picks among the same stocks. Chance (same volatility): each pick swapped for a random stock from its own fiftieth of the day's stocks by trailing 63-day volatility. Tail hits that were gainers: of the picks that landed among the 10 biggest gainers or the 10 biggest losers, the share that were gainers (picking volatile stocks with no sense of direction lands in both). The top decile is the tenth of stocks that rose most that month; a random pick lands there 10% of the time.

### lgbm by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 6.9 | 0.56 | 3.67 | 3.1 | 27% / 29% | +1.47% vs +0.53% | +0.042 |
| 2023 | 11 | 2.4 | 0.56 | 2.88 | 5.1 | 14% / 24% | +1.21% vs +1.12% | -0.023 |

### lgbm_null by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 2.9 | 0.56 | 1.74 | 1.6 | 13% / 17% | +1.00% vs +0.53% | +0.035 |
| 2023 | 11 | 2.4 | 0.56 | 2.94 | 3.1 | 12% / 31% | -2.27% vs +1.12% | -0.052 |

### momentum by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 3.0 | 0.56 | 1.37 | 2.0 | 25% / 15% | +2.26% vs +0.53% | +0.043 |
| 2023 | 11 | 5.0 | 0.56 | 2.30 | 2.0 | 33% / 15% | +7.20% vs +1.12% | +0.055 |

### reversal by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 7.0 | 0.56 | 5.14 | 10.0 | 24% / 29% | +4.07% vs +0.53% | +0.044 |
| 2023 | 11 | 2.0 | 0.56 | 5.48 | 12.0 | 23% / 45% | -5.03% vs +1.12% | -0.045 |

### high_vol by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 5.0 | 0.56 | 6.67 | 12.0 | 25% / 34% | +0.35% vs +0.53% | -0.030 |
| 2023 | 11 | 9.0 | 0.56 | 6.67 | 14.0 | 25% / 35% | +0.86% vs +1.12% | -0.056 |

### low_vol by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 0.0 | 0.56 | 0.00 | 0.0 | 5% / 4% | -0.88% vs +0.53% | +0.030 |
| 2023 | 11 | 0.0 | 0.56 | 0.00 | 0.0 | 2% / 4% | +0.26% vs +1.12% | +0.056 |

### oracle by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 3.0 | 0.56 | 0.74 | 0.0 | 18% / 2% | +6.69% vs +0.53% | +0.199 |
| 2023 | 11 | 2.0 | 0.56 | 0.64 | 0.0 | 20% / 1% | +8.21% vs +1.12% | +0.184 |

## Month-level view and the collapsed picks

| Scores | Hits vs same-volatility chance, month-level t | Months ahead of the average stock | Median month's excess | Share of the total excess from the 5 best months | Excess without those 5 (t) | Pick weight on stocks down >50% over the past year | Their return (pooled; averaged by month) | The other picks' return |
|---|---|---|---|---|---|---|---|---|
| lgbm | +1.24 | 12 of 22 | +1.12% | 418% | -2.14% (-1.85) | 56.5% | +1.70%; +0.16% | +0.87% |
| lgbm_null | +0.43 | 7 of 22 | -2.48% | n/a | -4.11% (-4.93) | 25.7% | +0.79%; -3.43% | -1.12% |
| momentum | +1.63 | 11 of 22 | +1.52% | 109% | -0.45% (-0.28) | 0.0% | n/a; n/a | +4.73% |
| reversal | -0.48 | 6 of 22 | -2.70% | n/a | -6.33% (-3.94) | 73.6% | +0.95%; +0.17% | -4.46% |
| high_vol | +0.26 | 8 of 22 | -2.56% | n/a | -6.43% (-3.92) | 75.9% | +0.92%; +1.63% | -0.37% |
| low_vol | n/a | 9 of 22 | -0.84% | n/a | -2.87% (-2.90) | 0.0% | n/a; n/a | -0.31% |
| oracle | +1.98 | 21 of 22 | +5.65% | 47% | +4.57% (5.97) | 9.5% | +25.96%; +22.04% | +5.50% |

## Against comparable stocks

Each pick against the 20 eligible stocks it didn't pick that are closest to it that day in percentile rank of trailing volatility and of past-year fall (stockai.h30.metrics.comparable_excess). Stocks that swung or fell alike as a group cancel out. The 5 best months are those with the highest raw excess.

| Scores | Excess over comparable stocks per month (t) | Months ahead | Without the 5 best months (t) |
|---|---|---|---|
| lgbm | +1.12% (1.03) | 13 of 22 | +0.06% (0.07) |
| lgbm_null | -1.96% (-2.29) | 8 of 22 | -2.95% (-3.16) |
| momentum | +3.46% (1.96) | 15 of 22 | +0.95% (0.61) |
| reversal | -2.31% (-1.03) | 10 of 22 | -4.76% (-2.16) |
| high_vol | -1.14% (-0.62) | 8 of 22 | -4.37% (-3.76) |
| low_vol | -0.73% (-2.14) | 7 of 22 | -1.16% (-3.13) |
| oracle | +6.07% (5.96) | 21 of 22 | +4.12% (5.70) |

## A simple monthly account in the picks (equal weight, from one graded month to the next)

| Scores | $10,000 at 0 bps | at 5 bps | at 20 bps | at 50 bps | Every eligible stock, equal weight (no costs) |
|---|---|---|---|---|---|
| lgbm | $11,815 | $11,584 | $10,917 | $9,691 | $11,434 |
| lgbm_null | $7,750 | $7,607 | $7,195 | $6,433 | $11,434 |
| momentum | $23,424 | $23,224 | $22,632 | $21,492 | $11,434 |
| reversal | $6,837 | $6,700 | $6,305 | $5,582 | $11,434 |
| high_vol | $8,209 | $8,136 | $7,921 | $7,508 | $11,434 |
| low_vol | $9,245 | $9,174 | $8,963 | $8,556 | $11,434 |
| oracle | $45,810 | $44,909 | $42,307 | $37,528 | $11,434 |

## The $10,000 account: top 10 positive scores, rebalanced every month (the grader)

| Scores | Random match | Years beating random, p < 0.1 (0 / 5 / 20 / 50 bps) | Years worse than random, p > 0.9 (0 bps) | Mean yearly return (0 / 5 / 20 / 50 bps) | Random median (0 / 20 bps) | $10,000 after 14 years at 20 bps | SPY: mean year, $10,000 after 14 years | Rank IC z (holding-period returns) |
|---|---|---|---|---|---|---|---|---|
| lgbm | uniform | 1 / 1 / 1 / 1 | 1 | +4.0% / +2.9% / -0.3% / -6.5% | +7.7% / +3.2% | $9,123 | +20.5%, $14,490 | 0.97 |
| lgbm (volatility-matched) | volatility | 1 / 1 / 1 / 1 | 1 | +4.0% / +2.9% / -0.3% / -6.5% | +0.0% / -4.2% | $9,123 | +20.5%, $14,490 | 0.97 |
| lgbm, ties at random, seed 1 | uniform | 0 / 0 / 0 / 0 | 0 | -1.6% / -2.7% / -5.8% / -11.7% | +7.9% / +3.3% | $8,878 | +20.5%, $14,490 | 0.98 |
| lgbm, ties at random, seed 1 (volatility-matched) | volatility | 0 / 0 / 0 / 0 | 0 | -1.6% / -2.7% / -5.8% / -11.7% | -1.8% / -5.9% | $8,878 | +20.5%, $14,490 | 0.98 |
| lgbm, ties at random, seed 2 | uniform | 0 / 0 / 0 / 0 | 1 | -7.3% / -8.3% / -11.2% / -16.8% | +7.8% / +3.2% | $7,849 | +20.5%, $14,490 | 1.11 |
| lgbm, ties at random, seed 2 (volatility-matched) | volatility | 0 / 0 / 0 / 0 | 0 | -7.3% / -8.3% / -11.2% / -16.8% | -0.6% / -4.8% | $7,849 | +20.5%, $14,490 | 1.11 |
| lgbm, ties at random, seed 3 | uniform | 1 / 1 / 1 / 1 | 1 | +8.5% / +7.3% / +3.9% / -2.7% | +8.2% / +3.6% | $10,470 | +20.5%, $14,490 | 1.02 |
| lgbm, ties at random, seed 3 (volatility-matched) | volatility | 1 / 1 / 1 / 1 | 0 | +8.5% / +7.3% / +3.9% / -2.7% | -0.5% / -4.7% | $10,470 | +20.5%, $14,490 | 1.02 |
| lgbm, ties at random, seed 4 | uniform | 0 / 0 / 0 / 0 | 1 | -8.1% / -9.1% / -12.0% / -17.6% | +7.4% / +2.8% | $7,737 | +20.5%, $14,490 | 1.21 |
| lgbm, ties at random, seed 4 (volatility-matched) | volatility | 0 / 0 / 0 / 0 | 0 | -8.1% / -9.1% / -12.0% / -17.6% | -0.7% / -4.9% | $7,737 | +20.5%, $14,490 | 1.21 |
| lgbm, ties at random, seed 5 | uniform | 0 / 0 / 0 / 0 | 0 | +6.3% / +5.1% / +1.8% / -4.5% | +8.0% / +3.5% | $10,367 | +20.5%, $14,490 | 1.00 |
| lgbm, ties at random, seed 5 (volatility-matched) | volatility | 0 / 0 / 0 / 0 | 0 | +6.3% / +5.1% / +1.8% / -4.5% | -0.3% / -4.5% | $10,367 | +20.5%, $14,490 | 1.00 |
| lgbm_null | uniform | 0 / 0 / 0 / 0 | 2 | -20.2% / -20.9% / -22.9% / -26.9% | +9.6% / +5.6% | $5,682 | +20.5%, $14,490 | -1.29 |
| lgbm_null, ties at random, seed 1 | uniform | 0 / 0 / 0 / 0 | 1 | -5.4% / -6.3% / -9.1% / -14.5% | +10.1% / +5.7% | $7,502 | +20.5%, $14,490 | -1.14 |
| lgbm_null, ties at random, seed 2 | uniform | 0 / 0 / 0 / 0 | 1 | -12.9% / -13.8% / -16.4% / -21.4% | +9.9% / +5.5% | $6,818 | +20.5%, $14,490 | -1.03 |
| lgbm_null, ties at random, seed 3 | uniform | 0 / 0 / 0 / 0 | 1 | -11.4% / -12.4% / -15.0% / -20.1% | +9.7% / +5.3% | $6,744 | +20.5%, $14,490 | -1.14 |
| lgbm_null, ties at random, seed 4 | uniform | 0 / 0 / 0 / 0 | 2 | -29.1% / -29.8% / -32.0% / -36.1% | +9.7% / +5.3% | $4,525 | +20.5%, $14,490 | -1.46 |
| lgbm_null, ties at random, seed 5 | uniform | 0 / 0 / 0 / 0 | 1 | -10.1% / -11.0% / -13.7% / -18.8% | +10.3% / +5.9% | $7,283 | +20.5%, $14,490 | -1.29 |
| momentum | uniform | 3 / 3 / 3 / 3 | 6 | +17.7% / +17.1% / +15.5% / +12.4% | +13.1% / +11.0% | $28,936 | +15.0%, $65,036 | 1.38 |
| momentum (volatility-matched) | volatility | 3 / 3 / 3 / 3 | 5 | +17.7% / +17.1% / +15.5% / +12.4% | +12.8% / +10.7% | $28,936 | +15.0%, $65,036 | 1.38 |
| reversal | uniform | 7 / 7 / 7 / 7 | 4 | +28.2% / +26.8% / +22.7% / +14.8% | +13.0% / +8.2% | $48,087 | +15.0%, $65,036 | 1.83 |
| reversal (volatility-matched) | volatility | 5 / 5 / 5 / 5 | 3 | +28.2% / +26.8% / +22.7% / +14.8% | +14.0% / +9.1% | $48,087 | +15.0%, $65,036 | 1.83 |
| high_vol | uniform | 6 / 6 / 6 / 6 | 4 | +25.8% / +25.2% / +23.4% / +19.9% | +12.9% / +10.8% | $80,644 | +15.0%, $65,036 | -1.15 |
| low_vol | uniform | 0 / 0 / 0 / 0 | 14 | +0.0% / +0.0% / +0.0% / +0.0% | +0.0% / +0.0% | $10,000 | +15.0%, $65,036 | 1.15 |
| oracle | uniform | 14 / 14 / 14 / 14 | 0 | +117.9% / +115.4% / +108.1% / +94.2% | +13.1% / +8.0% | $195,722,680 | +15.0%, $65,036 | 116.73 |

## The checks fixed before the Sharadar run

Written down in test-results-30d.md (Next steps) and docs/sharadar.md before any Sharadar result: under the main delisting rule the picks beat comparable stocks with t >= 2, the sign-flip null doesn't, and the grader's account beats SPY at 20 bps. On Stooq data they are shown for reference only.

| Scores | Comparable-stock t >= 2 | Its null's comparable-stock t < 2 | Account at 20 bps vs SPY | All three |
|---|---|---|---|---|
| lgbm | 1.03: no | -2.29: yes | $8,878 (median of 5 random tie-breaks) vs $14,490: no | no |

