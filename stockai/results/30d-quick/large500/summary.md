# stockai-30d results (horizon 21 trading days, top 10, subset large500)

Test years 2022-2023, one graded decision every 21 trading days (22 in all). Rebalance days left ungraded: 1 whose month would overlap the next graded month, 1 whose month reaches the locked year.

## Top-10 hits on the month's 10 biggest gainers

| Scores | Graded days | Hits per day | Chance (uniform) | Chance (same volatility) | z vs uniform | z vs same volatility | Days tied at the cut | Hits among the 10 biggest losers per day | Chance of a loser hit (same volatility) | Tail hits that were gainers | Picks in the top / bottom decile | Picks' mean month return vs all | Rank IC (t; days > 0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 22 | 0.563 | 0.218 | 0.627 | +3.53 | -0.44 | 19 | 0.743 | 0.588 | 43% | 15.0% / 15.6% | +0.81% vs +1.08% (t -0.3) | +0.0240 (0.8; 15) |
| lgbm_null | 22 | 0.795 | 0.218 | 0.627 | +5.91 | +1.15 | 22 | 0.503 | 0.489 | 61% | 18.5% / 14.7% | +3.86% vs +1.08% (t 1.5) | +0.0026 (0.1; 8) |
| momentum | 22 | 0.909 | 0.218 | 0.639 | +7.08 | +1.77 | 0 | 0.364 | 0.494 | 71% | 23.6% / 15.9% | +3.76% vs +1.08% (t 1.3) | +0.0419 (1.0; 13) |
| reversal | 22 | 1.091 | 0.218 | 0.840 | +8.95 | +1.47 | 0 | 0.682 | 0.892 | 62% | 25.0% / 16.4% | +3.55% vs +1.08% (t 1.5) | +0.0132 (0.4; 13) |
| high_vol | 22 | 1.909 | 0.218 | 1.808 | +17.33 | +0.42 | 0 | 1.909 | 1.929 | 50% | 28.2% / 32.3% | +2.97% vs +1.08% (t 0.6) | +0.0011 (0.0; 12) |
| low_vol | 22 | 0.000 | 0.218 | 0.000 | -2.24 | n/a | 0 | 0.045 | 0.045 | 0% | 2.3% / 6.8% | +0.35% vs +1.08% (t -0.7) | -0.0011 (-0.0; 10) |
| oracle | 22 | 0.318 | 0.218 | 0.270 | +1.02 | +0.50 | 0 | 0.045 | 0.193 | 88% | 15.5% / 3.6% | +4.42% vs +1.08% (t 4.3) | +0.1843 (15.2; 22) |

Chance (uniform): hits expected from 10 random picks among the same stocks. Chance (same volatility): each pick swapped for a random stock from its own fiftieth of the day's stocks by trailing 63-day volatility. Tail hits that were gainers: of the picks that landed among the 10 biggest gainers or the 10 biggest losers, the share that were gainers (picking volatile stocks with no sense of direction lands in both). The top decile is the tenth of stocks that rose most that month; a random pick lands there 10% of the time.

### lgbm by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 7.4 | 2.43 | 8.38 | 9.0 | 23% / 19% | +1.38% vs +0.71% | +0.078 |
| 2023 | 11 | 5.0 | 2.38 | 5.42 | 7.4 | 7% / 12% | +0.25% vs +1.44% | -0.030 |

### lgbm_null by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 5.4 | 2.43 | 3.75 | 2.3 | 14% / 9% | +3.34% vs +0.71% | +0.043 |
| 2023 | 11 | 12.1 | 2.38 | 10.04 | 8.7 | 24% / 20% | +4.39% vs +1.44% | -0.038 |

### momentum by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 5.0 | 2.43 | 5.64 | 3.0 | 17% / 17% | +0.94% vs +0.71% | +0.028 |
| 2023 | 11 | 15.0 | 2.38 | 8.42 | 5.0 | 30% / 15% | +6.59% vs +1.44% | +0.056 |

### reversal by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 12.0 | 2.43 | 7.89 | 5.0 | 27% / 14% | +4.89% vs +0.71% | +0.080 |
| 2023 | 11 | 12.0 | 2.38 | 10.59 | 10.0 | 23% / 19% | +2.22% vs +1.44% | -0.054 |

### high_vol by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 18.0 | 2.43 | 17.11 | 18.0 | 26% / 32% | +1.95% vs +0.71% | -0.002 |
| 2023 | 11 | 24.0 | 2.38 | 22.67 | 24.0 | 30% / 33% | +3.98% vs +1.44% | +0.005 |

### low_vol by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 0.0 | 2.43 | 0.00 | 0.0 | 4% / 7% | +0.21% vs +0.71% | +0.002 |
| 2023 | 11 | 0.0 | 2.38 | 0.00 | 1.0 | 1% / 6% | +0.49% vs +1.44% | -0.005 |

### oracle by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 2.0 | 2.43 | 2.31 | 1.0 | 15% / 2% | +4.17% vs +0.71% | +0.191 |
| 2023 | 11 | 5.0 | 2.38 | 3.62 | 0.0 | 15% / 5% | +4.67% vs +1.44% | +0.178 |

## Month-level view and the collapsed picks

| Scores | Hits vs same-volatility chance, month-level t | Months ahead of the average stock | Median month's excess | Share of the total excess from the 5 best months | Excess without those 5 (t) | Pick weight on stocks down >50% over the past year | Their return (pooled; averaged by month) | The other picks' return |
|---|---|---|---|---|---|---|---|---|
| lgbm | -0.60 | 8 of 22 | -1.16% | n/a | -2.04% (-3.63) | 39.1% | +0.73%; +1.81% | +0.87% |
| lgbm_null | +1.15 | 11 of 22 | -0.00% | 131% | -1.13% (-2.18) | 14.5% | +7.41%; -2.60% | +3.26% |
| momentum | +1.12 | 11 of 22 | -0.16% | 143% | -1.49% (-1.66) | 0.0% | n/a; n/a | +3.76% |
| reversal | +1.29 | 13 of 22 | +1.55% | 110% | -0.32% (-0.29) | 33.6% | +3.90%; +2.72% | +3.38% |
| high_vol | +1.31 | 11 of 22 | -0.19% | 274% | -4.25% (-2.37) | 47.3% | +0.97%; -1.00% | +4.75% |
| low_vol | n/a | 10 of 22 | -0.89% | n/a | -2.56% (-2.69) | 0.0% | n/a; n/a | +0.35% |
| oracle | +0.46 | 19 of 22 | +1.95% | 60% | +1.72% (3.49) | 5.5% | +15.12%; +15.83% | +3.80% |

## Against comparable stocks

Each pick against the 20 eligible stocks it didn't pick that are closest to it that day in percentile rank of trailing volatility and of past-year fall (stockai.h30.metrics.comparable_excess). Stocks that swung or fell alike as a group cancel out. The 5 best months are those with the highest raw excess.

| Scores | Excess over comparable stocks per month (t) | Months ahead | Without the 5 best months (t) |
|---|---|---|---|
| lgbm | -0.36% (-0.65) | 13 of 22 | -0.86% (-1.49) |
| lgbm_null | +2.88% (1.43) | 12 of 22 | -0.99% (-1.58) |
| momentum | +1.73% (0.88) | 10 of 22 | -0.50% (-0.61) |
| reversal | +2.19% (1.95) | 16 of 22 | +0.84% (0.76) |
| high_vol | +1.89% (0.80) | 10 of 22 | -2.56% (-1.50) |
| low_vol | -0.10% (-0.27) | 10 of 22 | -0.71% (-1.91) |
| oracle | +2.78% (4.30) | 17 of 22 | +1.66% (3.40) |

## A simple monthly account in the picks (equal weight, from one graded month to the next)

| Scores | $10,000 at 0 bps | at 5 bps | at 20 bps | at 50 bps | Every eligible stock, equal weight (no costs) |
|---|---|---|---|---|---|
| lgbm | $11,259 | $11,068 | $10,513 | $9,482 | $12,203 |
| lgbm_null | $20,093 | $19,817 | $19,011 | $17,492 | $12,203 |
| momentum | $19,079 | $18,926 | $18,473 | $17,598 | $12,203 |
| reversal | $18,892 | $18,524 | $17,458 | $15,499 | $12,203 |
| high_vol | $14,015 | $13,926 | $13,665 | $13,157 | $12,203 |
| low_vol | $10,695 | $10,596 | $10,304 | $9,743 | $12,203 |
| oracle | $24,484 | $23,991 | $22,569 | $19,961 | $12,203 |

