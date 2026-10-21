# stockai-30d results (horizon 21 trading days, top 10, subset no_collapse)

Test years 2022-2023, one graded decision every 21 trading days (22 in all). Rebalance days left ungraded: 1 whose month would overlap the next graded month, 1 whose month reaches the locked year.

## Top-10 hits on the month's 10 biggest gainers

| Scores | Graded days | Hits per day | Chance (uniform) | Chance (same volatility) | z vs uniform | z vs same volatility | Days tied at the cut | Hits among the 10 biggest losers per day | Chance of a loser hit (same volatility) | Tail hits that were gainers | Picks in the top / bottom decile | Picks' mean month return vs all | Rank IC (t; days > 0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 22 | 0.205 | 0.055 | 0.196 | +3.02 | +0.09 | 21 | 0.265 | 0.198 | 44% | 14.0% / 18.8% | +0.42% vs +0.83% (t -0.4) | +0.0115 (0.5; 9) |
| lgbm_null | 22 | 0.165 | 0.055 | 0.174 | +2.22 | -0.11 | 21 | 0.356 | 0.170 | 32% | 11.0% / 19.8% | -0.69% vs +0.83% (t -1.8) | -0.0086 (-0.3; 10) |
| momentum | 22 | 0.500 | 0.055 | 0.206 | +8.97 | +3.16 | 0 | 0.273 | 0.203 | 65% | 31.8% / 16.8% | +4.73% vs +0.83% (t 1.7) | +0.0355 (1.1; 16) |
| reversal | 22 | 0.182 | 0.055 | 0.331 | +2.56 | -1.27 | 0 | 0.409 | 0.304 | 31% | 20.9% / 25.0% | -0.38% vs +0.83% (t -0.9) | +0.0084 (0.3; 14) |
| high_vol | 22 | 0.682 | 0.055 | 0.581 | +12.64 | +0.66 | 0 | 0.682 | 0.514 | 50% | 20.9% / 31.8% | -0.06% vs +0.83% (t -0.4) | -0.0274 (-0.5; 10) |
| low_vol | 22 | 0.000 | 0.055 | 0.000 | -1.10 | n/a | 0 | 0.000 | 0.026 | n/a | 3.6% / 3.6% | -0.31% vs +0.83% (t -1.2) | +0.0274 (0.5; 12) |
| oracle | 22 | 0.091 | 0.055 | 0.068 | +0.73 | +0.42 | 0 | 0.000 | 0.047 | 100% | 16.4% / 0.9% | +5.55% vs +0.83% (t 6.4) | +0.1812 (31.9; 22) |

Chance (uniform): hits expected from 10 random picks among the same stocks. Chance (same volatility): each pick swapped for a random stock from its own fiftieth of the day's stocks by trailing 63-day volatility. Tail hits that were gainers: of the picks that landed among the 10 biggest gainers or the 10 biggest losers, the share that were gainers (picking volatile stocks with no sense of direction lands in both). The top decile is the tenth of stocks that rose most that month; a random pick lands there 10% of the time.

### lgbm by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 2.0 | 0.62 | 1.87 | 1.0 | 16% / 16% | +0.46% vs +0.52% | +0.047 |
| 2023 | 11 | 2.5 | 0.58 | 2.45 | 4.8 | 12% / 22% | +0.38% vs +1.15% | -0.024 |

### lgbm_null by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 0.5 | 0.62 | 0.65 | 2.1 | 8% / 10% | +0.02% vs +0.52% | +0.027 |
| 2023 | 11 | 3.2 | 0.58 | 3.18 | 5.8 | 14% / 30% | -1.40% vs +1.15% | -0.045 |

### momentum by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 5.0 | 0.62 | 1.55 | 2.0 | 29% / 16% | +2.26% vs +0.52% | +0.025 |
| 2023 | 11 | 6.0 | 0.58 | 2.98 | 4.0 | 35% / 17% | +7.20% vs +1.15% | +0.046 |

### reversal by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 1.0 | 0.62 | 2.93 | 3.0 | 19% / 17% | +0.93% vs +0.52% | +0.055 |
| 2023 | 11 | 3.0 | 0.58 | 4.34 | 6.0 | 23% / 33% | -1.69% vs +1.15% | -0.039 |

### high_vol by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 6.0 | 0.62 | 5.79 | 6.0 | 18% / 29% | -1.88% vs +0.52% | -0.009 |
| 2023 | 11 | 9.0 | 0.58 | 6.99 | 9.0 | 24% / 35% | +1.75% vs +1.15% | -0.046 |

### low_vol by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 0.0 | 0.62 | 0.00 | 0.0 | 5% / 4% | -0.88% vs +0.52% | +0.009 |
| 2023 | 11 | 0.0 | 0.58 | 0.00 | 0.0 | 2% / 4% | +0.26% vs +1.15% | +0.046 |

### oracle by test year

| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | Picks' month return vs all | Rank IC |
|---|---|---|---|---|---|---|---|---|
| 2022 | 11 | 2.0 | 0.62 | 0.87 | 0.0 | 16% / 1% | +5.32% vs +0.52% | +0.184 |
| 2023 | 11 | 0.0 | 0.58 | 0.64 | 0.0 | 16% / 1% | +5.78% vs +1.15% | +0.178 |

## Month-level view and the collapsed picks

| Scores | Hits vs same-volatility chance, month-level t | Months ahead of the average stock | Median month's excess | Share of the total excess from the 5 best months | Excess without those 5 (t) | Pick weight on stocks down >50% over the past year | Their return (pooled; averaged by month) | The other picks' return |
|---|---|---|---|---|---|---|---|---|
| lgbm | +0.11 | 7 of 22 | -1.22% | n/a | -2.34% (-3.40) | 0.0% | n/a; n/a | +0.42% |
| lgbm_null | -0.16 | 7 of 22 | -1.82% | n/a | -3.08% (-4.45) | 0.0% | n/a; n/a | -0.69% |
| momentum | +1.85 | 11 of 22 | +1.39% | 110% | -0.52% (-0.34) | 0.0% | n/a; n/a | +4.73% |
| reversal | -1.18 | 7 of 22 | -1.78% | n/a | -3.78% (-4.12) | 0.0% | n/a; n/a | -0.38% |
| high_vol | +0.88 | 7 of 22 | -3.01% | n/a | -5.84% (-4.70) | 0.0% | n/a; n/a | -0.06% |
| low_vol | n/a | 9 of 22 | -0.64% | n/a | -2.77% (-3.05) | 0.0% | n/a; n/a | -0.31% |
| oracle | +0.25 | 20 of 22 | +4.11% | 48% | +3.19% (6.68) | 0.0% | n/a; n/a | +5.55% |

## Against comparable stocks

Each pick against the 20 eligible stocks it didn't pick that are closest to it that day in percentile rank of trailing volatility and of past-year fall (stockai.h30.metrics.comparable_excess). Stocks that swung or fell alike as a group cancel out. The 5 best months are those with the highest raw excess.

| Scores | Excess over comparable stocks per month (t) | Months ahead | Without the 5 best months (t) |
|---|---|---|---|
| lgbm | -0.08% (-0.13) | 13 of 22 | -0.98% (-1.79) |
| lgbm_null | -1.63% (-1.69) | 7 of 22 | -3.19% (-3.99) |
| momentum | +3.43% (1.98) | 15 of 22 | +1.08% (0.68) |
| reversal | -1.82% (-1.26) | 9 of 22 | -3.61% (-2.37) |
| high_vol | -1.20% (-0.66) | 9 of 22 | -4.43% (-2.79) |
| low_vol | -0.74% (-2.15) | 7 of 22 | -1.17% (-3.14) |
| oracle | +4.32% (6.91) | 22 of 22 | +3.02% (7.18) |

## A simple monthly account in the picks (equal weight, from one graded month to the next)

| Scores | $10,000 at 0 bps | at 5 bps | at 20 bps | at 50 bps | Every eligible stock, equal weight (no costs) |
|---|---|---|---|---|---|
| lgbm | $10,470 | $10,276 | $9,715 | $8,679 | $11,512 |
| lgbm_null | $8,114 | $7,984 | $7,605 | $6,899 | $11,512 |
| momentum | $23,424 | $23,224 | $22,632 | $21,492 | $11,512 |
| reversal | $8,410 | $8,230 | $7,713 | $6,771 | $11,512 |
| high_vol | $7,936 | $7,841 | $7,563 | $7,034 | $11,512 |
| low_vol | $9,245 | $9,174 | $8,963 | $8,556 | $11,512 |
| oracle | $31,073 | $30,449 | $28,647 | $25,345 | $11,512 |

