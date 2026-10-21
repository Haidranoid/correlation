#### Check first: the headline numbers before any haircut

| | The file | Reproduced here |
|---|---|---|
| Picks' month return vs the average stock (raw scores, tied picks share) | +3.21% vs +1.03% (t 2.8) | +3.21% vs +1.03% (t 2.77) |
| Extra over comparable stocks | +2.05 points (t 3.6) | +2.05 (t 3.57) |
| $10,000 after 14 years, tie-break seed 2 (the median run) | $230,343 at 20 bps; $100,417 at 50 bps | $230,343 at 20 bps; $100,417 at 50 bps; $348,313 at 5 bps (every yearly return within 7e-13 points of the grader's) |
| SPY over the same 14 years | $65,036 | $65,036 (Stooq store); $65,715 total return (pubdata) |
| Seed 2's own picks, 162 months (the grid starts here) | | +3.08% vs +1.03% (t 2.62); +1.87 over comparable stocks (t 3.14) |

#### Part 1: raw disappearance rates (2,264 stocks alive on 2025-06-27)

| Stocks on 2025-06-27 | Stocks | Disappeared | Over the window | Per month |
|---|---|---|---|---|
| All | 2,264 | 123 | 5.4% | 0.37% |
| Past 12 months below −80% | 8 | 3 | 37.5% | 3.08% |
| −80% to −50% | 87 | 7 | 8.0% | 0.56% |
| −50% to 0 | 826 | 51 | 6.2% | 0.42% |
| Above 0 | 1,343 | 62 | 4.6% | 0.31% |
| Price below $5 | 40 | 8 | 20.0% | 1.47% |
| Price $5 or more | 2,224 | 115 | 5.2% | 0.35% |

#### Part 1: the hazard model

| Regressor at 2025-06-27 | Coefficient | SE | z | Odds ratio |
|---|---|---|---|---|
| Intercept | -3.052 | 0.111 | -27.46 | 0.05 |
| 12-month return (per sd) | -0.023 | 0.109 | -0.21 | 0.98 |
| Down more than 50% (flag) | -0.060 | 0.420 | -0.14 | 0.94 |
| 63-day volatility (per sd) | +0.032 | 0.086 | +0.37 | 1.03 |
| Log price (per sd) | -0.488 | 0.134 | -3.64 | 0.61 |
| Log median dollar volume, 63 days (per sd) | -0.276 | 0.126 | -2.19 | 0.76 |

AUC 0.67, McFadden R² 0.050; m = 316 / 21 = 15.05.

Calibration by tenth of predicted probability (predicted / actual over the window): 1.4 / 1.3; 2.2 / 3.5; 2.8 / 2.2; 3.5 / 3.5; 4.3 / 6.2; 5.1 / 4.0; 6.1 / 6.2; 7.3 / 4.9; 9.0 / 8.4; 12.7 / 14.1.

Monthly hazards the model gives in 2010-2023: picks 0.54% (median 0.46%, 90th percentile 0.97%); their comparable stocks 0.53%; the average eligible stock 0.40%; collapsed picks (32.5% of picks) 0.77%.

#### Part 1: stocks that ended early in the newer store

| Ticker | Last close | Return from 2025-06-27 to the last close | Last 21 trading days | 12-month return at 2025-06-27 |
|---|---|---|---|---|
| CRNX | 2026-08-31 | +183.5% | +1.7% | -33% |
| CSAN | 2026-09-18 | -44.0% | +8.3% | -49% |
| DBRG | 2026-09-29 | +55.8% | +0.4% | -23% |
| GBTG | 2026-09-28 | +47.3% | +0.4% | +1% |
| HLX | 2026-09-01 | +68.8% | +14.5% | -45% |
| LEG | 2026-08-26 | -0.5% | -14.3% | -17% |

#### Part 2: the haircut grid (seed 2 picks, 162 months; accounts over 14 years)

| L | k | Extra over comparable stocks, points a month (t) | Extra over the average stock (t) | $10,000 at 5 bps | at 20 bps | Rule passes |
|---|---|---|---|---|---|---|
| none | 0 | +1.87 (3.14) | +2.04 (2.62) | $348,313 | $230,343 | yes |
| -30% | 1 | +1.85 (3.14) | +1.98 (2.56) | $258,996 | $171,151 | yes |
| -30% | 2 | +1.66 (2.85) | +1.78 (2.35) | $192,430 | $127,069 | yes |
| -30% | 3 | +1.47 (2.55) | +1.59 (2.12) | $142,860 | $94,266 | yes |
| -55% | 1 | +1.85 (3.14) | +1.94 (2.52) | $207,328 | $136,932 | yes |
| -55% | 2 | +1.52 (2.62) | +1.61 (2.13) | $123,161 | $81,239 | yes |
| -55% | 3 | +1.20 (2.09) | +1.29 (1.73) | $73,015 | $48,099 | **no** |
| -100% | 1 | +1.85 (3.14) | +1.88 (2.45) | $138,785 | $91,571 | yes |
| -100% | 2 | +1.28 (2.21) | +1.31 (1.74) | $54,970 | $36,186 | **no** |
| -100% | 3 | +0.71 (1.24) | +0.74 (1.00) | $21,641 | $14,213 | **no** |

SPY: $65,715 total return (the bar); $65,036 in the Stooq store. Benchmarks use k = 1.

#### Break-even k (base hazards)

| L | Comparable-stock t falls to 2 at k = | 20 bps account falls to SPY at k = | Rule passes up to k = |
|---|---|---|---|
| -30% | 4.79 | 4.20 | 4.20 |
| -55% | 3.16 | 2.41 | 2.41 |
| -100% | 2.22 | 1.36 | 1.36 |

#### Crisis sensitivity

| L | k | Hazards doubled in months with SPY 20% below its 1-year high: comparable (t); 20 bps account | Hazards doubled in every month: comparable (t); 20 bps account |
|---|---|---|---|
| -30% | 1 | +1.85 (3.14); $169,597 | +1.83 (3.14); $127,068 |
| -30% | 2 | +1.65 (2.84); $124,753 | +1.44 (2.54); $69,875 |
| -30% | 3 | +1.45 (2.53); $91,678 | +1.06 (1.92); $38,300 **fails** |
| -55% | 1 | +1.85 (3.14); $135,172 | +1.83 (3.14); $81,238 |
| -55% | 2 | +1.51 (2.61); $79,141 | +1.17 (2.07); $28,419 **fails** |
| -55% | 3 | +1.17 (2.06); $46,228 **fails** | +0.52 (0.94); $9,859 **fails** |
| -100% | 1 | +1.84 (3.13); $89,771 | +1.82 (3.13); $36,186 **fails** |
| -100% | 2 | +1.26 (2.18); $34,755 **fails** | +0.68 (1.21); $5,548 **fails** |
| -100% | 3 | +0.67 (1.19); $13,365 **fails** | -0.46 (-0.83); $830 **fails** |

Break-even k for comparable t = 2 at L = -30%: 4.66 (drawdown months doubled), 2.87 (all doubled).
Break-even k for comparable t = 2 at L = -55%: 3.10 (drawdown months doubled), 2.06 (all doubled).
Break-even k for comparable t = 2 at L = -100%: 2.18 (drawdown months doubled), 1.60 (all doubled).

Without log price (L = -30%, k = 1): comparable +1.85 (t 3.14), account $171,764 at 20 bps; passes: True.
Without log price (L = -55%, k = 2): comparable +1.52 (t 2.62), account $81,361 at 20 bps; passes: True.

Verdict: Worth the Sharadar rerun; against the Stooq SPY: Worth the Sharadar rerun; drawdown crisis: Worth the Sharadar rerun; all hazards doubled: Unresolved; without log price: Worth the Sharadar rerun.
