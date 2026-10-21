Comparison years: 2022-2023 (only those a model has scores for). stockGPT seeds in the ensemble: [0]. Fingerprints graded: 8b5ac076267c83ab. IC z and long-short z: Stouffer over the years of the judge's yearly t. Returns: the $10,000 account's yearly return, averaged over the years. Picks' vol decile: mean trailing 63-day volatility decile of the names held (0 = calmest tenth of the day's candidates, 9 = most volatile), random accounts in brackets.

### Gate G1 (stockGPT ensemble, daily, delay 0, uniform random accounts)

| Check | Value | Passed |
|---|---|---|
| gross signal: rank IC in actual returns, combined z >= 2 | z = 4.44 (mean daily IC +0.0220, t > 0 in 2 of 2 years); census units z = 3.36 (IC +0.0144) | yes |
| nothing on the sign-flip null: combined z between -2 and 2 for both ICs | actual returns z = -0.17 (IC -0.0009); census units z = 1.03 (IC +0.0047); test years: 1 | yes |
| no tripwire: large-cap daily rank IC vs next-day return <= 0.05 | top 500 by dollar volume: mean +0.0259 (t 3.96), largest year +0.0303 | yes |
| **G1** | all three checks present | **PASSED** |

### Headline

| Model | Rebalance, delay, random match | Years | IC census (z) | IC actual returns (z; years > 0) | Long-short bps/day (z) | Years beating random, p < 0.1 (0 / 5 / 20 bps) | Picks, mean yearly return (0 / 5 / 20 bps) | Random median (0 / 5 / 20 bps) | SPY | Picks' vol decile (random) | Names picked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stockGPT, ensemble of the seeds | daily, delay 0, uniform | 2 | +0.0144 (3.36) | +0.0220 (4.44; 2) | 5.49 (1.59) | 0 / 0 / 0 | +17.6% / -2.6% / -44.6% | +12.6% / -6.6% / -46.6% | +20.5% | 1.3 (4.5) | 17.4 |
| stockGPT, ensemble of the seeds | weekly, delay 0, uniform | 2 | -0.0122 (-1.60) | +0.0080 (0.92; 2) | -3.31 (-1.11) | 0 / 0 / 0 | +0.3% / -3.5% / -14.1% | -0.7% / -4.5% / -15.0% | +20.5% | 1.3 (4.5) | 17.8 |
| stockGPT ensemble as daily percentile ranks (always 20 names), extra | daily, delay 0, uniform | 2 | +0.0144 (3.36) | +0.0220 (4.44; 2) | 5.49 (1.59) | 0 / 0 / 0 | +14.1% / -8.2% / -52.3% | +7.9% / -13.0% / -54.5% | +20.5% | 1.6 (4.5) | 20.0 |
| LightGBM | daily, delay 0, uniform | 2 | +0.0492 (6.90) | -0.0161 (-2.27; 0) | -1.96 (-0.43) | 0 / 0 / 0 | -2.6% / -15.5% / -44.8% | +8.5% / -5.8% / -38.3% | +20.5% | 6.2 (4.5) | 20.0 |
| LightGBM | weekly, delay 0, uniform | 2 | +0.1144 (8.57) | -0.0240 (-1.86; 0) | -5.63 (-1.41) | 0 / 0 / 0 | -14.9% / -17.9% / -26.5% | +8.0% / +4.1% / -6.6% | +20.5% | 6.0 (4.5) | 20.0 |
| LightGBM, actual-return target (added after the POC run), extra | daily, delay 0, uniform | 2 | -0.0100 (-1.80) | +0.0181 (2.82; 2) | 1.85 (0.39) | 1 / 1 / 1 | +28.8% / +4.1% / -45.1% | +8.0% / -12.6% / -53.7% | +20.5% | 6.6 (4.5) | 20.0 |
| LightGBM, actual-return target (added after the POC run), extra | weekly, delay 0, uniform | 2 | -0.0455 (-3.75) | -0.0007 (0.23; 1) | -3.57 (-0.29) | 1 / 1 / 1 | +13.2% / +8.1% / -6.1% | +8.0% / +3.2% / -10.2% | +20.5% | 6.4 (4.5) | 20.0 |
| 5-day reversal | weekly, delay 0, uniform | 2 | +0.0353 (3.85) | +0.0067 (0.71; 2) | 0.28 (0.03) | 0 / 0 / 0 | -14.0% / -18.0% / -28.8% | +8.1% / +3.2% / -10.3% | +20.5% | 8.3 (4.5) | 20.0 |
| 5-day reversal | daily, delay 0, uniform | 2 | +0.0186 (3.48) | +0.0102 (1.62; 2) | 6.56 (1.37) | 1 / 1 / 1 | +4.8% / -5.5% / -30.7% | +8.0% / -2.6% / -28.6% | +20.5% | 8.3 (4.5) | 20.0 |
| 1-day census reversal (-v[d]), extra | daily, delay 0, uniform | 2 | +0.0166 (3.28) | +0.0171 (3.38; 2) | 5.59 (1.70) | 1 / 1 / 1 | +16.7% / -8.5% / -56.0% | +8.4% / -14.8% / -58.7% | +20.5% | 5.1 (4.5) | 20.0 |
| 12-1 momentum | weekly, delay 0, uniform | 2 | -0.1088 (-7.38) | +0.0282 (1.72; 2) | 5.62 (1.16) | 1 / 1 / 1 | +28.5% / +27.3% / +24.0% | +8.3% / +7.3% / +4.5% | +20.5% | 7.6 (4.5) | 20.0 |
| 12-1 momentum | monthly, delay 0, uniform | 2 | -0.2116 (-7.45) | +0.0462 (1.48; 2) | 4.86 (0.82) | 1 / 1 / 1 | +31.5% / +30.9% / +29.2% | +8.2% / +7.7% / +6.2% | +20.5% | 7.6 (4.5) | 20.0 |

Percentile variant: rank ICs and long-short identical to the original scores in every year: yes; names picked per day 17.4 (original) against 20.0; years in which the original account held cash (under 1 name a day on average): 0.

### Every variant of the GPT ensemble, LightGBM and the reversal rules

| Model | Rebalance, delay, random match | Years | IC census (z) | IC actual returns (z; years > 0) | Long-short bps/day (z) | Years beating random, p < 0.1 (0 / 5 / 20 bps) | Picks, mean yearly return (0 / 5 / 20 bps) | Random median (0 / 5 / 20 bps) | SPY | Picks' vol decile (random) | Names picked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stockGPT, ensemble of the seeds | daily, delay 0, uniform | 2 | +0.0144 (3.36) | +0.0220 (4.44; 2) | 5.49 (1.59) | 0 / 0 / 0 | +17.6% / -2.6% / -44.6% | +12.6% / -6.6% / -46.6% | +20.5% | 1.3 (4.5) | 17.4 |
| stockGPT, ensemble of the seeds | daily, delay 0, volatility | 2 | +0.0144 (3.36) | +0.0220 (4.44; 2) | 5.49 (1.59) | 0 / 0 / 0 | +17.6% / -2.6% / -44.6% | +16.4% / -3.6% / -45.2% | +20.5% | 1.3 (1.4) | 17.4 |
| stockGPT, ensemble of the seeds | daily, delay 1, uniform | 2 | +0.0144 (3.36) | +0.0220 (4.44; 2) | 5.49 (1.59) | 0 / 0 / 0 | +1.7% / -15.9% / -52.4% | +2.2% / -15.3% / -51.7% | +20.5% | 1.3 (4.5) | 17.4 |
| stockGPT, ensemble of the seeds | daily, delay 1, volatility | 2 | +0.0144 (3.36) | +0.0220 (4.44; 2) | 5.49 (1.59) | 0 / 0 / 0 | +1.7% / -15.9% / -52.4% | +2.8% / -14.9% / -51.8% | +20.5% | 1.3 (1.4) | 17.4 |
| stockGPT, ensemble of the seeds | weekly, delay 0, uniform | 2 | -0.0122 (-1.60) | +0.0080 (0.92; 2) | -3.31 (-1.11) | 0 / 0 / 0 | +0.3% / -3.5% / -14.1% | -0.7% / -4.5% / -15.0% | +20.5% | 1.3 (4.5) | 17.8 |
| stockGPT, ensemble of the seeds | weekly, delay 0, volatility | 2 | -0.0122 (-1.60) | +0.0080 (0.92; 2) | -3.31 (-1.11) | 1 / 1 / 1 | +0.3% / -3.5% / -14.1% | -1.9% / -5.7% / -16.1% | +20.5% | 1.3 (1.4) | 17.8 |
| stockGPT, ensemble of the seeds | weekly, delay 1, uniform | 2 | -0.0122 (-1.60) | +0.0080 (0.92; 2) | -3.31 (-1.11) | 0 / 0 / 0 | -3.2% / -6.9% / -17.1% | -1.6% / -5.3% / -15.6% | +20.5% | 1.4 (4.5) | 17.8 |
| stockGPT, ensemble of the seeds | weekly, delay 1, volatility | 2 | -0.0122 (-1.60) | +0.0080 (0.92; 2) | -3.31 (-1.11) | 0 / 0 / 0 | -3.2% / -6.9% / -17.1% | -3.2% / -6.8% / -17.1% | +20.5% | 1.4 (1.4) | 17.8 |
| stockGPT ensemble as daily percentile ranks (always 20 names), extra | daily, delay 0, uniform | 2 | +0.0144 (3.36) | +0.0220 (4.44; 2) | 5.49 (1.59) | 0 / 0 / 0 | +14.1% / -8.2% / -52.3% | +7.9% / -13.0% / -54.5% | +20.5% | 1.6 (4.5) | 20.0 |
| stockGPT ensemble as daily percentile ranks (always 20 names), extra | daily, delay 0, volatility | 2 | +0.0144 (3.36) | +0.0220 (4.44; 2) | 5.49 (1.59) | 0 / 0 / 0 | +14.1% / -8.2% / -52.3% | +13.7% / -8.6% / -52.5% | +20.5% | 1.6 (1.7) | 20.0 |
| stockGPT ensemble as daily percentile ranks (always 20 names), extra | daily, delay 1, uniform | 2 | +0.0144 (3.36) | +0.0220 (4.44; 2) | 5.49 (1.59) | 0 / 0 / 0 | +5.6% / -14.9% / -55.6% | +8.7% / -12.3% / -54.0% | +20.5% | 1.6 (4.5) | 20.0 |
| stockGPT ensemble as daily percentile ranks (always 20 names), extra | weekly, delay 0, uniform | 2 | -0.0122 (-1.60) | +0.0080 (0.92; 2) | -3.31 (-1.11) | 0 / 0 / 0 | +4.0% / -0.4% / -12.6% | +7.9% / +3.4% / -9.2% | +20.5% | 1.6 (4.5) | 20.0 |
| LightGBM | daily, delay 0, uniform | 2 | +0.0492 (6.90) | -0.0161 (-2.27; 0) | -1.96 (-0.43) | 0 / 0 / 0 | -2.6% / -15.5% / -44.8% | +8.5% / -5.8% / -38.3% | +20.5% | 6.2 (4.5) | 20.0 |
| LightGBM | daily, delay 0, volatility | 2 | +0.0492 (6.90) | -0.0161 (-2.27; 0) | -1.96 (-0.43) | 0 / 0 / 0 | -2.6% / -15.5% / -44.8% | +7.5% / -7.0% / -39.5% | +20.5% | 6.2 (6.1) | 20.0 |
| LightGBM | daily, delay 1, uniform | 2 | +0.0492 (6.90) | -0.0161 (-2.27; 0) | -1.96 (-0.43) | 0 / 0 / 0 | -9.1% / -20.9% / -47.9% | +9.4% / -5.0% / -37.7% | +20.5% | 6.2 (4.5) | 20.0 |
| LightGBM | daily, delay 1, volatility | 2 | +0.0492 (6.90) | -0.0161 (-2.27; 0) | -1.96 (-0.43) | 0 / 0 / 0 | -9.1% / -20.9% / -47.9% | +7.7% / -6.7% / -39.2% | +20.5% | 6.2 (6.2) | 20.0 |
| LightGBM | weekly, delay 0, uniform | 2 | +0.1144 (8.57) | -0.0240 (-1.86; 0) | -5.63 (-1.41) | 0 / 0 / 0 | -14.9% / -17.9% / -26.5% | +8.0% / +4.1% / -6.6% | +20.5% | 6.0 (4.5) | 20.0 |
| LightGBM | weekly, delay 0, volatility | 2 | +0.1144 (8.57) | -0.0240 (-1.86; 0) | -5.63 (-1.41) | 0 / 0 / 0 | -14.9% / -17.9% / -26.5% | +4.5% / +0.8% / -9.8% | +20.5% | 6.0 (5.9) | 20.0 |
| LightGBM | weekly, delay 1, uniform | 2 | +0.1144 (8.57) | -0.0240 (-1.86; 0) | -5.63 (-1.41) | 0 / 0 / 0 | -9.5% / -12.7% / -21.8% | +9.1% / +5.3% / -5.5% | +20.5% | 6.0 (4.5) | 20.0 |
| LightGBM | weekly, delay 1, volatility | 2 | +0.1144 (8.57) | -0.0240 (-1.86; 0) | -5.63 (-1.41) | 0 / 0 / 0 | -9.5% / -12.7% / -21.8% | +9.1% / +5.2% / -5.7% | +20.5% | 6.0 (5.9) | 20.0 |
| LightGBM, actual-return target (added after the POC run), extra | daily, delay 0, uniform | 2 | -0.0100 (-1.80) | +0.0181 (2.82; 2) | 1.85 (0.39) | 1 / 1 / 1 | +28.8% / +4.1% / -45.1% | +8.0% / -12.6% / -53.7% | +20.5% | 6.6 (4.5) | 20.0 |
| LightGBM, actual-return target (added after the POC run), extra | daily, delay 0, volatility | 2 | -0.0100 (-1.80) | +0.0181 (2.82; 2) | 1.85 (0.39) | 1 / 1 / 1 | +28.8% / +4.1% / -45.1% | +6.4% / -14.0% / -54.7% | +20.5% | 6.6 (6.6) | 20.0 |
| LightGBM, actual-return target (added after the POC run), extra | daily, delay 1, uniform | 2 | -0.0100 (-1.80) | +0.0181 (2.82; 2) | 1.85 (0.39) | 1 / 1 / 1 | +3.5% / -16.3% / -55.7% | +9.3% / -11.5% / -53.0% | +20.5% | 6.6 (4.5) | 20.0 |
| LightGBM, actual-return target (added after the POC run), extra | daily, delay 1, volatility | 2 | -0.0100 (-1.80) | +0.0181 (2.82; 2) | 1.85 (0.39) | 1 / 1 / 1 | +3.5% / -16.3% / -55.7% | +5.5% / -14.7% / -55.0% | +20.5% | 6.6 (6.6) | 20.0 |
| LightGBM, actual-return target (added after the POC run), extra | weekly, delay 0, uniform | 2 | -0.0455 (-3.75) | -0.0007 (0.23; 1) | -3.57 (-0.29) | 1 / 1 / 1 | +13.2% / +8.1% / -6.1% | +8.0% / +3.2% / -10.2% | +20.5% | 6.4 (4.5) | 20.0 |
| LightGBM, actual-return target (added after the POC run), extra | weekly, delay 0, volatility | 2 | -0.0455 (-3.75) | -0.0007 (0.23; 1) | -3.57 (-0.29) | 0 / 0 / 0 | +13.2% / +8.1% / -6.1% | +14.5% / +9.3% / -5.0% | +20.5% | 6.4 (6.4) | 20.0 |
| LightGBM, actual-return target (added after the POC run), extra | weekly, delay 1, uniform | 2 | -0.0455 (-3.75) | -0.0007 (0.23; 1) | -3.57 (-0.29) | 1 / 1 / 1 | +23.3% / +17.8% / +2.5% | +9.1% / +4.2% / -9.1% | +20.5% | 6.4 (4.5) | 20.0 |
| LightGBM, actual-return target (added after the POC run), extra | weekly, delay 1, volatility | 2 | -0.0455 (-3.75) | -0.0007 (0.23; 1) | -3.57 (-0.29) | 0 / 0 / 0 | +23.3% / +17.8% / +2.5% | +19.8% / +14.4% / -0.4% | +20.5% | 6.4 (6.4) | 20.0 |
| 5-day reversal | daily, delay 0, uniform | 2 | +0.0186 (3.48) | +0.0102 (1.62; 2) | 6.56 (1.37) | 1 / 1 / 1 | +4.8% / -5.5% / -30.7% | +8.0% / -2.6% / -28.6% | +20.5% | 8.3 (4.5) | 20.0 |
| 5-day reversal | weekly, delay 0, uniform | 2 | +0.0353 (3.85) | +0.0067 (0.71; 2) | 0.28 (0.03) | 0 / 0 / 0 | -14.0% / -18.0% / -28.8% | +8.1% / +3.2% / -10.3% | +20.5% | 8.3 (4.5) | 20.0 |
| 5-day reversal | weekly, delay 0, volatility | 2 | +0.0353 (3.85) | +0.0067 (0.71; 2) | 0.28 (0.03) | 0 / 0 / 0 | -14.0% / -18.0% / -28.8% | +2.0% / -2.7% / -15.5% | +20.5% | 8.3 (8.2) | 20.0 |
| 1-day census reversal (-v[d]), extra | daily, delay 0, uniform | 2 | +0.0166 (3.28) | +0.0171 (3.38; 2) | 5.59 (1.70) | 1 / 1 / 1 | +16.7% / -8.5% / -56.0% | +8.4% / -14.8% / -58.7% | +20.5% | 5.1 (4.5) | 20.0 |
| 1-day census reversal (-v[d]), extra | daily, delay 0, volatility | 2 | +0.0166 (3.28) | +0.0171 (3.38; 2) | 5.59 (1.70) | 1 / 1 / 1 | +16.7% / -8.5% / -56.0% | +10.0% / -13.8% / -58.5% | +20.5% | 5.1 (5.1) | 20.0 |

### Seeds

| Model | Rebalance, delay, random match | Years | IC census (z) | IC actual returns (z; years > 0) | Long-short bps/day (z) | Years beating random, p < 0.1 (0 / 5 / 20 bps) | Picks, mean yearly return (0 / 5 / 20 bps) | Random median (0 / 5 / 20 bps) | SPY | Picks' vol decile (random) | Names picked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stockGPT, seed 0 | daily, delay 0, uniform | 2 | +0.0144 (3.36) | +0.0220 (4.44; 2) | 5.49 (1.59) | 0 / 0 / 0 | +17.6% / -2.6% / -44.6% | +12.6% / -6.6% / -46.6% | +20.5% | 1.3 (4.5) | 17.4 |

### Controls (judge, daily, delay 0, uniform)

| Model | Rebalance, delay, random match | Years | IC census (z) | IC actual returns (z; years > 0) | Long-short bps/day (z) | Years beating random, p < 0.1 (0 / 5 / 20 bps) | Picks, mean yearly return (0 / 5 / 20 bps) | Random median (0 / 5 / 20 bps) | SPY | Picks' vol decile (random) | Names picked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stockGPT, ensemble of the seeds | daily, delay 0, uniform | 2 | +0.0144 (3.36) | +0.0220 (4.44; 2) | 5.49 (1.59) | 0 / 0 / 0 | +17.6% / -2.6% / -44.6% | +12.6% / -6.6% / -46.6% | +20.5% | 1.3 (4.5) | 17.4 |
| GPT ensemble, scores used 1 day late | daily, delay 0, uniform | 2 | +0.0004 (0.10) | +0.0122 (2.47; 2) | -0.05 (-0.01) | 0 / 0 / 0 | +1.4% / -16.2% / -52.6% | +2.5% / -15.1% / -51.7% | +20.5% | 1.3 (4.5) | 17.3 |
| canary: GPT ensemble, scores 20 days stale | daily, delay 0, uniform | 2 | +0.0045 (1.09) | +0.0088 (1.77; 2) | 0.04 (-0.05) | 0 / 0 / 0 | +10.0% / -8.4% / -46.8% | +17.3% / -1.8% / -42.2% | +20.5% | 1.4 (4.5) | 16.6 |
| canary: GPT ensemble, dates permuted within year | daily, delay 0, uniform | 2 | +0.0041 (0.92) | +0.0102 (2.11; 2) | 2.50 (0.69) | 0 / 0 / 0 | +3.8% / -15.3% / -53.9% | +3.7% / -15.2% / -53.4% | +20.5% | 1.6 (4.5) | 17.4 |
| canary: GPT ensemble, scores a year (250 days) stale | daily, delay 0, uniform | 1 | +0.0028 (0.58) | +0.0068 (1.21; 1) | -1.54 (-0.42) | 0 / 0 / 0 | +7.2% / -9.6% / -45.9% | +6.9% / -9.7% / -45.7% | +27.0% | 1.9 (4.5) | 14.8 |
| static reference: -log volatility (252 days) | daily, delay 0, uniform | 2 | +0.0011 (0.24) | +0.0169 (1.61; 2) | -0.40 (-0.02) | 0 / 0 / 0 | +0.1% / -0.3% / -1.6% | +8.3% / +7.8% / +6.4% | +20.5% | 0.0 (4.5) | 20.0 |
| 1-day census reversal (-v[d]), extra | daily, delay 0, uniform | 2 | +0.0166 (3.28) | +0.0171 (3.38; 2) | 5.59 (1.70) | 1 / 1 / 1 | +16.7% / -8.5% / -56.0% | +8.4% / -14.8% / -58.7% | +20.5% | 5.1 (4.5) | 20.0 |
| reference canary: 1-day reversal, 20 days stale | daily, delay 0, uniform | 2 | +0.0002 (0.15) | +0.0005 (0.26; 1) | 1.53 (0.72) | 1 / 1 / 1 | +13.1% / -11.3% / -57.3% | +7.7% / -15.4% / -58.9% | +20.5% | 5.3 (4.5) | 20.0 |
| reference canary: 1-day reversal, dates permuted | daily, delay 0, uniform | 2 | -0.0027 (-0.11) | -0.0018 (-0.19; 1) | -2.46 (-0.40) | 0 / 0 / 0 | -14.2% / -33.1% / -68.3% | +8.0% / -15.5% / -59.6% | +20.5% | 4.8 (4.5) | 20.0 |
| stockGPT sign-flip null (seed 0) | daily, delay 0, uniform | 1 | +0.0047 (1.03) | -0.0009 (-0.17; 0) | 3.82 (0.85) | 0 / 0 / 0 | -1.2% / -22.2% / -62.0% | +7.0% / -15.5% / -58.5% | +14.1% | 6.1 (4.5) | 20.0 |

Canary readings (|combined z| < 2 counts as about 0):

- canary: GPT ensemble, scores 20 days stale: census units IC +0.0045 (z 1.09): about 0; actual returns IC +0.0088 (z 1.77): about 0
- canary: GPT ensemble, dates permuted within year: census units IC +0.0041 (z 0.92): about 0; actual returns IC +0.0102 (z 2.11): NOT about 0 (the static -log volatility score alone: actual returns IC +0.0169, z 1.61; census units z 0.24)
- canary: GPT ensemble, scores a year (250 days) stale: census units IC +0.0028 (z 0.58): about 0; actual returns IC +0.0068 (z 1.21): about 0
- reference canary: 1-day reversal, 20 days stale: census units IC +0.0002 (z 0.15): about 0; actual returns IC +0.0005 (z 0.26): about 0
- reference canary: 1-day reversal, dates permuted: census units IC -0.0027 (z -0.11): about 0; actual returns IC -0.0018 (z -0.19): about 0

### Large-cap tripwire (top 500 by 63-day dollar volume, daily rank IC vs next-day return)

| Scores | Mean daily IC | t | Days | Largest yearly mean | Tripped (> 0.05) |
|---|---|---|---|---|---|
| gpt_ensemble | +0.0259 | 3.96 | 503 | +0.0303 | no |
| gpt_null | +0.0003 | 0.04 | 251 | +0.0003 | no |
| lgbm | -0.0134 | -1.37 | 503 | -0.0134 | no |
| lgbm_ret | +0.0243 | 3.01 | 503 | +0.0274 | no |
| reversal5 | +0.0157 | 1.88 | 503 | +0.0221 | no |
| vreversal1 | +0.0212 | 3.07 | 503 | +0.0280 | no |

stockGPT ensemble by year: 2022 +0.0215; 2023 +0.0303

### What gpt_ensemble's scores resemble (inputs only: mean daily rank correlation over test days)

| Test year | v[d] | log volatility (252 days) | lgbm | reversal5 | vreversal1 | momentum_12_1 | seed0 |
|---|---|---|---|---|---|---|---|
| 2022 | -0.182 | -0.236 | -0.015 | +0.030 | +0.182 | +0.200 | +1.000 |
| 2023 | -0.612 | -0.507 | +0.117 | +0.149 | +0.612 | +0.139 | +1.000 |
| all | -0.397 | -0.372 | +0.051 | +0.090 | +0.397 | +0.169 | +1.000 |

### What lgbm's scores resemble (inputs only: mean daily rank correlation over test days)

| Test year | v[d] | log volatility (252 days) | momentum_12_1 | reversal5 | vreversal1 |
|---|---|---|---|---|---|
| 2022 | -0.129 | -0.070 | -0.570 | +0.263 | +0.129 |
| 2023 | -0.111 | -0.020 | -0.732 | +0.257 | +0.111 |
| all | -0.120 | -0.045 | -0.651 | +0.260 | +0.120 |

### What lgbm_ret's scores resemble (inputs only: mean daily rank correlation over test days)

| Test year | v[d] | log volatility (252 days) | momentum_12_1 | reversal5 | vreversal1 | gpt_ensemble | lgbm |
|---|---|---|---|---|---|---|---|
| 2022 | -0.303 | -0.223 | +0.300 | +0.181 | +0.303 | +0.272 | +0.041 |
| 2023 | -0.296 | -0.454 | +0.349 | +0.162 | +0.296 | +0.457 | -0.054 |
| all | -0.299 | -0.339 | +0.324 | +0.171 | +0.299 | +0.365 | -0.007 |

### stockGPT ensemble by test year (daily, delay 0, uniform random accounts)

| Test year | Val IC actual (mean of seeds) | Val IC v | Test IC census | Test IC actual | Long-short bps/day | Picks 0 bps | p | Picks 5 bps | p | Picks 20 bps | p | Random median (0 bps) | SPY | QQQ | Picks' vol decile | Names picked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022-06-28..2023-06-27 | +0.0247 | +0.0231 | +0.0110 (t 1.9) | +0.0193 (t 2.8) | 3.03 | +19.7% | 0.390 | +1.0% | 0.397 | -39.3% | 0.415 | +17.1% | +14.1% | +25.3% | 1.0 | 14.8 |
| 2023-06-28..2024-06-27 | +0.0190 | +0.0125 | +0.0177 (t 2.8) | +0.0246 (t 3.5) | 7.96 | +15.5% | 0.210 | -6.3% | 0.218 | -49.9% | 0.238 | +8.0% | +27.0% | +33.2% | 1.7 | 20.0 |

### stockGPT ensemble by test year (daily, delay 0, volatility-matched random accounts)

| Test year | Test IC census | Test IC actual | Long-short bps/day | Picks 0 bps | p | Picks 5 bps | p | Picks 20 bps | p | Random median (0 bps) | SPY | QQQ | Picks' vol decile | Names picked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022-06-28..2023-06-27 | +0.0110 (t 1.9) | +0.0193 (t 2.8) | 3.03 | +19.7% | 0.447 | +1.0% | 0.447 | -39.3% | 0.447 | +19.3% | +14.1% | +25.3% | 1.0 | 14.8 |
| 2023-06-28..2024-06-27 | +0.0177 (t 2.8) | +0.0246 (t 3.5) | 7.96 | +15.5% | 0.388 | -6.3% | 0.388 | -49.9% | 0.388 | +13.6% | +27.0% | +33.2% | 1.7 | 20.0 |

### Null by year

| Test year | Val IC actual | Val IC vs own (flipped) target | Test IC census | Test IC actual | Long-short bps/day | Picks 0 bps | p |
|---|---|---|---|---|---|---|---|
| 2022-06-28..2023-06-27 | -0.0012 | -0.0028 | +0.0047 (t 1.0) | -0.0009 (t -0.2) | 3.82 | -1.2% | 0.813 |

### LightGBM by test year (daily, delay 0, uniform random accounts)

| Test year | Val IC actual | Val IC v | Test IC census | Test IC actual | Long-short bps/day | Picks 0 bps | p | Picks 5 bps | p | Picks 20 bps | p | Random median (0 bps) | SPY | QQQ | Picks' vol decile | Names picked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022-06-28..2023-06-27 | -0.0148 | +0.0376 | +0.0448 (t 4.5) | -0.0139 (t -1.3) | 0.29 | -4.1% | 0.906 | -18.5% | 0.906 | -49.9% | 0.910 | +8.0% | +14.1% | +25.3% | 5.9 | 20.0 |
| 2023-06-28..2024-06-27 | -0.0036 | +0.0524 | +0.0536 (t 5.3) | -0.0182 (t -1.9) | -4.20 | -1.1% | 0.859 | -12.5% | 0.860 | -39.6% | 0.869 | +9.0% | +27.0% | +33.2% | 6.4 | 20.0 |

### LightGBM, actual-return target (added after the POC run), extra: by test year (daily, delay 0, uniform random accounts)

| Test year | Val IC actual | Val IC v | Test IC census | Test IC actual | Long-short bps/day | Picks 0 bps | p | Picks 5 bps | p | Picks 20 bps | p | Random median (0 bps) | SPY | QQQ | Picks' vol decile | Names picked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022-06-28..2023-06-27 | +0.0300 | +0.0011 | -0.0107 (t -1.3) | +0.0138 (t 1.4) | -0.05 | +10.9% | 0.353 | -10.8% | 0.364 | -53.6% | 0.385 | +7.4% | +14.1% | +25.3% | 6.8 | 20.0 |
| 2023-06-28..2024-06-27 | +0.0161 | -0.0043 | -0.0093 (t -1.2) | +0.0225 (t 2.6) | 3.74 | +46.8% | 0.000 | +19.1% | 0.000 | -36.5% | 0.000 | +8.7% | +27.0% | +33.2% | 6.5 | 20.0 |

### 5-day reversal by test year (weekly, delay 0, uniform random accounts)

| Test year | Test IC census | Test IC actual | Long-short bps/day | Picks 0 bps | p | Picks 5 bps | p | Picks 20 bps | p | Random median (0 bps) | SPY | QQQ | Picks' vol decile | Names picked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022-06-28..2023-06-27 | +0.0235 (t 1.4) | +0.0043 (t 0.3) | 1.02 | -23.2% | 1.000 | -26.8% | 1.000 | -36.5% | 1.000 | +7.9% | +14.1% | +25.3% | 8.1 | 20.0 |
| 2023-06-28..2024-06-27 | +0.0471 (t 4.0) | +0.0090 (t 0.7) | -0.45 | -4.8% | 0.935 | -9.1% | 0.935 | -21.0% | 0.938 | +8.4% | +27.0% | +33.2% | 8.4 | 20.0 |

### LightGBM training by year (validation only)

| Test year | Train rows | Best iteration | Val L2 / zero forecast | Val IC v | Val IC actual | Largest gain shares |
|---|---|---|---|---|---|---|
| 2022 | 3,999,247 | 30 | 0.9987 | +0.0376 | -0.0148 | ret_252 0.24, mkt_1 0.14, mkt_21 0.13 |
| 2023 | 3,999,327 | 29 | 0.9969 | +0.0524 | -0.0036 | ret_252 0.22, mkt_21 0.15, mkt_1 0.14 |

### LightGBM, actual-return target (added after the POC run), extra: training by year (validation only; the L2 is in units of its daily rank-gauss target)

| Test year | Train rows | Best iteration | Val L2 / zero forecast | Val IC v | Val IC actual | Largest gain shares |
|---|---|---|---|---|---|---|
| 2022 | 3,999,247 | 46 | 0.9988 | +0.0011 | +0.0300 | mkt_5 0.21, mkt_21 0.18, mkt_1 0.17 |
| 2023 | 3,999,327 | 17 | 0.9997 | -0.0043 | +0.0161 | mkt_5 0.21, mkt_1 0.17, mkt_21 0.17 |

### Timings

| Step (poc_log.txt) | Seconds |
|---|---|
| export | 0 |
| G0 | 4 |
| stockGPT walk-forward | 169 |
| sign-flip null | 80 |
| baselines | 55 |
| grading | 41 |
| controls | 51 |

stockGPT, per seed (train + score, all years): seed 0 2.7 min

| Test year | Train positions | Steps run / budget (mean of seeds) | Best step | Train s | Score s |
|---|---|---|---|---|---|
| 2022 | 11,018,748 | 472 / 704 | 118 | 51 | 23 |
| 2023 | 12,192,927 | 650 / 777 | 260 | 64 | 25 |

Null, per year (train + score): 2022 74 s

LightGBM: 27 s in all; per year (rows + fit + predict): 2022 7.6 s, 2023 7.6 s

LightGBM, actual-return target (extra): 27 s in all; per year (rows + fit + predict): 2022 7.8 s, 2023 6.5 s

### Ledger: every graded configuration

| File | Model | Variant | Years in comparison | All graded years |
|---|---|---|---|---|
| canary_permuted__r1_d0_uniform.json | canary: GPT ensemble, dates permuted within year | daily, delay 0, uniform | 2 | 2 |
| canary_shift20__r1_d0_uniform.json | canary: GPT ensemble, scores 20 days stale | daily, delay 0, uniform | 2 | 2 |
| canary_shift250__r1_d0_uniform.json | canary: GPT ensemble, scores a year (250 days) stale | daily, delay 0, uniform | 1 | 1 |
| canary_vreversal1_permuted__r1_d0_uniform.json | reference canary: 1-day reversal, dates permuted | daily, delay 0, uniform | 2 | 2 |
| canary_vreversal1_shift20__r1_d0_uniform.json | reference canary: 1-day reversal, 20 days stale | daily, delay 0, uniform | 2 | 2 |
| decay_shift1__r1_d0_uniform.json | GPT ensemble, scores used 1 day late | daily, delay 0, uniform | 2 | 2 |
| gpt_ensemble__r1_d0_uniform.json | stockGPT, ensemble of the seeds | daily, delay 0, uniform | 2 | 2 |
| gpt_ensemble__r1_d0_volatility.json | stockGPT, ensemble of the seeds | daily, delay 0, volatility | 2 | 2 |
| gpt_ensemble__r1_d1_uniform.json | stockGPT, ensemble of the seeds | daily, delay 1, uniform | 2 | 2 |
| gpt_ensemble__r1_d1_volatility.json | stockGPT, ensemble of the seeds | daily, delay 1, volatility | 2 | 2 |
| gpt_ensemble__r5_d0_uniform.json | stockGPT, ensemble of the seeds | weekly, delay 0, uniform | 2 | 2 |
| gpt_ensemble__r5_d0_volatility.json | stockGPT, ensemble of the seeds | weekly, delay 0, volatility | 2 | 2 |
| gpt_ensemble__r5_d1_uniform.json | stockGPT, ensemble of the seeds | weekly, delay 1, uniform | 2 | 2 |
| gpt_ensemble__r5_d1_volatility.json | stockGPT, ensemble of the seeds | weekly, delay 1, volatility | 2 | 2 |
| gpt_ensemble_pct__r1_d0_uniform.json | stockGPT ensemble as daily percentile ranks (always 20 names), extra | daily, delay 0, uniform | 2 | 2 |
| gpt_ensemble_pct__r1_d0_volatility.json | stockGPT ensemble as daily percentile ranks (always 20 names), extra | daily, delay 0, volatility | 2 | 2 |
| gpt_ensemble_pct__r1_d1_uniform.json | stockGPT ensemble as daily percentile ranks (always 20 names), extra | daily, delay 1, uniform | 2 | 2 |
| gpt_ensemble_pct__r5_d0_uniform.json | stockGPT ensemble as daily percentile ranks (always 20 names), extra | weekly, delay 0, uniform | 2 | 2 |
| gpt_null__r1_d0_uniform.json | stockGPT sign-flip null (seed 0) | daily, delay 0, uniform | 1 | 1 |
| gpt_seed0__r1_d0_uniform.json | stockGPT, seed 0 | daily, delay 0, uniform | 2 | 2 |
| lgbm__r1_d0_uniform.json | LightGBM | daily, delay 0, uniform | 2 | 2 |
| lgbm__r1_d0_volatility.json | LightGBM | daily, delay 0, volatility | 2 | 2 |
| lgbm__r1_d1_uniform.json | LightGBM | daily, delay 1, uniform | 2 | 2 |
| lgbm__r1_d1_volatility.json | LightGBM | daily, delay 1, volatility | 2 | 2 |
| lgbm__r5_d0_uniform.json | LightGBM | weekly, delay 0, uniform | 2 | 2 |
| lgbm__r5_d0_volatility.json | LightGBM | weekly, delay 0, volatility | 2 | 2 |
| lgbm__r5_d1_uniform.json | LightGBM | weekly, delay 1, uniform | 2 | 2 |
| lgbm__r5_d1_volatility.json | LightGBM | weekly, delay 1, volatility | 2 | 2 |
| lgbm_ret__r1_d0_uniform.json | LightGBM, actual-return target (added after the POC run), extra | daily, delay 0, uniform | 2 | 2 |
| lgbm_ret__r1_d0_volatility.json | LightGBM, actual-return target (added after the POC run), extra | daily, delay 0, volatility | 2 | 2 |
| lgbm_ret__r1_d1_uniform.json | LightGBM, actual-return target (added after the POC run), extra | daily, delay 1, uniform | 2 | 2 |
| lgbm_ret__r1_d1_volatility.json | LightGBM, actual-return target (added after the POC run), extra | daily, delay 1, volatility | 2 | 2 |
| lgbm_ret__r5_d0_uniform.json | LightGBM, actual-return target (added after the POC run), extra | weekly, delay 0, uniform | 2 | 2 |
| lgbm_ret__r5_d0_volatility.json | LightGBM, actual-return target (added after the POC run), extra | weekly, delay 0, volatility | 2 | 2 |
| lgbm_ret__r5_d1_uniform.json | LightGBM, actual-return target (added after the POC run), extra | weekly, delay 1, uniform | 2 | 2 |
| lgbm_ret__r5_d1_volatility.json | LightGBM, actual-return target (added after the POC run), extra | weekly, delay 1, volatility | 2 | 2 |
| lowvol__r1_d0_uniform.json | static reference: -log volatility (252 days) | daily, delay 0, uniform | 2 | 2 |
| momentum_12_1__r21_d0_uniform.json | 12-1 momentum | monthly, delay 0, uniform | 2 | 17 |
| momentum_12_1__r5_d0_uniform.json | 12-1 momentum | weekly, delay 0, uniform | 2 | 17 |
| reversal5__r1_d0_uniform.json | 5-day reversal | daily, delay 0, uniform | 2 | 17 |
| reversal5__r5_d0_uniform.json | 5-day reversal | weekly, delay 0, uniform | 2 | 17 |
| reversal5__r5_d0_volatility.json | 5-day reversal | weekly, delay 0, volatility | 2 | 17 |
| vreversal1__r1_d0_uniform.json | 1-day census reversal (-v[d]), extra | daily, delay 0, uniform | 2 | 17 |
| vreversal1__r1_d0_volatility.json | 1-day census reversal (-v[d]), extra | daily, delay 0, volatility | 2 | 17 |
