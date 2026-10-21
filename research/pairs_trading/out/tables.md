Primary configuration: export_div_all

### Tests (committed capital, net of 5 bps a side)

| Test | Number | Pass needs | Result |
|---|---|---|---|
| 1. 2006–2019 mean monthly net return | +0.002% a month (+0.03% a year), Newey-West t +0.04 (plain t +0.04), 167 months | t ≥ 2 | Fail |
| 2. Beats random pairs, 2006–2019 | book +0.03% a year; random books median +1.55, 95th percentile +3.09; 94% of random books did as well | above the 95th percentile | Fail |
| 3a. 2020–2025-06, same sign | +0.239% a month (+2.87% a year), t +1.38 | same sign as test 1 | Pass |
| 3b. The 16 unseen months, same sign | +0.248% a month (+2.97% a year), t +3.00, 16 months | same sign as test 1 | Pass |

### Book returns by period (% a month; Newey-West t, 6 lags)

| Prices | Dividends | Stocks | Period | Committed gross | t | Committed net 5 bps | t | Employed gross | Employed net 5 bps | t | FF5 + UMD alpha, net, % a year | t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| export | fixed | all | 2006-2019 | +0.057 | +0.89 | +0.002 | +0.04 | +0.063 | +0.007 | +0.10 | +0.23 | +0.40 |
| export | fixed | all | 2020-2025-06 | +0.295 | +1.67 | +0.239 | +1.38 | +0.299 | +0.241 | +1.39 | +3.90 | +1.90 |
| export | fixed | all | 2006-2025-06 | +0.125 | +1.78 | +0.070 | +1.01 | +0.130 | +0.074 | +1.05 | +1.14 | +1.48 |
| export | fixed | 500 largest | 2006-2019 | +0.042 | +0.60 | -0.010 | -0.14 | +0.044 | -0.009 | -0.13 | +0.01 | +0.01 |
| export | fixed | 500 largest | 2020-2025-06 | +0.137 | +1.51 | +0.083 | +0.94 | +0.144 | +0.088 | +0.96 | +1.84 | +2.17 |
| export | fixed | 500 largest | 2006-2025-06 | +0.069 | +1.21 | +0.017 | +0.29 | +0.073 | +0.019 | +0.32 | +0.41 | +0.75 |
| export | none | all | 2006-2019 | +0.057 | +0.89 | +0.002 | +0.04 | +0.063 | +0.007 | +0.10 | +0.23 | +0.40 |
| export | none | all | 2020-2025-06 | +0.322 | +1.83 | +0.266 | +1.55 | +0.326 | +0.268 | +1.55 | +4.19 | +2.06 |
| export | none | all | 2006-2025-06 | +0.133 | +1.88 | +0.078 | +1.12 | +0.138 | +0.082 | +1.15 | +1.23 | +1.57 |
| export | none | 500 largest | 2006-2019 | +0.042 | +0.60 | -0.010 | -0.14 | +0.044 | -0.009 | -0.13 | +0.01 | +0.01 |
| export | none | 500 largest | 2020-2025-06 | +0.107 | +1.10 | +0.054 | +0.58 | +0.111 | +0.057 | +0.57 | +1.59 | +1.70 |
| export | none | 500 largest | 2006-2025-06 | +0.061 | +1.05 | +0.008 | +0.15 | +0.063 | +0.010 | +0.16 | +0.33 | +0.59 |
| store | fixed | all | 2006-2019 | +0.057 | +0.88 | +0.002 | +0.03 | +0.062 | +0.007 | +0.10 | +0.23 | +0.39 |
| store | fixed | all | 2020-2025-06 | +0.295 | +1.67 | +0.239 | +1.38 | +0.299 | +0.241 | +1.39 | +3.90 | +1.90 |
| store | fixed | all | 2006-2025-06 | +0.125 | +1.78 | +0.070 | +1.01 | +0.130 | +0.074 | +1.04 | +1.14 | +1.48 |
| store | fixed | 500 largest | 2006-2019 | +0.042 | +0.60 | -0.010 | -0.14 | +0.044 | -0.009 | -0.13 | +0.01 | +0.01 |
| store | fixed | 500 largest | 2020-2025-06 | +0.137 | +1.51 | +0.083 | +0.94 | +0.144 | +0.088 | +0.96 | +1.84 | +2.17 |
| store | fixed | 500 largest | 2006-2025-06 | +0.069 | +1.21 | +0.017 | +0.29 | +0.073 | +0.019 | +0.32 | +0.41 | +0.75 |
| store | none | all | 2006-2019 | +0.057 | +0.88 | +0.002 | +0.03 | +0.062 | +0.007 | +0.10 | +0.23 | +0.39 |
| store | none | all | 2020-2025-06 | +0.322 | +1.83 | +0.266 | +1.55 | +0.326 | +0.268 | +1.55 | +4.19 | +2.06 |
| store | none | all | 2006-2025-06 | +0.133 | +1.88 | +0.078 | +1.11 | +0.138 | +0.081 | +1.14 | +1.23 | +1.57 |
| store | none | 500 largest | 2006-2019 | +0.042 | +0.60 | -0.010 | -0.14 | +0.044 | -0.009 | -0.13 | +0.01 | +0.01 |
| store | none | 500 largest | 2020-2025-06 | +0.107 | +1.10 | +0.054 | +0.58 | +0.111 | +0.057 | +0.57 | +1.59 | +1.70 |
| store | none | 500 largest | 2006-2025-06 | +0.061 | +1.05 | +0.008 | +0.14 | +0.063 | +0.010 | +0.16 | +0.32 | +0.58 |

### Against 1,000 random-pair books (committed capital, % a year)

| Stocks | Period | Costs | Book | Random median | Random 5th–95th percentile | Book − median | Random books ≥ book |
|---|---|---|---|---|---|---|---|
| all | 2006-2019 | 0 bps | +0.69 | +1.94 | +0.36 to +3.48 | -1.25 | 89.0% |
| all | 2006-2019 | 5 bps | +0.03 | +1.55 | -0.03 to +3.09 | -1.52 | 93.9% |
| all | 2020-2025-06 | 0 bps | +3.54 | -0.62 | -4.14 to +2.94 | +4.16 | 2.7% |
| all | 2020-2025-06 | 5 bps | +2.87 | -1.03 | -4.54 to +2.53 | +3.90 | 3.5% |
| all | 2006-2025-06 | 0 bps | +1.50 | +1.20 | -0.41 to +2.67 | +0.30 | 38.7% |
| all | 2006-2025-06 | 5 bps | +0.84 | +0.81 | -0.80 to +2.27 | +0.04 | 48.3% |
| 500 largest | 2006-2019 | 0 bps | +0.51 | +1.60 | +0.25 to +3.00 | -1.09 | 90.0% |
| 500 largest | 2006-2019 | 5 bps | -0.12 | +1.21 | -0.14 to +2.61 | -1.33 | 94.6% |
| 500 largest | 2020-2025-06 | 0 bps | +1.64 | -1.57 | -4.62 to +1.38 | +3.20 | 3.5% |
| 500 largest | 2020-2025-06 | 5 bps | +1.00 | -1.99 | -5.03 to +0.96 | +2.99 | 4.9% |
| 500 largest | 2006-2025-06 | 0 bps | +0.83 | +0.72 | -0.65 to +1.98 | +0.11 | 43.8% |
| 500 largest | 2006-2025-06 | 5 bps | +0.20 | +0.32 | -1.04 to +1.58 | -0.12 | 55.7% |

### Sub-periods, committed capital (% a year)

| Period | Gross, dividends fixed | Gross, no fix | Net 5 bps, fixed | t | Random median, gross |
|---|---|---|---|---|---|
| 2006-2009 | +2.50 | +2.50 | +1.80 | +0.86 | +8.22 |
| 2010-2014 | +0.89 | +0.89 | +0.24 | +0.29 | -1.24 |
| 2015-2019 | -0.94 | -0.94 | -1.57 | -2.07 | +0.10 |
| 2020-2022-06 | +5.95 | +5.95 | +5.23 | +1.23 | +1.68 |
| 2022-07-2025-06 | +1.58 | +2.17 | +0.96 | +0.99 | -2.41 |

### By calendar year, committed capital, gross (% a year)

| Year | Book | Random median | Book − median |
|---|---|---|---|
| 2006 | +0.79 | +5.02 | -4.23 |
| 2007 | -3.49 | -6.60 | +3.10 |
| 2008 | +3.03 | -1.02 | +4.05 |
| 2009 | +9.52 | +35.08 | -25.56 |
| 2010 | +0.49 | +1.31 | -0.81 |
| 2011 | -0.78 | -2.77 | +1.99 |
| 2012 | +2.87 | +0.17 | +2.70 |
| 2013 | -0.77 | -4.77 | +4.00 |
| 2014 | +2.65 | -0.14 | +2.79 |
| 2015 | -2.09 | -7.09 | +5.00 |
| 2016 | +1.02 | +9.46 | -8.44 |
| 2017 | +0.47 | -1.96 | +2.43 |
| 2018 | -0.03 | +0.20 | -0.22 |
| 2019 | -4.06 | -0.52 | -3.54 |
| 2020 | +13.50 | +2.75 | +10.75 |
| 2021 | +0.10 | +4.36 | -4.26 |
| 2022 | +1.44 | -4.26 | +5.70 |
| 2023 | +2.36 | +2.27 | +0.08 |
| 2024 | -0.34 | -6.86 | +6.52 |
| 2025 (to 06-27) | +4.64 | -3.28 | +7.92 |

### Trading statistics (per six-month portfolio of 20 pairs)

| | GGR 1963–2002, top 20, wait one day | Ours, formations of 2006–2019 | Ours, 2020–2025-06 | 500 largest, 2006–2019 |
|---|---|---|---|---|
| Universe (median stocks) |  | 1033 | 2005 | 499 |
| Pairs that open (of 20) | 19.30 | 19.54 | 19.38 | 19.53 |
| Round trips per pair | 1.96 | 1.62 | 1.63 | 1.55 |
| Months open per pair | 3.76 | 4.10 | 4.10 | 4.12 |
| 2σ trigger (normalized price gap) | 0.053 | 0.040 | 0.041 | 0.043 |
| Pairs that never open |  | 0.023 | 0.031 | 0.024 |
| Trades closed at the period end, not by converging |  | 0.504 | 0.496 | 0.524 |
| Opened pairs that never converge |  | 0.469 | 0.461 | 0.492 |
| Random pairs: share that open |  | 0.883 | 0.888 | 0.889 |
| Random pairs: round trips per pair |  | 0.96 | 0.97 | 0.97 |
| Random pairs: median 2σ trigger |  | 0.260 | 0.326 | 0.242 |
| Random pairs: trades not converging |  | 0.815 | 0.815 | 0.815 |

### Data check: export closes vs the store's raw closes (test 1's statistic)

| Dividends | Stocks | Period | Export: mean, t | Store: mean, t | Same sign | |Δt| | Within 1 |
|---|---|---|---|---|---|---|---|
| fixed | all | 2006-2019 | +0.0025, +0.04 | +0.0021, +0.03 | yes | 0.006 | yes |
| fixed | all | 2020-2025-06 | +0.2392, +1.38 | +0.2392, +1.38 | yes | 0.000 | yes |
| fixed | 500 largest | 2006-2019 | -0.0100, -0.14 | -0.0102, -0.14 | yes | 0.003 | yes |
| fixed | 500 largest | 2020-2025-06 | +0.0835, +0.94 | +0.0835, +0.94 | yes | 0.000 | yes |
| none | all | 2006-2019 | +0.0025, +0.04 | +0.0021, +0.03 | yes | 0.006 | yes |
| none | all | 2020-2025-06 | +0.2664, +1.55 | +0.2664, +1.55 | yes | 0.000 | yes |
| none | 500 largest | 2006-2019 | -0.0100, -0.14 | -0.0102, -0.14 | yes | 0.003 | yes |
| none | 500 largest | 2020-2025-06 | +0.0544, +0.58 | +0.0544, +0.58 | yes | 0.000 | yes |

File-name number (export_div_all, 2006–2025-06, 0 bps): book +1.50% a year − random median +1.20 = +0.3 points

### The 16 unseen months (new export and store; blocks from 2025-06-30)

| Prices | Dividends | Stocks | Months | Committed gross, % a month | Committed net 5 bps | t | Employed net 5 bps | Random median gross, % a year | Book − median, gross, pts | Random books ≥ book | Alpha net, % a year | t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| export | fixed | all | 16 (2025-06-30..2026-10-19) | +0.309 | +0.248 | +3.00 | +0.253 | -0.71 | +4.42 | 19.1% | n/a | n/a |
| export | fixed | 500 largest | 16 (2025-06-30..2026-10-19) | -0.014 | -0.068 | -0.57 | -0.078 | -2.49 | +2.32 | 29.2% | n/a | n/a |
| export | none | all | 16 (2025-06-30..2026-10-19) | +0.259 | +0.201 | +2.11 | +0.201 | -1.21 | +4.32 | 20.8% | n/a | n/a |
| export | none | 500 largest | 16 (2025-06-30..2026-10-19) | -0.158 | -0.210 | -1.87 | -0.224 | -2.53 | +0.64 | 43.4% | n/a | n/a |
| store | fixed | all | 16 (2025-06-30..2026-10-19) | +0.309 | +0.248 | +3.00 | +0.253 | -0.71 | +4.42 | 19.1% | n/a | n/a |
| store | fixed | 500 largest | 16 (2025-06-30..2026-10-19) | -0.014 | -0.068 | -0.57 | -0.078 | -2.49 | +2.32 | 29.2% | n/a | n/a |
| store | none | all | 16 (2025-06-30..2026-10-19) | +0.259 | +0.201 | +2.11 | +0.201 | -1.21 | +4.32 | 20.8% | n/a | n/a |
| store | none | 500 largest | 16 (2025-06-30..2026-10-19) | -0.158 | -0.210 | -1.87 | -0.224 | -2.53 | +0.64 | 43.4% | n/a | n/a |

export_div_all: monthly committed net (%): [0.783, 0.725, -0.609, -0.244, 0.29, 0.095, 0.042, 1.004, 0.997, -0.216, 1.032, -0.281, 0.4, 0.02, -0.341, 0.268]; new export 2006–2019 committed net -0.007% a month, t -0.11
export_div_big: monthly committed net (%): [-0.978, 0.273, -0.407, -0.281, -0.601, 0.353, -0.172, -0.455, 0.355, 0.445, 1.158, -1.476, 1.349, -0.01, -0.888, 0.242]; new export 2006–2019 committed net -0.017% a month, t -0.24
export_nodiv_all: monthly committed net (%): [0.651, 0.742, -0.672, -0.395, 0.436, -0.006, 0.069, 0.682, 0.638, 0.16, 1.102, -0.112, 0.568, -0.251, -0.601, 0.207]; new export 2006–2019 committed net -0.007% a month, t -0.11
export_nodiv_big: monthly committed net (%): [-1.045, 0.441, -0.449, -0.428, -0.987, 0.111, -0.504, -0.513, -0.161, 0.062, 0.729, -1.048, 1.116, -0.031, -0.766, 0.114]; new export 2006–2019 committed net -0.017% a month, t -0.24
store_div_all: monthly committed net (%): [0.783, 0.725, -0.609, -0.244, 0.29, 0.095, 0.042, 1.004, 0.997, -0.216, 1.032, -0.281, 0.4, 0.02, -0.341, 0.268]; new export 2006–2019 committed net -0.007% a month, t -0.11
store_div_big: monthly committed net (%): [-0.978, 0.273, -0.407, -0.281, -0.601, 0.353, -0.172, -0.455, 0.355, 0.445, 1.158, -1.476, 1.349, -0.01, -0.888, 0.242]; new export 2006–2019 committed net -0.017% a month, t -0.24
store_nodiv_all: monthly committed net (%): [0.651, 0.742, -0.672, -0.395, 0.436, -0.006, 0.069, 0.682, 0.638, 0.16, 1.102, -0.112, 0.568, -0.251, -0.601, 0.207]; new export 2006–2019 committed net -0.007% a month, t -0.11
store_nodiv_big: monthly committed net (%): [-1.045, 0.441, -0.449, -0.428, -0.987, 0.111, -0.504, -0.513, -0.161, 0.062, 0.729, -1.048, 1.116, -0.031, -0.766, 0.114]; new export 2006–2019 committed net -0.017% a month, t -0.24

