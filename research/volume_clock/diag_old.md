# Volume clock diagnostics (added after seeing tests 1–2), old export

## Why the clock misses

| Period | Events | Hit ±3 (the rule) | Hit ±10 | Volume ≥ 2× within ±3 (chance) | Nearest spike before c−3 | c−3..c−1 | c | c+1..c+3 | after c+3 | none within ±10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2006-2019 | 7478 | 30.1% | 44.5% | 68.5% (34.2%) | 4.4% | 5.3% | 16.0% | 8.8% | 10.0% | 55.5% |
| 2020-2025 | 5588 | 26.6% | 43.3% | 65.6% (35.4%) | 5.0% | 4.9% | 14.9% | 6.8% | 11.6% | 56.7% |

## Log against simple returns (per event; SE clustered by entry day)

| Period | Events | Log mean (the rule) | Simple mean | t | Simple − own stock outside | t | Log − own stock outside | t | Simple: hits / misses |
|---|---|---|---|---|---|---|---|---|---|
| 2006-2019 | 7477 | -19.0 (t -1.75) | +19.9 | +1.90 | +2.5 | +0.24 | -17.8 | -1.64 | -49.9 (t -1.82) / +49.9 (t +5.43) |
| 2020-2025 | 5727 | -7.1 (t -0.42) | +54.6 | +3.21 | +33.3 | +1.93 | +12.4 | +0.74 | +86.5 (t +2.06) / +43.1 (t +2.64) |

Returns in bps over the 7 trading days from close c−6 to close c+1. "Own stock outside" is the same stock's mean 7-day return over its in-universe days outside its predicted windows in the same period.

