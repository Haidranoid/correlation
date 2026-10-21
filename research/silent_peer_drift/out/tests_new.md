## Tests (new export)

Outcomes are signed by the direction of A's news move, market-relative log returns (stock minus the export's `market`). "t+1" is the skipped day (close t -> close t+1); 5, 20 and 62 days run from the close of t+1 to the close of t+6, t+21 and t+63. t statistics are clustered by event date (month clusters in brackets).

### The unseen months 2025-06-30 .. 2026-10-19

News days dated in the period: silent 810, middle 755, loud 788, unclassified 525 (counts include events without a 20-day outcome).

**Signed drift by class**

| Class | Events | t+1 (t) | 5 days (t) | 20 days (t) [month t] | 62 days (t) |
|---|---|---|---|---|---|
| silent | 780 | -0.07% (-0.32) | -0.64% (-2.00) | -0.49% (-1.00) [-0.77] | -1.06% (-0.87) |
| middle | 718 | -0.38% (-1.96) | -0.54% (-1.89) | -1.46% (-2.70) [-2.70] | -0.47% (-0.40) |
| loud | 752 | -0.11% (-0.51) | +0.08% (+0.20) | -0.86% (-1.16) [-1.28] | -2.98% (-2.00) |
| unclassified | 508 | -0.07% (-0.25) | +0.18% (+0.27) | -0.86% (-0.92) [-0.77] | -1.81% (-1.25) |
| all news days | 2,758 | -0.16% (-1.40) | -0.27% (-1.15) | -0.91% (-2.82) [-2.22] | -1.58% (-2.17) |

**Silent minus loud**

| Horizon | Silent events | Loud events | Difference | t, date clusters | t, month clusters |
|---|---|---|---|---|---|
| t+1 | 810 | 788 | +0.04% | +0.13 | +0.14 |
| 5 days | 806 | 786 | -0.72% | -1.44 | -1.69 |
| 20 days | 780 | 752 | +0.37% | +0.39 | +0.50 |
| 62 days | 689 | 669 | +1.92% | +1.03 | +1.03 |

**By size half (A's 60-day median dollar volume against the median of the previous year's news days), 20 days**

| Half | Silent (t) | Middle (t) | Loud (t) | Silent minus loud (t) [month t] |
|---|---|---|---|---|
| large | -0.47% (-0.66, n 391) | -1.65% (-2.35, n 403) | -1.86% (-2.41, n 458) | +1.39% (+1.28) [+1.27] |
| small | -0.52% (-0.78, n 389) | -1.22% (-1.65, n 315) | +0.69% (+0.57, n 294) | -1.21% (-0.83) [-0.89] |

**Silent events by news direction, 20 days** (the long side is good news; the short side is bad news)

| Side | Events | Signed drift (t) |
|---|---|---|
| good news (long) | 299 | -1.13% (-1.55) |
| bad news (short) | 481 | -0.10% (-0.16) |

**The no-news contrast:** same-size moves (>= 3 sd60) on volume <= 1.5x the 60-day median, no news day in the previous 5 days, first in 5 days

| Group | Events | t+1 (t) | 5 days (t) | 20 days (t) [month t] | 62 days (t) |
|---|---|---|---|---|---|
| quiet big moves | 1,746 | -0.13% (-1.00) | -0.15% (-0.75) | -0.35% (-0.59) [-0.57] | -0.62% (-0.90) |
| all news days (above) | 2,758 | -0.16% (-1.40) | -0.27% (-1.15) | -0.91% (-2.82) | -1.58% (-2.17) |

### Sensitivity: minimum co-movers for E (the rule is 3), silent minus loud at 20 days

| Minimum co-movers | The unseen months 2025-06-30 .. 2026-10-19 |
|---|---|
| 1 | +0.37% (+0.42; n 888 / 866) |
| 3 | +0.37% (+0.39; n 780 / 752) |
| 5 | +0.26% (+0.26; n 722 / 688) |
| 10 | +1.02% (+1.01; n 583 / 546) |

### Year by year, 20 days

| Year | Silent | Middle | Loud | Silent minus loud (t) |
|---|---|---|---|---|
| 2006 | +0.75% (n 64) | -0.43% | +0.67% (n 77) | +0.08% (+0.08) |
| 2007 | -1.46% (n 131) | +1.22% | -1.06% (n 132) | -0.41% (-0.34) |
| 2008 | +0.45% (n 132) | -0.78% | -3.71% (n 170) | +4.16% (+1.65) |
| 2009 | -0.17% (n 140) | -1.31% | -0.41% (n 92) | +0.25% (+0.11) |
| 2010 | +0.09% (n 141) | +0.67% | +0.77% (n 180) | -0.67% (-0.68) |
| 2011 | -0.08% (n 135) | +0.62% | +0.20% (n 170) | -0.28% (-0.28) |
| 2012 | -0.70% (n 213) | -0.73% | +0.49% (n 152) | -1.19% (-1.54) |
| 2013 | -0.01% (n 130) | -0.16% | +0.48% (n 146) | -0.50% (-0.68) |
| 2014 | +0.24% (n 149) | -0.77% | -0.22% (n 187) | +0.45% (+0.47) |
| 2015 | -0.11% (n 284) | -0.55% | -0.10% (n 235) | -0.01% (-0.01) |
| 2016 | -0.66% (n 366) | +0.17% | +0.47% (n 354) | -1.13% (-1.23) |
| 2017 | +0.54% (n 366) | +0.08% | +0.08% (n 328) | +0.45% (+0.81) |
| 2018 | -0.24% (n 332) | +0.15% | -0.37% (n 343) | +0.13% (+0.19) |
| 2019 | -0.41% (n 352) | +0.40% | +0.33% (n 304) | -0.74% (-0.73) |
| 2020 | -0.39% (n 360) | -0.10% | -0.77% (n 501) | +0.38% (+0.17) |
| 2021 | -1.82% (n 453) | -1.28% | +0.55% (n 381) | -2.37% (-1.89) |
| 2022 | +0.35% (n 446) | +0.33% | -2.76% (n 442) | +3.11% (+2.42) |
| 2023 | -0.70% (n 537) | +0.40% | +0.94% (n 528) | -1.64% (-1.81) |
| 2024 | -0.17% (n 535) | -0.01% | +0.41% (n 564) | -0.58% (-0.65) |
| 2025 | -0.66% (n 603) | -0.60% | -1.28% (n 562) | +0.63% (+0.60) |
| 2026 | -0.91% (n 408) | -2.52% | -1.09% (n 446) | +0.18% (+0.14) |
