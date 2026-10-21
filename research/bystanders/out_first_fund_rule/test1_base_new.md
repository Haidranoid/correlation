## Test 1: base rate (new export)

News days (vol >= 4x 60-day median, |move| >= 3 sd, first in 5 days) and solo news days, by year; co-movers per solo news day; bystander events per month.

| Year | News days | Solo | Solo with >= 1 co-mover | Co-movers per solo day | Bystanders / month | of which moved with A >= 1.5 sd | Participants / month |
|---|---|---|---|---|---|---|---|
| 2003 | 165 | 141 | 82 | 4.1 | 38 | 3 | 3 |
| 2004 | 294 | 223 | 104 | 3.0 | 44 | 3 | 2 |
| 2005 | 733 | 680 | 126 | 0.8 | 35 | 4 | 1 |
| 2006 | 1,057 | 907 | 419 | 3.3 | 193 | 12 | 11 |
| 2007 | 1,205 | 943 | 448 | 3.6 | 213 | 14 | 15 |
| 2008 | 823 | 613 | 379 | 5.6 | 206 | 19 | 14 |
| 2009 | 535 | 454 | 352 | 9.9 | 293 | 16 | 15 |
| 2010 | 838 | 641 | 434 | 7.7 | 307 | 25 | 18 |
| 2011 | 1,005 | 737 | 420 | 4.7 | 220 | 20 | 13 |
| 2012 | 1,058 | 828 | 588 | 7.4 | 386 | 31 | 28 |
| 2013 | 1,211 | 957 | 536 | 3.5 | 221 | 17 | 14 |
| 2014 | 1,394 | 1,045 | 561 | 3.3 | 228 | 18 | 13 |
| 2015 | 1,489 | 1,107 | 746 | 5.5 | 391 | 32 | 27 |
| 2016 | 1,592 | 1,046 | 817 | 8.3 | 559 | 37 | 40 |
| 2017 | 1,748 | 1,261 | 958 | 7.4 | 601 | 39 | 41 |
| 2018 | 1,772 | 1,274 | 895 | 5.8 | 467 | 42 | 36 |
| 2019 | 1,548 | 1,180 | 918 | 8.3 | 629 | 49 | 41 |
| 2020 | 1,646 | 947 | 736 | 10.2 | 636 | 44 | 38 |
| 2021 | 1,951 | 1,515 | 1,166 | 10.9 | 1070 | 78 | 75 |
| 2022 | 1,491 | 1,066 | 918 | 13.8 | 932 | 75 | 64 |
| 2023 | 1,982 | 1,338 | 1,175 | 12.0 | 1006 | 69 | 81 |
| 2024 | 2,420 | 1,583 | 1,308 | 8.7 | 874 | 61 | 68 |
| 2025 | 2,412 | 1,595 | 1,337 | 9.9 | 1000 | 76 | 79 |
| 2026 | 1,793 | 1,194 | 986 | 11.3 | 1128 | 87 | 94 |

- Bystander events (unique stock-days, expiry days dropped): 136,756; after 2010: 124,488.0.
- Bystanders per calendar month since 2010: min 53, median 541, max 1737 (202 months). Kill line: about 20.

**Sympathy move against the ordinary-response prediction (beta x A move).** Slope of B's move on the prediction, through the origin, and the mean ratio of the two in A's direction:

| Group | Rows | Slope of move on prediction | Mean move in A's direction / mean prediction in A's direction | Mean abs move (sd units) |
|---|---|---|---|---|
| All co-movers on solo news days | 177,920 | 0.08 | 0.10 | 0.85 |
| Bystanders (vol <= 1.5x) | 146,096 | 0.06 | 0.08 | 0.71 |
| Participants (vol >= 2x) | 10,319 | 0.26 | 0.28 | 2.21 |
| Bystanders that moved with A >= 1.5 sd | 10,667 | 0.53 | 0.68 | 2.00 |

- Size of the "moved with A >= 1.5 sd" bystander moves: median |move| 2.89%, interquartile 2.12-4.13%.
- All bystanders: median |move| 0.96%; median A move 10.2%; median correlation 0.40; median ordinary slope 0.39.
- Bystander events with a news-day response (A has >= 8 earlier solo news days): 39,584 of 136,756 (29%).
