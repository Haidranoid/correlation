## Test 1: base rate (old export)

News days (vol >= 4x 60-day median, |move| >= 3 sd, first in 5 days) and solo news days, by year; co-movers per solo news day; bystander events per month.

| Year | News days | Solo | Solo with >= 1 co-mover | Co-movers per solo day | Bystanders / month | of which moved with A >= 1.5 sd | Participants / month |
|---|---|---|---|---|---|---|---|
| 2003 | 178 | 153 | 94 | 4.6 | 46 | 4 | 3 |
| 2004 | 308 | 230 | 110 | 3.2 | 49 | 4 | 2 |
| 2005 | 793 | 735 | 138 | 0.9 | 41 | 5 | 2 |
| 2006 | 1,134 | 968 | 464 | 3.4 | 217 | 14 | 12 |
| 2007 | 1,292 | 1,013 | 491 | 3.7 | 233 | 15 | 16 |
| 2008 | 876 | 651 | 398 | 5.4 | 218 | 20 | 15 |
| 2009 | 574 | 486 | 378 | 10.0 | 322 | 18 | 16 |
| 2010 | 901 | 688 | 480 | 7.9 | 332 | 27 | 21 |
| 2011 | 1,094 | 792 | 469 | 5.0 | 252 | 23 | 15 |
| 2012 | 1,144 | 885 | 642 | 7.7 | 430 | 35 | 29 |
| 2013 | 1,306 | 1,026 | 582 | 3.6 | 243 | 18 | 15 |
| 2014 | 1,526 | 1,129 | 610 | 3.4 | 250 | 20 | 15 |
| 2015 | 1,631 | 1,217 | 829 | 5.7 | 449 | 36 | 32 |
| 2016 | 1,718 | 1,137 | 902 | 8.8 | 642 | 40 | 47 |
| 2017 | 1,916 | 1,368 | 1,056 | 7.6 | 668 | 42 | 48 |
| 2018 | 1,943 | 1,355 | 954 | 6.0 | 510 | 45 | 39 |
| 2019 | 1,695 | 1,279 | 1,003 | 8.5 | 693 | 52 | 48 |
| 2020 | 1,789 | 1,037 | 821 | 10.4 | 711 | 48 | 43 |
| 2021 | 2,142 | 1,652 | 1,271 | 10.8 | 1160 | 82 | 80 |
| 2022 | 1,635 | 1,164 | 1,008 | 13.9 | 1023 | 81 | 74 |
| 2023 | 2,208 | 1,493 | 1,307 | 12.1 | 1117 | 74 | 92 |
| 2024 | 2,639 | 1,728 | 1,429 | 8.8 | 964 | 69 | 74 |
| 2025 | 1,190 | 792 | 644 | 8.9 | 897 | 55 | 71 |

- Bystander events (unique stock-days, expiry days dropped): 132,241; after 2010: 118,728.0.
- Bystanders per calendar month since 2010: min 84, median 590, max 1933 (186 months). Kill line: about 20.

**Sympathy move against the ordinary-response prediction (beta x A move).** Slope of B's move on the prediction, through the origin, and the mean ratio of the two in A's direction:

| Group | Rows | Slope of move on prediction | Mean move in A's direction / mean prediction in A's direction | Mean abs move (sd units) |
|---|---|---|---|---|
| All co-movers on solo news days | 171,859 | 0.08 | 0.10 | 0.85 |
| Bystanders (vol <= 1.5x) | 141,392 | 0.06 | 0.08 | 0.71 |
| Participants (vol >= 2x) | 9,832 | 0.23 | 0.27 | 2.20 |
| Bystanders that moved with A >= 1.5 sd | 10,042 | 0.54 | 0.69 | 2.00 |

- Size of the "moved with A >= 1.5 sd" bystander moves: median |move| 2.81%, interquartile 2.08-3.97%.
- All bystanders: median |move| 0.94%; median A move 9.9%; median correlation 0.40; median ordinary slope 0.39.
- Bystander events with a news-day response (A has >= 8 earlier solo news days): 37,447 of 132,241 (28%).
