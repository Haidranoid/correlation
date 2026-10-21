# Survivorship stress test: decisions fixed before any outcome was computed (2026-10-21)

Written after reading the plan and stockai's code, before Part 1 or Part 2 produced any number.
Anything decided later is added at the bottom under "Later", marked as such.

## Reproduction (before any haircut)
- Headline +3.21% vs +1.03% (t 2.8) and +2.05 over comparable stocks (t 3.57) come from the raw
  `stockai/runs/30d/lgbm/scores.npy` with stockai's tie-sharing weights (evaluate.py / metrics.py). Reproduced
  with a numpy copy of that code.
- The dollar figures ($230,343 at 20 bps = median of 5 tie-breaks) come from seed 2,
  `stockai/runs/30d/jitter/lgbm_s2.npy`. The grader's account (holdout.rs) is reproduced in numpy: each June-28
  year separately, decision days h0-1..t-2, top 10 positive scores among the year's liquid candidates
  (descending score, stable), rebalanced every 21 days, daily equal weight 1/10, a missing daily return counts 0,
  turnover cost per side on names bought and sold. Must match the JSON's yearly final_pct at 0/5/20/50 bps.

## Part 1
- Universe: stockai's `judge_universe` (90% of 756 days with a census value, upper-median adjusted close >= $5 and
  median dollar volume >= $5M over 252 days) evaluated with h0 = 5911, i.e. on data through 2025-06-27 (the
  export's last column). Features at column 5910 (2025-06-27).
- Same-ticker match: the old ticker is looked up in the newer store. It is the same company only if the newer
  series also passes the constant-ratio test over 2025-04-01..2025-06-27 (below); otherwise the ticker was reused
  or the data broke, and the old company is treated as not found under its ticker (it can still be a rename).
- Constant-ratio test (renames and same-ticker): at least 40 common trading days in 2025-04-01..2025-06-27, and at
  least 95% of the daily ratios new/old within 0.5% of their median. (95% rather than 100% so a single bad print
  in either store does not turn a survivor into a disappearance; the count failing the stricter "every day" test
  is reported.)
- Rename candidates: newer-store symbols whose ticker does not exist in the older store (Stooq moves the whole
  history to the new ticker). Best match = smallest median absolute deviation from the constant ratio; ties and
  multiple matches are reported.
- A found company (same ticker or rename) disappeared if it has no close after 2025-06-27, or its last close in
  the newer store is before 2026-10-01 (early ender).
- Observed outcome for early enders: newer-store last close / newer-store close on 2025-06-27 (last close on or
  before it), both on the same adjustment basis; also the return over its last 21 trading days.
- Window length m: trading days d with 2025-06-27 < d < 2026-10-01 on the newer store's SPY calendar, / 21.
- Regressors at 2025-06-27: 12-month return close[d]/close[d-252]-1 (stockai's past_year_ratio - 1); a flag for
  that ratio < 0.5; stockai's 63-day volatility (sd of daily log returns, >= 42 obs); log close; log median
  dollar volume over days d-62..d. The four continuous ones are standardized with the 2025-06-27 universe's mean
  and sd; those constants are stored and applied unchanged to formation-day features in Part 2.
- A missing regressor (rare) is set to the fit sample's mean (z = 0), flag 0 if the 12-month return is missing;
  counts reported.
- Fit: numpy IRLS logistic regression with intercept. If it fails to converge (e.g. separation), a ridge of 1e-3
  on the slopes. Predicted p clipped to [1e-6, 1-1e-6]. h = 1-(1-p)^(1/m).
- Applying every disappearance the L of a performance delisting (mergers included) is the plan's pessimism; kept.

## Part 2
- Picks: seed 2's picks as the grader made them (top 10 positive scores among the year's candidates) on the 162
  graded decision days of evaluate.py (the same days as the file's tables).
- 162-month statistics are computed exactly as metrics.py does, with weight 1 per seed-2 pick (no ties remain),
  eligible = universe stocks with a finite 21-day forward return and finite score. The seed-2 unadjusted
  comparable excess is the grid's starting point (it is not exactly +2.05, which is the raw-score tie-shared one).
  A seed-2 pick without a finite forward return (if any) is dropped from that month's sum (as evaluate would not
  have it eligible), and counted.
- h_i from the hazard model at the decision day. Adjusted return (1 - k h_i) r_i + k h_i L, with k h_i capped at 1.
- Comparable stocks: the same 20 nearest neighbours as metrics.comparable_excess (neighbours fixed); each
  neighbour's return adjusted with k = 1 and its own h. Average stock: every eligible stock adjusted with k = 1.
- Account: the grader's daily account, with each rebalance block's growth factor increased by
  (1/10) * sum over held picks of k h_i,b (L - r_i,b), where r_i,b is the pick's compounded grader return over the
  block and h_i,b = 1 - (1 - h_i)^(len_b / 21) for a block of len_b trading days (short year-end blocks). At k = 0
  this is exactly the grader's account. Costs as the grader charges them. Years compound as in evaluate.py.
- SPY: the plan's comparison is run against SPY total return (pubdata.etf_adjclose, Yahoo adjusted close) over the
  same 14 June-June spans (close of h0-1 to close of t-1); the grader's Stooq-store SPY ($65,036) is reported too,
  and any cell where the two disagree is flagged.
- Pass in a cell: adjusted comparable t >= 2 and adjusted 20 bps account > SPY. Verdict from the plan's table
  only (no null condition, which is not in this test's rule).
- Break-even k per L: bisection on continuous k in [0, 100] for comparable t = 2; ">100" or "< 0" reported as
  such. The k where the 20 bps account falls to SPY is also reported (extra).
- Crisis sensitivity: the plan says double hazards in 2008-2009, but no test month falls in 2008-2009 (tests run
  2010-06..2024-06), so the literal rule changes nothing; reported as such. Pre-registered substitute: double every
  hazard (picks and benchmarks) in months whose decision day has the Stooq-store SPY close >= 20% below its
  trailing 252-day high; and as a bound, double every hazard in every month. The substitute is the one quoted.
- Extra sensitivity, not part of the verdict: the hazard model without log price, because 2010-2023 closes are
  split-adjusted as of 2025 (a stock that later split looks cheap, one that later reverse-split looks expensive).

## Later (after the first Part 1 run, before any Part 2 number)
- **Universe needs a close on 2025-06-27.** The first run counted 22 candidates with no close on 2025-06-27 as
  disappeared (ALTR, AZPN, DFS, X, ITCI, JWN, ...: acquired or delisted in the months before, still inside the
  756-day eligibility window). They stopped trading before the window starts, and the monthly test's eligible set
  needs a close on the decision day (finite forward return), so they are dropped from the universe. Rates with
  them are reported as a sensitivity.
- **Constant-ratio test tolerates adjustment steps.** The first run's level test failed 16 obvious survivors under
  their own ticker (UNH, HON, FDX, BDX, CAG, DD, ...): the newer store's adjusted closes step at each 2025
  dividend ex-date, so new/old is piecewise constant (UNH 0.993 -> 1.000 in June; OXLC steps ~2% a month). The
  test used for both same-ticker checks and renames is now: >= 40 common days in 2025-04-01..2025-06-27 and at
  most 4 days on which the ratio new/old moves by more than 0.5% from the previous common day (a step: an ex-date
  or a bad print). Unrelated series almost never stay within 0.5% of each other's daily move on 36 of 40+ days.
  Rename best match: fewest steps, then smallest median daily change. Counts under the plan's literal level test
  are reported as a sensitivity.
- An old ticker with fewer than 40 window closes in the older store can't be tested; its same ticker in the newer
  store is then taken as the same company (the plan's literal rule). Count reported.
- **A price match needs a price that moves.** The second run matched ESGR -> IPEX (plus BXBL, SYSB, NTRL) and
  FYBR -> NTRL: both were pinned at a takeover price (2 and 3 days with a daily move above 0.5% in the window), so
  any flat series "matches". The other 34 renames are confirmed independently (the newer ticker's history starts
  on the old ticker's first date, and the step test also passes over 2024-07-01..2025-06-27). Rule added: the
  ratio test is informative only if the old series has at least 20 days in the window with a daily move above
  0.5% (the real renames have 37 or more); a flatter old series gets no rename match, and its same ticker (if
  any) is taken as the same company (the plan's literal rule), counted as untestable.
