# Hypothesis B of `experiments/-3.1-luca-market-geometry.md`: results (B1, B2, B3, B5)

Part B of the plan, run 2026-10-23 by one of two agents working in this folder (the other runs hypothesis A). This file holds the novelty check, every decision the plan left open, and the results. The plan file itself is not edited here; the merge writes the experiment file.

## Novelty check (2026-10-23, web search before any test; it changed no rules)

Searched: Ricci curvature of stock correlation networks as a forecaster of returns or volatility, out of sample; curvature as an early-warning signal; log-Euclidean or matrix-log covariance forecasts judged by minimum-variance portfolios.

- **Sandhu, Georgiou & Tannenbaum** (arXiv 1505.05182; *Science Advances* 2016): 388 S&P 500 stocks, 1998-01..2013-07, windows of 22 and 132 days sliding one day. The network is the minimum spanning tree on d = √(2(1 − c)) plus every link with c ≥ ξ = 0.85. Ollivier-Ricci with the one-step random-walk measure (no mass kept on the node) and hop distance. "During periods of crisis there is an increase in Ollivier-Ricci curvature"; they call that state "more robust" and normal markets "fragile". Contemporaneous only.
- **Samal et al. 2021** (*R. Soc. Open Sci.*, arXiv 2009.12335): S&P 500 and Nikkei 225, 1985–2016, τ = 22-day windows shifted by 5 days. Network: the MST on √(2(1 − C)) plus every edge with C ≥ 0.75 (0.55, 0.65 and 0.85 in the supplement). Forman-Ricci is their Eq. 4, the weighted form of Sreejith et al. 2016 without a triangle term (unweighted: 4 − deg u − deg v); the edge and node weights used for the market networks are not stated in the text. Ollivier uses a measure uniform over neighbours. Average Forman curvature "shows a local minimum (more negative) during the bubble" and a "significant decrease" in the 2011 crash; they cite Ollivier as rising in crises. No lead-time, out-of-sample or forecasting test; forecasting is named only as a hope.
- **Sánchez García & Gherghe 2024** (arXiv 2405.07134, "On the Ollivier-Ricci curvature as fragility indicator of the stock markets"): analytical and empirical properties and parameter sensitivity of the Sandhu indicator; the abstract reports no forecasting test.
- **Wang, Zhao, Zhang, Feng & Lin 2023** (*Complexity*, arXiv 2204.06692): CSI 300 constituents, 2005–2020, Ricci curvature with a hybrid forecasting model, presented as an indicator of future risk and volatility. Seen in abstract only; it is the nearest to B2's risk target, but not a nested out-of-sample test against volatility yardsticks and plain graph statistics.
- **An IEEE Access paper (2026)** on a neural Ricci-flow framework for flash crashes: abstract only, no comparable design visible.
- **Bucci, Palma & Zhang** (arXiv 2412.09517, "Geometric Deep Learning for Realized Covariance Matrix Forecasting"): a HAR extended to positive-definite matrices through a Fréchet mean, daily realized covariances of the 50 largest S&P 500 stocks. The abstract claims better accuracy than traditional approaches; whether minimum-variance portfolios are evaluated isn't visible from the abstract. The arXiv page shows no journal; the plan cites *J. Financial Econometrics*, which this search couldn't confirm.
- **arXiv 2303.16151** ("Forecasting Large Realized Covariance Matrices: The Benefits of Factor Models and Shrinkage"): a VHAR on log-transformed matrices, judged by minimum-variance portfolios, comparable to the best DCC-type benchmark. The closest published cousin to B5, though it doesn't isolate the representation with the same grid, windows and shrinkage.
- **Not found:** a test of whether network curvature adds to forecasts beyond volatility, average correlation, the absorption ratio and plain graph statistics (B2); a curvature timing rule against random timings at the same exposure (B3); or a paired log-Euclidean against Euclidean comparison that changes only the representation (B5).

**Effect on the rules: none.** The searches fixed only values the plan says to read from the papers before running (the published graph's threshold, which sign is the crisis sign, which Forman version Samal et al. used). Those are listed below.

## Decisions (every point the plan left open, and every deviation)

### Before results were seen (fixed 2026-10-23 before the pilot or any curvature, forecast or portfolio number)

**Data**
1. **SPY is Yahoo's adjusted close (`pubdata.etf_adjclose(['SPY'])`, aligned with `pubdata.on_days`), everywhere in B.** The plan's data row names SPY from the store, but the store's SPY starts 2005-02-25, so the plan's own dates (B2's 60 monthly observations before 2008-01, which need SPY from 2003-01) can't hold on it. The store is also back-adjusted for dividends only to about late 2022. Yahoo's series starts 1993, is the benchmark `research/after_costs/group_b/market_timing.py` already uses, and covers the unseen months (last bar 2026-10-20). Its daily returns are checked against the store's SPY over 2005-03..2025-06 and the agreement reported.
2. **T-bill:** `pubdata.rf_daily` (FRED DGS3MO ÷ 100 ÷ 252). Cash earns it in every rule and baseline; excess returns subtract the sum of log(1 + rf) over the same days.
3. **Fund prices** (B2's fund targets, B3's fund risk scaling) come from `c2eval.twins(E)['F']`, the store's ETFs, with daily log returns beyond ±50% treated as missing, as c2eval does. They are dividend-adjusted only to about late 2022; fund results are reported, never pass/fail beyond the plan's secondary tests.

**The curvature series (`curvature.py`)**

4. **Calendar and universe:** `tw['rebal']` (export day 252 + 21k), the plan's month-ends, 270 on the old export (2002-12-31..2025-06-13) and 286 on the new. The universe is reimplemented line for line from `states()` in `research/market_timing/common.py` (which doesn't return the chosen stocks): drop `excluded(E, tw)[k]`; keep stocks with a daily log return on ≥ 90% of the 63 days, median dollar volume > 0 and sd > 0; take the 500 with the highest median dollar volume. **Check:** the average correlation recomputed here must equal `states_old.npz`'s to 1e-9 in every month, which validates the universe.
5. **Correlation:** as `states()`: each stock's 63 log returns demeaned over its own finite days, missing days 0, divided by its sd and normalized, so C = Z Zᵀ.
6. **Primary graph:** the m = ⌊0.05 × n(n − 1)/2⌋ most-correlated pairs; 6,237 when n = 500. Before 2005-06 the export has only 314–320 eligible stocks, and the same 5% rule applies to them (about 2,500 edges). Ties are broken by a stable sort on the upper triangle.
7. **Edge length:** L = √(2(1 − ρ)), clipped at 0.
8. **Augmented Forman (primary):** F(x, y) = 4 − deg x − deg y + 3 t(x, y), t = triangles on the edge, averaged over edges. **Check:** the edge average must equal 4 − Σd²/m + 9T/m to rounding.
9. **Samal et al.'s Forman (the replication):** their Eq. 4 with node weights 1: F(e) = 2 − Σ over the other edges e′ at x of √(w_e / w_e′) − the same at y. Their edge weights aren't stated; the edge weight here is the distance L, the quantity their MST and Ollivier paths use. This is a guess at their choice and is labelled so. It is computed on both graphs and is reported only.
10. **Ollivier-Ricci:** κ(x, y) = 1 − W₁(m_x, m_y) ÷ d(x, y), m_x = ½ on x and ½ ÷ deg x on each neighbour. Ground distance: all-pairs weighted shortest paths with edge lengths L (scipy Dijkstra). L is a Euclidean metric on standardized returns, so an edge's shortest path is the edge itself and d(x, y) = L. W₁ is solved exactly by POT's network simplex (`ot.emd2`) on the cost submatrix of the two supports.
11. **The crisis ("fragile") sign, from the papers:** Ollivier **rises** in crises (Sandhu et al.; Samal et al. cite the same), and Forman becomes **more negative** in crashes and bubbles (Samal et al.). So the calm side is **below** the trailing median for Ollivier and **above** it for Forman. The two point opposite ways. Sandhu et al. call high curvature "robust" and normal markets "fragile". The plan's "fragile" means crash-associated, so its calm side is the side normal months sit on.
12. **Published-style graph (reported):** the MST on L plus every pair with ρ ≥ 0.75 (Samal et al.'s main value; Sandhu et al. used 0.85), on the same 63-day window. The MST is included because both papers include it. Augmented Forman, Samal's Forman, Ollivier (with the plan's ½ mass and L, so only the graph differs) and the plain statistics are computed on it.
13. **Plain statistics:** Σd² ÷ m, T ÷ m, the edges' mean ρ, and the largest connected component's share of the n stocks.
14. **Group curvature:** a group is the stocks among the month's n with the same twin fund (`tw['fund_row'][k]`). Its value is the mean of the per-edge curvature over edges with both ends in the group, and it needs ≥ 10 such edges. Also stored: the internal edge count, the mean ρ over all member pairs, and the size. Each group's series is standardized by fund, (x − mean) ÷ sd of its own values at the previous 60 month-ends where it had one, needing at least 24.
15. **Shift null:** each stock's 63 window returns are circularly shifted by its own random integer in 1..62 (seed 20261023 + 1000k), then the whole pipeline is recomputed: C, the 5% graph, both curvatures and the statistics. One draw a month.
16. **Factor null:** within the window each stock's demeaned, missing-as-0 returns are regressed by OLS on the export's equal-weight `market` and its twin fund's daily log return (market only for stocks without a twin), both demeaned the same way. The returns are rebuilt as fitted + the residual circularly shifted by a random 1..62 (seed 20261023 + 1000k + 1), then the pipeline is recomputed. The export's `market` exists from 2001-12-31, so this doesn't depend on SPY. One draw a month.
17. **Rewiring:** double-edge swaps on the real primary graph. Two edges (a, b), (c, d) are picked uniformly and, with probability ½ each, become (a, d), (c, b) or (a, c), (b, d); a swap that makes a self-loop or a duplicate is rejected. **Lengths travel with the swap:** the new edges take the two old lengths, which keeps the multiset of lengths. A round is m attempted swaps, and T is counted after each round. The chain runs at least 10 rounds and stops once |mean of T over the last 5 rounds − mean over the 5 before| ≤ the sd (ddof 1) of the 5 before, with a cap of 200 rounds. Seed 20261023 + 1000k + 2. Then Forman and Ollivier are recomputed on the rewired graph.
18. **Odd/even:** window returns at positions 0, 2, 4, … (32 days) and 1, 3, 5, … (31 days), each standardized as in 5, on the same stocks, the 5% rule, both curvatures. Their correlation across months is reported, along with the correlation of month-to-month changes.
19. **252-day series:** the same selection rule and pipeline on a 252-day window: ≥ 90% coverage of the 252 days, the 500 most traded by median dollar volume over the 252 days, minus excluded.
20. **Parallelism:** the per-edge transport problems run in 3 worker processes with 1 BLAS thread each, so the job stays inside the 3-core budget. The workers return exactly what a serial loop would.

**B1 (`tests.py`)**

21. Each curvature series (Ollivier, augmented Forman, Samal's Forman, and the published graph's Ollivier and Forman) is regressed by OLS on (a) average correlation, the absorption ratio and SPY's 63-day log realized volatility, and (b) the four plain statistics, over all months 2002-12..2025-06 and separately over 2008–2019. R² and the residual's sd ÷ the series' sd are reported.

**B2 (`tests.py`)**

22. **Realized volatility:** RV = √(Σr² ÷ n) of daily log returns, not demeaned. The predictors are log RV of SPY over the 21 and 63 days ending at t₀. The targets cover closes t₀ + 1 .. t₀ + 22, i.e. the 21 returns of days t₀ + 2 .. t₀ + 22:
    - direction: SPY log return − Σ log(1 + rf);
    - risk: log RV;
    - reported: maximum drawdown from the close of t₀ + 1.
23. **Expanding window:** the forecast for month k is fitted by OLS (with intercept) on every month j whose target ended at or before t₀ₖ (t₀ⱼ + 22 ≤ t₀ₖ, so j ≤ k − 2) and has every input. A month is scored only with at least 60 such rows; the primary score covers decision dates 2008-01-01..2019-12-31. 2020-01..2025-06 and the unseen months are reported with the same expanding fits.
24. **Models:** (1) baseline: log RV21, log RV63, average correlation, absorption ratio; (2) + Σd²/m, T/m, mean edge ρ, LCC share; (3) model 2 + mean Ollivier; (4) baseline + mean augmented Forman. Reported only: 2 against 1; 3 against 1; and baseline + Samal's Forman, + the published graph's Ollivier, + the published graph's augmented Forman, each against 1.
25. **Clark-West:** a_t = (y − f₁)² − [(y − f₂)² − (f₁ − f₂)²], with f₁ the smaller model. The t statistic is mean ÷ Newey-West se with 2 lags, the p is one-sided normal, and the pass is t ≥ 3. The bootstrap p is a moving-block bootstrap of months with 6-month blocks and 10,000 draws on a − mean(a): the share of draws whose mean is ≥ the observed mean. Also reported: the out-of-sample R² of the larger model against the smaller one, and the MSE improvement with a 90% interval (Newey-West ±1.645 se, and the bootstrap's 5th–95th percentiles). Seed 20261023.
26. **Fund targets (reported, secondary):** a pooled panel over (month, fund) with a group value. Inputs: the fund's own log RV21 and RV63 and its group's mean pair ρ (baseline); + log of the internal edge count (model 2; log because counts run from 10 to thousands); + the group's standardized Ollivier (model 3); baseline + the group's standardized augmented Forman (model 4). The targets are the SPY ones computed on the fund's prices. A fund-month needs every daily return in the target and input windows. Clark-West is averaged across funds within each month, then Newey-West over months.
27. **Drawdowns:** every SPY peak-to-trough decline of ≥ 15% on daily closes from 2003-01 to 2025-06 (a new peak resets). For each, reported: the Ollivier and Forman z-scores against their trailing 60 month-ends at 6 and 3 months before the peak, at the peak and at the trough, and the first month-end from 12 months before the peak on where the z-score is beyond +1 on the crisis side, in months relative to the peak.

**B3 (`money.py`)**

28. **The rule's curvature** is the primary graph's mean Ollivier (the plan's "curvature" is singular, and Forman is read as graph statistics). It holds 100% SPY when Ollivier is strictly below its trailing median (`trailing_median`: previous 60 month-ends, at least 24), otherwise 50%. The Forman rule (100% when strictly above its median) is reported. If Ollivier couldn't be run, Forman would be the rule; that didn't happen if B3 below names Ollivier.
29. **Timing:** decided at the close of t₀, bought at the close of t₀ + 1, held as buy-and-hold to the close of t₀ + 22. Cash earns the T-bill, and SPY is Yahoo's. The months are the calendar points with t₀ from 2005-04-01 whose t₀ + 22 fits in the old export: 2005-04-04..2025-05-14, 242 months, the same months as `-3.2`. Trading at t₀ (`-3.2`'s timing) and cash at 0 are reported as well.
30. **Costs:** 5 bps (and 20) on every traded SPY dollar: the first buy and each month's trade from the drifted weight back to the target, charged at the close of t₀ + 1. Turnover, volatility and maximum drawdown are reported.
31. **Baselines:**
    - constant: the rule's mean weight, reset monthly;
    - volatility threshold: 100% when SPY's 21-day RV is strictly below its trailing median, else 50%;
    - volatility-managed: min(1, c ÷ var₂₁), with c by bisection so the mean weight equals the rule's on the window judged, as `-3.2`;
    - `-3.2`'s average-correlation rule (100% when strictly above its median) and absorption rule (100% when strictly below), on `research/market_timing`'s saved states (read only), rerun with this timing and cash;
    - 10,000 random orderings of the rule's own weights (seed 20261021 as `-3.2`).
32. **Pass (secondary):** over 2005-04..2019-12, the development period, the paired net-of-5-bps monthly return difference is positive against the constant allocation, the volatility threshold and the better of the two `-3.2` rules. "Better" means the higher net compound return over the same months. Each difference gets a Newey-West (2 lags) t and a one-sided p, with the 6-month block bootstrap p beside it. All three must pass, after Holm across B's secondary tests at 0.05 one-sided. A2, the other agent's secondary, joins the family at the merge and can only make it stricter. 2020-01..2025-06 and the whole span are reported.
33. **The file-name number** (as `-3.2`): the rule's compound annual return before costs minus the median of the 10,000 random timings, 2005-04..2025-06. **After costs:** the rule's return at 5 bps a side with T-bill cash, compounded by calendar year (2005 from the close of 2005-04-05, the first trade, to 2025's last held day), minus SPY's adjusted total return over the same days, averaged over the years, as `research/after_costs/group_b/market_timing.py` does. The cash-at-0 version is reported.
34. **Fund risk scaling:**
    - the funds with a group value in month k, weighted ∝ 1 ÷ forecast volatility;
    - three forecasts: exp of B2's fund risk model 3 (with group Ollivier), exp of model 2 (without it), and trailing 63-day RV;
    - bought at the close of t₀ + 1, buy-and-hold to t₀ + 22, 5 bps on turnover;
    - months are B2's fund scoring months (2008–2019 primary, 2020–2025-06 reported).

    **Pass:** the monthly realized variance (Σ of squared daily portfolio returns) of the curvature version is below the no-curvature version's: paired difference, Newey-West t, one-sided p, under Holm across B's secondaries. Sharpe ratios (excess of the T-bill) are reported.

**B5 (`covariance.py`)**

35. **Calendar and list:** the rebalance points with t₀ ≥ 756, as `research/correlation_portfolios/build.py`. The list is `universe()` from `research/correlation_portfolios/common.py`'s 500 largest, keeping stocks with ≥ 227 of the 252 returns over t₀ − 251..t₀, exactly MV-LW's list. It is imported read-only and checked to give MV-LW's weights.
36. **Windows:** returns t₀ − 20..t₀, t₀ − 62..t₀, t₀ − 251..t₀. Each is demeaned over its finite values with missing values 0 (`demean_fill`) and shrunk by `estimators.ledoit_wolf_cc`. LW-252 is S₂₅₂ itself, MV-LW's estimator.
37. **Combinations:** the 66 simplex points in steps of 0.1. Euclidean: Σ wᵢSᵢ. Log-Euclidean: exp(Σ wᵢ log Sᵢ), with log and exp by symmetric eigendecomposition. The three vertices are the same matrix in both representations.
38. **Loss:** the realized variance of the capped minimum-variance portfolio (`estimators.minvar_capped`, cap 2%, the MV-LW solver) held at fixed weights from the close of t₀ + 1. It is Σ over days t₀ + 2..t₀ + 22 of (Σᵢ wᵢ Rᵢ,d)², with R the simple daily returns of forward-filled export closes (a missing day is a 0 return).
39. **Warm start:** the grid solves pass `w0` from a neighbouring grid point. The solver's active-set finish verifies the KKT conditions, so this changes speed, not the answer.
40. **Weights:** at month k, the grid point with the lowest mean loss over months j with t₀ⱼ + 22 ≤ t₀ₖ (j ≤ k − 2), ties to the first in grid order. Until 12 such months exist, the weights are (⅓, ⅓, ⅓), solved separately. Scoring starts at the first month with 12 months of history.
41. **Test:** d_t = loss(Euclidean) − loss(log-Euclidean), so positive favours log-Euclidean. Mean, Newey-West (2 lags) t, one-sided p, 6-month block bootstrap p (10,000 draws, seed 20261023) and the 90% interval, over decision dates 2006-01-01..2019-12-31. **Pass:** t ≥ 3. Both are also compared with LW-252 the same way. 2020-01..2025-06 and the unseen months are reported.
42. **QLIKE (reported):** tr(Σ̂⁻¹Σ) − log det(Σ̂⁻¹Σ) − N, with Σ the LW-shrunk covariance of the 21 log returns of days t₀ + 2..t₀ + 22 on the same list.
43. **Affine-invariant step (reported):** B A⁻¹ B, with A the shrunk 63-day covariance ending at t₀ − 21 and B the one ending at t₀, on the current list. It is computed as B^½ (B^½ A⁻¹ B^½) B^½ for symmetry. Its smallest eigenvalue and condition number are reported, along with its loss and QLIKE.

**Unseen months (`unseen.py`)**

44. Run once at the end on `c2eval.EXPORT_NEW`, with the frozen code, after checking that the new export's first 5,911 days equal the old export's. The months are the calendar points with t₀ ≥ 2025-06-27 whose t₀ + 22 fits by 2026-10-19. Expanding fits and B5's weight history continue from all earlier months on the new export.

**Added after the pilot and the construction checks, still before any curvature, forecast or portfolio result**

45. **The pilot decided the full design runs.** Every Ollivier series was run exactly as fixed above, with no shortcut (timings below).
46. **The factor null has no value in the first 3 months** (2002-12-31, 2003-01-31, 2003-03-04). The export's `market` is NaN for its first 252 days, so those windows lack a full market series.
47. **The fund panel needs 60 distinct months in its training set,** which is the plan's "60 monthly observations" applied to the panel. Group z-scores need 24 earlier values first, so fund scoring starts later than 2008-01; the first scored month is reported. The fund risk scaling uses the same months.
48. **The fund forecasts behind the risk scaling** are made for every fund-month with a group value and finite inputs at t₀, whether or not its target window later turns out complete. Requiring a complete future window would look ahead.
49. **B's secondary family for Holm** has 6 members:
    - model 3 against 2 for SPY's maximum drawdown;
    - model 3 against 2 for the funds' excess return, log RV and maximum drawdown;
    - B3's market rule, whose p is the largest of its three one-sided p-values, since all three must pass;
    - B3's fund risk scaling.

    Comparisons 2 against 1, 3 against 1 and 4 against 1, and the published-graph and Samal models, are reported with raw p only. A2 joins the family at the merge.

### After results were seen

50. **Bug fixed, B3's Holm entry:** `money.py` first put p = 1 into the Holm family for the market rule whenever one of its three differences was negative. Decision 49 says the entry is the largest of the three one-sided p-values (0.956). It was fixed and rerun; no conclusion changed.
51. **Bug fixed, cash at 0:** the cash-at-0 variant's random timings still earned the T-bill. It was fixed and rerun: the rule minus random moved from −0.6 to −0.2 points. This number is reported only.
52. **An invalid extra statistic, kept and flagged:** besides the plan's odd/even level correlation, `tests.py` reports the correlation of the two halves' month-to-month changes (−0.33 for Ollivier). That number is not a stability measure. The calendar steps 21 days, an odd number, so a window's even positions are the previous window's odd days, and the changes compare overlapping days crosswise. Only the level correlation is read.
53. **Diagnostics added after results, reported only** (`b_extra.py`):
    - the stability checks and B1 on the 500-stock months only;
    - SPY drawdowns with the peak reset after each trough, which separates the 2010 and 2011 corrections that the pre-registered definition (decision 27) folds into the 2007–2012 underwater period;
    - the base rate of curvature spikes, needed to read the drawdown table.
54. **QLIKE with an undefined target:** in 3 months (2025-02-12, 2025-04-14, 2025-05-14) a stock on the list has no closes in the forward window, so the forward Ledoit-Wolf matrix, QLIKE's target, is NaN. `covariance.py` now leaves those months out of QLIKE only, and says how many months it used. The primary loss, which uses forward-filled closes, is unaffected.

## Results (run 2026-10-23)

**In short:** none of hypothesis B's tests passes. Curvature of the stock graph doesn't improve forecasts of SPY's next-month return (B2 direction, t +0.21) or its next-month volatility (B2 risk, t −0.87) beyond volatility, average correlation, the absorption ratio and plain graph statistics. A curvature timing rule matched random timing (−0.3 points a year, p 0.61) and trailed SPY by 3.05 points a year after costs. The curved-geometry idea for covariances fared no better: log-Euclidean and Euclidean combinations built portfolios with the same risk (B5, t −0.15). Curvature itself is mostly plain graph statistics renamed: four counting statistics explain 92% of the Ollivier series and all of Forman's.

### The pilot (the plan's gate)

- **What was timed:** every month of 2008, the most crowded year for both graphs, plus one calm month (2017-06), in one process on the shared machine (load about 43–78 on 24 cores):
  - Ollivier-Ricci on the 5% graph (6,237 exact transport problems, the largest node degree 113–170): 1.4–2.0 s a month, mean 1.7 s;
  - Ollivier on the published-style graph: 0.1 s in calm months, up to 13.0 s in 2008-12 with 16,129 edges;
  - Forman and the plain statistics: under 0.1 s;
  - degree-preserving rewiring: 0.1 s (10–14 rounds).
- **Extrapolated:** all 7 Ollivier passes on 5% graphs over 286 months (old and new export) is 0.9 h in one process, 0.3 h on 3. The published graph adds 0.13 h.
- **B5:** 15–17 s a month: 3 Ledoit-Wolf fits, 66 matrix exponentials, and 132 capped minimum-variance solves of about 0.12 s each. Warm starts barely helped. That makes 1.1–1.2 h in one process for the 262 months, 0.4 h on 3.
- **Decision:** everything runs as written. No pass was dropped or approximated, and the GPU wasn't needed.
- **Actual:** `curvature.py old` took 23 min on 3 processes (load 60–78); `covariance.py old` took 35 min on 3 processes.

### The tests, in order

| # | Test | Kind | Measure | Value | Bar | Result |
|---|---|---|---|---|---|---|
| 1 | B2 direction, 2008-02..2019-12 | primary | Clark-West t, model 3 (+ Ollivier) vs 2, SPY's next 21-day excess return | **t +0.21**, one-sided p 0.417 (bootstrap 0.372) | t ≥ 3 | **Fail** |
| 2 | B2 risk, 2008-02..2019-12 | primary | the same, SPY's next 21-day log realized volatility | **t −0.87**, p 0.807 (bootstrap 0.822) | t ≥ 3 | **Fail** |
| 3 | B5, 2006-02..2019-12 | primary | paired Diebold-Mariano t, Euclidean − log-Euclidean minimum-variance realized variance | **t −0.15**, p 0.560 (bootstrap 0.598); 12.37% against 12.38% annualized | t ≥ 3 | **Fail** |
| 4 | B3 market rule, 2005-04..2019-12 | secondary | paired monthly net difference against the constant 73% allocation, the volatility threshold and the absorption rule (the better `-3.2` rule here) | −0.083%, −0.079% and −0.100% a month (t −1.04, −1.22, −1.70); largest p 0.956 | all three > 0 under Holm | **Fail** |
| 5 | B3 fund risk scaling, 2010-08..2019-12 | secondary | paired monthly realized variance, without curvature − with | **−7.3 × 10⁻⁶** (the curvature version was riskier), t −3.84, p 1.000 | > 0 under Holm | **Fail** |
| 6 | B2's other targets, 2008–2019 | secondary | model 3 vs 2: SPY's maximum drawdown; the funds' excess return, log volatility, maximum drawdown | t +0.04; −1.29, −0.25, −0.60 | Holm | **Fail** |

- **The verdict stops at test 1.** Each test still has its own conclusion, and every number was computed.
- **Raw one-sided p-values for the merge's Holm correction** across A1, B2 direction, B2 risk and B5: B2 direction **0.4173**, B2 risk **0.8071**, B5 **0.5597**.
- **B's secondary family, Holm-adjusted (6 tests; A2 joins at the merge):**
  - SPY maximum drawdown: p 0.486;
  - the funds' excess return, log volatility and maximum drawdown: p 0.901, 0.599 and 0.725;
  - B3's market rule: p 0.956;
  - B3's fund risk scaling: p 1.000.

  Every Holm-adjusted p is 1.000, so all six fail.

### The number for the file name

- **Improvement over random: −0.3 points a year** before costs, over 242 months 2005-04..2025-06:
  - B3's rule's compound annual return (+8.0%) minus the median of 10,000 random timings (+8.3%);
  - each random timing reorders the rule's own 100% and 50% months, so its 74% average exposure is the same;
  - random p = 0.61.
  - Breakdown: 2005-04..2019-12 −0.8 (p 0.77); 2020-01..2025-06 +1.1 (p 0.30).
- **After costs: −3.05 points a year.** The rule's yearly return at 5 bps a side with T-bill cash, minus SPY's adjusted total return over the same days, averaged over 21 calendar years. The span runs from 2005-04-06, the first day after the first trade at the close of 2005-04-05, to 2025-06-16. With cash at 0 it is −3.40.
- **So the file would be `-3.1-luca-market-geometry.md`** (−3.0549 rounded to one decimal). The rename is the merge's job.

### The curvature series

**Construction checks:**
- **Universe:** the 500 stocks rebuilt here reproduce `research/market_timing`'s states exactly: the average correlation and the absorption ratio differ by 0.0 in all 270 months, and the universe sizes are equal.
- **Forman closed form:** the edge mean of the augmented Forman curvature equals 4 − Σd²/m + 9T/m to 5 × 10⁻¹⁴, in all 8 series.
- **Rewiring:** every rewired graph kept every stock's degree. The chain stopped after 10–14 rounds (median 10).
- **Graph sizes:**
  - 270 months, 2002-12-31..2025-06-13;
  - the primary graph has 2,457–6,237 edges (n = 314–320 before 2005-06, 500 after);
  - the published-style graph has 324–44,925 edges, swelling in crises as a threshold graph does.
- **Values, 500-stock months:**
  - mean Ollivier 0.152 (sd 0.037, range 0.067..0.275);
  - mean augmented Forman −27.5 (sd 10.5);
  - Samal et al.'s Forman −120 (sd 19).

**B1: what curvature is made of** (OLS over all 270 months; 2008–2019 in brackets):

| Series | R² on avg correlation, absorption ratio, SPY 63-day vol | Residual sd ÷ series sd | R² on the 4 plain graph statistics | Residual sd ÷ series sd | Correlation with avg correlation |
|---|---|---|---|---|---|
| Ollivier, 5% graph | 0.25 (0.32) | 0.86 | **0.92** (0.94) | 0.28 | +0.42 |
| Augmented Forman, 5% graph | 0.39 (0.37) | 0.78 | **1.000** (exact) | 0.00 | −0.25 |
| Samal et al.'s Forman, 5% graph | 0.30 (0.49) | 0.84 | 0.99999 | 0.004 | −0.54 |
| Ollivier, published graph | 0.46 (0.44) | 0.74 | 0.61 (0.60) | 0.62 | +0.56 |
| Augmented Forman, published graph | 0.23 (0.17) | 0.88 | 0.14 (0.05) | 0.93 | −0.36 |
| Samal et al.'s Forman, published graph | 0.63 (0.67) | 0.61 | 0.56 (0.58) | 0.66 | −0.77 |

- **On the fixed-edge graph, curvature is the plain statistics renamed.**
  - Forman is an exact function of them, as the plan derived.
  - Samal et al.'s distance-weighted Forman is almost one too (R² 0.99999).
  - Four counting numbers reproduce 92% of Ollivier's variance. On the 500-stock months alone it is 93%.
- **The market yardsticks explain much less,** a quarter to two fifths. B2 decides whether the remainder is useful; it isn't.

**The four checks** (each answers its own question; none shows forecasting value):

| Check | Ollivier: real | Ollivier: null | Real − null (t) | Real above null | Forman: real | Forman: null | Real − null (t) | Real above null |
|---|---|---|---|---|---|---|---|---|
| Shift null (each stock shifted by its own 1–62 days) | +0.149 | −0.081 | +0.231 (t +69) | 100% of 270 months | −26.8 | −32.3 | +5.5 (t +5.0) | 67% |
| Factor null (market + twin-fund fit, residuals shifted) | +0.149 | **+0.161** | **−0.013 (t −11.9)** | 12% of 267 | −27.0 | −32.1 | +5.1 (t +10.8) | 88% |
| Degree-preserving rewiring | +0.149 | +0.110 | +0.039 (t +19.2) | 99.6% | −26.8 | −65.4 | +38.6 (t +35.7) | 100% |

- **Shared structure: yes.** Shifting each stock's returns in time destroys the curvature (Ollivier falls from +0.15 to −0.08). Stocks share structure, as everyone knows.
- **More than the market and sectors: no, for Ollivier.** Returns rebuilt from each stock's market and twin-fund loadings, with its own residuals scrambled in time, give *more* Ollivier curvature than the real returns, in 88% of months. Their series correlates 0.94 with the real one. The part of Ollivier curvature that ordinary factors don't produce is slightly negative. Forman is 5.1 less negative in the real data than in the factor rebuild because the factor graph's edges crowd onto hubs more (Σd²/m 131 against 119), even though it also has more triangles (10.6 against 9.8 per edge).
- **More than degrees: yes.** The real graph has 9.8 triangles per edge against 5.5 after rewiring, which is all that moves Forman (+38.6). Ollivier is 0.039 higher on the real graph.
- **Stable: partly.**
  - Curvature from a window's odd days against its even days correlates 0.53 across months for Ollivier and 0.64 for Forman (0.46 and 0.62 on the 500-stock months).
  - Each half against the full window: 0.81–0.88.
  - The same series on 252-day windows correlates only 0.32 (Ollivier) and 0.40 (Forman) with the 63-day series.
  - So a good part of each month's value is estimation noise or depends on the window. (The changes' correlation, −0.33, is an artefact; see decision 52.)

### B2: does curvature add to forecasts? (expanding OLS, Clark-West)

- **Scoring months:** decision dates 2008-02-04..2019-12-06, 143 months. 2008-01-03 had 59 earlier months whose targets had ended, one short of the 60 required.
- **2020-01..2025-05:** 65 months, reported.
- **How to read the tables:**
  - "OOS R² vs smaller" is 1 − SSE(larger) ÷ SSE(smaller);
  - the MSE gain is the mean of the smaller model's squared error minus the larger model's, with a Newey-West 90% interval;
  - every t is Clark-West with Newey-West (2 lags), and p is one-sided.

**SPY's next 21-day excess return (direction):**

| Comparison | 2008–2019 t | p | Bootstrap p | OOS R² vs smaller | MSE gain (90%) | 2020–2025-06 t | p |
|---|---|---|---|---|---|---|---|
| **3 vs 2: + Ollivier (the key test)** | **+0.21** | **0.417** | 0.372 | −0.25% | −7.0e-06 (−8.4e-05..+7.0e-05) | −1.21 | 0.887 |
| 4 vs 1: + Forman | −0.42 | 0.661 | 0.678 | −1.33% | −3.4e-05 (−1.3e-04..+6.4e-05) | −0.52 | 0.698 |
| 2 vs 1: + plain statistics | −1.33 | 0.909 | 0.896 | −7.44% | −1.9e-04 (−3.2e-04..−6.3e-05) | −1.46 | 0.928 |
| 3 vs 1 | −0.88 | 0.811 | 0.810 | −7.71% | −2.0e-04 (−3.7e-04..−2.9e-05) | −1.49 | 0.932 |
| + Samal et al.'s Forman vs 1 | −0.02 | 0.509 | 0.452 | −0.62% | −1.6e-05 | −0.89 | 0.812 |
| + published-graph Ollivier vs 1 | −0.91 | 0.818 | 0.817 | −3.07% | −7.9e-05 | +0.77 | 0.222 |
| + published-graph Forman vs 1 | −0.25 | 0.600 | 0.689 | −6.38% | −1.6e-04 | +0.31 | 0.380 |

- **No model forecasts direction:** against the historical mean of excess returns, the out-of-sample R² over 2008–2019 is −8.8% for the baseline, −16.9% for model 2, −17.2% for model 3 and −10.3% for model 4. Every added input made the forecast worse. 2020–2025-06 is the same: −5.6% to −9.6%.

**SPY's next 21-day log realized volatility (risk):**

| Comparison | 2008–2019 t | p | Bootstrap p | OOS R² vs smaller | MSE gain (90%) | 2020–2025-06 t | p |
|---|---|---|---|---|---|---|---|
| **3 vs 2: + Ollivier (the key test)** | **−0.87** | **0.807** | 0.822 | −0.65% | −1.3e-03 (−3.3e-03..+6.7e-04) | −0.49 | 0.689 |
| 4 vs 1: + Forman | +2.74 | 0.003 | 0.003 | +4.07% | +8.1e-03 (+1.8e-03..+1.4e-02) | −0.75 | 0.775 |
| 2 vs 1: + plain statistics | +0.89 | 0.187 | 0.141 | −1.42% | −2.8e-03 (−1.6e-02..+1.0e-02) | −1.01 | 0.843 |
| 3 vs 1 | +0.71 | 0.240 | 0.195 | −2.08% | −4.1e-03 (−1.7e-02..+8.8e-03) | −1.24 | 0.893 |
| + Samal et al.'s Forman vs 1 | −0.52 | 0.698 | 0.632 | −0.70% | −1.4e-03 | −0.69 | 0.755 |
| + published-graph Ollivier vs 1 | −0.44 | 0.670 | 0.689 | −2.95% | −5.8e-03 | +2.37 | 0.009 |
| + published-graph Forman vs 1 | −0.83 | 0.796 | 0.767 | −1.96% | −3.9e-03 | −1.38 | 0.916 |

- **Risk is forecastable, curvature or not.** Against the historical mean, the out-of-sample R² over 2008–2019 is +38.2% for the baseline (SPY's own volatility, average correlation, the absorption ratio), +37.4% for model 2, +36.9% for model 3 and +40.7% for model 4. This is the "temperature" half of Luca's analogy, and the old yardsticks already deliver it.
- **Forman against the baseline** (reported, not in any family) is the one comparison that came near: t 2.74 over 2008–2019, below the bar of 3. It reversed in 2020–2025-06 (t −0.75; R² 32.8% against the baseline's 37.1%). The plan reads it as "graph statistics help", and model 2, which holds the same statistics, didn't help (t 0.89).
- **The published-graph Ollivier** helped the risk forecast in 2020–2025-06 (t 2.37), after doing nothing in 2008–2019 (t −0.44). It is one of 14 reported comparisons per period, uncorrected.

**SPY's maximum drawdown over the next 21 days (secondary):** model 3 vs 2 t +0.04 (p 0.486) over 2008–2019 and −1.22 over 2020–2025-06. Forman vs baseline t +1.31.

**The twin funds** (pooled panel: 3,927 fund-months, 124 funds, 1–23 groups a month with at least 10 internal edges; Clark-West averaged across funds each month, Newey-West over months):

| Target | Comparison | Months scored 2008–2019 | Fund-months | t | p | OOS R² vs smaller | 2020–2025-06 t | p |
|---|---|---|---|---|---|---|---|---|
| Excess return | 3 vs 2: + group Ollivier | 2010-08..2019-12 (112) | 1,299 | −1.29 | 0.901 | −3.30% | +0.13 | 0.447 |
| Excess return | 4 vs 1: + group Forman | 2010-08..2019-12 | 1,299 | −1.51 | 0.934 | −3.63% | +0.40 | 0.344 |
| Excess return | 2 vs 1: + internal edges | 2008-02..2019-12 (142) | 2,311 | −0.13 | 0.551 | −0.16% | +2.30 | 0.011 |
| Log volatility | 3 vs 2 | 2010-08..2019-12 | 1,299 | −0.25 | 0.599 | −1.45% | +0.07 | 0.470 |
| Log volatility | 4 vs 1 | 2010-08..2019-12 | 1,299 | +0.08 | 0.469 | −0.96% | +0.38 | 0.353 |
| Log volatility | 2 vs 1 | 2008-02..2019-12 | 2,311 | +1.53 | 0.063 | +0.11% | +1.43 | 0.076 |
| Maximum drawdown | 3 vs 2 | 2010-08..2019-12 | 1,299 | −0.60 | 0.725 | −2.77% | +1.82 | 0.034 |
| Maximum drawdown | 4 vs 1 | 2010-08..2019-12 | 1,299 | −0.92 | 0.822 | −2.76% | +2.74 | 0.003 |
| Maximum drawdown | 2 vs 1 | 2008-02..2019-12 | 2,311 | −0.64 | 0.738 | −0.24% | +2.46 | 0.007 |

- **The first fund month with 60 training months and standardized group curvature is 2010-08** (decision 47).
- **The training sets differ:** models 1 and 2 need no z-score, so they train on more fund-months than models 3 and 4. Each comparison scores the same fund-months, those where both forecasts exist. It is a secondary that fails either way.
- **No fund comparison helps in 2008–2019.** Several do in 2020–2025-06, mostly for drawdowns (t 1.8–2.7). That period isn't a test, and those comparisons were never in the family.

**Curvature around SPY's drawdowns of 15% or more** (z-score against the previous 60 month-ends; the crisis side is + for Ollivier and − for Forman):

| Peak → trough | Depth | Ollivier z: 6 months before, 3 before, at the peak, at the trough | First month beyond +1, from the peak | Forman z (same four) | First beyond −1 |
|---|---|---|---|---|---|
| 2007-10-09 → 2009-03-09 | 55.2% | +0.43, −0.15, −0.07, +0.61 | +6 (after the peak) | −1.57, −1.84, −1.84, −0.64 | −12 |
| 2018-09-20 → 2018-12-24 | 19.3% | +0.85, +1.22, −1.42, +0.06 | −5 | −0.04, +0.55, +0.14, −0.48 | none |
| 2020-02-19 → 2020-03-23 | 33.7% | −0.15, +1.58, +0.57, +0.57 | −3 | −0.55, +2.06, −0.04, −0.50 | −11 |
| 2022-01-03 → 2022-10-12 | 24.5% | −0.21, +2.02, −0.56, +0.41 | −12 | +0.32, +2.52, −0.01, −0.28 | none |
| 2025-02-19 → 2025-04-08 | 18.8% | −1.09, −0.23, −0.98, −0.37 | none | −0.56, +0.27, −0.29, −0.21 | −10 |

- **Spikes before peaks are what chance gives.**
  - Ollivier is beyond +1 in 15% of all months, and at least once in 71% of all 13-month spans, the span searched here.
  - Forman: 19% and 61%.
  - At the peaks themselves, Ollivier was below its trailing mean in 4 of 5 episodes.
- **In 2008 Ollivier rose only after the peak.** It crossed +1 six months after the market top.
- **The 2010 and 2011 corrections** (−15.7% and −18.6%, reported with the peak reset after each trough, decision 53): Ollivier z at the peak was +0.02 and −0.92.
- **Survivor-only data:** the stock universe lacks firms that later failed, so every crisis row above is possibly biased.

### B3: money tests

**The market rule** (100% SPY when Ollivier is below its trailing median, 50% otherwise; decided at t₀, bought at the close of t₀ + 1, cash at the T-bill, SPY = Yahoo adjusted close; 61 switches in 242 months):

| 2005-04..2025-06, 242 months | Annual, before costs | At 5 bps | At 20 bps | Sharpe (5 bps, over the T-bill) | Volatility | Max drawdown | Average exposure | Turnover a year |
|---|---|---|---|---|---|---|---|---|
| **The rule (Ollivier)** | **+8.0%** | +8.0% | +7.7% | 0.51 | 13.4% | −31.3% | 74% | 1.61 |
| Random timing, median of 10,000 | +8.3% | | | | | | 74% | |
| Buy and hold SPY | +10.5% | +10.4% | +10.4% | 0.52 | 19.3% | −55.2% | 100% | 0.05 |
| Constant 74% | +8.4% | +8.4% | +8.4% | 0.52 | 14.2% | −44.0% | 74% | 0.12 |
| Volatility threshold (100% when 21-day vol is below its median) | +8.2% | +8.1% | +7.9% | 0.56 | 12.2% | −31.3% | 75% | 1.75 |
| Volatility-managed SPY, same exposure | +7.5% | +7.4% | +7.1% | 0.55 | 11.1% | −22.5% | 74% | 2.06 |
| `-3.2`'s average-correlation rule | +8.4% | +8.3% | +8.1% | 0.45 | 17.1% | −54.4% | 73% | 1.03 |
| `-3.2`'s absorption rule | +8.2% | +8.2% | +8.1% | 0.56 | 12.4% | −32.6% | 75% | 0.78 |
| The Forman rule (reported) | +7.7% | +7.6% | +7.3% | 0.46 | 14.4% | −42.3% | 74% | 1.48 |

**Paired net differences** (rule − baseline, mean monthly net-of-5-bps return, Newey-West t, one-sided p):

| Baseline | 2005-04..2019-12 (the test) | t | p | Bootstrap p | 2005-04..2025-06 | t | 2020-01..2025-06 | t |
|---|---|---|---|---|---|---|---|---|
| Constant allocation | −0.083% | −1.04 | 0.850 | 0.853 | −0.044% | −0.59 | +0.054% | +0.34 |
| Volatility threshold | −0.079% | −1.22 | 0.888 | 0.955 | −0.008% | −0.12 | +0.187% | +1.21 |
| The better `-3.2` rule (absorption in 2005–2019; average correlation in the other two windows) | −0.100% | −1.70 | 0.956 | 0.982 | −0.059% | −0.46 | +0.108% | +0.47 |
| Volatility-managed (reported) | −0.071% | −0.63 | 0.736 | 0.837 | +0.051% | +0.48 | +0.377% | +1.47 |
| Buy and hold (reported) | −0.286% | −1.99 | 0.977 | 0.979 | −0.248% | −2.08 | −0.146% | −0.68 |

**By period** (random timing reordered within each; before-cost annual return; Sharpe at 5 bps):

| Period | Months | Rule | Buy and hold | Median random | Rule − random | p | Rule Sharpe | B&H Sharpe | Vol-threshold Sharpe | Vol-managed Sharpe | Rule exposure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2005-04..2009 | 57 | +2.3% | +1.2% | +2.3% | −0.0 | 0.50 | 0.02 | 0.06 | 0.06 | 0.15 | 62% |
| 2010–2014 | 60 | +8.7% | +14.5% | +11.1% | −2.4 | 0.93 | 0.70 | 0.93 | 0.80 | 0.81 | 77% |
| 2015–2019 | 60 | +8.3% | +12.2% | +9.8% | −1.5 | 0.89 | 0.66 | 0.85 | 0.84 | 0.86 | 78% |
| 2020–2025-06 | 65 | +12.6% | +13.6% | +11.4% | +1.1 | 0.30 | 0.65 | 0.57 | 0.61 | 0.37 | 79% |

- **Its smaller drawdown is no better than the plain volatility rules'.** It lost 31%, against 55% for SPY and 44% for the constant allocation. The volatility threshold also lost 31%, volatility management 22% and the absorption rule 33%. Its return is what random orderings of its months earn.
- **Timing doesn't change it:**
  - traded at the close of t₀ (the upper bound), the rule is +8.0% against +8.3% for random (−0.3, p 0.61);
  - with cash at 0, +7.7% against +7.9% (−0.2, p 0.57).
- **After costs by calendar year** (rule at 5 bps − SPY, points): 2005 −3.6, 2006 −6.2, 2007 +0.1, 2008 +17.5, 2009 −12.1, 2010 −11.4, 2011 −4.1, 2012 −9.9, 2013 −2.6, 2014 −3.6, 2015 −4.6, 2016 −2.8, 2017 −2.5, 2018 +2.2, 2019 −12.4, 2020 +3.3, 2021 −12.3, 2022 +4.4, 2023 +1.5, 2024 −0.0, 2025 (to 06-16) −5.3. The mean is −3.05.

**Fund risk scaling** (twin funds with a group value each month, weighted by 1 ÷ forecast volatility, bought at the close of t₀ + 1, 5 bps on turnover):

| 2010-08..2019-12, 113 months, median 12 funds | Volatility | Sharpe | Annual | Turnover a year | Mean monthly realized variance |
|---|---|---|---|---|---|
| With group curvature (B2's fund model 3) | 14.80% | 0.68 | +10.0% | 3.90 | 1.8290 × 10⁻³ |
| Without it (model 2) | 14.77% | 0.68 | +10.0% | 3.92 | 1.8217 × 10⁻³ |
| Trailing 63-day volatility | 14.74% | 0.69 | +10.2% | 3.85 | 1.8146 × 10⁻³ |

- **Curvature made the portfolio slightly riskier, not calmer.**
  - Without minus with: −7.3 × 10⁻⁶ a month, t −3.84; the pass needed a positive difference.
  - In 2020-01..2025-05 (65 months, median 10 funds): t −2.17, with volatility 21.09% with curvature against 21.07% without.
  - Plain trailing volatility did best in both periods.
- **The sizes are tiny,** 0.03 points of annual volatility. Every version is an inverse-volatility basket of the same 10–12 sector funds (the median per month); only the weights differ.

### B5: the covariance manifold (does the representation matter?)

**Construction:**
- **Months:** 245 calendar months, 2004-12-31..2025-05-14, on MV-LW's own list. It held 298–301 stocks until 2007 (its 756-day data rule meets the export's roughly 320 stocks before 2005) and 500 from 2008.
- **Check:** LW-252's minimum-variance weights here equal the saved MV-LW weights of `-4.6-correlation-portfolios.md` in all 245 months, with a maximum difference of 0.0 and the same stock lists.
- **Solver:** the worst KKT violation over all of about 33,000 solves is 1.5 × 10⁻¹⁵.
- **Scoring** starts 2006-02-01, the first month with 12 months of loss history: 167 months to 2019-12 and 65 months for 2020-01..2025-05.

**The test** (the loss is the realized variance of the capped minimum-variance portfolio over the 21 days after the t₀ + 1 trade; volatilities annualized from the mean monthly loss):

| Period | Months | Euclidean | Log-Euclidean | LW-252 (MV-LW) | Affine step B A⁻¹ B | Euclidean − log-Euclidean: t | p | Bootstrap p |
|---|---|---|---|---|---|---|---|---|
| **2006-02..2019-12 (the test)** | 167 | 12.37% | 12.38% | 12.38% | 13.01% | **−0.15** | **0.560** | 0.598 |
| 2020-01..2025-05 | 65 | 16.18% | 16.16% | 16.53% | 16.32% | +0.37 | 0.357 | 0.352 |
| 2006–2009 | 47 | 16.90% | 16.93% | 16.92% | 17.74% | −0.21 | 0.585 | 0.608 |
| 2010–2014 | 60 | 10.37% | 10.31% | 10.38% | 10.79% | +1.33 | 0.092 | 0.112 |
| 2015–2019 | 60 | 9.72% | 9.78% | 9.71% | 10.39% | −1.85 | 0.968 | 0.978 |

- **B5 fails.** The mean monthly loss difference (Euclidean − log-Euclidean) over 2006–2019 is −1.9 × 10⁻⁶, against a typical monthly variance of 1.3 × 10⁻³ (12.4% a year):
  - Newey-West 90% interval −2.3 × 10⁻⁵..+1.9 × 10⁻⁵;
  - bootstrap interval −2.1 × 10⁻⁵..+1.4 × 10⁻⁵;
  - log-Euclidean was lower in 50% of months.

  The two representations build almost the same portfolio, because both choose mostly the 252-day matrix:
  - Euclidean: (w₂₁, w₆₃, w₂₅₂) = (0.1, 0, 0.9) in 126 of 167 months, mean (0.09, 0.06, 0.85);
  - log-Euclidean: (0.2, 0, 0.8) in 133, mean (0.18, 0.04, 0.78).
- **Both against LW-252, MV-LW's existing estimator:**
  - 2006–2019: no difference (LW − log-Euclidean t −0.04, LW − Euclidean t +0.20);
  - 2020–2025-05: both combinations were a little calmer than LW-252, 16.2% against 16.5% (t +1.54 and +1.19; bootstrap p 0.011 and 0.018). That period is reported, not tested.
- **QLIKE** (reported; against the next 21 days' shrunk covariance):
  - means over 2006–2019: Euclidean 627, log-Euclidean 631, LW-252 800, affine step 1,240;
  - the difference Euclidean − log-Euclidean has t −0.52;
  - by period it changes sign: t −1.95 in 2006–2009, +14.0 in 2010–2014, +2.42 in 2015–2019, −5.41 in 2020–2025-05.
  - In 3 months of 2025 a listed stock has no forward closes, so the QLIKE target is undefined. Those months are left out of QLIKE only (decision 54).
- **The affine-invariant step B A⁻¹ B is valid mathematics, not a useful forecast.**
  - Its smallest eigenvalue stays positive (7 × 10⁻⁹ at worst), so it is a covariance in every month.
  - Its condition number is a median 1.2 × 10⁵ and at worst 1.6 × 10⁹.
  - Its portfolio was riskier than LW-252's: 13.0% against 12.4% over 2006–2019, t −3.32.
  - Its QLIKE is the worst of the four.
  - Stepping along the geodesic past the latest covariance extrapolates estimation noise, so it is statistically unstable and economically unjustified.
- **What the failure shows:** the flat log-Euclidean representation is no better or worse than plain matrix averaging for building minimum-variance portfolios here. It says nothing about market curvature either way.


### The 16 unseen months (new export, run once at the end with the frozen code)

These months were already read by about 65 experiments, so they are reported, never confirming. The months are the 15 calendar points 2025-07-16..2026-09-16 whose 22 days after t₀ fit by 2026-10-19. Every fit and B5's weight history used all earlier months of the new export.

- **The new export agrees with the old over the old span.** Its first 5,911 days equal the old export's. The series correlate 0.998 (Ollivier), 0.996 (Forman) and 0.9999 (average correlation). Its universe matches its own `states_new.npz` exactly, and every construction check passes again (Forman's closed form to 7 × 10⁻¹⁴; degrees kept).

| Test | Unseen months | Value | Old-export result |
|---|---|---|---|
| B2 direction, model 3 vs 2 | 15 | Clark-West t +0.45 (p 0.325); OOS R² vs model 2 +0.27% | fail (t +0.21) |
| B2 risk, model 3 vs 2 | 15 | t +1.20 (p 0.116; bootstrap p 0.024); OOS R² +1.35% | fail (t −0.87) |
| B5, Euclidean − log-Euclidean | 15 | t −1.31 (p 0.905); MV volatility 8.05% Euclidean, 8.13% log-Euclidean, 8.04% LW-252, 8.85% affine | fail (t −0.15) |
| B3 rule − median random timing | 15 | +0.9 points annualized (p 0.35) | −0.3 |
| B3 rule − constant / volatility threshold / absorption rule | 15 | +0.075%, −0.123%, −0.109% a month (t +0.60, −0.73, −0.60) | all negative |
| B3 after costs vs SPY | 15 | −2.0 points, annualized over the span (−1.2 by the two calendar pieces) | −3.05 |
| B3 fund risk scaling, without − with curvature | 15 | t −2.19 (curvature riskier again): 11.93% against 11.90% | fail (t −3.84) |

- **B2:**
  - Ollivier's sign turned positive for both targets, on 15 observations; neither reaches t 2.
  - Among the reported comparisons, the published-graph Forman helped direction (t 2.83). It had been negative in 2008–2019 (t −0.25), and it is one of 21 comparisons.
  - The fund panel (108 fund-months) shows nothing: t −0.63..+1.13.
- **B3:**
  - The rule held 50% in July 2025, 100% from August 2025 to April 2026, then 50% from May 2026.
  - It returned +16.2% a year before costs against SPY's +18.2%, the constant 80% allocation's +15.2% and the volatility threshold's +18.1%.
  - Random timing's median was +15.3%. The better `-3.2` rule here was the absorption rule (+17.9%).
  - The constant and volatility-managed baselines are set to the rule's exposure on these 15 months themselves, whereas `-3.2` reused the old span's c.
- **B5:**
  - Log-Euclidean had the lower realized variance in 20% of the months.
  - Both combinations matched LW-252: LW − Euclidean t −0.08, LW − log-Euclidean t −0.75.
  - The affine step was again the riskiest (t −3.98 against LW-252).
  - QLIKE again favoured Euclidean (t −2.55).


### Frozen for forward confirmation (tag pending commit)

Brandon hasn't asked for commits, so there is no git tag yet. These are the sha256 hashes of the final scripts, taken 2026-10-23 20:54 CDT, before the unseen months were run. Every result above was reproduced with these exact files: `tests.py`, `money.py`, the `covariance.py` report and `b_extra.py` were rerun after the last edit, and their outputs are identical. The environment is `research/market_geometry/.venv` (git-ignored, made by `uv venv --python 3.12`): Python 3.12.12, numpy 2.3.5, scipy 1.16.3, POT 0.9.6.

```
53abcc686ad6f5eaeb6d587f83ba123fc374065093880665385561d7f8229dac  b_common.py
a57e4c7454bdf0ce90224ea8279711d6c29bb9d557b370bda3824cef6f4932f7  pilot.py
5ed0583aa74a63c0a265b0f57b49ca78eda789a3b5c9e3ab11aaa735fee605de  curvature.py
1f695f223e0bf01c87479aa414e07500bf8a3e5fadb43ffb31f13943444b0c81  tests.py
0044d72aa1acf3259e5bed9dc13caeb88856fa8f8afe11fc6bd7de8bb8b5084c  money.py
13c3d78fd0b2e549b0b0b9e8692dd02f3688b1b44905248715545613f0a21263  covariance.py
5a7f60e6224c277484709b99be5490c2a3d0ee3848018b3ae7b664d59a3cd9f7  unseen.py
f838d961b3e15c576ef41b75acde4a98885d66a97f42f433bbeba0ea02a5fdef  b_extra.py
ba982e5771e01806a2c98818a03fb96e9353d9c11984f020487995f40e7ef746  b_artifact.py
```

`b_artifact.py` only builds the explainer page's data. It changed after the unseen run, when one sentence of its text was corrected (the B3 drawdown comparison), so its hash above is the final file's; it was 23735c5c… at the freeze. No test script changed after the freeze.

- **Forward test: pending.** Closes after 2026-10-19 don't exist yet. `forward.py` isn't written; it is the plan's last file and needs the tag. The first reading is at 12 months and the deciding one at 36, each primary at one-sided p < 0.05 after Holm across the four.
- **This depends on downloads:** Stooq downloads are manual, so data accrues only if someone keeps downloading, at least monthly, with `c2 update` run on each.
- **Nothing to confirm from B:** all three of B's primaries failed, so the forward log would only record that they keep failing.


### Not run

- **B4** (do markets follow geodesics of the macro state space S?): deferred by the plan until A, B2 and the pilot are done and its model is written into the plan file. It was not run.
- **The directional fund rotation:** the plan runs it only if B2's fund-level return forecast passes. That forecast failed (t −1.29), so it was not run.
- **Nothing else was skipped or approximated.** Every Ollivier series ran with exact transport. Every check (shift null, factor null, degree-preserving rewiring, odd/even halves, 252-day windows) ran in every month, except the factor null's first 3 months (decision 46).

### Files and runtime

**Code** (`research/market_geometry/`; run each under `systemd-run --user --scope -p MemoryMax=5G env OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 .venv/bin/python -I -B`, in this order):
- `b_common.py`: the settings, the universe (states() reimplemented), the graphs, both curvatures, rewiring, and the statistics (Newey-West, block bootstrap, Holm);
- `pilot.py`: the timing gate, writing `pilot.txt`;
- `curvature.py old|new`: the series and the checks, writing `curvature_old.txt`;
- `tests.py old`: B1, the checks' summary, B2 and the drawdowns, writing `tests_old.txt` / `.json`;
- `money.py old`: B3, writing `money_old.txt` / `.json`;
- `covariance.py old [report]`: B5, writing `covariance_old.txt` / `.json`;
- `b_extra.py`: the after-results diagnostics (decision 53), writing `extra_old.txt` / `.json`;
- `unseen.py`: the 16 months, once; it runs `curvature.py new` and `covariance.py new` first, and writes `unseen_B.txt` / `.json`;
- `b_artifact.py`: writes `artifact_data_B.json` for the explainer page.

**Large files** (`~/.local/share/correlation2/experiments/market_geometry/b/`):
- `curv_<tag>.npz` (per-month series);
- `edges_<tag>.pkl` (per-edge primary-graph curvatures, 26 MB);
- `groups_<tag>.pkl`;
- `fund_forecasts_<tag>.pkl`;
- `cov_<tag>.npz` (all 66 + 66 candidates each month);
- the logs.

**Runtime** (wall clock on the shared machine, load 43–78 on 24 cores):
- pilot 4 min;
- `curvature.py old` 23 min;
- `tests.py` and `money.py` about 1 min each;
- `covariance.py old` 35 min;
- `unseen.py` 52 min (`curvature.py new` 22 min, `covariance.py new` 29 min, the scoring 1 min);
- the whole of B, from the first search to this file: about 2 h 25 min (19:28–21:53 CDT).
