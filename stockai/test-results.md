# stockai: test results and implementation choices

## Where things stand (read this first)

**Terms used here:** G1 is PLAN's gate for stockGPT v1 (milestone M2, after G0's pipeline check): a real gross signal, nothing on a null model, no leakage tripwire. M1 is the judge-upgrades-and-baselines milestone, whose gate is that the canaries (below) drop to chance. A seed is one set of 14 yearly models trained from a different random start; the ensemble averages the 3 seeds' scores. v is correlation2's daily move: a stock's daily return net of the market's, minus that leftover's average over the past year, in units of its own typical daily move (item 1 below). bps are hundredths of a percent, charged on each trade. p is the share of 1,000 random accounts that did as well. Uniform random accounts pick at random among the same candidate stocks; volatility-matched ones pick stocks as calm or as volatile as the strategy's picks. LightGBM is a standard tree-based machine-learning model, here given 21 simple price features. The 5-day reversal rule buys last week's biggest losers; 12-1 momentum buys the past year's winners, skipping the last month. SPY and QQQ are the S&P 500 and Nasdaq-100 index funds.

**The proof of concept ran end to end on correlation2's data.** stockGPT v1 is a 0.85M-parameter GPT that reads 256 days of correlation2's daily move tokens and predicts the next day's.
- **Training:** one model per test year from 2010-06-28 to 2024-06-27, each trained only on earlier data, with 3 seeds each: 42 models.
- **Grading:** its daily scores were graded by correlation2's $10,000 holdout test, next to LightGBM, the 5-day reversal rule and 12-1 momentum. A second LightGBM, trained on actual next-day returns, was added during verification, after the POC run ("extra").
- **Time:** 61 minutes for the whole run (`scripts/poc.sh`), measured before the LightGBM extra was added (it adds about 2 minutes).
- **Locked year:** nothing graded, trained on or looked at the locked year (2024-06-28 on).
- **Checked:** independent agents re-derived the judge, the data and the numbers here (see "Changes made during verification").

**The answer:**
- **Gate G1 passed. stockGPT learned a real, small, gross next-day signal.**
  - Rank IC (each day's correlation between its ranking of the stocks and their next-day ranking; 0 is chance) against next-day actual returns: **+0.0124, combined z = 5.85, positive in 13 of 14 years.** Combined z pools the 14 yearly t-statistics; 2 or more is unlikely to be chance. That is the size the literature predicted (0.005–0.02).
  - The sign-flip null (the same model trained on data whose moves got a random sign each day) showed nothing (z = 0.08).
  - On the 500 largest stocks the IC is +0.0129, under the 0.05 leakage tripwire (a large-cap IC above 0.05 counts as a bug) but positive, so the signal isn't only in small caps.
- **Its daily account earns less than SPY before costs and loses money after them (the weekly one keeps +8.3% a year at 5 bps and −0.7% at 20 bps, against SPY's +15.0%), and it is no better than turnover-matched random accounts at any cost** (1,000 accounts that pick at random among the same stocks and trade as often).

  | | Years beating random at p < 0.1 (0 / 5 / 20 bps) | Mean yearly return (0 / 5 / 20 bps) | Random median (0 / 5 / 20 bps) |
  |---|---|---|---|
  | **Daily** | **4 / 4 / 3 of 14**, uniform random accounts; **3 / 3 / 3** matched on volatility | +9.3% / −4.7% / −35.9% | +6.2% / −7.3% / −37.2% |
  | **Weekly** | 4 / 4 / 4 (5 / 5 / 5 volatility-matched) | +11.4% / +8.3% / −0.7% | +6.5% / +3.5% / −4.8% |

  - **Chance alone gives about 1.4 years** (p < 0.1 means fewer than 10% of the random accounts did as well; 4 or more of 14: 4.4% if the years are independent). The three cost columns are one test, not three: the random accounts trade as often, so p moves by at most 0.04. Years fall in the other tail too (p > 0.9: 2 uniform, against 4 below 0.1; 4 volatility-matched, against 3), and all 14 yearly p's combined give z = +0.8 at 0 bps (+0.7 at 5 bps, +0.5 at 20 bps; +0.5 volatility-matched).
  - **SPY over the same 14 years:** +15.0% a year.
  - **Why costs win:** the daily account replaces about 61% of the names it holds each day (2,810 buys a year), so costs of c bps per trade (1 bp = 0.01%) take about 1.2 × c bps/day, about 15% a year at 5 bps. The long-short check (top tenth minus bottom tenth) earns 2.44 bps/day gross (z = 1.81); its two books replace 59% of their names a day, so costs take about 2.4 × c bps/day, and 1 bp erases it.
- **A one-line rule matches its IC, though not its account.** Score = minus today's move (−v[d]): IC +0.0118 (z = 5.70), but 2 / 2 / 2 years beating random and +6.2% / −16.6% / −59.6% a year.
  - What stockGPT learned is mostly that one-day reversal plus a lean toward calm stocks: over all scored stocks, its scores correlate −0.31 with today's move and −0.47 with the stock's volatility.
  - Its picks' mean volatility decile (trailing 63-day volatility among the day's candidates, 0 = calmest, 9 = most volatile) is 2.0, against 4.5 for random accounts.
- **The 5-day reversal rule and LightGBM do better on money. A LightGBM trained on actual next-day returns has the strongest daily IC in the POC and beats random about as often as the 5-day rule (10 of 14 years daily, 6 weekly), but at 20 bps its daily account loses more than theirs:**

  | Model | IC, actual returns (z) | Years beating random at p < 0.1 (0 / 5 / 20 bps) | Mean yearly return (0 / 5 / 20 bps) |
  |---|---|---|---|
  | 5-day reversal, weekly | +0.0145 (3.05) | **6 / 6 / 6** (7 / 7 / 7 volatility-matched) | **+27.9% / +22.0% / +5.8%** |
  | 5-day reversal, daily | +0.0088 (3.67) | 10 / 10 / 9 | +28.0% / +14.8% / −17.1% |
  | LightGBM, daily | −0.0022 (−0.77) | 8 / 8 / 8 (5 / 5 / 5 volatility-matched) | +27.8% / +10.2% / −29.2% |
  | LightGBM on actual returns, daily (extra) | **+0.0210 (9.15)** | **10 / 10 / 10**; a day late: 4 / 4 / 4 | +49.2% / +20.9% / −35.7% |
  | LightGBM on actual returns, weekly (extra) | +0.0145 (2.71) | 6 / 6 / 6; a day late: **7 / 7 / 7** | +26.3% / +20.5% / +4.6% (a day late: +25.0% / +19.3% / +3.7%) |

  - **LightGBM on actual returns:** its IC is positive in 14 of 14 years. But trading one day later cuts its daily wins from 10 to 4 of 14 years: the edge is very short-lived, like reversal, and part of close-to-close reversal is bid-ask bounce (a close that prints at the bid looks like a drop that recovers the next day), which a real trader can't capture.
  - **Weekly, it is level with the 5-day reversal rule, and a day late it still beats random in 7 of 14 years** (6 against volatility-matched accounts). Of the weekly models graded a day late it holds up best (stockGPT: 3 years, the v-trained LightGBM: 4); the rules weren't graded a day late. Even so, at 20 bps it earns +4.6% a year (+3.7% a day late), against SPY's +15.0%.
  - **Its large-cap tripwire fired on one year** (2021: +0.0503 against the 0.05 limit). A leak check found no leak: all 14 years' models refit bit for bit, each year's model comes out unchanged when the data after its last validation outcome is garbled, 46 sampled days (168,633 scores) rebuild bit for bit from data garbled after d, and 2021's excess is a volatility tilt in the one year where low volatility alone crosses 0.05 ("Leakage tripwire" below).
  - LightGBM trained on v wins on picks despite a slightly negative IC in actual returns: its lean on k (item 1 below) costs −0.0117, which cancels the +0.0095 that its volatility tilt and the rest of its ranking earn (below).
- **Nothing in the POC compounds to more than SPY after 20 bps costs** (SPY: +550% over the 14 years). 12-1 momentum comes closest (+319% compounded weekly, +288% monthly). Rebalanced monthly it makes +15.6% a year on average, level with SPY's +15.0%, and it is ahead of SPY in 7 of 14 years.
- **The canaries did not drop to about 0, so M1's gate is not met as written. They removed the timing part; a slower part remains, and it is not a leak.** The canaries are the same scores made stale or shuffled in time, which removes any timing information.
  - **Shifted 20 days later** (stale): IC z = 2.38 against correlation2's v ("census units", item 1 below) and 3.02 against actual returns.
  - **Dates permuted within each year:** z = 3.00 and 3.22.
  - **Shifted a year later:** about 0 (z = 0.44 and 1.59). So did one-day reversal's own canaries, in IC (z −0.39 to 1.27).
  - **Reading:** about 40–55% of the IC comes from a slowly varying property of each stock. About half of that part is the low-volatility tilt; the other half (+0.003 in actual returns, z 3) is a slower property orthogonal to volatility and to the census artifact (k, item 1 below), still unexplained, and gone when the scores are a year stale. A 20-day-stale score can't contain day d+1, so it isn't a leak.

**Three things that change how to read the numbers:**
1. **Census-unit IC is not all signal.**
   - correlation2's variable v subtracts each stock's trailing 252-day mean residual before scaling, so part of tomorrow's v (call it k) is known today. That part is not in actual returns. (docs/c2-export.md said v was the residual ÷ sd; now corrected.)
   - A score that leans against last year's performance gets census-unit IC for free: LightGBM gets z = +24.0 in census units against −0.77 in actual returns, and 12-1 momentum gets −19.8 against +2.5.
   - stockGPT doesn't use it: it correlates −0.10 with k, only through its low-volatility tilt, and with k removed from the outcome its census IC is +0.0176 (z 9.8).
   - So G1 is judged in actual returns.
2. **The judge buys only positive scores, and stockGPT's expected values are often all negative.**
   - The scores' overall level drifts by seed and year. The ensemble's scores were negative for every candidate on 244 of the 251 days of 2020–21 (one or two positive names on the other 7), so that account sat almost entirely in cash. Seed 0 sat fully in cash in 2020, and seed 1 in 2018.
   - Fewer than 20 positive names on some days cost a few names in 2010, 2011, 2016 and 2022: 19.6 names held a day over the 13 invested years, 18.2 counting 2020.
   - An extra variant ranks each day's scores into percentiles, so the account always holds the top 20. It leaves every IC identical and doesn't help: 3 / 3 / 3 years beating random.
3. **The data is point in time except for two inputs** (`docs/c2-export.md`): the bad-print repair looks at the next day's close, and the $5 price test uses prices adjusted for later splits, so stocks that later split heavily (NVDA in 2010–2017 and 2019) are left out of early universes. Both affect every model and the random accounts alike.

**Run it yourself** (from `~/programming/stockai`):

| Command | What it does |
|---|---|
| `scripts/poc.sh --quick` | Test years 2022–2023, one seed, the null on 2022: about 7 minutes on a free GPU (401 s; 633 s when another job shared the GPU), 6.3 GiB peak memory. Writes `results/poc-quick/` |
| `scripts/poc.sh` | The full POC: 61 minutes with the GPU partly shared, measured before the LightGBM extra was added (about 2 minutes more). Writes `results/poc/` (`summary.md` has every table) and `runs/poc/` (scores, checkpoints, logs) |
| `scripts/poc.sh --steps grade,controls,summary` | Regrade existing scores (147 s for the 46 files) |
| `uv run pytest -q` | 77 tests, 36 s |

**Your call:**
1. **M4 next, or stop the stockGPT line.** M4 means weekly and monthly targets with turnover limits, for stockGPT and LightGBM on actual returns alike. stockGPT passed G1, but as a strategy it is a weaker version of the 5-day reversal rule (though it beats the one-day −v[d] rule on money), and PLAN's G2 requires beating LightGBM, which is now a higher bar.
2. **The positive-score rule.** It suits correlation2's families, whose scores are signed predictions. For model scores, should the judge buy the top 20 whatever their sign? I kept the rule and reported the ranked variant beside it.
3. **Fresh data.** Sharadar (PLAN §5.3) includes delisted stocks and corporate actions, so it would also fix survivorship and the split-adjusted universe. (The earlier item 3, LightGBM on actual returns, was done during verification.)

---

A running log of what was measured and what was chosen because of it.
- **Machine:** Intel Core Ultra 9 285K (24 cores), 64 GB RAM, NVIDIA RTX 5080 16 GB (driver 595.99), shared with other sessions during these runs.
- **Software:** Python 3.12.12, torch 2.14.1+cu130, numpy 2.5.4, scipy 1.18.1, lightgbm 4.7.0; Rust 1.90.
- **Data:** correlation2's export of `Plan::new(&store, Config::full()).var` at commit 9335126. 8,503 stocks × 5,911 days, 2001-12-31 to 2025-06-27, fingerprint 8b5ac076267c83ab, in `~/.local/share/correlation2/export/full`.
- **Raw outputs:** `results/poc/`: one judge JSON per graded configuration, the printed tables in `tables/`, `g0.json`, `summary.md` and `summary.json`.

## G0: the pipeline check (M0)

**The judge.** `judge/target/release/stockai-judge` is correlation2's holdout test at commit 9335126, forked into `judge/` so correlation2 itself is not modified. Its data and census crates are pinned git dependencies.

G0 asks whether it grades a file of scores exactly as correlation2 grades its own reversal family. All of these held before anything else ran. They are rerun by `scripts/poc.sh` and recorded in `results/poc/g0.json`.

| Check | Result |
|---|---|
| correlation2's own binary (built at 9335126) against the judge: `--family reversal --lookback 5 --rebalance 5 --walk-forward --start 2023-06-28` | **22 of 22 lines identical** (17 year rows and the summary lines; the judge only appends columns) |
| Python reversal scores (`stockai.baselines.reversal`, lookback 5) against the judge's own (`--dump-scores`), wherever the judge computed one | **1,079,142 of 1,079,142 cells bitwise equal** over 866 rebalance days: 0 ulp, no +0.0/−0.0 mismatches, nothing to explain |
| Those Python scores graded with `--family scores --rebalance 5`, against `--family reversal --lookback 5 --rebalance 5` (both with `--walk-forward --start 2023-06-28 --costs 0,5,20 --random 1000 --json`) | **792 of 792 JSON fields identical**: every per-year number of the 17 years, the summary and the random-account checks. The printed tables are identical too |
| From the judge's build report (G0a, G0b; not rerun here) | The same table match at 21/21, 5/1 and `--costs 0`. The dumped scores graded through `--family scores` are identical in all 792 fields at delay 0, delay 1 and with volatility matching. 16 of 16 Rust tests pass (rerun during verification: still 16 of 16) |
| Extra: the judge's two rank ICs recomputed in Python from the export alone (GPT scores, 2023 daily and 2022 weekly) | Equal to about 1e-17 (summation order), with the same 1,968 and 1,969 candidates. This check's output wasn't kept, but during verification two independent recomputations matched the judge's yearly ICs for 2010–2022 to within 6e-17 |

## stockGPT v1 (M2)

### What was trained

PLAN §3.2, as built in `stockai/gpt/`, with no settings changed for this run. Every setting below was fixed before any test year was graded.

| | |
|---|---|
| **Tokens** | v in 101 bins of 0.1 (−5.0 to 5.0) plus a MISSING token |
| **Daily inputs** | Alongside each token: the market's move, the stock's standardized log trailing-252-day volatility, and a volatility-missing flag |
| **Model** | Pre-LayerNorm nanoGPT decoder: 4 layers, 4 heads, width 128, context 256 days, dropout 0.1; 852,480 parameters |
| **Training** | Next-token cross-entropy on windows sampled on the GPU from training days only. AdamW, learning rate 1e-3 decaying to 1e-4 (cosine), warmup min(200, 10% of steps), batch 256, bf16, deterministic kernels. Budget: 4 passes over the training positions, at least 400 steps (400 for 2010–2016, up to 776 for 2023) |
| **Early stopping** | About 6 evaluations per year, patience 3. Each evaluation measures the daily rank IC against next-day actual returns on the 252 validation days, liquid stocks only (close ≥ $5, 63-day median dollar volume ≥ $5M) |
| **Splits** | `stockai/splits.py`: training from 2005-01-03; validation is the 252 trading days before a 5-day gap before the test year; training ends 5 days before validation begins; then the test year |
| **Score** | Expected v at d+1 under the predicted bin probabilities, for every stock with a finite v[d]. The ensemble is the mean of the 3 seeds' scores |

### By test year (ensemble, daily, delay 0)

**What the columns are:**
- **Validation IC:** at each seed's chosen checkpoint, averaged over the 3 seeds. Early stopping picked the evaluation with the highest value, so it is biased upward.
- **Test ICs:** the judge's yearly mean daily rank IC, with its t.
- **Picks:** the account's return over the year, with p = the share of 1,000 turnover-matched random accounts that did as well. The volatility-matched p is at 0 bps.
- **Names:** names held per day.

| Test year | Val IC, actual | Test IC, census (t) | Test IC, actual (t) | Long-short bps/day | Picks 0 bps (p) | 5 bps (p) | 20 bps (p) | p, vol-matched | Random median | SPY | QQQ | Names |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2010–11 | +0.0095 | +0.0131 (3.0) | +0.0052 (1.1) | 2.04 | +11.9% (0.95) | +1.7% (0.95) | −23.6% (0.96) | 0.94 | +22.6% | +19.7% | +23.2% | 19 |
| 2011–12 | +0.0116 | +0.0083 (1.3) | +0.0195 (1.9) | 4.62 | +2.1% (0.09) | −7.6% (0.09) | −31.4% (0.10) | 0.22 | −6.3% | +6.3% | +14.9% | 19 |
| 2012–13 | +0.0198 | +0.0133 (2.2) | +0.0161 (2.0) | 3.95 | +28.0% (0.52) | +16.4% (0.52) | −12.5% (0.53) | 0.07 | +28.4% | +23.7% | +14.7% | 20 |
| 2013–14 | +0.0118 | +0.0116 (2.0) | +0.0117 (1.8) | 1.90 | +24.4% (0.59) | +8.9% (0.60) | −26.9% (0.62) | 0.39 | +25.9% | +24.0% | +33.8% | 20 |
| 2014–15 | +0.0144 | +0.0130 (1.9) | +0.0212 (2.8) | 9.25 | +10.6% (0.28) | −3.0% (0.29) | −34.6% (0.30) | 0.41 | +6.5% | +9.2% | +17.8% | 20 |
| 2015–16 | +0.0226 | +0.0060 (0.8) | +0.0149 (1.8) | 6.40 | +7.6% (0.01) | −5.6% (0.01) | −36.4% (0.01) | 0.05 | −9.6% | −2.8% | −5.4% | 20 |
| 2016–17 | +0.0242 | +0.0000 (0.0) | +0.0027 (0.2) | −3.66 | +9.6% (0.94) | −7.6% (0.94) | −44.7% (0.95) | 0.97 | +21.1% | +23.3% | +36.4% | 19 |
| 2017–18 | +0.0157 | +0.0015 (0.2) | −0.0020 (−0.3) | −4.61 | +7.6% (0.83) | −5.9% (0.83) | −37.1% (0.84) | 0.69 | +13.6% | +13.7% | +24.0% | 20 |
| 2018–19 | −0.0000 | +0.0170 (2.5) | +0.0191 (2.5) | 7.00 | +15.2% (0.04) | −3.5% (0.04) | −43.4% (0.05) | 0.15 | +2.3% | +10.4% | +10.8% | 20 |
| 2019–20 | +0.0163 | +0.0128 (1.9) | +0.0097 (1.3) | 7.52 | +29.6% (0.00) | +7.7% (0.00) | −38.3% (0.00) | 0.01 | −7.3% | +5.0% | +29.8% | 20 |
| 2020–21 | +0.0053 | +0.0077 (1.0) | +0.0020 (0.2) | −11.01 | −0.0% (0.77) | −0.1% (0.77) | −0.2% (0.77) | 0.91 | +0.2% | +44.3% | +46.3% | 0 |
| 2021–22 | +0.0219 | +0.0078 (0.9) | +0.0060 (0.6) | 0.93 | −21.3% (0.74) | −37.1% (0.75) | −67.9% (0.78) | 0.28 | −16.7% | −7.6% | −15.9% | 20 |
| 2022–23 | +0.0323 | +0.0075 (1.1) | +0.0221 (2.4) | 4.83 | −5.9% (0.66) | −21.8% (0.67) | −55.2% (0.69) | 0.91 | −3.1% | +14.1% | +25.3% | 18 |
| 2023–24 | +0.0201 | +0.0159 (2.5) | +0.0259 (3.5) | 4.95 | +11.2% (0.38) | −8.8% (0.38) | −49.8% (0.41) | 0.40 | +8.8% | +27.0% | +33.2% | 20 |

**Over the 14 years:**
- **Rank IC:** census units +0.0097 (z = 5.67, positive in 14 of 14 years); actual returns +0.0124 (z = 5.85, positive in 13).
- **Long-short:** 2.44 bps/day (z = 1.81).
- **Hit rate:** 50.0%.
- **The account:** mean yearly return +9.3%, compounded +214% (SPY +550%, QQQ +1,110%). At 5 bps that is −4.7% a year (−56% compounded); at 20 bps, −35.9% a year, about −100% compounded.
- **Validation predicts little:** the validation IC was positive in 13 of 14 years (2018's was zero). Across years its size barely predicts the test IC (correlation 0.15).
- **Combined z:** the 14 yearly t's (each year's mean daily IC ÷ its sd × √days) summed and divided by √14. Newey-West t's (which allow for correlation between neighboring days) with 1–10 lags change it by +1% to +11% (the lag-1 autocorrelation of the daily IC is −0.01), so autocorrelation doesn't inflate it.

### Variants: weekly, one-day delay, volatility-matched random accounts

Same scores, regraded:

| Variant | Random accounts | Years beating random (0 / 5 / 20 bps) | Picks, mean yearly (0 / 5 / 20 bps) | Random median (0 / 5 / 20 bps) |
|---|---|---|---|---|
| **Daily, delay 0** | uniform | 4 / 4 / 3 | +9.3% / −4.7% / −35.9% | +6.2% / −7.3% / −37.2% |
| Daily, delay 0 | volatility | 3 / 3 / 3 | +9.3% / −4.7% / −35.9% | +6.5% / −7.1% / −37.2% |
| Daily, delay 1 (trade a day later) | uniform | 2 / 2 / 1 | +9.9% / −4.2% / −35.5% | +6.4% / −7.0% / −36.9% |
| Daily, delay 1 | volatility | 2 / 2 / 2 | +9.9% / −4.2% / −35.5% | +7.5% / −6.2% / −36.7% |
| **Weekly, delay 0** | uniform | 4 / 4 / 4 | +11.4% / +8.3% / −0.7% | +6.5% / +3.5% / −4.8% |
| Weekly, delay 0 | volatility | 5 / 5 / 5 | +11.4% / +8.3% / −0.7% | +7.4% / +4.4% / −4.1% |
| Weekly, delay 1 | uniform | 3 / 3 / 3 | +8.7% / +5.6% / −3.0% | +6.4% / +3.4% / −4.9% |
| Weekly, delay 1 | volatility | 2 / 2 / 2 | +8.7% / +5.6% / −3.0% | +7.3% / +4.3% / −4.1% |
| Percentile ranks, daily (extra, always 20 names) | uniform | 3 / 3 / 3 | +14.1% / −2.3% / −38.2% | +13.3% / −2.9% / −38.4% |
| Percentile ranks, daily | volatility | 3 / 3 / 3 | +14.1% / −2.3% / −38.2% | +11.3% / −4.7% / −39.6% |
| Percentile ranks, daily, delay 1 | uniform | 1 / 1 / 1 | +12.1% / −4.0% / −39.0% | +12.9% / −3.2% / −38.5% |
| Percentile ranks, weekly | uniform | 4 / 4 / 4 | +14.0% / +10.4% / +0.2% | +13.3% / +9.7% / −0.4% |

**Reading the variants:**
- **Weekly:** the IC in actual returns holds (+0.0106, z = 2.43), but in census units it falls to +0.0029 (z = 0.98), and the long-short to 0.01 bps/day. A next-day model doesn't forecast 5-day moves. The weekly account's advantage is mostly cost: it trades about a fifth as often (587 names bought a year against 2,810).
- **Delay:** trading a day later cuts the years beating random from 4 to 2 (daily) and from 4 to 3 (weekly).
- **Volatility matching:** the random accounts then hold names as calm as the picks (decile 1.9 instead of 4.5), but their median moves by only −0.1 to +1.1 points (+0.4 to +1.1 at 0 bps; −2.0 to −1.2 for the percentile variant), and the years beating random change by at most one. Single years can move a lot (2012: p 0.52 to 0.07).
  - The match is exact on the buy day's volatility decile but coarse within it: the matched accounts' raw volatility is within about 5% of the picks'.
  - While names are held, the matched accounts' decile drifts up by as much as 0.25 (daily) and 0.36 (weekly), because the picks stay calm while their random stand-ins drift toward the middle.
- **Uniform p is slightly conservative at 5 and 20 bps:** a uniform random account can buy back a name it sold the same day, so it trades about 1% less than the strategy and pays slightly less cost.

### Seeds and reproducibility

| Seed | IC, actual (z) | IC, census (z) | Long-short bps/day (z) | Years beating random (0 / 5 / 20 bps) | Picks, mean yearly | Years in cash, or with fewer names |
|---|---|---|---|---|---|---|
| 0 | +0.0122 (6.10) | +0.0105 (6.34) | 3.08 (2.30) | 4 / 4 / 4 | +14.2% / −1.0% / −34.4% | cash 2020; 15–19 names in 2011, 2016, 2022 |
| 1 | +0.0111 (5.22) | +0.0073 (4.09) | 2.47 (1.59) | 2 / 2 / 2 | +11.1% / −2.3% / −32.0% | cash 2018; about 3 names in 2010 and 2020; 18 names in 2011 |
| 2 | +0.0116 (5.46) | +0.0083 (4.88) | 2.10 (1.69) | 4 / 3 / 3 | +10.9% / −4.3% / −37.9% | 10–19 names in 2010, 2016, 2020, 2022, 2023 |
| Ensemble | +0.0124 (5.85) | +0.0097 (5.67) | 2.44 (1.81) | 4 / 4 / 3 | +9.3% / −4.7% / −35.9% | near-cash 2020 (8 name-days); 18–19 names in 2010, 2011, 2016, 2022 |

**Seeds:** the three agree on the ranking. Their daily rank correlations with the ensemble are 0.93, 0.87 and 0.90. The lowest is seed 1 in 2018 (0.22), the year its model's scores were all negative; next come seeds 1 and 2 in 2010 and seed 2 in 2023 (0.62–0.72).

**Reproducibility:** training uses deterministic CUDA kernels. An earlier, interrupted run of this same step had finished seeds 0 and 1 and seed 2's 2010; this run reproduced every one of those score files bit for bit. So did the null, LightGBM and the rule baselines:
- 12,663,511 scores per seed;
- the reruns had different GPU contention;
- this comparison is from the builder's report; no hash log was kept.

During verification a fresh quick run (`scripts/poc.sh --quick`, from empty folders, on a free GPU) reproduced all 17 score files it shares with the earlier quick run byte for byte (equal sha256). Its shared judge outputs are identical apart from the folder name in their path fields, and its G1 result is identical.

### What the model learned (inputs only)

Mean daily rank correlation of the ensemble's scores over the test days (`results/poc/diagnostics_gpt_ensemble.json`):

| Today's move v[d] | Log volatility (252 days) | −v[d] rule | 5-day reversal | LightGBM | 12-1 momentum |
|---|---|---|---|---|---|
| −0.31 (years: −0.04 to −0.70) | −0.47 (−0.93 to +0.10) | +0.31 | +0.08 | +0.12 | +0.07 |

**The mix changes from year to year:**
- 2016's model is almost pure low volatility (−0.93), and 2021's is almost pure one-day reversal (−0.70 with v[d], +0.10 with volatility).
- Reversal is what every published model of this kind found, so PLAN §4.3's "reversal rediscovery" check holds.
- **Validation cross-entropy:** +6.6 to +177 millibits per prediction better than the training histogram. 2010's +156 to +177 is the regime shift after 2008, as the builder found. The sign-flip null gains about as much (+23 to +37), so the cross-entropy gain is mostly about the size of moves, not their direction.

## Baselines

Same judge, same test years 2010–2023 (the rules have scores from 2007, but every comparison here uses the GPT's 14 years). All are point in time (apart from the export's two exceptions, `docs/c2-export.md`), and all write NaN from the locked year's first decision.

| Model | Rebalance | Rank IC, actual returns (z; years > 0) | Rank IC, census units (z) | Long-short bps/day (z) | Years beating random (0 / 5 / 20 bps), uniform | Volatility-matched | Picks, mean yearly (0 / 5 / 20 bps) | Random median | Picks' vol decile |
|---|---|---|---|---|---|---|---|---|---|
| **stockGPT ensemble** | daily | +0.0124 (5.85; 13) | +0.0097 (5.67) | 2.44 (1.81) | 4 / 4 / 3 | 3 / 3 / 3 | +9.3% / −4.7% / −35.9% | +6.2% / −7.3% / −37.2% | 1.9 |
| stockGPT ensemble | weekly | +0.0106 (2.43; 10) | +0.0029 (0.98) | 0.01 (0.33) | 4 / 4 / 4 | 5 / 5 / 5 | +11.4% / +8.3% / −0.7% | +6.5% / +3.5% / −4.8% | 1.9 |
| −v[d] (1-day census reversal, extra) | daily | +0.0118 (5.70; 11) | +0.0126 (6.08) | 1.92 (1.90) | 2 / 2 / 2 | 2 / 2 / 2 | +6.2% / −16.6% / −59.6% | +13.2% / −10.7% / −56.2% | 5.0 |
| **5-day reversal** | weekly | +0.0145 (3.05; 11) | +0.0335 (8.07) | 3.63 (2.46) | 6 / 6 / 6 | 7 / 7 / 7 | +27.9% / +22.0% / +5.8% | +13.4% / +8.2% / −6.0% | 7.9 |
| 5-day reversal (extra) | daily | +0.0088 (3.67; 13) | +0.0168 (8.09) | 5.37 (2.57) | 10 / 10 / 9 | – | +28.0% / +14.8% / −17.1% | +13.3% / +1.8% / −26.2% | 7.9 |
| **LightGBM** | daily | −0.0022 (−0.77; 5) | +0.0556 (23.97) | 5.49 (2.99) | 8 / 8 / 8 | 5 / 5 / 5 | +27.8% / +10.2% / −29.2% | +13.4% / −2.0% / −36.6% | 6.0 |
| LightGBM | weekly | −0.0037 (−0.78; 7) | +0.1140 (22.28) | 1.03 (0.32) | 6 / 6 / 6 | 6 / 6 / 6 | +17.8% / +13.5% / +1.7% | +13.4% / +9.4% / −1.9% | 5.9 |
| **LightGBM on actual returns** (extra, added during verification) | daily | +0.0210 (9.15; 14) | +0.0012 (0.37) | 10.24 (6.10) | 10 / 10 / 10 | 11 / 11 / 11 | +49.2% / +20.9% / −35.7% | +13.4% / −7.8% / −50.5% | 5.7 |
| LightGBM on actual returns (extra) | weekly | +0.0145 (2.71; 9) | −0.0228 (−5.31) | 3.79 (2.32) | 6 / 6 / 6 | 6 / 6 / 6 | +26.3% / +20.5% / +4.6% | +13.4% / +8.2% / −5.8% | 5.7 |
| 12-1 momentum | weekly | +0.0171 (2.53; 12) | −0.1135 (−19.76) | 1.46 (1.14) | 5 / 5 / 5 | – | +18.7% / +17.6% / +14.4% | +13.2% / +12.2% / +9.1% | 7.7 |
| 12-1 momentum | monthly | +0.0174 (1.38; 12) | −0.2217 (−22.37) | 1.39 (0.98) | 5 / 5 / 5 | – | +17.7% / +17.2% / +15.6% | +13.2% / +12.7% / +11.2% | 7.7 |

**SPY:** +15.0% a year over these 14 years, +550% compounded. **QQQ:** +20.6% a year, +1,110%.

**Picks' vol decile:** the mean trailing 63-day volatility decile of the names held (0 = calmest tenth of the day's candidates, 9 = most volatile; random accounts 4.5). stockGPT's 1.9 counts the near-cash 2020 as decile 0 from 8 name-days; over the 13 invested years it is 2.0.

**The 5-day reversal rule:**
- **Compounded:** +1,315% at 0 bps, +628% at 5 bps and −1% at 20 bps (weekly), against +330%, +187% and −15% for the GPT, weekly.
- **Fading:** in five years from 2011 to 2020 no random account did as well, and it lost to the random median in each of the last three years, as correlation2 found.

**LightGBM.** It is trained on the same next-day v as the GPT, and early-stopped on validation L2. A few observations:
- **It has learned the census-unit artifact:**
  - Its validation IC averaged +0.058 against v but +0.001 against actual returns.
  - Its biggest feature by gain is the 252-day return in 9 of 14 years.
  - Its scores correlate −0.68 with 12-1 momentum.
- **Even so, its top 20 do well:**
  - Mean yearly +27.8% at 0 bps, beating random in 8 of 14 years, and 5 when the random accounts match its volatile picks (decile 6.0).
  - A long-short of 5.49 bps/day (z = 2.99), next to an overall IC that is negative.
  - Its scores lean against last year's winners (−0.68 with 12-1 momentum) and toward this week's losers (+0.21 with the 5-day rule).
- **Its near-zero IC in actual returns is its k loading cancelling everything else** (a daily rank regression run during verification, using k, the part of v[d+1] known at d; next section): its −0.0022 is −0.0117 from its loading on k (correlation +0.77; k's own IC in actual returns is −0.0145), +0.0033 from its volatility tilt, and +0.0062 (z 8.6) from the rest of its ranking. So the part of its ranking unrelated to k and volatility does carry a real signal.

**LightGBM on actual returns (extra, added during verification).** The same LightGBM, features and early stopping, trained instead on the next day's actual return, turned into a daily rank score (`--target ret`: each day's training rows are ranked and mapped onto a bell curve). It was added after the POC results were seen, so it is reported as an extra beside the planned baseline, not instead of it.
- **The strongest gross signal in the POC:** IC in actual returns +0.0210 (z = 9.15), positive in all 14 years, and positive in validation in all 14 years too (mean +0.0223). Long-short 10.24 bps/day (z = 6.10).
- **Daily money:** it beat random in 10 / 10 / 10 years (11 / 11 / 11 volatility-matched), with +49.2% / +20.9% / −35.7% a year. At 20 bps, "beating random" means losing 35.7% a year against random's 50.5%, because it replaces about 84% of its names a day (4,207 buys a year). The compounded +14,391% at 0 bps comes mostly from 2019–20 (+103.7%) and 2020–21 (+196.1%); the mean yearly return is the fairer summary.
- **A day late, most of it is gone:** 4 / 4 / 4 years, +21.5% / −1.4% / −47.4% a year. Its IC over all scored stocks decays smoothly: +0.033 for the next day, +0.021 for the day after, +0.016 five days out. The part specific to the next day (0.012) is smaller than the −v[d] rule's (0.029), so this is the normal decay of a reversal-like signal, not a spike at d+1.
- **Weekly:** 6 / 6 / 6 years beating random (6 / 6 / 6 volatility-matched), +26.3% / +20.5% / +4.6% a year, level with the 5-day rule's +27.9% / +22.0% / +5.8%. A day late: 7 / 7 / 7 (6 / 6 / 6 volatility-matched), +25.0% / +19.3% / +3.7%. Unlike the 5-day rule, it didn't fade at the end: it beat random in 2021 and 2023 (p 0.03 and 0.06), though not in 2022 (+2.5% against a random median of +7.7%, p 0.72), while the 5-day rule lost to the random median in each of the last three years. Against volatility-matched accounts the contrast shrinks: it beat random only in 2021 (p 0.02), and the 5-day rule was at the matched median in 2021 and 2023.
- **What it resembles** (`results/poc/diagnostics_lgbm_ret.json`): +0.35 with stockGPT, −0.36 with volatility, +0.23 with 12-1 momentum, −0.24 with today's move, +0.09 with the v-target LightGBM. Its scores as a whole lean calm, but its top 20 lean slightly volatile (decile 5.7, against 4.5 for random accounts).
- **Census units misjudge it the other way:** its census IC is +0.0012 (z 0.37) daily and −0.0228 (z −5.31) weekly. It leans toward last year's winners (−0.30 with k, negative in all 14 years), and census units mark that lean down: in the split used above for the v-trained LightGBM, k takes −0.0117 off its daily census IC and −0.0308 off its weekly one. Against v[d+1] − k[d], the census outcome without its known part, it scores +0.0235 daily (z 11.9, positive in 14 of 14 years) and +0.025 weekly (z 5.5), close to its daily IC in actual returns (+0.0210) and above its weekly one (+0.0145) (measured during verification).
- **Its large-cap tripwire fired on 2021; a leak check found no leak** ("Leakage tripwire" below).

### Census-unit IC is not all signal

correlation2's census variable is

v[d] = (residual[d] − mean of residual[d−252 .. d−1]) ÷ sd of residual[d−252 .. d−1], clipped at ±5,

where the residual is the stock's log return minus beta × the market's. The window ends the day before, so at decision day d the mean that will be subtracted from tomorrow's residual is already known.

**What follows:**
- **A free component:** minus the trailing-year mean residual (in sd units) is a component of v[d+1] that is fully predictable, and it isn't a return anyone earns.
- **How large:** for a stock whose true mean residual is steady, that component has a cross-sectional spread of about 1/√252 ≈ 0.06 of v's own. So a score that knows it reaches a census IC near 0.06 at one day, 0.14 at five days and 0.29 at 21: its share grows with √(holding days).
- **Measured directly** (during verification): call the component k. k[d] = −(mean ÷ sd) of the residuals of days d−251..d, all known at the close of d. It has a cross-sectional spread of 0.061. As a score on the judge's universe over 2010–2023, it gets census ICs of +0.055 at one day (z 20.4, positive in all 14 years), +0.127 at five and +0.244 at 21. That is a little under 0.063, 0.141 and 0.289, because the trailing mean also carries a real effect of the opposite sign: k's IC in actual returns is −0.0145 (z −4.7).
- **The evidence:**
  - LightGBM's census IC: +0.056 daily and +0.114 weekly, with about zero in actual returns.
  - 12-1 momentum: −0.11 weekly and −0.22 monthly in census units, with +0.017 in actual returns.
  - LightGBM's scores correlate +0.77 with k, and 12-1 momentum's −0.89. With k[d] removed from the outcome (v[d+1] − k[d]), LightGBM's daily census IC goes from +0.0556 to −0.0007 (actual returns: −0.0022), and momentum's weekly one from −0.113 to +0.019 (actual returns: +0.017).
  - correlation2's own 21-day reversal control (census z = 13.04, long-short z = 1.15) is the same thing: over 2010–2023 its census IC is +0.088 (z 12.6), of which k accounts for +0.071; against v[d+1] − k[d] it is +0.011 (z 1.3).

**Consequences:**
- stockGPT doesn't exploit it. Its scores correlate −0.10 with k (seeds −0.09 to −0.11), which is what its low-volatility tilt implies; given −log volatility, its weight on k is +0.007, and k accounts for +0.0005 of its +0.0097 census IC. The artifact works against it: against v[d+1] − k[d] its IC is +0.0176 (z 9.8). It isn't that the trailing mean is out of reach (the mean of the last 252 values in its context correlates −0.56 with k); the model just doesn't use it.
- LightGBM on actual returns isn't rewarded for it in training, and its lean toward last year's winners lowers its census IC (above). Against v[d+1] − k[d] the order of the two LightGBMs reverses: +0.0235 for the one trained on actual returns, −0.0007 for the one trained on v, as in actual returns.
- Census-unit IC stays in the tables, but G1 and every comparison here use actual returns. Because k[d] is known at decision time, v[d+1] − k[d] (the census outcome minus its known part) would make a cleaner census-unit outcome (not built here).
- `docs/c2-export.md` now gives the exact definition.

## Controls

All graded by the judge, daily, delay 0, uniform random accounts, over the same 14 years unless noted.

| Control | IC, census units (z) | IC, actual returns (z) | Long-short bps/day (z) |
|---|---|---|---|
| stockGPT ensemble | +0.0097 (5.67) | +0.0124 (5.85) | 2.44 (1.81) |
| Its scores used a day late (signal decay) | +0.0024 (1.75) | +0.0061 (2.87) | −0.21 (0.04) |
| **Canary: scores 20 days stale** | +0.0041 (**2.38**) | +0.0067 (**3.02**) | −0.31 (−0.38) |
| **Canary: dates permuted within each year** (seeded) | +0.0040 (**3.00**) | +0.0067 (**3.22**) | −0.17 (0.39) |
| Canary: scores a year (250 days) stale (2011–2023) | +0.0012 (0.44) | +0.0039 (1.59) | −0.82 (−1.03) |
| Static reference: −log volatility (252 days) | +0.0019 (0.62) | +0.0128 (2.88) | −2.01 (−0.89) |
| −v[d] (a pure one-day timing signal) | +0.0126 (6.08) | +0.0118 (5.70) | 1.92 (1.90) |
| Its canary: 20 days stale | +0.0025 (1.27) | +0.0025 (1.21) | 2.84 (2.13) |
| Its canary: dates permuted | −0.0025 (−0.39) | −0.0011 (0.09) | −1.70 (−0.57) |
| **Sign-flip null** (seed 0; 2012, 2017, 2022) | +0.0012 (0.21) | +0.0008 (0.08) | 0.33 (−0.26) |

### Score canaries (PLAN M1's gate)

**They did not drop to about 0.** They are designed to remove timing information, and they did:
- **Timing:** the census-unit IC falls from +0.0097 to +0.0024 with a one-day delay.
- **What stays:** a slower part, about +0.004 in census units and +0.007 in actual returns, survives a 20-day staleness and a within-year shuffle alike. It fades within a year (250 days stale: z = 0.44 and 1.59).
- **About half of it is the low-volatility tilt; the other half is a slower property orthogonal to both volatility and the census artifact** (measured during verification).
  - On the judge's universe the scores correlate −0.34 with log volatility (−0.47 is over every scored stock).
  - In a daily rank regression of the 20-day-stale canary on −log volatility and k (the part of v[d+1] known at d), its actual-return IC of +0.0067 splits into +0.0034 from the tilt, +0.0003 from k and +0.0030 (z 3.0) orthogonal to both. The permuted canary: +0.0032, +0.0002 and +0.0033 (z 4.0).
  - The orthogonal part is gone when the scores are a year stale (−0.0002); the tilt is not (+0.0048).
- **In census units the tilt seems to explain nothing (−log volatility alone gets z = 0.62), but that is the census artifact.** Low-volatility stocks have higher trailing mean residuals (−log volatility correlates −0.29 with k), so v[d+1] subtracts more from them. Against v[d+1] − k[d], the census outcome without its known part, −log volatility earns +0.0235 (z 7.8), and the canary's +0.0120 splits into +0.0087 from the tilt and the same orthogonal +0.0030 (z 3.0) as in actual returns.
  - So the census-unit residual is not a census-only property, and k doesn't carry it (k's share of the canary's census IC is +0.0005).
  - What the orthogonal part is remains open: no single slow trait tried explains it (from the v series: trailing means, the volatility of v, skewness, the share of tail days; from prices and trading: unchanged closes, idiosyncratic volatility, price, dollar volume).

**Why this is not a leak:**
- **The stale canary is point in time:** a score computed with data through d−20 can't contain day d+1.
- **The permuted canary keeps no more than the stale one.** Permuting does put later days' scores, whose 256-day context can include the outcome day, onto earlier days; that is a limit of this canary for long-context models, but it adds nothing here.
- **The canaries themselves work, in IC.** The −v[d] rule's canaries fall to about 0 (z −0.39 to 1.27).
- **Only in IC:** the 20-day-stale −v[d] keeps a long-short of 2.84 bps/day (z = 2.13) and beat random in 6 / 6 / 5 years (+21.8% a year against +13.1% random), more than the live rule. A 20-day-stale score can't contain day d+1, so this is a slower effect at a 20-day lag, not a leak; it is unexplained here.

**So:** the canaries confirm the timing part is real and fades within days. They can't remove a real, persistent property, and this model has two: a low-volatility tilt and a slower part not yet identified. M1's gate ("the canaries collapse to chance") is not met as written.

### Leakage tripwire

Each test day: the 500 stocks with the largest mean dollar volume over the previous 63 days (point in time) among those with a score, ranked against the next day's actual return. A mean above 0.05 counts as a bug until proven otherwise.

| Scores | Mean daily rank IC | t | Largest yearly mean | Tripped |
|---|---|---|---|---|
| stockGPT ensemble | +0.0129 | 4.82 | +0.0293 (2023) | no |
| sign-flip null | +0.0012 | 0.31 | +0.0084 | no |
| LightGBM | −0.0009 | −0.27 | +0.0156 | no |
| 5-day reversal | +0.0108 | 3.52 | +0.0298 | no |
| −v[d] | +0.0149 | 5.76 | +0.0362 | no |
| LightGBM on actual returns (extra) | +0.0249 | 9.00 | +0.0503 (2021) | **yes: 2021** (not a leak, below) |

PLAN's cross-entropy half of the tripwire ("or a large drop in cross-entropy") is not automated. Read from `train_log.jsonl`, stockGPT's validation cross-entropy gain was 6.6–177 millibits per prediction (the large 2010 values come from the post-2008 regime shift), and the null gained about as much in its years (23–37).

**LightGBM on actual returns tripped on 2021, and it is not a leak.** PLAN counts a trip as a bug until shown otherwise, so during verification a separate agent tried to prove a leak, and couldn't. G1 is unaffected: it reads only stockGPT's tripwire.
- **Every stored score at day d depends only on data through d.**
  - Refitting reproduced all 14 years bit for bit (12,554,720 scores).
  - Rescoring 46 days (each year's top large-cap day and 2 random days; for 2021, the top 5) from copies of the export garbled from d+1 on gave the same 168,633 scores bit for bit, although day d+1 had changed.
  - Each year's model, retrained from data garbled after its last validation outcome, came out bit-identical in all 14 years.
- **Once its volatility tilt is set aside, 2021 looks like the other years.** Its large-cap IC of +0.0503 splits into +0.0093 from a steady lean toward calm stocks, +0.0186 from that lean shifting with the volatility factor's daily payoff, and +0.0224 from the rest; the rest averages +0.0203 in the other 13 years.
  - The shifting lean comes from day-d market inputs: 74% of it is linear in the past market moves the model sees, and refit with the market features held constant, the model's 2021 large-cap IC drops to +0.0283.
  - 2021 is the one year in 21 (2003–2023) where the −log volatility score alone exceeds 0.05 on large caps (+0.0566).
- **No market timing:** the market series the model reads is the day-d average of correlation2's liquid stocks (recomputed independently, it matches to 4.5e-10, and it correlates −0.05 with the next day's average). Across all years, the model's daily tilt correlates +0.015 with the next day's market move; 2021's −0.130 is the lowest of the 14 years, but level with 2020's −0.127 and 2013's −0.126 (the highest is 2019's +0.18).
- **A sign-flip null** (the target multiplied by a random ±1 per day, the same for every stock; only the target is flipped, where stockGPT's null flips its inputs too; 2 seeds, test years 2010–2022) didn't trip (large-cap means −0.0051 and −0.0005, no year above +0.016) and found no IC in actual returns (z −1.94 and −0.43). In census units it read z −2.21 and −5.23, so by the rule G1 applies to stockGPT's null (\|z\| < 2 for both ICs) it would not count as clean. Its accounts beat uniform random accounts in 4 / 4 / 3 and 2 / 2 / 2 of its 13 years, about as often as stockGPT's 4 / 4 / 3 of 14. Measured during verification, neither number points to a leak, which would show up as skill in actual returns:
  - Each seed's 13 yearly models share one random sign pattern, and each year's training days are 81–93% those of the year before, so they share accidental leans, and the combined z, which treats the years as independent, overstates how unusual those leans are.
  - Seed 1 leans slightly toward last year's winners (−0.12 with k, negative in all 13 years), which census units mark down: k accounts for −2.8 to −4.0 of its −5.23, depending on the split, and against v[d+1] − k[d] it reads z −0.8.
  - Seed 0 has no consistent lean on k (+0.02). Against v[d+1] − k[d] it reads z −2.8, of which a lean toward volatile stocks (11 of 13 years) accounts for −1.7.
  - Both often give tied scores (4–10 distinct values a day in several years), and the judge orders ties by column, which is ticker order. Ticker order on its own scores z −3.2 against v[d+1] − k[d] (−0.5 in census units, −1.3 in actual returns), so a fixed accidental ordering can reach \|z\| ≈ 3 against that outcome; how much of the nulls' readings the tie order accounts for wasn't measured.
  - This kind of null is a weak test for a tree model; the bit-for-bit tests above are the real check.
- **2021 is barely over the line:** 0.0503 against 0.05, well inside its own noise (95% interval 0.024 to 0.077). Given this model's own year-to-year spread, some year out of 14 crossing 0.05 with no leak is about 10–30% likely.
- **A better rule (not changed here):** apply the 0.05 limit to the all-years mean (+0.0249 for this model), or to each year's IC after removing volatility and beta.

### Sign-flip null

v was multiplied by a random ±1 per date, the same for every stock, in both inputs and targets (`--null sign-flip`, seed 0). It was trained exactly like the real model for test years 2012, 2017 and 2022.

| Test year | Val IC against its own (flipped) target | Test IC, census (t) | Test IC, actual (t) | Long-short bps/day |
|---|---|---|---|---|
| 2012–13 | −0.0038 | −0.0067 (−1.5) | −0.0040 (−0.8) | −6.18 |
| 2017–18 | −0.0002 | +0.0055 (0.8) | +0.0072 (1.1) | 3.36 |
| 2022–23 | −0.0028 | +0.0047 (1.0) | −0.0009 (−0.2) | 3.82 |

Combined: z = 0.21 (census) and 0.08 (actual returns). It beat random in 0 of 3 years. Nothing.

**Also verified by the builders' tests, on synthetic data:**
- the model finds a planted reversal (IC +0.086 against a ceiling near 0.095, the builder's measurement; the test itself asserts an IC above 0.05);
- the scores don't change when any data after the decision day is altered;
- training batches never touch a day outside the split.

## Gate G1 (PLAN §7)

Computed by `stockai.poc.summarize` on the ensemble, daily, delay 0, uniform random accounts:

| Check | Value | Passed |
|---|---|---|
| Gross signal: rank IC in actual returns, combined z ≥ 2 | z = 5.85 (mean +0.0124, positive in 13 of 14 years); census units z = 5.67 | yes |
| Nothing on the null: combined \|z\| < 2 for both ICs | actual returns z = 0.08; census units z = 0.21 (3 years) | yes |
| No tripwire: large-cap daily rank IC ≤ 0.05 | +0.0129 (largest year +0.0293) | yes |
| **G1** | | **passed** |

**The comparison PLAN asks for:**
- stockGPT's gross signal is as strong as one-day reversal's (−v[d]: z = 5.70), and stronger than the 5-day rule's in IC z (5.85 against 3.05 weekly and 3.67 daily; the weekly rule's mean IC, +0.0145, is the larger).
- It is weaker than the 5-day reversal rule (daily and weekly) on every money measure, and weaker than LightGBM's top 20. It does better than the one-day −v[d] rule on money: 4 / 4 / 3 years beating random against 2 / 2 / 2, and +9.3% / −4.7% / −35.9% a year against +6.2% / −16.6% / −59.6%.
- LightGBM on actual returns (extra, added during verification) is stronger on both counts: IC z = 9.15 against 5.85, and 10 / 10 / 10 years beating random against 4 / 4 / 3. A day late, though, its daily account beats random in only 4 / 4 / 4 years.
- PLAN's note on G1 ("weaker but real signal is a finding, not a failure") is where this lands.

## Timings

- **Full run:** 3,643 s (61 min), before the LightGBM extra was added (its training takes 116 s, its gradings a few seconds each).

  | Step | Time |
  |---|---|
  | G0 | 10 s |
  | stockGPT walk-forward (42 models) | 3,229 s (54 min) |
  | Sign-flip null (3 models) | 188 s |
  | Baselines | 106 s (LightGBM 105 s for 14 years at 8 threads) |
  | Grading (27 files) | 42 s |
  | Controls (6 more graded files, tripwire, diagnostics) | 68 s |

  Another session's census runs used the GPU during parts of seed 0. (The step table in `summary.md` now shows the final regrade's grading and controls times, 61 s and 86 s, next to the full run's other steps.)
- **Regrade with the final script** (`--steps grade,controls,summary`, 46 files including the LightGBM extra): 147 s. An earlier regrade of 38 files took 135 s.
- **LightGBM on actual returns** (extra, run on its own during verification): 116 s for 14 years at 8 threads.
- **stockGPT per seed:** 20.6, 16.2 and 16.8 min. Per year (mean of seeds), training + scoring:

  | Year | Train | Score |
  |---|---|---|
  | 2010 (1.8M training positions, 400 steps) | 42 s | 13 s |
  | 2023 (12.2M positions, 776-step budget) | 83 s | 34 s |

  About 80% of training time is the 6 validation passes.
- **The judge:** about 1–3 s per walk-forward with 1,000 random accounts at 3 costs (the printed tables' timings, rounded to whole seconds).
- **Quick run:** 401 s (6 min 40 s) and 6.3 GiB peak memory (maximum resident set, from `/usr/bin/time`), with the final script, from empty `results/poc-quick-new/` and `runs/poc-quick-new/`, on a free GPU. An earlier quick run, before the LightGBM extra and with another session's job on the GPU, took 633 s. Both give G1 passed on 2022–2023 with one seed: IC in actual returns +0.0220, z = 4.44.
- **Tests:** 77 in 36 s (`uv run pytest -q`, rerun during verification).

## Changes made while running the POC

- **The earlier run.** An earlier run of this step wrote `scripts/poc.sh`, `stockai/poc/` (G0 checks, controls, diagnostics, summary), the extra −v[d] baseline (`stockai/baselines/vreversal.py`) and their tests, then stopped partway through the GPT's seed 2. I reviewed that code, kept it, and reran everything from scratch. Its finished files matched bit for bit.
- **`scripts/poc.sh`:**
  - The body now runs inside `main()`, so editing the script during an hour-long run can't change what the run executes.
  - Added the percentile variant (extra), the year-stale canary, momentum in the diagnostics, and a LightGBM diagnostic.
- **`stockai/poc/controls.py`:**
  - The tripwire picks its 500 stocks from what is known at d (score, trailing dollar volume), then drops the ones without a next-day return. Before, it selected among stocks with a known outcome.
  - Added `percentile`.
- **`stockai/poc/summarize.py`:** G1 is now computed, not eyeballed. Added canary readings, per-year tables for LightGBM and the 5-day rule, the picks' volatility decile and names held, timings, and a check that the percentile variant's rank measures are identical.
- **`docs/c2-export.md`:** the census variable's definition now includes the trailing-mean subtraction, and the command names are the real ones (`stockai-judge export`, `stockai-judge holdout`).
- **Tests:** one new test for `percentile`.
- **Nothing else during the POC run:** nothing in `stockai/gpt/`, `stockai/baselines/` (other than the earlier run's vreversal), `stockai/data.py`, `splits.py`, `synthetic.py` or `judge/` was changed. Nothing in correlation2 was touched. Nothing was committed. (What changed during verification is in the next section.)

**Configurations tried (PLAN §4.4):**
- **stockGPT v1 — one configuration, 42 fits here plus 3 null fits.** The builder made its choices on validation data only: 5 configurations and 10 fits, which set the 400-step floor (from the builder's report; not recorded in the repo).
- **LightGBM — two configurations, 14 fits each:** the planned target v[d+1], and the next-day return (`--target ret`), added during verification after the POC results were seen. The second is reported as an extra beside the first, not instead of it.
- **Rules:** no fitting.
- **Graded:** 46 files in `results/poc/`, all listed in the ledger at the end of `summary.md` (38 from the POC run, 8 from the LightGBM extra). The percentile variant was added after the cash year was seen; it is reported beside the primary result, not instead of it.

## Changes made during verification

After the POC run, independent agents checked it. The three verifiers only reported; they changed nothing. The changes below were made afterwards, on purpose, and the outputs were regraded. The verifiers' scripts and their JSON and log outputs are kept in `runs/poc/verification/` (not in git); their large intermediate files (rebuilt scores, k, saved models) were not kept.

**What was checked, and by whom:**
- **The judge** (one agent): re-implemented in Python, from the Rust source, the universe, the account and its costs, the delay, the uniform and volatility-matched random accounts, the hit rate, the long-short check and the locked-year guard, and compared them with the judge's output for selected years from 2016 to 2023, daily and weekly, at delay 0 and 1, with both kinds of random account. Every number compared matched bit for bit. It confirmed that `--delay 1` holds picks decided at d from the close of d+1 (earning d+1 to d+2), and reran the judge's 16 Rust tests (all pass). One day it couldn't check under the locked-year rule: the 252nd day of the 2023–24 row, which earns the return into the 2024-06-27 close.
- **The data** (one agent): rebuilt the market series and v from the export's closes and dollar volumes with correlation2's rules; all 15,169,056 values of v match bit for bit. It also replicated the judge's IC code, measured k (the part of v[d+1] known at d) and decomposed the canaries.
- **The claims** (one agent): regenerated every table in this file as it stood before the LightGBM extra (38 graded files) from the judge JSONs and training logs, and recomputed every 14-year total. A final recheck traced the rewritten text, the LightGBM extra's rows included, to the raw outputs.
- **The LightGBM extra:** one agent built it; another tried to prove it leaks ("Leakage tripwire"). A third rebuilt k and the null's scores bit for bit, matched the judge's ICs to 1e-14, and split their census-unit ICs into k's share and the rest.
- **A fresh quick run** from empty folders with the final scripts (`results/poc-quick-new/`, `runs/poc-quick-new/`): all 17 score files it shares with the earlier quick run are byte-identical, the shared judge outputs are identical apart from the folder name in their path fields, and G1 is identical. The comparison, with both runs' sha256s, is in `runs/poc/verification/finish/quick_rerun/compare_report.json`; the earlier run's folders were deleted afterwards.

**What they found** (fixed in this file unless noted):
- **Readings that were wrong or overstated:** the canaries' persistent part is about half volatility tilt, not nearly all; the census-unit residual is the census artifact, not a census-only property; LightGBM's near-zero IC is two parts cancelling, not "noise in the middle"; stockGPT beats the −v[d] rule on money; the 2.4 × c cost belongs to the long-short check, not the account (which pays about 1.2 × c); 2020–21 was near-cash, not all cash; the volatility-matched median moves −0.1 to +1.1 points, not 0.3 to 0.9; the picks' decile is 2.0 over the invested years, not "the calmest fifth"; the stale −v[d] canary keeps its money result; the tripwire's cross-entropy half isn't automated.
- **Context that was missing:** what "chance gives about 1.4" means, what combined z is, and the volatility match's coarseness.
- **Two look-aheads in correlation2's data:** the bad-print repair and the split-adjusted $5 test ("Notes for correlation2", `docs/c2-export.md`).
- **Numbers with no recorded output behind them:** now labelled as the builder's, or replaced with measurements (machine, test count and time, quick-run time and memory).
- **The export's `meta.json`** described v without the trailing-mean subtraction.

**What changed:**
- `stockai/baselines/lgbm.py`: a `--target ret` option. The default (`v`) is unchanged: a rerun of 2023 and the fresh quick run's 2022–2023 match the earlier scores bit for bit, and the existing LightGBM tests pass unmodified. 11 new tests in `tests/test_baselines_lgbm_ret.py` (77 tests in all). `scripts/poc.sh` now trains, grades (8 ways), tripwires and diagnoses it, and `stockai/poc/summarize.py` adds its rows; without its files the summary is byte-identical to before. Scores: `runs/poc/baselines/lgbm_ret/`.
- `judge/src/export.rs`: the `variable` text written into `meta.json` now states the trailing-mean subtraction, the adjusted closes and the bad-print repair's look-ahead. The export was rewritten; its data files are bitwise identical and its fingerprint is unchanged (8b5ac076267c83ab).
- `results/poc/` was regraded with the final script: 46 files, 147 s, ending `EXIT 0` (`runs/poc-regrade2.console.log`).
- Docs: this file, `README.md`, `PLAN.md` (status, decisions, §5.2), `docs/c2-export.md` and the `stockai/data.py` docstring.
- Nothing in correlation2 was touched, nothing was committed, and nothing graded, trained on or looked at the locked year.

## Not done

- **The locked year and the sealed exam (M7).** By design.
- **M1 judge upgrades not built:**
  - per-stock trading costs (EDGE spreads) and the 10 bps level;
  - results value-weighted and by size bucket;
  - a ledger with a deflated Sharpe ratio. (The volatility-matched accounts, the delay option and IC in actual returns were built.)
- **Baselines not run:** 21-day reversal; the zero-forecast and bin-histogram baselines, which give no ranking to grade; CatBoost.
- **LightGBM on v[d+1] − k[d],** the census target without its known part. (LightGBM on actual returns was done during verification.)
- **A tripwire rule that allows for factor tilts** (the 2021 trip), and an explanation of the canaries' orthogonal part and of the stale −v[d] canary's money result.
- **Ties in the judge:** tied scores are ordered by column, which is ticker order, in the rank IC and in the picks. That can matter only for scores with few distinct values, such as the LightGBM nulls (4–10 a day in some years) and the actual-return LightGBM's 2017 model (73 a day); its effect wasn't measured. Average ranks would remove it from the IC. The judge's rank code is kept identical to correlation2's holdout (G0), so it is not changed here.
- **Calibration (PLAN §4.2).**
- **The null on more seeds and years.**
- **`torch.compile`:** not used. The check exists and passes, but eager was fast enough.
- **No tuning of stockGPT on test years, and no search over sizes or contexts.**
- **Not started:** v2 (5- and 21-day targets, turnover-limited picks), v3 and the M3 outside models (Kronos, FinText, Chronos-2, TimesFM).
- **Survivorship:** Stooq's data has survivors only (PLAN §5.2): no stock in the export has its last close before 2024-03-01. That bias affects the picks and the random accounts alike.

## Notes for correlation2

Three findings from this verification apply to correlation2's own holdout at commit 9335126. correlation2 itself was not changed.
1. **The $5 universe test looks ahead.** Stooq's closes are split- and dividend-adjusted as of the download (data through 2025-06-27), so the median-price test, for the market's liquid set and for the holdout universe, uses price levels that reflect later splits, including splits in the locked year (AVGO's and LRCX's 10:1 splits of 2024-07-15 and 2024-10-03).
   - Stocks that later split heavily are left out of early universes: NVDA, trading $100M–$2.2B a day, in 2010–2017 and 2019; AVGO and LRCX in 2010–2014; CMG in 2010–2011. 24–59 stocks a year with a median dollar volume of $5M or more fail on price alone.
   - Returns are ratios, so v changes only through which stocks enter the market average.
   - A fix: test the price on unadjusted closes. Stooq's files carry only adjusted ones, so this needs unadjusted prices or split dates from another source (for example Sharadar's corporate-action tables, stockai's PLAN §5.3).
2. **The bad-print repair looks one day ahead.** `Panel::repair_bad_prints` (crates/c2-data/src/panel.rs) blanks close[r] when it is more than 2× away from close[r−1] (up or down) and close[r+1] comes back within about 10% of close[r−1]. So whether close[d], v[d] and v[d+1] exist depends on close[d+1]. It touches 72 cells before 2024-06-27, 3 of them on graded universe days. Small, but a causal repair would judge a print with data through r only, or blank the reverting day's outcome instead.
3. **Census-unit rank IC contains a part known at d.** v[d+1] subtracts the trailing-year mean residual, so k[d] = −(mean ÷ sd) of the residuals of days d−251..d is part of v[d+1].
   - As a score, k alone gets census IC +0.055 (z 20.4) and actual-return IC −0.0145. Slow scores that lean with k gain census-unit IC this way, and those that lean against it lose it (12-1 momentum, −0.89 with k: census IC −0.11 weekly against +0.017 in actual returns).
   - correlation2's 21-day reversal control (census z = 13.04) is mostly k: over 2010–2023 its census IC is +0.088, of which k accounts for +0.071, and against v[d+1] − k[d] it is +0.011 (z 1.3).
   - k[d] is known at decision time, so v[d+1] − k[d], the outcome minus its known part, makes a clean census-unit outcome.
