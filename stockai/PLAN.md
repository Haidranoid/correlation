# stockai plan

**Status (2026-10-19): the proof of concept is done:** M0 (export, judge, G0 passed), the parts of M1 it needed (volatility-matched random accounts, a one-day delay, IC in actual returns, the reversal, momentum and LightGBM baselines, canaries, tripwire) and M2 (stockGPT v1, 14 walk-forward years × 3 seeds, sign-flip null). G1 passed, but stockGPT's daily $10,000 account loses money at 5 and 20 bps per trade, and it is no better than turnover-matched random accounts at any cost level (it beats them at p < 0.1 in 4 / 4 / 3 of 14 years; combined over the years, z ≈ 0.8 at 0 bps and 0.5 at 20 bps). M1's canary gate did not pass as written: the 20-day-stale and permuted canaries keep z = 2.4–3.2 (about half low-volatility tilt, half a slower property not yet identified; not a leak). A LightGBM trained on actual next-day returns, added during verification, has the strongest signal (IC z = 9.15), but nothing in the POC compounds to more than SPY after 20 bps. See `test-results.md`.

Written 2026-10-18. The research behind it is in [the report](<reports/AI models for stock price prediction.md>) and `research_notes/`. Measurements and decisions will go in `test-results.md` once building starts, as in correlation2.

## Where things stand (read this first)

**What stockai is.** A test of one idea: train an AI model on price history the way an LLM is trained on text, so it predicts each stock's next move the way an LLM predicts the next word. Then judge it by correlation2's standard: predict a year it never saw, trade $10,000, and compare against SPY, QQQ and random-guess accounts after trading costs.

**LLM, diffusion, or something else?**
- **Build a small GPT-style model from scratch.** It is the same machinery as an LLM, but its "words" are a stock's daily moves instead of text. It is small (1–20 million parameters, against billions for a chat LLM) and trains in minutes on the RTX 5080.
- **Not a text LLM.** Feeding numbers to a ChatGPT-style model adds nothing. In the main test of this (Tan et al., NeurIPS 2024), replacing the LLM with a single attention layer did as well or better and trained over 100 times faster. Text LLMs have also memorized prices up to their training cutoff, so they can't be tested honestly on that history.
- **Not diffusion, except later as an add-on.** Diffusion models generate whole realistic price paths. That suits scenarios and risk, but nothing shows they pick direction better, and they are tens to thousands of times slower per forecast.
- **"Something else" is the bar to beat.** Gradient-boosted trees (LightGBM) and the 5-day reversal rule are cheap and hard to beat. Every AI result is compared against them.

**What has been done.** This exact idea has been tried, at scale:
- **StockGPT (2024)** trained a 0.93M-parameter GPT on US daily returns.
- **Rahimikia, Ni & Wang (Nov 2025)** trained transformer forecasting models (Chronos and TimesFM designs) from scratch on daily returns, and tested them on about 10,000 US stocks over 2001–2023. Before costs, long-short Sharpe ratios reached 3.7–7.7.
- **At 20 bps per trade, every model lost money,** including the tree model they compared against. The apparent edge was mostly short-term reversal in small stocks: plain linear regression scored a Sharpe of about 4.9 before costs.
- **Kronos (2025)** is an open model pretrained on 12 billion price bars, with data ending June 2024, so correlation2's held-out year (from 2024-06-28) is a clean test for it.
- **The untested gap:** nobody has published results for the version correlation2's test measures. That is a long-only account on liquid stocks with low turnover, holding for days or weeks. stockai can fill that gap.

**What to expect.**
- **Daily trading:** most likely the literature's result. Real but small signal (about 0.005–0.02 correlation between the model's daily ranking of liquid US stocks and their next-day ranking) that trading costs eat.
- **Why costs win at daily trading:** replacing the picks every day needs a next-day edge of more than twice the one-way cost, which is 10–40 bps at 5–20 bps per trade. Price-only models on liquid stocks show a few bps to about 10. Holding for H days divides the edge needed per day by about H.
- **The plan's aim:** so, weekly or longer holds, liquid stocks and rare trading, where costs bite least.
- **One year proves little:** a strategy with a Sharpe of 1, tested for one year, has a t-statistic near 1, which luck produces easily. So every model is graded across 14 walk-forward years, and every configuration tried is counted.
- **Value either way:** a clean answer on data no model has seen, as correlation2 gave for lead-lag.

**What happens first (M0).**
- Set up Python and PyTorch, and check the GPU.
- Export correlation2's prices for Python.
- Teach `c2 census holdout` to grade a file of model predictions.
- Prove the grading is right by reproducing correlation2's own reversal result exactly.

**Decisions for you (§10):**
1. Buy fresh data? Recommended: Sharadar, $39 for one month. It would also fix the survivors-only universe and the split-adjusted $5 test (§5.2).
2. Which holding period to aim for. The POC points to weekly or longer: at 20 bps every model and reversal rule traded daily lost money, and the strongest daily edge mostly vanished a day late (`test-results.md`).
3. Is it OK to add the grading features stockai needs to correlation2? For now they live in a pinned fork in `judge/`; see §10.

**New after the POC** (`test-results.md`, "Your call"): go on to M4 (weekly and monthly targets with turnover limits, for stockGPT and for LightGBM on actual returns alike) or stop the stockGPT line. G2 requires beating LightGBM, which is now a higher bar.

## 1. How this works, in plain terms

An LLM is trained by playing one game billions of times:
1. It reads some text.
2. It gives a probability to every possible next word.
3. It sees the real next word.
4. It adjusts its internal numbers (its *parameters*) to be less surprised next time.

The average surprise is the *loss*, and training is driving it down.

stockai plays the same game with stocks:

| LLM | stockai |
|---|---|
| A token is a piece of a word | A token is one stock's move on one day, as one of about 100 size bins ("down 1.2 typical days", "up 0.3 typical days") |
| A sentence | One stock's last 256 trading days (about a year) |
| Predicts the next word | Gives a probability for each bin of tomorrow's move |
| Trained on internet text | Trained on 20 years of daily moves of thousands of stocks |
| Writes a paragraph one word at a time | Can "write" a possible next month one day at a time; many samples give a spread of outcomes |

**Why moves, not prices.** A model asked for tomorrow's *price* looks brilliant by guessing today's price: it is within about 1% on most days and worthless. Of 612 deep-learning papers on investing, 73% forecast prices, and only 4% of those compare against that "tomorrow equals today" guess.

stockai predicts the *move*, in correlation2's units: the stock's daily return minus the part the market explains, minus that leftover's own average over the past year, divided by its own typical daily move over the past year, clipped at ±5. A "1" means one typical day's move. That unit means the same thing for every stock and every decade, which a single model needs. Because the past year's average is subtracted, part of tomorrow's value is already known today, so stockai judges signals in actual returns as well (test-results.md).

**Terms used below:**

| Term | Meaning here |
|---|---|
| Parameters | The model's adjustable numbers. More can learn more, and can also memorize noise |
| Training / validation / test years | Years the model learns from / years used to choose settings and when to stop / years it never sees until graded |
| Overfitting | Learning the noise of the training years. It shows as validation loss rising while training loss keeps falling |
| Walk-forward | Retraining for each test year using only earlier years, so every test year from 2010 on gets an honest out-of-sample prediction |
| Rank IC | Correlation between the model's ranking of stocks on a day and their actual ranking over the next day (or week). 0 is chance |
| Leakage | Any way information from the future reaches the model. The most common reason published results look too good |

## 2. What has already been done

| Work | What it is | What happened | For stockai |
|---|---|---|---|
| [StockGPT](https://arxiv.org/abs/2404.05101) (Mai, 2024) | 0.93M-parameter GPT over 402 return bins, trained on NYSE daily returns 1926–2000, tested 2001–2023 | **Before costs:** 119% a year, Sharpe 6.5 (equal weight, daily long-short). **Value-weighted:** Sharpe 1.0. **Value-weighted, trading a day later:** Sharpe 0.2. No code, weights or replication | The exact idea. The result is small-cap reversal before costs |
| [Re(Visiting) TSFMs in Finance](https://arxiv.org/abs/2511.18578) (Rahimikia, Ni & Wang, Nov 2025) | Chronos- and TimesFM-design transformers trained from scratch on daily returns (94 countries), plus zero-shot foundation models; tested on ~10,000 US stocks, 2001–2023. Released 613 [FinText models](https://huggingface.co/FinText) (Apache-2.0), one per year 2000–2023 | **Before costs:** long-short Sharpe 3.7–7.7. **At 20 bps:** all negative, including CatBoost (6.44 → −3.62). **OLS before costs:** ~4.9 | The closest prior test. Its yearly models are a ready-made walk-forward comparison |
| [Kronos](https://github.com/shiyu-coder/Kronos) (Shi et al., [arXiv 2508.02739](https://arxiv.org/abs/2508.02739), AAAI-26) | GPT-style model over OHLCV candles. 12.1B bars from 45 exchanges, data to June 2024. Open weights (MIT): mini 4.1M, small 24.7M, base 102.3M | **Authors' rank IC:** ~0.025–0.06, mostly intraday and non-US. **Independent checks:** direction near random on large US stocks. Its CSV fine-tuning script leaks future data ([#227](https://github.com/shiyu-coder/Kronos/issues/227)) | The pretrained model to beat; clean on correlation2's holdout year |
| General forecasting models (Chronos-2, TimesFM-2.5, Toto, Moirai) | Pretrained on mixed public series; data cutoffs mostly not disclosed | Zero-shot on US daily returns: R² below zero, ~50–51% direction | Cheap controls, with a contamination caveat |
| Text LLMs on numbers (LLMTime, Time-LLM; Tan et al. 2024) | Reuse chat-style LLMs for series | Removing the LLM matched or beat it; 100× slower; Time-LLM needs 32–39 GB of GPU memory | Not used |
| LLMs reading news (Lopez-Lira & Tang 2023 and later) | GPT-4 scores headlines | Sharpe 2.97 before costs (Oct 2021–May 2024), falling from 6.54 to 1.22 over the period; unprofitable at 20 bps round trip | Different data (text). Out of scope unless §9's pivot is chosen |
| Diffusion and flow matching (TimeGrad, Diffusion-TS, FTS-Diffusion, Sundial) | Generate realistic paths | TimeGrad takes 0.87–6.7 s per forecast against ~3 ms for a one-step model. The one controlled 2026 test of synthetic-data augmentation found it hurt direction prediction | Optional add-on for scenarios (M6) |
| Benchmarks and competitions (Qlib, Kaggle, Numerai) | Many models on real market data | **Best correlations:** about 0.01–0.05. **Trees:** about as good as deep models, with less run-to-run variation. **Replication:** deep models' reported gains often fail it (MASTER: 0.0215 against 0.064 claimed). **US daily returns (Nov 2025):** CatBoost beat the foundation models | LightGBM is the bar |

**Would anyone publish a model that worked?** Often not: a working edge with limited capacity is worth more traded quietly. But the literature isn't silent:
- **The signal is published.** Gross Sharpe ratios of 4–7 are in print. Costs are what kill it, and they hit everyone.
- **Firms that do profit from prediction rely on more.** Renaissance and market makers win on near-zero costs, speed, private and intraday data, and thousands of small signals combined.
- **Known edges decay.** Published anomalies earn about 58% less after publication (McLean & Pontiff, 2016), and correlation2 watched 5-day reversal fade after 2021.

So stockai doesn't rely on anyone's word: it tests on data no one has looked at.

## 3. The model

### 3.1 Recommendation

**Main line:** a small decoder-only transformer trained from scratch on correlation2's daily move tokens ("stockGPT"). It cannot be contaminated, because it only ever sees the years before each test year.

**Compared against, through the same judge:**
- **Rules:** 5-day and 21-day reversal, and 12-1 momentum.
- **LightGBM** on simple price features (the "something else").
- **Kronos,** zero-shot and fine-tuned (leak fixed). Its one clean test is correlation2's holdout year (plus the sealed exam, §4.4).
- **FinText models,** walk-forward: each test year uses the model pretrained through the previous year.
- **Chronos-2 and TimesFM-2.5,** zero-shot, as general controls (cutoffs undisclosed, so they don't count as evidence).

### 3.2 First version (stockGPT v1)

| Choice | v1 | Why |
|---|---|---|
| Tokens | correlation2's daily move (market-residual, minus its trailing 252-day mean, ÷ trailing 252-day typical move, clipped ±5), cut into ~100 bins 0.1 wide, plus a "missing" token | Same unit across stocks and decades; Kronos's tests found a discrete next-token loss beat plain regression on rank IC |
| Extra inputs per day | the market's move that day, and the stock's typical-move level | The tokens have the market and volatility removed; these give them back as context |
| Model | nanoGPT-style decoder, 4–6 layers, width 128–256: ~1–5M parameters | US stock data is ~20M tokens; at about 20 tokens per parameter that points to 1–5M (5–20M with all markets). The largest finance study found data, not size, was the limit |
| Context | 256 trading days | About a year: covers reversal, momentum and volatility clustering |
| Training | next-token cross-entropy over all stocks; AdamW, bf16, dropout; 3 random seeds; stop when validation rank IC (in actual returns) stops improving | Deep models vary run to run, and three seeds show how much. Cross-entropy can keep falling while the model learns only volatility, which says nothing about direction |
| Prediction → score | expected next move from the predicted bin probabilities | A score per stock per day is all the judge needs. Weekly and monthly holds get their own models trained directly on 5- and 21-day moves (v2). Rolling a daily model forward compounds its errors, and StockGPT's daily signal faded within a day |
| Walk-forward | one model per one-year test period from 2010-06-28 to 2024-06-27, trained on data ending 5 trading days before it | Each model takes minutes, so 14 years × 3 seeds is cheap. Stooq is too thin before 2005 to start earlier |

### 3.3 Later versions

- **v2, cost-aware (M4):**
  - Models trained directly on 5- and 21-day moves, with ranking-style training and an option to train only on liquid stocks.
  - Turnover-limited picks: hold 50 names and replace at most 5 a day, or hold each pick at least 5 days.
  - That turns over about 10% of the book a day, costing roughly 2.5–10% a year at 5–20 bps, against 25–100% for replacing everything daily.
- **v3, market-aware (M5):** each day the model also attends across all stocks at once, plus market and sector context.
  - This tests what correlation2's straight-line pairs couldn't: nonlinear, conditional and many-leaders-to-one-target effects.
  - It also adds the other Stooq markets as training data, each tagged with a market ID, and is graded on US stocks only.
- **v4, generative extras (M6, optional):** sampled one-month paths shown as forecast fans in c2-gui, and volatility forecasts, the one thing price models reliably predict, scored against simple volatility baselines.

## 4. How it will be judged

### 4.1 The judge is correlation2

- **New input:** `c2 census holdout` gets `--family scores --scores <file>`. The file holds one score per stock per day, aligned to the exported panel; NaN means no score.
- **Same test as everything correlation2 graded:**
  - $10,000, the 20 highest positive scores, equal weight, rebalanced every 1, 5 or 21 days.
  - Flat costs per trade (0, 5 and 20 bps today; M1 adds 10 bps and per-stock costs).
  - 1,000 random-guess accounts with the same turnover and stock pool.
  - Comparisons with SPY, QQQ and buy-and-hold of all candidates.
  - Hit rate, rank IC and the top-minus-bottom-10% long-short check.
  - `--walk-forward` gives the table by year.
- **One judge for every model:** stockGPT, Kronos, LightGBM and the rules alike. Model code never grades itself.
- **Upgrades in M1, before any model is graded:**
  - **Random accounts matched on volatility as well as turnover.** This is on correlation2's own to-do list. A model that learns reversal tilts toward volatile stocks, and in a strong year that tilt alone beats uniform random picks.
  - **Each stock's own trading cost.** Its bid-ask spread is estimated from daily prices (the EDGE estimator, in the `bidask` package) and floored at a few bps for large caps. Flat 0, 5, 10 and 20 bps are still reported; a flat cost flatters small caps.
  - **A next-day trading option.** Trade at the next session, and report a variant that waits one more day. StockGPT lost 78% of its equal-weighted return to a one-day delay.
  - **Results in actual returns,** value-weighted and by size bucket, as well as in correlation2's market-adjusted units. correlation2's market-adjusted rank IC was positive in 18 of 19 years, while its long-short in actual returns earned 0.45 bps a day (z = 0.92).
  - **A ledger** of every configuration ever graded, with a deflated Sharpe ratio that corrects for the number of tries. correlation2's SPEC §7.4 calls for this, but it isn't built yet.

### 4.2 Model-level checks

- **Loss against no-skill:** validation loss compared with always predicting the stock's own historical spread of moves. The gap is what the model actually knows. Expect it to be tiny.
- **Rank IC by year:** with t-statistics and the share of years positive.
- **Calibration:** when it says 60% chance of up, it should be up about 60% of the time.

### 4.3 Controls (tests of the test)

- **Reversal rediscovery.** Every published model of this kind found short-term reversal. If v1 shows no gross signal at all, suspect a bug in the pipeline before concluding anything.
- **Sign-flip null.** Train on data whose moves are multiplied by a random ±1 per date, correlation2's null. Any skill there is a bug or chance.
- **Planted signal.** Synthetic data with a known lead-lag; the model must find it.
- **Leakage canaries.** Shifting the targets by a day, or permuting the dates, must collapse every score to chance.
- **Leakage tripwire.** A daily rank IC above 0.05 on large caps, or a large drop in cross-entropy, is treated as a bug until proven otherwise. A common cause is a volatility or beta estimate that used the forecast day.
- **Compiled-model check.** At the start of every run, compare outputs with and without `torch.compile`: an open PyTorch bug ([#191433](https://github.com/pytorch/pytorch/issues/191433), reproduced on an RTX 5080) silently corrupts this exact kind of model under bf16.

### 4.4 Rules against fooling ourselves

- **Point in time.** Decisions at day d's close use data through day d only, including every normalization statistic. Kronos's fine-tuning leak was exactly a normalization computed over the forecast window.
- **No contaminated grading.** No pretrained model is graded on a period its pretraining could include.
- **Log everything.** Every experiment is logged, failures included (`test-results.md`, the ledger and a local tracker), and any result is reported with the number of configurations tried.
- **Statistics.** A new signal needs t > 3 across the walk-forward years, after the deflated-Sharpe correction. Settings are chosen on validation and walk-forward years only.
- **Two locked periods, each run once at the end (M7):**
  - correlation2's holdout year, 2024-06-28 to 2025-06-27. It is Kronos's clean year, but only semi-clean for us, because correlation2's results on it are already known.
  - The sealed exam: data after 2025-06-27, which needs fresh data. It is opened after one configuration is frozen and committed.

## 5. Data

### 5.1 In hand

| Source (Stooq bulk zips, ending 2025-06-27) | Rows | Notes |
|---|---|---|
| US | 26.6M | 11,531 symbols, of which 8,503 are stocks (20.2M rows); already in correlation2's store |
| Poland | 24.1M | includes funds and NewConnect |
| Japan | 15.0M | Tokyo Stock Exchange |
| UK | 10.8M | London Stock Exchange |
| Hong Kong | 7.6M | HKEX |
| World | 18.3M | currencies, bonds, crypto, indices: not stocks |
| **Total** | **~103M** | Kronos pretrained on ~12B; ten years of minute bars would reach that scale |

### 5.2 Known problems

- **Survivors only.** No delisted companies: SIVB, BBBY, FRC and TWTR are missing. About 3.3–3.5% of US firms delist for cause each year, typically losing 30–55% on the way out. A model that learns to buy small-cap losers looks better than it is, and the held-out year misses its own delistings too. Measured in the export: no stock's last close is before 2024-03-01 (only 4 before 2024-06-01), and ATVI, XLNX and CELG are missing too.
- **Thin before 2005.** Stooq has US data for 316 stocks in 2000 and 2,138 in 2005, and its older prices carry pre-decimal tick artifacts. So training starts in 2005, and stockGPT's walk-forward starts with the 2010 test year.
- **Adjusted prices.** Stooq adjusts for splits and dividends as of the download, so old prices shift between snapshots. Snapshots must be joined by daily returns, not price levels.
  - **The adjustment is also a look-ahead in anything that uses price levels.** correlation2's $5 median-price test (the market's liquid set and the judge's universe) sees later splits, including the locked year's (AVGO's and LRCX's 10:1 splits of 2024-07-15 and 2024-10-03). So NVDA is out of the universe in 2010–2017 and 2019, AVGO and LRCX in 2010–2014, and CMG in 2010–2011.
  - Dollar volume is unaffected by splits but scaled down by later dividends. Returns, and so v, are ratios and are unaffected, except through which stocks enter the market average.
  - Fix: price tests on unadjusted prices (Sharadar, decision 1). See `docs/c2-export.md`, "Point in time".
- **Bad prints.** correlation2's panel builder already repairs them, and stockai reuses it. The repair looks one day ahead: it blanks a close that moves more than 2× (up or down) and comes back within about 10% the next day, so whether close[d] exists depends on close[d+1]. It touches 72 cells before 2024-06-27, 3 of them on graded universe days.
- **Stale.** The data ends 2025-06-27.

### 5.3 Fresh and better data (decision 1)

| Option | Cost | What it gives | What it fixes |
|---|---|---|---|
| **Sharadar "Prices", full history** (recommended) | $39 for one month ($299/yr), personal license | Daily US prices from 1998 to now, including delisted stocks, with ticker, corporate-action and S&P 500 membership tables | Staleness, survivorship, thin pre-2005 history, and a sealed exam period |
| Sharadar "Prices", 5 years | $9 for one month | 2021 to now | Staleness and a sealed exam period |
| Stooq bulk zip, downloaded by hand | free | Same format as now; how current it is wasn't confirmed | Staleness only |
| Norgate Platinum | $346.50 per 6 months | From 1990, delisted stocks, historical index membership; Windows updater | Same as Sharadar, more explicitly guaranteed |
| Intraday (later, only if daily fails) | Massive $29–79 per month; MarketParquet $79 one-time | 5–10+ years of minute bars, 28–90 GB | Data scale, at a disk cost (171 GB free) |

**Before relying on Sharadar,** check on day one that SIVB, BBBY, FRC and TWTR are present. Also check whether the license lets you keep the data after cancelling, which wasn't confirmed.

### 5.4 How the years are used

- **Training:** from 2005. With Sharadar, from 2002, after US exchanges moved to decimal prices in 2001.
- **Development:** walk-forward one-year test periods from 2010-06-28 to 2024-06-27. Each model is trained only on data ending 5 trading days before its test year. The FinText models are graded each year with the version trained through the previous year.
- **Validation:** for each test year, the 252 trading days just before that 5-day gap.
  - It is used for early stopping and for choosing settings.
  - Training uses the data before it, with its own 5-day gap.
  - The test year is used for neither.
- **Locked year:** 2024-06-28 to 2025-06-27, run once (M7). Kronos is graded only here and in the sealed exam, because its pretraining covers every earlier year.
- **Sealed exam:** 2025-06-28 to the latest data, only with fresh data, run once (M7).

## 6. Software and hardware

- **Python 3.12 via uv.** Kronos pins pandas 2.2.2, and some tools have no wheels past 3.12. Each outside model (Kronos, Chronos-2/TimesFM, FinText) gets its own uv environment so their pinned dependencies don't collide.
- **PyTorch 2.14.1** (released 2026-09-30), using the default Linux wheel, which is built for CUDA 13.0 and supports the RTX 5080. Avoid the cu126 build. PyTorch 2.15 (due 2026-10-28) moves to CUDA 13.2, which the 595.99 driver also handles.
- **What works on the RTX 5080:** bf16, `torch.compile` (with the check in §4.3) and PyTorch's built-in attention. Skip the separate `flash-attn` package, which fails to build for this GPU.
- **Code base:** [nanoGPT](https://github.com/karpathy/nanoGPT)'s ~300-line model and training files as the starting point. The repo is frozen but the easiest to read.
- **Experiment tracking:** MLflow (local) or Trackio. No cloud account needed.
- **Licenses:**
  - Kronos code and weights are MIT; chronos-forecasting and FinText are Apache-2.0.
  - TimesFM 3.0 and Moirai 2.0 weights are non-commercial: fine for personal research.
  - The PyPI package named `kronos` is an unrelated Django app. Install Kronos from its GitHub repo.
- **Compute (estimates; no measured RTX 50-series benchmarks were found):**
  - One pass over the US data takes seconds to a minute at 1M parameters, minutes at 10M, and 5–20 minutes at 100M.
  - 16 GB of GPU memory is plenty up to ~100M parameters, which includes fine-tuning Kronos-base.
  - The training data fits on the GPU (under 1 GB), so windows are sampled there directly, with no data-loading workers.
  - Kronos's released inference code recomputes the whole window for every sample. 5,000 stocks over the locked year take an estimated 15–20 GPU-hours with Kronos-base and 4–5 hours with Kronos-small, so start with small. Small and base see 512 bars (about 2 years); mini sees 2,048.
- **Reproducibility:** commit `uv.lock`, and log the git commit, config, data hash and seed with every run.
- **Disk:** daily data, exports and checkpoints fit in a few GB. Intraday would need 28–90 GB of the 171 GB free.

## 7. Milestones

Each milestone ends with a measured result in `test-results.md`. Gates are where the plan can stop.

| | Build | Proves | Gate |
|---|---|---|---|
| **M0** Setup and judge | **stockai:** uv project; GPU, bf16 and compile checks; data loader. **correlation2:** `c2 export` (census variable, actual returns, closes, dollar volumes, liquidity, dates, tickers as `.npy`); `c2 census holdout --family scores` | The plumbing is right | **G0:** reversal scores computed in Python and graded through `--family scores` reproduce `--family reversal` exactly, walk-forward table included |
| **M1** Judge upgrades and baselines | **Judge:** the §4.1 upgrades. **Baselines, all walk-forward:** zero forecast; the bin histogram; 5-day and 21-day reversal; 12-1 momentum; LightGBM or CatBoost on simple price features (moves over 1/5/21/63/252 days, volatility, volume change, market move) | The bar, measured the strict way | The canaries collapse to chance |
| **M2** stockGPT v1 | Tokenizer, model, trainer; walk-forward predictions into a scores file; the §4.3 controls | Whether the model learns anything | **G1:** gross signal present (rank IC t ≥ 2; applied as combined z ≥ 2 over the walk-forward years, in actual returns), nothing on the null, no tripwire. Published models found reversal-strength signal, so a result far below that means checking the pipeline first. Weaker but real signal is a finding, not a failure |
| **M3** Outside models | FinText walk-forward; Chronos-2 and TimesFM-2.5 as controls; Kronos-small fine-tuned on data through 2024-06-27 with the leak fixed (graded only in M7) | How existing models compare | none |
| **M4** Survive costs | **Model:** v2. **Picks:** hold 50 and replace at most 5 a day, or a 5-day minimum hold; weekly and monthly rebalancing; liquid names | Whether any signal pays | **G2:** after each stock's own costs, beats SPY, QQQ and random accounts matched on turnover and volatility in most walk-forward years; beats LightGBM; t > 3 after counting every configuration. If not, stop the trading line and write it up; §9 lists the next options |
| **M5** Market-aware | v3: cross-stock attention, market and sector context, other markets | Effects correlation2 couldn't test | same as G2 |
| **M6** Generative extras (optional) | Forecast fans in c2-gui; volatility forecasts against trailing volatility and GARCH | What the model is good for besides direction | none |
| **M7** Locked runs | Freeze and commit one configuration. Run it once on the locked year, alongside Kronos (zero-shot and fine-tuned), FinText 2023 and the baselines, and once on the sealed exam. Daily picks go into correlation2's forward log (`c2 update`, `c2 forward-report`) | The real answer | **G3:** the configuration is committed before either run |

**A month, roughly:**
- **Week 1:** M0–M1.
- **Week 2:** M2–M3.
- **Week 3:** M4–M5.
- **Week 4:** M6–M7 and the write-up.

Fresh data needs to arrive before M7.

## 8. Using the Claude budget well

- **Long runs in the background:** training runs go in the background, and results land in `test-results.md`.
- **Parallel sub-agents:** independent jobs, such as M3's outside models, can run as parallel sub-agents.
- **Workflows for big sweeps:** sweeps (model size × context length × seeds × horizons) can run as a multi-agent workflow. That uses a lot of tokens, so it only happens if you ask ("use a workflow").

## 9. Expectations, risks, and what counts as success

- **Most likely outcome:** the literature's, and correlation2's. Real gross signal, gone after costs, indistinguishable from random accounts with the same turnover.
- **Success means passing G2, then holding up in both locked runs (M7).** That is: beating random accounts matched on turnover and volatility, after each stock's own costs, reported with the number of configurations tried.
- **Small accounts pay real costs.** Measured retail round trips on small orders cost 0.07–0.46% against the quote midpoint, depending on the broker, and 0.62% at the full quoted spread. A $10,000 account avoids moving the price, not paying the spread.
- **A smaller win:** volatility forecasts that beat simple baselines.
- **Main risks and their fixes:**
  - Survivorship: buy Sharadar.
  - Overfitting through many tries: the ledger, the t > 3 bar and the locked runs.
  - Leakage: the controls and the tripwire.
  - Run-to-run noise: three seeds.
  - The compile bug: the per-run check.
- **If G2 fails, your call:**
  - Intraday data ($29–79, 28–90 GB).
  - News text scored by an LLM whose training ends before the test. The research found thin, decaying signal there.
  - Stop at volatility and risk forecasting.

## 10. Decisions for you

1. **Fresh data.** Recommended: Sharadar "Prices", full history, $39 for one month. Download everything that month. It fixes staleness and survivorship, fills in the thin pre-2005 history, and gives the sealed exam. Cheaper: $9 for the 5-year tier, which gives the exam only. Free: a hand-downloaded Stooq bulk zip, which is survivors only and of unconfirmed date.
2. **Holding period.** The literature says costs win at daily trading. I'd aim for weekly, and test daily and monthly too.
3. **correlation2 additions.** stockai needs these added to correlation2:
   - `c2 export` and `--family scores` (M0).
   - The §4.1 grading upgrades (M1): volatility-matched random accounts, per-stock costs, next-day trading, results in actual returns, and the ledger. The volatility-matched accounts and the ledger were already on correlation2's own to-do list.

   **What was done for the POC (2026-10-19):** another session was in the middle of reviewing correlation2, so correlation2 was left untouched. The export and the grader live in `judge/` as a fork of its holdout test, built against correlation2's crates pinned at commit 9335126. G0 shows the fork grades exactly as correlation2 does. Still recommended: move `export`, `--family scores` and the upgrades into correlation2 once that review lands, so there is one judge, then re-pin. correlation2's variable code was refactored on its review branch, so a re-pin should rerun G0 and compare exports.
4. **Intraday data.** Not now; only if daily fails (§9).

## 11. Repo layout

```
stockai/
  PLAN.md           this plan
  README.md
  test-results.md   measurements and decisions (from M0)
  reports/          the research report
  research_notes/   the notes behind it
  pyproject.toml    uv project, Python 3.12 (M0)
  docs/c2-export.md the data and scores contract with the judge
  judge/            stockai-judge (Rust): export and holdout, a fork of correlation2's test pinned at 9335126
  stockai/          data.py, splits.py, synthetic.py; gpt/ (stockGPT); baselines/ (rules, LightGBM); poc/ (G0, controls, summary)
  tests/            pytest
  scripts/poc.sh    the proof of concept, end to end
  results/          judge outputs and summaries per run (poc/; poc-quick/ from --quick; poc-quick-new/ holds the verification's quick run)
  external/         one uv environment per outside model (Kronos, Chronos-2/TimesFM, FinText), not built yet
  runs/             scores, checkpoints and logs (not in git)
```
