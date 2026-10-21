# GPT-style and pretrained time-series foundation models for price/return forecasting — dated catalog (as of 2026-10-18)

Research date 2026-10-18. The user's holdout is 2024-06-28..2025-06-27 (daily US stocks, Stooq data ending 2025-06-27). Tags: **[authors]** means a claim by the model's own authors; **[independent]** means a third-party paper; **[community]** means unreviewed GitHub reports; "(search summary)" means a fact seen only in a search-engine summary, which is weaker evidence. "Weights public" dates come from Hugging Face `createdAt` metadata where available. All dates are YYYY-MM-DD.

## Kronos (Shi et al., 2025): tokenizer, sizes, open weights, corpus and cutoff, results, fine-tuning, license, follow-ups

### Takeaway
Kronos (Tsinghua; arXiv 2508.02739, 2025-08-02; AAAI-26, published 2026-03-14) is the only open foundation model built on the user's own data type: OHLCV bars quantized into discrete tokens and predicted autoregressively, like an LLM. MIT weights are public for mini (4.1M), small (24.7M) and base (102.3M); large (499.2M) is unreleased. The paper says its pretraining data "extends up to June 2024", so at most one bar (2024-06-28) could overlap the user's holdout. That makes it the cleanest finance-pretrained zero-shot candidate. The authors' gains are small in absolute terms (avg. price RankIC about 0.025, return RankIC about 0.06), independent evidence is mixed, and the fine-tuning code had a look-ahead bug that is only half fixed.

### Cited Findings
**Release, venue, license**
- arXiv 2508.02739 was posted 2025-08-02. Fine-tuning scripts were released 2025-08-17. README news (2025-11-10): "accepted by AAAI 2026" — [GitHub README](https://github.com/shiyu-coder/Kronos)
- AAAI-26 Vol. 40 No. 30, pp. 25366–25373, published 2026-03-14, DOI 10.1609/aaai.v40i30.39730. Authors Yu Shi, Zongliang Fu, Shuo Chen, Bohan Zhao, Wei Xu, Changshui Zhang, Jian Li (Tsinghua IIIS / Dept. of Automation) — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/39730)
- Also a poster at the NeurIPS 2025 "Generative AI in Finance" workshop — [NeurIPS 2025](https://neurips.cc/virtual/2025/132504)
- MIT license. Repo created 2025-07-01, about 41.1k stars and 6.9k forks, last push 2026-04-13 (checked 2026-10-18) — [GitHub API](https://api.github.com/repos/shiyu-coder/Kronos)
- HF repos were created 2025-06-30/07-01 (before the arXiv paper) and last modified 2025-09-09, all license MIT — [HF API](https://huggingface.co/api/models/NeoQuasar/Kronos-small)
- HF org NeoQuasar hosts Kronos-mini, Kronos-small, Kronos-base, Kronos-Tokenizer-2k and Kronos-Tokenizer-base (3.96M each). All were last updated 2025-09-09. No Kronos-large or newer Kronos version is listed. The same org also hosts unrelated "WindFM" wind-power models — [HF NeoQuasar](https://huggingface.co/NeoQuasar)
- Downloads (last month, as shown 2026-10-18): Tokenizer-base 2.42M, small 1.17M, base 713k, mini 432k — [HF NeoQuasar](https://huggingface.co/NeoQuasar)

**Tokenizer and architecture (how prices become tokens)**
- The tokenizer is a Transformer autoencoder with Binary Spherical Quantization (BSQ):
  - Input is D=6 features per bar (open, high, low, close, volume, amount), one token per bar.
  - Each token has k=20 bits (2^20 vocabulary), split into a coarse and a fine 10-bit subtoken (1,024 values each).
  - Each feature is z-scored and clipped to [−5, 5].
  - Missing volume/amount is zero-imputed, and both are randomly zeroed in 5% of training samples.
  - Encoder and decoder have 3 layers each, d=256 — [arXiv HTML](https://arxiv.org/html/2508.02739)
- Predictor: a decoder-only causal Transformer (RoPE, pre-LN RMSNorm).
  - A coarse head predicts the first subtoken; a fine head predicts the second, conditioned on the sampled coarse subtoken.
  - Learned minute/hour/weekday/day/month embeddings are added — [arXiv HTML](https://arxiv.org/html/2508.02739)
- Sizes (Table 1), all with 512-token max context:

  | Variant | Layers | d_model | Params |
  |---|---|---|---|
  | small | 8 | 512 | 24.7M |
  | base | 12 | 832 | 102.3M |
  | large | 18 | 1664 | 499.2M |

  [arXiv HTML](https://arxiv.org/html/2508.02739)
- Kronos-mini: 4.1M parameters, Kronos-Tokenizer-2k, context 2048. Large: "not released". Volume and amount are optional at inference and zero-filled if missing — [GitHub README](https://github.com/shiyu-coder/Kronos)
- Output is probabilistic: it samples future token paths. Defaults: T=0.6, top-p 0.9, N=10 paths for price forecasting — [arXiv HTML Table 6](https://arxiv.org/html/2508.02739)
- The released inference loop has no KV cache. Each step re-runs the full window (up to max_context) for batch × sample_count paths, and the predictor returns the mean of the sampled paths — [model/kronos.py](https://github.com/shiyu-coder/Kronos/blob/master/model/kronos.py)

**Pretraining corpus and date range**
- ">12 billion" K-line records. Appendix Table 13 totals 96,569 assets and 12.11B observations.
  - Exchange count is inconsistent: "45 global exchanges" in the abstract vs "over 40 exchanges across more than 30 countries" in Appendix B.
  - Frequency count is inconsistent: 7 in the text vs 12 listed in Table 4 (1min…Weekly).
  - Asset classes are equities, crypto, futures and FX, with crypto/futures/FX upsampled — [arXiv HTML](https://arxiv.org/html/2508.02739)
- US slice (Table 13), timeframes 1min/5min/15min/30min/1h/D/W:
  - Nasdaq: stocks + ETFs, 8,725 assets, 2,478,662,459 observations, from 2000-01-01.
  - NYSE: 7,073 assets, 2,133,143,549 observations, from 2000-01-01 — [arXiv HTML](https://arxiv.org/html/2508.02739)
- Other rows: Binance 997 assets, 1,237,002,843 obs, from 2021-01-31; FX 1,023 pairs, 462,434,562 obs, from 2020-01-31 — [arXiv HTML](https://arxiv.org/html/2508.02739)
- Cutoff: "The pre-training data for Kronos extends up to June 2024", and "our test period for all tasks begins in July 2024". No overall start date or test end date is given — [arXiv HTML](https://arxiv.org/html/2508.02739)
- An independent 2026 paper restates that the "Kronos pretraining corpus ends in June 2024" and chose its test window to avoid it — [arXiv 2607.26792](https://arxiv.org/html/2607.26792v1)
- Data provenance: a maintainer (2025-09-19) says the data comes from vendors and exchanges ("WIND, Tsanghi, Binance, etc.") and cannot be shared for licensing reasons. The evaluation code is "tightly coupled" to it and is not released — [Issue #100](https://github.com/shiyu-coder/Kronos/issues/100)
- Pretraining compute is not reported — [arXiv HTML](https://arxiv.org/html/2508.02739)

**Reported results [authors]** (test from July 2024; no end date given)
- Headline: +93% price-forecasting RankIC over the leading TSFM and +87% over the best non-pretrained baseline; 9% lower volatility MAE; 22% better synthetic-K-line fidelity — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/39730)
- Evaluation markets:
  - Stocks: XSHG, XNAS, XJPX, XNSE, XKRX, XHKG (in-distribution), and XIDX, XKLS, XTAI (out-of-distribution).
  - Crypto: all Binance spot pairs. FX: more than 1,000 pairs.
  - Daily setting: lookback 40, horizon 12 bars — [arXiv HTML Table 8](https://arxiv.org/html/2508.02739)
- Price forecasting, average IC / RankIC:
  - Kronos-S 0.0431 / 0.0254, B 0.0435 / 0.0258, L 0.0440 / 0.0267.
  - Best full-shot baseline: TimesNet 0.0317 / 0.0143.
  - Zero-shot TSFMs: Time-MoE-B 0.0174 / 0.0138, TimesFM 0.0052 / 0.0021, Chronos 0.0021–0.0060 IC, Moirai about −0.0005 IC — [arXiv HTML Tables 14–15](https://arxiv.org/html/2508.02739)
- Return forecasting, average IC / RankIC:
  - Kronos-S 0.0665 / 0.0622, B 0.0682 / 0.0634, L 0.0702 / 0.0675.
  - Best full-shot: DLinear 0.0439 / 0.0423.
  - Zero-shot: Time-MoE-S 0.0495 / 0.0477, Moirai 0.0527–0.0533 RankIC, Chronos-L 0.0357 / 0.0286 — [arXiv HTML Tables 16–17](https://arxiv.org/html/2508.02739)
- Realized volatility, MAE / R²: Kronos-S/B/L 0.0384 / 0.2490, 0.0372 / 0.2470, 0.0370 / 0.2624; GARCH 0.0405 / 0.2323; ARCH 0.0448 / 0.2281 — [arXiv HTML Table 18](https://arxiv.org/html/2508.02739)
- Investment simulation:
  - Setup: Qlib, Chinese A-shares, CSI300 (k=50, n=5) and CSI800 (k=200, n=10), top-k/drop-n, 5-day minimum hold, H=10, 90-day lookback, 0.15% cost.
  - Average AER / IR: S 0.1789 / 1.4222, B 0.1889 / 1.5217, L 0.2084 / 1.6491. No Sharpe; backtest dates not given — [arXiv HTML Table 10](https://arxiv.org/html/2508.02739)
- The paper says the finance-specific predecessors PLUTUS and DELPHYNE "[have not] released their code or models" — [arXiv HTML](https://arxiv.org/html/2508.02739)

**Fine-tuning code and known bugs**
- `finetune/` is a Qlib A-share pipeline (preprocess → `train_tokenizer.py` → `train_predictor.py` → `qlib_test.py` backtest, multi-GPU via torchrun) — [GitHub README](https://github.com/shiyu-coder/Kronos)
- `finetune_csv/` (added 2025-10-09) fine-tunes on any CSV of timestamps, open, high, low, close, volume, amount (volume and amount may be 0). It trains the tokenizer and then the predictor (`train_sequential.py`) — [finetune_csv README](https://github.com/shiyu-coder/Kronos/tree/master/finetune_csv)
- Issue #227 (2026-03-17): "the dataset leaks future information". Per-window normalization statistics were computed on the full window, including the forecast horizon — [Issue #227](https://github.com/shiyu-coder/Kronos/issues/227)
  - Fixed in `finetune/dataset.py` by commit 79d6d40 (2026-04-11), merged 2026-04-13 — [commit](https://github.com/shiyu-coder/Kronos/commit/79d6d403d8999de4aea8b388ca10905c8bf3de33)
  - `finetune_csv/finetune_base_model.py` on master (last changed 2025-10-09) still computes mean/std over the full lookback+predict+1 window as of 2026-10-18 — [source](https://github.com/shiyu-coder/Kronos/blob/master/finetune_csv/finetune_base_model.py)
  - Inference (`KronosPredictor`) normalizes only over the supplied history — [model/kronos.py](https://github.com/shiyu-coder/Kronos/blob/master/model/kronos.py)
- Issue #231 (2026-04-09): the greedy path called the integer `top_k` as a function. Fix merged 2026-04-13 — [Issue #231](https://github.com/shiyu-coder/Kronos/issues/231)
- A maintainer says the signals show "inversion/regression" (reversal) behaviour, lose mainly in trending markets, and that the model is at a "GPT-2" stage with no post-training — [Issue #33](https://github.com/shiyu-coder/Kronos/issues/33)

**Install and inference cost**
- `requirements.txt` has `torch>=2.0.0` and pins `pandas==2.2.2`, `matplotlib==3.9.3`, `einops==0.8.1`, `huggingface_hub==0.33.1`, `safetensors==0.6.2`. The README says Python 3.10+ — [requirements.txt](https://github.com/shiyu-coder/Kronos/blob/master/requirements.txt)
- None of these pins has a Python 3.14 wheel: pandas 2.2.2 covers cp39–cp312 only and matplotlib 3.9.3 stops at cp313 — [PyPI pandas 2.2.2](https://pypi.org/project/pandas/2.2.2/#files); [PyPI matplotlib 3.9.3](https://pypi.org/project/matplotlib/3.9.3/#files)
- No official VRAM or throughput figures. The weights are 4.1M–102.3M params, under 0.5 GB in fp32 — [GitHub README](https://github.com/shiyu-coder/Kronos)

**Independent and community evidence**
- [independent] Yu et al. (London Business School et al., arXiv 2607.26792, 2026-07-29):
  - Data: 12 large caps (BMW, Mercedes, VW, BP, PetroChina, Shell, Goldman, JPM, LSEG, Google, Meta, Microsoft), daily OHLCV 2016-07-23..2026-07-23. The test window 2025-01-17..2026-07-23 is after the Kronos cutoff.
  - Average quantile loss (AQL; a price-level quantile loss, not a directional or trading metric) after their KQSP fix: zero-shot Kronos 6.55 vs a per-stock trained ANN 6.56, not significantly different. MAE 15.73 vs 15.96.
  - Other TSFMs did worse: TabPFN 6.86, Chronos 7.68, Chronos-2 7.68, TimesFM-2.5 7.83, Time-MoE 7.92, Moirai 8.12, TimesFM 8.33 — [arXiv HTML](https://arxiv.org/html/2607.26792v1)
- [community] 2026-06-10, a reproducible Kronos-base test on BTCUSDT/HYPEUSDT direction at several horizons, plus 4,682 hourly 24h "probability up" forecasts taken from the public demo's git history. Directional accuracy was "close to random" and the probabilities were "somewhat overconfident" — [Issue #325](https://github.com/shiyu-coder/Kronos/issues/325)
- [community] 2026-05-24, after a month of tests and tuning, one user concludes that predicting with it is "worse than flipping a coin" (27 comments) — [Issue #299](https://github.com/shiyu-coder/Kronos/issues/299)
- [independent, negative] Mesfin (arXiv 2605.17724, 2026-05-18), a "Kronos-inspired" setup: GBM and LSTM on MNQ 5-min bars, 944 days 2021–2025, walk-forward. Nothing beat the 51.8% base rate (best 50.89%, permutation p=0.135) — [arXiv](https://arxiv.org/abs/2605.17724)
- Conflict: a 2026-02-22 practitioner blog says "Kronos is not yet publicly available". HF and GitHub show weights public since 2025, so the blog is wrong on this point — [Kinlay blog](https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/)

**Newer versions and follow-ups (through 2026-10-18)**
- No Kronos v2 or Kronos-large release was found (HF org, README). A user's request for Kronos-large access (Issue #325, 2026-06) has no visible maintainer reply — [HF NeoQuasar](https://huggingface.co/NeoQuasar); [Issue #325](https://github.com/shiyu-coder/Kronos/issues/325)
- KiT (Zhang & Li, arXiv 2609.34507, 2026-09-28) is a competitor K-line foundation model:
  - Model: flow-matching Diffusion Transformer; S 31.0M, M 101.0M, L 283.7M. Each bar becomes 5 log-ratio features (gap, body, two shadows, volume vs causal EMA).
  - Data: 3,914 A-shares, 888 US equities and 28 crypto, about 3B bars at 1m–1d, 2016-08-01..2026-04-10.
  - Split: train to 2023-12-31, validation 2024-01-01..2025-12-31, test 2026-01-01..2026-04-10. Trained on 128 H200s — [arXiv HTML](https://arxiv.org/html/2609.34507v1)
  - [authors] Return RankIC 0.0571 vs fine-tuned Kronos-base 0.0521; volatility RankIC 0.660 vs Sundial 0.528 and GARCH 0.522. A-share backtest 2025-05..2026-04, net of costs: +65.2%/yr (Sharpe 1.70) on 5-min bars.
  - The tables print RankICs with negative signs and some means don't match their rows. Code and weights are "will be available", i.e. not released — [arXiv HTML](https://arxiv.org/html/2609.34507v1)

### Inferences
- Contamination: the last trading day of June 2024 is Friday 2024-06-28, so at most the first holdout bar can be inside Kronos's pretraining data. The paper does not give the exact last date. Zero-shot Kronos on the user's 2024-06-28..2025-06-27 holdout is therefore effectively clean. Few other pretrained models can say this.
- Fit: Kronos takes OHLCV bars directly. Stooq has no "amount" column, and the model accepts it zero-filled. Kronos-mini's 2048-bar context is about 8 years of daily bars. Small and base are limited to 512 bars, about 2 years.
- The authors' evaluation is mostly intraday and non-US. Their only trading test uses CSI300/800 with 0.15% costs and gives no Sharpe. Nothing in the paper shows US daily bars surviving 5–20 bp costs.
- The leakage bug affects the user's own fine-tuning, not zero-shot inference. Use `finetune/dataset.py` from after 2026-04-11, or patch `finetune_csv` to normalize on the lookback only. The paper does not disclose whether the authors' own pretraining or Qlib backtests used the leaky normalization.
- Cost estimate (mine, not measured). Kronos-base is about 0.2 GFLOP per token, so about 105 GFLOP per 512-bar window per sample per step.
  - 5,000 symbols × 10 samples × 1 step ≈ 5.3 PFLOP. On an RTX 5080 (~56.3 TFLOPS FP32 peak — [Puget Systems](https://www.pugetsystems.com/parts/Video-Card/NVIDIA-GeForce-RTX-5080-16GB-16122/)) that is about 1.5 min at peak, or about 4–5 min per trading day at realistic utilization.
  - A 250-day holdout is about 15–20 GPU-hours for base and about 4–5 h for small. Multi-step horizons multiply this by roughly the number of steps, because there is no KV cache.

### Gaps
- The exact end date of the pretraining data within June 2024, the overall start date, the test-period end date, and the backtest window dates are not disclosed.
- No pretraining compute and no official VRAM or throughput figures.
- The pretraining data and evaluation code are unreleased (vendor licenses), so the authors' numbers cannot be reproduced exactly.
- No independent cost-aware US cross-sectional daily backtest of Kronos was found.
- The zero-shot volatility table (Table 19) and the synthetic-generation tables (Tables 20–21: discriminative score, TSTR IC/RankIC) were not extracted, so the comparators behind the "9% lower MAE" and "22% fidelity" headlines are unverified here.

## StockGPT (Dat Mai, 2024)

### Takeaway
StockGPT (arXiv 2404.05101; v1 2024-04-07, v3 2024-10-23) is a 0.93M-parameter nanoGPT-style decoder over 402 return-bin tokens. It was trained on CRSP NYSE daily returns 1926–2000 and tested 2001–2023. The headline 119%/yr, Sharpe 6.5 is a pre-cost, equal-weight, daily-rebalanced long-short. It shrinks to a Sharpe of 1.0 when value-weighted and 0.2 when value-weighted with a one-day skip. No code or weights were released, but the recipe is simple enough to rebuild.

### Cited Findings
- Versions v1 2024-04-07, v2 2024-04-09, v3 2024-10-23 (current). Single author. q-fin.CP. CC BY 4.0 — [arXiv](https://arxiv.org/abs/2404.05101)
- Tokenization: daily return × 10,000 → integer basis points → 50-bp bins closed on the right. Returns of −100% or worse go to one bin and returns above +100% are capped. That gives 402 tokens (indices 0–401), each represented by its bin midpoint. Example: (−2.4%, 0, 0, 5%, 4.8%) → (196, 200, 200, 210, 210) — [arXiv HTML v3](https://arxiv.org/html/2404.05101v3)
- Data: CRSP common stocks (share codes 10/11) on NYSE/AMEX/NASDAQ, about 70M observations.
  - Train about 50M (1926–2000): 1926–1990 for fitting, 1991–2000 for tuning and validation.
  - Test about 20M (2001–2023).
  - The main model is trained on NYSE only (validation loss 2.55 vs 2.72 for all exchanges) — [arXiv HTML v3](https://arxiv.org/html/2404.05101v3)
- Model: decoder-only, 4 blocks, 4 heads, embedding 128, context 256 days, dropout 0.2, about 0.93M parameters. Trained for 10,000 steps × batch 64 on a MacBook M2. Trained once and never retrained over the test period — [arXiv HTML v3](https://arxiv.org/html/2404.05101v3)
- Horizons: the daily model predicts t+1; a separate monthly model predicts the mean of the next 20 days — [arXiv HTML v3](https://arxiv.org/html/2404.05101v3)
- Daily results 2001–2023 [authors] — [arXiv HTML v3](https://arxiv.org/html/2404.05101v3):
  - Fama-MacBeth slope 0.50 (t=25.18).
  - Equal-weight long-short: 119.1%/yr, Sharpe 6.5, max drawdown −23.7%. With a $5 price filter: 73.8%, Sharpe 4.7.
  - Value-weight: 27.0%/yr, Sharpe 1.0.
  - Skip-one-day horizon: equal-weight 26.1% (Sharpe 1.7), value-weight 5.8% (Sharpe 0.2).
  - Author's worst case of 400% turnover × 5 bp: still 69%/yr equal-weight.
- Monthly results [authors]: equal-weight 13.5%/yr (Sharpe 0.9) vs the best factor (short-term reversal) at 8.8% (Sharpe 0.7). Alpha about 15–16%/yr vs 11 factors (t≈4.7) — [arXiv HTML v3](https://arxiv.org/html/2404.05101v3)
- No code or data availability statement. The paper thanks Karpathy for public GPT lecture code — [arXiv HTML v3](https://arxiv.org/html/2404.05101v3). GitHub repos named "StockGPT" are unrelated LLM chatbots — [GitHub search](https://github.com/search?q=StockGPT&type=repositories)
- Related independent result: on the same kind of data (CRSP daily excess returns, 2001–2023), every daily-rebalanced long-short strategy, models and benchmarks alike, turns negative at 20 bps. For example, CatBoost's average Sharpe falls from about 6.44 (0 bps) to −3.62 (20 bps) — [Rahimikia et al., arXiv PDF §5.2.4](https://arxiv.org/pdf/2511.18578)

### Inferences
- The decay pattern (equal-weight ≫ value-weight; the edge mostly gone when skipping one day) fits short-term reversal and bid-ask bounce in small caps. That is the same kind of edge the user's correlation2 project saw vanish under 5–20 bp costs.
- As a template for "next token = next return bin", StockGPT is the simplest fully reproducible design. It can be trained on Stooq data ending 2024-06-27 in hours on one GPU, with zero contamination. Expect its gross signal to sit mainly in illiquid names.

### Gaps
- No SSRN or journal version newer than arXiv v3 was found, and no independent replication.

## General time-series foundation models (Chronos, TimesFM, Lag-Llama, Moirai, Timer/Sundial, Time-MoE, TiRex, Toto, MOMENT, TimeGPT, 2025–2026 newcomers) and finance-specific TSFMs

### Takeaway
Only three general TSFM families tokenize values into a discrete vocabulary: Chronos-T5 (4,096 bins), Kronos (BSQ codes) and StockGPT (return bins). Every other model embeds continuous patches or points. Most of the decoder-only "GPT-style" models are next-patch predictors: TimesFM, Timer/Sundial, Moirai-MoE/2.0, Toto, Time-MoE (point-wise) and Lag-Llama (lag vectors). Finance is a negligible share of every general corpus. The only finance content is M4's finance category, Monash Bitcoin-influencer and FRED-MD sets, and exchange rates, except Sundial/Timer-S1, which add an undisclosed 10.5B-point "Finance (Ours)" set.

As of 2026-10-18 the newest open single models at the top of general leaderboards are Mistral's Timestral (2026-10-14), Google's TimesFM 3.0 (2026-08-24, non-commercial weights) and The Forecasting Company's t0-gamma (2026-10-12). None publishes a full data cutoff. Independent finance studies find near-zero or negative out-of-sample R² for every zero-shot TSFM on daily stock returns.

### Cited Findings

#### Catalog (one row per version; details and more sources in the family blocks below)

| Version | Weights public | Input representation | Architecture | Params | Pretraining data (size; latest disclosed date) | Finance series in corpus | Output | License |
|---|---|---|---|---|---|---|---|---|
| Kronos mini/small/base (large unreleased) | 2025-06-30/07-01 (updated 2025-09-09) | 1 BSQ token per OHLCVA bar (2×10-bit subtokens) | decoder-only | 4.1M / 24.7M / 102.3M (499.2M) | 12.1B bars, 45 exchanges; "up to June 2024" | yes: NYSE/Nasdaq stocks+ETFs from 2000, crypto, FX, futures | sampled paths | MIT [src](https://arxiv.org/html/2508.02739) |
| StockGPT | no weights (paper 2024-04-07) | 402 return-bin tokens (50 bp) | decoder-only | 0.93M | CRSP 1926–2000 | yes | categorical next token | n/a [src](https://arxiv.org/html/2404.05101v3) |
| Chronos-T5 tiny…large | 2024-02-21/28 | mean scaling + 4,094 uniform bins (4,096 vocab) | T5 encoder-decoder | 8M/20M/46M/200M/710M | ~84B obs + synthetic; latest dated set 2022 | M4 finance-category series | sampled (20 paths) | Apache-2.0 [src](https://arxiv.org/html/2403.07815) |
| Chronos-Bolt tiny…base | 2024-11-25 | patches of 16 | T5 enc-dec, direct multi-step | 9M/21M/48M/205M | "nearly 100B" obs; datasets undisclosed | not disclosed | 9 quantiles | Apache-2.0 [src](https://huggingface.co/amazon/chronos-bolt-base) |
| Chronos-2 (base; small; synth) | 2025-10-30; 2025-12-03; 2025-11-24 | patches of 16 + standardize→arcsinh | encoder-only, group attention, multivariate + covariates, ctx 8,192 | 120M; 28M | Chronos subset + GiftEvalPretrain subset + synthetic; no dates | M4 D/H/M/W (Bitcoin/FRED-MD unconfirmed) | 21 quantiles | Apache-2.0 [src](https://arxiv.org/html/2510.15821) |
| TimesFM 1.0 | 2024-05-03 | patches 32 in / 128 out, RevIN | decoder-only | 200M | Trends 2007–2021/22, Wiki pageviews 2012–Nov 2023, synthetic, M4 etc. | M4 | point (+experimental quantiles) | Apache-2.0 [src](https://arxiv.org/html/2310.10688) |
| TimesFM 2.0 | 2024-12-20 | patches 32/128 | decoder-only, ctx 2,048 | 500M | 1.0 set + 20 LOTSA sets; no cutoff stated | M4 | point + 10 exp. quantiles | Apache-2.0 [src](https://huggingface.co/google/timesfm-2.0-500m-pytorch) |
| TimesFM 2.5 | 2025-09-02 | patches 32, ctx 16k | decoder-only | 200M (+30M quantile head) | GiftEvalPretrain + Wiki (Nov 2023) + Trends (EoY 2022) + synthetic | via GiftEvalPretrain: Bitcoin, FRED-MD, M1/M3, NN5, CIF | quantiles to 1k horizon | Apache-2.0 [src](https://huggingface.co/google/timesfm-2.5-200m-pytorch) |
| TimesFM 3.0 | 2026-08-24 | patches 32 in / 64 out, ≤32 variates | decoder-only, variate attention, CPM | 330M | same sources (GiftEvalPretrain minus fev-bench overlap) + ">1 trillion" points total | same as 2.5 | point + 9 quantiles | non-commercial weights [src](https://huggingface.co/google/timesfm-3.0-pytorch) |
| Lag-Llama | 2024-02-07 | per-step vector of lagged values + date features | decoder-only (LLaMA) | 2.45M | 20 datasets (~352M windows), latest 2021 | none (Exchange Rate held out) | Student-t | Apache-2.0 [src](https://arxiv.org/html/2310.08278) |
| Moirai 1.0-R / 1.1-R | 2024-02-11 / 2024-06-14 | multi-size patches, instance norm | masked encoder, any-variate attention | 14M / 91M / 311M | LOTSA 27.6B obs (released Mar 2024) | Econ/Fin 0.10%: Bitcoin-influencers, FRED-MD, M1/M3/M4, NN5, Tourism, CIF, GoDaddy | mixture distribution | CC-BY-NC-4.0 [src](https://arxiv.org/html/2402.02592) |
| Moirai-MoE | 2024-11-01 | patch 16 | decoder-only MoE (32 experts, top-2) | 11M act/117M; 86M act/935M | LOTSA | as Moirai 1.0 | mixture | CC-BY-NC-4.0 [src](https://arxiv.org/html/2410.10469) |
| Moirai 2.0-R small | 2025-08-06 | patch 16 | decoder-only, quantile + multi-token prediction | 11.4M (paper also 87.1M, 305M unreleased) | GiftEvalPretrain + GIFT train + Chronos-mixup + KernelSynth + internal telemetry "starting from January 2024" (~1 yr) | GiftEvalPretrain finance + M4 | 9 quantiles | CC-BY-NC-4.0 [src](https://arxiv.org/html/2511.11698) |
| Timer / timer-base-84m | paper 2024-02; checkpoint n.d. | continuous patches of 96 | decoder-only GPT-style | ≤67M (paper); 84M | UTSD ≤1B pts; 260B (LOTSA+UTSD) for 84m | GoDaddy, Bitcoin* | point | Apache-2.0 [src](https://huggingface.co/thuml/timer-base-84m) |
| Timer-XL | arXiv 2024-10 | patches of 96 | decoder-only, TimeAttention, multivariate | not captured | UTSD + LOTSA (231B) + ERA5-Large | LOTSA sets | point | not captured [src](https://arxiv.org/html/2410.04803) |
| Sundial (Timer-3) | weights 2025-05 | patches of 16 | decoder-only + TimeFlow flow-matching head | 32M/128M/444M (only 128M public) | TimeBench 1,032B incl. "Finance (Ours)" 10.5B, undated | yes, undisclosed | generative samples | Apache-2.0 [src](https://arxiv.org/html/2502.00816) |
| Timer-S1 | 2026-03 | patches of 16 | decoder MoE + serial-token prediction | 8.3B total / 0.75B active | TimeBench + continued pretraining | yes, undisclosed | 9 quantiles | Apache-2.0 [src](https://huggingface.co/thuml/Timer-S1) |
| Time-MoE base/large (ultra unreleased) | 2024-10 | point-wise continuous tokens | decoder-only MoE | 50M act/113M; 200M act/453M (1.1B/2.4B) | Time-300B, 309B obs; finance 0.0001% | Exchange Rate, Bitcoin, FRED-MD, CIF, NN5, Tourism | point (Huber) | Apache-2.0 [src](https://arxiv.org/html/2409.16040) |
| TiRex 1.x | 2025-05-29 | patches of 32 | xLSTM (sLSTM) | 35M | Chronos+TSMixup (30M series), GP synthetic (15M), GiftEval-pretrain subset (2.5M) | none named | point + 9 quantiles | NXAI community license [src](https://arxiv.org/html/2505.23719) |
| TiRex-2 | 2026-07-01 | patches | xLSTM time mixer + attention variate mixer, covariates | 82.5M (38.4M univariate) | LOTSA + Chronos (decontaminated variants) | none named | quantiles | HF says apache-2.0; gated "Pro" tier reported [src](https://huggingface.co/NX-AI/TiRex-2) |
| Toto 1.0 | 2025-05-20 | patches of 64 | decoder-only, time + variate attention | 151M | ~2.36T pts: 43% Datadog observability, GIFT-Eval pretrain, Chronos, synthetic | none | Student-T mixture | Apache-2.0 [src](https://arxiv.org/html/2505.14766) |
| Toto 2.0 | 2026-05-19 | patches of 32 | decoder-only, u-μP | 4M/22M/313M/1B/2.5B | 5.04T pts, observability + synthetic only, "more recent" periods | none | 9 quantiles | Apache-2.0 [src](https://arxiv.org/html/2605.20119) |
| MOMENT | 2024-05-30 (PyPI) | patches of 8, 512 input | T5 encoder, masked reconstruction | 40M/125M/385M | Time-series Pile, 1.23B timestamps | Exchange-rate, Monash sets | point via trained linear head | MIT [src](https://arxiv.org/html/2402.03885) |
| TimeGPT-1 / -2 / -2.1 | API only (2023; 2025-10-14) | not disclosed | encoder-decoder transformer (v1) | not disclosed | >100B pts incl. "finance … banking", no dates; v2 undisclosed | yes, unnamed | conformal intervals | closed [src](https://arxiv.org/pdf/2310.03589) |
| TabPFN-TS (TabPFN v2/2.5/3/3.5 priors) | PyPI 2025-01-07 | tabular calendar features | TabPFN transformer | not disclosed | synthetic only (base priors) | none | point + probabilistic | code Apache-2.0; weights 2.5+ non-commercial [src](https://github.com/PriorLabs/tabpfn-time-series) |
| IBM TTM r2 / FlowState r1 | 2024 / 2025-08-07 | patches (MLP-Mixer) / no patching (SSM) | MLP-Mixer / SSM encoder + functional-basis decoder | ~1M / 9–18M | public sets (~700M samples) / GIFT-pretrain + Chronos subsets + synthetic | TTM: Bitcoin | point / quantiles | Apache-2.0 [src](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) |
| Timestral 5M/28M/123M/209M | 2026-10-14 | 32-step patches, ctx up to 32,768 | 24 layers, d=768 (209M) | 5M–209M | not captured | not captured | 21 quantiles | "Modified MIT" (no rights above $20M monthly revenue) [src](https://huggingface.co/mistralai/Timestral-209M) |
| t0 alpha/beta/gamma | 2026-09/10 | not captured | alternating time/variate attention | 102M / 256M / ~256M | "curated public data" + synthetic; no dates | not disclosed | quantiles | t0-gamma Apache-2.0 [src](https://arxiv.org/abs/2609.24559) |
| FinCast | 2025-08-20 (HF Vincent05R/FinCast; paper 2025-08-27) | patches + instance norm | decoder-only, sparse MoE (4 experts, top-2) | ~1B | ~20.5B points; dates not stated | yes: crypto, FX, futures, stocks, macro | point + quantiles | Apache-2.0 [src](https://arxiv.org/html/2508.19609v1) |
| FinText-TSFM (613 models) | 2025-11 | Chronos tokens / TimesFM patches | as base models | 8M–46M | daily excess returns 1990–2023, one model per year 2000–2023 | yes (equities) | as base | Apache-2.0 (card checked) [src](https://huggingface.co/FinText) |
| PFN timesfm-1.0-200m-fin | 2024-08-02 | patches (TimesFM 1.0) | decoder-only | 200M | +~90–100M finance points, series ending before 2023-01-01 | yes: S&P500, TOPIX500, FX, crypto, indices | point | CC-BY-NC-SA-4.0 [src](https://arxiv.org/html/2412.09880v1) |
| DELPHYNE (Bloomberg) | not released | patches of 32 | encoder, any-variate attention | 12 layers, d=768 | LOTSA + ~15% finance; finance data through end-2019 | yes | Student-T mixture | n/a [src](https://arxiv.org/html/2506.06288v1) |
| LENS (ex-PLUTUS) | not released | invertible patch embedding | encoder-decoder TimeFormer | 0.2B / 1B / 5B | 100B+ finance obs; dates not stated | yes | point | n/a [src](https://arxiv.org/html/2408.10111) |
| KiT | not released | 5 log-ratio features per bar | Diffusion Transformer (flow matching) | 31M / 101M / 284M | ~3B bars, 2016-08-01..2026-04-10 (train ≤2023-12-31) | yes | samples | n/a [src](https://arxiv.org/html/2609.34507v1) |

#### Amazon Chronos family
- Chronos-T5 tokenization: mean scaling s = mean(|x|) over the context, then uniform quantization into 4,094 bins over [−15, 15] (plus PAD and EOS, 4,096 tokens). The config has context 512, prediction 64, 20 samples, top-k 50 — [arXiv HTML](https://arxiv.org/html/2403.07815); [config](https://huggingface.co/amazon/chronos-t5-large/raw/main/config.json)
- Chronos-T5 corpus: 28 datasets, about 890K series, about 84B observations, plus 10M TSMixup and 1M KernelSynth series.
  - Dated spans: Mexico City Bikes 2010–2022, Wiki Daily 2007–2022, Weatherbench 1979–2018.
  - Exchange Rate (1990–2016), FRED-MD, NN5 and CIF were used for evaluation only — [arXiv HTML](https://arxiv.org/html/2403.07815)
- Chronos-T5 training cost: one 8×A100 node; mini 7.68 h ($252) up to large 63.05 h ($2,066). The paper was published in TMLR 2024 — [arXiv HTML](https://arxiv.org/html/2403.07815); [TMLR](https://mlanthology.org/tmlr/2024/ansari2024tmlr-chronos)
- Chronos-Bolt: patch 16, context 2,048, 9 quantiles. [authors] "Up to 250 times faster and 20 times more memory-efficient" than Chronos-T5 — [HF card](https://huggingface.co/amazon/chronos-bolt-base). Maintainer (2025-10-21): "Currently we are not planning to release the training data for Chronos-Bolt" — [HF discussion](https://huggingface.co/datasets/autogluon/chronos_datasets/discussions/4)
- Chronos-2 data: the 23 real datasets in Table 6 have no date columns. The corpus does "not overlap with the test portions of any GIFT-Eval task" but has "partial overlap with the training portions" of some GIFT-Eval datasets — [arXiv HTML](https://arxiv.org/html/2510.15821)
- Chronos-2 speed and history: [authors] about 300 series/s on one A10G (batch 1,024, context 2,048, horizon 64). arXiv 2510.15821 v1 is dated 2025-10-17; README "20 Oct 2025: Chronos-2 released" — [arXiv HTML](https://arxiv.org/html/2510.15821); [GitHub](https://github.com/amazon-science/chronos-forecasting)
- Fine-tuning: official `scripts/training/train.py` for Chronos-T5 — [GitHub scripts](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts). AutoGluon `fine_tune=True` for Chronos-2, LoRA by default (search summary) — [AutoGluon source](https://auto.gluon.ai/stable/_modules/autogluon/timeseries/models/chronos/chronos2.html)
- Package: `chronos-forecasting` 2.3.2 (2026-09-08), Python >=3.10, `torch<3,>=2.2`, `transformers<6,>=4.41` — [PyPI](https://pypi.org/pypi/chronos-forecasting/json)

#### Google TimesFM family
- TimesFM 1.0 corpus:
  - Google Trends: ~22k queries, 2007–2021 (hourly 2018–2019), about 0.5B points.
  - Wikipedia pageviews: Jan 2012–Nov 2023, about 300B points.
  - 3M synthetic series, plus M4, Electricity, Traffic, Weather, Favorita, LibCity.
  - 20 layers, d=1280 — [arXiv HTML](https://arxiv.org/html/2310.10688)
- The 2.5 and 3.0 cards list "Wikimedia Pageviews, cutoff Nov 2023", "Google Trends top queries, cutoff EoY 2022", GiftEvalPretrain and synthetic/augmented data — [TimesFM 2.5 card](https://huggingface.co/google/timesfm-2.5-200m-pytorch); [TimesFM 3.0 card](https://huggingface.co/google/timesfm-3.0-pytorch)
- TimesFM 3.0 (blog 2026-08-31): 330M, multivariate with past and future covariates, ">1 trillion time points" (sources not broken down), available in BigQuery `AI.FORECAST`. Weights are under `timesfm-non-commercial-license-v1.0` — [Google Research blog](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/); [GitHub](https://github.com/google-research/timesfm)
- Fine-tuning: an official LoRA/PEFT example (2026-04-09) at `timesfm-forecasting/examples/finetuning/`. 1.0/2.0 weights need `timesfm==1.3.0` (Python <3.12) — [GitHub](https://github.com/google-research/timesfm); [PyPI 1.3.0](https://pypi.org/pypi/timesfm/1.3.0/json)
- Speed: [authors] TimesFM 3.0 on MLX (Apple M4 Max) runs 666 series/s at batch 32, context 512, horizon 64. No GPU figures — [GitHub](https://github.com/google-research/timesfm)

#### Lag-Llama
- Design: 8 layers, 9 heads, context 32 points plus lags (quarterly…second). Robust median/IQR scaling and a Student-t head. Fine-tuning scripts (2024-04-16) recommend sweeping context 32–1024 — [arXiv HTML](https://arxiv.org/html/2310.08278); [GitHub](https://github.com/time-series-foundation-models/lag-llama)
- [authors] Exchange Rate CRPS: zero-shot 0.011, fine-tuned 0.009, best supervised 0.008 — [arXiv HTML](https://arxiv.org/html/2310.08278)

#### Salesforce Moirai family
- LOTSA: 27,646,462,733 observations; Econ/Fin is 0.10% (23 datasets, 24.9M obs). Monash "Bitcoin" is 18 daily influencer series (hash rate, tweets…), not prices — [arXiv HTML](https://arxiv.org/html/2402.02592); [Zenodo](https://zenodo.org/records/5121965)
- [authors] On in-distribution Monash Bitcoin, Moirai's MAE (~1.6–1.9E+18) is worse than naive (7.78E+17) — [arXiv HTML](https://arxiv.org/html/2402.02592)
- Moirai 2.0 internal data, verbatim: "derived from internal salesforce telemetry data… at a daily granularity… covers a period of approximately one year, starting from January 2024" (~2.15M series, ~1.48B obs) — [arXiv HTML](https://arxiv.org/html/2511.11698)
- Also found: `Salesforce/moirai-agent` (2025-12-29, 3B, qwen2-based, CC-BY-NC) and a gated `moirai-1.5-llama-kvcache` (2025-06-26). No Moirai 2.1/3.0 — [HF model listing](https://huggingface.co/api/models?author=Salesforce&search=moirai)
- Package: `uni2ts` 2.0.0 pins `torch<2.5` and `numpy~=1.26.0` — [PyPI](https://pypi.org/pypi/uni2ts/json)

#### THUML Timer / Timer-XL / Sundial / Timer-S1
- Sundial's TimeBench (1,032B points) includes "Finance (Ours)" (10.5B, 1.02%) and "IoT (Ours)" (5.8B), with no description or dates. ERA5 makes up most of the corpus (Daily 406B) — [arXiv HTML](https://arxiv.org/html/2502.00816)
- Sundial release: weights on HF 2025-05, ICML 2025 Oral. Fine-tuning code is "on its way". Speed on a CPU (M1 Pro): 249 ms for 672→16, 949 ms for 2880→720 with 20 samples — [GitHub](https://github.com/thuml/Sundial); [HF card](https://huggingface.co/thuml/sundial-base-128m)
- Timer-S1 (paper 2026-03-05): 8.3B total / 0.75B active parameters, "at least 40GB VRAM" recommended. [authors] GIFT-Eval MASE 0.693, CRPS 0.485 — [HF card](https://huggingface.co/thuml/Timer-S1); [arXiv HTML](https://arxiv.org/html/2603.04791)

#### Time-MoE
- Time-300B: 48.2M series, 309.09B observations, 90.5% Nature. The Finance domain is 1,715 series / 413,696 obs: Exchange Rate, CIF 2016, Bitcoin, FRED-MD, NN5, Tourism — [arXiv HTML](https://arxiv.org/html/2409.16040)
- Fine-tuning: official scripts on JSONL `{"sequence": [...]}`, including a `--from_scratch` flag. Base inference memory is 226.70 MB (BF16) and training memory 1.77 GB; pretraining used 128 A100-80G — [GitHub](https://github.com/Time-MoE/Time-MoE); [arXiv HTML](https://arxiv.org/html/2409.16040)

#### TiRex
- [authors] On an A40 (ctx 2048, pred 32), TiRex is ">11×" faster than TimesFM-2.0, ">4×" faster than Chronos-Bolt-Base and ">2176×" faster than TabPFN-TS — [arXiv HTML](https://arxiv.org/html/2505.23719)
- No self-service fine-tuning ("contact us"). TiRex-2 fine-tuning is a "Pro" feature. The CUDA kernels need compute capability ≥8.0 — [GitHub](https://github.com/NX-AI/tirex); [HF TiRex](https://huggingface.co/NX-AI/TiRex); [HF TiRex-2](https://huggingface.co/NX-AI/TiRex-2)

#### Toto
- Toto 1.0: 151M parameters, patch 64, context 4,096, Student-T mixture output, Apache-2.0, no documented fine-tuning — [HF card](https://huggingface.co/Datadog/Toto-Open-Base-1.0)
- Toto 2.0 (2026-05-19): 4M–2.5B. "Public data is excluded from pretraining" (base checkpoints). The 2.5B-FT checkpoint adds GIFT-Eval pretrain and train data — [arXiv HTML](https://arxiv.org/html/2605.20119)

#### MOMENT
- Encoder-only masked reconstruction; forecasting needs a trained linear head. Large took 404 GPU-hours on one RTX A6000. Official fine-tuning and PEFT tutorials exist — [arXiv HTML](https://arxiv.org/html/2402.03885); [HF card](https://huggingface.co/AutonLab/MOMENT-1-large)

#### Nixtla TimeGPT
- Closed weights, API or enterprise self-hosting only. TimeGPT-2 (2025-10-14) comes in mini, standard and pro variants and discloses no training data. Search-summary pricing: $0 for 50 calls/mo, $99 for 10k, $399 for 100k — [Nixtla blog](https://www.nixtla.io/blog/timegpt-2-announcement); [pricing page (404 when fetched)](https://docs.nixtla.io/docs/getting-started-timegpt_subscription_plans_and_pricing)

#### Finance-specific pretrained models (beyond Kronos and StockGPT)
- FinCast (Univ. of Sydney; arXiv 2508.19609, 2025-08-27; CIKM 2025):
  - Model: decoder-only, sparse MoE, about 1B parameters, with learnable frequency embeddings. Loss combines Huber, trend-consistency, quantile and MoE-auxiliary terms ("PQ-loss").
  - Data: about 20.5B points (Table 1), about 2.2M series (2.4M in the text), seconds to monthly. The date range is not stated.
  - Compute: 8× H200, 147,152 steps. It runs at full precision on an RTX 4060 8 GB — [arXiv HTML](https://arxiv.org/html/2508.19609v1)
- FinCast results [authors]:
  - Zero-shot average MSE / MAE: 0.1644 / 0.2397, vs TimesFM-500M 0.2411 / 0.2836, Chronos-small 0.1860 / 0.2537 and Time-MoE-large 0.1858 / 0.2571. Benchmark: 3,632 series; test periods not stated.
  - Supervised: US_71 (2016-01-04..2023-12-29) and US_14L (2005-01-04..2023-12-29); fine-tuned 0.2971 / 0.3505 vs PCIE 0.3261 / 0.3736 — [arXiv HTML](https://arxiv.org/html/2508.19609v1)
- FinCast release: weights on HF `Vincent05R/FinCast`, Apache-2.0, Python 3.11+ and PyTorch 2.5+. LoRA/DoRA PEFT is "done" but untested. Last push 2026-06-22 — [GitHub](https://github.com/vincent05r/FinCast-fts); [GitHub API](https://api.github.com/repos/vincent05r/FinCast-fts)
- FinText-TSFM (Rahimikia, Ni & Wang; arXiv 2511.18578, 2025-11-23):
  - Data: daily excess returns 1990–2023, 94 countries, about 2B observations (CRSP + Compustat Global).
  - Training: yearly expanding-window pretraining of Chronos tiny/mini/small and TimesFM 8M/20M from scratch. About 50,000 GPU-hours on GH200 — [arXiv HTML](https://arxiv.org/html/2511.18578v1)
  - Release: 613 models (360 on HF), in US, Global and Augmented collections, one per year 2000–2023, named `Family_Size_Year_Collection` — [HF FinText](https://huggingface.co/FinText)
  - Example card `FinText/TimesFM_20M_2023_Augmented`: apache-2.0, 19.8M params, data "up to the year 2023" — [HF card](https://huggingface.co/FinText/TimesFM_20M_2023_Augmented)
- PFN TimesFM-fin (Fu (MIT), Hirano & Imajo (Preferred Networks); arXiv 2412.09880, 2024-12-13):
  - Method: continual pretraining of `google/timesfm-1.0-200m` on Yahoo Finance + Binance data (S&P500, TOPIX500, FX, commodities, indices, crypto), hourly and daily. Loss is MSE on log prices. Series end before 2023-01-01; test is 2023 onward.
  - Release: weights `pfnet/timesfm-1.0-200m-fin` (created 2024-08-02, cc-by-nc-sa-4.0) and code `pfnet-research/timesfm_fin` — [arXiv HTML](https://arxiv.org/html/2412.09880v1); [HF API](https://huggingface.co/api/models/pfnet/timesfm-1.0-200m-fin)
- DELPHYNE (Ding (CMU), Mittal & Gopal (Bloomberg); arXiv 2506.06288, 2025-05-12; NeurIPS 2025 GenAI-in-Finance workshop):
  - Model: encoder with any-variate attention, patch 32, Student-T mixture head. Trained on LOTSA (~85% of the mix) plus ~15% finance (FX, commodities, 10,511 stocks, 28,837 ETFs, 5-min bars of 15,817 securities). Finance data is "only through the end of 2019".
  - [authors] Best zero-shot NLL on 14 SPX stocks, 2021-01-04..2023-12-29. Not released (per Kronos) — [arXiv HTML](https://arxiv.org/html/2506.06288v1); [NeurIPS](https://neurips.cc/virtual/2025/132496)
- LENS, formerly PLUTUS (HKUST-GZ; arXiv 2408.10111, v1 2024-08-19, v3 2025-10-21): encoder-decoder, 0.2B / 1B / 5B, 100B+ finance observations (TAQ, CRSP daily, options, futures…). It calls itself "open-source" but gives no repository — [arXiv HTML](https://arxiv.org/html/2408.10111)

#### Independent evaluations on stock, crypto and FX data (metric, horizon, test period)
- Rahimikia, Ni & Wang (arXiv 2511.18578; US daily excess returns, next day, OOS 2001–2023, equal-weight daily-rebalanced decile long-short, pre-cost):
  - Main zero-shot results: Chronos-large R²_OOS −1.37% (window 512, 20.17%/yr); TimesFM-500M −2.80% (−1.47%/yr). CatBoost: −0.03%, 51.16% accuracy, Sharpe 6.79 at window 252 — [arXiv HTML](https://arxiv.org/html/2511.18578v1)
  - Extended zero-shot tables (B.4–B.7) cover Chronos-Bolt T/M/S/B, TimesFM 2.5, Moirai 1.1 S/B/L, Moirai 2, Kairos 10/23/50M, MOMENT S/B/L, Lag-Llama, TiRex, FlowState, TTM, Toto and Sundial. Every model has negative R²_OOS at every window, and full-sample directional accuracy is about 48.2–51.4% — [arXiv PDF appendix](https://arxiv.org/pdf/2511.18578)
  - Long-short Sharpe at window 512 (pre-cost): Toto 6.22, Sundial 4.21, MOMENT-L 3.95, Lag-Llama 3.87, TiRex 3.61, Moirai 2 2.32, TimesFM 2.5 2.15, Kairos-50M −0.58, Chronos-Bolt-Base −0.64, Moirai-S −3.48, TTM −4.31 (CatBoost 6.46). Toto's R²_OOS is −114% — [arXiv PDF appendix](https://arxiv.org/pdf/2511.18578)
  - Pretraining from scratch on finance data: Chronos-small Sharpe 5.42 (36.84%/yr) and TimesFM-20M 3.66 at window 512. With global data + JKP factors + augmentation, Chronos-small reaches 51.74% accuracy and Sharpe 6.78. Fine-tuning public models "mostly deteriorated" — [arXiv HTML](https://arxiv.org/html/2511.18578v1)
  - Costs (Table 12): at 20 bps every daily long-short strategy has a negative Sharpe (CatBoost average 6.44 → −3.62; at 40 bps −13.59). The best TSFM under mixed costs (~11–21 bps), JKP-augmented TimesFM-20M, is still at −1.87 — [arXiv PDF §5.2.4](https://arxiv.org/pdf/2511.18578)
  - The authors caution that public TSFMs "may" carry look-ahead bias from their pretraining data (footnote 17) — [arXiv PDF](https://arxiv.org/pdf/2511.18578)
- Noguer i Alonso & Pereira Franklin (arXiv 2606.27100, 2026-06-25):
  - Setup: Yahoo daily adjusted closes for AAPL, AMZN, GOOG, JPM and META, 2014-09-15..2026-02-15. Training to 2024-08-21, evaluation from 2024-08-22, H=20 business days, 10 rolling origins, L=512, MAE skill vs a zero-return random walk.
  - Results: TSFMs (TimeGPT, TimesFM-2.5, Moirai-2.0, Chronos, Chronos-2) win 8 of 10 tasks against NBEATS, NHITS, PatchTST, iTransformer and KAN. Gains over the random walk are "small and sparse" (skill about 10⁻³); the DM test rejects only for Chronos/AMZN and Moirai-2.0/GOOG. Contamination is not discussed — [arXiv](https://arxiv.org/abs/2606.27100); [arXiv HTML](https://arxiv.org/html/2606.27100)
- Yu et al. (arXiv 2607.26792, 2026-07-29), 12 large caps, daily OHLCV, test 2025-01-17..2026-07-23: Kronos zero-shot matches a per-stock ANN on AQL (6.55 vs 6.56). Generic TSFMs trail (Chronos-2 7.68, TimesFM-2.5 7.83, Moirai 8.12) — [arXiv HTML](https://arxiv.org/html/2607.26792v1)
- Valeyre & Aboura (arXiv 2412.09394, 2024-12-12): chronos-t5-tiny zero-shot, 100-day context, next-day residual returns of about 550 large US stocks (CRSP; FF/PCA/IPCA residuals), test 2002–2016.
  - Gross Sharpe 1.84 / 3.17 / 1.34, which turns negative at 3 bp slippage. Fine-tuned up to 3.97.
  - Comparison: CNN-Transformer 3.15 / 5.01 / 4.29; short-term reversal 2.23 / 4.16 / 2.31 — [arXiv HTML](https://arxiv.org/html/2412.09394v1)
- Łaniewski & Ślepaczuk (Univ. of Warsaw, ISD 2025): Chronos on Nasdaq-100 and S&P 500 daily, 1995–early 2025, rolling window. Zero-shot accuracy is "statistically comparable" to the best baselines, but trading performance "lags top benchmarks". Fine-tuning was worse than zero-shot — [AIS eLibrary](https://aisel.aisnet.org/isd2014/proceedings2025/datascience/18)
- Brini (arXiv 2607.05291, 2026-07-06): 9 zero-shot TSFMs vs 8 HAR-type models on realized volatility of 50 assets (equities, FX, futures), 3 horizons. Only TTM beats Log-HAR at every horizon, and narrowly; most short-horizon gains vanish after Mincer-Zarnowitz recalibration — [arXiv](https://arxiv.org/abs/2607.05291)
- Goel et al.:
  - Realized volatility (arXiv 2505.11163, 2025-05-16): zero-shot TimesFM is a "reasonable baseline". Incremental fine-tuning is "essential" and then beats econometric models (DM/GW tests) [their claim] — [arXiv](https://arxiv.org/abs/2505.11163)
  - VaR on the S&P 100 (arXiv 2410.11773): "zero-shot use is not optimal". The fine-tuned model beats GARCH on actual-over-expected ratios — [arXiv](https://arxiv.org/abs/2410.11773)
- Marconi (arXiv 2507.07296, 2025-07-09): TTM and Chronos on 10-year yield changes, EUR/USD volatility and equity spreads. TTM transfers well, but traditional models match or beat it in 2 of 3 tasks — [arXiv](https://arxiv.org/abs/2507.07296)
- Das, Goyal & Yadav (arXiv 2605.21504, 2026-05-08): Chronos-2 on Magnificent-7 equities and Treasury rates, monthly rolling 2000–2025 (RMSE/MAPE). Multivariate beats univariate; mixing equities and rates hurts — [arXiv](https://arxiv.org/abs/2605.21504)
- FinTSB (arXiv 2502.18834 v3, 2026-06-01): Chinese A-shares, 20 datasets, TopK-Drop with 0.1% fee. Chronos-bolt-base has the best ARR (0.366) and ASR (4.327); Localformer has the best IC (0.044); "No single model wins on every metric" — [arXiv HTML](https://arxiv.org/html/2502.18834)
- PFN TimesFM-fin [authors of the fine-tune]:
  - Market-neutral Sharpe, zero-cost PnL, test 2023+. S&P500: fine-tuned 1.68 vs original TimesFM 0.42, AR(1) 1.58, random 0.03. TOPIX500: 1.06 vs −1.75. Currencies: 0.25 vs AR(1) 0.88.
  - Break-even cost for S&P500: 0.60% — [arXiv HTML](https://arxiv.org/html/2412.09880v1)
- Coolsaet, Gallardo & Gao (arXiv 2605.23962, 2026-05-12): pretraining a transformer on the TSX index improves per-stock direction BCE from 0.69 to 0.64. XGBoost and ensembles still earn higher average daily returns — [arXiv](https://arxiv.org/abs/2605.23962)
- Weak evidence (search summaries only):
  - Lag-Llama on 3 Indonesian stocks: fine-tuned CRPS 0.0195 — [UTP KHub](https://khub.utp.edu.my/cris/display/n72840)
  - STQA benchmark: Time-MoE close-price MRE 0.0799 — [arXiv](https://arxiv.org/pdf/2609.06117)

### Inferences
- Anything that "predicts the next token" over prices must pick a value representation. The menu is:
  - discrete bins of a scaled value (Chronos-T5) or of returns (StockGPT);
  - learned vector-quantized codes of a whole bar (Kronos);
  - continuous patch regression with point or quantile heads (most others);
  - flow or diffusion heads (Sundial, KiT).
- The independent finance evidence does not separate these representations. Every zero-shot model has negative R² on next-day returns. Pre-cost portfolio Sharpe varies widely (−4 to +6), which mostly reflects rank-ordering and reversal capture, not accuracy.
- The strongest pre-cost portfolio signals (Toto, Sundial, Lag-Llama, TiRex at window 512) come from models that are bad at the level forecast (R² ≪ 0). They are probably cross-sectional reversal detectors. The cost analysis in the same study implies those signals do not survive 20 bps at daily turnover.
- For the user's standard ($10k account, SPY/QQQ and random-turnover-matched controls), published TSFM numbers cannot be transferred. Each candidate has to be run through the same harness.

### Gaps
- Not captured or not disclosed:
  - Timestral's training data and architecture details.
  - t0's HF repo IDs other than t0-gamma.
  - Timer-XL's parameter count.
  - Kairos weights and license.
  - TiRex-2's final license (HF apache-2.0 vs a reported gated "Pro" split).
- Missing benchmark data: numeric GIFT-Eval/fev-bench positions for FinCast, Kronos and the FinText models (none submitted, as far as found).
- No independent cost-aware evaluation of TimesFM 3.0, Timestral, t0, Toto 2.0 or TiRex-2 on equities (all released ≤8 weeks before this research, except Toto 2.0 and TiRex-2).

## Order-book / order-flow generative models (LOBS5, MarketGPT, MarS LMM, TRADES, TradeFM): what they generate, data needs, relevance to daily bars

### Takeaway
None of these models can be used with daily Stooq bars. All of them generate order or trade messages, need message-level feeds (LOBSTER L3, Nasdaq TotalView-ITCH, proprietary tick data, or Chinese A-share orders), and forecast seconds to minutes ahead. Their value to the user is design ideas: composite per-event tokens, asset-ID tokens, tokenizers fitted on early data only, and the "autoregressive trap" warning about free-running rollouts.

### Cited Findings
- LOBS5 (Nagy et al., arXiv 2309.00638, v1 2023-08-23; ICAIF '23)
  - What it generates: tokenized LOBSTER L3 messages (10 levels, 4 event types) that drive a JAX order-book simulator.
  - Tokenization: vocabulary of 12,011, with 22 tokens per message (digits grouped in threes; price as signed ticks from mid).
  - Model: 6 S5 state-space layers, about 6.3M params, context 500 messages (about 11k tokens).
  - Data: GOOG and INTC. Train 2022-07-01..2022-11-11, test 2022-12-14..2022-12-30.
  - Results: per-token perplexity 3.63 (GOOG) and 4.04 (INTC). Correlation of generated vs realized mid-returns ρ≈0.1–0.2 over about 100 messages — [arXiv HTML](https://arxiv.org/html/2309.00638)
  - Release: code github.com/peernagy/LOBS5 (MIT). No pretrained weights; data must be bought from LOBSTER — [GitHub API](https://api.github.com/repos/peernagy/LOBS5); [GitHub](https://github.com/peernagy/LOBS5)
- LOB-Bench (Nagy et al., ICML 2025, PMLR 267)
  - Scaled LOBS5 (~35M) beats a parametric baseline, a CGAN and RWKV-4/6 (~170M) on distributional metrics.
  - Errors grow with rollout length (the "autoregressive trap"), and generated data did not significantly improve a downstream mid-price-trend predictor.
  - Compute: LOBS5 took about 30.4 L40 GPU-days — [PMLR](https://proceedings.mlr.press/v267/nagy25a.html); [arXiv HTML](https://arxiv.org/html/2502.09172)
- MarketGPT (Wheeler & Varner, arXiv 2411.16585, 2024-11-25)
  - Model: GPT-2-style decoder, 12 layers, d=768, about 100M params.
  - Tokenization: 24 tokens per message, vocabulary 12,111 including 98 ticker IDs. Context 10,368 tokens.
  - Data: Nasdaq TotalView-ITCH 5.0, 8 days for 20 assets (91.1M messages, 2.19B tokens), fine-tuned on AAPL. Ran on one RTX 4090.
  - Results: about 7% of generated messages were invalid. It reproduces volatility clustering, but kurtosis is 4.93±1.99 vs 2.44 empirical — [arXiv HTML](https://arxiv.org/html/2411.16585)
  - Release: code MIT; weights at huggingface.co/aaronwheeler/MarketGPT-100m — [GitHub](https://github.com/aaron-wheeler/MarketGPT)
- Microsoft MarS / Large Market Model (arXiv 2409.07486, v1 2024-09-04, v2 2025-03-13; ICLR 2025)
  - Data: order-level data for the 500 most liquid Chinese stocks, 2017–2023.
  - Tokens: an order token (type × 32 price bins × 32 volume bins × 16 interval bins = 49,152 combinations) and a per-minute "order image" VQGAN token (codebook 8,192).
  - Models: an order model of 2M–1.02B and an order-batch model of 150M–3B, LLaMA2-style. The token count conflicts within the paper (16B vs 32B/10B) — [arXiv HTML](https://arxiv.org/html/2409.07486v2)
  - Release: only the 2M/5M/10M order models are public (HF Don-Don/mars-order-*, MIT; 10M created 2026-08-31). Larger models are "pending release approval". Running it needs Docker, Ray Serve and "complete order-level historical data" — [GitHub](https://github.com/microsoft/MarS); [HF API](https://huggingface.co/api/models/Don-Don/mars-order-10m)
- TRADES / DeepMarket (Berti, Prenkaj, Velardi; ECAI 2025; arXiv 2502.07071)
  - Model: a transformer DDPM that generates LOBSTER order tuples, conditioned on the last 256 orders plus snapshots. Parameter count not stated.
  - Data: TSLA and INTC, January 2015 (~24M samples).
  - Speed: one simulated hour takes about 6 h on an RTX 3090.
  - Release: code MIT; checkpoints on Google Drive — [arXiv HTML](https://arxiv.org/html/2502.07071v3); [GitHub](https://github.com/LeonardoBerti00/DeepMarket)
- TradeFM (J.P. Morgan AI Research; arXiv 2602.23784, 2026-02-27; ICML 2026)
  - Model: 524.4M-param Llama-style decoder (32 layers, context 1,024).
  - Tokenization: one composite token per trade event from a 16,384 vocabulary (interarrival × price-depth × volume × side × action), with scale-invariant features.
  - Data: 9K+ US equities, Feb 2024–Sep 2025, 19B+ tokens of proprietary tick trades; test from Jan 2025. Trained on 3× A100.
  - Release: code and weights not stated as released — [arXiv HTML](https://arxiv.org/html/2602.23784v1); [ICML](https://icml.cc/virtual/2026/71494)

### Inferences
- Relevance to daily OHLCV is none as models. The design lessons do transfer:
  - (1) One composite token per daily bar built from a few coarse bins of scale-invariant features (return, range, gap, log-volume ratio), as in TradeFM, with bin edges fitted on pre-cutoff data only.
  - (2) Asset-ID tokens for training on many symbols at once (MarketGPT).
  - (3) Judge the model on one-step-ahead forecasts re-anchored on real data each day, not on long free-running generations (LOB-Bench).
  - (4) Realistic stylized facts do not imply predictive skill.

### Gaps
- Not released or not disclosed: weights for LOBS5, TradeFM and the larger MarS models; MarketGPT's HF weight license.

## Forecasting benchmarks (GIFT-Eval, fev-bench, TIME, BOOM, FinTSB): 2025–2026 standings and finance content

### Takeaway
The big general benchmarks barely measure finance. GIFT-Eval's Econ/Fin domain is six anonymized M4 subsets. fev-bench's 12 Econ tasks are monthly or quarterly macro aggregates (FRED-MD/QD includes the S&P 500 level and VIX). The leakage-aware TIME benchmark has one Crypto dataset. None contains identifiable daily stock, FX or crypto price series, so leaderboard rank says almost nothing about daily-return skill. As observed 2026-10-18, the top open single models are Timestral-209M/123M (Mistral), t0-gamma, TimesFM-3 and Chronos-2. GIFT-Eval's overall top 10 is dominated by "agentic" ensembles.

### Cited Findings
**GIFT-Eval (Salesforce; arXiv 2410.10393)**
- Its Econ/Fin datasets are exactly M4 Yearly, Quarterly, Monthly, Weekly, Daily and Hourly (all univariate, short-horizon configs only) — [dataset_properties.csv](https://huggingface.co/spaces/Salesforce/GIFT-Eval/blob/main/results/dataset_properties.csv)
- M4's Finance category has 24,534 series (Daily 1,559), anonymized with no tickers — [M4-info.csv](https://github.com/Mcompetitions/M4-methods/blob/master/Dataset/M4-info.csv)
- Snapshot 2026-10-17/18: 153 models, 97 configs. Types: 48 zero-shot, 43 pretrained, 38 agentic, 11 deep-learning, 7 fine-tuned, 6 statistical — [HF Space config](https://salesforce-gift-eval.hf.space/config)
- Default overall top 10 by MASE_Rank (observed 2026-10-18), MASE / CRPS:
  1. STRIDE w/ Synapse (Google Cloud): 0.625 / 0.423, agentic
  2. limix_moe: 0.654 / 0.448, agentic
  3. EXAONE-Forecast-Agent (LG): 0.610 / 0.419, agentic
  4. OpenTSLM-TeeMoE: 0.656 / 0.449, agentic
  5. TW3Cast: 0.651 / 0.451, agentic
  6. LS-MoE: 0.650 / 0.445, agentic
  7. PERSONA-AI-T4: 0.651 / 0.444, agentic
  8. Timestral-209M: 0.658 / 0.446, zero-shot
  9. IK-QuadTSFM: 0.663 / 0.454, agentic
  10. LS-Agent: 0.655 / 0.448, agentic

  Six of the top 10 have no replication code — [HF Space config](https://salesforce-gift-eval.hf.space/config)
- Best single foundation models with no test leakage (MASE / CRPS):
  - Timestral-209M 0.658 / 0.446
  - Timestral-123M 0.659 / 0.448
  - TimesFM-3 0.667 / 0.456
  - t0-gamma 0.661 / 0.452
  - Timestral-28M 0.678 / 0.461
  - Toto-2.0-2.5B-FT 0.679 / 0.463

  [HF Space config](https://salesforce-gift-eval.hf.space/config)
- Familiar models, overall position and MASE / CRPS: Toto-2.0-2.5B #48 (0.696 / 0.476), Chronos-2 #56 (0.698 / 0.485), Timer-S1 #66, TimesFM-2.5 #70 (0.705 / 0.490), TiRex #73, Moirai2 #87, Seasonal_Naive #127 — [HF Space config](https://salesforce-gift-eval.hf.space/config)
- Econ/Fin (M4) tab: the top 10 are all agentic. The best single no-leak models are Timestral-123M, t0-gamma and Timestral-209M. Reference CRPS / MASE: Chronos-2 0.735 / 0.772, TimesFM-2.5 0.756 / 0.781, Auto_ARIMA 0.873 / 0.866. The leak-flagged timesfm_2_0_500m has the best Econ/Fin scores of all entries — [HF Space config](https://salesforce-gift-eval.hf.space/config)
- Leakage handling:
  - "TestData Leakage" is a binary column, added 2025-08-05. The "zero-shot" type requires no leakage and no GIFT-Eval train split.
  - 20 entries self-flag leakage, including Chronos (small/base/large), chronos_bolt (small/base), TimesFM 1.0/2.0, TTM, Lag-Llama and MoiraiAgent-leaking — [about.py](https://huggingface.co/spaces/Salesforce/GIFT-Eval/blob/main/src/about.py); [results](https://huggingface.co/spaces/Salesforce/GIFT-Eval/tree/main/results)
  - GIFT-Eval Pretrain is 230B points, 4.5M series and "no leakage" with the test split. It includes bitcoin_with_missing and fred_md but no stock datasets — [HF dataset](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)

**fev-bench (Shchur et al., arXiv 2509.26468; v1 2025-09-30 … v4 2026-06-30)**
- 100 tasks (46 with covariates) in 7 domains — [arXiv](https://arxiv.org/abs/2509.26468); [task_groups.py](https://huggingface.co/spaces/autogluon/fev-bench/blob/main/src/task_groups.py)
- The Econ group has 12 tasks: australian_tourism, fred_md_2025/cee and /macro, fred_qd_2025/cee and /macro, gvar, us_consumption ×3, world_co2_emissions, world_life_expectancy, world_tourism — [task_groups.py](https://huggingface.co/spaces/autogluon/fev-bench/blob/main/src/task_groups.py)
- The FRED macro targets include `S&P 500`, `VIXCLSx` and `EXUSUKx` (monthly/quarterly) — [arXiv](https://arxiv.org/abs/2509.26468); [task_groups.py](https://huggingface.co/spaces/autogluon/fev-bench/blob/main/src/task_groups.py); [tasks.yaml](https://github.com/autogluon/fev/blob/main/benchmarks/fev_bench/tasks.yaml)
- SQL leaderboard recomputed from the repo at commit f9e2c7d (2026-10-16), win % / skill %:
  1. Timestral-209M 89.2 / 50.4
  2. Timestral-123M 88.5 / 50.2
  3. t0-gamma 86.7 / 49.4
  4. Timestral-28M 81.5 / 48.4
  5. TimesFM-3 80.7 / 49.0
  6. Chronos-2 73.5 / 47.3
  7. timeh2o-v1-tiny 72.9 / 45.5
  8. Timestral-5M 71.9 / 46.2
  9. t0-beta 70.6 / 46.7
  10. Toto-2.0-2.5B 70.2 / 44.4

  The closed-API TimeRSI would rank 3rd if included. Skill scores match Mistral's published card; win rates are not cross-checked against the live app — [fev results](https://github.com/autogluon/fev/tree/main/benchmarks/fev_bench/results); [Timestral card](https://huggingface.co/mistralai/Timestral-209M)
- Leakage handling: a flagged `trained_on_this_dataset` result is replaced with Chronos-Bolt's score — [analysis.py](https://github.com/autogluon/fev/blob/main/src/fev/analysis.py)

**TIME (Qiao et al., arXiv 2602.12147, v2 2026-02-27; described as ICML 2026 in secondary mentions, venue not verified)**
- 50 freshly sourced datasets and 98 tasks for leakage-free zero-shot evaluation — [arXiv HTML](https://arxiv.org/html/2602.12147v2)
- Finance datasets: Crypto, US Term Structure, Oil Price; no stocks or FX — [arXiv HTML](https://arxiv.org/html/2602.12147v2)
- Top 3 in the paper: Chronos-2, TimesFM-2.5, TiRex. Leaderboard: HF Space Real-TSF/TIME-leaderboard — [arXiv HTML](https://arxiv.org/html/2602.12147v2)

**BOOM (Datadog)**: an observability-only benchmark (350M observations, 2,807 series), with no finance — [arXiv](https://arxiv.org/abs/2505.14766)

**FinTSB (arXiv 2502.18834 v3)**: a finance-specific benchmark on Chinese A-shares (20 datasets × 300 stocks × 250 days; TopK-Drop, 0.1% fee). Chronos-bolt-base has the best ARR and ASR; Localformer has the best IC; "No single model wins on every metric" — [arXiv HTML](https://arxiv.org/html/2502.18834)

**Leakage literature**
- Meyer et al. (arXiv 2510.13654 v3, 2026-02-25) distinguish direct leakage (reused datasets) from indirect leakage (correlated series over overlapping periods).
  - Documented leakage gave "over 50% better test scores".
  - In a COVID stock experiment, MAE was about 190 vs about 217 when 7 of 484 series leaked.
  - They recommend a model-aware temporal cutoff: test only after the model's last pretraining timestamp — [arXiv HTML](https://arxiv.org/html/2510.13654v3)

### Inferences
- A model's GIFT-Eval or fev-bench rank should not be used to pick a stock model. The finance content is anonymized M4 series or monthly macro aggregates, scored against seasonal naive, which rewards trend and seasonality extrapolation that daily returns do not have.
- Leakage flags are self-declared. The newest leaders (Timestral, t0, TimesFM-3) were all released after the user's holdout and do not publish full data cutoffs. Treat them as unverified for a 2024–25 test.

### Gaps
- No identifiable daily-equity benchmark exists among the major leaderboards.
- Not retrieved: live TIME leaderboard standings and the TIME Crypto dataset details.
- fev-bench win rates come from a recomputation, not the live app.

## Contamination risk: does each model's pretraining data overlap the 2024-06-28..2025-06-27 holdout or later?

### Takeaway
Only a handful of pretrained models are provably clean for the user's holdout. They are clean either because their weights were public before 2024-06-28, or because their disclosed cutoff precedes it:
- Released before the holdout: Chronos-T5, TimesFM 1.0, Lag-Llama, Moirai 1.0/1.1, MOMENT, and the Timer paper models.
- Disclosed cutoff before the holdout: Kronos (data "up to June 2024"), PFN TimesFM-fin (finance data before 2023), FinText-TSFM vintages ≤2023, and TabPFN-TS base priors (synthetic only).

Moirai 2.0 is the only model whose disclosed corpus overlaps the holdout (non-financial telemetry from Jan 2024 onward). Sundial/Timer-S1 (an undisclosed 10.5B-point "Finance (Ours)" set), TimeGPT (closed, finance included, cannot be pinned) and FinCast (finance data, no dates) are the highest-risk. Most 2025–2026 models (Chronos-Bolt, Chronos-2, TimesFM 2.0/2.5/3.0, TiRex, Time-MoE, Timestral, t0) are "not disclosed", though most list only public archives assembled before mid-2024.

### Cited Findings

| Model / version | Weights public | Latest disclosed real-data date | Finance in corpus | Verdict vs holdout | Source |
|---|---|---|---|---|---|
| Kronos mini/small/base | 2025-06-30/07-01 (updated 2025-09-09) | "up to June 2024" | yes (stocks incl. NYSE/Nasdaq, crypto, FX, futures) | Essentially no (≤1 bar, 2024-06-28, possible) | [arXiv](https://arxiv.org/html/2508.02739) |
| KiT | not released | data 2016-08..2026-04; train ≤2023-12-31; validation 2024–2025 | yes | train clean; model selection used holdout period | [arXiv](https://arxiv.org/html/2609.34507v1) |
| FinCast | 2025-08-20 (HF) | not disclosed | yes | Not disclosed (high risk) | [arXiv](https://arxiv.org/html/2508.19609v1) |
| FinText-TSFM `_2023` vintages | 2025-11 | "up to the year 2023" | yes (excess returns) | No | [HF card](https://huggingface.co/FinText/TimesFM_20M_2023_Augmented) |
| PFN timesfm-1.0-200m-fin | 2024-08-02 | finance series end before 2023-01-01 | yes | No | [arXiv](https://arxiv.org/html/2412.09880v1) |
| DELPHYNE | not released | finance "through the end of 2019" | yes | No (but no weights) | [arXiv](https://arxiv.org/html/2506.06288v1) |
| StockGPT | no weights | CRSP to 2000 (train) | yes | No (rebuild yourself) | [arXiv](https://arxiv.org/html/2404.05101v3) |
| Chronos-T5 | 2024-02-21/28 | 2022 | M4 finance category | No | [HF API](https://huggingface.co/api/models?author=amazon&search=chronos) |
| Chronos-Bolt | 2024-11-25 | not disclosed (data won't be released) | not disclosed | Not disclosed | [HF discussion](https://huggingface.co/datasets/autogluon/chronos_datasets/discussions/4) |
| Chronos-2 / small / synth | 2025-10-30 / 12-03 / 11-24 | no dates; listed real sources pre-2024 | M4 (+ possibly GiftEvalPretrain Bitcoin/FRED-MD) | Not disclosed; likely no (synth = synthetic only) | [arXiv](https://arxiv.org/html/2510.15821) |
| TimesFM 1.0 | 2024-05-03 | Wiki pageviews Nov 2023 | M4 | No | [arXiv](https://arxiv.org/html/2310.10688) |
| TimesFM 2.0 | 2024-12-20 | 1.0 set + LOTSA subsets; no cutoff | M4 | Not disclosed; likely no | [HF card](https://huggingface.co/google/timesfm-2.0-500m-pytorch) |
| TimesFM 2.5 | 2025-09-02 | Wiki Nov 2023; Trends EoY 2022; GiftEvalPretrain; synthetic | GiftEvalPretrain finance (Bitcoin-influencer, FRED-MD, …) | No for listed sources | [HF card](https://huggingface.co/google/timesfm-2.5-200m-pytorch) |
| TimesFM 3.0 | 2026-08-24 | same; ">1T points" not broken down | same | Not fully disclosed | [HF card](https://huggingface.co/google/timesfm-3.0-pytorch) |
| Lag-Llama | 2024-02-07 | 2021 | none in pretraining | No | [arXiv](https://arxiv.org/html/2310.08278) |
| Moirai 1.0-R / 1.1-R | 2024-02-11 / 2024-06-14 | 2023 (LOTSA released Mar 2024) | LOTSA Econ/Fin | No (1.0's weights file was re-uploaded 2024-08-22 as a "compatibility fix", so it is cleared on its data, not on the file date) | [HF API](https://huggingface.co/api/models?author=Salesforce&search=moirai) |
| Moirai-MoE | 2024-11-01 | LOTSA | LOTSA Econ/Fin | Likely no | [arXiv](https://arxiv.org/html/2410.10469) |
| Moirai 2.0-R small | 2025-08-06 | internal telemetry ~Jan 2024–~Jan 2025 | GiftEvalPretrain finance + M4 | Yes (temporal overlap; non-financial) | [arXiv](https://arxiv.org/html/2511.11698) |
| Timer (paper) / timer-base-84m / Timer-XL | Feb 2024 / n.d. / Oct 2024 | not disclosed | GoDaddy, Bitcoin*; LOTSA sets | No / likely no | [HF card](https://huggingface.co/thuml/timer-base-84m) |
| Sundial / Timer-S1 | 2025-05 / 2026-03 | not disclosed | "Finance (Ours)" 10.5B, undescribed | High risk (not disclosed) | [arXiv](https://arxiv.org/html/2502.00816) |
| Time-MoE | 2024-10 | not disclosed | Exchange Rate, Bitcoin, FRED-MD, CIF, NN5 | Likely no | [arXiv](https://arxiv.org/html/2409.16040) |
| TiRex 1.x / TiRex-2 | 2025-05 / 2026-07 | not disclosed | none named | Likely no | [arXiv](https://arxiv.org/html/2505.23719) |
| Toto 1.0 / Toto 2.0 | 2025-05 / 2026-05 | undated Datadog observability ("more recent" for 2.0) | none | Strict temporal overlap likely (non-financial); finance no | [arXiv](https://arxiv.org/html/2605.20119) |
| MOMENT | 2024-05-30 (PyPI) | Pile, pre-2024 | Exchange-rate, Monash | No | [PyPI](https://pypi.org/pypi/momentfm/json) |
| TimeGPT-1/-2/-2.1 | API | not disclosed | "finance … banking" | High risk; the server model cannot be pinned | [arXiv](https://arxiv.org/pdf/2310.03589) |
| TabPFN-TS base priors | 2025–2026 | synthetic only | none | No (by construction); TS-3 finetune not disclosed | [GitHub](https://github.com/PriorLabs/tabpfn-time-series) |
| TTM r1/r2/r2.1, FlowState | 2024 / 2025-08 | not disclosed | Bitcoin (TTM) | Likely no | [HF card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) |
| Timestral, t0, Kairos | 2026-10 / 2026-09–10 / 2025 | not disclosed | Kairos: unnamed "financial … signals" | Not disclosed | [Timestral](https://huggingface.co/mistralai/Timestral-209M); [t0](https://arxiv.org/abs/2609.24559) |

- Independent finance studies whose test windows overlap the holdout use post-holdout models without discussing contamination. Example: Noguer i Alonso & Pereira Franklin evaluate from 2024-08-22 with TimeGPT, Chronos-2, Moirai-2.0 and TimesFM-2.5 — [arXiv](https://arxiv.org/abs/2606.27100)
- Rahimikia et al. built their own yearly-cutoff models specifically because public TSFMs "may" carry look-ahead bias (footnote 17) — [arXiv PDF](https://arxiv.org/pdf/2511.18578)
- Leakage literature recommends testing only after the model's last pretraining timestamp, and treating correlated series over the same period as indirect leakage — [Meyer et al.](https://arxiv.org/html/2510.13654v3)
- Weak evidence (search summary only): a 2026 study of finance TSFMs trained on annual data vintages across 14 equity markets reportedly found post-origin exposure did not by itself inflate accuracy — [arXiv PDF](https://arxiv.org/pdf/2609.20554)

### Inferences
- For a strict test, use only models in the "No" rows, or models the user pretrains or fine-tunes on data ending 2024-06-27. Of those, Kronos is the only one pretrained on stock price bars. The FinText 2023 vintages and PFN TimesFM-fin are the only ones pretrained on equity returns or prices.
- "Not disclosed" models with pre-2024 public sources (Chronos-2, TimesFM-2.5, TiRex, Time-MoE) are probably clean of price leakage, but they cannot be certified. Run them as secondary controls and report them that way.
- Stooq's survivorship bias (only names listed in 2025) is a separate look-ahead channel. It applies to any model the user trains or evaluates and is not fixed by a clean pretraining cutoff.

### Gaps
- No model except Kronos, PFN, FinText, DELPHYNE and Moirai 2.0 states an explicit real-data end date. Per-dataset date ranges for LOTSA, GiftEvalPretrain and Chronos-2's table are unpublished.
- The content of TimeBench's "Finance (Ours)", TimeGPT-2's data, and Timestral's and t0's corpora are undisclosed.

## Feasibility on the user's machine (RTX 5080 16 GB sm_120, Python 3.14, 61 GB RAM)

### Takeaway
PyTorch itself is not the blocker. PyPI has cp314 Linux wheels from torch 2.9.0 (2025-10-15) through 2.14.1 (2026-09-30), and Blackwell has been supported since 2.7 with CUDA 12.8+. The blockers are model packages pinning old dependencies. Use one uv-managed Python 3.11/3.12 venv per model. Chronos-2/Chronos, TimesFM 2.5/3.0, toto-2, tirex-ts and tabpfn-time-series should resolve on 3.14 (untested). Every relevant open model except Timer-S1 fits in 16 GB for inference.

### Cited Findings
- PyTorch 2.7 added Blackwell support with prebuilt CUDA 12.8 wheels (prototype) — [PyTorch blog](https://pytorch.org/blog/pytorch-2-7/)
- Python 3.14 wheels and CUDA versions: torch 2.9.0 (2025-10-15) has cp314 manylinux x86_64 wheels built against CUDA runtime 12.8.90. 2.14.1 (2026-09-30) depends on `nvidia-cudnn-cu13` — [PyPI torch](https://pypi.org/project/torch/#files)
- A PyTorch developer notice (search snippet) says 2.14 is the last release with CUDA 12.x wheels and that CUDA 13 builds cover sm_100 and sm_120 — [dev-discuss](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
- Blocked or pinned on Python 3.14:
  - Kronos: pandas==2.2.2 (≤cp312), matplotlib==3.9.3 (≤cp313) — [requirements.txt](https://github.com/shiyu-coder/Kronos/blob/master/requirements.txt)
  - timesfm 1.3.0 (needed for the 1.0/2.0 weights): Python <3.12 — [PyPI](https://pypi.org/pypi/timesfm/1.3.0/json)
  - uni2ts / Moirai: torch<2.5 (no sm_120) and numpy~=1.26 — [PyPI](https://pypi.org/pypi/uni2ts/json)
  - Lag-Llama: gluonts ≤0.14.4 needs numpy 1.x — [PyPI gluonts](https://pypi.org/pypi/gluonts/0.14.4/json)
  - momentfm: numpy==1.25.2, transformers==4.33.3 — [PyPI](https://pypi.org/pypi/momentfm/json)
  - granite-tsfm: Python <3.14 — [PyPI](https://pypi.org/pypi/granite-tsfm/json)
  - toto-ts: torch==2.7.0, numpy==1.26.4 — [PyPI](https://pypi.org/pypi/toto-ts/json)
  - Timer/Sundial cards and the Time-MoE README: transformers==4.40.1, whose tokenizers 0.19 wheels stop at cp312 — [PyPI tokenizers 0.19.1](https://pypi.org/pypi/tokenizers/0.19.1/json)
- Unpinned enough for Python 3.14 (per package metadata):
  - chronos-forecasting 2.3.2 (Python >=3.10, torch<3,>=2.2) — [PyPI](https://pypi.org/pypi/chronos-forecasting/json)
  - timesfm 2.0.2 / 3.0.2 — [PyPI](https://pypi.org/pypi/timesfm/3.0.2/json)
  - toto-2 2.0.0 (Python >=3.12) — [PyPI](https://pypi.org/pypi/toto-2/json)
  - tirex-ts 1.4.2 — [PyPI](https://pypi.org/pypi/tirex-ts/json)
  - tabpfn-time-series 1.3.0 — [PyPI](https://pypi.org/pypi/tabpfn-time-series/json)
- VRAM and throughput reported by authors:
  - Time-MoE base: 226.70 MB inference (BF16), 1.77 GB training — [arXiv](https://arxiv.org/html/2409.16040)
  - FinCast (1B): runs on an RTX 4060 8 GB at full precision — [arXiv](https://arxiv.org/html/2508.19609v1)
  - Timer-S1: "at least 40GB VRAM" recommended — [HF card](https://huggingface.co/thuml/Timer-S1)
  - Chronos-2: about 300 series/s on an A10G — [arXiv](https://arxiv.org/html/2510.15821)
  - MarketGPT (100M): trained and sampled on one RTX 4090 — [arXiv](https://arxiv.org/html/2411.16585)
- The TiRex CUDA kernels need compute capability ≥8.0; sm_120 is untested. TiRex-2 ships cu128 environments — [HF TiRex](https://huggingface.co/NX-AI/TiRex); [HF TiRex-2](https://huggingface.co/NX-AI/TiRex-2)

### Inferences
- Practical recipe (not tested here):
  - `uv python install 3.12`, then one venv per model, with torch ≥2.7 from a cu128+ (or default CUDA 13) index.
  - Check that `torch.cuda.get_arch_list()` includes sm_120 before benchmarking.
  - Kronos runs in a 3.12 venv, or on 3.14 if pandas/matplotlib are unpinned.
- Compute is not the binding constraint. A StockGPT-size model (0.93M) or a Kronos-small fine-tune trains in hours. Cross-sectional inference over about 5–11k symbols per day is minutes for Chronos-2/TimesFM-2.5 and about 4–5 min per day for Kronos-base at N=10 samples (my estimate, from the Kronos section).

### Gaps
- No package was actually installed or benchmarked on this machine. sm_120 kernels for TiRex/xLSTM and FlashAttention are unverified.

## Ranked shortlist: models most worth trying for daily US stock data (Stooq OHLCV, holdout 2024-06-28..2025-06-27)

### Takeaway
Ranked by worth testing under the user's own harness, not by expected profit. The best independent evidence says daily long-short TSFM signals die at about 20 bps (Rahimikia et al.), and Chronos-tiny's dies at 3 bp (Valeyre & Aboura). That matches the user's correlation2 result.
1. **Kronos-small/base** (zero-shot, then fine-tuned): the only open, finance-pretrained, OHLCV-token LLM with a disclosed cutoff before the holdout.
2. **A self-trained StockGPT-style GPT** on Stooq bars ending 2024-06-27: the actual project goal, and contamination-free by construction.
3. **FinText-TSFM 2023 vintages**: the only public pretrained models trained on daily equity returns with yearly cutoffs.
4. **Chronos-2 and TimesFM-2.5** as general-purpose controls.
5. **Toto 1.0** as a zero-shot cross-sectional ranker.

### Cited Findings
- **1. Kronos-small/base** (MIT; 24.7M/102.3M; context 512; mini 4.1M has context 2048)
  - Pretrained on 12.1B bars including NYSE/Nasdaq stocks and ETFs from 2000, with data "up to June 2024" — [arXiv HTML](https://arxiv.org/html/2508.02739)
  - Has CSV fine-tuning code (patch the normalization leak first) — [finetune_csv](https://github.com/shiyu-coder/Kronos/tree/master/finetune_csv); [Issue #227](https://github.com/shiyu-coder/Kronos/issues/227)
  - It is the best zero-shot TSFM in the one independent post-cutoff daily-stock test (AQL 6.55 vs ANN 6.56; Chronos-2 7.68). AQL is a price-level quantile loss, not a directional or trading metric — [arXiv 2607.26792](https://arxiv.org/html/2607.26792v1)
  - Counter-evidence: community reports near-random direction on crypto — [Issue #325](https://github.com/shiyu-coder/Kronos/issues/325). Maintainers say its signals are reversal-like — [Issue #33](https://github.com/shiyu-coder/Kronos/issues/33)
- **2. Self-trained GPT on discretized returns** (StockGPT recipe: 402 × 50-bp tokens, 4 layers, d=128, context 256, 0.93M params, trained on a laptop) — [arXiv HTML v3](https://arxiv.org/html/2404.05101v3)
  - Optional upgrades from the LOB literature: a composite per-bar token (TradeFM) and ticker-ID tokens (MarketGPT) — [TradeFM](https://arxiv.org/html/2602.23784v1); [MarketGPT](https://arxiv.org/html/2411.16585)
  - Ready-made from-scratch pipelines exist: Time-MoE `--from_scratch` and Kronos `finetune_csv` — [Time-MoE](https://github.com/Time-MoE/Time-MoE)
  - Finance-only pretraining with yearly cutoffs is what closed the gap to CatBoost in the largest study — [arXiv HTML](https://arxiv.org/html/2511.18578v1)
- **3. FinText-TSFM `_2023` checkpoints** (TimesFM 8M/20M, Chronos tiny/mini/small; US / Global / Augmented; Apache-2.0)
  - Data "up to the year 2023", so they are clean for the holdout — [HF FinText](https://huggingface.co/FinText); [HF card](https://huggingface.co/FinText/TimesFM_20M_2023_Augmented)
  - [authors of the suite] JKP-augmented TimesFM-20M has a pre-cost Sharpe of 7.68 at window 512, and the least-negative net Sharpe under mixed costs (−1.87) — [arXiv PDF Table 12](https://arxiv.org/pdf/2511.18578)
- **4. Chronos-2 (120M; small 28M) and TimesFM-2.5 (200M)**
  - Apache-2.0; their packages should resolve on Python 3.14 per package metadata (untested) — [PyPI chronos](https://pypi.org/pypi/chronos-forecasting/json); [PyPI timesfm](https://pypi.org/pypi/timesfm/2.0.2/json)
  - Fine-tuning: LoRA paths exist — [TimesFM GitHub](https://github.com/google-research/timesfm)
  - Benchmarks: top-3 on the leakage-aware TIME benchmark — [TIME](https://arxiv.org/html/2602.12147v2)
  - Data: TimesFM-2.5 lists real-data cutoffs of Nov 2023 and EoY 2022 — [HF card](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
  - Finance evidence is weak: TimesFM-2.5 R²_OOS < 0 with Sharpe 2.15 pre-cost — [arXiv PDF](https://arxiv.org/pdf/2511.18578); Chronos-2 AQL trails Kronos — [arXiv 2607.26792](https://arxiv.org/html/2607.26792v1)
- **5. Toto 1.0** (151M, Apache-2.0)
  - Best zero-shot long-short in Rahimikia's extended tables (Sharpe 6.22 pre-cost at window 512, vs CatBoost 6.46), despite R²_OOS of −114% — [arXiv PDF](https://arxiv.org/pdf/2511.18578)
  - No finance data in its corpus, but its Datadog data is undated — [arXiv HTML](https://arxiv.org/html/2505.14766)
  - The toto-ts pins require a Python 3.12 venv — [PyPI](https://pypi.org/pypi/toto-ts/json)
- **Not shortlisted, and why:**
  - TimeGPT: closed and unpinnable; finance data with undisclosed dates — [arXiv](https://arxiv.org/pdf/2310.03589)
  - Sundial and Timer-S1: undisclosed "Finance (Ours)" data; Timer-S1 needs about 40 GB — [arXiv](https://arxiv.org/html/2502.00816); [HF](https://huggingface.co/thuml/Timer-S1)
  - Moirai 2.0: holdout-period telemetry, CC-BY-NC, and uni2ts is blocked on sm_120 — [arXiv](https://arxiv.org/html/2511.11698); [PyPI](https://pypi.org/pypi/uni2ts/json)
  - FinCast: no data dates; 1B params — [arXiv](https://arxiv.org/html/2508.19609v1)
  - KiT, DELPHYNE, LENS: no weights — [KiT](https://arxiv.org/abs/2609.34507); [Kronos paper](https://arxiv.org/html/2508.02739)
  - LOB models (LOBS5, MarketGPT, MarS, TRADES, TradeFM): need message-level data — [MarS](https://github.com/microsoft/MarS)
  - Timestral, t0, TimesFM-3: post-holdout releases with undisclosed or partial cutoffs; Timestral has a revenue-cap license and TimesFM-3 a non-commercial one — [Timestral license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE); [TimesFM GitHub](https://github.com/google-research/timesfm)

### Inferences
- Order of work that fits the user's standard:
  - (a) Run Kronos-small zero-shot on the holdout, one-step-ahead re-anchored daily, into the existing $10k / SPY / QQQ / random-turnover harness with 5–20 bp costs.
  - (b) Fine-tune Kronos on Stooq bars ending 2024-06-27, with the leak patched.
  - (c) Train the StockGPT-style GPT from scratch on the same cutoff.
  - (d) Use FinText-2023, Chronos-2 and TimesFM-2.5 as controls.
  - (e) Always include CatBoost and short-term-reversal baselines. These were the strongest non-TSFM baselines in the independent studies.
- Make turnover a design variable, not an afterthought. Every published daily-rebalanced edge here goes negative at 20 bps or less, so weekly/monthly horizons or large-cap-only universes are where any surviving signal would have to appear. This is an inference from the cited cost results, not something shown for these models.

### Gaps
- No study has run any of these models through a test equivalent to the user's: a $10k long-only account vs SPY/QQQ and turnover-matched random accounts, 2024-06-28..2025-06-27. Their usefulness on that test is unknown until it is run.
