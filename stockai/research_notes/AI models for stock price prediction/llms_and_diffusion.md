# Text LLMs and diffusion/flow-matching models vs small autoregressive time-series transformers for stock return prediction (research notes, current as of 2026-10-18)

Labels used below. **[AUTHORS]** means a number reported by a method's own authors on their own setup. **[INDEPENDENT]** means the result comes from a third party's benchmark, ablation, replication or critique. **[SECONDARY]** means news coverage or a search-engine summary or snippet, not the primary document. Dates are arXiv submission or revision dates unless stated otherwise. User context assumed throughout: one RTX 5080 (16 GB VRAM), Stooq daily OHLCV to 2025-06-27, no news/text data, held-out test year 2024-06-28..2025-06-27, and a prior project (correlation2) that found no edge after 5-20 bps costs.

## 1. Text LLMs applied to numeric price series (LLMTime, Time-LLM, GPT4TS/One Fits All, LLM4TS, Tan et al. NeurIPS 2024, and 2025-2026 follow-ups): do they add value over small purpose-built models, and at what cost?

### Takeaway
The strongest independent evidence (Tan et al., NeurIPS 2024) finds that the language model is the dead weight in published "LLM-for-time-series" forecasters. Deleting it, or replacing it with a single attention layer, matched or beat the full model in most of 26 comparisons per method, while cutting parameters by up to ~30,000x and, for Time-LLM, training time from 3-50 hours to 1-2 minutes. A February 2026 preprint argues LLMs do help in large cross-domain settings. A controlled study revised in March 2026 finds LLM backbones "do not consistently surpass" models pretrained on time series. On stock returns specifically, a June 2026 benchmark found even purpose-built time-series foundation models beat a random walk only rarely ("small and sparse" gains).

### Cited Findings
**Mechanisms and authors' own claims**
- LLMTime (Gruver, Finzi, Qiu, Wilson; arXiv v3 revised 2024-08-12). Prices are written as text: digits separated by spaces for GPT-3, values separated by commas, decimal point dropped (e.g., 0.123, 1.23, 12.3 becomes " 1 2 , 1 2 3 , 1 2 3 0 , 1 2 3 0 0"). The LLM samples ~20 continuations, and the pointwise median is the forecast. — [LLMTime](https://arxiv.org/html/2310.07820)
- LLMTime practical limits. GPT-3 was not run on the Informer datasets "because of the cost of API queries". LLaMA-2 70B was limited to a 4,096-token context. Only ~300 observations fit in context for the Darts datasets, and GPT-3 history was truncated to 500 observations on Monash. — [LLMTime](https://arxiv.org/html/2310.07820)
- LLMTime finds GPT-4 forecasts worse than GPT-3 ("GPT-4 has a forecasting error (CRPS) significantly larger than GPT-3's"), and chat/RLHF-tuned LLaMA-2 models worse than base models. Appendix B uses a 2022 TSMC closing-price series (246 observations) as a post-training-cutoff test. — [LLMTime](https://arxiv.org/html/2310.07820)
- Time-LLM (Jin et al.; arXiv v2 2024-01-29). It splits a normalized series into patches. Cross-attention maps them onto learned "text prototypes" built from the frozen Llama-7B's word embeddings, and a "Prompt-as-Prefix" of dataset description and statistics is prepended. Only the input reprogramming and output projection are trained (<6.6M trainable parameters). **[AUTHORS]** claims include 12% and 20% average gains over GPT4TS and TimesNet, 1.4% MSE reduction vs PatchTST, and SOTA in "36 out of 40" cases. — [Time-LLM](https://arxiv.org/html/2310.01728)
- Time-LLM hardware. "All experiments conducted on NVIDIA A100-80G GPUs." In its Table 7 (ETTh1), the full Llama variant uses 32,136-39,004 MiB of GPU memory at 0.517-0.697 s/iteration. The reprogramming-only variant uses 3,678-4,176 MiB. — [Time-LLM](https://arxiv.org/html/2310.01728)
- GPT4TS / "One Fits All" (Zhou et al.; NeurIPS 2023 Spotlight). This is a "Frozen Pretrained Transformer": the self-attention and feed-forward blocks of a pretrained language or image model stay frozen, and other parts are fine-tuned. **[AUTHORS]** claim comparable or SOTA results across forecasting, classification, anomaly detection and few-shot tasks. — [One Fits All](https://arxiv.org/abs/2302.11939)
- LLM4TS (Chang et al.; ACM TIST 2025, arXiv v6 2025-02-20). It uses two-stage fine-tuning: first align a pretrained LLM to time series, then fine-tune for forecasting. **[AUTHORS]** claim it beats from-scratch models in full-shot settings and ranks first in few-shot, with ablations showing the pretrained weights matter. — [LLM4TS](https://arxiv.org/abs/2308.08469)

**Independent ablation: Tan et al. (Tan, Merrill, Gupta, Althoff, Hartvigsen; NeurIPS 2024; arXiv v2 2024-10-26)**
- **[INDEPENDENT]** Three LLM forecasters were tested (Time-LLM with LLaMA, CALF with GPT-2, OneFitsAll with GPT-2) on 13 datasets × 2 metrics. Each was compared with variants that remove the LLM, replace it with one attention layer, or replace it with one transformer block. The ablations beat Time-LLM in 26/26 cases, CALF in 22/26 and OneFitsAll in 19/26. — [Tan et al. 2024](https://arxiv.org/html/2406.16964); [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract.html)
- **[INDEPENDENT]** Compute, Table 4 (parameters; training time):
  - Time-LLM, ETTh1: 6,652M params and 181 min with the LLM, vs 0.198M and 0.99 min without it.
  - Time-LLM, Weather: 6,642M and 3,003 min, vs 0.198M and 1.91 min.
  - OneFitsAll, ETTh1: 85M and 7.36 min, vs 3M and 0.27 min. On Weather: 86M and 152 min, vs 4M and 16 min.
  - CALF, ETTh1: 180M and 3.28 min, vs 8M and 0.35 min.
  - Inference took on average 28.2× (Time-LLM), 2.3× (OneFitsAll) and 1.2× (CALF) longer than the ablated models.
  - Time-LLM ran on an A100 80GB. The GPT-2 methods ran on an RTX A6000 48GB.
  - — [Tan et al. 2024](https://arxiv.org/html/2406.16964)
- **[INDEPENDENT]** Pretraining did not help:
  - In few-shot (10% data), Time-LLM and its no-LLM variant each won 8 of 16 comparisons.
  - Randomly initialized CALF beat the pretrained one 8 to 3.
  - Shuffling the input sequence hurt the LLM models no more than the ablations, so LLMs show "no unique ability" to represent sequential dependencies.
  - The recommended baseline is "PAttn": instance normalization, patching, one attention layer, then a linear head.
  - — [Tan et al. 2024](https://arxiv.org/html/2406.16964)

**2025-2026 follow-ups**
- Qiu, Tong, Sun, Ma, Zhang, Shen, "Rethinking the Role of LLMs in Time Series Forecasting" (arXiv v1 2026-02-16, v2 2026-03-02, cs.CL, no venue listed). The study covers 8 billion observations, 17 scenarios and 4 horizons. It claims LLMs improve forecasting, "especially large gains in cross-domain generalization", that pre-alignment beats post-alignment in >90% of tasks, and that pretrained knowledge matters most under distribution shift. It argues earlier negative results came from narrow evaluation settings. It makes no statement on compute cost. — [Qiu et al. 2026](https://arxiv.org/abs/2602.14744)
- Zhang, Feng, Li, Lin, Li, Jia, "From Tokenizer Bias to Backbone Capability: A Controlled Study of LLMs for Time Series Forecasting" (v1 2025-04-09, v2 2026-03-06). Tokenizer/detokenizer pairs trained on small datasets overfit and mask the backbone's ability. With controlled pretraining, the LLM backbone "shows some promise" but "does not consistently surpass" models trained on large-scale time-series data. — [Zhang et al. 2025/2026](https://arxiv.org/abs/2504.08818)

**Stock-specific evidence (price history as input)**
- **[INDEPENDENT]** Noguer i Alonso & Franklin, "Pretrained Time-Series Foundation Models for Financial Return Forecasting" (2026-06-25):
  - Models: TimeGPT, TimesFM-2.5, Moirai-2.0, Chronos and Chronos-2, against from-scratch NBEATS, NHITS, PatchTST, iTransformer and KAN, plus random walks.
  - Data: returns of AAPL, AMZN, GOOG, JPM and META.
  - Gains over the random walk were "small and sparse". A Diebold-Mariano test rejected equal accuracy only for Chronos on AMZN and Moirai-2.0 on GOOG.
  - Pretrained models won 8 of 10 task-level comparisons vs from-scratch models.
  - — [Noguer i Alonso & Franklin 2026](https://arxiv.org/abs/2606.27100)
- **[INDEPENDENT]** Sodha, "When Reasoning Fails: Evaluating 'Thinking' LLMs for Stock Prediction" (2025-11-05). Compares gpt-5 with a 512-token reasoning budget, gpt-4o-mini, ridge and random forest, using rolling 48-month train / 1-month test windows, a one-day horizon, and NIFTY universes of 5-36 stocks. The thinking LLM's ranking quality fell as the universe grew, its outputs needed ex-post winsorizing and blending, and "after realistic transaction costs" it showed no net advantage. — [Sodha 2025](https://arxiv.org/abs/2511.08608)
- **[AUTHORS]** Valeyre & Aboura (v1 2024-12-12, v2 2025-11-03) report that Chronos (zero-shot and fine-tuned) long/short portfolios on large US single stocks "generate alpha". They also note "significant room for improvement" vs specialized smaller models. Period, Sharpe and cost treatment are not stated in the abstract. — [Valeyre & Aboura](https://arxiv.org/abs/2412.09394)
- A review "from a hedge-fund perspective" (Zhang & Zhang, 2026-04-10, IEEE Conference on AI 2026) lists data leakage, dataset/horizon design and "limits of stock price predictability" as understated pitfalls in LLM stock-forecasting papers. — [Zhang & Zhang 2026](https://arxiv.org/abs/2605.05211)
- **[INDEPENDENT]** Memorization: Lopez-Lira, Tang & Zhu (v1 2025-04-20, v2 2025-12-15):
  - LLMs recall exact economic and financial values from before their knowledge cutoff.
  - Instructions to respect historical boundaries do not stop recall-level accuracy, and masking fails because models reconstruct entities and dates from minimal context.
  - Memorization extends to embeddings, and no recall is observed after the cutoff.
  - Forecasts over training-period data "cannot be trusted".
  - — [Memorization Problem](https://arxiv.org/abs/2504.14765)
- StockBench's authors report that GPT-5, without search, correctly described AAPL's 2021 price trend, which is why they used a post-cutoff test window. — [StockBench](https://arxiv.org/html/2510.02209v1)
- Leaderboard snapshot: a paper that froze its model selection on 2026-04-09 reports Chronos-2 (120M params) ranked first in the pretrained section of the GIFT-Eval leaderboard. That is a purpose-built time-series model, not a text LLM. — [Laglil et al. 2026](https://arxiv.org/html/2607.23146). **[SECONDARY]** GIFT-Eval itself covers 23 datasets, ~144,000 series and 177M points. — [GIFT-Eval](https://arxiv.org/pdf/2410.10393)

### Inferences
- For stockai's price-only goal, a text LLM is the worst-value option. Independent ablations show the LLM adds no accuracy. It costs roughly 10× to ~1,500× more training time (e.g., ~50 h vs ~2 min on Weather for Time-LLM) and 1.2-28× more inference time. The default Time-LLM configuration needs 32-39 GB of GPU memory, about twice the RTX 5080's 16 GB, so it will not run unmodified on the user's card.
- Contamination makes text-LLM backtests on the user's data almost uninterpretable. Any LLM whose training data extends past 2025-06-27 has seen the entire Stooq history, including the held-out year. Prompting it with real tickers or dates becomes a memorization test, per the Lopez-Lira/Tang/Zhu evidence.
- LLMTime-style zero-shot prompting fits only ~300-500 observations per series in context and needs ~20 sampled continuations per forecast. For thousands of stocks × hundreds of days, the cost is API-scale or needs a 70B model, which cannot fit in 16 GB without heavy quantization.
- The pro-LLM 2026 result (Qiu et al.) concerns generalizing across domains and distribution shifts in standard forecasting benchmarks. It does not show that an LLM extracts tradeable signal from near-random daily returns. No paper found tests that.
- GPT-2-sized variants (85-180M params) fit easily on 16 GB, but Tan et al. show their ablated versions (3-13M params) do as well.

### Gaps
- I found no independent study that runs LLMTime, Time-LLM, GPT4TS or LLM4TS on a large daily US-equity cross-section with transaction costs. Evidence for the "add value?" question comes from non-financial benchmarks plus small stock samples.
- The Noguer i Alonso & Franklin sample period, horizon and cost treatment are not in the abstract.
- Qiu et al. (2026) does not state compute cost. It is not peer-reviewed as listed, and its author affiliations were not shown.
- The fetched arXiv pages for LLMTime and Time-LLM list no venue. They are commonly cited as NeurIPS 2023 and ICLR 2024, but I did not confirm this.
- I did not fetch the live GIFT-Eval leaderboard for October 2026; only the 2026-04-09 snapshot is reported.

## 2. Finance LLMs and LLM trading agents (FinGPT, BloombergGPT, FinMem, TradingAgents, and 2025-2026 systems): how good is the evidence (backtest length, costs, contamination)?

### Takeaway
The early finance LLMs (BloombergGPT, FinGPT) were evaluated on NLP tasks, not on trading. The agent papers reporting big returns (FinMem, TradingAgents) used 3-6-month backtests on 3-5 large-cap stocks, did not discuss transaction costs, and used LLMs that may have seen the test period. When tested longer and broader (FINSABER: 20 years, 100+ symbols), after model cutoffs (StockBench), or live (Alpha Arena, AI-Trader, HKU Agentic Trader), the reported edge shrinks to roughly buy-and-hold or to noisy, model-dependent wins and losses.

### Cited Findings
**Finance LLMs**
- BloombergGPT (Wu et al.; v1 2023-03-30, v3 2023-12-21): a 50B-parameter model trained on 363B tokens of Bloomberg financial data plus 345B general tokens. It was evaluated on financial NLP benchmarks (sentiment, NER, QA) and general LLM benchmarks. No trading backtest or return forecast is reported. — [BloombergGPT](https://arxiv.org/abs/2303.17564)
- FinGPT (Yang, Liu, Wang; v1 2023-06-09, v2 2025-11-15; IJCAI 2023 FinLLM Symposium) is a data-centric open-source pipeline with LoRA fine-tuning. It lists robo-advising and algorithmic trading as potential applications, but the abstract reports no quantitative trading result. — [FinGPT](https://arxiv.org/abs/2306.06031)

**Agent papers reporting large returns ([AUTHORS])**
- FinMem (Yu et al.; v2 2023-12-03):
  - Test window 2022-10-06 to 2023-04-10 (~6 months), training from 2021-08-17. Five tickers: TSLA, NFLX, AMZN, MSFT, COIN. Backbone: GPT-4-Turbo.
  - Reported cumulative returns: TSLA +61.8% (buy-and-hold −18.6%), COIN +35.0% (B&H −30.0%), MSFT +23.3% (B&H +14.7%).
  - In a backbone ablation on a different TSLA window, Llama2-70b-chat returned −52.7%.
  - The visible text does not mention transaction costs or knowledge-cutoff contamination.
  - — [FinMem](https://arxiv.org/html/2311.13743v2)
- TradingAgents (Xiao, Sun, Luo, Wang; v7 2025-06-03; workshop oral):
  - Backtest 2024-01-01 to 2024-03-29 (3 months). The results table covers 3 stocks (AAPL, GOOGL, AMZN); the text names 5.
  - Models: gpt-4o-mini, gpt-4o and o1-preview, with "11 LLM calls & 20+ tool calls/prediction".
  - Reported cumulative return 23-27% with Sharpe 5.6-8.2. A footnote calls the Sharpe "exceptionally high" due to "few pullbacks" in the short window.
  - The visible text does not discuss transaction costs or slippage. It claims no look-ahead in data feeds but does not discuss overlap between LLM training data and 2024.
  - — [TradingAgents](https://arxiv.org/html/2412.20138v7)

**Independent re-evaluations and critiques**
- **[INDEPENDENT]** FINSABER (Li, Kim, Cucuringu, Ma; v1 2025-05-11, v6 2026-06-26; KDD 2026 Datasets & Benchmarks, Oral):
  - Over "two decades" and "100+ symbols", previously reported LLM advantages "deteriorate significantly".
  - LLM strategies are too conservative in bull markets (they underperform passive benchmarks) and too aggressive in bear markets (heavy losses).
  - Prior evaluations were inflated by survivorship and data-snooping bias.
  - — [FINSABER](https://arxiv.org/abs/2505.07078)
- **[INDEPENDENT]** "Profit Mirage" (Li, Zeng, Xing, Xu, Xu; 2025-10-09): LLM trading agents' backtested returns collapse once the test window moves past the model's knowledge cutoff, an effect of information leakage. The authors release FinLake-Bench for leakage-robust evaluation and propose FactFin, a counterfactual-perturbation method of their own. — [Profit Mirage](https://arxiv.org/abs/2510.07920); commentary [UT McCombs](https://news.mccombs.utexas.edu/research/keeping-humans-in-the-loop/)
- **[INDEPENDENT]** Memorization (Lopez-Lira, Tang, Zhu, v2 2025-12-15): over periods covered by training data, "any observed output is consistent with both genuine skill and memorization". — [Memorization Problem](https://arxiv.org/abs/2504.14765)

**Post-cutoff and live tests ([INDEPENDENT] benchmarks; results are model-specific)**
- StockBench (Chen et al., Tsinghua/BUPT, 2025-10-02):
  - Window 2025-03-03 to 2025-06-30 (82 trading days), chosen to fall after mainstream LLM cutoffs. Top-20 DJIA stocks, $100k start, 3 seeds.
  - Equal-weight buy-and-hold returned +0.4% (max drawdown −15.2%). The 13 LLM agents ranged from −2.8% (GPT-OSS-20B) to +2.5% (Qwen3-235B-Think). GPT-5 returned +0.3% and Claude-4-Sonnet +2.2%.
  - Conclusion: "most models struggle to outperform" buy-and-hold, and static financial-QA skill does not transfer to trading.
  - — [StockBench](https://arxiv.org/html/2510.02209v1)
- **[SECONDARY]** Alpha Arena (Nof1) Season 1, about 2025-10-18 to 2025-11-03 (sources conflict on the end date and the coin set). Crypto, $10,000 per model, six models. Qwen3 Max won with about +22%, DeepSeek was second at about $10,489, and the other four lost money, with GPT-5 worst at about $3,733. Nof1 said results are not a definitive ranking. — [Forklog](https://forklog.com/en/four-out-of-six-ai-models-suffer-losses-in-trading-tournament/); [GNcrypto](https://www.gncrypto.news/news/qwen-wins-alpha-arena-season-1-with-22-percent-returns/)
- **[SECONDARY]** Alpha Arena Season 1.5 (US equities, Nov to early Dec 2025, 8 models, four contest modes). An undisclosed model later identified as Grok 4.20 was reportedly the only profitable model, at about +12.1%. The Season 1 winner Qwen3 Max reportedly lost heavily. Reports differ on details. — [Forklog](https://forklog.com/en/ai-model-grok-4-2-triumphs-in-trading-tournament/); [AIhola](https://aihola.com/article/grok-wins-alpha-arena-trading)
- AI-Trader (Fan et al., 2025-12-01). A "fully-automated, live, and data-uncontaminated" benchmark across US stocks, A-shares and crypto with six LLMs. "General intelligence does not automatically translate to effective trading capability"; most agents showed poor returns and weak risk management. — [AI-Trader](https://arxiv.org/abs/2512.10971)
- HKU Business School "Agentic Trader" (press release 2026-07-14). Six weeks of live trading from April 2026 in FX, the S&P index and precious metals. Ten models each started with $100,000: GPT-5.4, Claude Opus 4.6, Gemini 3.1 Pro Preview, Qwen3.5 Plus, Kimi K2.5, GLM, Seed-2.0-Lite, DeepSeek V3.2, MiniMax and Grok-4.1 Fast. Qwen was best at about +9.9% and DeepSeek worst at −15.10% (the release's risk section instead names Kimi K2.5 as strongest over the six weeks, an internal inconsistency). The release says results "should not be interpreted as a definitive measure of long-term investment capability". — [HKU press release](https://lead.hku.hk/press/news_detail_29241.html)
- InvestLogicBench (Jiang et al., v1 2026-08-06, v2 2026-08-20). Built from 201,247 decisions by 151 real investors and tested on four LLMs. Logical plausibility is about 4/5 but event grounding only 0.8-2.8/5; "return and process quality do not always agree". — [InvestLogicBench](https://arxiv.org/abs/2608.06108)

### Inferences
Evidence-quality summary:

| System | Test length | Symbols | Costs modeled? | Contamination handled? |
|---|---|---|---|---|
| FinMem | ~6 months (Oct 2022-Apr 2023) | 5 | Not mentioned | Not discussed. The paper was posted Nov 2023, after the test window. |
| TradingAgents | 3 months (Q1 2024) | 3 (5 named) | Not mentioned | Data feeds only, not LLM training data |
| StockBench | 4 months (Mar-Jun 2025) | 20 | Not stated | Yes (post-cutoff window) |
| FINSABER | ~20 years | 100+ | Not stated in abstract | Survivorship/snooping addressed |
| Live arenas | 2-6 weeks | Few assets | Live markets (cost handling not verified) | Yes (live), but tiny samples |

- Agents need many calls per decision (TradingAgents: 11 LLM calls plus 20+ tool calls per prediction). A daily decision for each of thousands of stocks over a backtest year is impractical on cost and latency, and agents cannot be backtested honestly on the user's 2005-2025 data because of contamination.
- The live results resemble the random-guess accounts in the user's correlation2 harness: large spreads between models (e.g., −63% to +22% in Alpha Arena Season 1) over a few weeks are what luck plus leverage produces. None of these sources shows a persistent edge.
- What the agent literature is good for: tooling for reading filings and news, research assistance, and explaining decisions. It is not evidence that LLMs forecast prices.

### Gaps
- I did not verify the training cutoffs of GPT-4-Turbo, gpt-4o or o1-preview against the FinMem and TradingAgents test windows from primary sources. The overlap is plausible but unconfirmed here.
- Alpha Arena figures come only from secondary news. I did not read Nof1's own leaderboard.
- Abstracts for FINSABER, StockBench and AI-Trader do not state transaction-cost assumptions. AI-Trader's evaluation period is not stated.
- I could not confirm from a primary source the commonly repeated claim that GPT-4 matched or beat BloombergGPT on financial NLP tasks.
- Agent Market Arena (arXiv 2510.11695, Oct 2025) appeared only in a search snippet ("agent frameworks behave in clearly different ways"); I did not read it.

## 3. LLMs on news and text for return prediction (Lopez-Lira & Tang; Chen, Kelly & Xiu; later work): what predictability, how fast it decayed, look-ahead concerns, and what data a hobbyist would need

### Takeaway
LLM-scored news predicts next-day returns better than older NLP. Lopez-Lira & Tang report a pre-cost Sharpe of 2.97 for a GPT-4 headline strategy (Oct 2021-May 2024). It is concentrated in small stocks and bad news, decayed from Sharpe 6.54 (2021Q4) to 1.22 (Jan-May 2024), and is unprofitable at 20 bps round-trip costs. Chen, Kelly & Xiu find LLM text features beat word-based methods across 16 markets, and a discussant cites a net Sharpe of about 1.5 after 10-20 bps costs. Look-ahead bias appears modest when tested with chronologically consistent models. This is a different input from price history, and the user would need timestamped news to try it.

### Cited Findings
**Lopez-Lira & Tang, "Can ChatGPT Forecast Stock Price Movements?"** (arXiv v6 stamped 2025-10-28; the paper's own header reads "This Version: August 24, 2026", which is inconsistent; the acknowledgments name an editor, but publication was not confirmed). All from [Lopez-Lira & Tang](https://arxiv.org/html/2304.07619):
- Data:
  - 159,137 firm-headline-date observations covering 4,123 US common stocks, Oct 2021-May 2024.
  - Headlines were web-scraped by ticker and matched to RavenPack, with filters of relevance 100 and event similarity >90 days.
  - About 82% are overnight news (before 9am or after 4pm).
  - Returns come from CRSP daily and TAQ intraday data.
- Trade timing: news before 9am is traded open-to-close the same day. News after the close is traded from the next open to the next close.
- GPT-4 results, overnight news:
  - Initial-reaction hit rate 93.3%, but that reaction is "not tradable by most investors".
  - Drift (the tradable part): hit rate 58%, 0.34%/day, Sharpe 2.97 before costs.
  - Long leg Sharpe 0.78, short leg Sharpe 2.01.
  - Long-short cumulative return was about 700% before costs, Oct 2021-May 2024.
- Model ladder (overnight drift Sharpe):
  - GPT-3.5: 1.66. BART-Large: 1.05. Llama2-70b: 0.97. FinBERT: −0.33.
  - GPT-1, GPT-2 and BERT: negative.
  - Embedding baselines: OpenAI embeddings 2.53, MiniLM 1.42.
- Concentration: much stronger in small firms (small-firm interaction coefficient 0.404, t=4.75) and in negative news. There is little drift after earnings and partnership news, and underreaction to insider trades, dividends and healthcare-conference news.
- Decay ("suggestive"): annualized Sharpe fell from 6.54 (2021Q4) to 3.68 (2022), 2.33 (2023) and 1.22 (Jan-May 2024). The authors link this to rising LLM adoption without establishing causation.
- Costs (round trip):
  - 5 bps leaves >300% cumulative, 10 bps >100%, and 20 bps makes it unprofitable.
  - Daily rebalancing has about 190% daily turnover, with Sharpe 1.29 at 10 bps.
- Look-ahead control: the GPT-4 version used was trained on data through Sept 2021, before the sample starts. Llama-2's later cutoff (Sept 2022) may overstate its results, which the authors acknowledge.

**Chen, Kelly & Xiu, "Expected Returns and Large Language Models"** (authors per a Wharton slide deck dated 2023 in its citations; the SSRN page I tried returned HTTP 403)
- Scope: 16 global equity markets, news in 13 languages. LLMs (ChatGPT, LLaMA, LLaMA2, RoBERTa, BERT) are compared with word-based models (Word2vec, SESTM, LMMD) using ridge, lasso, random forest and neural nets. Ridge regression maps text features to one-day-ahead returns, and the predictions are used to sort stocks. — [Discussion slides, S. Z. Li, Wharton Jacobs Levy Center, Sept 2024](https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2024/09/ExpectedReturnAndLLM_Discussion_SophiaZhengziLi_NoPause.pdf); [Kelly slides](https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2024/09/Kelly-WhartonJL.pdf)
- Main findings per the discussant: "Returns respond slowly to news", and LLMs beat word-based models in sentiment and return prediction. Using news from 9am on day t-1 to 9am on day t to predict the 9am-9am return on day t+1, with costs of 10 bps (large stocks) and 20 bps (small stocks), the "net (after TC) annual Sharpe Ratio is around 1.5". The discussant asks whether predictability is "front-loaded", i.e., concentrated in the first minutes after 9am when trading is hard, and notes significantly lower long-short returns for large firms. — [Discussion slides](https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2024/09/ExpectedReturnAndLLM_Discussion_SophiaZhengziLi_NoPause.pdf)
- **[SECONDARY]** Seminar abstracts say a strategy using fresh "news alerts" yields "even higher Sharpe ratios", that LLM benefits are largest for articles with negation or complex narratives, and that slow price response indicates "market inefficiencies and limits-to-arbitrage". — [Univ. of Sydney seminar listing](https://business.sydney.edu.au/events/research/2024/finance/dacheng-xiu); [ESSEC webinar](https://essec-amundi-chair.essec.edu/webinars/webinar-1-september-2024)
- Benchmark cited by the discussant (not LLM-based): Jiang, Li & Wang (2021 JFE) "news momentum" from intraday news-driven returns. Over 2000-2019 it earned an equal-weighted annual return of 34% and value-weighted 25%. It "survives transaction cost", and "a leading hedge fund still actively trades on it". — [Discussion slides](https://jacobslevycenter.wharton.upenn.edu/wp-content/uploads/2024/09/ExpectedReturnAndLLM_Discussion_SophiaZhengziLi_NoPause.pdf)

**Look-ahead bias**
- Glasserman & Lin (2023-09-29) separate "look-ahead bias", where the LLM learned the returns that followed a news story, from the "distraction effect", where general knowledge of a company distorts the sentiment reading. In-sample, anonymized headlines performed better, so distraction dominates, especially for large firms. Out of sample, look-ahead is not a concern but distraction may remain. — [Glasserman & Lin](https://arxiv.org/abs/2309.17322)
- He, Lv, Manela & Wu, "Chronologically Consistent Large Language Models" (v3 2025-07-06). ChronoBERT and ChronoGPT are trained only on text available at each point in time. On next-day return prediction from news they reach Sharpe ratios "comparable to a much larger Llama model", and the authors conclude look-ahead bias is "modest". — [ChronoGPT](https://arxiv.org/abs/2502.21206)
- Memorization of exact values before the cutoff, and the failure of masking, apply to text inputs too. — [Memorization Problem](https://arxiv.org/abs/2504.14765)

**Data a hobbyist could use**
- FNSPID (2024-02-09): 15.7M time-aligned financial news records and 29.7M price records for 4,775 S&P 500 companies, 1999-2023, from 4 news websites. Code and data are on GitHub. — [FNSPID](https://arxiv.org/abs/2402.06698)
- Alpaca historical news API: Benzinga news from 2015, about 130+ articles per day, one endpoint for stocks and crypto. The free-plan status was not stated on the page. — [Alpaca docs](https://docs.alpaca.markets/docs/historical-news-data)
- Lopez-Lira & Tang relied on RavenPack, a commercial vendor, for relevance filtering and matching. — [Lopez-Lira & Tang](https://arxiv.org/html/2304.07619)

### Inferences
- The predictability is real but thin and fragile. The tradable drift is a fraction of the untradable reaction at the moment of release, it lives in small caps and on the short side (borrow costs, short constraints), and Lopez-Lira's version dies at 20 bps round trip. That is the same cost band where the user's correlation2 signals died.
- A hobbyist test would need:
  - timestamped headlines (time of day matters for the pre-open / intraday / after-close rules);
  - ticker mapping;
  - daily open and close prices, which Stooq OHLCV provides for open-to-close returns;
  - a scoring model.
- FNSPID ends in 2023, so it does not cover the user's held-out year (2024-06-28..2025-06-27). Benzinga via Alpaca does.
- Any LLM used to score news for 2024-2025 must have a training cutoff before the test window, or be chronologically consistent like ChronoGPT. Otherwise the backtest measures memorization.
- Here the LLM acts as a text classifier or feature extractor, not a price forecaster. Small local models keep part of the signal (MiniLM embeddings: drift Sharpe 1.42 vs GPT-4's 2.97), and those run easily on a 16 GB GPU. This is the one setting where an LLM-family model has documented incremental value for returns.

### Gaps
- I could not read Chen-Kelly-Xiu's own tables (SSRN blocked), so exact gross and net Sharpe figures by market and weighting, and the paper's own decay analysis, are unverified beyond the discussant's ~1.5 net figure.
- Lopez-Lira & Tang's sections on speed of price response were in the truncated part of the text I read.
- I found no 2026 replication of headline-based LLM signals over a post-2024 window.

## 4. Diffusion and flow-matching models for time-series forecasting (TimeGrad, CSDI, TSDiff, Diffusion-TS, mr-Diff, TimeDiT, Sundial's TimeFlow, and later work): accuracy vs autoregressive/transformer baselines, and sampling cost

### Takeaway
Diffusion forecasters produce full probability distributions and sample paths, and can be strong on short-horizon probabilistic metrics. The independent ProbTS benchmark (NeurIPS 2024) finds them weak at long horizons, with CSDI hitting time and memory limits. Classic diffusion forecasters (TimeGrad, CSDI, SSSD) are ~30× to ~2,200× slower at inference than a one-step generator on the same benchmark (TimeGrad 0.87-6.7 s per forecast vs ~3 ms). Newer ones (mr-Diff, TimeDiff) are only ~3-12× slower. In 2025-2026 the frontier moved to autoregressive transformers with a light flow-matching head (Sundial, ICML 2025 Oral), which blurs the "diffusion vs autoregressive transformer" choice. In Sundial's own ablation the diffusion objective was worse than both flow matching and plain MSE.

### Cited Findings
**Methods (mostly [AUTHORS])**
- TimeGrad (Rasul et al.; ICML 2021) is autoregressive over time. At each time step it samples the next value with a denoising diffusion model, turning white noise into a sample via Langevin dynamics. It claimed SOTA multivariate probabilistic forecasting. — [TimeGrad](https://arxiv.org/abs/2101.12072)
- CSDI (Tashiro, Song, Song, Ermon; NeurIPS 2021) is a conditional score-based diffusion model built for imputation. It claims 40-65% gains over probabilistic imputation methods, and the authors say it is "competitive" for forecasting. — [CSDI](https://arxiv.org/abs/2107.03502)
- TSDiff (Kollovieh et al.; code at github.com/amazon-science; v3 2023-11-22) is an unconditional diffusion model that is conditioned at inference via "self-guidance" and reported as competitive with task-specific conditional forecasters. It can refine other models' forecasts. Forecasters trained on its synthetic samples beat those trained on other generators' samples and "sometimes even" those trained on real data. — [TSDiff](https://arxiv.org/abs/2307.11494)
- Diffusion-TS (Yuan & Qiao; v3 2024-10-21) is mainly a generator: an encoder-decoder transformer with decomposed representations and a Fourier loss, extended to forecasting and imputation "without any model changes". — [Diffusion-TS](https://arxiv.org/abs/2403.01742)
- mr-Diff (Shen, Chen, Kwok; ICLR 2024) is a non-autoregressive, coarse-to-fine diffusion: it generates the coarsest trend first and adds finer detail conditioned on it. — [ICLR 2024 poster](https://iclr.cc/virtual/2024/poster/17883); [paper PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/d64740dd69bcc90ba225a182984b81ba-Paper-Conference.pdf)
- TimeDiT (Cao, Ye, Zhang, Liu; v2 2025-02-11; ICML 2024 workshop) is a diffusion-transformer "proto-foundation model" for forecasting, imputation, anomaly detection and generation. It argues that autoregressive transformers "overlook uncertainty". — [TimeDiT](https://arxiv.org/abs/2409.02322)
- Sundial (Liu et al.; v1 2025-02-02, v4 2026-05-09; ICML 2025 Oral):
  - Method: a transformer pre-trained on TimeBench (~1 trillion time points). A small MLP flow-matching head ("TimeFlow loss") turns Gaussian noise into a sample of the next patch, conditioned on the transformer's per-patch representation. There is "no discrete tokenization".
  - Sampling: the default is 20 samples × 50 steps.
  - **[AUTHORS]** Results: GIFT-Eval MASE 0.673 (first) and CRPS 0.472 (second), vs TimesFM 0.680/0.465 and Chronos 0.786/0.551. Ablation zero-shot average MSE: TimeFlow 0.290, MSE loss 0.296, diffusion objective 0.314.
  - Speed: "35× inference speedup" vs Chronos on FEV, and 20 samples × 50 steps in "nearly one second on a CPU".
  - — [Sundial abs](https://arxiv.org/abs/2502.00816); [Sundial HTML](https://arxiv.org/html/2502.00816); [Sundial GitHub](https://github.com/thuml/Sundial)
- Sundial GitHub: the checkpoint is `sundial-base-128m`. On an Apple M1 Pro CPU it takes 249 ms (1 sample) to 949 ms (20 samples, 720-step horizon). Fine-tuning code is not yet released. — [Sundial GitHub](https://github.com/thuml/Sundial)

**Independent benchmark: ProbTS (Zhang et al.; NeurIPS 2024 Datasets & Benchmarks; v5 2024-10-21)**
- **[INDEPENDENT]** Horizons and decoding:
  - Long-horizon point forecasters "tend to lose their competitive performance in short-term scenarios".
  - Existing probabilistic models face "considerable challenges" at long horizons.
  - TimeGrad (autoregressive) accumulates errors at long horizons but "excels in capturing strong seasonality, outperforming models like PatchTST" on Traffic.
  - CSDI's long-term performance "is less than optimal".
  - The CRPS advantage of distributional models is largest on strongly non-Gaussian data.
  - — [ProbTS](https://arxiv.org/html/2310.07446v5)
- **[INDEPENDENT]** Cost: "excessive time and memory consumption of CSDI" left its results missing for Traffic-L at all horizons and Electricity-L at 336/720. Configuration: TimeGrad uses 100 diffusion steps and CSDI 50. Hardware: single V100. — [ProbTS](https://arxiv.org/html/2310.07446v5); listing [ProbTS abs](https://arxiv.org/abs/2310.07446)

**Sampling cost numbers**
- Inference time per forecast on univariate ETTh1. TARFVAE's authors copied the baseline rows from the mr-Diff paper; they were measured by mr-Diff's authors on hardware not confirmed here. Only the TARFVAE row was re-run, on an Nvidia A6000 48 GB:

  | Model | Time per forecast (horizon 96 → 720) |
  |---|---|
  | TimeGrad | 870 → 6,724 ms |
  | SSSD | 419 → 2,517 ms |
  | CSDI | 90 → 513 ms |
  | TimeDiff | 16 → 35 ms |
  | mr-Diff | 8 → 22 ms |
  | TARFVAE, a one-step flow-based VAE (1-200 samples) | about 3 ms |

  The TARFVAE paper notes diffusion's "iterative denoising process incurs substantial computational overhead". — [TARFVAE 2025-11-28](https://arxiv.org/html/2511.22853)
- **[SECONDARY]** Surveys: a 2025 deep-learning forecasting survey says early time-series diffusion models (TimeGrad, CSDI, SSSD) have slow inference and high complexity ([arXiv 2503.10198](https://arxiv.org/pdf/2503.10198)). A Sept 2026 probabilistic-forecasting survey rates DDPM/score-based sampling as high cost, rates latent diffusion low-to-moderate, and lists sampling efficiency as an open direction ([arXiv 2609.13345](https://arxiv.org/pdf/2609.13345)). Diffusion forecasting survey: [Su et al. 2025](https://arxiv.org/abs/2507.14507).
- Leaderboard context: as of 2026-04-09, the top pretrained model on GIFT-Eval was Chronos-2, not a diffusion model. — [Laglil et al. 2026](https://arxiv.org/html/2607.23146)
- **[SECONDARY]** 2026 activity includes DAD4TS (May 2026), a diffusion model for small-data forecasting compared against PatchTST, iTransformer and others; the snippet showed no timing numbers. — [DAD4TS](https://arxiv.org/pdf/2605.17866)

### Inferences
- The "(b) diffusion vs (c) autoregressive transformer" framing is outdated. TimeGrad is autoregressive with a diffusion head, and Sundial is an autoregressive transformer with a flow-matching head that reports the best GIFT-Eval MASE. The practical choice is between a point/quantile output head and a generative head on the same small transformer.
- For next-day returns, a generative head describes the spread of outcomes (fat tails, volatility clustering). That helps with sizing, risk and stop placement, but it does not create a directional edge when the conditional mean is near zero.
- Cost scaled to stockai: a backtest year over ~3,000 stocks × ~250 days is ~750,000 forecasts. At the published per-forecast times, unbatched, that is ~181 GPU-hours for TimeGrad (0.87 s), ~19 h for CSDI (0.09 s), ~1.7 h for mr-Diff (8 ms) and ~38 min for a ~3 ms one-step model. Batching on an RTX 5080 would cut these, but the relative gaps (~3× to ~300× at the 96-step horizon) would largely remain. This is arithmetic on published per-forecast figures (hardware for the copied baselines unconfirmed), not a measurement.
- Probabilistic metrics (CRPS) reward getting the whole distribution right. For daily returns a well-calibrated distribution centered near zero scores well while carrying no trading signal, so the trading-based test (vs random-guess accounts) remains the decisive one.

### Gaps
- I found no independent benchmark of diffusion or flow-matching forecasters on daily equity returns.
- I did not read the ProbTS efficiency appendix (D.7), so its numeric cost table is unreported here.
- The fetched pages show no venue for TSDiff or Diffusion-TS (commonly cited as NeurIPS 2023 and ICLR 2024; unconfirmed here).
- TimeBench's share of financial data is not stated in the parts of Sundial I read.
- I did not check the live GIFT-Eval leaderboard as of Oct 2026.

## 5. Diffusion and other generative models in finance (FTS-Diffusion, DiffsFormer, TRADES, QuantGAN, TimeGAN, newer work): what they are used for, and whether synthetic-data augmentation improves out-of-sample return prediction

### Takeaway
In finance, generative models are used mainly for synthetic data, matching stylized facts, scenario and stress generation, limit-order-book simulation, and training-data augmentation, rather than for direct forecasting. Forecasting gains from augmentation are reported almost only by the methods' own authors: FTS-Diffusion up to 17.9% lower prediction error, DiffsFormer +7.2%/+27.8% relative annualized return on Chinese indices. The one independent, controlled 2026 test found augmentation helps variance-dominated tasks such as volatility forecasting but degrades near-efficient directional prediction, which is the user's task.

### Cited Findings
**Finance generators ([AUTHORS])**
- FTS-Diffusion (Huang, Chen, Qiao; ICLR 2024): pattern recognition for "irregular and scale-invariant" patterns, a diffusion generator of pattern segments, and a pattern-transition model. **[AUTHORS]** Augmenting real training data with its synthetic data cuts downstream stock-prediction error "by as much as 17.9%" in two experiments. No independent replication found. — [ICLR 2024 paper PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/f90fc76b199fe6b0ec2a51aaf72c3277-Paper-Conference.pdf); [mlanthology](https://mlanthology.org/iclr/2024/huang2024iclr-generative); [ICLR spotlight page](https://ICLR.cc/virtual/2024/poster/19166)
- DiffsFormer (Gao et al.; arXiv 2024-02-05, no venue listed): a diffusion transformer pretrained on a large source domain that "edits" target-domain stock-factor samples for augmentation. **[AUTHORS]** Across 8 backbone models it reports relative improvements of 7.2% (CSI300) and 27.8% (CSI800) in annualized return ratio. The abstract does not mention transaction costs. — [DiffsFormer](https://arxiv.org/abs/2402.06656)
- TRADES (Berti, Prenkaj, Velardi; ECAI 2025; v3 2025-11-20): a transformer denoising diffusion model that generates limit-order-book order flow and reacts to an experimental agent, aimed at strategy calibration and market-impact studies. **[AUTHORS]** It improves the "predictive score" (train a predictor on synthetic data, test on real) by 3.27 and 3.48 over prior SOTA on two stocks; the units are unclear from the abstract. It releases the DeepMarket open-source framework and a synthetic LOB dataset. — [TRADES](https://arxiv.org/abs/2502.07071)
- Quant GANs (Wiese, Knobloch, Korn, Kretschmer; Quantitative Finance 2020): a GAN built on temporal convolutional networks that reproduces volatility clusters, leverage effects and serial autocorrelation at small and large lags. — [Quant GANs](https://arxiv.org/abs/1907.06673)
- TimeGAN (Yoon, Jarrett, van der Schaar; NeurIPS 2019): combines a learned embedding space with supervised and adversarial objectives. Reports better similarity and "predictive ability" than benchmarks. — [TimeGAN](https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html)
- Kronos (Shi et al.; arXiv 2025-08-02; AAAI 2026 per repo):
  - A "foundation model for the language of financial markets": a transformer pre-trained autoregressively on >12 billion K-line (OHLCV bar) records from 45 exchanges.
  - **[AUTHORS]** Zero-shot claims: +93% RankIC vs the leading time-series foundation model, +87% vs the best non-pretrained baseline, 9% lower volatility MAE, and +22% "generative fidelity" for synthetic K-lines.
  - Open weights: Kronos-mini 4.1M, small 24.7M and base 102.3M parameters. Kronos-large (499.2M) is not open.
  - The repo warns its fine-tuning pipeline is "not a production-ready quantitative trading system", and that a real backtest must "meticulously model transaction costs, slippage, and market impact".
  - — [Kronos arXiv](https://arxiv.org/abs/2508.02739); [Kronos GitHub](https://github.com/shiyu-coder/Kronos)
- TSDiff (non-finance datasets): forecasters trained on its synthetic samples "sometimes even" beat those trained on real data. — [TSDiff](https://arxiv.org/abs/2307.11494)
- Podobiński & Chudziak (2026-02-19): a transformer GAN (TTS-GAN) augmenting Bitcoin and S&P 500 price data "significantly improves" LSTM forecasting accuracy. No size-matched control is mentioned in the abstract. — [TTS-GAN augmentation](https://arxiv.org/abs/2602.17865)

**Independent, controlled test of augmentation**
- **[INDEPENDENT]** Sohm, Dezons, Sellami, Ninou, Pincon, "Improving Machine Learning Performance with Synthetic Augmentation" (2026-04-16):
  - Generators: bootstrap, copulas, VAEs, diffusion and TimeGAN.
  - Controls: a "size-matched null augmentation" separates real information gains from sample-size effects, and a block permutation test handles temporal dependence.
  - Data: Markov-switching simulations, high-frequency option trades and a daily equity panel.
  - Finding: "synthetic augmentation is beneficial only in variance-dominant regimes" such as persistent volatility forecasting. It "degrades performance in bias-dominant settings, such as near-efficient directional prediction".
  - — [Sohm et al. 2026](https://arxiv.org/abs/2604.14498)
- **[SECONDARY]** A Jan 2026 preprint (MarketGANs) argues most existing work treats synthetic data as local augmentation that reproduces marginal distributions, and rarely addresses joint return distributions, cross-sectional dependence or portfolio-relevant risk. — [MarketGANs](https://arxiv.org/html/2601.17773v1)

### Inferences
- For a next-day or next-week directional model, synthetic augmentation is unlikely to help and may hurt (Sohm et al.). The positive results mostly come from method authors, without size-matched nulls or cost-adjusted returns.
- Where generative models fit in stockai:
  - stress-testing a finished strategy on plausible alternative paths with realistic tails and volatility clustering;
  - volatility or risk forecasting;
  - simulating order books for execution research.
- The user's random-guess-account benchmark is already a simpler and more transparent null than synthetic markets.
- Kronos is the closest published version of "an LLM for prices": OHLCV bars become discrete tokens and a transformer predicts the next token. Its open mini/small/base weights (4-102M params) fit easily on 16 GB and make a ready-made pretrained baseline. Its RankIC claims are relative improvements on the authors' own benchmarks, not cost-adjusted trading results.

### Gaps
- I found no independent replication of FTS-Diffusion, DiffsFormer or Kronos.
- I could not check DiffsFormer's cost treatment or full protocol, or the units of TRADES's predictive score, without the full papers.
- Sohm et al.'s effect sizes for the daily equity panel are not in the abstract.

## 6. Plain-language explanation: how "next-token prediction" maps onto price data in each approach

### Takeaway
Only LLMTime (prices written as digit text), Chronos (scaled values bucketed into a vocabulary of bins) and Kronos (OHLCV bars turned into learned discrete tokens) literally do next-token prediction on prices. Time-LLM, GPT4TS and PatchTST-style models read patches of numbers and output numbers directly; diffusion models denoise a whole future window from random noise; and flow matching (Sundial) generates the next patch as continuous values.

### Cited Findings
- LLMTime: digits separated by spaces and time steps by commas. The LLM's next-token distribution over digits is sampled many times (e.g., 20), and the median gives the forecast. — [LLMTime](https://arxiv.org/html/2310.07820)
- Chronos (Ansari et al.; v3 2024-11-04): "tokenizes time series values using scaling and quantization into a fixed vocabulary". T5-family models (~20M-710M params) are trained "via the cross-entropy loss", the same loss used for words. — [Chronos](https://arxiv.org/abs/2403.07815)
- Kronos: a tokenizer "quantizes continuous, multi-dimensional K-line data (OHLCV) into hierarchical discrete tokens", and an autoregressive transformer is pre-trained on them. — [Kronos GitHub](https://github.com/shiyu-coder/Kronos)
- Time-LLM: patches are mapped by cross-attention onto text-prototype embeddings, passed through a frozen Llama with a text prompt prefix, and projected to numeric forecasts. — [Time-LLM](https://arxiv.org/html/2310.01728)
- GPT4TS: a pretrained transformer's self-attention and feed-forward blocks are frozen and the remaining parts fine-tuned for time-series tasks. — [One Fits All](https://arxiv.org/abs/2302.11939)
- PatchTST: "A Time Series is Worth 64 Words". Sub-series patches act as tokens for a transformer. **[SECONDARY]** Patching reportedly cut running time ~22× on Traffic. — [PatchTST](https://ar5iv.labs.arxiv.org/html/2211.14730)
- TimeGrad: an autoregressive model that, at each step, turns white noise into a sample of the next value by iterative denoising. — [TimeGrad](https://arxiv.org/abs/2101.12072)
- CSDI / non-autoregressive diffusion: denoising conditioned on observed values fills in or forecasts the missing part. ProbTS configs use 50 (CSDI) and 100 (TimeGrad) denoising steps. — [CSDI](https://arxiv.org/abs/2107.03502); [ProbTS](https://arxiv.org/html/2310.07446v5)
- Sundial: a flow-matching head turns Gaussian noise into the next patch in K steps (e.g., 50), conditioned on the transformer's summary of past patches, with "no discrete tokenization". — [Sundial](https://arxiv.org/html/2502.00816)
- Autoregressive (one-step-at-a-time) generation accumulates errors over long horizons (TimeGrad), while one-shot decoding gains an advantage as the horizon lengthens. — [ProbTS](https://arxiv.org/html/2310.07446v5)

### Inferences
Plain-language version for someone new to ML. This is synthesis based on the cited mechanisms above.

**1. What an LLM does with words.** Text is chopped into tokens (word pieces) from a fixed vocabulary. The model reads the tokens so far and outputs a probability for every possible next token. Training nudges it to give high probability to the token that actually came next (cross-entropy loss). To generate, you sample a token, append it, and repeat. That loop is "autoregressive".

**2. Making prices look like tokens.** There are three main recipes:
- **Digits as text (LLMTime).** Write 101.25, 101.90, ... as " 1 0 1 2 5 , 1 0 1 9 0 ," and let an off-the-shelf chatbot-style model continue the string. The model never knows these are prices; it pattern-matches digit strings. That is why tokenizers and chat tuning matter, and why the model may simply recall famous price histories it saw in training.
- **Bins as vocabulary (Chronos).** Rescale each series, then cut the value range into a fixed set of buckets. Each day's value becomes a bucket ID, which works exactly like a word ID. A transformer (T5 family) is trained on a large corpus of series to predict the next bucket, with the same cross-entropy loss used for words. It outputs a probability for every bucket, so you get a full distribution for tomorrow.
- **Learned bar-tokens (Kronos).** A small network learns to compress each day's whole OHLCV bar into a small set of discrete codes at coarse and fine levels ("hierarchical discrete tokens"). A GPT-like model then predicts the next bar's codes. This is the closest thing to "an LLM, but for candlesticks".

**3. Skipping discrete tokens (PatchTST, Sundial, Time-LLM, GPT4TS).** Group, say, 16 consecutive values into a "patch". Turn each patch into a vector, the way an LLM turns a word into a vector, and let the transformer read the patch-vectors. The output is then either numbers directly (a regression head trained to minimize squared error, as in PatchTST), or a generated sample of the next patch (Sundial's flow-matching head). Time-LLM and GPT4TS reuse a pretrained language model's frozen layers as the middle of this pipeline. Tan et al. show that middle can be swapped for one attention layer without loss.

**4. Diffusion is not next-token prediction.** Instead of writing left to right, a diffusion model starts with pure random noise in the shape of the whole future window (e.g., the next 5 daily returns). It repeatedly "denoises" it, 50-100 times, guided by the price history, until it looks like a plausible future. Run it 100 times and you get 100 different plausible futures. TimeGrad is a hybrid that goes one day at a time but uses a small diffusion process to generate each day.

**5. Flow matching** (Sundial's TimeFlow) is a faster cousin of diffusion. It learns a smooth path from noise to data, so fewer, cheaper steps produce a sample.

**6. What "predict the next price" should mean for trading.**
- Tomorrow's price is almost exactly today's price, so a model that outputs "same as today" (the random walk) already has tiny price error. Low price error therefore proves nothing.
- The useful target is the next-day or next-week return (percentage change), or its rank across stocks.
- The test is whether trading on it beats the random walk, buy-and-hold and same-turnover random accounts after costs, as in correlation2's harness.
- The 2026 foundation-model benchmark found pretrained models beat the random walk on returns only rarely ([Noguer i Alonso & Franklin 2026](https://arxiv.org/abs/2606.27100)).

### Gaps
- The fetched Chronos abstract does not state the bin count or exact scaling rule, and the fetched Kronos repo text does not give the tokenizer's code sizes. These specifics are omitted.

## 7. Synthesis question from the objective: when (if ever) should stockai use (a) a text LLM, (b) a diffusion or flow-matching model, or (c) a small purpose-built autoregressive time-series transformer, for next-day and next-week returns from price/volume history, and what does each cost on one RTX 5080?

### Takeaway
For price/volume-only next-day and next-week returns on one RTX 5080, choose (c): a small autoregressive or patch transformer (~0.2-25M parameters, minutes to hours to train), optionally with a quantile or flow-matching head, benchmarked against open pretrained models (Kronos, Chronos, Sundial) and a random walk; use (a) only if news text is added, and (b) for scenarios, stress tests and volatility rather than directional signals. Architecture matters less than the base rate: independent 2025-2026 evidence and the user's own correlation2 null both say price-only daily signals are weak and do not survive 5-20 bps costs.

### Cited Findings
**Compute evidence relevant to a 16 GB consumer GPU**

| Approach | Example | Size | Reported hardware / time | Source |
|---|---|---|---|---|
| (a) LLM reprogramming | Time-LLM (Llama-7B) | 6,642-6,652M total in Tan et al. (Time-LLM's own Table 7 lists ~3,404.5M for its full-Llama variant); <6.6M trainable | A100-80GB; 181 min (ETTh1), 3,003 min (Weather); 32,136-39,004 MiB GPU memory | [Tan et al.](https://arxiv.org/html/2406.16964), [Time-LLM](https://arxiv.org/html/2310.01728) |
| (a) GPT-2-based | OneFitsAll / CALF | 85-180M | RTX A6000 48GB; 3.3-152 min | [Tan et al.](https://arxiv.org/html/2406.16964) |
| (a) zero-shot prompting | LLMTime | GPT-3 API or LLaMA-2 70B | API cost limited experiments; 4,096-token context | [LLMTime](https://arxiv.org/html/2310.07820) |
| (c) LLM removed / PAttn-like | Tan et al. ablations | 0.2-13M | 0.27-24 min | [Tan et al.](https://arxiv.org/html/2406.16964) |
| (b) diffusion | TimeGrad / CSDI / mr-Diff | n/a | 870-6,724 / 90-513 / 8-22 ms per forecast (measured by mr-Diff's authors; hardware unconfirmed) | [TARFVAE (copied from mr-Diff)](https://arxiv.org/html/2511.22853) |
| (b)+(c) AR + flow head | Sundial-base | ~128M | 249-949 ms on an M1 Pro CPU | [Sundial GitHub](https://github.com/thuml/Sundial) |
| (c) AR on OHLCV tokens | Kronos mini/small/base | 4.1M / 24.7M / 102.3M | Multi-GPU `torchrun` fine-tuning scripts | [Kronos GitHub](https://github.com/shiyu-coder/Kronos) |

- QLoRA reduces memory enough "to finetune a 65B parameter model on a single 48GB GPU" with 4-bit NF4 quantization. — [QLoRA](https://arxiv.org/abs/2305.14314)
- RTX 5080 (Blackwell, sm_120) needs PyTorch ≥ 2.7.0 built with CUDA 12.8. 2.7.0 was the first stable release with native sm_120 support. — [Salad PyTorch/RTX 50 guide](https://docs.salad.com/container-engine/tutorials/machine-learning/pytorch-rtx5090.md); [PyTorch forum](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)

**Predictability base rates (price-only or near-price inputs)**
- Pretrained time-series models vs the random walk on US mega-cap returns: "small and sparse" gains (2026-06-25). — [Noguer i Alonso & Franklin](https://arxiv.org/abs/2606.27100)
- A reasoning LLM (gpt-5) shows no net advantage after costs for one-day stock ranking (2025-11-05). — [Sodha](https://arxiv.org/abs/2511.08608)
- Augmentation degrades near-efficient directional prediction (2026-04-16). — [Sohm et al.](https://arxiv.org/abs/2604.14498)
- Post-cutoff LLM agents ≈ buy-and-hold (Mar-Jun 2025). — [StockBench](https://arxiv.org/html/2510.02209v1)
- Even text-based LLM signals die at ~20 bps round-trip costs and decayed over 2021-2024. — [Lopez-Lira & Tang](https://arxiv.org/html/2304.07619)

### Inferences
**(a) Text LLM**
- Use only with text inputs (news, filings), as a classifier or embedder feeding a small model, scored with a model whose cutoff precedes the test window.
- Do not use it as the forecaster on raw numbers. Independent ablations show no accuracy gain at ~10-1,500× the training time and 1.2-28× the inference time. The default Time-LLM memory footprint (32-39 GB) exceeds 16 GB. Memorization makes backtests on 2005-2025 data untrustworthy.
- Good for: turning news into features, writing and reviewing code, summarizing filings.

**(b) Diffusion / flow matching**
- Use for distributions and scenarios: synthetic stress paths, volatility and risk forecasting, order-book simulation, and generating many plausible futures for position sizing.
- For forecasting, prefer a flow-matching or quantile head on a small autoregressive transformer (the Sundial pattern) over a standalone diffusion forecaster. Classic diffusion forecasters (TimeGrad, CSDI, SSSD) are ~30-2,200× slower to sample than a one-step generator, and an independent benchmark shows them weak at longer horizons.
- Do not expect synthetic data to rescue a directional model.

**(c) Small autoregressive or patch transformer: the default for stockai**
1. Start with a PAttn/PatchTST-sized model (~0.2-10M params) on returns and volume features. It trains in minutes per configuration on the RTX 5080, so walk-forward retraining and many seeds are affordable.
2. If the "next-token" framing is wanted literally, use Kronos-style or Chronos-style tokenization, or fine-tune the open Kronos-small/base weights. Note their 512-token context for small/base, and that the repo's fine-tuning scripts assume multi-GPU `torchrun` (one GPU should work with `nproc_per_node=1`, untested here).
3. Use zero-shot Chronos, Sundial or Kronos as free baselines before training anything.
4. Keep correlation2's evaluation: a held-out year, $10k, costs of 5-20 bps, SPY/QQQ, and same-turnover random accounts.

**Expectation setting.** Nothing found in 2024-2026 shows a price-only model of any family producing a cost-surviving daily edge on liquid US stocks. Treat a positive result as a likely bug or leak until it survives the random-account test.

### Gaps
- No source measured any of these models on an RTX 5080 specifically. GPU numbers are from A100, A6000 and V100 cards, plus CPU timings for Sundial.
- No study found compares (a), (b) and (c) head-to-head on the same daily US-equity cross-section with costs. The recommendation combines evidence from separate benchmarks.
- I found no direct evidence for price-only next-week (5-day) horizons specifically. The cited next-week evidence is text-based (Jiang-Li-Wang news momentum, via discussant slides).
