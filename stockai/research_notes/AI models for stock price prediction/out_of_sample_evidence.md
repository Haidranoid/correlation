# Out-of-sample evidence on ML/AI stock-return prediction from historical prices (gross vs net of costs), and the pitfalls that inflate published results

Research date: 2026-10-18.
- Every number carries its test period and universe.
- "Gross" means before transaction costs; "net" means after.
- "(secondary)" marks a figure I saw only in a summary, press release or blog, not in the paper itself.
- Numbers I derived myself from published figures appear only under Inferences, and are labelled as derivations.

## 1. Independent evaluations of time-series foundation models (TSFMs) on stock returns: zero-shot vs fine-tuned vs pre-trained from scratch (2024-2026)

### Takeaway
Off-the-shelf TSFMs are close to useless zero-shot on stock returns. This covers Chronos, TimesFM, Moirai, TimeGPT and about a dozen others. Out-of-sample R² is below zero, daily directional accuracy is about 50-51%, and skill against a random walk is of order 0.001. Fine-tuning helps little.

Pre-training from scratch on return data does better. On US daily data (2001-2023), equal-weighted long-short gross Sharpe ratios reach about 3.7-7.7. Charging 20 bps per trade, or a size-dependent 11-21 bps, turns every model negative, including the CatBoost benchmark.

The only fine-tuned-TSFM trading result on S&P 500 stocks (2023, zero costs) has a Sharpe of 1.68. A plain AR(1) model gets 1.58 on the same test.

### Cited Findings
**Rahimikia, Ni & Wang, "Re(Visiting) Time Series Foundation Models in Finance"** (SSRN/arXiv, Nov 2025; FoFI 2026 conference version dated Nov 2025; not peer-reviewed)

*Design*
- Data: daily firm-level excess returns, 1990-2023, across 94 countries. The US subset is CRSP common stocks on NYSE/AMEX/NASDAQ with price ≥ $5. The bottom 5% of market cap per country-day is excluded, and returns are winsorized to [-1, 1] — [arXiv 2511.18578 v1](https://arxiv.org/html/2511.18578v1); [SSRN 5770562](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5770562)
- Target: next-day excess return. Out-of-sample period 2001-2023, with models retrained yearly on an expanding window. The US OOS sample is 18,144,743 observations on 10,171 securities. Input windows are 5, 21, 252 and 512 days. Portfolios are equal-weighted decile long-short, rebalanced daily — [arXiv 2511.18578 v1](https://arxiv.org/html/2511.18578v1)

*Zero-shot (US)*
- Chronos (large), window 512: OOS R² −1.37% (vs a zero forecast), directional accuracy "just above 51%".
- TimesFM (500M): R² −2.80%, accuracy "just below 50%".
- Gross long-short annualized returns: Chronos (large) 20.17% (Sharpe 2.92), TimesFM (500M) −1.47%, CatBoost 47.25%.
- "Across all window sizes, off-the-shelf pre-trained TSFMs deliver markedly lower R² than benchmarks" — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf)
- An extended zero-shot test of 12 more models (Chronos-Bolt, TimesFM 2.5, Moirai 1/2, Kairos, Moment, Lag-Llama, TiRex, FlowState, TTM, Toto, Sundial) finds "pronounced weakness in goodness-of-fit" — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf)

*Fine-tuned*
- Fine-tuning "yields limited improvements and fails to close the performance gap with benchmarks". "Most fine-tuned TSFM performance deteriorates, except for Chronos (large)", and even that "does not translate into economic gains" — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf)

*Pre-trained from scratch*
- Models are pre-trained yearly using only data available up to that point. The authors say this avoids the look-ahead bias that "may arise when employing off-the-shelf TSFMs".
- Chronos (small) R² improves from −77.07% to −3.18% at window 5, and from −1.27% to −0.59% at window 512. It still trails the benchmarks on goodness-of-fit.
- Gross annualized return and Sharpe at window 512: Chronos (small) 36.84% / 5.42; TimesFM (20M) 30.36% / 3.66.
- With factor and synthetic-data augmentation, Chronos (small) reaches the paper's best directional accuracy, 51.74% (CatBoost: 51.16%), with 41.89% / Sharpe 6.78 (CatBoost: 47.25% / 6.46) — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf)

*Benchmarks (US, gross, equal-weighted daily long-short, 2001-2023)*
- CatBoost: 42.5-47.3% annualized, about 10.6-11.6 bps per day, annual volatility about 6.9-7.4%, Sharpe 5.74-6.79 depending on window.
- Plain OLS with Huber loss earns about 45% annualized with Sharpe about 4.9 (Table 5).
- The best benchmark R² is only −0.03% (CatBoost, window 512, accuracy 51.16%) — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf)

*Where the predictability sits (CatBoost)*
- Small caps: R² 0.51%, accuracy 52.29%. Large caps: R² −0.37%, accuracy 50.76%.
- "Across all model classes … the long leg of the portfolio consistently outperforms the short leg" — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf)

*Transaction costs (Table 12; EW daily long-short; all TSFMs pre-trained from scratch)*
- Scenarios: 0, 20 and 40 bps, plus a "mixed" scheme following Frazzini et al. (2012): about 21.3 bps for small caps and 11.2 bps for large caps.
- "Moving from 0 bps to 20 bps (and beyond) drives all long–short strategies to negative Sharpe ratios across window sizes, regardless of model class."
- CatBoost's average Sharpe: about 6.44 at 0 bps, −3.62 at 20 bps, −13.59 at 40 bps.
- The most cost-resilient model, TimesFM (20M) with JKP-factor augmentation at window 512: gross Sharpe 7.68, −2.03 at 20 bps, −11.65 at 40 bps, −1.87 under mixed costs.
- Every entry in the US, global, JKP-augmented and synthetic-augmented panels is negative at 20 bps and under mixed costs — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf)

*Decay, non-US results, code*
- "The cross-model annual average Sharpe ratio declines materially over time … The benchmark also exhibits a clear and persistent negative trend across all window sizes." TSFMs decline more slowly at long windows — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf)
- The authors report "broadly consistent results" in seven non-US markets: Hong Kong, Taiwan, South Korea, Germany, UK, India and Australia — [arXiv 2511.18578](https://arxiv.org/html/2511.18578v1)
- Code: [GitHub DeepIntoStreams/TSFM_Finance](https://github.com/DeepIntoStreams/TSFM_Finance). The author's site says 600+ finance-pretrained TSFMs are released on FinText.ai — [rahimikia.com](https://www.rahimikia.com/)

**Noguer i Alonso & Franklin, "Pretrained Time-Series Foundation Models for Financial Return Forecasting"** (arXiv 2606.27100, posted 2026-06-25)
- Universe: five US mega-caps (GOOG, AAPL, AMZN, JPM, META), daily data 2014-09-15 to 2026-02-15.
- Evaluation: 2024-08-22 to Feb 2026, 20-business-day horizon, context of 512, 10 rolling origins.
- Models: six zero-shot TSFMs (TimeGPT, TimeGPT-LH, TimesFM-2.5, Moirai-2.0, Chronos v1, Chronos-2), against per-ticker NBEATS, NHITS, PatchTST, iTransformer and KAN, plus random walk and seasonal naive.
- Results: TSFMs take 8 of 10 task wins, but skill scores against the random walk are "of order 10^-3" and gains are "small and sparse". A one-sided Diebold-Mariano test rejects only for Chronos/AMZN and Moirai-2.0/GOOG.
- Conclusion: the paper does not support TSFMs as "reliable sources of statistically robust alpha". It measures MAE only, reports no trading results or costs, and does not discuss pretraining contamination — [arXiv 2606.27100](https://arxiv.org/html/2606.27100)

**Fu, Hirano & Imajo, "Financial Fine-tuning a Large Time Series Model"** (arXiv 2412.09880, Dec 2024)
- Method: continual pre-training of TimesFM on about 90-100M points (TOPIX500 and S&P 500 stocks, FX, crypto and others), with data ending before 2023-01-01. Testing runs from 2023 on.
- Market-neutral Sharpe at horizon 128, assuming "no trading costs":

  | Market | Fine-tuned | Original TimesFM | Random | AR(1) |
  |---|---|---|---|---|
  | S&P 500 | 1.68 | 0.42 | 0.03 | 1.58 |
  | TOPIX500 | 1.06 | −1.75 | 0.11 | −0.82 |

- The fine-tuned model's S&P 500 annualized return is 3.6%. Its "neutral cost" (the cost level that zeroes returns) is 0.60% — [arXiv 2412.09880](https://arxiv.org/html/2412.09880v1)

**Chen, Chen, Chen, Huang & Zhang, "Does Training on Future Data Pay? Look-Ahead Bias in Forecasting with Pretrained Models"** (arXiv 2609.20554, 2026-09-17)
- Setup: five sets of financial TSFMs with independently trained annual vintages (US, global and factor-augmented), across 14 equity markets.
- Result: vintages trained past the forecast origin "generally reduce accuracy". MSFE is higher in 18 of 20 US model-horizon combinations. Median certainty-equivalent returns, exposed minus point-in-time, are −1.77 points a year in the US and −2.14 points internationally — [arXiv 2609.20554](https://arxiv.org/pdf/2609.20554)

**How much finance is in generic TSFM pretraining corpora**
- Estimated by the Kronos authors: Chronos 0.45%, Moirai 0.10%, MOMENT 1.60%, Sundial 1.02%, Time-MoE <0.01%, Timer 0.03% (Kronos: 100%) — [Kronos arXiv PDF](https://arxiv.org/pdf/2508.02739v1)
- Time-MoE's own paper puts finance at 1,715 sequences, or 0.0001% of Time-300B, none of it equities — [Time-MoE](https://arxiv.org/pdf/2409.16040)
- Chronos lists CIF 2016 and exchange rates as its finance data — [Chronos](https://arxiv.org/pdf/2403.07815)
- Zero-shot RankIC of generic TSFMs on financial price forecasting, averaged over assets and frequencies in the Kronos paper:
  - Moirai: 0.0000-0.0009
  - TimesFM: 0.0021
  - Chronos: 0.0047-0.0078
  - Time-MoE: 0.013-0.014
  - Moment: negative
  - Source: [Kronos arXiv HTML](https://arxiv.org/html/2508.02739v1)

**Volatility forecasting (adjacent task, same pattern)**
- Brini (arXiv 2607.05291, Jul 2026): 9 zero-shot TSFMs vs 8 HAR-family models on 50 equity/FX/futures assets. Averaged across assets, only Tiny Time Mixers beats Log-HAR at every horizon, and narrowly — [arXiv 2607.05291](https://arxiv.org/abs/2607.05291)
- Finance Research Letters (2026), "Foundation models do not beat simple volatility benchmarks: Evidence from 5000 Chinese stocks": zero-shot Kronos reportedly ranks last every year 2023-2025 against HAR-RV. A blog reports its errors at about 2.4-2.6× HAR-RV (secondary) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1544612326011347); [Substack](https://aidriventrading.substack.com/p/kronos-ai-and-the-real-test-promise)

### Inferences
- Rahimikia et al.'s from-scratch Chronos/TimesFM, and StockGPT (Section 2), are essentially the model stockai plans to build: a GPT-style model trained from scratch on daily returns. The best near-independent evidence for this model class on US data (2001-2023):
  - gross equal-weighted daily long-short Sharpe of about 4-8;
  - directional accuracy of about 51-52%;
  - negative net Sharpe at 20 bps per trade, and under size-based realistic costs.
- That is the same outcome correlation2 reached: nothing survived 5-20 bps. stockai should expect it too unless it changes *what is traded* (large caps, low turnover, multi-day holds), not just the model.
- Break-even cost (my derivation from Rahimikia Tables 5 and 12). CatBoost earns about 46%/yr gross at about 7% volatility, and its Sharpe falls from about +6.4 to −3.6 when 20 bps is added. That implies about 3.5%/yr of cost per bp of per-trade cost, so it breaks even near 13 bps per trade. The paper's own estimate for small caps, where the predictability sits, is about 21 bps.
- Plain OLS gets a gross Sharpe of about 4.9 in the same equal-weighted daily setup. Most of the "gross alpha" in daily cross-sectional prediction is therefore generic short-horizon reversal and liquidity provision (Sections 3 and 5) that any model picks up.
- Zero-shot TSFMs are useful only as a sanity baseline. A stockai model that cannot beat zero-shot Chronos, a zero-return forecast and an AR(1) has no edge. Fu et al.'s fine-tuned TimesFM (Sharpe 1.68) barely beats AR(1) (1.58).
- Compute is not the bottleneck for stockai (my inference). The models behind this evidence are small: StockGPT is about 0.93M parameters; Rahimikia's from-scratch Chronos and TimesFM models are about 8M-50M; Kronos-small is 24.7M and Kronos-base 102.3M. Models of that size train comfortably on one 16 GB RTX 5080. The binding constraints are signal-to-noise, costs and data quality.
- Contamination is a smaller problem for numeric return series than for LLMs. Generic TSFM corpora contain little finance, and Chen et al. (2026) find post-origin training *hurt* accuracy. Off-the-shelf weights used on past periods still break point-in-time discipline. Training stockai from scratch with a strict time split, as Rahimikia et al. did, avoids the issue.

### Gaps
- Not extracted from Rahimikia et al.: non-US numbers, value-weighted results, turnover, and whether "20 bps" applies per one-way trade or per unit of turnover.
- No peer-reviewed independent replication of any TSFM long-short result on US stocks.
- No TSFM study reports a positive net-of-cost daily Sharpe on a broad US universe.
- Noguer i Alonso & Franklin's point MAE values were not in the extracted text.

## 2. StockGPT (Dat Mai, 2024) and Kronos (Shi et al., 2025): specific claims and independent tests

### Takeaway
**StockGPT.** The headline result is 119%/yr, Sharpe 6.5, daily, 2001-2023. It is gross, equal-weighted and concentrated in small stocks:
- 27%/yr (Sharpe 1.0) value-weighted, with a −76% drawdown;
- 26% (Sharpe 1.7) if trading waits a single day.

Its only cost scenario is 5 bps per trade, unrealistically low for its universe. I found no independent replication (searches through Oct 2026).

**Kronos.** The headline claims are RankIC +93%, and +18-21%/yr annualized excess return. Both come from the authors' own long-only tests on Chinese A-shares over a single window starting July 2024. The two independent checks I found are both negative:
- 13 US large caps at a 30-minute horizon: no better than always predicting "up";
- volatility forecasting: ranked last against HAR-RV.

### Cited Findings
**StockGPT** (arXiv 2404.05101; v1 2024-04-07, v2 2024-04-09, v3 2024-10-23; single author; no journal listed)

*Model and data*
- Decoder-only GPT: 4 blocks, 4 heads, embedding size 128, context 256, about 0.93M parameters.
- Daily returns in bps are binned into 50-bp intervals, giving 402 tokens.
- Trained on CRSP NYSE/AMEX/NASDAQ common stocks 1926-2000 (main model NYSE only); out of sample 2001-2023 — [arXiv v3](https://arxiv.org/html/2404.05101v3); [abs](https://arxiv.org/abs/2404.05101)

*Daily strategy*
- Portfolios are formed "at the market close of each trading day" and held to the next close. Stocks "below the 10th percentile market value" are removed.
- Equal-weighted long-short: 119.1%/yr, Sharpe 6.5 (no price filter); 73.8% / 4.7 with a $5 price filter.
- Value-weighted: 27.0% / Sharpe 1.0, or 18.8% / 0.8 with the $5 filter. The VW maximum drawdown is −76.3% — [arXiv v3](https://arxiv.org/html/2404.05101v3)

*Size dependence*
- "Without the market cap restriction, the StockGPT-based portfolio would earn 230% annually with a Sharpe ratio of 10."
- Removing stocks below the 30th size percentile gives "50% annually with a Sharpe ratio of 2.9" — [arXiv v3](https://arxiv.org/html/2404.05101v3)

*One-day delay*
- Skipping one day cuts the equal-weighted result to 26.1% / Sharpe 1.7.
- The Fama-MacBeth slope falls from 0.50 (t = 25.18) at horizon 1 to 0.09 (t = 10.20) at horizon 2 — [arXiv v3](https://arxiv.org/html/2404.05101v3)

*Costs and exposures*
- All results are gross. The only cost analysis is a "worst-case" scenario: 400% turnover with "each trade costs 5 basis points" leaves 69%/yr net. No break-even analysis is given.
- The daily portfolio loads on short-term reversal (EW β = 0.57, t = 11.66) — [arXiv v3](https://arxiv.org/html/2404.05101v3)

*Monthly variant*
- Equal-weighted long-short earns about 13.5%/yr, Sharpe 0.9-1.0. With the bottom half of stocks removed it earns about 10%, Sharpe 0.7. No cost or turnover analysis — [arXiv v3](https://arxiv.org/html/2404.05101v3)

*Replication status*
- I found none for 2024-2026 in arXiv, SSRN, Semantic Scholar or alphaXiv searches; only summaries and blogs — [alphaXiv](https://www.alphaxiv.org/overview/2404.05101v3); [Semantic Scholar](https://www.semanticscholar.org/paper/StockGPT:-A-GenAI-Model-for-Stock-Prediction-and-Mai/47b5630c5b8407f0b07047ee21e41664dd102ab9)

**Kronos** (Shi, Fu, Chen, Zhao, Xu, Zhang & Li; arXiv 2508.02739, 2025-08-02; AAAI-26 proceedings, 2026-03-14)

*Model*
- A tokenizer quantizes OHLCV(A) candles (binary spherical quantization); a decoder-only transformer is then pre-trained autoregressively on the tokens.
- Sizes: 24.7M (small), 102.3M (base), 499.2M (large) parameters.
- Corpus: "over 12 billion K-line records from 45 global exchanges" (the appendix says "over 40 exchanges across more than 30 countries"), at 1-minute to weekly frequencies.
- Pre-training data "extends up to June 2024", and all tests begin July 2024 — [arXiv abs](https://arxiv.org/abs/2508.02739); [PDF](https://arxiv.org/pdf/2508.02739v1); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39730)

*Claims*
- RankIC +93% vs the leading TSFM and +87% vs the best non-pre-trained baseline; volatility MAE −9%; synthetic-data fidelity +22% — [arXiv abs](https://arxiv.org/abs/2508.02739)

*Absolute IC levels*
- Averaged over assets and frequencies, mixing intraday and daily data from stock, crypto and FX markets.
- Kronos small / base / large: IC 0.0431 / 0.0435 / 0.0440; RankIC 0.0254 / 0.0258 / 0.0267.
- Best full-shot baseline (TimesNet): IC 0.0317, RankIC 0.0143 — [arXiv HTML](https://arxiv.org/html/2508.02739v1)

*Investment simulation*
- Universe: Chinese A-shares from Qlib, with "no additional filtering".
- Strategy: daily top-k/drop-n, long-only, equal-weighted. CSI 300 uses k = 50 with at most 5 swaps a day; CSI 800 uses k = 200 with at most 10. Minimum hold is 5 days; cost is 0.15% per trade. Performance is measured as excess return over the index.
- Average annualized excess return / information ratio across CSI 300 and CSI 800:
  - Kronos-small: 17.89% / 1.42
  - Kronos-base: 18.89% / 1.52
  - Kronos-large: 20.84% / 1.65
  - Moirai-large: 15.77% / 1.25
  - Moment-large: 16.81% / 1.37
  - TimesFM: 10.72% / 1.19
  - Chronos-large: −3.02% / −0.34
- Source: [arXiv HTML](https://arxiv.org/html/2508.02739v1); [PDF](https://arxiv.org/pdf/2508.02739v1)

*Data cleaning*
- The authors note that raw K-lines are "frequently contaminated by artifacts stemming from low liquidity, price limits, or data feed errors". They filter price jumps, illiquid stretches and stagnant prices — [PDF](https://arxiv.org/pdf/2508.02739v1)

*Independent tests*
- (a) A non-peer-reviewed GitHub report, secondary via Substack: 13 large US stocks, Jan-May 2025, Kronos-small, 390 forecasts at a 30-minute horizon, 0.1% cost.
  - Directional accuracy was 54.6%, against 54.4% for always predicting "up" (Δ = 0.26 pp, p ≈ 0.92). The average profit per trade after costs was 0.049%.
- (b) The FRL 2026 volatility study (Section 1).
- I found no replication of the A-share backtest or of the RankIC claims — [Substack](https://aidriventrading.substack.com/p/kronos-ai-and-the-real-test-promise)

### Inferences
- StockGPT's own numbers show most of its edge is untradeable. The VW gross return is 27%, a one-day delay removes about 80% of the EW return, and the portfolio loads heavily on reversal, the signature of bid-ask bounce and liquidity provision (Sections 3 and 5).
- Cost check (my derivation, using the paper's own 400%/day turnover):
  - At 5 bps: 20 bps a day, or about 50%/yr. That already exceeds the 27% VW gross return, if VW turnover is similar.
  - At the about 21 bps small-cap cost used by Rahimikia et al.: about 85 bps a day, or about 215%/yr, far above the 119% EW gross.
  - StockGPT is not evidence of a net edge.
- Kronos's simulation measures long-only excess return against a Chinese index over about one year, in a retail-heavy market with different microstructure (Section 3, Leippold et al.). That is weak evidence for US performance against SPY/QQQ net of costs. Its turnover-limited construction (at most 5 of 50 names swapped a day, 5-day minimum hold) is worth copying.
- Any Kronos backtest on data before July 2024 is in-sample for its weights. stockai's held-out year (2024-06-28 to 2025-06-27) starts right at that boundary.
- The independent US test, where Kronos did no better than "always up", is exactly correlation2's kind of baseline. Always compare against always-long, zero-return, AR(1)/reversal and random-pick-same-turnover portfolios.

### Gaps
- No independent StockGPT replication, and no net-of-cost StockGPT results for VW portfolios or large caps only.
- Kronos's backtest end date was not stated in the extracted text; the paper was posted 2025-08-02.
- No Kronos result exists on US daily cross-sectional long-short returns net of costs.
- The GitHub report behind the US Kronos test was not verified at source.

## 3. Academic ML return-prediction benchmarks: inputs, horizon, OOS period, gross vs net, turnover, and where returns concentrate

### Takeaway
Published ML Sharpe ratios are overwhelmingly gross. They are 2-6× higher for equal-weighted portfolios that include small or illiquid stocks than for value-weighted or large-cap portfolios. Price-based daily and weekly signals turn over 800-1000% a month.

Where costs are modelled, net performance halves or turns negative: Fischer-Krauss, Krauss-Do-Huck, Guijarro-Ordonez-Pelger-Zanotti, Jensen-Kelly-Malamud-Pedersen and Rahimikia et al. Daily S&P 500 ML statistical-arbitrage profits disappeared after about 2010.

What survives costs is mainly:
- monthly, value-weighted strategies using hundreds of firm characteristics, not price alone (about 0.6-1.4% a month net, 2005-2021);
- low-volatility residual-reversal strategies on the largest about 550 stocks, with net Sharpe of about 1 but only about 5%/yr unlevered.

### Cited Findings
**Gu, Kelly & Xiu (2020, RFS), "Empirical Asset Pricing via Machine Learning"** (NBER WP 25398, revised Sep 2019)
- Inputs: 94 characteristics (many price, liquidity and volatility based), interactions with 8 macro variables, and 74 industry dummies.
- Universe and horizon: about 30,000 US stocks, 1957-2016; out of sample 1987-2016; monthly horizon — [NBER PDF](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf)
- Monthly OOS R² for all stocks: NN3 0.40%, NN4 0.39%, RF 0.33%, GBRT 0.34%, OLS-3 0.16%, full OLS −3.46%.
- By size: top-1000 stocks 0.70% (NN3), bottom-1000 0.45%. Annual-horizon R² is 3.40-3.60% for NN3/NN4 — [NBER PDF](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf)
- Portfolios (gross):
  - NN4 VW decile long-short: 2.3%/month (27.1%/yr), volatility 20.1%, Sharpe 1.35.
  - EW Sharpe 2.45; EW excluding stocks below the 20th NYSE size percentile, 1.69.
  - Turnover 110-130%/month for NN1-NN5. For reference, a short-term-reversal decile spread turns over 172.6%/month and a size spread 22.9%/month.
  - The dominant signals are "variations on momentum, liquidity, and volatility". No net-of-cost results are reported — [NBER PDF](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf)
- R² benchmark: "a roughly 3% inflation in out-of-sample R²s if performance is benchmarked against historical averages". OLS-3 scores 3.74%/month against the historical mean, but 0.16% against zero — [NBER PDF](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf)

**Jiang, Kelly & Xiu (2023, JF 78: 3193-3249), "(Re-)Imag(in)ing Price Trends"** (CNN on price-chart images)
- Inputs: images of 5, 20 or 60 days of OHLC prices, volume and moving average, for US stocks.
- Training: once, on 1993-2000; the model is then held fixed for 2001-2019. It predicts the up/down probability over the next 5, 20 or 60 days — [WP PDF (NUS AIDF copy)](https://www.aidf.nus.edu.sg/wp-content/uploads/2022/02/Xiu-Re-Imagining-Price-Trends.pdf); [Yale listing](https://economics.yale.edu/research/re-imagining-price-trends)
- Monthly horizon:
  - Decile long-short Sharpe "as high as 2.4 for equal-weight portfolios and 0.5 for value-weight portfolios". One-week reversal gets 1.2 (EW) and 0.3 (VW).
  - Accuracy "in excess of 53%" — [WP PDF](https://www.aidf.nus.edu.sg/wp-content/uploads/2022/02/Xiu-Re-Imagining-Price-Trends.pdf)
- Weekly horizon (I5/R5, gross):
  - EW: 0.83%/week, Sharpe 7.15, turnover 862%/month.
  - VW: 0.23%/week, Sharpe 1.49, turnover 947%/month.
  - Weekly reversal: 2.84 (EW) and 0.77 (VW).
  - The authors: "Rather than interpreting these results as trading results that are achievable in practice, the one week strategy is meant to demonstrate the impressive predictive strength of images" — [WP PDF](https://www.aidf.nus.edu.sg/wp-content/uploads/2022/02/Xiu-Re-Imagining-Price-Trends.pdf)
- Secondary (BSIC blog):
  - For the 500 largest stocks the Sharpe "drops to slightly above 1".
  - The blog's own replication on 20 large US stocks (test 2021-2025, 20-day target) gave a gross long-short Sharpe of 0.42, against 0.78 for the S&P 500 — [BSIC](https://bsic.it/technical-analysis-on-steroids/)

**Fischer & Krauss (2018, EJOR), LSTM on the S&P 500** (FAU DP 11/2017)
- Design:
  - Inputs: a 240-day sequence of each stock's daily returns.
  - Universe: survivor-bias-free S&P 500 constituents from month-end lists; trading Dec 1992 to Oct 2015.
  - Target: beats the cross-sectional median tomorrow. Portfolio: long top 10, short bottom 10, rebalanced daily — [EconStor PDF](https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf)
- Results, gross vs net at 5 bps per half-turn:
  - LSTM: 0.46%/day gross (Sharpe 5.8), 0.26% net.
  - Random forest: 0.43% gross, 0.23% net.
  - DNN: 0.32% gross, 0.12% net.
  - Logistic: 0.26% gross, 0.06% net (not significant).
  - Accuracy of the selected stocks: 54.3% — [EconStor PDF](https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf)
- 100,000 random "monkey" long-short portfolios average zero; the best earns 0.05%/day gross.
- 2010/01-2015/10 was "a time of deterioration". After costs the random forest "destroys more than 1 USD in value", and the LSTM is "able to keep capital approximately constant" — [EconStor PDF](https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf)
- Source of profits: high-volatility stocks with a "short-term reversal return profile".
  - A simple reversal rule earns 0.23%/day gross but 3 bps net. It explains R² of 52.1% (long leg) and 53.6% (short leg) of LSTM leg returns.
  - The remaining LSTM alpha is 31 bps/day gross and 11 bps net — [EconStor PDF](https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf)

**Krauss, Do & Huck (2017, EJOR), DNN, gradient-boosted trees, random forest and ensemble on the S&P 500** (FAU DP 03/2016)
- Same universe, with lagged returns as inputs, 1992-2015. The ensemble long-short (k = 10) earns 0.45%/day gross and 0.25% net at 5 bps per half-turn (73%/yr net) — [EconStor PDF](https://www.econstor.eu/bitstream/10419/130166/1/856307327.pdf)
- Ensemble net results by sub-period:
  - 12/92-03/01: over 200%/yr, Sharpe 6.7.
  - 04/01-08/08: 22%/yr, Sharpe 0.51.
  - 09/08-12/09: over 400%/yr.
  - 01/10-10/15: every model −14% to −25%/yr net, "although … all our models still produce positive mean returns prior to transaction costs" — [EconStor PDF](https://www.econstor.eu/bitstream/10419/130166/1/856307327.pdf)

**Guijarro-Ordonez, Pelger & Zanotti, "Deep Learning Statistical Arbitrage"** (arXiv 2106.04028, v2 Oct 2022)
- Universe: daily returns of "roughly the 550 largest and most liquid stocks in the U.S.". Residuals cover 1998-2016; out-of-sample trading runs Jan 2002 to Dec 2016. A CNN+Transformer works on residuals from FF, PCA or IPCA factor models — [arXiv PDF](https://arxiv.org/pdf/2106.04028v2)
- Gross: Sharpe up to about 4.2 (IPCA, K = 5: Sharpe 4.16, mean 8.7%, volatility 2.1%). A parametric OU-threshold benchmark reaches about 0.2-1.2 — [arXiv PDF](https://arxiv.org/pdf/2106.04028v2)
- With frictions of 5 bps per transaction plus 1 bp per short position per period, and no market impact: Sharpe 0.94-1.24 and mean 4.8-6.6%/yr (vs 3.2-4.2 gross on IPCA residuals) — [arXiv PDF](https://arxiv.org/pdf/2106.04028v2)

**Kelly, Malamud & Zhou (2024, JF), "The Virtue of Complexity in Return Prediction", and critiques**
- Method: US market timing using 12,000 random Fourier features of 15 predictors, on rolling 12-month training windows.
- Result over CRSP 1926-2020: OOS Sharpe gain vs buy-and-hold of "roughly 0.47 per annum with t-statistics near 3.0", despite "substantially negative" OOS R² — [NBER w30217 PDF](https://nber.org/system/files/working_papers/w30217/w30217.pdf)
- Nagel (Aug 2025; NBER w34104 / BFI WP 2025-104):
  - With features far outnumbering observations, the forecast "reduces to a recency-weighted average" of the 12 training returns. That is "a volatility-timed momentum strategy" that "happened to perform well in historical data".
  - On artificial data with reversals, the method "still constructs the same volatility-timed momentum strategy, which then performs poorly" — [BFI PDF](https://bfi.uchicago.edu/wp-content/uploads/2025/08/BFI_WP_2025-104.pdf); [NBER](https://www.nber.org/papers/w34104)
- Other critiques, as summarized by Rahimikia et al.:
  - Buncic (2025): results are "largely driven by modeling choices, notably the zero-intercept restriction and the aggregation method".
  - Cartea et al. (2025): a "limited virtue of complexity" when features are noisy.
  - Kelly & Malamud (2025) reply that the findings "remain robust" — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf); [SNU slides on the debate](https://fre.snu.ac.kr/wp-content/uploads/sites/9/2025/09/250924_Virtue-of-Complexity.pdf)

**Kelly, Kuznetsov, Malamud & Xu, "Artificial Intelligence Asset Pricing Models"** (NBER w33351, Jan 2025, revised Jun 2026; not peer-reviewed)
- Inputs: monthly JKP characteristics (132), US stocks with CRSP share codes 10-12, 1963-2022. Nano caps below the 1st NYSE size percentile are dropped. Out of sample 1968-2022.
- OOS SDF-portfolio Sharpe, gross. The paper contains no transaction-cost analysis.

  | Model | 1968-2022 | Post-2002 |
  |---|---|---|
  | FF6 | 1.01 | 0.74 |
  | HXZ | 1.77 | 0.95 |
  | BSV linear | 3.60 | 2.03 |
  | DKKM | 3.87 | 2.41 |
  | Linear attention | 3.89 | 2.25 |
  | MLP | 4.31 | 3.07 |
  | Transformer | 4.57 | 3.37 |

  Source: [NBER PDF](https://www.nber.org/system/files/working_papers/w33351/w33351.pdf)

**Avramov, Cheng & Metzker (2023, Management Science 69(5): 2587-2619), "Machine Learning vs. Economic Restrictions"**
- Deep-learning signals "extract profitability from difficult-to-arbitrage stocks" and from high limits-to-arbitrage periods.
- Excluding microcaps, distressed firms or high-volatility episodes considerably weakens profits. Trading costs reduce them further because of high turnover and extreme positions.
- Profits remain in long positions and in recent years — [HUJI CRIS abstract](https://cris.huji.ac.il/en/publications/machine-learning-vs-economic-restrictions-evidence-from-stock-ret/)
- Press release (secondary): VW risk-adjusted returns of 0.75-1.87%/month fall 62% excluding microcaps, 68% excluding non-rated firms and 80% excluding distressed firms — [finews.asia](https://finews.asia/finance/32603-cuhk-ai-based-stock-picking-overrated)

**Jensen, Kelly, Malamud & Pedersen, "Machine Learning and the Implementable Efficient Frontier"** (SFI WP 22-63, Aug 2022; listed as RFS in press, May 2026)
- A Markowitz-ML portfolio has a gross Sharpe of "roughly 2.0", but net of costs it "immediately dives into negative expected return territory".
- The cause is "excessive reliance on fleeting small-scale characteristics (e.g., 1-month reversal for small stocks)". Standard portfolio sorts are "also not implementable".
- Setting: 1981-2020, $10B AUM by 2020. A cost-aware "Portfolio-ML" fixes much of this — [AEA-hosted PDF](http://www.aeaweb.org/conference/2024/program/paper/SYTF3aaz); [SFI](https://www.sfi.ch/en/publications/machine-learning-and-the-implementable-efficient-frontier)

**Azevedo, Hoegner & Velikov, "The Expected Returns on Machine-Learning Strategies"** (AFA conference paper; the PDF shows no date, but it cites 2023 work and its sample ends Dec 2021)
- Design: up to 320 published anomalies (Chen-Zimmermann data) combined with 9 ML models. Monthly, VW with NYSE-breakpoint deciles, out of sample 2005-2021, with costs from high-frequency effective spreads.
- OOS R² ranges from 0.05% (OLS-Huber) to 0.76% (LSTM1).
- Gross returns reach 1.63%/month (Sharpe 0.32-1.11). Costs are about 20-25 bps a month at 60-70% one-sided monthly turnover.
- Net returns reach 1.42%/month (LSTM1, t = 3.99); ENET earns 0.64%. Three of nine strategies become insignificant, and cost-mitigation techniques "do not improve net anomaly performance" — [AFA PDF](https://afajof.org/management/viewp.php?n=75544)

**International follow-ups**
- China (Leippold, Wang & Zhou, JFE 2022):
  - "liquidity emerges as the most important predictor", unlike in the US.
  - "The retail investors' dominating presence positively affects short-term predictability, particularly for small stocks".
  - Performance stays "economically significant after transaction costs" — [IDEAS](https://ideas.repec.org/a/eee/jfinec/v145y2022i2p64-82.html)
- 46 markets with 148 characteristics (Cakici, Fieberg, Metko & Zaremba, JEDC 2023): ML extracts predictability mainly from momentum, reversal, value and size. Results depend on firm size and on the limits to arbitrage in each market (abstract via search) — [ICM repository](https://open.icm.edu.pl/handle/123456789/24915)

**Where daily and weekly price-based profits come from**
- Short-term reversal returns can be read as compensation for liquidity provision. Expected returns and conditional Sharpe ratios "increase enormously along with the VIX" in turmoil such as 2007-09 — [Nagel 2012, NBER w17653](https://nber.org/papers/w17653)
- Higher volatility goes with "faster, initially stronger reversals" — [Dai, Medhat, Novy-Marx & Rizova, NBER w30917 / FAJ 2024](https://www.nber.org/papers/w30917)
- This fits the 1999-2000 and 2008 spikes in Krauss-Do-Huck and StockGPT's 2009 jump.

**AQR practitioner view (Israel, Kelly & Moskowitz, JOIM 2020, "Can Machines 'Learn' Finance?")**
- "Return prediction—is a small data problem." For large investors "the monthly frequency is a sensible starting point … anything more frequent, even weekly let alone intra-day, quickly becomes prohibitive as trading costs rack up".
- The Fama-French 100 portfolios have a 71% average pairwise correlation, and 3 principal components explain at least 80% of their covariance — [AQR PDF](https://www.Aqr.Com/-/media/AQR/Documents/Alternative-Thinking/AQR-Alternative-Thinking-2Q19-Can-Machines-Learn-Finance.pdf?sc_lang=en)

### Inferences
- **Weighting and size.** EW Sharpe divided by VW Sharpe is about 2-5×: GKX 2.45 vs 1.35, JKX weekly 7.15 vs 1.49, StockGPT 6.5 vs 1.0. Size screens remove much of the rest: StockGPT goes from Sharpe 10 with no screen to 6.5 with a 10th-percentile screen and 2.9 with a 30th-percentile screen. An equal-weighted backtest over all Stooq symbols would be dominated by the stocks with the highest costs and the worst survivorship problems (Section 5).
- **Break-even costs** (my derivation; assumes turnover counts both sides of each leg, as in GKX/JKX, so treat as order of magnitude):
  - JKX weekly VW: 0.23%/week against about 190% of each leg traded per week, two legs → about 6 bps one-way.
  - JKX weekly EW: about 24 bps.
  - Fischer-Krauss: 0.46%/day against 4 half-turns a day → about 11.5 bps per half-turn over 1992-2015. After 2010, 5 bps already took the LSTM to roughly zero.
- **The typical edge.** The daily price-only literature on liquid US stocks converges on a gross edge of a few bps to about 10 bps per trade, about the same as realistic costs. The positive net results come from:
  - the 1990s;
  - crisis periods;
  - large-cap residual reversal with net Sharpe of about 1 but about 5%/yr unlevered (GPZ). Shorting and leverage cost extra for a $10,000 account.
  - monthly, fundamentals-heavy strategies (Azevedo et al.).

  None of these is a price-only daily next-token model.
- **Where the gross alpha sits.** Even linear models reach a daily EW Sharpe of about 5 (Rahimikia), and the profits load on reversal and VIX. The gross alpha in daily return prediction is mostly paid for supplying liquidity to small, volatile stocks. A model that predicts "next price" will rediscover this first, and it is exactly the component that bid-ask spreads absorb.

### Gaps
- No net-of-cost results exist for GKX's NN portfolios.
- For Avramov-Cheng-Metzker I have only press-release magnitudes.
- For JKX I used the working-paper tables, not the published JF version.
- I did not read the full Buncic or Kelly-Malamud 2025 papers.
- AIPM's Sharpe ratios are gross SDF-portfolio statistics, not tradable net returns.

## 4. Achievable daily/weekly rank IC from price/volume-only signals (US), realistic costs by liquidity tier, and how fast net performance falls as costs rise

### Takeaway
Rigorous US-specific RankIC numbers for price/volume-only signals are scarce. What exists points to:
- daily IC of about 0.001-0.006 for standard ML models on S&P 500 stocks (2021-2024);
- about 51-52% daily directional accuracy across the CRSP cross-section (2001-2023);
- RankIC of about 0.03-0.06 at a 5-day horizon on China's CSI 300, a more predictable, retail-heavy market.

Realistic one-way costs:
- about 1-10 bps for large and mega caps (spread half-widths of 7-10 bps VW on average over 1983-2018; NYSE effective-spread mode of about 5 bps in 2014);
- about 11 bps (large) to 21 bps (small) of measured institutional market impact;
- 80-120+ bps for microcaps.

Daily strategies trade nearly 100% of capital a day, so each 1 bp of per-trade cost removes roughly 2.5-5%/yr. Published net results flip from strongly positive at 0 bps to negative at 10-20 bps.

### Cited Findings
**Information coefficients and accuracy**
- AlphaAgent (arXiv 2502.16789, 2025):
  - Setup: S&P 500, OHLCV-only inputs, next-day target. Train 2015-2019, validate 2020, test 2021-01 to 2025-01, with a top-50/drop-5 strategy and a 0.0005 sell fee.
  - Daily IC / annualized excess return on the test period:

    | Model | IC | Excess return |
    |---|---|---|
    | LightGBM | 0.0011 | −2.64% |
    | Transformer | 0.0013 | −4.55% |
    | LSTM | 0.0028 | −1.51% |
    | TRA | −0.0003 | −8.51% |
    | StockMixer | 0.0030 | −2.49% |
    | AlphaForge | 0.0026 | +2.45% |
    | LLM-mined AlphaAgent factors | 0.0056 | +8.74% |

  - The same baselines on China's CSI 500 have IC 0.012-0.021 — [arXiv 2502.16789](https://arxiv.org/html/2502.16789)
- AlphaSchema (arXiv 2607.26642, Jul 2026): China CSI 300, 5-day close-to-close target, test 2023-2025.
  - RankIC: Alpha158 0.0504, LSTM 0.0587, GRU 0.0565, Transformer 0.0507, XGBoost 0.0373, Alpha360 0.0292. IC is 0.021-0.038.
  - Annualized excess returns after China-level costs (0.05% buy, 0.15% sell) are 2.5-11.9% — [arXiv 2607.26642](https://arxiv.org/pdf/2607.26642)
- Kronos: average RankIC 0.025-0.027 (IC about 0.043) across many markets and frequencies — [arXiv](https://arxiv.org/html/2508.02739v1)
- Rahimikia et al.: daily US directional accuracy 51.16% (CatBoost); 52.29% for small caps and 50.76% for large caps — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf)
- Fischer-Krauss: 54.3% accuracy on the top/bottom-10 picks, S&P 500, 1992-2015 — [EconStor](https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf)
- A cautionary 2025 example (arXiv 2510.14156):
  - Setup: 110 large S&P 500 stocks ("top 10 by market capitalization from each of the 11 GICS sectors"), 2015-2024, with daily return and turnover inputs.
  - Reported: daily Spearman IC of about 0.073-0.077 and Sharpe about 0.66-0.75.
  - Its reported ICIR of about 8.7 does not match its own IC and IC-SD columns, which imply about 0.76. Cost treatment is not stated — [arXiv 2510.14156](https://arxiv.org/html/2510.14156)
- Cross-Market Alpha (arXiv 2601.06499, v3 Sep 2026): 168 Alpha191 short-horizon price-volume signals on point-in-time S&P 500 constituents, 2002-2022, monthly VW decile portfolios.
  - Only 3 signals survive every robustness check, against 17 significant in the baseline. They cluster in volume-price interaction and short-term mean reversion.
  - The paper does "not evaluate turnover, transaction costs, or net-of-cost portfolio performance" — [arXiv 2601.06499](https://arxiv.org/html/2601.06499v3)
- Fundamental law of active management: IR = IC × √breadth — [CFI](https://corporatefinanceinstitute.com/resources/capital-markets/fundamental-law-of-active-management/)

**Cost levels by liquidity tier**
- Frazzini, Israel & Moskowitz (Dec 2012 NBER draft, "Trading Costs of Asset Pricing Anomalies"):
  - Data: "nearly a trillion dollars" of live trades by a large manager, 19 developed markets, 1998-2011.
  - Mean market impact 12.18 bps; 11.21 bps on average for large-cap trades and 21.27 bps for small-cap trades.
  - Short-term reversal runs at 305%/month turnover with 18.4 bps impact, about 6.75%/yr of cost. It "do[es] not survive transactions costs at sizes greater than $9 billion in the U.S." (break-even is given as $9.5B in the body) — [NBER draft PDF](https://conference.nber.org/conf_papers/f68262/f68262.pdf)
- Novy-Marx & Velikov (RFS 2016, author version):
  - Setup: US VW deciles with NYSE breakpoints, 1963-2013, costs from Hasbrouck Gibbs effective spreads.
  - Anomalies with turnover under 50%/month are mostly profitable net of costs after cost mitigation. Strategies with more than 90% monthly turnover always cost more than 1% a month.
  - Short-run reversal: 0.37%/month gross, 90.9% turnover, 1.65% cost, −1.28% net.
  - Industry momentum: 0.93% gross, −0.29% net.
  - Momentum: 1.33% gross, 0.68% net — [author PDF](https://mysimon.rochester.edu/novy-marx/research/ToAatTC.pdf)
- Chen & Velikov (JFQA 2023 / FEDS 2020):
  - The average anomaly's expected return is 4 bps/month net of effective spreads and post-publication decay; the best net "at best, 10 bps".
  - In 2014, NYSE effective spreads had a mode of about 5 bps and a mean of 16 bps, with about 20% of NYSE stocks above 20 bps. Anomaly portfolios paid a mean of 67 bps.
  - Costs "plummet in the 2000's" with decimalization and electronic trading — [JFQA](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/zeroing-in-on-the-expected-returns-of-anomalies/945133D5A3ECEEAF466AEE91551FD225); [FEDS 2020](https://www.federalreserve.gov/econres/feds/files/2020039pap.pdf)
- Medhat & Schmeling (RFS 2022): VW average half-spreads, 1983-2018.
  - Megacap strategy legs 0.07-0.08%; large-cap legs 0.08-0.10%; full-universe short-term-momentum legs 0.17-0.24%.
  - Microcap-heavy reversal legs 0.84-1.22%, so reversal's 1.94%/month cost "subsumes" its gross return — [City Univ. PDF](https://openaccess.city.ac.uk/id/eprint/31278/1/MS_short_term_mom_v27.pdf)
- Retail execution (Schwarz, Barber, Huang, Jorion & Odean, "Actual Retail Price", JF 2025; CQA slides):
  - Setup: 85,000 market orders worth $16M across 6 accounts at 5 brokers, Dec 2021-Jun 2022, on 128 stratified stocks. Orders were about $100 each (robust at $1,000-$5,000), bought and then sold about 30 minutes later.
  - Round-trip cost against the NBBO midpoint: TD Ameritrade −0.072%, E*Trade −0.197%, Fidelity −0.234%, Robinhood −0.314%, IBKR Lite −0.444%, IBKR Pro −0.462%. Paying the full quoted spread would cost −0.619% — [CQA slides](https://cqa.org/wp-content/uploads/2024/04/Huang_Actual-Retail-Price_CQA.pdf)

**Estimating spreads from free daily OHLC data (relevant to Stooq)**
- Corwin-Schultz (high/low over two days; 2009 NBER draft): monthly correlation with TAQ 0.873, against 0.694 for Roll — [NBER draft](https://projects.nber.org/confer/2009/mms09/Corwin_Schultz.pdf)
- Abdi-Ranaldo (close/high/low; RFS 2017): average monthly cross-sectional correlation with daily TAQ 0.74, against 0.37-0.65 for other estimators, over 1993-2015 — [PDF](https://www.alexandria.unisg.ch/server/api/core/bitstreams/dfb85399-5bcf-465c-8b63-9bfc97ad2ee0/content)
- EDGE (Ardia, Guidotti & Kroencke, JFE 2024; open/high/low/close):
  - Over 1,637,621 CRSP-TAQ stock-months, May 1993-Dec 2021, correlation with TAQ is 78.86% for EDGE, against 68.13% for Abdi-Ranaldo and 45.55% for Corwin-Schultz.
  - For large caps (Oct 2003-Dec 2021), daily-bar EDGE is negative in 41% of stock-months and 35 bps too high on average.
  - Implementations: the `bidask` package (CRAN/PyPI), which also includes the Roll, Corwin-Schultz and Abdi-Ranaldo estimators — [EDGE PDF](https://irf.fhnw.ch/bitstreams/e8c469d2-9c64-4004-8e87-53ee091f0cb1/download); [bidask](https://ftp.fau.de/cran/web/packages/bidask/readme/README.html)

**How fast net performance falls as costs rise**
- Rahimikia CatBoost (daily, EW, US 2001-2023): average Sharpe +6.44 at 0 bps, −3.62 at 20 bps, −13.59 at 40 bps — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf)
- Fischer-Krauss LSTM: 0.46%/day becomes 0.26% at 5 bps per half-turn — [EconStor](https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf)
- Krauss-Do-Huck ensemble: 0.45%/day becomes 0.25% — [EconStor](https://www.econstor.eu/bitstream/10419/130166/1/856307327.pdf)
- GPZ: Sharpe of about 4 becomes about 1 at 5 bps — [arXiv](https://arxiv.org/pdf/2106.04028v2)
- StockGPT: 119%/yr becomes 69% with 400% turnover at 5 bps — [arXiv](https://arxiv.org/html/2404.05101v3)
- Fine-tuned TimesFM on the S&P 500 in 2023: a 0.60% "neutral cost" zeroes it — [arXiv](https://arxiv.org/html/2412.09880v1)
- Novy-Marx-Velikov reversal: 0.37% gross becomes −1.28%/month net — [PDF](https://mysimon.rochester.edu/novy-marx/research/ToAatTC.pdf)

### Inferences
- **IC to Sharpe (my arithmetic).** A daily cross-section of 500 stocks gives breadth of about 126,000 a year (√ ≈ 355). An IC of 0.005 then implies a gross IR of about 1.8, and 0.02 about 7, before costs and ignoring cross-correlation. Tiny daily ICs therefore produce spectacular gross Sharpe ratios, and costs decide the outcome.
- **Plausible targets for stockai's held-out year:**
  - daily RankIC of 0.005-0.02 on liquid US names;
  - directional accuracy of 50.5-52%.
  - A daily RankIC above about 0.05 on large caps should be treated as a leakage bug until proven otherwise. AlphaAgent's S&P 500 baselines sit at 0.001-0.003; China's CSI 300 at 0.03-0.06.
- **Cost model for a $10,000 account (my synthesis):**
  - S&P 500-type names: one-way cost of about 1-5 bps (half-spread). Price improvement at good brokers can cut this, but Schwarz et al. show wide dispersion across brokers.
  - Mid and small caps: 10-25 bps.
  - Microcaps and sub-$5 stocks: 50-120+ bps.
  - Per-stock estimates for the Stooq universe can come from Abdi-Ranaldo or EDGE on daily OHLC (good for small/mid caps). For large caps, use a floor of a few bps instead, because daily-bar estimators are noisy there.
  - At about 100% daily turnover, each 1 bp costs about 2.5-5%/yr (252 days × 1-2 units of capital traded per day).
- **Design implication.** To survive 5-20 bps, a daily signal needs either:
  - per-trade edge well above 10 bps (not seen in the literature for liquid names after 2010), or
  - turnover cut by holding 5-20 days with entry/exit hysteresis (Kronos's top-k/drop-n with a 5-day minimum; Novy-Marx-Velikov's buy/hold spread). Novy-Marx-Velikov and Azevedo et al. find mitigation lowers gross returns roughly as much as it saves in costs.

### Gaps
- No peer-reviewed, US-specific, multi-year table of daily or weekly RankIC for price/volume-only signals on point-in-time universes.
- The common practitioner rule "IC ≈ 0.05 is good" could not be verified from a primary source.
- The 2018 Frazzini-Israel-Moskowitz "Trading Costs" paper (more data, 21 markets) could not be read (SSRN blocked).
- Schwarz et al.'s liquid-vs-illiquid split was not in the slide text.
- I found no recent (2023-2026) primary table of effective spreads by US market-cap tier.

## 5. Decay of return predictability over time and after publication; survivorship/delisting bias

### Takeaway
**Decay is large and well documented.**
- Published predictors earn 26% less out of sample and 58% less after publication (US).
- After costs, decimalization and post-publication decay, the average anomaly nets about 4 bps a month.
- Short-term reversal fell from 1.38%/day (1995) to 0.13%/day (2007) gross. Value-weighted, it has been insignificant since 2003 (0.69%/yr, t 0.27, for Jul 2003-Jun 2018). Over 1963-2018 the reversal spread is only about 2 bps/month among the 500 largest stocks, against 2.04%/month in microcaps.
- Daily ML S&P 500 strategies went from over 200%/yr net (1990s) to negative after 2010.

**Survivorship and delisting bias is material for exactly the strategies a "next-price" model learns.** Missing performance-related delisting returns are large and negative; the standard imputations are −30% (NYSE/AMEX) and −55% (Nasdaq). About 3.3-3.5% of US listed firms delist for cause each year. Delisting returns account for about 40% of momentum profits, and they inflate value and loser-buying strategies.

### Cited Findings
**Post-publication and out-of-sample decay**
- McLean & Pontiff (JF 2016): 97 predictors from 79 papers, sample ending 2013, mostly EW long-short before costs.
  - Average return is 0.582%/month in-sample, 0.402% out of sample before publication, and 0.264% after publication.
  - That is 26% lower out of sample and 58% lower after publication, attributing 32% to publication-informed trading — [PDF](https://www.gwern.net/doc/economics/2016-mclean.pdf)
- Jacobs & Müller (Jan 2017 WP, 231 anomalies, 39 markets, 1980-2015; JFE 2020): US returns fall 36% post-sample and 60-65% post-publication. None of the 38 other markets shows a reliable post-publication decline — [WP PDF](https://wp.lancs.ac.uk/fofi2018/files/2018/03/FoFI-2018-0174-Sebastian-M%C3%BCller2.pdf)
- Hou, Xue & Zhang (RFS 2020): 452 anomalies with microcaps controlled via NYSE breakpoints and VW.
  - 65% fail at |t| > 1.96, including 96% of the trading-frictions category; 82% fail the 2.78 multiple-test hurdle.
  - Microcaps are 61% of firms but 3.28% of market cap — [IDEAS](https://ideas.repec.org/a/oup/rfinst/v33y2020i5p2019-2133..html); [NBER WP](https://www.nber.org/system/files/working_papers/w23394/w23394.pdf)
- Chen & Velikov (JFQA 2023): 204 anomalies; the average anomaly nets 4 bps/month after costs and post-publication effects. FEDS 2020 version: 8 bps/month net post-publication and post-2005, and 4 bps VW — [JFQA](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/zeroing-in-on-the-expected-returns-of-anomalies/945133D5A3ECEEAF466AEE91551FD225); [FEDS](https://www.federalreserve.gov/econres/feds/files/2020039pap.pdf)
- Arnott, Harvey, Kalesnik & Linnainmaa ("Alice's Adventures in Factorland", JPM 2019; VW, scaled to 10% volatility, gross):
  - Short-term reversal: 5.34%/yr (t 3.96) over 1963-2018, but 0.69%/yr (t 0.27; CAPM alpha −2.12%) over Jul 2003-Jun 2018.
  - Momentum: 5.48% vs 0.77%.
  - Across 46 factors, average returns fall from 6.6% in the last 10 in-sample years to 3.2% in the first 10 out-of-sample years — [PDF](https://www.econ.queensu.ca/sites/econ.queensu.ca/files/Econ%20870/Arnott%2C%20et%20al%20Alices%20Adventures%20in%20Factorland%2C%202019.pdf)
- Chordia, Subrahmanyam & Tong (JAE 2014): a portfolio of prominent anomalies saw returns "approximately halved after decimalization" — [IDEAS](https://ideas.repec.org/a/eee/jaecon/v58y2014i1p41-58.html)

**Short-term reversal specifically**
- Khandani & Lo (Nov 2007): the Lo-MacKinlay daily contrarian strategy on CRSP stocks priced $5-$2,000, gross and frictionless.
  - Average daily return: 1.38% (1995), 0.44% (2000), 0.15% (2006), 0.13% (2007 through Aug).
  - Largest size decile: 0.04% (1995), 0.00% (2006), −0.04% (2007). Smallest decile: 3.57% in 1995.
  - The strategy supplies liquidity: it earns the spread rather than paying it — [NY Fed PDF](https://www.newyorkfed.org/medialibrary/media/research/conference/2007/liquidity/Khandani_Lo.pdf)
- Medhat & Schmeling (RFS 2022; 1963-2018, VW, gross unless stated):
  - Reversal is −1.41%/month in the lowest-turnover decile; the highest-turnover decile shows short-term momentum of +1.37%/month.
  - Reversal is −2.04%/month among microcaps but only −2 bps/month among the largest 500 stocks.
  - Net of costs, reversal's 1.94%/month cost subsumes it, while short-term momentum nets 1.00% — [PDF](https://openaccess.city.ac.uk/id/eprint/31278/1/MS_short_term_mom_v27.pdf)
- Conrad, Gultekin & Kaul (JBES 1997): contrarian profits "are largely generated by the bid-ask bounce". Using bid prices "eliminates all profits from price reversals for NASDAQ-NMS stocks and most of the profits for NYSE/AMEX stocks", and the rest "disappear at trivial levels of transaction costs" — [IDEAS](https://ideas.repec.org/a/bes/jnlbes/v15y1997i3p379-86.html)

**Momentum**
- Daniel & Moskowitz (NBER 2014; US 1927-2013):
  - Winners-minus-losers earns 17.9%/yr with 30.0% volatility and monthly skew −4.70. Worst months: Aug 1932 −74.36%, Jan 2001 −49.19%, Apr 2009 −45.52%.
  - In Mar-May 2009 losers rose 163%, against 8% for winners — [NBER PDF](https://www.nber.org/system/files/working_papers/w20439/w20439.pdf)

**ML-specific decay**
- Krauss-Do-Huck (S&P 500): ensemble net returns were over 200%/yr in 1992-2001, 22%/yr in 2001-2008, and −14% to −25%/yr for all models in 2010-2015. The authors cite "increasing popularization of machine learning and rising computing power" — [EconStor](https://www.econstor.eu/bitstream/10419/130166/1/856307327.pdf)
- Fischer-Krauss: "moderation" in 2001-2009 and "deterioration" in 2010-2015 — [EconStor](https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf)
- Rahimikia et al.: the cross-model Sharpe "declines materially over time" across 2001-2023 — [FoFI 2026 PDF](http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-020-Eghbal-Rahimikia.pdf)

**Backtest-to-live degradation (practitioner)**
- Suhonen, Lennkh & Perez (JPM 2017; secondary): about 215 bank alternative-beta indices from 17 banks. The median Sharpe falls 73% from backtest to live (about 1.20 to 0.31; live period roughly 2005-2015).
  - Only 18 of 215 had a live Sharpe at or above their backtest; 30% had a negative live Sharpe.
  - The most complex strategies lost over 30 pp more — [Hedge Fund Alpha](https://hedgefundalpha.com/strategies/alternative-beta-strategies-complexity-magical-backtests/); [CXO Advisory](https://www.cxoadvisory.com/big-ideas/live-performance-of-alternative-beta-products/)

**Survivorship and delisting bias**
- Shumway (JF 1997): correct delisting returns are missing for "most" negative-reason delistings since 1962, and the omitted returns are "large" — [IDEAS](https://ideas.repec.org/a/bla/jfinan/v52y1997i1p327-40.html)
  - The standard imputations are −30% for NYSE/AMEX and −55% for Nasdaq (secondary, SAFE data guide) — [SAFE guide](https://datacenter.safe-frankfurt.de/documents/Data_Description_CRSP_Shumway_20250317.pdf)
- Shumway & Warther (JF 1999): missing Nasdaq delisting returns are "large and negative". After correction there is "no evidence that there ever was a size effect on Nasdaq" — [IDEAS](https://ideas.repec.org/a/bla/jfinan/v54y1999i6p2361-2379.html)
- Delisting frequency (Doidge, Karolyi & Stulz, NBER WP 21181): US delists were 7.29% of prior-year listed firms per year in 1975-1996 and 9.49% in 1997-2012 (408 vs 520 a year). Delists for cause were 3.25% vs 3.50%; mergers 3.92% vs 5.64% — [NBER PDF](https://www.nber.org/system/files/working_papers/w21181/w21181.pdf)
- Eisdorfer (2008): about 40% of momentum profit comes from delisting returns, mostly bankruptcies — [IDEAS](https://ideas.repec.org/a/eee/finmar/v11y2008i2p160-179.html)
- Beaver, McNichols & Price (2007), via Alpha Architect (secondary): deep value returns 30.1%/yr without delisting returns but 24.6% with them; the most expensive firms go from 10.2% to 2.4% — [Alpha Architect](https://alphaarchitect.com/2016/08/dealing-with-delistings-a-critical-aspect-for-stock-selection-research/)
- Ranse (arXiv 2026; India NIFTY Smallcap 250, 2016-2025, EW): survivor-only backtests overstate annual returns by 4.94 pp and Sharpe ratios by 0.097 — [arXiv 2603.19380](https://arxiv.org/abs/2603.19380)
- Fischer-Krauss and Krauss-Do-Huck explicitly rebuild survivor-free S&P 500 membership from month-end constituent lists — [EconStor](https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf)

### Inferences
- **Stooq survivorship flows directly into the held-out-year test.** The test runs 2024-06-28 to 2025-06-27. Any stock that delisted during that year (or before) is missing from both the training data and the test-year universe.
  - With about 3.3-3.5%/yr of US firms delisting for cause at about −30% to −55% missing returns, an equal-weighted all-stock universe is overstated by very roughly 1-2 pp a year (my arithmetic, 3.5% × 30-55%).
  - Portfolios tilted toward recent losers, small or illiquid names or distressed stocks are overstated by much more. These are the long legs of reversal-type strategies, which is what daily return models learn.
  - Mergers (about 4-6%/yr) are also missing. Their usually positive terminal returns bias other strategies the other way.
- **The random-account comparison helps but is not enough.** Random accounts drawn from the same pool share the pool's survivorship bias. A strategy that systematically buys losers gets *extra* bias that random picks do not.
  - Mitigation: restrict the test universe to large, liquid names at the start of the test (low delisting hazard).
  - Report a sensitivity in which a plausible share of holdings suffers −30% to −55% terminal returns.
- **Expect decay inside the 20-year training window.** Short-horizon reversal, the main gross-alpha source for price-only models, was large before about 2003 and is about zero for large caps since then. A model trained on 2005-2024 and tested on 2024-2025 will partly learn effects that no longer exist. Weight recent years more, or test sub-period stability before trusting a single held-out year.

### Gaps
- No US study directly compares survivor-only and full-CRSP results for daily ML strategies.
- I did not read Shumway's own frequency and magnitude figures; the −30% and −55% are secondary.
- I lack Chordia et al. figures beyond "approximately halved".
- Momentum and reversal results after 2018 are not covered by these sources.
- The full Beaver-McNichols-Price paper was not read.

## 6. Methodological pitfalls in ML/deep-learning stock-prediction papers, with evidence of how common each is

### Takeaway
The pitfalls are pervasive and, where they have been counted, common. Measured prevalence:
- 73.4% of 612 deep-learning asset-management papers (2018-2023) forecast prices, and only 4.0% of the price-forecast papers compare with a naive random-walk forecast.
- Only 13% of 49 recent stock-prediction studies (2020-2024) report both statistical and financial metrics.
- About 20% share code.
- Leakage has been documented in 294 papers across 17 fields, rising to 648 papers in 30 fields; I found no finance-specific audit.

Effects where they have been measured:
- Backtest Sharpe explains about 2% of the variance in out-of-sample Sharpe across 888 live algorithms.
- 65% of published anomalies fail replication once microcaps are controlled.
- LLMs memorize historical index levels (S&P 500 MAPE 0.61% before the knowledge cutoff vs 16.87% after).

### Cited Findings
**(1) Price-level targets ("tomorrow equals today") and missing naive baselines**
- Elliot & Hsu (arXiv 1710.05751, 2017): on daily S&P 500 index levels, the best LSTM's MAE was 21.43 against 10.54 for the last-value forecast ("the martingale performed twice as well"). On a single stock, the LSTM "ends up just predicting whatever the previous value was" — [arXiv](https://arxiv.org/pdf/1710.05751)
- Beck, Dovern & Vogl (Applied Intelligence 55, 2025): across ARIMA, ETS, BVAR, LSTM, HCNN, DeepVAR, TFT and XGBoost, "no method consistently outperforms the naive (no-change) forecast" on stock-price and FX data. ML studies "mostly compare the performance solely relative to other complex models" — [FAU CRIS](https://cris.fau.de/publications/333940040)
- Prevalence, Reis, Serra & Gama (arXiv 2503.01591, 2025-03-03):
  - 612 deep-learning asset-management articles (2018-2023). Price forecasting is 73.4% of them, and 66.6% of those predict only one step ahead.
  - A naive baseline appears in 4.0% of price-forecast articles (2.8% excluding electricity), rising to 15.6% in higher-ranked journals.
  - The authors: "the main baseline … should be a random walk approach" — [arXiv](https://arxiv.org/pdf/2503.01591v1)
- Prevalence, Jiang (arXiv 2003.01859, 2020; ESWA 2021): of 124 deep-learning stock papers, 54 are daily "regression" papers predicting "the specific value of the prices" — [arXiv](https://arxiv.org/pdf/2003.01859)
- Sezer et al. (2005-2019 review): price prediction is "the most studied financial application", "mostly considered as a regression problem" — [arXiv](https://arxiv.org/pdf/1911.13288)

**(2) Leakage: normalization, temporal, cross-validation**
- Kapoor & Narayanan (Patterns 2023): 22 papers across 17 fields report leakage "collectively affecting 294 papers". The 2024 project page lists 41 papers across 30 fields affecting 648 papers, none of them in finance or economics.
- Their taxonomy:
  - L1.1 no test set
  - L1.2 pre-processing on train and test (includes normalizing with full-sample statistics)
  - L1.3 feature selection on train and test
  - L1.4 duplicates
  - L2 illegitimate features
  - L3.1 temporal leakage
  - L3.2 train/test non-independence
  - L3.3 sampling bias
  - Source: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/); [project page](https://reproducible.cs.princeton.edu/)
- López de Prado ("The 10 Reasons Most ML Funds Fail", 2018): k-fold CV fails on serially correlated features with overlapping labels, and "leakage leads to false discoveries". The fix is purging and embargoing — [GARP PDF](https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf)
- Albelali & Ahmed (arXiv 2512.06932, Dec 2025; climate data): building sequences before splitting under 10-fold CV inflated the RMSE gain by up to 20.5%, against typically under 5% for 2- or 3-way chronological splits — [arXiv](https://arxiv.org/pdf/2512.06932)
- Arnott, Harvey & Markowitz: "out of sample is not really out of sample", because researchers have lived through the hold-out period — [SSRN preprint copy](https://www.smallake.kr/wp-content/uploads/2023/04/SSRN-id3275654.pdf)
- Benchmark choice for R²: GKX note a roughly 3% inflation in OOS R² when the benchmark is the historical mean rather than zero — [NBER PDF](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf)

**(3) Survivorship bias**
- FINSABER (arXiv 2505.07078 v6, KDD '26): "A common cause is using today's S&P 500 constituents as the historical investment universe."
  - Its review of 9 LLM-investing methods finds evaluations of 3 months to 2 years on 3-100 symbols, with only 3 of 9 releasing code.
  - Over 2004-2024, "traditional strategies like Buy and Hold consistently rank among the top performers" — [arXiv](https://arxiv.org/html/2505.07078v6)
- Survivorship is not mentioned in five deep-learning reviews (Jiang 2020, Sezer 2019, Olorunnimbe & Viktor 2022, Akşehir & Kılıç 2024, Reis 2025). That is an absence, not a measured rate. Magnitudes are in Section 5.

**(4) Data snooping, multiple testing and backtest overfitting**
- Harvey, Liu & Zhu (RFS 2016): 316 factors; a new factor needs t > 3.0; "most claimed research findings in financial economics are likely false" — [PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF)
- Bailey, Borwein, López de Prado & Zhu (Notices AMS 2014):
  - "After trying only seven independent strategy configurations, the expected maximum SR IS is 1 for a two-year long backtest, while the expected SR OOS is 0."
  - With 5 years of data, "no more than forty-five independent model configurations should be tried" — [AMS](https://www.ams.org/notices/201405/rnoti-p458.pdf)
- Probability of Backtest Overfitting (Bailey et al.): on a random walk, an optimized strategy with in-sample Sharpe 1.27 still has about 53% negative out-of-sample Sharpe, a PBO of 55% — [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- Deflated Sharpe ratio — [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- Harvey & Liu "Backtesting" (JPM 2015): the multiple-testing haircut is "almost always more than and sometimes much larger than 50%" when Sharpe is below 0.4 — [PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF)
- Wiecki, Campbell, Lent & Stauth (Quantopian, 2016): across 888 algorithms with at least 6 months out of sample, backtest Sharpe barely predicts live Sharpe (R² < 0.025; Pearson R² = 0.02). More backtest days goes with a larger shortfall. In-sample was 2010 to mid-2015, a bull market; out-of-sample was mid-2015 to Feb 2016 — [PDF](https://c.mql5.com/forextsd/forum/208/All%20that%20Glitters%20Is%20Not%20Gold%20-%20Comparing%20Backtest%20and%20Out-of-Sample%20Performance%20on%20a%20Large%20Cohort%20o.pdf)
- Hou, Xue & Zhang: "widespread p-hacking"; 64% of 447 anomalies are insignificant, and 85% at t ≥ 3 (NBER 2017 version) — [NBER](https://www.nber.org/papers/w23394)
- Bajgrowicz & Scaillet (JFE 2012), on technical trading rules over DJIA daily data 1897-2011 with false-discovery-rate control (abstract via search):
  - "an investor would never have been able to select ex ante the future best-performing rules";
  - "even in-sample, the performance is completely offset by the introduction of low transaction costs" — [IDEAS](https://ideas.repec.org:443/a/eee/jfinec/v106y2012i3p473-491.html)
- Nagel (2025) shows that an apparent ML success, the KMZ "virtue of complexity", can be a mechanical artifact: volatility-timed momentum (Section 3) — [BFI PDF](https://bfi.uchicago.edu/wp-content/uploads/2025/08/BFI_WP_2025-104.pdf)
- Arnott-Harvey-Markowitz's backtesting protocol covers:
  - multiple-testing logs ("keep track of what is tried");
  - choosing winsorization and other data choices in advance;
  - cross-validation;
  - model dynamics, complexity and research culture.
  - Source: [preprint copy](https://www.smallake.kr/wp-content/uploads/2023/04/SSRN-id3275654.pdf)

**(5) Ignoring transaction costs and economic evaluation**
- Akşehir & Kılıç (PeerJ CS 2024): of 49 studies (2020-2024), only 13% use both statistical and financial metrics — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11623133/)
- Reis et al. 2025: only 22 of 48 algorithmic-trading articles compare with a passive portfolio such as buy-and-hold. Buy-and-hold baselines appear in 36.7% of portfolio-management articles — [arXiv](https://arxiv.org/pdf/2503.01591v1)
- Jiang 2020: "The transaction cost is often omitted or simplified" — [arXiv](https://arxiv.org/pdf/2003.01859)
- Quantified consequences are in Sections 1-4: Rahimikia (Sharpe +6.4 at 0 bps → −3.6 at 20 bps), Jensen et al. (net negative), Fischer-Krauss and Krauss-Do-Huck (negative after 2010), Novy-Marx-Velikov (reversal −1.28%/month net).

**(6) Look-ahead and pretraining contamination (LLMs and TSFMs)**
- Lopez-Lira, Tang & Zhu (arXiv 2504.14765; v1 2025-04-20, v2 2025-12-15), GPT-4o with a knowledge cutoff of 2023-10:

  | Measure | Before cutoff | After cutoff |
  |---|---|---|
  | S&P 500 daily level, MAPE | 0.61% | 16.87% |
  | Daily direction accuracy | 80.58% | 45.70% |

  - Instructing the model to respect a fake cutoff fails.
  - It identifies the company in 100% of anonymized Apple, Meta and Microsoft earnings calls — [arXiv](https://arxiv.org/html/2504.14765v2)
- Glasserman & Lin (arXiv 2309.17322, 2023): ChatGPT's training through Sep 2021 leaves only Oct 2021 onward for clean tests. Anonymized headlines did *better* in-sample, which they call a "distraction effect" — [arXiv](https://arxiv.org/pdf/2309.17322)
- Sarkar & Vafa (2024) find look-ahead bias in earnings-call risk-factor and election tasks; prompting fixes are limited — [NBER conf. PDF](https://conference.nber.org/conf_papers/f205530/f205530.pdf)
- He, Lv, Manela & Wu (arXiv 2502.21206, 2025): point-in-time ChronoBERT/ChronoGPT reach Sharpe 4.80/4.92 (EW, 2008-2023), comparable to Llama-3.1-8B. They conclude look-ahead bias is "modest" in their application — [arXiv](https://arxiv.org/pdf/2502.21206)
- TSFMs:
  - Kronos's pretraining runs through June 2024, so earlier backtests are in-sample — [arXiv](https://arxiv.org/pdf/2508.02739)
  - Generic TSFM corpora hold 0.0001-1.6% finance data (Section 1).
  - Meyer et al. (arXiv 2510.13654) document two leakage channels in TSFM benchmarks: reused datasets and temporal overlap of correlated series — [arXiv](https://arxiv.org/pdf/2510.13654)
  - Chen et al. (2026) find post-origin training generally *lowers* accuracy for financial TSFMs, a counterpoint for numeric return series — [arXiv](https://arxiv.org/pdf/2609.20554)

**(7) Other common pitfalls**
- Equal-weighted, microcap-dominated portfolios:
  - GKX EW Sharpe 2.45 vs VW 1.35; StockGPT 6.5 vs 1.0; JKX weekly 7.15 vs 1.49 (Section 3).
  - Microcaps are 61% of firms but 3.28% of market cap (Hou-Xue-Zhang) — [NBER](https://www.nber.org/system/files/working_papers/w23394/w23394.pdf)
- Trading at the close used to build the signal: StockGPT's EW return drops from 119% to 26% with a one-day delay — [arXiv](https://arxiv.org/html/2404.05101v3). Bid-ask bounce creates spurious reversal — [Conrad-Gultekin-Kaul](https://ideas.repec.org/a/bes/jnlbes/v15y1997i3p379-86.html)
- Single-regime or short test windows:
  - Wiecki's bull-market in-sample period.
  - FINSABER's 3-month to 2-year LLM evaluations.
  - Kronos's roughly one-year test.
  - Noguer i Alonso's 5 stocks over about 18 months.
- Code and reproducibility:
  - Only 7 of 35 reviewed DL papers (20%) provide source code — [Olorunnimbe & Viktor, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9245389/)
  - A FINSABER re-run of FinMem (Oct 2022-Apr 2023) gave Sharpe 0.927 / 0.404 (GPT-4o-mini / GPT-4o) on TSLA vs 2.679 reported, and MSFT −0.554 / 0.792 vs 1.440. Part of the gap may come from the different backbone model — [arXiv](https://arxiv.org/html/2505.07078v6)
- Metric inconsistencies:
  - A 2025 transformer paper's ICIR does not match its own IC table (Section 4).
  - A 2025 single-author paper reports OOS R² of 11-12% for US large caps, an order of magnitude above GKX's 0.4-0.7%/month, without a comparable definition on its abstract page — [arXiv 2505.01575](https://arxiv.org/abs/2505.01575)
  - KMZ show that R² and Sharpe can disagree in sign — [NBER](https://nber.org/system/files/working_papers/w30217/w30217.pdf)
- Data quality: raw OHLCV "frequently contaminated by artifacts stemming from low liquidity, price limits, or data feed errors" (the Kronos authors filter jumps, illiquid stretches and stale prices) — [arXiv](https://arxiv.org/pdf/2508.02739v1)

### Inferences
**Pitfall checklist for stockai**, with evidence of prevalence and the safeguard. This is my synthesis; each row maps to evidence above.

| # | Pitfall | How common / how big | Safeguard for stockai |
|---|---|---|---|
| 1 | Predicting price levels and scoring RMSE/R² on levels | 73% of DL asset papers forecast prices; naive baseline in 4% (Reis 2025); LSTM 2× worse than last value (Elliot-Hsu) | Predict next-day *return* or its cross-sectional rank. Score with RankIC, accuracy and portfolio P&L. Always report a zero-return / last-price baseline and an AR(1)/reversal baseline. |
| 2 | Normalization and feature leakage | 294 → 648 papers across fields (Kapoor-Narayanan); no finance audit | Fit scalers, token bins and vocabularies on training data only. Compute features strictly from data up to t. Unit-test that shifting the future does not change features. |
| 3 | Random or k-fold splits; overlapping labels | CV inflates gains up to 20.5% vs <5% (Albelali-Ahmed); López de Prado #8 | Walk-forward only, with purge and embargo around boundaries. One locked final year (2024-06-28 to 2025-06-27), touched once. |
| 4 | Survivorship and delisting | Common (today's S&P 500 lists, FINSABER); −30% to −55% missing returns; about 40% of momentum profits (Eisdorfer) | Stooq is survivor-only. Restrict to liquid large caps at each test start, report sensitivity to imputed delisting losses, and distrust loser-buying gains. |
| 5 | Data snooping, many architectures and hyperparameters | 7 tries → spurious in-sample Sharpe of 1 over 2 years (Bailey et al.); backtest explains 2% of live Sharpe variance (Wiecki); 73% Sharpe decay live (Suhonen) | Log every trial. Report deflated Sharpe / PBO. Hold the random-account comparison fixed (correlation2). Require t > 3 (HLZ). |
| 6 | Ignoring costs and slippage | Only 13% of studies use financial plus statistical metrics (Akşehir-Kılıç); costs flip Sharpe +6 to −4 (Rahimikia) | Per-stock spread estimates (Abdi-Ranaldo/EDGE on Stooq OHLC) with a floor of a few bps for megacaps; report P&L at 0/5/10/20 bps; include borrow cost if shorting. |
| 7 | Same-close execution and bid-ask bounce | StockGPT −78% with a one-day delay; bounce explains most reversal profits (Conrad-Gultekin-Kaul) | Signal at close t, execute at open or close of t+1. Report a skip-day variant. |
| 8 | EW, microcap-dominated portfolios | EW/VW Sharpe 2-6× (GKX, JKX, StockGPT) | Report VW and size-bucketed results. Apply price ≥ $5 and size/ADV filters. |
| 9 | Pretraining contamination (pretrained weights or LLMs) | GPT-4o S&P 500 MAPE 0.61% vs 16.87% across its cutoff (Lopez-Lira et al.); Kronos trained to Jun 2024 | Train from scratch with time-split corpora. Never evaluate pretrained weights on periods before their cutoff. |
| 10 | Short or single-regime tests | Common in LLM and TSFM papers (FINSABER; Kronos about 1 year) | A one-year Sharpe has a standard error of about 1 (my calculation: SE ≈ √((1+SR²/2)/years)). Add multi-year walk-forward and sub-period results; a single good year is weak evidence. |
| 11 | Bad ticks, splits and stale prices in free data | Kronos filters for these; they create fake reversals | Clean jumps and suspend illiquid days. Check that results survive dropping the top 1% of absolute returns. |
| 12 | Irreproducible claims | 20% share code (Olorunnimbe-Viktor); FinMem Sharpe 2.68 → 0.40-0.93 on re-run | Keep code and data snapshots; rerun from scratch before believing a result. |

- **What the literature predicts for the correlation2-style test.** A daily next-return GPT on Stooq US data is likely to show:
  - a held-out-year RankIC of about 0.00-0.02;
  - accuracy of about 50-52%;
  - gross long-short profits concentrated in small or illiquid names and recent losers;
  - net returns at 5-20 bps per trade that are near zero or negative, and statistically indistinguishable from random accounts with the same turnover.
- **The most plausible route to a tradeable signal** is a ranking model restricted to liquid large caps, with multi-day holds and turnover controls, judged against SPY/QQQ and random-turnover accounts net of realistic costs. Even then, expected net Sharpe is modest (GPZ about 1 on the largest 550 stocks, 2002-2016).

### Gaps
- No finance-specific audit counts full-sample normalization, random splits or survivorship in DL stock papers.
- No systematic reproduction-rate study of DL stock papers exists.
- Henrique et al. (2019) and Kumbure et al. (2022) review counts were not accessible (ScienceDirect blocked).
- A "1 of 22 papers adjusts for fees" figure from a 2025 review was seen only in a search summary and is not used.
- I found no quantitative prevalence for: trading at the signal close, ignoring short-borrow costs, or EW-vs-VW reporting in DL papers.
- Sullivan, Timmermann & White (JF 1999) on 7,846 technical rules: secondary summaries conflict on whether any rule stayed significant, so it is not cited above.
