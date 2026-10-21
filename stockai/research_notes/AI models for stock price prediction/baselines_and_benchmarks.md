# Baselines and Benchmarks: the bar a new daily/weekly stock-return model must beat (price/volume only)

Research date: 2026-10-18. Every number below carries its market and period. "Snippet" means the figure came from a search-engine summary of a page that could not be fetched directly (Kaggle pages return only a title to the fetch tool), so treat it as unverified. Figures marked "computed by this researcher" are my own calculations from public data files, not published statistics.

## 1. Microsoft Qlib benchmarks (Alpha158 / Alpha360): the bar, and how much deep models beat gradient-boosted trees

### Takeaway
Qlib's only published benchmark is China A-shares (CSI300, test 2017-01-01 to 2020-08-01, label = next-day close-to-close return). On it the bar is IC about 0.045-0.052, Rank IC about 0.05-0.067, daily ICIR about 0.35-0.42, and roughly 9-12% annualized excess return over CSI300 after costs with an information ratio of about 1.0-1.37. Examples: LightGBM on Alpha158 IC 0.0448 / IR 1.02; DoubleEnsemble on Alpha158 IC 0.0521 / IR 1.34; HIST on Alpha360 IC 0.0522 / Rank IC 0.0667 / IR 1.37. Trees win on hand-built features (Alpha158). Recurrent and graph models win by a modest margin on raw price/volume windows (Alpha360), which is the case closest to this project. There is no US-market table.

### Cited Findings
**Setup (what the numbers mean)**
- Results are for "China's A shared-stock & CSI300 data", and each cell is the mean ± std over 20 runs with different random seeds. The maintainers add that "We have very limited resources to implement and finetune the models… some models may have greater potential than what it looks like in the table". Also: "The backtest start from 0.8.0 is quite different from previous version." — [Qlib benchmarks README](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
- Benchmark config: market csi300, benchmark SH000300. Train 2008-01-01 to 2014-12-31, valid 2015-01-01 to 2016-12-31, test/backtest 2017-01-01 to 2020-08-01. Strategy `TopkDropoutStrategy` (topk 50, n_drop 5). Costs: open_cost 0.0005, close_cost 0.0015, min_cost 5, limit_threshold 0.095, deal_price close, account 100,000,000. — [LightGBM Alpha158 workflow YAML](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)
- Label for both Alpha158 and Alpha360 is `Ref($close, -2)/Ref($close, -1) - 1`, the 1-day return from the t+1 close to the t+2 close. — [qlib/contrib/data/handler.py](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py)
- Both feature sets use only `$open, $high, $low, $close, $volume, $vwap`, so they can be rebuilt from Stooq OHLCV (VWAP must be approximated). Alpha158 windows are 5/10/20/30/60 days. — [qlib/contrib/data/loader.py](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)
- The README describes the two sets this way: "Alpha158 is a tabular dataset… Each feature are carefully designed by human (a.k.a feature engineering)" and "Alpha360 contains raw price and volue data without much feature engineering." — [Qlib benchmarks README](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
- The "Annualized Return / Information Ratio / Max Drawdown" columns come from `1day.excess_return_with_cost`. They are excess return over the CSI300 benchmark, after costs. — [examples/run_all_model.py](https://github.com/microsoft/qlib/blob/main/examples/run_all_model.py)
- Metric definitions in the README: IC^(t) = corr(ŷ^(t), ret^(t)); Rank IC = correlation of ranks; ICIR = mean(IC)/std(IC), computed on daily IC and not annualized. — [Qlib benchmarks README](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
- Dating: the README was last changed 2023-08-01. The last results added were KRNN and Sandwich (Alpha360) on 2023-05-26. — [README commit history](https://github.com/microsoft/qlib/commits/main/examples/benchmarks/README.md). The newest Qlib release, v0.9.8 (2026-10-08), did not change the tables. — [Qlib releases](https://github.com/microsoft/qlib/releases)
- The benchmarks folder holds 25 model directories: ADARNN, ADD, ALSTM, CatBoost, DoubleEnsemble, GATs, GRU, GeneralPtNN, HIST, IGMTF, KRNN, LSTM, LightGBM, Linear, Localformer, MLP, SFM, Sandwich, TCN, TCTS, TFT, TRA, TabNet, Transformer, XGBoost. None of them have US results. — [benchmarks folder](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)

**CSI300, Alpha158 (all values from the [Qlib benchmarks README](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md); AR = annualized excess return with cost)**

| Model | IC | ICIR | Rank IC | Rank ICIR | AR | IR | MDD |
|---|---|---|---|---|---|---|---|
| DoubleEnsemble (LGBM base) | 0.0521 | 0.4223 | 0.0502 | 0.4117 | 0.1158 | 1.3432 | -0.0920 |
| LightGBM | 0.0448 | 0.3660 | 0.0469 | 0.3877 | 0.0901 | 1.0164 | -0.1038 |
| XGBoost | 0.0498 | 0.3779 | 0.0505 | 0.4131 | 0.0780 | 0.9070 | -0.1168 |
| CatBoost | 0.0481 | 0.3366 | 0.0454 | 0.3311 | 0.0765 | 0.8032 | -0.1092 |
| Linear | 0.0397 | 0.3000 | 0.0472 | 0.3531 | 0.0692 | 0.9209 | -0.1509 |
| MLP | 0.0376 | 0.2846 | 0.0429 | 0.3220 | 0.0895 | 1.1408 | -0.1103 |
| TRA | 0.0440 | 0.3535 | 0.0540 | 0.4451 | 0.0718 | 1.0835 | -0.0760 |
| TRA (20 selected features) | 0.0404 | 0.3197 | 0.0490 | 0.4047 | 0.0649 | 1.0091 | -0.0860 |
| GATs (20 feat.) | 0.0349 | 0.2511 | 0.0462 | 0.3564 | 0.0497 | 0.7338 | -0.0777 |
| ALSTM (20 feat.) | 0.0362 | 0.2789 | 0.0463 | 0.3661 | 0.0470 | 0.6992 | -0.1072 |
| LSTM (20 feat.) | 0.0318 | 0.2367 | 0.0435 | 0.3389 | 0.0381 | 0.5561 | -0.1207 |
| GRU (20 feat.) | 0.0315 | 0.2450 | 0.0428 | 0.3440 | 0.0344 | 0.5160 | -0.1017 |
| Localformer | 0.0356 | 0.2756 | 0.0468 | 0.3784 | 0.0438 | 0.6600 | -0.0952 |
| SFM | 0.0379 | 0.2959 | 0.0464 | 0.3825 | 0.0465 | 0.5672 | -0.1282 |
| Transformer | 0.0264 | 0.2053 | 0.0407 | 0.3273 | 0.0273 | 0.3970 | -0.1101 |
| TCN | 0.0279 | 0.2181 | 0.0421 | 0.3429 | 0.0262 | 0.4133 | -0.1090 |
| TFT (20 feat.) | 0.0358 | 0.2160 | 0.0116 | 0.0720 | 0.0847 | 0.8131 | -0.1824 |
| TabNet | 0.0204 | 0.1554 | 0.0333 | 0.2552 | 0.0227 | 0.3676 | -0.1089 |

The "20 selected features" are picked by LightGBM feature importance. — [Qlib benchmarks README](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)

**CSI300, Alpha360 (raw 60-day OHLCV/VWAP windows; same source)**

| Model | IC | ICIR | Rank IC | Rank ICIR | AR | IR | MDD |
|---|---|---|---|---|---|---|---|
| HIST | 0.0522 | 0.3530 | 0.0667 | 0.4576 | 0.0987 | 1.3726 (±0.27) | -0.0681 |
| IGMTF | 0.0480 | 0.3589 | 0.0606 | 0.4773 | 0.0946 | 1.3509 | -0.0716 |
| TRA | 0.0485 | 0.3787 | 0.0587 | 0.4756 | 0.0920 | 1.2789 (±0.42) | -0.0834 |
| TCTS (GRU base) | 0.0508 | 0.3931 | 0.0599 | 0.4756 | 0.0893 | 1.2256 | -0.0857 |
| GATs | 0.0476 | 0.3508 | 0.0598 | 0.4604 | 0.0824 | 1.1079 | -0.0894 |
| AdaRNN | 0.0464 | 0.3619 | 0.0539 | 0.4287 | 0.0753 | 1.0200 | -0.0936 |
| GRU | 0.0493 | 0.3772 | 0.0584 | 0.4638 | 0.0720 | 0.9730 (±0.33) | -0.0821 |
| ADD | 0.0430 | 0.3188 | 0.0559 | 0.4301 | 0.0667 | 0.8992 | -0.0855 |
| LSTM | 0.0448 | 0.3474 | 0.0549 | 0.4366 | 0.0647 | 0.8963 | -0.0875 |
| ALSTM | 0.0497 | 0.3829 | 0.0599 | 0.4736 | 0.0626 | 0.8651 | -0.0994 |
| TCN | 0.0441 | 0.3301 | 0.0519 | 0.4130 | 0.0604 | 0.8295 | -0.1018 |
| LightGBM | 0.0400 | 0.3037 | 0.0499 | 0.4042 | 0.0558 | 0.7632 (±0.00) | -0.0659 |
| DoubleEnsemble | 0.0390 | 0.2946 | 0.0486 | 0.3836 | 0.0462 | 0.6151 | -0.0915 |
| XGBoost | 0.0394 | 0.2909 | 0.0448 | 0.3679 | 0.0344 | 0.4527 | -0.1004 |
| CatBoost | 0.0378 | 0.2714 | 0.0467 | 0.3659 | 0.0292 | 0.3781 | -0.0862 |
| Localformer | 0.0404 | 0.2932 | 0.0542 | 0.4110 | 0.0246 | 0.3211 | -0.1095 |
| MLP | 0.0273 | 0.1870 | 0.0396 | 0.2910 | 0.0029 | 0.0274 | -0.1385 |
| Sandwich | 0.0258 | 0.1924 | 0.0337 | 0.2624 | 0.0005 | 0.0001 | -0.1752 |
| Transformer | 0.0114 | 0.0716 | 0.0327 | 0.2248 | -0.0270 | -0.3378 | -0.1653 |
| TabNet | 0.0099 | 0.0593 | 0.0290 | 0.1887 | -0.0369 | -0.3892 | -0.2145 |
| KRNN | 0.0173 | 0.1210 | 0.0270 | 0.2018 | -0.0465 | -0.5415 | -0.2919 |

**CSI500 (partial table, same split)**: LightGBM Alpha158 IC 0.0399, ICIR 0.4065, Rank IC 0.0482, AR 0.1284, IR 1.5650. LightGBM Alpha360 IC 0.0400, Rank IC 0.0536, AR 0.0505, IR 0.7658. MLP Alpha158 IC 0.0229, AR 0.0043. The DoubleEnsemble CSI500 rows show exactly the same AR/IR/MDD as Linear (0.0382 / 0.1723 / -0.4876), which looks like a copy error in the README. — [Qlib benchmarks README](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)

**Third-party re-run in the same Qlib setting (NeurIPS 2025)**
- R&D-Agent(Q) (arXiv 2505.15155; v1 2025-05-21, v2 2025-09-25) uses CSI300 with train 2008-2014, valid 2015-2016, test 2017-01-01 to 2020-08-01. Its label is the 1-day forward return, cross-sectionally z-scored, with a daily long-short ranking strategy with costs. Its Table 1 reports: LightGBM IC 0.0277, ARR 0.0397, IR 0.5664; XGBoost IC 0.0291; CatBoost IC 0.0279, IR 0.7008; DoubleEnsemble IC 0.0294, IR 0.7968; Transformer IC 0.0317; PatchTST IC 0.0247, ARR 0.0571, IR 0.7191; iTransformer IC 0.0270, ARR 0.0979, IR 1.2337; Mamba IC 0.0281; TRA IC 0.0404, IR 1.0091; MASTER IC 0.0215, Rank IC 0.0296, ARR 0.0896, IR 1.3406. Factor libraries with LightGBM: Alpha158 IC 0.0341 / ARR 0.0570 / IR 0.8459; Alpha360 IC 0.0420 / ARR 0.0438; Alpha101 IC 0.0308. The best LLM-agent configuration (o3-mini) reached IC 0.0532, ARR 0.1421, IR 1.7382. — [arXiv 2505.15155 (HTML v2)](https://arxiv.org/html/2505.15155v2); [abstract](https://arxiv.org/abs/2505.15155)
- Caveat: that table's GRU, LSTM, ALSTM, GATs and TRA rows are identical to the README's "Alpha158 (with selected 20 features)" rows. Its LightGBM (IC 0.0277) is far below the README's Alpha158 LightGBM (0.0448). So its baselines appear to use a reduced 20-factor input, and some rows were copied rather than re-run. — compare [arXiv 2505.15155](https://arxiv.org/html/2505.15155v2) with [Qlib README](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
- Qlib also ships "Rolling Retraining" baselines (examples/benchmarks_dynamic) for non-stationary markets, and promotes RD-Agent for LLM-driven factor and model search (released 2024-08-08). — [Qlib main README](https://github.com/microsoft/qlib/blob/main/README.md)

### Inferences
- **Deep vs trees depends on the input.** On Alpha158 the best tree models (DoubleEnsemble IC 0.0521, XGBoost 0.0498, LightGBM 0.0448) beat the best deep model (TRA 0.0440). On Alpha360 the best deep model (HIST IC 0.0522, Rank IC 0.0667, IR 1.37) beats LightGBM (0.0400 / 0.0499 / 0.76) by about +0.012 IC and +0.017 Rank IC. Yet HIST on raw windows only ties DoubleEnsemble on engineered features. The evidence says "learned sequence features ≈ human features + GBDT", not "deep >> trees".
- **Seed noise.** Deep-model IR std is ±0.25-0.47 across 20 seeds, versus ±0.00 for GBDT. A single-seed deep result within about 0.3-0.4 IR of LightGBM cannot be told apart from it.
- **Implied cost drag (my arithmetic from the YAML).** TopkDropout 50/5 replaces up to 5 of 50 names a day, about 10% one-way daily turnover. At 5 bps on buys and 15 bps on sells that is about 0.02% a day, roughly 5% a year. The roughly 9-12% annualized excess returns are therefore net of a drag of about that size.
- **Translating to stockai.** The minimum credible bar is "beat LightGBM or XGBoost trained on Alpha158-style features computed from the same Stooq OHLCV, using the same walk-forward split". The stretch bar is "match HIST/TRA-class sequence models (Rank IC about 0.06) on raw windows". Absolute IC levels from CSI300 2017-2020, a retail-heavy market with ±10% price limits, probably overstate what liquid US large caps allow. That is an inference; I found no US Qlib table to confirm it.
- The test window ends 2020-08-01. The published tables cannot show whether these rankings held in 2021-2026.

### Gaps
- No US-market (S&P 500 / Russell) Qlib-style benchmark table exists in the benchmarks folder. Qlib can collect US data via its Yahoo collector, but I found no official US results.
- I did not examine the small US academic datasets that deep stock-prediction papers commonly use (for example the NASDAQ/NYSE 2013-2017 relational stock-ranking sets and the ACL18 StockNet set). These are the usual "US versions" in that literature; their size and realism were not checked here.
- No published Qlib benchmark covers 2021-2026 test periods.
- MASTER, FactorVAE and Stockformer are not in Qlib's benchmarks folder. A MASTER port exists only in the SJTU-Quant fork (see section 3).
- I did not verify who produced the HIST and IGMTF rows (added in PR #1040, 2022-04-10). If the original authors produced them, the rows are not independent tests.

## 2. Competition evidence (Jane Street 2021 and 2024-25, JPX 2022, Ubiquant 2022, G-Research Crypto 2021-22, Optiver, Numerai)

### Takeaway
Where the winning method is known, it was one of three things:
- LightGBM alone (G-Research crypto, used by all of the top 3);
- LightGBM + TabNet (Ubiquant);
- a pure neural net on proprietary anonymized features (Jane Street 2021: supervised autoencoder + MLP; the documented Jane Street 2024 top solution: GRU + online updates).

No transformer won any competition I could verify. Feature engineering, time-ordered validation with gaps and, in Jane Street 2024, online updating mattered more than architecture. Achievable scores are tiny:
- weighted Pearson about 0.03 on 15-minute crypto returns (G-Research, scored on 3 months of live data);
- a correlation-based score of about 0.13 with 300 proprietary features on China A-shares (Ubiquant);
- sample-weighted zero-mean R² about 0.01 (Jane Street 2024);
- per-era CORR about 0.025-0.03 with per-era Sharpe 1.1-1.3 for Numerai's tutorial LightGBM-style model.

Public-to-private drops and leaderboard swings were common.

### Cited Findings
**Jane Street Market Prediction (Kaggle, launched 2020-11-24, $100K, nine months including live forecasting)**
- Jane Street announced it as a contest "based on a real problem with real financial data" with a $100K prize pool. — [Jane Street blog, 2020-11-24](https://blog.janestreet.com/announcing-our-market-prediction-kaggle-competition-index/)
- The winning team was led by Yirun Zhang, a King's College London PhD student, which "obtain[ed] the highest score within the competition" of "the nine-month competition". — [KCL news, 2021-09-23](https://www.kcl.ac.uk/news/kings-student-success-in-market-prediction-ai-competition)
- The winning approach was a supervised autoencoder plus MLP. Autoencoder features were concatenated with the raw anonymized features and fed into an MLP trained jointly. It added Gaussian noise before the encoder, swish activations, batch norm and dropout, and early stopping on the MLP's loss only. — [Numerai forum write-up of the Jane Street winner](https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338)

**Jane Street Real-Time Market Data Forecasting (Kaggle; training phase 2024-10-14 to 2025-01-13, then live forecasting; $50K first prize)**
- The target is `responder_6`. The metric is a sample-weighted zero-mean R², and submissions run through a timestep-by-timestep API with lagged responders supplied. The features are anonymized production-style data. — [Kaggle competition page](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) (snippet)
- A reimplementation repo states that top public-leaderboard scores were about 0.005-0.01. — [Wy66wy reproduction repo](https://github.com/Wy66wy/Jane-Street-Real-Time-Market-Data-Forecasting) (secondary)
- A top public solution (rank not verified; it links to Kaggle discussion 556542) uses the following:
  - Model: a GRU with a sequence of one trading day, in a 3-layer variant and a "1-layer GRU + 2 linear layers" variant, each run on 3 seeds and averaged. Ensemble public LB 0.0112 versus 0.0105 for the best single model.
  - Features: all original features plus per-time market averages and 1000-step rolling means/stds of 16 selected features (+0.002 CV).
  - Auxiliary targets: 4 responders as auxiliary targets (+0.001).
  - Online learning: one gradient update per day at inference with LR 0.0003. It added "+0.008" on CV.
  - What failed: "MLP, time-series transformers, cross-symbol attention and embeddings didn't work for me."
  - Its CV table: GRU1 with no auxiliary targets and no online learning scored 0.0112 CV average but only 0.0011 on a 200-day-gap test. With online learning it scored 0.0136 on the same gap test, and the full 3-seed ensemble averaged 0.0222 CV and 0.0162 on the gap test.
  - Hardware: "~100GB RAM, 12GB GPU RAM".
  — [solution.md](https://github.com/evgeniavolkova/kagglejanestreet/blob/master/solution.md); [repo README](https://github.com/evgeniavolkova/kagglejanestreet)
- Kaggle published "Winners Walkthroughs" videos for this competition with Evgeniia Grigoreva and with Team Patrick Yam. I could not read placements or methods from the pages. — [video 1](https://www.youtube.com/watch?v=lXYC0c7wJFU); [video 2](https://www.youtube.com/watch?v=lfzzPZZyzjE)

**JPX Tokyo Stock Exchange Prediction (Kaggle; 2022-04-05 to 2022-10-07 JST)**
- Entrants ranked about 2,000 Japanese stocks. The score was the Sharpe ratio of the daily return spread between portfolios of the top-200 and bottom-200 predicted stocks. There were 19,264 participants. — [JPX release, 2022-12-06](https://www.jpx.co.jp/english/corporate/news/news-releases/6020/20221206-01.html) (snippet; the page returned 403 to the fetch tool)
- First place went to Shoki Sakai (team "kisa031", Shizuoka University), who presented his algorithm at the 2022-11-10 award ceremony. — [Shizuoka University news](https://www.inf.shizuoka.ac.jp/english/news/detail.html?CN=154751) (snippet; the page returned 404 when fetched)
- Community LightGBM notebooks show private scores in the 0.19-0.39 range. For example, "LightGBM Ranker Introduction" scored 0.388 private. — [Kaggle notebook](https://www.kaggle.com/code/bturan19/lightgbm-ranker-introduction/data) (snippet)

**Ubiquant Market Prediction (Kaggle, 2022; China A-shares, 300 anonymized features)**
- The first-place team, "K I Y" ("Our Betting Strategy"), used LightGBM + TabNet, averaging 5 folds of each, and described the choice as deliberately simple. It added 100 features chosen by correlation with the target to the 300 base features. For the LightGBM single model this moved CV 0.141 → 0.154 and public LB 0.141 → 0.149. Training used the last 2.4M rows of train + supplemental data. Feature and parameter selection used PurgedGroupTimeSeries and TimeSeriesSplit. Their other submission (no supplemental data, plus MLP and CatBoost) "overfit more" and scored 0.115796. The leaderboard lists the gold score as 0.126987. — [1st-place write-up](https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy); [discussion 338220](https://kaggle.com/competitions/ubiquant-market-prediction/discussion/338220) (snippet)

**G-Research Crypto Forecasting (Kaggle, Nov 2021 to May 2022)**
- The task was to forecast short-term returns of 14 crypto coins, scored by weighted Pearson correlation. Each team's two chosen models ran for 3 months on live data. Results: 1st Meme Lord Capital (Jose and Eduardo, $50K), 2nd Nathaniel Maddux, 3rd GABA. G-Research reported that "all of our top three used a LightGBM model" and that "all of our top competitors believed it was their feature engineering that contributed most to their wins", with "a far greater impact on their final scores than model development". The leaderboard saw "many big jumps and precipitous drops, particularly early in the competition." — [G-Research wrap-up, 2022-05-30](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)
- The final leaderboard shows Meme Lord Capital first with a score of 0.0326. The 2nd-place post describes a single LightGBM GBDT regressor with squared loss and no ensembling beyond boosting, says "good N-fold cross validation is essential", and notes the winner finished "with a big lead". — [leaderboard](https://www.kaggle.com/competitions/g-research-crypto-forecasting/leaderboard); [2nd-place discussion](https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/323098) (both snippet)

**Optiver: Trading at the Close (Kaggle, 2023-09-20 to 2024-03-22) and Realized Volatility (2021)**
- The task was to forecast Nasdaq closing-auction price moves relative to an Optiver synthetic index, scored by MAE. — [competition page](https://www.kaggle.com/competitions/optiver-trading-at-the-close) (snippet)
- Final top-10 MAE spans 5.40 ("hyd", 1st) to 5.44 (10th), a spread of 0.04. — [clist.by standings mirror](https://clist.by/standings/optiver-trading-at-the-close-finance-tabular-meancolumnwisemae-46336195/)
- Other placings, all snippet: 9th place used XGBoost with an MAE objective and 157 features, retrained only twice in the private period ([9th-place write-up](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/adam-9th-place-solution)). 89th blended LightGBM with a Transformer-encoder NN over 8 folds ([write-up](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486038)). 50th used LightGBM + CatBoost ([write-up](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/486666)).
- Optiver Realized Volatility (2021): the 1st-place write-up is titled "Nearest Neighbors". — [discussion 274970](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970) (title only)

**Numerai tournament (obfuscated features, 20-day targets)**
- CORR is computed as follows: predictions are ranked, then gaussianized ((rank-0.5)/n → inverse normal CDF); the target is centered; both are raised to sign(x)·|x|^1.5 "to accentuate the tails"; then the Pearson correlation is taken. Only ranks matter ("your prediction's distribution does not matter"). Variants are CORR20V2 (20-day target) and CORR60. — [Numerai docs: CORR](https://docs.numer.ai/numerai-tournament/scoring/correlation-corr); [definitions](https://docs.numer.ai/numerai-tournament/scoring/definitions)
- The "hello_numerai" tutorial model's validation results were CORR mean 0.0245, std 0.021, Sharpe 1.1379, max drawdown 0.0637 on v4.3 data, and 0.0293 / 0.021 / 1.335 / 0.0329 on v5.0 "Atlas" data. Numerai called it "a drastic improvement in performance with no change to the underlying model." — [Numerai forum, 2024-07-17](https://forum.numer.ai/t/v5-atlas-data-release/7576)
- Validation Sharpe is the mean of per-era correlations divided by their standard deviation (staff-confirmed). — [Numerai forum](https://forum.numer.ai/t/submission-core-metrics/744) (snippet). One analysis found MMC Sharpe about half of CORR Sharpe. — [Numerai forum](https://forum.numer.ai/t/mmc-and-other-metrics-for-all-targets/6839) (snippet)

**DRW Crypto Market Prediction (Kaggle, 2025)**
- The metric is Pearson correlation. A forum thread reports that the organizers invalidated submissions after malpractice and reissued the dataset. — [fast.ai forum](https://forums.fast.ai/t/95-accuracy-notebooks-in-the-recent-drw-crypto-market-prediction-kaggle-comptetition/121591) (snippet; date and edition not verified)

### Inferences
- **Which scores are a bar for price-only models.** Ubiquant (about 0.13) and Jane Street (R² about 0.01) used proprietary anonymized features, so they are not a bar for a price-only model. G-Research (crypto market data, live-scored) and JPX (a portfolio-Sharpe score on about 2,000 Japanese stocks, close to the user's own evaluation framing) are the nearest comparators. G-Research's roughly 0.03 live correlation, won with LightGBM plus features, is the most honest single number. The JPX feature set (whether it included fundamentals) was not verified here.
- **Red flag.** A per-period cross-sectional correlation of 0.01-0.05 is competition-winning. A price-only model claiming daily IC above about 0.1 on liquid US stocks should be treated as leakage until proven otherwise.
- **Decay without updating.** The Jane Street 2024 numbers (gap-test score 0.0011 without online learning vs 0.0136 with it) show how fast a static model decays on a far-ahead test. A held-out-year test of stockai should allow scheduled retraining or online updates inside the year, as was done there. Otherwise a frozen model will understate what a maintained one could do.
- **Architecture lesson for an "LLM-like" model.** In one documented top solution of the most recent large competition (Jane Street 2024), a GRU with online updates was used. Its author reports that MLPs, time-series transformers and cross-symbol attention "didn't work". That is one competitor's experience, not a competition-wide result. The verified winning methods elsewhere were LightGBM + TabNet (Ubiquant) and plain LightGBM (G-Research). Optiver placings below first used GBDT and GBDT + NN blends; the first-place method is unknown.
- **Hardware.** The Jane Street 2024 solution needed about 100 GB RAM and 12 GB VRAM. The user's 61 GB RAM and 16 GB VRAM are enough for GRU-scale models if data are streamed or held in float16, which is an inference.

### Gaps
- The Jane Street 2024-25 final leaderboard (final standings dated about 2025-07-12 per clist.by profile listings) and the confirmed winners' methods were not retrievable. Kaggle discussion and writeup pages render as title-only to the fetch tool.
- I found no final utility score for Jane Street 2021, no winning method or score for JPX (first place kisa031), no first-place method for Optiver Trading at the Close ("hyd") or G-Research (Meme Lord Capital), and no results or methods for DRW 2025 or other 2025 Kaggle finance contests (for example Hull Tactical, Mitsui).
- No exact Ubiquant metric definition was retrieved, and no public-LB versus private-LB rank-change ("shake-up") statistics for any competition.
- Current (2025-2026) Numerai CORR20v2 benchmark-model figures and Numerai fund returns were not found.

## 3. Cross-sectional deep architectures (MASTER, FactorVAE, HIST, Stockformer, 2024-2026 transformers and foundation models): do the gains survive independent tests?

### Takeaway
Reported gains from 2021-2024 cross-sectional deep models rest mostly on China A-share CSI300/CSI800 tests that end around 2020-2022. The one third-party re-run I found (R&D-Agent(Q), NeurIPS 2025) got MASTER IC 0.0215, against about 0.064 in the original paper. MASTER's own authors say their published test data were mis-processed. Independent 2025 evidence on US daily excess returns ranks CatBoost above time-series foundation models on R² (all R² ≤ 0), and above zero-shot or fine-tuned TSFMs on long-short Sharpe; financially pretrained TSFMs roughly match it. I found no independent replication of FactorVAE, HIST or Stockformer.

### Cited Findings
**MASTER (AAAI 2024; arXiv 2312.15235)**
- The paper's CSI300 table reports MASTER at IC 0.064 ± 0.003 and RankICIR 0.48 ± 0.04. — [arXiv 2312.15235](https://arxiv.org/pdf/2312.15235) (snippet; table not read directly); [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/27767)
- The authors' README says:
  - "Our original experiments were conducted in a complex business codebase… The original code is confidential."
  - The "previous published valid & test data are problematic as we mistakenly used the training processors… valid & test data per day contains 95% of all stocks, but in our original experiment, we used all stocks… our access has expired and we cannot dump the original correct valid & test data again."
  - It substitutes open-source data, with results for seeds 0-4 in `model/performance.xlsx`, and "We did not tune on hyperparameters."
  - The Qlib port lacks "DropExtremeLabel" (dropping the 5% most extreme labels in training) and uses different market indices.
  - A notice dated 2025-06-26 says validation in the linked Qlib version used `learn_processor` instead of `infer_processor`.
  - Model input is N stocks × T=8 days × F=222 (158 Alpha158 factors + 63 market-information features + label).
  — [SJTU-DMTai/MASTER README](https://github.com/SJTU-DMTai/MASTER)
- Third-party run on the Qlib CSI300 2017-2020 test: MASTER IC 0.0215, ICIR 0.1925, Rank IC 0.0296, Rank ICIR 0.2486, but ARR 0.0896, IR 1.3406, MDD -0.0851. In the same table, Transformer reached IC 0.0317 and DoubleEnsemble 0.0294. Baselines there appear to use a 20-factor input (see section 1). — [arXiv 2505.15155 Table 1](https://arxiv.org/html/2505.15155v2)

**Time-series foundation models on daily excess returns (independent, 2025)**
- Rahimikia, Ni and Wang, "Re(Visiting) Time Series Foundation Models in Finance" (arXiv 2511.18578, 2025-11-23). Data: daily excess returns 1990-2023 for 94 countries, with expanding-window out-of-sample tests 2001-2023 and the main test on US data. Benchmarks: OLS, Lasso, Ridge, Elastic Net, PCR, XGBoost, CatBoost, LightGBM, NN-S and NN-L. — [arXiv HTML](https://arxiv.org/html/2511.18578v1); [abstract](https://arxiv.org/abs/2511.18578v1)
- Out-of-sample R²: the linear model averages -0.47%; CatBoost averages -0.10% (window 512: -0.03%, directional accuracy 51.16%, F1 0.49). Zero-shot Chronos-large scored -1.37% (accuracy just above 51%) and zero-shot TimesFM-500M -2.80% (accuracy just below 50%). Chronos-small pretrained from scratch on financial data improved from -77.07% to -3.18% at window 5, and from -1.27% to -0.59% at window 512, still behind the benchmarks. — [arXiv 2511.18578](https://arxiv.org/html/2511.18578v1)
- Equal-weighted decile long-short portfolios, rebalanced daily, before transaction costs:
  - CatBoost (window 252): 46.50% a year, Sharpe 6.79.
  - Zero-shot (window 512): Chronos-large 20.17% a year; TimesFM-500M -1.47%.
  - From scratch (window 512): Chronos-small 36.84%, Sharpe 5.42; TimesFM-20M 30.36%, Sharpe 3.66.
  - Scaled-data Chronos-small with factors and synthetic augmentation: 41.89%, Sharpe 6.78, versus CatBoost 47.25%, Sharpe 6.46.
  - The paper's conclusions: "Ensemble models are the strongest benchmarks, with CatBoost delivering the best risk-adjusted performance"; off-the-shelf TSFMs "perform poorly"; performance degrades over time for all models, more slowly for TSFMs.
  — [arXiv 2511.18578](https://arxiv.org/html/2511.18578v1)

**Decoder-only "next-token" return model (2026)**
- VAIOM (Ma and Chen, arXiv 2607.13929, 2026-07-15) is a 0.9M-parameter decoder-only transformer. It takes continuous inputs and outputs a categorical distribution over volatility-normalized return buckets, trained with cross-entropy. Data are 1-hour FX bars: fit before 2024, selected on 2024H2, tested on two 2025 halves without refitting. Every model across 3 seeds beat a fixed single-bar LightGBM on likelihood, gaining +0.029 and +0.043 bits per event. The smallest architecture had the best validation likelihood. The abstract reports no IC, Sharpe or cost-adjusted results. — [arXiv 2607.13929](https://arxiv.org/abs/2607.13929)

**ML strategies after realistic frictions (US)**
- Azevedo, Hoegner and Velikov, "The Expected Returns on Machine-Learning Strategies" (working paper 2023; AEA 2025 program), used 320 anomalies. Transaction costs, post-publication decay and post-decimalization liquidity cut performance by about 57% on average. LSTM-based strategies stayed the most profitable, with net out-of-sample returns up to 1.42% a month and a six-factor net alpha of 1.20% (t = 3.46). Cost-mitigation techniques lowered turnover but "do not improve net anomaly performance". These are monthly, characteristics-based strategies, not price-only. — [AEA 2025 program](https://www.aeaweb.org/conference/2025/program/paper/TihddefB); [Quantpedia summary](https://quantpedia.com/the-expected-returns-of-machine-learning-strategies/)
- Fischer and Krauss (European Journal of Operational Research, 2018), LSTM on S&P 500 constituents 1992-2015 at daily horizon: "daily returns of 0.46 percent and a Sharpe ratio of 5.8 prior to transaction costs". "The outperformance relative to the general market is very clear from 1992 to 2009, but as of 2010, excess returns seem to have been arbitraged away." — [repository abstract record](https://expeditiorepositorio.utadeo.edu.co/entities/publication/b0470fa3-a2f0-4f9f-ab8d-29b9596d050b) (abstract only; journal text not verified); a later benchmark of these figures is in [Ghosh, Neufeld, Sahoo 2020, arXiv 2004.10178](https://arxiv.org/pdf/2004.10178)

**Image-CNN price patterns, US 1993-2019 (Jiang, Kelly, Xiu, Journal of Finance 2023)**
- Citation: J. Finance 78: 3193-3249. — [DOI 10.1111/jofi.13268](https://doi.org/10.1111/jofi.13268); [SSRN 3756587](https://papers.ssrn.com/abstract=3756587)
- Two secondary sources conflict, and I could not read the primary tables:
  - CXO Advisory (2021-01-21, working-paper era): extreme-decile hedge gross Sharpe "as high as 2.5 (0.6)" for equal-weighted (value-weighted). The nearest competitor, 1-week reversal, scored 1.3 (EW) and 0.3 (VW). Out-of-sample direction accuracy was 52%-53%. — [CXO Advisory](https://www.cxoadvisory.com/?p=34917)
  - BSIC (2026-03-01): equal-weighted long-short Sharpe 7.2, 6.8 and 4.9 for the 5-, 20- and 60-day image models, falling to "slightly above 1" for the 500 largest stocks. Training covers 1993-2000 and out-of-sample 2001-2019. — [BSIC](https://bsic.it/technical-analysis-on-steroids/)
  - Both are gross of costs, and they probably describe different paper versions, horizons or tables.

### Inferences
- **The recurring pattern for daily or weekly cross-sectional price signals in US equities.** Out-of-sample R² is at or below zero and direction accuracy is about 51-53%. Equal-weighted decile Sharpe before costs is very large (5-7). It falls to about 0.6-1 when value-weighted or restricted to large caps, and costs then decide the outcome. A stockai result that looks like "51.5% accuracy, EW Sharpe 6 gross" is the normal starting point, not evidence of an edge.
- **For the "predict the next price like the next token" design.** Generic pretrained TSFMs do not transfer zero-shot. Pretraining from scratch on financial data closes most of the gap to CatBoost but does not clearly beat it on US daily R². A bucketed-return decoder can beat LightGBM on log-likelihood (FX hourly), but that has not been shown to turn into after-cost equity profits. The bar is therefore CatBoost/LightGBM on the same inputs, scored on both likelihood and after-cost portfolio results.
- **Independent re-runs tend to shrink headline IC gains.** MASTER's paper IC was about 0.064, but the third-party re-run measured 0.0215, near the bottom of that table. The same re-run's IR of 1.34 was among the highest in the table. I did not retrieve the paper's own IR, so no before/after IR comparison is possible. IC and portfolio metrics can diverge, so both must be reported.

### Gaps
- No independent replication found for FactorVAE (AAAI 2022), HIST, Stockformer, or other 2024-2026 cross-sectional transformers on either Chinese or US data. One search was run for FactorVAE/HIST and found nothing.
- AlphaDiverse (arXiv 2609.29014, 2026) reportedly runs MASTER as a baseline with medians across 3 seeds and a LightGBM capped at 500 trees; its numbers were not retrieved.
- Not retrieved: the transaction-cost section (5.2.4) and international results of the TSFM paper; the primary JKX tables (5-day Sharpe, value-weighted results, turnover); and Kronos and other financial "K-line" foundation models (2025), which were not examined here.

## 4. Simple price-only rule baselines in US equities (1-day/1-week reversal, 12-1 momentum, industry momentum, low volatility): 2015-2026 performance and capacity after costs

### Takeaway
Gross of costs, the classic price-only rules were weak in US equities over 2015-2026:
- value-weighted 1-month reversal was about 0 (Sharpe 0.02 in 2015-2025) and negative since 2020;
- the daily-rebalanced 20-day reversal factor fell from Sharpe 2.42 (1963-2014) to 0.38 (2015-2025);
- UMD 12-2 momentum Sharpe was 0.16 (CAPM alpha 6.1% a year, t = 1.55);
- industry momentum Sharpe was 0.26-0.40;
- low volatility worked only as a defensive long-only, beta-adjusted play (Sharpe 0.88 versus the market's 0.76; CAPM alpha 2.5% a year, t = 1.4).

After realistic costs, rules turning over more than 90% a month lost 0.3-1.3% a month historically (1963-2013), and the average published anomaly nets about 4 bps a month. "Beats reversal or momentum gross" is therefore not a bar. "Beats SPY after 5-20 bps per trade" is.

### Cited Findings
**Computed by this researcher from the Ken French Data Library (files built from the 202608 CRSP database; value-weighted NYSE/AMEX/NASDAQ; survivorship-free; gross of costs; 2026 = January-August only).**
- Sources: [French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), files F-F_Research_Data_Factors, F-F_Momentum_Factor, F-F_ST_Reversal_Factor, F-F_ST_Reversal_Factor_daily, Portfolios_Formed_on_VAR and 49_Industry_Portfolios (all under `.../ken.french/ftp/`).
- Run on 2026-10-18 with a session-scratchpad script that may not persist. The method below is complete enough to reproduce the numbers.
- Statistics: annualized mean = monthly mean × 12 (daily × 252); Sharpe = mean/sd × √12 (√252); CAPM alpha from OLS on Mkt-RF.
- Definitions, per the file headers:
  - ST_Rev monthly: sort on prior 1-month return.
  - ST_Rev daily file: "Prior return is measured from day -20 to -1", "constructed daily". This is a 20-day reversal rebalanced daily, not a 1-day or 1-week reversal.
  - Mom: "Prior return is measured from month -12 to -2".
  - VAR portfolios: total return variance over 60 days up to month-end, at least 20 observations.
  - Industry momentum is my own construction from the 49 value-weighted industries: long the top 10 and short the bottom 10 by past return, held 1 month.

| Strategy (gross, VW) | Jul 1963-2014 | 2000-2014 | 2015-2025 | 2020-Aug 2026 |
|---|---|---|---|---|
| Market excess (Mkt-RF) | 6.12%/yr, SR 0.40 | 4.04%, SR 0.26 | 11.87%, SR 0.76 | 13.08%, SR 0.75 |
| UMD momentum (12-2), monthly | 8.20%, SR 0.56, t 4.02 | 2.03%, SR 0.10 | 2.18%, SR 0.16, t 0.53; CAPM α 6.09% (t 1.55) | 2.71%, SR 0.18 |
| ST_Rev (1-month), monthly rebalance | 5.91%, SR 0.54, t 3.90 | 3.21%, SR 0.22 | 0.17%, SR 0.02, t 0.05; CAPM α -3.02% | -5.43%, SR -0.41 |
| ST_Rev (days -20..-1), daily rebalance | 26.98%, SR 2.42, t 17.3 | 12.58%, SR 0.80 | 6.02%, SR 0.38, t 1.26 | (2015-Aug 2026: 5.85%, SR 0.37) |
| Low vol: Lo20 - Hi20 (60-day variance quintiles) | 2.78%, SR 0.13; CAPM α 8.32% (t 3.45), β -0.91 | 6.00%, SR 0.21 | -1.78%, SR -0.07; CAPM α 8.08% (t 1.28), β -0.83 | -12.10%, SR -0.37; CAPM α 1.01% |
| Low vol long-only: Lo20 minus T-bill | 6.10%, SR 0.51; α 1.89% (t 2.43) | 5.88%, SR 0.53 | 11.35%, SR 0.88; α 2.50% (t 1.44), β 0.75 | 11.04%, SR 0.77; α 1.71% |
| Industry momentum 12-2 (49 industries, top 10 - bottom 10) | 8.59%, SR 0.54, t 3.90 | 3.11%, SR 0.16 | 3.67%, SR 0.26, t 0.86 | 4.17%, SR 0.26 |
| Industry momentum, 1-month formation | 8.38%, SR 0.64 | 3.40%, SR 0.21 | 4.96%, SR 0.40, t 1.34 | 11.22%, SR 0.82, t 2.11 |

- Calendar-year compounded monthly factor returns, 2015 through 2025 and then 2026 January-August (same computation):
  - UMD: 21.1, -18.2, 3.8, 10.7, -5.5, -0.9, -2.9, 20.2, -20.5, 18.5, -2.2, 3.9.
  - ST_Rev monthly: 1.3, 5.5, 3.5, 3.9, 17.1, -7.4, 4.0, -6.5, 4.7, -8.3, -18.0, -7.5.
  - ST_Rev daily-rebalanced: -1.5, 14.4, 3.8, 4.7, 13.3, 11.6, -1.7, 4.2, -7.3, 3.7, 11.0, 0.7.
  - Low-vol Lo20-Hi20: 13.3, -9.2, 5.1, 17.5, -2.0, -27.1, 31.8, 53.3, -46.4, -17.1, -27.3, -35.0.
  - The French file's own annual ST_Rev for 2025 is -20.84, built from annual portfolio returns, which is why it differs from my compounded figure. — [French ST_Rev file](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

**Trading costs and capacity: Novy-Marx and Velikov, "A Taxonomy of Anomalies and Their Trading Costs" (Review of Financial Studies 2016; value-weighted, July 1963 to December 2013; Hasbrouck effective spreads)**
- Table 3, monthly figures [t-stat]:

| Strategy | Gross return | One-sided turnover | Cost | Net return |
|---|---|---|---|---|
| Momentum | 1.33% | 34.5% | 0.65% | 0.68% [2.45] |
| Idiosyncratic volatility | 0.63% | 24.6% | 0.52% | 0.11% [0.37] |
| Industry momentum (past-month industry return) | 0.93% | 90.1% | 1.22% | -0.29% [-1.20] |
| Short-run reversals | 0.37% | 90.9% | 1.65% | -1.28% [-6.02] |
| Industry-relative reversals | 0.98% | 90.3% | 1.78% | -0.80% |
| High-frequency combo | 1.61% | 91.0% | 1.45% | 0.16% [1.11] |
| Seasonality | 0.84% | 91.1% | 1.46% | -0.62% |

  "The cost of trading the high-turnover strategies (those with monthly turnover in excess of 90% per month)… always exceeds 1% per month." — [NMV PDF](https://mysimon.rochester.edu/novy-marx/research/ToAatTC.pdf); [NBER w20721](https://www.nber.org/papers/w20721)
- Table 6, with buy/hold bands (10%/50% for high-turnover rules): short-run reversals net -0.72% [-3.78] at 57.1% turnover; industry momentum net -0.02%; high-frequency combo net 0.35% [2.83]. Momentum with a 10%/20% band nets 0.85% [3.35] at 18.8% turnover. — [NMV PDF](https://mysimon.rochester.edu/novy-marx/research/ToAatTC.pdf)
- Rule of thumb: most anomalies with one-sided monthly turnover below 50% keep significant net spreads; few higher-turnover strategies do. — [NBER abstract](https://www.nber.org/papers/w20721)
- Table 7, capacity before the last trader's profits vanish (December 2012, one-sided, 10/50 rules on NYSE breaks): momentum $5.16B, high-frequency combo $0.38B, industry-relative reversals (low-vol) $0.18B. Size, with capitalization breaks, reaches about $169B. — [NMV PDF](https://mysimon.rochester.edu/novy-marx/research/ToAatTC.pdf)

**Anomaly returns after costs and decay**
- Chen and Velikov, "Zeroing in on the Expected Returns of Anomalies" (JFQA 2023), covering 204 anomalies. After effective spreads, post-publication decay and modern trading conditions, the average anomaly's expected return is 4 bps a month. The strongest net at best 10 bps after controlling for data mining, and combinations net about 20 bps. Price impact is excluded, so these figures are optimistic. The earlier 2020 Fed working paper, with 120 anomalies, found 8 bps. — [Penn State abstract](https://pure.psu.edu/en/publications/zeroing-in-on-the-expected-returns-of-anomalies/); [Fed WP 2020](https://www.federalreserve.gov/econres/feds/zeroing-in-on-the-expected-returns-of-anomalies.htm); [JFQA](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/zeroing-in-on-the-expected-returns-of-anomalies/945133D5A3ECEEAF466AEE91551FD225)
- Blitz, Hanauer, Honarvar, Huisman and van Vliet, "Beyond Fama-French Factors: Alpha from Short-Term Signals" (Financial Analysts Journal 2023; MSCI World, December 1985 to December 2021):
  - Single short-term signals (short-term reversal, short-term momentum, analyst revisions, risk, monthly seasonality) average more than 6% a year gross six-factor alpha, but less than -2% net at 25 bps per single trip.
  - A composite signal reaches more than 12% gross, and a "buy 10 / hold 50" rule lifts net alpha above 6%.
  - The authors report the result holds out of sample, after publication, across regions, and with implementation lags of several days.
  - Figures are from the Robeco and CFA summaries; the full paper is paywalled. — [CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/2023/beyond-fama-french-factors-alpha-from-short-term-signals); [Robeco summary](https://www.robeco.com/en-uk/insights/2022/05/beyond-fama-french-alpha-from-short-term-signals); [RePEc](https://ideas.repec.org/a/taf/ufajxx/v79y2023i4p96-117.html)

### Inferences
- **The market is the real benchmark.** Holding the market (roughly SPY) earned Sharpe about 0.76 over 2015-2025 (French Mkt-RF). No classic price-only long-short rule matched that even gross in this window. A stockai strategy has to clear about Sharpe 0.8 net, or beat SPY/QQQ total return at similar risk.
- **Consistent with correlation2.** The prior project's "reversal positive control was detected but faded" matches these numbers: the daily-rebalanced 20-day reversal Sharpe fell from 2.42 to 0.38, and 1-month reversal has been about zero or negative since 2015.
- **Cost arithmetic for the user's 5-20 bps per trade (my arithmetic).** Annual cost ≈ 2 × one-way turnover per rebalance × cost per side × rebalances per year.
  - Full daily replacement: 2 × 1 × c × 252 = 25-101% a year at c = 5-20 bps.
  - Full weekly replacement: 5.2-20.8% a year.
  - Qlib-style 10% daily turnover: 2.5-10% a year.
  - A daily model must therefore pick names with an expected next-day edge of more than 2c (10-40 bps round trip) per traded name, or trade much less often.
- **Survivorship bias.** Stooq covers only firms listed in 2025, so a Stooq backtest of these same rules should come out better than the CRSP-based numbers above. Run the rules on Stooq and compare them with these numbers to see roughly how large the bias is.
- **Spread estimates.** For a $10,000 account, market impact is negligible, but spreads in small caps are wide. Equal-weighted or small-cap results, which look best gross, are where costs bite hardest. Estimating effective spreads from daily OHLC is a possible next step; it was not researched here.

### Gaps
- No post-2015 US performance figures for 1-day or 1-week reversal. The French library has only 1-month and 20-day reversal. These need to be computed from CRSP or Stooq. The only 1-week reversal figure found covers an earlier window: gross Sharpe 1.3 equal-weighted / 0.3 value-weighted, US 2001-2019, from the CXO summary of Jiang-Kelly-Xiu (see section 3; [CXO Advisory](https://www.cxoadvisory.com/?p=34917)).
- No published net-of-cost estimates per rule after 2013 (NMV ends in 2013; Chen-Velikov is an aggregate). No post-2012 capacity estimates.
- No published 2015-2026 net results for low-volatility or betting-against-beta strategies were retrieved. My low-vol rows are gross quintile spreads, not BAB.
- My industry-momentum construction (top/bottom 10 of 49) is a simple proxy, not Moskowitz-Grinblatt's exact method.

## 5. Evaluation practice (IC/rank IC/ICIR, decile spreads, turnover-adjusted returns, walk-forward, purged/embargoed CV, probabilistic metrics, significance tests)

### Takeaway
Practitioners score daily cross-sectional forecasts with per-date IC and rank IC and their ratio, ICIR = mean/std (Numerai calls this the "Sharpe" of per-era correlation). They then build after-cost top-k or decile portfolios against a benchmark and report excess return, IR, max drawdown and turnover. Validation is strictly chronological walk-forward with purging and embargo gaps of at least the label horizon, plus multiple seeds and multiple-testing corrections (Deflated Sharpe Ratio, probability of backtest overfitting, t > 3). Distributional scoring (log-likelihood, CRPS) is rare in equity benchmarks; the only instance found is a 2026 FX paper scored in bits per event.

### Cited Findings
- **Qlib's standard pair of evaluations.** Signal metrics are IC, ICIR, Rank IC and Rank ICIR. Portfolio metrics are annualized excess return with cost, IR and MDD from a TopkDropout backtest, each as mean ± std over 20 seeds. — [Qlib README](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md); [run_all_model.py](https://github.com/microsoft/qlib/blob/main/examples/run_all_model.py). Signal analysis annualizes with `ann_scaler: 252`. — [YAML](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml). For non-stationarity, Qlib provides "Rolling Retraining" baselines. — [Qlib main README](https://github.com/microsoft/qlib/blob/main/README.md)
- **Numerai.** Rank-gaussianized, tail-weighted Pearson correlation per era, at 20- and 60-day horizons, plus MMC measured against the stake-weighted metamodel. — [Numerai docs](https://docs.numer.ai/numerai-tournament/scoring/correlation-corr). Validation Sharpe is the mean of per-era correlations divided by their std. — [Numerai forum](https://forum.numer.ai/t/submission-core-metrics/744) (snippet)
- **Competition validation designs that worked:**
  - Jane Street 2024: two time-series folds of 200 dates each, plus a model tested on the last 200 dates after a 200-date gap "to simulate the private dataset scenario". — [solution.md](https://github.com/evgeniavolkova/kagglejanestreet/blob/master/solution.md)
  - Ubiquant winner: PurgedGroupTimeSeries + TimeSeriesSplit for feature and parameter selection. — [1st-place write-up](https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy) (snippet)
  - G-Research 2nd place: "good N-fold cross validation is essential." — [discussion](https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/323098) (snippet)
- **Purging and embargo (López de Prado 2017; Advances in Financial Machine Learning, 2018).** Purging drops training samples whose label windows overlap test labels. The embargo drops a buffer after each test block to cover serial correlation. — [Wikipedia: Purged cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation); [LuxAlgo explainer](https://www.luxalgo.com/library/concept/purged-cross-validation/). An embargo of about 1% of bars is a common rule of thumb. — [Return on Effort blog](https://returnoneffort.beehiiv.com/p/mitigating-overfitting-advanced-crossvalidation) (secondary)
- **Combinatorial purged CV (CPCV).** Test sets are every combination of k of N groups (for example N=6, k=2 gives 15 splits). This yields many out-of-sample backtest paths, a distribution of Sharpe ratios, and an estimate of the probability of backtest overfitting (PBO). — [fynance docs](https://fynance.readthedocs.io/en/latest/generated/fynance.data.combinatorial_purged_cv.html)
- **Deflated Sharpe Ratio (Bailey and López de Prado, Journal of Portfolio Management 2014).** Corrects a Sharpe ratio for selection bias under multiple testing and for non-normal returns. — [SSRN 2460551](https://papers.ssrn.com/abstract=2460551)
- **Probability of Backtest Overfitting (Bailey, Borwein, López de Prado, Zhu; Journal of Computational Finance).** Estimated with combinatorially symmetric cross-validation (CSCV). — [SSRN 2326253](https://papers.ssrn.com/abstract=2326253)
- **Higher significance bar.** Harvey, Liu and Zhu (2016) recommend a t-statistic hurdle of 3.0 rather than 2.0 for new factors. This is a secondhand description. — [arXiv 2603.20319](https://arxiv.org/pdf/2603.20319)
- **Cost-aware reporting standard.** Report gross return, turnover, effective-spread costs, net return and net alpha, and use buy/hold bands to cut turnover. — [Novy-Marx and Velikov 2016](https://mysimon.rochester.edu/novy-marx/research/ToAatTC.pdf). Adjust also for post-publication decay and data mining. — [Chen and Velikov 2023](https://pure.psu.edu/en/publications/zeroing-in-on-the-expected-returns-of-anomalies/); [Azevedo, Hoegner, Velikov](https://www.aeaweb.org/conference/2025/program/paper/TihddefB)
- **Foundation-model evaluation.** Expanding window, out-of-sample 2001-2023, reporting R²_OOS, directional accuracy, F1 and equal-weighted decile long-short portfolios. — [arXiv 2511.18578](https://arxiv.org/html/2511.18578v1). Likelihood (bits per event) against LightGBM across 3 seeds, with test halves in 2025 and no refitting. — [VAIOM, arXiv 2607.13929](https://arxiv.org/abs/2607.13929)

### Inferences
- **Recommended stockai protocol (synthesis of the above):**
  1. Per-date rank IC and ICIR on next-day and next-5-day returns, reported by year and by market-cap bucket.
  2. Top-k and decile long-short plus long-only portfolios at 5, 10 and 20 bps per side, with turnover, against SPY and QQQ.
  3. The same pipeline run with LightGBM/CatBoost on Alpha158-style OHLCV features, and with the simple rules from section 4, as baselines.
  4. Walk-forward retraining (monthly or weekly) with an embargo of at least the label horizon.
  5. At least 5 seeds.
  6. Deflated Sharpe or PBO, counting every configuration tried.
  7. The user's existing random-account null with matched turnover and stock pool.
  8. For a "next-token" distributional model: log-likelihood or CRPS against a calibrated baseline, plus a check that likelihood gains turn into after-cost IC (VAIOM shows likelihood gains alone are not trading evidence).
- **Statistical power of a one-year hold-out (my arithmetic).** The t-stat of mean IC ≈ daily ICIR × √(number of days). Over about 250 days, a Qlib-best ICIR of 0.4 gives t ≈ 6.3, but an ICIR of 0.1 gives only t ≈ 1.6. For portfolio returns, t ≈ annual Sharpe × √years, so a Sharpe-1 strategy over one held-out year has t ≈ 1 and cannot be told apart from luck. The user's single-year test can reject weak claims but cannot confirm modest edges. Multi-year walk-forward tests over 2015-2025 are needed.

### Gaps
- No literature on random-portfolio or permutation tests (for example Burns' random portfolios, White's Reality Check, Hansen's SPA, Romano-Wolf) was retrieved in this session. These are standard tools, but no sources are cited here.
- No equity-return benchmark using CRPS or log-likelihood as its main metric was found. VAIOM (FX, 2026) is the only likelihood-scored example.
- López de Prado's primary texts were not consulted directly; the purge, embargo and CPCV descriptions come from Wikipedia, documentation and explainer pages.
