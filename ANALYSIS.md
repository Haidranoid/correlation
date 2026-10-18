# correlation v1: analysis and viability test

This reviews the original Rails app (`~/programming/correlation`, GitHub `bhoult/correlation`), checks its data, tests its core idea properly on that data, and surveys what others have done. `SPEC.md` is the plan for the Rust rebuild that follows from it.

## Verdict

- **v1 never measured whether its idea works, and its saved results are artifacts.** It analyzed only 3 pairs of stocks. 95% of its 1,246 "significant" correlations are offsets of more than a year, fitted on as few as 10 data points and picked from about 14,000 offsets tried per pair.
- **The core idea, tested properly, did not beat random guessing.** The idea is that one stock's daily moves predict another's over the next 1 to 5 trading days. The test learned correlations from June 2021 to June 2024 and treated June 2024 to June 2025 as the unseen future:
  - The strongest stored pairs were no stronger than what pure chance produces from 23 million tries.
  - In the holdout year, their predictions were right 52.3% of the time, against 49.6% for random pairs.
  - A $10,000 account trading on them made 15.6% before costs. SPY made 13.6% and QQQ 14.3%. But 24.5% of random-guess accounts did at least as well (p = 0.245), so the win is consistent with luck.
  - Trading costs of 5 basis points per trade turned the result into −8.0%.
- **The pipeline itself works.** It recovers a known effect: the average large cap's move predicts small caps the next day (r = +0.20 in training, +0.08 in the holdout). Lead-lag exists at the portfolio level; pairs of individual stocks are too noisy.
- **Published work points elsewhere.** Lead-lag that survives scrutiny shows up in *clusters* of stocks (Bennett, Cucuringu & Reinert 2022) and in *economic links* at weekly to monthly horizons: customer to supplier (Cohen & Frazzini 2008), and firms covered by the same analysts (Ali & Hirshleifer 2020). Lower turnover is what gives those a chance after costs.
- **Is a rebuild worth it?** Not as "v1 but faster". The pair search is now a 0.2-second numpy job. It is worth it as a fast, honest testing platform for those better-founded signal families, where every number comes with its holdout result and its odds against random guessing. Expect the likely answer to still be "no tradable edge after costs". The platform's value is reaching that answer, or finding the exception, cheaply and reliably.

## 1. What v1 is

A Rails 8 app backed by PostgreSQL:
- **Data:** imports Stooq bulk zips; 26.5M daily prices for 11,431 US symbols, 1962 to 2025-06-27.
- **Company info:** fetched from FMP, Alpha Vantage, Finnhub and IEX Cloud.
- **Correlation:** computed between two chosen symbols in Ruby, with an inline-C kernel, parallelized with the `parallel` gem.
- **Web UI:** a dashboard, stock detail pages and interactive explainer graphics.

Analysis runs only from rake tasks. No screen starts an analysis or browses correlations.

### Worth keeping
- **Stooq bulk import** as a free source, with data types derived from the folder structure inside the archive.
- **Multi-provider company info:** a provider adapter with field normalization; a status table per symbol and provider (found / not_found / rate_limited / error + last checked) that suppresses retries; resumable batch runs.
- **The vocabulary:** windows, offsets, leader and lagger, an effect-size framing. It needs redefining, though (section 2).
- **The UI ideas:** a stat-tile dashboard, zoomable charts, drill-down from a stock to its partners, and compact `[timestamp, value]` series.
- **The interactive explainers:** draggable correlation scatter, window and offset sliders, Cohen bands. Today they run on random data and should use real data.
- **Ops habits:** marking symbols inactive rather than deleting them, with dry-run and reactivation; rotating backups.

### Broken or fragile (from a full code survey)
- **Ingest:**
  - Every file outside the world zip is typed "unknown". That threw away the stock/ETF/exchange labels, so ETFs and stocks are mixed.
  - Market codes don't match the country table, so countries come out as "Unknown".
  - Hyphenated tickers are dropped (100 in the US zip).
  - A ticker listed on two markets collides on a unique index.
  - On a fresh database the foreign key to `stock_infos` makes the import store nothing while reporting success.
  - The importer `rm -rf`s its own `data/` folder, source zips included.
- **Company info:**
  - A "not found" from the first provider skips all the others.
  - FMP empty results are retried forever.
  - Rate limits are printed but not enforced.
  - Invented placeholder data (sector guessed from ticker length) is saved as fact.
  - Finnhub `employees` is filled from shares outstanding.
  - IEX Cloud no longer operates (it shut down in 2024).
- **Analysis:**
  - Offsets are calendar days matched exactly, while windows are trading days, so weekend offsets lose data.
  - Negative offsets are documented but never tested.
  - The per-offset sweep re-queries both full histories from Postgres for every offset.
  - `correlation:list` and `data_cleanup:reactivate` crash on dropped columns.
- **UI:**
  - Correlations are sorted by signed value, so strong negative ones never show.
  - The "recent prices" fallback returns the oldest 30 rows.
  - Markdown and visualization HTML are injected unescaped.
  - Any logged-in user can download full database dumps.
- **CI** runs `rails test`, but the suite is RSpec, so the specs never ran in CI.

## 2. Why v1 could not tell whether it works

| Problem | Evidence | Effect |
|---|---|---|
| Overlapping windows | N-day % changes computed every day (`StockDataService.calculate_percentage_changes`) | Neighbouring samples share N−1 days, so 1,000 points of 30-day changes hold only about 33 independent observations; strength and significance are overstated |
| Offset shorter than the window leaks | A 40-day change ending at t vs one ending at t+7 share 33 days | "Predictive" correlation is mostly same-day co-movement |
| Picking the best of thousands of tries | `find_best_correlation` tests offsets 0..max (about 14,000 for MSFT/AAPL) and keeps \|r\| ≥ 0.3 | Top saved result: "MSFT predicts AAPL 14,201 days (39 years) later, r = −0.80 on 38 points" |
| No chance baseline, no multiple-testing correction | The only bar is \|r\| ≥ 0.3 and n ≥ 10 | Noise passes as signal |
| No out-of-sample test | Each correlation is one number over the whole history | No way to know whether it persists |
| Price-level framing, no market adjustment | Raw % changes | Most stocks correlate positively just by moving with the market. v1's 7 offset-0 results average r = 0.40 over about 9,880 days: real, but inflated by overlapping windows and trivial |
| Speed | Per-pair, per-offset scalar loops with database I/O | All pairs × 5 lags at v1's own 50 ms per correlation on 24 cores: **about 16 days**. The same work as matrix products: **0.2 seconds** in numpy, and on the RTX 5080 **7 ms per lag** for all 11,431 stocks (`research/gpu_bench`). The full v1-style sweep (all stocks, 20 years, lags 1–250) is about 10 s on the GPU |

What v1 stored: 3 pairs, 6 window combinations, 1,246 rows. 7 rows have offset 0, 54 have offsets of 1 to 30 days, and **1,185 have offsets over 365 days** (average 236 points, minimum 10).

## 3. The data

**Adjustment:** Stooq prices are adjusted for both splits and dividends. Known split dates show no jump:

| Stock | Split | Prices around it |
|---|---|---|
| NVDA | 10:1, 2024-06-10 | 120.86 → 121.76 |
| CMG | 50:1, 2024-06-26 | 65.66 → 65.86 |
| AVGO | 10:1, 2024-07-15 | 169.51 → 170.86 |
| SMCI | 10:1, 2024-10-01 | 41.64 → 40.55 |
| WMT | 3:1, 2024-02-26 | 58.52 → 59.60 |

AAPL on 2020-01-02 is 72.72. The actual close was $300.35 (75.09 after the 4:1 split), so dividends are adjusted in as well. That makes the prices total-return series, the right basis for return correlations and for comparing with SPY and QQQ.

**Contamination:**
- **ZVZZT**, a NASDAQ test ticker, sits in Stooq's "nasdaq stocks" folder, passes volume filters and moves 3,000× in a day.
- **69 one-day bad prints** since mid-2021: a jump of more than 2× that undoes itself the next day.
- Most of the other 1,019 split-sized moves since 2021 are real crashes in small biotechs and similar stocks (e.g. SAVA −84%). Keep those, because a strategy has to survive them.

**Recording changes over time:**

| Decade | Symbols | Days with exactly 0% change | Rows priced under $1 after adjustment | Close-only rows |
|---|---|---|---|---|
| 1960s | 2 | 22.5% | 0% | 4.7% |
| 1970s | 40 | 29.2% | 47.7% | 9.9% |
| 1980s | 198 | 16.7% | 19.3% | 2.1% |
| 1990s | 307 | 10.2% | 5.4% | 0.9% |
| 2000s | 3,012 | 3.0% | 1.0% | 1.5% |
| 2010s | 6,039 | 2.9% | 0.5% | 1.5% |
| 2020s | 11,431 | 2.2% | 2.3% | 2.7% |

Before decimalization (2001), prices moved in 1/8 and 1/16 steps. Decades of adjustment then shrink old prices to pennies. Together those produce fake "no change" days and stair-stepped returns. History before about 2005 is also thin and contains only survivors.

**Survivorship:** Stooq ships only symbols listed when the snapshot was taken. 11,183 of the 11,431 symbols were still trading in June 2025, and only 7,889 have data in December 2021. Stocks that died before 2025 are missing, which flatters any long-only backtest.

**Staleness:** the data ends 2025-06-27, about 16 months ago. Since early 2026 Stooq's downloads need an API key obtained through a CAPTCHA and have daily limits; the bulk snapshots are still offered.

## 4. The holdout experiment

Script: `research/leadlag_holdout.py` (numpy, 9 s). Output: `research/results.txt`.

**Design** (decisions made from training data only):
- Training: 2021-06-28 to 2024-06-27 (755 days). Holdout: 2024-06-28 to 2025-06-27 (250 days).
- Universe: 2,155 US common stocks with price ≥ $5, median dollar volume ≥ $5M and ≥ 95% of days traded. ETFs and test tickers excluded, bad prints repaired.
- Daily log returns, minus each stock's beta to the equal-weight market (betas from training), standardized and clipped at ±5 sd.
- Lead-lag correlation for lags of 1 to 5 trading days, as one matrix product per lag: 23.2 million leader/target/lag combinations. The top 2,000 are stored.
- Chance level: the same search after shifting the targets' timeline, which destroys any real lead-lag but keeps everything else.
- Each holdout day, every stock's score is the sum over stored pairs of r × the leader's standardized move k days earlier. The account buys the top 20 predicted gainers equal-weight at the close and rebalances daily.
- Baselines: SPY and QQQ buy-and-hold, an equal-weight hold of all candidates, and 1,000 random guessers picking 20 of the same candidates each day under the same rules and costs.

**Results**

| Check | Result | Reading |
|---|---|---|
| Strongest stored \|r\| | 0.142 to 0.212 | |
| Chance level, same search | best 0.189, 2,000th best 0.133; theory ≈ 0.212 | The stored pairs are what chance produces |
| Stored pairs above the best chance result | 15 of 2,000 | |
| Direction (\|reverse r\| / \|forward r\|) | 0.17, vs 0.15 for chance pairs | Selection alone produces this; no evidence of real direction |
| Same sign in the holdout | 52.3% (random pairs 49.6%) | Tiny |
| Mean signed holdout r | +0.0028 (one pair's sd is 0.063) | Economically nil |
| By target liquidity (least / middle / most liquid third) | 51.7% / 52.4% / 52.8% | Not a stale-price artifact, just weak |
| Positive control: big caps (t) → small caps (t+1) | +0.197 training, +0.081 holdout (reverse +0.010 / −0.032) | The method detects real lead-lag where it exists |
| Picks beating the market next day | 51.6% of 4,997 picks | Not a valid significance test: picks on the same day share that day's market, so they aren't independent. Use the day-level IC below |
| Daily rank IC | +0.0034, t = +1.45 | Not significant |

**$10,000 for the holdout year**

| Strategy | Final | Return | Random guessers (median) | Share of random guessers at least as good |
|---|---|---|---|---|
| Buy and hold SPY | $11,361 | 13.6% | | |
| Buy and hold QQQ | $11,432 | 14.3% | | |
| Buy and hold all 1,099 candidates | $10,902 | 9.0% | | |
| Correlation picks, no costs | $11,561 | 15.6% | $10,868 | 24.5% (p = 0.245) |
| Correlation picks, 5 bps per trade | $9,199 | −8.0% | $8,502 | 19.6% |
| Correlation picks, 20 bps per trade | $4,629 | −53.7% | $4,071 | 8.3% |

The 20 bps row looks better against random only because the picks repeat from day to day a little more, so they trade less. That isn't better prediction. Daily rebalancing of 20 names costs about 2 × 5 bps × 250 days ≈ 25% a year at 5 bps, which no signal of this size can cover.

**Caveats:**
- It's one year and one configuration (daily horizon, lags 1 to 5, top 2,000).
- The universe has survivorship bias. That inflates all long-only rows equally, so the comparison with random guessers is fair; the comparison with SPY favours the stocks.
- Trading at the same close used to compute the signal is slightly optimistic.

## 5. What others do

**Research**
- **Bennett, Cucuringu & Reinert (2022)**, *Lead-lag detection and network clustering for multivariate time series with an application to the US equity market* (Machine Learning). The rigorous version of v1's idea:
  - 5,325 NYSE stocks, 2000 to 2019 (CRSP, so no survivorship bias); lead-lag metrics over lags −5 to +5 (distance correlation worked best).
  - Stocks clustered by directed flow imbalance using Hermitian random-walk clustering; leading-cluster returns used to forecast lagging clusters.
  - Rolling 1-year lookback, refit every 2 months.
  - Result: Sharpe 0.62 vs 0.40 for the S&P 500, mean 2.4 bps a day. **No transaction costs**; the authors say costs "may be significant".
  - Cluster persistence fell in later years. They cite Xia et al. (2018): individual lead-lag relationships last about a year.
- **Ali & Hirshleifer (2020)**, *Shared analyst coverage: Unifying momentum spillover effects* (JFE). Firms covered by the same analysts are "connected". A connected-firm momentum factor earns 1.68% a month alpha (t = 9.67) and absorbs industry, customer, supplier, geographic and technology momentum. Monthly horizon.
- **Cohen & Frazzini (2008)**, *Economic links and predictable returns* (JF). Customer returns predict supplier returns because attention is limited. Links come from mandatory disclosure of customers that make up more than 10% of sales.
- **Pu, Roberts, Dong & Zohren (2023)**, *Network Momentum across Asset Classes*. A learned sparse network over 64 futures; momentum spillover gives Sharpe 1.5 and 22% a year (2000 to 2022) after volatility scaling. A different universe, but a useful method: a dynamic, sparse, interpretable graph.
- **Decision-time leakage (arXiv 2605.23959, 2026).** A one-switch benchmark: deliberately introduce leakage and measure the gain in rank IC and the 5 bps Sharpe. The same idea becomes v2's "leakage canary".

**Tools**
- **Microsoft Qlib.** Config-driven pipelines, cached datasets, IC and group-return analysis, excess return gross and net of costs, rolling retraining as a baseline, a data health check. The model for v2's reports.
- **NautilusTrader.** A Rust-core, event-driven backtester with a Parquet data catalog. Its Rust API is still changing and needs Rust 1.94 (this machine has 1.90). Overkill for daily portfolio simulation; worth revisiting only for execution realism.
- **cudarc 0.19.10** supports CUDA 13.0 to 13.4 with runtime library loading. **Burn/CubeCL** compiles one kernel to CUDA, ROCm, Metal, WGSL or the CPU, so it's the portable alternative. **eframe/egui 0.36.1** is current and renders with wgpu by default.

**Data sources** (prices from vendor or comparison pages, October 2026; check before buying)

| Source | Delisted stocks | History | Notes | Cost |
|---|---|---|---|---|
| Stooq bulk zips (current) | No | 1962+ | Split- and dividend-adjusted; API key and CAPTCHA since 2026 | Free |
| MarketParquet | Yes | 2000+, through 2026-10-16 | Parquet by date, daily to 1-minute bars; split-adjusted (dividends not stated); upstream source not named | $79 once (+$15/mo to stay current); last year free |
| Sharadar via Nasdaq Data Link | Yes | 1998+ | Several adjustment methods, corporate actions | ~$9–39/mo personal |
| Norgate Data | Yes | 1990+ | Point-in-time S&P 500 / Russell membership | ~$346 per 6 months (older figure) |
| EODHD | Yes | 30+ years | Delisted flag per exchange (60k US entries) | Subscription |
| Massive (formerly Polygon) | Reportedly yes | 2003+ on a starter plan | Daily flat files on paid plans | ~$29/mo (second-hand) |
| Alpha Vantage `LISTING_STATUS` | List only | Any past date | Free point-in-time list of listed/delisted symbols; v1 already has a key | Free |
| CRSP | Yes | 1925+ | The academic standard | Institutional |

## 6. Recommendation

1. **Build v2 as a hypothesis-testing platform, not a pair finder.** The center is the walk-forward and holdout harness with random-guess odds, chance levels, cost sensitivity and leakage canaries. Signal families plug into it.
2. **Test the better-founded families first:** cluster lead-lag, peer/connected-firm momentum (peers from long-run same-day correlation, which *is* stable), and known anomalies (short-term reversal, 12-1 momentum) as calibration. Favour weekly or monthly rebalancing.
3. **Fix the data before trusting any result.** Get survivorship-free history (MarketParquet at $79 is the cheapest first try; validate it against the Stooq zips on overlapping dates), current through 2026, so the final holdout is the most recent year. Keep dividends consistent between the strategy and the benchmarks.
4. **Gate the GUI on results.** If no family beats random guessers after 5 bps costs in walk-forward across many years, stop or pivot to risk and diversification tooling, where same-day correlation is genuinely useful.

## Sources

- Bennett, Cucuringu, Reinert (2022): [arXiv 2201.08283](https://arxiv.org/pdf/2201.08283), [Springer](https://link.springer.com/10.1007/s10994-022-06250-4)
- Ali & Hirshleifer (2020): [SSRN](https://papers.ssrn.com/abstract=3015582), [NBER w25201](https://www.nber.org/system/files/working_papers/w25201/w25201.pdf)
- Cohen & Frazzini (2008): [AQR summary](https://www.aqr.com/insights/research/journal-article/economic-links-and-predictable-returns)
- Pu, Roberts, Dong, Zohren (2023): [arXiv 2308.11294](https://arxiv.org/pdf/2308.11294)
- Decision-time leakage benchmark: [arXiv 2605.23959](https://arxiv.org/pdf/2605.23959)
- Qlib: [github.com/microsoft/qlib](https://github.com/microsoft/qlib)
- NautilusTrader: [Rust backtests](https://nautilustrader.io/docs/latest/how_to/run_rust_backtest), [backtesting concepts](https://nautilustrader.io/docs/latest/concepts/backtesting/)
- cudarc features: [docs.rs/crate/cudarc](https://docs.rs/crate/cudarc/latest/features); eframe: [docs.rs/crate/eframe](https://docs.rs/crate/eframe/latest); CubeCL: [README](https://docs.rs/crate/cubecl/0.11.0-pre.1/source/README.md); Rust GPU ecosystem: [NVIDIA cuda-rust appendix](https://nvidia.github.io/cuda-rust/appendix/ecosystem.html)
- Data: [MarketParquet](https://marketparquet.com), [MarketParquet vs Quandl/Sharadar](https://marketparquet.com/vs/quandl), [EODHD delisted data](https://eodhd.com/financial-apis/delisted-stock-companies-data-2), [Norgate toolkit](https://concretumgroup.com/the-complete-norgate-data-toolkit-three-python-workflows-every-quant-researcher-needs/), [Alpha Vantage delisted stocks](https://www.macroption.com/alpha-vantage-delisted-stocks/), [Massive day aggregates](https://massive.com/docs/flat-files/stocks/day-aggregates.md), [Stooq API key change (pandas-datareader #1012)](https://github.com/pydata/pandas-datareader/issues/1012), [R-bloggers on delisted tickers](https://r-bloggers.com/2021/08/when-yahoo-finance-doesnt-have-de-listed-tickers-needed)
