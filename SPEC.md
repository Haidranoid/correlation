# correlation2: specification for the Rust rebuild

Status: draft 1, 2026-10-18. Background and evidence: `ANALYSIS.md`. Reference experiment: `research/leadlag_holdout.py` and `research/results.txt`.

## Status as of 2026-10-18 (first version built)

Measurements, results and reasons are in `test-results.md`; read that first. This spec is the original plan. Where the build differs, measurement changed the plan and the reason is recorded there.

**Built:**
- **M0:** engine benchmarks; cuBLAS/cuBLASLt in five precisions; AVX-VNNI assembly kernel.
- **M1:** Stooq ingest, store, calendar, quality checks.
- **M2:** CPU/GPU engine validated against the Python oracle.
- **M2a:** full census.
- **The holdout harness:** $10,000 test, turnover-matched random guessers, walk-forward, long-short diagnostic; families A (census pairs at 1-, 5- and 21-day targets), C (peer momentum) and E (reversal, as a positive control); `replicate`.
- **M3, first version:** `c2 update`, forward log, `forward-report`, `install-timer` (not enabled).
- **M7, first version:** `c2-gui` with Status, Census, Holdout and Data tabs.

**Changed by measurement:**
- **No block-additive sums (§6A optimization 1).** Both devices are limited by writing results, not by arithmetic, so each window is recomputed directly with fused thresholding.
- **Sign-flip null, not time-shift (§6A, §7.1).** The time-shift null misaligned volatility regimes and produced a false 17× excess.
- **Thresholds chosen on the GPU,** using |r| histograms, with one host sync per window.
- **FDR needs at least 10 entries,** with the analytic null as a floor and a placebo rate recorded.
- **Calendar from long-lived stocks, not SPY.** Stooq's SPY starts on 2005-02-25 and misses 2011-02-17.
- **Five crates, not seven.** The store lives in `c2-data`; census, holdout and analysis live in `c2-census`; `c2-update` lives in `c2-cli`.
- **"Refine as new data arrives" currently means a dated rerun** (a full census takes ~6 minutes). Per-pair running sums are stored for a future live tracker.

**Remaining:**
- census CPU worker
- idle mode
- families B (cluster lead-lag) and D (size)
- validated survivorship-free data source and automatic fetching
- ledger with deflated Sharpe (§7.4)
- locked-holdout counter (§7.5)
- GUI explore/learn screens (§9)

## 1. Purpose

correlation2 is a native desktop research tool that answers one question honestly and fast:

> Can relationships between stocks, learned only from the past, pick stocks that beat buying the S&P 500 (SPY) or the Nasdaq-100 (QQQ) by a margin that random guessing can't explain, after trading costs, in time the model has never seen?

Every result the app shows carries three things: its out-of-sample (holdout) number, the share of random-guess portfolios that did at least as well, and its result after costs. A number without these is never displayed as a finding.

The evidence so far (ANALYSIS.md) is that v1's core idea, daily pair lead-lag, does not beat random guessing. v2 is therefore a platform for testing several signal families, not a faster pair finder. A legitimate and likely outcome is "no tradable edge"; the app must reach that answer cheaply and convincingly, or find the exception.

### Non-goals
- No live trading, broker connections or order routing.
- No intraday strategies in v2.0. Daily bars only, though the data layer should not rule out intraday later.
- No multi-user server and no web UI.
- Platform: Linux (Ubuntu 26.04, RTX 5080) first. Windows and macOS are nice-to-haves via the CPU backend.

## 2. Glossary

| Term | Meaning |
|---|---|
| Residual return | A stock's daily log return minus beta × the market's return, i.e. the part not explained by the whole market moving |
| Lag k | The leader's move on day t is compared with the target's move on trading day t+k. Always trading days, never calendar days |
| Training window | Data the model may learn from for a given decision date. Always strictly before that date |
| Holdout | Time the model never saw while being built. The **locked holdout** is the most recent 12 months, reserved for the final answer (section 7.5) |
| Walk-forward | Repeatedly train on the past and test on the next period, stepping forward through many years |
| Random guessers | Portfolios built by the same rules and costs as the strategy but with random picks. Their spread of outcomes is "luck" |
| p (vs random) | Share of random guessers that did at least as well as the strategy |
| IC | Rank correlation, across stocks on one day, between predicted and actual next-period residual return. Averaged over days |
| FDR | False discovery rate. Controls the share of "discoveries" that are noise when many things are tested |
| bps | Basis points; 1 bp = 0.01%. Costs are given per side (per buy or per sell) |

## 3. Architecture

A Cargo workspace. The layout below is the target. The first version starts with five crates, `c2-data` (including storage), `c2-engine`, `c2-eval`, `c2-cli` and `c2-gui`, and splits out `c2-store` and `c2-update` when the scheduled update job (M3) arrives.

```
correlation2/
  crates/
    c2-data     ingest adapters, cleaning, symbol master, quality scoring, Parquet storage, panel cache
    c2-update   scheduled data refresh and the forward prediction log (headless)
    c2-engine   returns, residuals, lagged correlation, top-K, clustering, signals; CPU and CUDA backends
    c2-eval     walk-forward harness, portfolio simulator, random guessers, statistics, leakage canaries
    c2-store    SQLite ledger (experiments, symbol master, provider status) + Parquet result arrays
    c2-gui      eframe/egui desktop app
    c2-cli      headless commands: update, run, report, holdout
  research/     reference Python experiments; their outputs are acceptance targets
  config/       TOML experiment configs
```

### Rules every crate follows
- **Point-in-time access.** Any code that makes a decision for date D receives a `PitView` that cannot return data dated after D. Section 7.1 has the details.
- **Reproducibility.** A result is identified by (config hash, data snapshot id, code version) and can be re-run bit for bit, apart from GPU floating-point ordering, which stays within stated tolerances.
- **No silent data changes.** Cleaning and repairs are logged with before and after values, and raw downloads are never modified.
- **Responsive GUI.** Compute runs on worker threads with progress, ETA and cancel, and the GUI never blocks.

## 4. Data layer (`c2-data`)

### 4.1 Sources (one adapter trait, `PriceSource`)

| Adapter | Role | Notes |
|---|---|---|
| `StooqZip` | Free baseline and cross-check | Reads the existing `d_*_txt.zip` files directly (US, world, JP, UK, HK, PL, HU, macro). Keeps folder-path labels (exchange, stock/etf). Survivor-only. Since 2026 new downloads need an API key via CAPTCHA, so this adapter imports zips the user downloads manually |
| `MarketParquet` | Recommended primary | US stocks and ETFs from 2000, delisted included, Parquet partitioned by date (`by_date/{asset}_{timeframe}/YYYY-MM-DD.parquet`; columns timestamp, symbol, asset_type, OHLCV). Split-adjusted; whether dividends are adjusted is not stated, see 4.4. $79 once, plus $15/mo to stay current; the last year is free. **Must pass validation (4.6) before use** |
| `EodhdApi`, `SharadarApi`, `NorgateExport` | Optional | Survivorship-free alternatives, behind the same trait |
| `AlphaVantageListing` | Universe history | `LISTING_STATUS` with `state=delisted` and `date=` gives point-in-time listed and delisted symbols. v1's `.env` already has a key |
| `SecEdgar` | Industry codes | Free SIC codes and company names |

Every download is stored untouched under `data/raw/<source>/<download-date>/`. Snapshots are never deleted automatically. Kept Stooq snapshots also build a point-in-time universe over time.

### 4.2 Canonical storage
- **Prices:** Parquet, one file per year and source; columns `date, symbol_id, open, high, low, close, volume, adj_factor, div_cash, source`. Keep unadjusted prices when the source provides them, and compute adjusted series from corporate actions.
- **Symbol master (SQLite):** stable `symbol_id`; ticker history with validity dates; `kind` (common, etf, adr, preferred, warrant, unit, right, test, other); exchange; listing and delisting dates; sector and industry with their source; data-quality flags.
- **Panel cache:** for an analysis window, a dense `f32` matrix [trading days × symbols] aligned to the NYSE trading calendar (generated, not inferred from the data), with NaN for missing values. Memory-mapped from `cache/panels/<hash>.bin` with a small header. Full US history from 2000 is about 6,700 days × 12,000 symbols × 4 B ≈ 320 MB per field.

### 4.3 Cleaning rules (each one logged and switchable)
1. **Drop exchange test tickers** (`^Z[A-Z]ZZT$`, `TEST` variants, `ZXYZ…`). ZVZZT is in Stooq's "nasdaq stocks" folder.
2. **Classify kind** from source labels plus ticker suffix rules (`-W`, `-WS`, `-U`, `-R`, preferred series). By default analysis uses common stocks only.
3. **Repair one-day bad prints.** A move of more than 2× that reverses within a day is blanked; the experiment found 69 since mid-2021. Keep real crashes.
4. **Never forward-fill.** Missing stays missing. Flag stale prices: 3 or more identical closes with zero volume.
5. **Era and precision flags.** Default analysis start is **2005-01-01**: decimalization was in 2001, and before that coverage is thin and survivor-only. Flag series whose share of 0% days exceeds 10% or whose adjusted price is under $1 (quantized returns).
6. **Fix v1's mistakes:**
   - keep hyphenated tickers
   - key instruments by (exchange, ticker)
   - never delete inputs
   - never invent company data

### 4.4 Adjustments and dividends
- **Returns are total-return** (split- and dividend-adjusted). Stooq already is (AAPL 2020-01-02 = 72.72 vs 75.09 split-only).
- **If a source is split-adjusted only,** dividends come from that source's corporate actions or a second source, and total return is computed from them.
- **Benchmarks (SPY, QQQ) are compared on the same basis as the strategy.** A setting chooses total return (default) or price return, and the report states which.

### 4.5 Universe builder
A universe is defined as of each decision date from data on or before that date:
- kind
- minimum price
- minimum median dollar volume over a lookback
- minimum share of days traded
- listing age
- quality flags

With a survivorship-free source, stocks that later delist are included and their delisting return is applied: last price, or −100% if the source says so. The GUI shows universe size over time.

### 4.6 Validation of a new source
Before a source becomes primary, compare it with the Stooq zips on overlapping dates (2000-2025-06-27) for the same symbols:
- close agreement, with \|diff\| < 0.5% on 99% of days after aligning adjustments
- split dates
- dividend handling
- missing-day counts
- delisted-stock coverage

These go out as a report, plus spot checks against the known splits (NVDA 2024-06-10, CMG 2024-06-26, AVGO 2024-07-15, SMCI 2024-10-01, WMT 2024-02-26).

### 4.7 Quality report
For each symbol:
- coverage %
- repairs (with dates)
- share of 0% days
- median dollar volume
- flags
- source disagreements

Plus a decade-by-decade summary, like ANALYSIS.md section 3.

## 5. Scheduled data updates and the forward log (`c2-update`)

### 5.1 The update job
`c2 update` is headless, idempotent and safe to re-run. Order of steps:
1. **Fetch new daily bars** since the last stored date from the configured sources: MarketParquet daily files, EODHD or Sharadar if configured, or a newly dropped Stooq zip in `data/inbox/`.
2. **Refresh listing status** (`AlphaVantageListing`), plus corporate actions and dividends where the source provides them.
3. **Validate the new rows:**
   - schema
   - no future dates
   - no duplicate (symbol, date)
   - price-jump sanity (section 4.3)
   - day-over-day universe size change under 5%, else hold for review
   - cross-source agreement when two sources overlap
   
   Failures quarantine the batch under `data/quarantine/` instead of loading it.
4. **Append** to canonical storage, record a new data snapshot id, and incrementally extend affected panel caches.
5. **Refresh company info** for new or changed symbols through the provider pipeline (section 10), within rate limits.
6. **Write the forward prediction log** (5.2).
7. **Record a run summary** in the ledger: rows added, symbols added/delisted, quarantines, duration, errors. Send a desktop notification (`notify-send`) on failure or quarantine.

### 5.2 Forward prediction log
- **The idea:** a prediction recorded before its outcome is known is the only test that can't have been fitted after the fact.
- **Data must be current first:**
  - The update records the latest bar date it holds. The forward log is written only when that date equals the current trading day.
  - If today's bar hasn't arrived (sources publish "after the close" at varying times), the job retries hourly up to a configured limit, then flags a gap for that day instead of predicting from stale data.
  - Each log row stores the snapshot's last bar date, so scoring can verify it.
- **When:** after each update, for every **tracked experiment** (a frozen config the user marks for forward tracking). The app computes the next trading day's predictions and portfolio from data up to today's close.
- **Where:** an append-only table (`forward_log`: date, experiment id, config hash, data snapshot id, predicted scores, picks, target weights, written_at).
- **Immutability:** rows are never updated; corrections are new rows.
- **Scoring:** on later updates, logged predictions are scored against realized returns. The GUI shows an accumulating "live" $10,000 account per tracked experiment next to SPY, QQQ and a random-guesser band, with the date tracking began.
- **Persistence:** this keeps running for months and becomes the strongest evidence the app produces.

### 5.3 Scheduling
- **Linux:** a systemd user timer installed by `c2 install-timer`: `OnCalendar=Mon..Fri 18:30 America/New_York` plus a catch-up run at login (`Persistent=true`). Runs `c2 update` with logs to the journal and to `logs/update-<date>.log`.
- **The GUI** shows last run time, status and next run, and has "Update now".
- **Missed runs** catch up automatically, because the job always fetches since the last stored date.

## 6. Signal families (`c2-engine`, trait `Signal`)

Every family produces, for each decision date, a score per stock from a `PitView`. All are evaluated by the same harness.

| Family | Idea | Default horizon | Why include |
|---|---|---|---|
| A. Pair lead-lag | v1's idea done right: residual daily returns, lags 1..L in trading days, store top pairs with FDR control against a timeline-shift null, score = Σ r × leader move k days ago | 1 day | Baseline; reproduces the experiment (null so far) |
| B. Cluster lead-lag | Bennett et al. 2022: directed lead-lag network (summed cross-correlation over lags, optionally distance correlation), Hermitian clustering into leader/lagger groups, forecast lagger clusters from smoothed leader returns; 1-year lookback, refit every 2 months | 1–5 days | Published Sharpe 0.62 before costs (2000–2019); test it net of costs on recent data |
| C. Peer momentum | Peers = top n by long-window **same-day** residual correlation (stable; v1's offset-0 r ≈ 0.40); score = peers' past 1-week or 1-month return minus own. Optional peers from sector/industry or shared analyst coverage | 1 week, 1 month | Closest free proxy for connected-firm momentum (Ali & Hirshleifer 2020: 1.68%/month alpha); low turnover |
| D. Size lead-lag | Large-cap basket's move → small caps next day | 1 day | Positive control (training r = +0.20); probably untradable, which is useful to show |
| E. Calibration anomalies | Short-term reversal (1 day, 1 week), 12-1 momentum, low volatility | Varies | Known effects to calibrate against. A new signal that can't beat these adds nothing |

Each family declares its parameters, so the experiment designer can build grids. Every evaluated configuration is recorded (section 7.4).

## 6A. The correlation census (first-version centerpiece)

The owner's position, and the reason v2 exists: the hypothesis can't be judged until every significant lead-lag correlation has been collected, across all time periods and a reasonable variety of time windows. The census does that collection, and the evaluation harness (section 7) then tests predictions built only from census entries found **before** each decision date.

### What it enumerates
- **Universe:** each period's liquid common-stock universe (section 4.5), chosen point-in-time from data in that period.
- **Return windows** (v1's window1/window2): the leader's return over h₁ trading days and the target's over h₂, with non-overlapping sampling on a grid with step max(h₁, h₂). Default horizons {1, 5, 10, 21, 63}. Pairs (h₁, h₂) are all combinations with h₂ ≤ h₁ plus the diagonal.
- **Lags:** in trading days, from 0 to a horizon-dependent maximum (h₂ ≤ 5: up to 63; h₂ = 21: up to 504, i.e. 2 years; h₂ = 63: up to 756). Lag 0 is kept as the co-movement reference.
- **Periods:** rolling windows of {1, 2, 3, 5} years stepped every quarter, from 2005 (or the data start) to the last date. Full history is a separate period.
- **For each (period, h₁, h₂, lag):** one matrix product of the leader matrix with the target matrix over all pairs, plus chance-level matrices from 3–4 timeline-shifted targets (section 7.1). The chance spread is never taken as narrower than the analytic Fisher-z normal (sd 1/√(n−3)).
- **Minimum observations:** a matrix is computed only when the period holds at least 30 non-overlapping observations at that window and lag (configurable). For example, h₂ = 63 in a 1-year period has only 4. Skipped combinations are recorded.

### What it keeps
- **Significant entries.** Fisher z for every pair. Benjamini–Hochberg at q = 0.05 against the empirical chance distribution of the same matrix; entries passing are stored with their q-value.
- **The top K per matrix** regardless of significance (default 200, flagged), so the strongest relationships can always be browsed.
- **A summary row per matrix:**
  - counts above \|z\| = 3, 4 and 5, real vs chance
  - estimated number of true relationships (excess over chance)
  - the lag-structure similarity with the previous period (`research/lag_spectrum.py`)

### Storage
Parquet, partitioned by (h₁, h₂), with rows:

`period_start, period_end, h1, h2, lag, leader_id, target_id, r, n, z, q, sxy, sx, sy, sxx, syy, flags`

Summaries go in the SQLite ledger. Sizes are reported before a run starts, with a cap the user can change.

### Compute
The census is a queue of independent tasks, one per (period length, h₁, h₂, lag batch). Each task walks its periods in date order on one worker, keeping the previous period's full matrix so the period-to-period similarity can be computed; stored top-K entries can't provide that. That makes ~150 tasks, enough to share between devices. GPU and CPU workers pull tasks from the queue at the same time:
- **GPU:** strided-batched SGEMM over lags, then an on-device threshold-and-compact kernel returning only entries above the storage threshold, plus histograms for the chance comparison.
- **CPU:** the same pipeline with `gemm` + `rayon`.

Rough size: about 300 periods × ~15 window pairs × ~60 lags ≈ 270,000 matrix products. That's roughly 1.4 ms each at N ≈ 5,000 on the RTX 5080 (more with chance matrices), so **on the order of 15–30 minutes for the full census on the GPU**, against months for v1's approach. Progress, ETA, pause and resume (completed tasks are recorded) are required.

### Optimizations (owner requirement: use every one that measurably helps)
1. **Additive blocks instead of recomputing windows.**
   - Time is cut into fixed 63-trading-day blocks. For each block, leader/target window pair and lag, compute the raw cross-product matrix plus per-stock sums and sums of squares.
   - Pearson r for any window is exact from summed block statistics, so every rolling 1-, 2-, 3- and 5-year window is a running sum: prefix sums minus prefix sums, memory-bound adds instead of new matrix products. This removes about 44× of the arithmetic.
   - Chance matrices pair leader block b with target block b+s for fixed whole-block offsets, so they are additive too.
   - Returns are scaled by each stock's trailing volatility (known point-in-time) before quantizing, so r is the correlation of volatility-normalized returns. That is a standard choice and is stated in reports.
2. **Precision modes, chosen by measurement.**
   - GPU: FP32 (cuBLAS SGEMM), TF32, FP16/BF16, and INT8 tensor-core GEMM with INT32 accumulation (cuBLASLt).
   - Standardized returns clipped at ±5 sd quantize to INT8 with a scale of 25 per sd. The quantization error is far below the sampling noise of r, and INT32 sums are exact and additive across blocks.
   - Default: the fastest mode whose r differs from an f64 reference by < 2e-3. Every stored entry is recomputed in FP32 or f64.
3. **CPU kernels.**
   - The Core Ultra 9 285K has AVX2, FMA and AVX-VNNI (`vpdpbusd`, u8 × s8 → s32), but not AVX-VNNI-INT8. The CPU INT8 path is a register-blocked micro-kernel written with `core::arch::asm!`, using the +128 offset trick corrected by column sums.
   - It is benchmarked against the `gemm`/`faer` f32 kernel; whichever is faster is used.
   - Work is split across all 24 cores (8 P-cores, 16 E-cores) with work stealing, so slower E-cores don't stall a batch.
4. **Batching and fusion.**
   - All lags of one (block, window pair) go in one batched GEMM.
   - Window returns are built on the device from per-stock cumulative sums.
   - Threshold, histogram and compaction run fused in one pass over each matrix, so full matrices never cross PCIe.
5. **Everything resident.** Prices load from a memory-mapped binary file (no database in the hot path). Block statistics live in GPU memory while in use, sized to free VRAM (the GPU is shared with local LLM servers).

6. **Optional third worker: NPU / integrated GPU.**
   - The CPU's Intel NPU ("AI Boost", architecture 3720, ~13 TOPS INT8) and its integrated Intel GPU can run fixed-shape INT8 matrix products through OpenVINO 2026.4. It is already installed in `~/programming/experiments/npu-test/.venv`; the NPU needs `LD_LIBRARY_PATH=/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu`.
   - That's a few percent of the RTX 5080's INT8 throughput. Add it as a scheduler worker only if a measured run shows ≥ 3% more total census throughput.
   - Block-sized products (64 × N) suit its static shapes.

### Incremental refinement as new data arrives (owner requirement)
- **Fixed history.** Each period's results are final once its data is complete. New data only creates new periods (windows ending in the new data); history is never recomputed.
- **Running sums, not just r.** Each stored entry keeps its sufficient statistics (Σxy, Σx, Σy, Σx², Σy², n), so windows can be extended or merged exactly without touching prices.
- **Daily.** The update job (section 5) adds each new trading day to the running sums of every tracked pair in the current, still-open windows. That's O(pairs), microseconds, and gives an exact, up-to-date r for the live windows.
- **Every block (63 trading days).** A full all-pairs pass over the new block finds pairs that weren't tracked before and closes the finished windows as new census periods.
- **Data revisions.** When a source revises past prices, the snapshot diff identifies the affected blocks, and only those blocks and the windows containing them are recomputed.

### Running alongside normal use (owner requirement)
- **Stop at any time, lose almost nothing.** The census commits progress after every unit (one block-step of one task), writing output atomically (temp file + rename) and recording completed units. Stopping, crashing or rebooting loses at most the unit in flight.
- **Pause and resume:**
  - from the CLI (`c2 census pause|resume|status`, via a control file)
  - from the GUI
  - with Ctrl-C (graceful)
  
  While paused, workers release the GPU entirely (memory freed, context dropped), so games or other GPU apps get it back.
- **Optional idle mode:** run at full speed only after the desktop has been idle for N minutes (GNOME's `org.gnome.Mutter.IdleMonitor`), and pause within seconds of keyboard or mouse activity.
- **Resource caps** (CPU threads, VRAM, GPU on/off) can be changed while running.

### What it answers directly
- **Replication:** for a pair in period P's stored top K, how often is it in the top K again, or at least the same sign with \|z\| ≥ 2, in the next non-overlapping period? Measured against the same rate for chance-matrix entries.
  - It is defined on the stored top-K entries because BH at q = 0.05 will store nothing for most matrices when there is little real structure.
  - That emptiness is itself a result, reported as such.
- **Lag spectrum over time:** for each (window pair, lag), the number of excess real relationships in each period.
- **Then prediction (section 7):** scores built from past census entries only (e.g. pairs that replicated in two or more past periods), simulated with $10,000 against SPY, QQQ and random guessers.

## 7. Evaluation harness (`c2-eval`)

### 7.1 Point-in-time guarantees
- **`PitView` is the only way signals read data.** It wraps the panel and a decision date and exposes rows strictly before the trade's information cutoff. Rows are sliced by index, so out-of-range access is a hard error.
- **The universe for date D** comes from `PitView(D)`.
- **Execution timing is explicit:**
  - `close-to-close`: the signal uses closes through t−1 and trades at the t−1 close. Optimistic, labelled as such.
  - `next-open` (default when open prices exist): the signal uses closes through t−1, trades at the open of t, and holds open to open.
- **Leakage canaries, run in CI and on demand:**
  - a deliberately leaky signal (peeks one day ahead) must show a large, obviously impossible edge
  - a correct signal evaluated on shuffled outcomes must look like random guessing
  
  Both must pass or the harness refuses to report.

### 7.2 Walk-forward
- **Train and refit:** a rolling or expanding training window (default 3 years rolling), refit every R trading days (daily, weekly or monthly), predict horizon H.
- **History covered:** 2006 to the last data date, minus the locked holdout.
- **Reports:**
  - a per-calendar-year table of strategy vs SPY, QQQ and the random-guesser median and p
  - the share of years that beat random guessers at p < 0.1
  - consistency matters more than the total

### 7.3 Portfolio simulator ("given $10,000")
- **Start:** $10,000 cash.
- **Construction:**
  - long-only top-N equal weight (default N = 20)
  - long-only score-weighted
  - long-short market-neutral (diagnostic: isolates the signal from market direction)
- **Rebalancing:** frequency per family horizon, with optional turnover buffer (keep a holding while it stays in the top N + b).
- **Costs:**
  - per-side bps (default runs at 0, 5, 10 and 20)
  - a spread estimate per stock from high-low prices (Corwin–Schultz) as the realistic case
  - optional per-trade minimum
- **Shares:** fractional by default (most brokers support it, and it's what the reference result in 7.6 uses: $500 per name). Whole shares are an option; with $10,000 they leave stocks priced above a position's size unbuyable, so the report lists those exclusions. Cash earns zero.
- **Benchmarks:**
  - SPY buy-and-hold
  - QQQ buy-and-hold
  - equal-weight universe
  - random guessers: same N, candidates, rebalance schedule and cost model with random picks. Default 10,000 runs on the GPU (1,000 on the CPU). Reported as a histogram with the strategy marked, and p = share at least as good

### 7.4 Statistics and the multiple-testing ledger
- **Prediction quality:**
  - hit rate vs 50% (binomial test)
  - daily IC and rank IC, ICIR, t-stat
  - decile spread returns
  - turnover
- **Money:**
  - final value
  - CAGR
  - max drawdown
  - Sharpe with Lo/Opdyke standard errors
  - excess return vs SPY and QQQ with a block-bootstrap confidence interval
- **The ledger:**
  - every configuration ever evaluated (family, params, data snapshot, period) is a ledger row
  - p-values are reported raw and FDR-adjusted across all ledger rows in the same family
  - the Deflated Sharpe Ratio (Bailey & López de Prado) accounts for how many configurations were tried
  - the GUI always shows "tested so far: n configurations"

### 7.5 The locked holdout
- **Locking:** the most recent 12 months of data are locked by default. Walk-forward and experiment runs stop before them.
- **Running it:** "Run final holdout" needs a frozen config and a confirmation, and each run is recorded with a counter in the GUI ("holdout used 3 times"). Reusing the holdout turns it into training data, so the counter is the honesty check.
- **Moving it:** when the update job brings in new data the lock can move forward, and older holdout results are kept with their dates.
- **The headline screen:** $10,000 → final value for the strategy, SPY, QQQ and the random-guesser distribution, at each cost level.

### 7.6 Reference results (acceptance targets)
Reproduce `research/results.txt` with the same settings: Stooq data, the 2021-06-28 / 2024-06-27 / 2025-06-27 split, lags 1 to 5, top 2,000, 20 picks. Restrict to v1's symbol set (the 11,431 symbols in the restored database) so the universe matches. With hyphenated tickers kept, the universe grows slightly (e.g. BRK-B joins), and that run is reported separately.

| Measure | Target | Tolerance |
|---|---|---|
| Universe size | 2,155 | exact |
| Tests | 23,209,350 | exact |
| Stored \|r\| range | 0.142–0.212 | ±0.001 |
| Holdout same-sign share | 52.3% | ±0.3 pt |
| $10k, no costs | $11,561 | ±0.5% |
| SPY / QQQ | $11,361 / $11,432 | ±0.1% |
| Random-guesser p, no costs | ~0.25 | ±0.05 (random seed) |

## 8. Compute engine and performance

### 8.1 Where the time actually goes
- **Plain correlation is no longer the bottleneck.**
  - **CPU** (numpy, 24-core Core Ultra 9 285K): all 11,431 × 11,431 pairs for one lag over 755 days took 0.15–0.21 s (1.0–1.35 TFLOP/s). The 2,155-stock, 5-lag search took 0.2 s.
  - **GPU** (RTX 5080, cuBLAS SGEMM through cudarc, `research/gpu_bench`; checked against a CPU calculation, max error ~1e-9):

| Stocks | Days | GPU per lag | Throughput |
|---|---|---|---|
| 2,155 | 755 | 0.29 ms | 24.5 TFLOP/s |
| 5,000 | 755 | 1.39 ms | 27.2 TFLOP/s |
| 11,431 | 755 | 7.0 ms | 28.1 TFLOP/s |
| 11,431 | 5,000 | 41.9 ms | 31.2 TFLOP/s |

- **What the full test costs.** These are matrix-product times only. Top-K extraction is memory-bound, about 0.1–0.6 ms per matrix, and the panel is uploaded once.

| Workload | GPU | CPU only |
|---|---|---|
| One all-pairs search, 5,000 stocks, 3 years, lags 1–20 | 28 ms | ~0.7 s |
| Walk-forward 2006–2025, weekly refit (~1,000 refits), same search | ~28 s | ~12 min |
| Same, plus 100 chance repeats per refit | ~47 min | ~20 h |
| v1's "everything": 11,431 stocks, 20 years, lags 1–250 | ~10 s | ~5.5 min |
| v1's actual offset range (up to ~14,000 days) | ~10 min | ~5 h |

v1's estimates were years in Ruby and about 6 months in C, for a much smaller job: pairs one at a time, re-read from the database.

- **The real limits now are statistical and memory.** A larger search raises the chance bar (the best \|r\| from luck grows with the number of tries), so searching more makes real effects harder to detect. One full lag matrix is 523 MB, so results must be reduced on the device; they can never all be stored.
- **The heavy work is:**
  - **Chance-level nulls:** thousands of timeline-shifted or block-bootstrapped repeats of each search.
  - **Walk-forward refits:** e.g. weekly refits over 20 years = about 1,000 refits × lags × families.
  - **Parameter grids and families B and C:** clustering needs eigen-decompositions of N × N matrices.
  - **Random guessers:** 10,000 portfolios × 5,000 days × rebalance logic.
  - **Full-universe scaling:** about 5,000 liquid stocks from 2005 with survivorship-free data, and more if less-liquid names are included.

This is where both the GPU and all CPU cores are used.

### 8.2 Kernels
- **Standardized panel:** `Z` [T × N] in `f32`, column-major. Means and variances accumulate in `f64`.
- **Lagged cross-correlation:** `C_k = Z[0:T−k]ᵀ · Z[k:T] / (T−k)` as SGEMM. All lags in one strided-batched GEMM.
- **Rolling windows:** recompute per refit by GEMM (simplest, and fast enough on the GPU). Incremental rank-k updates are an optimization only if profiling says so.
- **Top-K and threshold extraction on the device:**
  - Each 11k × 11k `f32` matrix is 523 MB, so matrices are never copied back whole.
  - The kernel does a per-row top-k, then a global merge, plus counts above thresholds for FDR.
  - Tile by column blocks when N × N × lags exceeds VRAM.
- **Nulls:** circular time shifts and stationary block bootstrap indices generated on the device. The same GEMM and top-K pipeline runs in batches.
- **Clustering (family B):** Hermitian adjacency from antisymmetric flow, top eigenvectors (cuSOLVER `heevd` on the GPU, `faer` on the CPU), k-means on the eigenvectors.
- **Simulator:** random-guesser portfolios with one GPU thread block per batch of runs. Picks via per-run random keys and partial sort. Costs from turnover masks.

### 8.3 Backends
- **Backend trait:** one implementation per backend; `CpuBackend` is the reference implementation.
- **`CpuBackend`:** `faer` GEMM and eigensolvers, `rayon` across all 24 cores. Always available.
- **`CudaBackend`:**
  - `cudarc` 0.19.x (CUDA 13.0–13.4, runtime library loading): cuBLAS SGEMM / strided batched, cuSOLVER, cuRAND, and NVRTC-compiled custom kernels for top-K, nulls and the simulator.
  - The RTX 5080 is compute capability 12.0 (Blackwell), which needs CUDA ≥ 12.8.
- **Using both at once:**
  - A scheduler splits batched work (null repeats, walk-forward refits, guesser batches) between the GPU and a CPU worker pool, sized by measured throughput, so neither idles.
  - CPU threads also do ingest, panel building, statistics and reporting while the GPU computes.
  - Transfers use pinned memory with double buffering.
- **Correctness:** CPU and GPU results must agree (\|Δr\| < 1e-4; identical top-K sets except ties at the boundary). A `c2 selftest` command checks this on startup in debug builds.
- **Portability later:** a `CubeclBackend` (Burn's kernel layer: CUDA/ROCm/Metal/WGSL), only if needed. The backend trait keeps that door open.

### 8.4 CUDA toolchain on this machine
- CUDA libraries currently exist only inside `~/model-lab/qwen3.8-flash-next/cuda-env/lib` (cuBLAS 13.4.1, CUDA runtime 13.2.86); there is no system CUDA and no `nvcc` on PATH. The driver is 595.99.
- **Proven working** (`research/gpu_bench`):
  - `cudarc = { version = "0.19", default-features = false, features = ["std", "cublas", "driver", "dynamic-loading", "cuda-13020"] }`
  - run with `LD_LIBRARY_PATH` pointing at that directory
  - in 0.19 the host/device copies are `clone_htod` / `clone_dtoh`
- **Possible 2× speedup, unmeasured:** TF32 tensor-core math for SGEMM. Its ~1e-3 relative precision is fine for screening, with the top candidates recomputed in FP32.
- **Setup options:** install the CUDA 13 runtime system-wide, or have the launcher set `LD_LIBRARY_PATH` to a configured CUDA library directory.
- **At startup:** the app probes for CUDA. If it's missing, it runs on the CPU backend with a visible banner.
- **GPU sharing:** the GPU is shared with local LLM servers (Strata on :8080 holds VRAM when loaded), so the engine checks free VRAM before allocating and tiles accordingly.
- **Toolchain:** Rust 1.90 is installed. Pin the MSRV to what egui 0.36 and cudarc 0.19 require, and check both.

### 8.5 Performance targets (RTX 5080, 24-core CPU)

| Workload | Target |
|---|---|
| 5,000 stocks × 5 lags × 3 years, one search | < 50 ms GPU, < 1 s CPU |
| Same search + 1,000-repeat chance null | < 1 min |
| Walk-forward 2006–2025, weekly refit, one family-A config | < 5 min |
| 10,000 random guessers × 250 days | < 10 s |
| Load panel cache for 2005–today | < 2 s |
| `c2 update` daily run, excluding network | < 30 s |

## 9. GUI (`c2-gui`)

Built with eframe/egui 0.36 (wgpu renderer) and `egui_plot`. One native window with a dark theme, compute on background threads, and a status bar showing data freshness, backend (GPU/CPU) and running jobs.

| Screen | Contents |
|---|---|
| **Dashboard** | Stat tiles (symbols, data range, last update, tracked experiments, holdout status); live forward-log accounts vs SPY/QQQ; recent ledger entries |
| **Data** | Sources and snapshots; update history and quarantines; "Update now"; symbol master browser; per-symbol quality report with a price chart marking repairs and gaps; decade quality table |
| **Universe** | Filter builder with a live count over time |
| **Explore** | Stock page: prices, returns, company info, peers (strongest same-day correlations), lead-lag partners of **both signs**. Pair viewer: scatter of leader(t) vs target(t+k), rolling correlation with training/holdout shading. Correlation heatmap: clustered ordering, downsampled to a texture with zoom. Cluster/network view for family B |
| **Experiment** | Pick a family, parameters (single or grid), walk-forward and simulator settings, then run with progress, ETA, cancel, and GPU/CPU utilization |
| **Results** | Report per run (equity curve vs SPY/QQQ, random-guesser histogram with the strategy marked, per-year table, IC over time, decile returns, cost sensitivity curve); ledger table with raw and FDR-adjusted p and configs tested; side-by-side compare |
| **Holdout** | The locked year: run-final-holdout with confirmation and usage counter; the $10,000 headline at each cost level |
| **Forward log** | Tracked experiments, their live accounts since tracking began, daily picks, scored outcomes |
| **Learn** | v1's explainers rebuilt on real data (draggable correlation scatter, window/offset sliders), plus two new ones: how the best \|r\| from chance grows with the number of tests, and what leakage does to a backtest |

## 10. Company info (from v1, fixed)
- **Providers** behind one adapter: FMP, Alpha Vantage, Finnhub. Drop IEX Cloud, which no longer operates. SEC EDGAR for SIC codes.
- **Status table:** per (symbol, provider): found, not_found, rate_limited, error, last_checked.
- **Fallthrough:** a `not_found` falls through to the next provider (v1 skipped them), and empty results are cached as not_found with a retry interval.
- **Rate limits** are enforced with a token bucket per provider. Daily quotas are respected, and a run resumes later.
- **Never store guessed data.** Unknown stays unknown.
- **Corrected mappings:** e.g. Finnhub employees.
- **Keys:** in headers where the API supports it; stored in the OS keyring or `.env`, never in the ledger.

## 11. Storage and configuration
- **SQLite (`c2.db`):**
  - symbol master
  - provider status
  - experiment ledger
  - holdout usage
  - forward log
  - update runs
- **Parquet:** prices, panels' source data, and large result arrays (score matrices, equity curves).
- **TOML:** experiment configs. A config plus a data snapshot id fully determines a run.
- **Data directory:** `~/.local/share/correlation2/` by default (raw, quarantine, canonical, cache, logs), configurable. Price stores live in its `stores/<name>/`, one per ingest (`stooq-<last trading day>`), with `stores/CURRENT` naming the default and `--store`/`C2_STORE` choosing another; census results and outside data stay in the data directory and can be read against any store (since 2026-10-20).

## 12. Milestones

Each milestone ends with a number that can be checked.

| # | Milestone | Done when |
|---|---|---|
| M0 | Workspace and toolchain | Workspace builds; `c2 selftest` runs SGEMM on CPU and GPU and they agree; CUDA probe works with and without the libraries. Head start: `research/gpu_bench` already does the GPU half |
| M1 | Data layer from Stooq zips | Symbol master with kinds from the US zip: 8,523 stock files (4,756 NASDAQ, 3,451 NYSE, 316 NYSE American) and 3,027 ETF files, hyphenated tickers kept, ZVZZT dropped; bad prints since 2021-06 repaired (69 on v1's symbol set); decade quality table matches ANALYSIS.md |
| M2 | CPU engine + simulator + random guessers | Reproduces section 7.6 within tolerance |
| M2a | Correlation census (section 6A), CPU + GPU together | Full census on the Stooq data with both backends running; resumable; replication and lag-spectrum summaries; the daily-lag rows agree with `research/lag_spectrum.txt` |
| M3 | Scheduled updates | `c2 update` + systemd timer; MarketParquet adapter validated against Stooq (4.6); data current through the latest trading day; quarantine path tested with a corrupted file |
| M4 | CUDA backend | Matches M2 within tolerance; section 8.5 targets measured and recorded |
| M5 | Walk-forward + ledger + canaries | Per-year table 2006 onward for family A; leakage canaries pass in CI; FDR-adjusted p in reports |
| M6 | Families B–E | Each evaluated walk-forward and in the ledger, with the calibration anomalies as reference rows |
| M7 | GUI | All screens in section 9 working on real data |
| M8 | Forward log | Tracked experiments logging daily via the timer; live accounts scored |
| M9 | Final answer | Locked holdout on the most recent 12 months (survivorship-free data, current through the last update) run once per frozen config; report vs SPY, QQQ and random guessers at 0/5/20 bps |

**Decision gates:**
- **After M2:** confirm family A's null result on the new data source before going further.
- **After M6:** if no family beats random guessers at p < 0.1 after 5 bps costs in most walk-forward years, stop or pivot. A pivot would mean risk and diversification tooling built on same-day correlation, before building GUI polish.

## 13. Keep / drop from v1

| v1 | v2 |
|---|---|
| Stooq bulk import | Keep as an adapter; keep path labels; never delete zips |
| Rails + Postgres | Drop. Parquet + SQLite, single native binary |
| Pairwise loop with DB per offset | Drop. Batched GEMM on panels |
| Overlapping N-day % changes, calendar-day offsets | Drop. Non-overlapping residual returns, trading-day lags |
| \|r\| ≥ 0.3 significance | Drop. Chance nulls, FDR, holdout, random guessers |
| Multi-provider company info + status table | Keep, with the fixes in section 10 |
| Dashboard tiles, charts, stock drill-down | Keep, in egui |
| Interactive explainers | Keep, on real data, plus multiple-testing and leakage explainers |
| Inactive marking with dry-run | Keep as quality flags in the symbol master |
| Rotating DB backups | Replace with immutable raw snapshots + reproducible derived data |
| Login, admin users, web docs | Drop (single-user desktop) |

## 14. Risks and open questions
- **Data:**
  - MarketParquet is new and doesn't name its upstream source, so validation (4.6) is mandatory.
  - Dividend handling differs by source.
  - Free survivorship-free history is limited.
- **Overfitting by iteration:** the ledger, FDR and the holdout counter reduce it but can't eliminate it. The forward log is the real defense.
- **Cost realism:** small-cap spreads can be far above 20 bps, so the Corwin–Schultz estimate should be the default for "realistic".
- **Regime change:** lead-lag relationships last about a year (Xia et al., cited by Bennett et al.), so long training windows may hurt; walk-forward should test window lengths.
- **GPU environment:** CUDA library location, VRAM shared with LLM servers. The CPU fallback must always work.
- **Scope:** the GUI is large. Build it after M2–M6 show whether there is anything worth exploring.
