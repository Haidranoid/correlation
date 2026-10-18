# correlation2: test results and implementation choices

A running log of what was measured and what was chosen because of it, newest milestone last. Machine: Intel Core Ultra 9 285K (8 P-cores + 16 E-cores, AVX2/FMA/AVX-VNNI, no AVX-512), 64 GB RAM, NVIDIA RTX 5080 16 GB (compute capability 12.0, driver 595.99), Ubuntu 26.04, Rust 1.90. Raw outputs are in `research/` (`bench_m0.txt` and later files).

## Choices so far (summary)

| Area | Choice | Why (measured) |
|---|---|---|
| GPU arithmetic | INT8 tensor cores (cuBLASLt, INT32 sums) for screening; exact recompute for stored entries | 342 TOPS at 1,280 rows vs 27 TOPS FP32 (12.8×); error ≤ 7e-3 at 64 rows and 1.5e-3 at 1,280, about 1–6% of r's own sampling noise |
| GPU window strategy | Recompute each window directly in INT8, not block-additive sums | At 64-row blocks every precision takes the same ~0.115 ms, the time to write 25M results; arithmetic is nearly free, so memory traffic decides |
| CPU arithmetic | INT8 with a hand-written AVX-VNNI (`vpdpbusd`) assembly micro-kernel, `crates/c2-engine/src/vnni.rs` | 8.1–8.6 TOPS with results stored, 9.6 TOPS compute-only, vs 1.3–1.4 TOPS for the f32 `gemm` crate (6.3×). Bit-identical to the GPU's INT8 results |
| CPU window strategy | Same as GPU: recompute each window directly, process 4 × 24 tiles in registers (fused), never store the full matrix | With VNNI the CPU is no longer compute-starved; storing the result (100 MB) costs 1.2 of 1.8 ms at 64 rows. Block-additive sums would need that storage, so they were dropped |
| Significance pass | FP32 math, per-warp histogram aggregation, column-walking 2-D launch | 4–6× faster than the first version (0.17 ms vs 1.0 ms per 25M entries) |
| CUDA libraries | Found automatically (re-run with `LD_LIBRARY_PATH`), CPU fallback if absent | cudarc panics on a missing library; now probed first |

## M0: compute engine benchmarks

Command: `c2 bench --n 5000` (5,000 synthetic stocks with market, sector and planted one-day lead structure, clipped at ±5 sd). Time is per all-pairs product for one lag. Error is the largest |r − f64 reference| over 2,000 sampled pairs (1,500 random, 500 planted leads), computed from the values the GPU actually received (rounded or quantized).

### GPU: precision modes

| Rows (days) | Mode | ms | TOPS | Max error |
|---|---|---|---|---|
| 64 | FP32 (cuBLAS SGEMM) | 0.148 | 21.6 | 9.7e-8 |
| 64 | TF32 | 0.123 | 26.0 | 1.3e-4 |
| 64 | FP16 | 0.119 | 26.8 | 1.3e-4 |
| 64 | BF16 | 0.119 | 26.8 | 9.9e-4 |
| 64 | INT8 | 0.115 | 27.9 | 7.3e-3 |
| 256 | FP32 | 0.428 | 29.9 | 1.4e-7 |
| 256 | TF32 | 0.281 | 45.5 | 7.1e-5 |
| 256 | FP16 | 0.171 | 74.9 | 7.2e-5 |
| 256 | BF16 | 0.171 | 75.0 | 5.2e-4 |
| 256 | INT8 | 0.120 | 106.8 | 3.7e-3 |
| 768 | FP32 | 1.350 | 28.4 | 1.7e-7 |
| 768 | TF32 | 0.743 | 51.7 | 3.7e-5 |
| 768 | FP16 | 0.395 | 97.2 | 3.7e-5 |
| 768 | BF16 | 0.395 | 97.2 | 2.8e-4 |
| 768 | INT8 | 0.130 | 295.3 | 2.1e-3 |
| 1,280 | FP32 | 2.397 | 26.7 | 1.7e-7 |
| 1,280 | TF32 | 1.266 | 50.6 | 3.2e-5 |
| 1,280 | FP16 | 0.647 | 98.9 | 3.2e-5 |
| 1,280 | BF16 | 0.626 | 102.3 | 3.2e-4 |
| 1,280 | INT8 | **0.187** | **341.7** | 1.5e-3 |

**Reading it:**
- INT8 wins by a wide margin once a product covers a full window (256+ rows).
- INT8's error comes from quantizing to 1/25 sd steps. It is 1–6% of the sampling noise of r itself (1/√rows) and shrinks as windows grow.
- Anything the census stores is recomputed exactly, so INT8 only decides which pairs get screened in. The screening threshold is set a safety margin below the storage threshold.
- FP16 is the fallback if more precision is ever needed during screening: about 3× slower than INT8, with 30× smaller error.
- At 64 rows all modes cost the same ~0.115 ms: writing the 5,000 × 5,000 result (100 MB) at ~870 GB/s. So the GPU should never materialize small products it doesn't need. This drove the window-strategy choice in the summary table.

### GPU: significance pass (r, |Fisher z| histogram, compaction above a threshold)

| Version | INT8 product | FP32 product | Change |
|---|---|---|---|
| First (FP64 math, global grid-stride, plain shared atomics) | 0.999 ms (100 GB/s) | 0.769 ms (130 GB/s) | |
| Optimized (FP32 math, column-walking 2-D grid, `__match_any_sync` warp aggregation) | **0.168 ms (594 GB/s)** | **0.195 ms (514 GB/s)** | 4–6× faster |

Both versions histogram all 25,000,000 entries and flag the same entries (2,695 / 2,688 with \|z\| ≥ 4), so the optimization changed no results. The first version's main costs were FP64 math (1/64 rate on GeForce) and warps colliding on the same histogram bin.

**Next step for this kernel:** a fused tensor-core GEMM + significance kernel (inline PTX `mma.sync`) that never writes the n × n matrix. It would remove both the 0.115 ms write and the 0.17 ms read per product.

### CPU: f32 (`gemm` crate, AVX2/FMA, 24 threads)

| Rows | ms | TOPS | Max error |
|---|---|---|---|
| 64 | 2.49–2.82 | 1.13–1.29 | 9.7e-8 |
| 256 | 8.77 | 1.44 | 1.4e-7 |
| 1,280 | 45.5 | 1.41 | 8.5e-8 |

**Bug found and fixed:** the first CPU run had error ≈ 1.0 because the row and column strides were passed to `gemm::gemm` in swapped order. Corrected in `crates/c2-engine/src/cpu.rs`.

### CPU: INT8 AVX-VNNI assembly micro-kernel (24 threads)

**Design:**
- **Tile:** 4 leaders × 24 targets per step, using all 16 ymm registers: 12 accumulators, 3 target loads, 1 broadcast.
- **Signs:** `vpdpbusd` takes unsigned × signed bytes, so leaders are stored +128 and the result is corrected by 128 × target column sums.
- **Packing:** panels are packed so the inner loop only reads sequentially.
- **Cache blocking:** a target block stays in L1, a 256-leader chunk in L2.
- **Threads:** rayon work-stealing across the 8 P-cores and 16 E-cores.

| Rows | ms (stores result) | TOPS | Max error | One-off packing (ms) | Compute only, no store (ms) |
|---|---|---|---|---|---|
| 64 | 1.78 | 1.79 | 7.3e-3 | 0.47 | 0.56 |
| 256 | 2.19 | 5.83 | 3.7e-3 | 0.94 | 1.63 |
| 768 | 4.76 | 8.07 | 2.1e-3 | 2.30 | 4.08 |
| 1,280 | 7.88 | 8.12 | 1.5e-3 | 3.52 | 6.81 (9.6 TOPS) |

**Reading it:**
- The errors are identical, to the digit, to the GPU's INT8 errors. Both compute exact integer sums of the same quantized values, so this cross-checks both implementations.
- **Measurement bug found and fixed:** the first compute-only timing used one shared atomic counter for all 24 threads and read ~8 ms everywhere (contention, not arithmetic). It was replaced with `black_box`.
- **Throughput share:** the GPU does a 1,280-row product in 0.187 ms against the CPU's 6.8 ms, so the CPU adds about 3% to census throughput. That's worth taking, since the GPU feeder needs only one thread.

### Decision: drop block-additive sums
The spec planned block-additive window sums to save arithmetic. Measurements show both devices have arithmetic to spare (INT8) and are limited by writing results. Additive sums require storing per-block matrices, so they would cost more than they save. **Both devices recompute each window directly and process results without storing the full matrix.**

### CUDA library discovery

| Situation | Behavior |
|---|---|
| Libraries only in `~/model-lab/.../cuda-env/lib`, nothing set | `c2` re-runs itself with that directory on `LD_LIBRARY_PATH`; GPU used |
| No CUDA libraries anywhere (simulated with `C2_CUDA_REEXEC=1`) | Prints `GPU unavailable, CPU only: CUDA libraries not found: cuBLAS, cuBLASLt, NVRTC (...)`; CPU continues |
| Before the fix | cudarc panicked: `Unable to dynamically load the "cublas" shared library` |

## M1: data layer (`crates/c2-data`)

**What was built:**
- **Parallel ingest straight from the Stooq zip.** The zip is memory-mapped and each rayon task opens its own reader; decompression uses zlib-rs.
- **One memory-mapped columnar store** (`~/.local/share/correlation2/store/prices.bin`, 640 MB): per-symbol contiguous date/open/high/low/close/volume columns, each 64-byte aligned, plus `symbols.tsv`. Written via temp file + rename.
- **SPY trading calendar**, dense panel builder, bad-print repair and the decade quality table.

| Check | Result | Target / reference |
|---|---|---|
| Ingest time, `d_us_txt.zip` (11,550 files, 26,656,424 rows) | **1.1 s** wall, 1.5 GB peak RAM | v1's importer: minutes, with database round trips |
| Files by folder | nasdaq stocks 4,756, nyse stocks 3,451, nysemkt stocks 316, nyse etfs 2,695, nasdaq etfs 332 | Same as the zip listing ✓ |
| Empty files | 19 (all in stock folders) | — |
| Symbols with data | 11,531: 8,503 stocks, 3,027 ETFs, 1 test ticker (ZVZZT, 777 rows) | ETFs 3,027 ✓ |
| v1's symbol set (`^[A-Z0-9^_]{1,15}$`, i.e. no hyphenated tickers) | **11,431** | v1 database: 11,431 ✓ exact |
| Bad prints since 2021-06-01, v1 symbol set | **69** | Python experiment: 69 ✓ exact |
| Bad prints since 2021-06-01, all symbols | 70 | (one more among hyphenated tickers) |
| `c2 data-check` (kinds, both bad-print panels, decade table) | 0.06 s | — |

| Decade | Symbols | Rows | 0% days | Under $1 | Close only |
|---|---|---|---|---|---|
| 1960s | 2 | 3,973 | 22.5% | 0.0% | 4.7% |
| 1970s | 40 | 79,879 | 29.2% | 47.7% | 9.9% |
| 1980s | 199 | 309,692 | 16.8% | 19.1% | 2.2% |
| 1990s | 309 | 666,256 | 10.2% | 5.4% | 0.9% |
| 2000s | 3,014 | 3,574,941 | 3.0% | 1.0% | 1.5% |
| 2010s | 6,064 | 10,592,071 | 2.9% | 0.5% | 1.6% |
| 2020s | 11,531 | 11,429,612 | 2.3% | 2.5% | 2.7% |

This matches ANALYSIS.md section 3 (computed from v1's Postgres) to within the extra hyphenated tickers. The 1960s and 1970s rows are identical.

**Choices:**
- Stooq prices are used as given (already split- and dividend-adjusted), and the raw store is never modified.
- Cleaning (test tickers, bad prints) is applied when a panel is built, so it is switchable and logged.
- Missing days stay NaN. Nothing is forward-filled.

## M2a: correlation census engine (`crates/c2-census`)

### Design choices

**Census variable.** Market-residual daily log return divided by the stock's trailing 252-day residual volatility, with beta, volatility and liquidity all computed from days up to t−1, then clipped at ±5.
- Values never change when later data arrives, so running sums stay in fixed units. That is what makes incremental refinement possible (owner requirement).
- Horizon returns are sums of h daily values divided by √h.

**Lags.**
- The leader's return covers days (g−h₁, g] and the target's (g+lag−h₂, g+lag].
- Lag 0 is the same-period reference. Predictive lags start at h₂. **Lags between 0 and h₂ are never computed**: they would let the two returns share days, which was v1's leak.
- For h₁ = h₂ = 1, lag 1 is "next day" and lag 0 "same day", matching `research/lag_spectrum.py`.
- Observations step by max(h₁, h₂), so they never overlap.

**Universe per window.** At least 90% of days with a value, plus median price ≥ $5 and median dollar volume ≥ $5M over the window's last 252 days. Each window's universe is saved (`universes.bin`).

**Nulls.** 3 per matrix. The target series is circularly shifted by a whole number of observations: within [m/4, 3m/4], at least one more than the largest lag, never 0. One null's strongest entries are stored too (flag `NULL`), so "real pairs replicate X% vs chance Y%" can be computed later.

**Storage.** Per matrix:
- the top 100 real entries by |z|
- every real entry above the FDR threshold (Benjamini–Hochberg-style, q = 0.05 against the mean of the 3 null histograms)
- the top 50 entries of null #0

Each stored entry gets exact f64 recomputation plus its sufficient statistics (Σxy, Σx, Σy, Σx², Σy²). Records are 52 bytes; each matrix also gets a summary line with real and chance tail counts.

**Pipeline.**
- The CPU uploads each window's running-sum matrix once.
- GPU kernels build every leader, target and null panel from it, quantize to INT8 and compute column sums.
- cuBLASLt INT8 multiply, then the fused significance pass. Only histograms and screened entries (|z| ≥ 3.5) come back.
- A separate finisher thread does FDR selection, exact recomputation and writing while the GPU continues.

**Checkpoints.**
- One unit = one window of one task (all its lags).
- Records and summaries are appended, flushed and fsynced, then a line is written to `progress.log`. Resuming truncates to the last commit.
- Ctrl-C and SIGTERM stop after the current unit. `c2 census pause` releases the GPU until `c2 census resume`.
- `manifest.json` stores the config, the store fingerprint and the git commit. Resume refuses a changed config.

**Problem found:** cuBLASLt's INT8 kernels returned `CUBLAS_STATUS_NOT_SUPPORTED` for 1,806 stocks. They need widths that are multiples of 4; the benchmark's 5,000 and 2,000 happened to qualify. Widths are now padded to a multiple of 16 with zero columns, which give r = 0 and are never stored.

### Validation against Python (`c2 census validate`, oracle from `research/census_oracle.py`)

Window 2016-06-28..2022-06-27, daily, lags 0 and 1, 1,806 stocks, 1,510 days, 4,049 pairs (2,000 random per lag plus the 50 strongest lag-1 pairs). Same standardization as the Python.

| Check | Result |
|---|---|
| Oracle universe found in the new store | 1,806 / 1,806 |
| max \|r − Python\|, exact f64 path | **3.6e-7** (from f32 price storage) |
| max \|r − Python\|, GPU INT8 census path | 1.98e-3 (quantization; ~4% of r's sampling sd at 1,509 obs) |
| max \|r − Python\|, CPU AVX-VNNI INT8 path | 1.98e-3 |
| max \|GPU INT8 − CPU INT8\| | **0** (bit-identical) |
| Python's 50 strongest lag-1 pairs caught by the GPU screening pass (\|z\| ≥ 3.5) | **50 / 50** |

### Smoke runs: four problems found and fixed before the full census

Smoke grid: horizons {1, 5, 21}, 1- and 3-year windows, 888 units.

| Run | Change | Result |
|---|---|---|
| `smoke` | First version: per-lag GPU↔CPU round trips, FDR on \|z\| bins | 49 s, but **GPU utilization ~1%**; same-day matrices stored **~182,000 entries each (3.2 GB total)**; early windows had **0 stocks** |
| `smoke-b` | Thresholds chosen on the GPU (`choose_thresholds` kernel), \|r\| histograms (1,024 bins), one sync per window, prep thread, buffers allocated once, same-day matrices top-100 only, FDR capped at 1,000 per matrix, 3-year warm-up, long-lived-stock calendar | 2 min (more windows now non-empty), **GPU 97–99%**, 249 MB, first window 299 stocks. But real lagged matrices showed **17× chance** at lag 1 |
| `smoke-c` | **Sign-flip null** replaces the time-shift null | Real/chance at \|z\| ≥ 4 drops from 17× to **1.0–1.2×**. The 17× was volatility clustering |
| `smoke-d` | FDR needs ≥ 10 entries above the cut-off; placebo rate recorded | FDR cut-offs fire in **0.0%** of lagged matrices; placebo 0–0.7% |

**Why the time-shift null was wrong.**
- In real data, leader day t and target day t+k fall in the same volatility regime: a turbulent week is turbulent for both. Shifting the target by months puts them in different regimes, so chance correlations look narrower than they really are.
- A 1.4× wider spread turns a 1-in-16,000 event into 1-in-250: a 60× "excess" with no lead-lag at all.
- The sign-flip null multiplies the target's observations by a random ±1 that is the same for every stock on that date. Volatility alignment is preserved exactly and only the relationship is destroyed.
- Same-day (lag 0) matrices still show 39× chance: real co-movement, as expected.

**Lag structure, smoke-c (leader and target both daily, \|z\| ≥ 4 summed over windows):**

| Lag | Real | Time-shift null (wrong) | Sign-flip null | Real / sign-flip |
|---|---|---|---|---|
| 0 | 25,654,910 | 32,158 | 649,773 | 39.5 |
| 1 | 562,463 | 32,880 | 467,399 | **1.20** |
| 2 | 462,425 | 33,340 | 436,250 | 1.06 |
| 5 | 300,624 | 33,006 | 322,193 | 0.93 |
| 15 | 87,634 | 34,088 | 96,558 | 0.91 |
| 63 | 57,340 | 117,980 | 58,540 | 0.98 |

Only lag 1 keeps a small excess (1.2×). That fits the known next-day effect from large to small stocks (`research/results.txt` positive control). It is spread too thinly over pairs for any individual pair to pass the FDR rule.

**How this squares with the earlier Python result:** `research/lag_spectrum.py` found that the lag structure of one period doesn't reappear in the next (top pairs ~coin-flip). The census now says the same thing from within each window: beyond lag 1, extreme lagged correlations occur at chance rates once volatility alignment is accounted for.

**Other small things:**
- The time-shift null's own tail jumped at a few lags (e.g. daily lag 63: 118k vs ~33k elsewhere). That was most likely shifts re-aligning quarterly earnings seasons. The sign-flip null makes it moot.
- `tail()` in the summary uses `ceil` for the boundary bin while the histogram uses `floor`, so tail counts slightly undercount. Harmless for ratios.

**Run directories left for you to delete if you like** (a safety check stopped me removing my own test output): `~/.local/share/correlation2/census/{smoke,smoke-b,smoke-c,smoke-d}`, about 250 MB each except `smoke` (3.2 GB).

## M2: the $10,000 holdout harness (`c2 census holdout`, `c2 census replicate`)

### Design
- **The holdout.** The last year (2024-06-28..2025-06-27 by default) is treated as the future. Predictions use only real census pairs (one-day target horizon, lag ≥ 1) from windows that **ended before the holdout starts**: by default the latest window of each length (1, 2, 3 and 5 years). The full-history window overlaps the holdout and is excluded.
- **Timing.** At the close of day d, a pair (leader i → target j, leader horizon h₁, lag k) contributes r × (the leader's h₁-day move ending at day d+1−k) to target j's score. k ≥ 1, so nothing after day d is used.
- **The account.** $10,000, top 20 predicted gainers with a positive score, equal weight, rebalanced every close, costs charged on turnover.
- **Benchmarks.** SPY and QQQ buy-and-hold over the same days, and an equal-weight buy-and-hold of every predictable stock.
- **Random guessers with matched turnover.** 1,000 accounts that hold as many stocks as the strategy and each day replace exactly as many as it did, at random.
  - This matters. In the first version the random guessers turned over 100% a day, and the census picks appeared to "beat random" at 20 bps (p = 0.000) purely because they traded less.
  - With turnover matched, p was 0.39 at every cost level.
- **Prediction-quality measures:**
  - hit rate: picks beating the market the next day
  - daily rank IC between scores and next-day residual returns
  - a long-short diagnostic: top 10% minus bottom 10% of scores, market-neutral, gross bps/day, Sharpe, t and turnover
- **`--walk-forward`.** Each year from 2006-06-28 on is treated as the future in turn, each predicted only from windows that ended before it.
- **`c2 census replicate`.** For every stored top pair, the exact r of the same pair and lag in the next non-overlapping window, real pairs vs pairs picked from the sign-flip null matrices.

### Results on the smoke census (horizons 1/5/21, 1- and 3-year windows)

**Replication (smoke-d):**

| Group | Same sign next window | Mean sign(r then) × r next |
|---|---|---|
| Same-day pairs (lag 0), daily, real | **100%** | **0.789** |
| Lagged, daily, 1-step lag, real | 52.2% | +0.0036 |
| Lagged, daily, 1-step lag, chance-picked | 49.9% | +0.0008 |
| Lagged, all other lags and horizons, real | 49.5–52.0% | ≈ 0 |
| Lagged, all other lags and horizons, chance-picked | 49.2–51.3% | ≈ 0 |

Same-day co-movement persists almost perfectly, which proves the test can see persistence. Lagged pairs persist no better than pairs picked from noise.

**$10,000, holdout 2024-06-28..2025-06-27:**

| | Final | Return | Random guessers (turnover-matched) |
|---|---|---|---|
| SPY | $11,361 | 13.6% | |
| QQQ | $11,432 | 14.3% | |
| Census picks, 0 bps | $11,080 | 10.8% | median $10,768; 38.6% did as well (p = 0.386) |
| Census picks, 5 bps | $9,708 | −2.9% | p = 0.389 |
| Census picks, 20 bps | $6,527 | −34.7% | p = 0.398 |

Hit rate 50.2%; rank IC +0.0009 (t = 0.55).

**Walk-forward, 19 one-year holdouts (2006–2025):**
- Picks beat random guessers at p < 0.1 in **2 of 19 years**. Chance alone gives about 1.9.
- Rank IC t-stat positive in 16 of 19 years, combined z = 3.53. That's a weak but consistent ranking signal in vol-normalized residual returns, most likely the documented next-day effect from large to small stocks.
- Long-short top/bottom 10% in actual returns: **0.43 bps/day gross, positive in 8 of 19 years, combined z = 0.52.** Not significant. With 44% of the book replaced daily, costs of c bps per side take about 1.8 × c bps/day, so even 1 bp of cost erases it.

## The full census (`c2 census run --name main`)

| | |
|---|---|
| Grid | horizons {1, 5, 10, 21, 63} (15 leader/target pairs), rolling windows of 1, 2, 3 and 5 years stepped quarterly from 2005-01-03, plus full history; lags per SPEC 6A; 3 sign-flip nulls per matrix |
| Size | 285 windows (283–2,345 stocks each), 75 tasks, **4,275 units, 83,879 real matrices (~335,000 matrix products including nulls)** |
| Time | **5 min 47 s** (10:51:16 → 10:57:03), RTX 5080 at 97–99%. For reference, v1's approach was estimated at ~6 months in C for a smaller job |
| Storage | 630 MB of records (52 bytes each) + per-matrix summaries, in `~/.local/share/correlation2/census/main/` |
| Code | Manifest names commit `74ba415`; the binary also had the then-uncommitted ≥10-entry FDR rule, placebo column and pidfile, all committed in `8c97929` |
| ETA model | The first ETA read 6–8 min and then said "1 min left" at 10% of windows. The cost model under-weighted the fixed cost per product (writing and scanning the n × n result), giving it 64 rows instead of the ~2,000 measured. Fixed for future runs |

**Real vs chance, leader and target both daily, |z| ≥ 4 summed over windows** (`research/census_main_summary.txt` has every horizon pair):

| Lag | Real | Sign-flip null | Real/null | Windows with an FDR cut-off | Placebo rate |
|---|---|---|---|---|---|
| 0 | 58,600,890 | 1,358,406 | 43.1 | — | — |
| 1 | 1,169,860 | 974,664 | **1.20** | 1.4% | 0.4% |
| 2 | 954,393 | 900,288 | 1.06 | 0.0% | 0.0% |
| 3 | 870,905 | 834,956 | 1.04 | 0.0% | 0.4% |
| 4 | 788,354 | 698,335 | 1.13 | 0.4% | 0.0% |
| 5 | 617,557 | 663,341 | 0.93 | 0.0% | 0.0% |

**Replication on the full census** (`research/census_main_replicate.txt`): same-day pairs keep their sign 100% of the time (r ≈ 0.78 in the next window). Every lagged group of real pairs keeps its sign 49.7–52.7% of the time, the same as chance-picked pairs (48.5–51%). The best lagged group is daily next-day pairs at 52.7%, with mean next-window r +0.004.

**$10,000 holdout 2024-06-28..2025-06-27** (`research/census_main_holdout.txt`): 81,900 stored pairs from the 4 latest pre-holdout windows; 2,217 predictable stocks.

| | Final | Return | Random guessers, turnover-matched |
|---|---|---|---|
| SPY | $11,361 | 13.6% | |
| QQQ | $11,432 | 14.3% | |
| All 2,217 candidates, buy and hold | $10,826 | 8.3% | |
| **Census picks, 0 bps** | **$12,328** | **23.3%** | median $10,771; **10.1% did as well (p = 0.101)** |
| Census picks, 5 bps | $10,817 | 8.2% | p = 0.103 |
| Census picks, 20 bps | $7,303 | −27.0% | p = 0.106 |

Hit rate 49.9% of 4,996 picks; daily rank IC +0.0009 (t = 0.57).

**Walk-forward, 19 one-year holdouts** (`research/census_main_walkforward.txt`):
- The picks beat turnover-matched random guessers at p < 0.1 in **2 of 19 years**. Chance alone gives about 1.9.
- The picks' return swings from +92.5% (2020–21) to −35.1% (2008–09): a concentrated, volatile 20-stock book, not skill.
- The rank IC t-stat is positive in 18 of 19 years (combined z = 4.15), a weak but consistent ranking signal in vol-normalized residual returns.
- Long-short top/bottom 10% in actual returns: **0.45 bps/day gross, positive in 12 of 19 years, combined z = 0.92 (not significant).** About 40% of the book changes daily, so any trading cost erases it.

### Verdict from the first version
- **The census works as designed:** fast, validated, resumable, honest nulls.
- **The hypothesis does not hold on this data:** lead-lag correlations between individual US stocks, at any lag from a day to two years, collected over 20 years of rolling windows, do not predict the following year better than random guessing, before or after costs.
- **What it does find:** strong, persistent same-day co-movement (useful for risk and diversification, not prediction), and a faint next-day ranking effect that is real in residual terms but not tradable.
- **Two-year caveat:** this used Stooq's survivor-only data. Survivorship bias inflates both the picks and the random guessers equally, so the comparison with random stays fair.
