# correlation2: test results and implementation choices

## Where things stand (read this first)

**Latest (2026-10-20):**
- **Current census:** `main-2025-06-27`, the default via `census/LATEST`. It has 36 move-length pairs (A and B each over 1, 3, 5, 10, 21 or 63 days) and 358.7 billion real correlations. See "Follow-up: your terms, a wider census…".
- **Code review fixes:** all 15 findings of a full code review, plus the verifiers' later findings, are fixed and verified. See "Code review fixes" at the end. The forecast and "% per 1%" are now relative to the market. Each run is read by its own settings, periods and stock list, and a second writer can't share a run.
- **Outside data:** 1,753 non-stock series were tested with the census's method. The series cover weather, climate, crypto, markets and macro, politics, news sentiment, Google Trends, prediction markets, prosperity, population and immigration, and space weather. None leads US stocks beyond chance. See "Outside data" at the end.
- **Corrected forecast track record** (market-relative): 61 large caps pooled, direction right 50.9%, mean correlation +0.018. That is a faint tilt with no practical value.
- **Out of sample:** fresh Stooq data through 2026-10-19, now the price store `stooq-2026-10-19`, covers the 16 months the census never saw. The census's picks made +21.0% before costs against +20.3% for the median random account (47.5% did as well; SPY made +24.6%). The 61 large-cap forecasts were right 51.7% of the time against 50.0% by chance (95% range 47.9–55.5%). No edge. See `research/oos_2026-10-19.txt`.
- **Price stores, split from the other data (2026-10-20):** stock prices now live apart from the census results and the outside data, one store per download, in `~/.local/share/correlation2/stores/`. There are two: `stooq-2025-06-27`, the data the census was made on and still the default, and `stooq-2026-10-19`, the fresh download. `stores/CURRENT` names the default. `c2 store list` shows the stores and `c2 store use NAME` changes the default; the GUI switches within seconds. `c2 --store NAME …` or `C2_STORE=NAME` uses another store for one command. `c2 ingest` and `c2 update` now build a new store and make it the default, and keep the old one. The census reads any store: `c2 --store stooq-2026-10-19 census holdout --start 2025-06-30` reproduces the out-of-sample result exactly, without a copy of the census.

The summary below describes the first census (`main`, 15 move-length pairs). Its conclusions still hold.

**The full correlation census has finished.** Every pair of liquid US stocks (283–2,345 per window), at every return horizon from 1 to 63 days, at lags from 1 day to 3 years, in rolling 1/2/3/5-year windows from 2005 to mid-2025. That's 83,879 real correlation matrices with 3 chance matrices each.
- **Time:** 5 min 47 s on the RTX 5080, against v1's estimated ~6 months in C.
- **Output:** `~/.local/share/correlation2/census/main/`.

**The answer to the question:**
- **The test you specified:** predict the last year (2024-06-28..2025-06-27) using only correlations found before it, and trade $10,000.
- **Before costs:** the picks ended at **$12,328 (+23.3%)**, against SPY's $11,361 (+13.6%) and QQQ's $11,432 (+14.3%).
- **But that isn't skill:** **10% of random-guess accounts** (same turnover, same stock pool) did as well (p = 0.10). The picks beat the market on 49.9% of days. At 5 bps per trade the picks made +8.2%; at 20 bps, −27%.
- **And it isn't stable:** the reduced smoke census, with the same rule and the same year, gave +10.8% (p = 0.39). Only the grid of horizons and windows differed.
- **Every year from 2006 on, the same test:** the picks beat random guessers in 2 of 19 years, which is what chance gives.
- **Pair by pair:** the strongest lagged pairs in one window keep their sign in the next window 50–53% of the time, the same as pairs picked from pure noise. Same-day co-movement, by contrast, persists almost perfectly.
- **Longer holding periods:** with weekly and monthly targets the census was null too (2 of 19 and 1 of 17 years beating random).
- **The test can detect a real signal:** a known anomaly (5-day short-term reversal) clearly beat random in 8 of 18 years (long-short 4.4 bps/day, z = 2.9), and the test tracked it fading after 2021. So the census's "no" is not a blind spot in the test.

So on this data the hypothesis does not hold. Details and every table follow below.

**Commands** (the `c2` binary is installed at `~/.local/bin/c2`; rebuild with `cargo build --release` in `~/programming/correlation2`):

| Command | What it does |
|---|---|
| `c2 census run --name main` | Run or resume the census (Ctrl-C stops cleanly after the current window) |
| `c2 census status` / `stop` / `pause` / `resume` | Check, stop, or pause (pausing frees the GPU) a running census |
| `c2 census holdout [--start 2024-06-28] [--walk-forward]` | The $10,000 test, one year or every year |
| `c2 census holdout --family peers [--walk-forward]` | The same test with peer momentum (monthly rebalance) |
| `c2 census holdout --family census --rebalance 5\|21` | Census pairs at weekly or monthly target horizons |
| `c2 census holdout --family reversal --lookback 5` | Positive control (short-term reversal) |
| `c2 census replicate` | Do stored pairs keep their sign in the next window? |
| `c2 census validate` | Engine vs the Python oracle |
| `c2 ingest`, `c2 data-check`, `c2 bench` | Build a new price store and make it the default, check the data, benchmark the hardware |
| `c2 store list` / `c2 store use NAME` | List the price stores; make one the default |
| `c2 --store NAME …` | Run one command on another price store (`C2_STORE=NAME` does the same) |
| `c2 update` / `c2 forward-report` / `c2 install-timer [--enable]` | Data refresh, forward-log scoring, the weekday timer |
| `c2-gui` | The native GUI |

**Also tested: peer momentum** (signal family C, the economic-link-style hypothesis from published research), using the census's same-day co-movement network:
- it beat random guessers in 3 of 18 years (chance gives about 1.8)
- its ranking signal was, if anything, slightly reversed (z = −2.79)
- nothing tradable

**Native GUI:** `c2-gui` (or **correlation2** in the app grid) has Status, Census, Holdout and Data tabs. Every tab was checked with screenshots, but you haven't tried it yet.

**Data updates and the forward log (first version):**
- `c2 update` ingests a Stooq zip dropped in `~/.local/share/correlation2/inbox/` into a new price store, which becomes the default (the old store is kept). It reruns the census if the data changed, and appends the next period's picks to an append-only forward log.
- `c2 forward-report` scores those picks against SPY once their holding period has passed.
- Tested end to end in a scratch data directory.
- The timer is **not enabled**; run `c2 install-timer --enable` if you want it.
- Fetching is still manual (Stooq needs a CAPTCHA) until a source like MarketParquet is validated.

**Not done yet** (from SPEC.md):
- census CPU worker (the GPU finishes in 6 minutes, so it would add ~3%)
- idle-detection mode
- signal families B and D (cluster lead-lag, the size control)
- a random guesser matched on volatility (a stricter baseline)
- survivorship-free data
- a point-in-time $5 price rule (it needs a split history; see "Last checks" under "Code review fixes")

**Your call:**
- **Fresher data:** the 16 months after the census are now in hand as the store `stooq-2026-10-19`. `c2 store use stooq-2026-10-19` makes it the default; the next `c2 update` then reruns the census on it (about 12 minutes) and starts the forward log. Delisted stocks and prices as they were at the time still need a paid source: Sharadar ($39 for a month of full history) or MarketParquet.
- **Direction:** continue with the cluster lead-lag family (B), or pivot the tool toward same-day co-movement, which is real and persistent and suits risk and diversification work.
- **The weekday update timer:** enable it with `~/.local/bin/c2 install-timer --enable` (run it from `~/.local/bin` so the unit points at the installed binary), or leave it off.

---

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
- **One memory-mapped columnar store** (`~/.local/share/correlation2/store/prices.bin`, 640 MB; since 2026-10-20 one per download, in `stores/<name>/`): per-symbol contiguous date/open/high/low/close/volume columns, each 64-byte aligned, plus `symbols.tsv`. Written via temp file + rename.
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

Only lag 1 keeps a small excess (1.2×). It is spread too thinly over pairs for any individual pair to pass the FDR rule within one window. Later check (2026-10-18), on the full-history window of `main-2025-06-27`: the next-day pairs that do pass are almost all stale prices or time zones. Of 127, 97 have a thinly traded closed-end fund as B (Nuveen municipal funds such as NVG, NZF, NAC), and 17 have a foreign company's US shares as B (MUFG, HMC, HSBC). In the leader, follower order, the leader trades more in 80% of pairs.

**How this squares with the earlier Python result:** `research/lag_spectrum.py` found that the lag structure of one period doesn't reappear in the next (top pairs ~coin-flip). The census now says the same thing from within each window: beyond lag 1, extreme lagged correlations occur at chance rates once volatility alignment is accounted for.

**Other small things:**
- The time-shift null's own tail jumped at a few lags (e.g. daily lag 63: 118k vs ~33k elsewhere). That was most likely shifts re-aligning quarterly earnings seasons. The sign-flip null makes it moot.
- `tail()` in the summary uses `ceil` for the boundary bin while the histogram uses `floor`, so tail counts slightly undercount. Harmless for ratios.

**Run directories:** the smoke runs (`smoke`, `smoke-b`, `smoke-c`, `smoke-d`) were deleted on 2026-10-19; their results stay in this file. The first 3.2 GB smoke run had already been replaced by its re-run.

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
- Rank IC t-stat positive in 16 of 19 years, combined z = 3.53. That's a weak but consistent ranking signal in vol-normalized residual returns. Its source is *not* the next-day effect from large to small stocks; see the correction under the full census's caveat 3.
- Long-short top/bottom 10% in actual returns: **0.43 bps/day gross, positive in 8 of 19 years, combined z = 0.52.** Not significant. With 44% of the book replaced daily, costs of c bps per side take about 1.8 × c bps/day, so even 1 bp of cost erases it.

## The full census (`c2 census run --name main`)

| | |
|---|---|
| Grid | horizons {1, 5, 10, 21, 63} (15 leader/target pairs), rolling windows of 1, 2, 3 and 5 years stepped quarterly from 2005-01-03, plus full history; lags per SPEC 6A; 3 sign-flip nulls per matrix |
| Size | 285 windows (283–2,345 stocks each), 75 tasks, **4,275 units, 83,879 real matrices (~335,000 matrix products including nulls)** |
| Time | **5 min 47 s** (10:51:16 → 10:57:03 as written in `run.log`; those log times are one hour ahead of this machine's clock, since they were written as UTC−4 on a UTC−5 system, fixed since), RTX 5080 at 97–99%. For reference, v1's approach was estimated at ~6 months in C for a smaller job |
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

**Three caveats on these numbers:**
1. **Configuration sensitivity.** The same holdout year, with the same rule, gave +10.8% (p = 0.39) on the smoke census and +23.3% (p = 0.10) on the full census. Only the set of horizons and window lengths differed. A result that moves that much with an arbitrary choice is noise. SPEC §7.4's ledger and deflated Sharpe ratio exist for exactly this; with every configuration counted, the 23.3% is not evidence.
2. **The random baseline is generous to the strategy.** Picks are chosen by score size, and score size favors stocks with many strong stored pairs, which tend to be the volatile ones. Uniform random picks from the same pool don't share that tilt. In a strong year a volatility tilt beats a uniform draw on its own (note the picks swinging +92% / −35% while the random medians don't), so p = 0.10 overstates how unusual 23.3% is. A random guesser matched on volatility or pick frequency would be the stricter test.
3. **What the rank IC (z = 4.15) is.** Corrected 2026-10-19. Two readings were offered here: the next-day effect from large to small stocks, and a mechanical lag-1 effect from leftover market exposure. A decomposition of the same 19 yearly ICs contradicts both:
   - **Not next-day:** offset-1 pairs alone give z = +0.19. Offsets of 2 days and more give +4.17, spread across 2–5, 6–21 and 22–63 days (+1.81 / +2.60 / +2.26).
   - **Sign-dependent:** it vanishes when the pairs' signs are ignored (−1.45) or randomized (−0.41), and comes mostly from negative-r pairs (+4.29, against +1.69 for positive-r).
   - **Not the stock's own reversal:** rank correlation with own-stock reversal is about −0.01.
   - **Not only thin stocks:** it tilts toward illiquid targets but isn't confined to them (terciles +3.81 / +2.87 / +1.79), and the top contributors are liquid names (SNA, AEM, A, VZ, EMR).
   - **Also in the smoke census:** smoke-d shows the same pattern.

   So it is a faint, consistent cross-sectional tilt of unexplained origin. It is still not monetizable: the long-short result in actual returns is near zero, and the picks beat random guessers in only 2 of 19 years.

### Verdict from the first version
- **The census works as designed:** fast, validated, resumable, honest nulls.
- **The hypothesis does not hold on this data:** lead-lag correlations between individual US stocks, at any lag from a day to two years, collected over 20 years of rolling windows, do not predict the following year better than random guessing, before or after costs.
- **What it does find:** strong, persistent same-day co-movement (useful for risk and diversification, not prediction), and a faint next-day ranking effect that is real in residual terms but not tradable.
- **Two-year caveat:** this used Stooq's survivor-only data. Survivorship bias inflates both the picks and the random guessers equally, so the comparison with random stays fair.

## Signal family C: peer ("connected-firm") momentum (`c2 census holdout --family peers`)

**Why this test.** It's the published, economically motivated form of lead-lag: Ali & Hirshleifer (2020) found that connected firms' past returns predict a stock's next-month return. The census supplied the network, because same-day co-movement is the one structure that persists (100% sign replication).

**Rule, fixed in advance:**
- **Peers:** each stock's 20 strongest positive same-day correlates (census variable) over the 3 years before the holdout.
- **Score:** the peers' mean log return over the last 21 trading days.
- **Account:** top 20 stocks, rebalanced every 21 trading days.
- **Comparisons:** the same $10,000 account, turnover-matched random guessers and benchmarks as before.

**Harness regression check:** after refactoring the harness for rebalancing periods, the census family reproduced its earlier numbers exactly (+23.3%, p = 0.101, hit 49.9%, IC t +0.57).

**2024-06-28..2025-06-27** (`research/peer_momentum_holdout.txt`):
- picks +12.7% (SPY +13.6%, QQQ +14.3%)
- random median +7.8%; **30% of random guessers did as well (p = 0.30)**
- +11.4% at 5 bps and +7.7% at 20 bps (monthly rebalancing keeps costs small)
- rank IC t −0.42; long-short t 0.20

**Walk-forward, 18 years** (`research/peer_momentum_walkforward.txt`; 2006–07 was skipped because the 3-year peer window reaches back before the census variable starts):
- The picks beat random guessers at p < 0.1 in **3 of 18 years**. Chance gives about 1.8 (binomial p ≈ 0.27).
- **Rank IC positive in only 6 of 18 years, combined z = −2.79.** If anything a peer *reversal* in residual terms, concentrated in 2016–2023. That's the opposite of the published effect, consistent with anomalies fading after publication.
- Long-short top/bottom 10% in actual returns: 0.51 bps/day, positive in 10 of 18 years, combined **z = 0.15**. Nothing tradable.

**Conclusion.** Neither the statistical family (census lead-lag pairs) nor the economic-link-style family (peer momentum from the census co-movement network) beats random guessing on this data. No parameter variants were tried: trying several and reporting the best would be the overfitting the harness exists to prevent.

## M7 (first version): native GUI (`c2-gui`)

- **Built with** eframe/egui 0.33.3 (the newest version that supports Rust 1.90) and egui_plot 0.34, rendering with OpenGL by default.
- **Launch:** `~/.local/bin/c2-gui`, or **correlation2** in the GNOME app grid (`~/.local/share/applications/correlation2.desktop`; delete that file to remove the launcher).
- **Runs without CUDA.** Census runs are started as a separate detached `c2` process, so closing the window never interrupts one.

| Tab | What it does | Verified (screenshots via the self-check mode) |
|---|---|---|
| Status | Census state, progress bar, ETA; Start/resume, Pause (frees the GPU), Resume, Stop | Shows the finished `main` census, 4,275 of 4,275 |
| Census | Real vs chance by lag for any horizon pair, as a plot and a table; ticker lookup of a stock's strongest stored pairs | Daily table matches the CLI (lag 1: 1.20, 1.4% FDR); AAPL lookup returns pairs with windows, partners, lags, r and z |
| Holdout | $10,000 test for either family, with stocks held, rebalance, costs and random-guesser count as inputs; results table; equity curves (picks, SPY, QQQ, random median) | Matches the CLI exactly (+23.3%, p = 0.101) |
| Data | Price-store summary and decade quality table | Matches `c2 data-check` |

**Bugs found by the screenshots and fixed:**
- **Random-median curve stuck at $0:** the path was pre-filled with zeros, which blocked the real median from being computed.
- **Date and cost fields cut off:** egui sizes grid columns from the previous frame, so the fields never grew. Fixed with a minimum column width.

**Self-check mode:** `C2_GUI_SCREENSHOT=<prefix> C2_GUI_TAB=status|census|holdout|data [C2_GUI_TICKER=AAPL] c2-gui` opens the tab, waits for its data, saves `<prefix>.rgba` and exits. `research/rgba_to_png.py <prefix>` converts the capture to PNG.

**Also fixed:** census log times were written as UTC−4 while this machine is UTC−5, so `run.log` times from today's runs are one hour ahead of local time. Log lines now use the system's offset.

## Closing two gaps: longer target horizons, and a positive control for the money test

**1. Census pairs with weekly and monthly targets.** The first holdouts traded only daily-target pairs. The census also stored target horizons of 5, 10, 21 and 63 days, so I ran the same walk-forward with the target horizon equal to the holding period (`--rebalance 5` and `--rebalance 21`).
- Timing: the target's return over (d, d+h₂] is predicted from the leader's move ending at d + h₂ − lag. lag ≥ h₂, so that is at or before d.
- Pre-declared, one run each, no tuning. Files: `research/walkforward_census_5.txt`, `walkforward_census_21.txt`.

| Target horizon | Years beating random at p < 0.1 | Rank IC (residual) | Long-short in actual returns |
|---|---|---|---|
| 1 day | 2 of 19 (chance ≈ 1.9) | z = 4.15 | 0.45 bps/day, z = 0.92 |
| 5 days | 2 of 19 | z = 3.29 | −0.44 bps/day, z = −1.11 |
| 21 days | 1 of 17 | z = 4.55 | 0.21 bps/day, z = 0.68 |

Null at every holding period in actual returns. The residual rank IC is consistently positive at every horizon. It is the same small effect discussed above (large→small diffusion, or leftover market exposure), and it never turns into money.

**2. A positive control for the money side.** Until now the holdout harness had only ever said "no", so a broken harness would have produced the same verdict. Own-stock short-term reversal (score = minus the stock's own log return over the last L days, the broadest known short-horizon anomaly) is the control (`--family reversal`).

| Control | Years beating random at p < 0.1 | Rank IC | Long-short in actual returns |
|---|---|---|---|
| **5-day reversal, weekly** | **8 of 18** (chance ≈ 1.8) | positive 18/18, **z = 8.96** | **4.41 bps/day gross, positive 16/18, z = 2.94** |
| 21-day reversal, monthly | 4 of 18 | positive 18/18, z = 13.04 | 0.91 bps/day, z = 1.15 |

By year, the 5-day reversal was strong from 2008 to 2021: long-only picks beat random at p ≈ 0.000 in seven of those years (e.g. +64.4% vs +28.0% random in 2012–13). It **has faded since 2021**: long-only picks lost money in each year since, while the long-short stayed slightly positive. That is what published anomalies typically do.

**What this means for the verdict.** The same harness, data and random guessers that find nothing in the census clearly detect a known effect, and track it fading. **The census's null result is not a blind spot of the test.**

## M3 (first version): data updates and the forward log (`c2 update`, `c2 forward-report`, `c2 install-timer`)

**What `c2 update` does:**
1. Ingests any Stooq zip in `~/.local/share/correlation2/inbox/` (since 2026-10-20 into a new price store in `stores/`, which becomes the default; the old store is kept). Downloading is manual for now, because Stooq requires a CAPTCHA since early 2026; MarketParquet would allow automatic fetching once it's validated (ANALYSIS.md).
2. Archives the zip under `raw/stooq/<last data day>/`.
3. If the store changed, reruns the full census as `main-<last data day>` (~6 min) and points `census/LATEST` at it.
4. Appends the next period's picks for both families (census pairs daily, peer momentum monthly) to `forward_log.jsonl`: one line per data day and family, never rewritten, written before the outcomes exist.

`c2 forward-report` scores entries whose holding period has passed, against SPY. `c2 install-timer` prints systemd user units for weekday evenings at 18:30 New York time (catching up after downtime); `c2 install-timer --enable` installs and starts them. **Not enabled; that's your decision.** Run it from `~/.local/bin/c2` so the unit points at the installed binary.

**On "refine as new data arrives":** a full census takes ~6 minutes, so new data currently means a fresh dated run rather than appending windows to an old one. Each stored pair keeps its running sums (Σxy, Σx, Σy, Σx², Σy²), so a live tracker that updates current windows day by day can be added later.

**Resume guard (bug fixed):** progress is recorded by window position. Re-ingesting newer data and resuming an old run would have filed records under the wrong windows without any error. Resume now refuses when the store's row count, the census variable's date range or the window count differ. The manifest also records `git_dirty`, so a binary built from uncommitted code is visible (the gap found earlier with run `main`).

**End-to-end test** (all in a scratch data directory via `C2_DATA_DIR`; the real store and census untouched):

| Step | Result |
|---|---|
| `c2 ingest --until 2025-03-31` | 26,015,779 rows; 417 symbols that listed later are empty |
| Smoke census `smoke-base` | 876 windows |
| `c2 update` with an empty inbox | "no new zips"; logged 20 census and 20 peer picks for the period after 2025-03-31 |
| Copy of the full zip into the inbox, `c2 update --smoke` | ingested 26,656,424 rows; archived the zip; detected the change (last day 2025-03-31 → 2025-06-27); ran census `smoke-2025-06-27` (2 min); updated LATEST; logged picks for 2025-06-27. Original zip untouched |
| Resume `smoke-base` on the new data | **Refused**: "the price data changed … (rows 26015779 -> 26656424, …, windows 146 -> 148); start a new run with a different --name" |
| Windows, old vs new run | 146 → 148; first 77 identical. Ids of the longer windows shift by the added windows, so ids are run-specific; key windows by (length, start) for any future cross-run tracking |
| `c2 forward-report` | 2025-03-31 census picks −0.30% vs SPY +0.28% (1 day); peer picks +4.21% vs SPY −0.87% (21 days); 2025-06-27 entries pending |
| `c2 install-timer` | Prints both units (nothing installed) |

## Follow-up: browsing relationships by stock (after the user returned)

**`c2 census stock TICKER [--significant]`** lists every stored lagged relationship involving a stock, as leader or follower, strongest first. Each row shows:
- the partner and lag in days
- the move in price terms: the follower's typical move beyond the market per 1% move of the leader, i.e. r × the ratio of the two stocks' return volatilities, so it has the same sign as r
- the census r and z
- whether it passed the significance test
- how many pairs at least that strong pure chance produced in the same window

The chance column is what separates real relationships from the many large-but-random ones.

**GUI:**
- **Strongest** lists stocks ranked by number × strength (sum of |r| over their stored relationships), with relationship count, mean and max |r|, significant count (leads/follows), and how many separate windows the significant ones come from. A sort control switches to significant-only, strongest single correlation, or most relationships. Clicking a stock opens its timeline.
- **Stock timeline** is drawn on a days-ahead axis only (the first version put leaders at negative days, which read as "correlation in the past"). A switch picks the direction: *what moves this stock later* (other stocks move today, this one follows k days later; these feed the forecast) or *what this stock moves later*. Partners are placed at their lag. Height is the % move per 1% (or r). Significant relationships are orange; chance-level ones grey. Filters: significant only, window (all / most recent / full history / any single window), return horizons, number shown. Selecting a relationship plots its r in every census window where it was stored, so persistent relationships and one-off events look different.

**What the significant relationships turned out to be.** 895 stored pairs passed the significance test (444 stocks); noise passed the same test at a similar matrix-level rate (82 placebo firings vs 47 real matrices). They fall into two groups:
- **Municipal-bond closed-end funds** (Nuveen NAD, NAC, NZF, NEA, NVG, filed by Stooq as NYSE stocks) leading each other by 1 day over the full 2005–2025 history. This is persistent: thinly traded funds with similar holdings, catching up a day late. Real, but hard to trade.
- **Clusters from the 2020 crash windows,** mostly at a 13-day lag: regional banks (BOH 110 pairs, COLB, CBSH, FHB) and utilities (SO, AWR, ATO, EVRG, CMS). The crash came in two waves about 13 trading days apart (late February and March 9–16, 2020). That really isn't chance within those windows, but it is a one-off event, not a repeatable relationship.

**The number × strength ranking is led by very volatile stocks** (SAVA, RHP, TXT…), largely through long-horizon windows with few observations (33–50), where chance alone reaches |r| 0.5–0.8.

## Follow-up: forecasting a stock's price from its relationships (`c2 census forecast`, GUI Forecast tab)

**Method.** As of a chosen day, take every stored relationship where another stock (the leader) moves first and the selected stock follows `lag` days later. Use the most recent census window that had ended by that day, optionally only windows from the last N years.
- The leader's actual moves so far, over its h1-day return horizon, are scaled by beta = r × (follower volatility ÷ leader volatility) in that window. That gives a predicted h2-day move for the follower, landing `lag` days after the leader's move, spread evenly over its h2 days.
- Each relationship covering a future day is a separate estimate of that day's return, so the forecast for the day is their **mean**. The means are accumulated into a price path.
- The first draft **summed** the relationships and produced paths like −82% in a month for AAPL. Summing N estimates of the same thing multiplies the size by about N.

**Track record built in.** The same forecast is remade every 21 trading days in the past, each time using only windows that had ended by then, and compared with what actually happened 21 days later. The CLI and the GUI both show it.

**Results:**

| | direction right | predicted vs actual correlation | typical size predicted vs actual |
|---|---|---|---|
| AAPL (200 forecasts) | 55.0% | +0.08 | 0.26% vs 6.30% |
| BOH (206 forecasts) | 53.4% | +0.13 | 0.20% vs 5.94% |
| NAD (95 forecasts) | 51.6% | −0.17 | 0.12% vs 2.41% |
| 61 large caps pooled (13,700 forecasts) | 50.2% | mean per-stock +0.005 (sd 0.09); positive for 32 of 61 | 0.17% vs 5.64% |
| same, windows from the last 3 years only | 49.6% | 0.000 | 0.23% vs 5.63% |

The forecast line comes out nearly flat: hundreds to thousands of relationships per stock disagree and mostly cancel. The little that remains has no measurable skill. That matches the census (real ≈ chance), the replication test (50–53% same sign) and the holdout.

**GUI Forecast tab:**
- Controls: ticker (shared with the timeline tab), as-of date with month/year steps, days ahead (5–252), significant-only, window age, and share price or value of $X.
- Plot: the forecast path, the actual path when the as-of date is in the past, and the stock's typical ±1 sd range for scale.
- Below the plot:
  - a bar chart of when the predicted moves land, with the number of relationships covering each day
  - the track-record line, coloured yellow when it is within what luck produces
  - the relationships with the largest share of the forecast

## Follow-up: your terms, a wider census, the Explainer, and Status statistics

**Terminology now used in the app and CLI:**

| Your term | Earlier name | Meaning |
|---|---|---|
| move length (A over 5 days, B over 5 days) | horizon h1 / h2, "returns" | how long a move is measured for each stock |
| offset | lag | trading days from the end of A's move to the end of B's move |
| correlation period (e.g. "756 days") | census window | the stretch of history a correlation is measured over: rolling 252/504/756/1260-trading-day periods, one starting every quarter, plus the whole 2005–2025 history (5,154 days) |

**In your words.** Every relationship is now also counted straight from prices, on the same non-overlapping observations the census uses. The counts are: how often B rose after A rose by at least its typical (1 sd) move, how often after A fell by that much, and how often B rose overall. They are given both inside the correlation period and over the rest of the history. The second set is the honest check, because the first comes from the very period that made the pair look strong. For example, NZF → NAD daily over the full history: after NZF rose 1% or more, NAD rose next day 185 of 306 times (60%), against 51% normally.

**Wider census (`main-2025-06-27`, now the default via `census/LATEST`):**
- **Grid:** move lengths 1, 3, 5, 10, 21 and 63 trading days, with every combination including B longer than A. That is 36 pairs, up from 15.
- **Code change:** the observation grid now starts at max(A, B) − 1, so B's move never reaches before the period. For the old pairs this is the same as before.
- **Run:** 358.7 billion real correlations (plus 3× as many for the chance baseline) in 0h11m.
- **Verification:** records for A 1 → B 5, A 3 → B 21 and A 5 → B 21 reproduce exactly from prices (`c2 census explain`): daily values identical, r within 2e-8 (float storage).
- **Result:** unchanged.
  - Across all lagged matrices: 1.02× as many real pairs as chance at |z| ≥ 4, and 1.05× at |z| ≥ 5.
  - The significance test fired in 94 matrices, but on pure chance (the placebo) it fired in 163.
  - No new move-length pair stands out. Ratios run 0.98–1.13. The only higher ratios come from 63-day pairs, which have just 12–25 matrices.

**A bug found and fixed on the way:**
- **What happened:** the first wide run crashed at 71%. The GPU kernels scan the padded matrix width (a multiple of 16), but the per-column target statistics for the padding columns were never cleared between work units. A narrower unit inherited stale statistics there, which produced a junk correlation in a padding column.
- **Fix:** the statistics are now cleared every unit, and padding entries are filtered out.
- **Effect on the first census (`main`):** it had the same stale values, but they only fed its real and chance histograms equally (both use the same target statistics). It never stored a padding pair, since that would have crashed. The new census supersedes it anyway.

**Explainer tab and `c2 census explain`:** picks any A, B, move lengths, offset and correlation period, then recomputes from prices:
- each day's return, market return, beta, residual, trailing sd and census value, compared with the census's own values (identical in every case tried)
- every observation's raw and census-unit moves
- the scatter, r, z and the stored record
- the chance comparison for that matrix
- the plain-terms counts
- the % move per 1% used by the timeline and the forecast

The Stock timeline and Forecast tabs open it with the relationship pre-filled.

**Status tab additions:**
- **Run history:** every census run with its kind, data through, start time, time spent computing (summed over sessions from `run.log`), correlations and GPU multiply-adds. Crashed runs show as interrupted.
- **Last full census.**
- **Data updates:** a note on updates, plus the update log. Incremental updates are not built: new data means a full rerun of about 11 minutes.
- **Charts for the census in use:**
  - strength of stored correlations, real vs chance
  - real/chance by move lengths, by offset and by period length
  - significant relationships by year
  - per-stock average |r|, significant count and stored count

## Code review fixes (2026-10-18/19)

A full code review (`/code-review max`) of the forecast, your-terms, wider-census, Explainer and Status work found 15 verified problems. Its candidate list held 29 more correctness items, 6 efficiency items and 8 cleanups. All of them were fixed in three rounds by parallel agents, each in its own git worktree with its own files. The work was merged on branch `review-fixes`, then checked by four verifiers and a fresh code review. The verifiers reran every original reproduction against the real data, read-only, or in sandbox copies. A third round fixed what they found, and a final verification round rechecked everything.

**What changed that you will notice:**
- **Each census run is read by its own description** (`c2_census::meta::RunMeta`): its own settings, task numbering, correlation periods (mapped by date) and stock list.
  - Each run keeps a copy of the store's `symbols.tsv`, and stocks are matched by ticker. `c2 ingest`, `c2 update` and the new `c2 census snapshot-symbols` save it; it was saved for the 7 existing runs on 2026-10-18.
  - Old runs (`main`, 75 tasks) are read correctly again. replicate, holdout and walk-forward reproduce `research/census_main_*.txt` number for number.
  - After a data refresh, correlation periods the new data doesn't cover and delisted stocks are skipped and counted, not misread and not refused.
- **One writer per run.** `run::run` holds an exclusive lock and writes the pid, so a second census on the same run is refused instantly.
  - `status`, `stop`, `pause` and `resume` without `--name` act on the running census. A census started by `c2 update` shows as running, not "interrupted".
  - LATEST is written atomically and fsynced. `c2 update --smoke` never changes it.
  - `c2 update` runs the full census whenever LATEST is out of date for the store, even with an empty inbox. It never gets stuck on a same-day run it can't resume.
  - The forward log is written only from a census of the day's own data, one row per day and family, under a file lock.
- **The forecast and every "% per 1%" are relative to the market,** with one shared implementation (`moves.rs`) built on the census's repaired closes. That fixes the market leak (BOH as of 2008-10-10: +0.39% became −1.36%) and the bad prints (NUE's "typical daily move" 20.76% became 1.99%).
  - The number shown is exactly the multiplier applied. It is bit-identical in the forecast, the Stock timeline and Explainer step 7 for all 34,133 contributions checked.
  - Forecasts skip full-history periods, which the track record can't grade. Ties go to the longest period, not to file order.
  - Each contribution row shows the part of A's moves landing in the forecast's days, so its arithmetic holds: (1 + X/100)^M − 1.
- **Explainer:**
  - It uses the run's own periods and stocks.
  - It says when the census never computed a relationship, and why: too few observations, an offset off the census grid, the stock didn't qualify, the period isn't covered, or the stock is gone since.
  - Offset 0 is explainable as the same-day reference.
  - The self-check now compares the recomputed sums with the census's stored ones. On 2,184 random stored pairs the largest difference was 5.9e-8.
- **GUI:**
  - Every job and cache is keyed by run (and store version). Switching runs can't mix them up or crash the Holdout.
  - Failed jobs show errors instead of endless spinners.
  - The store is reloaded when `c2 update` replaces it.
  - The Forecast tab reads a stock once: an option change takes 33 ms (was 0.7 s), and walking back 12 months 0.37 s (was 8.4 s).
  - The Strongest tab takes 0.2 s (was 1.7 s). Status statistics take 0.12 s and 80 MB (was 1.3 s and 1.8 GB).
- **Census engine:** padding entries are dropped before selection. A CPU recount of `main-2025-06-27`'s tail counts matches exactly, so that census was unaffected. Ties at the keep boundary now break by matrix index, so reruns are byte-identical.

**Corrected forecast track records** (21-day forecasts remade every 21 days; direction right / correlation of predicted with actual):

| | Before the fixes (raw moves) | After (market-relative), `main-2025-06-27` |
|---|---|---|
| AAPL | 55.0% / +0.08 | 56.5% / +0.05 |
| BOH | 53.4% / +0.13 | 54.4% / −0.12 |
| NAD | 51.6% / −0.17 | 48.4% / −0.24 (reliably wrong; the GUI shows it red) |
| 61 large caps pooled | 50.2% / +0.005 | 50.9% / +0.018 (40 of 61 positive) |
| same, periods ending in the last 3 years only | 49.6% / 0.000 | 50.9% / +0.013 |

- The pooled 50.9% is 0.9 points above the 50.1% the up/down base rates alone give.
- Allowing for stocks forecast on the same dates moving together, it's borderline: p = 1.7% to 8% depending on the permutation test.
- Predicted moves average 0.10%, against actual moves of 4.70%. This is a faint tilt with no practical value.

**A finding about the census's chance comparison.** The outside-data test injected 220 pure-noise series. At 63-day moves they show about 2.4× as many strong correlations as the sign-flip null predicts, and 1.2× at 21-day moves. The census's own 63→63 ratio is 2.41. So that bar on the Status chart is an artifact: the null is too lenient for persistent series at long move lengths. Possible fixes are noise calibration series in every census, or a block sign flip at long move lengths. Neither is built yet.

**Costs:** CLI `forecast`/`stock` now peak at 1.27 GB (was 0.37), and the GUI at about 1.7 GB (was 0.83), because the market-relative maths needs the census variable for all stocks.

**Polish after the final verification (2026-10-19):**
- **Forecast cap.** Each of A's moves counts as at most 5 of its typical moves in that relationship's correlation period, because the census clips at 5 and never measured a relationship beyond that. Capped rows are marked.
  - It fixes absurd rows such as MSTR <- OKLO, which said +6,268,142,759% and now says +242.84%. OKLO's 2022–2024 history is its pre-merger SPAC's flat price. Stooq carries SPAC histories under the later ticker; removing those stretches from the census would need a rerun.
  - Across every stock's latest forecast, 3.3% of rows are capped. The largest implied row went from 6.27 billion % to 1,587%, and rows beyond ±100% from 1,262 to 535.
  - The pooled track record is unchanged at 50.9% (mean correlation +0.015). AAPL is 58.0% / +0.050, BOH 54.4% / −0.123 → 55.6% / −0.238 as of 2008-10-10, and NAD 48.4% / −0.236.
- **No false alarms after a data refresh.** When today's data lacks a stock the run had, the census's market average changes, and every recomputed value differs slightly from what was stored. The Explainer now names the missing stock and shows the difference as expected (yellow), not as a fault.
- **Superseded relationships are marked.** A relationship's shorter period is never used by the forecast when a longer period of the same relationship ends the same day; NAD has 100 such rows and AAPL 288. These rows are marked "(a longer period is used)" in `c2 census stock`, and with an asterisk in the Stock timeline. The Explainer says why.
- **Simultaneous starts.** The census lock is taken before anything is logged, so the loser of two simultaneous starts is recorded as refused.
- **Smaller fixes:** thousands separators everywhere, verdict wording by which side of 50% the hit rate is on, size-less rows kept off the "% per 1%" chart, singular forms, and "correlation-period computations" instead of "windows".

**Last checks (2026-10-19):** the final Explainer verification passed every item but one, now fixed, and noted a few smaller things:
- **One damaged results file no longer blocks a whole run.** In runs older than the manifest's task list, an unreadable or corrupted results file made every read of the run fail with a misleading "has records but no summary lines". Now only that file's task fails, and the message names the file and the real error.
  - Other tasks still explain, in the CLI and the GUI.
  - The Forecast and Stock views name the file that stops them, and resuming such a run is refused.
  - Tested on sandbox copies with unreadable, garbled and truncated files, including `main-2025-06-27` with one of its 180 summary files unreadable.
- **Wording.** The GUI says "relative to the market" rather than "beyond the market", and defines it once per screen: a stock's move minus the part its usual sensitivity to the market (its beta) accounts for. The Explainer's intro no longer claims every step is checked against stored values, since daily values aren't stored. The CLI shows "no price" instead of "NaN%".
- **Speed.** The older `smoke` run opens in 0.55 s instead of 1.1 s, because its census variable is built once.
- **Open finding, not fixed: the $5 price rule looks ahead.** A stock counts in a correlation period only if its median price was at least $5. That rule uses the store's prices, which are adjusted for every later split and dividend, so whether a stock counted in a past period depends on what happened after it.
  - Example: AAPL's 2009-01-05 close is $2.73 in the store (it traded near $95 then), so AAPL was left out of the 2009 periods.
  - This changes which stocks were compared in older periods, not how. Recent periods, which later splits barely touch, show no edge either (the track record over periods ending in the last 3 years is 50.9%), so the verdict stands.
  - A fix needs a split history, which the Stooq bulk files don't include, and a census rerun.

## Outside data: does anything outside the stock market lead US stocks?

**Data** (`~/.local/share/correlation2/external/`; each source has a `catalog_*.csv`, a `SOURCES_*.md` with sources, terms, timing rules and caveats, normalized `series/*.csv` and untouched `raw/` downloads):

| Catalog | Series | What |
|---|---|---|
| weather | 239 | 30 NOAA US stations plus London, Frankfurt and Sydney (temperature, rain, snow, wind, degree days); CPC population-weighted degree days; NAO/AO/PNA/AAO, ENSO, MJO; hurricanes, tornadoes; US Drought Monitor; ERA5 points (New York sunshine, Corn Belt, Kansas wheat, Central Valley, Brazil coffee, Brazil/Argentina soy, Tokyo) |
| markets | 255 | 71 crypto (59 coins, Coinbase, Bitcoin on-chain, Fear & Greed); 30 world indices and 19 equity ETFs; government yields in 14 countries; FX; commodities; oil, gas and inventories; credit; volatility indices; economic releases |
| other | 79 | space weather and sunspots (placebo); moon phase and daylight; earthquakes; population (monthly US; annual states, regions and world); flu and COVID; 37 Wikipedia topics; geopolitical risk; gasoline and electricity demand; TSA |
| politics | 212 | state EPU (50 + DC), 19 country EPUs, US categorical EPU, Twitter and UK EPU; geopolitical risk for 44 countries; partisan conflict; protests (US by state from 2017; 12 countries 1995–2023); elections and changes of power; state partisan control; Iowa Electronic Markets; approval; shutdowns and debt limit; Federal Register; COVID stringency |
| news | 135 | daily headline tone (VADER, Loughran-McDonald) and crisis-word counts from Wikipedia Current Events (point-in-time), Hacker News, Reuters (2007–2014), ABC Australia, Times of India, Irish Times, HuffPost, All the News |
| trends | 88 | Google Trends, weekly from 2004: Preis-2013 terms, FEARS terms, crypto, jobs, weather, market terms, 20 tickers; 8 daily key terms |
| predictions | 111 | Kalshi and Polymarket as 20 continuous rolling series (Fed next meeting, CPI, jobs, GDP, recession, shutdown, House and Senate control) from 2021; 91 single contracts (finite-life, chosen with hindsight; browsing only) |
| prosperity | 342 | county unemployment and home values (distributions across all counties plus 50 counties), state unemployment and coincident indexes, OECD unemployment, QCEW wages, BEA income, SAIPE income and poverty, World Bank and WID inequality |
| migration | 292 | CBP border encounters (monthly from 1999), refugees, DHS removals and arrests, green cards; EU asylum (monthly); Canada, Australia and UK flows; county and state population growth with net international and domestic migration (annual) |

**Method** (`crates/c2-census/examples/external_test.rs`; results in `research/external_test_results.txt`):
- **Same as the census:** move lengths, offsets, observation grid, sign-flip null, false-discovery rule and placebo, against the census's own stock universes.
  - Periods: the four non-overlapping 1260-day correlation periods (2005–2025) plus full history.
  - It reproduced 350 stored census correlations to 3e-8.
- **Timing:** each observation counts only from the day it was publicly known by the 4 pm New York close, a conservative bound per series. For example, weather counts from the next day, jobless claims 5 days after the week, and monthly releases at their worst historical publication delay.
- **Standardizing:**
  - Seasonal series become anomalies against earlier years only.
  - Everything is standardized by a trailing year, as stocks are.
  - Sparse event series need at least 10 non-zero moves in a period.
- **Kept apart:** annual series (about 20 changes in 20 years, often all on one release day) and finite-life contracts are reported separately, not in the headline.
- **Noise baseline:** 220 pure-noise series run through the same pipeline.

**Results** (run `ext-all-2`: 1,753 series, 15.6 billion real correlations against 46.7 billion chance, 2 min 03 s on the GPU):

| Direction, predictive offsets | Real/chance at \|z\| ≥ 4 | Noise baseline | Same sign in the next 5-year period | Picks from the null copies |
|---|---|---|---|---|
| outside data -> stocks | 1.04 | 1.04 | 50.7% (n 292,506) | 50.1% |
| stocks -> outside data | 1.05 | 1.06 | | |

- **By category, outside data -> stocks, predictive:** 0.81–1.10 everywhere, apart from two small, explained exceptions. Climate indices score 1.27 (two near-duplicate weekly Niño series); monthly population scores 1.59 (a calendar alignment with a monthly-paying fund).
  - News 0.92, politics 1.02, prosperity 1.05, migration 1.01, Google Trends 1.03.
  - Annual series 1.00.
- **Controls:** all pass.
  - Same day: oil -> energy stocks z +7.6 to +32.7; gold -> miners +24.9 to +52.2; gas -> producers +3.7 to +14.9.
  - SPY -> Asian indices the next day: r +0.15 to +0.54.
  - Synthetic leads are found at their exact offsets.
  - Placebos look like chance: sunspots, solar flux, Wikipedia "Cat"/"Philosophy".
- **What survives is mechanical:**
  - US-listed ETFs leading thinly traded closed-end funds and small banks by a day (stale prices, r ≈ 0.13).
  - Data released on a fixed calendar lining up with funds' payout dates or companies' calendars. The 50 state indexes released together and the monthly-paying PDI, CLM and CRF funds are one example; quarterly state income and Garmin another. Moon phase lines up with dividend dates the same way.
  - In plain terms these are flat: GRMN rose 52.5% of the time after the income series' top quarter and 55.0% after its bottom quarter, against 53.0% normally.
- **Prediction markets** need their own test, because they start in 2021. Using the census's 252-day periods from April 2022 to April 2025 (`--short-periods`), the rolling series score 1.03× chance, and 50.4% keep their sign next year (null picks 49.2%). That is the same as crypto (1.05×, 49.5%), daily Google Trends (1.00×, 49.3%), 100 random other series (1.02×, 49.8%) and noise (1.01×, 49.9%). They do move with stocks on the same day (1.77×), because their rolls fall on FOMC, CPI and jobs-report days.

**Verdict:** none of the outside data tested leads US stocks beyond chance, and nothing found is tradable.

To rerun:
- `cargo run --release -p c2-census --example external_test -- --run <name> --noise 220`
- add `--short-periods --short-from 2022-04-01` for the recent-years test
- `--browse <run> --series <id> --min-z 4` lists a series' stored relationships
