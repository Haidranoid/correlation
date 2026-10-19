//! `c2`: correlation2 command line.

use std::time::Instant;

mod update;

use anyhow::Result;
use c2_engine::{ColumnStats, cpu, pearson, reference_r, synth};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "c2", about = "correlation2: stock lead-lag census and holdout testing")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Time the all-pairs cross-product in every precision on the GPU and CPU, checking each
    /// against an exact f64 reference.
    Bench {
        /// Number of stocks.
        #[arg(long, default_value_t = 5000)]
        n: usize,
        /// Rows (trading days) per product, comma-separated. 64 = one census block.
        #[arg(long, value_delimiter = ',', default_value = "64,256,768,1280")]
        rows: Vec<usize>,
        /// Skip the CPU measurements.
        #[arg(long)]
        no_cpu: bool,
    },
    /// Build the price store from Stooq bulk zips (default: v1's d_us_txt.zip).
    Ingest {
        /// Stooq zip files.
        #[arg(long = "zip")]
        zips: Vec<std::path::PathBuf>,
        /// Markets to keep (top-level Stooq folders).
        #[arg(long, value_delimiter = ',', default_value = "us")]
        markets: Vec<String>,
        /// Drop rows after this date (YYYY-MM-DD): reproduce what was known then, or test updates.
        #[arg(long)]
        until: Option<String>,
    },
    /// Ingest new zips from <data dir>/inbox, rerun the census if the data changed, and append
    /// today's picks to the forward log. Run on a schedule by `c2 install-timer`.
    Update {
        /// Zip files to ingest instead of the inbox.
        #[arg(long = "zip")]
        zips: Vec<std::path::PathBuf>,
        /// Use the reduced census grid (for testing).
        #[arg(long)]
        smoke: bool,
        /// Skip rerunning the census.
        #[arg(long)]
        no_census: bool,
    },
    /// Score forward-log picks whose holding period has passed, against SPY.
    ForwardReport,
    /// Print (or with --enable, install and start) a systemd user timer that runs `c2 update` on
    /// weekday evenings.
    InstallTimer {
        #[arg(long)]
        enable: bool,
    },
    /// Print the store's acceptance numbers: symbol kinds, bad prints, decade quality table.
    DataCheck,
    /// The correlation census: run, check on, pause or resume it, or validate the engine.
    Census {
        #[command(subcommand)]
        action: CensusCmd,
    },
}

#[derive(Subcommand)]
enum CensusCmd {
    /// Run or resume a census (Ctrl-C stops after committing; run again to resume).
    Run {
        /// Reduced grid (horizons 1, 5, 21; 1- and 3-year windows) for a quick end-to-end run.
        /// Only for a new run: an existing one always continues with its own settings.
        #[arg(long)]
        smoke: bool,
        /// Run name (directory under <data dir>/census/).
        #[arg(long, default_value_t = c2_census::run::latest_run())]
        name: String,
    },
    /// Show progress and ETA.
    Status {
        /// Census to show (default: the census running now if there is exactly one, else the
        /// default census named in census/LATEST).
        #[arg(long)]
        name: Option<String>,
    },
    /// Pause after the current window and release the GPU.
    Pause {
        /// Census to pause (default: the census running now if there is exactly one, else the
        /// default census named in census/LATEST).
        #[arg(long)]
        name: Option<String>,
    },
    /// Continue a paused census.
    Resume {
        /// Census to continue (default: the census running now if there is exactly one, else
        /// the default census named in census/LATEST).
        #[arg(long)]
        name: Option<String>,
    },
    /// The $10,000 holdout test: predict the last year from census pairs found before it.
    Holdout {
        #[arg(long, default_value_t = c2_census::run::latest_run())]
        name: String,
        /// First day treated as the future.
        #[arg(long, default_value = "2024-06-28")]
        start: String,
        /// Stocks held each day.
        #[arg(long, default_value_t = 20)]
        picks: usize,
        /// Trading costs to test, basis points per side.
        #[arg(long, value_delimiter = ',', default_value = "0,5,20")]
        costs: Vec<f64>,
        /// Random-guess accounts to compare against.
        #[arg(long, default_value_t = 1000)]
        random: usize,
        /// Use pairs from every pre-holdout window, not just the latest of each length.
        #[arg(long)]
        all_windows: bool,
        /// Walk forward: treat each year from this year's start month onward as the future in
        /// turn (one-year holdouts ending at the last data day), and print a table by year.
        #[arg(long)]
        walk_forward: bool,
        /// Signal family: `census` (stored lead-lag pairs; the target horizon equals --rebalance),
        /// `peers` (peer momentum) or `reversal` (own-stock short-term reversal, a positive control).
        #[arg(long, default_value = "census")]
        family: String,
        /// Trading days between rebalances (default: 1 for census pairs, 21 for peers).
        #[arg(long)]
        rebalance: Option<usize>,
        /// Peer momentum: peers per stock, return lookback (days), correlation window (days).
        #[arg(long, default_value_t = 20)]
        peers: usize,
        #[arg(long, default_value_t = 21)]
        lookback: usize,
        #[arg(long, default_value_t = 756)]
        peer_window: usize,
    },
    /// Recompute one relationship from prices step by step and compare with what the census stored.
    Explain {
        /// Stock A, which moves first.
        leader: String,
        /// Stock B, which follows.
        follower: String,
        #[arg(long, default_value_t = c2_census::run::latest_run())]
        name: String,
        /// A's move length in trading days.
        #[arg(long, default_value_t = 1)]
        h1: usize,
        /// B's move length in trading days.
        #[arg(long, default_value_t = 1)]
        h2: usize,
        /// Offset in trading days from the end of A's move to the end of B's.
        #[arg(long, default_value_t = 1)]
        offset: usize,
        /// Correlation period in trading days (252, 504, 756, 1260; 0 = full history).
        #[arg(long, default_value_t = 0)]
        period: usize,
        /// Use the correlation period ending on or before this day (default: the latest).
        #[arg(long)]
        end: Option<String>,
    },
    /// Forecast a stock's move relative to the market from its leaders' recent moves.
    Forecast {
        ticker: String,
        #[arg(long, default_value_t = c2_census::run::latest_run())]
        name: String,
        /// Forecast as of this day (default: the last day of data). A past date shows what happened.
        #[arg(long)]
        as_of: Option<String>,
        /// Use only relationships that passed the significance test.
        #[arg(long)]
        significant: bool,
        /// Trading days ahead.
        #[arg(long, default_value_t = 21)]
        days: usize,
        /// Use only relationships from correlation periods that ended within this many years.
        #[arg(long)]
        max_age_years: Option<f64>,
    },
    /// A stock's lead-lag relationships: who it leads and follows, at what lag, and by how much.
    Stock {
        ticker: String,
        #[arg(long, default_value_t = c2_census::run::latest_run())]
        name: String,
        /// Only pairs that passed the false-discovery test.
        #[arg(long)]
        significant: bool,
        /// Most rows per direction.
        #[arg(long, default_value_t = 25)]
        limit: usize,
    },
    /// Replication test: do stored pairs keep their sign in the next non-overlapping window?
    Replicate {
        #[arg(long, default_value_t = c2_census::run::latest_run())]
        name: String,
    },
    /// Stop a running census cleanly (it finishes the window in progress); `run` resumes it.
    Stop {
        /// Census to stop (default: the census running now if there is exactly one, else the
        /// default census named in census/LATEST).
        #[arg(long)]
        name: Option<String>,
    },
    /// Check the engine against research/census_oracle.json (exact r from Python).
    Validate {
        #[arg(long, default_value = "research/census_oracle.json")]
        oracle: std::path::PathBuf,
    },
    /// Save each census run's stock list: a copy of the store's symbols.tsv, which the stock
    /// numbers in its results refer to, for every run that has none yet and whose list the
    /// store still holds. `c2 ingest` and `c2 update` do this before they rebuild the store.
    SnapshotSymbols,
}

fn main() -> Result<()> {
    #[cfg(feature = "cuda")]
    c2_engine::gpu::reexec_with_cuda_libraries();
    match Cli::parse().cmd {
        Cmd::Bench { n, rows, no_cpu } => bench(n, &rows, !no_cpu),
        Cmd::Ingest { zips, markets, until } => ingest(zips, &markets, until),
        Cmd::Update { zips, smoke, no_census } => update::update(update::UpdateOptions { zips, smoke, no_census }),
        Cmd::ForwardReport => update::forward_report(),
        Cmd::InstallTimer { enable } => update::install_timer(enable),
        Cmd::DataCheck => data_check(),
        Cmd::Census { action } => census(action),
    }
}

/// Lag-1 leader and target panels (`kp x n`, zero-padded to a multiple of 32 rows) cut from a
/// `k + 1` day synthetic panel, plus that panel for the reference.
fn lag1_panels(n: usize, k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let t = k + 1;
    let full = synth::returns_panel(t, n, 42 + k as u64);
    let kp = k.div_ceil(32) * 32;
    let (mut x, mut y) = (vec![0f32; kp * n], vec![0f32; kp * n]);
    for j in 0..n {
        x[j * kp..j * kp + k].copy_from_slice(&full[j * t..j * t + k]);
        y[j * kp..j * kp + k].copy_from_slice(&full[j * t + 1..j * t + 1 + k]);
    }
    (x, y, full, kp)
}

/// Pairs to check: random ones plus planted sector leads (leader i < 20, target in its sector).
fn sample_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut rng = synth::Rng::new(7);
    let mut pairs: Vec<(usize, usize)> = (0..1500)
        .map(|_| ((rng.next_u64() % n as u64) as usize, (rng.next_u64() % n as u64) as usize))
        .collect();
    pairs.extend((0..500).map(|q| (q % 20, 20 + (q * 20 + q % 20) % (n - 20))));
    pairs
}

/// Largest |r - reference| over `pairs`, given a cross-product lookup and the panels it came from.
fn max_error(cross: impl Fn(usize, usize) -> f64, xs: &[f32], ys: &[f32], kp: usize, k: usize, n: usize, full: &[f32], pairs: &[(usize, usize)]) -> f64 {
    let (sx, sy) = (ColumnStats::of(xs, kp, n, 0..k), ColumnStats::of(ys, kp, n, 0..k));
    pairs
        .iter()
        .map(|&(i, j)| {
            let r = pearson(cross(i, j), sx.sum[i], sy.sum[j], sx.sum_sq[i], sy.sum_sq[j], k as f64);
            (r - reference_r(full, k + 1, i, j, 1)).abs()
        })
        .fold(0.0, f64::max)
}

fn bench(n: usize, rows: &[usize], with_cpu: bool) -> Result<()> {
    let pairs = sample_pairs(n);
    println!("{n} stocks; times are per all-pairs product (one lag); error = max |r - f64 reference| over {} pairs\n", pairs.len());

    #[cfg(feature = "cuda")]
    if let Err(e) = bench_gpu(n, rows, &pairs) {
        println!("GPU unavailable, CPU only: {e:#}");
    }

    if with_cpu {
        println!("\nCPU ({} threads, gemm f32)", rayon::current_num_threads());
        println!("{:>6} {:>10} {:>9} {:>10}", "rows", "ms", "TOPS", "error");
        for &k in rows {
            let (x, y, full, kp) = lag1_panels(n, k);
            let mut out = vec![0f32; n * n];
            cpu::xprod_f32(&x, &y, kp, n, kp, &mut out);
            let reps = if k <= 256 { 5 } else { 2 };
            let start = Instant::now();
            for _ in 0..reps {
                cpu::xprod_f32(&x, &y, kp, n, kp, &mut out);
            }
            let secs = start.elapsed().as_secs_f64() / reps as f64;
            let err = max_error(|i, j| out[i * n + j] as f64, &x, &y, kp, k, n, &full, &pairs);
            println!("{:>6} {:>10.2} {:>9.2} {:>10.1e}", k, secs * 1e3, 2.0 * (n * n * kp) as f64 / secs / 1e12, err);
        }

        println!("\nCPU ({} threads, INT8 AVX-VNNI assembly micro-kernel)", rayon::current_num_threads());
        println!("{:>6} {:>10} {:>9} {:>10} {:>10} {:>12}", "rows", "ms", "TOPS", "error", "pack ms", "no-store ms");
        for &k in rows {
            let (x, y, full, kp) = lag1_panels(n, k);
            let (xq, yq): (Vec<i8>, Vec<i8>) = (x.iter().map(|&v| c2_engine::quantize(v)).collect(), y.iter().map(|&v| c2_engine::quantize(v)).collect());
            let start = Instant::now();
            let packed = c2_engine::vnni::Packed::new(&xq, &yq, kp, n, kp);
            let pack_secs = start.elapsed().as_secs_f64();
            let mut out = vec![0i32; n * n];
            packed.xprod(&mut out);
            let reps = if k <= 256 { 5 } else { 2 };
            let start = Instant::now();
            for _ in 0..reps {
                packed.xprod(&mut out);
            }
            let secs = start.elapsed().as_secs_f64() / reps as f64;
            // Compute only (tiles visited, nothing written): the cost a fused census epilogue starts from.
            let start = Instant::now();
            for _ in 0..reps {
                packed.for_each_tile(|_, _, t| {
                    std::hint::black_box(t);
                });
            }
            let compute_secs = start.elapsed().as_secs_f64() / reps as f64;
            let (xs, ys): (Vec<f32>, Vec<f32>) = (xq.iter().map(|&v| v as f32).collect(), yq.iter().map(|&v| v as f32).collect());
            let err = max_error(|i, j| out[i * n + j] as f64, &xs, &ys, kp, k, n, &full, &pairs);
            println!("{:>6} {:>10.2} {:>9.2} {:>10.1e} {:>10.2} {:>12.2}", k, secs * 1e3, 2.0 * (n * n * kp) as f64 / secs / 1e12, err, pack_secs * 1e3, compute_secs * 1e3);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn bench_gpu(n: usize, rows: &[usize], pairs: &[(usize, usize)]) -> Result<()> {
    use c2_engine::gpu::{DevProduct, Gpu, HIST_BINS, Precision};

    let gpu = Gpu::new()?;
    let (free, total) = gpu.memory()?;
    println!("GPU: {} ({:.1} of {:.1} GB free)", gpu.name()?, free as f64 / 1e9, total as f64 / 1e9);
    println!("{:>6} {:>6} {:>10} {:>9} {:>10}", "rows", "mode", "ms", "TOPS", "error");
    for &k in rows {
        let (x, y, full, kp) = lag1_panels(n, k);
        for p in Precision::ALL {
            let (dx, dy) = (gpu.upload(&x, p)?, gpu.upload(&y, p)?);
            let mut c = gpu.alloc_product(p, n)?;
            let plan = if p == Precision::Fp32 { None } else { Some(gpu.plan(p, n, kp)?) };
            gpu.xprod(p, plan.as_ref(), &dx, &dy, &mut c, n, kp)?;
            gpu.stream.synchronize()?;
            let reps = if k <= 256 { 50 } else { 20 };
            let start = Instant::now();
            for _ in 0..reps {
                gpu.xprod(p, plan.as_ref(), &dx, &dy, &mut c, n, kp)?;
            }
            gpu.stream.synchronize()?;
            let secs = start.elapsed().as_secs_f64() / reps as f64;

            // Error is measured on the values the GPU actually saw (rounded or quantized).
            let seen = |v: &[f32]| -> Vec<f32> {
                match p {
                    Precision::Fp16 => v.iter().map(|&a| half::f16::from_f32(a).to_f32()).collect(),
                    Precision::Bf16 => v.iter().map(|&a| half::bf16::from_f32(a).to_f32()).collect(),
                    Precision::Int8 => v.iter().map(|&a| c2_engine::quantize(a) as f32).collect(),
                    _ => v.to_vec(),
                }
            };
            let (xs, ys) = (seen(&x), seen(&y));
            let err = match &c {
                DevProduct::F32(d) => {
                    let h = gpu.stream.clone_dtoh(d)?;
                    max_error(|i, j| h[i + j * n] as f64, &xs, &ys, kp, k, n, &full, pairs)
                }
                DevProduct::I32(d) => {
                    let h = gpu.stream.clone_dtoh(d)?;
                    max_error(|i, j| h[i + j * n] as f64, &xs, &ys, kp, k, n, &full, pairs)
                }
            };
            println!("{:>6} {:>6} {:>10.3} {:>9.1} {:>10.1e}", k, p.name(), secs * 1e3, 2.0 * (n * n * kp) as f64 / secs / 1e12, err);

            // The census pass that follows each product: r, |z| histogram, compaction.
            if k == rows[0] && matches!(p, Precision::Fp32 | Precision::Int8) {
                let (sxh, syh) = (ColumnStats::of(&xs, kp, n, 0..k), ColumnStats::of(&ys, kp, n, 0..k));
                let factors = gpu.pass_factors(&sxh, &syh, k as f64)?;
                let s = &gpu.stream;
                let cap = 1 << 20;
                let mut hist = s.alloc_zeros::<u32>(HIST_BINS)?;
                let mut count = s.alloc_zeros::<u32>(1)?;
                let (mut oi, mut or) = (s.alloc_zeros::<i64>(cap)?, s.alloc_zeros::<f32>(cap)?);
                let run = |hist: &mut _, count: &mut _, oi: &mut _, or: &mut _| {
                    gpu.census_pass(&c, n, k as f64, &factors, 16.0, 4.0, cap, hist, count, oi, or)
                };
                run(&mut hist, &mut count, &mut oi, &mut or)?;
                s.synchronize()?;
                let start = Instant::now();
                for _ in 0..20 {
                    s.memset_zeros(&mut hist)?;
                    s.memset_zeros(&mut count)?;
                    run(&mut hist, &mut count, &mut oi, &mut or)?;
                }
                s.synchronize()?;
                let secs = start.elapsed().as_secs_f64() / 20.0;
                let (h, cnt) = (s.clone_dtoh(&hist)?, s.clone_dtoh(&count)?);
                let total: u64 = h.iter().map(|&v| v as u64).sum();
                println!("{:>6} {:>6} {:>10.3}   census pass ({:.0} GB/s read; {} of {} entries histogrammed; {} with |z| >= 4)",
                    "", "", secs * 1e3, (n * n * 4) as f64 / secs / 1e9, total, n * n, cnt[0]);
            }
        }
    }
    Ok(())
}

fn store_dir() -> std::path::PathBuf {
    c2_data::data_dir().join("store")
}

fn ingest(mut zips: Vec<std::path::PathBuf>, markets: &[String], until: Option<String>) -> Result<()> {
    use anyhow::Context;
    if zips.is_empty() {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from).unwrap_or_default();
        zips.push(home.join("programming/correlation/data/stooq/d_us_txt.zip"));
    }
    let paths: Vec<&std::path::Path> = zips.iter().map(|p| p.as_path()).collect();
    let markets: Vec<&str> = markets.iter().map(|s| s.as_str()).collect();
    let dir = store_dir();
    let until = until.map(|u| c2_data::parse_date(&u)).transpose()?;
    // Census results hold store ids, which a rebuilt store renumbers: first give every run
    // without one a copy of the symbol table its ids refer to (as `c2 update` does).
    println!("Stock lists of the census runs, saved before the store is rebuilt:");
    snapshot_symbols().context("could not save the stock lists of the existing census runs, so the store was not rebuilt")?;
    let r = c2_data::ingest::ingest_stooq_until(&paths, &markets, &dir, until)?;
    println!("Ingested {} files ({} empty) into {}: {} rows in {:.1}s", r.files, r.empty, dir.display(), r.rows, r.seconds);
    for (folder, n) in &r.by_folder {
        println!("  {folder:<22} {n:>6} files");
    }
    Ok(())
}

fn data_check() -> Result<()> {
    use c2_data::{Calendar, Kind, Panel, Store};
    let store = Store::open(&store_dir())?;
    println!("{} symbols, {} rows", store.symbols.len(), store.rows());
    let mut kinds = std::collections::BTreeMap::new();
    for s in &store.symbols {
        *kinds.entry((s.kind.as_str(), s.exchange.as_str())).or_insert(0usize) += 1;
    }
    for ((kind, exch), n) in &kinds {
        println!("  {kind:<6} {exch:<8} {n:>6}");
    }
    for s in store.symbols.iter().filter(|s| s.kind == Kind::Test) {
        println!("  test ticker excluded: {} ({} rows)", s.ticker, s.rows);
    }

    // Bad prints since 2021-06-01 on the SPY calendar, all symbols and v1's symbol set
    // (v1's importer kept only tickers matching ^[A-Z0-9^_]{1,15}$, so no hyphens).
    let cal = Calendar::from_symbol(&store, "SPY")?;
    let rows = cal.range(c2_data::parse_date("2021-06-01")?, i32::MAX);
    let v1 = |t: &str| !t.is_empty() && t.len() <= 15 && t.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'^' || b == b'_');
    for (label, ids) in [
        ("all symbols", (0..store.symbols.len()).collect::<Vec<_>>()),
        ("v1 symbol set", (0..store.symbols.len()).filter(|&i| v1(&store.symbols[i].ticker)).collect()),
    ] {
        let mut panel = Panel::build(&store, &cal, rows.clone(), &ids);
        println!("Bad prints repaired since 2021-06-01, {label} ({} symbols): {}", ids.len(), panel.repair_bad_prints());
    }

    // The census calendar (long-lived stocks) against SPY where both exist.
    let tdays = Calendar::trading_days(&store)?;
    let spy_first = cal.days[0];
    let a: Vec<i32> = tdays.days.iter().copied().filter(|&d| d >= spy_first).collect();
    let extra = a.iter().filter(|d| cal.days.binary_search(d).is_err()).count();
    let missing = cal.days.iter().filter(|d| a.binary_search(d).is_err()).count();
    println!("Trading calendar: {} days from {}; since SPY's first day ({}): {} days vs SPY's {}, {extra} extra, {missing} missing",
             tdays.days.len(), c2_data::date_string(tdays.days[0]), c2_data::date_string(spy_first), a.len(), cal.days.len());

    println!("\n{:>7} {:>8} {:>11} {:>10} {:>9} {:>11}", "decade", "symbols", "rows", "0% days", "under $1", "close only");
    for r in c2_data::quality::decade_table(&store) {
        let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
        println!("{:>6}s {:>8} {:>11} {:>9.1}% {:>8.1}% {:>10.1}%", r.decade, r.symbols, r.rows,
                 pct(r.zero_return, r.with_prev), pct(r.under_1, r.rows), pct(r.close_only, r.rows));
    }
    Ok(())
}

fn census(action: CensusCmd) -> Result<()> {
    use c2_census::run;
    match action {
        CensusCmd::Validate { oracle } => {
            let store = c2_data::Store::open(&store_dir())?;
            let r = c2_census::validate::validate(&store, &oracle)?;
            println!("Oracle window: {} stocks ({} not in store: {:?}), {} days, {} pairs", r.universe, r.missing.len(), r.missing, r.days, r.pairs);
            println!("  max |r - oracle|, exact f64 path:       {:.2e}", r.max_err_exact);
            println!("  max |r - oracle|, GPU INT8 census path: {:.2e}", r.max_err_gpu_int8);
            println!("  max |r - oracle|, CPU INT8 (AVX-VNNI):  {:.2e}", r.max_err_cpu_int8);
            println!("  max |GPU INT8 - CPU INT8|:              {:.2e}", r.max_cpu_gpu_diff);
            println!("  oracle's 50 strongest lag-1 pairs screened in by the GPU pass: {}/{}", r.top50_found, r.top50_total);
            Ok(())
        }
        CensusCmd::SnapshotSymbols => {
            println!("Stock lists of the census runs in {}:", c2_data::data_dir().join("census").display());
            snapshot_symbols()
        }
        CensusCmd::Holdout { name, start, picks, costs, random, all_windows, walk_forward, family, rebalance, peers, lookback, peer_window } => {
            let fam = match family.as_str() {
                "census" => c2_census::holdout::Family::CensusPairs,
                "peers" => c2_census::holdout::Family::PeerMomentum { peers, lookback, window: peer_window },
                "reversal" => c2_census::holdout::Family::Reversal { lookback, window: peer_window },
                other => anyhow::bail!("unknown family {other} (census, peers or reversal)"),
            };
            let rebalance = rebalance.unwrap_or(match family.as_str() { "peers" => 21, "reversal" => lookback, _ => 1 });
            let store = c2_data::Store::open(&store_dir())?;
            let dir = run::run_dir(&name);
            let t0 = Instant::now();
            // The run's own settings, correlation periods, stocks and task numbering, on today's data.
            let meta = c2_census::meta::RunMeta::load(&dir)?;
            let plan = run::Plan::for_run(&store, &meta)?;
            if walk_forward {
                let (y0, mo, d) = c2_data::civil_from_days(c2_data::parse_date(&start)?);
                let last = *plan.var.days.last().unwrap();
                println!("Walk-forward: each year treated as the future, predicted only from census windows that ended before it.");
                println!("{:<24} {:>7} {:>9} {:>9} {:>10} {:>10} {:>10} {:>8} {:>8} {:>9} {:>8}", "holdout", "pairs", "SPY", "QQQ", "picks", "random med", "p (0 bps)", "hit %", "IC t", "LS bps/d", "LS t");
                let mut beat = 0;
                let mut years = 0;
                let (mut ic_ts, mut ls_ts, mut ls_bps, mut ls_turn): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = Default::default();
                for y in (2006..=y0).rev().collect::<Vec<_>>().into_iter().rev() {
                    let s = c2_data::days_from_civil(y, mo, d);
                    let e = c2_data::days_from_civil(y + 1, mo, d);
                    if e > last + 7 && y != y0 {
                        continue;
                    }
                    let opt = c2_census::holdout::Options { family: fam, rebalance, start: s, end: e, picks, costs_bps: vec![0.0], random_runs: random, cash: 10_000.0, all_windows };
                    let Ok(r) = c2_census::holdout::holdout(&store, &plan, &dir, &opt) else { continue };
                    if r.pairs == 0 {
                        continue;
                    }
                    let pct = |v: f64| 100.0 * (v / 10_000.0 - 1.0);
                    let bench = |i: usize| r.benchmarks.get(i).map(|b| pct(b.final_value)).unwrap_or(f64::NAN);
                    let (_, strat, med, p) = r.strategy[0];
                    years += 1;
                    beat += usize::from(p < 0.1);
                    println!("{:<24} {:>7} {:>8.1}% {:>8.1}% {:>9.1}% {:>9.1}% {:>10.3} {:>7.1}% {:>8.2} {:>9.2} {:>8.2}",
                        format!("{}..{}", c2_data::date_string(r.first_day), c2_data::date_string(r.last_day)), r.pairs,
                        bench(0), bench(1), pct(strat), pct(med), p, 100.0 * r.hit_rate, r.ic_t, 1e4 * r.ls_mean, r.ls_t);
                    ic_ts.push(r.ic_t);
                    ls_ts.push(r.ls_t);
                    ls_bps.push(1e4 * r.ls_mean);
                    ls_turn.push(r.ls_turnover);
                }
                println!("Years where the picks beat random guessers at p < 0.1: {beat} of {years} (chance alone: about {:.1})  [{:.0}s]", 0.1 * years as f64, t0.elapsed().as_secs_f64());
                let stouffer = |v: &[f64]| v.iter().sum::<f64>() / (v.len() as f64).sqrt();
                let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
                println!("Rank IC: t positive in {} of {years} years; combined across years z = {:.2}",
                    ic_ts.iter().filter(|&&t| t > 0.0).count(), stouffer(&ic_ts));
                println!("Long-short top/bottom 10%: mean {:.2} bps/day gross, positive in {} of {years} years, combined z = {:.2}; turnover {:.0}% of the book per rebalance, so costs of c bps per side take about {:.2} x c bps/day",
                    mean(&ls_bps), ls_bps.iter().filter(|&&v| v > 0.0).count(), stouffer(&ls_ts), 100.0 * mean(&ls_turn), 2.0 * 2.0 * mean(&ls_turn) / rebalance as f64);
                return Ok(());
            }
            let opt = c2_census::holdout::Options { family: fam, rebalance, start: c2_data::parse_date(&start)?, end: i32::MAX, picks, costs_bps: costs, random_runs: random, cash: 10_000.0, all_windows };
            let r = c2_census::holdout::holdout(&store, &plan, &dir, &opt)?;
            println!("Holdout {} .. {} ({} trading days), {:.0}s", c2_data::date_string(r.first_day), c2_data::date_string(r.last_day), r.days, t0.elapsed().as_secs_f64());
            println!("Family {family}, rebalanced every {rebalance} trading day(s): {} pairs from {} pre-holdout window(s); {} stocks can be picked",
                r.pairs, r.windows_used, r.candidates);
            println!("Picks beat the market over their holding period: {:.1}% of {} picks (coin flip 50%)", 100.0 * r.hit_rate, r.hits);
            println!("Rank correlation of prediction vs outcome (IC): mean {:+.4}, t = {:+.2}", r.ic_mean, r.ic_t);
            println!("Long-short top/bottom 10%: {:.2} bps/day gross, Sharpe {:.2}, t = {:.2}, book turnover {:.0}% per rebalance",
                1e4 * r.ls_mean, r.ls_sharpe, r.ls_t, 100.0 * r.ls_turnover);
            println!("\n{:<44} {:>12} {:>9}", "", "final value", "return");
            for b in &r.benchmarks {
                println!("{:<44} {:>12.0} {:>8.1}%", b.label, b.final_value, 100.0 * (b.final_value / 10_000.0 - 1.0));
            }
            for (c, s, med, share) in &r.strategy {
                let label = if *c == 0.0 { "no trading costs".to_string() } else { format!("{c} bps per trade") };
                println!("{:<44} {:>12.0} {:>8.1}%   random guessers: median {:.0}, {:.1}% did at least as well (p = {:.3})",
                    format!("Picks, {label}"), s, 100.0 * (s / 10_000.0 - 1.0), med, 100.0 * share, share);
            }
            Ok(())
        }
        CensusCmd::Explain { leader, follower, name, h1, h2, offset, period, end } => {
            let store = c2_data::Store::open(&store_dir())?;
            let t0 = Instant::now();
            let plan = run::Plan::new(&store, c2_census::grid::Config::full())?;
            let end = match end { Some(d) => c2_data::parse_date(&d)?, None => i32::MAX };
            let window = c2_census::explain::find_window(&plan, period, end).ok_or_else(|| anyhow::anyhow!("no {period}-day correlation periods"))?;
            let req = c2_census::explain::Request { leader: leader.to_uppercase(), follower: follower.to_uppercase(), h1, h2, lag: offset, window, threshold: None };
            let e = c2_census::explain::explain(&store, &plan, &run::run_dir(&name), &req)?;
            println!("(census variable for all stocks built in {:.1}s)", t0.elapsed().as_secs_f64());
            println!("correlation period: {} trading days, {} .. {}; {} stocks qualified; {} in: {}, {} in: {}", e.period_days, c2_data::date_string(e.period_first),
                c2_data::date_string(e.period_last), e.universe, e.a.ticker, e.a.in_universe, e.b.ticker, e.b.in_universe);
            println!("daily census values recomputed from prices vs the census: max difference {} {}, {} {}", e.a.ticker, e.a.max_diff, e.b.ticker, e.b.max_diff);
            println!("{} observations ({}-day moves of {} ending {} trading days before the end of {}-day moves of {}):", e.n, h1, e.a.ticker, offset, h2, e.b.ticker);
            for o in e.obs.iter().take(3) {
                println!("  {} {} -> {}: {:+.2}% (x {:+.3});  {} {} -> {}: {:+.2}% (y {:+.3})", e.a.ticker, c2_data::date_string(e.days[o.a_end - h1]), c2_data::date_string(e.days[o.a_end]),
                    100.0 * o.a_move.exp_m1(), o.x, e.b.ticker, c2_data::date_string(e.days[o.b_end - h2]), c2_data::date_string(e.days[o.b_end]), 100.0 * o.b_move.exp_m1(), o.y);
            }
            println!("r = {:+.6}, z = {:+.3}; r of raw % moves = {:+.4}; % move per 1% = {:+.3}", e.r, e.z, e.r_raw, e.move_per_pct);
            match &e.stored {
                Some(s) => println!("stored by the census: r = {:+.6} (difference {:.2e}), screening r {:+.4}, z {:+.3}, {}{}", s.record.r, (s.record.r as f64 - e.r).abs(), s.record.r_int8, s.record.z,
                    if s.top { "among the strongest" } else { "" }, if s.significant { ", significant" } else { "" }),
                None => println!("not stored by the census for this period (not among that matrix's strongest)"),
            }
            if e.matrix.found {
                println!("matrix: {} stocks x {} stocks, {} observations; pairs at |z|>=3/4/5: real {:?}, chance (sign-flip) {:?}; significance cut-off |r| {}",
                    e.matrix.stocks, e.matrix.stocks, e.matrix.obs, e.matrix.real, e.matrix.null, e.matrix.fdr_r.map(|v| format!("{v:.4}")).unwrap_or("none".into()));
            }
            println!("chance alone would give about {:.1} of {:.0} pairs at |z| >= {:.2}", e.chance_analytic, e.pairs_tested, e.z.abs());
            if let Some(w) = &e.words {
                let pc = |a: u32, b: u32| if b > 0 { format!("{a} of {b} ({:.0}%)", 100.0 * a as f64 / b as f64) } else { "no cases".into() };
                for (label, c) in [("in the period", &w.period), ("rest of history", &w.rest)] {
                    println!("{label}: after {} rose >= {:.2}%: {} rose {}; after it fell >= {:.2}%: {} rose {}; {} rose {} overall", e.a.ticker, 100.0 * w.up.exp_m1(), e.b.ticker,
                        pc(c.a_up_b_up, c.a_up), -100.0 * w.down.exp_m1(), e.b.ticker, pc(c.a_down_b_up, c.a_down), e.b.ticker, pc(c.b_up, c.cases));
                }
            }
            Ok(())
        }
        CensusCmd::Forecast { ticker, name, as_of, significant, days, max_age_years } => {
            let store = c2_data::Store::open(&store_dir())?;
            let t = ticker.trim().to_uppercase();
            let as_of = match as_of { Some(d) => c2_data::parse_date(&d)?, None => i32::MAX };
            let dir = run::run_dir(&name);
            // The census variable with the run's own settings.
            let meta = c2_census::meta::RunMeta::load(&dir)?;
            let cal = c2_data::Calendar::trading_days(&store)?;
            let var = c2_census::variable::production(&store, &cal, meta.cfg.start)?;
            let data = c2_census::forecast::prepare(&store, &var, &dir, &t)?;
            let f = c2_census::forecast::forecast_from(&data, as_of, &c2_census::forecast::Options {
                significant_only: significant,
                horizon: days,
                max_age: max_age_years.map(|y| (y * 252.0) as usize),
            })?;
            println!("{t} as of {}: ${:.2}; {} relationships where {t} follows another stock{}; typical daily move relative to the market {:.2}%", c2_data::date_string(f.as_of),
                f.last_price, f.relationships, if significant { " (significant only)" } else { "" }, 100.0 * f.daily_sd);
            println!("Moves are relative to the market; the $ columns are the price those moves give if the market is flat.");
            println!("{:>5} {:>11} {:>12} {:>11} {:>12} {:>9}", "day", "predicted", "if flat", "actual", "if flat", "covering");
            for (j, (p, a)) in f.predicted_pct.iter().zip(&f.actual_pct).enumerate() {
                if j < 5 || (j + 1) % 5 == 0 || j + 1 == f.predicted_pct.len() {
                    let ad = a.map(|v| format!("{v:+.2}%")).unwrap_or_default();
                    let ap = a.map(|v| format!("${:.2}", f.last_price * (1.0 + v / 100.0))).unwrap_or_default();
                    println!("{:>5} {:>10.2}% {:>12} {:>11} {:>12} {:>9}", j + 1, p, format!("${:.2}", f.last_price * (1.0 + p / 100.0)), ad, ap, f.covering[j]);
                }
            }
            println!("\nLargest shares of the forecast at day {}:", f.predicted_pct.len());
            for c in f.contributions.iter().take(10) {
                println!("  {:<6} offset {:>3}d (moves {}d -> {}d, r {:+.3}{}): its move {:+.2}% x {:+.3}% per 1% implies {:+.2}% for {t}; share {:+.3} pts   [{}]",
                    c.leader, c.lag, c.h1, c.h2, c.r, if c.significant { ", significant" } else { "" }, c.leader_move_pct, c.move_per_pct, c.implied_pct, c.share_pct, c.window);
            }
            match &f.skill {
                Some(s) => println!("\nTrack record: the same forecast made every 21 trading days in the past ({} forecasts, {}-day horizon), against {t}'s actual move relative to the market: \
                    direction right {:.1}% of the time (of {} with a direction); predicted vs actual correlation {:+.3} (slope {:+.2}); average size predicted {:.2}% vs actual {:.2}%.",
                    s.forecasts, s.horizon, 100.0 * s.hit_rate, s.directional, s.corr, s.slope, s.mean_abs_predicted_pct, s.mean_abs_actual_pct),
                None => println!("\nTrack record: too few past forecasts to judge."),
            }
            println!("\nRelative to the market means a stock's move minus its usual share of the market's move (beta x market), which is what the census correlates.");
            println!("Each leader's moves are multiplied by that relationship's % per 1% (the same number `c2 census stock` shows).");
            Ok(())
        }
        CensusCmd::Stock { ticker, name, significant, limit } => {
            let store = c2_data::Store::open(&store_dir())?;
            let t = ticker.trim().to_uppercase();
            let dir = run::run_dir(&name);
            // The census variable with the run's own settings.
            let meta = c2_census::meta::RunMeta::load(&dir)?;
            let cal = c2_data::Calendar::trading_days(&store)?;
            let var = c2_census::variable::production(&store, &cal, meta.cfg.start)?;
            let rows = c2_census::analyze::stock_profile_with(&store, &var, &dir, &t, significant)?;
            let sig = rows.iter().filter(|r| r.significant).count();
            if significant {
                println!("{t}: {sig} lead-lag relationships passed the significance test across all correlation periods");
            } else {
                println!("{t}: {} stored lead-lag relationships across all correlation periods, {sig} passed the significance test", rows.len());
            }
            for leads in [true, false] {
                let part: Vec<_> = rows.iter().filter(|r| r.leads == leads).take(limit).collect();
                println!("\n{t} {} (strongest first):", if leads { "leads" } else { "follows" });
                if part.is_empty() {
                    println!("  none");
                }
                for r in part {
                    let (a, b) = if leads { (t.as_str(), r.partner.as_str()) } else { (r.partner.as_str(), t.as_str()) };
                    let size = r.move_per_pct.map(|m| format!("{b} {m:+.2}% per 1% {a} move, relative to the market")).unwrap_or_else(|| "size n/a".into());
                    let moves = format!("{a} over {}d -> {b} over {}d", r.h1, r.h2);
                    let words = r.words.map(|w| {
                        let c = w.period;
                        let rate = |x: u32, n: u32| if n > 0 { format!("{:.0}%", 100.0 * x as f64 / n as f64) } else { "-".into() };
                        format!("; after {a} rose {:.1}%+, {b} rose {}/{} ({}) vs {} normally", 100.0 * w.up.exp_m1(), c.a_up_b_up, c.a_up, rate(c.a_up_b_up, c.a_up), rate(c.b_up, c.cases))
                    }).unwrap_or_default();
                    println!("  {a} -> {b:<6} offset {:>3} days: {size}; r {:+.3}, z {:+.1}; chance made {:.0} pairs with |z| >= {} in that period{}{words}   [{moves}; correlation period {} days to {}, {} observations]",
                        r.lag, r.r, r.z, r.chance, r.chance_z, if r.significant { "  SIGNIFICANT" } else { "" }, r.period_days, r.period_end, r.obs);
                }
            }
            println!("\nr is the census correlation (market-adjusted, volatility-normalized returns); the % per 1% is that relationship in price terms:");
            println!("the follower's move relative to the market per 1% move of the leader relative to the market, in that correlation period");
            println!("(the multiplier `c2 census forecast` applies). The 'after ... rose' counts use plain price moves. 'Chance made N' is how many pairs");
            println!("at least that strong appeared in the same period when the relationship was scrambled; if N is not far below 1, the pair is likely noise.");
            println!("The offset runs from the end of the leader's move to the end of the follower's. `c2 census explain` shows any one step by step.");
            Ok(())
        }
        CensusCmd::Replicate { name } => {
            let store = c2_data::Store::open(&store_dir())?;
            let dir = run::run_dir(&name);
            let t0 = Instant::now();
            // The run's own settings, correlation periods, stocks and task numbering, on today's data.
            let meta = c2_census::meta::RunMeta::load(&dir)?;
            let plan = run::Plan::for_run(&store, &meta)?;
            let table = c2_census::analyze::replicate(&plan, &dir)?;
            println!("Replication in the next non-overlapping window ({:.0}s). 'Chance' rows are pairs picked from sign-flip null matrices.", t0.elapsed().as_secs_f64());
            println!("{:>4} {:>4} {:>16} {:>7} {:>10} {:>11} {:>16} {:>14}", "h1", "h2", "lag", "kind", "pairs", "same sign", "mean |r| then", "mean r next*");
            for ((h1, h2, b, null), r) in &table {
                println!("{h1:>4} {h2:>4} {b:>16} {:>7} {:>10} {:>10.1}% {:>16.4} {:>14.4}",
                    if *null { "chance" } else { "real" }, r.pairs, 100.0 * r.same_sign as f64 / r.pairs.max(1) as f64,
                    r.abs_then / r.pairs.max(1) as f64, r.signed_next / r.pairs.max(1) as f64);
            }
            println!("* sign(r then) x r next: positive means the relationship persisted in its original direction");
            Ok(())
        }
        CensusCmd::Stop { name } => {
            let name = control_target(name);
            let dir = run::run_dir(&name);
            match run::running_pid(&dir) {
                Some(pid) => {
                    // SIGTERM: the census's handler commits the window in progress, then exits.
                    let sent = std::process::Command::new("kill").arg(pid.to_string()).status()?.success();
                    anyhow::ensure!(sent, "could not signal census {name}'s process (pid {pid})");
                    println!("asked census {name} (pid {pid}) to stop; it commits the window in progress first. `c2 census run --name {name}` resumes.");
                }
                None => println!("census {name} is not running"),
            }
            Ok(())
        }
        CensusCmd::Status { name } => {
            let name = control_target(name);
            let dir = existing_run_dir(&name)?;
            match run::running_pid(&dir) {
                Some(pid) => println!("process: running (pid {pid})"),
                None => println!("process: not running"),
            }
            let Ok(text) = std::fs::read_to_string(dir.join("status.json")) else {
                println!("census {name}: no progress recorded yet (control: {})", run::control_state(&dir));
                return Ok(());
            };
            let s: serde_json::Value = serde_json::from_str(&text)?;
            let age = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs().saturating_sub(s["updated_unix"].as_u64().unwrap_or(0));
            println!("census {name}: {} (control: {}), {}/{} windows, {:.2}% of work, ETA {}, status written {age}s ago",
                s["state"].as_str().unwrap_or("?"), run::control_state(&dir), s["units_done"], s["units_total"],
                100.0 * s["fraction_done"].as_f64().unwrap_or(0.0), run::hms(s["eta_seconds"].as_f64().unwrap_or(-1.0)));
            Ok(())
        }
        CensusCmd::Pause { name } => {
            let name = control_target(name);
            let dir = existing_run_dir(&name)?;
            run::set_control(&dir, "pause")?;
            match run::running_pid(&dir) {
                Some(pid) => println!("pausing census {name} (pid {pid}) after the current window (the GPU is released while paused); `c2 census resume` continues it"),
                None => println!("census {name} is not running; when it next starts it waits, paused, until `c2 census resume --name {name}`"),
            }
            Ok(())
        }
        CensusCmd::Resume { name } => {
            let name = control_target(name);
            let dir = existing_run_dir(&name)?;
            let was_paused = run::control_state(&dir) == "pause";
            run::set_control(&dir, "run")?;
            match run::running_pid(&dir) {
                Some(pid) => println!("resuming census {name} (pid {pid})"),
                None if was_paused => println!("census {name} is not running; it is no longer paused, so it runs as soon as it is started"),
                None => println!("census {name} is not running and was not paused"),
            }
            Ok(())
        }
        CensusCmd::Run { smoke, name } => {
            let dir = run::run_dir(&name);
            // Fail before the plan is built. What keeps a second writer out is the lock that
            // `run::run` holds on the run (it also writes the run's pid file).
            if let Some(pid) = run::running_pid(&dir) {
                anyhow::bail!("census {name} is already running (pid {pid})");
            }
            let cfg = run_settings(&name, &dir, smoke)?;
            let store = c2_data::Store::open(&store_dir())?;
            let log = census_log(&dir, None);
            let plan = census_plan(&store, cfg, &name, &dir, &log)?;
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let s2 = stop.clone();
            ctrlc::set_handler(move || {
                eprintln!("stopping after the current window...");
                s2.store(true, std::sync::atomic::Ordering::Relaxed);
            })?;
            run::run(&store, &plan, &dir, &stop, &log)
        }
    }
}

/// The census a control command (status, stop, pause, resume) acts on: `--name`, else the
/// census running now if there is exactly one (a census `c2 update` started is not the default
/// until it finishes), else the default census (census/LATEST).
fn control_target(name: Option<String>) -> String {
    if let Some(name) = name {
        return name;
    }
    let running = c2_census::run::running_runs();
    if let [one] = &running[..] {
        return one.clone();
    }
    let latest = c2_census::run::latest_run();
    if !running.is_empty() {
        eprintln!("{} censuses are running ({}), so this acts on the default census, {latest}; use --name to choose", running.len(), running.join(", "));
    }
    latest
}

/// `<data dir>/census/<name>`, which must exist.
fn existing_run_dir(name: &str) -> Result<std::path::PathBuf> {
    let dir = c2_census::run::run_dir(name);
    anyhow::ensure!(dir.is_dir(), "there is no census {name} ({} does not exist)", dir.display());
    Ok(dir)
}

/// The settings `c2 census run` uses: an existing run continues with its own (its manifest's);
/// a new one gets the full grid, or with `--smoke` the reduced one.
fn run_settings(name: &str, dir: &std::path::Path, smoke: bool) -> Result<c2_census::grid::Config> {
    use c2_census::grid::Config;
    let new = if smoke { Config::smoke() } else { Config::full() };
    if !dir.join("manifest.json").exists() {
        return Ok(new);
    }
    let meta = c2_census::meta::RunMeta::load(dir)?;
    anyhow::ensure!(
        !smoke || meta.config_text == new.to_text(),
        "census {name} already exists with other settings than --smoke's ({}); --smoke only chooses the settings of a new run, so leave it out to continue {name} with its own, or choose a new --name",
        meta.config_text
    );
    Ok(meta.cfg)
}

/// The log of a census run, shared by `c2 census run` and `c2 update`: each message, stamped
/// with the local time, is printed and appended to `<dir>/run.log` (and to `also`, if given).
/// `analyze::run_history` reads run.log: a "plan:" line opens a session (see `census_plan`),
/// and the time stamps after it measure how long the census computed.
fn census_log(dir: &std::path::Path, also: Option<std::path::PathBuf>) -> impl Fn(&str) + Sync + use<> {
    let files: Vec<std::path::PathBuf> = std::iter::once(dir.join("run.log")).chain(also).collect();
    move |msg: &str| {
        use std::io::Write;
        let line = format!("[{}] {msg}", hms_now());
        println!("{line}");
        for path in &files {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "{line}");
            }
        }
    }
}

/// Builds the plan of a census about to run (or resume) in `dir` and logs its "plan:" line,
/// which opens a session in run.log.
fn census_plan(store: &c2_data::Store, cfg: c2_census::grid::Config, name: &str, dir: &std::path::Path, log: &dyn Fn(&str)) -> Result<c2_census::run::Plan> {
    std::fs::create_dir_all(dir)?;
    let t0 = Instant::now();
    let plan = c2_census::run::Plan::new(store, cfg)?;
    let units: usize = plan.tasks.iter().map(|t| t.windows.len()).sum();
    let sizes: Vec<usize> = plan.universes.iter().map(|u| u.len()).collect();
    log(&format!("plan: {} stocks in the variable, {} windows (universe {}..{} stocks), {} tasks, {} units; built in {:.1}s",
        plan.var.n, plan.windows.len(), sizes.iter().min().unwrap_or(&0), sizes.iter().max().unwrap_or(&0),
        plan.tasks.len(), units, t0.elapsed().as_secs_f64()));
    if c2_census::run::control_state(dir) == "pause" {
        log(&format!("census {name} is paused: it waits, without the GPU, until `c2 census resume --name {name}`"));
    }
    Ok(plan)
}

/// Gives every census run without its own copy of its stock list (the store's symbols.tsv,
/// which the stock numbers in its results refer to) a copy, where the store still holds that
/// list, and prints what happened to each run.
fn snapshot_symbols() -> Result<()> {
    use c2_census::meta::Snapshot;
    let done = c2_census::meta::snapshot_symbols_for_all_runs()?;
    if done.is_empty() {
        println!("  (no census runs)");
    }
    for (name, outcome) in done {
        let what = match outcome {
            Snapshot::Written => "written: saved a copy of the store's symbols.tsv, which is still the one it was made with",
            Snapshot::AlreadyThere => "already there: it has its own copy",
            Snapshot::Unrecoverable => "unrecoverable: it has no copy, and the store has been rebuilt since it ran, so its results can't be matched to stocks",
        };
        println!("  {name:<32} {what}");
    }
    Ok(())
}

/// Local time of day for log lines. The UTC offset comes from `date +%z` once (std has no time
/// zones); falls back to UTC.
fn hms_now() -> String {
    static OFFSET: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
    let offset = *OFFSET.get_or_init(|| {
        let z = std::process::Command::new("date").arg("+%z").output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();
        let z = z.trim();
        let sign = if z.starts_with('-') { -1 } else { 1 };
        let digits: String = z.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() == 4 {
            sign * (digits[..2].parse::<i64>().unwrap_or(0) * 3600 + digits[2..].parse::<i64>().unwrap_or(0) * 60)
        } else {
            0
        }
    });
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let local = secs as i64 + offset;
    format!("{:02}:{:02}:{:02}", (local / 3600).rem_euclid(24), (local / 60).rem_euclid(60), local.rem_euclid(60))
}
