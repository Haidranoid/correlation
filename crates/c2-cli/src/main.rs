//! `c2`: correlation2 command line.

use std::time::Instant;

mod update;

use anyhow::Result;
use c2_engine::{ColumnStats, cpu, pearson, reference_r, synth};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "c2", about = "correlation2: stock lead-lag census and holdout testing")]
struct Cli {
    /// Price store to use: a name in <data dir>/stores, or a path (default: the store named in
    /// stores/CURRENT; the C2_STORE environment variable does the same).
    #[arg(long, global = true)]
    store: Option<String>,
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
    /// Build a price store from Stooq bulk zips (default: v1's d_us_txt.zip) in <data dir>/stores
    /// and make it the default store. The stores already there are kept.
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
        /// The new store's name (default: stooq-<its last trading day>). A store of that name is
        /// replaced.
        #[arg(long)]
        name: Option<String>,
    },
    /// Ingest new zips from <data dir>/inbox into a new default store, rerun the census if the data
    /// changed, and append today's picks to the forward log. Run on a schedule by `c2 install-timer`.
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
    /// The price stores: list them, or choose the default one.
    Store {
        #[command(subcommand)]
        action: StoreCmd,
    },
    /// The correlation census: run, check on, pause or resume it, or validate the engine.
    Census {
        #[command(subcommand)]
        action: CensusCmd,
    },
}

#[derive(Subcommand)]
enum StoreCmd {
    /// List the price stores in <data dir>/stores: last trading day, symbols, rows, size.
    List,
    /// Make a store (a name in <data dir>/stores, or a path) the default one, which every command
    /// and the GUI then use.
    Use { name: String },
}

#[derive(Subcommand)]
enum CensusCmd {
    /// Run or resume a census (Ctrl-C stops after committing; run again to resume).
    Run {
        /// Reduced grid (move lengths 1, 5 and 21 days; 252- and 756-day correlation periods) for a
        /// quick end-to-end run.
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
    /// Pause after the correlation period in progress and release the GPU.
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
        /// Use pairs from every correlation period that ended before the holdout, not just the
        /// latest of each length.
        #[arg(long)]
        all_windows: bool,
        /// Walk forward: treat each year from this year's start month onward as the future in
        /// turn (one-year holdouts ending at the last data day), and print a table by year.
        #[arg(long)]
        walk_forward: bool,
        /// Signal family: `census` (stored lead-lag pairs; B's move length equals --rebalance),
        /// `peers` (peer momentum) or `reversal` (own-stock short-term reversal, a positive control).
        #[arg(long, default_value = "census")]
        family: String,
        /// Trading days between rebalances (default: 1 for census pairs, 21 for peers).
        #[arg(long)]
        rebalance: Option<usize>,
        /// Peer momentum: peers per stock, return lookback (days), correlation period (days).
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
    /// A stock's lead-lag relationships: who it leads and follows, at what offset, and by how much.
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
    /// Replication test: do stored pairs keep their sign in the next correlation period of the same
    /// length (the one that starts after theirs ends)?
    Replicate {
        #[arg(long, default_value_t = c2_census::run::latest_run())]
        name: String,
    },
    /// Stop a running census cleanly (it finishes the correlation period in progress); `run`
    /// resumes it.
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
    /// store in use still holds. `c2 ingest`, `c2 update` and `c2 store use` do this before they
    /// change the default store.
    SnapshotSymbols,
}

fn main() -> Result<()> {
    #[cfg(feature = "cuda")]
    c2_engine::gpu::reexec_with_cuda_libraries();
    let cli = Cli::parse();
    if let Some(store) = &cli.store {
        // These two make a new store the default and work on that; a chosen one would be ignored.
        match cli.cmd {
            Cmd::Ingest { .. } => anyhow::bail!("`c2 ingest` builds a new store, so --store doesn't apply (--name names the new store)"),
            Cmd::Update { .. } => anyhow::bail!("`c2 update` works on the default store, so --store doesn't apply (`c2 store use NAME` changes the default)"),
            _ => c2_data::stores::choose(store)?,
        }
    }
    match cli.cmd {
        Cmd::Bench { n, rows, no_cpu } => bench(n, &rows, !no_cpu),
        Cmd::Ingest { zips, markets, until, name } => ingest(zips, &markets, until, name),
        Cmd::Update { zips, smoke, no_census } => update::update(update::UpdateOptions { zips, smoke, no_census }),
        Cmd::ForwardReport => update::forward_report(),
        Cmd::InstallTimer { enable } => update::install_timer(enable),
        Cmd::DataCheck => data_check(),
        Cmd::Store { action } => store(action),
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
    println!("{n} stocks; times are per all-pairs product (one offset); error = max |r - f64 reference| over {} pairs\n", pairs.len());

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
    c2_data::store_dir()
}

fn ingest(mut zips: Vec<std::path::PathBuf>, markets: &[String], until: Option<String>, name: Option<String>) -> Result<()> {
    use anyhow::Context;
    if zips.is_empty() {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from).unwrap_or_default();
        zips.push(home.join("programming/correlation/data/stooq/d_us_txt.zip"));
    }
    let paths: Vec<&std::path::Path> = zips.iter().map(|p| p.as_path()).collect();
    let markets: Vec<&str> = markets.iter().map(|s| s.as_str()).collect();
    let until = until.map(|u| c2_data::parse_date(&u)).transpose()?;
    // Census results hold store ids, which another store numbers differently: first give every
    // run without one a copy of the symbol table its ids refer to (as `c2 update` does).
    println!("Stock lists of the census runs, saved before the default store changes:");
    snapshot_symbols().context("could not save the stock lists of the existing census runs, so no store was built")?;
    let before = c2_data::stores::default_name();
    let b = c2_data::stores::build_stooq(&paths, &markets, until, name.as_deref())?;
    let r = &b.report;
    println!("Ingested {} files ({} empty) into {}: {} rows in {:.1}s", r.files, r.empty, b.dir.display(), r.rows, r.seconds);
    for (folder, n) in &r.by_folder {
        println!("  {folder:<22} {n:>6} files");
    }
    println!(
        "Store {} (last trading day {}{}) is now the default store{}.",
        b.name,
        c2_data::date_string(b.last_day),
        if b.replaced { "; it replaced the store of that name" } else { "" },
        match before {
            Some(old) if old != b.name => format!(", in place of {old}, which is kept"),
            _ => String::new(),
        }
    );
    if std::env::var_os("C2_STORE").is_some() {
        println!("Note: C2_STORE is set, so commands run with it still use the store it names.");
    }
    Ok(())
}

fn store(action: StoreCmd) -> Result<()> {
    match action {
        StoreCmd::List => {
            let (in_use, how) = c2_data::stores::selected();
            let default = c2_data::stores::default_name();
            let stores = c2_data::stores::list();
            println!("Price stores in {}:", c2_data::stores::stores_dir().display());
            if stores.is_empty() {
                println!("  (none; `c2 ingest` builds one)");
            } else {
                println!("  {:<24} {:>10} {:>8} {:>12} {:>8}", "name", "last day", "symbols", "rows", "size");
            }
            for s in &stores {
                let mark = if default.as_deref() == Some(s.name.as_str()) { "  (default)" } else { "" };
                match c2_data::Store::open(&s.dir) {
                    Ok(st) => {
                        let bytes: u64 = ["prices.bin", "symbols.tsv"].iter().filter_map(|f| std::fs::metadata(s.dir.join(f)).ok()).map(|m| m.len()).sum();
                        println!("  {:<24} {:>10} {:>8} {:>12} {:>5} MB{mark}", s.name, c2_data::date_string(c2_data::stores::last_day(&st)), st.symbols.len(), st.rows(), bytes / 1_000_000);
                    }
                    Err(e) => println!("  {:<24} can't be opened: {e:#}{mark}", s.name),
                }
            }
            println!("In use: {} ({}).", in_use.display(), how.describe());
            Ok(())
        }
        StoreCmd::Use { name } => {
            use anyhow::Context;
            c2_data::stores::find(&name)?;
            println!("Stock lists of the census runs, saved before the default store changes:");
            snapshot_symbols().context("could not save the stock lists of the existing census runs, so the default store was not changed")?;
            let dir = c2_data::stores::set_default(&name)?;
            println!("The default store is now {} ({}).", c2_data::stores::name(&dir), dir.display());
            Ok(())
        }
    }
}

fn data_check() -> Result<()> {
    use c2_data::{Calendar, Kind, Panel, Store};
    let (dir, how) = c2_data::stores::selected();
    let store = Store::open(&dir)?;
    println!("Store {} ({}, {})", c2_data::stores::name(&dir), how.describe(), dir.display());
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
            println!("Oracle correlation period: {} stocks ({} not in store: {:?}), {} days, {} pairs", r.universe, r.missing.len(), r.missing, r.days, r.pairs);
            println!("  max |r - oracle|, exact f64 path:       {:.2e}", r.max_err_exact);
            println!("  max |r - oracle|, GPU INT8 census path: {:.2e}", r.max_err_gpu_int8);
            println!("  max |r - oracle|, CPU INT8 (AVX-VNNI):  {:.2e}", r.max_err_cpu_int8);
            println!("  max |GPU INT8 - CPU INT8|:              {:.2e}", r.max_cpu_gpu_diff);
            println!("  oracle's 50 strongest offset-1 pairs screened in by the GPU pass: {}/{}", r.top50_found, r.top50_total);
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
            let gaps = gaps(&plan);
            if walk_forward {
                let (y0, mo, d) = c2_data::civil_from_days(c2_data::parse_date(&start)?);
                let last = *plan.var.days.last().unwrap();
                println!("Walk-forward: each year treated as the future, predicted only from census correlation periods that ended before it.");
                for line in gaps.lines("pairs with it are left out", "pairs with them are left out") {
                    println!("{line}.");
                }
                println!("{:<24} {:>7} {:>9} {:>9} {:>10} {:>10} {:>10} {:>8} {:>8} {:>9} {:>8}", "holdout", "pairs", "SPY", "QQQ", "picks", "random med", "p (0 bps)", "hit %", "IC t", "LS bps/d", "LS t");
                let mut beat = 0;
                let mut years = 0;
                // Years whose holdout failed: (period, error). Each is shown in the table.
                let mut failed: Vec<(String, String)> = Vec::new();
                let mut no_pairs = 0;
                let (mut ic_ts, mut ls_ts, mut ls_bps, mut ls_turn): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = Default::default();
                for y in (2006..=y0).rev().collect::<Vec<_>>().into_iter().rev() {
                    let s = c2_data::days_from_civil(y, mo, d);
                    let e = c2_data::days_from_civil(y + 1, mo, d);
                    if e > last + 7 && y != y0 {
                        continue;
                    }
                    let opt = c2_census::holdout::Options { family: fam, rebalance, start: s, end: e, picks, costs_bps: vec![0.0], random_runs: random, cash: 10_000.0, all_windows };
                    let r = match c2_census::holdout::holdout(&store, &plan, &dir, &opt) {
                        Ok(r) => r,
                        Err(err) => {
                            let (period, msg) = (format!("{}..{}", c2_data::date_string(s), c2_data::date_string(e - 1)), format!("{err:#}"));
                            match failed.iter().find(|(_, m)| *m == msg) {
                                Some((first, _)) => println!("{period:<24} failed: the same error as {first}"),
                                None => println!("{period:<24} failed: {msg}"),
                            }
                            failed.push((period, msg));
                            continue;
                        }
                    };
                    let period = format!("{}..{}", c2_data::date_string(r.first_day), c2_data::date_string(r.last_day));
                    if r.pairs == 0 {
                        no_pairs += 1;
                        println!("{period:<24} no census pairs: none with B moving over {} from correlation periods that ended before it{}", days_text(rebalance), holdout_left_out(&r));
                        continue;
                    }
                    let pct = |v: f64| 100.0 * (v / 10_000.0 - 1.0);
                    let bench = |i: usize| r.benchmarks.get(i).map(|b| pct(b.final_value)).unwrap_or(f64::NAN);
                    let (_, strat, med, p) = r.strategy[0];
                    years += 1;
                    beat += usize::from(p < 0.1);
                    println!("{:<24} {:>7} {:>8.1}% {:>8.1}% {:>9.1}% {:>9.1}% {:>10.3} {:>7.1}% {:>8.2} {:>9.2} {:>8.2}{}",
                        period, r.pairs, bench(0), bench(1), pct(strat), pct(med), p, 100.0 * r.hit_rate, r.ic_t, 1e4 * r.ls_mean, r.ls_t, holdout_left_out(&r));
                    ic_ts.push(r.ic_t);
                    ls_ts.push(r.ls_t);
                    ls_bps.push(1e4 * r.ls_mean);
                    ls_turn.push(r.ls_turnover);
                }
                if years == 0 {
                    let empty = if no_pairs > 0 { format!(", {no_pairs} had no census pairs from correlation periods that ended before them") } else { String::new() };
                    match failed.first() {
                        Some((period, msg)) => anyhow::bail!("no year of the walk-forward could be tested ({} failed{empty}); the first failure, {period}: {msg}", failed.len()),
                        None => anyhow::bail!("no year of the walk-forward could be tested: none had census pairs from correlation periods that ended before it"),
                    }
                }
                if !failed.is_empty() || no_pairs > 0 {
                    let mut why = Vec::new();
                    if !failed.is_empty() {
                        why.push(format!("{} failed", failed.len()));
                    }
                    if no_pairs > 0 {
                        why.push(format!("{no_pairs} had no census pairs"));
                    }
                    println!("{} year(s) could not be tested ({}, above); the summary covers the other {years}.", failed.len() + no_pairs, why.join(", "));
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
            for line in gaps.lines("pairs with it are left out", "pairs with them are left out") {
                println!("{line}.");
            }
            if r.periods_unavailable > 0 || r.pairs_skipped > 0 {
                let mut parts = Vec::new();
                if r.periods_unavailable > 0 {
                    parts.push(format!("{} correlation period(s) that ended before the holdout but aren't covered by today's price data{}", r.periods_unavailable,
                        if all_windows { "" } else { " (for each length the latest covered one is used)" }));
                }
                if r.pairs_skipped > 0 {
                    parts.push(format!("{} stored pair(s) with a stock that is no longer in today's data", r.pairs_skipped));
                }
                println!("Left out: {}.", parts.join("; "));
            }
            if r.pairs == 0 {
                // Census pairs only: nothing was predicted, so there is nothing to score.
                println!("Family {family}, rebalanced every {rebalance} trading day(s): no census pairs. No stored relationship with B moving over {} comes from a correlation period that ended before the holdout, so nothing was predicted.",
                    days_text(rebalance));
            } else {
                println!("Family {family}, rebalanced every {rebalance} trading day(s): {} pairs from {} correlation period(s) that ended before the holdout; {} stocks can be picked",
                    r.pairs, r.windows_used, r.candidates);
                println!("Picks beat the market over their holding period: {:.1}% of {} picks (coin flip 50%)", 100.0 * r.hit_rate, r.hits);
                println!("Rank correlation of prediction vs outcome (IC): mean {:+.4}, t = {:+.2}", r.ic_mean, r.ic_t);
                println!("Long-short top/bottom 10%: {:.2} bps/day gross, Sharpe {:.2}, t = {:.2}, book turnover {:.0}% per rebalance",
                    1e4 * r.ls_mean, r.ls_sharpe, r.ls_t, 100.0 * r.ls_turnover);
            }
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
            use c2_census::explain::{self, Check};
            let days = |d: usize| if d == 1 { "1 day".to_string() } else { format!("{d} days") };
            let tdays = |d: usize| if d == 1 { "1 trading day".to_string() } else { format!("{d} trading days") };
            let store = c2_data::Store::open(&store_dir())?;
            let t0 = Instant::now();
            // The run's own correlation periods, stocks and tasks: `window` is the run's period id.
            let meta = c2_census::meta::RunMeta::load(&run::run_dir(&name))?;
            let plan = run::Plan::for_run(&store, &meta)?;
            let built = t0.elapsed().as_secs_f64();
            let end = match end { Some(d) => Some(c2_data::parse_date(&d)?), None => None };
            // Only a period today's data covers can be explained; with no --end, the latest one.
            let (window, passed_over) = explain::find_period(&plan, &meta, period, end)?;
            let req = explain::Request { leader: leader.to_uppercase(), follower: follower.to_uppercase(), h1, h2, lag: offset, window, threshold: None };
            let t1 = Instant::now();
            let e = explain::explain_with(&store, &plan, &meta, &req)?;
            let took = t1.elapsed().as_secs_f64();
            let (a, b) = (e.a.ticker.as_str(), e.b.ticker.as_str());
            println!("(census {name}: census variable for all stocks built in {built:.1}s; this explanation took {:.1} ms)", 1000.0 * took);
            let qualified = |s: &explain::DaySteps| if s.in_universe { "qualified" } else { "did NOT qualify" };
            let kind = if plan.windows[window].length == 0 { format!("full-history correlation period ({} trading days)", e.period_days) } else { format!("{}-day correlation period", e.period_days) };
            if let Some(note) = &passed_over {
                println!("note: {note}");
            }
            println!("{kind} {} .. {}: {} stocks qualified; {a} {}, {b} {}", c2_data::date_string(e.period_first), c2_data::date_string(e.period_last), e.universe, qualified(&e.a), qualified(&e.b));
            if let Some(note) = &e.variable_note {
                println!("note: {note}");
            }
            if e.same_day {
                println!("offset 0 is the same-day reference: {b}'s move over {} ends the same day as {a}'s over {}. The census computes it to compare its later offsets with; it is not a prediction.", days(h2), days(h1));
            }
            match &e.census_computed {
                Ok(()) => println!("the census computed this matrix: {} stocks, each against every other, over {} observations", e.matrix.stocks, e.matrix.obs),
                Err(why) => println!("the census never computed this matrix: {why}"),
            }
            let ending = if offset == 0 { "the same day as".to_string() } else { format!("{} before the end of", tdays(offset)) };
            println!("{} observations ({h1}-day moves of {a} ending {ending} {h2}-day moves of {b}):", e.n);
            // A move over days without a price has no % (the GUI says "no price" too).
            let pct = |m: f64| if m.is_finite() { format!("{:+.2}%", 100.0 * m.exp_m1()) } else { "no price".to_string() };
            for o in e.obs.iter().take(3) {
                println!("  {} {} -> {}: {} (x {:+.3});  {} {} -> {}: {} (y {:+.3})", e.a.ticker, c2_data::date_string(e.days[o.a_end - h1]), c2_data::date_string(e.days[o.a_end]),
                    pct(o.a_move), o.x, e.b.ticker, c2_data::date_string(e.days[o.b_end - h2]), c2_data::date_string(e.days[o.b_end]), pct(o.b_move), o.y);
            }
            println!("r = {:+.6}, z = {:+.3}; r of raw % moves = {:+.4}", e.r, e.z, e.r_raw);
            match &e.stored {
                Some(s) => {
                    let test = if s.significant { ", passed the significance test" } else if offset > 0 { ", did not pass the significance test" } else { "" };
                    println!("stored by the census: r = {:+.6} (difference {:.2e}), screening r {:+.4}, z {:+.3}{}{test}", s.record.r, (s.record.r as f64 - e.r).abs(), s.record.r_int8, s.record.z,
                        if s.top { ", among the strongest of its matrix" } else { "" });
                }
                None => match e.never_computed() {
                    Some(why) => println!("not stored: the census never computed this pair, because {why}. The r above is what it would have been."),
                    None => println!("not stored: the census computed this pair but kept only the strongest {} of its matrix and any that passed the significance test", meta.cfg.keep_real),
                },
            }
            match &e.check {
                // Under a note (values recomputed today can differ, e.g. a stock gone from today's
                // data), a disagreement is expected, not a fault.
                Check::Record(c) => {
                    let verdict = match (c.agrees(), &e.variable_note) {
                        (true, None) => "agree".to_string(),
                        (true, Some(_)) => "agree (though with today's data they could differ slightly; see the note above)".to_string(),
                        (false, Some(_)) => "differ slightly, as expected with today's data (see the note above)".to_string(),
                        (false, None) => "DIFFER".to_string(),
                    };
                    println!("checked against the census's record: {} observations here, {} stored; sums of x, y, x^2, y^2 and xy {verdict} (largest relative difference {:.1e}; the record keeps about 7 digits)",
                        c.obs, c.stored_obs, c.max_rel_diff);
                }
                Check::Summary(c) => println!("nothing is stored for this pair to check against; the matrix's summary line {}: {} stocks and {} observations there, {} and {} here",
                    if c.agrees() { "agrees" } else { "DIFFERS" }, c.stored_stocks, c.stored_obs, c.stocks, c.obs),
                Check::Nothing => println!("nothing stored to check against: the census never computed this matrix"),
            }
            if e.matrix.found {
                let cut = match e.matrix.fdr_r { Some(v) => format!("{v:.4}"), None if offset == 0 => "none (no significance test at offset 0)".into(), None => "none (no pair passed)".into() };
                println!("matrix: pairs at |z|>=3/4/5: real {:?}, chance (sign-flip) {:?}; significance cut-off |r| {cut}", e.matrix.real, e.matrix.null);
            }
            println!("chance alone would give about {:.1} of {:.0} pairs at |z| >= {:.2}", e.chance_analytic, e.pairs_tested, e.z.abs());
            match &e.move_size {
                Some(s) => println!("size: {b}'s move relative to the market per 1% move of {a}'s relative to the market = r {:+.6} x ({:.2}% / {:.2}%) = {:+.4}, over {} observations",
                    e.move_size_r, 100.0 * s.sd_b, 100.0 * s.sd_a, s.per_pct, s.pairs),
                None => println!("size: fewer than {} observations have both moves, so there is no size", c2_census::moves::MIN_PAIRS),
            }
            // How forecast::rel_path applies it: per leader move, spread over B's predicted days,
            // only the days inside the forecast counting, on log changes.
            match (&e.forecast_use, &e.move_size) {
                (Ok(()), Some(s)) => {
                    let (m, step) = (s.per_pct, h1.max(h2));
                    let unless = if e.stored.as_ref().is_some_and(|s| s.significant) { "" } else { "; with \"significant relationships only\" it leaves this one out, since it didn't pass the significance test" };
                    println!("how the Forecast uses it (on as-of days when this is the most recent correlation period of {a} -> {b} at these move lengths and offset; of periods ending on the same day it takes the longest, and no longer one of this relationship ends that day{unless}): each of {a}'s recent {h1}-day moves relative to the market (the latest ending on the as-of day, then every {} before it) predicts {b}'s move relative to the market over the {} ending {} after it: a move of X% predicts (1 + X/100)^{m:.4} - 1 for {b} (the % per 1% applies to log changes, so this is about X x {m:.4} only for small moves), spread evenly over those days. Only the days that fall in the forecast's days ahead count, and each day's forecast is the average over the relationships covering it. E.g. a typical move of {a}, {:+.2}%, predicts {:+.2}% for {b}.",
                        days(step), days(h2), tdays(offset), 100.0 * s.sd_a.exp_m1(), 100.0 * (m * s.sd_a).exp_m1());
                }
                (Err(why), _) => println!("not used by the Forecast: {why}"),
                (Ok(()), None) => {}
            }
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
            // The run's own census variable, from its reading plan, which also says what today's
            // data can't reproduce of the run.
            let meta = c2_census::meta::RunMeta::load(&dir)?;
            let (data, gaps) = {
                let plan = run::Plan::for_run(&store, &meta)?;
                (c2_census::forecast::prepare(&store, &plan.var, &dir, &t)?, gaps(&plan))
            };
            let f = c2_census::forecast::forecast_from(&data, as_of, &c2_census::forecast::Options {
                significant_only: significant,
                horizon: days,
                max_age: max_age_years.map(|y| (y * 252.0) as usize),
            })?;
            println!("{t} as of {}: ${:.2}; {} relationships where {t} follows another stock{}; typical daily move relative to the market {:.2}%", c2_data::date_string(f.as_of),
                f.last_price, f.relationships, if significant { " (significant only)" } else { "" }, 100.0 * f.daily_sd);
            for line in gaps.lines("relationships with it are left out", "relationships with them are left out") {
                println!("{line}.");
            }
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
                println!("  {} offset {}d (moves {}d -> {}d, r {:+.3}{}): {}'s moves landing in these days {:+.2}% -> {t} {:+.2}% (at {:+.3}% per 1%, applied to log changes; for small moves Y ≈ X x M); share {:+.3} pts   [{}]",
                    c.leader, c.lag, c.h1, c.h2, c.r, if c.significant { ", significant" } else { "" }, c.leader, c.leader_move_in_horizon_pct, c.implied_pct, c.move_per_pct, c.share_pct, c.window);
                // X is the capped move already; say when a move was capped (api_numbers.md, polish).
                let cap = c2_census::forecast::MOVE_CAP;
                match c.clipped_moves {
                    0 => {}
                    1 => println!("      capped: 1 of {}'s {}-day moves was far outside anything seen in this correlation period (typical {:.2}%), so it counts as {cap:.0} typical moves",
                        c.leader, c.h1, c.leader_typical_move_pct),
                    n => println!("      capped: {n} of {}'s {}-day moves were far outside anything seen in this correlation period (typical {:.2}%), so each counts as {cap:.0} typical moves",
                        c.leader, c.h1, c.leader_typical_move_pct),
                }
            }
            println!("A leader's moves \"landing in these days\" are its market-relative moves whose predicted days fall in the days ahead; a move whose predicted days run past the last day counts in proportion to the days inside. The implied move is (1 + X/100)^M - 1: the % per 1% applies to log changes, so it is about X x M only for small moves.");
            if f.contributions.iter().any(|c| c.clipped_moves > 0) {
                let cap = c2_census::forecast::MOVE_CAP;
                println!("A leader's move more than {cap:.0} of its typical moves in a relationship's correlation period is counted as {cap:.0}: the census clips at {cap:.0}, so it never measured the relationship beyond that.");
            }
            match &f.skill {
                Some(s) => println!("\nTrack record: the same forecast made every 21 trading days in the past ({} forecasts, each {} trading days ahead), against {t}'s actual move relative to the market: \
                    direction right {:.1}% of the time (of {} with a direction); predicted vs actual correlation {:+.3} (slope {:+.2}); average size predicted {:.2}% vs actual {:.2}%.",
                    s.forecasts, s.horizon, 100.0 * s.hit_rate, s.directional, s.corr, s.slope, s.mean_abs_predicted_pct, s.mean_abs_actual_pct),
                None => println!("\nTrack record: too few past forecasts to judge."),
            }
            println!("\nRelative to the market means a stock's move minus its usual share of the market's move (beta x market), which is what the census correlates.");
            println!("Each relationship's % per 1% is the number `c2 census stock` shows for it; the forecast uses rolling correlation periods only, never the full-history one.");
            Ok(())
        }
        CensusCmd::Stock { ticker, name, significant, limit } => {
            let store = c2_data::Store::open(&store_dir())?;
            let t = ticker.trim().to_uppercase();
            let dir = run::run_dir(&name);
            // The run's own census variable, from its reading plan, which also says what today's
            // data can't reproduce of the run.
            let meta = c2_census::meta::RunMeta::load(&dir)?;
            let (rows, gaps) = {
                let plan = run::Plan::for_run(&store, &meta)?;
                (c2_census::analyze::stock_profile_with(&store, &plan.var, &dir, &t, significant)?, gaps(&plan))
            };
            let full_history: std::collections::HashSet<u32> = meta.windows.iter().filter(|w| w.length == 0).map(|w| w.id as u32).collect();
            // Of a relationship's periods that end on the same day, the forecast uses the longest one
            // it can (a rolling period with a size, `forecast::rel_order`); a shorter one's multiplier
            // is never applied. Rows listed here: all stored ones, or with --significant the
            // significant ones, as `c2 census forecast --significant` sees them.
            let period_of = |r: &c2_census::analyze::ProfileRow| meta.windows.get(r.window_id as usize).map(|w| (w.length, w.last));
            let relationship = |r: &c2_census::analyze::ProfileRow, last: i32| (r.leads, r.partner.clone(), r.h1, r.h2, r.lag, last);
            let mut longest: std::collections::HashMap<(bool, String, u8, u8, u16, i32), usize> = std::collections::HashMap::new();
            for r in rows.iter().filter(|r| r.move_per_pct.is_some()) {
                if let Some((length, last)) = period_of(r).filter(|(length, _)| *length > 0) {
                    let e = longest.entry(relationship(r, last)).or_default();
                    *e = (*e).max(length);
                }
            }
            let longer_used = |r: &c2_census::analyze::ProfileRow| period_of(r).is_some_and(|(length, last)| length > 0 && longest.get(&relationship(r, last)).is_some_and(|&l| l > length));
            let sig = rows.iter().filter(|r| r.significant).count();
            let scope = if gaps.periods == 0 { "across all correlation periods".to_string() } else { format!("across the {} correlation periods that today's data covers", gaps.of - gaps.periods) };
            if significant {
                println!("{t}: {sig} lead-lag relationships passed the significance test {scope}");
            } else {
                println!("{t}: {} stored lead-lag relationships {scope}, {sig} passed the significance test", rows.len());
            }
            for line in gaps.lines("rows with it show no size", "rows with one of them show no size") {
                println!("{line}.");
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
                    let period = if full_history.contains(&r.window_id) {
                        format!("full-history correlation period, {} days to {} (not used by the forecast)", r.period_days, r.period_end)
                    } else if longer_used(r) {
                        format!("correlation period {} days to {} (a longer period is used)", r.period_days, r.period_end)
                    } else {
                        format!("correlation period {} days to {}", r.period_days, r.period_end)
                    };
                    println!("  {a} -> {b:<6} offset {:>3} days: {size}; r {:+.3}, z {:+.1}; chance made {:.0} pairs with |z| >= {} in that period{}{words}   [{moves}; {period}, {} observations]",
                        r.lag, r.r, r.z, r.chance, r.chance_z, if r.significant { "  SIGNIFICANT" } else { "" }, r.obs);
                }
            }
            println!("\nr is the census correlation (market-adjusted, volatility-normalized returns); the % per 1% is that relationship in price terms:");
            println!("the follower's move relative to the market per 1% move of the leader relative to the market, in that correlation period.");
            println!("For rows from rolling correlation periods it is the multiplier `c2 census forecast{}` applies, unless a longer correlation period of the same relationship ends the same day:",
                if significant { " --significant" } else { "" });
            println!("of those the forecast uses the longest (the shorter ones say \"a longer period is used\"). It doesn't use the full-history period.");
            println!("The 'after ... rose' counts use plain price moves. 'Chance made N' is how many pairs at least that strong appeared in the same period");
            println!("when the relationship was scrambled; if N is not far below 1, the pair is likely noise.");
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
            let rep = c2_census::analyze::replication(&store, &plan, &dir)?;
            println!("Replication in the next correlation period of the same length ({:.0}s). 'Chance' rows are pairs picked from sign-flip null matrices.", t0.elapsed().as_secs_f64());
            for line in gaps(&plan).lines("records with it are left out", "records with them are left out") {
                println!("{line}.");
            }
            let mut left_out = Vec::new();
            if rep.records_unavailable > 0 {
                left_out.push(format!("{} stored record(s) whose correlation period, or the next one of its length, today's data doesn't cover ({} such rolling period(s))",
                    rep.records_unavailable, rep.periods_unavailable));
            }
            if rep.records_missing_stock > 0 {
                left_out.push(format!("{} stored record(s) with a stock no longer in today's data", rep.records_missing_stock));
            }
            if !left_out.is_empty() {
                println!("Left out: {}.", left_out.join("; "));
            }
            println!("{:>4} {:>4} {:>16} {:>7} {:>10} {:>11} {:>16} {:>14}", "h1", "h2", "offset", "kind", "pairs", "same sign", "mean |r| then", "mean r next*");
            for ((h1, h2, b, null), r) in &rep.table {
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
                    println!("asked census {name} (pid {pid}) to stop; it first commits the correlation period in progress. `c2 census run --name {name}` resumes.");
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
            println!("census {name}: {} (control: {}), {} of {} correlation-period computations, {:.2}% of the work, ETA {}, status written {age}s ago",
                s["state"].as_str().unwrap_or("?"), run::control_state(&dir), s["units_done"], s["units_total"],
                100.0 * s["fraction_done"].as_f64().unwrap_or(0.0), run::hms(s["eta_seconds"].as_f64().unwrap_or(-1.0)));
            Ok(())
        }
        CensusCmd::Pause { name } => {
            let name = control_target(name);
            let dir = existing_run_dir(&name)?;
            run::set_control(&dir, "pause")?;
            match run::running_pid(&dir) {
                Some(pid) => println!("pausing census {name} (pid {pid}) after the correlation period in progress (the GPU is released while paused); `c2 census resume` continues it"),
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
            // Every guard runs before anything is logged, so a refused start opens no session in
            // run.log (`refuse` notes it there instead). Fail before the plan is built if the run
            // is live; what keeps a second writer out is the lock `run::run` holds (it also
            // writes the run's pid file).
            if let Some(pid) = run::running_pid(&dir) {
                return Err(refuse(&dir, anyhow::anyhow!("census {name} is already running (pid {pid})")));
            }
            let cfg = run_settings(&name, &dir, smoke).map_err(|e| refuse(&dir, e))?;
            let store = c2_data::Store::open(&store_dir())?;
            let t0 = Instant::now();
            let plan = run::Plan::new(&store, cfg)?;
            let built = t0.elapsed().as_secs_f64();
            // The writer lock decides between starts made at the same moment, before anything
            // is logged; the loser is refused here.
            let lock = run::lock_run(&dir).map_err(|e| refuse(&dir, e))?;
            // The checks `run::run` makes before resuming (settings, price data, task list,
            // correlation periods and their stocks), made here, under the lock, so a refusal
            // comes before the plan line.
            run::resume_check(&dir, &plan, &store).map_err(|e| refuse(&dir, e))?;
            let log = census_log(&dir, None);
            log_plan(&plan, built, &name, &dir, &log)?;
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let s2 = stop.clone();
            ctrlc::set_handler(move || {
                eprintln!("stopping after the correlation period in progress...");
                s2.store(true, std::sync::atomic::Ordering::Relaxed);
            })?;
            run::run_locked(&store, &plan, &dir, lock, &stop, &log)
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

/// Logs the "plan:" line that opens a census session in run.log, for a plan built in `built`
/// seconds that has passed every guard (`run::resume_check`), so a refused attempt never opens
/// a session. Says so when the run is paused.
fn log_plan(plan: &c2_census::run::Plan, built: f64, name: &str, dir: &std::path::Path, log: &dyn Fn(&str)) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let units: usize = plan.tasks.iter().map(|t| t.windows.len()).sum();
    let sizes: Vec<usize> = plan.universes.iter().map(|u| u.len()).collect();
    log(&format!("plan: {} stocks in the variable, {} correlation periods ({}..{} stocks each), {} tasks, {} units; built in {built:.1}s",
        plan.var.n, plan.windows.len(), sizes.iter().min().unwrap_or(&0), sizes.iter().max().unwrap_or(&0), plan.tasks.len(), units));
    if c2_census::run::control_state(dir) == "pause" {
        log(&format!("census {name} is paused: it waits, without the GPU, until `c2 census resume --name {name}`"));
    }
    Ok(())
}

/// Notes a refused start of the census in `dir` in its run.log (if the run exists) and returns
/// the error. The line deliberately has no "[HH:MM:SS]" stamp at its start: `analyze::run_history`
/// counts computing time from stamped lines, and a stamped refusal would stretch the last session
/// to the moment of the refusal.
fn refuse(dir: &std::path::Path, err: anyhow::Error) -> anyhow::Error {
    if dir.is_dir()
        && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("run.log"))
    {
        use std::io::Write;
        let _ = writeln!(f, "refused {}: {err:#}", local_date_time());
    }
    err
}

/// "1 day" or "N days".
fn days_text(n: usize) -> String {
    if n == 1 { "1 day".into() } else { format!("{n} days") }
}

/// A walk-forward row's note of what its holdout left out ("" when nothing).
fn holdout_left_out(r: &c2_census::holdout::Report) -> String {
    let mut parts = Vec::new();
    if r.periods_unavailable > 0 {
        parts.push(format!("{} uncovered correlation period(s)", r.periods_unavailable));
    }
    if r.pairs_skipped > 0 {
        parts.push(format!("{} pair(s) with a stock no longer in today's data", r.pairs_skipped));
    }
    if parts.is_empty() { String::new() } else { format!("   [left out: {}]", parts.join(", ")) }
}

/// What today's data can't reproduce of the run a reading plan (`Plan::for_run`) was built for.
struct Gaps {
    /// Correlation periods today's data doesn't cover exactly, of all the run's periods.
    periods: usize,
    of: usize,
    /// The run's stocks no longer in today's data (distinct; `Plan::missing_stocks`).
    stocks: usize,
    /// Why recomputed values may differ from the stored ones (`Plan::variable_note`).
    note: Option<String>,
}

fn gaps(plan: &c2_census::run::Plan) -> Gaps {
    let stocks: std::collections::HashSet<u32> = plan.missing_stocks.iter().flatten().copied().collect();
    Gaps { periods: plan.unavailable().0, of: plan.windows.len(), stocks: stocks.len(), note: plan.variable_note.clone() }
}

impl Gaps {
    /// The lines to print, `then_one` / `then_many` saying what happens to what involves the
    /// missing stock / stocks; none when the run is fully reproducible on today's data.
    fn lines(&self, then_one: &str, then_many: &str) -> Vec<String> {
        let mut out = Vec::new();
        match self.periods {
            0 => {}
            1 => out.push(format!("1 of the run's {} correlation periods isn't covered exactly by today's price data, so it is left out", self.of)),
            n => out.push(format!("{n} of the run's {} correlation periods aren't covered exactly by today's price data, so they are left out", self.of)),
        }
        match self.stocks {
            0 => {}
            1 => out.push(format!("1 of the run's stocks is no longer in today's data, so {then_one}")),
            n => out.push(format!("{n} of the run's stocks are no longer in today's data, so {then_many}")),
        }
        out.extend(self.note.clone());
        out
    }
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
            Snapshot::Unrecoverable => "unrecoverable: it has no copy, and the store in use is not the one it ran on, so its results can't be matched to stocks",
        };
        println!("  {name:<32} {what}");
    }
    Ok(())
}

/// Local time of day for log lines. The UTC offset comes from `date +%z` once (std has no time
/// zones); falls back to UTC.
fn hms_now() -> String {
    hms_of(local_seconds())
}

/// Local date and time, "YYYY-MM-DD HH:MM:SS".
fn local_date_time() -> String {
    let local = local_seconds();
    format!("{} {}", c2_data::date_string(local.div_euclid(86_400) as i32), hms_of(local))
}

fn hms_of(local: i64) -> String {
    format!("{:02}:{:02}:{:02}", (local / 3600).rem_euclid(24), (local / 60).rem_euclid(60), local.rem_euclid(60))
}

/// Seconds since 1970 in local time (see `hms_now`).
fn local_seconds() -> i64 {
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
    secs as i64 + offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gap_lines_agree_in_number() {
        let one = Gaps { periods: 1, of: 16, stocks: 1, note: None }.lines("pairs with it are left out", "pairs with them are left out");
        assert_eq!(one, [
            "1 of the run's 16 correlation periods isn't covered exactly by today's price data, so it is left out",
            "1 of the run's stocks is no longer in today's data, so pairs with it are left out",
        ]);
        let many = Gaps { periods: 4, of: 16, stocks: 22, note: Some("a note".into()) }.lines("pairs with it are left out", "pairs with them are left out");
        assert_eq!(many, [
            "4 of the run's 16 correlation periods aren't covered exactly by today's price data, so they are left out",
            "22 of the run's stocks are no longer in today's data, so pairs with them are left out",
            "a note",
        ]);
        assert!(Gaps { periods: 0, of: 16, stocks: 0, note: None }.lines("", "").is_empty());
    }
}
