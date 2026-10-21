//! `stockai-judge`: the two pieces of correlation2 that stockai needs, built against
//! correlation2's crates pinned at commit 9335126, without modifying correlation2.
//!
//! - `export`: writes `plan.var` of `Plan::new(&store, Config::full())` (the census variable,
//!   closes, dollar volumes, the market return, days and tickers) as `.npy` files for Python
//!   (stockai/docs/c2-export.md).
//! - `holdout`: correlation2's $10,000 holdout test (`c2 census holdout`), forked, with a
//!   `scores` family that grades a file of model scores. Nothing in the locked year
//!   (2024-06-28 .. 2025-06-27) or later is graded without `--locked`.
//! - `import-sharadar` and `check-sharadar`: Sharadar's bulk tables into a store of the same kind,
//!   with delistings and as-traded closes beside it (docs/sharadar.md), and the day-one checks.
//!   `export` and `holdout` find a store through C2_DATA_DIR (`$C2_DATA_DIR/store`), as before.
//! - `export-external`: correlation2's external series (weather, politics, news, ...) as census-style
//!   daily variables on the export's days, timed as correlation2 times them (docs/external.md).

mod check;
mod export;
mod external;
mod extras;
mod holdout;
mod npy;
mod sharadar;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use c2_data::{civil_from_days, date_string, days_from_civil, parse_date};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{Value, json};

use holdout::{DelistRule, DelistStats, Family, Matching, Options, PriceTest, RandomMatch, Report, ScoreMatrix};

/// The correlation2 commit this judge is built from (Cargo.toml pins the same revision).
pub const C2_COMMIT: &str = "93351262d1f155aa049b39198cc17c59f66e0867";

/// First day of the locked year (2024-06-28 .. 2025-06-27), graded only with `--locked`.
fn locked_from() -> i32 {
    days_from_civil(2024, 6, 28)
}

#[derive(Parser)]
#[command(name = "stockai-judge", version, about = "stockai's judge: correlation2's export and $10,000 holdout test (correlation2 at 9335126), with a scores family")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write the census variable and prices for Python (docs/c2-export.md): plan.var of
    /// Plan::new(store, Config::full()), as `c2 census holdout` builds it for a full census.
    Export {
        /// Output directory (default: <C2_DATA_DIR or ~/.local/share/correlation2>/export/full).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// The $10,000 holdout test, forked from `c2 census holdout`: trade $10,000 on the highest
    /// positive scores over a year the scores never saw, against SPY, QQQ and random guessers.
    Holdout(Box<HoldoutArgs>),
    /// Import Sharadar's bulk tables (stocks, funds, tickers, actions) into a correlation2 price store
    /// with close_unadj.f32, delist.tsv and source.json beside it (docs/sharadar.md). Export and
    /// grade it with C2_DATA_DIR set to the store's parent directory.
    ImportSharadar {
        /// Directory holding the four tables as .csv, .csv.gz or .zip (scripts/sharadar_download.sh
        /// writes ~/.local/share/stockai/sharadar/raw/<date>/).
        #[arg(long)]
        raw: PathBuf,
        /// Store directory to write (e.g. ~/.local/share/stockai/sharadar/store). A directory holding
        /// a store this importer did not write is refused.
        #[arg(long)]
        out: PathBuf,
        /// Keep only stocks whose last price is on or after this date (the Stooq-like control:
        /// 2025-06-27, Stooq's last day). Funds are kept.
        #[arg(long)]
        survivors_only: Option<String>,
    },
    /// Write correlation2's external series as census-style daily variables on the export's trading
    /// days (docs/external.md): ext_v.npy, ext_raw.npy, ext_meta.tsv and meta.json, with the reading,
    /// timing and standardization of correlation2's external_test.rs at 310ea71.
    ExportExternal {
        /// correlation2's external data: catalog_*.csv and series/*.csv are read, nothing is written
        /// (default ~/.local/share/correlation2/external).
        #[arg(long)]
        external: Option<PathBuf>,
        /// Output directory (default ~/.local/share/stockai/external). Refused inside
        /// ~/.local/share/correlation2.
        #[arg(long)]
        out: Option<PathBuf>,
        /// An export of the same store whose days.npy the days must equal (default
        /// <C2_DATA_DIR>/export/full; compared when present).
        #[arg(long)]
        export: Option<PathBuf>,
    },
    /// Day-one checks of a Sharadar download (docs/sharadar.md): SIVB, BBBY, FRC and TWTR, delistings
    /// by year and kind, the action codes, and daily returns against the Stooq store. Reads nothing
    /// in the locked year. Exits with status 1 if a check fails.
    CheckSharadar {
        /// Directory holding the four tables, as for import-sharadar.
        #[arg(long)]
        raw: PathBuf,
        /// correlation2's Stooq store for the price comparison (default ~/.local/share/correlation2/store;
        /// skipped if absent).
        #[arg(long)]
        stooq: Option<PathBuf>,
        /// Last day of the price comparison (at most 2024-06-27, before the locked year).
        #[arg(long, default_value = "2024-06-27")]
        until: String,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum RandomMatchArg {
    /// correlation2's guessers: the strategy's number of names and replacements, drawn uniformly.
    Uniform,
    /// Also the volatility decile (trailing 63-day realized volatility) of every name bought.
    Volatility,
}

#[derive(Clone, Copy, ValueEnum)]
enum DelistArg {
    /// Pre-registered: a buyout at its last price, a failure or an unknown delisting a total loss.
    Main,
    /// Every delisting at its last price (optimistic bound).
    LastPrice,
    /// Every delisting a total loss (harsh bound).
    TotalLoss,
}

#[derive(Clone, Copy, ValueEnum)]
enum PriceTestArg {
    /// As-traded closes (close_unadj.f32 of a Sharadar store).
    Traded,
    /// The variable's adjusted closes (correlation2's test).
    Adjusted,
}

#[derive(clap::Args)]
struct HoldoutArgs {
    /// First day treated as the future (walk-forward: the start month and day of every year,
    /// from 2006 up to this date's year).
    #[arg(long, default_value = "2023-06-28")]
    start: String,
    /// First day after a single holdout (exclusive). Default: the end of the data, capped at
    /// 2024-06-28 unless --locked.
    #[arg(long)]
    end: Option<String>,
    /// Stocks held each day.
    #[arg(long, default_value_t = 20)]
    picks: usize,
    /// Trading costs to test, basis points per side (the walk-forward's main columns use the first).
    #[arg(long, value_delimiter = ',', default_value = "0,5,20")]
    costs: Vec<f64>,
    /// Random-guess accounts to compare against.
    #[arg(long, default_value_t = 1000)]
    random: usize,
    /// Walk forward: treat each year from 2006 to --start's year as the future in turn, and print
    /// a table by year.
    #[arg(long)]
    walk_forward: bool,
    /// Signal family: `scores` (a file of model scores, --scores) or `reversal` (own-stock
    /// short-term reversal, correlation2's positive control).
    #[arg(long, default_value = "scores")]
    family: String,
    /// Scores file for --family scores: .npy, float32 or float64, shape [N, T] of the export;
    /// scores[j, d] is decided at the close of day d; NaN = no score.
    #[arg(long)]
    scores: Option<PathBuf>,
    /// Export directory whose meta.json fingerprint the scores must match (default: a meta.json
    /// next to the scores file, if there is one).
    #[arg(long)]
    export: Option<PathBuf>,
    /// Trading days between rebalances (default: --lookback for reversal, 1 for scores).
    #[arg(long)]
    rebalance: Option<usize>,
    /// Reversal: return lookback in trading days.
    #[arg(long, default_value_t = 21)]
    lookback: usize,
    /// Days of history before the holdout for the liquid universe (>= 90% with census values).
    #[arg(long, default_value_t = 756)]
    peer_window: usize,
    /// The account acts this many trading days after each decision (holds nothing before the first).
    #[arg(long, default_value_t = 0)]
    delay: usize,
    /// How the random accounts are matched to the strategy.
    #[arg(long, value_enum, default_value = "uniform")]
    random_match: RandomMatchArg,
    /// Also write the results as JSON.
    #[arg(long)]
    json: Option<PathBuf>,
    /// Include the daily account curve in a single holdout's JSON.
    #[arg(long)]
    json_curve: bool,
    /// Write the scores the family computed (f64 [N, T], NaN where none) to this .npy file.
    #[arg(long)]
    dump_scores: Option<PathBuf>,
    /// Allow grading the locked year (2024-06-28 onward). Only for the single final run.
    #[arg(long)]
    locked: bool,
    /// How a held stock that stops trading counts (docs/sharadar.md). Default: main when the store
    /// has delist.tsv (Sharadar); a store without it (Stooq) has no delisting logic and refuses this.
    #[arg(long, value_enum)]
    delist: Option<DelistArg>,
    /// Closes for the universe's $5 median-price test. Default: traded when the store has
    /// close_unadj.f32 (Sharadar), else adjusted.
    #[arg(long, value_enum)]
    price_test: Option<PriceTestArg>,
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Export { out } => run_export(out),
        Cmd::Holdout(args) => run_holdout(*args),
        Cmd::ImportSharadar { raw, out, survivors_only } => run_import(&raw, &out, survivors_only.as_deref()),
        Cmd::ExportExternal { external, out, export } => run_export_external(external, out, export),
        Cmd::CheckSharadar { raw, stooq, until } => {
            let until = parse_date(&until)?;
            ensure!(until < locked_from(), "--until {} is in the locked year; the checks read nothing from 2024-06-28 on", date_string(until));
            let stooq = stooq.unwrap_or_else(|| std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join(".local/share/correlation2/store"));
            if !check::run(&raw, &stooq, until)? {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

fn run_export_external(external: Option<PathBuf>, out: Option<PathBuf>, export_dir: Option<PathBuf>) -> Result<()> {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    let c2_home = home.join(".local/share/correlation2");
    let external = external.unwrap_or_else(|| c2_home.join("external"));
    let out = std::path::absolute(out.unwrap_or_else(|| home.join(".local/share/stockai/external")))?;
    // Nothing is written inside correlation2's data (or the external data read).
    let real = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    for dir in [&c2_home, &external] {
        ensure!(!out.starts_with(dir) && !real(&out).starts_with(real(dir)), "--out {} is inside {}, which is read only here", out.display(), dir.display());
    }
    let t0 = Instant::now();
    let dir = store_dir();
    let store = c2_data::Store::open(&dir)?;
    let plan = c2_census::run::Plan::new(&store, c2_census::grid::Config::full())?;
    let days = &plan.var.days;
    let fp = export::fingerprint_hex(days, &plan.var.ids);
    // The export of the same store must have exactly these days.
    let export_dir = export_dir.unwrap_or_else(|| c2_data::data_dir().join("export").join("full"));
    if export_dir.join("days.npy").exists() {
        let npy::Data::I4(theirs) = npy::read(&export_dir.join("days.npy"))?.data else { bail!("{}: days.npy is not int32", export_dir.display()) };
        ensure!(theirs == *days, "{}/days.npy differs from the store's trading days ({} vs {} days): export the store again", export_dir.display(), theirs.len(), days.len());
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(export_dir.join("meta.json"))?)?;
        ensure!(meta["fingerprint"] == fp.as_str(), "{}: fingerprint {}, the store's variable has {fp}", export_dir.display(), meta["fingerprint"]);
        println!("The store's trading days equal {}/days.npy ({} days, {} .. {}, fingerprint {fp})", export_dir.display(), days.len(), date_string(days[0]), date_string(days[days.len() - 1]));
    } else {
        println!("No export at {} to compare the days with; fingerprint {fp}", export_dir.display());
    }
    let s = external::export(days, &fp, &external, &out)?;
    println!(
        "Exported {} external series x {} days to {} ({:.0} MB) in {:.1}s",
        s.k,
        s.t,
        s.out.display(),
        s.bytes as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    println!("By status: {}", s.statuses.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", "));
    println!("By category: {}", s.categories.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", "));
    println!("Series with a value per trading day (mean over the year; all / headline):");
    for (y, all, head) in &s.live {
        println!("  {y}: {all:.0} / {head:.0}");
    }
    if !s.warnings.is_empty() {
        println!("{} warnings (in meta.json): {}", s.warnings.len(), s.warnings.iter().take(5).cloned().collect::<Vec<_>>().join("; "));
    }
    Ok(())
}

fn run_import(raw: &Path, out: &Path, survivors_only: Option<&str>) -> Result<()> {
    let t0 = Instant::now();
    let survivors = survivors_only
        .map(|s| sharadar::strict_date(s.as_bytes()).with_context(|| format!("--survivors-only {s:?}: expected YYYY-MM-DD")))
        .transpose()?;
    let imported = sharadar::import(raw, out, survivors)?;
    for line in &imported.log {
        println!("{line}");
    }
    println!("import-sharadar took {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn store_dir() -> PathBuf {
    c2_data::data_dir().join("store")
}

fn run_export(out: Option<PathBuf>) -> Result<()> {
    let out = out.unwrap_or_else(|| c2_data::data_dir().join("export").join("full"));
    let dir = store_dir();
    let store = c2_data::Store::open(&dir)?;
    let t0 = Instant::now();
    let plan = c2_census::run::Plan::new(&store, c2_census::grid::Config::full())?;
    let built = t0.elapsed().as_secs_f64();
    let extras = extras::Extras::load(&store, &dir, &plan.var)?;
    let s = export::export(&store, &plan, &dir, &out, C2_COMMIT, extras.as_ref())?;
    println!(
        "Exported plan.var to {}: {} stocks x {} days ({} .. {}), fingerprint {}; {} finite census values (sum {:.6}); {:.0} MB in {:.1}s (plan built in {:.1}s)",
        s.dir.display(),
        s.n,
        s.t,
        s.first_day,
        s.last_day,
        s.fingerprint,
        s.finite_v,
        s.sum_v,
        s.bytes as f64 / 1e6,
        t0.elapsed().as_secs_f64(),
        built
    );
    if let Some(x) = &extras {
        println!(
            "Sharadar store: also close_unadj.npy, delist_day.npy and delist_kind.npy; {} delisted stocks among the columns ({} buyout, {} failure, {} unknown); {} last closes before their last price date (by up to {} days); {} delisted without any close (left as listed)",
            x.delisted.iter().sum::<usize>(),
            x.delisted[0],
            x.delisted[1],
            x.delisted[2],
            x.moved,
            x.moved_max_days,
            x.without_close
        );
    }
    Ok(())
}

/// Rule (a) for a single holdout: nothing from 2024-06-28 on is graded without --locked. An
/// explicit --end past it is refused; the default end (end of data) is capped there.
/// Returns the exclusive end and a note to print.
fn single_holdout_end(start: i32, end: Option<i32>, locked: bool) -> Result<(i32, Option<String>)> {
    let lock = locked_from();
    if locked {
        return Ok((end.unwrap_or(i32::MAX), None));
    }
    ensure!(
        start < lock,
        "--start {} is in the locked year or later (from 2024-06-28), which is graded once at the end with --locked; development runs never pass --locked",
        date_string(start)
    );
    match end {
        Some(e) if e > lock => bail!(
            "--end {} reaches past 2024-06-27 into the locked year; use --end 2024-06-28 or earlier (--locked is only for the single final run)",
            date_string(e)
        ),
        Some(e) => Ok((e, None)),
        None => Ok((lock, Some("(end of data capped at 2024-06-28: the locked year 2024-06-28 .. 2025-06-27 is graded only with --locked)".into()))),
    }
}

/// Rule (a) for the walk-forward: its last period [y0-mm-dd, (y0+1)-mm-dd) must end by
/// 2024-06-28 (exclusive) unless --locked. Earlier periods end earlier.
fn walk_forward_guard(start: i32, locked: bool) -> Result<()> {
    if locked {
        return Ok(());
    }
    let (y0, mo, d) = civil_from_days(start);
    let (s, e) = (days_from_civil(y0, mo, d), days_from_civil(y0 + 1, mo, d));
    ensure!(
        e <= locked_from(),
        "--walk-forward --start {} grades {} .. {} last, which reaches past 2024-06-27 into the locked year; use --start 2023-06-28 or earlier (--locked is only for the single final run)",
        date_string(start),
        date_string(s),
        date_string(e - 1)
    );
    Ok(())
}

/// Checks a scores file against the export it was made from: `--export DIR`'s meta.json
/// (required), else a meta.json next to the file (if any).
fn check_scores_meta(scores: &Path, export: Option<&Path>, fingerprint: &str, n: usize, t: usize) -> Result<String> {
    let store = store_dir();
    let (meta, required) = match export {
        Some(dir) => (dir.join("meta.json"), true),
        None => (scores.parent().unwrap_or(Path::new("")).join("meta.json"), false),
    };
    if !meta.exists() {
        ensure!(!required, "--export {}: no meta.json there", meta.parent().unwrap_or(Path::new("")).display());
        return Ok("no meta.json next to the scores file; fingerprint not checked (pass --export DIR to check it)".into());
    }
    let m: Value = serde_json::from_str(&std::fs::read_to_string(&meta)?).with_context(|| format!("reading {}", meta.display()))?;
    let Some(f) = m.get("fingerprint").and_then(Value::as_str) else {
        ensure!(!required, "{}: no fingerprint", meta.display());
        return Ok(format!("{} has no fingerprint; not checked", meta.display()));
    };
    let hint = match m.get("store_dir").and_then(Value::as_str) {
        Some(d) if Path::new(d) != store => format!(
            " (that export was made from the store in {d}, but the grader read {}: set C2_DATA_DIR to the store's parent directory)",
            store.display()
        ),
        _ => String::new(),
    };
    ensure!(
        f == fingerprint,
        "{}: fingerprint {f}, but this store's variable has {fingerprint}: the scores were made for a different export (re-export, re-score){hint}",
        meta.display()
    );
    if let (Some(mn), Some(mt)) = (m.get("n").and_then(Value::as_u64), m.get("t").and_then(Value::as_u64)) {
        ensure!((mn as usize, mt as usize) == (n, t), "{}: n, t = {mn}, {mt}, but the variable is {n} x {t}", meta.display());
    }
    Ok(format!("fingerprint {fingerprint} matches {}", meta.display()))
}

/// Scores computed by the family, gathered over every graded holdout into one [N, T] matrix.
struct Dump {
    t: usize,
    m: Vec<f64>,
}

impl Dump {
    fn add(&mut self, r: &Report) {
        let Some(k) = &r.kept else { return };
        for (i, s) in k.scores.iter().enumerate() {
            if let Some(s) = s {
                for (c, &x) in s.iter().enumerate() {
                    self.m[k.candidates[c] * self.t + k.days[i]] = x;
                }
            }
        }
    }
}

fn pct(v: f64) -> f64 {
    100.0 * (v / 10_000.0 - 1.0)
}

fn num(x: f64) -> Value {
    // serde_json writes non-finite numbers as null.
    json!(x)
}

fn matching_json(m: &Matching) -> Value {
    json!({
        "mode": m.mode.name(),
        "strategy_buys": m.strategy_buys,
        "random_buys_mean": num(m.random_buys_mean),
        "random_buys_min": m.random_buys_min,
        "random_buys_max": m.random_buys_max,
        "strategy_buy_decile": num(m.strategy_buy_decile),
        "random_buy_decile": num(m.random_buy_decile),
        "strategy_held_decile": num(m.strategy_held_decile),
        "random_held_decile": num(m.random_held_decile),
        "unmatched_buys": m.unmatched,
    })
}

fn delist_json(d: &DelistStats) -> Value {
    let kinds = |v: &[usize; 3]| json!({"buyout": v[0], "failure": v[1], "unknown": v[2]});
    json!({
        "rule": d.rule.name(),
        "candidates_delisted": kinds(&d.candidates),
        "candidates_dead_before_start": d.dead_before,
        "picks_held_over_delisting": kinds(&d.held),
        "random_held_over_delisting_mean": {"buyout": num(d.random_held[0]), "failure": num(d.random_held[1]), "unknown": num(d.random_held[2])},
        "picks_bought_after_last_close": d.bought_dead,
        "random_bought_after_last_close_mean": num(d.random_bought_dead),
    })
}

/// One line on a holdout's delistings.
fn delist_line(d: &DelistStats) -> String {
    format!(
        "candidates delisted {}/{}/{} (buyout/failure/unknown), dead from the start {}; picks held over a delisting {}/{}/{}, random accounts {:.2}/{:.2}/{:.2}; bought after their last close: picks {}, random {:.2}",
        d.candidates[0],
        d.candidates[1],
        d.candidates[2],
        d.dead_before,
        d.held[0],
        d.held[1],
        d.held[2],
        d.random_held[0],
        d.random_held[1],
        d.random_held[2],
        d.bought_dead,
        d.random_bought_dead
    )
}

fn bench_pct(r: &Report, ticker: Option<&str>) -> Value {
    r.benchmarks.iter().find(|b| b.ticker == ticker).map(|b| num(pct(b.final_value))).unwrap_or(Value::Null)
}

fn report_json(r: &Report, with_curve: bool) -> Value {
    let mut v = json!({
        "first_day": date_string(r.first_day),
        "last_day": date_string(r.last_day),
        "days": r.days,
        "pairs": r.pairs,
        "windows_used": r.windows_used,
        "candidates": r.candidates,
        "benchmarks": r.benchmarks.iter().map(|b| json!({"label": b.label, "ticker": b.ticker, "final_value": num(b.final_value), "return_pct": num(pct(b.final_value))})).collect::<Vec<_>>(),
        "strategy": r.strategy.iter().map(|&(c, s, med, p)| json!({"cost_bps": c, "final": num(s), "final_pct": num(pct(s)), "random_median": num(med), "random_median_pct": num(pct(med)), "p": num(p)})).collect::<Vec<_>>(),
        "hit_rate": num(r.hit_rate),
        "hits": r.hits,
        "ic_mean": num(r.ic_mean),
        "ic_t": num(r.ic_t),
        "ic_ret_mean": num(r.ic_ret_mean),
        "ic_ret_t": num(r.ic_ret_t),
        "ls_mean": num(r.ls_mean),
        "ls_bps_per_day": num(1e4 * r.ls_mean),
        "ls_sharpe": num(r.ls_sharpe),
        "ls_t": num(r.ls_t),
        "ls_turnover": num(r.ls_turnover),
        "random_check": matching_json(&r.matching),
    });
    if let Some(d) = &r.delisting {
        v["delisting"] = delist_json(d);
    }
    if with_curve {
        v["curve"] = r
            .curve
            .iter()
            .map(|&(d, p, spy, qqq, med)| json!({"day": date_string(d), "picks": num(p), "spy": num(spy), "qqq": num(qqq), "random_median": num(med)}))
            .collect::<Vec<_>>()
            .into();
    }
    v
}

fn year_json(s: i32, e: i32, r: &Report) -> Value {
    let mut v = json!({
        "start": date_string(s),
        "end": date_string(e),
        "first_day": date_string(r.first_day),
        "last_day": date_string(r.last_day),
        "days": r.days,
        "candidates": r.candidates,
        "pairs": r.pairs,
        "spy_pct": bench_pct(r, Some("SPY")),
        "qqq_pct": bench_pct(r, Some("QQQ")),
        "hold_all_pct": bench_pct(r, None),
        "costs": r.strategy.iter().map(|&(c, s, med, p)| json!({"cost_bps": c, "final": num(s), "final_pct": num(pct(s)), "random_median_pct": num(pct(med)), "p": num(p)})).collect::<Vec<_>>(),
        "hit_rate": num(r.hit_rate),
        "hits": r.hits,
        "ic_mean": num(r.ic_mean),
        "ic_t": num(r.ic_t),
        "ic_ret_mean": num(r.ic_ret_mean),
        "ic_ret_t": num(r.ic_ret_t),
        "ls_bps_per_day": num(1e4 * r.ls_mean),
        "ls_sharpe": num(r.ls_sharpe),
        "ls_t": num(r.ls_t),
        "ls_turnover": num(r.ls_turnover),
        "random_check": matching_json(&r.matching),
    });
    if let Some(d) = &r.delisting {
        v["delisting"] = delist_json(d);
    }
    v
}

fn write_json(path: &Path, v: &Value) -> Result<()> {
    npy::write_atomic(path, |w| {
        use std::io::Write;
        w.write_all(serde_json::to_string_pretty(v)?.as_bytes())?;
        w.write_all(b"\n")?;
        Ok(())
    })
}

fn run_holdout(a: HoldoutArgs) -> Result<()> {
    let family = a.family.as_str();
    ensure!(matches!(family, "scores" | "reversal"), "unknown family {family} (scores or reversal; this fork has no census or peers family)");
    if family == "scores" {
        ensure!(a.scores.is_some(), "--family scores needs --scores FILE.npy");
    } else {
        ensure!(a.scores.is_none() && a.export.is_none(), "--scores and --export are for --family scores");
    }
    ensure!(!a.costs.is_empty(), "--costs needs at least one value");
    ensure!(a.picks >= 1 && a.random >= 1, "--picks and --random must be at least 1");
    ensure!(!a.json_curve || a.json.is_some(), "--json-curve needs --json");
    let rebalance = a.rebalance.unwrap_or(if family == "reversal" { a.lookback } else { 1 });
    let random_match = match a.random_match {
        RandomMatchArg::Uniform => RandomMatch::Uniform,
        RandomMatchArg::Volatility => RandomMatch::Volatility,
    };
    let start = parse_date(&a.start)?;
    let (end, note) = if a.walk_forward {
        ensure!(a.end.is_none(), "--end is for a single holdout; the walk-forward's periods are one year each");
        walk_forward_guard(start, a.locked)?;
        (i32::MAX, None)
    } else {
        single_holdout_end(start, a.end.as_deref().map(parse_date).transpose()?, a.locked)?
    };

    let dir = store_dir();
    let store = c2_data::Store::open(&dir)?;
    let t0 = Instant::now();
    let plan = c2_census::run::Plan::new(&store, c2_census::grid::Config::full())?;
    let (n, t) = (plan.var.n, plan.var.t);
    // A Sharadar store's sidecars switch on the delisting rule and the as-traded $5 test by default.
    let extras = extras::Extras::load(&store, &dir, &plan.var)?;
    let delist = match (a.delist, &extras) {
        (Some(_), None) => bail!(
            "--delist needs a store with delist.tsv (a Sharadar store); {} has none. Set C2_DATA_DIR to the Sharadar store's parent directory",
            dir.display()
        ),
        (Some(r), Some(_)) => Some(match r {
            DelistArg::Main => DelistRule::Main,
            DelistArg::LastPrice => DelistRule::LastPrice,
            DelistArg::TotalLoss => DelistRule::TotalLoss,
        }),
        (None, Some(_)) => Some(DelistRule::Main),
        (None, None) => None,
    };
    let price_test = match (a.price_test, &extras) {
        (Some(PriceTestArg::Traded), None) => bail!("--price-test traded needs as-traded closes (close_unadj.f32 of a Sharadar store); {} has none", dir.display()),
        (Some(PriceTestArg::Traded), Some(_)) | (None, Some(_)) => PriceTest::Traded,
        (Some(PriceTestArg::Adjusted), _) | (None, None) => PriceTest::Adjusted,
    };
    let fingerprint = export::fingerprint_hex(&plan.var.days, &plan.var.ids);
    let mut checked = None;
    let matrix = match &a.scores {
        Some(path) => {
            checked = Some(check_scores_meta(path, a.export.as_deref(), &fingerprint, n, t)?);
            let m = ScoreMatrix::load(path, n, t)?;
            ensure!(m.finite() > 0, "{}: every score is NaN", path.display());
            Some(m)
        }
        None => None,
    };
    let fam = match &matrix {
        Some(m) => Family::Scores { matrix: m, window: a.peer_window },
        None => Family::Reversal { lookback: a.lookback, window: a.peer_window },
    };
    let mut dump = a.dump_scores.as_ref().map(|_| Dump { t, m: vec![f64::NAN; n * t] });
    let mut options = json!({
        "family": family,
        "scores": a.scores.as_ref().map(|p| p.display().to_string()),
        "scores_dtype": matrix.as_ref().map(|m| m.dtype),
        "export": a.export.as_ref().map(|p| p.display().to_string()),
        "fingerprint_check": checked,
        "lookback": (family == "reversal").then_some(a.lookback),
        "peer_window": a.peer_window,
        "rebalance": rebalance,
        "picks": a.picks,
        "costs_bps": a.costs,
        "random": a.random,
        "random_match": random_match.name(),
        "delay": a.delay,
        "cash": 10_000.0,
        "start": a.start,
        "end": (!a.walk_forward).then(|| if end == i32::MAX { "end of data".to_string() } else { date_string(end) }),
        "walk_forward": a.walk_forward,
        "locked": a.locked,
        "dump_scores": a.dump_scores.as_ref().map(|p| p.display().to_string()),
    });
    // Only a Sharadar store adds these, so a Stooq run's JSON is as before.
    if let Some(x) = &extras {
        options["delist"] = delist.map(DelistRule::name).into();
        options["price_test"] = price_test.name().into();
        options["store"] = json!({
            "dir": dir.display().to_string(),
            "importer": x.source["importer"],
            "lastupdated_max": x.source["lastupdated_max"],
            "survivors_only": x.source["survivors_only"],
        });
    }
    let header = |mode: &str| {
        json!({
            "tool": concat!("stockai-judge ", env!("CARGO_PKG_VERSION")),
            "c2_commit": C2_COMMIT,
            "mode": mode,
            "variable": {"n": n, "t": t, "first_day": date_string(plan.var.days[0]), "last_day": date_string(plan.var.days[t - 1]), "fingerprint": fingerprint},
            "options": options,
        })
    };
    let opt_for = |s: i32, e: i32| Options {
        family: fam,
        rebalance,
        start: s,
        end: e,
        picks: a.picks,
        costs_bps: a.costs.clone(),
        random_runs: a.random,
        cash: 10_000.0,
        delay: a.delay,
        random_match,
        keep_scores: a.dump_scores.is_some(),
        extras: extras.as_ref(),
        delist,
        price_test,
    };
    if let Some(c) = &checked {
        eprintln!("{c}");
    }
    if let Some(x) = &extras {
        println!(
            "Sharadar store {} (survivors only: {}): delisting rule {}, $5 test on {} closes",
            dir.display(),
            x.source["survivors_only"].as_str().unwrap_or("no"),
            delist.map_or("none", DelistRule::name),
            if price_test == PriceTest::Traded { "as-traded" } else { "adjusted" }
        );
    }

    if a.walk_forward {
        let (y0, mo, d) = civil_from_days(start);
        let last = *plan.var.days.last().unwrap();
        let costs = &a.costs;
        println!("Walk-forward: each year treated as the future, predicted only from census windows that ended before it.");
        let mut head = format!(
            "{:<24} {:>7} {:>9} {:>9} {:>10} {:>10} {:>10} {:>8} {:>8} {:>9} {:>8}",
            "holdout",
            "pairs",
            "SPY",
            "QQQ",
            "picks",
            "random med",
            format!("p ({} bps)", costs[0]),
            "hit %",
            "IC t",
            "LS bps/d",
            "LS t"
        );
        head += &format!(" {:>8}", "ICret t");
        for c in &costs[1..] {
            head += &format!(" {:>10} {:>8}", format!("picks {c}bp"), format!("p {c}bp"));
        }
        println!("{head}");
        let mut beat = vec![0usize; costs.len()];
        let mut years = 0;
        let (mut ic_ts, mut ic_ret_ts, mut ls_ts, mut ls_bps, mut ls_turn): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = Default::default();
        let (mut graded, mut skipped) = (Vec::new(), Vec::new());
        let mut delist_lines = Vec::new();
        for y in 2006..=y0 {
            let s = days_from_civil(y, mo, d);
            let e = days_from_civil(y + 1, mo, d);
            if e > last + 7 && y != y0 {
                continue;
            }
            let r = match holdout::holdout(&store, &plan, &opt_for(s, e)) {
                Ok(r) => r,
                Err(err) => {
                    skipped.push(json!({"start": date_string(s), "end": date_string(e), "reason": err.to_string()}));
                    eprintln!("skipped {} .. {}: {err}", date_string(s), date_string(e));
                    continue;
                }
            };
            if r.pairs == 0 {
                skipped.push(json!({"start": date_string(s), "end": date_string(e), "reason": "no pairs"}));
                continue;
            }
            let bench = |i: usize| r.benchmarks.get(i).map(|b| pct(b.final_value)).unwrap_or(f64::NAN);
            let (_, strat, med, p) = r.strategy[0];
            years += 1;
            for (b, &(_, _, _, p)) in beat.iter_mut().zip(&r.strategy) {
                *b += usize::from(p < 0.1);
            }
            let mut row = format!(
                "{:<24} {:>7} {:>8.1}% {:>8.1}% {:>9.1}% {:>9.1}% {:>10.3} {:>7.1}% {:>8.2} {:>9.2} {:>8.2}",
                format!("{}..{}", date_string(r.first_day), date_string(r.last_day)),
                r.pairs,
                bench(0),
                bench(1),
                pct(strat),
                pct(med),
                p,
                100.0 * r.hit_rate,
                r.ic_t,
                1e4 * r.ls_mean,
                r.ls_t
            );
            row += &format!(" {:>8.2}", r.ic_ret_t);
            for &(_, s, _, p) in &r.strategy[1..] {
                row += &format!(" {:>9.1}% {:>8.3}", pct(s), p);
            }
            println!("{row}");
            ic_ts.push(r.ic_t);
            ic_ret_ts.push(r.ic_ret_t);
            ls_ts.push(r.ls_t);
            ls_bps.push(1e4 * r.ls_mean);
            ls_turn.push(r.ls_turnover);
            if let Some(dm) = dump.as_mut() {
                dm.add(&r);
            }
            if let Some(d) = &r.delisting {
                delist_lines.push(format!("{:<24} {}", format!("{}..{}", date_string(r.first_day), date_string(r.last_day)), delist_line(d)));
            }
            graded.push(year_json(s, e, &r));
        }
        let stouffer = |v: &[f64]| v.iter().sum::<f64>() / (v.len() as f64).sqrt();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        println!(
            "Years where the picks beat random guessers at p < 0.1: {} of {years} (chance alone: about {:.1})  [{:.0}s]",
            beat[0],
            0.1 * years as f64,
            t0.elapsed().as_secs_f64()
        );
        println!("Rank IC: t positive in {} of {years} years; combined across years z = {:.2}", ic_ts.iter().filter(|&&t| t > 0.0).count(), stouffer(&ic_ts));
        println!(
            "Long-short top/bottom 10%: mean {:.2} bps/day gross, positive in {} of {years} years, combined z = {:.2}; turnover {:.0}% of the book per rebalance, so costs of c bps per side take about {:.2} x c bps/day",
            mean(&ls_bps),
            ls_bps.iter().filter(|&&v| v > 0.0).count(),
            stouffer(&ls_ts),
            100.0 * mean(&ls_turn),
            2.0 * 2.0 * mean(&ls_turn) / rebalance as f64
        );
        println!(
            "Rank IC in actual returns: t positive in {} of {years} years; combined across years z = {:.2}",
            ic_ret_ts.iter().filter(|&&t| t > 0.0).count(),
            stouffer(&ic_ret_ts)
        );
        println!(
            "Years where the picks beat random guessers at p < 0.1, by cost: {}",
            costs.iter().zip(&beat).map(|(c, b)| format!("{c} bps: {b} of {years}")).collect::<Vec<_>>().join("; ")
        );
        if !delist_lines.is_empty() {
            println!("Delistings ({} rule):", delist.map_or("no", DelistRule::name));
            for l in &delist_lines {
                println!("{l}");
            }
        }
        if a.delay > 0 || random_match == RandomMatch::Volatility {
            println!(
                "Options: the account trades {} trading day(s) after each decision; random guessers matched on {}",
                a.delay,
                if random_match == RandomMatch::Volatility { "turnover and volatility decile" } else { "turnover" }
            );
        }
        if let (Some(path), Some(dm)) = (&a.dump_scores, &dump) {
            npy::write(path, &[n, t], &dm.m)?;
            eprintln!("wrote the computed scores to {}", path.display());
        }
        if let Some(path) = &a.json {
            let mut out = header("walk-forward");
            out["years"] = graded.into();
            out["skipped"] = skipped.into();
            out["summary"] = json!({
                "years": years,
                "beat_random_p_lt_0_1": costs.iter().zip(&beat).map(|(c, b)| json!({"cost_bps": c, "years": b})).collect::<Vec<_>>(),
                "ic_t_positive_years": ic_ts.iter().filter(|&&t| t > 0.0).count(),
                "ic_z": num(stouffer(&ic_ts)),
                "ic_ret_t_positive_years": ic_ret_ts.iter().filter(|&&t| t > 0.0).count(),
                "ic_ret_z": num(stouffer(&ic_ret_ts)),
                "ls_bps_per_day_mean": num(mean(&ls_bps)),
                "ls_positive_years": ls_bps.iter().filter(|&&v| v > 0.0).count(),
                "ls_z": num(stouffer(&ls_ts)),
                "ls_turnover_mean": num(mean(&ls_turn)),
                "ls_cost_bps_per_day_per_bps": num(2.0 * 2.0 * mean(&ls_turn) / rebalance as f64),
            });
            write_json(path, &out)?;
        }
        return Ok(());
    }

    if let Some(note) = &note {
        println!("{note}");
    }
    let r = holdout::holdout(&store, &plan, &opt_for(start, end))?;
    println!("Holdout {} .. {} ({} trading days), {:.0}s", date_string(r.first_day), date_string(r.last_day), r.days, t0.elapsed().as_secs_f64());
    println!(
        "Family {family}, rebalanced every {rebalance} trading day(s): {} pairs from {} pre-holdout window(s); {} stocks can be picked",
        r.pairs, r.windows_used, r.candidates
    );
    println!("Picks beat the market over their holding period: {:.1}% of {} picks (coin flip 50%)", 100.0 * r.hit_rate, r.hits);
    println!("Rank correlation of prediction vs outcome (IC): mean {:+.4}, t = {:+.2}", r.ic_mean, r.ic_t);
    println!("Rank correlation of prediction vs actual return (IC in returns): mean {:+.4}, t = {:+.2}", r.ic_ret_mean, r.ic_ret_t);
    println!(
        "Long-short top/bottom 10%: {:.2} bps/day gross, Sharpe {:.2}, t = {:.2}, book turnover {:.0}% per rebalance",
        1e4 * r.ls_mean,
        r.ls_sharpe,
        r.ls_t,
        100.0 * r.ls_turnover
    );
    if a.delay > 0 {
        println!("The account trades {} trading day(s) after each decision (it holds nothing for the first {})", a.delay, a.delay);
    }
    if let Some(d) = &r.delisting {
        println!("Delistings ({} rule): {}", d.rule.name(), delist_line(d));
    }
    let m = &r.matching;
    println!(
        "Random guessers ({}): {:.1} names bought on average ({}..{}; the picks bought {}); mean volatility decile of buys {:.2} (picks {:.2}), of holdings {:.2} (picks {:.2}){}",
        m.mode.name(),
        m.random_buys_mean,
        m.random_buys_min,
        m.random_buys_max,
        m.strategy_buys,
        m.random_buy_decile,
        m.strategy_buy_decile,
        m.random_held_decile,
        m.strategy_held_decile,
        if m.mode == RandomMatch::Volatility { format!("; {} buys unmatched", m.unmatched) } else { String::new() }
    );
    println!("\n{:<44} {:>12} {:>9}", "", "final value", "return");
    for b in &r.benchmarks {
        println!("{:<44} {:>12.0} {:>8.1}%", b.label, b.final_value, 100.0 * (b.final_value / 10_000.0 - 1.0));
    }
    for (c, s, med, share) in &r.strategy {
        let label = if *c == 0.0 { "no trading costs".to_string() } else { format!("{c} bps per trade") };
        println!(
            "{:<44} {:>12.0} {:>8.1}%   random guessers: median {:.0}, {:.1}% did at least as well (p = {:.3})",
            format!("Picks, {label}"),
            s,
            100.0 * (s / 10_000.0 - 1.0),
            med,
            100.0 * share,
            share
        );
    }
    if let (Some(path), Some(dm)) = (&a.dump_scores, dump.as_mut()) {
        dm.add(&r);
        npy::write(path, &[n, t], &dm.m)?;
        eprintln!("wrote the computed scores to {}", path.display());
    }
    if let Some(path) = &a.json {
        let mut out = header("holdout");
        out["report"] = report_json(&r, a.json_curve);
        write_json(path, &out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(s: &str) -> i32 {
        parse_date(s).unwrap()
    }

    #[test]
    fn locked_year_is_refused_without_locked() {
        // Walk-forward: the development periods end 2024-06-27; anything later is refused.
        assert!(walk_forward_guard(day("2023-06-28"), false).is_ok());
        assert!(walk_forward_guard(day("2015-06-28"), false).is_ok());
        assert!(walk_forward_guard(day("2023-06-27"), false).is_ok());
        for s in ["2023-06-29", "2023-12-01", "2024-06-28", "2025-01-01"] {
            let e = walk_forward_guard(day(s), false).unwrap_err().to_string();
            assert!(e.contains("locked year"), "{s}: {e}");
        }
        assert!(walk_forward_guard(day("2024-06-28"), true).is_ok());

        // Single holdout: the default end is capped at 2024-06-28; explicit later ends are refused.
        let (e, note) = single_holdout_end(day("2023-06-28"), None, false).unwrap();
        assert_eq!(e, day("2024-06-28"));
        assert!(note.unwrap().contains("capped"));
        assert_eq!(single_holdout_end(day("2023-06-28"), Some(day("2024-06-28")), false).unwrap(), (day("2024-06-28"), None));
        assert_eq!(single_holdout_end(day("2020-06-28"), Some(day("2021-06-28")), false).unwrap().0, day("2021-06-28"));
        assert!(single_holdout_end(day("2023-06-28"), Some(day("2024-06-29")), false).is_err());
        assert!(single_holdout_end(day("2024-06-28"), None, false).is_err());
        assert!(single_holdout_end(day("2024-07-15"), Some(day("2024-08-01")), false).is_err());
        assert_eq!(single_holdout_end(day("2024-06-28"), None, true).unwrap(), (i32::MAX, None));
    }

    #[test]
    fn fingerprint_check_reads_meta_next_to_scores_or_from_export() {
        let dir = std::env::temp_dir().join(format!("stockai-judge-meta-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let scores = dir.join("s.npy");
        // No meta.json: not checked (unless --export is given).
        assert!(check_scores_meta(&scores, None, "00ff", 2, 3).unwrap().contains("not checked"));
        assert!(check_scores_meta(&scores, Some(&dir), "00ff", 2, 3).is_err());
        std::fs::write(dir.join("meta.json"), r#"{"format": "c2-export-1", "n": 2, "t": 3, "fingerprint": "00ff"}"#).unwrap();
        assert!(check_scores_meta(&scores, None, "00ff", 2, 3).unwrap().contains("matches"));
        assert!(check_scores_meta(&scores, Some(&dir), "00ff", 2, 3).unwrap().contains("matches"));
        let e = check_scores_meta(&scores, None, "abcd", 2, 3).unwrap_err().to_string();
        assert!(e.contains("different export"), "{e}");
        assert!(check_scores_meta(&scores, None, "00ff", 2, 4).is_err());
    }
}
