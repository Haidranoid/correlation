//! External series vs US stocks: does non-stock data move first and US stocks follow?
//!
//! A census-faithful lead-lag test of everything in `<data>/external/catalog_*.csv` (regional
//! weather, climate indices, commodities, rates, FX, world equity, crypto, macro, space
//! weather, attention, politics, ...; format in the download agents' shared spec:
//! `series/<id>.csv` with `date,value`) against the census's own US stocks.
//!
//! Run:
//!   cargo run --release -p c2-census --example external_test -- --run NAME
//!   cargo run --release -p c2-census --example external_test -- --selftest
//!   cargo run --release -p c2-census --example external_test -- --browse NAME --series wx_ --min-z 5
//!
//! It reads (never writes) the price store and the census run, and writes only under
//! `<data>/external/results/<run name>/`. Every `catalog_*.csv` present is included, so a
//! rerun after more catalogs arrive picks them up.
//!
//! Method, as the census does it unless stated:
//! * Correlation periods: read through `RunMeta` and chosen by length and date, never by id (a
//!   new census renumbers them): the chain of non-overlapping consecutive 1260-day periods that
//!   starts on the census's first day, plus the full-history period, mapped onto the calendar by
//!   date, with the run's own stock universes. Stocks: the production census variable.
//! * An external series becomes a census-style variable on the trading calendar without
//!   look-ahead. An observation dated d with `known_after_days` k counts from the first trading
//!   day on or after d + k (calendar days). `logdiff`/`diff`: the move on trading day t is the
//!   change in the latest known level since day t-1 (release days carry the whole move, other
//!   days 0). `anomaly`/`anomaly_log1p`: the value (or ln(1 + value)) minus the mean of the
//!   15-day spans centred on the same date in each of the previous 10 years (at least 3
//!   years); the anomalies of the calendar days that land on one trading day are averaged.
//!   `level`: the value minus its mean over the previous 252 trading days. Then, exactly like
//!   the census's stock variable: minus the trailing 252-day mean and divided by the trailing
//!   252-day sd (both through t-1, at least 126 values), clipped at +-5. A series is live while
//!   its latest observation is recent for its frequency; elsewhere it is missing (0 in sums).
//! * An external series joins a period when >= 90% of the period's trading days have a value
//!   and >= 10 of them carry a non-zero move (a release or an event). Annual series (one
//!   release a year, often all on one date: a calendar pulse) and finite-life single
//!   prediction-market contracts are reported apart from the headline numbers.
//! * Grid: all 36 move-length pairs and every offset in `grid::lags(h2)` with >= 30
//!   observations; A's move over (e-h1, e], B's over (e+offset-h2, e+offset]; sums divided by
//!   sqrt(move length). Null: the census's sign flip of B's observations, 3 copies, seeded
//!   exactly like the census (`null_seed(task, period, offset, copy)`).
//! * GPU, FP32: one N x N product per (pair, offset, copy), N = externals + stocks; three
//!   blocks are kept: ext->stock, ext->ext (off-diagonal) and stock->ext.
//! * False-discovery rule (q = 0.05) and placebo exactly as the census's `choose_thresholds`
//!   and `placebo_cutoff`, per block (pairs = the block's pair count) and per block x category.
//!
//! `entries.bin` layout (little-endian, 24 bytes per entry, every real and null |z| >= 3.5):
//!   u16 period, u8 h1, u8 h2, u16 offset, u8 copy (0 real, 1-3 null), u8 direction
//!   (0 ext->stock, 1 ext->ext, 2 stock->ext), u32 A id, u32 B id, u16 observations, u16 0,
//!   f32 r. Ids: stocks are store ids; externals are 1,000,000 + the row in `externals.csv`.
//!   z = atanh(r) * sqrt(observations - 3).

// Tables are indexed by direction and offset class throughout; index loops read best there.
#![allow(clippy::needless_range_loop)]

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use c2_census::grid;
use c2_census::meta::{IdMap, RunMeta};
use c2_census::run::{FLAG_NULL, RECORD_BYTES, Record};
use c2_census::variable::{self, Variable};
use c2_data::{Calendar, Store, date_string, days_from_civil};
use c2_engine::gpu::{DevPanel, DevProduct, Gpu, Precision};
use rayon::prelude::*;

const EXT_BASE: u32 = 1_000_000;
const R_BINS: usize = 1024;
const NULLS: usize = 3;
const STORE_Z: f64 = 3.5;
const COVERAGE: f64 = 0.9;
const MIN_ACTIVE: usize = 10;
const SYN: &str = "synthetic";
/// Category for finite-life single-contract series (flagged in the catalog's notes).
const SINGLE: &str = "single_contract";
/// Pure-noise calibration series (`--noise N`): the baseline every ratio is compared with.
const NOISE: &str = "noise_test";
/// Annual series: one release a year, often all on the same date, so they act as one calendar
/// pulse that standardization pushes to +-5. Reported apart unless `--include-annual`.
const ANNUAL: &str = "annual";
/// Short-period mode's baselines: Google Trends daily series and a random sample of the rest
/// (crypto keeps its own category).
const TRENDS_DAILY: &str = "trends_daily";
const SAMPLE: &str = "sample_others";
const SYN_LOAD: f64 = 0.3;
/// Out-of-sample tallies of relationships picked from the noise externals (real data).
const NOISE_SRC: usize = 10;
const DIRS: [&str; 3] = ["ext->stock", "ext->ext", "stock->ext"];
/// Offset classes: 0 = offset 0; 1 = offset >= 1 beyond the follower's reporting delay; 2 = (B an
/// external) offset >= 1 but shorter than B's move length plus its reporting delay and
/// measurement span, so B's move can describe the same physical days as A's: timing, not a lead.
const OFFSETS: [&str; 3] = ["same day", "predictive", "within delay"];

const USAGE: &str = "external_test: does non-stock data lead US stock moves? (census-faithful test)

  cargo run --release -p c2-census --example external_test -- [options]

  --run NAME        results go to <data>/external/results/NAME (default ext-<today>)
  --census NAME     census run giving the correlation periods and stock universes (default main-2025-06-27)
  --external DIR    external data root with catalog_*.csv and series/ (default <data>/external)
  --length L        rolling length of the non-overlapping correlation periods (default: the census's longest, 1260)
  --periods LIST    correlation period ids of the run (default: chosen by length and date: the chain of
                    non-overlapping L-day periods from the census's first day, plus full history)
  --include-annual  count annual series in the headline numbers (default: reported apart)
  --short-periods   prediction-market mode: the census's non-overlapping 252-day periods from --short-from
                    (default 2021-07-01) on, no full history; headline = rolling prediction-market series;
                    baselines reported apart: Google Trends daily, crypto, a random --sample N (default 100)
                    of the other series, and the --noise series; finite-life contracts reported apart
  --pairs LIST      move-length pairs as A:B, e.g. 1:1,5:21 (default all 36)
  --synthetic-only  skip the catalogs; only the injected synthetic series (pipeline check)
  --noise N         add N pure-noise calibration series (reported apart from the headline; 220 recommended)
  --catalogs LIST   only these catalogs, e.g. data-markets,data-other,data-weather (default: every catalog_*.csv)
  --selftest        run the self-checks only
  --browse RUN [--series TEXT] [--period ID] [--min-z Z] [--null] [--limit N]
                    print a finished run's stored entries (|z| >= 3.5), strongest first
";

// ---------------------------------------------------------------------------------------------
// Arguments

struct Args {
    run: String,
    census: String,
    external: PathBuf,
    synthetic_only: bool,
    selftest: bool,
    /// Correlation period ids to test (default: chosen by length and date, see `select_periods`).
    periods: Option<Vec<usize>>,
    /// Rolling length of the non-overlapping correlation periods (default: the census's longest).
    length: Option<usize>,
    /// Count annual series in the headline numbers too.
    include_annual: bool,
    /// Short-period mode: the census's non-overlapping 252-day periods from `short_from` on, for
    /// series too young for 1260-day periods (prediction markets), with matched baselines.
    short: bool,
    short_from: i32,
    sample: usize,
    pairs: Option<Vec<(usize, usize)>>,
    browse: Option<Vec<String>>,
    command: String,
    noise: usize,
    /// Only these catalogs (names after `catalog_`); empty = every catalog present.
    catalogs: Vec<String>,
}

fn next_val(it: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    it.next().with_context(|| format!("{name} needs a value"))
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        run: String::new(),
        census: "main-2025-06-27".into(),
        external: c2_data::data_dir().join("external"),
        synthetic_only: false,
        selftest: false,
        periods: None,
        length: None,
        include_annual: false,
        short: false,
        short_from: days_from_civil(2021, 7, 1),
        sample: 100,
        pairs: None,
        browse: None,
        command: std::env::args().skip(1).collect::<Vec<_>>().join(" "),
        noise: 0,
        catalogs: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--run" => a.run = next_val(&mut it, "--run")?,
            "--census" => a.census = next_val(&mut it, "--census")?,
            "--external" => a.external = next_val(&mut it, "--external")?.into(),
            "--synthetic-only" => a.synthetic_only = true,
            "--length" => a.length = Some(next_val(&mut it, "--length")?.parse()?),
            "--include-annual" => a.include_annual = true,
            "--short-periods" => a.short = true,
            "--short-from" => a.short_from = c2_data::parse_date(&next_val(&mut it, "--short-from")?)?,
            "--sample" => a.sample = next_val(&mut it, "--sample")?.parse()?,
            "--noise" => a.noise = next_val(&mut it, "--noise")?.parse()?,
            "--catalogs" => a.catalogs = next_val(&mut it, "--catalogs")?.split(',').map(|s| s.trim().to_string()).collect(),
            "--selftest" => a.selftest = true,
            "--periods" => {
                a.periods = Some(next_val(&mut it, "--periods")?.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>()?);
            }
            "--pairs" => {
                let v = next_val(&mut it, "--pairs")?;
                let mut pairs = Vec::new();
                for p in v.split(',') {
                    let (x, y) = p.split_once(':').with_context(|| format!("pair {p:?} is not A:B"))?;
                    pairs.push((x.trim().parse()?, y.trim().parse()?));
                }
                a.pairs = Some(pairs);
            }
            "--browse" => {
                a.browse = Some(it.by_ref().collect());
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown option {other}\n{USAGE}"),
        }
    }
    if a.run.is_empty() {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
        a.run = format!("ext-{}", date_string((now / 86_400) as i32));
    }
    ensure!(a.run.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)), "run name {:?}: use letters, digits, - _ .", a.run);
    Ok(a)
}

fn say(t0: &Instant, msg: &str) {
    eprintln!("[{:7.1}s] {msg}", t0.elapsed().as_secs_f64());
}

/// Thousands separators.
fn th(v: f64) -> String {
    let n = v.round() as i64;
    let s = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    if n < 0 { format!("-{out}") } else { out }
}

fn ratio(real: f64, chance: f64) -> String {
    if chance > 0.0 { format!("{:.2}", real / chance) } else if real > 0.0 { "inf".into() } else { "-".into() }
}

fn pct(a: f64, b: f64) -> String {
    if b > 0.0 { format!("{:.1}%", 100.0 * a / b) } else { "-".into() }
}

// ---------------------------------------------------------------------------------------------
// CSV (RFC 4180: quotes, doubled quotes, CRLF, BOM)

fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let (mut rows, mut row, mut field) = (Vec::new(), Vec::new(), String::new());
    let (mut in_q, mut any) = (false, false);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_q {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_q = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' => {
                in_q = true;
                any = true;
            }
            ',' => {
                row.push(std::mem::take(&mut field));
                any = true;
            }
            '\r' => {}
            '\n' => {
                if any || !field.is_empty() {
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                }
                any = false;
            }
            _ => {
                field.push(c);
                any = true;
            }
        }
    }
    if any || !field.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

// ---------------------------------------------------------------------------------------------
// External series

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transform {
    LogDiff,
    Diff,
    Anomaly,
    AnomalyLog1p,
    Level,
}

impl Transform {
    fn parse(s: &str) -> Option<Transform> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "logdiff" => Transform::LogDiff,
            "diff" => Transform::Diff,
            "anomaly" => Transform::Anomaly,
            "anomaly_log1p" => Transform::AnomalyLog1p,
            "level" => Transform::Level,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Transform::LogDiff => "logdiff",
            Transform::Diff => "diff",
            Transform::Anomaly => "anomaly",
            Transform::AnomalyLog1p => "anomaly_log1p",
            Transform::Level => "level",
        }
    }
}

#[derive(Clone, Debug)]
struct Meta {
    id: String,
    name: String,
    category: String,
    subcategory: String,
    region: String,
    frequency: String,
    units: String,
    catalog: String,
    transform: Transform,
    known_after: i32,
}

struct Series {
    meta: Meta,
    obs: Vec<(i32, f64)>,
}

/// One external series on the census variable's rows.
struct ExtVar {
    /// Census-style variable; NaN where unavailable.
    v: Vec<f32>,
    /// Unstandardized move per row: log change (logdiff), change (diff), mean anomaly
    /// (anomaly kinds) or mean value (level); 0 on live rows without a new observation, NaN
    /// where the series is not live.
    raw: Vec<f64>,
    /// Trading rows a series may go without a new observation and still count as live.
    gap: usize,
}

/// Rows of staleness allowed by frequency: the larger of the catalog's and the one the median
/// spacing of the first 100 observations implies (so later data can't change earlier days).
fn live_gap(freq: &str, obs: &[(i32, f64)]) -> usize {
    let named = match freq.trim().to_ascii_lowercase().as_str() {
        "daily" => 10,
        "weekly" => 15,
        "monthly" => 50,
        "quarterly" => 140,
        "annual" | "yearly" => 300,
        _ => 0,
    };
    let mut gaps: Vec<i32> = obs.windows(2).take(100).map(|w| w[1].0 - w[0].0).collect();
    let inferred = if gaps.is_empty() {
        300
    } else {
        let mid = gaps.len() / 2;
        match *gaps.select_nth_unstable(mid).1 {
            ..=4 => 10,
            5..=10 => 15,
            11..=45 => 50,
            46..=120 => 140,
            _ => 300,
        }
    };
    named.max(inferred)
}

/// For each observation, whether the level change since the previous one counts as a move:
/// the gap may be at most the 90th-percentile spacing of the previous 250 gaps plus 2 days
/// (weekends and holidays for weekday series, a late month for monthly ones), but exactly 1 day
/// for series observed every day, so a missing day (a prediction market's contract roll, an
/// outage) is never bridged. Causal: it looks only at earlier gaps.
fn bridged(obs: &[(i32, f64)]) -> Vec<bool> {
    let mut out = vec![true; obs.len()];
    let mut win: VecDeque<i32> = VecDeque::new();
    let mut counts: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
    for i in 1..obs.len() {
        let gap = obs[i].0 - obs[i - 1].0;
        if win.len() >= 20 {
            let target = (win.len() * 9 / 10).min(win.len() - 1);
            let (mut seen, mut p90) = (0usize, 0i32);
            for (&g, &c) in &counts {
                seen += c;
                if seen > target {
                    p90 = g;
                    break;
                }
            }
            out[i] = gap <= if p90 <= 1 { 1 } else { p90 + 2 };
        }
        win.push_back(gap);
        *counts.entry(gap).or_default() += 1;
        if win.len() > 250 {
            let old = win.pop_front().unwrap();
            let c = counts.get_mut(&old).unwrap();
            *c -= 1;
            if *c == 0 {
                counts.remove(&old);
            }
        }
    }
    out
}

/// Trading rows by which an external's value can trail the days it describes: its
/// known_after_days (calendar days, an upper bound in rows) plus its measurement span.
fn delay_rows(m: &Meta) -> usize {
    let span = match m.frequency.as_str() {
        "weekly" => 5,
        "monthly" => 21,
        "quarterly" => 63,
        "annual" | "yearly" => 252,
        _ => 0,
    };
    m.known_after.max(0) as usize + span
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 { 29 } else { 28 }
        }
    }
}

/// Monday = 0 .. Sunday = 6.
fn weekday(d: i32) -> i32 {
    (d + 3).rem_euclid(7)
}

/// Per-observation anomalies: x - mean of x over the 15 days centred on the same date in each
/// of the previous 10 years (x = value, or ln(1 + value)); needs at least 3 such years.
fn anomalies(obs: &[(i32, f64)], log1p: bool) -> Vec<(i32, f64)> {
    let xs: Vec<(i32, f64)> = obs
        .iter()
        .filter_map(|&(d, v)| {
            let x = if log1p {
                if v > -1.0 { v.ln_1p() } else { return None }
            } else {
                v
            };
            x.is_finite().then_some((d, x))
        })
        .collect();
    let (Some(&(d0, _)), Some(&(d1, _))) = (xs.first(), xs.last()) else { return Vec::new() };
    let mut dense = vec![f64::NAN; (d1 - d0 + 1) as usize];
    for &(d, x) in &xs {
        dense[(d - d0) as usize] = x;
    }
    let get = |d: i32| if d < d0 || d > d1 { f64::NAN } else { dense[(d - d0) as usize] };
    xs.iter()
        .filter_map(|&(d, x)| {
            let (y, m, dd) = c2_data::civil_from_days(d);
            let (mut s, mut c, mut years) = (0.0, 0usize, 0usize);
            for yy in y - 10..y {
                let centre = days_from_civil(yy, m, dd.min(days_in_month(yy, m)));
                let mut any = false;
                for o in -7..=7 {
                    let v = get(centre + o);
                    if v.is_finite() {
                        s += v;
                        c += 1;
                        any = true;
                    }
                }
                years += usize::from(any);
            }
            (years >= 3).then(|| (d, x - s / c as f64))
        })
        .collect()
}

/// The census's standardization (variable.rs, second pass), on any daily series: minus the
/// trailing mean, divided by the trailing sd (rows d-252..d-1, at least 126 values), clipped at
/// +-5. The trailing sums are recomputed for every row (no running-sum drift), and a trailing
/// stretch without a single non-zero value gives no value at all (sparse indicators can't
/// divide by a zero or round-off sd).
fn standardize(x: &[f64]) -> Vec<f32> {
    let (lb, min_obs, clip) = (variable::LOOKBACK, variable::MIN_OBS, variable::CLIP as f64);
    let mut out = vec![f32::NAN; x.len()];
    out.par_iter_mut().enumerate().for_each(|(d, o)| {
        if !x[d].is_finite() {
            return;
        }
        let (mut s, mut ss, mut k, mut nz) = (0f64, 0f64, 0usize, 0usize);
        for &v in &x[d.saturating_sub(lb)..d] {
            if v.is_finite() {
                s += v;
                ss += v * v;
                k += 1;
                nz += usize::from(v != 0.0);
            }
        }
        if k < min_obs || nz == 0 {
            return;
        }
        let kf = k as f64;
        let var = (ss - s * s / kf) / kf;
        if var > 1e-12 * (ss / kf) && var > 0.0 {
            *o = ((x[d] - s / kf) / var.sqrt()).clamp(-clip, clip) as f32;
        }
    });
    out
}

fn build_ext(days: &[i32], s: &Series) -> ExtVar {
    let t = days.len();
    let tf = s.meta.transform;
    let gap = live_gap(&s.meta.frequency, &s.obs);
    let obs: Vec<(i32, f64)> = match tf {
        Transform::LogDiff => s.obs.iter().filter(|o| o.1 > 0.0).map(|&(d, v)| (d, v.ln())).collect(),
        Transform::Diff | Transform::Level => s.obs.clone(),
        Transform::Anomaly => anomalies(&s.obs, false),
        Transform::AnomalyLog1p => anomalies(&s.obs, true),
    };
    // Row at which each observation becomes usable: the first trading day >= date + k.
    let avail: Vec<usize> = obs.iter().map(|o| days.partition_point(|&d| d < o.0.saturating_add(s.meta.known_after))).collect();
    let bridge = bridged(&obs);
    let (mut x, mut raw) = (vec![f64::NAN; t], vec![f64::NAN; t]);
    let mut i = 0;
    let mut level: Option<f64> = None;
    let mut last_date: Option<i32> = None;
    let mut last: Option<usize> = None;
    let mut trail: VecDeque<(usize, f64)> = VecDeque::new();
    let mut tsum = 0f64;
    for row in 0..t {
        let (mut sum, mut cnt, mut newest, mut broken) = (0f64, 0usize, f64::NAN, false);
        while i < obs.len() && avail[i] <= row {
            // A level change across a gap longer than the series' normal spacing (a contract
            // roll, an outage) is not a move.
            broken |= last_date.is_some() && !bridge[i];
            last_date = Some(obs[i].0);
            sum += obs[i].1;
            cnt += 1;
            newest = obs[i].1;
            i += 1;
        }
        if cnt > 0 {
            last = Some(row);
        }
        let live = row > 0 && last.is_some_and(|l| row - l <= gap);
        while trail.front().is_some_and(|&(r, _)| r + variable::LOOKBACK < row) {
            tsum -= trail.pop_front().unwrap().1;
        }
        match tf {
            Transform::LogDiff | Transform::Diff => {
                if cnt > 0 {
                    if let (Some(l), true, false) = (level, row > 0, broken) {
                        x[row] = newest - l;
                    }
                    level = Some(newest);
                } else if live && level.is_some() {
                    x[row] = 0.0;
                }
                raw[row] = x[row];
            }
            Transform::Anomaly | Transform::AnomalyLog1p => {
                if cnt > 0 && row > 0 {
                    x[row] = sum / cnt as f64;
                } else if live {
                    x[row] = 0.0;
                }
                raw[row] = x[row];
            }
            Transform::Level => {
                let enough = trail.len() >= 10;
                if cnt > 0 {
                    let val = sum / cnt as f64;
                    if row > 0 && enough {
                        x[row] = val - tsum / trail.len() as f64;
                        raw[row] = val;
                    }
                    trail.push_back((row, val));
                    tsum += val;
                } else if live && enough {
                    x[row] = 0.0;
                    raw[row] = 0.0;
                }
            }
        }
    }
    ExtVar { v: standardize(&x), raw, gap }
}

fn read_series(path: &Path) -> Result<(Vec<(i32, f64)>, usize)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let (mut obs, mut bad) = (Vec::new(), 0usize);
    for (n, line) in text.lines().enumerate() {
        let line = line.trim().trim_start_matches('\u{feff}');
        if line.is_empty() || (n == 0 && line.to_ascii_lowercase().starts_with("date")) {
            continue;
        }
        let mut f = line.split(',');
        let (Some(d), Some(v)) = (f.next(), f.next()) else {
            bad += 1;
            continue;
        };
        match (c2_data::parse_date(d.trim().trim_matches('"')), v.trim().trim_matches('"').parse::<f64>()) {
            (Ok(d), Ok(v)) if v.is_finite() => obs.push((d, v)),
            _ => bad += 1,
        }
    }
    obs.sort_by_key(|o| o.0);
    let before = obs.len();
    obs.dedup_by_key(|o| o.0);
    let dups = before - obs.len();
    Ok((obs, bad + dups))
}

/// A catalog note flagging a finite-life single-contract series. The predictions catalog marks
/// them with an upper-case "FINITE-LIFE" ("FINITE-LIFE single contract", "FINITE-LIFE, FINAL 90
/// DAYS ONLY", "FINITE-LIFE PLAY-MONEY FORECAST"); its continuous series built by stitching
/// contracts say "ROLLING (finite-life contracts stitched)" and are not flagged. Lower-case
/// mentions count unless the note describes a rolling or stitched series or negates them.
fn single_contract(notes: &str) -> bool {
    if notes.contains("FINITE-LIFE") {
        return true;
    }
    let n = notes.to_ascii_lowercase();
    if n.contains("rolling") || n.contains("stitched") {
        return false;
    }
    ["finite-life", "finite life", "single-contract", "single contract"].iter().any(|p| {
        n.match_indices(p).any(|(i, _)| {
            let before = n[..i].trim_end();
            !["not a", "not", "no", "non-", "non"].iter().any(|neg| before.ends_with(neg))
        })
    })
}

/// One release a year: by the catalog's frequency, or a median spacing of 300 days or more.
fn is_annual(m: &Meta, obs: &[(i32, f64)]) -> bool {
    if matches!(m.frequency.as_str(), "annual" | "yearly") {
        return true;
    }
    let mut gaps: Vec<i32> = obs.windows(2).map(|w| w[1].0 - w[0].0).collect();
    if gaps.is_empty() {
        return false;
    }
    let mid = gaps.len() / 2;
    *gaps.select_nth_unstable(mid).1 >= 300
}

/// Per catalog: (name, series listed, series loaded).
type CatalogCounts = Vec<(String, usize, usize)>;

fn read_catalogs(dir: &Path, only: &[String], warn: &mut Vec<String>) -> Result<(Vec<Series>, CatalogCounts)> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("no external data folder {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("catalog_") && n.ends_with(".csv")))
        .collect();
    files.sort();
    let (mut out, mut summary, mut seen) = (Vec::new(), Vec::new(), HashSet::new());
    for f in files {
        let cname = f.file_stem().and_then(|s| s.to_str()).unwrap_or("").trim_start_matches("catalog_").to_string();
        if !only.is_empty() && !only.contains(&cname) {
            warn.push(format!("catalog {cname} present but not included in this run (--catalogs)"));
            continue;
        }
        let rows = parse_csv(&std::fs::read_to_string(&f)?);
        let Some(header) = rows.first() else { continue };
        let col = |name: &str| header.iter().position(|h| h.trim().eq_ignore_ascii_case(name));
        let (Some(ci_id), Some(ci_tf), Some(ci_k)) = (col("id"), col("transform"), col("known_after_days")) else {
            warn.push(format!("{cname}: catalog lacks id/transform/known_after_days columns; skipped"));
            continue;
        };
        let cols = [col("name"), col("category"), col("subcategory"), col("region"), col("frequency"), col("units"), col("notes")];
        let (mut listed, mut used) = (0, 0);
        for r in &rows[1..] {
            let get = |i: Option<usize>| i.and_then(|i| r.get(i)).map(|s| s.trim().to_string()).unwrap_or_default();
            let id = get(Some(ci_id));
            if id.is_empty() {
                continue;
            }
            listed += 1;
            if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.') {
                warn.push(format!("{cname}: id {id:?} has unexpected characters; skipped"));
                continue;
            }
            if !seen.insert(id.clone()) {
                warn.push(format!("{cname}: duplicate id {id}; skipped"));
                continue;
            }
            let Some(tf) = Transform::parse(&get(Some(ci_tf))) else {
                warn.push(format!("{cname}: {id}: unknown transform {:?}; skipped", get(Some(ci_tf))));
                continue;
            };
            let k = match get(Some(ci_k)).parse::<f64>() {
                Ok(k) if k.is_finite() && k >= 0.0 => k.ceil() as i32,
                _ => {
                    warn.push(format!("{cname}: {id}: bad known_after_days {:?}; skipped", get(Some(ci_k))));
                    continue;
                }
            };
            let (obs, bad) = match read_series(&dir.join("series").join(format!("{id}.csv"))) {
                Ok(o) => o,
                Err(e) => {
                    warn.push(format!("{cname}: {id}: {e:#}; skipped"));
                    continue;
                }
            };
            if bad > 0 {
                warn.push(format!("{cname}: {id}: {bad} unreadable or duplicate rows dropped"));
            }
            if obs.len() < 2 {
                warn.push(format!("{cname}: {id}: only {} observations; skipped", obs.len()));
                continue;
            }
            let mut category = get(cols[1]).to_ascii_lowercase();
            if category.is_empty() || category == SYN || category == SINGLE || category == NOISE || category == ANNUAL {
                category = "other".into();
            }
            let mut subcategory = get(cols[2]);
            // Finite-life single-contract series (prediction markets) are reported apart from the
            // headline numbers.
            if single_contract(&get(cols[6])) {
                subcategory = format!("{category}/{subcategory}");
                category = SINGLE.into();
            }
            let meta = Meta {
                id: id.clone(),
                name: get(cols[0]),
                category,
                subcategory,
                region: get(cols[3]),
                frequency: get(cols[4]).to_ascii_lowercase(),
                units: get(cols[5]),
                catalog: cname.clone(),
                transform: tf,
                known_after: k,
            };
            out.push(Series { meta, obs });
            used += 1;
        }
        summary.push((cname, listed, used));
    }
    Ok((out, summary))
}

// ---------------------------------------------------------------------------------------------
// Synthetic series: positive controls (a stock's census moves shifted in time, plus noise) and
// negative controls (noise of every transform and timing kind). Kept in memory only.

struct Rng(u64);

impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
    fn poisson(&mut self, lambda: f64) -> f64 {
        let (l, mut k, mut p) = ((-lambda).exp(), 0.0, 1.0);
        loop {
            p *= self.uniform();
            if p <= l {
                return k;
            }
            k += 1.0;
        }
    }
}

fn syn_meta(id: &str, name: String, sub: &str, freq: &str, tf: Transform, k: i32) -> Meta {
    Meta {
        id: id.into(),
        name,
        category: SYN.into(),
        subcategory: sub.into(),
        region: "-".into(),
        frequency: freq.into(),
        units: "-".into(),
        catalog: "synthetic".into(),
        transform: tf,
        known_after: k,
    }
}

/// What a synthetic positive control was built from: B's (or A's) census variable `shift` rows
/// later (positive: the external moves first) or earlier (negative: the external follows).
#[derive(Clone)]
struct SynSpec {
    id: String,
    ticker: String,
    shift: isize,
}

const NOISE_KINDS: usize = 11;

/// One noise series of each timing and transform kind: (id suffix, description, frequency,
/// transform, known_after_days, observations).
fn noise_kind(kind: usize, rng: &mut Rng, all: &[i32], weekdays: &[i32]) -> (&'static str, &'static str, &'static str, Transform, i32, Vec<(i32, f64)>) {
    let doy = |d: i32| (d - days_from_civil(c2_data::civil_from_days(d).0, 1, 1)) as f64;
    let mut lv = 0f64;
    match kind {
        0 => ("rw_cal_k0", "random walk on calendar days, 2% gaps", "daily", Transform::LogDiff, 0,
            all.iter().filter_map(|&d| { lv += 0.01 * rng.normal(); (rng.uniform() >= 0.02).then_some((d, 100.0 * lv.exp())) }).collect()),
        1 => ("rw_cal_k1", "random walk on calendar days, known next day", "daily", Transform::LogDiff, 1,
            all.iter().map(|&d| { lv += 0.01 * rng.normal(); (d, 100.0 * lv.exp()) }).collect()),
        2 => ("fat_wd", "fat-tailed random walk on weekdays", "daily", Transform::LogDiff, 0,
            weekdays.iter().map(|&d| { lv += 0.01 * rng.normal() * if rng.uniform() < 0.05 { 4.0 } else { 1.0 }; (d, 100.0 * lv.exp()) }).collect()),
        3 => {
            let (mut s2, mut prev) = (1f64, 0f64);
            ("garch_wd", "GARCH(1,1) random walk on weekdays", "daily", Transform::LogDiff, 0,
                weekdays.iter().map(|&d| { s2 = 0.05 + 0.10 * prev * prev + 0.85 * s2; prev = s2.sqrt() * rng.normal(); lv += 0.01 * prev; (d, 100.0 * lv.exp()) }).collect())
        }
        4 => ("weekly_diff_k5", "weekly random walk dated Saturdays, known 5 days later", "weekly", Transform::Diff, 5,
            all.iter().filter(|&&d| weekday(d) == 5).map(|&d| { lv += rng.normal(); (d, lv) }).collect()),
        5 => ("weekly_level_k7", "weekly level dated by week start (Sunday), known 7 days later (Trends-like)", "weekly", Transform::Level, 7,
            all.iter().filter(|&&d| weekday(d) == 6).map(|&d| (d, 50.0 + 10.0 * rng.normal())).collect()),
        6 => ("monthly_k30", "monthly random walk dated the 1st, known 30 days later", "monthly", Transform::LogDiff, 30,
            all.iter().filter(|&&d| c2_data::civil_from_days(d).2 == 1).map(|&d| { lv += 0.01 * rng.normal(); (d, 100.0 * lv.exp()) }).collect()),
        7 => {
            let mut ar = 0f64;
            ("temp_k1", "seasonal temperature-like series, known next day", "daily", Transform::Anomaly, 1,
                all.iter().map(|&d| { ar = 0.7 * ar + 3.0 * rng.normal(); (d, 12.0 + 10.0 * (std::f64::consts::TAU * (doy(d) - 105.0) / 365.25).sin() + ar) }).collect())
        }
        8 => ("precip_k1", "zero-heavy seasonal precipitation-like series, known next day", "daily", Transform::AnomalyLog1p, 1,
            all.iter().map(|&d| { let wet = 0.35 + 0.15 * (std::f64::consts::TAU * doy(d) / 365.25).sin(); (d, if rng.uniform() < wet { 5.0 * (0.8 * rng.normal()).exp() } else { 0.0 }) }).collect()),
        9 => ("counts_k1", "daily Poisson(0.7) counts, known next day", "daily", Transform::Level, 1, all.iter().map(|&d| (d, rng.poisson(0.7))).collect()),
        _ => ("sparse_k0", "sparse 0/1 event indicator on weekdays (3% of days)", "daily", Transform::Level, 0,
            weekdays.iter().map(|&d| (d, f64::from(u8::from(rng.uniform() < 0.03)))).collect()),
    }
}

fn synthetic(var: &Variable, store: &Store, col_of: &HashMap<usize, usize>, extra: usize) -> Result<(Vec<Series>, Vec<SynSpec>)> {
    let mut out = Vec::new();
    let mut specs = Vec::new();
    let col = |t: &str| -> Result<usize> {
        let id = store.id(t).with_context(|| format!("{t} not in the store"))?;
        col_of.get(&id).copied().with_context(|| format!("{t} not in the census variable"))
    };
    for (id, ticker, shift, seed) in [("syn_lead1_jnj", "JNJ", 1isize, 11u64), ("syn_lead5_pg", "PG", 5, 12), ("syn_follow3_ko", "KO", -3, 13)] {
        let c = var.col(&var.v, col(ticker)?);
        let mut rng = Rng(seed);
        let mut lv = 0f64;
        let obs = (0..var.t)
            .map(|t| {
                let s = t as isize + shift;
                let v = if s >= 0 && (s as usize) < var.t { c[s as usize] } else { f32::NAN };
                lv += 0.01 * (SYN_LOAD * if v.is_finite() { v as f64 } else { 0.0 } + rng.normal());
                (var.days[t], 100.0 * lv.exp())
            })
            .collect();
        let name = if shift > 0 {
            format!("synthetic: {ticker}'s census move {shift} trading days later x{SYN_LOAD} + noise (moves first)")
        } else {
            format!("synthetic: {ticker}'s census move {} trading days earlier x{SYN_LOAD} + noise (follows)", -shift)
        };
        out.push(Series { meta: syn_meta(id, name, "positive", "daily", Transform::LogDiff, 0), obs });
        specs.push(SynSpec { id: id.into(), ticker: ticker.into(), shift });
    }
    {
        let mut rng = Rng(21);
        let ex: Vec<f64> = (0..var.t).map(|_| rng.normal()).collect();
        let (mut lx, mut ly, mut ox, mut oy) = (0f64, 0f64, Vec::new(), Vec::new());
        for t in 0..var.t {
            lx += 0.01 * ex[t];
            ly += 0.01 * (SYN_LOAD * if t >= 2 { ex[t - 2] } else { 0.0 } + rng.normal());
            ox.push((var.days[t], 100.0 * lx.exp()));
            oy.push((var.days[t], 100.0 * ly.exp()));
        }
        out.push(Series { meta: syn_meta("syn_eex_x", "synthetic: random walk that syn_eex_y2 follows".into(), "positive", "daily", Transform::LogDiff, 0), obs: ox });
        out.push(Series {
            meta: syn_meta("syn_eex_y2", format!("synthetic: syn_eex_x's move 2 trading days later x{SYN_LOAD} + noise"), "positive", "daily", Transform::LogDiff, 0),
            obs: oy,
        });
    }
    // Negative controls, plus `extra` more noise series in a regular category (a null calibration
    // of the whole pipeline at the real data's scale; development runs only).
    let (d0, d1) = (days_from_civil(1998, 1, 1), days_from_civil(2025, 9, 30));
    let all: Vec<i32> = (d0..=d1).collect();
    let weekdays: Vec<i32> = all.iter().copied().filter(|&d| weekday(d) < 5).collect();
    let mut rng = Rng(777);
    for kind in 0..NOISE_KINDS {
        let (suffix, desc, freq, tf, k, obs) = noise_kind(kind, &mut rng, &all, &weekdays);
        out.push(Series { meta: syn_meta(&format!("noise_{suffix}"), format!("synthetic noise: {desc}"), "noise", freq, tf, k), obs });
    }
    for i in 0..extra {
        let mut rng = Rng(10_000 + i as u64);
        let (suffix, desc, freq, tf, k, obs) = noise_kind(i % NOISE_KINDS, &mut rng, &all, &weekdays);
        let mut meta = syn_meta(&format!("noisetest{i:04}_{suffix}"), format!("calibration noise: {desc}"), suffix, freq, tf, k);
        meta.category = NOISE.into();
        out.push(Series { meta, obs });
    }
    Ok((out, specs))
}

// ---------------------------------------------------------------------------------------------
// The census run: correlation periods, universes, task numbering, null seeds

struct Window {
    id: usize,
    /// Rolling length in trading days, or 0 for full history.
    length: usize,
    first: i32,
    last: i32,
    /// Variable rows, found by date (`RunMeta::window_rows_in`); empty when today's calendar
    /// doesn't cover the period exactly.
    rows: Range<usize>,
    covered: bool,
    /// Variable columns of the census universe (filled for the periods tested).
    stocks: Vec<usize>,
}

struct Census {
    meta: RunMeta,
    windows: Vec<Window>,
    /// (h1, h2, length) -> the run's own task id.
    tasks: HashMap<(usize, usize, usize), usize>,
    /// The run's store ids -> today's store ids.
    ids: IdMap,
}

impl Census {
    /// Today's store id for a store id of the run.
    fn current(&self, run_id: u32) -> Option<usize> {
        self.ids.current(run_id)
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The census's sign-flip seed (run.rs `null_seed`), so null copy t of (task, period, offset)
/// flips exactly the observations the census flipped.
fn null_seed(task: usize, window: usize, lag: usize, t: usize) -> u32 {
    (splitmix((task as u64) << 40 ^ (window as u64) << 20 ^ (lag as u64) << 4 ^ t as u64) as u32) | 1
}

struct Check {
    ok: bool,
    what: String,
}

fn check(checks: &mut Vec<Check>, ok: bool, what: String) {
    eprintln!("  [{}] {what}", if ok { "ok" } else { "FAIL" });
    checks.push(Check { ok, what });
}

/// Reads the census run through `RunMeta`: its settings, its own correlation periods (mapped onto
/// today's variable by date), task numbering and stock ids (translated to today's store).
fn load_census(meta: RunMeta, store: &Store, var: &Variable, checks: &mut Vec<Check>) -> Result<Census> {
    let cfg = &meta.cfg;
    ensure!(cfg.nulls == NULLS, "census {} has {} null copies; this test needs the census's {NULLS}", meta.name, cfg.nulls);
    let ids = meta.id_map(store)?;
    let windows: Vec<Window> = meta
        .windows
        .iter()
        .map(|w| {
            let rows = meta.window_rows_in(&var.days, w.id);
            Window { id: w.id, length: w.length, first: w.first, last: w.last, covered: rows.is_some(), rows: rows.unwrap_or(0..0), stocks: Vec::new() }
        })
        .collect();
    let tasks: HashMap<(usize, usize, usize), usize> = meta.tasks()?.iter().map(|t| ((t.h1, t.h2, t.length), t.id)).collect();
    check(checks, true, format!("census run {}: {} correlation periods and {} tasks from its own files; settings \"{}\"; stock ids {}",
        meta.name, windows.len(), tasks.len(), meta.config_text,
        if ids.is_identity() { "identical to today's store".to_string() } else { format!("translated to today's store through tickers ({} untranslated)", ids.untranslated()) }));
    Ok(Census { meta, windows, tasks, ids })
}

/// The correlation periods to test, chosen by length and dates from the run's own list (never by
/// id, which a new census renumbers): the chain of non-overlapping `length`-day periods that
/// starts on the census's first trading day, each starting the trading day after the previous one
/// ends, plus the full-history period from that day. Returns (chain, full history).
fn select_periods(c: &Census, var: &Variable, length: usize, from: Option<i32>, checks: &mut Vec<Check>) -> Result<(Vec<usize>, Option<usize>)> {
    let census_first = var.days.partition_point(|&d| d < c.meta.cfg.start);
    // The chain starts with the first `length`-day period starting on or after `from` (default:
    // the census's first day).
    let first_row = match from {
        None => census_first,
        Some(f) => c
            .windows
            .iter()
            .filter(|w| w.length == length && w.covered && w.first >= f)
            .map(|w| w.rows.start)
            .min()
            .with_context(|| format!("census {} has no {length}-day period starting on or after {}", c.meta.name, date_string(f)))?,
    };
    ensure!(first_row < var.t, "the census starts after today's data ends");
    let rolling: Vec<&Window> = c.windows.iter().filter(|w| w.length == length && w.covered).collect();
    ensure!(!rolling.is_empty(), "census {} has no {length}-day correlation periods that today's calendar covers (lengths: {:?})", c.meta.name, c.meta.cfg.lengths);
    let starts: Vec<&&Window> = rolling.iter().filter(|w| w.rows.start == first_row).collect();
    ensure!(starts.len() == 1, "census {}: {} {length}-day periods start on its first day {}; expected exactly 1", c.meta.name, starts.len(), date_string(var.days[first_row]));
    let mut chain = vec![starts[0].id];
    loop {
        let end = c.windows[*chain.last().unwrap()].rows.end;
        let next: Vec<&&Window> = rolling.iter().filter(|w| w.rows.start == end).collect();
        match next.len() {
            0 => break,
            1 => chain.push(next[0].id),
            k => bail!("census {}: {k} {length}-day periods start on {}", c.meta.name, date_string(var.days[end])),
        }
    }
    // Validate: exact length, contiguous, non-overlapping, in date order.
    for (k, &id) in chain.iter().enumerate() {
        let w = &c.windows[id];
        ensure!(w.rows.len() == length, "period {id} spans {} trading days, not {length}", w.rows.len());
        ensure!(var.days[w.rows.start] == w.first && var.days[w.rows.end - 1] == w.last, "period {id}'s dates don't match its rows");
        if k > 0 {
            let prev = &c.windows[chain[k - 1]];
            ensure!(w.rows.start == prev.rows.end && w.first > prev.last, "periods {} and {id} are not consecutive", prev.id);
        }
    }
    let fulls: Vec<&Window> = if from.is_some() { Vec::new() } else { c.windows.iter().filter(|w| w.length == 0 && w.covered && w.rows.start == census_first).collect() };
    ensure!(fulls.len() <= 1, "census {} has {} full-history periods", c.meta.name, fulls.len());
    let full = fulls.first().map(|w| w.id);
    let span = |id: usize| format!("{} ({}..{})", id, date_string(c.windows[id].first), date_string(c.windows[id].last));
    check(checks, chain.len() >= 2, format!("correlation periods chosen by length and date, not id: {} non-overlapping consecutive {length}-day periods from {} ({}){}",
        chain.len(), from.map(|f| format!("the first starting on or after {}", date_string(f))).unwrap_or_else(|| "the census's first day".into()), chain.iter().map(|&id| span(id)).collect::<Vec<_>>().join(", "),
        full.map(|f| format!(", and full history {}", span(f))).unwrap_or_else(|| ", no full-history period".into())));
    Ok((chain, full))
}

/// Fills a tested period's universe: the run's store ids -> today's store ids -> variable columns.
fn fill_universe(c: &mut Census, id: usize, store: &Store, col_of: &HashMap<usize, usize>) -> Result<()> {
    ensure!(c.windows[id].covered, "census {}: today's calendar doesn't cover correlation period {id} exactly", c.meta.name);
    let mut cols = Vec::with_capacity(c.meta.universes[id].len());
    for &rid in &c.meta.universes[id] {
        let ticker = c.meta.stocks().ok().and_then(|s| s.ticker(rid)).unwrap_or("?").to_string();
        let sid = c.current(rid).with_context(|| format!("census {}: period {id}: stock {ticker} isn't in today's store", c.meta.name))?;
        cols.push(*col_of.get(&sid).with_context(|| format!("census {}: period {id}: stock {} isn't in today's census variable", c.meta.name, store.symbols[sid].ticker))?);
    }
    c.windows[id].stocks = cols;
    Ok(())
}

/// Exact r for leader prefix `pa` and target prefix `pb` (prefix sums over a period's rows,
/// missing as 0) on the census grid: a copy of run.rs `exact`.
fn exact(pa: &[f64], pb: &[f64], h1: usize, h2: usize, lag: usize, m: usize, seed: u32) -> f64 {
    let (s, g0) = (h1.max(h2), h1.max(h2) - 1);
    let (sh1, sh2) = ((h1 as f64).sqrt(), (h2 as f64).sqrt());
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for q in 0..m {
        let e = g0 + q * s;
        let x = (pa[e + 1] - pa[e + 1 - h1]) / sh1;
        let et = e + lag;
        let mut y = (pb[et + 1] - pb[et + 1 - h2]) / sh2;
        if seed != 0 {
            y *= c2_engine::sign_flip(seed, q as u32);
        }
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    c2_engine::pearson(sxy, sx, sy, sxx, syy, m as f64)
}

fn zscore(r: f64, m: usize) -> f64 {
    r.clamp(-0.999_999, 0.999_999).atanh() * ((m as f64 - 3.0).max(1.0)).sqrt()
}

fn prefix_of(v: &[f32]) -> Vec<f64> {
    let mut p = vec![0f64; v.len() + 1];
    for (r, &x) in v.iter().enumerate() {
        p[r + 1] = p[r] + if x.is_finite() { x as f64 } else { 0.0 };
    }
    p
}

/// Recomputes stored census records (real and null) from today's variable and this program's
/// grid and seeds: proves the stock side and the null are the census's.
fn replicate(c: &Census, var: &Variable, col_of: &HashMap<usize, usize>, chain: &[usize], full: Option<usize>, checks: &mut Vec<Check>) -> Result<()> {
    // Two move-length pairs per period, rotating through those with observations at this length.
    let mut cases: Vec<(usize, usize, usize)> = Vec::new();
    for (k, &w) in chain.iter().enumerate().take(4) {
        let len = c.windows[w].rows.len();
        let ok: Vec<(usize, usize)> = [(1, 1), (5, 21), (21, 5), (3, 10), (10, 1), (5, 5), (3, 1), (1, 5)].into_iter().filter(|&(a, b)| grid::observations(len, a, b, 0) >= c.meta.cfg.min_obs).collect();
        for j in 0..2.min(ok.len()) {
            let (h1, h2) = ok[(2 * k + j) % ok.len()];
            cases.push((w, h1, h2));
        }
    }
    if let Some(f) = full {
        cases.push((f, 63, 63));
        cases.push((f, 1, 1));
    }
    let (mut count, mut worst, mut nulls) = (0usize, 0f64, 0usize);
    for &(wid, h1, h2) in &cases {
        let win = &c.windows[wid];
        let Some(&task) = c.tasks.get(&(h1, h2, win.length)) else { continue };
        let Ok(bytes) = std::fs::read(c.meta.task_path(task, "rec")) else { continue };
        let recs: Vec<Record> = bytes.chunks_exact(RECORD_BYTES).map(Record::from_bytes).filter(|r| r.window as usize == wid).collect();
        let pick = |null: bool| -> Vec<Record> {
            let v: Vec<Record> = recs.iter().filter(|r| (r.flags & FLAG_NULL != 0) == null).copied().collect();
            let step = (v.len() / 25).max(1);
            v.into_iter().step_by(step).take(25).collect()
        };
        for rec in pick(false).into_iter().chain(pick(true)) {
            let col = |rid: u32| c.current(rid).and_then(|s| col_of.get(&s).copied());
            let (Some(i), Some(j)) = (col(rec.leader), col(rec.target)) else { bail!("a census record's stock is not in today's census variable") };
            let pa = prefix_of(&var.col(&var.v, i)[win.rows.clone()]);
            let pb = prefix_of(&var.col(&var.v, j)[win.rows.clone()]);
            let (lag, m) = (rec.lag as usize, rec.obs as usize);
            ensure!(m == grid::observations(win.rows.len(), h1, h2, lag), "observation count differs from the census's");
            let null = rec.flags & FLAG_NULL != 0;
            let seed = if null { null_seed(task, wid, lag, 0) } else { 0 };
            let r = exact(&pa, &pb, h1, h2, lag, m, seed);
            worst = worst.max((r - rec.r as f64).abs());
            count += 1;
            nulls += usize::from(null);
        }
    }
    check(checks, worst < 2e-6 && count > 100, format!("census records reproduced: {count} stored stock->stock correlations ({nulls} of them null copies) in {} (period, move-length) cases, largest |r difference| {worst:.1e}", cases.len()));
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// One correlation period on the GPU

struct ExtInfo {
    meta: Meta,
    var: ExtVar,
    cat: usize,
    /// Per requested period: (coverage, active rows, joins).
    per: Vec<(f64, usize, bool)>,
}

#[derive(Clone, Copy)]
struct Entry {
    period: u16,
    h1: u8,
    h2: u8,
    lag: u16,
    copy: u8,
    dir: u8,
    a: u32,
    b: u32,
    m: u16,
    r: f32,
}

impl Entry {
    fn z(&self) -> f64 {
        zscore(self.r as f64, self.m as usize)
    }
    fn bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.period.to_le_bytes());
        out.push(self.h1);
        out.push(self.h2);
        out.extend_from_slice(&self.lag.to_le_bytes());
        out.push(self.copy);
        out.push(self.dir);
        out.extend_from_slice(&self.a.to_le_bytes());
        out.extend_from_slice(&self.b.to_le_bytes());
        out.extend_from_slice(&self.m.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&self.r.to_le_bytes());
    }
    fn from_bytes(b: &[u8]) -> Entry {
        let u16_at = |p: usize| u16::from_le_bytes([b[p], b[p + 1]]);
        let u32_at = |p: usize| u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        Entry {
            period: u16_at(0),
            h1: b[2],
            h2: b[3],
            lag: u16_at(4),
            copy: b[6],
            dir: b[7],
            a: u32_at(8),
            b: u32_at(12),
            m: u16_at(16),
            r: f32::from_le_bytes(b[20..24].try_into().unwrap()),
        }
    }
}

/// Counts for one (period, pair, offset, direction, category).
struct MRow {
    period: usize,
    h1: usize,
    h2: usize,
    lag: usize,
    m: usize,
    dir: usize,
    cat: usize,
    pairs: u64,
    real: [u64; 3],
    /// Summed over the 3 null copies.
    null: [u64; 3],
    /// This category's real (null copy 2) entries above the block's false-discovery (placebo) cut-off.
    fdr_n: u64,
    placebo_n: u64,
    /// The rule run within this category alone.
    cfdr: Option<usize>,
    cfdr_n: u64,
    cplacebo: Option<usize>,
    cplacebo_n: u64,
}

/// One block of one product: the census's "matrix".
struct FRow {
    period: usize,
    h1: usize,
    h2: usize,
    lag: usize,
    m: usize,
    dir: usize,
    pairs: u64,
    fdr: Option<usize>,
    fdr_n: u64,
    placebo: Option<usize>,
    placebo_n: u64,
}

struct PeriodOut {
    id: usize,
    e: usize,
    s: usize,
    rows: Vec<MRow>,
    mats: Vec<FRow>,
    entries: Vec<Entry>,
    products: usize,
    secs: f64,
    gpu_secs: f64,
    scan_secs: f64,
    max_dr: f64,
    checked: usize,
}

struct Factors {
    sx: Vec<f64>,
    ax: Vec<f64>,
    my: Vec<f64>,
    by: Vec<f64>,
}

fn factors(sx: &[f64], sxx: &[f64], sy: &[f64], syy: &[f64], m: usize) -> Factors {
    let mf = m as f64;
    let inv = |s: f64, ss: f64| {
        let v = ss - s * s / mf;
        if v > 1e-9 * mf { 1.0 / v.sqrt() } else { 0.0 }
    };
    Factors {
        sx: sx.to_vec(),
        ax: sx.iter().zip(sxx).map(|(&s, &q)| inv(s, q)).collect(),
        my: sy.iter().map(|&s| s / mf).collect(),
        by: sy.iter().zip(syy).map(|(&s, &q)| inv(s, q)).collect(),
    }
}

struct Scan {
    hist: Vec<u32>,
    cnt: Vec<u64>,
    stored: Vec<(u32, u32, f32)>,
}

impl Scan {
    fn new(ncat: usize) -> Scan {
        Scan { hist: vec![0; 3 * ncat * R_BINS], cnt: vec![0; 3 * ncat * 3], stored: Vec::new() }
    }
    fn merge(mut self, o: Scan) -> Scan {
        for (a, b) in self.hist.iter_mut().zip(&o.hist) {
            *a += b;
        }
        for (a, b) in self.cnt.iter_mut().zip(&o.cnt) {
            *a += b;
        }
        self.stored.extend(o.stored);
        self
    }
    #[inline]
    fn tally(&mut self, slot: usize, r: f64, thr: &[f64; 4], i: usize, j: usize) {
        let a = r.abs().min(0.999_999);
        self.hist[slot * R_BINS + ((a * R_BINS as f64) as usize).min(R_BINS - 1)] += 1;
        if a >= thr[0] {
            self.cnt[slot * 3] += 1;
            if a >= thr[1] {
                self.cnt[slot * 3 + 1] += 1;
                if a >= thr[2] {
                    self.cnt[slot * 3 + 2] += 1;
                }
            }
            if a >= thr[3] {
                self.stored.push((i as u32, j as u32, r as f32));
            }
        }
    }
}

/// r for every entry of the three blocks of an N x N product `c` (c[i + j*n]: A = column i of
/// X, B = column j of Y; externals are columns 0..e): |r| histograms, |z| >= 3/4/5 counts and
/// entries >= 3.5, per (direction, category). Slot = direction * ncat + category; an ext->ext
/// pair involving a series of an excluded category (index >= excl0) counts under that category.
#[allow(clippy::too_many_arguments)]
fn scan(c: &[f32], n: usize, e: usize, f: &Factors, colcat: &[usize], excl0: usize, ncat: usize, thr: &[f64; 4]) -> Scan {
    let cols: Vec<usize> = (0..n).collect();
    let chunk = n.div_ceil(rayon::current_num_threads() * 3).max(1);
    cols.par_chunks(chunk)
        .map(|js| {
            let mut sc = Scan::new(ncat);
            for &j in js {
                let col = &c[j * n..(j + 1) * n];
                let (myj, byj) = (f.my[j], f.by[j]);
                if j < e {
                    let cj = colcat[j];
                    for i in 0..e {
                        if i == j {
                            continue;
                        }
                        let r = (col[i] as f64 - f.sx[i] * myj) * f.ax[i] * byj;
                        sc.tally(ncat + ee_cat(colcat[i], cj, excl0), r, thr, i, j);
                    }
                    for i in e..n {
                        let r = (col[i] as f64 - f.sx[i] * myj) * f.ax[i] * byj;
                        sc.tally(2 * ncat + cj, r, thr, i, j);
                    }
                } else {
                    for i in 0..e {
                        let r = (col[i] as f64 - f.sx[i] * myj) * f.ax[i] * byj;
                        sc.tally(colcat[i], r, thr, i, j);
                    }
                }
            }
            sc
        })
        .reduce(|| Scan::new(ncat), Scan::merge)
}

/// Category of an ext->ext pair: A's, unless either series is in an excluded category.
#[inline]
fn ee_cat(ci: usize, cj: usize, excl0: usize) -> usize {
    if ci >= excl0 || cj >= excl0 { ci.max(cj) } else { ci }
}

/// Complementary error function (run.rs; Numerical Recipes erfcc).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.265_512_23 + t * (1.000_023_68 + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06
        + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))))).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

/// The census's false-discovery cut-off (`choose_thresholds`): the lowest |r| bin b where the
/// real count at or above b is >= 10 and expected chance / real <= q, expected chance being the
/// larger of the nulls' mean count and the normal-theory count for `pairs` pairs. With null 2
/// as "real" against nulls 1 and 3 it is the census's placebo (`placebo_cutoff`).
fn fdr_bin(real: &[u32], nulls: &[&[u32]], pairs: f64, m: usize, q: f64) -> Option<usize> {
    let zs = ((m as f64 - 3.0).max(1.0)).sqrt();
    let (mut cr, mut cn, mut best) = (0f64, 0f64, None);
    for b in (0..R_BINS).rev() {
        cr += real[b] as f64;
        cn += nulls.iter().map(|h| h[b] as f64).sum::<f64>() / nulls.len().max(1) as f64;
        let r = (b as f64 / R_BINS as f64).min(0.999_999);
        let analytic = pairs * erfc(r.atanh() * zs / std::f64::consts::SQRT_2);
        if cr >= 10.0 && cn.max(analytic) / cr <= q {
            best = Some(b);
        }
    }
    best
}

fn above(h: &[u32], b: Option<usize>) -> u64 {
    b.map(|b| h[b..].iter().map(|&v| v as u64).sum()).unwrap_or(0)
}

struct Ctx<'a> {
    var: &'a Variable,
    ext: &'a [ExtInfo],
    ncat: usize,
    /// Categories from this index on (single-contract, synthetic) stay out of headline numbers.
    excl0: usize,
    census: &'a Census,
    pairs: Vec<(usize, usize)>,
    min_obs: usize,
    q: f64,
}

fn run_period(gpu: &Gpu, cx: &Ctx, p: &Window, exts_in: &[usize], t0: &Instant) -> Result<PeriodOut> {
    let started = Instant::now();
    let var = cx.var;
    let (e, s) = (exts_in.len(), p.stocks.len());
    let n = e + s;
    let w = p.rows.len();
    let ld = w + 1;
    let (ncat, excl0) = (cx.ncat, cx.excl0);
    let mut out = PeriodOut { id: p.id, e, s, rows: vec![], mats: vec![], entries: vec![], products: 0, secs: 0.0, gpu_secs: 0.0, scan_secs: 0.0, max_dr: 0.0, checked: 0 };
    if e == 0 || s == 0 {
        return Ok(out);
    }
    let colcat: Vec<usize> = exts_in.iter().map(|&x| cx.ext[x].cat).collect();
    let gid: Vec<u32> = exts_in.iter().map(|&x| EXT_BASE + x as u32).chain(p.stocks.iter().map(|&j| var.ids[j] as u32)).collect();
    let mut prefix = vec![0f64; ld * n];
    prefix.par_chunks_mut(ld).enumerate().for_each(|(c, pc)| {
        for r in 0..w {
            let row = p.rows.start + r;
            let v = if c < e { cx.ext[exts_in[c]].var.v[row] } else { var.v[p.stocks[c - e] * var.t + row] };
            pc[r + 1] = pc[r] + if v.is_finite() { v as f64 } else { 0.0 };
        }
    });
    // Pair counts per (direction, category).
    let mut npairs = vec![0u64; 3 * ncat];
    for i in 0..e {
        npairs[colcat[i]] += s as u64;
        npairs[2 * ncat + colcat[i]] += s as u64;
        for j in 0..e {
            if i != j {
                npairs[ncat + ee_cat(colcat[i], colcat[j], excl0)] += 1;
            }
        }
    }
    let st = gpu.stream.clone();
    let kmax = w;
    let mut dx = DevPanel::F32(st.alloc_zeros::<f32>(kmax * n)?);
    let mut dy = DevPanel::F32(st.alloc_zeros::<f32>(kmax * n)?);
    let mut dc = gpu.alloc_product(Precision::Fp32, n)?;
    let (mut hx, mut hy, mut hyr) = (vec![0f32; kmax * n], vec![0f32; kmax * n], vec![0f32; kmax * n]);
    // The product comes back into page-locked (cached, not write-combined) host memory: about
    // twice the transfer rate of pageable memory. Every element is written before it is read.
    let mut hc_pin = unsafe { gpu.ctx.alloc_pinned_with_flags::<f32>(n * n, 0)? };
    let mut rng = Rng(p.id as u64 * 7919);
    for &(h1, h2) in &cx.pairs {
        let (step, g0) = (h1.max(h2), h1.max(h2) - 1);
        let lags: Vec<(usize, usize)> = grid::lags(h2).into_iter().map(|k| (k, grid::observations(w, h1, h2, k))).filter(|&(_, m)| m >= cx.min_obs).collect();
        if lags.is_empty() {
            continue;
        }
        let task = *cx.census.tasks.get(&(h1, h2, p.length)).with_context(|| format!("no census task for ({h1},{h2},{})", p.length))?;
        let m0 = lags[0].1;
        let (sh1, sh2) = ((h1 as f64).sqrt(), (h2 as f64).sqrt());
        // A's panel (observations 0..m0) and its running column sums.
        let xs: Vec<(Vec<f64>, Vec<f64>)> = hx[..m0 * n]
            .par_chunks_mut(m0)
            .enumerate()
            .map(|(c, xc)| {
                let pc = &prefix[c * ld..(c + 1) * ld];
                let (mut cs, mut css) = (vec![0f64; m0 + 1], vec![0f64; m0 + 1]);
                for q in 0..m0 {
                    let en = g0 + q * step;
                    let x = ((pc[en + 1] - pc[en + 1 - h1]) / sh1) as f32;
                    xc[q] = x;
                    cs[q + 1] = cs[q] + x as f64;
                    css[q + 1] = css[q] + (x as f64) * (x as f64);
                }
                (cs, css)
            })
            .collect();
        let tg = Instant::now();
        if let DevPanel::F32(d) = &mut dx {
            st.memcpy_htod(&hx[..m0 * n], d)?;
        }
        out.gpu_secs += tg.elapsed().as_secs_f64();
        for (li, &(lag, m)) in lags.iter().enumerate() {
            let sx: Vec<f64> = xs.iter().map(|c| c.0[m]).collect();
            let sxx: Vec<f64> = xs.iter().map(|c| c.1[m]).collect();
            let thr = [3.0, 4.0, 5.0, STORE_Z].map(|z| (z / ((m as f64 - 3.0).max(1.0)).sqrt()).tanh());
            let mut hists: Vec<Vec<u32>> = Vec::with_capacity(NULLS + 1);
            let mut cnts: Vec<Vec<u64>> = Vec::with_capacity(NULLS + 1);
            // B's real panel once per offset; a null copy flips the sign of whole observations, which
            // is exact in f32 and leaves each column's sum of squares unchanged.
            let ysr: Vec<(f64, f64)> = hyr[..m0 * n]
                .par_chunks_mut(m0)
                .enumerate()
                .map(|(c, yc)| {
                    let pc = &prefix[c * ld..(c + 1) * ld];
                    let (mut sy, mut syy) = (0f64, 0f64);
                    for q in 0..m0 {
                        if q < m {
                            let et = g0 + q * step + lag;
                            let y = ((pc[et + 1] - pc[et + 1 - h2]) / sh2) as f32;
                            yc[q] = y;
                            sy += y as f64;
                            syy += (y as f64) * (y as f64);
                        } else {
                            yc[q] = 0.0;
                        }
                    }
                    (sy, syy)
                })
                .collect();
            let syy: Vec<f64> = ysr.iter().map(|v| v.1).collect();
            for copy in 0..=NULLS {
                let seed = if copy == 0 { 0 } else { null_seed(task, p.id, lag, copy - 1) };
                let sy: Vec<f64> = if copy == 0 {
                    ysr.iter().map(|v| v.0).collect()
                } else {
                    let signs: Vec<f32> = (0..m).map(|q| c2_engine::sign_flip(seed, q as u32) as f32).collect();
                    hy[..m0 * n]
                        .par_chunks_mut(m0)
                        .zip(hyr[..m0 * n].par_chunks(m0))
                        .map(|(yc, yr)| {
                            let mut sy = 0f64;
                            for q in 0..m0 {
                                let y = if q < m { yr[q] * signs[q] } else { 0.0 };
                                yc[q] = y;
                                sy += y as f64;
                            }
                            sy
                        })
                        .collect()
                };
                let tg = Instant::now();
                if let DevPanel::F32(d) = &mut dy {
                    st.memcpy_htod(if copy == 0 { &hyr[..m0 * n] } else { &hy[..m0 * n] }, d)?;
                }
                gpu.xprod(Precision::Fp32, None, &dx, &dy, &mut dc, n, m0)?;
                if let DevProduct::F32(d) = &dc {
                    st.memcpy_dtoh(d, &mut hc_pin)?;
                }
                st.synchronize()?;
                let hc = hc_pin.as_slice()?;
                out.gpu_secs += tg.elapsed().as_secs_f64();
                out.products += 1;
                let ts = Instant::now();
                let f = factors(&sx, &sxx, &sy, &syy, m);
                let sc = scan(hc, n, e, &f, &colcat, excl0, ncat, &thr);
                // Spot check against exact f64 sums, real and first null copy, first offset.
                if li == 0 && copy <= 1 {
                    for _ in 0..48 {
                        let (i, j) = match rng.u64() % 3 {
                            0 => (rng.u64() as usize % e, e + rng.u64() as usize % s),
                            1 => (rng.u64() as usize % e, rng.u64() as usize % e),
                            _ => (e + rng.u64() as usize % s, rng.u64() as usize % e),
                        };
                        if i == j {
                            continue;
                        }
                        let rg = (hc[i + j * n] as f64 - f.sx[i] * f.my[j]) * f.ax[i] * f.by[j];
                        let rx = exact(&prefix[i * ld..(i + 1) * ld], &prefix[j * ld..(j + 1) * ld], h1, h2, lag, m, seed);
                        out.max_dr = out.max_dr.max((rg - rx).abs());
                        out.checked += 1;
                    }
                }
                for &(i, j, r) in &sc.stored {
                    let (i, j) = (i as usize, j as usize);
                    let dir = if i < e && j >= e { 0 } else if i < e { 1 } else { 2 };
                    out.entries.push(Entry { period: p.id as u16, h1: h1 as u8, h2: h2 as u8, lag: lag as u16, copy: copy as u8, dir, a: gid[i], b: gid[j], m: m as u16, r });
                }
                hists.push(sc.hist);
                cnts.push(sc.cnt);
                out.scan_secs += ts.elapsed().as_secs_f64();
            }
            // False-discovery rule and placebo per block (headline categories) and per block x category.
            for d in 0..3 {
                let sl = |copy: usize, g: usize| &hists[copy][(d * ncat + g) * R_BINS..(d * ncat + g + 1) * R_BINS];
                let mut mh = vec![vec![0u32; R_BINS]; NULLS + 1];
                for (copy, h) in mh.iter_mut().enumerate() {
                    for g in 0..excl0 {
                        for (a, b) in h.iter_mut().zip(sl(copy, g)) {
                            *a += b;
                        }
                    }
                }
                let mpairs: u64 = (0..excl0).map(|g| npairs[d * ncat + g]).sum();
                let (fdr, placebo) = if mpairs > 0 {
                    (fdr_bin(&mh[0], &[&mh[1], &mh[2], &mh[3]], mpairs as f64, m, cx.q), fdr_bin(&mh[2], &[&mh[1], &mh[3]], mpairs as f64, m, cx.q))
                } else {
                    (None, None)
                };
                if mpairs > 0 {
                    out.mats.push(FRow { period: p.id, h1, h2, lag, m, dir: d, pairs: mpairs, fdr, fdr_n: above(&mh[0], fdr), placebo, placebo_n: above(&mh[2], placebo) });
                }
                for g in 0..ncat {
                    let pairs = npairs[d * ncat + g];
                    if pairs == 0 {
                        continue;
                    }
                    let cf = fdr_bin(sl(0, g), &[sl(1, g), sl(2, g), sl(3, g)], pairs as f64, m, cx.q);
                    let cp = fdr_bin(sl(2, g), &[sl(1, g), sl(3, g)], pairs as f64, m, cx.q);
                    let cnt = |copy: usize, t: usize| cnts[copy][(d * ncat + g) * 3 + t];
                    out.rows.push(MRow {
                        period: p.id, h1, h2, lag, m, dir: d, cat: g, pairs,
                        real: [cnt(0, 0), cnt(0, 1), cnt(0, 2)],
                        null: [0, 1, 2].map(|t| (1..=NULLS).map(|c| cnt(c, t)).sum()),
                        fdr_n: if g >= excl0 { 0 } else { above(sl(0, g), fdr) },
                        placebo_n: if g >= excl0 { 0 } else { above(sl(2, g), placebo) },
                        cfdr: cf, cfdr_n: above(sl(0, g), cf), cplacebo: cp, cplacebo_n: above(sl(2, g), cp),
                    });
                }
            }
        }
        say(t0, &format!("  period {}: move lengths {h1}/{h2} done ({} offsets), {} products so far, {} entries stored", p.id, lags.len(), out.products, th(out.entries.len() as f64)));
    }
    out.secs = started.elapsed().as_secs_f64();
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Analysis on the CPU: exact r anywhere, out-of-sample, controls, plain terms

struct Data<'a> {
    var: &'a Variable,
    ext: &'a [ExtInfo],
    col_of: &'a HashMap<usize, usize>,
    ext_pre: Vec<Vec<f64>>,
    stock_pre: HashMap<usize, Vec<f64>>,
    /// Per requested period: members (gids).
    members: HashMap<usize, HashSet<u32>>,
    windows: &'a [Window],
    store: &'a Store,
    excl0: usize,
}

impl Data<'_> {
    fn pre(&self, gid: u32) -> Option<&[f64]> {
        if gid >= EXT_BASE { self.ext_pre.get((gid - EXT_BASE) as usize).map(|v| v.as_slice()) } else { self.stock_pre.get(&(gid as usize)).map(|v| v.as_slice()) }
    }
    fn name(&self, gid: u32) -> String {
        if gid >= EXT_BASE { self.ext[(gid - EXT_BASE) as usize].meta.id.clone() } else { self.store.symbols[gid as usize].ticker.clone() }
    }
    fn is_syn(&self, gid: u32) -> bool {
        gid >= EXT_BASE && self.ext[(gid - EXT_BASE) as usize].meta.category == SYN
    }
    /// Offset class of an entry (see `OFFSETS`).
    fn offset_class(&self, e: &Entry) -> usize {
        if e.lag == 0 {
            0
        } else if e.dir != 0 && (e.lag as usize) < e.h2 as usize + delay_rows(&self.ext[(e.b - EXT_BASE) as usize].meta) {
            2
        } else {
            1
        }
    }
    /// The category an entry is counted under (as in the GPU scan).
    fn entry_cat(&self, e: &Entry) -> usize {
        let cat = |g: u32| self.ext[(g - EXT_BASE) as usize].cat;
        match e.dir {
            0 => cat(e.a),
            1 => ee_cat(cat(e.a), cat(e.b), self.excl0),
            _ => cat(e.b),
        }
    }
    fn stock_gid(&self, ticker: &str) -> Option<u32> {
        self.store.id(ticker).filter(|id| self.col_of.contains_key(id)).map(|id| id as u32)
    }
    /// Exact (r, z, m) in period `wid`, or None if either series is not in its universe.
    fn stat(&self, a: u32, b: u32, wid: usize, h1: usize, h2: usize, lag: usize) -> Option<(f64, f64, usize)> {
        let mem = self.members.get(&wid)?;
        if !mem.contains(&a) || !mem.contains(&b) {
            return None;
        }
        let win = &self.windows[wid];
        let m = grid::observations(win.rows.len(), h1, h2, lag);
        if m < 30 {
            return None;
        }
        let (pa, pb) = (self.pre(a)?, self.pre(b)?);
        let r = exact(&pa[win.rows.start..=win.rows.end], &pb[win.rows.start..=win.rows.end], h1, h2, lag, m, 0);
        Some((r, zscore(r, m), m))
    }
    /// A's census-style h-day sum ending at row e (for ranking A's moves).
    fn std_move(&self, gid: u32, e: usize, h: usize) -> Option<f64> {
        let p = self.pre(gid)?;
        (e + 1 >= h && e + 1 < p.len()).then(|| p[e + 1] - p[e + 1 - h])
    }
    /// Unstandardized move over rows (e-h, e]: log change of the repaired close for a stock,
    /// the external's raw moves summed; None if any day is missing.
    fn raw_move(&self, gid: u32, e: usize, h: usize) -> Option<f64> {
        if e + 1 < h || e >= self.var.t {
            return None;
        }
        if gid >= EXT_BASE {
            let raw = &self.ext[(gid - EXT_BASE) as usize].var.raw;
            let v: f64 = raw[e + 1 - h..=e].iter().sum();
            v.is_finite().then_some(v)
        } else {
            if e < h {
                return None;
            }
            let c = self.var.col(&self.var.close, *self.col_of.get(&(gid as usize))?);
            let (a, b) = (c[e - h], c[e]);
            (a.is_finite() && b.is_finite()).then(|| (b as f64 / a as f64).ln())
        }
    }
    /// Did B rise over its move? Stocks: repaired close up; logdiff/diff/anomaly externals: raw
    /// move above zero; level externals: above their recent average (standardized move > 0).
    fn b_up(&self, gid: u32, e: usize, h: usize) -> Option<bool> {
        if gid >= EXT_BASE && self.ext[(gid - EXT_BASE) as usize].meta.transform == Transform::Level {
            self.raw_move(gid, e, h)?;
            return self.std_move(gid, e, h).map(|v| v > 0.0);
        }
        self.raw_move(gid, e, h).map(|v| v > 0.0)
    }
    fn describe_raw(&self, gid: u32, v: f64) -> String {
        if gid < EXT_BASE {
            return format!("{:+.1}%", 100.0 * v.exp_m1());
        }
        let x = &self.ext[(gid - EXT_BASE) as usize].meta;
        match x.transform {
            Transform::LogDiff => format!("{:+.1}%", 100.0 * v.exp_m1()),
            Transform::Diff => format!("{v:+.3} {}", x.units),
            Transform::Anomaly => format!("{v:+.2} {} vs normal (summed over the days)", x.units),
            Transform::AnomalyLog1p => format!("{v:+.2} in ln(1+x) vs normal (summed over the days)"),
            Transform::Level => format!("{v:.2} {} (summed)", x.units),
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Quarters {
    cases: u32,
    base_up: u32,
    top: u32,
    top_up: u32,
    bot: u32,
    bot_up: u32,
    top_typ: f64,
    bot_typ: f64,
}

/// Plain-terms counts on the census grid: A's h1-day move ranked into quarters (by its
/// census-style value), B's h2-day move ending `lag` days after A's ends, up or not.
fn quarters(obs: &[(f64, f64, bool)]) -> Option<Quarters> {
    if obs.len() < 8 {
        return None;
    }
    let mut xs: Vec<f64> = obs.iter().map(|o| o.0).collect();
    xs.sort_by(|a, b| a.total_cmp(b));
    let (lo, hi) = (xs[(xs.len() - 1) / 4], xs[(3 * (xs.len() - 1)).div_ceil(4)]);
    let mut q = Quarters { cases: obs.len() as u32, ..Default::default() };
    let (mut tr, mut br) = (Vec::new(), Vec::new());
    for &(x, raw, up) in obs {
        q.base_up += u32::from(up);
        if x >= hi {
            q.top += 1;
            q.top_up += u32::from(up);
            tr.push(raw);
        }
        if x <= lo {
            q.bot += 1;
            q.bot_up += u32::from(up);
            br.push(raw);
        }
    }
    let med = |v: &mut Vec<f64>| if v.is_empty() { f64::NAN } else { let k = v.len() / 2; *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1 };
    q.top_typ = med(&mut tr);
    q.bot_typ = med(&mut br);
    Some(q)
}

/// (in the period, rest of history from the census start) for one relationship; the rest walks
/// the same grid and skips observations that share a day with the period (as `analyze::words`).
#[allow(clippy::too_many_arguments)]
fn plain(d: &Data, a: u32, b: u32, wid: usize, h1: usize, h2: usize, lag: usize, lo: usize) -> (Option<Quarters>, Option<Quarters>) {
    let win = &d.windows[wid];
    let (step, g0) = (h1.max(h2), h1.max(h2) - 1);
    let m = grid::observations(win.rows.len(), h1, h2, lag);
    let one = |en: usize| -> Option<(f64, f64, bool)> {
        Some((d.std_move(a, en, h1)?, d.raw_move(a, en, h1)?, d.b_up(b, en + lag, h2)?))
    };
    let inside: Vec<(f64, f64, bool)> = (0..m).filter_map(|q| one(win.rows.start + g0 + q * step)).collect();
    let first_e = win.rows.start + g0;
    let mut en = first_e % step;
    while en < lo + step {
        en += step;
    }
    let mut rest = Vec::new();
    let touches = |x: usize, y: usize| x < win.rows.end && y >= win.rows.start;
    while en + lag < d.var.t {
        if !touches(en + 1 - h1, en) && !touches(en + lag + 1 - h2, en + lag)
            && let Some(o) = one(en)
        {
            rest.push(o);
        }
        en += step;
    }
    (quarters(&inside), quarters(&rest))
}

#[derive(Default, Clone)]
struct Oos {
    n: usize,
    dropped: usize,
    same: usize,
    sum_sr: f64,
    rep2: usize,
    rep3: usize,
}

impl Oos {
    fn add(&mut self, o: &Oos) {
        self.n += o.n;
        self.dropped += o.dropped;
        self.same += o.same;
        self.sum_sr += o.sum_sr;
        self.rep2 += o.rep2;
        self.rep3 += o.rep3;
    }
}

struct Survivor {
    from: usize,
    to: usize,
    e: Entry,
    zp: f64,
    rq: f64,
    zq: f64,
}

/// Next-period exact (r, z) for each entry (None: A or B left the next period's universe).
fn next_r(d: &Data, sel: &[&Entry], to: usize) -> Vec<Option<(f64, f64)>> {
    sel.par_iter().map(|en| d.stat(en.a, en.b, to, en.h1 as usize, en.h2 as usize, en.lag as usize).map(|(r, z, _)| (r, z))).collect()
}

impl Oos {
    fn tally(&mut self, en: &Entry, next: Option<(f64, f64)>) {
        let Some((rq, zq)) = next else {
            self.dropped += 1;
            return;
        };
        self.n += 1;
        let sp = (en.r as f64).signum();
        let same = sp * rq > 0.0;
        self.same += usize::from(same);
        self.sum_sr += sp * rq;
        self.rep2 += usize::from(same && zq.abs() >= 2.0);
        self.rep3 += usize::from(same && zq.abs() >= 3.0);
    }
}

// ---------------------------------------------------------------------------------------------
// Self-checks of the external variable

fn selftests(days: &[i32], checks: &mut Vec<Check>) {
    // CSV parser.
    let rows = parse_csv("\u{feff}a,\"b,c\",d\r\n\"x\"\"y\",,z\n\nlast");
    check(checks, rows == vec![vec!["a", "b,c", "d"], vec!["x\"y", "", "z"], vec!["last"]], "CSV parser: quotes, doubled quotes, CRLF, BOM, blank lines".into());
    let meta = |tf: Transform, k: i32, freq: &str| syn_meta("t", "test".into(), "test", freq, tf, k);
    let row = |d: i32| days.binary_search(&d).ok();
    // A Friday whose next trading day is the following Monday, in the middle of the data.
    let fri = days.iter().copied().find(|&d| d > days_from_civil(2012, 3, 1) && weekday(d) == 4 && row(d + 3).is_some()).unwrap();
    let (fr, mo) = (row(fri).unwrap(), row(fri + 3).unwrap());
    let all: Vec<i32> = (days_from_civil(1999, 1, 1)..=days_from_civil(2025, 9, 1)).collect();
    // 1. logdiff on calendar days with a jump on that Friday: k=0 lands Friday, k=1 Monday.
    let jump: Vec<(i32, f64)> = all.iter().map(|&d| (d, if d >= fri { 110.0 } else { 100.0 })).collect();
    let v0 = build_ext(days, &Series { meta: meta(Transform::LogDiff, 0, "daily"), obs: jump.clone() });
    let v1 = build_ext(days, &Series { meta: meta(Transform::LogDiff, 1, "daily"), obs: jump });
    let j = 1.1f64.ln();
    check(checks, (v0.raw[fr] - j).abs() < 1e-12 && v0.raw[mo] == 0.0 && v1.raw[fr] == 0.0 && (v1.raw[mo] - j).abs() < 1e-12,
        format!("timing: a level change dated Friday {} lands that Friday with known_after_days 0 and on Monday {} with 1", date_string(fri), date_string(fri + 3)));
    // 2. Weekly series dated by week start (Sunday) with known_after_days 7 (Google Trends): the
    // week's value counts only from the first trading day after the week ends.
    let sun0 = fri + 2 - 14;
    let weekly: Vec<(i32, f64)> = all.iter().copied().filter(|&d| weekday(d) == 6).map(|d| (d, if d >= sun0 { 60.0 } else { 50.0 })).collect();
    let vw = build_ext(days, &Series { meta: meta(Transform::Diff, 7, "weekly"), obs: weekly });
    let land = days.partition_point(|&d| d < sun0 + 7);
    let quiet = (land.saturating_sub(30)..land).all(|r| vw.raw[r] == 0.0);
    check(checks, (vw.raw[land] - 10.0).abs() < 1e-12 && quiet && days[land] > sun0 + 6,
        format!("timing: a weekly value for the week starting Sunday {} (known_after_days 7) lands on {}, the first trading day after the week ends", date_string(sun0), date_string(days[land])));
    // 3. Weekend observations with known_after_days 1 are averaged onto Monday.
    let lv: Vec<(i32, f64)> = all.iter().map(|&d| (d, if d == fri { 1.0 } else if d == fri + 1 { 2.0 } else if d == fri + 2 { 3.0 } else if d == fri + 3 { 10.0 } else { 0.0 })).collect();
    let vl = build_ext(days, &Series { meta: meta(Transform::Level, 1, "daily"), obs: lv });
    check(checks, (vl.raw[mo] - 2.0).abs() < 1e-12 && (vl.raw[mo + 1] - 10.0).abs() < 1e-12,
        "timing: Friday, Saturday and Sunday values known the next day are averaged onto Monday; Monday's lands Tuesday".into());
    // 3a. Finite-life flags as the predictions catalog writes them.
    let flags = [
        ("FINITE-LIFE single contract (crypto): \"Will Bitcoin hit $100k in 2024?\"", true),
        ("FINITE-LIFE PLAY-MONEY FORECAST, not a real-money market", true),
        ("FINITE-LIFE, FINAL 90 DAYS ONLY: \"Which party will win the 2020 election?\"", true),
        ("ROLLING (finite-life contracts stitched): Kalshi KXFED strike ladder", false),
        ("ROLLING by election cycle: House control after 2024, then 2026", false),
        ("a continuous index, not a single contract", false),
        ("one finite-life contract", true),
        ("finite-life single contract RECSSNBER-23 (open 2022-11-07, close 2024-01-25, settled: no)", true),
        ("finite-life single contract; play-money forecast, not a real-money market", true),
        ("ROLLING continuous series (successive contracts stitched): Kalshi KXFED/FED", false),
    ];
    check(checks, flags.iter().all(|&(n, want)| single_contract(n) == want), "finite-life flags: single contracts marked FINITE-LIFE are excluded from the headline; ROLLING (stitched) series are not".into());
    // 3b. Roll gaps: a series observed every day that misses a day doesn't bridge the gap (a
    // prediction market's contract roll is not a move); a weekday series bridges a holiday.
    let skip = fri + 3; // Monday
    let daily: Vec<(i32, f64)> = all.iter().copied().filter(|&d| d != skip).map(|d| (d, if d > skip { 120.0 } else { 100.0 })).collect();
    let vd = build_ext(days, &Series { meta: meta(Transform::LogDiff, 0, "daily"), obs: daily });
    let wkd: Vec<(i32, f64)> = all.iter().copied().filter(|&d| weekday(d) < 5 && d != skip).map(|d| (d, if d > skip { 120.0 } else { 100.0 })).collect();
    let vk = build_ext(days, &Series { meta: meta(Transform::LogDiff, 0, "daily"), obs: wkd });
    let tue = row(skip + 1).unwrap();
    check(checks, vd.raw[tue].is_nan() && vd.raw[tue + 1] == 0.0 && (vk.raw[tue] - 1.2f64.ln()).abs() < 1e-12,
        "roll gaps: a 7-day series missing a day doesn't count the jump across it as a move; a weekday series bridges a 1-day holiday".into());
    // 4. No look-ahead: changing everything dated on or after a cut date leaves every earlier
    // trading day's value bit-identical, for every transform.
    let mut rng = Rng(99);
    let cut = days_from_civil(2013, 6, 14);
    let mut ok_all = true;
    let mut detail = String::new();
    for (tf, k, freq) in [(Transform::LogDiff, 0, "daily"), (Transform::Diff, 5, "weekly"), (Transform::Anomaly, 1, "daily"), (Transform::AnomalyLog1p, 1, "daily"), (Transform::Level, 7, "weekly"), (Transform::LogDiff, 30, "monthly")] {
        let ds: Vec<i32> = match freq {
            "weekly" => all.iter().copied().filter(|&d| weekday(d) == 6).collect(),
            "monthly" => all.iter().copied().filter(|&d| c2_data::civil_from_days(d).2 == 1).collect(),
            _ => all.iter().copied().filter(|_| rng.uniform() > 0.03).collect(),
        };
        let draw = |rng: &mut Rng, d: i32| -> f64 {
            let doy = (d % 365) as f64;
            match tf {
                Transform::LogDiff => 100.0 * (0.2 * rng.normal()).exp(),
                Transform::AnomalyLog1p => if rng.uniform() < 0.6 { 0.0 } else { 3.0 * rng.uniform() },
                _ => 10.0 * (doy / 58.0).sin() + rng.normal(),
            }
        };
        let base: Vec<(i32, f64)> = ds.iter().map(|&d| (d, draw(&mut rng, d))).collect();
        let mut pert: Vec<(i32, f64)> = Vec::new();
        for &(d, v) in &base {
            if d < cut {
                pert.push((d, v));
            } else if rng.uniform() > 0.2 {
                pert.push((d, draw(&mut rng, d)));
            }
        }
        let a = build_ext(days, &Series { meta: meta(tf, k, freq), obs: base });
        let b = build_ext(days, &Series { meta: meta(tf, k, freq), obs: pert });
        let first = days.partition_point(|&d| d < cut + k);
        let same = (0..first).all(|r| a.v[r].to_bits() == b.v[r].to_bits() && a.raw[r].to_bits() == b.raw[r].to_bits());
        let finite = (0..first).filter(|&r| a.v[r].is_finite()).count();
        let differs = (first..days.len()).any(|r| a.v[r].to_bits() != b.v[r].to_bits());
        ok_all &= same && differs && finite > 1000;
        let first_diff = (0..first).find(|&r| a.v[r].to_bits() != b.v[r].to_bits() || a.raw[r].to_bits() != b.raw[r].to_bits());
        let _ = write!(detail, "{} k={k}: {}{}; ", tf.name(), if same { format!("{finite} earlier values identical") } else { format!("earlier value CHANGED at {}", date_string(days[first_diff.unwrap()])) }, if differs { "" } else { " (but later ones didn't change!)" });
    }
    check(checks, ok_all, format!("no look-ahead: values dated >= {} changed or dropped; {detail}", date_string(cut)));
    // 5. Anomaly recovery: a seasonal series plus noise gives back the noise.
    let mut rng = Rng(5);
    let noise: Vec<f64> = all.iter().map(|_| rng.normal()).collect();
    let doy = |d: i32| (d - days_from_civil(c2_data::civil_from_days(d).0, 1, 1)) as f64;
    let obs: Vec<(i32, f64)> = all.iter().zip(&noise).map(|(&d, &e)| (d, 15.0 + 12.0 * (std::f64::consts::TAU * doy(d) / 365.25).sin() + e)).collect();
    let an = anomalies(&obs, false);
    let nmap: HashMap<i32, f64> = all.iter().copied().zip(noise.iter().copied()).collect();
    let pairs: Vec<(f64, f64)> = an.iter().filter(|o| o.0 >= days_from_civil(2004, 1, 1)).map(|o| (o.1, nmap[&o.0])).collect();
    let corr = {
        let n = pairs.len() as f64;
        let (sx, sy) = pairs.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
        let (sxx, syy, sxy) = pairs.iter().fold((0.0, 0.0, 0.0), |a, p| (a.0 + p.0 * p.0, a.1 + p.1 * p.1, a.2 + p.0 * p.1));
        c2_engine::pearson(sxy, sx, sy, sxx, syy, n)
    };
    let first_an = an.first().map(|o| o.0).unwrap_or(0);
    check(checks, corr > 0.98 && first_an >= days_from_civil(2002, 1, 1) - 7,
        format!("anomaly: seasonal cycle removed (correlation with the injected noise {corr:.3}); first anomaly {} (needs 3 earlier years)", date_string(first_an)));
    // 6. Sparse 0/1 indicator: bounded, no blow-up; an event-free trailing year gives no value.
    let mut rng = Rng(8);
    let sparse: Vec<(i32, f64)> = all.iter().copied().filter(|&d| weekday(d) < 5).map(|d| (d, if d < days_from_civil(2010, 1, 1) && rng.uniform() < 0.01 { 1.0 } else { 0.0 })).collect();
    let vs = build_ext(days, &Series { meta: meta(Transform::Level, 0, "daily"), obs: sparse });
    let maxabs = vs.v.iter().filter(|v| v.is_finite()).fold(0f32, |a, v| a.max(v.abs()));
    let late = row(days_from_civil(2012, 6, 1)).or_else(|| Some(days.partition_point(|&d| d < days_from_civil(2012, 6, 1)))).unwrap();
    check(checks, maxabs <= 5.0 && vs.v[late].is_nan() && vs.v.iter().all(|v| !v.is_infinite()),
        format!("sparse 0/1 events: values bounded by +-5 (max {maxabs:.2}), none once a year passes without an event"));
}

// ---------------------------------------------------------------------------------------------
// Browse a finished run

fn browse(args: &[String]) -> Result<()> {
    let run = args.first().context("--browse needs a run name")?;
    let dir = c2_data::data_dir().join("external/results").join(run);
    let (mut series, mut period, mut min_z, mut null, mut limit) = (String::new(), None, 4.0, false, 50usize);
    let mut it = args.iter().skip(1).cloned();
    while let Some(x) = it.next() {
        match x.as_str() {
            "--series" => series = next_val(&mut it, "--series")?,
            "--period" => period = Some(next_val(&mut it, "--period")?.parse::<u16>()?),
            "--min-z" => min_z = next_val(&mut it, "--min-z")?.parse()?,
            "--null" => null = true,
            "--real" => null = false,
            "--limit" => limit = next_val(&mut it, "--limit")?.parse()?,
            o => bail!("unknown browse option {o}"),
        }
    }
    let ext = parse_csv(&std::fs::read_to_string(dir.join("externals.csv"))?);
    let names: Vec<String> = ext.iter().skip(1).map(|r| r[1].clone()).collect();
    let store = Store::open(&c2_data::data_dir().join("store"))?;
    let name = |g: u32| if g >= EXT_BASE { names.get((g - EXT_BASE) as usize).cloned().unwrap_or_default() } else { store.symbols.get(g as usize).map(|s| s.ticker.clone()).unwrap_or_default() };
    let bytes = std::fs::read(dir.join("entries.bin"))?;
    let mut sel: Vec<Entry> = bytes
        .chunks_exact(24)
        .map(Entry::from_bytes)
        .filter(|e| (e.copy > 0) == null && period.is_none_or(|p| p == e.period) && e.z().abs() >= min_z)
        .filter(|e| series.is_empty() || name(e.a).contains(&series) || name(e.b).contains(&series))
        .collect();
    sel.sort_by(|a, b| b.z().abs().total_cmp(&a.z().abs()));
    println!("period  A moves first            B follows                moves  offset  obs       r      z  {}", if null { "null copy" } else { "" });
    for e in sel.iter().take(limit) {
        println!("{:>6}  {:<24} {:<24} {:>2}/{:<2}  {:>6}  {:>4}  {:+.3}  {:+5.1}  {}", e.period, name(e.a), name(e.b), e.h1, e.h2, e.lag, e.m, e.r, e.z(), if null { e.copy.to_string() } else { DIRS[e.dir as usize].to_string() });
    }
    println!("{} matching entries", sel.len());
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Main

fn main() -> Result<()> {
    c2_engine::gpu::reexec_with_cuda_libraries();
    let args = parse_args()?;
    if let Some(b) = &args.browse {
        return browse(b);
    }
    let t0 = Instant::now();
    let data = c2_data::data_dir();
    let mut timing: Vec<(String, f64)> = Vec::new();
    let mut lap = Instant::now();
    let mut mark = |what: &str, lap: &mut Instant| {
        timing.push((what.to_string(), lap.elapsed().as_secs_f64()));
        *lap = Instant::now();
    };
    say(&t0, "loading the price store, the census run and building the census variable");
    let store = Store::open(&data.join("store"))?;
    let cal = Calendar::trading_days(&store)?;
    let meta = RunMeta::load(&data.join("census").join(&args.census))?;
    let cfg = meta.cfg.clone();
    let var = variable::production(&store, &cal, cfg.start)?;
    let col_of: HashMap<usize, usize> = var.ids.iter().enumerate().map(|(j, &id)| (id, j)).collect();
    mark("census variable", &mut lap);
    say(&t0, "self-checks");
    let mut checks = Vec::new();
    selftests(&var.days, &mut checks);
    let mut census = load_census(meta, &store, &var, &mut checks)?;
    let length = args.length.unwrap_or_else(|| cfg.lengths.iter().copied().max().unwrap_or(1260));
    let length = if args.short { args.length.unwrap_or(252) } else { length };
    let (chain, full) = select_periods(&census, &var, length, args.short.then_some(args.short_from), &mut checks)?;
    let period_ids: Vec<usize> = match &args.periods {
        Some(p) => p.clone(),
        None => chain.iter().copied().chain(full).collect(),
    };
    for &id in &period_ids {
        ensure!(id < census.windows.len(), "census {} has no correlation period {id}", census.meta.name);
        fill_universe(&mut census, id, &store, &col_of)?;
    }
    replicate(&census, &var, &col_of, &chain, full, &mut checks)?;
    mark("self-checks", &mut lap);
    if args.selftest {
        let gpu = Gpu::new()?;
        let (n, k) = (300usize, 700usize);
        let mut rng = Rng(3);
        let x: Vec<f32> = (0..n * k).map(|_| rng.normal() as f32).collect();
        let y: Vec<f32> = (0..n * k).map(|_| rng.normal() as f32).collect();
        let (dx, dy) = (gpu.upload(&x, Precision::Fp32)?, gpu.upload(&y, Precision::Fp32)?);
        let mut dc = gpu.alloc_product(Precision::Fp32, n)?;
        gpu.xprod(Precision::Fp32, None, &dx, &dy, &mut dc, n, k)?;
        let DevProduct::F32(c) = &dc else { bail!("unexpected product type") };
        let c = gpu.stream.clone_dtoh(c)?;
        let mut worst = 0f64;
        for i in 0..n {
            for j in 0..n {
                let d: f64 = (0..k).map(|q| x[i * k + q] as f64 * y[j * k + q] as f64).sum();
                worst = worst.max((d - c[i + j * n] as f64).abs() / (k as f64).sqrt());
            }
        }
        check(&mut checks, worst < 1e-4, format!("GPU FP32 X^T Y ({} on {}) matches f64 sums: largest error {worst:.1e} sd units", n, gpu.name()?));
        let bad = checks.iter().filter(|c| !c.ok).count();
        println!("{} checks, {bad} failed", checks.len());
        return if bad == 0 { Ok(()) } else { bail!("{bad} self-checks failed") };
    }
    ensure!(checks.iter().all(|c| c.ok), "self-checks failed; not running");

    // External series.
    say(&t0, "reading the external catalogs");
    let mut warn = Vec::new();
    let (mut series, catalogs) = if args.synthetic_only { (Vec::new(), Vec::new()) } else { read_catalogs(&args.external, &args.catalogs, &mut warn)? };
    // Short-period mode: the prediction-market series in the headline; matched baselines (Google
    // Trends daily, crypto, a seeded random sample of the other series) as their own categories,
    // reported apart like the noise.
    let mut baseline_cats: Vec<&str> = Vec::new();
    if args.short {
        let (mut keep, mut others) = (Vec::new(), Vec::new());
        for mut s in series.drain(..) {
            if s.meta.catalog == "data-predictions" || s.meta.category == "prediction_market" || s.meta.category == SINGLE {
                keep.push(s);
            } else if s.meta.catalog == "data-trends" && s.meta.frequency == "daily" {
                s.meta.subcategory = format!("{}/{}", s.meta.category, s.meta.subcategory);
                s.meta.category = TRENDS_DAILY.into();
                keep.push(s);
            } else if s.meta.category == "crypto" {
                keep.push(s);
            } else if !is_annual(&s.meta, &s.obs) && s.meta.frequency != "quarterly" {
                others.push(s);
            }
        }
        others.sort_by(|a, b| a.meta.id.cmp(&b.meta.id));
        let mut rng = Rng(2021);
        for i in (1..others.len()).rev() {
            let j = (rng.u64() % (i as u64 + 1)) as usize;
            others.swap(i, j);
        }
        for mut s in others.into_iter().take(args.sample) {
            s.meta.subcategory = format!("{}/{}", s.meta.category, s.meta.subcategory);
            s.meta.category = SAMPLE.into();
            keep.push(s);
        }
        series = keep;
        baseline_cats = vec!["crypto", TRENDS_DAILY, SAMPLE];
    }
    let n_catalog = series.len();
    if !args.include_annual {
        for s in series.iter_mut().filter(|s| s.meta.category != SINGLE && is_annual(&s.meta, &s.obs)) {
            s.meta.subcategory = format!("{}/{}", s.meta.category, s.meta.subcategory);
            s.meta.category = ANNUAL.into();
        }
    }
    let (syn_series, syn_specs) = synthetic(&var, &store, &col_of, args.noise)?;
    series.extend(syn_series);
    let mut cats: Vec<String> = series
        .iter()
        .map(|s| s.meta.category.clone())
        .filter(|c| c != SYN && c != SINGLE && c != NOISE && c != ANNUAL && !baseline_cats.contains(&c.as_str()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let excl0 = cats.len();
    for special in baseline_cats.iter().copied().chain([NOISE, ANNUAL, SINGLE]) {
        if series.iter().any(|s| s.meta.category == special) {
            cats.push(special.into());
        }
    }
    let noise_cat = cats.iter().position(|c| c == NOISE);
    cats.push(SYN.into());
    let syn = cats.len() - 1;
    let periods: Vec<&Window> = period_ids.iter().map(|&p| &census.windows[p]).collect();
    let vars: Vec<ExtVar> = series.par_iter().map(|s| build_ext(&var.days, s)).collect();
    let ext: Vec<ExtInfo> = series
        .into_iter()
        .zip(vars)
        .map(|(s, v)| {
            let per = periods
                .iter()
                .map(|p| {
                    let len = p.rows.len();
                    let have = v.v[p.rows.clone()].iter().filter(|x| x.is_finite()).count();
                    let active = v.raw[p.rows.clone()].iter().filter(|x| x.is_finite() && **x != 0.0).count();
                    (have as f64 / len as f64, active, have as f64 >= COVERAGE * len as f64 && active >= MIN_ACTIVE)
                })
                .collect();
            let cat = cats.iter().position(|c| *c == s.meta.category).unwrap();
            ExtInfo { meta: s.meta, var: v, cat, per }
        })
        .collect();
    let exts_in: Vec<Vec<usize>> = (0..periods.len()).map(|pi| (0..ext.len()).filter(|&x| ext[x].per[pi].2).collect()).collect();
    for (pi, p) in periods.iter().enumerate() {
        say(&t0, &format!("period {} ({}..{}): {} external series join, {} stocks", p.id, date_string(p.first), date_string(p.last), exts_in[pi].len(), p.stocks.len()));
    }
    mark("external series", &mut lap);

    // GPU.
    let pairs: Vec<(usize, usize)> = args.pairs.clone().unwrap_or_else(|| cfg.horizon_pairs());
    let cx = Ctx { var: &var, ext: &ext, ncat: cats.len(), excl0, census: &census, pairs: pairs.clone(), min_obs: cfg.min_obs, q: cfg.fdr_q };
    let label = |id: usize| -> String {
        let w = &census.windows[id];
        format!("{id} ({}-{})", c2_data::civil_from_days(w.first).0, c2_data::civil_from_days(w.last).0)
    };
    let gpu = Gpu::new()?;
    let (free, total) = gpu.memory()?;
    say(&t0, &format!("GPU {} ({} MB free of {} MB)", gpu.name()?, free >> 20, total >> 20));
    let mut outs = Vec::new();
    for (pi, p) in periods.iter().enumerate() {
        let o = run_period(&gpu, &cx, p, &exts_in[pi], &t0)?;
        say(&t0, &format!("period {}: {} products in {:.0}s (GPU {:.0}s, scan {:.0}s), spot check |dr| {:.1e} over {}", p.id, o.products, o.secs, o.gpu_secs, o.scan_secs, o.max_dr, o.checked));
        outs.push(o);
    }
    drop(gpu);
    mark("GPU correlations", &mut lap);

    // Exact r anywhere: global prefix sums of every series used.
    let mut members: HashMap<usize, HashSet<u32>> = HashMap::new();
    let mut stock_cols: BTreeSet<usize> = BTreeSet::new();
    for (pi, p) in periods.iter().enumerate() {
        let mut set: HashSet<u32> = exts_in[pi].iter().map(|&x| EXT_BASE + x as u32).collect();
        for &j in &p.stocks {
            set.insert(var.ids[j] as u32);
            stock_cols.insert(j);
        }
        members.insert(p.id, set);
    }
    let stock_pre: HashMap<usize, Vec<f64>> = stock_cols.par_iter().map(|&j| (var.ids[j], prefix_of(var.col(&var.v, j)))).collect();
    let ext_pre: Vec<Vec<f64>> = ext.par_iter().map(|x| prefix_of(&x.var.v)).collect();
    let d = Data { var: &var, ext: &ext, col_of: &col_of, ext_pre, stock_pre, members, windows: &census.windows, store: &store, excl0 };
    let lo = var.days.partition_point(|&x| x < cfg.start);

    // Out-of-sample: each period of the chain against the next one.
    say(&t0, "out-of-sample check");
    let by_period: HashMap<usize, &PeriodOut> = outs.iter().map(|o| (o.id, o)).collect();
    let transitions: Vec<(usize, usize)> = chain.windows(2).map(|w| (w[0], w[1])).filter(|(a, b)| by_period.contains_key(a) && by_period.contains_key(b)).collect();
    // (from, to, dir, predictive, rule, copy) -> tally
    let mut oos: HashMap<(usize, usize, usize, usize, usize, usize), Oos> = HashMap::new();
    let mut oos_cat: HashMap<(usize, bool, usize), Oos> = HashMap::new(); // (category, null?, dir) at |z| >= 4, predictive
    let mut survivors: Vec<Survivor> = Vec::new();
    let mut null_surv = [0usize; NULLS];
    let mut noise_surv = 0usize;
    // Per direction: (real, null copies summed, noise externals) survivors.
    let mut surv_dir = [[0usize; 3]; 3];
    // ext->stock predictive at |z| >= 4, by (A is a US-listed ETF, picked from a null copy).
    let mut oos_etf: HashMap<(bool, bool), Oos> = HashMap::new();
    let is_etf = |g: u32| g >= EXT_BASE && ext[(g - EXT_BASE) as usize].meta.id.starts_with("etf_");
    for &(from, to) in &transitions {
        let src = by_period[&from];
        for copy in 0..=NULLS {
            for dir in 0..3 {
                for pred in 0..3 {
                    if dir == 0 && pred == 2 {
                        continue;
                    }
                    // Every non-synthetic entry, strongest first; headline ones exclude single-contract series too.
                    let mut cand: Vec<(f64, &Entry)> = src
                        .entries
                        .iter()
                        .filter(|e| e.copy as usize == copy && e.dir as usize == dir && !d.is_syn(e.a) && !d.is_syn(e.b) && d.offset_class(e) == pred)
                        .map(|e| (e.z().abs(), e))
                        .collect();
                    cand.sort_by(|a, b| b.0.total_cmp(&a.0));
                    let z4: Vec<&Entry> = cand.iter().filter(|c| c.0 >= 4.0).map(|c| c.1).collect();
                    let next4 = next_r(&d, &z4, to);
                    let (mut o4, mut ot, mut n4, mut nt) = (Oos::default(), Oos::default(), Oos::default(), Oos::default());
                    for (en, nx) in z4.iter().zip(&next4) {
                        let cat = d.entry_cat(en);
                        let head = cat < excl0;
                        if head {
                            o4.tally(en, *nx);
                            if dir == 0 && pred == 1 {
                                oos_etf.entry((is_etf(en.a), copy > 0)).or_default().tally(en, *nx);
                            }
                        }
                        if copy == 0 && Some(cat) == noise_cat {
                            n4.tally(en, *nx);
                            if pred == 1 && nx.is_some_and(|(rq, zq)| rq * en.r as f64 > 0.0 && zq.abs() >= 3.0) {
                                noise_surv += 1;
                                surv_dir[dir][2] += 1;
                            }
                        }
                        if pred == 1 {
                            let g = if dir == 2 { en.b } else { en.a };
                            oos_cat.entry((d.ext[(g - EXT_BASE) as usize].cat, copy > 0, dir)).or_default().tally(en, *nx);
                            if let (true, Some((rq, zq))) = (head, *nx)
                                && rq * en.r as f64 > 0.0
                                && zq.abs() >= 3.0
                            {
                                if copy == 0 {
                                    survivors.push(Survivor { from, to, e: **en, zp: en.z(), rq, zq });
                                    surv_dir[dir][0] += 1;
                                } else {
                                    null_surv[copy - 1] += 1;
                                    surv_dir[dir][1] += 1;
                                }
                            }
                        }
                    }
                    let top: Vec<&Entry> = cand.iter().map(|c| c.1).filter(|e| d.entry_cat(e) < excl0).take(1000).collect();
                    for (en, nx) in top.iter().zip(next_r(&d, &top, to)) {
                        ot.tally(en, nx);
                    }
                    oos.insert((from, to, dir, pred, 0, copy), o4);
                    oos.insert((from, to, dir, pred, 1, copy), ot);
                    if copy == 0 && noise_cat.is_some() {
                        let topn: Vec<&Entry> = cand.iter().map(|c| c.1).filter(|e| Some(d.entry_cat(e)) == noise_cat).take(1000).collect();
                        for (en, nx) in topn.iter().zip(next_r(&d, &topn, to)) {
                            nt.tally(en, nx);
                        }
                        oos.insert((from, to, dir, pred, 0, NOISE_SRC), n4);
                        oos.insert((from, to, dir, pred, 1, NOISE_SRC), nt);
                    }
                }
            }
        }
    }
    mark("out-of-sample", &mut lap);

    // Results directory and summary.
    let out_dir = args.external.join("results").join(&args.run);
    std::fs::create_dir_all(&out_dir)?;
    say(&t0, &format!("writing {}", out_dir.display()));
    let census_base = census_baseline(&census, &period_ids, &pairs)?;
    let mut sum = String::new();
    let all_rows: Vec<&MRow> = outs.iter().flat_map(|o| o.rows.iter()).collect();
    let all_mats: Vec<&FRow> = outs.iter().flat_map(|o| o.mats.iter()).collect();
    let catname = |g: usize| cats[g].as_str();

    writeln!(sum, "External series vs US stocks: census-faithful lead-lag test")?;
    writeln!(sum, "run {} ({}), census run {}, data {}", args.run, date_string((std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs() / 86_400) as i32), args.census, args.external.display())?;
    writeln!(sum, "command: cargo run --release -p c2-census --example external_test -- {}", args.command)?;
    writeln!(sum)?;
    writeln!(sum, "How to read this: A moves first over its move length, B follows over its move length, and the offset is the")?;
    writeln!(sum, "number of trading days from the end of A's move to the end of B's. Offset 0 = the same day (no prediction);")?;
    writeln!(sum, "offset >= 1 = predictive. 'chance' is the mean of the 3 sign-flip null copies (the census's null); a real/chance")?;
    writeln!(sum, "ratio near 1 means no more strong correlations than noise produces.")?;
    writeln!(sum)?;
    writeln!(sum, "== Method (rules chosen) ==")?;
    for line in [
        "Stocks: the production census variable and the census run's own stock universe for each correlation period.",
        "Timing: an observation dated d with known_after_days k counts from the first trading day on or after d + k",
        "  (calendar days); the weekly Trends-style series dated by week start with k = 7 land on the first trading day",
        "  after the week ends (self-checked).",
        "logdiff/diff: the move on trading day t is the change in the latest known level since t-1, so release days",
        "  carry the whole move and other days are 0. A change across a gap longer than the series' normal spacing",
        "  (the 90th percentile of its previous 250 gaps plus 2 days; exactly 1 day for series observed every day, so a",
        "  prediction market's roll gap or an outage is never bridged) is not counted as a move.",
        "anomaly/anomaly_log1p: value (or ln(1 + value)) minus the mean over the 15 days centred on the same date in each",
        "  of the 10 previous years (at least 3 years); anomalies landing on the same trading day are averaged.",
        "level: value (averaged over the days landing on a trading day) minus its mean over the previous 252 trading days.",
        "Standardization, exactly as the census does for stocks: minus the trailing 252-day mean, divided by the trailing",
        "  252-day sd (both through t-1, at least 126 values), clipped at +-5. No value when the trailing year holds no",
        "  non-zero move at all, so a sparse 0/1 indicator can never divide by a zero or round-off sd.",
        "Live: a series has values while its latest observation is recent for its frequency (10 trading days for daily,",
        "  15 weekly, 50 monthly, 140 quarterly, 300 annual); elsewhere it is missing, which counts as 0 in sums.",
        "Joining a period: >= 90% of the period's trading days have a value AND >= 10 of them carry a non-zero move (so a",
        "  single event can't dominate a correlation through a tiny trailing sd; the +-5 clip bounds each day too).",
        "Kept outside every headline number and reported as their own categories: single_contract (notes marked",
        "  FINITE-LIFE: one prediction-market contract with a finite life, not a rolling series), annual (one release a",
        "  year, often all on one date: a calendar pulse that standardization pushes to +-5; --include-annual counts them),",
        "  the noise calibration series and the synthetic controls.",
        "At offset 0 with equal move lengths, ext->stock and stock->ext hold the same correlations transposed.",
        "Grid, sums / sqrt(move length), >= 30 observations, the 3 sign-flip null copies (census seeds), the false-",
        "  discovery rule and its placebo: copies of the census's code; checked against stored census records below.",
    ] {
        writeln!(sum, "{line}")?;
    }
    writeln!(sum)?;

    // Catalogs and series.
    writeln!(sum, "== What was tested ==")?;
    for (c, listed, used) in &catalogs {
        writeln!(sum, "catalog {c}: {listed} series listed, {used} loaded")?;
    }
    writeln!(sum, "plus {} synthetic control series injected in memory (excluded from every headline number)", ext.len() - n_catalog)?;
    for w in &warn {
        writeln!(sum, "  note: {w}")?;
    }
    writeln!(sum)?;
    write!(sum, "{:<16} {:>6}", "category", "series")?;
    for p in &periods {
        write!(sum, " {:>9}", format!("in {}", p.id))?;
    }
    writeln!(sum, "   transforms")?;
    for (g, c) in cats.iter().enumerate() {
        let xs: Vec<&ExtInfo> = ext.iter().filter(|x| x.cat == g).collect();
        if xs.is_empty() {
            continue;
        }
        write!(sum, "{:<16} {:>6}", c, xs.len())?;
        for pi in 0..periods.len() {
            write!(sum, " {:>9}", xs.iter().filter(|x| x.per[pi].2).count())?;
        }
        let mut tf: BTreeSet<String> = BTreeSet::new();
        for x in &xs {
            tf.insert(format!("{}/{}", x.meta.transform.name(), if x.meta.frequency.is_empty() { "?" } else { &x.meta.frequency }));
        }
        writeln!(sum, "   {}", tf.into_iter().collect::<Vec<_>>().join(", "))?;
    }
    writeln!(sum)?;
    // Considered but not tested: every catalog series that joined no period, with the reason.
    let catalog_ext: Vec<&ExtInfo> = ext.iter().filter(|x| x.meta.category != SYN && x.meta.category != NOISE).collect();
    let untested: Vec<(&ExtInfo, String)> = catalog_ext.iter().filter_map(|x| not_tested(x).map(|r| (*x, r))).collect();
    writeln!(sum, "Considered but not tested: {} of {} catalog series joined no correlation period. A series joins a period only", untested.len(), catalog_ext.len())?;
    writeln!(sum, "with values on >= 90% of its trading days and >= {MIN_ACTIVE} non-zero moves there; annual data (about 5 changes per")?;
    writeln!(sum, "{length}-day period, about 20 in full history) mostly can't pass, so it is reported here rather than dropped.")?;
    let full_idx: Vec<usize> = periods.iter().enumerate().filter(|(_, p)| p.length == 0).map(|(i, _)| i).collect();
    let full_only = |x: &ExtInfo| x.per.iter().enumerate().all(|(i, p)| p.2 == full_idx.contains(&i)) && !full_idx.is_empty();
    writeln!(sum, "{:<16} {:>10} {:>10} {:>12} {:>12} {:>12}   annual series (joined / considered)", "category", "considered", "joined", "full hist only", "too sparse", "too short")?;
    for c in &cats {
        let xs: Vec<&&ExtInfo> = catalog_ext.iter().filter(|x| &x.meta.category == c).collect();
        if xs.is_empty() {
            continue;
        }
        let sparse = untested.iter().filter(|(x, r)| &x.meta.category == c && r.starts_with("too sparse")).count();
        let short = untested.iter().filter(|(x, r)| &x.meta.category == c && r.starts_with("too short")).count();
        let annual: Vec<&&&ExtInfo> = xs.iter().filter(|x| matches!(x.meta.frequency.as_str(), "annual" | "yearly")).collect();
        let annual_joined = annual.iter().filter(|x| x.per.iter().any(|p| p.2)).count();
        writeln!(sum, "{:<16} {:>10} {:>10} {:>12} {:>12} {:>12}   {}", c, xs.len(), xs.len() - sparse - short, xs.iter().filter(|x| full_only(x)).count(), sparse, short,
            if annual.is_empty() { "-".to_string() } else { format!("{annual_joined} / {}", annual.len()) })?;
    }
    for (label, key) in [("Too sparse to test", "too sparse"), ("Too short or gappy", "too short")] {
        let ids: Vec<String> = untested.iter().filter(|(_, r)| r.starts_with(key)).map(|(x, _)| format!("{} ({})", x.meta.id, if x.meta.frequency.is_empty() { "?" } else { &x.meta.frequency })).collect();
        if !ids.is_empty() {
            let shown: Vec<String> = ids.iter().take(60).cloned().collect();
            writeln!(sum, "{label} ({}): {}{}", ids.len(), shown.join(", "), if ids.len() > 60 { format!(", ... and {} more (externals.csv, column status)", ids.len() - 60) } else { String::new() })?;
        }
    }
    writeln!(sum)?;
    writeln!(sum, "Correlation periods (the census's own, with its stock universes):")?;
    for (pi, p) in periods.iter().enumerate() {
        let o = &outs[pi];
        writeln!(sum, "  {} {}..{} ({} trading days): {} external series + {} stocks", p.id, date_string(p.first), date_string(p.last), p.rows.len(), o.e, o.s)?;
    }
    let (mut tested, mut nulls_n) = (0u64, 0u64);
    for r in all_rows.iter().filter(|r| r.cat < excl0) {
        tested += r.pairs;
        nulls_n += r.pairs * NULLS as u64;
    }
    let matrices = all_mats.len();
    let lag_count: usize = pairs.iter().map(|&(_, h2)| grid::lags(h2).len()).sum();
    writeln!(sum, "Grid: {} move-length pairs (A and B each over 1, 3, 5, 10, 21, 63 days), {} offsets in all (0 plus the census's", pairs.len(), lag_count)?;
    writeln!(sum, "predictive offsets for each B move length), only where >= {} non-overlapping observations fit the period", cfg.min_obs)?;
    writeln!(sum, "(63-day moves fit only in full history). {} block matrices; {} real correlations of the catalog series tested,", th(matrices as f64), th(tested as f64))?;
    writeln!(sum, "{} null-copy correlations as chance. Stock->stock pairs are not recomputed (the census has them).", th(nulls_n as f64))?;
    let products: usize = outs.iter().map(|o| o.products).sum();
    let (gs, ss): (f64, f64) = outs.iter().fold((0.0, 0.0), |a, o| (a.0 + o.gpu_secs, a.1 + o.scan_secs));
    writeln!(sum, "GPU: {products} FP32 products ({gs:.0}s transfer + multiply, {ss:.0}s scanning on the CPU).")?;
    writeln!(sum)?;

    // Self-checks.
    writeln!(sum, "== Self-checks ==")?;
    for c in &checks {
        writeln!(sum, "[{}] {}", if c.ok { "ok" } else { "FAIL" }, c.what)?;
    }
    let (dr, nck) = outs.iter().fold((0f64, 0usize), |a, o| (a.0.max(o.max_dr), a.1 + o.checked));
    writeln!(sum, "[{}] GPU FP32 r vs exact f64 recomputation: {nck} spot checks (real and null), largest |r difference| {dr:.1e}", if dr < 1e-4 { "ok" } else { "FAIL" })?;
    writeln!(sum)?;

    // Headline real vs chance.
    writeln!(sum, "== Real vs chance (all periods and move lengths pooled) ==")?;
    writeln!(sum, "Headline = the catalogs' series except single-contract{} series. 'noise externals' = {} pure-noise", if args.include_annual { "" } else { " and annual" }, ext.iter().filter(|x| Some(x.cat) == noise_cat).count())?;
    writeln!(sum, "series of the same timing and transform kinds, run through the same pipeline against the same stocks: the")?;
    writeln!(sum, "ratio that 'no relationship at all' produces. The census's null is not exactly calibrated at long move lengths")?;
    writeln!(sum, "(see the move-length table), so compare real ratios with the noise ratios, not with 1.")?;
    writeln!(sum, "{:<12} {:<11} {:>14} {:>14} {:>6}   {:>12} {:>12} {:>6}   {:>10} {:>10} {:>6}", "direction", "offset", "|z|>=3 real", "chance", "ratio", "|z|>=4 real", "chance", "ratio", "|z|>=5", "chance", "ratio")?;
    let agg_in = |cat_ok: &dyn Fn(usize) -> bool, f: &dyn Fn(&MRow) -> bool| -> ([f64; 3], [f64; 3], u64) {
        let (mut r, mut c, mut p) = ([0f64; 3], [0f64; 3], 0u64);
        for row in all_rows.iter().filter(|x| cat_ok(x.cat) && f(x)) {
            for t in 0..3 {
                r[t] += row.real[t] as f64;
                c[t] += row.null[t] as f64 / NULLS as f64;
            }
            p += row.pairs;
        }
        (r, c, p)
    };
    let agg = |f: &dyn Fn(&MRow) -> bool| agg_in(&|g| g < excl0, f);
    let aggn = |f: &dyn Fn(&MRow) -> bool| agg_in(&|g| Some(g) == noise_cat, f);
    for dir in 0..3 {
        for pred in 0..2 {
            let (r, c, _) = agg(&|x| x.dir == dir && usize::from(x.lag > 0) == pred);
            writeln!(sum, "{:<12} {:<11} {:>14} {:>14} {:>6}   {:>12} {:>12} {:>6}   {:>10} {:>10} {:>6}", DIRS[dir], OFFSETS[pred], th(r[0]), th(c[0]), ratio(r[0], c[0]), th(r[1]), th(c[1]), ratio(r[1], c[1]), th(r[2]), th(c[2]), ratio(r[2], c[2]))?;
        }
    }
    writeln!(sum, "For comparison, the census's own stock->stock matrices for the same periods and move lengths:")?;
    for pred in 0..2 {
        let (mut r, mut c) = ([0f64; 3], [0f64; 3]);
        for ((_, p), b) in &census_base {
            if usize::from(*p) == pred {
                for t in 0..3 {
                    r[t] += b.real[t];
                    c[t] += b.null[t];
                }
            }
        }
        writeln!(sum, "{:<12} {:<11} {:>14} {:>14} {:>6}   {:>12} {:>12} {:>6}   {:>10} {:>10} {:>6}", "stock->stock", OFFSETS[pred], th(r[0]), th(c[0]), ratio(r[0], c[0]), th(r[1]), th(c[1]), ratio(r[1], c[1]), th(r[2]), th(c[2]), ratio(r[2], c[2]))?;
    }
    // Split by offset class from the stored entries (|z| >= 4).
    let mut tab: HashMap<(bool, usize, usize), [f64; 2]> = HashMap::new();
    for o in &outs {
        for e in o.entries.iter().filter(|e| e.z().abs() >= 4.0 && !d.is_syn(e.a) && !d.is_syn(e.b)) {
            let cat = d.entry_cat(e);
            let noise = Some(cat) == noise_cat;
            if cat >= excl0 && !noise {
                continue;
            }
            let t = tab.entry((noise, e.dir as usize, d.offset_class(e))).or_default();
            if e.copy == 0 {
                t[0] += 1.0;
            } else {
                t[1] += 1.0 / NULLS as f64;
            }
        }
    }
    let tab_line = |noise: bool, dir: usize, k: usize| -> String {
        let v = tab.get(&(noise, dir, k)).copied().unwrap_or_default();
        format!("{}/{} = {}", th(v[0]), th(v[1]), ratio(v[0], v[1]))
    };
    let mut split = String::new();
    writeln!(split, "Offsets split by timing (|z| >= 4, from the stored entries). 'within delay': a stock->ext or ext->ext offset shorter")?;
    writeln!(split, "than B's move length plus B's known_after_days and measurement span (5 rows weekly, 21 monthly, ...), so B's")?;
    writeln!(split, "move can describe the same physical days as A's: timing, not a lead. 'predictive' is only what lies beyond that.")?;
    writeln!(split, "{:<12} {:<13} {:>26} {:>26}", "direction", "offset class", "real/chance", "noise externals")?;
    for dir in 0..3 {
        for k in 0..3 {
            if dir == 0 && k == 2 {
                continue;
            }
            writeln!(split, "{:<12} {:<13} {:>26} {:>26}", DIRS[dir], OFFSETS[k], tab_line(false, dir, k), tab_line(true, dir, k))?;
        }
    }
    if noise_cat.is_some() {
        writeln!(sum, "Noise externals (the baseline):")?;
        for dir in 0..3 {
            for pred in 0..2 {
                let (r, c, _) = aggn(&|x| x.dir == dir && usize::from(x.lag > 0) == pred);
                writeln!(sum, "{:<12} {:<11} {:>14} {:>14} {:>6}   {:>12} {:>12} {:>6}   {:>10} {:>10} {:>6}", DIRS[dir], OFFSETS[pred], th(r[0]), th(c[0]), ratio(r[0], c[0]), th(r[1]), th(c[1]), ratio(r[1], c[1]), th(r[2]), th(c[2]), ratio(r[2], c[2]))?;
            }
        }
    }
    writeln!(sum)?;
    sum.push_str(&split);
    writeln!(sum)?;
    writeln!(sum, "By correlation period, |z| >= 4:")?;
    writeln!(sum, "{:<8} {:<12} {:>24} {:>24} {:>17}   stock->stock predictive (census)", "period", "direction", "same day real/chance", "predictive real/chance", "noise predictive")?;
    for p in &periods {
        for dir in 0..3 {
            let (r0, c0, _) = agg(&|x| x.period == p.id && x.dir == dir && x.lag == 0);
            let (r1, c1, _) = agg(&|x| x.period == p.id && x.dir == dir && x.lag > 0);
            let (rn, cn, _) = aggn(&|x| x.period == p.id && x.dir == dir && x.lag > 0);
            let base = census_base.get(&(p.id, true)).map(|b| format!("{}/{} = {}", th(b.real[1]), th(b.null[1]), ratio(b.real[1], b.null[1]))).unwrap_or_default();
            writeln!(sum, "{:<8} {:<12} {:>24} {:>24} {:>17}   {}", p.id, DIRS[dir], format!("{}/{} = {}", th(r0[1]), th(c0[1]), ratio(r0[1], c0[1])), format!("{}/{} = {}", th(r1[1]), th(c1[1]), ratio(r1[1], c1[1])), ratio(rn[1], cn[1]), if dir == 0 { base } else { String::new() })?;
        }
    }
    writeln!(sum)?;
    writeln!(sum, "By category, predictive offsets only, |z| >= 4 (category = the external series' category; for ext->ext, A's):")?;
    writeln!(sum, "{:<16} {:>26} {:>26} {:>26}", "category", "ext->stock real/chance", "ext->ext real/chance", "stock->ext real/chance")?;
    for (g, c) in cats.iter().enumerate() {
        let mut line = format!("{c:<16}");
        let mut any = false;
        for dir in 0..3 {
            let rows: Vec<&&MRow> = all_rows.iter().filter(|x| x.cat == g && x.dir == dir && x.lag > 0).collect();
            let (r, ch) = rows.iter().fold((0f64, 0f64), |a, x| (a.0 + x.real[1] as f64, a.1 + x.null[1] as f64 / NULLS as f64));
            any |= !rows.is_empty();
            write!(line, " {:>26}", if rows.is_empty() { "-".into() } else { format!("{}/{} = {}", th(r), th(ch), ratio(r, ch)) })?;
        }
        if any {
            writeln!(sum, "{line}")?;
        }
    }
    writeln!(sum)?;
    writeln!(sum, "By category, same day (offset 0), |z| >= 4:")?;
    for (g, c) in cats.iter().enumerate() {
        let mut line = format!("{c:<16}");
        let mut any = false;
        for dir in 0..3 {
            let rows: Vec<&&MRow> = all_rows.iter().filter(|x| x.cat == g && x.dir == dir && x.lag == 0).collect();
            let (r, ch) = rows.iter().fold((0f64, 0f64), |a, x| (a.0 + x.real[1] as f64, a.1 + x.null[1] as f64 / NULLS as f64));
            any |= !rows.is_empty();
            write!(line, " {:>26}", if rows.is_empty() { "-".into() } else { format!("{}/{} = {}", th(r), th(ch), ratio(r, ch)) })?;
        }
        if any {
            writeln!(sum, "{line}")?;
        }
    }
    writeln!(sum)?;
    if args.short {
        writeln!(sum, "== Like-for-like comparison (short periods, ext->stock unless noted) ==")?;
        writeln!(sum, "Each group run through the same {length}-day periods, move lengths, offsets and null. 'same sign next' = picked at")?;
        writeln!(sum, "|z| >= 4 in one period, same sign in the next (null = picked from the null copies, the 50% baseline).")?;
        writeln!(sum, "{:<18} {:>7} {:>16} {:>20} {:>20} {:>17} {:>14} {:>26} {:>15}", "group", "series", "joined by period", "predictive |z|>=3", "predictive |z|>=4", "same day |z|>=4", "FDR / placebo", "same sign next: real/null", "stock->ext pred")?;
        for (g, c) in cats.iter().enumerate() {
            if g == syn {
                continue;
            }
            let members: Vec<&ExtInfo> = ext.iter().filter(|x| x.cat == g).collect();
            let joined: Vec<String> = (0..periods.len()).map(|pi| members.iter().filter(|x| x.per[pi].2).count().to_string()).collect();
            let rows_of = |dir: usize, pred: bool| -> Vec<&&MRow> { all_rows.iter().filter(|x| x.cat == g && x.dir == dir && (x.lag > 0) == pred).collect() };
            let tot = |rs: &[&&MRow], t: usize| -> (f64, f64) { rs.iter().fold((0.0, 0.0), |a, x| (a.0 + x.real[t] as f64, a.1 + x.null[t] as f64 / NULLS as f64)) };
            let (p, sd, sp) = (rows_of(0, true), rows_of(0, false), rows_of(2, true));
            let ((r3, c3), (r4, c4), (s4, cs4), (q4, cq4)) = (tot(&p, 0), tot(&p, 1), tot(&sd, 1), tot(&sp, 1));
            let fired = p.iter().filter(|x| x.cfdr.is_some()).count();
            let placebo = p.iter().filter(|x| x.cplacebo.is_some()).count();
            let (o, on) = (oos_cat.get(&(g, false, 0)).cloned().unwrap_or_default(), oos_cat.get(&(g, true, 0)).cloned().unwrap_or_default());
            writeln!(sum, "{:<18} {:>7} {:>16} {:>20} {:>20} {:>17} {:>14} {:>26} {:>15}", c, members.len(), joined.join("/"),
                format!("{}/{} = {}", th(r3), th(c3), ratio(r3, c3)), format!("{}/{} = {}", th(r4), th(c4), ratio(r4, c4)),
                ratio(s4, cs4), format!("{fired}/{placebo} of {}", p.len()),
                format!("{} ({}) / {} ({})", pct(o.same as f64, o.n as f64), th(o.n as f64), pct(on.same as f64, on.n as f64), th(on.n as f64)),
                ratio(q4, cq4))?;
        }
        writeln!(sum, "Prediction-market series by period (joined = >= 90% coverage and >= {MIN_ACTIVE} non-zero moves):")?;
        for x in ext.iter().filter(|x| x.meta.category == "prediction_market") {
            let j: Vec<String> = periods.iter().zip(&x.per).map(|(w, p)| format!("{}:{}", w.id, if p.2 { "yes".to_string() } else { format!("{:.0}%", 100.0 * p.0) })).collect();
            writeln!(sum, "  {:<28} {}", x.meta.id, j.join("  "))?;
        }
        let sc_joined = ext.iter().filter(|x| x.meta.category == SINGLE && x.per.iter().any(|p| p.2)).count();
        writeln!(sum, "Finite-life single contracts: {} loaded, {sc_joined} cover 90% of some {length}-day period (reported apart).", ext.iter().filter(|x| x.meta.category == SINGLE).count())?;
        writeln!(sum)?;
    }
    writeln!(sum, "By move-length pair (A/B), predictive offsets, |z| >= 4: real/chance = ratio [noise externals' ratio]")?;
    writeln!(sum, "{:<8} {:>32} {:>32} {:>32}", "moves", DIRS[0], DIRS[1], DIRS[2])?;
    for &(h1, h2) in &pairs {
        let mut line = format!("{:<8}", format!("{h1}/{h2}"));
        for dir in 0..3 {
            let (r, c, n) = agg(&|x| x.h1 == h1 && x.h2 == h2 && x.dir == dir && x.lag > 0);
            let (rn, cn, _) = aggn(&|x| x.h1 == h1 && x.h2 == h2 && x.dir == dir && x.lag > 0);
            write!(line, " {:>32}", if n == 0 { "-".into() } else { format!("{}/{} = {} [{}]", th(r[1]), th(c[1]), ratio(r[1], c[1]), ratio(rn[1], cn[1])) })?;
        }
        writeln!(sum, "{line}")?;
    }
    writeln!(sum)?;

    // False discovery vs placebo.
    writeln!(sum, "== False-discovery rule (q = 0.05) vs its placebo ==")?;
    writeln!(sum, "Run exactly as the census does, on each block matrix (one period, move-length pair, offset and direction;")?;
    writeln!(sum, "pairs = the block's pair count). The placebo runs the same rule with null copy 2 standing in for the real")?;
    writeln!(sum, "matrix: how often it fires with nothing there.")?;
    writeln!(sum, "{:<12} {:<11} {:>9} {:>14} {:>14} {:>16} {:>16}", "direction", "offset", "matrices", "rule fired", "placebo fired", "entries passing", "placebo entries")?;
    for dir in 0..3 {
        for pred in 0..2 {
            let ms: Vec<&&FRow> = all_mats.iter().filter(|m| m.dir == dir && usize::from(m.lag > 0) == pred).collect();
            let fired = ms.iter().filter(|m| m.fdr.is_some()).count();
            let pfired = ms.iter().filter(|m| m.placebo.is_some()).count();
            let n: u64 = ms.iter().map(|m| m.fdr_n).sum();
            let pn: u64 = ms.iter().map(|m| m.placebo_n).sum();
            writeln!(sum, "{:<12} {:<11} {:>9} {:>14} {:>14} {:>16} {:>16}", DIRS[dir], OFFSETS[pred], ms.len(), format!("{fired} ({})", pct(fired as f64, ms.len() as f64)), format!("{pfired} ({})", pct(pfired as f64, ms.len() as f64)), th(n as f64), th(pn as f64))?;
        }
    }
    if let Some(nc) = noise_cat {
        writeln!(sum, "The same rule on the noise externals' blocks alone (the baseline):")?;
        for dir in 0..3 {
            for pred in 0..2 {
                let rows: Vec<&&MRow> = all_rows.iter().filter(|x| x.cat == nc && x.dir == dir && usize::from(x.lag > 0) == pred).collect();
                let f = rows.iter().filter(|x| x.cfdr.is_some()).count();
                let pf = rows.iter().filter(|x| x.cplacebo.is_some()).count();
                let n: u64 = rows.iter().map(|x| x.cfdr_n).sum();
                let pn: u64 = rows.iter().map(|x| x.cplacebo_n).sum();
                writeln!(sum, "{:<12} {:<11} {:>9} {:>14} {:>14} {:>16} {:>16}", DIRS[dir], OFFSETS[pred], rows.len(), format!("{f} ({})", pct(f as f64, rows.len() as f64)), format!("{pf} ({})", pct(pf as f64, rows.len() as f64)), th(n as f64), th(pn as f64))?;
            }
        }
    }
    writeln!(sum, "Census stock->stock, same periods and move lengths, predictive: rule fired in {}, placebo in {}, of {} matrices.",
        census_base.iter().filter(|k| k.0.1).map(|k| k.1.fdr).sum::<usize>(), census_base.iter().filter(|k| k.0.1).map(|k| k.1.placebo).sum::<usize>(), census_base.iter().filter(|k| k.0.1).map(|k| k.1.matrices).sum::<usize>())?;
    writeln!(sum)?;
    writeln!(sum, "Run within each category alone (predictive offsets): matrices where the rule fired / placebo fired, entries passing / placebo entries")?;
    writeln!(sum, "{:<16} {:>30} {:>30} {:>30}", "category", DIRS[0], DIRS[1], DIRS[2])?;
    for (g, c) in cats.iter().enumerate() {
        let mut line = format!("{c:<16}");
        let mut any = false;
        for dir in 0..3 {
            let rows: Vec<&&MRow> = all_rows.iter().filter(|x| x.cat == g && x.dir == dir && x.lag > 0).collect();
            any |= !rows.is_empty();
            let f = rows.iter().filter(|x| x.cfdr.is_some()).count();
            let pf = rows.iter().filter(|x| x.cplacebo.is_some()).count();
            let n: u64 = rows.iter().map(|x| x.cfdr_n).sum();
            let pn: u64 = rows.iter().map(|x| x.cplacebo_n).sum();
            write!(line, " {:>30}", if rows.is_empty() { "-".into() } else { format!("{f}/{pf} of {}, {}/{}", rows.len(), th(n as f64), th(pn as f64)) })?;
        }
        if any {
            writeln!(sum, "{line}")?;
        }
    }
    writeln!(sum)?;

    // Out-of-sample.
    writeln!(sum, "== Out-of-sample: same sign in the next {length}-day period? ==")?;
    writeln!(sum, "Relationships (A, B, move lengths, offset) picked in one period are recomputed exactly in the next. Picked from")?;
    writeln!(sum, "the real data vs picked from null copy 1 (the baseline: its sign is noise by construction, so ~50%) vs picked")?;
    writeln!(sum, "from the real data of the pure-noise externals ('noise'). 'kept' = both series still in the next period's")?;
    writeln!(sum, "universe; 'rep' = same sign and |z| >= 2 / >= 3 in the next period.")?;
    for rule in 0..2 {
        writeln!(sum, "{}", if rule == 0 { "Picked: every |z| >= 4" } else { "Picked: the top 1,000 by |z| per period, direction and offset class" })?;
        writeln!(sum, "{:<12} {:<11} {:<7} {:>9} {:>11} {:>9} {:>9} {:>8} {:>8}", "direction", "offset", "picked", "kept", "same sign", "mean r", "rep z>=2", "rep z>=3", "dropped")?;
        for dir in 0..3 {
            for pred in 0..3 {
                if dir == 0 && pred == 2 {
                    continue;
                }
                for (label, copies) in [("real", vec![0usize]), ("null 1", vec![1]), ("null 2-3", vec![2, 3]), ("noise", vec![NOISE_SRC])] {
                    if label == "noise" && noise_cat.is_none() {
                        continue;
                    }
                    let mut t = Oos::default();
                    for &(f, to) in &transitions {
                        for &c in &copies {
                            if let Some(o) = oos.get(&(f, to, dir, pred, rule, c)) {
                                t.add(o);
                            }
                        }
                    }
                    writeln!(sum, "{:<12} {:<11} {:<7} {:>9} {:>11} {:>9} {:>9} {:>8} {:>8}", DIRS[dir], OFFSETS[pred], label, th(t.n as f64), pct(t.same as f64, t.n as f64),
                        if t.n > 0 { format!("{:+.4}", t.sum_sr / t.n as f64) } else { "-".into() }, pct(t.rep2 as f64, t.n as f64), pct(t.rep3 as f64, t.n as f64), th(t.dropped as f64))?;
                }
            }
        }
        writeln!(sum)?;
    }
    writeln!(sum, "Per period transition, predictive, every |z| >= 4: real same sign (n) vs null 1 (n) vs noise externals (n)")?;
    for &(f, to) in &transitions {
        let mut line = format!("  {f}->{to}:");
        for dir in 0..3 {
            let get = |c: usize| oos.get(&(f, to, dir, 1, 0, c)).cloned().unwrap_or_default();
            let (r, n, z) = (get(0), get(1), get(NOISE_SRC));
            write!(line, "  {} {} ({}) vs {} ({}) vs {} ({})", DIRS[dir], pct(r.same as f64, r.n as f64), th(r.n as f64), pct(n.same as f64, n.n as f64), th(n.n as f64), pct(z.same as f64, z.n as f64), th(z.n as f64))?;
        }
        writeln!(sum, "{line}")?;
    }
    writeln!(sum)?;
    writeln!(sum, "By category, predictive, every |z| >= 4, all transitions: real vs null 1-3 same-sign share")?;
    writeln!(sum, "{:<16} {:>28} {:>28} {:>28}", "category", DIRS[0], DIRS[1], DIRS[2])?;
    for (g, c) in cats.iter().enumerate() {
        if g == syn {
            continue;
        }
        let mut line = format!("{c:<16}");
        let mut any = false;
        for dir in 0..3 {
            let (r, n) = (oos_cat.get(&(g, false, dir)).cloned().unwrap_or_default(), oos_cat.get(&(g, true, dir)).cloned().unwrap_or_default());
            any |= r.n + n.n > 0;
            write!(line, " {:>28}", format!("{} ({}) vs {} ({})", pct(r.same as f64, r.n as f64), th(r.n as f64), pct(n.same as f64, n.n as f64), th(n.n as f64)))?;
        }
        if any {
            writeln!(sum, "{line}")?;
        }
    }
    writeln!(sum)?;

    // Controls.
    say(&t0, "controls");
    let mut ctrl = String::new();
    let controls_csv = controls(&d, &periods, &syn_specs, &outs, &transitions, &mut ctrl)?;
    std::fs::write(out_dir.join("controls.csv"), controls_csv)?;
    writeln!(sum, "== Controls ==")?;
    sum.push_str(&ctrl);
    writeln!(sum)?;

    // Survivors.
    say(&t0, "plain-terms counts");
    survivors.sort_by(|a, b| b.zp.abs().min(b.zq.abs()).total_cmp(&a.zp.abs().min(a.zq.abs())));
    let nsurv = survivors.len();
    let null_mean = null_surv.iter().sum::<usize>() as f64 / NULLS as f64;
    writeln!(sum, "== Strongest predictive relationships that survive the out-of-sample check ==")?;
    // Survivors per million predictive pairs tested in the periods they were picked from.
    let sources: HashSet<usize> = transitions.iter().map(|t| t.0).collect();
    let tested_in = |ok: &dyn Fn(usize) -> bool| -> f64 { all_rows.iter().filter(|x| ok(x.cat) && x.lag > 0 && sources.contains(&x.period)).map(|x| x.pairs as f64).sum() };
    let (pr, pn) = (tested_in(&|g| g < excl0), tested_in(&|g| Some(g) == noise_cat));
    writeln!(sum, "Survive = |z| >= 4 in one period and the same sign with |z| >= 3 in the next. Real: {} relationships; picked from", th(nsurv as f64))?;
    writeln!(sum, "the null copies instead, the same test passes {} on average ({:?} per copy); picked from the noise externals, {}.", th(null_mean), null_surv, th(noise_surv as f64))?;
    writeln!(sum, "Per million predictive pairs tested: real {:.2}, null copies {:.2}, noise externals {:.2}: the numbers to beat.",
        1e6 * nsurv as f64 / pr.max(1.0), 1e6 * null_mean / pr.max(1.0), 1e6 * noise_surv as f64 / pn.max(1.0))?;
    writeln!(sum, "By direction, per million predictive pairs tested (real / null copies, mean / noise externals):")?;
    for dir in 0..3 {
        let tested_dir = |ok: &dyn Fn(usize) -> bool| -> f64 { all_rows.iter().filter(|x| ok(x.cat) && x.dir == dir && x.lag > 0 && sources.contains(&x.period)).map(|x| x.pairs as f64).sum() };
        let (pr_d, pn_d) = (tested_dir(&|g| g < excl0), tested_dir(&|g| Some(g) == noise_cat));
        let v = surv_dir[dir];
        writeln!(sum, "  {:<11} {} survivors ({:.2} per million) / {:.1} ({:.2}) / {} ({:.2})", DIRS[dir], th(v[0] as f64), 1e6 * v[0] as f64 / pr_d.max(1.0),
            v[1] as f64 / NULLS as f64, 1e6 * v[1] as f64 / NULLS as f64 / pr_d.max(1.0), v[2], 1e6 * v[2] as f64 / pn_d.max(1.0))?;
    }
    let etf = |g: u32| g >= EXT_BASE && ext[(g - EXT_BASE) as usize].meta.id.starts_with("etf_");
    let es_ext = survivors.iter().filter(|s| s.e.dir == 0 && !etf(s.e.a)).count();
    writeln!(sum, "ext->stock survivors led by US-listed ETFs (market data from the price store, not outside information): {}; by all other external series: {es_ext}.",
        survivors.iter().filter(|s| s.e.dir == 0 && etf(s.e.a)).count())?;
    let mut surv_csv = format!("from,to,direction,a,b,h1,h2,offset,obs,r_from,z_from,r_to,z_to,{}\n", period_ids.iter().map(|p| format!("z_{p}")).collect::<Vec<_>>().join(","));
    let track_of = |e: &Entry| -> Vec<String> {
        period_ids.iter().map(|&w| d.stat(e.a, e.b, w, e.h1 as usize, e.h2 as usize, e.lag as usize).map(|s| format!("{:+.1}", s.1)).unwrap_or_else(|| "-".into())).collect()
    };
    let track_label = period_ids.iter().map(|&p| label(p)).collect::<Vec<_>>().join(" / ");
    for sv in &survivors {
        let e = &sv.e;
        let _ = writeln!(surv_csv, "{},{},{},{},{},{},{},{},{},{:.4},{:.2},{:.4},{:.2},{}", sv.from, sv.to, DIRS[e.dir as usize], csv_field(&d.name(e.a)), csv_field(&d.name(e.b)), e.h1, e.h2, e.lag, e.m, e.r, sv.zp, sv.rq, sv.zq, track_of(e).join(","));
    }
    let groups: [(usize, Option<bool>, &str); 4] = [
        (0, Some(false), "ext->stock, external series other than US-listed ETFs"),
        (0, Some(true), "ext->stock, US-listed ETFs leading stocks"),
        (1, None, "ext->ext"),
        (2, None, "stock->ext"),
    ];
    for (dir, want_etf, title) in groups {
        writeln!(sum)?;
        writeln!(sum, "-- {title} (strongest first, best move lengths and offset per pair, at most 10) --")?;
        let mut shown: HashSet<(u32, u32)> = HashSet::new();
        let mut lines = 0;
        for sv in survivors.iter().filter(|s| s.e.dir as usize == dir && want_etf.is_none_or(|w| etf(s.e.a) == w)) {
            let e = &sv.e;
            if lines >= 10 || !shown.insert((e.a, e.b)) {
                continue;
            }
            lines += 1;
            let (h1, h2, lag) = (e.h1 as usize, e.h2 as usize, e.lag as usize);
            writeln!(sum, "{}. {} -> {}: A over {h1} day(s), B over {h2} day(s), offset {lag}. Found in {} (r {:+.3}, z {:+.1}), next period {} (r {:+.3}, z {:+.1}).",
                lines, d.name(e.a), d.name(e.b), sv.from, e.r, sv.zp, sv.to, sv.rq, sv.zq)?;
            writeln!(sum, "   z in {track_label}: {}", track_of(e).join(" / "))?;
            let (inside, rest) = plain(&d, e.a, e.b, sv.from, h1, h2, lag, lo);
            let bn = d.name(e.b);
            for (label, q) in [(format!("inside {}", sv.from), inside), ("rest of history (2005-2025, sharing no day with it)".to_string(), rest)] {
                let Some(q) = q else { continue };
                writeln!(sum, "   {label}: after {}'s {h1}-day move was in its top quarter (typically {}), {bn} rose over the {h2} day(s) ending {lag} day(s) after it in {} of {} cases ({}),",
                    d.name(e.a), d.describe_raw(e.a, q.top_typ), q.top_up, q.top, pct(q.top_up as f64, q.top as f64))?;
                writeln!(sum, "     after its bottom quarter (typically {}) in {} of {} ({}), vs {} of all {} cases normally.", d.describe_raw(e.a, q.bot_typ), q.bot_up, q.bot, pct(q.bot_up as f64, q.bot as f64), pct(q.base_up as f64, q.cases as f64), q.cases)?;
            }
        }
        if lines == 0 {
            writeln!(sum, "(none)")?;
        }
    }
    writeln!(sum)?;

    // Series with the most excess against stocks (stored entries, |z| >= 4): ext->stock predictive
    // with the series as A, and stock->ext beyond the reporting delay with the series as B.
    let mut per_series: HashMap<u32, [f64; 4]> = HashMap::new(); // ES real, ES chance, SE real, SE chance
    for o in &outs {
        for e in o.entries.iter().filter(|e| e.dir != 1 && e.z().abs() >= 4.0 && d.offset_class(e) == 1) {
            let w = if e.copy == 0 { 1.0 } else { 1.0 / NULLS as f64 };
            let (g, k) = if e.dir == 0 { (e.a, 0) } else { (e.b, 2) };
            per_series.entry(g).or_default()[k + usize::from(e.copy > 0)] += w;
        }
    }
    let mut ps: Vec<(u32, [f64; 4])> = per_series.into_iter().collect();
    ps.sort_by(|a, b| (b.1[0] - b.1[1]).total_cmp(&(a.1[0] - a.1[1])));
    writeln!(sum, "== Series with the most excess strong predictive correlations with stocks (|z| >= 4, all periods) ==")?;
    writeln!(sum, "ext->stock with the series moving first, and stock->ext beyond its reporting delay with the series following.")?;
    writeln!(sum, "{:<30} {:<14} {:>10} {:>10} {:>7}   {:>10} {:>10} {:>7}", "series", "category", "ext->stock", "chance", "ratio", "stock->ext", "chance", "ratio")?;
    for (g, v) in ps.iter().filter(|x| !d.is_syn(x.0) && Some(ext[(x.0 - EXT_BASE) as usize].cat) != noise_cat).take(25) {
        let x = &ext[(g - EXT_BASE) as usize];
        writeln!(sum, "{:<30} {:<14} {:>10} {:>10} {:>7}   {:>10} {:>10} {:>7}", x.meta.id, catname(x.cat), th(v[0]), th(v[1]), ratio(v[0], v[1]), th(v[2]), th(v[3]), ratio(v[2], v[3]))?;
    }
    if let Some((g, v)) = ps.iter().find(|x| x.0 >= EXT_BASE && Some(ext[(x.0 - EXT_BASE) as usize].cat) == noise_cat) {
        writeln!(sum, "For reference, the noise series with the largest ext->stock excess: {} {} real vs {} chance ({}).", ext[(g - EXT_BASE) as usize].meta.id, th(v[0]), th(v[1]), ratio(v[0], v[1]))?;
    }
    writeln!(sum)?;

    // Verdict numbers (the interpretation is written by hand under them).
    let (rp, cp, _) = agg(&|x| x.dir == 0 && x.lag > 0);
    let mut real_ss = Oos::default();
    let mut null_ss = Oos::default();
    for &(f, to) in &transitions {
        for c in 0..=NULLS {
            if let Some(o) = oos.get(&(f, to, 0, 1, 0, c)) {
                if c == 0 { real_ss.add(o) } else { null_ss.add(o) }
            }
        }
    }
    let (rpn, cpn, _) = aggn(&|x| x.dir == 0 && x.lag > 0);
    let mut noise_ss = Oos::default();
    for &(f, to) in &transitions {
        if let Some(o) = oos.get(&(f, to, 0, 1, 0, NOISE_SRC)) {
            noise_ss.add(o);
        }
    }
    writeln!(sum, "== The decisive numbers ==")?;
    writeln!(sum, "ext->stock, predictive offsets: {} real vs {} chance correlations at |z| >= 4 (ratio {}); pure-noise externals {}.", th(rp[1]), th(cp[1]), ratio(rp[1], cp[1]), ratio(rpn[1], cpn[1]))?;
    writeln!(sum, "Noise externals picked at |z| >= 4: same sign in the next period {} of {} ({}).", th(noise_ss.same as f64), th(noise_ss.n as f64), pct(noise_ss.same as f64, noise_ss.n as f64))?;
    // US-listed ETFs (market data from the price store) vs every other external series.
    let mut etf_tab: HashMap<bool, [f64; 2]> = HashMap::new();
    for o in &outs {
        for e in o.entries.iter().filter(|e| e.dir == 0 && e.lag > 0 && e.z().abs() >= 4.0 && d.entry_cat(e) < excl0) {
            let t = etf_tab.entry(is_etf(e.a)).or_default();
            if e.copy == 0 { t[0] += 1.0 } else { t[1] += 1.0 / NULLS as f64 }
        }
    }
    for etf in [false, true] {
        let v = etf_tab.get(&etf).copied().unwrap_or_default();
        let (r, n) = (oos_etf.get(&(etf, false)).cloned().unwrap_or_default(), oos_etf.get(&(etf, true)).cloned().unwrap_or_default());
        writeln!(sum, "ext->stock predictive, {}: {} real vs {} chance at |z| >= 4 ({}); next-period same sign {} of {} ({}) vs null-picked {} ({}); mean signed r {:+.4} vs {:+.4}.",
            if etf { "US-listed ETFs only" } else { "all external series except US-listed ETFs" }, th(v[0]), th(v[1]), ratio(v[0], v[1]),
            th(r.same as f64), th(r.n as f64), pct(r.same as f64, r.n as f64), th(n.n as f64), pct(n.same as f64, n.n as f64),
            r.sum_sr / r.n.max(1) as f64, n.sum_sr / n.n.max(1) as f64)?;
    }
    writeln!(sum, "ext->stock, predictive, picked at |z| >= 4: same sign in the next period {} of {} times ({}), null-picked {} ({}); mean signed r {:+.4} vs {:+.4}.",
        th(real_ss.same as f64), th(real_ss.n as f64), pct(real_ss.same as f64, real_ss.n as f64), th(null_ss.n as f64), pct(null_ss.same as f64, null_ss.n as f64),
        real_ss.sum_sr / real_ss.n.max(1) as f64, null_ss.sum_sr / null_ss.n.max(1) as f64)?;
    writeln!(sum)?;

    // Timing.
    mark("summary", &mut lap);
    writeln!(sum, "== Timing ==")?;
    for (w, s) in &timing {
        writeln!(sum, "{w:<20} {s:>8.1}s")?;
    }
    writeln!(sum, "{:<20} {:>8.1}s", "total", t0.elapsed().as_secs_f64())?;
    writeln!(sum)?;
    writeln!(sum, "== Files ==")?;
    writeln!(sum, "externals.csv  every series: catalog row, transform, timing, coverage and whether it joined each period")?;
    writeln!(sum, "counts.csv     per period x move lengths x offset x direction x category: pairs, |z|>=3/4/5 real and chance, false-discovery results")?;
    writeln!(sum, "fdr.csv        per block matrix: the false-discovery cut-off and placebo, entries passing")?;
    writeln!(sum, "oos.csv        out-of-sample tallies")?;
    writeln!(sum, "survivors.csv  every relationship that survived the out-of-sample check, with its z in every period")?;
    writeln!(sum, "controls.csv   the control relationships, every period")?;
    writeln!(sum, "entries.bin    every real and null entry with |z| >= 3.5 (24-byte records; layout in the example's header)")?;
    writeln!(sum, "Browse: cargo run --release -p c2-census --example external_test -- --browse {} --series <id or ticker> --min-z 4", args.run)?;

    // CSV outputs.
    let mut ex = String::from("row,id,name,category,subcategory,region,catalog,transform,known_after_days,frequency,units,live_gap_rows,first,last,observations,status");
    for p in &periods {
        let _ = write!(ex, ",coverage_{0},active_{0},joins_{0}", p.id);
    }
    ex.push('\n');
    for (i, x) in ext.iter().enumerate() {
        let s = &x.meta;
        let (first, last) = series_span(&x.var, &var.days);
        let status = not_tested(x).unwrap_or_else(|| {
            let joined: Vec<String> = periods.iter().zip(&x.per).filter(|(_, p)| p.2).map(|(w, _)| w.id.to_string()).collect();
            format!("joins {}", joined.join(" "))
        });
        let _ = write!(ex, "{i},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}", s.id, csv_field(&s.name), csv_field(catname(x.cat)), csv_field(&s.subcategory), csv_field(&s.region), s.catalog, s.transform.name(), s.known_after, csv_field(&s.frequency), csv_field(&s.units), x.var.gap, first, last, x.var.raw.iter().filter(|v| v.is_finite() && **v != 0.0).count(), csv_field(&status));
        for pr in &x.per {
            let _ = write!(ex, ",{:.3},{},{}", pr.0, pr.1, u8::from(pr.2));
        }
        ex.push('\n');
    }
    std::fs::write(out_dir.join("externals.csv"), ex)?;
    let mut cc = String::from("period,h1,h2,offset,obs,direction,category,pairs,real_z3,real_z4,real_z5,chance_z3,chance_z4,chance_z5,block_fdr_passing,block_placebo_passing,cat_fdr_r,cat_fdr_passing,cat_placebo_r,cat_placebo_passing\n");
    for r in &all_rows {
        let b = |x: Option<usize>| x.map(|b| format!("{:.4}", b as f64 / R_BINS as f64)).unwrap_or_default();
        let _ = writeln!(cc, "{},{},{},{},{},{},{},{},{},{},{},{:.2},{:.2},{:.2},{},{},{},{},{},{}", r.period, r.h1, r.h2, r.lag, r.m, DIRS[r.dir], csv_field(catname(r.cat)), r.pairs, r.real[0], r.real[1], r.real[2],
            r.null[0] as f64 / NULLS as f64, r.null[1] as f64 / NULLS as f64, r.null[2] as f64 / NULLS as f64, r.fdr_n, r.placebo_n, b(r.cfdr), r.cfdr_n, b(r.cplacebo), r.cplacebo_n);
    }
    std::fs::write(out_dir.join("counts.csv"), cc)?;
    let mut fc = String::from("period,h1,h2,offset,obs,direction,pairs,fdr_r,fdr_passing,placebo_r,placebo_passing\n");
    for m in &all_mats {
        let b = |x: Option<usize>| x.map(|b| format!("{:.4}", b as f64 / R_BINS as f64)).unwrap_or_default();
        let _ = writeln!(fc, "{},{},{},{},{},{},{},{},{},{},{}", m.period, m.h1, m.h2, m.lag, m.m, DIRS[m.dir], m.pairs, b(m.fdr), m.fdr_n, b(m.placebo), m.placebo_n);
    }
    std::fs::write(out_dir.join("fdr.csv"), fc)?;
    let mut oc = String::from("from,to,direction,offset_class,pick,copy,kept,same_sign,mean_signed_r,rep_z2,rep_z3,dropped\n");
    let mut keys: Vec<_> = oos.keys().copied().collect();
    keys.sort();
    for k in keys {
        let o = &oos[&k];
        let _ = writeln!(oc, "{},{},{},{},{},{},{},{},{:.5},{},{},{}", k.0, k.1, DIRS[k.2], OFFSETS[k.3], if k.4 == 0 { "z>=4" } else { "top1000" }, k.5, o.n, o.same, o.sum_sr / o.n.max(1) as f64, o.rep2, o.rep3, o.dropped);
    }
    std::fs::write(out_dir.join("oos.csv"), oc)?;
    std::fs::write(out_dir.join("survivors.csv"), surv_csv)?;
    let mut eb = Vec::new();
    for o in &outs {
        for e in &o.entries {
            e.bytes(&mut eb);
        }
    }
    std::fs::write(out_dir.join("entries.bin"), eb)?;
    std::fs::write(out_dir.join("summary.txt"), &sum)?;
    print!("{sum}");
    say(&t0, &format!("done: {}", out_dir.join("summary.txt").display()));
    Ok(())
}

/// Why a series joined no correlation period (None if it joined one).
fn not_tested(x: &ExtInfo) -> Option<String> {
    if x.per.iter().any(|p| p.2) {
        return None;
    }
    let max_active = x.per.iter().map(|p| p.1).max().unwrap_or(0);
    let max_cov = x.per.iter().map(|p| p.0).fold(0f64, f64::max);
    Some(if max_active < MIN_ACTIVE {
        format!("too sparse to test: at most {max_active} non-zero moves in any correlation period (needs {MIN_ACTIVE})")
    } else {
        format!("too short or gappy: values on at most {:.0}% of a period's trading days (needs 90%)", 100.0 * max_cov)
    })
}

fn series_span(v: &ExtVar, days: &[i32]) -> (String, String) {
    let first = v.v.iter().position(|x| x.is_finite());
    let last = v.v.iter().rposition(|x| x.is_finite());
    match (first, last) {
        (Some(a), Some(b)) => (date_string(days[a]), date_string(days[b])),
        _ => (String::new(), String::new()),
    }
}

struct Base {
    real: [f64; 3],
    null: [f64; 3],
    matrices: usize,
    fdr: usize,
    placebo: usize,
}

/// The census's own stock->stock counts for the same periods and move lengths, from its task
/// summary files: (period, predictive) -> totals.
fn census_baseline(c: &Census, periods: &[usize], pairs: &[(usize, usize)]) -> Result<HashMap<(usize, bool), Base>> {
    let mut out: HashMap<(usize, bool), Base> = HashMap::new();
    let want: HashSet<usize> = periods.iter().copied().collect();
    let mut lens: BTreeSet<usize> = BTreeSet::new();
    for &p in periods {
        lens.insert(c.windows[p].length);
    }
    for &(h1, h2) in pairs {
        for &len in &lens {
            let Some(&task) = c.tasks.get(&(h1, h2, len)) else { continue };
            let Ok(text) = std::fs::read_to_string(c.meta.task_path(task, "csv")) else { continue };
            for line in text.lines().skip(1) {
                let f: Vec<&str> = line.split(',').collect();
                if f.len() < 18 {
                    continue;
                }
                let wid: usize = f[0].parse()?;
                if !want.contains(&wid) {
                    continue;
                }
                let lag: usize = f[3].parse()?;
                let b = out.entry((wid, lag > 0)).or_insert(Base { real: [0.0; 3], null: [0.0; 3], matrices: 0, fdr: 0, placebo: 0 });
                for t in 0..3 {
                    b.real[t] += f[6 + t].parse::<f64>()?;
                    b.null[t] += f[9 + t].parse::<f64>()?;
                }
                b.matrices += 1;
                b.fdr += usize::from(!f[12].is_empty());
                b.placebo += usize::from(!f[17].trim().is_empty());
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Controls

fn find_ext(d: &Data, pats: &[&str], exclude: &[&str]) -> Vec<u32> {
    let mut out: Vec<u32> = d
        .ext
        .iter()
        .enumerate()
        .filter(|(_, x)| x.meta.category != SYN)
        .filter(|(_, x)| {
            let s = format!("{} {}", x.meta.id, x.meta.name).to_ascii_lowercase();
            pats.iter().any(|p| s.contains(p)) && !exclude.iter().any(|p| s.contains(p))
        })
        .map(|(i, _)| EXT_BASE + i as u32)
        .collect();
    out.sort();
    out
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Writes the controls' report into `s`; returns their rows as CSV.
fn controls(d: &Data, periods: &[&Window], syn: &[SynSpec], outs: &[PeriodOut], transitions: &[(usize, usize)], s: &mut String) -> Result<String> {
    let mut csv = String::from("control,a,b,direction,h1,h2,offset,period,r,z,obs\n");
    let pids: Vec<usize> = periods.iter().map(|p| p.id).collect();
    // Same-day sector controls: external vs stocks' market-relative moves.
    let sector = |s: &mut String, csv: &mut String, label: &str, pats: &[&str], excl: &[&str], tickers: &[&str], need: f64| -> Result<()> {
        let exts = find_ext(d, pats, excl);
        writeln!(s, "{label}:")?;
        if exts.is_empty() {
            writeln!(s, "  NOT RUN: no external series matching {pats:?}")?;
            return Ok(());
        }
        let stocks: Vec<u32> = tickers.iter().filter_map(|t| d.stock_gid(t)).collect();
        for &x in exts.iter().take(4) {
            let k = d.ext[(x - EXT_BASE) as usize].meta.known_after;
            // A series known the same day shows the link at offset 0 ext->stock; one known k days
            // later lands k trading days late, so the same link shows as stock->ext at offset k.
            let kk = (k.max(1) as usize).min(10);
            let mut line = format!("  {} (known_after_days {k}): median z over {} stocks", d.name(x), stocks.len());
            let mut pass = true;
            let mut any = false;
            for &p in &pids {
                let (mut z0, mut z1) = (Vec::new(), Vec::new());
                for &b in &stocks {
                    if let Some((r, z, m)) = d.stat(x, b, p, 1, 1, 0) {
                        z0.push(z);
                        let _ = writeln!(csv, "{label},{},{},ext->stock,1,1,0,{p},{r:.4},{z:.2},{m}", d.name(x), d.name(b));
                    }
                    if let Some((r, z, m)) = d.stat(b, x, p, 1, 1, kk) {
                        z1.push(z);
                        let _ = writeln!(csv, "{label},{},{},stock->ext,1,1,{kk},{p},{r:.4},{z:.2},{m}", d.name(b), d.name(x));
                    }
                }
                if z0.is_empty() {
                    write!(line, "; {p}: -")?;
                    continue;
                }
                any = true;
                let (m0, m1) = (median(&mut z0), median(&mut z1));
                let key = if k == 0 { m0 } else { m1 };
                pass &= key >= need;
                write!(line, "; {p}: ext->stock offset 0 {m0:+.1}, stock->ext offset {kk} {m1:+.1}")?;
            }
            writeln!(s, "{line}")?;
            writeln!(s, "    -> {}", if !any { "NOT RUN (not in any period)".to_string() } else if pass { format!("PASS (timing-consistent median z >= {need} in every period present)") } else { format!("FAIL (timing-consistent median z < {need} somewhere)") })?;
        }
        Ok(())
    };
    sector(s, &mut csv, "Oil vs energy stocks, same day (A over 1 day, B over 1 day)", &["dcoilwtico", "dcoilbrenteu", "etf_uso"], &[],
        &["XOM", "CVX", "COP", "OXY", "EOG", "DVN", "APA", "HES", "MRO", "HAL", "SLB", "MPC", "VLO"], 5.0)?;
    sector(s, &mut csv, "Gold vs gold miners, same day", &["etf_gld", "xauusd"], &[], &["NEM", "AEM", "KGC", "AU", "GFI", "RGLD", "WPM", "FNV", "GOLD", "HMY", "EGO", "IAG"], 5.0)?;
    sector(s, &mut csv, "Natural gas vs gas producers, same day", &["etf_ung", "dhhngsp"], &[], &["EQT", "RRC", "SWN", "AR", "CNX", "CTRA", "CHK", "COG", "GPOR", "SM"], 3.0)?;
    // SPY leading Asian indices by a day.
    writeln!(s, "SPY moves first, Asian indices follow the next day (ext->ext, 1/1 days, offset 1):")?;
    let mut spy = find_ext(d, &["etf_spy"], &[]);
    if spy.is_empty() {
        spy = find_ext(d, &["gspc", "sp500", "spx"], &["vix", "spyder", "volatility"]);
    }
    let mut asia = find_ext(d, &["nikkei", "_nkx", "n225", "hang seng", "_hsi", "kospi", "_ks11", "s&p/asx", "_axjo", "_aord", "taiex", "_twii", "shanghai", "_shc", "sensex", "nifty", "straits times", "_sti"], &["etf_", "ishares", "msci"]);
    asia.retain(|&g| d.ext[(g - EXT_BASE) as usize].meta.category == "world_equity");
    if spy.is_empty() || asia.is_empty() {
        writeln!(s, "  NOT RUN: SPY {} / Asian index {} series", spy.len(), asia.len())?;
    } else {
        for &x in spy.iter().take(1) {
            for &y in asia.iter().take(8) {
                let mut line = format!("  {} -> {} (known_after_days {}):", d.name(x), d.name(y), d.ext[(y - EXT_BASE) as usize].meta.known_after);
                let mut pass = true;
                let mut any = false;
                for &p in &pids {
                    let (Some(n1), Some(n0)) = (d.stat(x, y, p, 1, 1, 1), d.stat(x, y, p, 1, 1, 0)) else { continue };
                    any = true;
                    pass &= n1.1 >= 5.0 && n1.0 > 0.0;
                    let _ = writeln!(csv, "spy_asia,{},{},ext->ext,1,1,1,{p},{:.4},{:.2},{}", d.name(x), d.name(y), n1.0, n1.1, n1.2);
                    write!(line, " {p}: next day r {:+.2} z {:+.1} (same day z {:+.1});", n1.0, n1.1, n0.1)?;
                }
                writeln!(s, "{line} -> {}", if !any { "NOT RUN" } else if pass { "PASS" } else { "FAIL" })?;
            }
        }
    }
    // Heating degree days vs natural gas.
    writeln!(s, "Weather: heating degree days (HDD) vs natural gas (ext->ext). Colder than normal = more HDD = gas up.")?;
    writeln!(s, "  Timing: a day's HDD is known k days later (k = 1 for stations, 2 for CPC), so the physical same-day link shows")?;
    writeln!(s, "  up as gas moving first with HDD following at offset k; 'weather known, gas follows' is the tradable direction.")?;
    let mut hdd = find_ext(d, &["cpc_hdd_gas_conus"], &[]);
    hdd.extend(find_ext(d, &["_hdd", "heating degree"], &["cpc_hdd_gas_conus"]));
    let gas = find_ext(d, &["etf_ung", "dhhngsp"], &[]);
    let (mut link, mut lead) = (f64::NEG_INFINITY, 0f64);
    if hdd.is_empty() || gas.is_empty() {
        writeln!(s, "  NOT RUN: {} degree-day series, {} natural gas series", hdd.len(), gas.len())?;
    } else {
        for &g in &gas {
            let kg = d.ext[(g - EXT_BASE) as usize].meta.known_after;
            writeln!(s, "  {} (known_after_days {kg}) vs {} HDD series; z by period for {}, and the median over all HDD series:", d.name(g), hdd.len(), d.name(hdd[0]))?;
            let tests: Vec<(String, bool, usize, usize, usize)> = {
                let kh = d.ext[(hdd[0] - EXT_BASE) as usize].meta.known_after.max(0) as usize;
                vec![
                    (format!("gas moves, HDD follows at offset {kh} (same physical day), 1-day moves"), false, 1, 1, kh),
                    ("gas moves, HDD follows, 5-day moves at offset 5".to_string(), false, 5, 5, 5),
                    ("HDD known, gas same trading day (offset 0), 1-day moves".to_string(), true, 1, 1, 0),
                    ("HDD known, gas follows next day (offset 1), 1-day moves".to_string(), true, 1, 1, 1),
                    ("HDD known, gas follows over 5 days (offset 5), 5-day moves".to_string(), true, 5, 5, 5),
                ]
            };
            for (label, weather_first, h1, h2, lag) in tests {
                let mut line = format!("    {label:<62}");
                for &p in &pids {
                    let st = if weather_first { d.stat(hdd[0], g, p, h1, h2, lag) } else { d.stat(g, hdd[0], p, h1, h2, lag) };
                    write!(line, " {p}: {}", st.map(|v| format!("{:+.1}", v.1)).unwrap_or_else(|| "-".into()))?;
                }
                let mut all: Vec<f64> = Vec::new();
                for &h in &hdd {
                    for &p in &pids {
                        let (a, b) = if weather_first { (h, g) } else { (g, h) };
                        if let Some((r, z, m)) = d.stat(a, b, p, h1, h2, lag) {
                            all.push(z);
                            let _ = writeln!(csv, "hdd_gas,{},{},ext->ext,{h1},{h2},{lag},{p},{r:.4},{z:.2},{m}", d.name(a), d.name(b));
                        }
                    }
                }
                let n3 = all.iter().filter(|z| **z >= 3.0).count();
                let med = median(&mut all);
                if weather_first {
                    lead = lead.max(med.abs());
                } else {
                    link = link.max(med);
                }
                writeln!(s, "{line}   all HDD: median {med:+.2} (n {}, {n3} with z >= 3)", all.len())?;
            }
        }
        writeln!(s, "    -> {} (physical same-day link: best median z {link:+.2}, expected >= 2); tradable lead: largest |median z| {lead:.2} (none expected: markets trade the forecasts)",
            if link >= 2.0 { "PASS" } else { "WEAK" })?;
    }
    // Synthetic positive controls: found at the right offsets and move lengths?
    writeln!(s, "Synthetic positive controls (a stock's census moves shifted in time, x{SYN_LOAD}, plus noise; expected r = 0.29 x overlap):")?;
    for sp in syn {
        let Some(xi) = d.ext.iter().position(|x| x.meta.id == sp.id) else { continue };
        let x = EXT_BASE + xi as u32;
        let Some(b) = d.stock_gid(&sp.ticker) else { continue };
        let rho = SYN_LOAD / (1.0 + SYN_LOAD * SYN_LOAD).sqrt();
        let lead = sp.shift > 0;
        let (aa, bb) = if lead { (x, b) } else { (b, x) };
        let l = sp.shift.unsigned_abs() as isize;
        let mut worst = 0f64;
        let mut best: Option<(f64, usize, usize, usize, usize, f64)> = None;
        let mut checked = 0;
        for &p in &pids {
            for h1 in [1usize, 3, 5, 10, 21, 63] {
                for h2 in [1usize, 3, 5, 10, 21, 63] {
                    for lag in grid::lags(h2) {
                        let Some((r, z, m)) = d.stat(aa, bb, p, h1, h2, lag) else { continue };
                        // A's days (e-h1, e] carry B's days shifted by l; B's are (e+lag-h2, e+lag].
                        let (a0, a1, b0, b1) = (-(h1 as isize) + l, l, lag as isize - h2 as isize, lag as isize);
                        let overlap = (a1.min(b1) - a0.max(b0)).max(0) as f64;
                        let expect = rho * overlap / ((h1 * h2) as f64).sqrt();
                        worst = worst.max((r - expect).abs() / (4.0 / (m as f64).sqrt() + 0.03));
                        checked += 1;
                        if best.is_none_or(|bst| z.abs() > bst.0.abs()) {
                            best = Some((z, h1, h2, lag, p, expect));
                        }
                    }
                }
            }
        }
        let Some((bz, bh1, bh2, blag, bp, bexp)) = best else { continue };
        let (r11, ok11) = pids.iter().filter_map(|&p| d.stat(aa, bb, p, 1, 1, l as usize)).fold((Vec::new(), true), |mut acc, st| { acc.0.push(st.1); acc.1 &= st.1 >= 8.0; acc });
        let pass = worst <= 1.0 && ok11 && blag == l as usize && bexp >= 0.9 * rho;
        writeln!(s, "  {} ({}): strongest of {checked} grid points is {}/{} days at offset {blag} in {bp} (z {bz:+.1}, expected r {bexp:.2}); 1/1 at offset {l}: z {}; every grid point within tolerance of the expected r: {} -> {}",
            sp.id, if lead { format!("moves {l} days before {}", sp.ticker) } else { format!("follows {} by {l} days", sp.ticker) }, bh1, bh2,
            r11.iter().map(|z| format!("{z:+.1}")).collect::<Vec<_>>().join("/"), if worst <= 1.0 { "yes" } else { "no" }, if pass { "PASS" } else { "FAIL" })?;
    }
    if let (Some(xi), Some(yi)) = (d.ext.iter().position(|x| x.meta.id == "syn_eex_x"), d.ext.iter().position(|x| x.meta.id == "syn_eex_y2")) {
        let (x, y) = (EXT_BASE + xi as u32, EXT_BASE + yi as u32);
        let zs: Vec<f64> = pids.iter().filter_map(|&p| d.stat(x, y, p, 1, 1, 2).map(|v| v.1)).collect();
        let z0: Vec<f64> = pids.iter().filter_map(|&p| d.stat(x, y, p, 1, 1, 1).map(|v| v.1)).collect();
        let pass = !zs.is_empty() && zs.iter().all(|&z| z >= 8.0) && z0.iter().all(|z| z.abs() < 4.0);
        writeln!(s, "  syn_eex_y2 follows syn_eex_x by 2 days (ext->ext): 1/1 at offset 2 z {}; at offset 1 z {} -> {}", zs.iter().map(|z| format!("{z:+.1}")).collect::<Vec<_>>().join("/"), z0.iter().map(|z| format!("{z:+.1}")).collect::<Vec<_>>().join("/"), if pass { "PASS" } else { "FAIL" })?;
    }
    // Placebos: sunspots, lunar phase, and the synthetic noise series.
    writeln!(s, "Placebos (should look like chance against stocks): real vs chance at |z| >= 4 between the series and stocks (ext->stock and")?;
    writeln!(s, "stock->ext; ext->ext is left out because calendar series relate trivially to each other), predictive and same day; out-of-sample same sign")?;
    let placebo_groups: Vec<(String, Vec<u32>)> = vec![
        ("sunspots".into(), find_ext(d, &["sunspot", "silso"], &[])),
        ("solar radio flux F10.7".into(), find_ext(d, &["f107"], &[])),
        ("lunar phase".into(), find_ext(d, &["moon", "lunar"], &[])),
        ("Wikipedia 'Cat' and 'Philosophy' views".into(), find_ext(d, &["wiki_cat", "wiki_philosophy"], &[])),
        ("synthetic noise".into(), d.ext.iter().enumerate().filter(|(_, x)| x.meta.category == SYN && x.meta.subcategory == "noise").map(|(i, _)| EXT_BASE + i as u32).collect()),
    ];
    for (label, ids) in placebo_groups {
        if ids.is_empty() {
            writeln!(s, "  {label}: NOT RUN (no series)")?;
            continue;
        }
        let set: HashSet<u32> = ids.iter().copied().collect();
        let mut c = [0f64; 4];
        let mut same = (0usize, 0usize);
        for o in outs {
            for e in o.entries.iter().filter(|e| e.dir != 1 && (set.contains(&e.a) || set.contains(&e.b)) && e.z().abs() >= 4.0) {
                let k = if e.lag > 0 { 0 } else { 2 } + usize::from(e.copy > 0);
                c[k] += if e.copy == 0 { 1.0 } else { 1.0 / NULLS as f64 };
                if e.copy == 0
                    && e.lag > 0
                    && let Some(&(_, to)) = transitions.iter().find(|t| t.0 == e.period as usize)
                    && let Some((rq, _, _)) = d.stat(e.a, e.b, to, e.h1 as usize, e.h2 as usize, e.lag as usize)
                {
                    same.0 += usize::from(rq * e.r as f64 > 0.0);
                    same.1 += 1;
                }
            }
        }
        let ratio_p = if c[1] > 0.0 { c[0] / c[1] } else { f64::NAN };
        let ok = (c[1] < 20.0 || (0.75..=1.33).contains(&ratio_p)) && (same.1 < 30 || ((same.0 as f64 / same.1 as f64) - 0.5).abs() < 0.1);
        writeln!(s, "  {label} ({}): predictive {} real vs {} chance ({}), same day {} vs {} ({}); next-period same sign {} of {} ({}) -> {}",
            ids.iter().map(|&g| d.name(g)).collect::<Vec<_>>().join(", ").chars().take(120).collect::<String>(),
            th(c[0]), th(c[1]), ratio(c[0], c[1]), th(c[2]), th(c[3]), ratio(c[2], c[3]), same.0, same.1, pct(same.0 as f64, same.1 as f64), if ok { "PASS (looks like chance)" } else { "CHECK" })?;
    }
    Ok(csv)
}
