//! Step-by-step recomputation of one relationship from prices, for the GUI's Explainer and
//! `c2 census explain`, checked against what the census run stored.
//!
//! - Before it explains anything, `explain` decides whether the census computed the matrix the
//!   relationship belongs to (`Explained::census_computed`). The run's summary file is the record
//!   of what it computed; when it has no line for the matrix, the run's own task list, offsets,
//!   observation minimum and stock count give the reason.
//! - The daily standardization runs `variable::production`'s own per-column steps
//!   (`log_returns`, `residuals`, `standardize`) and keeps their intermediate values (beta,
//!   residual, trailing mean and sd). Being the census's own code, it can't disagree with the
//!   census variable, so the check that counts is against what the run stored
//!   (`Explained::check`): the stored record's observation sums when the census kept the pair,
//!   else the matrix's summary line.
//! - Of the run's files it reads only the one task's summary file and, of its record file, only
//!   the matrix's block.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

use crate::analyze::{Words, words};
use crate::grid;
use crate::meta::{RunMeta, RunWindow};
use crate::moves::{self, MoveSize};
use crate::run::{FLAG_FDR, FLAG_NULL, FLAG_TOP, Plan, RECORD_BYTES, Record, SUMMARY_HEADER};
use crate::variable;

pub struct Request {
    pub leader: String,
    pub follower: String,
    pub h1: usize,
    pub h2: usize,
    /// Offset in trading days from the end of A's move to the end of B's: 0 (the same-day
    /// reference) or at least `h2`.
    pub lag: usize,
    /// The run's correlation period id (`RunMeta::windows`; a `Plan::for_run` plan keeps the
    /// run's ids).
    pub window: usize,
    /// Leader move threshold for the plain-terms counts (0.03 = 3%); None = its typical move.
    pub threshold: Option<f64>,
}

/// One stock's daily steps over the shown rows.
#[derive(Default)]
pub struct DaySteps {
    pub ticker: String,
    pub close: Vec<f32>,
    /// ln(close / previous close).
    pub ret: Vec<f32>,
    /// Beta to the market from the previous 252 rows.
    pub beta: Vec<f64>,
    /// ret - beta * market.
    pub resid: Vec<f64>,
    /// Mean and sd of the residual over the previous 252 rows.
    pub mean: Vec<f64>,
    pub sd: Vec<f64>,
    /// (resid - mean) / sd clipped at +-5, recomputed here: the daily values the observations
    /// sum.
    pub v: Vec<f32>,
    /// Deprecated: the census variable's in-memory value for the same day. It comes from the
    /// same code as `v`, so it always equals it; the census doesn't store daily values.
    /// `Explained::check` compares with what the census stored.
    pub v_census: Vec<f32>,
    /// Deprecated: largest |v - v_census| over the shown rows (always 0, see `v_census`).
    pub max_diff: f32,
    /// In the correlation period's stocks as the run recorded them (its `universes.bin`).
    pub in_universe: bool,
}

pub struct Obs {
    /// Row (in the shown rows) where the leader's move ends; the follower's ends `lag` rows later.
    pub a_end: usize,
    pub b_end: usize,
    /// Raw log price moves.
    pub a_move: f64,
    pub b_move: f64,
    /// Census units: sum of the daily values `v` over the move length / sqrt(length); a day
    /// without a value counts as 0, as in the census.
    pub x: f64,
    pub y: f64,
}

pub struct Stored {
    pub record: Record,
    pub top: bool,
    pub significant: bool,
}

/// The matrix the relationship was tested in: every universe stock against every other at this
/// move-length pair, offset and period (its line in the run's summary file).
#[derive(Clone, Default)]
pub struct Matrix {
    /// The run has a summary line for it, i.e. the census computed it (`Explained::census_computed`).
    pub found: bool,
    pub stocks: usize,
    pub obs: usize,
    /// Real pairs at |z| >= 3, 4, 5 and the average of the sign-flip nulls.
    pub real: [f64; 3],
    pub null: [f64; 3],
    /// |r| above which pairs passed the false-discovery test, if any did (never at offset 0).
    pub fdr_r: Option<f64>,
    /// Records the census stored for the matrix: real pairs (its strongest `keep_real` plus any
    /// that passed the significance test) and sign-flip pairs.
    pub stored_real: usize,
    pub stored_null: usize,
}

/// Why the census never computed a matrix (every stock of a correlation period against every
/// other, at one pair of move lengths and one offset). `Display` gives the reason in words.
#[derive(Clone, Debug, PartialEq)]
pub enum NotComputed {
    /// The run computed nothing for these move lengths in correlation periods of this length
    /// (`length` 0: the full-history period). `horizons` are the run's move lengths;
    /// `shorter_only`: the run paired B's moves only with A moves at least as long (older runs).
    NoTask { run: String, h1: usize, h2: usize, length: usize, horizons: Vec<usize>, shorter_only: bool },
    /// The census doesn't test this offset for B's move length; `nearest` are the closest
    /// offsets it does test (below and above).
    OffGrid { lag: usize, h2: usize, nearest: Vec<usize> },
    /// The correlation period (`days` trading days) has room for `obs` observations at this
    /// offset; the census needs `min`.
    TooFewObservations { days: usize, obs: usize, min: usize },
    /// Fewer than two stocks qualified in the correlation period.
    TooFewStocks { stocks: usize },
    /// Nothing above rules it out, yet the run's summary has no line for it. `period_started`:
    /// the summary has lines for this correlation period at other offsets (its files are
    /// inconsistent); otherwise the run hasn't got to this period (unfinished or stopped).
    NoSummary { run: String, period_started: bool },
}

fn days(d: usize) -> String {
    if d == 1 { "1 day".to_string() } else { format!("{d} days") }
}

/// "a", "a and b", "a, b and c".
fn list_and(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

impl std::fmt::Display for NotComputed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotComputed::NoTask { run, h1, h2, length, horizons, shorter_only } => {
                let period = if *length == 0 { "the full-history correlation period".to_string() } else { format!("{length}-day correlation periods") };
                write!(f, "census {run} computed nothing for A moving over {} and B over {} in {period}", days(*h1), days(*h2))?;
                if !horizons.contains(h1) || !horizons.contains(h2) {
                    let hs: Vec<String> = horizons.iter().map(|h| h.to_string()).collect();
                    write!(f, " (its move lengths are {} days)", list_and(&hs))
                } else if *shorter_only && h2 > h1 {
                    write!(f, " (this older run paired B's moves only with A moves at least as long)")
                } else {
                    Ok(())
                }
            }
            NotComputed::OffGrid { lag, h2, nearest } => {
                let near: Vec<String> = nearest.iter().map(|k| k.to_string()).collect();
                let which = if nearest.len() == 1 { "offset it tests is" } else { "offsets it tests are" };
                write!(f, "the census doesn't test an offset of {} for B moves over {}; the nearest {which} {} days", days(*lag), days(*h2), list_and(&near))
            }
            NotComputed::TooFewObservations { days, obs, min } => {
                write!(f, "a {days}-day correlation period has room for only {obs} observations at this offset, and the census needs at least {min}")
            }
            NotComputed::TooFewStocks { stocks } => {
                write!(f, "only {stocks} stock{} qualified in this correlation period, and the census needs at least 2", if *stocks == 1 { "" } else { "s" })
            }
            NotComputed::NoSummary { run, period_started: false } => write!(
                f,
                "census {run} hasn't computed this correlation period at these move lengths (the run is unfinished, or was stopped before it got there)"
            ),
            NotComputed::NoSummary { run, period_started: true } => write!(
                f,
                "census {run}'s summary has lines for this correlation period at these move lengths, but none for this offset, although the census computes it (the run's files may be damaged)"
            ),
        }
    }
}

/// What the Explainer's numbers were checked against: what the census stored (it doesn't store
/// daily values).
#[derive(Clone, Debug)]
pub enum Check {
    /// The census stored this pair: the observation count and sums recomputed here against the
    /// record's.
    Record(SumsCheck),
    /// The census computed the matrix but didn't store this pair, so nothing is stored for the
    /// pair itself; the matrix's summary line against what is recomputed here.
    Summary(SummaryCheck),
    /// The census never computed the matrix: nothing stored to check against.
    Nothing,
}

/// Recomputed observation sums must agree with the record's f32 sums to this (relative, see
/// `SumsCheck::max_rel_diff`); f32 rounding alone is about 6e-8.
pub const SUMS_TOLERANCE: f64 = 1e-6;

#[derive(Clone, Debug)]
pub struct SumsCheck {
    /// Observations recomputed here, and in the record.
    pub obs: usize,
    pub stored_obs: usize,
    /// Sums of x, y, x^2, y^2 and xy over the observations: recomputed here, and as the record
    /// stores them (f32, read as f64).
    pub sums: [f64; 5],
    pub stored: [f64; 5],
    /// Largest difference between the two, each relative to its sum's natural size so that a
    /// sum near 0 can't inflate it: sum x against sqrt(m sum x^2), sum y against sqrt(m sum y^2),
    /// sum x^2 and sum y^2 against themselves, sum xy against sqrt(sum x^2 sum y^2). Infinite if
    /// a sum is not finite.
    pub max_rel_diff: f64,
}

impl SumsCheck {
    /// Same observation count, and every sum within `SUMS_TOLERANCE`.
    pub fn agrees(&self) -> bool {
        self.obs == self.stored_obs && self.max_rel_diff <= SUMS_TOLERANCE
    }
}

#[derive(Clone, Debug)]
pub struct SummaryCheck {
    /// The correlation period's stocks (the run's universe) and the observations recomputed
    /// here, against the summary line's.
    pub stocks: usize,
    pub stored_stocks: usize,
    pub obs: usize,
    pub stored_obs: usize,
}

impl SummaryCheck {
    pub fn agrees(&self) -> bool {
        self.stocks == self.stored_stocks && self.obs == self.stored_obs
    }
}

pub struct Explained {
    pub h1: usize,
    pub h2: usize,
    pub lag: usize,
    /// Offset 0: B's move ends the same day as A's. The census computes it as the same-period
    /// reference for its later offsets; it is not a prediction.
    pub same_day: bool,
    pub window: usize,
    pub period_first: i32,
    pub period_last: i32,
    pub period_days: usize,
    /// Stocks in the correlation period, as the run recorded them.
    pub universe: usize,
    /// Dates of the shown rows: the period plus the day before it (the first move's start).
    pub days: Vec<i32>,
    pub market: Vec<f64>,
    pub a: DaySteps,
    pub b: DaySteps,
    pub obs: Vec<Obs>,
    pub n: usize,
    /// Exact Pearson r of x and y (what the census stores), its Fisher z, and r of the raw moves.
    pub r: f64,
    pub z: f64,
    pub r_raw: f64,
    /// Whether the census computed this matrix (then `matrix` is its summary line), else why
    /// not. Decided before anything else; the pair is explained either way.
    pub census_computed: Result<(), NotComputed>,
    /// `moves::move_per_pct` with `move_size_r`: r x sd(B's market-relative h2-day moves) /
    /// sd(A's market-relative h1-day moves) over the period's observations, i.e. B's move
    /// relative to the market per 1% move of A relative to the market. For a stored pair it is
    /// exactly (bit for bit) the multiplier the Forecast applies and the timeline's "% per 1%".
    /// NaN if fewer than 10 observations have both moves.
    pub move_per_pct: f64,
    /// Its parts: the two standard deviations (log changes) and the observations used.
    pub move_size: Option<MoveSize>,
    /// The r `move_size` uses: the stored record's r when the census kept the pair (as the
    /// Forecast and the timeline do), else `r` recomputed here.
    pub move_size_r: f64,
    /// Ok when the Forecast uses this relationship (it multiplies A's latest move relative to
    /// the market by `move_per_pct`, as of days on which this is the most recent correlation
    /// period of the relationship), else why it doesn't.
    pub forecast_use: Result<(), String>,
    /// The census's record of this pair, when it stored it.
    pub stored: Option<Stored>,
    pub matrix: Matrix,
    /// What the numbers above were checked against in the run's stored data.
    pub check: Check,
    /// The plan's `variable_note`: set when values recomputed today may differ from those the run
    /// stored (its census variable started on another day, e.g. `smoke`), saying why. Show it with
    /// the correlation period (step 1) and next to `check`.
    pub variable_note: Option<String>,
    /// Pairs tested in the matrix and how many chance alone would give at |z| >= this one.
    pub pairs_tested: f64,
    pub chance_analytic: f64,
    pub words: Option<Words>,
}

impl Explained {
    /// Why the census never computed this pair, if it didn't: its matrix wasn't computed
    /// (`census_computed`), or A or B wasn't among the correlation period's stocks. None when it
    /// computed the pair, stored or not.
    pub fn never_computed(&self) -> Option<String> {
        if let Err(why) = &self.census_computed {
            return Some(why.to_string());
        }
        let out: Vec<String> = [&self.a, &self.b].iter().filter(|s| !s.in_universe).map(|s| s.ticker.clone()).collect();
        (!out.is_empty()).then(|| format!("{} did not qualify in this correlation period", list_and(&out)))
    }
}

/// Production standardization for variable column `j`, keeping the intermediates: the same
/// functions `variable::production` runs on every column.
fn day_steps(plan: &Plan, j: usize) -> DaySteps {
    let var = &plan.var;
    let c = var.col(&var.close, j);
    let ret = variable::log_returns(c);
    let res = variable::residuals(&ret, &var.market);
    let (mut mean, mut sd, mut v) = (vec![f64::NAN; var.t], vec![f64::NAN; var.t], vec![f32::NAN; var.t]);
    variable::standardize(&res.resid, &mut v, Some((&mut mean, &mut sd)));
    DaySteps { close: c.to_vec(), ret, beta: res.beta, resid: res.resid, mean, sd, v, v_census: var.col(&var.v, j).to_vec(), ..Default::default() }
}

fn column(plan: &Plan, store: &c2_data::Store, ticker: &str) -> Result<usize> {
    let id = store.id(ticker).with_context(|| format!("{ticker} is not in the price store"))?;
    plan.var.ids.iter().position(|&i| i == id).with_context(|| format!("{ticker} is not a common stock the census covers"))
}

/// "756-day correlation period 2022-04-08..2025-04-14".
fn describe_period(w: &RunWindow) -> String {
    let what = if w.length == 0 { "full-history".to_string() } else { format!("{}-day", w.length) };
    format!("{what} correlation period {}..{}", c2_data::date_string(w.first), c2_data::date_string(w.last))
}

/// The census's offsets for B moves of `h2` days nearest to `lag`: `lag` itself if the census
/// tests it, else the closest below and above it (one of them past either end).
pub fn nearest_offsets(h2: usize, lag: usize) -> Vec<usize> {
    let lags = grid::lags(h2);
    if lags.contains(&lag) {
        return vec![lag];
    }
    let below = lags.iter().copied().filter(|&k| k < lag).max();
    let above = lags.iter().copied().filter(|&k| k > lag).min();
    below.into_iter().chain(above).collect()
}

/// What decides whether the census computed a matrix, as plain values.
struct Facts<'a> {
    run: &'a str,
    horizons: &'a [usize],
    min_obs: usize,
    shorter_only: bool,
    /// The run has a task for (h1, h2, length), and its summary has lines for this period.
    task: bool,
    period_seen: bool,
    h1: usize,
    h2: usize,
    lag: usize,
    length: usize,
    /// The period's trading days, observations at this offset, and stocks.
    days: usize,
    obs: usize,
    stocks: usize,
}

/// Why the census has no summary line for a matrix: the first of the run's rules it fails, in
/// the order the census applies them, else `NoSummary`.
fn why_not(f: &Facts) -> NotComputed {
    if !f.task {
        return NotComputed::NoTask { run: f.run.to_string(), h1: f.h1, h2: f.h2, length: f.length, horizons: f.horizons.to_vec(), shorter_only: f.shorter_only };
    }
    if !grid::lags(f.h2).contains(&f.lag) {
        return NotComputed::OffGrid { lag: f.lag, h2: f.h2, nearest: nearest_offsets(f.h2, f.lag) };
    }
    if f.obs < f.min_obs {
        return NotComputed::TooFewObservations { days: f.days, obs: f.obs, min: f.min_obs };
    }
    if f.stocks < 2 {
        return NotComputed::TooFewStocks { stocks: f.stocks };
    }
    NotComputed::NoSummary { run: f.run.to_string(), period_started: f.period_seen }
}

/// A matrix's summary line, and where its records sit in the task's record file.
struct SummaryLine {
    matrix: Matrix,
    /// Index of its first record, and how many it has (`stored_real + stored_null`).
    first: u64,
    count: u64,
}

/// What a task's summary file says about one correlation period and offset.
struct SummaryFile {
    line: Option<SummaryLine>,
    /// It has a line for this correlation period at some offset.
    period_seen: bool,
    /// Records the whole file accounts for.
    records: u64,
}

/// Reads a task's summary file (`t<id>.csv`) by its header, whose column names are
/// `run::SUMMARY_HEADER`'s (older runs lack its last column), for the line of correlation period
/// `window` at offset `lag`. An empty file (no unit committed yet) has no lines, and an
/// unterminated last line (a census appending to it right now) is not read.
fn parse_summary(text: &str, window: usize, h1: usize, h2: usize, lag: usize) -> Result<SummaryFile> {
    let mut out = SummaryFile { line: None, period_seen: false, records: 0 };
    let complete = text.rfind('\n').map_or("", |i| &text[..=i]);
    let mut lines = complete.lines();
    let Some(header) = lines.next() else { return Ok(out) };
    let header: Vec<&str> = header.split(',').collect();
    let known: Vec<&str> = SUMMARY_HEADER.trim_end().split(',').collect();
    if let Some(bad) = header.iter().find(|c| !known.contains(c)) {
        bail!("unexpected summary column {bad:?}");
    }
    let col = |name: &str| header.iter().position(|c| *c == name).with_context(|| format!("the summary has no {name} column"));
    let (c_window, c_h1, c_h2, c_lag, c_stocks, c_obs) = (col("window")?, col("h1")?, col("h2")?, col("lag")?, col("stocks")?, col("obs")?);
    let c_real = [col("real_z3")?, col("real_z4")?, col("real_z5")?];
    let c_null = [col("null_z3")?, col("null_z4")?, col("null_z5")?];
    let (c_fdr, c_sr, c_sn) = (col("fdr_r")?, col("stored_real")?, col("stored_null")?);
    for (k, l) in lines.enumerate() {
        if l.is_empty() {
            continue;
        }
        let f: Vec<&str> = l.split(',').collect();
        anyhow::ensure!(f.len() == header.len(), "summary line {} has {} fields, its header {}", k + 2, f.len(), header.len());
        let int = |c: usize| f[c].parse::<u64>().with_context(|| format!("summary line {}: {:?} is not a count", k + 2, f[c]));
        let (stored_real, stored_null) = (int(c_sr)?, int(c_sn)?);
        if int(c_window)? == window as u64 {
            out.period_seen = true;
            if int(c_lag)? == lag as u64 && out.line.is_none() {
                let (lh1, lh2) = (int(c_h1)?, int(c_h2)?);
                anyhow::ensure!((lh1, lh2) == (h1 as u64, h2 as u64), "summary line {} holds move lengths ({lh1}, {lh2}), not the run's ({h1}, {h2}) for this file", k + 2);
                let num = |c: usize| f[c].parse::<f64>().unwrap_or(f64::NAN);
                let matrix = Matrix {
                    found: true,
                    stocks: int(c_stocks)? as usize,
                    obs: int(c_obs)? as usize,
                    real: c_real.map(num),
                    null: c_null.map(num),
                    fdr_r: f[c_fdr].parse().ok(),
                    stored_real: stored_real as usize,
                    stored_null: stored_null as usize,
                };
                out.line = Some(SummaryLine { matrix, first: out.records, count: stored_real + stored_null });
            }
        }
        out.records += stored_real + stored_null;
    }
    Ok(out)
}

/// The census's record of the real pair (`leader`, `target`) in the matrix of summary `line`,
/// from the task's record file `file` (`size` bytes). The census writes each summary line's
/// `stored_real + stored_null` records in line order, so when the file holds exactly the
/// `total` records its summary accounts for, only the matrix's block is read; otherwise (a
/// census still writing it, or a file of another layout) the whole file is scanned.
/// `key` = (correlation period, h1, h2, offset).
fn find_record<F: Read + Seek>(file: &mut F, size: u64, total: u64, line: &SummaryLine, key: (usize, usize, usize, usize), leader: u32, target: u32) -> Result<Option<Record>> {
    let ours = |r: &Record| (r.window as usize, r.h1 as usize, r.h2 as usize, r.lag as usize) == key;
    let wanted = |r: &Record| ours(r) && r.leader == leader && r.target == target && r.flags & FLAG_NULL == 0;
    let rb = RECORD_BYTES as u64;
    if size == total * rb {
        let mut block = vec![0u8; (line.count * rb) as usize];
        file.seek(SeekFrom::Start(line.first * rb))?;
        file.read_exact(&mut block)?;
        let records: Vec<Record> = block.chunks_exact(RECORD_BYTES).map(Record::from_bytes).collect();
        if records.iter().all(ours) {
            return Ok(records.into_iter().find(wanted));
        }
    }
    file.seek(SeekFrom::Start(0))?;
    let mut buf = vec![0u8; RECORD_BYTES * 16384];
    let mut fill = 0;
    loop {
        let n = file.read(&mut buf[fill..])?;
        if n == 0 {
            return Ok(None);
        }
        fill += n;
        let whole = fill / RECORD_BYTES * RECORD_BYTES;
        if let Some(r) = buf[..whole].chunks_exact(RECORD_BYTES).map(Record::from_bytes).find(wanted) {
            return Ok(Some(r));
        }
        buf.copy_within(whole..fill, 0);
        fill -= whole;
    }
}

/// Sums of x, y, x^2, y^2 and xy, accumulated in observation order as the census does.
fn obs_sums(pts: &[(f64, f64)]) -> [f64; 5] {
    let mut s = [0.0; 5];
    for &(x, y) in pts {
        s[0] += x;
        s[1] += y;
        s[2] += x * x;
        s[3] += y * y;
        s[4] += x * y;
    }
    s
}

fn pearson(pts: &[(f64, f64)]) -> f64 {
    let s = obs_sums(pts);
    c2_engine::pearson(s[4], s[0], s[1], s[2], s[3], pts.len() as f64)
}

/// The observation count and sums recomputed here (`sums`, over `obs` observations) against a
/// stored record's.
fn sums_check(obs: usize, sums: [f64; 5], rec: &Record) -> SumsCheck {
    let stored = [rec.sx, rec.sy, rec.sxx, rec.syy, rec.sxy].map(|v| v as f64);
    let m = obs as f64;
    // |sum x| <= sqrt(m sum x^2) and |sum xy| <= sqrt(sum x^2 sum y^2): each sum's natural size.
    let scale = [(m * sums[2]).sqrt(), (m * sums[3]).sqrt(), sums[2], sums[3], (sums[2] * sums[3]).sqrt()];
    let max_rel_diff = (0..5)
        .map(|k| {
            let d = (sums[k] - stored[k]).abs();
            if !d.is_finite() || !scale[k].is_finite() {
                f64::INFINITY
            } else if d == 0.0 {
                0.0
            } else {
                d / scale[k].max(f64::MIN_POSITIVE)
            }
        })
        .fold(0.0, f64::max);
    SumsCheck { obs, stored_obs: rec.obs as usize, sums, stored, max_rel_diff }
}

fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.265_512_23 + t * (1.000_023_68 + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06
        + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))))).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

/// `explain_with` for the run in `dir` (reads its `RunMeta`, about 10 ms).
pub fn explain(store: &c2_data::Store, plan: &Plan, dir: &Path, req: &Request) -> Result<Explained> {
    let meta = RunMeta::load(dir)?;
    explain_with(store, plan, &meta, req)
}

/// Explains one relationship of the run `meta` describes. `plan` is that run's plan
/// (`Plan::for_run`); `req.window` is one of the run's correlation period ids. Errors if the
/// plan was prepared for another run or its period of that id is not the run's (by date), if
/// today's data doesn't cover the period (saying why, `period_unavailable`), if a stock of the
/// period is no longer in today's data (`Plan::missing_stocks`) or is unknown, if the offset
/// lets B's move start before A's ends (0 < offset < h2), or if there is room for fewer than 3
/// observations. Otherwise it first decides whether the census computed the matrix
/// (`census_computed`), then explains the pair either way.
pub fn explain_with(store: &c2_data::Store, plan: &Plan, meta: &RunMeta, req: &Request) -> Result<Explained> {
    let (h1, h2, lag) = (req.h1, req.h2, req.lag);
    anyhow::ensure!(h1 >= 1 && h2 >= 1, "move lengths start at 1 day");
    if lag > 0 && lag < h2 {
        bail!(
            "an offset of {} would let B's move over {} start before A's ends, so the census never tests it; the nearest offsets it tests are 0 (the same-day reference) and {}",
            days(lag),
            days(h2),
            days(h2)
        );
    }
    let run = meta.name.as_str();
    if let Some(other) = plan.built_for()
        && !plan.is_for(meta)
    {
        bail!("this calculation was prepared for census {other}, not for census {run}; prepare it from the run itself");
    }
    let rw = meta.windows.get(req.window).with_context(|| format!("census {run} has no correlation period number {}", req.window))?;
    // A period today's data doesn't cover exactly can't be recomputed (`Plan::for_run` marks it).
    if let Some(why) = period_unavailable(plan, meta, req.window) {
        bail!("census {run}'s {} can't be explained with today's data: {why}", describe_period(rw));
    }
    // `req.window` is the run's id: the plan's period of that id must be the run's, by date.
    let rows = meta.window_rows_in(&plan.var.days, req.window).context("the correlation period is not covered")?;
    let w = plan
        .windows
        .get(req.window)
        .filter(|w| w.length == rw.length && rows == (w.start..w.end))
        .ok_or_else(|| anyhow!("the census data prepared for the Explainer doesn't match census {run}: its correlation period {} is not the run's {}", req.window, describe_period(rw)))?;
    // A stock the period had that today's data no longer has (delisted or dropped since) can't be
    // recomputed; say so rather than calling it unknown.
    let stocks = meta.stocks()?;
    let col = |ticker: &str| -> Result<usize> {
        if let Some(rid) = stocks.id(ticker)
            && plan.missing_stocks.get(req.window).is_some_and(|m| m.contains(&rid))
        {
            bail!(
                "{ticker} was one of census {run}'s stocks in its {}, but today's data no longer has it as a census stock (delisted or dropped since), so this pair can't be recomputed",
                describe_period(rw)
            );
        }
        column(plan, store, ticker)
    };
    let (ja, jb) = (col(&req.leader)?, col(&req.follower)?);
    anyhow::ensure!(ja != jb, "pick two different stocks");
    let (step, g0) = (h1.max(h2), h1.max(h2) - 1);
    let len = w.end - w.start;
    let m = grid::observations(len, h1, h2, lag);

    // Did the census compute this matrix? The task's summary file records what it computed; the
    // run's own rules say why not when it has no line for it.
    let universe_ids = &meta.universes[req.window];
    let summary = match meta.find_task(h1, h2, rw.length)? {
        Some(t) => {
            let path = meta.task_path(t.id, "csv");
            let text = std::fs::read_to_string(&path).with_context(|| format!("census {run}: read {}", path.display()))?;
            Some((t.id, parse_summary(&text, req.window, h1, h2, lag).with_context(|| format!("census {run}: {}", path.display()))?))
        }
        None => None,
    };
    let census_computed = match &summary {
        Some((_, s)) if s.line.is_some() => Ok(()),
        _ => Err(why_not(&Facts {
            run,
            horizons: &meta.cfg.horizons,
            min_obs: meta.cfg.min_obs,
            shorter_only: meta.tasks()?.iter().all(|t| t.h2 <= t.h1),
            task: summary.is_some(),
            period_seen: summary.as_ref().is_some_and(|(_, s)| s.period_seen),
            h1,
            h2,
            lag,
            length: rw.length,
            days: len,
            obs: m,
            stocks: universe_ids.len(),
        })),
    };
    if m < 3 {
        bail!("a {len}-day correlation period has room for only {m} observations at this offset, too few to correlate (the census needs at least {}, so it never computed this)", meta.cfg.min_obs);
    }

    // Records name stocks by the run's ids. The plan must number them as the run does
    // (`Plan::for_run`); one built for today's store differs after a re-ingest.
    let (ida, idb) = (plan.run_id_of_column(ja), plan.run_id_of_column(jb));
    for (ticker, id) in [(&req.leader, ida), (&req.follower, idb)] {
        if let (Some(want), Some(have)) = (stocks.id(ticker), id)
            && want != have
        {
            bail!("the census data prepared for the Explainer numbers {ticker} differently from census {run} (stock {have}, the run's {want}); rebuild it from the run");
        }
    }
    // Shown rows: the day before the period (the first move starts there) through its end.
    let row0 = w.start.checked_sub(1).context("the correlation period starts on the first day of data")?;
    let shown = |mut s: DaySteps, ticker: &str, id: Option<u32>| {
        s.ticker = ticker.to_string();
        s.in_universe = id.is_some_and(|id| universe_ids.contains(&id));
        for vec in [&mut s.beta, &mut s.resid, &mut s.mean, &mut s.sd] {
            *vec = vec[row0..w.end].to_vec();
        }
        for vec in [&mut s.close, &mut s.ret, &mut s.v, &mut s.v_census] {
            *vec = vec[row0..w.end].to_vec();
        }
        s.max_diff = s.v.iter().zip(&s.v_census).filter(|(a, b)| a.is_finite() || b.is_finite()).map(|(a, b)| if a.is_finite() && b.is_finite() { (a - b).abs() } else { f32::INFINITY }).fold(0.0, f32::max);
        s
    };
    let a = shown(day_steps(plan, ja), &req.leader, ida);
    let b = shown(day_steps(plan, jb), &req.follower, idb);
    // Observations exactly as the census forms them, from the values recomputed here: missing
    // values count as 0 in the sums.
    let sum = |v: &[f32], end: usize, h: usize| -> f64 { v[end + 1 - h..=end].iter().map(|&x| if x.is_finite() { x as f64 } else { 0.0 }).sum::<f64>() / (h as f64).sqrt() };
    let lnc = |c: &[f32], k: usize| (c[k] as f64).ln();
    let obs: Vec<Obs> = (0..m)
        .map(|q| {
            // In shown rows (offset 1 from the period's rows).
            let e = 1 + g0 + q * step;
            Obs {
                a_end: e,
                b_end: e + lag,
                a_move: lnc(&a.close, e) - lnc(&a.close, e - h1),
                b_move: lnc(&b.close, e + lag) - lnc(&b.close, e + lag - h2),
                x: sum(&a.v, e, h1),
                y: sum(&b.v, e + lag, h2),
            }
        })
        .collect();
    let xy: Vec<(f64, f64)> = obs.iter().map(|o| (o.x, o.y)).collect();
    let sums = obs_sums(&xy);
    let r = c2_engine::pearson(sums[4], sums[0], sums[1], sums[2], sums[3], m as f64);
    let z = r.clamp(-0.999_999, 0.999_999).atanh() * ((m as f64 - 3.0).max(1.0)).sqrt();
    let raw: Vec<(f64, f64)> = obs.iter().map(|o| (o.a_move, o.b_move)).filter(|p| p.0.is_finite() && p.1.is_finite()).collect();
    let r_raw = pearson(&raw);

    // The stored record, matched by the run's ids for the two stocks, from the matrix's block.
    let (matrix, stored) = match &summary {
        Some((tid, s)) => match &s.line {
            Some(line) => {
                let record = match (ida, idb) {
                    (Some(ia), Some(ib)) => {
                        let path = meta.task_path(*tid, "rec");
                        let mut file = std::fs::File::open(&path).with_context(|| format!("census {run}: read {}", path.display()))?;
                        let size = file.metadata()?.len();
                        find_record(&mut file, size, s.records, line, (req.window, h1, h2, lag), ia, ib).with_context(|| format!("census {run}: read {}", path.display()))?
                    }
                    _ => None,
                };
                (line.matrix.clone(), record.map(|record| Stored { top: record.flags & FLAG_TOP != 0, significant: record.flags & FLAG_FDR != 0, record }))
            }
            None => (Matrix::default(), None),
        },
        None => (Matrix::default(), None),
    };
    let check = match (&stored, census_computed.is_ok()) {
        (Some(s), _) => Check::Record(sums_check(m, sums, &s.record)),
        (None, true) => Check::Summary(SummaryCheck { stocks: universe_ids.len(), stored_stocks: matrix.stocks, obs: m, stored_obs: matrix.obs }),
        (None, false) => Check::Nothing,
    };

    // Size in price terms from the market-relative returns (`resid`), on the period's rows
    // (shown rows 1..=len), with the stored r when there is one: then exactly the Forecast's
    // multiplier and the timeline's "% per 1%" (same series, rows, grid and r).
    let move_size_r = stored.as_ref().map_or(r, |s| s.record.r as f64);
    let move_size = moves::move_size(&a.resid, &b.resid, move_size_r, h1, h2, lag, 1..1 + len);
    let move_per_pct = move_size.map_or(f64::NAN, |s| s.per_pct);
    // The Forecast's own filters (forecast::prepare).
    let forecast_use = if lag == 0 {
        Err("offset 0 is the same-day reference, not a prediction, so the Forecast doesn't use it".to_string())
    } else if rw.length == 0 {
        Err("the Forecast doesn't use the full-history correlation period (its track record could never check it)".to_string())
    } else if stored.is_none() {
        Err("the census didn't store this pair, and the Forecast uses only the relationships it stored".to_string())
    } else if move_size.is_none() {
        Err(format!("fewer than {} observations have both moves, so it has no size", moves::MIN_PAIRS))
    } else {
        Ok(())
    };

    let universe = universe_ids.len();
    let pairs_tested = (universe * universe.saturating_sub(1)) as f64;
    let chance_analytic = pairs_tested * erfc(z.abs() / std::f64::consts::SQRT_2);

    // Plain-terms counts on log closes over all variable rows.
    let all = |j: usize| -> Vec<f64> { plan.var.col(&plan.var.close, j).iter().map(|&c| if c > 0.0 { (c as f64).ln() } else { f64::NAN }).collect() };
    let lo = plan.var.days.partition_point(|&d| d < meta.cfg.start);
    let words = words(&all(ja), &all(jb), h1, h2, lag, w.start + g0, m, step, w.start, w.end, lo, plan.var.t, req.threshold);

    Ok(Explained {
        h1,
        h2,
        lag,
        same_day: lag == 0,
        window: req.window,
        period_first: plan.var.days[w.start],
        period_last: plan.var.days[w.end - 1],
        period_days: len,
        universe,
        days: plan.var.days[row0..w.end].to_vec(),
        market: plan.var.market[row0..w.end].to_vec(),
        a,
        b,
        obs,
        n: m,
        r,
        z,
        r_raw,
        census_computed,
        move_per_pct,
        move_size,
        move_size_r,
        forecast_use,
        stored,
        matrix,
        check,
        variable_note: plan.variable_note.clone(),
        pairs_tested,
        chance_analytic,
        words,
    })
}

/// Why today's data can't reproduce correlation period `id` of the run, or None when it covers
/// it exactly: its first and last days are trading days of the census variable with exactly its
/// length of rows between them (`Plan::for_run` marks the others unavailable).
pub fn period_unavailable(plan: &Plan, meta: &RunMeta, id: usize) -> Option<String> {
    let Some(w) = meta.windows.get(id) else { return Some(format!("census {} has no correlation period number {id}", meta.name)) };
    if plan.period_available(id) && meta.window_rows_in(&plan.var.days, id).is_some() {
        return None;
    }
    Some(why_uncovered(&plan.var.days, w))
}

/// Why the trading days `days` (a census variable's) don't cover period `w` exactly.
fn why_uncovered(days: &[i32], w: &RunWindow) -> String {
    let d = c2_data::date_string;
    let (first, last) = (days.first().copied().unwrap_or(i32::MAX), days.last().copied().unwrap_or(i32::MIN));
    if w.last > last {
        format!("it ends on {}, after today's data, which ends on {}", d(w.last), d(last))
    } else if w.first < first {
        format!("it starts on {}, before today's census data, which starts on {}", d(w.first), d(first))
    } else if days.binary_search(&w.first).is_err() {
        format!("its first day, {}, is not a trading day in today's data", d(w.first))
    } else if days.binary_search(&w.last).is_err() {
        format!("its last day, {}, is not a trading day in today's data", d(w.last))
    } else {
        let n = days.partition_point(|&x| x <= w.last) - days.partition_point(|&x| x < w.first);
        format!("today's data has {n} trading days from {} to {}, not the {} the census used", d(w.first), d(w.last), w.length)
    }
}

/// Which period of one length a request means. `of_len` holds that length's periods in date
/// order; `covered` says which ones today's data covers.
#[derive(Debug, PartialEq)]
enum Pick {
    /// The period asked for (by `end`, or the latest), covered.
    Covered(usize),
    /// With no `end`: the latest period isn't covered (`want`), the latest covered one is `chosen`.
    PassedOver { want: usize, chosen: usize },
    /// The period asked for isn't covered; `latest_covered` of that length, if any.
    Uncovered { want: usize, latest_covered: Option<usize> },
}

/// The period of `of_len` ending on or before `end` (else the earliest), or with no `end` the
/// latest one, falling back to the latest covered one only when no `end` was given. Indexes
/// into `of_len`; `of_len` must not be empty.
fn pick(of_len: &[&RunWindow], covered: impl Fn(&RunWindow) -> bool, end: Option<i32>) -> Pick {
    let latest_covered = (0..of_len.len()).rev().find(|&k| covered(of_len[k]));
    let want = match end {
        Some(end) => (0..of_len.len()).rev().find(|&k| of_len[k].last <= end).unwrap_or(0),
        None => of_len.len() - 1,
    };
    match (covered(of_len[want]), end, latest_covered) {
        (true, _, _) => Pick::Covered(want),
        (false, None, Some(chosen)) => Pick::PassedOver { want, chosen },
        (false, _, latest_covered) => Pick::Uncovered { want, latest_covered },
    }
}

/// The correlation period of `length` trading days (0 = full history) ending on or before `end`,
/// else the earliest one of that length, among those today's data covers
/// (`Plan::period_available`). None if it covers none of that length.
pub fn find_window(plan: &Plan, length: usize, end: i32) -> Option<usize> {
    let of_len: Vec<usize> = plan.windows.iter().filter(|w| w.length == length && plan.period_available(w.id)).map(|w| w.id).collect();
    of_len.iter().rev().find(|&&id| plan.var.days[plan.windows[id].end - 1] <= end).or(of_len.first()).copied()
}

/// The correlation period a request names, by the run's own dates: of `length` trading days
/// (0 = full history), the one ending on or before `end` (else the earliest of that length), or
/// with no `end` the latest one today's data covers. Errors name the run's lengths when it has no
/// period of `length`, and name the period and why when today's data doesn't cover the one asked
/// for. The note is set when, with no `end`, later periods of that length had to be passed over.
pub fn find_period(plan: &Plan, meta: &RunMeta, length: usize, end: Option<i32>) -> Result<(usize, Option<String>)> {
    let run = &meta.name;
    let d = c2_data::date_string;
    let of_len: Vec<&RunWindow> = meta.windows.iter().filter(|w| w.length == length).collect();
    if of_len.is_empty() {
        let mut have: Vec<usize> = meta.windows.iter().map(|w| w.length).collect();
        have.sort_by_key(|&l| if l == 0 { usize::MAX } else { l });
        have.dedup();
        let names: Vec<String> = have.iter().map(|&l| if l == 0 { "the full history (0)".to_string() } else { format!("{l} days") }).collect();
        let what = if length == 0 { "full-history correlation period".to_string() } else { format!("{length}-day correlation periods") };
        bail!(if names.is_empty() { format!("census {run} has no correlation periods") } else { format!("census {run} has no {what}; its correlation periods are {}", list_and(&names)) });
    }
    let covered = |w: &RunWindow| period_unavailable(plan, meta, w.id).is_none();
    let why = |w: &RunWindow| period_unavailable(plan, meta, w.id).unwrap_or_default();
    match pick(&of_len, covered, end) {
        Pick::Covered(k) => Ok((of_len[k].id, None)),
        Pick::PassedOver { want, chosen } => {
            let (want, chosen) = (of_len[want], of_len[chosen]);
            let note = format!("census {run}'s latest {} can't be explained with today's data: {}. Showing the latest one it covers, {}..{}.", describe_period(want), why(want), d(chosen.first), d(chosen.last));
            Ok((chosen.id, Some(note)))
        }
        Pick::Uncovered { want, latest_covered } => {
            let want = of_len[want];
            let hint = match latest_covered {
                Some(k) => format!("; the latest {length}-day one today's data covers ends on {}", d(of_len[k].last)),
                None => match meta.windows.iter().filter(|w| covered(w)).map(|w| w.last).max() {
                    Some(last) => format!("; the correlation periods today's data covers end on or before {}", d(last)),
                    None => "; today's data covers none of its correlation periods".to_string(),
                },
            };
            bail!("census {run}'s {} can't be explained with today's data: {}{hint}", describe_period(want), why(want))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rw(id: usize, length: usize, first: &str, last: &str) -> RunWindow {
        RunWindow { id, length, first: c2_data::parse_date(first).unwrap(), last: c2_data::parse_date(last).unwrap(), stocks: 10 }
    }

    #[test]
    fn uncovered_periods_say_why() {
        // Trading days 2025-01-02 .. 2025-01-10 without 2025-01-08.
        let days: Vec<i32> = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-09", "2025-01-10"].iter().map(|d| c2_data::parse_date(d).unwrap()).collect();
        let why = |w: RunWindow| why_uncovered(&days, &w);
        assert_eq!(why(rw(0, 3, "2025-01-09", "2025-01-14")), "it ends on 2025-01-14, after today's data, which ends on 2025-01-10");
        assert_eq!(why(rw(0, 3, "2024-12-30", "2025-01-03")), "it starts on 2024-12-30, before today's census data, which starts on 2025-01-02");
        assert!(why(rw(0, 3, "2025-01-08", "2025-01-10")).contains("first day, 2025-01-08, is not a trading day"));
        assert!(why(rw(0, 3, "2025-01-06", "2025-01-08")).contains("last day, 2025-01-08, is not a trading day"));
        // Both days are trading days, but a day in between is missing today (the run had 5 rows).
        assert_eq!(why(rw(0, 5, "2025-01-06", "2025-01-10")), "today's data has 4 trading days from 2025-01-06 to 2025-01-10, not the 5 the census used");
    }

    #[test]
    fn a_request_picks_its_period_and_falls_back_only_without_an_end() {
        let ws = [rw(3, 252, "2023-01-03", "2023-12-29"), rw(7, 252, "2023-04-03", "2024-03-28"), rw(9, 252, "2023-07-03", "2024-06-28")];
        let of_len: Vec<&RunWindow> = ws.iter().collect();
        let date = |d: &str| Some(c2_data::parse_date(d).unwrap());
        let all = |_: &RunWindow| true;
        let not_last = |w: &RunWindow| w.id != 9;
        let none = |_: &RunWindow| false;
        // Covered: the latest, the one ending by --end, the earliest if --end is before them all.
        assert_eq!(pick(&of_len, all, None), Pick::Covered(2));
        assert_eq!(pick(&of_len, all, date("2024-05-01")), Pick::Covered(1));
        assert_eq!(pick(&of_len, all, date("2020-01-01")), Pick::Covered(0));
        // The latest isn't covered: with no --end, the latest covered one; with --end, refused.
        assert_eq!(pick(&of_len, not_last, None), Pick::PassedOver { want: 2, chosen: 1 });
        assert_eq!(pick(&of_len, not_last, date("2024-06-28")), Pick::Uncovered { want: 2, latest_covered: Some(1) });
        assert_eq!(pick(&of_len, not_last, date("2024-06-27")), Pick::Covered(1));
        assert_eq!(pick(&of_len, none, None), Pick::Uncovered { want: 2, latest_covered: None });
    }

    #[test]
    fn nearest_offsets_bracket_the_request() {
        assert_eq!(nearest_offsets(5, 7), vec![5, 10]);
        assert_eq!(nearest_offsets(5, 10), vec![10]);
        assert_eq!(nearest_offsets(5, 3), vec![0, 5]);
        assert_eq!(nearest_offsets(5, 130), vec![125]);
        assert_eq!(nearest_offsets(21, 30), vec![21, 42]);
        assert_eq!(nearest_offsets(1, 13), vec![13]);
        assert_eq!(nearest_offsets(1, 0), vec![0]);
    }

    fn facts<'a>(horizons: &'a [usize]) -> Facts<'a> {
        Facts { run: "r", horizons, min_obs: 30, shorter_only: false, task: true, period_seen: true, h1: 21, h2: 21, lag: 21, length: 252, days: 252, obs: 11, stocks: 1773 }
    }

    #[test]
    fn why_not_gives_the_first_rule_that_fails_in_the_user_s_words() {
        let hs = [1, 3, 5, 10, 21, 63];
        // 21-day moves at offset 21 in a 252-day period: 11 observations (V14's example).
        assert_eq!(why_not(&facts(&hs)), NotComputed::TooFewObservations { days: 252, obs: 11, min: 30 });
        let off = why_not(&Facts { h1: 5, h2: 5, lag: 7, obs: 49, ..facts(&hs) });
        assert_eq!(off, NotComputed::OffGrid { lag: 7, h2: 5, nearest: vec![5, 10] });
        let none = why_not(&Facts { task: false, h1: 3, h2: 1, ..facts(&[1, 5, 10, 21, 63]) });
        assert!(matches!(none, NotComputed::NoTask { .. }));
        assert_eq!(why_not(&Facts { obs: 239, stocks: 1, ..facts(&hs) }), NotComputed::TooFewStocks { stocks: 1 });
        assert_eq!(why_not(&Facts { obs: 239, period_seen: false, ..facts(&hs) }), NotComputed::NoSummary { run: "r".into(), period_started: false });
        let older = why_not(&Facts { task: false, shorter_only: true, h1: 1, h2: 5, ..facts(&[1, 5, 21]) });
        let texts = [
            why_not(&facts(&hs)).to_string(),
            off.to_string(),
            none.to_string(),
            older.to_string(),
            NotComputed::TooFewStocks { stocks: 1 }.to_string(),
            NotComputed::NoSummary { run: "r".into(), period_started: false }.to_string(),
            NotComputed::NoSummary { run: "r".into(), period_started: true }.to_string(),
        ];
        assert!(texts[0].contains("only 11 observations") && texts[0].contains("at least 30"), "{}", texts[0]);
        assert!(texts[1].contains("offset of 7 days") && texts[1].contains("5 and 10 days"), "{}", texts[1]);
        assert!(texts[2].contains("its move lengths are 1, 5, 10, 21 and 63 days"), "{}", texts[2]);
        assert!(texts[3].contains("only with A moves at least as long"), "{}", texts[3]);
        // The user's vocabulary: no "window", "horizon" or "lag" in what is shown.
        for t in &texts {
            for word in ["window", "horizon", "lag"] {
                assert!(!t.to_lowercase().contains(word), "{t}");
            }
        }
    }

    fn line(window: usize, lag: usize, stocks: usize, obs: usize, real: usize, null: usize, placebo: bool) -> String {
        format!("{window},1,1,{lag},{stocks},{obs},5,1,0,3.0,0.3,0.0,{},{real},{null},40,0{}\n", if lag > 0 { "0.4385" } else { "" }, if placebo { "," } else { "" })
    }

    #[test]
    fn summary_is_read_by_header_and_locates_the_matrix_records() {
        // Today's 18 columns, and an older run's 17 (no placebo_fdr_r).
        for placebo in [true, false] {
            let header = if placebo { SUMMARY_HEADER.to_string() } else { SUMMARY_HEADER.replace(",placebo_fdr_r", "") };
            let text = format!("{header}{}{}{}{}", line(4, 0, 1773, 252, 100, 50, placebo), line(4, 13, 1773, 239, 120, 50, placebo), line(5, 0, 1800, 252, 100, 50, placebo), line(5, 13, 1800, 239, 100, 50, placebo));
            let s = parse_summary(&text, 4, 1, 1, 13).unwrap();
            let l = s.line.as_ref().unwrap();
            assert!(l.matrix.found && s.period_seen);
            assert_eq!((l.matrix.stocks, l.matrix.obs, l.matrix.stored_real, l.matrix.stored_null), (1773, 239, 120, 50));
            assert_eq!(l.matrix.fdr_r, Some(0.4385));
            assert_eq!((l.first, l.count, s.records), (150, 170, 150 + 170 + 150 + 150));
            // Offset 0: no significance test, so no cut-off.
            assert_eq!(parse_summary(&text, 5, 1, 1, 0).unwrap().line.unwrap().matrix.fdr_r, None);
            // A period it has at other offsets only, and one it doesn't have.
            let s = parse_summary(&text, 5, 1, 1, 7).unwrap();
            assert!(s.line.is_none() && s.period_seen);
            assert!(!parse_summary(&text, 6, 1, 1, 13).unwrap().period_seen);
            // A line the census is appending right now is not read yet.
            let s = parse_summary(&format!("{text}6,1,1,13,18"), 6, 1, 1, 13).unwrap();
            assert!(s.line.is_none() && !s.period_seen && s.records == 620);
        }
        // By name, not by position: swap two columns.
        let swapped = SUMMARY_HEADER.replace("stocks,obs", "obs,stocks");
        let text = format!("{swapped}4,1,1,13,239,1773,5,1,0,3.0,0.3,0.0,0.4385,120,50,40,0,\n");
        let l = parse_summary(&text, 4, 1, 1, 13).unwrap().line.unwrap();
        assert_eq!((l.matrix.stocks, l.matrix.obs), (1773, 239));
        // Unknown columns, a line that doesn't fit, move lengths other than the task's.
        assert!(parse_summary("window,h1,h2,lag,rows\n", 4, 1, 1, 13).is_err());
        assert!(parse_summary(&format!("{SUMMARY_HEADER}4,1,1,13\n"), 4, 1, 1, 13).is_err());
        assert!(parse_summary(&format!("{SUMMARY_HEADER}{}", line(4, 13, 10, 239, 1, 1, true)), 4, 5, 1, 13).is_err());
        // Nothing committed yet.
        let s = parse_summary("", 4, 1, 1, 13).unwrap();
        assert!(s.line.is_none() && !s.period_seen && s.records == 0);
    }

    fn rec(window: u32, lag: u16, leader: u32, target: u32, flags: u8) -> Record {
        Record { window, h1: 1, h2: 1, lag, flags, obs: 239, leader, target, r: 0.5, ..Default::default() }
    }

    #[test]
    fn record_is_read_from_the_matrix_block_else_found_by_scanning() {
        // Two matrices: (4, offset 0) with 2 records, (4, offset 13) with 3 (the last a null).
        let records = [rec(4, 0, 7, 9, FLAG_TOP), rec(4, 0, 1, 2, FLAG_TOP), rec(4, 13, 1, 2, FLAG_TOP), rec(4, 13, 7, 9, FLAG_TOP | FLAG_FDR), rec(4, 13, 3, 5, FLAG_NULL | FLAG_TOP)];
        let mut bytes = Vec::new();
        for r in &records {
            r.to_bytes(&mut bytes);
        }
        let matrix = |first, count| SummaryLine { matrix: Matrix::default(), first, count };
        let find = |bytes: &[u8], total: u64, leader: u32, target: u32| {
            let mut f = std::io::Cursor::new(bytes.to_vec());
            find_record(&mut f, bytes.len() as u64, total, &matrix(2, 3), (4, 1, 1, 13), leader, target).unwrap()
        };
        let r = find(&bytes, 5, 7, 9).unwrap();
        assert_eq!((r.lag, r.flags), (13, FLAG_TOP | FLAG_FDR));
        // Not in the matrix (it is at offset 0), and only as a null.
        assert!(find(&bytes, 5, 9, 7).is_none());
        assert!(find(&bytes, 5, 3, 5).is_none());
        // A file longer than its summary says (a census writing it): scanned whole.
        let mut longer = bytes.clone();
        rec(8, 1, 0, 0, 0).to_bytes(&mut longer);
        assert_eq!(find(&longer, 5, 7, 9).unwrap().lag, 13);
        // A block that isn't this matrix's (another layout): scanned whole.
        let mut wrong = Vec::new();
        for r in records.iter().rev() {
            r.to_bytes(&mut wrong);
        }
        assert_eq!(find(&wrong, 5, 7, 9).unwrap().lag, 13);
    }

    #[test]
    fn stored_sums_are_compared_relative_to_their_natural_size() {
        // Sum of x is nearly 0 relative to its terms: a plain relative difference would blow up.
        let pts: Vec<(f64, f64)> = (0..239).map(|q| { let x = ((q * 7919 % 101) as f64 - 50.0) / 29.0; (x, 0.3 * x + ((q * 31 % 17) as f64 - 8.0) / 9.0) }).collect();
        let mut s = obs_sums(&pts);
        s[0] = 1e-9;
        let as_stored = |s: [f64; 5]| Record { obs: 239, sx: s[0] as f32 + 3e-9, sy: s[1] as f32, sxx: s[2] as f32, syy: s[3] as f32, sxy: s[4] as f32, ..Default::default() };
        let c = sums_check(239, s, &as_stored(s));
        assert!(c.agrees() && c.max_rel_diff < 1e-7, "{c:?}");
        // A different observation count, or a sum off by 1e-4 of its size, does not agree.
        assert!(!sums_check(238, s, &as_stored(s)).agrees());
        let mut off = s;
        off[4] += 1e-4 * (s[2] * s[3]).sqrt();
        assert!(!sums_check(239, off, &as_stored(s)).agrees());
        let mut bad = s;
        bad[1] = f64::NAN;
        assert_eq!(sums_check(239, bad, &as_stored(s)).max_rel_diff, f64::INFINITY);
    }
}
