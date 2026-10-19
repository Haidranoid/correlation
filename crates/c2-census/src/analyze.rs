//! Reading census output, and the replication test: do stored pairs keep their sign in the next
//! window that does not overlap?
//!
//! Every reader takes what a run's files mean from the run itself (`meta::RunMeta`): its task
//! list (what each `tasks/t<id>` file holds), its correlation periods (mapped onto the census
//! variable's rows by date, never clamped: a period today's data doesn't cover exactly is left
//! out) and its stock list (records hold store ids as of the run; they are translated through
//! tickers, never used as today's store ids).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use c2_data::Store;
use rayon::prelude::*;

use crate::grid::{self, Config, Task, Window};
use crate::meta::{self, RunMeta};
use crate::run::{FLAG_FDR, FLAG_NULL, FLAG_TOP, Plan, RECORD_BYTES, Record};

/// All records of one task file.
pub fn read_records(path: &Path) -> Result<Vec<Record>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(bytes.chunks_exact(RECORD_BYTES).map(Record::from_bytes).collect())
}

/// Calls `f` on every record of a task file, in file order, reading the file in pieces so that
/// memory stays small however large it is. A trailing partial record (a write cut off by a
/// crash) is ignored, as `read_records` does.
pub(crate) fn visit_records(path: &Path, mut f: impl FnMut(&Record)) -> Result<()> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let mut buf = vec![0u8; RECORD_BYTES * 16384];
    let mut fill = 0;
    loop {
        let n = file.read(&mut buf[fill..]).with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            return Ok(());
        }
        fill += n;
        let whole = fill / RECORD_BYTES * RECORD_BYTES;
        for chunk in buf[..whole].chunks_exact(RECORD_BYTES) {
            f(&Record::from_bytes(chunk));
        }
        buf.copy_within(whole..fill, 0);
        fill -= whole;
    }
}

/// `visit_records` on the run's results file for `task`, checking that every record belongs to
/// it: the task's move lengths and one of its correlation periods. Errors otherwise, since the
/// run's files then don't hold what its task list says and nothing read from them can be trusted.
pub(crate) fn visit_task(meta: &RunMeta, task: &Task, mut f: impl FnMut(&Record)) -> Result<()> {
    let mut ours = vec![false; meta.windows.len()];
    for &w in &task.windows {
        ours[w] = true;
    }
    let mut stray: Option<Record> = None;
    visit_records(&meta.task_path(task.id, "rec"), |r| {
        if (r.h1 as usize, r.h2 as usize) == (task.h1, task.h2) && ours.get(r.window as usize) == Some(&true) {
            f(r);
        } else if stray.is_none() {
            stray = Some(*r);
        }
    })?;
    match stray {
        None => Ok(()),
        Some(r) => bail!(
            "census {}: results file t{} should hold {}, but it has a record for A over {} -> B over {} days in correlation period {}; the run's files don't match its task list",
            meta.name, task.id, meta::describe_task(task.h1, task.h2, task.length), r.h1, r.h2, r.window
        ),
    }
}

/// One line of a task's summary (`tasks/t<id>.csv`, see `run::SUMMARY_HEADER`): one matrix,
/// i.e. one correlation period and offset.
#[derive(Clone, Debug, Default)]
pub struct SummaryLine {
    pub window: usize,
    pub h1: usize,
    pub h2: usize,
    pub lag: usize,
    /// Stocks in the matrix, and observations per stock.
    pub stocks: usize,
    pub obs: usize,
    /// Real pairs at |z| >= 3, 4, 5, and the mean count over the sign-flip nulls.
    pub real: [f64; 3],
    pub null: [f64; 3],
    /// The false-discovery cut-off |r|; None when nothing passed.
    pub fdr_r: Option<f64>,
    pub stored_real: usize,
    pub stored_null: usize,
    /// The same test run on a sign-flip null matrix: its cut-off when it fired. None when it
    /// didn't fire, or when the run didn't record it (`Summary::placebo_recorded`).
    pub placebo_fdr_r: Option<f64>,
}

/// A task's summary lines.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub lines: Vec<SummaryLine>,
    /// False when the file has no `placebo_fdr_r` column (runs made before it existed: their
    /// summaries have 17 columns), or no header at all.
    pub placebo_recorded: bool,
}

/// Reads a task's summary by the names in its header, so every layout a run has used reads
/// correctly. A file with no lines (or no header) has no matrices. An incomplete last line (a
/// write cut off by a crash; committed lines always end in a newline) is ignored; any other
/// malformed line is an error.
pub fn read_summary(path: &Path) -> Result<Summary> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    parse_summary(&text, &path.display().to_string())
}

/// `read_summary` of a summary's text; `what` names it in errors.
fn parse_summary(text: &str, what: &str) -> Result<Summary> {
    let complete = match text.rfind('\n') {
        Some(k) => &text[..=k],
        None => "",
    };
    let mut lines = complete.lines();
    let Some(header) = lines.next() else { return Ok(Summary::default()) };
    let names: Vec<&str> = header.split(',').collect();
    let col = |n: &str| names.iter().position(|h| *h == n);
    let need = |n: &str| col(n).with_context(|| format!("{what}: its header has no {n} column"));
    let (cw, ch1, ch2, clag, cn, cobs) = (need("window")?, need("h1")?, need("h2")?, need("lag")?, need("stocks")?, need("obs")?);
    let real = [need("real_z3")?, need("real_z4")?, need("real_z5")?];
    let null = [need("null_z3")?, need("null_z4")?, need("null_z5")?];
    let (cfdr, csr, csn) = (need("fdr_r")?, need("stored_real")?, need("stored_null")?);
    let cplacebo = col("placebo_fdr_r");
    let out = lines
        .enumerate()
        .map(|(k, line)| {
            let f: Vec<&str> = line.split(',').collect();
            let bad = || anyhow!("{what} line {}: {line:?} doesn't fit its header", k + 2);
            let field = |c: usize| f.get(c).copied().ok_or_else(bad);
            let int = |c: usize| field(c)?.parse::<usize>().map_err(|_| bad());
            let num = |c: usize| field(c)?.parse::<f64>().map_err(|_| bad());
            let opt = |c: usize| -> Result<Option<f64>> {
                let v = field(c)?;
                if v.is_empty() { Ok(None) } else { v.parse().map(Some).map_err(|_| bad()) }
            };
            if f.len() != names.len() {
                return Err(bad());
            }
            Ok(SummaryLine {
                window: int(cw)?,
                h1: int(ch1)?,
                h2: int(ch2)?,
                lag: int(clag)?,
                stocks: int(cn)?,
                obs: int(cobs)?,
                real: [num(real[0])?, num(real[1])?, num(real[2])?],
                null: [num(null[0])?, num(null[1])?, num(null[2])?],
                fdr_r: opt(cfdr)?,
                stored_real: int(csr)?,
                stored_null: int(csn)?,
                placebo_fdr_r: match cplacebo {
                    Some(c) => opt(c)?,
                    None => None,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Summary { lines: out, placebo_recorded: cplacebo.is_some() })
}

/// `read_summary` of the run's summary for `task`, checking that every line belongs to it (its
/// move lengths and one of its correlation periods).
pub(crate) fn task_summary(meta: &RunMeta, task: &Task) -> Result<Summary> {
    let s = read_summary(&meta.task_path(task.id, "csv"))?;
    if let Some(l) = s.lines.iter().find(|l| (l.h1, l.h2) != (task.h1, task.h2) || !task.windows.contains(&l.window)) {
        bail!(
            "census {}: results file t{} should hold {}, but its summary has a line for A over {} -> B over {} days in correlation period {}; the run's files don't match its task list",
            meta.name, task.id, meta::describe_task(task.h1, task.h2, task.length), l.h1, l.h2, l.window
        );
    }
    Ok(s)
}

/// Refuses unless `plan` can read the run in `meta`.
///
/// A `Plan::for_run` plan must have been built for this run (`Plan::is_for`); what today's data
/// can't reproduce is marked in it (unavailable periods, missing stocks, a variable note) and
/// readers skip that, so it is accepted as it is. Any other plan (`Plan::new`) is right only
/// while nothing changed: its census variable must start on the run's first day, and its
/// correlation periods must be the run's, id by id, on the variable's rows (found by date).
/// Either way, with `store`, the plan must number the stocks as the run's records do in that
/// store (`Plan::for_run` translates through tickers; `Plan::new` is right only while the store
/// is the one the run was made with).
pub(crate) fn check_plan(plan: &Plan, meta: &RunMeta, store: Option<&Store>) -> Result<()> {
    let name = &meta.name;
    let prepare = "prepare it from the run itself (Plan::for_run)";
    match plan.built_for() {
        Some(other) if !plan.is_for(meta) => bail!("this calculation was prepared for census {other}, not for census {name}; {prepare}"),
        Some(_) => {}
        None => {
            let first = c2_data::date_string(plan.var.days[0]);
            if let Some(theirs) = meta.manifest["variable_days"][0].as_str()
                && theirs != first
            {
                bail!("census {name} was made with a census variable starting {theirs}, this one starts {first}, so its values can't be reproduced; {prepare}");
            }
        }
    }
    if plan.windows.len() != meta.windows.len() {
        bail!("this calculation was prepared for {} correlation periods, census {name} has {}; {prepare}", plan.windows.len(), meta.windows.len());
    }
    for (w, theirs) in plan.windows.iter().zip(&meta.windows) {
        let rows = meta.window_rows_in(&plan.var.days, theirs.id);
        let fits = w.length == theirs.length && if plan.period_available(theirs.id) { rows == Some(w.start..w.end) } else { rows.is_none() };
        if !fits {
            bail!(
                "this calculation was prepared for other correlation periods than census {name}'s (its period {} is {}..{}); {prepare}",
                theirs.id, c2_data::date_string(theirs.first), c2_data::date_string(theirs.last)
            );
        }
    }
    if let Some(store) = store {
        let ids = meta.id_map(store)?;
        if let Some(c) = (0..plan.var.n).find(|&c| plan.run_id_of_column(c) != ids.run(plan.var.ids[c])) {
            bail!(
                "census {name} numbers its stocks differently from this calculation (e.g. {}); the price store was rebuilt since the run, so {prepare}",
                store.symbols[plan.var.ids[c]].ticker
            );
        }
    }
    Ok(())
}

/// Exact r of leader column `i` and target column `j` (variable columns) in a window.
#[allow(clippy::too_many_arguments)]
fn r_in_window(plan: &Plan, min_obs: usize, w: &Window, h1: usize, h2: usize, lag: usize, i: usize, j: usize) -> Option<f64> {
    let m = grid::observations(w.end - w.start, h1, h2, lag);
    if m < min_obs {
        return None;
    }
    let (vi, vj) = (&plan.var.col(&plan.var.v, i)[w.start..w.end], &plan.var.col(&plan.var.v, j)[w.start..w.end]);
    let sum = |v: &[f32], end: usize, h: usize| -> f64 {
        v[end + 1 - h..=end].iter().map(|&x| if x.is_finite() { x as f64 } else { 0.0 }).sum::<f64>() / (h as f64).sqrt()
    };
    let (s, g0) = (h1.max(h2), h1.max(h2) - 1);
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for q in 0..m {
        let e = g0 + q * s;
        let (x, y) = (sum(vi, e, h1), sum(vj, e + lag, h2));
        sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
    }
    Some(c2_engine::pearson(sxy, sx, sy, sxx, syy, m as f64))
}

/// Replication summary for one (h1, h2, lag bucket, real/null) group.
#[derive(Default, Clone)]
pub struct Rep {
    pub pairs: usize,
    pub same_sign: usize,
    /// Sum of sign(r_then) * r_next.
    pub signed_next: f64,
    /// Sum of |r_then|.
    pub abs_then: f64,
}

/// Key: (h1, h2, lag bucket label, is_null).
pub type RepTable = Vec<((usize, usize, &'static str, bool), Rep)>;

fn bucket(lag: usize, h2: usize) -> &'static str {
    match lag / h2.max(1) {
        0 => "0 (same period)",
        1 => "1 step",
        2..=5 => "2-5 steps",
        6..=20 => "6-20 steps",
        _ => "21+ steps",
    }
}

/// For every stored top-K record (real and null) in each rolling window, the exact r of the same
/// pair at the same lag in the next window of that length that starts where this one ends.
/// Null records were selected on sign-flipped data, so their next-window r is the baseline for
/// "selected by chance".
///
/// The tasks (what each results file holds) come from the run in `dir`. `plan` supplies the
/// census variable, the period rows and the stock numbering, and must fit the run (see
/// `check_plan`): `Plan::for_run` does. Its stock numbering can only be checked against a store:
/// `replicate_run` does that too. `replication` also says what was skipped.
pub fn replicate(plan: &Plan, dir: &Path) -> Result<RepTable> {
    Ok(replicate_in(plan, &RunMeta::load(dir)?, None)?.table)
}

/// `replicate`, also checking that `plan` numbers the stocks as the run's records do in `store`
/// (a `Plan::new` plan doesn't once the store has been rebuilt since the run).
pub fn replicate_run(store: &Store, plan: &Plan, dir: &Path) -> Result<RepTable> {
    Ok(replicate_in(plan, &RunMeta::load(dir)?, Some(store))?.table)
}

/// The replication test (`replicate_run`) and what it had to leave out.
#[derive(Clone, Default)]
pub struct Replication {
    pub table: RepTable,
    /// Rolling correlation periods that today's data doesn't cover exactly
    /// (`Plan::period_available`): their records aren't tested, and none of them serves as the
    /// period after another.
    pub periods_unavailable: usize,
    /// Stored records left out because their period, or the period after it, is unavailable.
    pub records_unavailable: usize,
    /// Stored records left out because a stock of the pair is not in today's data
    /// (`Plan::missing_stocks`).
    pub records_missing_stock: usize,
}

/// `replicate_run`, with what it skipped: unavailable correlation periods and stocks today's
/// data no longer has are left out and counted, never refused.
pub fn replication(store: &Store, plan: &Plan, dir: &Path) -> Result<Replication> {
    replicate_in(plan, &RunMeta::load(dir)?, Some(store))
}

/// What became of one stored record in the replication test.
enum Fate {
    Tested((usize, usize, &'static str, bool), f64, f64),
    /// No later period of its length (the end of the history), or too few observations there.
    Untested,
    Unavailable,
    MissingStock,
}

fn replicate_in(plan: &Plan, meta: &RunMeta, store: Option<&Store>) -> Result<Replication> {
    check_plan(plan, meta, store)?;
    let mut out = Replication {
        periods_unavailable: meta.windows.iter().filter(|w| w.length > 0 && !plan.period_available(w.id)).count(),
        ..Default::default()
    };
    let mut table: HashMap<(usize, usize, &'static str, bool), Rep> = HashMap::new();
    for task in meta.tasks()? {
        if task.length == 0 {
            continue;
        }
        let mut records = Vec::new();
        visit_task(meta, task, |r| {
            if r.flags & FLAG_TOP != 0 {
                records.push(*r);
            }
        })?;
        // Next non-overlapping window of the same length: the one whose first day is the trading
        // day after this one's last (its first row is this one's end row). By date, so that
        // unavailable periods (empty rows) can be told apart; only from a covered period.
        let next: HashMap<usize, usize> = task
            .windows
            .iter()
            .filter(|&&a| plan.period_available(a))
            .filter_map(|&a| {
                let day = *plan.var.days.get(plan.windows[a].end)?;
                task.windows.iter().find(|&&b| meta.windows[b].first == day).map(|&b| (a, b))
            })
            .collect();
        let fates: Vec<Fate> = records
            .par_iter()
            .map(|r| {
                let a = r.window as usize;
                if !plan.period_available(a) {
                    return Fate::Unavailable;
                }
                let Some(&nw) = next.get(&a) else { return Fate::Untested };
                if !plan.period_available(nw) {
                    return Fate::Unavailable;
                }
                let (Some(i), Some(j)) = (plan.column_of_run_id(r.leader), plan.column_of_run_id(r.target)) else { return Fate::MissingStock };
                match r_in_window(plan, meta.cfg.min_obs, &plan.windows[nw], task.h1, task.h2, r.lag as usize, i, j) {
                    Some(rn) => Fate::Tested((task.h1, task.h2, bucket(r.lag as usize, task.h2), r.flags & FLAG_NULL != 0), r.r as f64, rn),
                    None => Fate::Untested,
                }
            })
            .collect();
        for fate in fates {
            match fate {
                Fate::Tested(key, then, next) => {
                    let e = table.entry(key).or_default();
                    e.pairs += 1;
                    e.same_sign += usize::from(then.signum() == next.signum());
                    e.signed_next += then.signum() * next;
                    e.abs_then += then.abs();
                }
                Fate::Untested => {}
                Fate::Unavailable => out.records_unavailable += 1,
                Fate::MissingStock => out.records_missing_stock += 1,
            }
        }
    }
    let mut rows: RepTable = table.into_iter().collect();
    rows.sort_by(|a, b| (a.0.0, a.0.1, a.0.2, a.0.3).cmp(&(b.0.0, b.0.1, b.0.2, b.0.3)));
    out.table = rows;
    Ok(out)
}

/// One stored relationship involving a chosen stock, with its size in price terms.
#[derive(Clone)]
pub struct ProfileRow {
    /// "first..last" dates of the census window (the correlation period).
    pub window: String,
    /// Length of the correlation period in trading days, and its last day.
    pub period_days: u32,
    pub period_end: String,
    pub window_id: u32,
    pub partner: String,
    /// True when the chosen stock is the leader.
    pub leads: bool,
    pub h1: u8,
    pub h2: u8,
    pub lag: u16,
    pub obs: u16,
    /// Census correlation (vol-normalized market-residual returns) and its Fisher z.
    pub r: f32,
    pub z: f32,
    /// Passed the false-discovery test in its matrix.
    pub significant: bool,
    /// Pairs at least about this strong that pure chance produced in the same matrix (mean of
    /// the sign-flip nulls, at the largest of |z| 3/4/5 below this pair's |z|).
    pub chance: f64,
    pub chance_z: u8,
    /// The tested relationship in price terms (`moves::move_per_pct`): r x (sd of the follower's
    /// market-relative h2-day moves / sd of the leader's market-relative h1-day moves) on the
    /// census's observations in the period, i.e. the follower's move relative to the market per
    /// 1% move of the leader relative to the market. Same sign as r, and exactly the multiplier
    /// the forecast applies to this relationship. None if fewer than 10 observations have both
    /// moves.
    pub move_per_pct: Option<f64>,
    /// The same relationship counted in plain terms from prices (raw moves of the closes the
    /// census uses, bad prints repaired): how often the follower rose after the leader's big
    /// moves, in the period and over the rest of the history.
    pub words: Option<Words>,
}

/// Counts on the census observation grid (one case per non-overlapping observation).
#[derive(Clone, Copy, Default, Debug)]
pub struct Counts {
    pub cases: u32,
    /// Follower rose over its move length.
    pub b_up: u32,
    /// Leader rose by at least the threshold, and of those, how often the follower rose.
    pub a_up: u32,
    pub a_up_b_up: u32,
    /// Leader fell by at least the threshold, and of those, how often the follower rose.
    pub a_down: u32,
    pub a_down_b_up: u32,
}

impl Counts {
    fn add(&mut self, x: f64, y: f64, up: f64, down: f64) {
        let b = u32::from(y > 0.0);
        self.cases += 1;
        self.b_up += b;
        if x >= up {
            self.a_up += 1;
            self.a_up_b_up += b;
        }
        if x <= down {
            self.a_down += 1;
            self.a_down_b_up += b;
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Words {
    /// Leader thresholds as log changes: it "rose" when its move >= up, "fell" when <= down.
    pub up: f64,
    pub down: f64,
    /// In the correlation period (what the census saw; selected for being strong there).
    pub period: Counts,
    /// Every other observation on the same grid in the census's history that shares no day
    /// with the period (the out-of-sample check).
    pub rest: Counts,
}

/// Plain-terms counts for one relationship. `pl` and `pt` are log closes of leader and follower
/// by row; observation q of the period ends the leader's move at row e = first_e + q * step
/// (q < m): leader move pl[e] - pl[e - h1], follower move pt[e + lag] - pt[e + lag - h2]. The
/// period covers return rows [p0, p1); `rest` walks the same grid over rows [lo, hi).
/// `threshold` is a fraction (0.03 = 3%); None uses the leader's own typical (1 sd) move in the
/// period.
#[allow(clippy::too_many_arguments)]
pub fn words(pl: &[f64], pt: &[f64], h1: usize, h2: usize, lag: usize, first_e: usize, m: usize, step: usize, p0: usize, p1: usize, lo: usize, hi: usize, threshold: Option<f64>) -> Option<Words> {
    let obs = |e: usize| -> Option<(f64, f64)> {
        let (x, y) = (pl.get(e)? - pl.get(e.checked_sub(h1)?)?, pt.get(e + lag)? - pt.get((e + lag).checked_sub(h2)?)?);
        (x.is_finite() && y.is_finite()).then_some((x, y))
    };
    let period: Vec<(f64, f64)> = (0..m).filter_map(|q| obs(first_e + q * step)).collect();
    if period.len() < 5 {
        return None;
    }
    let (up, down) = match threshold {
        Some(t) => ((1.0 + t).ln(), (1.0 - t.min(0.99)).ln()),
        None => {
            let n = period.len() as f64;
            let mx = period.iter().map(|o| o.0).sum::<f64>() / n;
            let sd = (period.iter().map(|o| (o.0 - mx) * (o.0 - mx)).sum::<f64>() / n).sqrt();
            (sd, -sd)
        }
    };
    let mut w = Words { up, down, ..Default::default() };
    for &(x, y) in &period {
        w.period.add(x, y, up, down);
    }
    // Rest of history: the same grid (same remainder mod step), skipping observations whose
    // return days (leader rows e-h1+1..=e, follower rows e+lag-h2+1..=e+lag) touch the period.
    let mut e = first_e % step;
    while e < lo + h1.max(h2) {
        e += step;
    }
    while e + lag < hi {
        let touches = |a: usize, b: usize| a < p1 && b >= p0;
        if !touches(e + 1 - h1, e) && !touches(e + lag + 1 - h2, e + lag) {
            if let Some((x, y)) = obs(e) {
                w.rest.add(x, y, up, down);
            }
        }
        e += step;
    }
    Some(w)
}

/// Every stored real lagged relationship involving `ticker` (as leader or target), strongest
/// |z| first; with `significant_only`, just those that passed the false-discovery test.
/// Builds the run's census variable first (`run::variable_for`, as `Plan::for_run` does; under a
/// second); `stock_profile_with` reuses one.
pub fn stock_profile(store: &Store, dir: &Path, ticker: &str, significant_only: bool) -> Result<Vec<ProfileRow>> {
    let meta = RunMeta::load(dir)?;
    let (var, _) = crate::run::variable_for(store, &meta)?;
    profile(store, &var, &meta, ticker, significant_only)
}

/// `stock_profile` with the run's census variable `var` (`Plan::var` of `Plan::for_run`, or
/// `run::variable_for`): sizes use its market-relative returns and the
/// plain-terms counts its repaired closes, on its rows.
///
/// What it reads comes from the run in `dir`: its task list (not whatever files `tasks/`
/// holds), its correlation periods mapped onto `var`'s rows by date, and its stock numbering,
/// translated through tickers into `store`'s. Relationships from a correlation period `var`
/// doesn't cover exactly are left out (`RunMeta::uncovered_windows` lists them); a partner that
/// today's store or `var` no longer has keeps its row, under the run's ticker, without sizes.
/// The chosen stock only has to be in the run.
pub fn stock_profile_with(store: &Store, var: &crate::variable::Variable, dir: &Path, ticker: &str, significant_only: bool) -> Result<Vec<ProfileRow>> {
    profile(store, var, &RunMeta::load(dir)?, ticker, significant_only)
}

/// The run's id of `ticker`, or an error saying why there is none.
fn run_id_for(store: &Store, meta: &RunMeta, ids: &meta::IdMap, ticker: &str) -> Result<u32> {
    meta.run_id_of(store, ids, ticker).ok_or_else(|| match store.id(ticker) {
        None => anyhow!("{ticker} is not in the price store"),
        Some(_) => anyhow!("{ticker} was not in the price store when census {} was made, so it has no results there", meta.name),
    })
}

/// Variable rows of each of the run's correlation periods (by date; None when `days` doesn't
/// cover it exactly), and the census's whole history on those rows: the full-history period,
/// else the union of the periods that are covered.
fn period_rows(meta: &RunMeta, days: &[i32]) -> (Vec<Option<Range<usize>>>, Range<usize>) {
    let rows: Vec<Option<Range<usize>>> = (0..meta.windows.len()).map(|w| meta.window_rows_in(days, w)).collect();
    let full = meta.windows.iter().find(|w| w.length == 0).and_then(|w| rows[w.id].clone());
    let all = full.unwrap_or_else(|| {
        let covered = || rows.iter().flatten();
        covered().map(|r| r.start).min().unwrap_or(0)..covered().map(|r| r.end).max().unwrap_or(0)
    });
    (rows, all)
}

fn profile(store: &Store, var: &crate::variable::Variable, meta: &RunMeta, ticker: &str, significant_only: bool) -> Result<Vec<ProfileRow>> {
    let stocks = meta.stocks()?;
    let ids = meta.id_map(store)?;
    let id = run_id_for(store, meta, &ids, ticker)?;
    let (exact, all) = period_rows(meta, &var.days);
    let (lo, hi) = (all.start, all.end);
    let per_task: Vec<Vec<(Record, f64, u8)>> = meta.tasks()?.par_iter().map(|task| task_rows(meta, task, id, significant_only)).collect::<Result<_>>()?;
    let records: Vec<(Record, f64, u8)> = per_task.into_iter().flatten().filter(|(r, _, _)| exact[r.window as usize].is_some()).collect();
    // Each involved stock's log closes and market-relative daily returns, by variable row,
    // built once per partner (in parallel) and dropped after its rows are made. Records name
    // stocks by run id: through the ticker to today's store id, then to the variable's column.
    let col_of: HashMap<usize, usize> = var.ids.iter().enumerate().map(|(c, &i)| (i, c)).collect();
    let series = |rid: u32| {
        ids.current(rid).and_then(|c| col_of.get(&c)).map(|&j| {
            let lc: Vec<f64> = var.col(&var.close, j).iter().map(|&c| if c > 0.0 { (c as f64).ln() } else { f64::NAN }).collect();
            (lc, var.market_relative(j))
        })
    };
    let mine = series(id);
    let mut by_partner: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, (r, _, _)) in records.iter().enumerate() {
        by_partner.entry(if r.leader == id { r.target } else { r.leader }).or_default().push(i);
    }
    let by_partner: Vec<(u32, Vec<usize>)> = by_partner.into_iter().collect();
    let mut rows: Vec<(usize, ProfileRow)> = by_partner
        .par_iter()
        .flat_map_iter(|(partner, idx)| {
            let theirs = series(*partner);
            let name = stocks.ticker(*partner).unwrap_or("?").to_string();
            let rows: Vec<(usize, ProfileRow)> = idx
                .iter()
                .map(|&i| {
                    let (r, chance_n, chance_z) = &records[i];
                    let p = &meta.windows[r.window as usize];
                    let span = exact[r.window as usize].clone().unwrap_or_default();
                    let (h1, h2, lag) = (r.h1 as usize, r.h2 as usize, r.lag as usize);
                    let (mut words_, mut move_per_pct) = (None, None);
                    if let (Some(me), Some(them)) = (&mine, &theirs) {
                        let ((pl, al), (pt, at)) = if r.leader == id { ((&me.0, &me.1), (&them.0, &them.1)) } else { ((&them.0, &them.1), (&me.0, &me.1)) };
                        let step = h1.max(h2);
                        let m = grid::observations(span.len(), h1, h2, lag);
                        // Leader moves end at span.start + max(h1, h2) - 1 + q * step.
                        words_ = words(pl, pt, h1, h2, lag, span.start + step - 1, m, step, span.start, span.end, lo, hi, None);
                        move_per_pct = crate::moves::move_per_pct(al, at, r.r as f64, h1, h2, lag, span.clone());
                    }
                    let row = ProfileRow {
                        window: format!("{}..{}", c2_data::date_string(p.first), c2_data::date_string(p.last)),
                        period_days: span.len() as u32,
                        period_end: c2_data::date_string(p.last),
                        window_id: r.window,
                        partner: name.clone(),
                        leads: r.leader == id,
                        h1: r.h1,
                        h2: r.h2,
                        lag: r.lag,
                        obs: r.obs,
                        r: r.r,
                        z: r.z,
                        significant: r.flags & FLAG_FDR != 0,
                        chance: *chance_n,
                        chance_z: *chance_z,
                        move_per_pct,
                        words: words_,
                    };
                    (i, row)
                })
                .collect();
            rows
        })
        .collect();
    // Task-file order, then strongest |z| first (a stable sort, so ties keep that order).
    rows.sort_by_key(|(i, _)| *i);
    let mut rows: Vec<ProfileRow> = rows.into_iter().map(|(_, row)| row).collect();
    rows.sort_by(|a, b| b.z.abs().total_cmp(&a.z.abs()));
    Ok(rows)
}

/// The real lagged records of one task involving the stock with run id `id`, with the chance
/// counts of their matrices (mean sign-flip null pairs at the largest of |z| 3/4/5 below each
/// one's |z|).
fn task_rows(meta: &RunMeta, task: &Task, id: u32, significant_only: bool) -> Result<Vec<(Record, f64, u8)>> {
    let mut recs = Vec::new();
    visit_task(meta, task, |r| {
        if r.flags & FLAG_NULL == 0 && r.lag > 0 && (r.leader == id || r.target == id) && (!significant_only || r.flags & FLAG_FDR != 0) {
            recs.push(*r);
        }
    })?;
    if recs.is_empty() {
        return Ok(Vec::new());
    }
    // Chance tail counts per (window, lag) from this task's summary.
    let chance: HashMap<(u32, u16), [f64; 3]> = task_summary(meta, task)?.lines.iter().map(|l| ((l.window as u32, l.lag as u16), l.null)).collect();
    Ok(recs
        .into_iter()
        .map(|r| {
            let (az, tails) = (r.z.abs(), chance.get(&(r.window, r.lag)).copied().unwrap_or([f64::NAN; 3]));
            let (chance_z, chance_n) = if az >= 5.0 { (5, tails[2]) } else if az >= 4.0 { (4, tails[1]) } else { (3, tails[0]) };
            (r, chance_n, chance_z)
        })
        .collect())
}

/// One stock's standing in the census: how many of its stored lagged relationships passed the
/// significance test, in how many windows, and its strongest.
#[derive(Clone)]
pub struct RankRow {
    pub ticker: String,
    pub significant: usize,
    pub leads: usize,
    pub follows: usize,
    /// Distinct census windows its significant relationships come from (1 often means a single
    /// event, such as the 2020 crash, rather than a lasting relationship).
    pub windows: usize,
    pub stored: usize,
    /// Sum, mean and maximum of |r| over all its stored relationships: the ranking score
    /// (sum = number of relationships x average strength).
    pub sum_abs_r: f64,
    pub mean_abs_r: f64,
    pub max_abs_r: f32,
    /// Sum of |r| over its significant relationships only.
    pub sig_sum_abs_r: f64,
    /// Strongest |z| among its significant relationships, else among all stored.
    pub best_z: f32,
    pub best_partner: String,
    pub best_leads: bool,
    pub best_lag: u16,
    pub best_r: f32,
    /// That strongest relationship's move lengths in trading days (A's, then B's, whichever
    /// stock leads) and its correlation period (the run's period id, as `ProfileRow::window_id`).
    pub best_h1: u8,
    pub best_h2: u8,
    pub best_window: u32,
    /// Strongest |z| among all stored relationships (mostly chance level).
    pub max_z_any: f32,
}

/// A stock's strongest relationship so far.
#[derive(Clone, Copy)]
struct Best {
    z: f32,
    other: u32,
    leads: bool,
    lag: u16,
    r: f32,
    h1: u8,
    h2: u8,
    window: u32,
}

impl Best {
    /// Keeps the stronger by |z|; on a tie the one met first (task order, then file order).
    fn keep(slot: &mut Option<Best>, cand: Best) {
        if slot.is_none_or(|b| cand.z > b.z) {
            *slot = Some(cand);
        }
    }
}

#[derive(Default)]
struct RankAcc {
    significant: usize,
    leads: usize,
    follows: usize,
    windows: HashSet<u32>,
    stored: usize,
    sum_abs_r: f64,
    max_abs_r: f32,
    sig_sum_abs_r: f64,
    best: Option<Best>,
    best_any: Option<Best>,
}

impl RankAcc {
    fn add(&mut self, r: &Record, other: u32, leads: bool) {
        self.stored += 1;
        self.sum_abs_r += r.r.abs() as f64;
        self.max_abs_r = self.max_abs_r.max(r.r.abs());
        let cand = Best { z: r.z.abs(), other, leads, lag: r.lag, r: r.r, h1: r.h1, h2: r.h2, window: r.window };
        Best::keep(&mut self.best_any, cand);
        if r.flags & FLAG_FDR != 0 {
            self.significant += 1;
            self.sig_sum_abs_r += r.r.abs() as f64;
            if leads { self.leads += 1 } else { self.follows += 1 }
            self.windows.insert(r.window);
            Best::keep(&mut self.best, cand);
        }
    }

    /// Adds the tallies of a later task.
    fn merge(&mut self, o: RankAcc) {
        self.significant += o.significant;
        self.leads += o.leads;
        self.follows += o.follows;
        self.windows.extend(o.windows);
        self.stored += o.stored;
        self.sum_abs_r += o.sum_abs_r;
        self.max_abs_r = self.max_abs_r.max(o.max_abs_r);
        self.sig_sum_abs_r += o.sig_sum_abs_r;
        for (slot, cand) in [(&mut self.best, o.best), (&mut self.best_any, o.best_any)] {
            if let Some(c) = cand {
                Best::keep(slot, c);
            }
        }
    }
}

/// Every stock with stored lagged relationships, most significant first (ties by ticker).
/// Reads the run's task files in parallel. Tickers come from the run's own stock list, so they
/// are right whatever has happened to the store since, and stocks it no longer has are ranked
/// too; `_store` isn't needed for that.
pub fn stock_ranking(_store: &Store, dir: &Path) -> Result<Vec<RankRow>> {
    let meta = RunMeta::load(dir)?;
    let stocks = meta.stocks()?;
    let parts: Vec<HashMap<u32, RankAcc>> = meta
        .tasks()?
        .par_iter()
        .map(|task| {
            let mut acc: HashMap<u32, RankAcc> = HashMap::new();
            visit_task(&meta, task, |r| {
                if r.flags & FLAG_NULL == 0 && r.lag > 0 {
                    acc.entry(r.leader).or_default().add(r, r.target, true);
                    acc.entry(r.target).or_default().add(r, r.leader, false);
                }
            })?;
            Ok(acc)
        })
        .collect::<Result<_>>()?;
    // In task order, so sums and ties never depend on how the work was split.
    let mut acc: HashMap<u32, RankAcc> = HashMap::new();
    for part in parts {
        for (id, a) in part {
            match acc.entry(id) {
                std::collections::hash_map::Entry::Occupied(mut e) => e.get_mut().merge(a),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(a);
                }
            }
        }
    }
    let name = |id: u32| stocks.ticker(id).unwrap_or("?").to_string();
    let mut rows: Vec<RankRow> = acc
        .into_iter()
        .map(|(id, a)| {
            let b = a.best.or(a.best_any).unwrap_or(Best { z: 0.0, other: 0, leads: true, lag: 0, r: 0.0, h1: 0, h2: 0, window: 0 });
            RankRow {
                ticker: name(id),
                significant: a.significant,
                leads: a.leads,
                follows: a.follows,
                windows: a.windows.len(),
                stored: a.stored,
                sum_abs_r: a.sum_abs_r,
                mean_abs_r: a.sum_abs_r / a.stored.max(1) as f64,
                max_abs_r: a.max_abs_r,
                sig_sum_abs_r: a.sig_sum_abs_r,
                best_z: b.z,
                best_partner: name(b.other),
                best_leads: b.leads,
                best_lag: b.lag,
                best_r: b.r,
                best_h1: b.h1,
                best_h2: b.h2,
                best_window: b.window,
                max_z_any: a.best_any.map_or(0.0, |b| b.z),
            }
        })
        .collect();
    rows.sort_by(|a, b| b.sum_abs_r.total_cmp(&a.sum_abs_r).then_with(|| a.ticker.cmp(&b.ticker)));
    Ok(rows)
}

/// Real and chance counts for one group of matrices (lagged ones only).
#[derive(Clone, Copy, Default, Debug)]
pub struct Tally {
    pub matrices: usize,
    pub correlations: f64,
    /// Real pairs at |z| >= 3, 4, 5, and the mean of the sign-flip nulls.
    pub real: [f64; 3],
    pub null: [f64; 3],
    pub significant: usize,
}

impl Tally {
    fn add(&mut self, corr: f64, real: [f64; 3], null: [f64; 3]) {
        self.matrices += 1;
        self.correlations += corr;
        for k in 0..3 {
            self.real[k] += real[k];
            self.null[k] += null[k];
        }
    }

    fn merge(&mut self, o: &Tally) {
        self.matrices += o.matrices;
        self.correlations += o.correlations;
        for k in 0..3 {
            self.real[k] += o.real[k];
            self.null[k] += o.null[k];
        }
        self.significant += o.significant;
    }
}

/// Whole-census statistics for the Status tab.
#[derive(Clone, Default)]
pub struct DatasetStats {
    pub nulls: usize,
    pub stocks: usize,
    pub periods: usize,
    pub tasks: usize,
    /// Every matrix (lag 0 included): correlations computed, real only.
    pub correlations_all: f64,
    pub matrices_all: usize,
    /// Lagged matrices only.
    pub lagged: Tally,
    pub fdr_matrices: usize,
    /// Lagged matrices where the false-discovery test, run on a sign-flip null matrix, fired.
    /// Only meaningful when `placebo_recorded`.
    pub placebo_matrices: usize,
    pub stored_real: usize,
    pub stored_null: usize,
    pub significant_pairs: usize,
    pub significant_stocks: usize,
    /// Keyed tallies: by (A move length, B move length), by period length, by period end year,
    /// and by offset as a multiple of B's move length.
    pub by_horizons: Vec<((usize, usize), Tally)>,
    pub by_length: Vec<(usize, Tally)>,
    pub by_year: Vec<(i32, Tally)>,
    pub by_offset_steps: Vec<(usize, Tally)>,
    /// |r| histograms (50 bins over 0..1) of stored real records and stored null records (both
    /// are each matrix's strongest; the null ones are the strongest of pure chance).
    pub hist_real: Vec<u64>,
    pub hist_null: Vec<u64>,
    /// Per stock (as leader or follower): ticker (the run's), stored relationships, mean |r| of
    /// them, significant. Most stored first, then by ticker.
    pub per_stock: Vec<(String, usize, f64, usize)>,
    /// False when the run's summaries have no placebo column (runs made before it existed, with
    /// 17-column summaries): the placebo test was not recorded, so `placebo_matrices` (0) says
    /// nothing.
    pub placebo_recorded: bool,
    /// Set when the run's stock list can't be recovered (the message says why): `per_stock`'s
    /// tickers are then empty. Everything else holds.
    pub stock_list_error: Option<String>,
}

/// One task's share of `DatasetStats`.
#[derive(Default)]
struct StatsPart {
    correlations_all: f64,
    matrices_all: usize,
    lagged: Tally,
    fdr_matrices: usize,
    placebo_matrices: usize,
    placebo_recorded: bool,
    by_h: HashMap<(usize, usize), Tally>,
    by_len: HashMap<usize, Tally>,
    by_year: HashMap<i32, Tally>,
    by_off: HashMap<usize, Tally>,
    stored_real: usize,
    stored_null: usize,
    hist_real: Vec<u64>,
    hist_null: Vec<u64>,
    per: HashMap<u32, (usize, f64, usize)>,
    sig_pairs: HashSet<(u32, u32)>,
    sig_by: HashMap<(usize, usize, usize, i32, usize), usize>,
}

/// Whole-census statistics of the run in `dir`: its summaries (by their header's column names)
/// and its records, read task file by task file in parallel and never all held at once. The
/// files are the run's task list's. Tickers come from the run's own stock list; `_store` isn't
/// needed for that.
pub fn dataset_stats(_store: &Store, dir: &Path) -> Result<DatasetStats> {
    let meta = RunMeta::load(dir)?;
    let tasks = meta.tasks()?;
    // A correlation period's length and the year it ended.
    let period = |w: usize| meta.windows.get(w).map_or((0, 0), |w| (w.length, c2_data::civil_from_days(w.last).0));
    let parts: Vec<StatsPart> = tasks
        .par_iter()
        .map(|task| -> Result<StatsPart> {
            let mut p = StatsPart { hist_real: vec![0; 50], hist_null: vec![0; 50], ..Default::default() };
            let summary = task_summary(&meta, task)?;
            p.placebo_recorded = summary.placebo_recorded || summary.lines.is_empty();
            for l in &summary.lines {
                let n = l.stocks as f64;
                let corr = n * (n - 1.0).max(0.0);
                p.correlations_all += corr;
                p.matrices_all += 1;
                if l.lag == 0 {
                    continue;
                }
                p.lagged.add(corr, l.real, l.null);
                let (len, year) = period(l.window);
                p.by_h.entry((l.h1, l.h2)).or_default().add(corr, l.real, l.null);
                p.by_len.entry(len).or_default().add(corr, l.real, l.null);
                p.by_year.entry(year).or_default().add(corr, l.real, l.null);
                p.by_off.entry(l.lag / l.h2.max(1)).or_default().add(corr, l.real, l.null);
                p.fdr_matrices += usize::from(l.fdr_r.is_some());
                p.placebo_matrices += usize::from(l.placebo_fdr_r.is_some());
            }
            // Records: stored counts, |r| histograms, per-stock numbers, significant tallies.
            visit_task(&meta, task, |r| {
                if r.lag == 0 {
                    return;
                }
                let bin = ((r.r.abs() * 50.0) as usize).min(49);
                if r.flags & FLAG_NULL != 0 {
                    p.stored_null += 1;
                    p.hist_null[bin] += 1;
                    return;
                }
                p.stored_real += 1;
                p.hist_real[bin] += 1;
                let sig = r.flags & FLAG_FDR != 0;
                for s in [r.leader, r.target] {
                    let e = p.per.entry(s).or_default();
                    e.0 += 1;
                    e.1 += r.r.abs() as f64;
                    e.2 += usize::from(sig);
                }
                if sig {
                    p.sig_pairs.insert((r.leader, r.target));
                    let (len, year) = period(r.window as usize);
                    *p.sig_by.entry((r.h1 as usize, r.h2 as usize, len, year, r.lag as usize / (r.h2 as usize).max(1))).or_default() += 1;
                }
            })?;
            Ok(p)
        })
        .collect::<Result<_>>()?;
    let mut st = DatasetStats { nulls: meta.cfg.nulls, periods: meta.windows.len(), tasks: tasks.len(), placebo_recorded: true, ..Default::default() };
    let (mut by_h, mut by_len, mut by_year, mut by_off) = (HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new());
    let (mut hist_real, mut hist_null) = (vec![0u64; 50], vec![0u64; 50]);
    let mut per: HashMap<u32, (usize, f64, usize)> = HashMap::new();
    let mut sig_pairs: HashSet<(u32, u32)> = HashSet::new();
    let mut sig_by: HashMap<(usize, usize, usize, i32, usize), usize> = HashMap::new();
    fn merge_into<K: std::hash::Hash + Eq>(to: &mut HashMap<K, Tally>, from: HashMap<K, Tally>) {
        for (k, t) in from {
            to.entry(k).or_default().merge(&t);
        }
    }
    // In task order, so the sums never depend on how the work was split.
    for p in parts {
        st.correlations_all += p.correlations_all;
        st.matrices_all += p.matrices_all;
        st.lagged.merge(&p.lagged);
        st.fdr_matrices += p.fdr_matrices;
        st.placebo_matrices += p.placebo_matrices;
        st.placebo_recorded &= p.placebo_recorded;
        merge_into(&mut by_h, p.by_h);
        merge_into(&mut by_len, p.by_len);
        merge_into(&mut by_year, p.by_year);
        merge_into(&mut by_off, p.by_off);
        st.stored_real += p.stored_real;
        st.stored_null += p.stored_null;
        for k in 0..50 {
            hist_real[k] += p.hist_real[k];
            hist_null[k] += p.hist_null[k];
        }
        for (s, (n, sum, sig)) in p.per {
            let e = per.entry(s).or_default();
            e.0 += n;
            e.1 += sum;
            e.2 += sig;
        }
        sig_pairs.extend(p.sig_pairs);
        for (k, c) in p.sig_by {
            *sig_by.entry(k).or_default() += c;
        }
    }
    for (&(h1, h2, len, year, off), &c) in &sig_by {
        by_h.entry((h1, h2)).or_insert_with(Tally::default).significant += c;
        by_len.entry(len).or_insert_with(Tally::default).significant += c;
        by_year.entry(year).or_insert_with(Tally::default).significant += c;
        by_off.entry(off).or_insert_with(Tally::default).significant += c;
        st.lagged.significant += c;
    }
    st.hist_real = hist_real;
    st.hist_null = hist_null;
    st.significant_pairs = sig_pairs.len();
    st.stocks = per.len();
    st.significant_stocks = per.values().filter(|p| p.2 > 0).count();
    let stocks = meta.stocks().map_err(|e| format!("{e:#}"));
    st.stock_list_error = stocks.as_ref().err().cloned();
    st.per_stock = per
        .into_iter()
        .map(|(id, (n, s, sig))| {
            let ticker = stocks.as_ref().ok().and_then(|t| t.ticker(id)).unwrap_or("").to_string();
            (ticker, n, s / n.max(1) as f64, sig)
        })
        .collect();
    st.per_stock.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    fn sorted<K: Ord + Copy>(m: HashMap<K, Tally>) -> Vec<(K, Tally)> {
        let mut v: Vec<_> = m.into_iter().collect();
        v.sort_by_key(|e| e.0);
        v
    }
    st.by_horizons = sorted(by_h);
    st.by_length = sorted(by_len);
    st.by_year = sorted(by_year);
    st.by_offset_steps = sorted(by_off);
    Ok(st)
}

/// One census run directory, for the Status tab's history.
#[derive(Clone, Default)]
pub struct RunInfo {
    pub name: String,
    /// "full" or "smoke test" (from the run's settings; "unknown" if they can't be read).
    pub kind: String,
    pub started_unix: u64,
    pub finished_unix: Option<u64>,
    /// As status.json says, except "running" while a process computes it (whatever the file
    /// says) and "interrupted" when the file says running but nothing does.
    pub state: String,
    pub fraction: f64,
    /// Time spent computing, summed over every session in run.log.
    pub seconds: f64,
    pub sessions: usize,
    /// Last day of data it covered.
    pub data_last_day: String,
    pub correlations_real: f64,
    pub correlations_null: f64,
    /// INT8 multiply-adds on the GPU (padded matrix sizes), real and null.
    pub multiply_adds: f64,
    /// `c2 update` started it after new data arrived.
    pub from_update: bool,
    /// The run the caller reads (`run_history_using`); for `run_history`, the default census.
    pub in_use: bool,
    /// The default census (`census/LATEST`).
    pub latest: bool,
    /// The process computing it, while one is (`run::running_pid`).
    pub pid: Option<u32>,
}

/// Every census run under the data directory, newest first; `in_use` marks the default census
/// (`census/LATEST`). `run_history_using` marks the run the caller reads instead.
pub fn run_history() -> Vec<RunInfo> {
    run_history_using(&crate::run::latest_run())
}

/// Every census run under the data directory, newest first; `in_use` marks the run named
/// `in_use` (the one the caller shows, e.g. the GUI's chosen run) and `latest` the default
/// census. A run whose files can't all be read is still listed, with what can be.
pub fn run_history_using(in_use: &str) -> Vec<RunInfo> {
    let root = c2_data::data_dir().join("census");
    let latest = crate::run::latest_run();
    let update_log = std::fs::read_to_string(c2_data::data_dir().join("update.log")).unwrap_or_default();
    let mut out: Vec<RunInfo> = std::fs::read_dir(&root)
        .map(|it| it.flatten().map(|e| e.path()).filter(|p| p.join("manifest.json").exists()).collect::<Vec<_>>())
        .unwrap_or_default()
        .par_iter()
        .map(|dir| run_info(dir, in_use, &latest, &update_log))
        .collect();
    out.sort_by(|a, b| b.started_unix.cmp(&a.started_unix).then_with(|| a.name.cmp(&b.name)));
    out
}

fn run_info(dir: &Path, in_use: &str, latest: &str, update_log: &str) -> RunInfo {
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
    let m: serde_json::Value = std::fs::read_to_string(dir.join("manifest.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let s: serde_json::Value = std::fs::read_to_string(dir.join("status.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let meta = RunMeta::load(dir).ok();
    // The run's settings as RunMeta reads them, else straight from the manifest's text.
    let cfg: Option<Config> = meta.as_ref().map(|m| m.cfg.clone()).or_else(|| Config::from_text(m["config"].as_str()?).ok());
    let nulls = cfg.as_ref().map_or(0, |c| c.nulls) as f64;
    let pid = crate::run::running_pid(dir);
    let mut state = s["state"].as_str().unwrap_or("not started").to_string();
    if pid.is_some() {
        state = "running".into();
    } else if state == "running" {
        // status.json says running but no process holds the run: it crashed or was killed.
        state = "interrupted".into();
    }
    let (seconds, sessions) = computing_time(&std::fs::read_to_string(dir.join("run.log")).unwrap_or_default());
    let mut info = RunInfo {
        kind: match &cfg {
            Some(c) if c.full_history => "full",
            Some(_) => "smoke test",
            None => "unknown",
        }
        .into(),
        started_unix: m["created_unix"].as_u64().unwrap_or(0),
        finished_unix: (state == "finished").then(|| s["updated_unix"].as_u64()).flatten(),
        fraction: s["fraction_done"].as_f64().unwrap_or(0.0),
        state,
        seconds,
        sessions,
        data_last_day: meta.as_ref().and_then(|m| m.windows.iter().map(|w| w.last).max()).map(c2_data::date_string).unwrap_or_default(),
        from_update: update_log.contains(&format!("running census {name} ")),
        in_use: name == in_use,
        latest: name == latest,
        pid,
        name,
        ..Default::default()
    };
    // Correlations computed, from every summary line: of the run's task files, or of every
    // summary file there when its task list is unknown (e.g. a crashed run).
    let summaries: Vec<std::path::PathBuf> = match meta.as_ref().map(|m| m.tasks().map(|t| t.iter().map(|t| m.task_path(t.id, "csv")).collect())) {
        Some(Ok(paths)) => paths,
        _ => std::fs::read_dir(dir.join("tasks"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("csv"))
            .collect(),
    };
    for path in summaries {
        let Ok(summary) = read_summary(&path) else { continue };
        for l in &summary.lines {
            let n = l.stocks as f64;
            let pairs = n * (n - 1.0).max(0.0);
            info.correlations_real += pairs;
            info.correlations_null += pairs * nulls;
            let np = crate::run::pad_width(l.stocks) as f64;
            info.multiply_adds += np * np * (l.obs.div_ceil(32) * 32) as f64 * (1.0 + nulls);
        }
    }
    info
}

/// Time spent computing, from a run's log, and the number of sessions: each session runs from a
/// "plan:" line to the last time stamp before the next one (a log with no "plan:" line at all is
/// one session from its first line to its last). Lines are "[HH:MM:SS] message", as both
/// `c2 census run` and `c2 update` write them; a time earlier than the line before it is taken
/// to be on the next day.
fn computing_time(log: &str) -> (f64, usize) {
    let (mut seconds, mut sessions, mut start, mut last) = (0.0, 0, None::<f64>, None::<f64>);
    let (mut first, mut prev, mut wrap) = (None::<f64>, 0.0, 0.0);
    for line in log.lines() {
        let (Some(stamp), Some(message)) = (line.get(1..9), line.get(10..)) else { continue };
        if !line.starts_with('[') || line.as_bytes()[9] != b']' {
            continue;
        }
        let f: Vec<f64> = stamp.split(':').filter_map(|v| v.parse().ok()).collect();
        if f.len() != 3 {
            continue;
        }
        let mut t = f[0] * 3600.0 + f[1] * 60.0 + f[2] + wrap;
        if t < prev {
            wrap += 86400.0;
            t += 86400.0;
        }
        prev = t;
        first.get_or_insert(t);
        if message.trim_start().starts_with("plan:") {
            if let (Some(s0), Some(l)) = (start, last) {
                seconds += l - s0;
            }
            start = Some(t);
            sessions += 1;
        }
        last = Some(t);
    }
    match (start.or(first), last) {
        (Some(s0), Some(l)) => {
            seconds += l - s0;
            (seconds, sessions.max(1))
        }
        _ => (0.0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computing_time_counts_sessions_from_either_writer() {
        // `c2 census run`: two sessions, the second across midnight; the time between them
        // doesn't count.
        let run = "[23:50:00] plan: 8503 stocks in the variable, 285 windows (universe 283..2345 stocks), 180 tasks, 10260 units; built in 0.5s\n\
                   [23:50:15] 56/10260 windows (1.46% of work), last unit 266 ms on GPU, ETA 0h17m01s\n\
                   [23:55:00] stopped: 3000/10260 windows committed\n\
                   [23:58:00] plan: 8503 stocks in the variable, 285 windows (universe 283..2345 stocks), 180 tasks, 10260 units; built in 0.5s\n\
                   [00:02:00] 6000/10260 windows (60.00% of work), last unit 266 ms on GPU, ETA 0h03m00s\n\
                   [00:05:00] finished: 10260/10260 windows committed\n";
        assert_eq!(computing_time(run), (300.0 + 420.0, 2));
        // `c2 update` writes the same lines into the run's log (its own line before the plan
        // line is not computing time).
        let update = "[18:30:02] running census main-2025-07-03 on the new data\n\
                      [18:30:03] plan: 8510 stocks in the variable, 286 windows (universe 283..2350 stocks), 180 tasks, 10296 units; built in 0.5s\n\
                      [18:30:18] 56/10296 windows (1.46% of work), last unit 266 ms on GPU, ETA 0h12m01s\n\
                      [18:42:03] finished: 10296/10296 windows committed\n";
        assert_eq!(computing_time(update), (720.0, 1));
        // A log with no plan line (an older writer) is one session.
        assert_eq!(computing_time("[10:00:00] 1/888 windows\n[10:00:30] finished: 888/888 windows committed\n"), (30.0, 1));
        assert_eq!(computing_time(""), (0.0, 0));
        assert_eq!(computing_time("no time stamp\n[1x:00:00] bad\n"), (0.0, 0));
    }

    #[test]
    fn summaries_read_by_header() {
        let old = "window,h1,h2,lag,stocks,obs,real_z3,real_z4,real_z5,null_z3,null_z4,null_z5,fdr_r,stored_real,stored_null,screened,truncated\n\
                   4,1,1,0,300,252,7336,2990,1574,261.0,9.7,0.3,,100,50,100,0\n\
                   4,1,1,1,300,251,3000,20,1,2900.5,19.0,0.7,0.2817,100,50,100,0\n";
        let new = format!("{}4,1,1,1,300,251,3000,20,1,2900.5,19.0,0.7,0.2817,120,50,140,0,0.3012\n4,1,1,2,300,250,1,0,0,1.0,0,0,,100,50,100,0,\n4,1,1,3,30", crate::run::SUMMARY_HEADER);
        let (sa, sb) = (parse_summary(old, "a").unwrap(), parse_summary(&new, "b").unwrap());
        assert!(!sa.placebo_recorded && sb.placebo_recorded);
        assert_eq!(sa.lines.len(), 2);
        let l = &sa.lines[1];
        assert_eq!((l.window, l.h1, l.h2, l.lag, l.stocks, l.obs, l.fdr_r, l.placebo_fdr_r), (4, 1, 1, 1, 300, 251, Some(0.2817), None));
        assert_eq!((l.real, l.null), ([3000.0, 20.0, 1.0], [2900.5, 19.0, 0.7]));
        // The cut-off last line (no newline) is ignored; an empty placebo field did not fire.
        assert_eq!(sb.lines.len(), 2);
        assert_eq!((sb.lines[0].placebo_fdr_r, sb.lines[0].stored_real, sb.lines[1].placebo_fdr_r, sb.lines[1].fdr_r), (Some(0.3012), 120, None, None));
        // A header alone, or nothing at all: no matrices.
        assert!(parse_summary(crate::run::SUMMARY_HEADER, "h").unwrap().lines.is_empty());
        assert!(parse_summary("", "e").unwrap().lines.is_empty());
        // A field that isn't a number, a line with too few fields, a header without a needed column.
        let short = "window,h1,h2,lag,stocks,obs,real_z3,real_z4,real_z5,null_z3,null_z4,null_z5,fdr_r,stored_real,stored_null\n";
        assert!(parse_summary(&format!("{short}4,1,1,x,1,1,1,1,1,1,1,1,,1,1\n"), "c").is_err());
        assert!(parse_summary(&format!("{short}4,1,1,1,1,1\n"), "d").is_err());
        assert!(parse_summary("window,h1,h2,lag\n4,1,1,1\n", "f").is_err());
    }
}
