//! Running the census: plan, GPU worker, finisher thread, checkpoints, status.
//!
//! Layout of a run directory (`<data dir>/census/<name>/`):
//!   manifest.json      config, store fingerprint, code version (written once; resume checks it)
//!   windows.csv        window id, length, first/last date, universe size
//!   universes.bin      per window: u32 count, then u32 store ids (the window's stocks)
//!   tasks/t<id>.rec    52-byte records (see `Record`), appended a window at a time
//!   tasks/t<id>.csv    one summary line per (window, lag) matrix
//!   progress.log       "task window rec_bytes csv_bytes" per committed window
//!   status.json        progress and ETA, rewritten every few seconds
//!   control            "run" or "pause" (`c2 census pause|resume`)
//!   symbols.tsv        copy of the store's symbol table the run's store ids refer to
//!   lock, pid          held / written by the one process running the census (`running_pid`)
//!
//! A unit of work is one window of one task (all its lags). The GPU thread computes it and hands
//! each lag's histograms and screened entries to the finisher thread, which recomputes stored
//! entries exactly, writes them and commits the unit. Stopping at any point loses at most the
//! units in flight; resuming truncates files to the last commit and continues.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use c2_data::{Calendar, Store};
use rayon::prelude::*;

use crate::grid::{self, Config, Task, Window};
use crate::meta::{self, RunMeta};
use crate::variable::{self, Variable};

pub const RECORD_BYTES: usize = 52;

/// Flags on a stored record.
pub const FLAG_TOP: u8 = 1;
pub const FLAG_FDR: u8 = 2;
pub const FLAG_NULL: u8 = 4;
pub const FLAG_TRUNCATED: u8 = 8;

/// One stored correlation. Values are in census-variable units (each horizon's sum divided by
/// sqrt(horizon)); the sums let windows be extended or merged exactly later.
#[derive(Clone, Copy, Debug, Default)]
pub struct Record {
    pub window: u32,
    pub h1: u8,
    pub h2: u8,
    pub lag: u16,
    pub flags: u8,
    pub obs: u16,
    pub leader: u32,
    pub target: u32,
    pub r: f32,
    pub r_int8: f32,
    pub z: f32,
    pub sxy: f32,
    pub sx: f32,
    pub sy: f32,
    pub sxx: f32,
    pub syy: f32,
}

impl Record {
    pub fn to_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.window.to_le_bytes());
        out.push(self.h1);
        out.push(self.h2);
        out.extend_from_slice(&self.lag.to_le_bytes());
        out.push(self.flags);
        out.push(0);
        out.extend_from_slice(&self.obs.to_le_bytes());
        out.extend_from_slice(&self.leader.to_le_bytes());
        out.extend_from_slice(&self.target.to_le_bytes());
        for v in [self.r, self.r_int8, self.z, self.sxy, self.sx, self.sy, self.sxx, self.syy] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    pub fn from_bytes(b: &[u8]) -> Record {
        let u32_at = |p: usize| u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        let f32_at = |p: usize| f32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        Record {
            window: u32_at(0),
            h1: b[4],
            h2: b[5],
            lag: u16::from_le_bytes([b[6], b[7]]),
            flags: b[8],
            obs: u16::from_le_bytes([b[10], b[11]]),
            leader: u32_at(12),
            target: u32_at(16),
            r: f32_at(20),
            r_int8: f32_at(24),
            z: f32_at(28),
            sxy: f32_at(32),
            sx: f32_at(36),
            sy: f32_at(40),
            sxx: f32_at(44),
            syy: f32_at(48),
        }
    }
}

/// The census the GUI and CLI use by default: the name in `census/LATEST`, else "main".
pub fn latest_run() -> String {
    std::fs::read_to_string(c2_data::data_dir().join("census/LATEST")).map(|s| s.trim().to_string()).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "main".into())
}

/// Makes `name` the default census (`census/LATEST`). The only writer of LATEST; atomic, so a
/// reader never sees it empty or half written, and durable (`write_durable`), so the change
/// survives a power loss once this returns.
pub fn set_latest(name: &str) -> Result<()> {
    anyhow::ensure!(!name.trim().is_empty() && !name.contains('/'), "{name:?} is not a census name");
    let dir = c2_data::data_dir().join("census");
    std::fs::create_dir_all(&dir)?;
    write_durable(&dir.join("LATEST"), name.trim().as_bytes())
}

pub fn run_dir(name: &str) -> PathBuf {
    c2_data::data_dir().join("census").join(name)
}

/// Everything decided before any work starts.
pub struct Plan {
    pub cfg: Config,
    pub var: Variable,
    /// Per correlation period id, its rows of `var`. Only meaningful where `period_ok` holds: an
    /// unavailable period keeps its id with an empty range (`start == end`: the row after its
    /// last day's position, kept within 1..var.t-1 so that `var.days[start]` and
    /// `var.days[end - 1]` are always valid), so nothing is ever computed over it.
    pub windows: Vec<Window>,
    /// Per window: variable column indices of its universe (stocks today's data still has).
    pub universes: Vec<Vec<usize>>,
    pub tasks: Vec<Task>,
    /// Estimated cost of each (task, window position).
    pub cost: Vec<Vec<f64>>,
    /// Per correlation period: true when today's data covers it exactly (`RunMeta::window_rows`).
    /// Always true for `Plan::new`. Readers skip the others (`period_available`).
    pub period_ok: Vec<bool>,
    /// Per correlation period: the run's stocks (as the run's store ids) that today's store or
    /// census variable no longer has; they are left out of `universes`. Empty for `Plan::new`.
    pub missing_stocks: Vec<Vec<u32>>,
    /// Set when values recomputed with this plan may differ from those the run stored, saying
    /// why: the run's census variable started on another day (older code). See `variable_for`.
    pub variable_note: Option<String>,
    /// Store id as the run's records hold it -> variable column, and back. The identity on
    /// `var.ids` for a new run; through tickers for `Plan::for_run`.
    run_col: HashMap<u32, usize>,
    col_run: Vec<Option<u32>>,
    /// The run `Plan::for_run` built this for: (name, the manifest's `created_unix`).
    built_for: Option<(String, u64)>,
}

impl Plan {
    /// The plan for starting (or resuming) a run with `cfg` on this store.
    pub fn new(store: &Store, cfg: Config) -> Result<Plan> {
        let cal = Calendar::trading_days(store)?;
        let var = variable::production(store, &cal, cfg.start)?;
        let first_row = var.days.partition_point(|&d| d < cfg.start);
        let windows = grid::windows(&cfg, first_row, var.t);
        let universes: Vec<Vec<usize>> = windows.par_iter().map(|w| universe(&var, w)).collect();
        let tasks = grid::tasks(&cfg, &windows);
        let cost = unit_costs(&cfg, &windows, &universes, &tasks);
        let col_run: Vec<Option<u32>> = var.ids.iter().map(|&id| Some(id as u32)).collect();
        let run_col = var.ids.iter().enumerate().map(|(c, &id)| (id as u32, c)).collect();
        let (period_ok, missing_stocks) = (vec![true; windows.len()], vec![Vec::new(); windows.len()]);
        Ok(Plan { cfg, var, windows, universes, tasks, cost, period_ok, missing_stocks, variable_note: None, run_col, col_run, built_for: None })
    }

    /// The plan of an existing run, rebuilt on today's store: the census variable from the
    /// current store with the run's settings and the run's own first day (`variable_for`), and
    /// the run's own correlation periods (by date), universes (`universes.bin`, translated to
    /// today's variable columns) and tasks. So `windows[id]` and `tasks[id]` mean what they
    /// meant in that run, and `column_of_run_id` turns a record's leader or target into a
    /// variable column.
    ///
    /// What today's data can't reproduce is marked, not refused: a correlation period it
    /// doesn't cover exactly is unavailable (`period_ok`), a stock it no longer has is left out
    /// of its periods and counted (`missing_stocks`), and a census variable that started on
    /// another day is noted (`variable_note`). Errors only when the run can't be read
    /// correctly at all: its task list or its stock list is unknown.
    pub fn for_run(store: &Store, meta: &RunMeta) -> Result<Plan> {
        let tasks = meta.tasks()?.to_vec();
        meta.stocks()?;
        let ids = meta.id_map(store)?;
        let cfg = meta.cfg.clone();
        let (var, variable_note) = variable_for(store, meta)?;
        let (windows, period_ok): (Vec<Window>, Vec<bool>) = meta
            .windows
            .iter()
            .map(|w| match meta.window_rows_in(&var.days, w.id) {
                Some(rows) => (Window { id: w.id, length: w.length, start: rows.start, end: rows.end }, true),
                None => {
                    // An empty range, placed so that both `var.days[start]` and `var.days[end - 1]`
                    // are valid rows (the run's dates are in `meta.windows`).
                    let at = var.days.partition_point(|&d| d <= w.last).clamp(1, var.t.saturating_sub(1).max(1));
                    (Window { id: w.id, length: w.length, start: at, end: at }, false)
                }
            })
            .unzip();
        let col_of: HashMap<usize, usize> = var.ids.iter().enumerate().map(|(c, &id)| (id, c)).collect();
        let (universes, missing_stocks): (Vec<Vec<usize>>, Vec<Vec<u32>>) = meta
            .universes
            .iter()
            .map(|u| {
                let (mut cols, mut missing) = (Vec::with_capacity(u.len()), Vec::new());
                for &rid in u {
                    match ids.current(rid).and_then(|cur| col_of.get(&cur)) {
                        Some(&c) => cols.push(c),
                        None => missing.push(rid),
                    }
                }
                (cols, missing)
            })
            .unzip();
        let cost = unit_costs(&cfg, &windows, &universes, &tasks);
        let col_run: Vec<Option<u32>> = var.ids.iter().map(|&id| ids.run(id)).collect();
        let run_col = col_run.iter().enumerate().filter_map(|(c, r)| r.map(|r| (r, c))).collect();
        let built_for = Some((meta.name.clone(), meta.manifest["created_unix"].as_u64().unwrap_or(0)));
        Ok(Plan { cfg, var, windows, universes, tasks, cost, period_ok, missing_stocks, variable_note, run_col, col_run, built_for })
    }

    pub fn total_cost(&self) -> f64 {
        self.cost.iter().flatten().sum()
    }

    /// True when correlation period `id` can be read: today's data covers it exactly.
    pub fn period_available(&self, id: usize) -> bool {
        self.period_ok.get(id).copied().unwrap_or(false)
    }

    /// Correlation periods today's data doesn't cover exactly (unavailable), and the run's
    /// stock places in period universes that today's data no longer has, summed over periods.
    pub fn unavailable(&self) -> (usize, usize) {
        (self.period_ok.iter().filter(|ok| !**ok).count(), self.missing_stocks.iter().map(|m| m.len()).sum())
    }

    /// The run `Plan::for_run` built this plan for; None for `Plan::new`.
    pub fn built_for(&self) -> Option<&str> {
        self.built_for.as_ref().map(|(name, _)| name.as_str())
    }

    /// True when `Plan::for_run` built this plan for the run `meta` describes (same name and
    /// creation time).
    pub fn is_for(&self, meta: &RunMeta) -> bool {
        self.built_for.as_ref().is_some_and(|(name, created)| *name == meta.name && *created == meta.manifest["created_unix"].as_u64().unwrap_or(0))
    }

    /// Variable column of a stock as the run's records name it (`Record::leader` / `target`).
    pub fn column_of_run_id(&self, run_id: u32) -> Option<usize> {
        self.run_col.get(&run_id).copied()
    }

    /// The run's store id of a variable column (what its records call that stock).
    pub fn run_id_of_column(&self, col: usize) -> Option<u32> {
        self.col_run.get(col).copied().flatten()
    }
}

/// The census variable a run's results are read against: today's store with the run's
/// settings, starting on the run's own first day (the manifest's `variable_days`). The second
/// value is set, saying why, when that first day differs from the one today's code starts the
/// variable on (runs made by older code, such as `smoke`, which started 2003-12-01): the
/// variable is then built from the run's first day when the calendar allows, because values
/// depend on where it starts, and not only through the phase of its 21-day liquidity
/// re-checks (`smoke` starts a whole number of re-checks later, yet built from today's first
/// day its r differ by up to 0.22 in periods ending 2007-2009). Values recomputed today may
/// still differ from those the run stored. The variable is built once.
pub fn variable_for(store: &Store, meta: &RunMeta) -> Result<(Variable, Option<String>)> {
    let cal = Calendar::trading_days(store)?;
    let build = |cal: &Calendar| variable::production(store, cal, meta.cfg.start);
    let first = cal.days.get(variable::first_row(&cal, meta.cfg.start)).copied().context("the price data has no trading days for the census variable")?;
    let today = c2_data::date_string(first);
    let name = &meta.name;
    let Some(theirs) = meta.manifest["variable_days"][0].as_str().map(String::from) else {
        let note = format!("census {name} doesn't record the first day of its census variable, so values recomputed today may differ from the stored ones");
        return Ok((build(&cal)?, Some(note)));
    };
    if theirs == today {
        return Ok((build(&cal)?, None));
    }
    // A later first day: start the calendar there, so the variable's rows start there too.
    if let Some(k) = c2_data::parse_date(&theirs).ok().and_then(|d| cal.days.binary_search(&d).ok()).filter(|_| theirs > today) {
        let from_theirs = build(&Calendar { days: cal.days[k..].to_vec() })?;
        if c2_data::date_string(from_theirs.days[0]) == theirs {
            let note = format!(
                "census {name} was made by older code whose census variable started on {theirs} (today's starts on {today}); it is rebuilt from {theirs} here, but values recomputed today may still differ from the stored ones"
            );
            return Ok((from_theirs, Some(note)));
        }
    }
    let note = format!("census {name} was made by older code whose census variable started on {theirs}; today's starts on {today}, so values recomputed today may differ from the stored ones");
    Ok((build(&cal)?, Some(note)))
}

/// Estimated cost of each (task, window position).
fn unit_costs(cfg: &Config, windows: &[Window], universes: &[Vec<usize>], tasks: &[Task]) -> Vec<Vec<f64>> {
    tasks
        .iter()
        .map(|t| {
            t.windows
                .iter()
                .map(|&wid| {
                    let (w, n) = (&windows[wid], universes[wid].len() as f64);
                    let rows: usize = grid::lags(t.h2)
                        .into_iter()
                        .map(|k| grid::observations(w.end - w.start, t.h1, t.h2, k))
                        .filter(|&m| m >= cfg.min_obs)
                        // Each product also pays a fixed cost (writing and scanning the n x n result)
                        // worth about 2,000 rows of INT8 arithmetic on the RTX 5080 (test-results.md, M0).
                        .map(|m| m.div_ceil(32) * 32 + 2000)
                        .sum();
                    n * n * rows as f64 * (1 + cfg.nulls) as f64
                })
                .collect()
        })
        .collect()
}

/// A window's stocks: >= 90% of days with a census value, and over the window's last 252 days
/// a median price >= $5 and median dollar volume >= $5M.
fn universe(var: &Variable, w: &Window) -> Vec<usize> {
    let (len, tail) = (w.end - w.start, (w.end - w.start).min(variable::LOOKBACK));
    (0..var.n)
        .filter(|&j| {
            let v = &var.col(&var.v, j)[w.start..w.end];
            if v.iter().filter(|x| x.is_finite()).count() * 10 < len * 9 {
                return false;
            }
            let (c, dv) = (&var.col(&var.close, j)[w.end - tail..w.end], &var.col(&var.dollar_volume, j)[w.end - tail..w.end]);
            let mut pc: Vec<f32> = c.iter().copied().filter(|x| x.is_finite()).collect();
            let mut pv: Vec<f32> = c.iter().zip(dv).filter(|(x, _)| x.is_finite()).map(|(_, v)| *v).collect();
            let med = |v: &mut Vec<f32>| {
                let mid = v.len() / 2;
                if v.is_empty() { 0.0 } else { *v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1 }
            };
            med(&mut pc) >= variable::MIN_PRICE && med(&mut pv) >= variable::MIN_DOLLAR_VOLUME
        })
        .collect()
}

fn config_text(cfg: &Config) -> String {
    cfg.to_text()
}

/// Writes `bytes` to `path` through a temporary file in the same directory and a rename, so
/// readers see the old or the new content, never a part.
pub(crate) fn write_atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("rename {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

fn write_atomic(path: &Path, text: &str) -> Result<()> {
    write_atomic_bytes(path, text.as_bytes())
}

/// `write_atomic_bytes`, made durable: the temporary file is flushed to disk before the rename
/// and its directory after it, so once this returns the new content survives a power loss
/// (without the first, a crash can leave the renamed file empty; without the second, the
/// rename itself can be lost).
pub(crate) fn write_durable(path: &Path, bytes: &[u8]) -> Result<()> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    {
        let mut f = File::create(&tmp).with_context(|| format!("write {}", tmp.display()))?;
        f.write_all(bytes).with_context(|| format!("write {}", tmp.display()))?;
        f.sync_all().with_context(|| format!("flush {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("rename {} to {}", tmp.display(), path.display()))?;
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    File::open(dir).and_then(|d| d.sync_all()).with_context(|| format!("flush the directory {}", dir.display()))?;
    Ok(())
}

/// What a run directory's `windows.csv` and `universes.bin` hold for a plan.
fn layout(plan: &Plan) -> (String, Vec<u8>) {
    let mut w = String::from("id,length,first,last,stocks\n");
    let mut u = Vec::new();
    for (win, uni) in plan.windows.iter().zip(&plan.universes) {
        w.push_str(&format!("{},{},{},{},{}\n", win.id, win.length, c2_data::date_string(plan.var.days[win.start]),
                            c2_data::date_string(plan.var.days[win.end - 1]), uni.len()));
        u.extend_from_slice(&(uni.len() as u32).to_le_bytes());
        for &j in uni {
            u.extend_from_slice(&(plan.var.ids[j] as u32).to_le_bytes());
        }
    }
    (w, u)
}

fn run_name(dir: &Path) -> String {
    dir.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string()
}

/// Whether `run::run(store, plan, dir, ..)` would accept `dir`: Ok when it holds no run yet (no
/// manifest; a fresh start) or a run `plan` can resume or find complete, else the error that
/// `run::run` would stop with, before doing anything. Like `run::run`, it refuses a plan made for
/// reading (`Plan::for_run`) that leaves something out. Reads only, and takes no lock (a run
/// being computed right now is `running_pid`'s business). For `c2 update`: when its dated run
/// name is already taken by a run this refuses, it can fall back to a free name instead of
/// failing.
pub fn resume_check(dir: &Path, plan: &Plan, store: &Store) -> Result<()> {
    writing_plan(plan)?;
    if !dir.join("manifest.json").exists() {
        return Ok(());
    }
    let (windows_csv, universes_bin) = layout(plan);
    check_resume(dir, plan, store, &windows_csv, &universes_bin)
}

/// Refuses a plan made for reading a run (`Plan::for_run`) that leaves something out (a
/// correlation period or a stock today's data doesn't have, or a census variable rebuilt from
/// another first day): computing with it would write that gap into a run.
fn writing_plan(plan: &Plan) -> Result<()> {
    let (periods, stocks) = plan.unavailable();
    if let Some(run) = plan.built_for()
        && (periods > 0 || stocks > 0 || plan.variable_note.is_some())
    {
        bail!(
            "this calculation was prepared for reading census {run} ({periods} correlation period(s) and {stocks} stock place(s) left out{}), so it can't compute or resume a census; plan the run afresh (Plan::new)",
            if plan.variable_note.is_some() { ", its census variable rebuilt" } else { "" }
        );
    }
    Ok(())
}

/// The resume guard: the run in `dir` (which has a manifest) must be `plan`'s own: same
/// settings, price data, task list, correlation periods and universes.
fn check_resume(dir: &Path, plan: &Plan, store: &Store, windows_csv: &str, universes_bin: &[u8]) -> Result<()> {
    let name = run_name(dir);
    let cfg = config_text(&plan.cfg);
    let days = serde_json::json!([c2_data::date_string(plan.var.days[0]), c2_data::date_string(*plan.var.days.last().unwrap())]);
    let old: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    if old["config"].as_str() != Some(cfg.as_str()) {
        bail!("census {name} was started with a different config:\n  {}\nnow:\n  {}", old["config"], cfg);
    }
    // Progress is recorded by window position; new data changes the window list, so resuming
    // would file records under the wrong windows.
    if old["store_rows"].as_u64() != Some(store.rows() as u64) || old["variable_days"] != days || old["windows"].as_u64() != Some(plan.windows.len() as u64) {
        bail!(
            "the price data changed since census {name} started (rows {} -> {}, days {} -> {}, correlation periods {} -> {}); \
             start a new run with a different --name",
            old["store_rows"], store.rows(), old["variable_days"], days, old["windows"], plan.windows.len()
        );
    }
    // Progress is also recorded by task id, and what a task id holds depends on the code that
    // made the run (the pairing of move lengths changed once already): the run's own task list
    // must be this plan's, or results would be filed under the wrong move lengths.
    let fresh = || format!("start a new run with a different --name (census {name} itself stays readable)");
    let meta = RunMeta::load(dir).with_context(|| format!("can't check census {name} before resuming it; {}", fresh()))?;
    let theirs = meta.tasks().with_context(|| format!("can't check which move lengths census {name}'s results files hold, so resuming could mix them; {}", fresh()))?;
    if let Some((_, why)) = meta.damaged().first() {
        bail!("{why}; census {name} can't be resumed until that file is readable again, or {}", fresh());
    }
    if let Some(id) = (0..theirs.len().max(plan.tasks.len())).find(|&id| {
        let key = |t: &Task| (t.h1, t.h2, t.length, t.windows.clone());
        theirs.get(id).map(key) != plan.tasks.get(id).map(key)
    }) {
        let say = |t: Option<&Task>| t.map(|t| meta::describe_task(t.h1, t.h2, t.length)).unwrap_or_else(|| "nothing".into());
        bail!(
            "census {name} numbers its results files differently from this version ({} files there, {} now; t{id} holds {} there but {} now), so resuming would mix them; {}",
            theirs.len(), plan.tasks.len(), say(theirs.get(id)), say(plan.tasks.get(id)), fresh()
        );
    }
    if std::fs::read_to_string(dir.join("windows.csv")).ok().as_deref() != Some(windows_csv)
        || std::fs::read(dir.join("universes.bin")).ok().as_deref() != Some(universes_bin)
    {
        bail!("census {name}'s correlation periods or their stocks differ from what this version would use, so resuming would mix them; {}", fresh());
    }
    Ok(())
}

/// Writes the run's description on first start (the manifest last, so a manifest means the
/// rest is there); on resume refuses unless the plan is the run's own (`check_resume`).
fn prepare_dir(dir: &Path, plan: &Plan, store: &Store) -> Result<()> {
    std::fs::create_dir_all(dir.join("tasks"))?;
    let manifest = dir.join("manifest.json");
    let cfg = config_text(&plan.cfg);
    let days = serde_json::json!([c2_data::date_string(plan.var.days[0]), c2_data::date_string(*plan.var.days.last().unwrap())]);
    let (windows_csv, universes_bin) = layout(plan);
    if manifest.exists() {
        return check_resume(dir, plan, store, &windows_csv, &universes_bin);
    }
    // The symbol table the run's store ids refer to: saved with the run, and fingerprinted.
    let symbols = std::fs::read(store.dir().join("symbols.tsv")).context("read the store's symbols.tsv")?;
    let same = std::str::from_utf8(&symbols).ok().and_then(|t| c2_data::store::parse_symbols(t).ok()).is_some_and(|s| {
        s.len() == store.symbols.len() && s.iter().zip(&store.symbols).all(|(a, b)| a.ticker == b.ticker && a.exchange == b.exchange)
    });
    anyhow::ensure!(same, "the store's symbols.tsv no longer matches the open store (was it rebuilt just now?); start the census again");
    let git = std::process::Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    let dirty = std::process::Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty());
    let m = serde_json::json!({
        "config": cfg,
        "git_dirty": dirty,
        "store": c2_data::stores::name(store.dir()),
        "store_symbols": store.symbols.len(),
        "store_rows": store.rows(),
        "symbols_tsv_fnv64": format!("{:016x}", meta::fnv64(&symbols)),
        "variable_days": days,
        "variable_stocks": plan.var.n,
        "windows": plan.windows.len(),
        "tasks": plan.tasks.len(),
        // What each tasks/t<id> file holds: [A's move length, B's move length, period length].
        "task_list": plan.tasks.iter().map(|t| [t.h1, t.h2, t.length]).collect::<Vec<_>>(),
        "git_commit": git.trim(),
        "created_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),
        "record_bytes": RECORD_BYTES,
        "variable": "market-residual daily log return / trailing 252-day residual sd (both through t-1), clipped +-5; horizon sums divided by sqrt(h)",
        "null": "sign-flip: target observations times a random +-1 shared by all stocks on that observation (keeps volatility alignment)",
    });
    write_atomic_bytes(&dir.join("symbols.tsv"), &symbols)?;
    write_atomic(&dir.join("windows.csv"), &windows_csv)?;
    write_atomic_bytes(&dir.join("universes.bin"), &universes_bin)?;
    write_atomic(&manifest, &serde_json::to_string_pretty(&m)?)?;
    Ok(())
}

/// Committed progress per task: (windows done, record bytes, summary bytes).
fn read_progress(dir: &Path) -> Result<HashMap<usize, (usize, u64, u64)>> {
    let mut done = HashMap::new();
    let Ok(f) = File::open(dir.join("progress.log")) else { return Ok(done) };
    for line in BufReader::new(f).lines() {
        let line = line?;
        let v: Vec<u64> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if v.len() == 4 {
            let e = done.entry(v[0] as usize).or_insert((0, 0, 0));
            *e = (e.0 + 1, v[2], v[3]);
        }
    }
    Ok(done)
}

pub fn control_state(dir: &Path) -> String {
    std::fs::read_to_string(dir.join("control")).map(|s| s.trim().to_string()).unwrap_or_else(|_| "run".into())
}

pub fn set_control(dir: &Path, state: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    write_atomic(&dir.join("control"), state)
}

/// Everything about one unit (window x task) that the GPU and the finisher share.
struct UnitData {
    h1: usize,
    h2: usize,
    w: usize,
    n: usize,
    /// GPU matrix width: n rounded up to a multiple of 16 (cuBLASLt INT8 needs it); the extra
    /// columns are all zero and never stored.
    np: usize,
    /// Prefix sums, `(w + 1) x n` column-major.
    prefix: Vec<f64>,
    /// Store ids of the universe columns.
    ids: Vec<u32>,
    /// (lag, observations, sign-flip null seeds) for each computed lag.
    lags: Vec<(usize, usize, Vec<u32>)>,
}

/// A unit prepared on the CPU, waiting for the GPU.
struct Prepared {
    task: usize,
    window: usize,
    pos: usize,
    unit: Arc<UnitData>,
}

/// A unit's raw GPU output, per lag `li`: histograms `[li][1 + nulls][R_BINS]`, cut-offs
/// `[li][3]`, counts `[li][2]` (real, null 0), compacted entries `[li][CAP_*]`.
struct UnitResult {
    task: usize,
    window: usize,
    pos: usize,
    gpu_ms: f64,
    unit: Arc<UnitData>,
    hist: Vec<u32>,
    thr: Vec<f32>,
    cnt: Vec<u32>,
    real: Vec<(i64, f32)>,
    null: Vec<(i64, f32)>,
}

pub const CAP_REAL: usize = 4096;
pub const CAP_NULL: usize = 1024;

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Seed for a sign-flip null: the target's observations are multiplied by a random +-1 that is
/// the same for every stock on an observation. That keeps volatility bursts aligned between
/// leader and target exactly as in the real pairing and destroys only the relationship itself.
/// (A time-shifted null misaligns volatility regimes, which makes real lagged correlations look
/// far more extreme than chance even with no lead-lag at all.)
fn null_seed(task: usize, window: usize, lag: usize, t: usize) -> u32 {
    (splitmix((task as u64) << 40 ^ (window as u64) << 20 ^ (lag as u64) << 4 ^ t as u64) as u32) | 1
}

/// Builds a unit's prefix sums and lag list on the CPU.
fn prepare(plan: &Plan, task: &Task, pos: usize) -> Prepared {
    let cfg = &plan.cfg;
    let wid = task.windows[pos];
    let win = &plan.windows[wid];
    let uni = &plan.universes[wid];
    let (w, n) = (win.end - win.start, uni.len());
    let (h1, h2) = (task.h1, task.h2);
    let mut lags: Vec<(usize, usize, Vec<u32>)> = grid::lags(h2)
        .into_iter()
        .map(|k| (k, grid::observations(w, h1, h2, k), Vec::new()))
        .filter(|&(_, m, _)| m >= cfg.min_obs)
        .collect();
    if n < 2 {
        lags.clear();
    }
    for (k, _, seeds) in lags.iter_mut() {
        *seeds = (0..cfg.nulls).map(|t| null_seed(task.id, wid, *k, t)).collect();
    }
    let ld = w + 1;
    let mut prefix = vec![0f64; if lags.is_empty() { 0 } else { ld * n }];
    prefix.par_chunks_mut(ld).zip(uni.par_iter()).for_each(|(p, &j)| {
        let col = &plan.var.col(&plan.var.v, j)[win.start..win.end];
        for r in 0..w {
            let v = col[r];
            p[r + 1] = p[r] + if v.is_finite() { v as f64 } else { 0.0 };
        }
    });
    let unit = UnitData {
        h1, h2, w, n, np: pad_width(n), prefix,
        ids: uni.iter().map(|&j| plan.var.ids[j] as u32).collect(),
        lags,
    };
    Prepared { task: task.id, window: wid, pos, unit: Arc::new(unit) }
}

/// Exact Pearson r and sums for leader column `i`, target column `j` at `lag`; `seed` non-zero
/// applies that sign-flip null to the target.
fn exact(u: &UnitData, i: usize, j: usize, lag: usize, m: usize, seed: u32) -> (f64, [f64; 5]) {
    let (s, g0, ld) = (u.h1.max(u.h2), u.h1.max(u.h2) - 1, u.w + 1);
    let (pi, pj) = (&u.prefix[i * ld..(i + 1) * ld], &u.prefix[j * ld..(j + 1) * ld]);
    let (sh1, sh2) = ((u.h1 as f64).sqrt(), (u.h2 as f64).sqrt());
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for q in 0..m {
        let e = g0 + q * s;
        let x = (pi[e + 1] - pi[e + 1 - u.h1]) / sh1;
        let et = g0 + q * s + lag;
        let mut y = (pj[et + 1] - pj[et + 1 - u.h2]) / sh2;
        if seed != 0 {
            y *= c2_engine::sign_flip(seed, q as u32);
        }
        sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
    }
    (c2_engine::pearson(sxy, sx, sy, sxx, syy, m as f64), [sxy, sx, sy, sxx, syy])
}

/// The false-discovery rule run with null 1 standing in for the real matrix (against the mean
/// of the other nulls and the analytic floor): how often the rule fires with nothing there.
/// Returns that cut-off in |r| if it fires.
fn placebo_cutoff(h: &[u32], nh: usize, n: usize, m: usize, q: f64) -> Option<f64> {
    if nh < 3 {
        return None;
    }
    let others: Vec<usize> = (1..nh).filter(|&t| t != 2).collect();
    let pairs = (n * n.saturating_sub(1)) as f64;
    let zs = ((m as f64 - 3.0).max(1.0)).sqrt();
    let (mut cr, mut cn, mut best) = (0f64, 0f64, None);
    for b in (0..R_BINS).rev() {
        cr += h[2 * R_BINS + b] as f64;
        cn += others.iter().map(|&t| h[t * R_BINS + b] as f64).sum::<f64>() / others.len() as f64;
        let r = (b as f64 / R_BINS as f64).min(0.999_999);
        let analytic = pairs * erfc(r.atanh() * zs / std::f64::consts::SQRT_2);
        if cr >= 10.0 && cn.max(analytic) / cr <= q {
            best = Some(b as f64 / R_BINS as f64);
        }
    }
    best
}

/// Complementary error function (Numerical Recipes erfcc; relative error < 1.2e-7).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.265_512_23 + t * (1.000_023_68 + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06
        + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))))).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

/// Sum of histogram bins at or above |z| for `m` observations.
fn tail(h: &[u32], z: f64, m: usize) -> u64 {
    let r = (z / ((m as f64 - 3.0).max(1.0)).sqrt()).tanh();
    let b = ((r * R_BINS as f64).ceil() as usize).min(R_BINS);
    h[b..].iter().map(|&v| v as u64).sum()
}

const R_BINS: usize = 1024;

/// Turns a unit's GPU output into records and summary lines.
fn finish_unit(res: &UnitResult, cfg: &Config, out: &mut Vec<u8>, csv: &mut String) {
    let u = &res.unit;
    let nh = 1 + cfg.nulls;
    // The GPU matrices are np x np; entries in padding columns (>= n) are not pairs. Their r is
    // exactly 0 (zero data and zero stats), so they only land in the lowest |r| bin and are only
    // pulled off the GPU when a cut-off is 0, i.e. when the matrix has fewer real pairs than it
    // keeps. Drop them before choosing what to keep, so the counts written agree with the records.
    let np = u.np as i64;
    let is_pair = |&(idx, _): &(i64, f32)| ((idx % np) as usize) < u.n && ((idx / np) as usize) < u.n;
    // (lag index, entry, null?, flags)
    let mut chosen: Vec<(usize, i64, f32, bool, u8)> = Vec::new();
    for (li, (k, m, _)) in u.lags.iter().enumerate() {
        let thr = &res.thr[li * 3..li * 3 + 3];
        let (rc, nc) = (res.cnt[li * 2] as usize, res.cnt[li * 2 + 1] as usize);
        let pulled = &res.real[li * CAP_REAL..li * CAP_REAL + rc.min(CAP_REAL)];
        let mut real: Vec<(i64, f32)> = pulled.iter().copied().filter(is_pair).collect();
        // Entries pulled off the GPU, padding excluded. Exact unless a cut-off of 0 met a matrix
        // too big for the cap, which needs fewer than keep_real pairs with |r| >= 1/1024 among
        // more than 64 stocks (with real data a cut-off of 0 means 10 stocks or fewer).
        let screened = rc - (pulled.len() - real.len());
        // Strongest first; equal |r| by matrix index, so which pairs a tie at the keep boundary
        // keeps, and the order records are written in, never depend on the GPU's order.
        let strongest = |a: &(i64, f32), b: &(i64, f32)| b.1.abs().total_cmp(&a.1.abs()).then(a.0.cmp(&b.0));
        real.sort_by(strongest);
        let fdr = if *k > 0 && thr[2] <= 1.0 { Some(thr[2]) } else { None };
        let mut stored = 0;
        for (rank, &(idx, r8)) in real.iter().enumerate() {
            let top = rank < cfg.keep_real;
            let sig = fdr.is_some_and(|t| r8.abs().min(0.999_999) >= t);
            if (!top && !sig) || stored >= cfg.keep_real + cfg.fdr_cap {
                continue;
            }
            let mut flags = 0;
            if top { flags |= FLAG_TOP; }
            if sig { flags |= FLAG_FDR; }
            if rc > CAP_REAL { flags |= FLAG_TRUNCATED; }
            chosen.push((li, idx, r8, false, flags));
            stored += 1;
        }
        let mut null: Vec<(i64, f32)> = res.null[li * CAP_NULL..li * CAP_NULL + nc.min(CAP_NULL)].iter().copied().filter(is_pair).collect();
        null.sort_by(strongest);
        let stored_null = null.len().min(cfg.keep_null);
        for &(idx, r8) in null.iter().take(cfg.keep_null) {
            chosen.push((li, idx, r8, true, FLAG_NULL | FLAG_TOP));
        }
        let h = &res.hist[li * nh * R_BINS..(li + 1) * nh * R_BINS];
        let real_h = &h[..R_BINS];
        let null_mean = |z: f64| (1..nh).map(|t| tail(&h[t * R_BINS..(t + 1) * R_BINS], z, *m) as f64).sum::<f64>() / (nh - 1).max(1) as f64;
        let placebo = if *k > 0 { placebo_cutoff(h, nh, u.n, *m, cfg.fdr_q) } else { None };
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{:.1},{:.1},{:.1},{},{},{},{},{},{}\n",
            res.window, u.h1, u.h2, k, u.n, m,
            tail(real_h, 3.0, *m), tail(real_h, 4.0, *m), tail(real_h, 5.0, *m), null_mean(3.0), null_mean(4.0), null_mean(5.0),
            fdr.map(|t| format!("{t:.4}")).unwrap_or_default(), stored, stored_null, screened, u8::from(rc > CAP_REAL),
            placebo.map(|t| format!("{t:.4}")).unwrap_or_default(),
        ));
    }
    let records: Vec<Record> = chosen
        .par_iter()
        .map(|&(li, idx, r8, null, flags)| {
            let (k, m, seeds) = &u.lags[li];
            let (i, j) = ((idx % np) as usize, (idx / np) as usize);
            let seed = if null { seeds[0] } else { 0 };
            let (r, s) = exact(u, i, j, *k, *m, seed);
            let zf = ((*m as f64 - 3.0).max(1.0)).sqrt();
            Record {
                window: res.window as u32,
                h1: u.h1 as u8,
                h2: u.h2 as u8,
                lag: *k as u16,
                flags,
                obs: *m as u16,
                leader: u.ids[i],
                target: u.ids[j],
                r: r as f32,
                r_int8: r8,
                z: (r.clamp(-0.999_999, 0.999_999).atanh() * zf) as f32,
                sxy: s[0] as f32, sx: s[1] as f32, sy: s[2] as f32, sxx: s[3] as f32, syy: s[4] as f32,
            }
        })
        .collect();
    for rec in &records {
        rec.to_bytes(out);
    }
}

pub const SUMMARY_HEADER: &str = "window,h1,h2,lag,stocks,obs,real_z3,real_z4,real_z5,null_z3,null_z4,null_z5,fdr_r,stored_real,stored_null,screened,truncated,placebo_fdr_r\n";

/// Status shared with `c2 census status` and the GUI.
fn write_status(dir: &Path, state: &str, done_units: usize, total_units: usize, cost_done: f64, cost_total: f64, rate: f64, started: Instant, note: &str) {
    let eta = if rate > 0.0 { (cost_total - cost_done) / rate } else { f64::NAN };
    let s = serde_json::json!({
        "state": state,
        "units_done": done_units,
        "units_total": total_units,
        "fraction_done": if cost_total > 0.0 { cost_done / cost_total } else { 0.0 },
        "eta_seconds": if eta.is_finite() { eta } else { -1.0 },
        "elapsed_seconds_this_session": started.elapsed().as_secs_f64(),
        "updated_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "note": note,
    });
    let _ = write_atomic(&dir.join("status.json"), &serde_json::to_string_pretty(&s).unwrap_or_default());
}

/// Runs (or resumes) the census in `dir` until finished or `stop` is set. Progress lines go to
/// `log` every ~15 s. Holds `<dir>/lock` throughout, so a second writer on the same run fails
/// at once ("census X is already running (pid N)"). Refuses, before touching `dir`, a plan made
/// for reading (`Plan::for_run`) that leaves out periods or stocks or rebuilt the variable.
/// `lock_run` + `run_locked` do the same in two steps, for a caller that must know it is the
/// writer before it logs anything for the run.
pub fn run(store: &Store, plan: &Plan, dir: &Path, stop: &AtomicBool, log: &(dyn Fn(&str) + Sync)) -> Result<()> {
    writing_plan(plan)?;
    run_locked(store, plan, dir, lock_run(dir)?, stop, log)
}

/// Takes the writer lock of the census run in `dir` (creating the directory), as `run` does: a
/// second writer fails at once ("census X is already running (pid N)"). Held until the returned
/// guard is dropped; pass it to `run_locked`.
pub fn lock_run(dir: &Path) -> Result<RunLock> {
    std::fs::create_dir_all(dir)?;
    RunLock::acquire(dir)
}

/// `run` for a caller that already holds the run's writer lock (`lock_run` on the same `dir`).
pub fn run_locked(store: &Store, plan: &Plan, dir: &Path, lock: RunLock, stop: &AtomicBool, log: &(dyn Fn(&str) + Sync)) -> Result<()> {
    writing_plan(plan)?;
    anyhow::ensure!(lock.pid_path.parent() == Some(dir), "the lock passed is for {}, not for census {}", lock.pid_path.display(), run_name(dir));
    let _lock = lock;
    prepare_dir(dir, plan, store)?;
    let progress = read_progress(dir)?;
    let total_units: usize = plan.tasks.iter().map(|t| t.windows.len()).sum();
    let cost_total = plan.total_cost();
    let mut cost_done = 0.0;
    let mut units_done = 0;
    // Truncate every task's files to its last commit.
    for t in &plan.tasks {
        let (wins, rec, csv) = progress.get(&t.id).copied().unwrap_or((0, 0, 0));
        let f = OpenOptions::new().create(true).write(true).truncate(false).open(dir.join(format!("tasks/t{}.rec", t.id)))?;
        f.set_len(rec)?;
        let f = OpenOptions::new().create(true).write(true).truncate(false).open(dir.join(format!("tasks/t{}.csv", t.id)))?;
        f.set_len(csv)?;
        cost_done += plan.cost[t.id][..wins.min(t.windows.len())].iter().sum::<f64>();
        units_done += wins.min(t.windows.len());
    }
    if units_done == total_units {
        write_status(dir, "finished", units_done, total_units, cost_done, cost_total, 0.0, Instant::now(), "");
        log("census already complete");
        return Ok(());
    }

    let started = Instant::now();
    let (prep_tx, prep_rx) = mpsc::sync_channel::<Prepared>(2);
    let (res_tx, res_rx) = mpsc::sync_channel::<UnitResult>(4);
    let cfg = plan.cfg.clone();
    let dir_f = dir.to_path_buf();
    std::thread::scope(|scope| -> Result<()> {
        // CPU: prefix sums for upcoming units, while the GPU works on the current one.
        let progress_ref = &progress;
        scope.spawn(move || {
            for task in &plan.tasks {
                let start = progress_ref.get(&task.id).map(|p| p.0).unwrap_or(0);
                for pos in start..task.windows.len() {
                    if stop.load(Ordering::Relaxed) || prep_tx.send(prepare(plan, task, pos)).is_err() {
                        return;
                    }
                }
            }
        });
        // CPU: exact recomputation, writing and committing finished units.
        let finisher = scope.spawn(move || -> Result<()> {
            let mut files: HashMap<usize, (File, File)> = HashMap::new();
            let mut prog = OpenOptions::new().create(true).append(true).open(dir_f.join("progress.log"))?;
            let (mut cost_done, mut units_done) = (cost_done, units_done);
            let (session_start_cost, mut last_status, mut last_log) = (cost_done, Instant::now(), Instant::now());
            for res in res_rx {
                let (mut buf, mut csv) = (Vec::new(), String::new());
                finish_unit(&res, &cfg, &mut buf, &mut csv);
                let (rf, cf) = match files.entry(res.task) {
                    std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
                    std::collections::hash_map::Entry::Vacant(v) => {
                        let rf = OpenOptions::new().append(true).open(dir_f.join(format!("tasks/t{}.rec", res.task)))?;
                        let mut cf = OpenOptions::new().append(true).open(dir_f.join(format!("tasks/t{}.csv", res.task)))?;
                        if cf.metadata()?.len() == 0 {
                            cf.write_all(SUMMARY_HEADER.as_bytes())?;
                        }
                        v.insert((rf, cf))
                    }
                };
                rf.write_all(&buf)?;
                cf.write_all(csv.as_bytes())?;
                rf.sync_data()?;
                cf.sync_data()?;
                writeln!(prog, "{} {} {} {}", res.task, res.window, rf.metadata()?.len(), cf.metadata()?.len())?;
                prog.sync_data()?;
                cost_done += plan.cost[res.task][res.pos];
                units_done += 1;
                let rate = (cost_done - session_start_cost) / started.elapsed().as_secs_f64().max(1e-3);
                if last_status.elapsed() > Duration::from_secs(2) {
                    write_status(&dir_f, "running", units_done, total_units, cost_done, cost_total, rate, started, "");
                    last_status = Instant::now();
                }
                if last_log.elapsed() > Duration::from_secs(15) {
                    let eta = (cost_total - cost_done) / rate.max(1e-9);
                    log(&format!("{units_done}/{total_units} correlation-period computations ({:.2}% of the work), last one {:.0} ms on the GPU, ETA {}",
                        100.0 * cost_done / cost_total, res.gpu_ms, hms(eta)));
                    last_log = Instant::now();
                }
            }
            let rate = (cost_done - session_start_cost) / started.elapsed().as_secs_f64().max(1e-3);
            let state = if units_done == total_units { "finished" } else { "stopped" };
            write_status(&dir_f, state, units_done, total_units, cost_done, cost_total, rate, started, "");
            log(&format!("{state}: {units_done}/{total_units} correlation-period computations committed"));
            Ok(())
        });
        let worker = gpu_worker(plan, prep_rx, res_tx, stop, dir, log);
        let fin = finisher.join().map_err(|_| anyhow::anyhow!("finisher thread panicked"))?;
        worker.and(fin)
    })
}

/// The one writer's hold on a run directory: an exclusive `flock` on `<dir>/lock` for as long
/// as this lives. The holder's pid is written into the lock file ("pid\n", read back only when
/// complete) and into `<dir>/pid` (for older readers). Dropping it removes `pid` if it is still
/// ours; the lock goes with the file. Taken by `run` itself, or by `lock_run`.
pub struct RunLock {
    file: File,
    pid_path: PathBuf,
}

impl RunLock {
    fn acquire(dir: &Path) -> Result<RunLock> {
        let name = run_name(dir);
        let path = dir.join("lock");
        let mut file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path).with_context(|| format!("open {}", path.display()))?;
        // A status check holds the lock for an instant while it tests it, so retry briefly.
        let mut tries = 0;
        loop {
            match file.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) if tries < 10 => {
                    tries += 1;
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    return Err(match holder_pid(dir) {
                        Some(pid) => anyhow!("census {name} is already running (pid {pid})"),
                        None => anyhow!("census {name} is already running (another process holds {})", path.display()),
                    });
                }
                Err(std::fs::TryLockError::Error(e)) => return Err(e).with_context(|| format!("lock {}", path.display())),
            }
        }
        // Older c2 binaries take no lock but leave their pid.
        if let Some(pid) = legacy_pid(dir) {
            bail!("census {name} is already running (pid {pid})");
        }
        let pid = std::process::id();
        file.set_len(0)?;
        file.write_all(format!("{pid}\n").as_bytes())?;
        file.sync_data()?;
        let pid_path = dir.join("pid");
        write_atomic(&pid_path, &pid.to_string())?;
        Ok(RunLock { file, pid_path })
    }
}

impl Drop for RunLock {
    fn drop(&mut self) {
        // Still holding the lock: clear the pid so no later reader can take it for a holder's.
        let _ = self.file.set_len(0);
        if std::fs::read_to_string(&self.pid_path).is_ok_and(|p| p.trim() == std::process::id().to_string()) {
            let _ = std::fs::remove_file(&self.pid_path);
        }
    }
}

/// True when some process holds `<dir>/lock` (tested by trying to take it for an instant).
fn lock_held(dir: &Path) -> bool {
    let Ok(f) = File::open(dir.join("lock")) else { return false };
    matches!(f.try_lock(), Err(std::fs::TryLockError::WouldBlock))
}

/// `pid` from `path` if it is a live `c2*` process that started before `path` was last written
/// (so not a pid reused since). `path` holds "pid", or "pid\n" when `complete_line`.
fn live_c2_pid(path: &Path, complete_line: bool) -> Option<u32> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = if complete_line { text.strip_suffix('\n')? } else { text.trim() };
    let pid: u32 = text.parse().ok().filter(|&p| p > 0)?;
    let exe = std::fs::read_link(format!("/proc/{pid}/exe")).ok()?;
    if !exe.file_name()?.to_str()?.starts_with("c2") {
        return None;
    }
    let written = std::fs::metadata(path).ok()?.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs_f64();
    match process_start_unix(pid) {
        Some(started) if started > written + 2.0 => None,
        _ => Some(pid),
    }
}

/// The pid of the process holding `<dir>/lock`, as it wrote it into the lock file (else into
/// `<dir>/pid`), checked to be a live c2 process that wrote it.
fn holder_pid(dir: &Path) -> Option<u32> {
    for _ in 0..10 {
        if let Some(pid) = live_c2_pid(&dir.join("lock"), true) {
            return Some(pid);
        }
        // Just locked, pid not written yet.
        std::thread::sleep(Duration::from_millis(10));
    }
    live_c2_pid(&dir.join("pid"), false)
}

/// Seconds since 1970 at which process `pid` started (Linux: /proc/stat's boot time plus the
/// process's start in clock ticks, which /proc reports at 100 per second).
fn process_start_unix(pid: u32) -> Option<f64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // Fields after the command name, which is in parentheses and may contain spaces; the start
    // time is field 22 overall, the 20th after the name.
    let after = &stat[stat.rfind(')')? + 1..];
    let ticks: f64 = after.split_whitespace().nth(19)?.parse().ok()?;
    let boot: f64 = std::fs::read_to_string("/proc/stat").ok()?.lines().find_map(|l| l.strip_prefix("btime "))?.trim().parse().ok()?;
    Some(boot + ticks / 100.0)
}

/// A live `c2*` process other than this one, named in `<dir>/pid` and started before the file
/// was written: how runs started by older binaries, which take no lock, show they are running.
fn legacy_pid(dir: &Path) -> Option<u32> {
    live_c2_pid(&dir.join("pid"), false).filter(|&pid| pid != std::process::id())
}

/// The pid of the process running the census in `dir`, if one is: whoever holds `<dir>/lock`
/// (taken by `run` for its whole duration), or, for runs started by older binaries, a live
/// `c2*` process named in `<dir>/pid`. Never returns 0.
pub fn running_pid(dir: &Path) -> Option<u32> {
    if lock_held(dir) { holder_pid(dir) } else { legacy_pid(dir) }
}

/// Names of the census runs that are running now (see `running_pid`), sorted.
pub fn running_runs() -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(c2_data::data_dir().join("census")) else { return Vec::new() };
    let mut out: Vec<String> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && running_pid(p).is_some())
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
        .collect();
    out.sort();
    out
}

/// cuBLASLt's INT8 kernels need matrix widths that are multiples of 4; use 16.
pub fn pad_width(n: usize) -> usize {
    n.div_ceil(16) * 16
}

pub fn hms(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "unknown".into();
    }
    let s = secs as u64;
    format!("{}h{:02}m{:02}s", s / 3600, s / 60 % 60, s % 60)
}

#[cfg(feature = "cuda")]
fn gpu_worker(
    plan: &Plan, prep: mpsc::Receiver<Prepared>, out: mpsc::SyncSender<UnitResult>, stop: &AtomicBool, dir: &Path,
    log: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    use c2_engine::gpu::{Gpu, LtPlan, Precision, R_BINS};
    use c2_engine::gpu::cudarc::driver::CudaSlice;
    let cfg = &plan.cfg;
    let nh = 1 + cfg.nulls;

    // Largest shapes in the plan, so buffers are allocated once.
    let (mut np_max, mut kp_max, mut pre_max, mut lags_max) = (16usize, 32usize, 1usize, 1usize);
    for t in &plan.tasks {
        for &wid in &t.windows {
            let w = plan.windows[wid].end - plan.windows[wid].start;
            let np = pad_width(plan.universes[wid].len());
            np_max = np_max.max(np);
            kp_max = kp_max.max(grid::observations(w, t.h1, t.h2, 0).div_ceil(32) * 32);
            pre_max = pre_max.max((w + 1) * plan.universes[wid].len());
            lags_max = lags_max.max(grid::lags(t.h2).len());
        }
    }
    struct Bufs {
        pre: CudaSlice<f64>,
        x: CudaSlice<i8>,
        y: CudaSlice<i8>,
        xs: CudaSlice<f32>, xq: CudaSlice<f32>, ys: CudaSlice<f32>, yq: CudaSlice<f32>,
        c_real: CudaSlice<i32>, c_null: CudaSlice<i32>, c_tmp: CudaSlice<i32>,
        f: c2_engine::gpu::PassFactors,
        f_null: c2_engine::gpu::PassFactors,
        f_tmp: c2_engine::gpu::PassFactors,
        hist: CudaSlice<u32>, thr: CudaSlice<f32>, cnt: CudaSlice<u32>,
        ri: CudaSlice<i64>, rr: CudaSlice<f32>, ni: CudaSlice<i64>, nr: CudaSlice<f32>,
    }
    let alloc = |g: &Gpu| -> Result<Bufs> {
        let s = &g.stream;
        Ok(Bufs {
            pre: s.alloc_zeros(pre_max)?,
            x: s.alloc_zeros(kp_max * np_max)?,
            y: s.alloc_zeros(kp_max * np_max)?,
            xs: s.alloc_zeros(np_max)?, xq: s.alloc_zeros(np_max)?, ys: s.alloc_zeros(np_max)?, yq: s.alloc_zeros(np_max)?,
            c_real: s.alloc_zeros(np_max * np_max)?, c_null: s.alloc_zeros(np_max * np_max)?, c_tmp: s.alloc_zeros(np_max * np_max)?,
            f: g.alloc_factors(np_max)?,
            f_null: g.alloc_factors(np_max)?,
            f_tmp: g.alloc_factors(np_max)?,
            hist: s.alloc_zeros(lags_max * nh * R_BINS)?,
            thr: s.alloc_zeros(lags_max * 3)?,
            cnt: s.alloc_zeros(lags_max * 2)?,
            ri: s.alloc_zeros(lags_max * CAP_REAL)?, rr: s.alloc_zeros(lags_max * CAP_REAL)?,
            ni: s.alloc_zeros(lags_max * CAP_NULL)?, nr: s.alloc_zeros(lags_max * CAP_NULL)?,
        })
    };
    let mut gpu: Option<(Gpu, Bufs)> = None;
    let mut plans: HashMap<(usize, usize), LtPlan> = HashMap::new();
    for p in prep {
        // Pause: release the GPU entirely until resumed.
        while control_state(dir) == "pause" && !stop.load(Ordering::Relaxed) {
            if gpu.take().is_some() {
                plans.clear();
                log("paused: GPU released");
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        if gpu.is_none() {
            let g = Gpu::new()?;
            let b = alloc(&g)?;
            gpu = Some((g, b));
        }
        let (g, b) = gpu.as_mut().unwrap();
        let st = g.stream.clone();
        let t0 = Instant::now();
        let u = &p.unit;
        let lags = u.lags.len();
        if lags == 0 {
            out.send(UnitResult { task: p.task, window: p.window, pos: p.pos, gpu_ms: 0.0, unit: p.unit.clone(),
                hist: vec![], thr: vec![], cnt: vec![], real: vec![], null: vec![] })?;
            continue;
        }
        let (n, np, ld) = (u.n, u.np, u.w + 1);
        let (h1, h2, step, g0) = (u.h1, u.h2, u.h1.max(u.h2), u.h1.max(u.h2) - 1);
        let m0 = u.lags[0].1;
        let kp = m0.div_ceil(32) * 32;
        st.memcpy_htod(&u.prefix, &mut b.pre.slice_mut(0..u.prefix.len()))?;
        st.memset_zeros(&mut b.hist)?;
        st.memset_zeros(&mut b.cnt)?;
        if !plans.contains_key(&(np, kp)) {
            plans.insert((np, kp), g.plan(Precision::Int8, np, kp)?);
        }
        let lt = &plans[&(np, kp)];
        // Leader panel once per unit; zero any padding columns left from a wider unit.
        st.memset_zeros(&mut b.x)?;
        st.memset_zeros(&mut b.y)?;
        // The kernels scan the padded width np; build_panel only writes stats for the n real
        // columns, so clear the rest or a narrower unit inherits the previous unit's stats in its
        // padding columns (zero stats give r = 0 there).
        for v in [&mut b.xs, &mut b.xq, &mut b.ys, &mut b.yq] {
            st.memset_zeros(v)?;
        }
        g.build_panel(&b.pre, ld, n, g0, step, 0, h1, m0, 0, 0, kp, &mut b.x, &mut b.xs, &mut b.xq)?;
        for (li, (k, m, seeds)) in u.lags.iter().enumerate() {
            let (k, m) = (*k, *m);
            let hb = li * nh * R_BINS;
            g.build_panel(&b.pre, ld, n, g0, step, k, h2, m, 0, 0, kp, &mut b.y, &mut b.ys, &mut b.yq)?;
            g.col_stats(&b.x, kp, m, np, &mut b.xs, &mut b.xq)?;
            g.factors(&b.xs, &b.xq, &b.ys, &b.yq, np, m, &mut b.f)?;
            g.xprod_i8(lt, &b.x, &b.y, &mut b.c_real)?;
            g.census_hist(&b.c_real, np, &b.f, &mut b.hist.slice_mut(hb..hb + R_BINS))?;
            for (t, &seed) in seeds.iter().enumerate() {
                // Sign flips change the target's sums, so each null gets its own factors.
                g.build_panel(&b.pre, ld, n, g0, step, k, h2, m, 0, seed, kp, &mut b.y, &mut b.ys, &mut b.yq)?;
                let (c, f) = if t == 0 { (&mut b.c_null, &mut b.f_null) } else { (&mut b.c_tmp, &mut b.f_tmp) };
                g.factors(&b.xs, &b.xq, &b.ys, &b.yq, np, m, f)?;
                g.xprod_i8(lt, &b.x, &b.y, c)?;
                let off = hb + (1 + t) * R_BINS;
                g.census_hist(c, np, f, &mut b.hist.slice_mut(off..off + R_BINS))?;
            }
            g.choose_thresholds(&b.hist.slice(hb..hb + nh * R_BINS), cfg.nulls, cfg.keep_real, cfg.keep_null, cfg.fdr_cap,
                cfg.fdr_q as f32, k > 0, n, m, &mut b.thr.slice_mut(li * 3..li * 3 + 3))?;
            let thr = b.thr.slice(li * 3..li * 3 + 3);
            g.census_compact(&b.c_real, np, &b.f, &thr, 0, &mut b.cnt.slice_mut(li * 2..li * 2 + 1), CAP_REAL,
                &mut b.ri.slice_mut(li * CAP_REAL..(li + 1) * CAP_REAL), &mut b.rr.slice_mut(li * CAP_REAL..(li + 1) * CAP_REAL))?;
            if !seeds.is_empty() {
                g.census_compact(&b.c_null, np, &b.f_null, &thr, 1, &mut b.cnt.slice_mut(li * 2 + 1..li * 2 + 2), CAP_NULL,
                    &mut b.ni.slice_mut(li * CAP_NULL..(li + 1) * CAP_NULL), &mut b.nr.slice_mut(li * CAP_NULL..(li + 1) * CAP_NULL))?;
            }
        }
        let hist = st.clone_dtoh(&b.hist.slice(0..lags * nh * R_BINS))?;
        let thr = st.clone_dtoh(&b.thr.slice(0..lags * 3))?;
        let cnt = st.clone_dtoh(&b.cnt.slice(0..lags * 2))?;
        let pairs = |idx: &CudaSlice<i64>, val: &CudaSlice<f32>, cap: usize| -> Result<Vec<(i64, f32)>> {
            let a = st.clone_dtoh(&idx.slice(0..lags * cap))?;
            let v = st.clone_dtoh(&val.slice(0..lags * cap))?;
            Ok(a.into_iter().zip(v).collect())
        };
        let real = pairs(&b.ri, &b.rr, CAP_REAL)?;
        let null = pairs(&b.ni, &b.nr, CAP_NULL)?;
        out.send(UnitResult {
            task: p.task, window: p.window, pos: p.pos, gpu_ms: t0.elapsed().as_secs_f64() * 1e3,
            unit: p.unit.clone(), hist, thr, cnt, real, null,
        })?;
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn gpu_worker(_: &Plan, _: mpsc::Receiver<Prepared>, _: mpsc::SyncSender<UnitResult>, _: &AtomicBool, _: &Path, _: &(dyn Fn(&str) + Sync)) -> Result<()> {
    bail!("this build has no GPU support")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A matrix of 5 stocks on a 16-wide GPU matrix: every cut-off is 0, so the GPU hands back
    /// all 240 entries, 220 of them padding with r = 0. The counts written must match the
    /// records: 20 real pairs and 20 null pairs, not 100 and 50 padded with non-pairs.
    #[test]
    fn padding_never_counts_as_stored() {
        let cfg = Config::smoke();
        let (n, w) = (5usize, 40usize);
        let mut prefix = vec![0f64; (w + 1) * n];
        for j in 0..n {
            for r in 0..w {
                let v = ((r * 7 + j * 13) % 11) as f64 - 5.0;
                prefix[j * (w + 1) + r + 1] = prefix[j * (w + 1) + r] + v;
            }
        }
        let lags = vec![(0, 40, vec![3, 5, 7]), (1, 39, vec![9, 11, 13])];
        let unit = UnitData { h1: 1, h2: 1, w, n, np: pad_width(n), prefix, ids: vec![10, 11, 12, 13, 14], lags };
        let np = unit.np as i64;
        let all: Vec<(i64, f32)> = (0..np * np)
            .filter(|&idx| idx % np != idx / np)
            .map(|idx| (idx, if idx % np < n as i64 && idx / np < n as i64 { 0.1 + (idx % 7) as f32 * 0.01 } else { 0.0 }))
            .collect();
        assert_eq!(all.len(), 240);
        let block = |cap: usize| -> Vec<(i64, f32)> {
            let mut v = vec![(0i64, 0f32); 2 * cap];
            for li in 0..2 {
                v[li * cap..li * cap + all.len()].copy_from_slice(&all);
            }
            v
        };
        let nh = 1 + cfg.nulls;
        let res = UnitResult {
            task: 0, window: 3, pos: 0, gpu_ms: 0.0, unit: Arc::new(unit),
            hist: vec![0; 2 * nh * R_BINS],
            thr: vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0],
            cnt: vec![240, 240, 240, 240],
            real: block(CAP_REAL),
            null: block(CAP_NULL),
        };
        let (mut out, mut csv) = (Vec::new(), String::new());
        finish_unit(&res, &cfg, &mut out, &mut csv);
        let recs: Vec<Record> = out.chunks_exact(RECORD_BYTES).map(Record::from_bytes).collect();
        assert!(recs.iter().all(|r| r.leader >= 10 && r.target >= 10 && r.leader != r.target));
        for (li, line) in csv.lines().enumerate() {
            let f: Vec<&str> = line.split(',').collect();
            let (real, null) = (recs.iter().filter(|r| r.lag as usize == li && r.flags & FLAG_NULL == 0).count(), recs.iter().filter(|r| r.lag as usize == li && r.flags & FLAG_NULL != 0).count());
            assert_eq!((real, null), (20, 20));
            // stored_real, stored_null, screened
            assert_eq!((f[13], f[14], f[15]), ("20", "20", "20"), "{line}");
        }
    }

    /// Equal |r| at the keep boundary: whatever order the GPU hands the entries back in, the
    /// same pairs are kept, in the same order (the lower matrix index first), byte for byte.
    #[test]
    fn ties_keep_the_same_pairs_in_the_same_order() {
        let cfg = Config { keep_real: 3, keep_null: 2, ..Config::smoke() };
        let (n, w) = (5usize, 40usize);
        let mut prefix = vec![0f64; (w + 1) * n];
        for j in 0..n {
            for r in 0..w {
                prefix[j * (w + 1) + r + 1] = prefix[j * (w + 1) + r] + (((r * 5 + j * 3) % 7) as f64 - 3.0);
            }
        }
        let unit = Arc::new(UnitData { h1: 1, h2: 1, w, n, np: pad_width(n), prefix, ids: vec![20, 21, 22, 23, 24], lags: vec![(0, 40, vec![3, 5, 7])] });
        let np = unit.np as i64;
        // The 20 pairs: two strong ones, then four tied at |r| 0.5 (signs differ) where the third
        // kept place falls, then weak ones.
        let pairs: Vec<(i64, f32)> = (0..np * np)
            .filter(|&idx| idx % np < n as i64 && idx / np < n as i64 && idx % np != idx / np)
            .enumerate()
            .map(|(k, idx)| (idx, match k { 0 | 1 => 0.9, 2..=5 => if k % 2 == 0 { 0.5 } else { -0.5 }, _ => 0.1 }))
            .collect();
        let run = |entries: &[(i64, f32)]| {
            let mut real = vec![(0i64, 0f32); CAP_REAL];
            real[..entries.len()].copy_from_slice(entries);
            let mut null = vec![(0i64, 0f32); CAP_NULL];
            null[..entries.len()].copy_from_slice(entries);
            let res = UnitResult {
                task: 0, window: 1, pos: 0, gpu_ms: 0.0, unit: unit.clone(),
                hist: vec![0; (1 + cfg.nulls) * R_BINS],
                thr: vec![0.0, 0.0, 2.0],
                cnt: vec![entries.len() as u32, entries.len() as u32],
                real, null,
            };
            let (mut out, mut csv) = (Vec::new(), String::new());
            finish_unit(&res, &cfg, &mut out, &mut csv);
            (out, csv)
        };
        let forward = run(&pairs);
        let mut reversed = pairs.clone();
        reversed.reverse();
        let mut shuffled = pairs.clone();
        shuffled.rotate_left(7);
        assert!(forward == run(&reversed) && forward == run(&shuffled));
        let recs: Vec<Record> = forward.0.chunks_exact(RECORD_BYTES).map(Record::from_bytes).collect();
        let kept = |null: bool| -> Vec<(u32, u32)> { recs.iter().filter(|r| (r.flags & FLAG_NULL != 0) == null).map(|r| (r.leader, r.target)).collect() };
        // The two strong pairs, then the tied pair with the lowest matrix index (target-major).
        let id = |idx: i64| (20 + (idx % np) as u32, 20 + (idx / np) as u32);
        assert_eq!(kept(false), vec![id(pairs[0].0), id(pairs[1].0), id(pairs[2].0)]);
        assert_eq!(kept(true), vec![id(pairs[0].0), id(pairs[1].0)]);
    }
}
