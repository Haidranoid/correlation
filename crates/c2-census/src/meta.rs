//! A census run's own description, read from its directory: settings, tasks, correlation
//! periods and stock list. Readers use this instead of rebuilding them from today's code and data.
//!
//! How each part is recovered:
//! - Settings: the manifest's `config` text (`Config::from_text`).
//! - Tasks (what `tasks/t<id>.rec` and `.csv` hold): the manifest's `task_list` (runs made by this
//!   version on), else each task's first summary line (its move lengths and correlation period,
//!   whose length `windows.csv` gives). Files that hold nothing (combinations too short for a
//!   single observation) are labeled by the one task numbering the code has used that fits every
//!   other file. A task that can't be identified is never guessed: the task list is then unknown
//!   and `tasks()` says why.
//! - Correlation periods: `windows.csv`, mapped onto a calendar by date (`window_rows`).
//! - Stocks: records hold store ids as of the run. The run's copy of the store's `symbols.tsv`, else
//!   the current store's table if its fingerprint still equals the manifest's. Otherwise the run's
//!   stock list can't be recovered and `stocks()` says so. `id_map` translates through tickers.
//!
//! Nothing here writes into a run directory except `snapshot_symbols_for_all_runs`, the explicit
//! step that `c2 update` and `c2 ingest` take before they replace the store.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use c2_data::{Calendar, Store, SymbolInfo};

use crate::grid::{Config, Task};
use crate::run::RECORD_BYTES;

/// FNV-1a, the fingerprint the manifest keeps of the store's `symbols.tsv` (`symbols_tsv_fnv64`).
pub fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

fn fingerprint(bytes: &[u8]) -> String {
    format!("{:016x}", fnv64(bytes))
}

fn current_symbols_path() -> PathBuf {
    c2_data::data_dir().join("store/symbols.tsv")
}

/// One correlation period as the run recorded it (a line of `windows.csv`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunWindow {
    pub id: usize,
    /// Rolling length in trading days; 0 for the full-history period.
    pub length: usize,
    /// First and last trading day (days since 1970).
    pub first: i32,
    pub last: i32,
    /// Stocks in its universe.
    pub stocks: usize,
}

/// Where a run's task list came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskSource {
    /// The manifest's `task_list`.
    Manifest,
    /// Derived from the task files' first summary lines (runs made before `task_list`); files
    /// holding nothing get the labels of the one numbering rule that fits all the others.
    Derived,
}

/// Where a run's stock list came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StockSource {
    /// The run's own copy of the store's table, `<run>/symbols.tsv`.
    Snapshot,
    /// The current store's `symbols.tsv`, whose fingerprint still equals the manifest's.
    CurrentStore,
}

/// The store's symbol table as it was when the run was made: record ids index it.
pub struct RunStocks {
    pub symbols: Vec<SymbolInfo>,
    pub source: StockSource,
    /// Ticker -> run id, for tickers that occur once.
    by_ticker: HashMap<String, u32>,
}

impl RunStocks {
    fn new(symbols: Vec<SymbolInfo>, source: StockSource) -> RunStocks {
        let mut count: HashMap<&str, usize> = HashMap::new();
        for s in &symbols {
            *count.entry(s.ticker.as_str()).or_default() += 1;
        }
        let by_ticker = symbols.iter().enumerate().filter(|(_, s)| count[s.ticker.as_str()] == 1).map(|(k, s)| (s.ticker.clone(), k as u32)).collect();
        RunStocks { symbols, source, by_ticker }
    }

    /// The ticker of a record's leader or target id.
    pub fn ticker(&self, run_id: u32) -> Option<&str> {
        self.symbols.get(run_id as usize).map(|s| s.ticker.as_str())
    }

    /// The run's id for a ticker (None if the run's store did not have it, or had it twice).
    pub fn id(&self, ticker: &str) -> Option<u32> {
        self.by_ticker.get(ticker).copied()
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

/// Translation between a run's store ids and another store's ids, through tickers. A ticker
/// that occurs more than once in either table is matched on (ticker, exchange) when that is
/// unique on both sides, else left untranslated.
pub struct IdMap {
    to_current: Vec<Option<u32>>,
    to_run: Vec<Option<u32>>,
    identity: bool,
}

impl IdMap {
    pub fn new(run: &[SymbolInfo], current: &[SymbolInfo]) -> IdMap {
        fn index(t: &[SymbolInfo]) -> HashMap<&str, Vec<u32>> {
            let mut m: HashMap<&str, Vec<u32>> = HashMap::new();
            for (k, s) in t.iter().enumerate() {
                m.entry(s.ticker.as_str()).or_default().push(k as u32);
            }
            m
        }
        let (ri, ci) = (index(run), index(current));
        let mut to_current = vec![None; run.len()];
        let mut to_run = vec![None; current.len()];
        for (k, s) in run.iter().enumerate() {
            let (Some(a), Some(b)) = (ri.get(s.ticker.as_str()), ci.get(s.ticker.as_str())) else { continue };
            let found = if a.len() == 1 && b.len() == 1 {
                Some(b[0])
            } else {
                let same = |t: &[SymbolInfo], ids: &[u32]| -> Vec<u32> { ids.iter().copied().filter(|&i| t[i as usize].exchange == s.exchange).collect() };
                let (a, b) = (same(run, a), same(current, b));
                (a.len() == 1 && b.len() == 1).then(|| b[0])
            };
            if let Some(c) = found {
                to_current[k] = Some(c);
                to_run[c as usize] = Some(k as u32);
            }
        }
        let identity = run.len() == current.len() && to_current.iter().enumerate().all(|(k, c)| *c == Some(k as u32));
        IdMap { to_current, to_run, identity }
    }

    /// Today's store id of a run's stock (a record's leader or target).
    pub fn current(&self, run_id: u32) -> Option<usize> {
        self.to_current.get(run_id as usize).copied().flatten().map(|c| c as usize)
    }

    /// The run's id of a stock in today's store.
    pub fn run(&self, current_id: usize) -> Option<u32> {
        self.to_run.get(current_id).copied().flatten()
    }

    /// True when every run id is the same stock's id today (the two tables list the same
    /// tickers in the same order), so record ids can be used as store ids directly.
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// Run ids with no counterpart in today's store.
    pub fn untranslated(&self) -> usize {
        self.to_current.iter().filter(|c| c.is_none()).count()
    }
}

/// Everything a run directory says about itself.
pub struct RunMeta {
    pub name: String,
    pub dir: PathBuf,
    pub manifest: serde_json::Value,
    /// The manifest's `config` text and the settings it describes.
    pub config_text: String,
    pub cfg: Config,
    /// Correlation periods by id (`windows.csv`).
    pub windows: Vec<RunWindow>,
    /// Per correlation period: its stocks as the run's store ids (`universes.bin`).
    pub universes: Vec<Vec<u32>>,
    pub task_source: TaskSource,
    tasks: std::result::Result<Vec<Task>, String>,
    stocks: std::result::Result<RunStocks, String>,
}

/// "A moves over 5 days -> B over 21 days, 756-day correlation periods".
pub fn describe_task(h1: usize, h2: usize, length: usize) -> String {
    let days = |d: usize| if d == 1 { "1 day".to_string() } else { format!("{d} days") };
    let period = if length == 0 { "the full-history correlation period".to_string() } else { format!("{length}-day correlation periods") };
    format!("A moves over {} -> B over {}, {period}", days(h1), days(h2))
}

impl RunMeta {
    /// `RunMeta::load` of `<data dir>/census/<name>`.
    pub fn load_name(name: &str) -> Result<RunMeta> {
        RunMeta::load(&crate::run::run_dir(name))
    }

    /// Reads a run directory. Fails if the manifest, `windows.csv` or `universes.bin` are missing
    /// or don't fit together. An unknown task list or unrecoverable stock list is not an error
    /// here; `tasks()` and `stocks()` report it to the readers that need them.
    pub fn load(dir: &Path) -> Result<RunMeta> {
        let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
        let text = std::fs::read_to_string(dir.join("manifest.json")).with_context(|| format!("census {name}: no manifest in {}", dir.display()))?;
        let manifest: serde_json::Value = serde_json::from_str(&text).with_context(|| format!("census {name}: manifest.json is not valid JSON"))?;
        let config_text = manifest["config"].as_str().with_context(|| format!("census {name}: the manifest records no settings"))?.to_string();
        let cfg = Config::from_text(&config_text).with_context(|| format!("census {name}"))?;
        let windows = read_windows(dir).with_context(|| format!("census {name}: windows.csv"))?;
        if let Some(n) = manifest["windows"].as_u64() {
            anyhow::ensure!(n as usize == windows.len(), "census {name}: the manifest records {n} correlation periods, windows.csv {}", windows.len());
        }
        let universes = read_universes(dir, &windows).with_context(|| format!("census {name}: universes.bin"))?;
        let (task_source, tasks) = task_list(dir, &manifest, &cfg, &windows);
        let stocks = read_stocks(&name, dir, &manifest);
        Ok(RunMeta { name, dir: dir.to_path_buf(), manifest, config_text, cfg, windows, universes, task_source, tasks, stocks })
    }

    /// The run's tasks by id: `tasks()[id]` is what `tasks/t<id>.*` hold. Errors if they can't
    /// be established (the message says why).
    pub fn tasks(&self) -> Result<&[Task]> {
        self.tasks.as_deref().map_err(|e| anyhow!("census {}: {e}", self.name))
    }

    pub fn task(&self, id: usize) -> Result<&Task> {
        self.tasks()?.get(id).ok_or_else(|| anyhow!("census {} has no results file t{id}", self.name))
    }

    /// The task holding (h1, h2) for correlation periods of `length` days (0 = full history).
    pub fn find_task(&self, h1: usize, h2: usize, length: usize) -> Result<Option<&Task>> {
        Ok(self.tasks()?.iter().find(|t| (t.h1, t.h2, t.length) == (h1, h2, length)))
    }

    /// `<run>/tasks/t<id>.<ext>` (`ext` is "rec" or "csv").
    pub fn task_path(&self, id: usize, ext: &str) -> PathBuf {
        self.dir.join(format!("tasks/t{id}.{ext}"))
    }

    /// Calendar rows of correlation period `wid`, found by date. None unless its first and last
    /// days are both in `cal` and, for a rolling period, exactly `length` rows lie between them
    /// (no clamping to the data, so nothing after the period's end can leak in).
    pub fn window_rows(&self, cal: &Calendar, wid: usize) -> Option<Range<usize>> {
        self.window_rows_in(&cal.days, wid)
    }

    /// `window_rows` against any sorted list of trading days (e.g. a census variable's `days`).
    pub fn window_rows_in(&self, days: &[i32], wid: usize) -> Option<Range<usize>> {
        let w = self.windows.get(wid)?;
        let (a, b) = (days.binary_search(&w.first).ok()?, days.binary_search(&w.last).ok()?);
        (b >= a && (w.length == 0 || b + 1 - a == w.length)).then_some(a..b + 1)
    }

    /// The run's stock list. Errors if it can't be recovered (the message says why).
    pub fn stocks(&self) -> Result<&RunStocks> {
        self.stocks.as_ref().map_err(|e| anyhow!("census {}: {e}", self.name))
    }

    /// Translation of this run's store ids to `store`'s ids and back. Build it once per store.
    pub fn id_map(&self, store: &Store) -> Result<IdMap> {
        Ok(IdMap::new(&self.stocks()?.symbols, &store.symbols))
    }

    /// The run's id of `ticker` (what its records call that stock): through today's store and
    /// `ids` (this run's `id_map(store)`), which also settles a ticker listed twice by exchange;
    /// else the run's own table, for a stock today's store no longer has. None if the run's
    /// store didn't have it (or its stock list can't be recovered).
    pub fn run_id_of(&self, store: &Store, ids: &IdMap, ticker: &str) -> Option<u32> {
        store.id(ticker).and_then(|c| ids.run(c)).or_else(|| self.stocks().ok()?.id(ticker))
    }

    /// Ids of the correlation periods that `days` (e.g. a census variable's) doesn't cover
    /// exactly (`window_rows_in` is None): readers leave their results out.
    pub fn uncovered_windows(&self, days: &[i32]) -> Vec<usize> {
        (0..self.windows.len()).filter(|&w| self.window_rows_in(days, w).is_none()).collect()
    }
}

fn read_windows(dir: &Path) -> Result<Vec<RunWindow>> {
    let text = std::fs::read_to_string(dir.join("windows.csv"))?;
    let mut lines = text.lines();
    anyhow::ensure!(lines.next() == Some("id,length,first,last,stocks"), "unexpected header");
    lines
        .enumerate()
        .map(|(k, l)| {
            let f: Vec<&str> = l.split(',').collect();
            anyhow::ensure!(f.len() == 5 && f[0].parse::<usize>().ok() == Some(k), "line {} is not correlation period {k}: {l:?}", k + 2);
            let w = RunWindow { id: k, length: f[1].parse()?, first: c2_data::parse_date(f[2])?, last: c2_data::parse_date(f[3])?, stocks: f[4].parse()? };
            anyhow::ensure!(w.first <= w.last, "correlation period {k} ends before it starts");
            Ok(w)
        })
        .collect()
}

fn read_universes(dir: &Path, windows: &[RunWindow]) -> Result<Vec<Vec<u32>>> {
    let b = std::fs::read(dir.join("universes.bin"))?;
    let u32_at = |p: usize| b.get(p..p + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()));
    let (mut pos, mut out) = (0usize, Vec::with_capacity(windows.len()));
    for w in windows {
        let n = u32_at(pos).with_context(|| format!("ends before correlation period {}", w.id))? as usize;
        anyhow::ensure!(n == w.stocks, "correlation period {} has {n} stocks here, {} in windows.csv", w.id, w.stocks);
        pos += 4;
        let ids: Option<Vec<u32>> = (0..n).map(|k| u32_at(pos + 4 * k)).collect();
        out.push(ids.with_context(|| format!("ends inside correlation period {}", w.id))?);
        pos += 4 * n;
    }
    anyhow::ensure!(pos == b.len(), "{} bytes left over after the last correlation period", b.len() - pos);
    Ok(out)
}

/// The windows of a task of `length`, as `grid::tasks` assigns them: every period of that
/// length, in id (date) order.
fn windows_of(windows: &[RunWindow], length: usize) -> Vec<usize> {
    windows.iter().filter(|w| w.length == length).map(|w| w.id).collect()
}

/// (window, h1, h2) of a task's first summary line, if it has one.
fn first_summary(path: &Path) -> Result<Option<(usize, usize, usize)>> {
    let Ok(f) = std::fs::File::open(path) else { return Ok(None) };
    let mut lines = BufReader::new(f).lines();
    let Some(header) = lines.next().transpose()? else { return Ok(None) };
    anyhow::ensure!(header.starts_with("window,h1,h2,lag,"), "{} has an unexpected header", path.display());
    let Some(row) = lines.next().transpose()? else { return Ok(None) };
    let f: Vec<&str> = row.split(',').collect();
    let num = |i: usize| f.get(i).and_then(|v| v.parse::<usize>().ok()).with_context(|| format!("{}: bad first line {row:?}", path.display()));
    Ok(Some((num(0)?, num(1)?, num(2)?)))
}

/// (window, h1, h2) of a task's first stored record, if it has one.
fn first_record(path: &Path) -> Option<(usize, usize, usize)> {
    let mut b = [0u8; RECORD_BYTES];
    std::fs::File::open(path).ok()?.read_exact(&mut b).ok()?;
    let r = crate::run::Record::from_bytes(&b);
    Some((r.window as usize, r.h1 as usize, r.h2 as usize))
}

fn task_list(dir: &Path, manifest: &serde_json::Value, cfg: &Config, windows: &[RunWindow]) -> (TaskSource, std::result::Result<Vec<Task>, String>) {
    if let Some(list) = manifest.get("task_list") {
        let parse = || -> std::result::Result<Vec<Task>, String> {
            let list = list.as_array().ok_or("the manifest's task_list is not a list")?;
            list.iter()
                .enumerate()
                .map(|(id, t)| {
                    let v: Vec<usize> = t.as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect()).unwrap_or_default();
                    let [h1, h2, length] = v[..] else { return Err(format!("the manifest's task_list entry {id} is {t}, not [h1, h2, length]")) };
                    Ok(Task { id, h1, h2, length, windows: windows_of(windows, length) })
                })
                .collect()
        };
        return (TaskSource::Manifest, parse().and_then(|t| check_tasks(dir, cfg, t)));
    }
    let derive = || -> std::result::Result<Vec<Task>, String> {
        let count = manifest["tasks"].as_u64().ok_or("the manifest records neither a task list nor a task count")? as usize;
        // Per task: (h1, h2, length) from its first summary line, or None if it holds nothing.
        let mut seen: Vec<Option<(usize, usize, usize)>> = Vec::with_capacity(count);
        for id in 0..count {
            let rec = dir.join(format!("tasks/t{id}.rec"));
            match first_summary(&dir.join(format!("tasks/t{id}.csv"))).map_err(|e| format!("{e:#}"))? {
                Some((wid, h1, h2)) => {
                    let w = windows.get(wid).ok_or_else(|| format!("results file t{id} names correlation period {wid}, which windows.csv doesn't have"))?;
                    // Its first record (when there is one) belongs to the same unit, so must agree.
                    if let Some(r) = first_record(&rec).filter(|&r| r != (wid, h1, h2)) {
                        return Err(format!("results file t{id}'s summary starts with period {wid} and move lengths ({h1}, {h2}), but its records start with period {} and ({}, {})", r.0, r.1, r.2));
                    }
                    seen.push(Some((h1, h2, w.length)));
                }
                None if std::fs::metadata(&rec).map_or(0, |m| m.len()) == 0 => seen.push(None),
                None => return Err(format!("results file t{id} has records but no summary lines")),
            }
        }
        let silent: Vec<usize> = (0..count).filter(|&id| seen[id].is_none()).collect();
        let label = if silent.is_empty() {
            seen.iter().map(|s| s.unwrap()).collect()
        } else {
            // Files that hold nothing don't say what they were for. That only matters as a label,
            // and only one numbering the code has used can be consistent with the rest: the run's
            // task count, every file that holds results, and nothing at all in each file whose
            // combination is too short for a single observation. Anything else stays unknown.
            let fits: Vec<Vec<(usize, usize, usize)>> = numberings(cfg, windows)
                .into_iter()
                .filter(|rule| rule.len() == count && rule.iter().zip(&seen).all(|(&t, s)| s.map_or_else(|| produces_nothing(cfg, windows, t), |s| s == t)))
                .collect();
            let [rule] = &fits[..] else {
                let shown: Vec<String> = silent.iter().take(8).map(|id| format!("t{id}")).collect();
                return Err(format!(
                    "what {} of its {count} results files hold can't be established ({}{}): they hold no results, the run predates the manifest's task list, and {}, so their move lengths would have to be guessed",
                    silent.len(), shown.join(", "), if silent.len() > shown.len() { ", ..." } else { "" },
                    if fits.is_empty() { "no task numbering this code has used explains them (an unfinished run?)" } else { "more than one task numbering fits" }
                ));
            };
            rule.clone()
        };
        Ok(label.into_iter().enumerate().map(|(id, (h1, h2, length))| Task { id, h1, h2, length, windows: windows_of(windows, length) }).collect())
    };
    (TaskSource::Derived, derive().and_then(|t| check_tasks(dir, cfg, t)))
}

/// The task numberings the code has used, as (h1, h2, length) by task id: every (h1, h2) pair
/// (from commit dc94b39 on), and before that only pairs whose B move is no longer than A's.
/// Both go pair by pair, each over the period lengths in order and then full history, skipping
/// lengths without periods (`grid::tasks`).
fn numberings(cfg: &Config, windows: &[RunWindow]) -> Vec<Vec<(usize, usize, usize)>> {
    let mut lengths = cfg.lengths.clone();
    if cfg.full_history {
        lengths.push(0);
    }
    lengths.retain(|&len| windows.iter().any(|w| w.length == len));
    let all = cfg.horizon_pairs();
    let older: Vec<(usize, usize)> = all.iter().copied().filter(|&(h1, h2)| h2 <= h1).collect();
    [all, older].iter().map(|pairs| pairs.iter().flat_map(|&(h1, h2)| lengths.iter().map(move |&len| (h1, h2, len))).collect()).collect()
}

/// True when the census computes nothing for (h1, h2, length): in every period of that length
/// either fewer than 2 stocks qualify or even the same-period offset has fewer than `min_obs`
/// observations. (For the full-history period, whose number of days is not recorded, only the
/// stock count can settle it.)
fn produces_nothing(cfg: &Config, windows: &[RunWindow], (h1, h2, length): (usize, usize, usize)) -> bool {
    let too_short = length > 0 && crate::grid::observations(length, h1, h2, 0) < cfg.min_obs;
    windows.iter().filter(|w| w.length == length).all(|w| w.stocks < 2 || too_short)
}

/// Consistency of a task list with the run's settings and files: known move lengths and period
/// lengths, no (h1, h2, length) twice, and no task files beyond the list.
fn check_tasks(dir: &Path, cfg: &Config, tasks: Vec<Task>) -> std::result::Result<Vec<Task>, String> {
    let mut seen: HashMap<(usize, usize, usize), usize> = HashMap::new();
    for t in &tasks {
        let length_ok = if t.length == 0 { cfg.full_history } else { cfg.lengths.contains(&t.length) };
        if !cfg.horizons.contains(&t.h1) || !cfg.horizons.contains(&t.h2) || !length_ok {
            return Err(format!("results file t{} holds {}, which the run's settings don't include", t.id, describe_task(t.h1, t.h2, t.length)));
        }
        if t.windows.is_empty() {
            return Err(format!("results file t{} holds {}, but the run has no such correlation periods", t.id, describe_task(t.h1, t.h2, t.length)));
        }
        if let Some(other) = seen.insert((t.h1, t.h2, t.length), t.id) {
            return Err(format!("results files t{other} and t{} both hold {}; the run was probably resumed under a different numbering", t.id, describe_task(t.h1, t.h2, t.length)));
        }
    }
    let mut extra: Vec<usize> = std::fs::read_dir(dir.join("tasks"))
        .map(|it| {
            it.flatten()
                .filter_map(|e| {
                    let name = e.file_name().into_string().ok()?;
                    let stem = name.strip_suffix(".rec").or_else(|| name.strip_suffix(".csv"))?;
                    stem.strip_prefix('t')?.parse::<usize>().ok()
                })
                .filter(|&id| id >= tasks.len())
                .collect()
        })
        .unwrap_or_default();
    extra.sort_unstable();
    extra.dedup();
    if let Some(&first) = extra.first() {
        return Err(format!(
            "it has results files beyond the {} its task list accounts for (t{first}{}); the run was probably resumed under a different numbering",
            tasks.len(), if extra.len() > 1 { format!(" and {} more", extra.len() - 1) } else { String::new() }
        ));
    }
    Ok(tasks)
}

fn read_stocks(name: &str, dir: &Path, manifest: &serde_json::Value) -> std::result::Result<RunStocks, String> {
    let want = manifest["symbols_tsv_fnv64"].as_str().unwrap_or("");
    let parse = |bytes: &[u8], what: &str| -> std::result::Result<Vec<SymbolInfo>, String> {
        let text = std::str::from_utf8(bytes).map_err(|_| format!("{what} is not text"))?;
        c2_data::store::parse_symbols(text).map_err(|e| format!("{what}: {e:#}"))
    };
    let snap = dir.join("symbols.tsv");
    if snap.exists() {
        let bytes = std::fs::read(&snap).map_err(|e| format!("can't read {}: {e}", snap.display()))?;
        if !want.is_empty() && fingerprint(&bytes) != want {
            return Err(format!("its copy of the stock list ({}) is not the one its manifest records (fingerprint {} vs {want})", snap.display(), fingerprint(&bytes)));
        }
        return Ok(RunStocks::new(parse(&bytes, "its symbols.tsv")?, StockSource::Snapshot));
    }
    let path = current_symbols_path();
    let bytes = std::fs::read(&path).map_err(|e| format!("it has no copy of its stock list, and the store's {} can't be read: {e}", path.display()))?;
    let now = fingerprint(&bytes);
    if want.is_empty() || now != want {
        return Err(format!(
            "its stock list can't be recovered: census {name} has no copy of the store's symbols.tsv, and the store has been rebuilt since it ran (symbol table fingerprint {} then, {now} now), so its stock numbers can't be matched to tickers",
            if want.is_empty() { "unrecorded" } else { want }
        ));
    }
    Ok(RunStocks::new(parse(&bytes, "the store's symbols.tsv")?, StockSource::CurrentStore))
}

/// What `snapshot_symbols_for_all_runs` did for one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Snapshot {
    /// Wrote `<run>/symbols.tsv` from the current store (its fingerprint matched).
    Written,
    /// The run already has its copy; left alone.
    AlreadyThere,
    /// No copy, and the current store no longer matches the run's fingerprint: its stock list
    /// is already lost; nothing written.
    Unrecoverable,
}

/// Before the store is replaced: gives every run without its own copy of the store's symbol
/// table a copy of the current one, but only where the manifest's fingerprint shows it is the
/// table the run was made with. Returns (run name, outcome) for every run directory.
pub fn snapshot_symbols_for_all_runs() -> Result<Vec<(String, Snapshot)>> {
    let root = c2_data::data_dir().join("census");
    let Ok(entries) = std::fs::read_dir(&root) else { return Ok(Vec::new()) };
    let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.join("manifest.json").is_file()).collect();
    dirs.sort();
    let current = std::fs::read(current_symbols_path()).ok();
    let now = current.as_deref().map(fingerprint);
    let mut out = Vec::new();
    for dir in dirs {
        let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
        let outcome = if dir.join("symbols.tsv").exists() {
            Snapshot::AlreadyThere
        } else {
            let m: serde_json::Value = std::fs::read_to_string(dir.join("manifest.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
            match (&current, &now, m["symbols_tsv_fnv64"].as_str()) {
                (Some(bytes), Some(now), Some(want)) if now == want => {
                    crate::run::write_atomic_bytes(&dir.join("symbols.tsv"), bytes).with_context(|| format!("save census {name}'s stock list"))?;
                    Snapshot::Written
                }
                _ => Snapshot::Unrecoverable,
            }
        };
        out.push((name, outcome));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(ticker: &str, exchange: &str) -> SymbolInfo {
        SymbolInfo { ticker: ticker.into(), exchange: exchange.into(), kind: c2_data::Kind::Stock, first: 0, last: 0, rows: 0, source: String::new() }
    }

    #[test]
    fn ids_translate_through_tickers() {
        let run = [sym("A", "nyse"), sym("AA", "nyse"), sym("B", "nasdaq"), sym("D", "nyse"), sym("D", "nasdaq")];
        let now = [sym("A", "nyse"), sym("AB", "nyse"), sym("B", "nasdaq"), sym("C", "nyse"), sym("D", "nasdaq")];
        let m = IdMap::new(&run, &now);
        assert_eq!((m.current(0), m.current(1), m.current(2), m.current(3), m.current(4)), (Some(0), None, Some(2), None, Some(4)));
        assert_eq!((m.run(0), m.run(1), m.run(2), m.run(3), m.run(4)), (Some(0), None, Some(2), None, Some(4)));
        assert!(!m.is_identity());
        assert_eq!(m.untranslated(), 2);
        assert!(IdMap::new(&run, &run).is_identity());
        let stocks = RunStocks::new(run.to_vec(), StockSource::Snapshot);
        assert_eq!((stocks.id("B"), stocks.id("D"), stocks.ticker(2)), (Some(2), None, Some("B")));
    }

    #[test]
    fn periods_need_exact_coverage() {
        let w = |id, length, first, last| RunWindow { id, length, first, last, stocks: 0 };
        let meta = RunMeta {
            name: "t".into(),
            dir: PathBuf::new(),
            manifest: serde_json::Value::Null,
            config_text: String::new(),
            cfg: Config::smoke(),
            windows: vec![w(0, 3, 10, 12), w(1, 3, 11, 14), w(2, 0, 10, 14), w(3, 3, 13, 16)],
            universes: vec![],
            task_source: TaskSource::Derived,
            tasks: Err(String::new()),
            stocks: Err(String::new()),
        };
        let days = [10, 11, 12, 14, 15];
        assert_eq!(meta.window_rows_in(&days, 0), Some(0..3));
        // 11..14 holds 3 rows here (day 13 is not a trading day), as the run had them.
        assert_eq!(meta.window_rows_in(&days, 1), Some(1..4));
        assert_eq!(meta.window_rows_in(&days, 2), Some(0..4));
        // First day missing, and last day beyond the data: never clamped.
        assert_eq!(meta.window_rows_in(&days, 3), None);
        assert_eq!(meta.window_rows_in(&[10, 11, 12, 13, 14], 1), None);
        assert_eq!(meta.window_rows_in(&days, 4), None);
    }
}
