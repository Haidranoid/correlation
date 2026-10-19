//! `c2-gui`: native desktop front end for correlation2.
//!
//! It drives the `c2` command (census runs happen in their own detached process, so closing the
//! window never interrupts one) and reads its output files. Tabs:
//!   Status   census progress and ETA; start, pause, resume, stop; every census run (when,
//!            how long, how much computed); charts of what the census in use found
//!   Strongest  stocks ranked by significant lead-lag relationships; click one for its timeline
//!   Stock    timeline of one stock's lead-lag partners on a days-ahead axis (which stocks' moves
//!            show up in it later, or where its own moves show up later, and by how much), with
//!            each relationship's history across census windows
//!   Forecast the stock's predicted price path from its leaders' recent moves, the actual path
//!            when the as-of date is in the past, and the forecast's track record
//!   Explainer one relationship recomputed from prices step by step and checked against what the
//!            census stored (the pair's record, else its matrix's summary), or why it never
//!            computed it
//!   Census   real vs chance by offset for each pair of move lengths; a ticker's strongest stored
//!            pairs
//!   Holdout  the $10,000 test for either signal family, with equity curves
//!   Data     price store summary and the decade quality table
//!
//! Everything read from a census run is computed on a background thread (`Job`) and cached under
//! a key that starts with the run's name, so switching runs on the Status tab never shows or
//! reuses another run's results.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use c2_census::holdout::{Family, Options, Report};
use c2_census::meta::RunMeta;
use c2_census::run::{self, FLAG_NULL, Plan};
use c2_data::Store;
use eframe::egui;
use c2_census::analyze::ProfileRow;
use c2_census::analyze::{Counts, DatasetStats, RunInfo, Words};
use c2_census::explain::{Check, Explained};
use c2_census::forecast::{Forecast, ForecastData};
use c2_census::grid;
use egui_plot::{Arrows, Bar, BarChart, HLine, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoint, PlotPoints, Points, Polygon, Text, VLine};

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Status,
    Strongest,
    Stock,
    Forecast,
    Explain,
    Census,
    Holdout,
    Data,
}

/// Real vs chance tail counts for one (h1, h2, lag), summed over windows.
#[derive(Default, Clone)]
struct LagRow {
    real_z4: f64,
    null_z4: f64,
    matrices: usize,
    fdr: usize,
}

struct PairRow {
    window: String,
    partner: String,
    role: &'static str,
    h1: u8,
    h2: u8,
    lag: u16,
    r: f32,
    z: f32,
    obs: u16,
}

/// Shown when a background job ends without delivering a result (it panicked or its thread died).
const STOPPED: &str = "the background job stopped unexpectedly";

/// A computation on a background thread, and its outcome cached under the key of the inputs it
/// was computed from. Every key starts with the census run's name, and changing the run clears
/// every job (`App::set_run`), so no result is ever reused or shown for another run. At most one
/// thread runs per job: a job for new inputs starts once the one in flight has finished, so rapid
/// input changes don't pile up scans of the census. A finished job is never restarted for the
/// same key by itself (an empty or failed result stays until the user asks again).
struct Job<K, T> {
    running: Option<(K, Receiver<Result<T, String>>)>,
    done: Option<(K, Result<T, String>)>,
    /// Bumped whenever `done` changes, so views derived from the result know to rebuild.
    generation: u64,
    /// When the job in flight started, and how long the last finished one took (from its start
    /// to its result arriving); self-checks report it.
    started: Instant,
    took: Duration,
}

impl<K, T> Default for Job<K, T> {
    fn default() -> Self {
        Job { running: None, done: None, generation: 0, started: Instant::now(), took: Duration::ZERO }
    }
}

impl<K: Clone + PartialEq, T: Send + 'static> Job<K, T> {
    /// Starts `work` for `key` on a new thread, abandoning any job in flight (its result is
    /// dropped when it arrives). A panic in `work` becomes an error result.
    fn start(&mut self, key: K, work: impl FnOnce() -> Result<T, String> + Send + 'static) {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_else(|p| {
                let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned());
                Err(msg.map_or_else(|| STOPPED.to_string(), |m| format!("{STOPPED}: {m}")))
            });
            let _ = tx.send(res);
        });
        self.running = Some((key, rx));
        self.started = Instant::now();
    }

    /// Collects the outcome of the job in flight if it has finished; true when one arrived. A
    /// thread that ended without sending anything (`Disconnected`) finishes the job with an error,
    /// so no spinner waits forever.
    fn poll(&mut self) -> bool {
        let res = match &self.running {
            None => return false,
            Some((_, rx)) => match rx.try_recv() {
                Ok(res) => res,
                Err(TryRecvError::Empty) => return false,
                Err(TryRecvError::Disconnected) => Err(format!("{STOPPED} without reporting an error")),
            },
        };
        if let Some((key, _)) = self.running.take() {
            self.done = Some((key, res));
            self.generation += 1;
            self.took = self.started.elapsed();
        }
        true
    }

    fn is_running(&self) -> bool {
        self.running.is_some()
    }

    /// The finished outcome for exactly these inputs, if there is one.
    fn get(&self, key: &K) -> Option<&Result<T, String>> {
        self.done.as_ref().filter(|(k, _)| k == key).map(|(_, r)| r)
    }

    /// True when a job for `key` should start now: none is in flight and none has finished for
    /// these inputs.
    fn wants(&self, key: &K) -> bool {
        self.running.is_none() && self.done.as_ref().is_none_or(|(k, _)| k != key)
    }

    /// Forgets the outcome and abandons any job in flight.
    fn clear(&mut self) {
        self.running = None;
        if self.done.take().is_some() {
            self.generation += 1;
        }
    }
}

/// (run, ticker, as-of day or None for the latest, days ahead, significant only, max age of a
/// correlation period in years or 0 for any).
type ForecastKey = (String, String, Option<i32>, usize, bool, usize);

/// (run, A, B, A's move length, B's move length, offset, correlation period id in the run, fixed
/// threshold in 1/100 %).
type ExplainKey = (String, String, String, usize, usize, usize, usize, Option<i64>);

/// (run, signal family, first holdout day, stocks held, rebalance days, costs, random accounts).
type HoldoutKey = (String, usize, String, usize, usize, String, usize);

/// A census run's own description (`RunMeta`: settings, periods, tasks, stock list) and its plan
/// on today's prices (`Plan::for_run`): the census variable and the run's own correlation
/// periods, universes and tasks, so period and task ids mean what they meant in that run.
struct RunPlan {
    meta: RunMeta,
    plan: Plan,
}

/// The one run plan kept in memory, with the run it is for and the generation of the price store
/// it was built from. Shared by the plan job (Stock timeline, Forecast, Explainer) and the
/// holdout's worker so that only one is built at a time.
type PlanSlot = Arc<Mutex<PlanEntry>>;

/// The slot's content: nothing, or ((run, store generation), its plan).
type PlanEntry = Option<((String, u64), Arc<RunPlan>)>;

/// The Strongest tab's data: the ranking, and per correlation period of the run (by its id) its
/// length and last day, for each row's strongest relationship.
struct Ranking {
    rows: Vec<c2_census::analyze::RankRow>,
    periods: Vec<(String, String)>,
}

/// (job generation, find text, sort) of the Strongest tab's prepared rows.
type RankViewKey = (u64, String, usize);

/// One prepared row of the Strongest tab: the stock and its other cells, already formatted.
struct RankLine {
    ticker: String,
    cells: [String; 13],
}

/// Self-check mode (`C2_GUI_SCREENSHOT=<path prefix>`, `C2_GUI_TAB=<tab>`): open a tab, wait for
/// its data (or its error), save the window's pixels and exit. After `C2_GUI_TIMEOUT` seconds
/// (default 600) it saves whatever is shown; it waits at least `C2_GUI_WAIT` seconds (default 0),
/// e.g. for a change made to the data while it runs. `C2_GUI_PLAN` builds the run's plan first
/// (as a Stock timeline, Forecast or Explainer visit would), for tabs that show what it notes.
struct Shot {
    prefix: String,
    frames: u32,
    requested: bool,
    started: Instant,
    timeout: Duration,
    min_wait: Duration,
    plan: bool,
}

/// Run name typing settles after this long without a keystroke (or on Enter).
const RUN_IDLE: Duration = Duration::from_millis(1200);

/// How often the price store's files are checked for a rebuild (`c2 update`, `c2 ingest`).
const STORE_CHECK: Duration = Duration::from_secs(3);

/// The price store's files as last seen: size, modification time and inode of `prices.bin` and
/// `symbols.tsv` (a rebuild renames new files over both).
type StoreStamp = Vec<(u64, Option<std::time::SystemTime>, u64)>;

/// A store opened in the background: the store, its summary line, the Data tab's table, and the
/// files' stamp it was opened from.
type OpenedStore = (Arc<Store>, String, Vec<c2_data::quality::DecadeRow>, Option<StoreStamp>);

fn store_dir() -> PathBuf {
    c2_data::data_dir().join("store")
}

/// The store's files as they are now; None while one is missing or a rebuild is writing them
/// (its temporary files exist).
fn store_stamp(dir: &Path) -> Option<StoreStamp> {
    use std::os::unix::fs::MetadataExt;
    if ["prices.bin.tmp", "symbols.tsv.tmp"].iter().any(|f| dir.join(f).exists()) {
        return None;
    }
    ["prices.bin", "symbols.tsv"].iter().map(|f| std::fs::metadata(dir.join(f)).ok().map(|m| (m.len(), m.modified().ok(), m.ino()))).collect()
}

/// Opens the price store in `dir` (stamped first, so a rebuild during the open shows up as a
/// change at the next check).
fn open_store(dir: &Path) -> Result<OpenedStore, String> {
    let stamp = store_stamp(dir);
    let store = Store::open(dir).map_err(|e| format!("{e:#}"))?;
    let decades = c2_data::quality::decade_table(&store);
    let note = format!("{} symbols, {} daily rows, in {}", store.symbols.len(), store.rows(), dir.display());
    Ok((Arc::new(store), note, decades, stamp))
}

/// A census process started from this window (Start / resume), watched until it exits so that a
/// refusal ("census X is already running") or a failure is shown, not just written to run.out.
struct Launch {
    run: String,
    child: std::process::Child,
    /// Its output file and the length it had before this process started writing to it.
    out: PathBuf,
    from: u64,
}

struct App {
    tab: Tab,
    /// The census run every tab reads (set from the Status tab's Run name box on Enter, when the
    /// box loses focus, or after a pause in typing).
    run_name: String,
    run_input: String,
    run_edited: Option<Instant>,
    store: Option<Arc<Store>>,
    store_note: String,
    decades: Vec<c2_data::quality::DecadeRow>,
    /// The store's files as opened, when they were last checked, a changed stamp waiting to read
    /// the same twice (a rebuild writes two files), the reopen in flight (key: the generation it
    /// makes), the generation of the store in use (plans are keyed by it), and the note shown
    /// after a reload.
    store_stamp: Option<StoreStamp>,
    store_checked: Instant,
    store_settling: Option<Option<StoreStamp>>,
    store_reload: Job<u64, OpenedStore>,
    store_gen: u64,
    store_changed: Option<(Instant, String)>,
    status: Option<serde_json::Value>,
    status_err: String,
    pid: Option<u32>,
    control: String,
    last_poll: Instant,
    message: String,
    /// The census process started from this window, until it exits, and what its exit said
    /// (true = an error).
    launch: Option<Launch>,
    launch_note: Option<(bool, String)>,
    // Census tab: the table of real vs chance by offset, the run it was read from, and what
    // reading it found when it found nothing to show.
    lag_table: BTreeMap<(usize, usize), BTreeMap<usize, LagRow>>,
    lag_table_run: String,
    lag_note: String,
    pair: (usize, usize),
    ticker: String,
    /// Key: (run, ticker).
    pairs: Job<(String, String), Vec<PairRow>>,
    // Holdout tab.
    family: usize,
    start: String,
    picks: usize,
    rebalance: usize,
    costs: String,
    random: usize,
    plan_slot: PlanSlot,
    /// The holdout's report, and what the run's plan on today's prices left out (`plan_note`).
    holdout: Job<HoldoutKey, (Report, Option<String>)>,
    holdout_started: Option<Instant>,
    shot: Option<Shot>,
    // Stock timeline tab.
    tl: Timeline,
    // Forecast tab.
    fc: ForecastView,
    // Explainer tab.
    ex: ExplainView,
    /// The current run's plan: the census variable and the run's own periods, universes and
    /// tasks, used by the Stock timeline, Forecast and Explainer tabs. Key: run.
    plan: Job<String, Arc<RunPlan>>,
    // Status tab statistics: of the run (key: run), and every run (key: refresh count, and the
    // run shown, which the list marks as in use).
    /// The run's statistics and how many of each matrix's strongest pairs it stores (`keep_real`).
    stats: Job<String, (DatasetStats, usize)>,
    runs: Job<(u64, String), Vec<RunInfo>>,
    runs_refresh: u64,
    /// Local time offset from UTC in seconds (from `date +%z`).
    utc_offset: i64,
    // Strongest tab. Key: run.
    rank: Job<String, Ranking>,
    rank_view: Option<(RankViewKey, Vec<RankLine>)>,
    rank_filter: String,
    /// 0 = count x strength (sum of |r|), 1 = significant (sum of |r| over significant ones),
    /// 2 = strongest single |r|, 3 = most relationships.
    rank_sort: usize,
}

/// Key of one relationship regardless of window: (partner, stock leads?, lag, h1, h2).
type RelKey = (String, bool, u16, u8, u8);

/// A stock's stored relationships, and which correlation periods of the run are the
/// full-history one (its `windows.csv` length 0).
struct TimelineData {
    rows: Vec<ProfileRow>,
    full_history: HashSet<u32>,
}

/// (job generation, stock first, significant only, period filter, move lengths, rows shown).
type TimelineViewKey = (u64, bool, bool, String, Option<(u8, u8)>, usize);

/// What the timeline shows for its current filters (rebuilt only when they change).
#[derive(Default)]
struct TimelineView {
    rows: Vec<ProfileRow>,
    /// Rolling correlation period lengths present, and the move-length pairs.
    lengths: Vec<u32>,
    horizons: Vec<(u8, u8)>,
    /// Significant relationships in the chosen direction.
    total_sig: usize,
    /// Last day of the most recent rolling correlation period among the stock's rows.
    recent_end: Option<String>,
    /// Length of the full-history correlation period, if the stock has rows there.
    full_days: Option<u32>,
}

#[derive(Default)]
struct Timeline {
    input: String,
    /// The ticker asked for (Show, Enter, a click on the Strongest tab, the Forecast tab); it
    /// loads when the Stock timeline tab is shown.
    want: Option<String>,
    /// Key: (run, ticker). The rows shown are always the ones loaded for the ticker in the key.
    job: Job<(String, String), TimelineData>,
    view: Option<(TimelineViewKey, TimelineView)>,
    significant_only: bool,
    /// "all", "recent" (the most recent rolling correlation period), "full" (the full-history
    /// period) or a rolling period length in days.
    window: String,
    /// (h1, h2) or None for all.
    horizons: Option<(u8, u8)>,
    max_shown: usize,
    show_r: bool,
    selected: Option<RelKey>,
    /// false: stocks whose moves show up in this one later (they move first); true: stocks this
    /// one moves first.
    stock_first: bool,
    /// Self-check (`C2_GUI_SELECT`): select the first row shown once the rows are there.
    select_first: bool,
}

#[derive(Default)]
struct ExplainView {
    leader: String,
    follower: String,
    leader_input: String,
    follower_input: String,
    h1: usize,
    h2: usize,
    lag: usize,
    /// Correlation period length in trading days (0 = all history) and which period of that
    /// length (index in date order).
    length: usize,
    pos: usize,
    /// A correlation period id of a run to jump to (from a row read from that run), or a
    /// (length, end day) to find, once the census variable is built.
    pending_window: Option<(String, usize)>,
    pending_end: Option<(usize, i32)>,
    /// The period shown, as (length, last day), so a run switch can find the same one by date.
    selected: Option<(usize, i32)>,
    threshold_fixed: bool,
    threshold_pct: f64,
    obs: usize,
    day: Option<usize>,
    raw_scatter: bool,
    job: Job<ExplainKey, Explained>,
    /// The request for the current choices (None until the census variable is ready).
    key: Option<ExplainKey>,
    /// Self-check (`C2_GUI_SCROLL`): scroll the steps down this far once they are shown.
    scroll: Option<f32>,
}

#[derive(Default)]
struct ForecastView {
    /// Ticker of the forecast (the ticker box is shared with the timeline tab).
    ticker: String,
    as_of: Option<i32>,
    as_of_text: String,
    horizon: usize,
    significant_only: bool,
    max_age_years: usize,
    /// Plot the value of this many dollars invested instead of the share price.
    amount: String,
    show_amount: bool,
    /// What forecasts for one stock need from the run, read once. Key: (run, ticker).
    data: Job<(String, String), Arc<ForecastData>>,
    /// The forecast for the current choices, made from `data` without rereading the run.
    job: Job<ForecastKey, Forecast>,
    /// A mistyped as-of date.
    input_error: String,
    /// Self-check (`C2_GUI_WALK`): steps of "◀ month" still to take, and when the first was taken
    /// and how many there are in all.
    walk: usize,
    walk_started: Option<(Instant, usize)>,
}

fn key(r: &ProfileRow) -> RelKey {
    (r.partner.clone(), r.leads, r.lag, r.h1, r.h2)
}

/// Decimal year of the last date in a "first..last" window label.
fn window_end_year(w: &str) -> f64 {
    let last = w.split("..").nth(1).unwrap_or("");
    let p: Vec<f64> = last.split('-').filter_map(|v| v.parse().ok()).collect();
    if p.len() == 3 { p[0] + (p[1] - 1.0) / 12.0 + (p[2] - 1.0) / 365.0 } else { f64::NAN }
}

fn c2_binary() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("c2"))).filter(|p| p.exists()).unwrap_or_else(|| "c2".into())
}

/// The last few non-empty lines `path` gained after its first `from` bytes.
fn output_since(path: &Path, from: u64) -> String {
    let bytes = std::fs::read(path).unwrap_or_default();
    let text = String::from_utf8_lossy(bytes.get(from as usize..).unwrap_or_default());
    let lines: Vec<&str> = text.lines().map(str::trim_end).filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(4)..].join("\n")
}

/// Spawns `cmd` with its output appended to `out` (not truncated: a refused start must not wipe
/// the output of the process that is running): the child, and `out`'s length before it.
fn spawn_logged(mut cmd: std::process::Command, out: &Path) -> std::io::Result<(std::process::Child, u64)> {
    let from = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    let f = std::fs::OpenOptions::new().create(true).append(true).open(out)?;
    cmd.stdout(f.try_clone()?).stderr(f);
    Ok((cmd.spawn()?, from))
}

/// What a census process started from this window said when it exited: (an error?, the text).
/// A refusal ("census X is already running (pid N)") or a failure shows its last lines; a clean
/// exit its last line ("finished", "stopped", "census already complete").
fn launch_outcome(l: &Launch, status: std::io::Result<std::process::ExitStatus>) -> (bool, String) {
    let said = output_since(&l.out, l.from);
    match status {
        Ok(s) if s.success() => (false, format!("Census '{}': {}", l.run, said.lines().last().unwrap_or("done"))),
        Ok(s) => (true, format!("Census '{}' exited with an error ({s}):\n{said}", l.run)),
        Err(e) => (true, format!("Lost track of the process for census '{}': {e}", l.run)),
    }
}

fn pct(v: f64) -> String {
    format!("{:+.1}%", 100.0 * (v / 10_000.0 - 1.0))
}

const ORANGE: egui::Color32 = egui::Color32::from_rgb(255, 160, 40);
const BLUE: egui::Color32 = egui::Color32::from_rgb(90, 170, 255);
const GREY: egui::Color32 = egui::Color32::from_rgb(140, 150, 170);
const YELLOW_NOTE: egui::Color32 = egui::Color32::from_rgb(230, 200, 90);
const GREEN_NOTE: egui::Color32 = egui::Color32::from_rgb(120, 220, 120);
const RED_NOTE: egui::Color32 = egui::Color32::from_rgb(240, 120, 100);

/// `text` as a sentence: capitalized, ending in a full stop.
fn sentence(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => format!("{}{}.", first.to_uppercase(), chars.as_str().trim_end_matches('.')),
        None => String::new(),
    }
}

fn days_label(d: usize) -> String {
    if d == 1 { "1 day".into() } else { format!("{d} days") }
}

fn tdays(d: usize) -> String {
    if d == 1 { "1 trading day".into() } else { format!("{d} trading days") }
}

fn moves_label(h1: usize, h2: usize) -> String {
    format!("A {} -> B {}", days_label(h1), days_label(h2))
}

fn share(a: u32, b: u32) -> String {
    if b == 0 { "no cases".into() } else { format!("{a} of {b} ({:.0}%)", 100.0 * a as f64 / b as f64) }
}

fn rate(a: u32, b: u32) -> f64 {
    if b == 0 { f64::NAN } else { 100.0 * a as f64 / b as f64 }
}

/// Large counts in words: 2.5 billion.
fn big(n: f64) -> String {
    if n >= 1e12 {
        format!("{:.2} trillion", n / 1e12)
    } else if n >= 1e9 {
        format!("{:.1} billion", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.1} million", n / 1e6)
    } else {
        format!("{n:.0}")
    }
}

/// 42325285 -> "42,325,285".
fn commas(n: f64) -> String {
    let digits = format!("{:.0}", n.abs());
    let mut out = String::new();
    for (k, c) in digits.chars().enumerate() {
        if k > 0 && (digits.len() - k) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0.0 { format!("-{out}") } else { out }
}

fn local_time(unix: u64, offset: i64) -> String {
    let t = unix as i64 + offset;
    let secs = t.rem_euclid(86_400);
    format!("{} {:02}:{:02}", c2_data::date_string(t.div_euclid(86_400) as i32), secs / 3600, secs % 3600 / 60)
}

/// The plan of census `run`: its own description read from its directory (`RunMeta`), and the
/// census variable built from today's prices with the run's own settings together with the run's
/// own correlation periods (matched by date), universes and tasks (`Plan::for_run`), so period
/// and task ids mean what they meant in that run. What today's prices don't cover is marked in
/// the plan, not refused (`Plan::period_available`, `missing_stocks`, `variable_note`). From the
/// shared slot when it holds this run's plan for this store (`store_gen`), else built now in its
/// place. Errors (naming the problem) only when the run can't be read.
fn plan_for_run(slot: &Mutex<PlanEntry>, store: &Store, store_gen: u64, run: &str, dir: &Path) -> Result<Arc<RunPlan>, String> {
    // The slot only ever holds nothing or a complete plan (it is emptied before a build and
    // filled after it succeeds), so a lock poisoned by a panicking worker is still usable.
    let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    let key = (run.to_string(), store_gen);
    if let Some((k, p)) = guard.as_ref()
        && *k == key
    {
        return Ok(p.clone());
    }
    // Free the other run's variable before building this one.
    *guard = None;
    let meta = RunMeta::load(dir).map_err(|e| format!("{e:#}"))?;
    let plan = Plan::for_run(store, &meta).map_err(|e| format!("{e:#}"))?;
    let rp = Arc::new(RunPlan { meta, plan });
    *guard = Some((key, rp.clone()));
    Ok(rp)
}

/// What the Explainer leaves out of a length's periods, or None when today's data covers them
/// all: `of_length` are the run's periods of `length` days (0 = full history) in date order,
/// `covered` the ones today's data covers. The latest left-out one is named with the explainer's
/// reason (`explain::period_unavailable`); as in `c2 census explain` without an end date, the
/// latest period today's data covers is named when later ones are passed over.
fn uncovered_note(rp: &RunPlan, length: usize, of_length: &[usize], covered: &[usize]) -> Option<String> {
    let left_out: Vec<usize> = of_length.iter().copied().filter(|id| !covered.contains(id)).collect();
    let &latest = left_out.last()?;
    let d = c2_data::date_string;
    let span = |id: usize| format!("{}..{}", d(rp.meta.windows[id].first), d(rp.meta.windows[id].last));
    let why = c2_census::explain::period_unavailable(&rp.plan, &rp.meta, latest).unwrap_or_default();
    let mut note = if length == 0 {
        format!("The run's full-history correlation period {} can't be explained with today's data: {why}.", span(latest))
    } else if left_out.len() == 1 {
        format!("One of the run's {} {length}-day correlation periods, {}, can't be explained with today's data, so it isn't offered: {why}.", of_length.len(), span(latest))
    } else {
        format!(
            "{} of the run's {} {length}-day correlation periods can't be explained with today's data, so they aren't offered. The latest of them, {}: {why}.",
            left_out.len(),
            of_length.len(),
            span(latest)
        )
    };
    if of_length.last() == Some(&latest)
        && let Some(&last_covered) = covered.last()
    {
        note.push_str(&format!(" The latest one today's data covers is {}.", span(last_covered)));
    }
    Some(note)
}

/// What the run's plan on today's prices leaves out, in words, or None when it leaves out
/// nothing: correlation periods today's data doesn't cover exactly, stocks it no longer has, and
/// why values recomputed today may differ from the stored ones (an older census variable).
fn plan_note(rp: &RunPlan) -> Option<String> {
    let plan = &rp.plan;
    let (periods, places) = plan.unavailable();
    let mut parts = Vec::new();
    if periods > 0 {
        let last = plan.var.days.last().map(|d| c2_data::date_string(*d)).unwrap_or_default();
        parts.push(format!(
            "{periods} of its {} correlation periods aren't covered exactly by today's price data (which ends {last}), so they are left out",
            plan.windows.len()
        ));
    }
    if places > 0 {
        let stocks: HashSet<u32> = plan.missing_stocks.iter().flatten().copied().collect();
        let (is, their) = if stocks.len() == 1 { ("is", "its") } else { ("are", "their") };
        parts.push(format!(
            "{} of its stocks {is} no longer in today's price data (in {places} of its periods' stock lists), so {their} pairs are left out",
            stocks.len()
        ));
    }
    if let Some(note) = &plan.variable_note {
        parts.push(note.clone());
    }
    (!parts.is_empty()).then(|| format!("Census '{}' read on today's price data: {}.", rp.meta.name, parts.join("; ")))
}

/// A correlation period of a run as the tables show it: its length ("756 days", or "all history")
/// and its last day.
fn period_label(w: &c2_census::meta::RunWindow) -> (String, String) {
    (if w.length == 0 { "all history".to_string() } else { format!("{} days", w.length) }, c2_data::date_string(w.last))
}

/// The timeline's rows for its filters `vk`: each relationship's strongest correlation period
/// among those passing them, strongest first. The order is fixed: ties are broken by the
/// relationship and the period, never by hash order.
fn timeline_view(data: &TimelineData, vk: &TimelineViewKey) -> TimelineView {
    let (stock_first, significant_only, window, horizons, max_shown) = (vk.1, vk.2, vk.3.as_str(), vk.4, vk.5);
    let rolling = |r: &ProfileRow| !data.full_history.contains(&r.window_id);
    // "The most recent": the latest end among the stock's rolling correlation periods; the
    // full-history period, which always ends last, is its own option.
    let recent_end = data.rows.iter().filter(|r| rolling(r)).map(|r| r.period_end.clone()).max();
    let full_days = data.rows.iter().find(|r| !rolling(r)).map(|r| r.period_days);
    let mut lengths: Vec<u32> = data.rows.iter().filter(|r| rolling(r)).map(|r| r.period_days).collect();
    lengths.sort_unstable();
    lengths.dedup();
    let mut pairs: Vec<(u8, u8)> = data.rows.iter().map(|r| (r.h1, r.h2)).collect();
    pairs.sort_unstable();
    pairs.dedup();
    let total_sig = data.rows.iter().filter(|r| r.significant && r.leads == stock_first).count();
    // The row kept for a relationship: highest |z|, then the latest, the longest, the lowest id.
    let better = |a: &ProfileRow, b: &ProfileRow| {
        a.z.abs()
            .total_cmp(&b.z.abs())
            .then_with(|| a.period_end.cmp(&b.period_end))
            .then_with(|| a.period_days.cmp(&b.period_days))
            .then_with(|| b.window_id.cmp(&a.window_id))
            .is_gt()
    };
    let mut best: BTreeMap<RelKey, &ProfileRow> = BTreeMap::new();
    for r in data
        .rows
        .iter()
        .filter(|r| r.leads == stock_first)
        .filter(|r| !significant_only || r.significant)
        .filter(|r| horizons.is_none_or(|h| (r.h1, r.h2) == h))
        .filter(|r| match window {
            "all" => true,
            "recent" => rolling(r) && Some(&r.period_end) == recent_end.as_ref(),
            "full" => !rolling(r),
            days => rolling(r) && r.period_days.to_string() == days,
        })
    {
        let e = best.entry(key(r)).or_insert(r);
        if better(r, e) {
            *e = r;
        }
    }
    let mut rows: Vec<&ProfileRow> = best.into_values().collect();
    rows.sort_by(|a, b| b.z.abs().total_cmp(&a.z.abs()).then_with(|| key(a).cmp(&key(b))));
    rows.truncate(max_shown);
    TimelineView { rows: rows.into_iter().cloned().collect(), lengths, horizons: pairs, total_sig, recent_end, full_days }
}

/// How a forecast's track record compares with luck: (colour, sentence). Two measures: the hit
/// rate over the `n_hit` forecasts where both the predicted and the actual move had a direction
/// (`Skill::directional`), and the predicted vs actual correlation over all `n_corr` forecasts.
/// Each is beyond luck when its z is 2 or more either way. The sentence names the measure that
/// did, and gives the other, so a record isn't described by a measure that didn't fire. Green:
/// one or both better than luck and none worse; red: one or both worse and none better; yellow:
/// neither, or the two in opposite directions. Skill can be negative: a record that points the
/// wrong way more often than luck explains is worse than chance, not "good".
fn track_record_verdict(hit_rate: f64, n_hit: usize, corr: f64, n_corr: usize) -> (egui::Color32, String) {
    let z_hit = (hit_rate - 0.5) / (0.25 / n_hit.max(1) as f64).sqrt();
    let z_corr = corr * (n_corr.max(1) as f64).sqrt();
    // Each measure: 1 better than luck, -1 worse, 0 within what luck produces.
    let side = |z: f64| if z >= 2.0 { 1 } else if z <= -2.0 { -1 } else { 0 };
    let (right, c) = (format!("{:.0}%", 100.0 * hit_rate), format!("{corr:+.2}"));
    let caveat = "with thousands of stocks some look this good by chance, so check the holdout before trusting it.";
    match (side(z_hit), side(z_corr)) {
        (0, 0) => (YELLOW_NOTE, format!("Both are within what luck produces (direction right {right} of the time, correlation {c}), so treat the forecast line as noise.")),
        (1, 1) => (GREEN_NOTE, format!("Both beat luck: its direction was right {right} of the time and its predicted sizes lined up with what happened (correlation {c}); {caveat}")),
        (1, 0) => (GREEN_NOTE, format!("Its direction was right more often than luck explains ({right} of the time), though its predicted sizes were within luck (correlation {c}); {caveat}")),
        (0, 1) => (GREEN_NOTE, format!("Its predicted sizes lined up with what happened beyond luck (correlation {c}), though its direction was right only {right} of the time, within luck; {caveat}")),
        (-1, -1) => (RED_NOTE, format!("Both are worse than luck: its direction was right only {right} of the time and its predicted sizes went the wrong way (correlation {c}). Don't use this forecast.")),
        (-1, 0) => (RED_NOTE, format!("Its direction was wrong more often than luck explains (right only {right} of the time), though its predicted sizes were within luck (correlation {c}). Don't use this forecast.")),
        (0, -1) => (RED_NOTE, format!("Its predicted sizes went the wrong way (correlation {c}, beyond luck), though its direction was right {right} of the time. Don't use this forecast.")),
        (1, -1) => (YELLOW_NOTE, format!("The two measures disagree: its direction was right more often than luck explains ({right} of the time), but its predicted sizes went the wrong way (correlation {c}, beyond luck); treat the forecast line as noise.")),
        _ => (YELLOW_NOTE, format!("The two measures disagree: its predicted sizes lined up with what happened beyond luck (correlation {c}), but its direction was wrong more often than luck explains (right only {right} of the time); treat the forecast line as noise.")),
    }
}

/// Step 5's verdict on one pair against its matrix's significance cut-off `cutoff` (|r|, as the
/// census's summary stores it, to 4 decimals). The census applies the test to its GPU estimate of
/// r (`r_int8`), so when the pair is stored its flag is the verdict; otherwise it is judged here
/// from the exact r. `tested` is false when a stock wasn't in the period's universe.
fn significance_verdict(cutoff: f64, exact_r: f64, stored: Option<(f32, bool)>, tested: bool) -> String {
    let lead = format!("The significance test passed pairs whose |r| is {cutoff:.4} or more here (above that line at most about 5% are expected to be chance).");
    let exact = exact_r.abs();
    let side = |v: f64| if v >= cutoff { "at or above" } else { "below" };
    match stored {
        Some((r8, passed)) => {
            let est = (r8 as f64).abs();
            let mut s = format!(
                "{lead} The census applies it to its GPU estimate of r, before the exact value is computed: for this pair |r| {est:.4}, so this pair {} \
                 (as step 4 says). The exact r, |r| {exact:.4}, is {} the cut-off",
                if passed { "passed" } else { "did not pass" },
                side(exact)
            );
            if (exact >= cutoff) != passed {
                s.push_str("; the two differ slightly because the GPU works in 8-bit arithmetic, and the census's decision is the one made on its estimate");
            }
            if (est >= cutoff) != passed {
                s.push_str(" (the cut-off is stored rounded to 4 decimals)");
            }
            s.push('.');
            s
        }
        None if !tested => format!(
            "{lead} The census never tested this pair in this period (see step 1). By the exact r recomputed here, |r| {exact:.4}, it would {}.",
            if exact >= cutoff { "pass" } else { "not pass" }
        ),
        None => format!(
            "{lead} The census didn't store this pair, so its own decision isn't on file; judged by the exact r recomputed here, |r| {exact:.4}, it {}. \
             (The census decides with its GPU estimate of r, which can differ slightly from the exact r.)",
            if exact >= cutoff { "passes" } else { "does not pass" }
        ),
    }
}

/// One line per period for the plain-terms counts.
fn words_text(w: &Words, a: &str, b: &str, h1: usize, h2: usize, lag: usize) -> Vec<String> {
    let (up, down) = (100.0 * w.up.exp_m1(), -100.0 * w.down.exp_m1());
    let ending = if lag == 0 { "ending the same day".to_string() } else { format!("ending {} later", tdays(lag)) };
    let line = |c: &Counts, intro: String| {
        format!(
            "{intro}: after {a} rose {up:.1}% or more over {} ({} times), {b} rose over the {} {ending} in {}. \
             After {a} fell {down:.1}% or more ({} times), {b} rose in {}. Overall {b} rose in {}.",
            days_label(h1), c.a_up, days_label(h2), share(c.a_up_b_up, c.a_up), c.a_down, share(c.a_down_b_up, c.a_down), share(c.b_up, c.cases)
        )
    };
    let mut out = vec![line(&w.period, format!("In the correlation period ({} observations)", w.period.cases))];
    if w.rest.cases == 0 {
        out.push("This correlation period is the whole history, so nothing is left over to check it against.".into());
    } else {
        out.push(line(&w.rest, format!("Over the rest of the history, outside this period ({} observations, the honest check)", w.rest.cases)));
    }
    out
}

fn day_cells(s: &c2_census::explain::DaySteps, market: f64, d: usize) -> [String; 7] {
    let f = |v: f64, text: String| if v.is_finite() { text } else { "none".to_string() };
    [
        format!("${:.2} -> ${:.2}", s.close[d - 1], s.close[d]),
        f(s.ret[d] as f64, format!("{:+.3}%", 100.0 * s.ret[d])),
        f(market, format!("{:+.3}%", 100.0 * market)),
        f(s.beta[d], format!("{:.3}", s.beta[d])),
        f(s.resid[d], format!("{:+.3}%", 100.0 * s.resid[d])),
        f(s.sd[d], format!("{:+.3}% / {:.3}%", 100.0 * s.mean[d], 100.0 * s.sd[d])),
        f(s.v[d] as f64, format!("{:+.3}", s.v[d])),
    ]
}

fn words_short(w: &Words) -> String {
    let c = &w.period;
    if c.a_up == 0 {
        return String::new();
    }
    format!("A +{:.1}% or more: B rose {}/{} ({:.0}%), normally {:.0}%", 100.0 * w.up.exp_m1(), c.a_up_b_up, c.a_up, rate(c.a_up_b_up, c.a_up), rate(c.b_up, c.cases))
}

/// Plot x-axis labels for categorical bars at integer positions.
fn category_axis(labels: Vec<String>) -> impl Fn(egui_plot::GridMark, &std::ops::RangeInclusive<f64>) -> String {
    move |m, _| {
        let k = m.value.round();
        if (m.value - k).abs() < 1e-6 && k >= 0.0 { labels.get(k as usize).cloned().unwrap_or_default() } else { String::new() }
    }
}

impl App {
    fn new() -> App {
        let dir = store_dir();
        // Store::open's error already says to run `c2 ingest`.
        let (store, store_note, decades, store_stamp) = match open_store(&dir) {
            Ok((s, note, d, stamp)) => (Some(s), note, d, stamp),
            Err(e) => (None, format!("No price store yet: {e}"), Vec::new(), store_stamp(&dir)),
        };
        // Self-checks can open another run than the one in use (`C2_GUI_RUN`).
        let run_name = std::env::var("C2_GUI_RUN").ok().filter(|r| !r.trim().is_empty()).unwrap_or_else(run::latest_run);
        App {
            tab: Tab::Status,
            run_input: run_name.clone(),
            run_name,
            run_edited: None,
            store,
            store_note,
            decades,
            store_stamp,
            store_checked: Instant::now(),
            store_settling: None,
            store_reload: Job::default(),
            store_gen: 0,
            store_changed: None,
            status: None,
            status_err: String::new(),
            pid: None,
            control: "run".into(),
            last_poll: Instant::now() - Duration::from_secs(10),
            message: String::new(),
            launch: None,
            launch_note: None,
            lag_table: BTreeMap::new(),
            lag_table_run: String::new(),
            lag_note: String::new(),
            pair: (1, 1),
            ticker: "AAPL".into(),
            pairs: Job::default(),
            family: 0,
            start: "2024-06-28".into(),
            picks: 20,
            rebalance: 1,
            costs: "0,5,20".into(),
            random: 1000,
            plan_slot: Arc::new(Mutex::new(None)),
            holdout: Job::default(),
            holdout_started: None,
            shot: std::env::var("C2_GUI_SCREENSHOT").ok().map(|prefix| Shot {
                prefix,
                frames: 0,
                requested: false,
                started: Instant::now(),
                timeout: Duration::from_secs(std::env::var("C2_GUI_TIMEOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(600)),
                min_wait: Duration::from_secs(std::env::var("C2_GUI_WAIT").ok().and_then(|s| s.parse().ok()).unwrap_or(0)),
                plan: std::env::var("C2_GUI_PLAN").is_ok(),
            }),
            tl: Timeline { input: "NAD".into(), window: "all".into(), max_shown: 40, ..Default::default() },
            fc: ForecastView { horizon: 21, amount: "10000".into(), ..Default::default() },
            ex: ExplainView {
                leader: "NZF".into(),
                follower: "NAD".into(),
                leader_input: "NZF".into(),
                follower_input: "NAD".into(),
                h1: 1,
                h2: 1,
                lag: 1,
                threshold_pct: 3.0,
                ..Default::default()
            },
            plan: Job::default(),
            stats: Job::default(),
            runs: Job::default(),
            runs_refresh: 0,
            utc_offset: std::process::Command::new("date")
                .arg("+%z")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|z| {
                    let z = z.trim();
                    let sign = if z.starts_with('-') { -1 } else { 1 };
                    let (h, m) = (z.get(1..3)?.parse::<i64>().ok()?, z.get(3..5)?.parse::<i64>().ok()?);
                    Some(sign * (h * 3600 + m * 60))
                })
                .unwrap_or(0),
            rank: Job::default(),
            rank_view: None,
            rank_filter: String::new(),
            rank_sort: 0,
        }
    }

    /// Switches every tab to census `name`: everything read from the previous run is dropped
    /// (results, jobs in flight, the census variable), and choices that name a correlation period
    /// carry over by date.
    fn set_run(&mut self, name: &str) {
        let name = name.trim();
        self.run_edited = None;
        if name.is_empty() || name == self.run_name {
            return;
        }
        self.run_name = name.to_string();
        self.clear_results();
        self.tl.selected = None;
        self.ex.pending_window = None;
        if let Some(sel) = self.ex.selected.take() {
            self.ex.pending_end = Some(sel);
        }
        self.message.clear();
        // Read the new run's status now, so the Start / Stop buttons reflect it this frame.
        self.last_poll = Instant::now() - Duration::from_secs(10);
        self.poll();
    }

    /// Drops everything read from the run or through the price store: results, jobs in flight,
    /// views made from them and the run's plan. The choices on each tab stay, and their results
    /// are read again when a tab asks for them.
    fn clear_results(&mut self) {
        self.rank.clear();
        self.rank_view = None;
        self.tl.job.clear();
        self.tl.view = None;
        self.fc.data.clear();
        self.fc.job.clear();
        self.plan.clear();
        // Free the plan's census variable now unless a worker is using the slot (its entry is
        // keyed by run and store, so a stale one is never reused).
        if let Ok(mut guard) = self.plan_slot.try_lock() {
            *guard = None;
        }
        self.ex.job.clear();
        self.ex.key = None;
        self.stats.clear();
        self.pairs.clear();
        self.lag_table.clear();
        self.lag_table_run.clear();
        self.lag_note.clear();
        self.holdout.clear();
        self.holdout_started = None;
    }

    /// Reopens the price store when its files change (`c2 update` and `c2 ingest` rebuild it), so
    /// new runs can be read and the newest periods aren't taken for gaps in the data. Checked every
    /// few seconds; the store is reopened in the background once its files read the same at two
    /// checks in a row with no rebuild writing them (a rebuild renames two files in, one after the
    /// other).
    fn watch_store(&mut self) {
        if self.store_checked.elapsed() < STORE_CHECK || self.store_reload.is_running() {
            return;
        }
        self.store_checked = Instant::now();
        let dir = store_dir();
        let now = store_stamp(&dir);
        if now == self.store_stamp {
            self.store_settling = None;
            return;
        }
        if now.is_some() && self.store_settling.as_ref() == Some(&now) {
            self.store_settling = None;
            self.store_reload.start(self.store_gen + 1, move || open_store(&dir));
        } else {
            self.store_settling = Some(now);
        }
    }

    /// Puts a reopened store in place of the old one and drops everything read through the old
    /// one. A store that can't be opened yet is tried again after the next change check.
    fn apply_store_reload(&mut self) {
        let Some((generation, res)) = self.store_reload.done.take() else { return };
        match res {
            Ok((store, note, decades, stamp)) => {
                self.store = Some(store);
                self.store_note = note;
                self.decades = decades;
                self.store_stamp = stamp;
                self.store_gen = generation;
                self.clear_results();
                // The Explainer keeps its period by date: which periods today's data covers (and
                // so their positions in its list) may have changed.
                if let Some(sel) = self.ex.selected {
                    self.ex.pending_end = Some(sel);
                }
                self.store_changed = Some((Instant::now(), "Price data changed; reloaded".to_string()));
            }
            Err(e) => self.store_changed = Some((Instant::now(), format!("Price data changed but can't be read yet ({e}); trying again"))),
        }
    }

    /// Collects every finished background job, whichever tab is shown. Self-checks report how
    /// long each took.
    fn poll_jobs(&mut self) {
        let timing = self.shot.is_some();
        let report = |what: &str, arrived: bool, took: Duration| {
            if timing && arrived {
                eprintln!("c2-gui self-check: {what} ready after {:.3} s", took.as_secs_f64());
            }
            arrived
        };
        report("Strongest ranking", self.rank.poll(), self.rank.took);
        report("Stock timeline", self.tl.job.poll(), self.tl.job.took);
        report("forecast data", self.fc.data.poll(), self.fc.data.took);
        // The as-of box shows the day of the forecast for the current choices (not an older one
        // that finished while those were being computed).
        if report("forecast", self.fc.job.poll(), self.fc.job.took)
            && let Some((k, Ok(f))) = &self.fc.job.done
            && *k == self.forecast_key()
        {
            self.fc.as_of_text = c2_data::date_string(f.as_of);
        }
        report("plan (census variable)", self.plan.poll(), self.plan.took);
        if report("explanation", self.ex.job.poll(), self.ex.job.took)
            && let Some((_, Ok(e))) = &self.ex.job.done
        {
            self.ex.obs = self.ex.obs.min(e.obs.len().saturating_sub(1));
        }
        report("run statistics", self.stats.poll(), self.stats.took);
        report("list of runs", self.runs.poll(), self.runs.took);
        report("stored pairs", self.pairs.poll(), self.pairs.took);
        report("holdout", self.holdout.poll(), self.holdout.took);
        if report("price store reopened", self.store_reload.poll(), self.store_reload.took) {
            self.apply_store_reload();
        }
    }

    /// True while any background job is computing (the window then checks for results often).
    fn jobs_running(&self) -> bool {
        self.rank.is_running()
            || self.tl.job.is_running()
            || self.fc.data.is_running()
            || self.fc.job.is_running()
            || self.plan.is_running()
            || self.ex.job.is_running()
            || self.stats.is_running()
            || self.runs.is_running()
            || self.pairs.is_running()
            || self.holdout.is_running()
            || self.store_reload.is_running()
    }

    /// Whether the shown tab has everything it asked for (a result or an error), for self-checks.
    fn tab_ready(&self) -> bool {
        let run = &self.run_name;
        let has_store = self.store.is_some();
        // The Stock timeline and the Forecast wait for the run's plan; if it can't be built,
        // they show that error and wait for nothing else.
        let plan_failed = matches!(self.plan.get(run), Some(Err(_)));
        match self.tab {
            Tab::Status => self.runs.get(&(self.runs_refresh, run.clone())).is_some() && (!has_store || self.stats.get(run).is_some()),
            Tab::Strongest => !has_store || (self.rank.get(run).is_some() && self.stats.get(run).is_some()),
            Tab::Stock => !has_store || self.tl.want.as_ref().is_none_or(|t| plan_failed || self.tl.job.get(&(run.clone(), t.clone())).is_some()),
            Tab::Forecast => {
                !has_store
                    || self.fc.ticker.is_empty()
                    || plan_failed
                    || matches!(self.fc.data.get(&(run.clone(), self.fc.ticker.clone())), Some(Err(_)))
                    || (self.fc.walk == 0 && self.fc.walk_started.is_none() && self.fc.job.get(&self.forecast_key()).is_some())
            }
            Tab::Explain => {
                !has_store
                    || match self.plan.get(run) {
                        None => false,
                        Some(Err(_)) => true,
                        // No key: no correlation period of the chosen length, so nothing to wait for.
                        Some(Ok(_)) => self.ex.key.as_ref().is_none_or(|k| self.ex.job.get(k).is_some()),
                    }
            }
            Tab::Census => !self.pairs.is_running(),
            Tab::Holdout => !has_store || (!self.holdout.is_running() && self.holdout.done.is_some()),
            Tab::Data => true,
        }
    }

    fn strongest_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Strongest lead-lag relationships");
        let Some(store) = self.store.clone() else {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.store_note);
            return;
        };
        let run = self.run_name.clone();
        // The run's statistics give the number of correlations tested (below).
        self.want_stats();
        if self.rank.wants(&run) {
            let dir = self.dir();
            self.rank.start(run.clone(), move || {
                let rows = c2_census::analyze::stock_ranking(&store, &dir).map_err(|e| format!("{e:#}"))?;
                // Each row's strongest relationship names its period by the run's own id.
                let periods = RunMeta::load(&dir).map_err(|e| format!("{e:#}"))?.windows.iter().map(period_label).collect();
                Ok(Ranking { rows, periods })
            });
        }
        let ranking = match self.rank.get(&run) {
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!("reading census '{run}' (a few seconds)…"));
                });
                return;
            }
            Some(Err(e)) => {
                ui.colored_label(egui::Color32::LIGHT_RED, format!("Could not rank the stocks of census '{run}': {e}"));
                if ui.button("Try again").clicked() {
                    self.rank.clear();
                }
                return;
            }
            Some(Ok(ranking)) => ranking,
        };
        let rank = &ranking.rows;
        if rank.is_empty() {
            ui.label(format!("Census '{run}' has no stored lead-lag relationships yet."));
            return;
        }
        let with_sig = rank.iter().filter(|r| r.significant > 0).count();
        const SORTS: [&str; 4] = [
            "number x strength (sum of |r|)",
            "significant relationships (sum of |r| over those)",
            "strongest single correlation",
            "most relationships",
        ];
        ui.horizontal(|ui| {
            ui.label("Find:");
            ui.add(egui::TextEdit::singleline(&mut self.rank_filter).desired_width(80.0));
            ui.label("Sort by:");
            egui::ComboBox::from_id_salt("rank_sort").selected_text(SORTS[self.rank_sort]).show_ui(ui, |ui| {
                for (k, s) in SORTS.iter().enumerate() {
                    ui.selectable_value(&mut self.rank_sort, k, *s);
                }
            });
        });
        // The number of lead-lag correlations this run tested, from the same run's statistics (the
        // Status tab's) once they have been read.
        let tested = match self.stats.get(&run) {
            Some(Ok((st, _))) => format!("when {} lead-lag correlations are tested, as in census '{run}'", big(st.lagged.correlations)),
            _ => "when every pair of stocks is tested at every offset, move length and correlation period".to_string(),
        };
        ui.label(format!(
            "Stocks with both many stored lead-lag correlations and strong ones rank highest (score = sum of |r|). Most stored correlations are within \
             what chance produces {tested}; {with_sig} stocks have relationships that passed the significance test ('sig.'), and \
             'periods' counts how many separate correlation periods those come from (1 usually means one event, such as the 2020 crash). The \
             columns from 'strongest' on describe the stock's strongest relationship and its correlation period. Click a stock for its timeline."
        ));
        // Filtered, sorted and formatted only when the result, the find text or the sort changes.
        let view_key: RankViewKey = (self.rank.generation, self.rank_filter.trim().to_uppercase(), self.rank_sort);
        if self.rank_view.as_ref().is_none_or(|(k, _)| *k != view_key) {
            let (filter, sort) = (&view_key.1, view_key.2);
            let mut rows: Vec<&c2_census::analyze::RankRow> = rank.iter().filter(|r| filter.is_empty() || r.ticker.contains(filter.as_str())).filter(|r| sort != 1 || r.significant > 0).collect();
            let score = |r: &c2_census::analyze::RankRow| match sort {
                1 => r.sig_sum_abs_r,
                2 => r.max_abs_r as f64,
                3 => r.stored as f64,
                _ => r.sum_abs_r,
            };
            rows.sort_by(|a, b| score(b).total_cmp(&score(a)).then_with(|| a.ticker.cmp(&b.ticker)));
            let lines = rows
                .iter()
                .take(1000)
                .map(|r| {
                    // The strongest relationship, laid out as on the Stock timeline: best_h1 is the
                    // move length of whichever stock moves first (A), best_h2 the follower's (B).
                    let (a, b) = if r.best_leads { (&r.ticker, &r.best_partner) } else { (&r.best_partner, &r.ticker) };
                    let (period, ending) = ranking.periods.get(r.best_window as usize).cloned().unwrap_or_default();
                    RankLine {
                        ticker: r.ticker.clone(),
                        cells: [
                            format!("{:.0}", if sort == 1 { r.sig_sum_abs_r } else { r.sum_abs_r }),
                            r.stored.to_string(),
                            format!("{:.3}", r.mean_abs_r),
                            format!("{:.3}", r.max_abs_r),
                            if r.significant > 0 { format!("{} ({}↗ {}↙)", r.significant, r.leads, r.follows) } else { "0".into() },
                            if r.significant > 0 { r.windows.to_string() } else { String::new() },
                            format!("{a} -> {b}"),
                            days_label(r.best_lag as usize),
                            format!("{} -> {}", days_label(r.best_h1 as usize), days_label(r.best_h2 as usize)),
                            format!("{:+.3}", r.best_r),
                            format!("{:.1}", r.best_z),
                            period,
                            ending,
                        ],
                    }
                })
                .collect();
            self.rank_view = Some((view_key, lines));
        }
        let mut open: Option<String> = None;
        if let Some((_, lines)) = &self.rank_view {
            egui::ScrollArea::both().id_salt("rank").show(ui, |ui| {
                egui::Grid::new("rank_grid").striped(true).show(ui, |ui| {
                    for h in ["#", "stock", "score", "relationships", "mean |r|", "max |r|", "sig.", "periods", "strongest: A -> B", "offset", "move lengths", "r", "z", "period", "ending"] {
                        ui.strong(h);
                    }
                    ui.end_row();
                    for (k, line) in lines.iter().enumerate() {
                        ui.label(format!("{}", k + 1));
                        if ui.link(&line.ticker).clicked() {
                            open = Some(line.ticker.clone());
                        }
                        for c in &line.cells {
                            ui.label(c);
                        }
                        ui.end_row();
                    }
                });
            });
        }
        if let Some(t) = open {
            self.tl.input = t.clone();
            self.tl.significant_only = self.rank_sort == 1;
            self.tl.window = "all".into();
            self.tl.horizons = None;
            self.request_timeline(&t);
            self.tab = Tab::Stock;
        }
    }

    /// Asks for `ticker`'s timeline (it loads when the Stock timeline tab is shown). The Forecast
    /// tab follows the same stock. Asking again for the stock shown reloads it.
    fn request_timeline(&mut self, ticker: &str) {
        let ticker = ticker.trim().to_uppercase();
        if ticker.is_empty() {
            return;
        }
        let key = (self.run_name.clone(), ticker.clone());
        if self.tl.job.get(&key).is_some() {
            self.tl.job.clear();
        }
        if self.tl.want.as_ref() != Some(&ticker) {
            self.tl.selected = None;
        }
        self.tl.want = Some(ticker.clone());
        self.fc.ticker = ticker;
    }

    /// Rebuilds the timeline's filtered rows when the loaded rows or the filters have changed.
    fn update_timeline_view(&mut self, job_key: &(String, String)) {
        let vk: TimelineViewKey = (self.tl.job.generation, self.tl.stock_first, self.tl.significant_only, self.tl.window.clone(), self.tl.horizons, self.tl.max_shown);
        if self.tl.view.as_ref().is_none_or(|(k, _)| *k != vk)
            && let Some(Ok(data)) = self.tl.job.get(job_key)
        {
            let view = timeline_view(data, &vk);
            self.tl.view = Some((vk, view));
        }
    }

    fn open_explainer(&mut self, leader: String, follower: String, h1: usize, h2: usize, lag: usize, window: usize) {
        self.ex.leader_input = leader.clone();
        self.ex.follower_input = follower.clone();
        self.ex.leader = leader;
        self.ex.follower = follower;
        self.ex.h1 = h1;
        self.ex.h2 = h2;
        self.ex.lag = lag;
        // The id is a correlation period of the run the row was read from: the current run.
        self.ex.pending_window = Some((self.run_name.clone(), window));
        self.ex.pending_end = None;
        self.ex.obs = 0;
        self.ex.day = None;
        self.tab = Tab::Explain;
    }

    fn stock_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Stock timeline");
        ui.horizontal(|ui| {
            ui.label("Ticker:");
            let resp = ui.add(egui::TextEdit::singleline(&mut self.tl.input).desired_width(80.0));
            let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if ui.add_enabled(self.store.is_some(), egui::Button::new("Show")).clicked() || enter {
                let t = self.tl.input.clone();
                self.request_timeline(&t);
            }
            if self.tl.job.is_running() {
                ui.spinner();
            }
        });
        let Some(store) = self.store.clone() else {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.store_note);
            return;
        };
        let Some(want) = self.tl.want.clone() else {
            ui.label("Enter a ticker to see which stocks' moves show up in it later, or where its own moves show up later, after how many days and by how much.");
            return;
        };
        let run = self.run_name.clone();
        let job_key = (run.clone(), want.clone());
        if self.tl.job.wants(&job_key) {
            // Sizes and plain-terms counts come from the run's census variable (built once per
            // run and shared with the Forecast and Explainer tabs).
            let plan = match self.need_plan() {
                Ok(Some(plan)) => plan,
                Ok(None) => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(format!("Building the census variable for every stock, for census '{run}' (a few seconds)…"));
                    });
                    return;
                }
                Err(e) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, format!("The Stock timeline can't read census '{run}': {e}"));
                    if ui.button("Try again").clicked() {
                        self.plan.clear();
                    }
                    return;
                }
            };
            let dir = self.dir();
            let t = want.clone();
            self.tl.job.start(job_key.clone(), move || {
                let rows = c2_census::analyze::stock_profile_with(&store, &plan.plan.var, &dir, &t, false).map_err(|e| format!("{e:#}"))?;
                let full_history = plan.meta.windows.iter().filter(|w| w.length == 0).map(|w| w.id as u32).collect();
                Ok(TimelineData { rows, full_history })
            });
        }
        let data = match self.tl.job.get(&job_key) {
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!("reading census '{run}' for {want}…"));
                });
                return;
            }
            Some(Err(e)) => {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
                return;
            }
            Some(Ok(data)) => data,
        };
        if data.rows.is_empty() {
            ui.label(format!("Census '{run}' stored no lead-lag relationships for {want}."));
            return;
        }
        // What the run's plan on today's prices left out (its relationships from uncovered
        // periods aren't shown).
        if let Some(note) = self.plan_note_for(&run) {
            ui.colored_label(YELLOW_NOTE, note);
        }
        // The ticker of the rows loaded (the job's key), never a newer one typed in the box.
        let t = want;
        self.update_timeline_view(&job_key);
        let Some((_, view)) = &self.tl.view else { return };
        let (lengths, horizons, total_sig) = (view.lengths.clone(), view.horizons.clone(), view.total_sig);
        let (recent_end, full_days) = (view.recent_end.clone(), view.full_days);
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.tl.stock_first, false, format!("What moves {t} later (other stocks move first)"));
            ui.radio_value(&mut self.tl.stock_first, true, format!("What {t} moves later ({t} moves first)"));
            ui.separator();
            if ui.button(format!("Forecast {t}'s price ▶")).clicked() {
                self.fc.ticker = t.clone();
                self.tab = Tab::Forecast;
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.tl.significant_only, format!("significant only ({total_sig})"));
            let wl = |w: &str| match w {
                "all" => "any (strongest of each)".to_string(),
                "recent" => match &recent_end {
                    Some(end) => format!("the most recent (ending {end})"),
                    None => "the most recent".to_string(),
                },
                "full" => match full_days {
                    Some(d) => format!("all history ({d} days)"),
                    None => "all history".to_string(),
                },
                d => format!("{d} days"),
            };
            ui.label("Correlation period:");
            egui::ComboBox::from_id_salt("tl_window").selected_text(wl(&self.tl.window)).show_ui(ui, |ui| {
                let full = full_days.map(|_| "full".to_string());
                for w in ["all".to_string(), "recent".to_string()].into_iter().chain(full).chain(lengths.iter().map(|l| l.to_string())) {
                    let label = wl(&w);
                    ui.selectable_value(&mut self.tl.window, w, label);
                }
            });
            let hl = |h: Option<(u8, u8)>| h.map(|(a, b)| moves_label(a as usize, b as usize)).unwrap_or_else(|| "any".into());
            ui.label("Move lengths:");
            egui::ComboBox::from_id_salt("tl_horizons").selected_text(hl(self.tl.horizons)).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.tl.horizons, None, hl(None));
                for h in &horizons {
                    ui.selectable_value(&mut self.tl.horizons, Some(*h), hl(Some(*h)));
                }
            });
            ui.add(egui::Slider::new(&mut self.tl.max_shown, 5..=200).text("shown"));
            ui.checkbox(&mut self.tl.show_r, "plot r instead of % move");
        });
        // The filters may have changed just now.
        self.update_timeline_view(&job_key);
        let Some((_, view)) = &self.tl.view else { return };
        let rows = &view.rows;
        if self.tl.select_first && !rows.is_empty() {
            self.tl.select_first = false;
            self.tl.selected = rows.first().map(key);
        }
        let (all_rows, full_history) = match self.tl.job.get(&job_key) {
            Some(Ok(data)) => (&data.rows, &data.full_history),
            _ => return,
        };
        let selected = self.tl.selected.clone();
        let stock_first = self.tl.stock_first;
        let show_r = self.tl.show_r;
        let mut explain: Option<ProfileRow> = None;
        egui::ScrollArea::vertical().id_salt("tl_scroll").show(ui, |ui| {
            ui.label(if stock_first {
                format!("A = {t}, which moves first; B = the stock that follows. A point at +k days: when {t} moves, that stock tends to move k trading days \
                         later (offset from the end of {t}'s move to the end of B's); height = how far B tends to move beyond what the market did, per 1% \
                         that {t} moves beyond the market. Orange = passed the significance test; grey = within what chance produces.")
            } else {
                format!("B = {t}, which follows; A = the stock that moves first. A point at +k days: when that stock moves, {t} tends to move k trading days \
                         later (offset from the end of A's move to the end of {t}'s); height = how far {t} tends to move beyond what the market did, per 1% \
                         that A moves beyond the market (the multiplier the Forecast tab applies, except to the full-history period, which it doesn't use). \
                         Orange = passed the significance test; grey = within what chance produces.")
            });
            let y_of = |r: &ProfileRow| if show_r { r.r as f64 } else { r.move_per_pct.unwrap_or(r.r as f64) };
            let x_of = |r: &ProfileRow| r.lag as f64;
            Plot::new("timeline")
                .height(280.0)
                .x_axis_label("offset: trading days later")
                .y_axis_label(if show_r { "r".to_string() } else { "% vs market, per 1% of A".to_string() })
                .legend(Legend::default())
                .show(ui, |p| {
                    p.vline(VLine::new("today", 0.0).color(egui::Color32::WHITE).width(2.0));
                    p.hline(HLine::new("no move", 0.0).color(egui::Color32::DARK_GRAY));
                    let pts = |sig: bool| -> Vec<[f64; 2]> { rows.iter().filter(|r| r.significant == sig).map(|r| [x_of(r), y_of(r)]).collect() };
                    p.points(Points::new("within chance", PlotPoints::from(pts(false))).color(GREY).radius(4.0).shape(MarkerShape::Circle).filled(true));
                    p.points(Points::new("significant", PlotPoints::from(pts(true))).color(ORANGE).radius(6.0).shape(MarkerShape::Diamond).filled(true));
                    for (k, r) in rows.iter().enumerate() {
                        let is_sel = selected.as_ref() == Some(&key(r));
                        if k < 25 || r.significant || is_sel {
                            let c = if is_sel { egui::Color32::YELLOW } else if r.significant { ORANGE } else { GREY };
                            // Alternate sides so partners at the same offset don't print on top of each other.
                            let anchor = if k % 2 == 0 { egui::Align2::LEFT_CENTER } else { egui::Align2::RIGHT_CENTER };
                            p.text(Text::new("", PlotPoint::new(x_of(r), y_of(r)), format!(" {} +{}d ", r.partner, r.lag)).color(c).anchor(anchor));
                        }
                    }
                    let today = if stock_first { format!("{t} moves") } else { "A moves".to_string() };
                    p.text(Text::new("", PlotPoint::new(0.0, 0.0), today).color(egui::Color32::WHITE).anchor(egui::Align2::LEFT_BOTTOM));
                });
            ui.strong("Relationships (click one for details)");
            // Scrolls sideways too, so the last columns stay reachable in a narrow window.
            egui::ScrollArea::both().id_salt("tl_rows").max_height(260.0).show(ui, |ui| {
                egui::Grid::new("tl_grid").striped(true).show(ui, |ui| {
                    for h in ["", "A -> B", "offset", "move lengths", "B per 1% of A\n(vs market)", "r", "z", "chance", "in your words (in the correlation period)", "period", "ending"] {
                        ui.strong(h);
                    }
                    ui.end_row();
                    for r in rows {
                        let k = key(r);
                        let sel = self.tl.selected.as_ref() == Some(&k);
                        if ui.selectable_label(sel, if r.significant { "★" } else { "·" }).clicked() {
                            self.tl.selected = Some(k.clone());
                        }
                        let arrow = if r.leads { format!("{t} -> {}", r.partner) } else { format!("{} -> {t}", r.partner) };
                        if ui.selectable_label(sel, arrow).clicked() {
                            self.tl.selected = Some(k);
                        }
                        ui.label(days_label(r.lag as usize));
                        ui.label(format!("{} -> {}", days_label(r.h1 as usize), days_label(r.h2 as usize)));
                        ui.label(r.move_per_pct.map(|m| format!("{m:+.2}%")).unwrap_or_default());
                        ui.label(format!("{:+.3}", r.r));
                        ui.label(format!("{:+.1}", r.z));
                        ui.label(format!("{:.0} at |z|≥{}", r.chance, r.chance_z));
                        ui.label(r.words.as_ref().map(words_short).unwrap_or_default());
                        ui.label(days_label(r.period_days as usize));
                        ui.label(&r.period_end);
                        ui.end_row();
                    }
                });
            });
            ui.add_space(4.0);
            let sel_row = self.tl.selected.as_ref().and_then(|k| rows.iter().find(|r| key(r) == *k)).cloned();
            match sel_row {
                None => {
                    ui.label("Select a relationship to see it in plain terms, step through how it is calculated, and see it across every correlation period: \
                              a persistent relationship shows up again and again with the same sign; a one-off event shows up once.");
                }
                Some(r) => {
                    let (a, b) = if r.leads { (t.clone(), r.partner.clone()) } else { (r.partner.clone(), t.clone()) };
                    let hist: Vec<&ProfileRow> = all_rows.iter().filter(|x| key(x) == key(&r)).collect();
                    ui.columns(2, |cols| {
                        let ui = &mut cols[0];
                        ui.strong(format!(
                            "{a} -> {b}: {a}'s move over {} against {b}'s move over {} ending {} later",
                            days_label(r.h1 as usize), days_label(r.h2 as usize), tdays(r.lag as usize)
                        ));
                        if ui.button("Explain how this is calculated ▶").clicked() {
                            explain = Some(r.clone());
                        }
                        ui.label(format!("Shown: the {}-day correlation period ending {} (r {:+.3} over {} observations).", r.period_days, r.period_end, r.r, r.obs));
                        if let Some(m) = r.move_per_pct {
                            // The Forecast uses rolling periods only: its track record could never
                            // check the full-history one.
                            let use_note = if full_history.contains(&r.window_id) {
                                "the Forecast tab doesn't apply it: it never uses the full-history correlation period, which its track record could never check"
                            } else {
                                "the multiplier the Forecast tab applies"
                            };
                            ui.label(format!(
                                "Size: when {a} moves 1% beyond what the market did over {}, {b} tends to move {m:+.2}% beyond the market over the {} \
                                 ending {} later ({use_note}).",
                                days_label(r.h1 as usize),
                                days_label(r.h2 as usize),
                                tdays(r.lag as usize)
                            ));
                        }
                        if let Some(w) = &r.words {
                            for s in words_text(w, &a, &b, r.h1 as usize, r.h2 as usize, r.lag as usize) {
                                ui.label(s);
                            }
                        }
                        let ui = &mut cols[1];
                        let same_sign = hist.iter().filter(|x| x.r.signum() == r.r.signum()).count();
                        ui.label(format!(
                            "Stored in {} correlation period(s); same sign in {same_sign}. Where it is absent, it was not among that period's strongest.",
                            hist.len()
                        ));
                        let mut pts: Vec<[f64; 2]> = hist.iter().map(|x| [window_end_year(&x.window), x.r as f64]).collect();
                        pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
                        Plot::new("history").height(170.0).x_axis_label("end of correlation period (year)").y_axis_label("r").show(ui, |p| {
                            p.hline(HLine::new("0", 0.0).color(egui::Color32::DARK_GRAY));
                            p.points(Points::new("r by period", PlotPoints::from(pts)).radius(5.0).color(egui::Color32::YELLOW).filled(true));
                        });
                    });
                }
            }
        });
        if let Some(r) = explain {
            let (a, b) = if r.leads { (t.clone(), r.partner.clone()) } else { (r.partner.clone(), t.clone()) };
            self.open_explainer(a, b, r.h1 as usize, r.h2 as usize, r.lag as usize, r.window_id as usize);
        }
    }

    fn forecast_key(&self) -> ForecastKey {
        (self.run_name.clone(), self.fc.ticker.clone(), self.fc.as_of, self.fc.horizon, self.fc.significant_only, self.fc.max_age_years)
    }

    fn forecast_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Forecast");
        if self.fc.ticker.is_empty() {
            self.fc.ticker = self.tl.want.clone().unwrap_or_else(|| self.tl.input.trim().to_uppercase());
        }
        let current = match &self.fc.job.done {
            Some((k, Ok(f))) if k.0 == self.run_name => Some(f.as_of),
            _ => None,
        };
        ui.horizontal_wrapped(|ui| {
            ui.label("Ticker:");
            let resp = ui.add(egui::TextEdit::singleline(&mut self.tl.input).desired_width(80.0));
            let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            // Depends on nothing but this tab's own job, so it works every time.
            if ui.add_enabled(self.store.is_some(), egui::Button::new("Forecast")).clicked() || enter {
                let t = self.tl.input.trim().to_uppercase();
                if !t.is_empty() {
                    self.fc.ticker = t.clone();
                    self.fc.input_error.clear();
                    // Asking again for the stock shown rereads the run for it (a run still being
                    // computed may have stored more since).
                    if self.fc.data.get(&(self.run_name.clone(), t.clone())).is_some() {
                        self.fc.data.clear();
                        self.fc.job.clear();
                    }
                    // The Stock timeline tab follows the same stock (it loads when shown).
                    if self.tl.want.as_ref() != Some(&t) {
                        self.tl.want = Some(t);
                        self.tl.selected = None;
                    }
                }
            }
            ui.separator();
            ui.label("As of:");
            let resp = ui.add(egui::TextEdit::singleline(&mut self.fc.as_of_text).desired_width(90.0));
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                match c2_data::parse_date(self.fc.as_of_text.trim()) {
                    Ok(d) => {
                        self.fc.as_of = Some(d);
                        self.fc.input_error.clear();
                    }
                    Err(_) => self.fc.input_error = "dates look like 2024-06-28".into(),
                }
            }
            for (label, step) in [("◀ year", -365), ("◀ month", -30), ("month ▶", 30), ("year ▶", 365)] {
                if ui.add_enabled(current.is_some(), egui::Button::new(label)).clicked() {
                    self.fc.as_of = current.map(|d| d + step);
                }
            }
            if ui.button("latest").clicked() {
                self.fc.as_of = None;
            }
            if self.fc.job.is_running() || self.fc.data.is_running() || self.plan.is_running() {
                ui.spinner();
            }
        });
        ui.horizontal_wrapped(|ui| {
            let hl = |h: usize| format!("{h} ({})", match h { 5 => "a week", 10 => "two weeks", 21 => "a month", 63 => "a quarter", 126 => "half a year", _ => "a year" });
            ui.label("Days ahead:");
            egui::ComboBox::from_id_salt("fc_h").selected_text(hl(self.fc.horizon)).show_ui(ui, |ui| {
                for h in [5, 10, 21, 63, 126, 252] {
                    ui.selectable_value(&mut self.fc.horizon, h, hl(h));
                }
            });
            ui.checkbox(&mut self.fc.significant_only, "significant relationships only");
            let al = |a: usize| if a == 0 { "any time".to_string() } else { format!("in the last {a} year{}", if a == 1 { "" } else { "s" }) };
            ui.label("Use correlation periods ending:");
            egui::ComboBox::from_id_salt("fc_age").selected_text(al(self.fc.max_age_years)).show_ui(ui, |ui| {
                for a in [0, 1, 3, 5] {
                    ui.selectable_value(&mut self.fc.max_age_years, a, al(a));
                }
            });
            ui.separator();
            ui.radio_value(&mut self.fc.show_amount, false, "share price");
            ui.radio_value(&mut self.fc.show_amount, true, "value of $");
            ui.add_enabled(self.fc.show_amount, egui::TextEdit::singleline(&mut self.fc.amount).desired_width(70.0));
        });
        let Some(store) = self.store.clone() else {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.store_note);
            return;
        };
        if self.fc.ticker.is_empty() {
            ui.label("Enter a ticker.");
            return;
        }
        if !self.fc.input_error.is_empty() {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.fc.input_error);
        }
        // Three steps, each computed once for its inputs: the run's plan (its census variable,
        // shared with the other tabs), what forecasts for this stock need from the run (read once
        // per run and stock), and the forecast for the current choices, made from that without
        // reading anything again. While a step runs, the last forecast of this run stays shown;
        // it names its own stock and day.
        let run = self.run_name.clone();
        let data_key = (run.clone(), self.fc.ticker.clone());
        let key = self.forecast_key();
        let mut waiting = None;
        if self.fc.data.wants(&data_key) {
            match self.need_plan() {
                Ok(Some(plan)) => {
                    let (dir, ticker) = (self.dir(), self.fc.ticker.clone());
                    self.fc.data.start(data_key.clone(), move || {
                        c2_census::forecast::prepare(&store, &plan.plan.var, &dir, &ticker).map(Arc::new).map_err(|e| format!("{e:#}"))
                    });
                }
                Ok(None) => waiting = Some(format!("Building the census variable for every stock, for census '{run}' (a few seconds)…")),
                Err(e) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, format!("The Forecast can't read census '{run}': {e}"));
                    if ui.button("Try again").clicked() {
                        self.plan.clear();
                    }
                    return;
                }
            }
        }
        match self.fc.data.get(&data_key) {
            None => {
                waiting.get_or_insert_with(|| format!("Reading census '{run}' for {} (under a second)…", self.fc.ticker));
            }
            Some(Err(e)) => {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
                return;
            }
            Some(Ok(data)) => {
                if self.fc.job.wants(&key) {
                    let data = data.clone();
                    let (_, _, as_of, horizon, significant_only, age) = key.clone();
                    self.fc.job.start(key.clone(), move || {
                        let opt = c2_census::forecast::Options { significant_only, horizon, max_age: (age > 0).then_some(age * 252) };
                        c2_census::forecast::forecast_from(&data, as_of.unwrap_or(i32::MAX), &opt).map_err(|e| format!("{e:#}"))
                    });
                }
            }
        }
        let f = match &self.fc.job.done {
            Some((k, Err(e))) if *k == key => {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
                return;
            }
            Some((k, Ok(f))) if k.0 == run => f,
            _ => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(waiting.unwrap_or_else(|| "Computing the forecast…".to_string()));
                });
                return;
            }
        };
        if let Some(w) = waiting {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(w);
            });
        }
        if let Some(note) = self.plan_note_for(&run) {
            ui.colored_label(YELLOW_NOTE, note);
        }
        let t = &f.ticker;
        let h = f.predicted_pct.len();
        let amount: f64 = self.fc.amount.trim().replace([',', '$'], "").parse().unwrap_or(10_000.0);
        let base = if self.fc.show_amount { amount } else { f.last_price };
        let val = |pct: f64| base * (1.0 + pct / 100.0);
        let day = c2_data::date_string(f.as_of);
        let end_pct = f.predicted_pct[h - 1];
        let typical = 100.0 * f.daily_sd * (h as f64).sqrt();
        let what = if self.fc.show_amount { format!("${amount:.0} in {t}") } else { t.to_string() };
        // Every move here is relative to the market: the census correlates moves beyond what the
        // market did, so that is what the forecast predicts and what it is checked against.
        ui.label(format!(
            "{what}: ${base:.2} on {day}. Forecast for the next {h} trading days: {t} moves {end_pct:+.2}% beyond what the market does, so ${:.2} if the \
             market is flat. It is the average of what {} relationships imply (another stock moved first and {t} tends to follow some days later; each \
             leader's move beyond the market so far is scaled by the relationship). For scale, {t}'s typical {h}-day move beyond the market is ±{typical:.1}%.",
            val(end_pct),
            f.relationships
        ));
        if let Some(Some(a)) = f.actual_pct.last() {
            ui.label(format!("What happened: {t} moved {a:+.2}% beyond what the market did over those {h} trading days (${:.2} had the market been flat).", val(*a)));
        } else if f.actual_pct.iter().any(|a| a.is_some()) {
            ui.label(format!("What happened so far ({t}'s move beyond the market) is drawn in blue; the data ends before the last day ahead."));
        }
        match &f.skill {
            Some(s) => {
                let (colour, verdict) = track_record_verdict(s.hit_rate, s.directional, s.corr, s.forecasts);
                let text = format!(
                    "Track record for {t}, with every move measured beyond the market: the same forecast made every 21 trading days in the past, each using \
                     only relationships known at the time: {} forecasts {}-day ahead; direction right {:.0}% of the {} times both moved (a coin flip gets 50%); \
                     predicted vs actual correlation {:+.2}; the forecasts averaged ±{:.2}% while {t} actually moved ±{:.2}% beyond the market on average. {verdict}",
                    s.forecasts, s.horizon, 100.0 * s.hit_rate, s.directional, s.corr, s.mean_abs_predicted_pct, s.mean_abs_actual_pct,
                );
                ui.colored_label(colour, text);
            }
            None => {
                ui.colored_label(YELLOW_NOTE, "Track record: too few past forecasts to judge this one.");
            }
        }
        let orange = egui::Color32::from_rgb(255, 160, 40);
        let blue = egui::Color32::from_rgb(90, 170, 255);
        let grey = egui::Color32::from_rgb(140, 150, 170);
        let pred: Vec<[f64; 2]> = std::iter::once([0.0, base]).chain(f.predicted_pct.iter().enumerate().map(|(j, v)| [(j + 1) as f64, val(*v)])).collect();
        let actual: Vec<[f64; 2]> = std::iter::once([0.0, base])
            .chain(f.actual_pct.iter().enumerate().filter_map(|(j, a)| a.map(|a| [(j + 1) as f64, val(a)])))
            .collect();
        let band = |sign: f64| -> Vec<[f64; 2]> { (0..=h).map(|j| [j as f64, base * (sign * f.daily_sd * (j as f64).sqrt()).exp()]).collect() };
        let range = "typical range (±1 sd of past moves beyond the market)";
        Plot::new("forecast_plot")
            .height(300.0)
            .x_axis_label(format!("trading days after {day}"))
            .y_axis_label(if self.fc.show_amount { "value if the market is flat ($)" } else { "price if the market is flat ($)" })
            .legend(Legend::default())
            .show(ui, |p| {
                p.hline(HLine::new("", base).color(egui::Color32::DARK_GRAY));
                p.line(Line::new(range, PlotPoints::from(band(1.0))).color(grey).style(LineStyle::dashed_loose()));
                p.line(Line::new(range, PlotPoints::from(band(-1.0))).color(grey).style(LineStyle::dashed_loose()));
                if actual.len() > 1 {
                    p.line(Line::new("actual, had the market been flat", PlotPoints::from(actual)).color(blue).width(2.0));
                }
                p.line(Line::new("forecast", PlotPoints::from(pred)).color(orange).width(2.5));
            });
        ui.label("When the predicted moves land: the forecast's change beyond the market on each day ahead (in %), from the relationships whose delay reaches that day.");
        let bars: Vec<Bar> = (0..h)
            .map(|j| {
                let prev = if j == 0 { 0.0 } else { f.predicted_pct[j - 1] };
                let d = (1.0 + f.predicted_pct[j] / 100.0) / (1.0 + prev / 100.0) * 100.0 - 100.0;
                let date = f.future_days[j].map(c2_data::date_string).unwrap_or_default();
                Bar::new((j + 1) as f64, d)
                    .width(0.8)
                    .fill(if d >= 0.0 { egui::Color32::from_rgb(110, 200, 120) } else { egui::Color32::from_rgb(230, 100, 90) })
                    .name(format!("day {} {date}: {d:+.3}% from {} relationships", j + 1, f.covering[j]))
            })
            .collect();
        Plot::new("forecast_days").height(110.0).x_axis_label("trading days ahead").y_axis_label("% vs market").show(ui, |p| {
            p.bar_chart(BarChart::new("daily forecast", bars));
        });
        ui.strong(format!("Largest shares of the forecast at day {h} ({} relationships say something about these days)", f.contributions.len()));
        // Each row multiplies out exactly: alone implies = (1 + move/100)^(% per 1%) - 1, with the
        // move that lands in the days shown (`leader_move_in_horizon_pct`), not all of the
        // leader's moves feeding the forecast (`leader_move_pct`, only on hover).
        ui.label(format!(
            "Leader's move: its moves beyond the market that land in the {h} days shown (a move whose predicted days run past the last day counts \
             in proportion). {t} per 1%: how far {t} tends to move beyond the market per 1% the leader moves beyond it (the Stock timeline's \
             '% per 1%'). Alone implies = (1 + move/100)^(% per 1%) - 1: the % per 1% applies to log changes, so it is about move x % per 1% \
             only for small moves."
        ));
        let mut open: Option<(String, String, usize, usize, usize, usize)> = None;
        egui::ScrollArea::vertical().id_salt("fc_rows").show(ui, |ui| {
            egui::Grid::new("fc_grid").striped(true).show(ui, |ui| {
                let per = format!("{t} per 1%");
                for c in ["leader", "offset", "move lengths", "r", "", "leader's move\nlanding in these days", &per, "alone implies", "share of forecast", "correlation period", ""] {
                    ui.strong(c);
                }
                ui.end_row();
                for c in f.contributions.iter().take(200) {
                    ui.label(&c.leader);
                    ui.label(format!("{} day{}", c.lag, if c.lag == 1 { "" } else { "s" }));
                    ui.label(format!("{} -> {}", days_label(c.h1), days_label(c.h2)));
                    ui.label(format!("{:+.3}", c.r));
                    ui.label(if c.significant { "★" } else { "" });
                    ui.label(format!("{:+.2}%", c.leader_move_in_horizon_pct))
                        .on_hover_text(format!("All its moves feeding the forecast, each counted whole: {:+.2}%", c.leader_move_pct));
                    ui.label(format!("{:+.3}%", c.move_per_pct));
                    ui.label(format!("{:+.2}%", c.implied_pct));
                    ui.label(format!("{:+.3} pts", c.share_pct));
                    ui.label(format!("{} days to {}", c.period_days, c.window.split("..").nth(1).unwrap_or("")));
                    if ui.small_button("explain").clicked() {
                        open = Some((c.leader.clone(), t.clone(), c.h1, c.h2, c.lag, c.window_id));
                    }
                    ui.end_row();
                }
            });
        });
        if let Some((a, b, h1, h2, lag, w)) = open {
            self.open_explainer(a, b, h1, h2, lag, w);
        }
    }

    /// The current run's plan (`plan_for_run`): its description, the census variable for every
    /// stock and the run's own correlation periods, universes and tasks, built in the background
    /// (in the slot it shares with the holdout). Ok(None) while it is being built.
    /// `plan_note` for `run`'s plan if one has been built (this builds none).
    fn plan_note_for(&self, run: &str) -> Option<String> {
        match self.plan.get(&run.to_string()) {
            Some(Ok(rp)) => plan_note(rp),
            _ => None,
        }
    }

    fn need_plan(&mut self) -> Result<Option<Arc<RunPlan>>, String> {
        let Some(store) = self.store.clone() else { return Err(self.store_note.clone()) };
        let run = self.run_name.clone();
        if self.plan.wants(&run) {
            let (slot, dir, r, generation) = (self.plan_slot.clone(), self.dir(), run.clone(), self.store_gen);
            self.plan.start(run.clone(), move || plan_for_run(&slot, &store, generation, &r, &dir));
        }
        match self.plan.get(&run) {
            None => Ok(None),
            Some(Ok(p)) => Ok(Some(p.clone())),
            Some(Err(e)) => Err(e.clone()),
        }
    }

    fn explain_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("How one correlation is calculated");
        let rp = match self.need_plan() {
            Ok(Some(rp)) => rp,
            Ok(None) => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!("Building the census variable for every stock, for census '{}' (a few seconds)…", self.run_name));
                });
                return;
            }
            Err(e) => {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
                if self.store.is_some() && ui.button("Try again").clicked() {
                    self.plan.clear();
                }
                return;
            }
        };
        let plan = &rp.plan;
        // The run's periods of a length that today's data covers, in date order: the ones that can
        // be recomputed and so can be chosen (the others are counted below the choices).
        let covered = |len: usize| -> Vec<usize> { plan.windows.iter().filter(|w| w.length == len && plan.period_available(w.id)).map(|w| w.id).collect() };
        // A period id from a row of this run (the plan's periods are the run's own).
        if let Some((run, w)) = self.ex.pending_window.take()
            && run == self.run_name
            && let Some(win) = plan.windows.get(w)
            && let Some(pos) = covered(win.length).iter().position(|&id| id == w)
        {
            self.ex.length = win.length;
            self.ex.pos = pos;
        }
        // A period by length and last day (a run switch, a store reload, a self-check): the
        // covered one of that length ending on or before that day, else the earliest covered
        // (`explain::find_window` picks only covered periods).
        if let Some((len, end)) = self.ex.pending_end.take()
            && let Some(w) = c2_census::explain::find_window(plan, len, end)
            && let Some(pos) = covered(len).iter().position(|&id| id == w)
        {
            self.ex.length = len;
            self.ex.pos = pos;
        }
        let mut lengths: Vec<usize> = plan.windows.iter().map(|w| w.length).collect();
        lengths.sort_by_key(|&l| if l == 0 { usize::MAX } else { l });
        lengths.dedup();
        // A run may lack the length or move lengths chosen before (smoke runs have no full
        // history): fall back to ones it has.
        if !lengths.contains(&self.ex.length)
            && let Some(&l) = lengths.last()
        {
            self.ex.length = l;
            self.ex.pos = usize::MAX;
        }
        // Only move lengths the run computed for periods of this length (older runs paired fewer:
        // B's move no longer than A's), and below, only the offsets the census uses for B's.
        let (h1s, h2s, h1, h2) = move_length_choices(&plan.tasks, self.ex.length, self.ex.h1, self.ex.h2);
        (self.ex.h1, self.ex.h2) = (h1, h2);
        let full_days = plan.windows.iter().find(|w| w.length == 0 && plan.period_available(w.id)).map(|w| w.end - w.start);
        let len_label = move |l: usize| match (l, full_days) {
            (0, Some(d)) => format!("{d} days (all history)"),
            (0, None) => "all history".to_string(),
            _ => format!("{l} days"),
        };
        // In the list of lengths, those of which today's data covers no period.
        let none_covered: Vec<usize> = lengths.iter().copied().filter(|&l| covered(l).is_empty()).collect();
        let len_choice = |l: usize| if none_covered.contains(&l) { format!("{} (none covered by today's price data)", len_label(l)) } else { len_label(l) };
        ui.label(
            "Pick two stocks, the length of move to measure for each, the offset between them and a correlation period. Every step below is \
             recomputed from prices and checked against what the census stored: the stored record's sums when it kept this pair, else its \
             summary of the matrix.",
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Stock A (moves first):");
            let r1 = ui.add(egui::TextEdit::singleline(&mut self.ex.leader_input).desired_width(70.0));
            ui.label("Stock B (follows):");
            let r2 = ui.add(egui::TextEdit::singleline(&mut self.ex.follower_input).desired_width(70.0));
            if r1.lost_focus() || r2.lost_focus() {
                self.ex.leader = self.ex.leader_input.trim().to_uppercase();
                self.ex.follower = self.ex.follower_input.trim().to_uppercase();
            }
            if ui.button("swap A and B").clicked() {
                std::mem::swap(&mut self.ex.leader_input, &mut self.ex.follower_input);
                std::mem::swap(&mut self.ex.leader, &mut self.ex.follower);
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("A's move over");
            egui::ComboBox::from_id_salt("ex_h1").selected_text(days_label(self.ex.h1)).show_ui(ui, |ui| {
                for &h in &h1s {
                    ui.selectable_value(&mut self.ex.h1, h, days_label(h));
                }
            });
            ui.label("B's move over");
            egui::ComboBox::from_id_salt("ex_h2").selected_text(days_label(self.ex.h2)).show_ui(ui, |ui| {
                for &h in &h2s {
                    ui.selectable_value(&mut self.ex.h2, h, days_label(h));
                }
            });
            // The offsets the census computes for B's move length: 0, the same-day reference, then
            // the predictive ones.
            let lags: Vec<usize> = grid::lags(self.ex.h2);
            self.ex.lag = nearest(&lags, self.ex.lag);
            let lag_label = |k: usize| if k == 0 { "0 days (same day, for reference)".to_string() } else { days_label(k) };
            ui.label("offset");
            egui::ComboBox::from_id_salt("ex_lag").selected_text(lag_label(self.ex.lag)).show_ui(ui, |ui| {
                for &k in &lags {
                    ui.selectable_value(&mut self.ex.lag, k, lag_label(k));
                }
            });
            ui.label("from the end of A's move to the end of B's");
        });
        let before = self.ex.length;
        ui.horizontal_wrapped(|ui| {
            ui.label("Correlation period:");
            egui::ComboBox::from_id_salt("ex_len").selected_text(len_choice(self.ex.length)).show_ui(ui, |ui| {
                for &l in &lengths {
                    ui.selectable_value(&mut self.ex.length, l, len_choice(l));
                }
            });
            if self.ex.length != before {
                self.ex.pos = usize::MAX;
            }
            // Only the periods today's data covers can be chosen.
            let ends: Vec<String> = covered(self.ex.length).iter().map(|&id| c2_data::date_string(rp.meta.windows[id].last)).collect();
            self.ex.pos = self.ex.pos.min(ends.len().saturating_sub(1));
            if ends.len() > 1 {
                ui.label("ending");
                let maxp = ends.len() - 1;
                ui.add(egui::Slider::new(&mut self.ex.pos, 0..=maxp).custom_formatter(move |v, _| ends.get(v.round() as usize).cloned().unwrap_or_default()));
            }
            if self.ex.job.is_running() {
                ui.spinner();
            }
        });
        let ids = covered(self.ex.length);
        let of_length: Vec<usize> = plan.windows.iter().filter(|w| w.length == self.ex.length).map(|w| w.id).collect();
        if let Some(note) = uncovered_note(&rp, self.ex.length, &of_length, &ids) {
            ui.colored_label(YELLOW_NOTE, note);
        }
        let window = ids.get(self.ex.pos).copied();
        self.ex.selected = window.map(|w| (plan.windows[w].length, rp.meta.windows[w].last));
        let run = self.run_name.clone();
        if h1s.is_empty() {
            ui.label(format!("Census '{run}' computed no lead-lag correlations for {} correlation periods.", len_label(self.ex.length)));
        }
        // Only move lengths the run has a task for with periods of this length (a choice changed
        // above is checked against the others on the next frame).
        let computed = plan.tasks.iter().any(|t| (t.h1, t.h2, t.length) == (self.ex.h1, self.ex.h2, self.ex.length));
        let key: Option<ExplainKey> = window.filter(|_| computed).map(|w| {
            let threshold = self.ex.threshold_fixed.then(|| (self.ex.threshold_pct * 100.0).round() as i64);
            (run.clone(), self.ex.leader.clone(), self.ex.follower.clone(), self.ex.h1, self.ex.h2, self.ex.lag, w, threshold)
        });
        self.ex.key = key.clone();
        if let Some(k) = &key
            && self.ex.job.wants(k)
            && let Some(store) = self.store.clone()
        {
            let (rp2, k2) = (rp.clone(), k.clone());
            self.ex.job.start(k.clone(), move || {
                let req = c2_census::explain::Request {
                    leader: k2.1,
                    follower: k2.2,
                    h1: k2.3,
                    h2: k2.4,
                    lag: k2.5,
                    window: k2.6,
                    threshold: k2.7.map(|t| t as f64 / 10_000.0),
                };
                c2_census::explain::explain_with(&store, &rp2.plan, &rp2.meta, &req).map_err(|e| format!("{e:#}"))
            });
        }
        if let (Some(k), Some((done_key, Err(err)))) = (&key, &self.ex.job.done)
            && k == done_key
        {
            ui.colored_label(egui::Color32::LIGHT_RED, err);
        }
        // The explanation for the current choices, else (while those are computed) the last one
        // of this run; it names its own stocks, moves and period.
        let done = self.ex.job.done.take();
        if let Some((k, Ok(e))) = &done
            && k.0 == self.run_name
        {
            self.explain_steps(ui, e, &rp);
        }
        self.ex.job.done = done;
    }

    /// The steps of one explanation of the run `rp` describes.
    fn explain_steps(&mut self, ui: &mut egui::Ui, e: &Explained, rp: &RunPlan) {
        // How many of each matrix's strongest pairs the run stores.
        let keep_real = rp.meta.cfg.keep_real;
        use c2_census::explain::DaySteps;
        let (a, b) = (e.a.ticker.clone(), e.b.ticker.clone());
        let (h1, h2, lag) = (e.h1, e.h2, e.lag);
        let step = h1.max(h2);
        let n_obs = e.obs.len();
        if n_obs == 0 {
            return;
        }
        let date = |row: usize| e.days.get(row).map(|d| c2_data::date_string(*d)).unwrap_or_default();
        let pct = |l: f64| if l.is_finite() { format!("{:+.2}%", 100.0 * l.exp_m1()) } else { "no price".to_string() };
        let date_axis = |days: Vec<i32>| move |m: egui_plot::GridMark, _: &std::ops::RangeInclusive<f64>| days.get(m.value.round().max(0.0) as usize).map(|d| c2_data::date_string(*d)).unwrap_or_default();
        let mut area = egui::ScrollArea::vertical().id_salt("ex_scroll");
        if let Some(y) = self.ex.scroll.take() {
            area = area.vertical_scroll_offset(y);
        }
        area.show(ui, |ui| {
            // Whether the census computed this at all, and what offset 0 is, before anything else.
            if let Err(why) = &e.census_computed {
                ui.add_space(6.0);
                ui.colored_label(YELLOW_NOTE, format!("The census never computed this: {why}. The steps below show what it would have found."));
            }
            if e.same_day {
                ui.add_space(6.0);
                ui.colored_label(
                    YELLOW_NOTE,
                    format!(
                        "Offset 0 is the same-day reference: {b}'s move ends the same day as {a}'s. The census computes it to compare its later \
                         offsets with; it is not a prediction."
                    ),
                );
            }
            // 1. Period and observations.
            ui.add_space(6.0);
            ui.heading("1. The correlation period and the moves being compared");
            let inu = |s: &DaySteps| if s.in_universe { "included".to_string() } else { "NOT included, so the census never tested it in this period".to_string() };
            ui.label(format!(
                "Correlation period: {} trading days, {} to {}. The census used {} stocks in it (each needs prices on at least 90% of its days and, over \
                 its last year, a median price of $5 or more and median daily trading of $5M or more). {a}: {}. {b}: {}.",
                e.period_days,
                c2_data::date_string(e.period_first),
                c2_data::date_string(e.period_last),
                e.universe,
                inu(&e.a),
                inu(&e.b)
            ));
            // What today's data can't reproduce of this period and of this run.
            let missing = rp.plan.missing_stocks.get(e.window).map_or(0, Vec::len);
            if missing > 0 {
                ui.colored_label(
                    YELLOW_NOTE,
                    format!(
                        "{missing} of the census's {} stocks in this period {} no longer in today's price data, so the census's results for {} can't be \
                         recomputed here.",
                        e.universe,
                        if missing == 1 { "is" } else { "are" },
                        if missing == 1 { "it" } else { "them" }
                    ),
                );
            }
            if let Some(note) = &e.variable_note {
                ui.colored_label(YELLOW_NOTE, format!("Note: {}", sentence(note)));
            }
            ui.label(format!(
                "Each observation pairs {a}'s move over {} with {b}'s move over {} that ends {}. Observations step forward {} at a time so no \
                 two share a day: {} observations in this period.",
                days_label(h1),
                days_label(h2),
                if lag == 0 { format!("the same day as {a}'s") } else { format!("{} after {a}'s ends", tdays(lag)) },
                days_label(step),
                n_obs
            ));
            let obs_resp = ui.add(
                egui::Slider::new(&mut self.ex.obs, 0..=n_obs - 1)
                    .text("observation to follow through the steps")
                    .custom_formatter(|v, _| format!("{}", v.round() as usize + 1)),
            );
            if obs_resp.changed() {
                self.ex.day = None;
            }
            let q = self.ex.obs.min(n_obs - 1);
            let o = &e.obs[q];
            ui.label(format!(
                "Observation {}: {a} from the close on {} to the close on {}: {}.   {b} from the close on {} to the close on {}: {}.",
                q + 1,
                date(o.a_end - h1),
                date(o.a_end),
                pct(o.a_move),
                date(o.b_end - h2),
                date(o.b_end),
                pct(o.b_move)
            ));
            let norm = |c: &[f32]| -> Vec<[f64; 2]> {
                let base = c.iter().copied().find(|v| v.is_finite() && *v > 0.0).unwrap_or(1.0) as f64;
                c.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, v)| [i as f64, 100.0 * *v as f64 / base]).collect()
            };
            let (pa, pb) = (norm(&e.a.close), norm(&e.b.close));
            let lo = pa.iter().chain(&pb).map(|p| p[1]).fold(f64::INFINITY, f64::min);
            let hi = pa.iter().chain(&pb).map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
            let band = |x0: usize, x1: usize| PlotPoints::from(vec![[x0 as f64, lo], [x1 as f64, lo], [x1 as f64, hi], [x0 as f64, hi]]);
            Plot::new("ex_prices")
                .height(220.0)
                .legend(Legend::default())
                .y_axis_label("price (period start = 100)")
                .x_axis_formatter(date_axis(e.days.clone()))
                .show(ui, |p| {
                    p.polygon(Polygon::new(format!("{a}'s move"), band(o.a_end - h1, o.a_end)).fill_color(ORANGE.gamma_multiply(0.3)).stroke(egui::Stroke::new(1.0, ORANGE)));
                    p.polygon(Polygon::new(format!("{b}'s move"), band(o.b_end - h2, o.b_end)).fill_color(BLUE.gamma_multiply(0.3)).stroke(egui::Stroke::new(1.0, BLUE)));
                    p.line(Line::new(a.clone(), PlotPoints::from(pa)).color(ORANGE));
                    p.line(Line::new(b.clone(), PlotPoints::from(pb)).color(BLUE));
                    p.arrows(
                        Arrows::new(format!("offset: {lag} trading days"), PlotPoints::from(vec![[o.a_end as f64, hi]]), PlotPoints::from(vec![[o.b_end as f64, hi]]))
                            .color(egui::Color32::YELLOW),
                    );
                });

            // 2. Daily census values.
            ui.add_space(8.0);
            ui.heading("2. From prices to the census value, one day at a time");
            ui.label(
                "The census doesn't correlate raw price moves. For each stock and day it removes the part of the move explained by the whole market, then \
                 rescales by that stock's own recent volatility, using only data from before that day:",
            );
            let nd = e.days.len();
            let mut day = self.ex.day.unwrap_or(o.a_end).clamp(1, nd - 1);
            let days2 = e.days.clone();
            let resp = ui.add(
                egui::Slider::new(&mut day, 1..=nd - 1)
                    .text("day")
                    .custom_formatter(move |v, _| days2.get(v.round() as usize).map(|d| c2_data::date_string(*d)).unwrap_or_default()),
            );
            if resp.changed() {
                self.ex.day = Some(day);
            }
            const STEPS: [&str; 7] = [
                "closing price: day before -> that day",
                "1. daily return r = ln(close / previous close)",
                "2. market return m that day (average of the liquid stocks)",
                "3. beta from the previous 252 days = cov(r, m) / var(m)",
                "4. residual e = r - beta x m (the stock's own move)",
                "5. mean / sd of e over the previous 252 days",
                "6. census value v = (e - mean) / sd, kept within +-5",
            ];
            let (ca, cb) = (day_cells(&e.a, e.market[day], day), day_cells(&e.b, e.market[day], day));
            egui::Grid::new("ex_day").striped(true).num_columns(3).show(ui, |ui| {
                ui.strong(format!("on {}", date(day)));
                ui.strong(&a);
                ui.strong(&b);
                ui.end_row();
                for k in 0..STEPS.len() {
                    ui.label(STEPS[k]);
                    ui.label(&ca[k]);
                    ui.label(&cb[k]);
                    ui.end_row();
                }
            });
            ui.label(
                "Recomputed here for every day of the period with the census's own code (the census doesn't store daily values; step 4 checks the \
                 result against what it stored).",
            );
            let series = |s: &DaySteps| -> Vec<[f64; 2]> { s.v.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, v)| [i as f64, *v as f64]).collect() };
            Plot::new("ex_vals")
                .height(140.0)
                .legend(Legend::default())
                .y_axis_label("census value")
                .x_axis_formatter(date_axis(e.days.clone()))
                .show(ui, |p| {
                    p.line(Line::new(a.clone(), PlotPoints::from(series(&e.a))).color(ORANGE));
                    p.line(Line::new(b.clone(), PlotPoints::from(series(&e.b))).color(BLUE));
                    p.vline(VLine::new("selected day", day as f64).color(egui::Color32::YELLOW));
                });

            // 3. Observations in census units.
            ui.add_space(8.0);
            ui.heading("3. Each observation in census units");
            ui.label(format!(
                "x = the sum of {a}'s daily census values over its {} / sqrt({h1}); y = the same for {b} over its {} / sqrt({h2}). Dividing by the square \
                 root of the length puts every move length on the same scale. Days without a value count as 0.",
                days_label(h1),
                days_label(h2)
            ));
            let heads = ["#", &format!("{a} from -> to"), &format!("{a} move"), "x", &format!("{b} from -> to"), &format!("{b} move"), "y"].map(|s| s.to_string());
            egui::Grid::new("ex_obs_head").num_columns(7).min_col_width(95.0).show(ui, |ui| {
                for h in &heads {
                    ui.strong(h);
                }
                ui.end_row();
            });
            let row_h = ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y;
            egui::ScrollArea::vertical().id_salt("ex_obs").max_height(170.0).show_rows(ui, row_h, n_obs, |ui, range| {
                egui::Grid::new("ex_obs_grid").num_columns(7).min_col_width(95.0).striped(true).show(ui, |ui| {
                    for i in range {
                        let o = &e.obs[i];
                        if ui.selectable_label(i == q, format!("{}", i + 1)).clicked() {
                            self.ex.obs = i;
                            self.ex.day = None;
                        }
                        ui.label(format!("{} -> {}", date(o.a_end - h1), date(o.a_end)));
                        ui.label(pct(o.a_move));
                        ui.label(format!("{:+.3}", o.x));
                        ui.label(format!("{} -> {}", date(o.b_end - h2), date(o.b_end)));
                        ui.label(pct(o.b_move));
                        ui.label(format!("{:+.3}", o.y));
                        ui.end_row();
                    }
                });
            });

            // 4. The correlation.
            ui.add_space(8.0);
            ui.heading("4. The correlation");
            ui.checkbox(&mut self.ex.raw_scatter, "plot raw % moves instead of census units");
            let raw = self.ex.raw_scatter;
            let point = |o: &c2_census::explain::Obs| if raw { [100.0 * o.a_move.exp_m1(), 100.0 * o.b_move.exp_m1()] } else { [o.x, o.y] };
            let pts: Vec<[f64; 2]> = e.obs.iter().map(point).filter(|p| p[0].is_finite() && p[1].is_finite()).collect();
            let n = pts.len().max(1) as f64;
            let (mx, my) = (pts.iter().map(|p| p[0]).sum::<f64>() / n, pts.iter().map(|p| p[1]).sum::<f64>() / n);
            let (sxy, sxx) = pts.iter().fold((0.0, 0.0), |(sxy, sxx), p| (sxy + (p[0] - mx) * (p[1] - my), sxx + (p[0] - mx) * (p[0] - mx)));
            let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
            let (x0, x1) = (pts.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min), pts.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max));
            let sel = point(o);
            let unit = if raw { " (%)" } else { " (census units)" };
            Plot::new("ex_scatter")
                .height(260.0)
                .legend(Legend::default())
                .x_axis_label(format!("{a}'s move{unit}"))
                .y_axis_label(if lag == 0 { format!("{b}'s move the same day{unit}") } else { format!("{b}'s move {lag} days later{unit}") })
                .show(ui, |p| {
                    p.hline(HLine::new("", 0.0).color(egui::Color32::DARK_GRAY));
                    p.vline(VLine::new("", 0.0).color(egui::Color32::DARK_GRAY));
                    p.points(Points::new("observations", PlotPoints::from(pts.clone())).color(GREY).radius(2.5).filled(true));
                    if x0.is_finite() {
                        p.line(Line::new("best-fit line", PlotPoints::from(vec![[x0, my + slope * (x0 - mx)], [x1, my + slope * (x1 - mx)]])).color(ORANGE).width(2.0));
                    }
                    if sel[0].is_finite() && sel[1].is_finite() {
                        p.points(Points::new("selected observation", PlotPoints::from(vec![sel])).color(egui::Color32::YELLOW).radius(6.0).filled(true));
                    }
                });
            ui.label(format!(
                "r = {:+.4} from n = {} observations. r measures how consistently {b}'s {}move lines up with {a}'s move: 0 = no relation, +1 = always \
                 the same direction in proportion, -1 = always the opposite. z = atanh(r) x sqrt(n - 3) = {:+.2}: how many standard errors r is from 0. \
                 One pair on its own reaches |z| of 2 by chance about 5% of the time, and 4 about 6 times in 100,000.",
                e.r,
                e.n,
                if lag == 0 { "" } else { "later " },
                e.z
            ));
            // What the census did with this pair: stored it, never computed it (and why), or
            // computed it without keeping it. Then what the numbers above were checked against.
            let never = e.never_computed();
            match (&e.stored, &never) {
                (Some(s), _) => ui.label(format!(
                    "The census stored this pair for this period: r = {:+.6} (recomputed here: {:+.6}; difference {:.1e}). Its GPU estimate of r (the fast \
                     8-bit first pass, which decides what is stored and applies the significance test) was {:+.4}{}; {}.",
                    s.record.r,
                    e.r,
                    (s.record.r as f64 - e.r).abs(),
                    s.record.r_int8,
                    if s.top { "; among the strongest in its matrix" } else { "" },
                    if e.same_day {
                        "offset 0 gets no significance test"
                    } else if s.significant {
                        "it passed the significance test"
                    } else {
                        "it did not pass the significance test"
                    }
                )),
                (None, Some(why)) => ui.label(format!("The census never computed this pair here: {why}. The r above is what it would have been.")),
                (None, None) => ui.label(format!(
                    "The census computed this pair but did not store it for this period: it keeps the strongest {keep_real} of each matrix plus any that \
                     pass the significance test."
                )),
            };
            match &e.check {
                Check::Record(c) if c.agrees() => ui.label(format!(
                    "Checked against the census's record of this pair: the same {} observations, and the sums of x, y, x·x, y·y and x·y it stored agree \
                     with the ones recomputed here {} (the record keeps about 7 significant digits).",
                    c.obs,
                    if c.max_rel_diff > 0.0 { format!("to {:.0e}", c.max_rel_diff) } else { "exactly".to_string() }
                )),
                // A disagreement is expected when this run's census variable started on another
                // day (`variable_note`, below); otherwise it is a real discrepancy, in red.
                Check::Record(c) => ui.colored_label(
                    if e.variable_note.is_some() { YELLOW_NOTE } else { egui::Color32::LIGHT_RED },
                    format!(
                        "Checked against the census's record of this pair: they DIFFER ({} observations here, {} stored; sums differ by up to {:.1e} of \
                         their size, where under 1e-6 is expected). The census stored something other than what its code computes from today's prices.",
                        c.obs, c.stored_obs, c.max_rel_diff
                    ),
                ),
                Check::Summary(c) if c.agrees() => ui.label(format!(
                    "Nothing is stored for this pair to check against; the matrix's summary agrees with what is recomputed here: {} stocks and {} \
                     observations.",
                    c.stored_stocks, c.stored_obs
                )),
                Check::Summary(c) => ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    format!(
                        "Nothing is stored for this pair to check against, and the matrix's summary DIFFERS: {} stocks and {} observations there, {} and \
                         {} here.",
                        c.stored_stocks, c.stored_obs, c.stocks, c.obs
                    ),
                ),
                Check::Nothing => ui.label("The census never computed this matrix, so nothing is stored to check against."),
            };
            if e.variable_note.is_some()
                && let Check::Record(c) = &e.check
            {
                ui.colored_label(
                    YELLOW_NOTE,
                    if c.agrees() {
                        "(They agree here, but this run's census variable started on another day, so values recomputed today can differ from the stored \
                         ones: see step 1.)"
                    } else {
                        "(Expected here: this run's census variable started on another day, so values recomputed today can differ from the stored ones: \
                         see step 1.)"
                    },
                );
            }
            ui.label(format!("For comparison, the correlation of the raw % moves (no market adjustment or volatility scaling) is {:+.3}.", e.r_raw));

            // 5. Chance.
            ui.add_space(8.0);
            ui.heading("5. Could it be chance?");
            let m = &e.matrix;
            let az = e.z.abs();
            let formula = if e.chance_analytic >= 1.0 { commas(e.chance_analytic) } else { format!("{:.3}", e.chance_analytic) };
            if let Err(why) = &e.census_computed {
                ui.label(format!(
                    "The census never computed this matrix ({why}), so there is no measured chance count. A bell-curve formula predicts about {formula} \
                     of the {} correlations at |z| of {az:.2} or more by chance alone; stock moves have fatter tails, so the real count would be higher.",
                    big(e.pairs_tested)
                ));
            } else {
                let k = if az >= 5.0 { 2 } else if az >= 4.0 { 1 } else { 0 };
                ui.label(format!(
                    "In this period, with these move lengths and this offset, the census correlated each of the {} stocks with every other: {} correlations. \
                     Pure chance, measured by running the same correlations on sign-shuffled data, produced {:.0} pairs at |z| of {} or more{}. A bell-curve \
                     formula would predict only {formula} at this pair's |z| of {az:.2}: stock moves have fatter tails than a bell curve, which is why the census \
                     measures chance instead of assuming it.",
                    e.universe,
                    big(e.pairs_tested),
                    m.null[k],
                    k + 3,
                    if az < 3.0 { format!(" (this pair's |z| is only {az:.2})") } else { String::new() }
                ));
                let mut real = Vec::new();
                let mut null = Vec::new();
                for k in 0..3 {
                    real.push(Bar::new(k as f64 - 0.2, m.real[k]).width(0.38).fill(ORANGE).name(format!("|z| ≥ {}: {:.0} real", k + 3, m.real[k])));
                    null.push(Bar::new(k as f64 + 0.2, m.null[k]).width(0.38).fill(GREY).name(format!("|z| ≥ {}: {:.0} by chance", k + 3, m.null[k])));
                }
                Plot::new("ex_chance")
                    .height(150.0)
                    .legend(Legend::default())
                    .y_axis_label("pairs")
                    .x_axis_formatter(category_axis(vec!["|z| ≥ 3".into(), "|z| ≥ 4".into(), "|z| ≥ 5".into()]))
                    .show(ui, |p| {
                        p.bar_chart(BarChart::new("real pairs", real).color(ORANGE));
                        p.bar_chart(BarChart::new("on sign-shuffled data (chance)", null).color(GREY));
                    });
                ui.label(format!(
                    "Measured in this matrix: real pairs at |z| of 3 / 4 / 5 or more: {:.0} / {:.0} / {:.0}. The same correlations on sign-shuffled data \
                     (each observation's B move flipped up or down at random, which keeps volatility but breaks any real link): {:.0} / {:.0} / {:.0}.",
                    m.real[0], m.real[1], m.real[2], m.null[0], m.null[1], m.null[2]
                ));
                match m.fdr_r {
                    // The census's own verdict when it stored the pair (the same flag as step 4).
                    Some(t) => ui.label(significance_verdict(t, e.r, e.stored.as_ref().map(|s| (s.record.r_int8, s.significant)), never.is_none())),
                    None if e.same_day => ui.label("The census doesn't run the significance test at offset 0 (the same-day reference)."),
                    None => ui.label("No pair in this matrix passed the significance test: nothing stood out from chance."),
                };
            }

            // 6. Plain terms.
            ui.add_space(8.0);
            ui.heading("6. In your words");
            ui.horizontal_wrapped(|ui| {
                ui.label("A big move is:");
                ui.radio_value(&mut self.ex.threshold_fixed, false, "A's typical move in the period (one standard deviation)");
                ui.radio_value(&mut self.ex.threshold_fixed, true, "a fixed size of");
                ui.add_enabled(self.ex.threshold_fixed, egui::Slider::new(&mut self.ex.threshold_pct, 0.5..=20.0).step_by(0.5).suffix("%"));
            });
            match &e.words {
                Some(w) => {
                    for s in words_text(w, &a, &b, h1, h2, lag) {
                        ui.label(s);
                    }
                    let groups: [(&str, fn(&Counts) -> f64); 3] = [
                        ("after A rose", |c: &Counts| rate(c.a_up_b_up, c.a_up)),
                        ("overall", |c: &Counts| rate(c.b_up, c.cases)),
                        ("after A fell", |c: &Counts| rate(c.a_down_b_up, c.a_down)),
                    ];
                    let mut per = Vec::new();
                    let mut rest = Vec::new();
                    for (k, (label, f)) in groups.iter().enumerate() {
                        per.push(Bar::new(k as f64 - 0.2, f(&w.period)).width(0.38).fill(ORANGE).name(format!("{label}, in the period")));
                        if w.rest.cases > 0 {
                            rest.push(Bar::new(k as f64 + 0.2, f(&w.rest)).width(0.38).fill(BLUE).name(format!("{label}, rest of history")));
                        }
                    }
                    Plot::new("ex_words")
                        .height(150.0)
                        .legend(Legend::default())
                        .y_axis_label(format!("% of times {b} rose"))
                        .x_axis_formatter(category_axis(groups.iter().map(|g| format!("{} {}", g.0.replace(" A ", &format!(" {a} ")), "")).collect()))
                        .show(ui, |p| {
                            p.hline(HLine::new("50%", 50.0).color(egui::Color32::DARK_GRAY));
                            p.bar_chart(BarChart::new("in the correlation period", per).color(ORANGE));
                            if !rest.is_empty() {
                                p.bar_chart(BarChart::new("rest of the history", rest).color(BLUE));
                            }
                        });
                    ui.label("Raw price moves include the whole market's moves, which is why B's normal rate of rising is usually a little above 50%.");
                }
                None => {
                    ui.label("Too few observations with prices to count.");
                }
            }

            // 7. Size, relative to the market: exactly the multiplier the Forecast tab applies.
            ui.add_space(8.0);
            ui.heading("7. Size: how far B moves beyond the market per 1% that A does (the timeline's '% per 1%')");
            match &e.move_size {
                Some(s) => {
                    ui.label(format!(
                        "% per 1% = r x (typical size of {b}'s {}-day moves beyond the market / typical size of {a}'s {}-day moves beyond the market) \
                         = {:+.4} x ({:.2}% / {:.2}%) = {:+.3}, over {} observations{}. So when {a} moves 1% beyond what the market did over {}, {b} \
                         tends to move {:+.3}% beyond the market over the {} ending {}.",
                        h2,
                        h1,
                        e.move_size_r,
                        100.0 * s.sd_b,
                        100.0 * s.sd_a,
                        s.per_pct,
                        s.pairs,
                        // The Forecast and the timeline use the r the census stored.
                        if e.stored.is_some() { ", with r as the census stored it" } else { "" },
                        days_label(h1),
                        s.per_pct,
                        days_label(h2),
                        if lag == 0 { "the same day".to_string() } else { format!("{} later", tdays(lag)) }
                    ));
                    match &e.forecast_use {
                        // As `forecast::rel_path` applies it: per move of A, on log changes, only the
                        // part of the predicted days inside the forecast's days ahead.
                        Ok(()) => ui.label(format!(
                            "How the Forecast tab uses it (on as-of days when this is the most recent correlation period of {a} -> {b} at these move \
                             lengths and offset; for each such relationship it uses the latest rolling period that ended by the as-of day): each of \
                             {a}'s recent {}-day moves beyond the market (the latest ending on the as-of day, then every {} before it) predicts {b}'s \
                             move beyond the market over the {} ending {} after it: a move of X% predicts (1 + X/100)^{:.3} - 1 for {b} (the % per 1% \
                             applies to log changes, so this is about X x {:.3} only for small moves), spread evenly over those days. Only the days that \
                             fall in the forecast's days ahead count, and each day's forecast is the average over the relationships covering it. For \
                             example a typical move of {a}, {:+.2}%, predicts {:+.2}% for {b}. The Stock timeline shows the same number as '% per 1%'.",
                            h1,
                            days_label(step),
                            days_label(h2),
                            tdays(lag),
                            s.per_pct,
                            s.per_pct,
                            100.0 * s.sd_a.exp_m1(),
                            100.0 * (s.per_pct * s.sd_a).exp_m1()
                        )),
                        // The reason names the Forecast itself ("the Forecast uses only ...").
                        Err(why) => ui.label(sentence(why)),
                    };
                }
                None => {
                    ui.label(format!("Fewer than {} observations have both moves, so there is no size.", c2_census::moves::MIN_PAIRS));
                }
            }
            ui.add_space(12.0);
        });
    }

    /// Starts reading the current run's statistics unless they are read or being read.
    fn want_stats(&mut self) {
        let run = self.run_name.clone();
        if let Some(store) = self.store.clone()
            && self.stats.wants(&run)
        {
            let dir = self.dir();
            self.stats.start(run, move || {
                let stats = c2_census::analyze::dataset_stats(&store, &dir).map_err(|e| format!("{e:#}"))?;
                let keep_real = RunMeta::load(&dir).map_err(|e| format!("{e:#}"))?.cfg.keep_real;
                Ok((stats, keep_real))
            });
        }
    }

    fn status_stats(&mut self, ui: &mut egui::Ui) {
        let run_name = self.run_name.clone();
        // Every run, marked with the one shown here (the Run name box) and the default census, and
        // the shown run's statistics.
        let runs_key = (self.runs_refresh, run_name.clone());
        if self.runs.wants(&runs_key) {
            let r = run_name.clone();
            self.runs.start(runs_key.clone(), move || Ok(c2_census::analyze::run_history_using(&r)));
        }
        self.want_stats();
        let off = self.utc_offset;
        let mut refresh = false;
        egui::ScrollArea::vertical().id_salt("status_scroll").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Census runs");
                if ui.button("refresh").clicked() {
                    refresh = true;
                }
                if self.runs.is_running() || self.stats.is_running() {
                    ui.spinner();
                }
            });
            // The list for this run, else (for the moment it takes to read) the last one, whose
            // in-use mark is then taken from the run name here, not from the list.
            let current = matches!(&self.runs.done, Some((k, _)) if *k == runs_key);
            let runs = match &self.runs.done {
                Some((k, Err(e))) if *k == runs_key => {
                    ui.colored_label(egui::Color32::LIGHT_RED, e);
                    return;
                }
                Some((_, Ok(runs))) => runs,
                _ => {
                    ui.label("Reading the census output…");
                    return;
                }
            };
            // Scrolls sideways when a long run name with its marks widens it past the window.
            egui::ScrollArea::horizontal().id_salt("runs_h").show(ui, |ui| {
                egui::Grid::new("runs_grid").striped(true).show(ui, |ui| {
                    for h in ["run", "kind", "data through", "started", "time computing", "correlations computed", "GPU multiply-adds", "state"] {
                        ui.strong(h);
                    }
                    ui.end_row();
                    for r in runs {
                        // "in use": the run this window shows (the Run name box); "default": the one
                        // the app and `c2` open when no run is named (census/LATEST).
                        let in_use = if current { r.in_use } else { r.name == run_name };
                        let marks: Vec<&str> = [(in_use, "in use"), (r.latest, "default"), (r.from_update, "new data")].iter().filter(|m| m.0).map(|m| m.1).collect();
                        let name = if marks.is_empty() { r.name.clone() } else { format!("{}  ({})", r.name, marks.join(", ")) };
                        ui.label(name);
                        ui.label(&r.kind);
                        ui.label(&r.data_last_day);
                        ui.label(local_time(r.started_unix, off));
                        ui.label(if r.sessions > 1 { format!("{} in {} sessions", run::hms(r.seconds), r.sessions) } else { run::hms(r.seconds) });
                        ui.label(format!("{} + {} chance", big(r.correlations_real), big(r.correlations_null)));
                        ui.label(big(r.multiply_adds));
                        ui.label(if r.state == "finished" {
                            format!("finished {}", r.finished_unix.map(|t| local_time(t, off)).unwrap_or_default())
                        } else {
                            format!("{} ({:.0}%)", r.state, 100.0 * r.fraction)
                        });
                        ui.end_row();
                    }
                });
            });
            let last_full = runs.iter().filter(|r| r.kind == "full" && r.finished_unix.is_some()).max_by_key(|r| r.finished_unix);
            ui.label(match last_full {
                Some(r) => format!(
                    "Last full census: '{}', finished {} after {} of computing: {} correlations ({} real, the rest on sign-shuffled data for the chance baseline).",
                    r.name,
                    local_time(r.finished_unix.unwrap_or(0), off),
                    run::hms(r.seconds),
                    big(r.correlations_real + r.correlations_null),
                    big(r.correlations_real)
                ),
                None => "No full census has finished yet.".into(),
            });
            ui.label(
                "Incremental updates from new data: not built yet. When `c2 update` finds new prices it reruns the whole census under a dated name \
                 (marked \"new data\" above) and the app switches to the newest run. The census keeps each stored pair's running sums so a later \
                 version can extend them with just the new days.",
            );
            let log = std::fs::read_to_string(c2_data::data_dir().join("update.log")).unwrap_or_default();
            if log.trim().is_empty() {
                ui.label("Data updates: none have run yet (the weekday update timer is not enabled; `c2 install-timer --enable` turns it on).");
            } else {
                ui.label("Recent data updates (update.log):");
                let lines: Vec<&str> = log.lines().collect();
                for l in &lines[lines.len().saturating_sub(8)..] {
                    ui.monospace(*l);
                }
            }
            ui.separator();
            ui.heading(format!("What census '{run_name}' found"));
            let (st, keep_real) = match self.stats.get(&run_name) {
                Some(Ok((st, keep_real))) => (st, *keep_real),
                Some(Err(e)) => {
                    ui.label(format!("This run has no output to summarize yet ({e})."));
                    return;
                }
                None if self.store.is_none() => {
                    ui.colored_label(egui::Color32::LIGHT_RED, format!("These statistics need the price store. {}", self.store_note));
                    return;
                }
                None => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(format!("Reading census '{run_name}'…"));
                    });
                    return;
                }
            };
            let lt = &st.lagged;
            if let Some(e) = &st.stock_list_error {
                ui.colored_label(YELLOW_NOTE, format!("The run's stock names can't be recovered, so the per-stock charts below have counts but no names: {e}"));
            }
            egui::Grid::new("st_head").striped(true).show(ui, |ui| {
                let mut row = |k: &str, v: String| {
                    ui.label(k);
                    ui.label(v);
                    ui.end_row();
                };
                row("correlations computed", format!("{} real, plus {} times as many on sign-shuffled data as the chance baseline", big(st.correlations_all), st.nulls));
                row("matrices (a correlation period x move lengths x offset)", commas(st.matrices_all as f64));
                row("correlation periods", st.periods.to_string());
                row("stocks with stored relationships", commas(st.stocks as f64));
                for (k, z) in [(1, 4), (2, 5)] {
                    row(
                        &format!("pairs at |z| of {z} or more (offsets ≥ 1)"),
                        format!("{} real vs {} on shuffled data: x{:.2}", commas(lt.real[k]), commas(lt.null[k]), lt.real[k] / lt.null[k].max(1.0)),
                    );
                }
                row("stored relationships", format!("{} (each matrix's strongest {keep_real} plus every significant one); {} chance ones stored for comparison", commas(st.stored_real as f64), commas(st.stored_null as f64)));
                row(
                    "significant (passed the false-discovery test)",
                    format!("{} relationships: {} distinct pairs, {} stocks, in {} of {} matrices", commas(lt.significant as f64), commas(st.significant_pairs as f64), commas(st.significant_stocks as f64), commas(st.fdr_matrices as f64), commas(lt.matrices as f64)),
                );
                row(
                    "the same test run on pure chance fired in",
                    if st.placebo_recorded {
                        format!("{} matrices (close to the line above means the 'significant' ones are mostly chance too)", st.placebo_matrices)
                    } else {
                        // Older runs' summaries have no column for it: 0 would be a false count.
                        "not recorded by this run (it was made before the census kept this count)".to_string()
                    },
                );
            });
            ui.add_space(6.0);
            let ratio = |t: &c2_census::analyze::Tally| t.real[1] / t.null[1].max(1.0);
            let ratio_plot = |ui: &mut egui::Ui, id: &str, title: &str, items: Vec<(String, c2_census::analyze::Tally)>| {
                ui.strong(title);
                let labels: Vec<String> = items.iter().map(|i| i.0.clone()).collect();
                let bars: Vec<Bar> = items
                    .iter()
                    .enumerate()
                    .map(|(k, (l, t))| {
                        Bar::new(k as f64, ratio(t)).width(0.7).fill(BLUE).name(format!("{l}: {:.0} real vs {:.0} chance at |z| ≥ 4; {} significant", t.real[1], t.null[1], t.significant))
                    })
                    .collect();
                Plot::new(id).height(160.0).y_axis_label("real / chance").x_axis_formatter(category_axis(labels)).show(ui, |p| {
                    p.hline(HLine::new("chance level", 1.0).color(egui::Color32::WHITE));
                    p.bar_chart(BarChart::new("real / chance", bars).color(BLUE));
                });
            };
            ui.columns(2, |cols| {
                let ui = &mut cols[0];
                ui.strong("Strength of the stored correlations (|r|)");
                let (tr, tn) = (st.hist_real.iter().sum::<u64>().max(1) as f64, st.hist_null.iter().sum::<u64>().max(1) as f64);
                let br: Vec<Bar> = st.hist_real.iter().enumerate().map(|(k, &v)| Bar::new((k as f64 + 0.5) / 50.0 - 0.0045, 100.0 * v as f64 / tr).width(0.009).fill(ORANGE)).collect();
                let bn: Vec<Bar> = st.hist_null.iter().enumerate().map(|(k, &v)| Bar::new((k as f64 + 0.5) / 50.0 + 0.0045, 100.0 * v as f64 / tn).width(0.009).fill(GREY)).collect();
                Plot::new("st_hist").height(160.0).legend(Legend::default()).x_axis_label("|r|").y_axis_label("% of stored").show(ui, |p| {
                    p.bar_chart(BarChart::new("real: each matrix's strongest", br).color(ORANGE));
                    p.bar_chart(BarChart::new("chance: strongest on shuffled data", bn).color(GREY));
                });
                ui.label("Real relationships beyond chance would push the orange bars further right than the grey ones.");
                ratio_plot(
                    &mut cols[1],
                    "st_h",
                    "Real / chance at |z| ≥ 4, by move lengths (A's move -> B's move, in trading days)",
                    st.by_horizons.iter().map(|((a, b), t)| (format!("{a}d->{b}d"), *t)).collect(),
                );
            });
            ui.columns(2, |cols| {
                ratio_plot(&mut cols[0], "st_off", "Real / chance at |z| ≥ 4, by offset (in multiples of B's move length)", st.by_offset_steps.iter().map(|(k, t)| (k.to_string(), *t)).collect());
                ratio_plot(
                    &mut cols[1],
                    "st_len",
                    "Real / chance at |z| ≥ 4, by correlation period length",
                    st.by_length.iter().map(|(l, t)| (if *l == 0 { "all".to_string() } else { format!("{l} days") }, *t)).collect(),
                );
            });
            ui.columns(2, |cols| {
                let ui = &mut cols[0];
                ui.strong("Significant relationships by the year their correlation period ended");
                let bars: Vec<Bar> = st
                    .by_year
                    .iter()
                    .map(|(y, t)| Bar::new(*y as f64, t.significant as f64).width(0.7).fill(ORANGE).name(format!("{y}: {} significant; real / chance at |z| ≥ 4 {:.2}", t.significant, ratio(t))))
                    .collect();
                Plot::new("st_year").height(160.0).y_axis_label("significant").show(ui, |p| {
                    p.bar_chart(BarChart::new("significant", bars).color(ORANGE));
                });
                let ui = &mut cols[1];
                ui.strong("Per stock: average strength (|r|) of its stored relationships");
                let mut hist = vec![0u32; 50];
                for s in &st.per_stock {
                    hist[((s.2 * 50.0) as usize).min(49)] += 1;
                }
                let bars: Vec<Bar> = hist.iter().enumerate().map(|(k, &v)| Bar::new((k as f64 + 0.5) / 50.0, v as f64).width(0.018).fill(BLUE)).collect();
                Plot::new("st_stock_r").height(160.0).x_axis_label("average |r|").y_axis_label("stocks").show(ui, |p| {
                    p.bar_chart(BarChart::new("stocks", bars).color(BLUE));
                });
            });
            ui.columns(2, |cols| {
                let bucket_plot = |ui: &mut egui::Ui, id: &str, title: &str, edges: &[(usize, usize, &str)], values: Vec<usize>, color: egui::Color32| {
                    ui.strong(title);
                    let bars: Vec<Bar> = edges
                        .iter()
                        .enumerate()
                        .map(|(k, (lo, hi, label))| {
                            let n = values.iter().filter(|&&v| v >= *lo && v <= *hi).count();
                            Bar::new(k as f64, n as f64).width(0.7).fill(color).name(format!("{label}: {n} stocks"))
                        })
                        .collect();
                    Plot::new(id).height(160.0).y_axis_label("stocks").x_axis_formatter(category_axis(edges.iter().map(|e| e.2.to_string()).collect())).show(ui, |p| {
                        p.bar_chart(BarChart::new("stocks", bars).color(color));
                    });
                };
                bucket_plot(
                    &mut cols[0],
                    "st_sig",
                    "Per stock: number of significant relationships",
                    &[(0, 0, "0"), (1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 100, "21-100"), (101, usize::MAX, "over 100")],
                    st.per_stock.iter().map(|s| s.3).collect(),
                    ORANGE,
                );
                bucket_plot(
                    &mut cols[1],
                    "st_stored",
                    "Per stock: number of stored relationships",
                    &[(1, 10, "1-10"), (11, 100, "11-100"), (101, 1000, "101-1,000"), (1001, 10_000, "1,001-10,000"), (10_001, usize::MAX, "over 10,000")],
                    st.per_stock.iter().map(|s| s.1).collect(),
                    BLUE,
                );
            });
        });
        if refresh {
            self.runs_refresh += 1;
            self.stats.clear();
        }
    }

    fn dir(&self) -> PathBuf {
        run::run_dir(&self.run_name)
    }

    /// Starts `c2 census run` for the current run in its own process group (closing the window
    /// doesn't stop it), its output appended to `<run>/run.out`, and watches it (`check_launch`).
    /// `run::run` refuses a second writer of a run; that refusal is then shown on the Status tab.
    fn start_census(&mut self) {
        let dir = self.dir();
        let _ = std::fs::create_dir_all(&dir);
        let out = dir.join("run.out");
        let mut cmd = std::process::Command::new(c2_binary());
        cmd.args(["census", "run", "--name", &self.run_name]);
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        self.launch_note = None;
        self.message.clear();
        match spawn_logged(cmd, &out) {
            Ok((child, from)) => self.launch = Some(Launch { run: self.run_name.clone(), child, out, from }),
            Err(e) => self.launch_note = Some((true, format!("Could not start c2 for census '{}': {e}", self.run_name))),
        }
        self.last_poll = Instant::now() - Duration::from_secs(10);
    }

    /// Once the census process started from here exits, says what it said (`launch_outcome`).
    fn check_launch(&mut self) {
        let status = match self.launch.as_mut().map(|l| l.child.try_wait()) {
            None | Some(Ok(None)) => return,
            Some(Ok(Some(status))) => Ok(status),
            Some(Err(e)) => Err(e),
        };
        if let Some(l) = self.launch.take() {
            self.launch_note = Some(launch_outcome(&l, status));
        }
        self.message.clear();
        self.last_poll = Instant::now() - Duration::from_secs(10);
    }

    fn poll(&mut self) {
        if self.last_poll.elapsed() < Duration::from_secs(2) {
            return;
        }
        self.last_poll = Instant::now();
        let dir = self.dir();
        // Whoever holds the run's lock (`run::run` takes it for its whole duration), or for runs
        // started by older binaries a live c2 process named in its pid file.
        self.pid = run::running_pid(&dir);
        self.control = run::control_state(&dir);
        match std::fs::read_to_string(dir.join("status.json")).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string())) {
            Ok(v) => {
                self.status = Some(v);
                self.status_err.clear();
            }
            Err(e) => {
                self.status = None;
                self.status_err = format!("no status for census '{}' yet ({e})", self.run_name);
            }
        }
    }

    fn status_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Correlation census");
        ui.horizontal(|ui| {
            ui.label("Run name:");
            // Every tab reads the run named here. It switches on Enter, when the box loses focus,
            // or after a pause in typing, not on every keystroke.
            let resp = ui.text_edit_singleline(&mut self.run_input);
            if resp.changed() {
                self.run_edited = Some(Instant::now());
            }
            if resp.lost_focus() {
                let name = self.run_input.clone();
                self.set_run(&name);
            }
            if self.run_input.trim() != self.run_name {
                ui.label(format!("(still showing '{}': press Enter or pause typing to switch)", self.run_name));
            }
        });
        ui.separator();
        match &self.status {
            Some(s) => {
                let frac = s["fraction_done"].as_f64().unwrap_or(0.0) as f32;
                // As the runs list reads it: running while a process computes the run (whatever
                // status.json says), interrupted when the file says running but nothing does.
                let file_state = s["state"].as_str().unwrap_or("?");
                let state = match (self.pid, file_state) {
                    (Some(_), _) => "running",
                    (None, "running") => "interrupted",
                    (None, other) => other,
                };
                ui.label(format!(
                    "State: {state}  ·  process: {}  ·  control: {}",
                    self.pid.map(|p| format!("running (pid {p})")).unwrap_or_else(|| "not running".into()),
                    self.control
                ));
                ui.add(egui::ProgressBar::new(frac).show_percentage());
                // An estimate of the time left only means something while it runs.
                let eta = if self.pid.is_some() { format!(" · ETA {}", run::hms(s["eta_seconds"].as_f64().unwrap_or(-1.0))) } else { String::new() };
                ui.label(format!("{} of {} work units (one pair of move lengths in one correlation period){eta}", s["units_done"], s["units_total"]));
            }
            None => {
                ui.label(&self.status_err);
            }
        }
        // What this run's plan on today's prices leaves out, once a tab has built it.
        if let Some(note) = self.plan_note_for(&self.run_name) {
            ui.colored_label(YELLOW_NOTE, note);
        }
        ui.add_space(8.0);
        // While a new run name is being typed, the buttons wait for it to be switched to, so they
        // never act on a different run than the one shown.
        let settled = self.run_input.trim() == self.run_name;
        // Only one process may write a run. A census process started from here that hasn't exited
        // counts as running too: it builds its plan before it takes the run's lock.
        let live = self.pid.is_some() || self.launch.as_ref().is_some_and(|l| l.run == self.run_name);
        ui.horizontal(|ui| {
            let dir = self.dir();
            if ui.add_enabled(settled && !live, egui::Button::new("Start / resume census")).clicked() {
                self.start_census();
            }
            if ui.add_enabled(settled, egui::Button::new("Pause (frees the GPU)")).clicked() {
                let _ = run::set_control(&dir, "pause");
                self.message = "pausing after the work unit in progress".into();
            }
            if ui.add_enabled(settled, egui::Button::new("Resume")).clicked() {
                let _ = run::set_control(&dir, "run");
                self.message = "resuming".into();
            }
            if ui.add_enabled(settled && self.pid.is_some(), egui::Button::new("Stop")).clicked()
                && let Some(pid) = self.pid
            {
                let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
                self.message = format!("asked pid {pid} to stop after committing the work unit in progress");
            }
        });
        if !self.message.is_empty() {
            ui.label(&self.message);
        }
        if let Some(l) = &self.launch
            && l.run == self.run_name
            && self.pid.is_none()
        {
            ui.label(format!("Starting census '{}' (pid {}): it builds its plan, then takes the run's lock…", l.run, l.child.id()));
        }
        if let Some((error, note)) = &self.launch_note {
            if *error {
                ui.colored_label(egui::Color32::LIGHT_RED, note);
            } else {
                ui.label(note);
            }
        }
        ui.add_space(4.0);
        ui.label("Stopping or pausing loses at most the work unit in progress; starting again resumes where it left off.");
        ui.separator();
        self.status_stats(ui);
    }

    fn load_lag_table(&mut self) {
        self.lag_table.clear();
        self.lag_table_run = self.run_name.clone();
        self.lag_note.clear();
        let entries = match std::fs::read_dir(self.dir().join("tasks")) {
            Ok(entries) => entries,
            Err(e) => {
                self.lag_note = format!("Census '{}' has no output to summarize yet ({e}).", self.run_name);
                return;
            }
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().and_then(|x| x.to_str()) != Some("csv") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let mut lines = text.lines();
            let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
            let col = |name: &str| header.iter().position(|h| *h == name);
            let (Some(ch1), Some(ch2), Some(clag), Some(cr4), Some(cn4), Some(cfdr)) =
                (col("h1"), col("h2"), col("lag"), col("real_z4"), col("null_z4"), col("fdr_r"))
            else {
                continue;
            };
            for line in lines {
                let f: Vec<&str> = line.split(',').collect();
                let num = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
                let key = (num(ch1) as usize, num(ch2) as usize);
                let row = self.lag_table.entry(key).or_default().entry(num(clag) as usize).or_default();
                row.real_z4 += num(cr4);
                row.null_z4 += num(cn4);
                row.matrices += 1;
                row.fdr += usize::from(f.get(cfdr).is_some_and(|v| !v.is_empty()));
            }
        }
        if self.lag_table.is_empty() {
            self.lag_note = format!("Census '{}' has no finished matrices to summarize yet.", self.run_name);
        }
        if !self.lag_table.contains_key(&self.pair)
            && let Some(k) = self.lag_table.keys().next()
        {
            self.pair = *k;
        }
    }

    /// Looks up the strongest stored pairs of the Census tab's ticker in the current run.
    fn start_find_pairs(&mut self) {
        let (dir, ticker) = (self.dir(), self.ticker.trim().to_uppercase());
        self.pairs.start((self.run_name.clone(), ticker.clone()), move || find_pairs(&dir, &ticker).map_err(|e| format!("{e:#}")));
    }

    fn census_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Real vs chance, by offset");
        // The table belongs to the run it was read from.
        if self.lag_table_run != self.run_name {
            self.lag_table.clear();
        }
        ui.horizontal(|ui| {
            if ui.button(if self.lag_table.is_empty() { "Load census summaries" } else { "Reload" }).clicked() {
                self.load_lag_table();
            }
            let pairs: Vec<(usize, usize)> = self.lag_table.keys().copied().collect();
            egui::ComboBox::from_label("move lengths, A / B (trading days)")
                .selected_text(format!("{} / {}", self.pair.0, self.pair.1))
                .show_ui(ui, |ui| {
                    for p in pairs {
                        ui.selectable_value(&mut self.pair, p, format!("{} / {}", p.0, p.1));
                    }
                });
        });
        if self.lag_table_run == self.run_name && !self.lag_note.is_empty() {
            ui.colored_label(YELLOW_NOTE, &self.lag_note);
        }
        if let Some(rows) = self.lag_table.get(&self.pair).cloned() {
            ui.label("Pairs with |z| ≥ 4, summed over all correlation periods. Real / chance near 1 means no lead-lag beyond chance (offset 0 is the same period, moving together).");
            let points: Vec<[f64; 2]> = rows.iter().filter(|(k, _)| **k > 0).map(|(k, r)| [*k as f64, r.real_z4 / r.null_z4.max(1.0)]).collect();
            Plot::new("ratio").height(220.0).legend(Legend::default()).show(ui, |p| {
                p.line(Line::new("real / chance (offsets ≥ 1)", PlotPoints::from(points)));
                p.line(Line::new("chance level", PlotPoints::from(vec![[0.0, 1.0], [rows.keys().last().copied().unwrap_or(1) as f64, 1.0]])));
            });
            egui::ScrollArea::vertical().max_height(220.0).id_salt("lags").show(ui, |ui| {
                egui::Grid::new("lag_grid").striped(true).show(ui, |ui| {
                    for h in ["offset", "real", "chance", "real/chance", "matrices", "matrices with significant pairs"] {
                        ui.strong(h);
                    }
                    ui.end_row();
                    for (k, r) in &rows {
                        ui.label(k.to_string());
                        ui.label(format!("{:.0}", r.real_z4));
                        ui.label(format!("{:.0}", r.null_z4));
                        ui.label(format!("{:.2}", r.real_z4 / r.null_z4.max(1.0)));
                        ui.label(r.matrices.to_string());
                        ui.label(format!("{:.1}%", 100.0 * r.fdr as f64 / r.matrices.max(1) as f64));
                        ui.end_row();
                    }
                });
            });
        }
        ui.separator();
        ui.heading("A stock's strongest stored pairs");
        ui.horizontal(|ui| {
            ui.label("Ticker:");
            ui.text_edit_singleline(&mut self.ticker);
            if ui.add_enabled(!self.pairs.is_running() && self.store.is_some(), egui::Button::new("Find")).clicked() {
                self.start_find_pairs();
            }
            if self.pairs.is_running() {
                ui.spinner();
            }
        });
        if self.store.is_none() {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.store_note);
        }
        let pair_rows: &[PairRow] = match &self.pairs.done {
            Some((k, Ok(rows))) if k.0 == self.run_name => rows,
            Some((k, Err(e))) if k.0 == self.run_name => {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
                &[]
            }
            _ => &[],
        };
        egui::ScrollArea::vertical().id_salt("pairs").show(ui, |ui| {
            egui::Grid::new("pair_grid").striped(true).show(ui, |ui| {
                for h in ["correlation period", "partner", "role", "A days", "B days", "offset", "r", "z", "observations"] {
                    ui.strong(h);
                }
                ui.end_row();
                for r in pair_rows {
                    ui.label(&r.window);
                    ui.label(&r.partner);
                    ui.label(r.role);
                    ui.label(r.h1.to_string());
                    ui.label(r.h2.to_string());
                    ui.label(r.lag.to_string());
                    ui.label(format!("{:+.3}", r.r));
                    ui.label(format!("{:+.1}", r.z));
                    ui.label(r.obs.to_string());
                    ui.end_row();
                }
            });
        });
    }

    fn holdout_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("$10,000 holdout test");
        ui.horizontal(|ui| {
            ui.label("Signal:");
            ui.radio_value(&mut self.family, 0, "census lead-lag pairs");
            ui.radio_value(&mut self.family, 1, "peer momentum");
        });
        egui::Grid::new("holdout_opts").min_col_width(120.0).show(ui, |ui| {
            ui.label("First day treated as the future");
            ui.add(egui::TextEdit::singleline(&mut self.start).desired_width(110.0));
            ui.end_row();
            ui.label("Stocks held");
            ui.add(egui::DragValue::new(&mut self.picks).range(2..=200));
            ui.end_row();
            ui.label("Rebalance every N trading days");
            ui.add(egui::DragValue::new(&mut self.rebalance).range(1..=63));
            ui.end_row();
            ui.label("Costs (bps per trade, comma-separated)");
            ui.add(egui::TextEdit::singleline(&mut self.costs).desired_width(110.0));
            ui.end_row();
            ui.label("Random-guess accounts");
            ui.add(egui::DragValue::new(&mut self.random).range(100..=10000));
            ui.end_row();
        });
        ui.horizontal(|ui| {
            let can_run = !self.holdout.is_running() && self.store.is_some();
            if ui.add_enabled(can_run, egui::Button::new("Run")).clicked() {
                self.start_holdout();
            }
            if let Some(t) = self.holdout_started
                && self.holdout.is_running()
            {
                ui.spinner();
                ui.label(format!("running on census '{}'… {:.0}s (the first run loads 20 years of data)", self.run_name, t.elapsed().as_secs_f64()));
            }
        });
        if self.store.is_none() {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.store_note);
        }
        // Only a result of the current run (a run switch clears it anyway).
        let (run, census_pairs, (r, note)) = match &self.holdout.done {
            Some((k, Ok(r))) if k.0 == self.run_name => (&k.0, k.1 == 0, r),
            Some((k, Err(e))) if k.0 == self.run_name => {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
                return;
            }
            _ => return,
        };
        ui.separator();
        if let Some(note) = note {
            ui.colored_label(YELLOW_NOTE, note);
        }
        let (first, last) = (c2_data::date_string(r.first_day), c2_data::date_string(r.last_day));
        if census_pairs && r.pairs == 0 {
            // Nothing was simulated: no correlation period of the needed kind ended before the
            // holdout. The benchmarks still hold.
            ui.label(format!(
                "Census '{run}': no census pairs for this holdout ({first} .. {last}): none of its correlation periods with B moves of the \
                 rebalance length ended before {first}, so nothing was simulated. The benchmarks over the same days:"
            ));
        } else {
            ui.label(format!(
                "Census '{run}': {first} .. {last} ({} trading days) · {} stocks could be picked · picks beat the market {:.1}% of the time · rank IC t = {:+.2} · long-short t = {:+.2}",
                r.days, r.candidates, 100.0 * r.hit_rate, r.ic_t, r.ls_t
            ));
        }
        if census_pairs && r.pairs > 0 {
            ui.label(format!("Picks come from {} stored pairs of {} correlation periods that ended before the holdout.", commas(r.pairs as f64), r.windows_used));
        }
        let mut left_out = Vec::new();
        if r.periods_unavailable > 0 {
            left_out.push(format!(
                "{} correlation periods that ended before the holdout but aren't covered exactly by today's price data (the latest covered ones were used instead)",
                r.periods_unavailable
            ));
        }
        if r.pairs_skipped > 0 {
            left_out.push(format!("{} stored pairs with a stock that is no longer in today's price data", commas(r.pairs_skipped as f64)));
        }
        if census_pairs && !left_out.is_empty() {
            ui.colored_label(YELLOW_NOTE, format!("Left out: {}.", left_out.join("; ")));
        }
        egui::Grid::new("holdout_results").striped(true).show(ui, |ui| {
            for h in ["", "final value", "return", "random guessers (median)", "share of random at least as good"] {
                ui.strong(h);
            }
            ui.end_row();
            for b in &r.benchmarks {
                ui.label(&b.label);
                ui.label(format!("${:.0}", b.final_value));
                ui.label(pct(b.final_value));
                ui.label("");
                ui.label("");
                ui.end_row();
            }
            for (c, s, med, share) in &r.strategy {
                ui.label(format!("Picks, {c} bps per trade"));
                ui.label(format!("${s:.0}"));
                ui.label(pct(*s));
                ui.label(format!("${med:.0}"));
                ui.label(format!("{:.1}%  (p = {share:.3})", 100.0 * share));
                ui.end_row();
            }
        });
        let series = |f: fn(&(i32, f64, f64, f64, f64)) -> f64| -> PlotPoints {
            PlotPoints::from(r.curve.iter().enumerate().filter(|(_, c)| f(c).is_finite()).map(|(k, c)| [k as f64, f(c)]).collect::<Vec<_>>())
        };
        if r.curve.is_empty() {
            return;
        }
        Plot::new("equity").legend(Legend::default()).show(ui, |p| {
            p.line(Line::new("picks (first cost level)", series(|c| c.1)));
            p.line(Line::new("SPY", series(|c| c.2)));
            p.line(Line::new("QQQ", series(|c| c.3)));
            p.line(Line::new("random guessers (median)", series(|c| c.4)));
        });
    }

    fn start_holdout(&mut self) {
        let Some(store) = self.store.clone() else { return };
        self.holdout_started = Some(Instant::now());
        let (run, dir, slot, generation) = (self.run_name.clone(), self.dir(), self.plan_slot.clone(), self.store_gen);
        let family = if self.family == 0 {
            Family::CensusPairs
        } else {
            Family::PeerMomentum { peers: 20, lookback: 21, window: 756 }
        };
        let start = self.start.clone();
        let (picks, rebalance, random) = (self.picks, self.rebalance, self.random);
        let costs: Vec<f64> = self.costs.split(',').filter_map(|c| c.trim().parse().ok()).collect();
        let key: HoldoutKey = (run.clone(), self.family, start.clone(), picks, rebalance, self.costs.clone(), random);
        self.holdout.start(key, move || {
            // This run's plan (`Plan::for_run`, built if the slot holds another run's): its own
            // correlation periods, task numbering and stock ids, by which its records are looked
            // up. The holdout refuses any other plan (e.g. a `Plan::new` one after a store rebuild).
            let rp = plan_for_run(&slot, &store, generation, &run, &dir)?;
            let plan = &rp.plan;
            let opt = Options {
                family,
                rebalance,
                start: c2_data::parse_date(&start).map_err(|e| format!("{e:#}"))?,
                end: i32::MAX,
                picks,
                costs_bps: if costs.is_empty() { vec![0.0] } else { costs },
                random_runs: random,
                cash: 10_000.0,
                all_windows: false,
            };
            let report = c2_census::holdout::holdout(&store, plan, &dir, &opt).map_err(|e| format!("{e:#}"))?;
            Ok((report, plan_note(&rp)))
        });
    }

    fn data_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Price store");
        ui.label(&self.store_note);
        ui.label("Stooq prices are split- and dividend-adjusted; test tickers are excluded and one-day bad prints repaired when panels are built.");
        ui.separator();
        egui::Grid::new("decades").striped(true).show(ui, |ui| {
            for h in ["decade", "symbols", "rows", "0% days", "under $1", "close only"] {
                ui.strong(h);
            }
            ui.end_row();
            for r in &self.decades {
                let p = |a: usize, b: usize| format!("{:.1}%", 100.0 * a as f64 / b.max(1) as f64);
                ui.label(format!("{}s", r.decade));
                ui.label(r.symbols.to_string());
                ui.label(r.rows.to_string());
                ui.label(p(r.zero_return, r.with_prev));
                ui.label(p(r.under_1, r.rows));
                ui.label(p(r.close_only, r.rows));
                ui.end_row();
            }
        });
    }
}

/// The strongest stored real pairs involving `ticker`, as leader or target, across all windows.
/// Records name stocks by the run's own store ids and periods by the run's own ids, so both are
/// read through the run's description (`RunMeta`), never today's store.
fn find_pairs(dir: &Path, ticker: &str) -> anyhow::Result<Vec<PairRow>> {
    let meta = RunMeta::load(dir)?;
    let stocks = meta.stocks()?;
    let id = stocks.id(ticker).ok_or_else(|| anyhow::anyhow!("{ticker} was not in the price store when census {} was made", meta.name))?;
    let label = |w: &c2_census::meta::RunWindow| format!("{}..{}", c2_data::date_string(w.first), c2_data::date_string(w.last));
    let mut rows = Vec::new();
    for task in meta.tasks()? {
        let path = meta.task_path(task.id, "rec");
        if !path.exists() {
            continue;
        }
        for r in c2_census::analyze::read_records(&path)? {
            if r.flags & FLAG_NULL != 0 || r.lag == 0 || (r.leader != id && r.target != id) {
                continue;
            }
            let (partner, role) = if r.leader == id { (r.target, "leads") } else { (r.leader, "follows") };
            rows.push(PairRow {
                window: meta.windows.get(r.window as usize).map(label).unwrap_or_default(),
                partner: stocks.ticker(partner).unwrap_or_default().to_string(),
                role,
                h1: r.h1,
                h2: r.h2,
                lag: r.lag,
                r: r.r,
                z: r.z,
                obs: r.obs,
            });
        }
    }
    rows.sort_by(|a, b| b.z.abs().total_cmp(&a.z.abs()));
    rows.truncate(200);
    Ok(rows)
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.check_launch();
        self.poll();
        self.watch_store();
        // Results arrive whichever tab is shown.
        self.poll_jobs();
        if self.shot.is_some() {
            self.walk_step();
        }
        // A pause in typing a run name switches to it.
        if self.run_edited.is_some_and(|t| t.elapsed() >= RUN_IDLE) {
            let name = self.run_input.clone();
            self.set_run(&name);
        }
        if self.shot.as_ref().is_some_and(|s| s.frames == 0) {
            self.setup_self_check();
        }
        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Status, "Status");
                ui.selectable_value(&mut self.tab, Tab::Strongest, "Strongest");
                ui.selectable_value(&mut self.tab, Tab::Stock, "Stock timeline");
                ui.selectable_value(&mut self.tab, Tab::Forecast, "Forecast");
                ui.selectable_value(&mut self.tab, Tab::Explain, "Explainer");
                ui.selectable_value(&mut self.tab, Tab::Census, "Census");
                ui.selectable_value(&mut self.tab, Tab::Holdout, "Holdout");
                ui.selectable_value(&mut self.tab, Tab::Data, "Data");
                // For a minute after the price store was reopened (or failed to reopen).
                if let Some((at, note)) = &self.store_changed
                    && at.elapsed() < Duration::from_secs(60)
                {
                    ui.separator();
                    ui.colored_label(YELLOW_NOTE, note);
                }
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::Status => self.status_tab(ui),
            Tab::Strongest => self.strongest_tab(ui),
            Tab::Stock => self.stock_tab(ui),
            Tab::Forecast => self.forecast_tab(ui),
            Tab::Explain => self.explain_tab(ui),
            Tab::Census => self.census_tab(ui),
            Tab::Holdout => self.holdout_tab(ui),
            Tab::Data => self.data_tab(ui),
        });
        // Self-check: once the tab has drawn and has its data or its error (or the time is up),
        // save the window and exit.
        if let Some(mut shot) = self.shot.take() {
            shot.frames += 1;
            if shot.plan {
                let _ = self.need_plan();
            }
            let ready = self.tab_ready() && (!shot.plan || self.plan.get(&self.run_name).is_some());
            let timed_out = shot.started.elapsed() >= shot.timeout;
            let waited = shot.started.elapsed() >= shot.min_wait;
            if !shot.requested && shot.frames > 30 && waited && (ready || timed_out) {
                if !ready {
                    eprintln!("c2-gui self-check: saving after {:.0}s although the tab is still waiting", shot.started.elapsed().as_secs_f64());
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
                shot.requested = true;
            }
            let image = ctx.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(img) = image {
                let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                let _ = std::fs::write(format!("{}.rgba", shot.prefix), bytes);
                let _ = std::fs::write(format!("{}.txt", shot.prefix), format!("{} {}", img.size[0], img.size[1]));
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            self.shot = Some(shot);
        }
        // While a background job computes, look for its result often, so that a quick one (a
        // forecast from data already read takes milliseconds) shows at once; while a run name is
        // being typed, wake up in time to switch to it.
        let wait = if self.jobs_running() {
            50
        } else if self.run_edited.is_some() {
            200
        } else {
            500
        };
        ctx.request_repaint_after(Duration::from_millis(wait));
    }
}

impl App {
    /// Self-check mode: opens the tab named by `C2_GUI_TAB` with the choices given in the other
    /// `C2_GUI_*` variables (`C2_GUI_RUN` picks the run, see `App::new`).
    fn setup_self_check(&mut self) {
        let env = |k: &str| std::env::var(k).ok();
        match env("C2_GUI_TAB").as_deref() {
            Some("census") => {
                self.tab = Tab::Census;
                self.load_lag_table();
                if let Some(t) = env("C2_GUI_TICKER") {
                    self.ticker = t;
                    self.start_find_pairs();
                }
            }
            Some("holdout") => {
                self.tab = Tab::Holdout;
                self.start_holdout();
            }
            Some("data") => self.tab = Tab::Data,
            Some("strongest") => self.tab = Tab::Strongest,
            Some("stock") => {
                self.tab = Tab::Stock;
                if let Some(t) = env("C2_GUI_TICKER") {
                    self.tl.input = t;
                }
                if let Some(w) = env("C2_GUI_WINDOW") {
                    self.tl.window = w;
                }
                self.tl.select_first = env("C2_GUI_SELECT").is_some();
                self.tl.significant_only = env("C2_GUI_SIGNIFICANT").is_some();
                let t = self.tl.input.clone();
                self.request_timeline(&t);
            }
            Some("forecast") => {
                self.tab = Tab::Forecast;
                self.fc.ticker = env("C2_GUI_TICKER").unwrap_or_else(|| "AAPL".into());
                self.tl.input = self.fc.ticker.clone();
                self.fc.as_of = env("C2_GUI_AS_OF").and_then(|d| c2_data::parse_date(&d).ok());
                self.fc.horizon = env("C2_GUI_DAYS").and_then(|d| d.parse().ok()).unwrap_or(21);
                self.fc.significant_only = env("C2_GUI_SIGNIFICANT").is_some();
                // Press "◀ month" this many times, each once the forecast before it is shown.
                self.fc.walk = env("C2_GUI_WALK").and_then(|d| d.parse().ok()).unwrap_or(0);
            }
            Some("explain") => {
                self.tab = Tab::Explain;
                if let Some(a) = env("C2_GUI_TICKER") {
                    self.ex.leader = a.clone();
                    self.ex.leader_input = a;
                }
                if let Some(b) = env("C2_GUI_FOLLOWER") {
                    self.ex.follower = b.clone();
                    self.ex.follower_input = b;
                }
                let num = |k: &str, d: usize| env(k).and_then(|v| v.parse().ok()).unwrap_or(d);
                self.ex.h1 = num("C2_GUI_H1", 1);
                self.ex.h2 = num("C2_GUI_H2", 1);
                self.ex.lag = num("C2_GUI_LAG", 1);
                self.ex.obs = num("C2_GUI_OBS", 0);
                let end = env("C2_GUI_END").and_then(|d| c2_data::parse_date(&d).ok()).unwrap_or(i32::MAX);
                self.ex.pending_end = Some((num("C2_GUI_PERIOD", 0), end));
                self.ex.scroll = env("C2_GUI_SCROLL").and_then(|v| v.parse().ok());
            }
            _ => self.tab = Tab::Status,
        }
    }

    /// Self-check (`C2_GUI_WALK`): once the forecast for the current choices is shown, takes the
    /// next "◀ month" step as that button does; reports how long the steps took in all.
    fn walk_step(&mut self) {
        if self.fc.walk == 0 && self.fc.walk_started.is_none() {
            return;
        }
        let shown = match self.fc.job.get(&self.forecast_key()) {
            None => return,
            Some(res) => res.as_ref().ok().map(|f| f.as_of),
        };
        match shown {
            Some(day) if self.fc.walk > 0 => {
                let steps = self.fc.walk;
                self.fc.walk_started.get_or_insert_with(|| (Instant::now(), steps));
                self.fc.walk -= 1;
                self.fc.as_of = Some(day - 30);
            }
            _ => {
                if let Some((t, steps)) = self.fc.walk_started.take() {
                    let day = shown.map(c2_data::date_string).unwrap_or_else(|| "(an error)".into());
                    eprintln!("c2-gui self-check: {steps} steps of \"◀ month\" took {:.3} s in all, ending as of {day}", t.elapsed().as_secs_f64());
                }
                self.fc.walk = 0;
            }
        }
    }
}

/// `v` if it is one of `options` (sorted), else the next larger one, else the largest (`v` itself
/// when there are none).
fn nearest(options: &[usize], v: usize) -> usize {
    if options.contains(&v) { v } else { options.iter().copied().find(|&x| x >= v).or(options.last().copied()).unwrap_or(v) }
}

/// The move lengths a run computed for correlation periods of `length` days, from its tasks:
/// A's (h1) choices, B's (h2) choices for the chosen A, and the choice nearest to (h1, h2).
fn move_length_choices(tasks: &[grid::Task], length: usize, h1: usize, h2: usize) -> (Vec<usize>, Vec<usize>, usize, usize) {
    let mut pairs: Vec<(usize, usize)> = tasks.iter().filter(|t| t.length == length).map(|t| (t.h1, t.h2)).collect();
    pairs.sort_unstable();
    pairs.dedup();
    let mut h1s: Vec<usize> = pairs.iter().map(|p| p.0).collect();
    h1s.dedup();
    let h1 = nearest(&h1s, h1);
    let h2s: Vec<usize> = pairs.iter().filter(|p| p.0 == h1).map(|p| p.1).collect();
    let h2 = nearest(&h2s, h2);
    (h1s, h2s, h1, h2)
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, std::env::var("C2_GUI_HEIGHT").ok().and_then(|h| h.parse().ok()).unwrap_or(780.0)])
            .with_title("correlation2"),
        ..Default::default()
    };
    eframe::run_native("correlation2", options, Box::new(|_cc| Ok(Box::new(App::new()))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait<K: Clone + PartialEq, T: Send + 'static>(job: &mut Job<K, T>) {
        let t = Instant::now();
        while !job.poll() {
            assert!(t.elapsed() < Duration::from_secs(10), "job never finished");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn job_reports_a_panic_as_an_error() {
        let mut job: Job<String, u32> = Job::default();
        job.start("run".into(), || panic!("index out of bounds: the len is 148 but the index is 148"));
        wait(&mut job);
        let err = job.get(&"run".to_string()).unwrap().as_ref().unwrap_err();
        assert!(err.starts_with(STOPPED) && err.contains("the index is 148"), "{err}");
        assert!(!job.is_running());
    }

    #[test]
    fn job_reports_a_vanished_thread_as_an_error() {
        let mut job: Job<String, u32> = Job::default();
        let (tx, rx) = channel();
        drop(tx);
        job.running = Some(("run".into(), rx));
        assert!(job.poll());
        assert!(job.get(&"run".to_string()).unwrap().as_ref().unwrap_err().starts_with(STOPPED));
        assert!(!job.is_running());
    }

    #[test]
    fn job_keeps_an_empty_result_and_waits_for_the_one_in_flight() {
        let mut job: Job<String, Vec<u32>> = Job::default();
        let key = "main".to_string();
        assert!(job.wants(&key));
        job.start(key.clone(), || Ok(Vec::new()));
        // While it runs, no other job starts, for these inputs or others.
        assert!(!job.wants(&key) && !job.wants(&"other".to_string()));
        wait(&mut job);
        assert!(job.get(&key).unwrap().as_ref().unwrap().is_empty());
        // An empty result is a result: nothing restarts it.
        assert!(!job.wants(&key));
        assert!(job.get(&"other".to_string()).is_none() && job.wants(&"other".to_string()));
        let g = job.generation;
        job.clear();
        assert!(job.generation > g && job.wants(&key));
    }

    #[test]
    fn track_record_handles_negative_skill() {
        // The review's case: reliably wrong over 200 forecasts.
        let (c, text) = track_record_verdict(0.35, 200, -0.25, 200);
        assert_eq!(c, RED_NOTE);
        assert!(text.contains("Both are worse than luck") && text.contains("right only 35%"), "{text}");
        // One significantly negative measure is enough.
        assert_eq!(track_record_verdict(0.5, 225, -0.14, 225).0, RED_NOTE);
        let (c, text) = track_record_verdict(0.60, 200, 0.25, 200);
        assert_eq!(c, GREEN_NOTE);
        assert!(text.contains("Both beat luck") && text.contains("holdout"), "{text}");
        assert!(track_record_verdict(0.52, 200, 0.05, 200).1.contains("within what luck produces"));
        // Mixed: direction right more often than luck, sizes moving the opposite way.
        let (c, text) = track_record_verdict(0.60, 200, -0.20, 200);
        assert_eq!(c, YELLOW_NOTE);
        assert!(text.contains("disagree"), "{text}");
    }

    #[test]
    fn track_record_names_the_measure_that_fired() {
        // Like ADM: only the correlation is beyond luck (wrong way); the direction was right 55%
        // of the time, so the text must not say it pointed the wrong way.
        let (c, text) = track_record_verdict(0.55, 200, -0.17, 200);
        assert_eq!(c, RED_NOTE);
        assert!(text.contains("predicted sizes went the wrong way (correlation -0.17, beyond luck)") && text.contains("right 55% of the time"), "{text}");
        assert!(!text.contains("wrong more often"), "{text}");
        // Like LAMR: only the correlation is beyond luck (right way), at 45% right.
        let (c, text) = track_record_verdict(0.45, 200, 0.17, 200);
        assert_eq!(c, GREEN_NOTE);
        assert!(text.contains("predicted sizes lined up") && text.contains("right only 45% of the time, within luck"), "{text}");
        // Only the direction, wrong way.
        let (c, text) = track_record_verdict(0.40, 200, 0.05, 200);
        assert_eq!(c, RED_NOTE);
        assert!(text.contains("direction was wrong more often than luck explains (right only 40%"), "{text}");
    }

    #[test]
    fn track_record_hit_rate_counts_only_forecasts_with_a_direction() {
        // 62% right is beyond luck over 100 forecasts with a direction (z 2.4), not over the 20
        // that had one out of 100 made (z 1.07); the correlation is over all 100.
        assert_eq!(track_record_verdict(0.62, 100, 0.0, 100).0, GREEN_NOTE);
        assert_eq!(track_record_verdict(0.62, 20, 0.0, 100).0, YELLOW_NOTE);
    }

    #[test]
    fn a_refused_start_shows_what_the_process_said() {
        let dir = std::env::temp_dir().join(format!("c2-gui-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("run.out");
        // The running census's own output stays, and isn't shown as the new process's.
        std::fs::write(&out, "[10:00:00] 5/888 windows (1.00% of work)\n").unwrap();
        let mut cmd = std::process::Command::new("sh");
        cmd.args(["-c", "echo 'Error: census x is already running (pid 4242)' >&2; exit 1"]);
        let (mut child, from) = spawn_logged(cmd, &out).unwrap();
        let status = child.wait();
        let (error, text) = launch_outcome(&Launch { run: "x".into(), child, out: out.clone(), from }, status);
        assert!(error && text.contains("Census 'x' exited with an error"), "{text}");
        assert!(text.contains("already running (pid 4242)") && !text.contains("windows"), "{text}");
        assert!(std::fs::read_to_string(&out).unwrap().starts_with("[10:00:00] 5/888"));
        // A clean exit shows its last line.
        let mut cmd = std::process::Command::new("sh");
        cmd.args(["-c", "echo '[10:01:00] plan: ...'; echo '[10:01:01] census already complete'"]);
        let (mut child, from) = spawn_logged(cmd, &out).unwrap();
        let status = child.wait();
        let (error, text) = launch_outcome(&Launch { run: "x".into(), child, out: out.clone(), from }, status);
        assert!(!error && text == "Census 'x': [10:01:01] census already complete", "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn explainer_offers_only_the_move_lengths_the_run_computed() {
        let task = |id, h1, h2, length| grid::Task { id, h1, h2, length, windows: Vec::new() };
        // An older run's pairing (B's move no longer than A's), 756-day periods only.
        let old: Vec<grid::Task> = [(1, 1), (5, 1), (5, 5), (21, 1), (21, 5), (21, 21)].iter().enumerate().map(|(k, &(a, b))| task(k, a, b, 756)).collect();
        assert_eq!(move_length_choices(&old, 756, 5, 21), (vec![1, 5, 21], vec![1, 5], 5, 5));
        assert_eq!(move_length_choices(&old, 756, 1, 5), (vec![1, 5, 21], vec![1], 1, 1));
        // A move length the run doesn't have moves to the next one it has.
        assert_eq!(move_length_choices(&old, 756, 3, 1), (vec![1, 5, 21], vec![1, 5], 5, 1));
        // No tasks for that period length: nothing to offer.
        assert_eq!(move_length_choices(&old, 252, 5, 5), (vec![], vec![], 5, 5));
    }

    #[test]
    fn significance_verdict_follows_the_census_flag() {
        // SO->COLB (1d/1d, offset 13, 252 days to 2020-04-08): exact r -0.437553, GPU estimate
        // -0.4389, cut-off 0.4385, flagged significant (step 4 says "passed").
        let s = significance_verdict(0.4385, -0.437553, Some((-0.43888, true)), true);
        assert!(s.contains("so this pair passed"), "{s}");
        assert!(!s.contains("not pass"), "{s}");
        assert!(s.contains("GPU estimate") && s.contains("0.4376") && s.contains("0.4389"), "{s}");
        assert!(s.contains("8-bit"), "{s}");
        // Stored, not significant, on the same side of the cut-off both ways: no caveat.
        let s = significance_verdict(0.4385, 0.30, Some((0.301, false)), true);
        assert!(s.contains("did not pass") && !s.contains("8-bit"), "{s}");
        // Not stored: judged from the exact r, saying so.
        let s = significance_verdict(0.4385, 0.43, None, true);
        assert!(s.contains("exact r") && s.contains("does not pass") && s.contains("GPU estimate"), "{s}");
        // Never tested.
        assert!(significance_verdict(0.4385, 0.5, None, false).contains("never tested"));
    }

    fn row(partner: &str, window_id: u32, period_days: u32, period_end: &str, z: f32) -> ProfileRow {
        ProfileRow {
            window: format!("2000-01-01..{period_end}"),
            period_days,
            period_end: period_end.into(),
            window_id,
            partner: partner.into(),
            leads: false,
            h1: 1,
            h2: 1,
            lag: 1,
            obs: 100,
            r: z / 10.0,
            z,
            significant: false,
            chance: 0.0,
            chance_z: 3,
            move_per_pct: None,
            words: None,
        }
    }

    #[test]
    fn timeline_most_recent_means_the_latest_rolling_period() {
        // Window 284 is the full history (it always ends last); 280-283 are rolling periods
        // ending 2025-04-14; 100 ended earlier.
        let data = TimelineData {
            rows: vec![
                row("NZF", 284, 5154, "2025-06-27", 12.6),
                row("NVG", 280, 252, "2025-04-14", 5.0),
                row("NAC", 283, 1260, "2025-04-14", 4.0),
                row("NEA", 100, 504, "2019-01-02", 9.0),
            ],
            full_history: HashSet::from([284]),
        };
        let view = |w: &str| timeline_view(&data, &(0, false, false, w.to_string(), None, 40));
        let recent = view("recent");
        assert_eq!(recent.recent_end.as_deref(), Some("2025-04-14"));
        let partners: Vec<&str> = recent.rows.iter().map(|r| r.partner.as_str()).collect();
        assert_eq!(partners, ["NVG", "NAC"]);
        let full = view("full");
        assert_eq!(full.rows.len(), 1);
        assert_eq!(full.rows[0].partner, "NZF");
        assert_eq!(full.full_days, Some(5154));
        // Rolling lengths only; the full history is its own option.
        assert_eq!(view("all").lengths, [252, 504, 1260]);
        assert_eq!(view("504").rows.len(), 1);
    }

    #[test]
    fn timeline_order_is_fixed_for_ties() {
        let rows = vec![row("BBB", 1, 252, "2020-01-02", 4.0), row("AAA", 2, 252, "2020-01-02", 4.0), row("CCC", 3, 252, "2020-01-02", -4.0)];
        let mut reversed = rows.clone();
        reversed.reverse();
        let order = |rows: Vec<ProfileRow>| -> Vec<String> {
            let data = TimelineData { rows, full_history: HashSet::new() };
            timeline_view(&data, &(0, false, false, "all".into(), None, 40)).rows.into_iter().map(|r| r.partner).collect()
        };
        assert_eq!(order(rows), ["AAA", "BBB", "CCC"]);
        assert_eq!(order(reversed), ["AAA", "BBB", "CCC"]);
        // The same relationship in two periods with equal |z|: the later period is kept.
        let data = TimelineData { rows: vec![row("AAA", 1, 252, "2019-01-02", 4.0), row("AAA", 2, 252, "2020-01-02", -4.0)], full_history: HashSet::new() };
        let v = timeline_view(&data, &(0, false, false, "all".into(), None, 40));
        assert_eq!(v.rows.len(), 1);
        assert_eq!(v.rows[0].window_id, 2);
    }
}
