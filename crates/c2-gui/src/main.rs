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
//!   Explainer one relationship recomputed from prices step by step: the daily values checked
//!            against the census's own code, the correlation against the one the census stored
//!   Census   real vs chance by lag for each horizon pair; a ticker's strongest stored pairs
//!   Holdout  the $10,000 test for either signal family, with equity curves
//!   Data     price store summary and the decade quality table
//!
//! Everything read from a census run is computed on a background thread (`Job`) and cached under
//! a key that starts with the run's name, so switching runs on the Status tab never shows or
//! reuses another run's results.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use c2_census::holdout::{Family, Options, Report};
use c2_census::run::{self, FLAG_NULL, Plan};
use c2_data::Store;
use eframe::egui;
use c2_census::analyze::ProfileRow;
use c2_census::analyze::{Counts, DatasetStats, RunInfo, Words};
use c2_census::explain::Explained;
use c2_census::forecast::Forecast;
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
}

impl<K, T> Default for Job<K, T> {
    fn default() -> Self {
        Job { running: None, done: None, generation: 0 }
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

/// The census variable is rebuilt per run: (run name, the config text in its manifest).
type PlanKey = (String, String);

/// The one census variable kept in memory, shared by the Explainer's plan job and the holdout's
/// worker so that only one is built at a time.
type PlanSlot = Arc<Mutex<Option<(PlanKey, Arc<Plan>)>>>;

/// A Strongest-tab row with the move lengths of its strongest relationship (h1, h2).
type RankEntry = (c2_census::analyze::RankRow, Option<(u8, u8)>);

/// (job generation, find text, sort) of the Strongest tab's prepared rows.
type RankViewKey = (u64, String, usize);

/// One prepared row of the Strongest tab: the stock and its other cells, already formatted.
struct RankLine {
    ticker: String,
    cells: [String; 8],
}

/// Self-check mode (`C2_GUI_SCREENSHOT=<path prefix>`, `C2_GUI_TAB=<tab>`): open a tab, wait for
/// its data (or its error), save the window's pixels and exit. After `C2_GUI_TIMEOUT` seconds
/// (default 600) it saves whatever is shown.
struct Shot {
    prefix: String,
    frames: u32,
    requested: bool,
    started: Instant,
    timeout: Duration,
}

/// Run name typing settles after this long without a keystroke (or on Enter).
const RUN_IDLE: Duration = Duration::from_millis(1200);

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
    status: Option<serde_json::Value>,
    status_err: String,
    pid: Option<u32>,
    control: String,
    last_poll: Instant,
    message: String,
    // Census tab: the lag table and the run it was read from.
    lag_table: BTreeMap<(usize, usize), BTreeMap<usize, LagRow>>,
    lag_table_run: String,
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
    holdout: Job<HoldoutKey, Report>,
    holdout_started: Option<Instant>,
    shot: Option<Shot>,
    // Stock timeline tab.
    tl: Timeline,
    // Forecast tab.
    fc: ForecastView,
    // Explainer tab, and the census variable it (and the holdout) needs. Key: run.
    ex: ExplainView,
    plan: Job<String, Arc<Plan>>,
    // Status tab statistics: of the run (key: run), and every run (key: refresh count).
    stats: Job<String, DatasetStats>,
    runs: Job<u64, Vec<RunInfo>>,
    runs_refresh: u64,
    /// Local time offset from UTC in seconds (from `date +%z`).
    utc_offset: i64,
    // Strongest tab. Key: run.
    rank: Job<String, Vec<RankEntry>>,
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
    job: Job<ForecastKey, Forecast>,
    /// A mistyped as-of date.
    input_error: String,
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

fn running_pid(dir: &Path) -> Option<u32> {
    let pid: u32 = std::fs::read_to_string(dir.join("pid")).ok()?.trim().parse().ok()?;
    let exe = std::fs::read_link(format!("/proc/{pid}/exe")).ok()?;
    exe.file_name()?.to_str()?.starts_with("c2").then_some(pid)
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

/// A run's manifest: its config text and the number of tasks it was planned with.
fn read_manifest(run: &str, dir: &Path) -> Result<(String, Option<u64>), String> {
    let text = std::fs::read_to_string(dir.join("manifest.json")).map_err(|e| format!("census '{run}' has no manifest ({e}); check the run name on the Status tab"))?;
    let m: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("census '{run}': unreadable manifest.json ({e})"))?;
    let config = m["config"].as_str().ok_or_else(|| format!("census '{run}': manifest.json has no config"))?;
    Ok((config.to_string(), m["tasks"].as_u64()))
}

/// The census configuration for a run's manifest config text. Until runs can be read back
/// exactly (round 2's RunMeta), the text only decides between the full and the smoke grid;
/// `check_periods` and `check_tasks` refuse a run the guess doesn't fit.
fn config_for(text: &str) -> c2_census::grid::Config {
    if text.contains("full_history=true") { c2_census::grid::Config::full() } else { c2_census::grid::Config::smoke() }
}

/// The census variable for `run`, from the shared slot when it holds this run's (with the same
/// manifest config), else built now in its place. Checked against the run's correlation periods
/// before it is returned.
fn plan_for_run(slot: &Mutex<Option<(PlanKey, Arc<Plan>)>>, store: &Store, run: &str, dir: &Path) -> Result<Arc<Plan>, String> {
    let (config, _) = read_manifest(run, dir)?;
    let key: PlanKey = (run.to_string(), config);
    let plan = {
        // The slot only ever holds nothing or a complete plan (it is emptied before a build and
        // filled after it succeeds), so a lock poisoned by a panicking worker is still usable.
        let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some((k, p)) if *k == key => p.clone(),
            _ => {
                // Free the other run's variable before building this one.
                *guard = None;
                let p = Arc::new(Plan::new(store, config_for(&key.1)).map_err(|e| format!("{e:#}"))?);
                *guard = Some((key, p.clone()));
                p
            }
        }
    };
    check_periods(&plan, run, dir)?;
    Ok(plan)
}

/// One row of a run's `windows.csv`: correlation period id, length (0 = full history), first
/// and last day.
fn run_periods(run: &str, dir: &Path) -> Result<Vec<(usize, usize, String, String)>, String> {
    let text = std::fs::read_to_string(dir.join("windows.csv")).map_err(|e| format!("census '{run}' has no list of correlation periods yet ({e})"))?;
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let num = |i: usize| f.get(i).and_then(|v| v.parse::<usize>().ok());
            match (num(0), num(1), f.get(2), f.get(3)) {
                (Some(id), Some(len), Some(a), Some(b)) => Ok((id, len, a.to_string(), b.to_string())),
                _ => Err(format!("census '{run}': unreadable line in windows.csv: {l}")),
            }
        })
        .collect()
}

/// The run's correlation periods must be exactly the plan's, id by id and date by date, or the
/// plan's period ids would point at other periods than the ones the run's records were filed under.
fn check_periods(plan: &Plan, run: &str, dir: &Path) -> Result<(), String> {
    let periods = run_periods(run, dir)?;
    let day = |row: usize| c2_data::date_string(plan.var.days[row]);
    let rebuilt = |w: &grid::Window| (w.id, w.length, day(w.start), day(w.end - 1));
    if periods.len() != plan.windows.len() {
        return Err(format!(
            "census '{run}' has {} correlation periods, but rebuilt from today's price data with its settings there are {}; this tab can't use that run yet",
            periods.len(),
            plan.windows.len()
        ));
    }
    for (p, w) in periods.iter().zip(&plan.windows) {
        if *p != rebuilt(w) {
            let r = rebuilt(w);
            return Err(format!(
                "census '{run}' doesn't match today's price data: its correlation period {} is {} days, {} to {}, but rebuilt today it is {} days, {} to {}; this tab can't use that run yet",
                p.0, p.1, p.2, p.3, r.1, r.2, r.3
            ));
        }
    }
    Ok(())
}

/// The holdout reads `tasks/t{id}.rec` by the plan's task numbering, so the run must have been
/// planned with the same move lengths, correlation period lengths and number of tasks.
fn check_tasks(plan: &Plan, run: &str, dir: &Path) -> Result<(), String> {
    let (config, tasks) = read_manifest(run, dir)?;
    let same_grid = config.contains(&format!("horizons={:?} ", plan.cfg.horizons)) && config.contains(&format!("lengths={:?} ", plan.cfg.lengths));
    if !same_grid || tasks != Some(plan.tasks.len() as u64) {
        return Err(format!(
            "census '{run}' was planned with a different set of move lengths or tasks ({} tasks) than this version numbers them ({} tasks: move lengths {:?}, correlation periods {:?}), so the holdout can't read it yet",
            tasks.map_or("an unknown number of".to_string(), |t| t.to_string()),
            plan.tasks.len(),
            plan.cfg.horizons,
            plan.cfg.lengths
        ));
    }
    Ok(())
}

/// Move lengths (h1, h2) of each row's strongest relationship, found by its stored record
/// (leader, follower, offset, r and z identify it). `RankRow` doesn't carry them yet.
fn rank_move_lengths(store: &Store, dir: &Path, rows: &[c2_census::analyze::RankRow]) -> Result<Vec<Option<(u8, u8)>>, String> {
    let mut wanted: HashMap<(u32, u32, u16, u32, u32), Vec<usize>> = HashMap::new();
    for (k, r) in rows.iter().enumerate() {
        let (Some(me), Some(other)) = (store.id(&r.ticker), store.id(&r.best_partner)) else { continue };
        let (leader, target) = if r.best_leads { (me, other) } else { (other, me) };
        wanted.entry((leader as u32, target as u32, r.best_lag, r.best_r.to_bits(), r.best_z.to_bits())).or_default().push(k);
    }
    let mut out = vec![None; rows.len()];
    let entries = std::fs::read_dir(dir.join("tasks")).map_err(|e| format!("{}: {e}", dir.display()))?;
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("rec") {
            continue;
        }
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        for chunk in bytes.chunks_exact(run::RECORD_BYTES) {
            let r = run::Record::from_bytes(chunk);
            if r.flags & FLAG_NULL != 0 || r.lag == 0 {
                continue;
            }
            if let Some(ks) = wanted.get(&(r.leader, r.target, r.lag, r.r.to_bits(), r.z.abs().to_bits())) {
                for &k in ks {
                    out[k] = Some((r.h1, r.h2));
                }
            }
        }
    }
    Ok(out)
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

/// Window ids of a run's full-history correlation period (length 0 in `windows.csv`).
fn full_history_ids(run: &str, dir: &Path) -> Result<HashSet<u32>, String> {
    Ok(run_periods(run, dir)?.into_iter().filter(|p| p.1 == 0).map(|p| p.0 as u32).collect())
}

/// How a forecast's track record compares with luck, from its hit rate and its predicted vs
/// actual correlation over `n` forecasts: (colour, sentence). Skill can be negative: a record
/// that points the wrong way more often than luck explains is worse than chance, not "good".
fn track_record_verdict(hit_rate: f64, corr: f64, n: usize) -> (egui::Color32, &'static str) {
    let n = n.max(1) as f64;
    let z_hit = (hit_rate - 0.5) / (0.25 / n).sqrt();
    let z_corr = corr * n.sqrt();
    let better = z_hit >= 2.0 || z_corr >= 2.0;
    let worse = z_hit <= -2.0 || z_corr <= -2.0;
    match (better, worse) {
        (false, false) => (YELLOW_NOTE, "That is within what luck produces, so treat the forecast line as noise."),
        (true, false) => (
            GREEN_NOTE,
            "That is better than luck usually produces for one stock, but with thousands of stocks some look this good by chance; check the holdout before trusting it.",
        ),
        (false, true) => (
            RED_NOTE,
            "That is worse than chance: these forecasts pointed the wrong way more than luck explains. Don't use this forecast.",
        ),
        (true, true) => (
            YELLOW_NOTE,
            "The two measures disagree (one better than luck, the other worse), so the record doesn't support the forecast; treat the forecast line as noise.",
        ),
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
    let line = |c: &Counts, intro: String| {
        format!(
            "{intro}: after {a} rose {up:.1}% or more over {} ({} times), {b} rose over the {} ending {} later in {}. \
             After {a} fell {down:.1}% or more ({} times), {b} rose in {}. Overall {b} rose in {}.",
            days_label(h1), c.a_up, days_label(h2), tdays(lag), share(c.a_up_b_up, c.a_up), c.a_down, share(c.a_down_b_up, c.a_down), share(c.b_up, c.cases)
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

fn day_cells(s: &c2_census::explain::DaySteps, market: f64, d: usize) -> [String; 8] {
    let f = |v: f64, text: String| if v.is_finite() { text } else { "none".to_string() };
    let same = s.v[d] == s.v_census[d] || (s.v[d].is_nan() && s.v_census[d].is_nan());
    [
        format!("${:.2} -> ${:.2}", s.close[d - 1], s.close[d]),
        f(s.ret[d] as f64, format!("{:+.3}%", 100.0 * s.ret[d])),
        f(market, format!("{:+.3}%", 100.0 * market)),
        f(s.beta[d], format!("{:.3}", s.beta[d])),
        f(s.resid[d], format!("{:+.3}%", 100.0 * s.resid[d])),
        f(s.sd[d], format!("{:+.3}% / {:.3}%", 100.0 * s.mean[d], 100.0 * s.sd[d])),
        f(s.v[d] as f64, format!("{:+.3}", s.v[d])),
        format!("{}{}", if s.v_census[d].is_finite() { format!("{:+.3}", s.v_census[d]) } else { "none".into() }, if same { "  (same)" } else { "  (DIFFERENT)" }),
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
        let dir = c2_data::data_dir().join("store");
        let (store, store_note, decades) = match Store::open(&dir) {
            Ok(s) => {
                let d = c2_data::quality::decade_table(&s);
                let note = format!("{} symbols, {} daily rows, in {}", s.symbols.len(), s.rows(), dir.display());
                (Some(Arc::new(s)), note, d)
            }
            Err(e) => (None, format!("No price store yet ({e:#}). Run `c2 ingest`."), Vec::new()),
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
            status: None,
            status_err: String::new(),
            pid: None,
            control: "run".into(),
            last_poll: Instant::now() - Duration::from_secs(10),
            message: String::new(),
            lag_table: BTreeMap::new(),
            lag_table_run: String::new(),
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
        self.rank.clear();
        self.rank_view = None;
        self.tl.job.clear();
        self.tl.view = None;
        self.tl.selected = None;
        self.fc.job.clear();
        self.plan.clear();
        // Free the old run's census variable now unless a worker is using the slot.
        if let Ok(mut guard) = self.plan_slot.try_lock() {
            *guard = None;
        }
        self.ex.job.clear();
        self.ex.key = None;
        self.ex.pending_window = None;
        if let Some(sel) = self.ex.selected.take() {
            self.ex.pending_end = Some(sel);
        }
        self.stats.clear();
        self.pairs.clear();
        self.lag_table.clear();
        self.lag_table_run.clear();
        self.holdout.clear();
        self.holdout_started = None;
        self.message.clear();
        // Read the new run's status now, so the Start / Stop buttons reflect it this frame.
        self.last_poll = Instant::now() - Duration::from_secs(10);
        self.poll();
    }

    /// Collects every finished background job, whichever tab is shown.
    fn poll_jobs(&mut self) {
        self.rank.poll();
        self.tl.job.poll();
        // The as-of box shows the day of the forecast for the current choices (not an older one
        // that finished while those were being computed).
        if self.fc.job.poll()
            && let Some((k, Ok(f))) = &self.fc.job.done
            && *k == self.forecast_key()
        {
            self.fc.as_of_text = c2_data::date_string(f.as_of);
        }
        self.plan.poll();
        if self.ex.job.poll()
            && let Some((_, Ok(e))) = &self.ex.job.done
        {
            self.ex.obs = self.ex.obs.min(e.obs.len().saturating_sub(1));
        }
        self.stats.poll();
        self.runs.poll();
        self.pairs.poll();
        self.holdout.poll();
    }

    /// Whether the shown tab has everything it asked for (a result or an error), for self-checks.
    fn tab_ready(&self) -> bool {
        let run = &self.run_name;
        let has_store = self.store.is_some();
        match self.tab {
            Tab::Status => self.runs.get(&self.runs_refresh).is_some() && (!has_store || self.stats.get(run).is_some()),
            Tab::Strongest => !has_store || (self.rank.get(run).is_some() && self.stats.get(run).is_some()),
            Tab::Stock => !has_store || self.tl.want.as_ref().is_none_or(|t| self.tl.job.get(&(run.clone(), t.clone())).is_some()),
            Tab::Forecast => !has_store || self.fc.ticker.is_empty() || self.fc.job.get(&self.forecast_key()).is_some(),
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
                let lengths = rank_move_lengths(&store, &dir, &rows)?;
                Ok(rows.into_iter().zip(lengths).collect())
            });
        }
        let rank = match self.rank.get(&run) {
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
            Some(Ok(rank)) => rank,
        };
        if rank.is_empty() {
            ui.label(format!("Census '{run}' has no stored lead-lag relationships yet."));
            return;
        }
        let with_sig = rank.iter().filter(|r| r.0.significant > 0).count();
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
            Some(Ok(st)) => format!("when {} lead-lag correlations are tested, as in census '{run}'", big(st.lagged.correlations)),
            _ => "when every pair of stocks is tested at every offset, move length and correlation period".to_string(),
        };
        ui.label(format!(
            "Stocks with both many stored lead-lag correlations and strong ones rank highest (score = sum of |r|). Most stored correlations are within \
             what chance produces {tested}; {with_sig} stocks have relationships that passed the significance test ('sig.'), and \
             'periods' counts how many separate correlation periods those come from (1 usually means one event, such as the 2020 crash). Click a stock for its timeline."
        ));
        // Filtered, sorted and formatted only when the result, the find text or the sort changes.
        let view_key: RankViewKey = (self.rank.generation, self.rank_filter.trim().to_uppercase(), self.rank_sort);
        if self.rank_view.as_ref().is_none_or(|(k, _)| *k != view_key) {
            let (filter, sort) = (&view_key.1, view_key.2);
            let mut rows: Vec<&RankEntry> = rank.iter().filter(|r| filter.is_empty() || r.0.ticker.contains(filter.as_str())).filter(|r| sort != 1 || r.0.significant > 0).collect();
            let score = |r: &c2_census::analyze::RankRow| match sort {
                1 => r.sig_sum_abs_r,
                2 => r.max_abs_r as f64,
                3 => r.stored as f64,
                _ => r.sum_abs_r,
            };
            rows.sort_by(|a, b| score(&b.0).total_cmp(&score(&a.0)).then_with(|| a.0.ticker.cmp(&b.0.ticker)));
            let lines = rows
                .iter()
                .take(1000)
                .map(|(r, lengths)| {
                    let (a, b) = if r.best_leads { (&r.ticker, &r.best_partner) } else { (&r.best_partner, &r.ticker) };
                    let rel = match lengths {
                        Some((h1, h2)) => format!(
                            "{a} moves over {}, {b} follows over {} ending {} later, r {:+.3}",
                            days_label(*h1 as usize),
                            days_label(*h2 as usize),
                            days_label(r.best_lag as usize),
                            r.best_r
                        ),
                        None => format!("{a} moves, {b} follows {} later, r {:+.3}", days_label(r.best_lag as usize), r.best_r),
                    };
                    RankLine {
                        ticker: r.ticker.clone(),
                        cells: [
                            format!("{:.0}", if sort == 1 { r.sig_sum_abs_r } else { r.sum_abs_r }),
                            r.stored.to_string(),
                            format!("{:.3}", r.mean_abs_r),
                            format!("{:.3}", r.max_abs_r),
                            if r.significant > 0 { format!("{} ({}↗ {}↙)", r.significant, r.leads, r.follows) } else { "0".into() },
                            if r.significant > 0 { r.windows.to_string() } else { String::new() },
                            rel,
                            format!("{:.1}", r.best_z),
                        ],
                    }
                })
                .collect();
            self.rank_view = Some((view_key, lines));
        }
        let mut open: Option<String> = None;
        if let Some((_, lines)) = &self.rank_view {
            egui::ScrollArea::vertical().id_salt("rank").show(ui, |ui| {
                egui::Grid::new("rank_grid").striped(true).show(ui, |ui| {
                    for h in ["#", "stock", "score", "relationships", "mean |r|", "max |r|", "sig.", "periods", "strongest relationship", "z"] {
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
            let dir = self.dir();
            let (r, t) = (run.clone(), want.clone());
            self.tl.job.start(job_key.clone(), move || {
                let rows = c2_census::analyze::stock_profile(&store, &dir, &t, false).map_err(|e| format!("{e:#}"))?;
                Ok(TimelineData { rows, full_history: full_history_ids(&r, &dir)? })
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
        let all_rows = match self.tl.job.get(&job_key) {
            Some(Ok(data)) => &data.rows,
            _ => return,
        };
        let selected = self.tl.selected.clone();
        let stock_first = self.tl.stock_first;
        let show_r = self.tl.show_r;
        let mut explain: Option<ProfileRow> = None;
        egui::ScrollArea::vertical().id_salt("tl_scroll").show(ui, |ui| {
            ui.label(if stock_first {
                format!("A = {t}, which moves first; B = the stock that follows. A point at +k days: when {t} moves, that stock tends to move k trading days \
                         later (offset from the end of {t}'s move to the end of B's); height = B's typical move beyond the market per 1% move of {t}. \
                         Orange = passed the significance test; grey = within what chance produces.")
            } else {
                format!("B = {t}, which follows; A = the stock that moves first. A point at +k days: when that stock moves, {t} tends to move k trading days \
                         later (offset from the end of A's move to the end of {t}'s); height = {t}'s typical move beyond the market per 1% move of A \
                         (these feed the Forecast tab). Orange = passed the significance test; grey = within what chance produces.")
            });
            let y_of = |r: &ProfileRow| if show_r { r.r as f64 } else { r.move_per_pct.unwrap_or(r.r as f64) };
            let x_of = |r: &ProfileRow| r.lag as f64;
            Plot::new("timeline")
                .height(280.0)
                .x_axis_label("offset: trading days later")
                .y_axis_label(if show_r { "r".to_string() } else { "% move per 1%".to_string() })
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
            egui::ScrollArea::vertical().id_salt("tl_rows").max_height(260.0).show(ui, |ui| {
                egui::Grid::new("tl_grid").striped(true).show(ui, |ui| {
                    for h in ["", "A -> B", "offset", "move lengths", "B per 1% of A", "r", "z", "chance", "in your words (in the correlation period)", "period", "ending"] {
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
                    // Asking again for the forecast shown recomputes it.
                    let key = self.forecast_key();
                    if self.fc.job.get(&key).is_some() {
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
            if self.fc.job.is_running() {
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
        let key = self.forecast_key();
        if !self.fc.ticker.is_empty() && self.fc.job.wants(&key) {
            let dir = self.dir();
            let (_, ticker, as_of, horizon, significant_only, age) = key.clone();
            self.fc.job.start(key.clone(), move || {
                let opt = c2_census::forecast::Options { significant_only, horizon, max_age: (age > 0).then_some(age * 252) };
                c2_census::forecast::forecast(&store, &dir, &ticker, as_of.unwrap_or(i32::MAX), &opt).map_err(|e| format!("{e:#}"))
            });
        }
        if !self.fc.input_error.is_empty() {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.fc.input_error);
        }
        // The forecast for the current choices, else (while that one is computed) the last one
        // of this run, which names its own stock and day.
        let f = match &self.fc.job.done {
            Some((k, Err(e))) if *k == key => {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
                return;
            }
            Some((k, Ok(f))) if k.0 == self.run_name => f,
            _ => {
                if self.fc.ticker.is_empty() {
                    ui.label("Enter a ticker.");
                } else {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(format!("Reading census '{}' for {} (about a second)…", self.run_name, self.fc.ticker));
                    });
                }
                return;
            }
        };
        let t = &f.ticker;
        let h = f.predicted_pct.len();
        let amount: f64 = self.fc.amount.trim().replace([',', '$'], "").parse().unwrap_or(10_000.0);
        let base = if self.fc.show_amount { amount } else { f.last_price };
        let val = |pct: f64| base * (1.0 + pct / 100.0);
        let day = c2_data::date_string(f.as_of);
        let end_pct = f.predicted_pct[h - 1];
        let typical = 100.0 * f.daily_sd * (h as f64).sqrt();
        let what = if self.fc.show_amount { format!("${amount:.0} in {t}") } else { format!("{t}") };
        ui.label(format!(
            "{what}: ${base:.2} on {day}. Forecast {h} trading days later: ${:.2} ({end_pct:+.2}%). It is the average of what {} relationships imply \
             (another stock moved first and {t} tends to follow some days later; each leader's actual move so far is scaled by the relationship). \
             For scale, {t}'s typical {h}-day move is ±{typical:.1}%.",
            val(end_pct),
            f.relationships
        ));
        if let Some(Some(a)) = f.actual_pct.last() {
            ui.label(format!("What happened: ${:.2} ({a:+.2}%) after {h} trading days.", val(*a)));
        } else if f.actual_pct.iter().any(|a| a.is_some()) {
            ui.label("What happened so far is drawn in blue; the data ends before the full horizon.");
        }
        match &f.skill {
            Some(s) => {
                let (colour, verdict) = track_record_verdict(s.hit_rate, s.corr, s.forecasts);
                let text = format!(
                    "Track record for {t}: the same forecast made every 21 trading days in the past, each using only relationships known at the time: {} forecasts \
                     {}-day ahead, direction right {:.0}% of the time (a coin flip gets 50%), predicted vs actual correlation {:+.2}; the forecasts averaged {:.2}% \
                     while {t} actually moved {:.2}% on average. {verdict}",
                    s.forecasts, s.horizon, 100.0 * s.hit_rate, s.corr, s.mean_abs_predicted_pct, s.mean_abs_actual_pct,
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
        Plot::new("forecast_plot")
            .height(300.0)
            .x_axis_label(format!("trading days after {day}"))
            .y_axis_label(if self.fc.show_amount { "value ($)" } else { "share price ($)" })
            .legend(Legend::default())
            .show(ui, |p| {
                p.hline(HLine::new("", base).color(egui::Color32::DARK_GRAY));
                p.line(Line::new("typical range (±1 sd of past moves)", PlotPoints::from(band(1.0))).color(grey).style(LineStyle::dashed_loose()));
                p.line(Line::new("typical range (±1 sd of past moves)", PlotPoints::from(band(-1.0))).color(grey).style(LineStyle::dashed_loose()));
                if actual.len() > 1 {
                    p.line(Line::new("actual", PlotPoints::from(actual)).color(blue).width(2.0));
                }
                p.line(Line::new("forecast", PlotPoints::from(pred)).color(orange).width(2.5));
            });
        ui.label(format!("When the predicted moves land: the forecast's change on each day ahead (in %), from the relationships whose delay reaches that day."));
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
        Plot::new("forecast_days").height(110.0).x_axis_label("trading days ahead").y_axis_label("% that day").show(ui, |p| {
            p.bar_chart(BarChart::new("daily forecast", bars));
        });
        ui.strong(format!("Largest shares of the forecast at day {h} ({} relationships say something about these days)", f.contributions.len()));
        let mut open: Option<(String, String, usize, usize, usize, usize)> = None;
        egui::ScrollArea::vertical().id_salt("fc_rows").show(ui, |ui| {
            egui::Grid::new("fc_grid").striped(true).show(ui, |ui| {
                for c in ["leader", "offset", "move lengths", "r", "", "leader's move", "alone implies", "share of forecast", "correlation period", ""] {
                    ui.strong(c);
                }
                ui.end_row();
                for c in f.contributions.iter().take(200) {
                    ui.label(&c.leader);
                    ui.label(format!("{} day{}", c.lag, if c.lag == 1 { "" } else { "s" }));
                    ui.label(format!("{} -> {}", days_label(c.h1), days_label(c.h2)));
                    ui.label(format!("{:+.3}", c.r));
                    ui.label(if c.significant { "★" } else { "" });
                    ui.label(format!("{:+.2}%", c.leader_move_pct));
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

    /// The census variable for every stock of the current run, built in the background (in the
    /// slot it shares with the holdout) and checked against the run's correlation periods.
    /// Ok(None) while it is being built.
    fn need_plan(&mut self) -> Result<Option<Arc<Plan>>, String> {
        let Some(store) = self.store.clone() else { return Err(self.store_note.clone()) };
        let run = self.run_name.clone();
        if self.plan.wants(&run) {
            let (slot, dir, r) = (self.plan_slot.clone(), self.dir(), run.clone());
            self.plan.start(run.clone(), move || plan_for_run(&slot, &store, &r, &dir));
        }
        match self.plan.get(&run) {
            None => Ok(None),
            Some(Ok(p)) => Ok(Some(p.clone())),
            Some(Err(e)) => Err(e.clone()),
        }
    }

    fn explain_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("How one correlation is calculated");
        let plan = match self.need_plan() {
            Ok(Some(plan)) => plan,
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
        // A period id from a row of this run (the plan's periods were checked to be the run's).
        if let Some((run, w)) = self.ex.pending_window.take()
            && run == self.run_name
            && let Some(win) = plan.windows.get(w)
        {
            self.ex.length = win.length;
            self.ex.pos = plan.windows.iter().filter(|x| x.length == win.length && x.id < w).count();
        }
        if let Some((len, end)) = self.ex.pending_end.take()
            && let Some(w) = c2_census::explain::find_window(&plan, len, end)
        {
            self.ex.length = len;
            self.ex.pos = plan.windows.iter().filter(|x| x.length == len && x.id < w).count();
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
        for h in [&mut self.ex.h1, &mut self.ex.h2] {
            if !plan.cfg.horizons.contains(h) {
                *h = plan.cfg.horizons.iter().copied().find(|&x| x >= *h).or(plan.cfg.horizons.last().copied()).unwrap_or(1);
            }
        }
        let full_days = plan.windows.iter().find(|w| w.length == 0).map(|w| w.end - w.start).unwrap_or(0);
        let len_label = move |l: usize| if l == 0 { format!("{full_days} days (all history)") } else { format!("{l} days") };
        ui.label(
            "Pick two stocks, the length of move to measure for each, the offset between them and a correlation period. Every step below is \
             recomputed from prices. The daily census values are checked against the census's own code (the census doesn't store daily values); \
             the correlation in step 4 is checked against the one the census stored.",
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
        let horizons = plan.cfg.horizons.clone();
        ui.horizontal_wrapped(|ui| {
            ui.label("A's move over");
            egui::ComboBox::from_id_salt("ex_h1").selected_text(days_label(self.ex.h1)).show_ui(ui, |ui| {
                for &h in &horizons {
                    ui.selectable_value(&mut self.ex.h1, h, days_label(h));
                }
            });
            ui.label("B's move over");
            egui::ComboBox::from_id_salt("ex_h2").selected_text(days_label(self.ex.h2)).show_ui(ui, |ui| {
                for &h in &horizons {
                    ui.selectable_value(&mut self.ex.h2, h, days_label(h));
                }
            });
            let lags: Vec<usize> = grid::lags(self.ex.h2).into_iter().filter(|&k| k > 0).collect();
            if !lags.contains(&self.ex.lag) {
                self.ex.lag = lags.iter().copied().find(|&k| k >= self.ex.lag).or(lags.last().copied()).unwrap_or(self.ex.h2);
            }
            ui.label("offset");
            egui::ComboBox::from_id_salt("ex_lag").selected_text(days_label(self.ex.lag)).show_ui(ui, |ui| {
                for &k in &lags {
                    ui.selectable_value(&mut self.ex.lag, k, days_label(k));
                }
            });
            ui.label("from the end of A's move to the end of B's");
        });
        let before = self.ex.length;
        ui.horizontal_wrapped(|ui| {
            ui.label("Correlation period:");
            egui::ComboBox::from_id_salt("ex_len").selected_text(len_label(self.ex.length)).show_ui(ui, |ui| {
                for &l in &lengths {
                    ui.selectable_value(&mut self.ex.length, l, len_label(l));
                }
            });
            if self.ex.length != before {
                self.ex.pos = usize::MAX;
            }
            let ends: Vec<String> = plan.windows.iter().filter(|w| w.length == self.ex.length).map(|w| c2_data::date_string(plan.var.days[w.end - 1])).collect();
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
        let window = plan.windows.iter().filter(|w| w.length == self.ex.length).nth(self.ex.pos).map(|w| w.id);
        self.ex.selected = window.map(|w| (plan.windows[w].length, plan.var.days[plan.windows[w].end - 1]));
        let run = self.run_name.clone();
        let key: Option<ExplainKey> = window.map(|w| {
            let threshold = self.ex.threshold_fixed.then(|| (self.ex.threshold_pct * 100.0).round() as i64);
            (run.clone(), self.ex.leader.clone(), self.ex.follower.clone(), self.ex.h1, self.ex.h2, self.ex.lag, w, threshold)
        });
        self.ex.key = key.clone();
        if let Some(k) = &key
            && self.ex.job.wants(k)
            && let Some(store) = self.store.clone()
        {
            let (plan2, dir, k2) = (plan.clone(), self.dir(), k.clone());
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
                c2_census::explain::explain(&store, &plan2, &dir, &req).map_err(|e| format!("{e:#}"))
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
            self.explain_steps(ui, e);
        }
        self.ex.job.done = done;
    }

    fn explain_steps(&mut self, ui: &mut egui::Ui, e: &Explained) {
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
        egui::ScrollArea::vertical().id_salt("ex_scroll").show(ui, |ui| {
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
            ui.label(format!(
                "Each observation pairs {a}'s move over {} with {b}'s move over {} that ends {} after {a}'s ends. Observations step forward \
                 {} at a time so no two share a day: {} observations in this period.",
                days_label(h1),
                days_label(h2),
                tdays(lag),
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
            const STEPS: [&str; 8] = [
                "closing price: day before -> that day",
                "1. daily return r = ln(close / previous close)",
                "2. market return m that day (average of the liquid stocks)",
                "3. beta from the previous 252 days = cov(r, m) / var(m)",
                "4. residual e = r - beta x m (the stock's own move)",
                "5. mean / sd of e over the previous 252 days",
                "6. census value v = (e - mean) / sd, kept within +-5",
                "   the census's own code, same day",
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
            let diff = |s: &DaySteps| if s.max_diff.is_finite() { format!("{}", s.max_diff) } else { "a value is missing on one side".into() };
            ui.label(format!(
                "Recomputed here for every day of the period and compared with what the census's own code computes from the same prices (the census \
                 doesn't store daily values; step 4 checks the stored correlation). Largest difference: {a} {}, {b} {}.",
                diff(&e.a),
                diff(&e.b)
            ));
            let series = |s: &DaySteps| -> Vec<[f64; 2]> { s.v_census.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, v)| [i as f64, *v as f64]).collect() };
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
                .y_axis_label(format!("{b}'s move {lag} days later{unit}"))
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
                "r = {:+.4} from n = {} observations. r measures how consistently {b}'s later move lines up with {a}'s move: 0 = no relation, +1 = always \
                 the same direction in proportion, -1 = always the opposite. z = atanh(r) x sqrt(n - 3) = {:+.2}: how many standard errors r is from 0. \
                 One pair on its own reaches |z| of 2 by chance about 5% of the time, and 4 about 6 times in 100,000.",
                e.r, e.n, e.z
            ));
            let missing: Vec<&str> = [&e.a, &e.b].iter().filter(|s| !s.in_universe).map(|s| s.ticker.as_str()).collect();
            match &e.stored {
                None if !missing.is_empty() => ui.label(format!(
                    "The census never computed this pair in this period because {} did not qualify (see step 1); the r above is what it would have been.",
                    missing.join(" and ")
                )),
                Some(s) => ui.label(format!(
                    "The census stored this pair for this period: r = {:+.6} (recomputed here: {:+.6}; difference {:.1e}). Its GPU estimate of r (the fast \
                     8-bit first pass, which decides what is stored and applies the significance test) was {:+.4}{}; it {} the significance test.",
                    s.record.r,
                    e.r,
                    (s.record.r as f64 - e.r).abs(),
                    s.record.r_int8,
                    if s.top { "; among the strongest in its matrix" } else { "" },
                    if s.significant { "passed" } else { "did not pass" }
                )),
                None => ui.label(
                    "The census computed this pair but did not store it for this period: it keeps the strongest 100 of each matrix plus any that pass \
                     the significance test.",
                ),
            };
            ui.label(format!("For comparison, the correlation of the raw % moves (no market adjustment or volatility scaling) is {:+.3}.", e.r_raw));

            // 5. Chance.
            ui.add_space(8.0);
            ui.heading("5. Could it be chance?");
            let m = &e.matrix;
            let az = e.z.abs();
            let formula = if e.chance_analytic >= 1.0 { format!("{:.0}", e.chance_analytic) } else { format!("{:.3}", e.chance_analytic) };
            if m.found {
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
            } else {
                ui.label(format!(
                    "Among the {} correlations in this matrix ({} stocks, each against every other), a bell-curve formula predicts about {formula} at |z| of {az:.2} \
                     or more by chance alone; stock moves have fatter tails, so the real chance count is higher.",
                    big(e.pairs_tested),
                    e.universe
                ));
            }
            if m.found {
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
                    Some(t) => ui.label(significance_verdict(t, e.r, e.stored.as_ref().map(|s| (s.record.r_int8, s.significant)), missing.is_empty())),
                    None => ui.label("No pair in this matrix passed the significance test: nothing stood out from chance."),
                };
            } else {
                ui.label("This census run has no summary for this matrix (the stocks may not have qualified, or this run used a smaller grid).");
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

            // 7. Size.
            ui.add_space(8.0);
            ui.heading("7. Size in price terms (used by the timeline and the Forecast tab)");
            let raw_moves: Vec<(f64, f64)> = e.obs.iter().map(|o| (o.a_move, o.b_move)).filter(|p| p.0.is_finite() && p.1.is_finite()).collect();
            let sd = |v: Vec<f64>| {
                let n = v.len().max(1) as f64;
                let mu = v.iter().sum::<f64>() / n;
                (v.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / n).sqrt()
            };
            let (sa, sb) = (sd(raw_moves.iter().map(|p| p.0).collect()), sd(raw_moves.iter().map(|p| p.1).collect()));
            ui.label(format!(
                "% move per 1% = r x (typical size of {b}'s {}-day moves / typical size of {a}'s {}-day moves) = {:+.4} x ({:.2}% / {:.2}%) = {:+.3}. \
                 So when {a} moves 1% more than usual over {}, {b} tends to move {:+.3}% more than usual over the {} ending {} later. \
                 The Forecast tab multiplies each leader's latest move by this number.",
                h2,
                h1,
                e.r,
                100.0 * sb,
                100.0 * sa,
                e.move_per_pct,
                days_label(h1),
                e.move_per_pct,
                days_label(h2),
                tdays(lag as usize)
            ));
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
            self.stats.start(run, move || c2_census::analyze::dataset_stats(&store, &dir).map_err(|e| format!("{e:#}")));
        }
    }

    fn status_stats(&mut self, ui: &mut egui::Ui) {
        let run_name = self.run_name.clone();
        // Every run (independent of the one chosen), and the chosen run's statistics: switching
        // runs rereads only the latter.
        if self.runs.wants(&self.runs_refresh) {
            self.runs.start(self.runs_refresh, || Ok(c2_census::analyze::run_history()));
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
            let runs = match self.runs.get(&self.runs_refresh) {
                Some(Ok(runs)) => runs,
                Some(Err(e)) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, e);
                    return;
                }
                None => {
                    ui.label("Reading the census output…");
                    return;
                }
            };
            egui::Grid::new("runs_grid").striped(true).show(ui, |ui| {
                for h in ["run", "kind", "data through", "started", "time computing", "correlations computed", "GPU multiply-adds", "state"] {
                    ui.strong(h);
                }
                ui.end_row();
                for r in runs {
                    let mut name = r.name.clone();
                    if r.in_use {
                        name.push_str("  (in use)");
                    }
                    if r.from_update {
                        name.push_str("  (new data)");
                    }
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
            let st = match self.stats.get(&run_name) {
                Some(Ok(st)) => st,
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
                row("stored relationships", format!("{} (each matrix's strongest 100 plus every significant one); {} chance ones stored for comparison", commas(st.stored_real as f64), commas(st.stored_null as f64)));
                row(
                    "significant (passed the false-discovery test)",
                    format!("{} relationships: {} distinct pairs, {} stocks, in {} of {} matrices", commas(lt.significant as f64), commas(st.significant_pairs as f64), commas(st.significant_stocks as f64), commas(st.fdr_matrices as f64), commas(lt.matrices as f64)),
                );
                row("the same test run on pure chance fired in", format!("{} matrices (close to the line above means the 'significant' ones are mostly chance too)", st.placebo_matrices));
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

    fn poll(&mut self) {
        if self.last_poll.elapsed() < Duration::from_secs(2) {
            return;
        }
        self.last_poll = Instant::now();
        let dir = self.dir();
        self.pid = running_pid(&dir);
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
                let state = s["state"].as_str().unwrap_or("?").to_string();
                ui.label(format!(
                    "State: {state}  ·  process: {}  ·  control: {}",
                    self.pid.map(|p| format!("running (pid {p})")).unwrap_or_else(|| "not running".into()),
                    self.control
                ));
                ui.add(egui::ProgressBar::new(frac).show_percentage());
                ui.label(format!(
                    "{} of {} work units (one pair of move lengths in one correlation period) · ETA {}",
                    s["units_done"], s["units_total"], run::hms(s["eta_seconds"].as_f64().unwrap_or(-1.0))
                ));
            }
            None => {
                ui.label(&self.status_err);
            }
        }
        ui.add_space(8.0);
        // While a new run name is being typed, the buttons wait for it to be switched to, so they
        // never act on a different run than the one shown.
        let settled = self.run_input.trim() == self.run_name;
        ui.horizontal(|ui| {
            let dir = self.dir();
            if ui.add_enabled(settled && self.pid.is_none(), egui::Button::new("Start / resume census")).clicked() {
                let _ = std::fs::create_dir_all(&dir);
                let log = std::fs::File::create(dir.join("run.out"));
                let mut cmd = std::process::Command::new(c2_binary());
                cmd.args(["census", "run", "--name", &self.run_name]);
                if let Ok(f) = log {
                    if let Ok(f2) = f.try_clone() {
                        cmd.stdout(f).stderr(f2);
                    }
                }
                use std::os::unix::process::CommandExt;
                cmd.process_group(0);
                self.message = match cmd.spawn() {
                    Ok(c) => format!("started census '{}' (pid {})", self.run_name, c.id()),
                    Err(e) => format!("could not start c2: {e}"),
                };
                self.last_poll = Instant::now() - Duration::from_secs(10);
            }
            if ui.add_enabled(settled, egui::Button::new("Pause (frees the GPU)")).clicked() {
                let _ = run::set_control(&dir, "pause");
                self.message = "pausing after the current window".into();
            }
            if ui.add_enabled(settled, egui::Button::new("Resume")).clicked() {
                let _ = run::set_control(&dir, "run");
                self.message = "resuming".into();
            }
            if ui.add_enabled(settled && self.pid.is_some(), egui::Button::new("Stop")).clicked() {
                if let Some(pid) = self.pid {
                    let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
                    self.message = format!("asked pid {pid} to stop after committing the current window");
                }
            }
        });
        if !self.message.is_empty() {
            ui.label(&self.message);
        }
        ui.add_space(4.0);
        ui.label("Stopping or pausing loses at most the work unit in progress; starting again resumes where it left off.");
        ui.separator();
        self.status_stats(ui);
    }

    fn load_lag_table(&mut self) {
        self.lag_table.clear();
        self.lag_table_run = self.run_name.clone();
        let Ok(entries) = std::fs::read_dir(self.dir().join("tasks")) else {
            self.message = "no census output yet".into();
            return;
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
        if !self.lag_table.contains_key(&self.pair)
            && let Some(k) = self.lag_table.keys().next()
        {
            self.pair = *k;
        }
    }

    /// Looks up the strongest stored pairs of the Census tab's ticker in the current run.
    fn start_find_pairs(&mut self) {
        let Some(store) = self.store.clone() else { return };
        let (dir, ticker) = (self.dir(), self.ticker.trim().to_uppercase());
        self.pairs.start((self.run_name.clone(), ticker.clone()), move || find_pairs(&store, &dir, &ticker).map_err(|e| format!("{e:#}")));
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
        let (run, r) = match &self.holdout.done {
            Some((k, Ok(r))) if k.0 == self.run_name => (&k.0, r),
            Some((k, Err(e))) if k.0 == self.run_name => {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
                return;
            }
            _ => return,
        };
        ui.separator();
        ui.label(format!(
            "Census '{run}': {} .. {} ({} trading days) · {} stocks could be picked · picks beat the market {:.1}% of the time · rank IC t = {:+.2} · long-short t = {:+.2}",
            c2_data::date_string(r.first_day), c2_data::date_string(r.last_day), r.days, r.candidates, 100.0 * r.hit_rate, r.ic_t, r.ls_t
        ));
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
        let (run, dir, slot) = (self.run_name.clone(), self.dir(), self.plan_slot.clone());
        let census_pairs = self.family == 0;
        let family = if census_pairs {
            Family::CensusPairs
        } else {
            Family::PeerMomentum { peers: 20, lookback: 21, window: 756 }
        };
        let start = self.start.clone();
        let (picks, rebalance, random) = (self.picks, self.rebalance, self.random);
        let costs: Vec<f64> = self.costs.split(',').filter_map(|c| c.trim().parse().ok()).collect();
        let key: HoldoutKey = (run.clone(), self.family, start.clone(), picks, rebalance, self.costs.clone(), random);
        self.holdout.start(key, move || {
            // This run's census variable (rebuilt if the slot holds another run's), checked against
            // the run's correlation periods, and for census pairs its task numbering, before any
            // record is looked up by them.
            let plan = plan_for_run(&slot, &store, &run, &dir)?;
            if census_pairs {
                check_tasks(&plan, &run, &dir)?;
            }
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
            c2_census::holdout::holdout(&store, &plan, &dir, &opt).map_err(|e| format!("{e:#}"))
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
fn find_pairs(store: &Store, dir: &Path, ticker: &str) -> anyhow::Result<Vec<PairRow>> {
    let id = store.id(ticker).ok_or_else(|| anyhow::anyhow!("{ticker} is not in the price store"))? as u32;
    let windows: Vec<String> = std::fs::read_to_string(dir.join("windows.csv"))?
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            format!("{}..{}", f.get(2).unwrap_or(&""), f.get(3).unwrap_or(&""))
        })
        .collect();
    let mut rows = Vec::new();
    for e in std::fs::read_dir(dir.join("tasks"))?.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("rec") {
            continue;
        }
        for r in c2_census::analyze::read_records(&path)? {
            if r.flags & FLAG_NULL != 0 || r.lag == 0 || (r.leader != id && r.target != id) {
                continue;
            }
            let (partner, role) = if r.leader == id { (r.target, "leads") } else { (r.leader, "follows") };
            rows.push(PairRow {
                window: windows.get(r.window as usize).cloned().unwrap_or_default(),
                partner: store.symbols.get(partner as usize).map(|s| s.ticker.clone()).unwrap_or_default(),
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
        self.poll();
        // Results arrive whichever tab is shown.
        self.poll_jobs();
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
            let ready = self.tab_ready();
            let timed_out = shot.started.elapsed() >= shot.timeout;
            if !shot.requested && shot.frames > 30 && (ready || timed_out) {
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
        // While a run name is being typed, wake up in time to switch to it.
        ctx.request_repaint_after(Duration::from_millis(if self.run_edited.is_some() { 200 } else { 500 }));
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
            }
            _ => self.tab = Tab::Status,
        }
    }
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
        let (c, text) = track_record_verdict(0.35, -0.25, 200);
        assert_eq!(c, RED_NOTE);
        assert!(text.contains("worse than chance"), "{text}");
        // One significantly negative measure is enough.
        assert_eq!(track_record_verdict(0.5, -0.14, 225).0, RED_NOTE);
        let (c, text) = track_record_verdict(0.60, 0.25, 200);
        assert_eq!(c, GREEN_NOTE);
        assert!(text.contains("better than luck"), "{text}");
        assert!(track_record_verdict(0.52, 0.05, 200).1.contains("within what luck produces"));
        // Mixed: direction right more often than luck, sizes moving the opposite way.
        let (c, text) = track_record_verdict(0.60, -0.20, 200);
        assert_eq!(c, YELLOW_NOTE);
        assert!(text.contains("disagree"), "{text}");
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
