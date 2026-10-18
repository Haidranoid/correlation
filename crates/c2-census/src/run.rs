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

use anyhow::{Result, bail};
use c2_data::{Calendar, Store};
use rayon::prelude::*;

use crate::grid::{self, Config, Task, Window};
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

pub fn run_dir(name: &str) -> PathBuf {
    c2_data::data_dir().join("census").join(name)
}

/// Everything decided before any work starts.
pub struct Plan {
    pub cfg: Config,
    pub var: Variable,
    pub windows: Vec<Window>,
    /// Per window: variable column indices of its universe.
    pub universes: Vec<Vec<usize>>,
    pub tasks: Vec<Task>,
    /// Estimated cost of each (task, window position).
    pub cost: Vec<Vec<f64>>,
}

impl Plan {
    pub fn new(store: &Store, cfg: Config) -> Result<Plan> {
        let cal = Calendar::trading_days(store)?;
        let var = variable::production(store, &cal, cfg.start)?;
        let first_row = var.days.partition_point(|&d| d < cfg.start);
        let windows = grid::windows(&cfg, first_row, var.t);
        let universes: Vec<Vec<usize>> = windows.par_iter().map(|w| universe(&var, w)).collect();
        let tasks = grid::tasks(&cfg, &windows);
        let cost = tasks
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
            .collect();
        Ok(Plan { cfg, var, windows, universes, tasks, cost })
    }

    pub fn total_cost(&self) -> f64 {
        self.cost.iter().flatten().sum()
    }
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
    format!(
        "horizons={:?} lengths={:?} full_history={} step={} start={} min_obs={} nulls={} keep_real={} keep_null={} floor_z={} fdr_q={}",
        cfg.horizons, cfg.lengths, cfg.full_history, cfg.step, c2_data::date_string(cfg.start), cfg.min_obs, cfg.nulls,
        cfg.keep_real, cfg.keep_null, cfg.floor_z, cfg.fdr_q
    )
}

/// FNV-1a, for fingerprinting files in the manifest.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

fn write_atomic(path: &Path, text: &str) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// Writes manifest, windows and universes on first start; on resume checks the config matches.
fn prepare_dir(dir: &Path, plan: &Plan, store: &Store) -> Result<()> {
    std::fs::create_dir_all(dir.join("tasks"))?;
    let manifest = dir.join("manifest.json");
    let cfg = config_text(&plan.cfg);
    if manifest.exists() {
        let old: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&manifest)?)?;
        if old["config"].as_str() != Some(cfg.as_str()) {
            bail!("run directory {} was started with a different config:\n  {}\nnow:\n  {}", dir.display(), old["config"], cfg);
        }
        return Ok(());
    }
    let symbols = std::fs::read(c2_data::data_dir().join("store/symbols.tsv")).unwrap_or_default();
    let git = std::process::Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    let m = serde_json::json!({
        "config": cfg,
        "store_symbols": store.symbols.len(),
        "store_rows": store.rows(),
        "symbols_tsv_fnv64": format!("{:016x}", fnv(&symbols)),
        "variable_days": [c2_data::date_string(plan.var.days[0]), c2_data::date_string(*plan.var.days.last().unwrap())],
        "variable_stocks": plan.var.n,
        "windows": plan.windows.len(),
        "tasks": plan.tasks.len(),
        "git_commit": git.trim(),
        "created_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),
        "record_bytes": RECORD_BYTES,
        "variable": "market-residual daily log return / trailing 252-day residual sd (both through t-1), clipped +-5; horizon sums divided by sqrt(h)",
        "null": "sign-flip: target observations times a random +-1 shared by all stocks on that observation (keeps volatility alignment)",
    });
    write_atomic(&manifest, &serde_json::to_string_pretty(&m)?)?;
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
    write_atomic(&dir.join("windows.csv"), &w)?;
    std::fs::write(dir.join("universes.bin"), u)?;
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
    let (s, g0, ld) = (u.h1.max(u.h2), u.h1 - 1, u.w + 1);
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
    // (lag index, entry, null?, flags)
    let mut chosen: Vec<(usize, i64, f32, bool, u8)> = Vec::new();
    for (li, (k, m, _)) in u.lags.iter().enumerate() {
        let thr = &res.thr[li * 3..li * 3 + 3];
        let (rc, nc) = (res.cnt[li * 2] as usize, res.cnt[li * 2 + 1] as usize);
        let mut real = res.real[li * CAP_REAL..li * CAP_REAL + rc.min(CAP_REAL)].to_vec();
        real.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
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
        let mut null = res.null[li * CAP_NULL..li * CAP_NULL + nc.min(CAP_NULL)].to_vec();
        null.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
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
            fdr.map(|t| format!("{t:.4}")).unwrap_or_default(), stored, stored_null, rc, u8::from(rc > CAP_REAL),
            placebo.map(|t| format!("{t:.4}")).unwrap_or_default(),
        ));
    }
    let np = u.np as i64;
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
/// `log` every ~15 s.
pub fn run(store: &Store, plan: &Plan, dir: &Path, stop: &AtomicBool, log: &(dyn Fn(&str) + Sync)) -> Result<()> {
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
                    log(&format!("{units_done}/{total_units} windows ({:.2}% of work), last unit {:.0} ms on GPU, ETA {}",
                        100.0 * cost_done / cost_total, res.gpu_ms, hms(eta)));
                    last_log = Instant::now();
                }
            }
            let rate = (cost_done - session_start_cost) / started.elapsed().as_secs_f64().max(1e-3);
            let state = if units_done == total_units { "finished" } else { "stopped" };
            write_status(&dir_f, state, units_done, total_units, cost_done, cost_total, rate, started, "");
            log(&format!("{state}: {units_done}/{total_units} windows committed"));
            Ok(())
        });
        let worker = gpu_worker(plan, prep_rx, res_tx, stop, dir, log);
        let fin = finisher.join().map_err(|_| anyhow::anyhow!("finisher thread panicked"))?;
        worker.and(fin)
    })
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
        let (h1, h2, step, g0) = (u.h1, u.h2, u.h1.max(u.h2), u.h1 - 1);
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
