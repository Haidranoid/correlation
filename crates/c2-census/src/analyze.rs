//! Reading census output, and the replication test: do stored pairs keep their sign in the next
//! window that does not overlap?

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::grid::{self, Window};
use crate::run::{FLAG_NULL, FLAG_TOP, Plan, RECORD_BYTES, Record};

/// All records of one task file.
pub fn read_records(path: &Path) -> Result<Vec<Record>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(bytes.chunks_exact(RECORD_BYTES).map(Record::from_bytes).collect())
}

/// Exact r of leader column `i` and target column `j` (variable columns) in a window.
fn r_in_window(plan: &Plan, w: &Window, h1: usize, h2: usize, lag: usize, i: usize, j: usize) -> Option<f64> {
    let m = grid::observations(w.end - w.start, h1, h2, lag);
    if m < plan.cfg.min_obs {
        return None;
    }
    let (vi, vj) = (&plan.var.col(&plan.var.v, i)[w.start..w.end], &plan.var.col(&plan.var.v, j)[w.start..w.end]);
    let sum = |v: &[f32], end: usize, h: usize| -> f64 {
        v[end + 1 - h..=end].iter().map(|&x| if x.is_finite() { x as f64 } else { 0.0 }).sum::<f64>() / (h as f64).sqrt()
    };
    let (s, g0) = (h1.max(h2), h1 - 1);
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
pub fn replicate(plan: &Plan, dir: &Path) -> Result<RepTable> {
    let col_of: HashMap<u32, usize> = plan.var.ids.iter().enumerate().map(|(c, &id)| (id as u32, c)).collect();
    let mut table: HashMap<(usize, usize, &'static str, bool), Rep> = HashMap::new();
    for task in &plan.tasks {
        if task.length == 0 {
            continue;
        }
        let path = dir.join(format!("tasks/t{}.rec", task.id));
        if !path.exists() {
            continue;
        }
        let records = read_records(&path)?;
        // Next non-overlapping window of the same length: the one starting at this one's end.
        let next: HashMap<usize, usize> = task
            .windows
            .iter()
            .filter_map(|&a| task.windows.iter().find(|&&b| plan.windows[b].start == plan.windows[a].end).map(|&b| (a, b)))
            .collect();
        let results: Vec<((usize, usize, &'static str, bool), f64, f64)> = records
            .par_iter()
            .filter(|r| r.flags & FLAG_TOP != 0)
            .filter_map(|r| {
                let nw = *next.get(&(r.window as usize))?;
                let (i, j) = (*col_of.get(&r.leader)?, *col_of.get(&r.target)?);
                let rn = r_in_window(plan, &plan.windows[nw], task.h1, task.h2, r.lag as usize, i, j)?;
                let key = (task.h1, task.h2, bucket(r.lag as usize, task.h2), r.flags & FLAG_NULL != 0);
                Some((key, r.r as f64, rn))
            })
            .collect();
        for (key, then, next) in results {
            let e = table.entry(key).or_default();
            e.pairs += 1;
            e.same_sign += usize::from(then.signum() == next.signum());
            e.signed_next += then.signum() * next;
            e.abs_then += then.abs();
        }
    }
    let mut out: RepTable = table.into_iter().collect();
    out.sort_by(|a, b| (a.0.0, a.0.1, a.0.2, a.0.3).cmp(&(b.0.0, b.0.1, b.0.2, b.0.3)));
    Ok(out)
}

/// One stored relationship involving a chosen stock, with its size in price terms.
#[derive(Clone)]
pub struct ProfileRow {
    pub window: String,
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
    /// The tested relationship in price terms: r x (target return volatility / leader return
    /// volatility) over the same observations, i.e. the target's typical % move beyond the market
    /// per 1% move of the leader. Same sign as r. None if prices are missing.
    pub move_per_pct: Option<f64>,
}

/// Every stored real lagged relationship involving `ticker` (as leader or target), strongest
/// |z| first; with `significant_only`, just those that passed the false-discovery test.
pub fn stock_profile(store: &c2_data::Store, dir: &Path, ticker: &str, significant_only: bool) -> Result<Vec<ProfileRow>> {
    use crate::run::FLAG_FDR;
    let id = store.id(ticker).ok_or_else(|| anyhow::anyhow!("{ticker} is not in the price store"))? as u32;
    let cal = c2_data::Calendar::trading_days(store)?;
    // Window id -> (first day, last day).
    let windows: Vec<(i32, i32)> = std::fs::read_to_string(dir.join("windows.csv"))?
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            Ok((c2_data::parse_date(f[2])?, c2_data::parse_date(f[3])?))
        })
        .collect::<Result<_>>()?;
    let mut rows = Vec::new();
    let mut tasks: Vec<PathBuf> = std::fs::read_dir(dir.join("tasks"))?.flatten().map(|e| e.path()).filter(|p| p.extension().and_then(|x| x.to_str()) == Some("rec")).collect();
    tasks.sort();
    for path in tasks {
        let records: Vec<Record> = read_records(&path)?
            .into_iter()
            .filter(|r| r.flags & FLAG_NULL == 0 && r.lag > 0 && (r.leader == id || r.target == id))
            .filter(|r| !significant_only || r.flags & FLAG_FDR != 0)
            .collect();
        if records.is_empty() {
            continue;
        }
        // Chance tail counts per (window, lag) from this task's summary.
        let mut chance: HashMap<(u32, u16), [f64; 3]> = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(path.with_extension("csv")) {
            let mut lines = text.lines();
            let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
            let col = |n: &str| header.iter().position(|h| *h == n);
            if let (Some(cw), Some(cl), Some(c3), Some(c4), Some(c5)) = (col("window"), col("lag"), col("null_z3"), col("null_z4"), col("null_z5")) {
                for line in lines {
                    let f: Vec<&str> = line.split(',').collect();
                    let num = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::NAN);
                    chance.insert((num(cw) as u32, num(cl) as u16), [num(c3), num(c4), num(c5)]);
                }
            }
        }
        for r in records {
            let (first, last) = windows[r.window as usize];
            let (az, tails) = (r.z.abs(), chance.get(&(r.window, r.lag)).copied().unwrap_or([f64::NAN; 3]));
            let (chance_z, chance_n) = if az >= 5.0 { (5, tails[2]) } else if az >= 4.0 { (4, tails[1]) } else { (3, tails[0]) };
            let (leader, target) = (store.symbols[r.leader as usize].ticker.clone(), store.symbols[r.target as usize].ticker.clone());
            rows.push(ProfileRow {
                window: format!("{}..{}", c2_data::date_string(first), c2_data::date_string(last)),
                partner: if r.leader == id { target.clone() } else { leader.clone() },
                leads: r.leader == id,
                h1: r.h1,
                h2: r.h2,
                lag: r.lag,
                obs: r.obs,
                r: r.r,
                z: r.z,
                significant: r.flags & FLAG_FDR != 0,
                chance: chance_n,
                chance_z,
                move_per_pct: vol_ratio(store, &cal, &leader, &target, first, last, r.h1 as usize, r.h2 as usize, r.lag as usize).map(|k| r.r as f64 * k),
            });
        }
    }
    rows.sort_by(|a, b| b.z.abs().total_cmp(&a.z.abs()));
    Ok(rows)
}

/// Ratio of the target's h2-day to the leader's h1-day log-return standard deviation on the
/// census's observation grid inside [first, last].
#[allow(clippy::too_many_arguments)]
fn vol_ratio(store: &c2_data::Store, cal: &c2_data::Calendar, leader: &str, target: &str, first: i32, last: i32, h1: usize, h2: usize, lag: usize) -> Option<f64> {
    let rows = cal.range(first, last);
    let (w0, w) = (rows.start, rows.end - rows.start);
    let series = |t: &str| -> Option<Vec<f64>> {
        let id = store.id(t)?;
        let (d, c) = (store.dates(id), store.close(id));
        // Log price on calendar rows w0-1 .. w0+w (one row before the window for the first return).
        Some((w0.saturating_sub(1)..w0 + w).map(|k| d.binary_search(&cal.days[k]).ok().map(|i| (c[i] as f64).ln()).unwrap_or(f64::NAN)).collect())
    };
    let (pl, pt) = (series(leader)?, series(target)?);
    let off = usize::from(w0 > 0);
    let (s, g0) = (h1.max(h2), h1 - 1);
    let m = crate::grid::observations(w, h1, h2, lag);
    let (mut sx, mut sy, mut sxx, mut syy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for q in 0..m {
        let g = g0 + q * s;
        let (le, ls) = (g + off, (g + off).checked_sub(h1)?);
        let (te, ts) = (g + lag + off, (g + lag + off).checked_sub(h2)?);
        let (x, y) = (pl.get(le)? - pl.get(ls)?, pt.get(te)? - pt.get(ts)?);
        if x.is_finite() && y.is_finite() {
            sx += x; sy += y; sxx += x * x; syy += y * y; n += 1.0;
        }
    }
    let (vx, vy) = (sxx - sx * sx / n, syy - sy * sy / n);
    (n >= 10.0 && vx > 0.0 && vy > 0.0).then(|| (vy / vx).sqrt())
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
    /// Strongest |z| among all stored relationships (mostly chance level).
    pub max_z_any: f32,
}

/// Every stock with stored lagged relationships, most significant first.
pub fn stock_ranking(store: &c2_data::Store, dir: &Path) -> Result<Vec<RankRow>> {
    use crate::run::FLAG_FDR;
    #[derive(Default)]
    struct Acc {
        significant: usize,
        leads: usize,
        follows: usize,
        windows: std::collections::HashSet<u32>,
        stored: usize,
        sum_abs_r: f64,
        max_abs_r: f32,
        sig_sum_abs_r: f64,
        best: Option<(f32, u32, bool, u16, f32)>,
        best_any: Option<(f32, u32, bool, u16, f32)>,
    }
    let mut acc: HashMap<u32, Acc> = HashMap::new();
    for e in std::fs::read_dir(dir.join("tasks"))?.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("rec") {
            continue;
        }
        for r in read_records(&path)? {
            if r.flags & FLAG_NULL != 0 || r.lag == 0 {
                continue;
            }
            for (me, other, leads) in [(r.leader, r.target, true), (r.target, r.leader, false)] {
                let a = acc.entry(me).or_default();
                a.stored += 1;
                a.sum_abs_r += r.r.abs() as f64;
                a.max_abs_r = a.max_abs_r.max(r.r.abs());
                let cand = (r.z.abs(), other, leads, r.lag, r.r);
                if a.best_any.is_none_or(|b| cand.0 > b.0) {
                    a.best_any = Some(cand);
                }
                if r.flags & FLAG_FDR != 0 {
                    a.significant += 1;
                    a.sig_sum_abs_r += r.r.abs() as f64;
                    if leads { a.leads += 1 } else { a.follows += 1 }
                    a.windows.insert(r.window);
                    if a.best.is_none_or(|b| cand.0 > b.0) {
                        a.best = Some(cand);
                    }
                }
            }
        }
    }
    let name = |id: u32| store.symbols.get(id as usize).map(|s| s.ticker.clone()).unwrap_or_default();
    let mut rows: Vec<RankRow> = acc
        .into_iter()
        .map(|(id, a)| {
            let (z, other, leads, lag, r) = a.best.or(a.best_any).unwrap_or((0.0, 0, true, 0, 0.0));
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
                best_z: z,
                best_partner: name(other),
                best_leads: leads,
                best_lag: lag,
                best_r: r,
                max_z_any: a.best_any.map(|b| b.0).unwrap_or(0.0),
            }
        })
        .collect();
    rows.sort_by(|a, b| b.sum_abs_r.total_cmp(&a.sum_abs_r));
    Ok(rows)
}
