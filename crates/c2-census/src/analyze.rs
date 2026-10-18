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
    /// The tested relationship in price terms: r x (target return volatility / leader return
    /// volatility) over the same observations, i.e. the target's typical % move beyond the market
    /// per 1% move of the leader. Same sign as r. None if prices are missing.
    pub move_per_pct: Option<f64>,
    /// The same relationship counted in plain terms from prices: how often the follower rose
    /// after the leader's big moves, in the period and over the rest of the history.
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
pub fn stock_profile(store: &c2_data::Store, dir: &Path, ticker: &str, significant_only: bool) -> Result<Vec<ProfileRow>> {
    use crate::run::FLAG_FDR;
    let id = store.id(ticker).ok_or_else(|| anyhow::anyhow!("{ticker} is not in the price store"))? as u32;
    let cal = c2_data::Calendar::trading_days(store)?;
    // Window id -> (first day, last day, calendar rows).
    let windows: Vec<(i32, i32, std::ops::Range<usize>, bool)> = std::fs::read_to_string(dir.join("windows.csv"))?
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let (a, b) = (c2_data::parse_date(f[2])?, c2_data::parse_date(f[3])?);
            Ok((a, b, cal.range(a, b), f[1] == "0"))
        })
        .collect::<Result<_>>()?;
    // The census's whole history: the full-history window, else the union of windows.
    let (lo, hi) = windows
        .iter()
        .find(|w| w.3)
        .map(|w| (w.2.start, w.2.end))
        .unwrap_or_else(|| (windows.iter().map(|w| w.2.start).min().unwrap_or(0), windows.iter().map(|w| w.2.end).max().unwrap_or(0)));
    let base = lo.saturating_sub(300);
    let mut records: Vec<(Record, f64, u8)> = Vec::new();
    let mut tasks: Vec<PathBuf> = std::fs::read_dir(dir.join("tasks"))?.flatten().map(|e| e.path()).filter(|p| p.extension().and_then(|x| x.to_str()) == Some("rec")).collect();
    tasks.sort();
    for path in tasks {
        let recs: Vec<Record> = read_records(&path)?
            .into_iter()
            .filter(|r| r.flags & FLAG_NULL == 0 && r.lag > 0 && (r.leader == id || r.target == id))
            .filter(|r| !significant_only || r.flags & FLAG_FDR != 0)
            .collect();
        if recs.is_empty() {
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
        for r in recs {
            let (az, tails) = (r.z.abs(), chance.get(&(r.window, r.lag)).copied().unwrap_or([f64::NAN; 3]));
            let (chance_z, chance_n) = if az >= 5.0 { (5, tails[2]) } else if az >= 4.0 { (4, tails[1]) } else { (3, tails[0]) };
            records.push((r, chance_n, chance_z));
        }
    }
    // Log closes on calendar rows base.. for every symbol involved.
    let mut ids: Vec<u32> = records.iter().flat_map(|(r, _, _)| [r.leader, r.target]).collect();
    ids.sort_unstable();
    ids.dedup();
    let prices: HashMap<u32, Vec<f64>> = ids
        .par_iter()
        .map(|&s| {
            let (d, c) = (store.dates(s as usize), store.close(s as usize));
            let v = (base..cal.days.len())
                .map(|k| d.binary_search(&cal.days[k]).ok().map(|i| c[i] as f64).filter(|p| *p > 0.0).map(f64::ln).unwrap_or(f64::NAN))
                .collect();
            (s, v)
        })
        .collect();
    let mut rows: Vec<ProfileRow> = records
        .par_iter()
        .map(|(r, chance_n, chance_z)| {
            let (first, last, range, _) = &windows[r.window as usize];
            let (h1, h2, lag) = (r.h1 as usize, r.h2 as usize, r.lag as usize);
            let (pl, pt) = (&prices[&r.leader], &prices[&r.target]);
            let step = h1.max(h2);
            let m = crate::grid::observations(range.len(), h1, h2, lag);
            // Grid rows relative to `base`: leader moves end at range.start + g0 + q * step.
            let first_e = range.start + step - 1 - base;
            let (p0, p1) = (range.start - base, range.end - base);
            let words = words(pl, pt, h1, h2, lag, first_e, m, step, p0, p1, lo - base, hi - base, None);
            let ratio = {
                let obs: Vec<(f64, f64)> = (0..m)
                    .filter_map(|q| {
                        let e = first_e + q * step;
                        let (x, y) = (pl.get(e)? - pl.get(e.checked_sub(h1)?)?, pt.get(e + lag)? - pt.get((e + lag).checked_sub(h2)?)?);
                        (x.is_finite() && y.is_finite()).then_some((x, y))
                    })
                    .collect();
                let n = obs.len() as f64;
                let (sx, sy) = (obs.iter().map(|o| o.0).sum::<f64>(), obs.iter().map(|o| o.1).sum::<f64>());
                let (vx, vy) = (obs.iter().map(|o| o.0 * o.0).sum::<f64>() - sx * sx / n, obs.iter().map(|o| o.1 * o.1).sum::<f64>() - sy * sy / n);
                (n >= 10.0 && vx > 0.0 && vy > 0.0).then(|| (vy / vx).sqrt())
            };
            let (leader, target) = (store.symbols[r.leader as usize].ticker.clone(), store.symbols[r.target as usize].ticker.clone());
            ProfileRow {
                window: format!("{}..{}", c2_data::date_string(*first), c2_data::date_string(*last)),
                period_days: range.len() as u32,
                period_end: c2_data::date_string(*last),
                window_id: r.window,
                partner: if r.leader == id { target } else { leader },
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
                move_per_pct: ratio.map(|k| r.r as f64 * k),
                words,
            }
        })
        .collect();
    rows.sort_by(|a, b| b.z.abs().total_cmp(&a.z.abs()));
    Ok(rows)
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
    /// Per stock (as leader or follower): stored relationships, mean |r| of them, significant.
    pub per_stock: Vec<(String, usize, f64, usize)>,
}

pub fn dataset_stats(store: &c2_data::Store, dir: &Path) -> Result<DatasetStats> {
    use crate::run::FLAG_FDR;
    let windows: Vec<(usize, i32)> = std::fs::read_to_string(dir.join("windows.csv"))?
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            Ok((f[1].parse::<usize>()?, f[3][..4].parse::<i32>()?))
        })
        .collect::<Result<_>>()?;
    let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let nulls = m["config"].as_str().and_then(|c| c.split("nulls=").nth(1)).and_then(|v| v.split(' ').next()).and_then(|v| v.parse().ok()).unwrap_or(3);
    let mut st = DatasetStats { nulls, periods: windows.len(), ..Default::default() };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir.join("tasks"))?.flatten().map(|e| e.path()).filter(|p| p.extension().and_then(|x| x.to_str()) == Some("csv")).collect();
    paths.sort();
    st.tasks = paths.len();
    let (mut by_h, mut by_len, mut by_year, mut by_off) = (HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new());
    let mut sig_by: HashMap<(usize, usize, usize, i32, usize), usize> = HashMap::new();
    let mut stocks = std::collections::HashSet::new();
    for path in &paths {
        let text = std::fs::read_to_string(path)?;
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            let num = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            let (w, h1, h2, lag, n) = (num(0) as usize, num(1) as usize, num(2) as usize, num(3) as usize, num(4));
            let corr = n * (n - 1.0).max(0.0);
            st.correlations_all += corr;
            st.matrices_all += 1;
            if lag == 0 {
                continue;
            }
            let real = [num(6), num(7), num(8)];
            let null = [num(9), num(10), num(11)];
            st.lagged.add(corr, real, null);
            let (len, year) = windows.get(w).copied().unwrap_or((0, 0));
            by_h.entry((h1, h2)).or_insert_with(Tally::default).add(corr, real, null);
            by_len.entry(len).or_insert_with(Tally::default).add(corr, real, null);
            by_year.entry(year).or_insert_with(Tally::default).add(corr, real, null);
            by_off.entry(lag / h2.max(1)).or_insert_with(Tally::default).add(corr, real, null);
            st.fdr_matrices += usize::from(!f.get(12).unwrap_or(&"").is_empty());
            st.placebo_matrices += usize::from(!f.get(17).unwrap_or(&"").is_empty());
        }
    }
    // Records: stored counts, |r| histograms, per-stock numbers, significant tallies.
    let recs: Vec<Vec<Record>> = paths.par_iter().map(|p| read_records(&p.with_extension("rec")).unwrap_or_default()).collect();
    st.hist_real = vec![0; 50];
    st.hist_null = vec![0; 50];
    let mut per: HashMap<u32, (usize, f64, usize)> = HashMap::new();
    let mut sig_pairs = std::collections::HashSet::new();
    for r in recs.iter().flatten().filter(|r| r.lag > 0) {
        let bin = ((r.r.abs() * 50.0) as usize).min(49);
        if r.flags & FLAG_NULL != 0 {
            st.stored_null += 1;
            st.hist_null[bin] += 1;
            continue;
        }
        st.stored_real += 1;
        st.hist_real[bin] += 1;
        let sig = r.flags & FLAG_FDR != 0;
        for s in [r.leader, r.target] {
            let e = per.entry(s).or_default();
            e.0 += 1;
            e.1 += r.r.abs() as f64;
            e.2 += usize::from(sig);
            stocks.insert(s);
        }
        if sig {
            sig_pairs.insert((r.leader, r.target));
            let (len, year) = windows.get(r.window as usize).copied().unwrap_or((0, 0));
            *sig_by.entry((r.h1 as usize, r.h2 as usize, len, year, r.lag as usize / (r.h2 as usize).max(1))).or_default() += 1;
        }
    }
    for (&(h1, h2, len, year, off), &c) in &sig_by {
        by_h.entry((h1, h2)).or_insert_with(Tally::default).significant += c;
        by_len.entry(len).or_insert_with(Tally::default).significant += c;
        by_year.entry(year).or_insert_with(Tally::default).significant += c;
        by_off.entry(off).or_insert_with(Tally::default).significant += c;
        st.lagged.significant += c;
    }
    st.significant_pairs = sig_pairs.len();
    st.stocks = stocks.len();
    st.significant_stocks = per.values().filter(|p| p.2 > 0).count();
    st.per_stock = per
        .into_iter()
        .map(|(id, (n, s, sig))| (store.symbols.get(id as usize).map(|x| x.ticker.clone()).unwrap_or_default(), n, s / n.max(1) as f64, sig))
        .collect();
    st.per_stock.sort_by(|a, b| b.1.cmp(&a.1));
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
    /// "full" or "smoke test".
    pub kind: String,
    pub started_unix: u64,
    pub finished_unix: Option<u64>,
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
    pub in_use: bool,
}

/// Every census run under the data directory, newest first.
pub fn run_history() -> Vec<RunInfo> {
    let root = c2_data::data_dir().join("census");
    let latest = crate::run::latest_run();
    let update_log = std::fs::read_to_string(c2_data::data_dir().join("update.log")).unwrap_or_default();
    let mut out: Vec<RunInfo> = std::fs::read_dir(&root)
        .map(|it| it.flatten().map(|e| e.path()).filter(|p| p.join("manifest.json").exists()).collect::<Vec<_>>())
        .unwrap_or_default()
        .par_iter()
        .map(|dir| {
            let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            let m: serde_json::Value = std::fs::read_to_string(dir.join("manifest.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
            let s: serde_json::Value = std::fs::read_to_string(dir.join("status.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
            let config = m["config"].as_str().unwrap_or("");
            let nulls: f64 = config.split("nulls=").nth(1).and_then(|v| v.split(' ').next()).and_then(|v| v.parse().ok()).unwrap_or(3.0);
            let mut state = s["state"].as_str().unwrap_or("not started").to_string();
            // "running" in status.json with no live c2 process means it crashed or was killed.
            let alive = std::fs::read_to_string(dir.join("pid"))
                .ok()
                .and_then(|p| p.trim().parse::<u32>().ok())
                .and_then(|pid| std::fs::read_link(format!("/proc/{pid}/exe")).ok())
                .is_some_and(|exe| exe.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("c2")));
            if state == "running" && !alive {
                state = "interrupted".into();
            }
            // Sessions: from a "plan:" line to the last timestamp before the next one.
            let (mut seconds, mut sessions, mut start, mut last) = (0.0, 0, None::<f64>, 0.0);
            let mut prev = 0.0;
            let mut wrap = 0.0;
            for line in std::fs::read_to_string(dir.join("run.log")).unwrap_or_default().lines() {
                let Some(ts) = line.get(1..9) else { continue };
                let f: Vec<f64> = ts.split(':').filter_map(|v| v.parse().ok()).collect();
                if f.len() != 3 {
                    continue;
                }
                let mut t = f[0] * 3600.0 + f[1] * 60.0 + f[2] + wrap;
                if t < prev {
                    wrap += 86400.0;
                    t += 86400.0;
                }
                prev = t;
                if line.contains("] plan:") {
                    if let Some(s0) = start {
                        seconds += last - s0;
                    }
                    start = Some(t);
                    sessions += 1;
                }
                last = t;
            }
            if let Some(s0) = start {
                seconds += last - s0;
            }
            let mut info = RunInfo {
                kind: if config.contains("full_history=true") { "full".into() } else { "smoke test".into() },
                started_unix: m["created_unix"].as_u64().unwrap_or(0),
                finished_unix: (state == "finished").then(|| s["updated_unix"].as_u64()).flatten(),
                fraction: s["fraction_done"].as_f64().unwrap_or(0.0),
                state,
                seconds,
                sessions,
                data_last_day: std::fs::read_to_string(dir.join("windows.csv")).ok().and_then(|t| t.lines().last().and_then(|l| l.split(',').nth(3)).map(String::from)).unwrap_or_default(),
                from_update: update_log.contains(&format!("running census {name} ")),
                in_use: name == latest,
                name,
                ..Default::default()
            };
            for e in std::fs::read_dir(dir.join("tasks")).into_iter().flatten().flatten() {
                if e.path().extension().and_then(|x| x.to_str()) != Some("csv") {
                    continue;
                }
                for line in std::fs::read_to_string(e.path()).unwrap_or_default().lines().skip(1) {
                    let f: Vec<f64> = line.split(',').take(6).map(|v| v.parse().unwrap_or(0.0)).collect();
                    if f.len() < 6 {
                        continue;
                    }
                    let (n, obs) = (f[4], f[5]);
                    let pairs = n * (n - 1.0).max(0.0);
                    info.correlations_real += pairs;
                    info.correlations_null += pairs * nulls;
                    let np = crate::run::pad_width(n as usize) as f64;
                    info.multiply_adds += np * np * ((obs as usize).div_ceil(32) * 32) as f64 * (1.0 + nulls);
                }
            }
            info
        })
        .collect();
    out.sort_by(|a, b| b.started_unix.cmp(&a.started_unix));
    out
}
