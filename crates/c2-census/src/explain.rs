//! Step-by-step recomputation of one stored relationship from prices, for the GUI's explainer.
//!
//! Nothing here feeds the census. The daily standardization is a line-for-line copy of
//! `variable::production` that also keeps its intermediate values (beta, residual, trailing mean
//! and sd), and the explainer reports how far its result is from the census's own values, so a
//! mismatch would show rather than hide.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::analyze::{Words, read_records, words};
use crate::grid;
use crate::run::{FLAG_FDR, FLAG_NULL, FLAG_TOP, Plan, Record};
use crate::variable::{CLIP, LOOKBACK, MIN_OBS};

pub struct Request {
    pub leader: String,
    pub follower: String,
    pub h1: usize,
    pub h2: usize,
    pub lag: usize,
    /// Census window (correlation period) id.
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
    /// (resid - mean) / sd clipped at +-5, recomputed here, and the census's own value.
    pub v: Vec<f32>,
    pub v_census: Vec<f32>,
    /// Largest |v - v_census| over the shown rows (0 means identical).
    pub max_diff: f32,
    pub in_universe: bool,
}

pub struct Obs {
    /// Row (in the shown rows) where the leader's move ends; the follower's ends `lag` rows later.
    pub a_end: usize,
    pub b_end: usize,
    /// Raw log price moves.
    pub a_move: f64,
    pub b_move: f64,
    /// Census units: sum of daily census values over the move length / sqrt(length).
    pub x: f64,
    pub y: f64,
}

pub struct Stored {
    pub record: Record,
    pub top: bool,
    pub significant: bool,
}

/// The matrix the relationship was tested in: every universe stock against every other at this
/// move-length pair, offset and period (from the run's summary file).
#[derive(Default)]
pub struct Matrix {
    pub found: bool,
    pub stocks: usize,
    pub obs: usize,
    /// Real pairs at |z| >= 3, 4, 5 and the average of the sign-flip nulls.
    pub real: [f64; 3],
    pub null: [f64; 3],
    /// |r| above which pairs passed the false-discovery test, if any did.
    pub fdr_r: Option<f64>,
}

pub struct Explained {
    pub h1: usize,
    pub h2: usize,
    pub lag: usize,
    pub window: usize,
    pub period_first: i32,
    pub period_last: i32,
    pub period_days: usize,
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
    /// r x sd(follower raw moves) / sd(leader raw moves): % move per 1%, as in the timeline.
    pub move_per_pct: f64,
    pub stored: Option<Stored>,
    pub matrix: Matrix,
    /// Pairs tested in the matrix and how many chance alone would give at |z| >= this one.
    pub pairs_tested: f64,
    pub chance_analytic: f64,
    pub words: Option<Words>,
}

/// Production standardization for variable column `j`, keeping the intermediates.
fn day_steps(plan: &Plan, j: usize) -> DaySteps {
    let var = &plan.var;
    let t = var.t;
    let c = var.col(&var.close, j);
    let market = &var.market;
    let mut r = vec![f32::NAN; t];
    for d in 1..t {
        r[d] = (c[d] / c[d - 1]).ln();
    }
    let (mut beta, mut resid) = (vec![f64::NAN; t], vec![f64::NAN; t]);
    let (mut sr, mut sm, mut srm, mut smm, mut k) = (0f64, 0f64, 0f64, 0f64, 0usize);
    let valid = |d: usize| r[d].is_finite() && market[d].is_finite();
    for d in 0..t {
        if d >= 1 && valid(d - 1) {
            let (a, b) = (r[d - 1] as f64, market[d - 1]);
            sr += a; sm += b; srm += a * b; smm += b * b; k += 1;
        }
        if d > LOOKBACK && valid(d - 1 - LOOKBACK) {
            let (a, b) = (r[d - 1 - LOOKBACK] as f64, market[d - 1 - LOOKBACK]);
            sr -= a; sm -= b; srm -= a * b; smm -= b * b; k -= 1;
        }
        if k >= MIN_OBS && valid(d) {
            let kf = k as f64;
            let var_m = smm - sm * sm / kf;
            if var_m > 0.0 {
                beta[d] = (srm - sr * sm / kf) / var_m;
                resid[d] = r[d] as f64 - beta[d] * market[d];
            }
        }
    }
    let (mut mean, mut sd, mut v) = (vec![f64::NAN; t], vec![f64::NAN; t], vec![f32::NAN; t]);
    let (mut s, mut ss, mut k) = (0f64, 0f64, 0usize);
    for d in 0..t {
        if d >= 1 && resid[d - 1].is_finite() {
            s += resid[d - 1]; ss += resid[d - 1] * resid[d - 1]; k += 1;
        }
        if d > LOOKBACK && resid[d - 1 - LOOKBACK].is_finite() {
            let e = resid[d - 1 - LOOKBACK];
            s -= e; ss -= e * e; k -= 1;
        }
        if k >= MIN_OBS && resid[d].is_finite() {
            let var = (ss - s * s / k as f64) / k as f64;
            if var > 1e-12 {
                mean[d] = s / k as f64;
                sd[d] = var.sqrt();
                v[d] = ((resid[d] - s / k as f64) / var.sqrt()).clamp(-CLIP as f64, CLIP as f64) as f32;
            }
        }
    }
    DaySteps { close: c.to_vec(), ret: r, beta, resid, mean, sd, v, v_census: var.col(&var.v, j).to_vec(), ..Default::default() }
}

fn column(plan: &Plan, store: &c2_data::Store, ticker: &str) -> Result<usize> {
    let id = store.id(ticker).with_context(|| format!("{ticker} is not in the price store"))?;
    plan.var.ids.iter().position(|&i| i == id).with_context(|| format!("{ticker} is not a common stock the census covers"))
}

/// The task file of a run holding (h1, h2) for a window, found through the summary files (task
/// numbering depends on the run's configuration), and the summary line for the lag.
fn find_matrix(dir: &Path, window: usize, h1: usize, h2: usize, lag: usize) -> Option<(PathBuf, Matrix)> {
    let prefix = format!("{window},{h1},{h2},{lag},");
    for e in std::fs::read_dir(dir.join("tasks")).ok()?.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("csv") {
            continue;
        }
        let text = std::fs::read_to_string(&path).ok()?;
        if let Some(line) = text.lines().find(|l| l.starts_with(&prefix)) {
            let f: Vec<&str> = line.split(',').collect();
            let num = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::NAN);
            // window,h1,h2,lag,stocks,obs,real_z3,real_z4,real_z5,null_z3,null_z4,null_z5,fdr_r,...
            let m = Matrix {
                found: true,
                stocks: num(4) as usize,
                obs: num(5) as usize,
                real: [num(6), num(7), num(8)],
                null: [num(9), num(10), num(11)],
                fdr_r: f.get(12).and_then(|v| v.parse().ok()),
            };
            return Some((path.with_extension("rec"), m));
        }
    }
    None
}

fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.265_512_23 + t * (1.000_023_68 + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06
        + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))))).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

pub fn explain(store: &c2_data::Store, plan: &Plan, dir: &Path, req: &Request) -> Result<Explained> {
    let (h1, h2, lag) = (req.h1, req.h2, req.lag);
    anyhow::ensure!(h1 >= 1 && h2 >= 1, "move lengths start at 1 day");
    anyhow::ensure!(lag >= h2, "the offset must be at least B's move length, or B's move would start before A's ends");
    let w = plan.windows.get(req.window).context("no such correlation period")?;
    let (ja, jb) = (column(plan, store, &req.leader)?, column(plan, store, &req.follower)?);
    anyhow::ensure!(ja != jb, "pick two different stocks");
    let (step, g0) = (h1.max(h2), h1.max(h2) - 1);
    let len = w.end - w.start;
    let m = grid::observations(len, h1, h2, lag);
    anyhow::ensure!(m >= 3, "a {len}-day period has room for only {m} observations at this offset");
    // Shown rows: the day before the period (the first move starts there) through its end.
    let row0 = w.start - 1;
    let shown = |mut s: DaySteps, ticker: &str, j: usize| {
        s.ticker = ticker.to_string();
        s.in_universe = plan.universes[req.window].contains(&j);
        for vec in [&mut s.beta, &mut s.resid, &mut s.mean, &mut s.sd] {
            *vec = vec[row0..w.end].to_vec();
        }
        for vec in [&mut s.close, &mut s.ret, &mut s.v, &mut s.v_census] {
            *vec = vec[row0..w.end].to_vec();
        }
        s.max_diff = s.v.iter().zip(&s.v_census).filter(|(a, b)| a.is_finite() || b.is_finite()).map(|(a, b)| if a.is_finite() && b.is_finite() { (a - b).abs() } else { f32::INFINITY }).fold(0.0, f32::max);
        s
    };
    let a = shown(day_steps(plan, ja), &req.leader, ja);
    let b = shown(day_steps(plan, jb), &req.follower, jb);
    // Observations exactly as the census forms them: missing values count as 0 in the sums.
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
                x: sum(&a.v_census, e, h1),
                y: sum(&b.v_census, e + lag, h2),
            }
        })
        .collect();
    let pearson = |pts: &[(f64, f64)]| {
        let n = pts.len() as f64;
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &(x, y) in pts {
            sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
        }
        c2_engine::pearson(sxy, sx, sy, sxx, syy, n)
    };
    let xy: Vec<(f64, f64)> = obs.iter().map(|o| (o.x, o.y)).collect();
    let r = pearson(&xy);
    let z = r.clamp(-0.999_999, 0.999_999).atanh() * ((m as f64 - 3.0).max(1.0)).sqrt();
    let raw: Vec<(f64, f64)> = obs.iter().map(|o| (o.a_move, o.b_move)).filter(|p| p.0.is_finite() && p.1.is_finite()).collect();
    let r_raw = pearson(&raw);
    let sd = |v: &mut dyn Iterator<Item = f64>| {
        let v: Vec<f64> = v.collect();
        let n = v.len() as f64;
        let mu = v.iter().sum::<f64>() / n;
        (v.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / n).sqrt()
    };
    let move_per_pct = r * sd(&mut raw.iter().map(|p| p.1)) / sd(&mut raw.iter().map(|p| p.0));

    let (path, matrix) = find_matrix(dir, req.window, h1, h2, lag).unwrap_or_default();
    let stored = if matrix.found {
        let (ida, idb) = (plan.var.ids[ja] as u32, plan.var.ids[jb] as u32);
        read_records(&path)?
            .into_iter()
            .find(|rec| rec.window as usize == req.window && rec.lag as usize == lag && rec.leader == ida && rec.target == idb && rec.flags & FLAG_NULL == 0)
            .map(|rec| Stored { top: rec.flags & FLAG_TOP != 0, significant: rec.flags & FLAG_FDR != 0, record: rec })
    } else {
        None
    };
    let universe = plan.universes[req.window].len();
    let pairs_tested = (universe * universe.saturating_sub(1)) as f64;
    let chance_analytic = pairs_tested * erfc(z.abs() / std::f64::consts::SQRT_2);

    // Plain-terms counts on log closes over all variable rows.
    let all = |j: usize| -> Vec<f64> { plan.var.col(&plan.var.close, j).iter().map(|&c| if c > 0.0 { (c as f64).ln() } else { f64::NAN }).collect() };
    let lo = plan.var.days.partition_point(|&d| d < plan.cfg.start);
    let words = words(&all(ja), &all(jb), h1, h2, lag, w.start + g0, m, step, w.start, w.end, lo, plan.var.t, req.threshold);

    Ok(Explained {
        h1,
        h2,
        lag,
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
        move_per_pct,
        stored,
        matrix,
        pairs_tested,
        chance_analytic,
        words,
    })
}

/// The correlation period of `length` trading days (0 = full history) ending on or before `end`,
/// else the earliest one of that length.
pub fn find_window(plan: &Plan, length: usize, end: i32) -> Option<usize> {
    let of_len: Vec<usize> = plan.windows.iter().filter(|w| w.length == length).map(|w| w.id).collect();
    of_len.iter().rev().find(|&&id| plan.var.days[plan.windows[id].end - 1] <= end).or(of_len.first()).copied()
}
