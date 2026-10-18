//! Reading census output, and the replication test: do stored pairs keep their sign in the next
//! window that does not overlap?

use std::collections::HashMap;
use std::path::Path;

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
