//! The holdout test: treat the last year as the future, predict it only from census correlations
//! found in windows that ended before it, and trade $10,000 on the predictions against SPY, QQQ
//! and random guessers.
//!
//! Timing: at the close of day d the account picks stocks to hold over day d+1. A stored pair
//! (leader i, target j, leader horizon h1, lag k, target horizon 1) says the leader's h1-day
//! move ending at day g predicts the target's move on day g + k. For day d+1 that is the leader's
//! move ending at g = d + 1 - k <= d (k >= 1), so only data through day d is used.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use c2_data::Store;
use rayon::prelude::*;

use crate::analyze::{check_plan, visit_task};
use crate::meta::RunMeta;
use crate::run::{FLAG_NULL, Plan};

/// Which signal makes the predictions.
#[derive(Clone, Copy, Debug)]
pub enum Family {
    /// Family A: stored census lead-lag pairs (one-day targets).
    CensusPairs,
    /// Positive control (family E): own-stock short-term reversal. Score = minus the stock's own
    /// log return over the last `lookback` days. A broad, documented anomaly; if the harness can
    /// find nothing here either, its other null results say less.
    Reversal { lookback: usize, window: usize },
    /// Family C: peer ("connected-firm") momentum. Each stock's peers are its `peers` strongest
    /// same-day correlates over the `window` trading days before the holdout (the census showed
    /// these persist); its score is the peers' mean log return over the last `lookback` days.
    PeerMomentum { peers: usize, lookback: usize, window: usize },
    /// Family B: cluster lead-lag (Bennett et al. 2022, `cluster`). Groups of stocks that move
    /// first and groups that follow, refit every `refit` days on the year before; each stock's
    /// score is its group's forecast for the next day.
    ClusterLeadLag(crate::cluster::Settings),
}

pub struct Options {
    pub family: Family,
    /// Trading days between rebalances (picks are held in between).
    pub rebalance: usize,
    pub start: i32,
    /// First day after the holdout (exclusive); `i32::MAX` runs to the end of the data.
    pub end: i32,
    pub picks: usize,
    pub costs_bps: Vec<f64>,
    pub random_runs: usize,
    pub cash: f64,
    /// Use every pre-holdout window instead of only the latest window of each length.
    pub all_windows: bool,
}

pub struct Outcome {
    pub label: String,
    pub final_value: f64,
}

pub struct Report {
    pub first_day: i32,
    pub last_day: i32,
    pub days: usize,
    /// Census pairs used. 0 for the census-pairs family means it had none for this holdout (no
    /// usable correlation period of the needed kind ended before it): nothing was simulated, so
    /// `strategy` and `curve` are empty and the prediction statistics are NaN; the benchmarks
    /// still hold. Callers report that as "no census pairs", not as a failure.
    pub pairs: usize,
    pub windows_used: usize,
    /// Census pairs only: rolling correlation periods that ended before the holdout but that
    /// today's data doesn't cover exactly (`Plan::period_available`), so they were left out
    /// (with the latest periods of each length, the latest covered one is used instead).
    pub periods_unavailable: usize,
    /// Census pairs only: stored pairs from the periods used that were left out because a stock
    /// of the pair is not in today's data (`Plan::missing_stocks`).
    pub pairs_skipped: usize,
    pub candidates: usize,
    pub benchmarks: Vec<Outcome>,
    /// (cost bps, strategy final, random median, share of random >= strategy)
    pub strategy: Vec<(f64, f64, f64, f64)>,
    pub hit_rate: f64,
    pub hits: usize,
    pub ic_mean: f64,
    pub ic_t: f64,
    /// Long-short diagnostic: each day long the top 10% of scores, short the bottom 10%, equal
    /// weight. Mean daily return (fraction), annualized Sharpe, t-stat, and mean daily turnover
    /// of the two books (fraction of gross replaced per day).
    pub ls_mean: f64,
    pub ls_sharpe: f64,
    pub ls_t: f64,
    pub ls_turnover: f64,
    /// Daily account values for plotting: (day, picks at the first cost level, SPY, QQQ,
    /// random guessers' median path at that cost).
    pub curve: Vec<(i32, f64, f64, f64, f64)>,
}

struct Pair {
    leader: usize,
    target: usize,
    h1: usize,
    lag: usize,
    r: f64,
}

fn rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn rank(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    for (k, &i) in idx.iter().enumerate() {
        r[i] = k as f64;
    }
    r
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    if saa > 0.0 && sbb > 0.0 { sab / (saa * sbb).sqrt() } else { 0.0 }
}

/// The holdout test on the census run in `dir`. For census pairs, the tasks (what each results
/// file holds) come from the run itself; `plan` supplies the census variable, the period rows and
/// the stock numbering, and must fit the run (`Plan::for_run` does; otherwise this refuses).
pub fn holdout(store: &Store, plan: &Plan, dir: &Path, opt: &Options) -> Result<Report> {
    let var = &plan.var;
    let h0 = var.days.partition_point(|&d| d < opt.start);
    // Rows [0, t) end at the holdout's last day; nothing after it is used.
    let t = var.days.partition_point(|&d| d < opt.end);
    anyhow::ensure!(h0 > 0 && h0 + 2 < t, "holdout outside the data");
    // Decision days d = h0-1 .. t-2 (hold over d+1 in the holdout); scores on rebalance days.
    let days: Vec<usize> = (h0 - 1..t - 1).collect();
    let rebalance = opt.rebalance.max(1);
    let ((candidates, scores, pairs_used, windows_used), skipped): (Scores, Skipped) = match opt.family {
        Family::CensusPairs => census_scores(store, plan, dir, opt, h0, t, &days, rebalance)?,
        Family::PeerMomentum { peers, lookback, window } => (peer_scores(var, h0, t, &days, rebalance, peers, lookback, window), Skipped::default()),
        Family::Reversal { lookback, window } => (reversal_scores(var, h0, t, &days, rebalance, lookback, window), Skipped::default()),
        Family::ClusterLeadLag(s) => (crate::cluster::scores(store, var, &days, rebalance, &s)?, Skipped::default()),
    };
    // Benchmarks over the same days (close of the day before the holdout to the last close).
    let bench = |ticker: &str| -> Option<f64> {
        let id = store.id(ticker)?;
        let (dates, close) = (store.dates(id), store.close(id));
        let price = |day: i32| dates.binary_search(&day).ok().map(|k| close[k] as f64);
        Some(opt.cash * price(var.days[t - 1])? / price(var.days[h0 - 1])?)
    };
    let mut benchmarks = Vec::new();
    for (label, ticker) in [("Buy and hold SPY (S&P 500)", "SPY"), ("Buy and hold QQQ (Nasdaq-100)", "QQQ")] {
        if let Some(v) = bench(ticker) {
            benchmarks.push(Outcome { label: label.into(), final_value: v });
        }
    }
    if matches!(opt.family, Family::CensusPairs) && pairs_used == 0 {
        // No census pairs: a result in its own right (e.g. an early walk-forward year, before any
        // correlation period of the needed kind ended), not a failure.
        return Ok(Report {
            first_day: var.days[h0],
            last_day: var.days[t - 1],
            days: days.len(),
            pairs: 0,
            windows_used,
            periods_unavailable: skipped.periods,
            pairs_skipped: skipped.pairs,
            candidates: 0,
            benchmarks,
            strategy: Vec::new(),
            hit_rate: f64::NAN,
            hits: 0,
            ic_mean: f64::NAN,
            ic_t: f64::NAN,
            ls_mean: f64::NAN,
            ls_sharpe: f64::NAN,
            ls_t: f64::NAN,
            ls_turnover: f64::NAN,
            curve: Vec::new(),
        });
    }
    let nc = candidates.len();
    anyhow::ensure!(nc >= 2 * opt.picks, "only {nc} stocks can be picked (history before the holdout too short?)");
    let vt = var.t;
    let ret = |j: usize, d: usize| -> f64 {
        let (a, b) = (var.close[j * vt + d], var.close[j * vt + d + 1]);
        if a.is_finite() && b.is_finite() && a > 0.0 { (b / a - 1.0) as f64 } else { 0.0 }
    };
    // Forward outcomes over a holding period (d, d + h]: raw return and census-variable sum.
    let fwd_ret = |j: usize, d: usize, h: usize| -> f64 {
        let e = (d + h).min(t - 1);
        let (a, b) = (var.close[j * vt + d], var.close[j * vt + e]);
        if a.is_finite() && b.is_finite() && a > 0.0 { (b / a - 1.0) as f64 } else { 0.0 }
    };
    let fwd_resid = |j: usize, d: usize, h: usize| -> f32 {
        let e = (d + h).min(t - 1);
        (d + 1..=e).map(|r| var.v[j * vt + r]).filter(|v| v.is_finite()).sum::<f32>()
            + if (d + 1..=e).all(|r| !var.v[j * vt + r].is_finite()) { f32::NAN } else { 0.0 }
    };

    // Strategy picks: on rebalance days the top `picks` predicted gainers with a positive score,
    // equal weight; held unchanged until the next rebalance.
    let mut picks: Vec<Vec<usize>> = Vec::with_capacity(days.len());
    for s in &scores {
        match s {
            Some(s) => {
                let mut order: Vec<usize> = (0..nc).filter(|&k| s[k] > 0.0).collect();
                order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
                order.truncate(opt.picks);
                picks.push(order);
            }
            None => picks.push(picks.last().cloned().unwrap_or_default()),
        }
    }
    let simulate = |chosen: &dyn Fn(usize) -> Vec<usize>, cost: f64| -> f64 {
        let (mut value, mut prev): (f64, Vec<usize>) = (opt.cash, Vec::new());
        for (k, &d) in days.iter().enumerate() {
            let now = chosen(k);
            let w = 1.0 / opt.picks as f64;
            let gross: f64 = now.iter().map(|&c| w * ret(candidates[c], d)).sum();
            let kept = now.iter().filter(|c| prev.contains(c)).count();
            let turnover = w * ((now.len() - kept) + (prev.len() - kept)) as f64;
            value *= 1.0 + gross - turnover * cost / 1e4;
            prev = now;
        }
        value
    };

    let path = |chosen: &dyn Fn(usize) -> Vec<usize>, cost: f64| -> Vec<f64> {
        let (mut value, mut prev): (f64, Vec<usize>) = (opt.cash, Vec::new());
        let mut out = Vec::with_capacity(days.len());
        for (k, &d) in days.iter().enumerate() {
            let now = chosen(k);
            let w = 1.0 / opt.picks as f64;
            let gross: f64 = now.iter().map(|&c| w * ret(candidates[c], d)).sum();
            let kept = now.iter().filter(|c| prev.contains(c)).count();
            let turnover = w * ((now.len() - kept) + (prev.len() - kept)) as f64;
            value *= 1.0 + gross - turnover * cost / 1e4;
            prev = now;
            out.push(value);
        }
        out
    };

    let mut strategy = Vec::new();
    let mut median_path: Vec<f64> = Vec::new();
    let mut paths_for_median: Vec<Vec<f64>> = Vec::new();
    for &c in &opt.costs_bps {
        let s = simulate(&|k| picks[k].clone(), c);
        // Random guessers with the strategy's turnover: each day they hold as many stocks as the
        // strategy and replace exactly as many as it did, at random. Otherwise costs alone would
        // separate them (random daily picks turn over ~100% a day).
        let mut finals: Vec<f64> = (0..opt.random_runs)
            .into_par_iter()
            .map(|run| {
                let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (run as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                let mut chosen: Vec<Vec<usize>> = Vec::with_capacity(days.len());
                let mut set: Vec<usize> = Vec::new();
                for k in 0..days.len() {
                    let target = picks[k].len().min(nc);
                    let replaced = if k == 0 { target } else { picks[k].iter().filter(|c| !picks[k - 1].contains(c)).count() };
                    let keep = target.saturating_sub(replaced).min(set.len());
                    // Drop random members down to `keep`, then add random non-members up to `target`.
                    while set.len() > keep {
                        let i = (rng(&mut state) % set.len() as u64) as usize;
                        set.swap_remove(i);
                    }
                    while set.len() < target {
                        let c = (rng(&mut state) % nc as u64) as usize;
                        if !set.contains(&c) {
                            set.push(c);
                        }
                    }
                    chosen.push(set.clone());
                }
                path(&|k| chosen[k].clone(), c)
            })
            .collect::<Vec<Vec<f64>>>()
            .into_iter()
            .enumerate()
            .map(|(run, p)| {
                if median_path.is_empty() {
                    if run == 0 {
                        paths_for_median.clear();
                    }
                    paths_for_median.push(p.clone());
                }
                *p.last().unwrap_or(&opt.cash)
            })
            .collect();
        if median_path.is_empty() && !paths_for_median.is_empty() {
            median_path = (0..days.len())
                .map(|k| {
                    let mut col: Vec<f64> = paths_for_median.iter().map(|p| p[k]).collect();
                    col.sort_by(|a, b| a.total_cmp(b));
                    col[col.len() / 2]
                })
                .collect();
        }
        finals.sort_by(|a, b| a.total_cmp(b));
        let median = finals[finals.len() / 2];
        let share = finals.iter().filter(|&&f| f >= s).count() as f64 / finals.len() as f64;
        strategy.push((c, s, median, share));
    }

    let hold_all: f64 = candidates
        .iter()
        .map(|&j| days.iter().map(|&d| 1.0 + ret(j, d)).product::<f64>())
        .sum::<f64>()
        / nc.max(1) as f64;
    benchmarks.push(Outcome { label: format!("Buy and hold all {nc} candidates"), final_value: opt.cash * hold_all });

    // Daily curves for plotting: picks (first cost level), SPY, QQQ, random median.
    let first_cost = opt.costs_bps.first().copied().unwrap_or(0.0);
    let pick_path = path(&|k| picks[k].clone(), first_cost);
    let bench_path = |ticker: &str| -> Vec<f64> {
        let Some(id) = store.id(ticker) else { return vec![f64::NAN; days.len()] };
        let (dates, close) = (store.dates(id), store.close(id));
        let price = |day: i32| dates.binary_search(&day).ok().map(|k| close[k] as f64);
        let base = price(var.days[days[0]]);
        days.iter().map(|&d| match (base, price(var.days[d + 1])) { (Some(b), Some(p)) => opt.cash * p / b, _ => f64::NAN }).collect()
    };
    let (spy_path, qqq_path) = (bench_path("SPY"), bench_path("QQQ"));
    let curve: Vec<(i32, f64, f64, f64, f64)> = days
        .iter()
        .enumerate()
        .map(|(k, &d)| (var.days[d + 1], pick_path[k], spy_path[k], qqq_path[k], *median_path.get(k).unwrap_or(&f64::NAN)))
        .collect();

    // Prediction quality on rebalance days, over the holding period: picks beating the market,
    // rank IC of scores vs forward census-variable sums, and a long-short decile diagnostic.
    // Stocks without a score that day (NaN: outside a cluster fit's universe) are left out.
    let (mut hits, mut tries) = (0usize, 0usize);
    let mut ics = Vec::new();
    let mut ls = Vec::new();
    let mut turn = Vec::new();
    let mut prev_books: Option<(Vec<usize>, Vec<usize>)> = None;
    for (k, &d) in days.iter().enumerate() {
        let Some(sc) = &scores[k] else { continue };
        for &c in &picks[k] {
            let v = fwd_resid(candidates[c], d, rebalance);
            if v.is_finite() {
                tries += 1;
                hits += usize::from(v > 0.0);
            }
        }
        let (a, b): (Vec<f64>, Vec<f64>) = (0..nc)
            .filter_map(|c| {
                let v = fwd_resid(candidates[c], d, rebalance);
                (v.is_finite() && sc[c].is_finite()).then_some((sc[c], v as f64))
            })
            .unzip();
        if a.len() > 10 {
            ics.push(corr(&rank(&a), &rank(&b)));
        }
        let mut order: Vec<usize> = (0..nc).filter(|&c| sc[c].is_finite()).collect();
        order.sort_by(|&x, &y| sc[y].total_cmp(&sc[x]));
        let (scored, q) = (order.len(), (order.len() / 10).max(1));
        if scored < 2 * q {
            continue;
        }
        let (long, short) = (order[..q].to_vec(), order[scored - q..].to_vec());
        let mean = |set: &[usize]| set.iter().map(|&c| fwd_ret(candidates[c], d, rebalance)).sum::<f64>() / set.len() as f64;
        ls.push(mean(&long) - mean(&short));
        if let Some((pl, ps)) = &prev_books {
            let changed = long.iter().filter(|c| !pl.contains(c)).count() + short.iter().filter(|c| !ps.contains(c)).count();
            turn.push(changed as f64 / (2 * q) as f64);
        }
        prev_books = Some((long, short));
    }
    let ls_mean = ls.iter().sum::<f64>() / ls.len().max(1) as f64;
    let ls_sd = (ls.iter().map(|x| (x - ls_mean).powi(2)).sum::<f64>() / ls.len().max(1) as f64).sqrt();
    let ls_turnover = turn.iter().sum::<f64>() / turn.len().max(1) as f64;
    let ic_mean = ics.iter().sum::<f64>() / ics.len().max(1) as f64;
    let ic_sd = (ics.iter().map(|x| (x - ic_mean).powi(2)).sum::<f64>() / ics.len().max(1) as f64).sqrt();
    Ok(Report {
        first_day: var.days[h0],
        last_day: var.days[t - 1],
        days: days.len(),
        pairs: pairs_used,
        windows_used,
        periods_unavailable: skipped.periods,
        pairs_skipped: skipped.pairs,
        candidates: nc,
        benchmarks,
        strategy,
        hit_rate: hits as f64 / tries.max(1) as f64,
        hits: tries,
        ic_mean,
        ic_t: ic_mean / ic_sd.max(1e-12) * (ics.len() as f64).sqrt(),
        // Per holding period: mean return as a daily equivalent, Sharpe annualized.
        ls_mean: ls_mean / rebalance as f64,
        ls_sharpe: ls_mean / ls_sd.max(1e-12) * (252.0 / rebalance as f64).sqrt(),
        ls_t: ls_mean / ls_sd.max(1e-12) * (ls.len() as f64).sqrt(),
        ls_turnover,
        curve,
    })
}

type Scores = (Vec<usize>, Vec<Option<Vec<f64>>>, usize, usize);

/// What census-pair scoring left out: correlation periods today's data doesn't cover, and
/// stored pairs with a stock today's data doesn't have.
#[derive(Clone, Copy, Debug, Default)]
struct Skipped {
    periods: usize,
    pairs: usize,
}

/// Family A scores: stored census pairs from windows that ended before the holdout. The run's
/// own task list says what each results file holds; the plan must fit the run (`check_plan`),
/// so its windows are the run's and `column_of_run_id` turns record ids into its columns.
/// Periods today's data doesn't cover exactly and pairs with a stock it no longer has are
/// skipped and counted.
#[allow(clippy::too_many_arguments)]
fn census_scores(store: &Store, plan: &Plan, dir: &Path, opt: &Options, h0: usize, t: usize, days: &[usize], rebalance: usize) -> Result<(Scores, Skipped)> {
    let var = &plan.var;
    let meta = RunMeta::load(dir)?;
    check_plan(plan, &meta, Some(store))?;
    // Windows that ended before the holdout (its first day): the latest of each length, or all
    // of them, among those today's data covers. The others are only counted, by their dates.
    let first_day = var.days.get(h0).copied().unwrap_or(i32::MAX);
    let periods = plan.windows.iter().filter(|w| w.length > 0 && meta.windows[w.id].last < first_day).count();
    let before: Vec<usize> = plan.windows.iter().filter(|w| w.length > 0 && plan.period_available(w.id) && w.end <= h0).map(|w| w.id).collect();
    let mut skipped = Skipped { periods: periods - before.len(), pairs: 0 };
    let mut use_window = vec![false; plan.windows.len()];
    if opt.all_windows {
        for &w in &before {
            use_window[w] = true;
        }
    } else {
        let mut latest: HashMap<usize, usize> = HashMap::new();
        for &w in &before {
            let e = latest.entry(plan.windows[w].length).or_insert(w);
            if plan.windows[w].end > plan.windows[*e].end {
                *e = w;
            }
        }
        for &w in latest.values() {
            use_window[w] = true;
        }
    }
    let windows_used = use_window.iter().filter(|&&u| u).count();
    // Real stored pairs whose B move length equals the holding period, at predictive offsets,
    // in task order, then file order (the order the scores are summed in).
    let h2 = rebalance;
    anyhow::ensure!(
        meta.cfg.horizons.contains(&h2),
        "census {} has no B move length of {h2} days (the rebalance period must be one of {:?})",
        meta.name,
        meta.cfg.horizons
    );
    let mut pairs = Vec::new();
    for task in meta.tasks()?.iter().filter(|t| t.h2 == h2 && t.length > 0) {
        // `visit_task` passes only records of the task's own periods, which `check_plan` made
        // the plan's, so `use_window` covers them.
        visit_task(&meta, task, |r| {
            if r.flags & FLAG_NULL != 0 || r.lag == 0 || !use_window[r.window as usize] {
                return;
            }
            match (plan.column_of_run_id(r.leader), plan.column_of_run_id(r.target)) {
                (Some(i), Some(j)) => pairs.push(Pair { leader: i, target: j, h1: task.h1, lag: r.lag as usize, r: r.r as f64 }),
                _ => skipped.pairs += 1,
            }
        })?;
    }
    let mut candidates: Vec<usize> = pairs.iter().map(|p| p.target).collect();
    candidates.sort_unstable();
    candidates.dedup();
    let cand_pos: HashMap<usize, usize> = candidates.iter().enumerate().map(|(k, &c)| (c, k)).collect();
    // Leader h-day sums of the census variable, by prefix sums (only leaders that are used).
    let vt = var.t;
    let mut leaders: Vec<usize> = pairs.iter().map(|p| p.leader).collect();
    leaders.sort_unstable();
    leaders.dedup();
    let prefix_of: HashMap<usize, Vec<f64>> = leaders
        .par_iter()
        .map(|&j| {
            let mut p = vec![0.0; t + 1];
            for d in 0..t {
                let v = var.v[j * vt + d];
                p[d + 1] = p[d] + if v.is_finite() { v as f64 } else { 0.0 };
            }
            (j, p)
        })
        .collect();
    let leader_move = |i: usize, end: usize, h: usize| {
        let p = &prefix_of[&i];
        (p[end + 1] - p[end + 1 - h]) / (h as f64).sqrt()
    };
    let nc = candidates.len();
    let scores = days
        .par_iter()
        .enumerate()
        .map(|(k, &d)| {
            (k % rebalance == 0).then(|| {
                let mut s = vec![0.0; nc];
                // The target's h2-day return starting at d+1 is (g + lag - h2, g + lag] with
                // g = d + h2 - lag; lag >= h2, so the leader's move ends at or before d.
                for p in &pairs {
                    let g = d + h2 - p.lag;
                    if g + 1 >= p.h1 {
                        s[cand_pos[&p.target]] += p.r * leader_move(p.leader, g, p.h1);
                    }
                }
                s
            })
        })
        .collect();
    Ok(((candidates, scores, pairs.len(), windows_used), skipped))
}

/// Family C scores: peers from same-day correlation of the census variable over the `window`
/// days before the holdout; score = mean peer log return over the last `lookback` days.
#[allow(clippy::too_many_arguments)]
fn peer_scores(var: &crate::variable::Variable, h0: usize, t: usize, days: &[usize], rebalance: usize, peers: usize, lookback: usize, window: usize) -> Scores {
    let vt = var.t;
    let w0 = h0.saturating_sub(window);
    let candidates = liquid_universe(var, h0, window);
    let (n, w) = (candidates.len(), h0 - w0);
    // Standardized panel (missing = 0) and the same-day correlation matrix.
    let mut z = vec![0f32; w * n];
    z.par_chunks_mut(w).zip(candidates.par_iter()).for_each(|(col, &j)| {
        for (r, out) in col.iter_mut().enumerate() {
            let v = var.v[j * vt + w0 + r];
            *out = if v.is_finite() { v } else { 0.0 };
        }
        let mean = col.iter().sum::<f32>() / w as f32;
        let sd = (col.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / w as f32).sqrt().max(1e-6);
        for x in col.iter_mut() {
            *x = (*x - mean) / sd;
        }
    });
    let mut c = vec![0f32; n * n];
    c2_engine::cpu::xprod_f32(&z, &z, w, n, w, &mut c);
    let peer_sets: Vec<Vec<usize>> = (0..n)
        .into_par_iter()
        .map(|j| {
            let mut order: Vec<usize> = (0..n).filter(|&i| i != j && c[j * n + i] > 0.0).collect();
            order.sort_by(|&a, &b| c[j * n + b].total_cmp(&c[j * n + a]));
            order.truncate(peers);
            order
        })
        .collect();
    let log_ret = |j: usize, d: usize| -> Option<f64> {
        let (a, b) = (var.close[j * vt + d.checked_sub(lookback)?], var.close[j * vt + d]);
        (a.is_finite() && b.is_finite() && a > 0.0).then(|| (b as f64 / a as f64).ln())
    };
    let scores = days
        .par_iter()
        .enumerate()
        .map(|(k, &d)| {
            (k % rebalance == 0 && d < t).then(|| {
                peer_sets
                    .iter()
                    .map(|ps| {
                        let v: Vec<f64> = ps.iter().filter_map(|&p| log_ret(candidates[p], d)).collect();
                        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
                    })
                    .collect()
            })
        })
        .collect();
    (candidates, scores, n * peers, 1)
}

/// Stocks with >= 90% of days with a census value over the `window` days before `h0`, and median
/// price >= $5 and median dollar volume >= $5M over the last year before it.
fn liquid_universe(var: &crate::variable::Variable, h0: usize, window: usize) -> Vec<usize> {
    let vt = var.t;
    let w0 = h0.saturating_sub(window);
    let tail = (h0 - w0).min(crate::variable::LOOKBACK);
    (0..var.n)
        .filter(|&j| {
            let v = &var.v[j * vt + w0..j * vt + h0];
            if h0 == w0 || v.iter().filter(|x| x.is_finite()).count() * 10 < (h0 - w0) * 9 {
                return false;
            }
            let mut pc: Vec<f32> = var.close[j * vt + h0 - tail..j * vt + h0].iter().copied().filter(|x| x.is_finite()).collect();
            let mut pv: Vec<f32> = var.dollar_volume[j * vt + h0 - tail..j * vt + h0].iter().copied().filter(|x| x.is_finite()).collect();
            let med = |v: &mut Vec<f32>| {
                let mid = v.len() / 2;
                if v.is_empty() { 0.0 } else { *v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1 }
            };
            med(&mut pc) >= crate::variable::MIN_PRICE && med(&mut pv) >= crate::variable::MIN_DOLLAR_VOLUME
        })
        .collect()
}

/// Positive-control scores: minus each stock's own log return over the last `lookback` days.
fn reversal_scores(var: &crate::variable::Variable, h0: usize, t: usize, days: &[usize], rebalance: usize, lookback: usize, window: usize) -> Scores {
    let vt = var.t;
    let candidates = liquid_universe(var, h0, window);
    let scores = days
        .par_iter()
        .enumerate()
        .map(|(k, &d)| {
            (k % rebalance == 0 && d < t).then(|| {
                candidates
                    .iter()
                    .map(|&j| match d.checked_sub(lookback) {
                        Some(a) => {
                            let (p0, p1) = (var.close[j * vt + a], var.close[j * vt + d]);
                            if p0.is_finite() && p1.is_finite() && p0 > 0.0 { -((p1 as f64 / p0 as f64).ln()) } else { 0.0 }
                        }
                        None => 0.0,
                    })
                    .collect()
            })
        })
        .collect();
    let n = candidates.len();
    (candidates, scores, n, 0)
}

/// Picks for the next holding period as of the last day in the data: what the strategy would buy
/// at today's close. Uses only data through that day (every census window ends at or before it).
/// Returns (ticker, score) for the top `picks` positive scores.
pub fn current_picks(store: &Store, plan: &Plan, dir: &Path, family: Family, rebalance: usize, picks: usize) -> Result<Vec<(String, f64)>> {
    let var = &plan.var;
    let t = var.t;
    let opt = Options { family, rebalance, start: var.days[t - 1], end: i32::MAX, picks, costs_bps: vec![0.0], random_runs: 0, cash: 10_000.0, all_windows: false };
    let days = [t - 1];
    let (candidates, scores, _, _) = match family {
        Family::CensusPairs => census_scores(store, plan, dir, &opt, t, t, &days, rebalance)?.0,
        Family::PeerMomentum { peers, lookback, window } => peer_scores(var, t, t, &days, rebalance, peers, lookback, window),
        Family::Reversal { lookback, window } => reversal_scores(var, t, t, &days, rebalance, lookback, window),
        Family::ClusterLeadLag(s) => crate::cluster::scores(store, var, &days, rebalance, &s)?,
    };
    let s = scores.into_iter().next().flatten().unwrap_or_default();
    let mut order: Vec<usize> = (0..s.len()).filter(|&k| s[k] > 0.0).collect();
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    order.truncate(picks);
    Ok(order.into_iter().map(|k| (store.symbols[var.ids[candidates[k]]].ticker.clone(), s[k])).collect())
}
