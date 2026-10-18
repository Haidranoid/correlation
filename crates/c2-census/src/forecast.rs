//! Price forecasts from a stock's lead-lag relationships.
//!
//! As of a chosen day, every stored relationship "leader L moves first, this stock follows `lag`
//! days later" turns L's moves over the last `lag` days into predicted moves for the days ahead:
//! a leader h1-day move x ending at day g predicts this stock's h2-day move over
//! (g + lag - h2, g + lag] of beta * x, where beta = r * (this stock's volatility / L's volatility)
//! in that relationship's census window. Each prediction is spread evenly over its h2 days; the
//! forecast for a day is the mean over the relationships that cover it, accumulated into a path. Only relationships from census windows that ended on or before
//! the as-of day are used, so a past as-of date can be checked against what actually happened.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use c2_data::{Calendar, Store};
use rayon::prelude::*;

use crate::analyze::read_records;
use crate::run::{FLAG_FDR, FLAG_NULL};

/// A relationship in which the chosen stock follows, with its size in log-return terms.
#[derive(Clone)]
struct Rel {
    leader: u32,
    lag: usize,
    h1: usize,
    h2: usize,
    r: f32,
    beta: f64,
    significant: bool,
    /// Calendar row of the window's last day.
    window_end: usize,
    window: String,
}

pub struct Contribution {
    pub leader: String,
    pub lag: usize,
    pub h1: usize,
    pub h2: usize,
    pub r: f32,
    pub significant: bool,
    pub window: String,
    /// The leader's moves feeding this forecast, summed (in %).
    pub leader_move_pct: f64,
    /// What this relationship alone predicts for the stock over the horizon (in %).
    pub implied_pct: f64,
    /// Its share of the combined forecast at the horizon (percentage points; shares add up to
    /// the forecast's log change).
    pub share_pct: f64,
}

pub struct Skill {
    pub forecasts: usize,
    pub horizon: usize,
    /// Correlation between predicted and actual moves over the horizon.
    pub corr: f64,
    /// Regression slope of actual on predicted: 1 would mean the forecasts are the right size,
    /// 0 that they carry no information.
    pub slope: f64,
    /// Share of forecasts that got the direction right (among non-zero forecasts).
    pub hit_rate: f64,
    pub mean_abs_predicted_pct: f64,
    pub mean_abs_actual_pct: f64,
}

pub struct Forecast {
    pub ticker: String,
    pub as_of: i32,
    pub last_price: f64,
    pub relationships: usize,
    /// Predicted and actual cumulative % change after 1..=horizon trading days (actual is None
    /// beyond the data).
    pub predicted_pct: Vec<f64>,
    pub actual_pct: Vec<Option<f64>>,
    /// Calendar dates of the days ahead, where known.
    pub future_days: Vec<Option<i32>>,
    /// How many relationships say something about each day ahead.
    pub covering: Vec<usize>,
    /// Standard deviation of the stock's daily log returns over the year before the as-of day.
    pub daily_sd: f64,
    pub contributions: Vec<Contribution>,
    pub skill: Option<Skill>,
}

/// Log prices of one symbol on calendar rows `from..to` (NaN where it did not trade).
fn log_prices(store: &Store, cal: &Calendar, id: usize, from: usize, to: usize) -> Vec<f64> {
    let (d, c) = (store.dates(id), store.close(id));
    (from..to)
        .map(|k| d.binary_search(&cal.days[k]).ok().map(|i| c[i] as f64).filter(|p| *p > 0.0).map(f64::ln).unwrap_or(f64::NAN))
        .collect()
}

struct Prepared {
    cal: Calendar,
    /// First calendar row covered by `prices`.
    base: usize,
    /// Log prices by symbol id over rows base.. (target and leaders).
    prices: HashMap<u32, Vec<f64>>,
    rels: Vec<Rel>,
    target: u32,
}

fn prepare(store: &Store, dir: &Path, ticker: &str, significant_only: bool) -> Result<Prepared> {
    let target = store.id(ticker).with_context(|| format!("{ticker} is not in the price store"))? as u32;
    let cal = Calendar::trading_days(store)?;
    let windows: Vec<(usize, usize, String)> = std::fs::read_to_string(dir.join("windows.csv"))?
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let (a, b) = (c2_data::parse_date(f[2])?, c2_data::parse_date(f[3])?);
            let rows = cal.range(a, b);
            Ok((rows.start, rows.end, format!("{}..{}", f[2], f[3])))
        })
        .collect::<Result<_>>()?;
    let base = windows.iter().map(|w| w.0).min().unwrap_or(0).saturating_sub(1);
    let end = cal.days.len();
    let mut records = Vec::new();
    for e in std::fs::read_dir(dir.join("tasks"))?.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("rec") {
            continue;
        }
        records.extend(read_records(&path)?.into_iter().filter(|r| {
            r.target == target && r.flags & FLAG_NULL == 0 && r.lag > 0 && (!significant_only || r.flags & FLAG_FDR != 0)
        }));
    }
    let mut ids: Vec<u32> = records.iter().map(|r| r.leader).collect();
    ids.push(target);
    ids.sort_unstable();
    ids.dedup();
    let prices: HashMap<u32, Vec<f64>> = ids.par_iter().map(|&id| (id, log_prices(store, &cal, id as usize, base, end))).collect();
    // beta = r * (target h2-day vol / leader h1-day vol) over the relationship's own window.
    let vol = |p: &[f64], start: usize, stop: usize, h: usize| -> f64 {
        let v: Vec<f64> = (start.max(base + h)..stop).step_by(h).map(|k| p[k - base] - p[k - h - base]).filter(|x| x.is_finite()).collect();
        let n = v.len() as f64;
        if n < 10.0 {
            return f64::NAN;
        }
        let m = v.iter().sum::<f64>() / n;
        (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).sqrt()
    };
    let rels: Vec<Rel> = records
        .par_iter()
        .filter_map(|r| {
            let (ws, we, label) = windows.get(r.window as usize)?;
            let (h1, h2) = (r.h1 as usize, r.h2 as usize);
            let ratio = vol(&prices[&target], *ws, *we, h2) / vol(&prices[&r.leader], *ws, *we, h1);
            ratio.is_finite().then(|| Rel {
                leader: r.leader,
                lag: r.lag as usize,
                h1,
                h2,
                r: r.r,
                beta: r.r as f64 * ratio,
                significant: r.flags & FLAG_FDR != 0,
                window_end: *we - 1,
                window: label.clone(),
            })
        })
        .collect();
    Ok(Prepared { cal, base, prices, rels, target })
}

pub struct Options {
    pub significant_only: bool,
    /// Trading days ahead.
    pub horizon: usize,
    /// Use only relationships from census windows that ended at most this many trading days
    /// before the as-of day.
    pub max_age: Option<usize>,
}

/// The relationships usable at calendar row `t0`: for each (leader, lag, h1, h2), the most recent
/// window that ended on or before t0 (and, with `max_age`, not too long before).
fn usable<'a>(p: &'a Prepared, t0: usize, max_age: Option<usize>) -> Vec<&'a Rel> {
    let oldest = max_age.map_or(0, |a| t0.saturating_sub(a));
    let mut best: HashMap<(u32, usize, usize, usize), &Rel> = HashMap::new();
    for r in p.rels.iter().filter(|r| r.window_end <= t0 && r.window_end >= oldest) {
        let e = best.entry((r.leader, r.lag, r.h1, r.h2)).or_insert(r);
        if r.window_end > e.window_end {
            *e = r;
        }
    }
    best.into_values().collect()
}

/// Per-day predicted log returns for days t0+1..=t0+horizon from one relationship (zero on days
/// it says nothing about), and the sum of the leader moves that fed them. A relationship covers a
/// day at most once: its observations step by max(h1, h2) >= h2, so target intervals never overlap.
fn rel_path(p: &Prepared, r: &Rel, t0: usize, horizon: usize) -> (Vec<f64>, f64) {
    let lp = &p.prices[&r.leader];
    let step = r.h1.max(r.h2);
    let mut daily = vec![0.0; horizon];
    let mut used = 0.0;
    let mut g = t0;
    // Leader observations g = t0, t0 - step, ... (all known at t0) whose target interval
    // (g + lag - h2, g + lag] overlaps (t0, t0 + horizon].
    while g + r.lag > t0 && g >= p.base + r.h1 {
        if g + r.lag - r.h2 < t0 + horizon {
            let x = lp[g - p.base] - lp[g - r.h1 - p.base];
            if x.is_finite() {
                used += x;
                let per_day = r.beta * x / r.h2 as f64;
                for day in (g + r.lag + 1 - r.h2)..=(g + r.lag) {
                    if day > t0 && day <= t0 + horizon {
                        daily[day - t0 - 1] += per_day;
                    }
                }
            }
        }
        if g < step {
            break;
        }
        g -= step;
    }
    (daily, used)
}

/// Each relationship covering a future day is a separate estimate of that day's return, so the
/// forecast for the day is their mean. Returns (mean daily log returns, relationships covering
/// each day, each relationship's (daily path, leader moves)).
fn combine(p: &Prepared, rels: &[&Rel], t0: usize, horizon: usize) -> (Vec<f64>, Vec<usize>, Vec<(Vec<f64>, f64)>) {
    let paths: Vec<(Vec<f64>, f64)> = rels.iter().map(|r| rel_path(p, r, t0, horizon)).collect();
    let mut sum = vec![0.0; horizon];
    let mut count = vec![0usize; horizon];
    for (daily, _) in &paths {
        for j in 0..horizon {
            if daily[j] != 0.0 {
                sum[j] += daily[j];
                count[j] += 1;
            }
        }
    }
    let mean = sum.iter().zip(&count).map(|(s, &c)| if c > 0 { s / c as f64 } else { 0.0 }).collect();
    (mean, count, paths)
}

/// Forecast for `ticker` as of `as_of` (the last trading day on or before it).
pub fn forecast(store: &Store, dir: &Path, ticker: &str, as_of: i32, opt: &Options) -> Result<Forecast> {
    let horizon = opt.horizon.max(1);
    let p = prepare(store, dir, ticker, opt.significant_only)?;
    let t0 = p.cal.days.partition_point(|&d| d <= as_of).checked_sub(1).context("as-of date is before the data")?;
    anyhow::ensure!(t0 >= p.base + 253, "as-of date is before the census");
    let tp = &p.prices[&p.target];
    let last_log = tp[t0 - p.base];
    anyhow::ensure!(last_log.is_finite(), "{ticker} has no price on {}", c2_data::date_string(p.cal.days[t0]));
    let rels = usable(&p, t0, opt.max_age);
    let (mean, covering, paths) = combine(&p, &rels, t0, horizon);
    let mut contributions = Vec::new();
    for (r, (daily, used)) in rels.iter().zip(&paths) {
        let own: f64 = daily.iter().sum();
        if own == 0.0 {
            continue;
        }
        let share: f64 = daily.iter().zip(&covering).map(|(d, &c)| if c > 0 { d / c as f64 } else { 0.0 }).sum();
        contributions.push(Contribution {
            leader: store.symbols[r.leader as usize].ticker.clone(),
            lag: r.lag,
            h1: r.h1,
            h2: r.h2,
            r: r.r,
            significant: r.significant,
            window: r.window.clone(),
            leader_move_pct: 100.0 * used.exp_m1(),
            implied_pct: 100.0 * own.exp_m1(),
            share_pct: 100.0 * share,
        });
    }
    contributions.sort_by(|a, b| b.share_pct.abs().total_cmp(&a.share_pct.abs()));
    let mut acc = 0.0;
    let predicted_pct: Vec<f64> = mean.iter().map(|d| { acc += d; 100.0 * acc.exp_m1() }).collect();
    let actual_pct: Vec<Option<f64>> = (1..=horizon)
        .map(|j| tp.get(t0 + j - p.base).copied().filter(|v| v.is_finite()).map(|v| 100.0 * (v - last_log).exp_m1()))
        .collect();
    // Typical size of the stock's own moves: sd of daily log returns over the year before t0.
    let rets: Vec<f64> = (t0 - 251..=t0).map(|k| tp[k - p.base] - tp[k - 1 - p.base]).filter(|x| x.is_finite()).collect();
    let n = rets.len().max(2) as f64;
    let m = rets.iter().sum::<f64>() / n;
    let daily_sd = (rets.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)).sqrt();
    let skill = skill(&p, horizon.min(21), t0, opt.max_age);
    Ok(Forecast {
        ticker: ticker.to_string(),
        as_of: p.cal.days[t0],
        last_price: last_log.exp(),
        relationships: rels.len(),
        future_days: (1..=horizon).map(|j| p.cal.days.get(t0 + j).copied()).collect(),
        predicted_pct,
        actual_pct,
        covering,
        daily_sd,
        contributions,
        skill,
    })
}

/// The same forecast made every 21 trading days in the past (each using only windows that had
/// ended by then), compared with what happened over the next `h` days. Only dates whose outcome
/// was known by `until` are used.
fn skill(p: &Prepared, h: usize, until: usize, max_age: Option<usize>) -> Option<Skill> {
    let tp = &p.prices[&p.target];
    let first = p.rels.iter().map(|r| r.window_end).min()?;
    let dates: Vec<usize> = (first..until.saturating_sub(h)).step_by(21).collect();
    let pairs: Vec<(f64, f64)> = dates
        .par_iter()
        .filter_map(|&t0| {
            let (a, b) = (tp[t0 - p.base], tp[t0 + h - p.base]);
            if !a.is_finite() || !b.is_finite() {
                return None;
            }
            let pred: f64 = combine(p, &usable(p, t0, max_age), t0, h).0.iter().sum();
            (pred != 0.0).then_some((pred, b - a))
        })
        .collect();
    if pairs.len() < 5 {
        return None;
    }
    let n = pairs.len() as f64;
    let (mx, my) = (pairs.iter().map(|q| q.0).sum::<f64>() / n, pairs.iter().map(|q| q.1).sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in &pairs {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    Some(Skill {
        forecasts: pairs.len(),
        horizon: h,
        corr: if sxx > 0.0 && syy > 0.0 { sxy / (sxx * syy).sqrt() } else { 0.0 },
        slope: if sxx > 0.0 { sxy / sxx } else { 0.0 },
        hit_rate: pairs.iter().filter(|(x, y)| x.signum() == y.signum()).count() as f64 / n,
        mean_abs_predicted_pct: 100.0 * pairs.iter().map(|q| q.0.abs()).sum::<f64>() / n,
        mean_abs_actual_pct: 100.0 * pairs.iter().map(|q| q.1.abs()).sum::<f64>() / n,
    })
}
