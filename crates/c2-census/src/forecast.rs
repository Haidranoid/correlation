//! Price forecasts from a stock's lead-lag relationships, relative to the market.
//!
//! As of a chosen day, every stored relationship "leader A moves first, this stock (B) follows
//! `lag` days later" turns A's recent moves into predicted moves of B. All moves are
//! market-relative (see `moves`): sums of the census's market residuals, which is what the census
//! correlates. A's h1-day move x ending at day g predicts B's h2-day move over
//! (g + lag - h2, g + lag] of move_per_pct * x, where move_per_pct = r * sd(B's moves) / sd(A's
//! moves) on the census's own observations in that relationship's correlation period
//! (`moves::move_per_pct`, the "% per 1%" the stock timeline and the Explainer show). Each of
//! A's moves is first capped at `MOVE_CAP` of its typical h1-day moves in that period: the census
//! clips daily values at that many typical days, so it never measured the relationship beyond
//! them, and a move far outside the period's range (such as a former blank-check company's first
//! moves as an operating one) would otherwise be extrapolated without limit. Each prediction is
//! spread evenly over its h2 days; the forecast for a day is the mean over the relationships
//! that cover it, accumulated into a path. So the forecast is B's move relative to
//! the market, which is its price path if the market stays flat. The actual line, the typical
//! daily move and the track record are market-relative too, so like is compared with like.
//!
//! For each (leader, offset, move lengths) a forecast uses the most recent rolling correlation
//! period that ended on or before the as-of day; if several ended that day, the longest, then
//! the lowest period id. The full-history period is never used: it ends on the last day of data,
//! so the track record could never test it. So a past as-of date can be checked against what
//! actually happened, and the track record grades exactly the method shown.
//!
//! `prepare` reads a run's records for one stock once (`ForecastData`); `forecast_from` then
//! makes forecasts for any as-of day and options without touching the disk.
//!
//! What the run's files mean comes from the run itself (`meta::RunMeta`): its task list, its
//! correlation periods (mapped onto the census variable's rows by date, never clamped, so a
//! period the data doesn't cover exactly is left out rather than made to end early, which would
//! let a forecast use correlations measured on days after its as-of day), and its stock list
//! (record ids are translated through tickers into today's store).

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use c2_data::{Calendar, Store};
use rayon::prelude::*;

use crate::analyze::visit_task;
use crate::meta::RunMeta;
use crate::moves;
use crate::run::{FLAG_FDR, FLAG_NULL, Record};
use crate::variable::{self, Variable};

/// A leader's move is capped at this many of its typical h1-day moves in the relationship's
/// correlation period before the multiplier applies: the census clips daily values at this many
/// typical days (`variable::CLIP`), so a relationship was never measured beyond it.
pub const MOVE_CAP: f64 = variable::CLIP as f64;

/// A relationship in which the chosen stock (B) follows.
#[derive(Clone)]
struct Rel {
    /// Index into `ForecastData::leaders`.
    leader: u32,
    lag: usize,
    h1: usize,
    h2: usize,
    r: f32,
    /// `moves::move_per_pct`: the multiplier applied to A's market-relative moves.
    beta: f64,
    /// A's typical h1-day market-relative move in the period (`moves::MoveSize::sd_a`, a log
    /// change); its moves are capped at `MOVE_CAP` of these.
    sd_a: f64,
    significant: bool,
    /// Variable row of the period's last day, and its length in trading days.
    window_end: usize,
    period_days: usize,
    window: String,
    window_id: usize,
}

/// The order of `ForecastData::rels`: by (leader, offset, move lengths), and within each the
/// preferred period first: the most recent end, then the longest, then the lowest id. Total, so
/// the order (and every sum over relationships) never depends on how the files were listed.
fn rel_order(a: &Rel, b: &Rel) -> std::cmp::Ordering {
    (a.leader, a.lag, a.h1, a.h2)
        .cmp(&(b.leader, b.lag, b.h1, b.h2))
        .then(b.window_end.cmp(&a.window_end))
        .then(b.period_days.cmp(&a.period_days))
        .then(a.window_id.cmp(&b.window_id))
        .then(a.r.total_cmp(&b.r))
        .then(a.significant.cmp(&b.significant))
}

struct Leader {
    ticker: String,
    /// Market-relative daily log returns by variable row.
    rel: Vec<f64>,
}

/// Everything forecasts for one stock need from one census run, loaded once by `prepare`.
/// Changing the as-of day, the number of days ahead, significant-only or the maximum age reuses
/// it (`forecast_from`).
pub struct ForecastData {
    /// The run directory and the stock (B) this was loaded for.
    pub dir: PathBuf,
    pub ticker: String,
    /// Stored relationships in which B follows that a forecast can use (every rolling period,
    /// significant or not).
    pub stored: usize,
    /// Trading days of the variable's rows.
    days: Vec<i32>,
    /// First row of the census's earliest correlation period.
    first_row: usize,
    /// B's repaired log closes and its market-relative daily log returns, by row.
    target_log: Vec<f64>,
    target_rel: Vec<f64>,
    leaders: Vec<Leader>,
    /// Sorted by `rel_order`.
    rels: Vec<Rel>,
}

pub struct Contribution {
    pub leader: String,
    pub lag: usize,
    pub h1: usize,
    pub h2: usize,
    pub r: f32,
    pub significant: bool,
    pub window: String,
    /// Census window (correlation period) id and its length in trading days.
    pub window_id: usize,
    pub period_days: usize,
    /// For reference: the leader's market-relative moves feeding this forecast, summed (in %),
    /// each counted whole even when only part of the days it predicts fall inside the horizon,
    /// and as they happened (before the cap). The row's arithmetic uses
    /// `leader_move_in_horizon_pct`.
    pub leader_move_pct: f64,
    /// The part of the leader's moves that lands in these days (in %): 100 (e^L - 1) with
    /// L = sum of x_i w_i, where x_i is one of the leader's market-relative h1-day moves (a log
    /// change, capped at +-`MOVE_CAP` x `leader_typical_move_pct`) and w_i the share of the h2
    /// days it predicts that fall inside the horizon.
    pub leader_move_in_horizon_pct: f64,
    /// How many of the leader's moves feeding this row were capped: each was more than
    /// `MOVE_CAP` typical moves (`leader_typical_move_pct`) from zero, far outside anything seen
    /// in this correlation period, so it counts as exactly `MOVE_CAP` typical moves, the most
    /// the census itself ever measured.
    pub clipped_moves: usize,
    /// The leader's typical h1-day move relative to the market in this correlation period (in %:
    /// 100 x the standard deviation of its log moves on the census's observations,
    /// `moves::MoveSize::sd_a`).
    pub leader_typical_move_pct: f64,
    /// The multiplier applied to them (`moves::move_per_pct`): the stock's move relative to the
    /// market per 1% move of the leader relative to the market. The stock timeline and the
    /// Explainer show this same number as "% per 1%".
    pub move_per_pct: f64,
    /// What this relationship alone predicts for the stock's move relative to the market over
    /// the horizon (in %): 100 (e^(move_per_pct x L) - 1), i.e.
    /// (1 + leader_move_in_horizon_pct / 100)^move_per_pct - 1. The multiplier applies to log
    /// changes, so only for small moves is this about leader_move_in_horizon_pct x move_per_pct.
    pub implied_pct: f64,
    /// Its share of the combined forecast at the horizon (percentage points; shares add up to
    /// the forecast's log change, 100 x ln(1 + predicted % / 100)).
    pub share_pct: f64,
}

/// The same forecast made in the past, compared with what happened; all moves relative to the
/// market.
pub struct Skill {
    pub forecasts: usize,
    pub horizon: usize,
    /// Correlation between predicted and actual moves over the horizon.
    pub corr: f64,
    /// Regression slope of actual on predicted: 1 would mean the forecasts are the right size,
    /// 0 that they carry no information.
    pub slope: f64,
    /// Share of forecasts that got the direction right, among those where both the predicted and
    /// the actual move are non-zero (a zero move has no direction, so it counts neither way).
    pub hit_rate: f64,
    /// How many forecasts the hit rate is over.
    pub directional: usize,
    pub mean_abs_predicted_pct: f64,
    pub mean_abs_actual_pct: f64,
}

/// A forecast of the stock's move relative to the market (its price path if the market stays
/// flat).
pub struct Forecast {
    pub ticker: String,
    pub as_of: i32,
    /// The stock's close on the as-of day (repaired the census's way).
    pub last_price: f64,
    pub relationships: usize,
    /// Predicted and actual cumulative % move relative to the market after 1..=horizon trading
    /// days. Actual is None from the first day whose market-relative return is unknown (no
    /// price that day or the day before), and beyond the data.
    pub predicted_pct: Vec<f64>,
    pub actual_pct: Vec<Option<f64>>,
    /// Calendar dates of the days ahead, where known.
    pub future_days: Vec<Option<i32>>,
    /// How many relationships cover each day ahead (make a prediction for it, even one of 0).
    pub covering: Vec<usize>,
    /// Standard deviation of the stock's market-relative daily log returns over the year up to
    /// the as-of day.
    pub daily_sd: f64,
    pub contributions: Vec<Contribution>,
    pub skill: Option<Skill>,
}

/// Loads what forecasts for `ticker` need from the census run in `dir`: every stored real lagged
/// relationship in which it follows (significant or not, so the significant-only option needs
/// no reread), its leaders' and its own market-relative daily returns, and its repaired closes.
/// `var` is the census variable of the run (`Plan::var` of `Plan::for_run`, or
/// `variable::production` with the run's start); its rows are the forecast's days.
///
/// The run's task list says which files to read (rolling periods only). Its correlation periods
/// are found on `var`'s rows by date (`RunMeta::window_rows_in`); its record ids are translated
/// through tickers into `store` and then `var`'s columns. Relationships from the full-history
/// period, from periods `var` doesn't cover exactly (`RunMeta::uncovered_windows`), with a
/// leader that today's store or `var` doesn't have, or with fewer than `moves::MIN_PAIRS`
/// observations are left out.
pub fn prepare(store: &Store, var: &Variable, dir: &Path, ticker: &str) -> Result<ForecastData> {
    let meta = RunMeta::load(dir)?;
    let stocks = meta.stocks()?;
    let ids = meta.id_map(store)?;
    let current = store.id(ticker).with_context(|| format!("{ticker} is not in the price store"))?;
    let col_of: HashMap<usize, usize> = var.ids.iter().enumerate().map(|(c, &id)| (id, c)).collect();
    let tc = *col_of.get(&current).with_context(|| format!("{ticker} is not a common stock the census covers"))?;
    let target = meta
        .run_id_of(store, &ids, ticker)
        .with_context(|| format!("{ticker} was not in the price store when census {} was made, so it has no results there", meta.name))?;
    // Variable rows (not calendar rows) of each correlation period, found by date.
    let rows: Vec<Option<Range<usize>>> = (0..meta.windows.len()).map(|w| meta.window_rows_in(&var.days, w)).collect();
    let first_row = rows.iter().flatten().map(|r| r.start).min().context("none of the run's correlation periods fall on this store's trading days")?;
    let records: Vec<Record> = meta
        .tasks()?
        .iter()
        .filter(|t| t.length > 0)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|task| {
            let mut out = Vec::new();
            visit_task(&meta, task, |r| {
                if r.target == target && r.flags & FLAG_NULL == 0 && r.lag > 0 {
                    out.push(*r);
                }
            })?;
            Ok(out)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    // Leaders by run id: today's variable column through the ticker, where there is one.
    let column = |rid: u32| ids.current(rid).and_then(|c| col_of.get(&c).copied());
    let mut leader_ids: Vec<u32> = records.iter().map(|r| r.leader).filter(|&rid| column(rid).is_some()).collect();
    leader_ids.sort_unstable();
    leader_ids.dedup();
    let index: HashMap<u32, u32> = leader_ids.iter().enumerate().map(|(i, &rid)| (rid, i as u32)).collect();
    let leaders: Vec<Leader> = leader_ids
        .par_iter()
        .map(|&rid| Leader { ticker: stocks.ticker(rid).unwrap_or("?").to_string(), rel: var.market_relative(column(rid).expect("leaders have a column")) })
        .collect();
    let target_rel = var.market_relative(tc);
    let mut rels: Vec<Rel> = records
        .par_iter()
        .filter_map(|r| {
            let p = meta.windows.get(r.window as usize).filter(|p| p.length > 0)?;
            let span = rows[r.window as usize].clone()?;
            let leader = *index.get(&r.leader)?;
            let (h1, h2, lag) = (r.h1 as usize, r.h2 as usize, r.lag as usize);
            if lag < h2 {
                return None;
            }
            let size = moves::move_size(&leaders[leader as usize].rel, &target_rel, r.r as f64, h1, h2, lag, span.clone())?;
            Some(Rel {
                leader,
                lag,
                h1,
                h2,
                r: r.r,
                beta: size.per_pct,
                sd_a: size.sd_a,
                significant: r.flags & FLAG_FDR != 0,
                window_end: span.end - 1,
                period_days: span.len(),
                window: format!("{}..{}", c2_data::date_string(p.first), c2_data::date_string(p.last)),
                window_id: r.window as usize,
            })
        })
        .collect();
    rels.sort_by(rel_order);
    let target_log = var.col(&var.close, tc).iter().map(|&c| if c > 0.0 { (c as f64).ln() } else { f64::NAN }).collect();
    Ok(ForecastData {
        dir: dir.to_path_buf(),
        ticker: ticker.to_string(),
        stored: rels.len(),
        days: var.days.clone(),
        first_row,
        target_log,
        target_rel,
        leaders,
        rels,
    })
}

pub struct Options {
    pub significant_only: bool,
    /// Trading days ahead.
    pub horizon: usize,
    /// Use only relationships from correlation periods that ended at most this many trading
    /// days before the as-of day.
    pub max_age: Option<usize>,
}

/// The relationships usable at row `t0`, one per (leader, offset, move lengths): the first in
/// `rel_order` among those whose period ended on or before t0 (with `max_age`, not too long
/// before; with significant-only, among the significant ones). Returned in `rel_order`.
fn usable<'a>(d: &'a ForecastData, t0: usize, opt: &Options) -> Vec<&'a Rel> {
    let oldest = opt.max_age.map_or(0, |a| t0.saturating_sub(a));
    let mut out: Vec<&Rel> = Vec::new();
    for r in &d.rels {
        if out.last().is_some_and(|l| (l.leader, l.lag, l.h1, l.h2) == (r.leader, r.lag, r.h1, r.h2)) {
            continue;
        }
        if r.window_end <= t0 && r.window_end >= oldest && (r.significant || !opt.significant_only) {
            out.push(r);
        }
    }
    out
}

/// One relationship's predicted market-relative log returns for days t0+1..=t0+horizon, which
/// of those days it covers (makes a prediction for, even one of 0), and the leader moves that
/// fed them: summed whole as they happened (`used`), and capped and weighted by the share of
/// each one's h2 days inside the horizon (`landed`, so the path's total is beta x landed), with
/// how many were capped.
struct RelPath {
    daily: Vec<f64>,
    covers: Vec<bool>,
    used: f64,
    landed: f64,
    clipped: usize,
}

/// A relationship covers a day at most once: its observations step by max(h1, h2) >= h2, so
/// target intervals never overlap.
fn rel_path(d: &ForecastData, r: &Rel, t0: usize, horizon: usize) -> RelPath {
    let a = &d.leaders[r.leader as usize].rel;
    let step = r.h1.max(r.h2);
    let cap = MOVE_CAP * r.sd_a;
    let mut p = RelPath { daily: vec![0.0; horizon], covers: vec![false; horizon], used: 0.0, landed: 0.0, clipped: 0 };
    // Leader moves ending at g = t0, t0 - step, ... (all known at t0) whose target interval
    // (g + lag - h2, g + lag] overlaps (t0, t0 + horizon].
    let mut g = t0;
    while g + r.lag > t0 && g + 1 >= r.h1 {
        if g + r.lag - r.h2 < t0 + horizon {
            let x = moves::move_sum(a, g, r.h1);
            if x.is_finite() {
                p.used += x;
                let x = if x.abs() > cap {
                    p.clipped += 1;
                    cap.copysign(x)
                } else {
                    x
                };
                let per_day = r.beta * x / r.h2 as f64;
                let mut inside = 0;
                for day in (g + r.lag + 1 - r.h2)..=(g + r.lag) {
                    if day > t0 && day <= t0 + horizon {
                        p.daily[day - t0 - 1] += per_day;
                        p.covers[day - t0 - 1] = true;
                        inside += 1;
                    }
                }
                p.landed += x * inside as f64 / r.h2 as f64;
            }
        }
        if g < step {
            break;
        }
        g -= step;
    }
    p
}

struct Combined {
    /// Mean predicted market-relative log return per day ahead (0 where nothing covers it).
    mean: Vec<f64>,
    /// Relationships covering each day.
    count: Vec<usize>,
    paths: Vec<RelPath>,
}

/// Each relationship covering a future day is a separate estimate of that day's return, so the
/// forecast for the day is their mean.
fn combine(d: &ForecastData, rels: &[&Rel], t0: usize, horizon: usize) -> Combined {
    let paths: Vec<RelPath> = rels.iter().map(|r| rel_path(d, r, t0, horizon)).collect();
    let mut sum = vec![0.0; horizon];
    let mut count = vec![0usize; horizon];
    for p in &paths {
        for j in 0..horizon {
            if p.covers[j] {
                sum[j] += p.daily[j];
                count[j] += 1;
            }
        }
    }
    let mean = sum.iter().zip(&count).map(|(s, &c)| if c > 0 { s / c as f64 } else { 0.0 }).collect();
    Combined { mean, count, paths }
}

/// Forecast for `ticker` as of `as_of` (the last trading day on or before it), from the census
/// run in `dir`. Builds the run's census variable and reads the run on every call; to make
/// several forecasts for one stock, call `prepare` once and then `forecast_from`.
pub fn forecast(store: &Store, dir: &Path, ticker: &str, as_of: i32, opt: &Options) -> Result<Forecast> {
    let cal = Calendar::trading_days(store)?;
    let var = variable::production(store, &cal, RunMeta::load(dir)?.cfg.start)?;
    forecast_from(&prepare(store, &var, dir, ticker)?, as_of, opt)
}

/// Forecast from loaded data as of `as_of` (the last trading day on or before it). No disk
/// access.
pub fn forecast_from(d: &ForecastData, as_of: i32, opt: &Options) -> Result<Forecast> {
    let horizon = opt.horizon.max(1);
    let t0 = d.days.partition_point(|&day| day <= as_of).checked_sub(1).context("as-of date is before the data")?;
    anyhow::ensure!(t0 >= d.first_row + 252, "as-of date is before the census");
    let last_log = d.target_log[t0];
    anyhow::ensure!(
        last_log.is_finite(),
        "{} has no price on {} (none in the data, or a one-day bad print the census ignores)",
        d.ticker,
        c2_data::date_string(d.days[t0])
    );
    let rels = usable(d, t0, opt);
    let c = combine(d, &rels, t0, horizon);
    let mut contributions = Vec::new();
    for (r, p) in rels.iter().zip(&c.paths) {
        if !p.covers.contains(&true) {
            continue;
        }
        let share: f64 = p.daily.iter().zip(&c.count).map(|(x, &n)| if n > 0 { x / n as f64 } else { 0.0 }).sum();
        contributions.push(Contribution {
            leader: d.leaders[r.leader as usize].ticker.clone(),
            lag: r.lag,
            h1: r.h1,
            h2: r.h2,
            r: r.r,
            significant: r.significant,
            window: r.window.clone(),
            window_id: r.window_id,
            period_days: r.period_days,
            leader_move_pct: 100.0 * p.used.exp_m1(),
            leader_move_in_horizon_pct: 100.0 * p.landed.exp_m1(),
            clipped_moves: p.clipped,
            leader_typical_move_pct: 100.0 * r.sd_a,
            move_per_pct: r.beta,
            // The sum of this relationship's daily path, up to rounding.
            implied_pct: 100.0 * (r.beta * p.landed).exp_m1(),
            share_pct: 100.0 * share,
        });
    }
    contributions.sort_by(|a, b| b.share_pct.abs().total_cmp(&a.share_pct.abs()));
    let mut acc = 0.0;
    let predicted_pct: Vec<f64> = c.mean.iter().map(|x| { acc += x; 100.0 * acc.exp_m1() }).collect();
    let (mut acc, mut known) = (0.0, true);
    let actual_pct: Vec<Option<f64>> = (1..=horizon)
        .map(|j| {
            let x = d.target_rel.get(t0 + j).copied().unwrap_or(f64::NAN);
            known &= x.is_finite();
            known.then(|| {
                acc += x;
                100.0 * acc.exp_m1()
            })
        })
        .collect();
    // Typical size of the stock's own market-relative moves: sd over the year up to t0.
    let rets: Vec<f64> = d.target_rel[t0 + 1 - 252..=t0].iter().copied().filter(|x| x.is_finite()).collect();
    let n = rets.len().max(2) as f64;
    let m = rets.iter().sum::<f64>() / n;
    let daily_sd = (rets.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)).sqrt();
    let skill = skill(d, horizon.min(21), t0, opt);
    Ok(Forecast {
        ticker: d.ticker.clone(),
        as_of: d.days[t0],
        last_price: last_log.exp(),
        relationships: rels.len(),
        future_days: (1..=horizon).map(|j| d.days.get(t0 + j).copied()).collect(),
        predicted_pct,
        actual_pct,
        covering: c.count,
        daily_sd,
        contributions,
        skill,
    })
}

/// The same forecast made every 21 trading days in the past (each using only periods that had
/// ended by then), compared with the stock's market-relative move over the next `h` days. Only
/// dates whose outcome was known by `until` are used, and only those some relationship covers.
fn skill(d: &ForecastData, h: usize, until: usize, opt: &Options) -> Option<Skill> {
    let first = d.rels.iter().filter(|r| r.significant || !opt.significant_only).map(|r| r.window_end).min()?;
    let dates: Vec<usize> = (first..until.saturating_sub(h)).step_by(21).collect();
    let pairs: Vec<(f64, f64)> = dates
        .par_iter()
        .filter_map(|&t0| {
            let actual = moves::move_sum(&d.target_rel, t0 + h, h);
            if !actual.is_finite() {
                return None;
            }
            let c = combine(d, &usable(d, t0, opt), t0, h);
            c.count.iter().any(|&n| n > 0).then(|| (c.mean.iter().sum(), actual))
        })
        .collect();
    let directional: Vec<&(f64, f64)> = pairs.iter().filter(|(x, y)| *x != 0.0 && *y != 0.0).collect();
    if pairs.len() < 5 || directional.len() < 5 {
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
        hit_rate: directional.iter().filter(|(x, y)| (*x > 0.0) == (*y > 0.0)).count() as f64 / directional.len() as f64,
        directional: directional.len(),
        mean_abs_predicted_pct: 100.0 * pairs.iter().map(|q| q.0.abs()).sum::<f64>() / n,
        mean_abs_actual_pct: 100.0 * pairs.iter().map(|q| q.1.abs()).sum::<f64>() / n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(leader: u32, window_end: usize, period_days: usize, window_id: usize) -> Rel {
        Rel { leader, lag: 2, h1: 1, h2: 1, r: 0.5, beta: 1.0, sd_a: f64::INFINITY, significant: true, window_end, period_days, window: String::new(), window_id }
    }

    fn data(rels: Vec<Rel>, leader_moves: &[f64]) -> ForecastData {
        let t = 400;
        let leaders = leader_moves.iter().map(|&x| Leader { ticker: format!("L{x}"), rel: vec![x; t] }).collect();
        let mut rels = rels;
        rels.sort_by(rel_order);
        ForecastData {
            dir: PathBuf::new(),
            ticker: "B".into(),
            stored: rels.len(),
            days: (0..t as i32).collect(),
            first_row: 0,
            target_log: vec![0.0; t],
            target_rel: vec![0.0; t],
            leaders,
            rels,
        }
    }

    #[test]
    fn a_zero_prediction_still_covers_its_day() {
        // Leader 0 did not move at all, leader 1 rose 1%: both predict day 1 (lag 2, 1-day
        // moves), so day 1's forecast is the mean of 0 and 0.01.
        let d = data(vec![rel(0, 300, 252, 1), rel(1, 300, 252, 1)], &[0.0, 0.01]);
        let opt = Options { significant_only: false, horizon: 2, max_age: None };
        let c = combine(&d, &usable(&d, 350, &opt), 350, 2);
        assert_eq!(c.count, vec![2, 2]);
        assert!((c.mean[0] - 0.005).abs() < 1e-15);
        let f = forecast_from(&d, 350, &opt).unwrap();
        assert_eq!(f.contributions.len(), 2);
        assert_eq!(f.covering, vec![2, 2]);
    }

    #[test]
    fn hit_rate_skips_zero_moves() {
        // A leader that always rose 1% predicts a rise every time. The stock's actual 2-day
        // moves: up after the first 6 forecast dates, down after the next 3, exactly 0 after the
        // last 3. A zero move has no direction, so the hit rate is 6 of 9, not 9 of 12.
        let mut d = data(vec![rel(0, 100, 252, 1)], &[0.01]);
        let dates: Vec<usize> = (100..348).step_by(21).collect();
        assert_eq!(dates.len(), 12);
        for (k, &t0) in dates.iter().enumerate() {
            d.target_rel[t0 + 1] = if k < 6 { 0.01 } else if k < 9 { -0.01 } else { 0.0 };
        }
        let s = skill(&d, 2, 350, &Options { significant_only: false, horizon: 2, max_age: None }).unwrap();
        assert_eq!((s.forecasts, s.directional), (12, 9));
        assert!((s.hit_rate - 6.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn same_end_prefers_the_longest_period_in_any_order() {
        // Runs number periods by length, so on a shared end day the longest has the highest id
        // (main-2025-06-27: 252, 756 and 1260 days with ids 77, 221 and 283 all end 2025-04-14;
        // 504 and 1260 days with ids 138 and 270 end 2022-01-06). Lowest id alone would pick the
        // shortest.
        let a = vec![rel(0, 300, 252, 77), rel(0, 300, 756, 221), rel(0, 300, 1260, 283), rel(0, 200, 504, 138), rel(0, 200, 1260, 270)];
        let mut b = a.clone();
        b.reverse();
        let opt = Options { significant_only: false, horizon: 5, max_age: None };
        for rels in [a, b] {
            let d = data(rels, &[0.01]);
            let u = usable(&d, 350, &opt);
            assert_eq!(u.len(), 1);
            assert_eq!((u[0].period_days, u[0].window_id), (1260, 283));
            // Before those ended, the longest of the older ones.
            let u = usable(&d, 250, &opt);
            assert_eq!((u[0].period_days, u[0].window_id), (1260, 270));
        }
    }

    #[test]
    fn a_row_multiplies_out() {
        // Offset 10, A over 1 day, B over 5 days, 7 days ahead. A's move on t0 (+0.3) predicts
        // B's days t0+6..t0+10, 2 of 5 inside; its move on t0-5 (-0.2) predicts t0+1..t0+5, all
        // inside. Counted whole they sum to +0.1, but what lands is L = 0.4 x 0.3 - 0.2 = -0.08,
        // and the relationship implies e^(2 L) - 1 at 2% per 1%.
        let mut r = rel(0, 300, 252, 1);
        (r.lag, r.h2, r.beta) = (10, 5, 2.0);
        let mut d = data(vec![r], &[0.0]);
        d.leaders[0].rel[350] = 0.3;
        d.leaders[0].rel[345] = -0.2;
        let f = forecast_from(&d, 350, &Options { significant_only: false, horizon: 7, max_age: None }).unwrap();
        let c = &f.contributions[0];
        let l = 0.4 * 0.3 - 0.2;
        assert!((c.leader_move_pct - 100.0 * 0.1f64.exp_m1()).abs() < 1e-9);
        assert!((c.leader_move_in_horizon_pct - 100.0 * f64::exp_m1(l)).abs() < 1e-9);
        assert!((c.implied_pct - 100.0 * f64::exp_m1(2.0 * l)).abs() < 1e-9);
        assert!((c.implied_pct / 100.0 + 1.0 - (1.0 + c.leader_move_in_horizon_pct / 100.0).powf(c.move_per_pct)).abs() < 1e-12);
        // The forecast's own path adds up to the same.
        assert!(((1.0 + f.predicted_pct[6] / 100.0).ln() - 2.0 * l).abs() < 1e-12);
    }

    #[test]
    fn a_move_far_outside_its_period_is_capped() {
        // As above, but A's typical 1-day move in the period is 1%, so its moves are capped at
        // +-5 typical moves (0.05). Its -50% move on t0 (2 of 5 predicted days inside) counts as
        // -0.05; its -3% move on t0-5 is used as it is. L = 0.4 x -0.05 - 0.03 = -0.05.
        let mut r = rel(0, 300, 252, 1);
        (r.lag, r.h2, r.beta, r.sd_a) = (10, 5, 2.0, 0.01);
        let mut d = data(vec![r], &[0.0]);
        d.leaders[0].rel[350] = -0.5;
        d.leaders[0].rel[345] = -0.03;
        let opt = Options { significant_only: false, horizon: 7, max_age: None };
        let f = forecast_from(&d, 350, &opt).unwrap();
        let c = &f.contributions[0];
        let l = 0.4 * -0.05 - 0.03;
        assert_eq!(c.clipped_moves, 1);
        assert!((c.leader_typical_move_pct - 1.0).abs() < 1e-12);
        // As it happened, counted whole: -0.53.
        assert!((c.leader_move_pct - 100.0 * f64::exp_m1(-0.53)).abs() < 1e-9);
        assert!((c.leader_move_in_horizon_pct - 100.0 * f64::exp_m1(l)).abs() < 1e-9);
        assert!((c.implied_pct - 100.0 * f64::exp_m1(2.0 * l)).abs() < 1e-9);
        assert!((c.implied_pct / 100.0 + 1.0 - (1.0 + c.leader_move_in_horizon_pct / 100.0).powf(c.move_per_pct)).abs() < 1e-12);
        assert!(((1.0 + f.predicted_pct[6] / 100.0).ln() - 2.0 * l).abs() < 1e-12);
        // Up to the cap nothing changes: +0.05 exactly is used as it is.
        d.leaders[0].rel[350] = 0.05;
        let c = &forecast_from(&d, 350, &opt).unwrap().contributions[0];
        assert_eq!(c.clipped_moves, 0);
        assert!((c.leader_move_in_horizon_pct - 100.0 * f64::exp_m1(0.4 * 0.05 - 0.03)).abs() < 1e-9);
    }
}
