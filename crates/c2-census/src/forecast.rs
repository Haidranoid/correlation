//! Price forecasts from a stock's lead-lag relationships, relative to the market.
//!
//! As of a chosen day, every stored relationship "leader A moves first, this stock (B) follows
//! `lag` days later" turns A's recent moves into predicted moves of B. All moves are
//! market-relative (see `moves`): sums of the census's market residuals, which is what the census
//! correlates. A's h1-day move x ending at day g predicts B's h2-day move over
//! (g + lag - h2, g + lag] of move_per_pct * x, where move_per_pct = r * sd(B's moves) / sd(A's
//! moves) on the census's own observations in that relationship's correlation period
//! (`moves::move_per_pct`, the "% per 1%" the stock timeline and the Explainer show). Each
//! prediction is spread evenly over its h2 days; the forecast for a day is the mean over the
//! relationships that cover it, accumulated into a path. So the forecast is B's move relative to
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

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use c2_data::{Calendar, Store};
use rayon::prelude::*;

use crate::grid::Config;
use crate::moves;
use crate::run::{FLAG_FDR, FLAG_NULL, RECORD_BYTES, Record};
use crate::variable::{self, Variable};

/// The records of one task file that `keep` accepts, read in pieces so that memory stays small
/// however large the file is.
pub(crate) fn scan_records(path: &Path, keep: impl Fn(&Record) -> bool) -> Result<Vec<Record>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let mut buf = vec![0u8; RECORD_BYTES * 16384];
    let (mut fill, mut out) = (0, Vec::new());
    loop {
        let n = file.read(&mut buf[fill..]).with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        fill += n;
        let whole = fill / RECORD_BYTES * RECORD_BYTES;
        out.extend(buf[..whole].chunks_exact(RECORD_BYTES).map(Record::from_bytes).filter(|r| keep(r)));
        buf.copy_within(whole..fill, 0);
        fill -= whole;
    }
    Ok(out)
}

/// A correlation period as a run's windows.csv lists it.
pub(crate) struct Period {
    /// Trading days; 0 for the full-history period.
    pub length: usize,
    pub first: i32,
    pub last: i32,
}

/// A run's correlation periods by id, from its windows.csv (id,length,first,last,stocks).
/// Round 2: `RunMeta`.
pub(crate) fn read_periods(dir: &Path) -> Result<Vec<Period>> {
    let path = dir.join("windows.csv");
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            let f: Vec<&str> = l.split(',').collect();
            anyhow::ensure!(f.len() >= 4 && f[0].parse::<usize>().ok() == Some(i), "{}: unexpected line {l:?}", path.display());
            Ok(Period { length: f[1].parse()?, first: c2_data::parse_date(f[2])?, last: c2_data::parse_date(f[3])? })
        })
        .collect()
}

/// Rows of the trading days `days` (e.g. `Variable::days`) that a period covers, matched by
/// date. None unless its first and last days are both trading days in `days` and, for a rolling
/// period, the rows number exactly its length: a period is never clamped onto other days.
/// Round 2: `RunMeta::window_rows`.
pub(crate) fn period_rows(days: &[i32], p: &Period) -> Option<Range<usize>> {
    let (a, b) = (days.binary_search(&p.first).ok()?, days.binary_search(&p.last).ok()?);
    let rows = a..b + 1;
    (a <= b && (p.length == 0 || rows.len() == p.length)).then_some(rows)
}

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
    /// The leader's market-relative moves feeding this forecast, summed (in %).
    pub leader_move_pct: f64,
    /// The multiplier applied to them (`moves::move_per_pct`): the stock's move relative to the
    /// market per 1% move of the leader relative to the market. The stock timeline and the
    /// Explainer show this same number as "% per 1%".
    pub move_per_pct: f64,
    /// What this relationship alone predicts for the stock's move relative to the market over
    /// the horizon (in %).
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
/// `var` is the census variable of the run (`Plan::var`, or `variable::production` with the
/// run's start); its rows are the forecast's days. Relationships from the full-history period,
/// from periods whose dates are not exactly `var`'s trading days (see `period_rows`), with a
/// leader outside `var`, or with fewer than `moves::MIN_PAIRS` observations are left out.
pub fn prepare(store: &Store, var: &Variable, dir: &Path, ticker: &str) -> Result<ForecastData> {
    let target = store.id(ticker).with_context(|| format!("{ticker} is not in the price store"))?;
    let col_of: HashMap<usize, usize> = var.ids.iter().enumerate().map(|(c, &id)| (id, c)).collect();
    let tc = *col_of.get(&target).with_context(|| format!("{ticker} is not a common stock the census covers"))?;
    let periods = read_periods(dir)?;
    let rows: Vec<Option<Range<usize>>> = periods.iter().map(|p| period_rows(&var.days, p)).collect();
    let first_row = rows.iter().flatten().map(|r| r.start).min().context("none of the run's correlation periods fall on this store's trading days")?;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir.join("tasks"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("rec"))
        .collect();
    paths.sort();
    let target = target as u32;
    let records: Vec<_> = paths
        .par_iter()
        .map(|path| scan_records(path, |r| r.target == target && r.flags & FLAG_NULL == 0 && r.lag > 0))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    let mut ids: Vec<u32> = records.iter().map(|r| r.leader).filter(|id| col_of.contains_key(&(*id as usize))).collect();
    ids.sort_unstable();
    ids.dedup();
    let index: HashMap<u32, u32> = ids.iter().enumerate().map(|(i, &id)| (id, i as u32)).collect();
    let leaders: Vec<Leader> = ids.par_iter().map(|&id| Leader { ticker: store.symbols[id as usize].ticker.clone(), rel: var.market_relative(col_of[&(id as usize)]) }).collect();
    let target_rel = var.market_relative(tc);
    let mut rels: Vec<Rel> = records
        .par_iter()
        .filter_map(|r| {
            let p = periods.get(r.window as usize).filter(|p| p.length > 0)?;
            let span = rows[r.window as usize].clone()?;
            let leader = *index.get(&r.leader)?;
            let (h1, h2, lag) = (r.h1 as usize, r.h2 as usize, r.lag as usize);
            if lag < h2 {
                return None;
            }
            let beta = moves::move_per_pct(&leaders[leader as usize].rel, &target_rel, r.r as f64, h1, h2, lag, span.clone())?;
            Some(Rel {
                leader,
                lag,
                h1,
                h2,
                r: r.r,
                beta,
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
/// of those days it covers (makes a prediction for, even one of 0), and the sum of the leader
/// moves that fed them.
struct RelPath {
    daily: Vec<f64>,
    covers: Vec<bool>,
    used: f64,
}

/// A relationship covers a day at most once: its observations step by max(h1, h2) >= h2, so
/// target intervals never overlap.
fn rel_path(d: &ForecastData, r: &Rel, t0: usize, horizon: usize) -> RelPath {
    let a = &d.leaders[r.leader as usize].rel;
    let step = r.h1.max(r.h2);
    let mut p = RelPath { daily: vec![0.0; horizon], covers: vec![false; horizon], used: 0.0 };
    // Leader moves ending at g = t0, t0 - step, ... (all known at t0) whose target interval
    // (g + lag - h2, g + lag] overlaps (t0, t0 + horizon].
    let mut g = t0;
    while g + r.lag > t0 && g + 1 >= r.h1 {
        if g + r.lag - r.h2 < t0 + horizon {
            let x = moves::move_sum(a, g, r.h1);
            if x.is_finite() {
                p.used += x;
                let per_day = r.beta * x / r.h2 as f64;
                for day in (g + r.lag + 1 - r.h2)..=(g + r.lag) {
                    if day > t0 && day <= t0 + horizon {
                        p.daily[day - t0 - 1] += per_day;
                        p.covers[day - t0 - 1] = true;
                    }
                }
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
/// run in `dir`. Builds the census variable and reads the run on every call; to make several
/// forecasts for one stock, call `prepare` once and then `forecast_from`.
pub fn forecast(store: &Store, dir: &Path, ticker: &str, as_of: i32, opt: &Options) -> Result<Forecast> {
    let cal = Calendar::trading_days(store)?;
    let var = variable::production(store, &cal, Config::full().start)?;
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
        let own: f64 = p.daily.iter().sum();
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
            move_per_pct: r.beta,
            implied_pct: 100.0 * own.exp_m1(),
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
        Rel { leader, lag: 2, h1: 1, h2: 1, r: 0.5, beta: 1.0, significant: true, window_end, period_days, window: String::new(), window_id }
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
    fn periods_map_by_exact_dates_only() {
        let days = [10, 11, 12, 15, 16, 17];
        let p = |length, first, last| Period { length, first, last };
        assert_eq!(period_rows(&days, &p(3, 11, 15)), Some(1..4));
        assert_eq!(period_rows(&days, &p(0, 10, 17)), Some(0..6));
        // Not a trading day, past the data, wrong length, reversed: never clamped.
        assert_eq!(period_rows(&days, &p(3, 11, 14)), None);
        assert_eq!(period_rows(&days, &p(3, 15, 18)), None);
        assert_eq!(period_rows(&days, &p(4, 11, 15)), None);
        assert_eq!(period_rows(&days, &p(0, 15, 11)), None);
    }

    #[test]
    fn same_end_prefers_the_longest_period_in_any_order() {
        let a = vec![rel(0, 300, 756, 9), rel(0, 300, 1260, 4), rel(0, 200, 1260, 2)];
        let mut b = a.clone();
        b.reverse();
        let opt = Options { significant_only: false, horizon: 5, max_age: None };
        for rels in [a, b] {
            let d = data(rels, &[0.01]);
            let u = usable(&d, 350, &opt);
            assert_eq!(u.len(), 1);
            assert_eq!((u[0].period_days, u[0].window_id), (1260, 4));
            // Before that period ended, the older one is used.
            assert_eq!(usable(&d, 250, &opt)[0].window_id, 2);
        }
    }
}
