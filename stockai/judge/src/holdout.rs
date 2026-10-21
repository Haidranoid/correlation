//! The holdout test, forked from correlation2's `crates/c2-census/src/holdout.rs` at commit
//! 9335126 so stockai can grade files of model scores without modifying correlation2. Treat a
//! year as the future, predict it only from data before each decision, and trade $10,000 on the
//! predictions against SPY, QQQ and random guessers.
//!
//! Timing (unchanged): at the close of day d the account picks stocks to hold over day d+1,
//! using data through day d only.
//!
//! Unchanged from correlation2: the liquid universe, the reversal family's scores, rebalancing,
//! the account and its costs, the uniform random guessers and their xorshift seeding, the
//! benchmarks, hit rate, rank IC and the long-short check. With `delay = 0` and
//! `RandomMatch::Uniform` the reversal family's report is bit-identical to correlation2's (the
//! tests below run both on the same data). The census-pairs and peer-momentum families are not
//! in this fork.
//!
//! New:
//! - `Family::Scores`: scores read from a `[N, T]` matrix aligned to `plan.var`. NaN means no
//!   score: never picked, and left out of both rank ICs and the long-short deciles.
//! - `delay`: the account acts `delay` trading days after each decision (it holds the picks
//!   decided `delay` days earlier, and nothing before the first). Hit rate, rank IC and the
//!   long-short check stay decision-day measures of the scores.
//! - Rank IC against the actual return over the holding period (`ic_ret_*`), next to the IC
//!   against forward census-variable sums.
//! - `RandomMatch::Volatility`: random accounts that match the strategy's turnover exactly and
//!   the volatility decile of every name it buys.
//! - Sharadar stores (docs/sharadar.md, with `Extras`): a delisting rule (`DelistRule`) for a held
//!   stock that stops trading, the same for the strategy and the random accounts; random accounts
//!   that never buy a stock after its last close (as the picks, which need a score, can't); and
//!   the $5 median-price test of the universe on as-traded closes (`PriceTest::Traded`). Without
//!   extras (a Stooq store) none of this exists and every number is as before.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use c2_census::run::Plan;
use c2_census::variable::{self, Variable};
use c2_data::Store;
use rayon::prelude::*;

use crate::extras::Extras;
use crate::npy;
use crate::sharadar::DelistKind;

/// Which signal makes the predictions.
#[derive(Clone, Copy)]
pub enum Family<'a> {
    /// Positive control (correlation2's family E): own-stock short-term reversal. Score = minus
    /// the stock's own log return over the last `lookback` days.
    Reversal { lookback: usize, window: usize },
    /// A file of scores: `matrix.get(j, d)` for variable column j, decided at the close of day d.
    /// Candidates are the same liquid universe as the reversal family's (`window` days).
    Scores { matrix: &'a ScoreMatrix, window: usize },
}

/// How the random guessers are matched to the strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandomMatch {
    /// correlation2's guessers: each day as many names as the strategy, replacing as many as it
    /// did, drawn uniformly from the candidates.
    Uniform,
    /// Also the volatility decile of every name the strategy buys (by trailing 63-day realized
    /// volatility among that day's candidates), mirroring the strategy slot by slot.
    Volatility,
}

impl RandomMatch {
    pub fn name(self) -> &'static str {
        match self {
            RandomMatch::Uniform => "uniform",
            RandomMatch::Volatility => "volatility",
        }
    }
}

/// What a held stock earns when it stops trading (docs/sharadar.md, "The delisting rule"). Its last
/// close is on decision day L: over day L+1 a position that loses everything returns -1, and after
/// that (or at once, at its last price) it returns 0 until it is sold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelistRule {
    /// Pre-registered: a buyout at its last price, a failure or an unknown delisting a total loss.
    Main,
    /// Every delisting at its last price (the optimistic bound; what a missing price did before).
    LastPrice,
    /// Every delisting a total loss (the harsh bound).
    TotalLoss,
}

impl DelistRule {
    pub fn name(self) -> &'static str {
        match self {
            DelistRule::Main => "main",
            DelistRule::LastPrice => "last-price",
            DelistRule::TotalLoss => "total-loss",
        }
    }

    /// Whether a delisting of this kind (`Extras::delist_kind`; 0 = listed) loses everything.
    pub fn loses(self, kind: i8) -> bool {
        match self {
            DelistRule::Main => kind == DelistKind::Failure.code() || kind == DelistKind::Unknown.code(),
            DelistRule::LastPrice => false,
            DelistRule::TotalLoss => kind > 0,
        }
    }
}

/// Which closes the universe's $5 median-price test uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PriceTest {
    /// The variable's closes, adjusted as of the download (correlation2's test).
    Adjusted,
    /// As-traded closes (`Extras::close_unadj`), so a later reverse split can't lift a stock over $5.
    Traded,
}

impl PriceTest {
    pub fn name(self) -> &'static str {
        match self {
            PriceTest::Adjusted => "adjusted",
            PriceTest::Traded => "traded",
        }
    }
}

pub struct Options<'a> {
    pub family: Family<'a>,
    /// Trading days between rebalances (picks are held in between).
    pub rebalance: usize,
    pub start: i32,
    /// First day after the holdout (exclusive); `i32::MAX` runs to the end of the data.
    pub end: i32,
    pub picks: usize,
    pub costs_bps: Vec<f64>,
    pub random_runs: usize,
    pub cash: f64,
    /// The account trades this many trading days after each decision (0: at the decision close).
    pub delay: usize,
    pub random_match: RandomMatch,
    /// Keep the scores, picks and holdings in the report (`Report::kept`).
    pub keep_scores: bool,
    /// The store's Sharadar sidecars, aligned to the variable: needed by `delist` and `PriceTest::Traded`.
    pub extras: Option<&'a Extras>,
    /// None: no delisting logic (a missing price is a 0 return, as in correlation2).
    pub delist: Option<DelistRule>,
    pub price_test: PriceTest,
}

pub struct Outcome {
    pub label: String,
    /// "SPY", "QQQ", or None for buy-and-hold of all candidates.
    pub ticker: Option<&'static str>,
    pub final_value: f64,
}

pub struct Report {
    pub first_day: i32,
    pub last_day: i32,
    pub days: usize,
    pub pairs: usize,
    pub windows_used: usize,
    pub candidates: usize,
    pub benchmarks: Vec<Outcome>,
    /// (cost bps, strategy final, random median, share of random >= strategy)
    pub strategy: Vec<(f64, f64, f64, f64)>,
    pub hit_rate: f64,
    /// Number of picks judged by the hit rate (correlation2 names this field `hits`).
    pub hits: usize,
    pub ic_mean: f64,
    pub ic_t: f64,
    /// Rank IC of the scores against the actual return over the holding period (the long-short
    /// check's `fwd_ret`, which counts a missing price as a 0 return).
    pub ic_ret_mean: f64,
    pub ic_ret_t: f64,
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
    /// How the random accounts compare with the strategy's trading.
    pub matching: Matching,
    pub kept: Option<Kept>,
    /// Delistings met in the holdout (with a delisting rule).
    pub delisting: Option<DelistStats>,
}

/// Delistings among the candidates and in the accounts over one holdout. Kinds are counted as
/// (buyout, failure, unknown).
#[derive(Clone, Debug)]
pub struct DelistStats {
    pub rule: DelistRule,
    /// Candidates whose last close is one of the holdout's decision days (they stop trading inside it).
    pub candidates: [usize; 3],
    /// Candidates whose last close came before the first decision day: in the universe from the
    /// year before, but dead from the start (held, they earn 0).
    pub dead_before: usize,
    /// The strategy's names held over the day after their last close.
    pub held: [usize; 3],
    /// The same for the random accounts, mean per account.
    pub random_held: [f64; 3],
    /// Names bought on a decision day after their last close: by the strategy (only if the scores
    /// score a dead stock), and by the random accounts, mean per account (0: they draw only stocks
    /// still trading).
    pub bought_dead: usize,
    pub random_bought_dead: f64,
}

/// The random accounts' trading against the strategy's (their sets do not depend on the cost).
#[derive(Clone, Debug)]
pub struct Matching {
    pub mode: RandomMatch,
    /// Names bought over the holdout: by the strategy, and by each random account.
    pub strategy_buys: usize,
    pub random_buys_mean: f64,
    pub random_buys_min: usize,
    pub random_buys_max: usize,
    /// Mean volatility decile (0 = calmest tenth of that day's candidates, 9 = most volatile)
    /// of the names bought, and of the names held (over days and names). Random: pooled over runs.
    pub strategy_buy_decile: f64,
    pub random_buy_decile: f64,
    pub strategy_held_decile: f64,
    pub random_held_decile: f64,
    /// Volatility matching: buys that could not be matched on decile (decile exhausted, or the
    /// strategy's name had no volatility estimate), so any non-member was bought. Summed over runs.
    pub unmatched: usize,
}

/// Decision-day detail, kept on request.
pub struct Kept {
    /// Variable columns of the candidates.
    pub candidates: Vec<usize>,
    /// Decision days (variable rows), one per account day.
    pub days: Vec<usize>,
    /// Scores by account day on rebalance days (by candidate position), else None.
    pub scores: Vec<Option<Vec<f64>>>,
    /// Picks decided each day, and the names held over the next day (the picks decided `delay`
    /// days earlier), as candidate positions. (Read by the tests.)
    #[cfg_attr(not(test), allow(dead_code))]
    pub picks: Vec<Vec<usize>>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub held: Vec<Vec<usize>>,
}

/// Scores aligned to `plan.var`: `[N, T]` in C order, row j = variable column j, column d = day d.
pub struct ScoreMatrix {
    pub n: usize,
    pub t: usize,
    pub dtype: &'static str,
    data: ScoreData,
}

enum ScoreData {
    F4(Vec<f32>),
    F8(Vec<f64>),
}

impl ScoreMatrix {
    pub fn from_f64(n: usize, t: usize, v: Vec<f64>) -> Result<ScoreMatrix> {
        ensure!(v.len() == n * t, "{} scores for [{n}, {t}]", v.len());
        let m = ScoreMatrix { n, t, dtype: "<f8", data: ScoreData::F8(v) };
        m.check_values()?;
        Ok(m)
    }

    pub fn from_f32(n: usize, t: usize, v: Vec<f32>) -> Result<ScoreMatrix> {
        ensure!(v.len() == n * t, "{} scores for [{n}, {t}]", v.len());
        let m = ScoreMatrix { n, t, dtype: "<f4", data: ScoreData::F4(v) };
        m.check_values()?;
        Ok(m)
    }

    /// Reads a `.npy` scores file: dtype `<f4` or `<f8`, shape `[n, t]`, C order.
    pub fn load(path: &Path, n: usize, t: usize) -> Result<ScoreMatrix> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path).with_context(|| format!("opening scores file {}", path.display()))?);
        let (h, _) = npy::read_header(&mut f).with_context(|| format!("{}: not a readable .npy file", path.display()))?;
        if h.descr != "<f4" && h.descr != "<f8" {
            bail!("{}: dtype {:?}; scores must be float32 or float64 ('<f4' or '<f8')", path.display(), h.descr);
        }
        ensure!(
            h.shape == [n, t],
            "{}: shape {:?}; scores must be [N, T] = [{n}, {t}], the export's (stocks, days) for this store",
            path.display(),
            h.shape
        );
        ensure!(!h.fortran_order, "{}: Fortran order; save the scores C-ordered (np.ascontiguousarray)", path.display());
        let m = match npy::read(path)?.data {
            npy::Data::F4(v) => ScoreMatrix::from_f32(n, t, v),
            npy::Data::F8(v) => ScoreMatrix::from_f64(n, t, v),
            npy::Data::I1(_) | npy::Data::I4(_) => unreachable!("dtype checked above"),
        };
        m.with_context(|| format!("scores file {}", path.display()))
    }

    /// The score of variable column `j` decided at the close of day `d`, widened to f64.
    #[inline]
    pub fn get(&self, j: usize, d: usize) -> f64 {
        match &self.data {
            ScoreData::F4(v) => v[j * self.t + d] as f64,
            ScoreData::F8(v) => v[j * self.t + d],
        }
    }

    /// Number of finite scores.
    pub fn finite(&self) -> usize {
        match &self.data {
            ScoreData::F4(v) => v.par_iter().filter(|x| x.is_finite()).count(),
            ScoreData::F8(v) => v.par_iter().filter(|x| x.is_finite()).count(),
        }
    }

    /// NaN is "no score"; an infinite score is an error.
    fn check_values(&self) -> Result<()> {
        let bad = match &self.data {
            ScoreData::F4(v) => v.par_iter().position_first(|x| x.is_infinite()),
            ScoreData::F8(v) => v.par_iter().position_first(|x| x.is_infinite()),
        };
        if let Some(k) = bad {
            bail!("an infinite score at stock column {}, day index {} ({}); use NaN for no score", k / self.t, k % self.t, self.get(k / self.t, k % self.t));
        }
        Ok(())
    }
}

fn rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// correlation2's per-run seed for the random guessers.
fn seed(run: usize) -> u64 {
    0x9E37_79B9_7F4A_7C15u64 ^ (run as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9)
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

pub fn holdout(store: &Store, plan: &Plan, opt: &Options) -> Result<Report> {
    let var = &plan.var;
    let h0 = var.days.partition_point(|&d| d < opt.start);
    // Rows [0, t) end at the holdout's last day; nothing after it is used.
    let t = var.days.partition_point(|&d| d < opt.end);
    anyhow::ensure!(h0 > 0 && h0 + 2 < t, "holdout outside the data");
    anyhow::ensure!(opt.random_runs > 0, "need at least one random guesser");
    ensure!(opt.delist.is_none() || opt.extras.is_some(), "a delisting rule needs the store's delist.tsv (a Sharadar store)");
    if let Some(x) = opt.extras {
        ensure!(
            x.close_unadj.len() == var.n * var.t && x.delist_day.len() == var.n && x.delist_kind.len() == var.n,
            "the store's extras are not aligned to the variable"
        );
    }
    let price: &[f32] = match opt.price_test {
        PriceTest::Adjusted => &var.close,
        PriceTest::Traded => &opt.extras.context("the as-traded $5 test needs close_unadj.f32 (a Sharadar store)")?.close_unadj,
    };
    // Decision days d = h0-1 .. t-2 (hold over d+1 in the holdout); scores on rebalance days.
    let days: Vec<usize> = (h0 - 1..t - 1).collect();
    let rebalance = opt.rebalance.max(1);
    let window = match opt.family {
        Family::Reversal { window, .. } | Family::Scores { window, .. } => window,
    };
    let candidates = liquid_universe(var, price, h0, window);
    let (scores, pairs_used, windows_used): (Vec<Option<Vec<f64>>>, usize, usize) = match opt.family {
        Family::Reversal { lookback, .. } => reversal_scores(var, &candidates, t, &days, rebalance, lookback),
        Family::Scores { matrix, .. } => file_scores(var, &candidates, matrix, t, &days, rebalance)?,
    };
    let nc = candidates.len();
    anyhow::ensure!(nc >= 2 * opt.picks, "only {nc} stocks can be picked (history before the holdout too short?)");
    anyhow::ensure!(pairs_used > 0, "no candidate has a finite score on the holdout's rebalance days");
    let vt = var.t;
    // The delisting rule, if any: (rule, each column's last-close day or -1, its kind).
    let delist = opt.delist.map(|rule| {
        let x = opt.extras.expect("checked above");
        (rule, x.delist_day.as_slice(), x.delist_kind.as_slice())
    });
    // loss_day[j]: the decision day whose next-day return is -1 (the rule makes the delisting a total loss), else -1.
    let loss_day: Option<Vec<i32>> = delist.map(|(rule, day, kind)| (0..var.n).map(|j| if rule.loses(kind[j]) { day[j] } else { -1 }).collect());
    let ret = |j: usize, d: usize| -> f64 {
        if loss_day.as_ref().is_some_and(|l| l[j] == d as i32) {
            return -1.0;
        }
        let (a, b) = (var.close[j * vt + d], var.close[j * vt + d + 1]);
        if a.is_finite() && b.is_finite() && a > 0.0 { (b / a - 1.0) as f64 } else { 0.0 }
    };
    // Forward outcomes over a holding period (d, d + h]: raw return and census-variable sum. Under a
    // delisting rule a stock whose last close L falls in [d, d + h) counts at close[L] or as -1.
    let fwd_ret = |j: usize, d: usize, h: usize| -> f64 {
        let e = (d + h).min(t - 1);
        let a = var.close[j * vt + d];
        if let Some((rule, day, kind)) = delist {
            let l = day[j];
            if l >= 0 && (d..e).contains(&(l as usize)) && a.is_finite() && a > 0.0 {
                return if rule.loses(kind[j]) { -1.0 } else { (var.close[j * vt + l as usize] / a - 1.0) as f64 };
            }
        }
        let b = var.close[j * vt + e];
        if a.is_finite() && b.is_finite() && a > 0.0 { (b / a - 1.0) as f64 } else { 0.0 }
    };
    // Under a delisting rule, in a sequence of daily sets: names held over the day after their last
    // close (by kind), and names bought on a day after it.
    let delist_hits = |sets: &[Vec<usize>]| -> DelistHits {
        let (mut by_kind, mut bought_dead) = ([0usize; 3], 0usize);
        if let Some((_, day, kind)) = delist {
            let none = Vec::new();
            for (k, set) in sets.iter().enumerate() {
                let (d, prev) = (days[k] as i32, if k == 0 { &none } else { &sets[k - 1] });
                for &c in set {
                    let j = candidates[c];
                    if day[j] == d && kind[j] > 0 {
                        by_kind[kind[j] as usize - 1] += 1;
                    }
                    if day[j] >= 0 && day[j] < d && !prev.contains(&c) {
                        bought_dead += 1;
                    }
                }
            }
        }
        (by_kind, bought_dead)
    };
    let fwd_resid = |j: usize, d: usize, h: usize| -> f32 {
        let e = (d + h).min(t - 1);
        (d + 1..=e).map(|r| var.v[j * vt + r]).filter(|v| v.is_finite()).sum::<f32>()
            + if (d + 1..=e).all(|r| !var.v[j * vt + r].is_finite()) { f32::NAN } else { 0.0 }
    };

    // Strategy picks: on rebalance days the top `picks` predicted gainers with a positive score,
    // equal weight; held unchanged until the next rebalance. (A NaN score is never > 0.)
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
    // What the account holds over day d+1: the picks decided `delay` days earlier, nothing before.
    let held: Vec<Vec<usize>> = (0..days.len()).map(|k| if k >= opt.delay { picks[k - opt.delay].clone() } else { Vec::new() }).collect();
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

    // Under a delisting rule the random accounts buy only stocks still trading on the decision day,
    // as the picks can (a stock past its last close has no score): per candidate, its last close.
    let alive = delist.map(|(_, day, _)| Alive { last: candidates.iter().map(|&j| if day[j] >= 0 { day[j] } else { i32::MAX }).collect(), days: &days });
    // Volatility deciles of the candidates on each account day (point in time; without the dead).
    let deciles = Deciles::new(var, &candidates, &days, alive.as_ref());
    let mut strategy = Vec::new();
    let mut median_path: Vec<f64> = Vec::new();
    let mut paths_for_median: Vec<Vec<f64>> = Vec::new();
    let mut run_stats: Vec<SetStats> = Vec::new();
    let mut run_delist: Vec<DelistHits> = Vec::new();
    for &c in &opt.costs_bps {
        let s = simulate(&|k| held[k].clone(), c);
        // Random guessers with the strategy's turnover: each day they hold as many stocks as the
        // strategy and replace exactly as many as it did, at random. Otherwise costs alone would
        // separate them (random daily picks turn over ~100% a day).
        let first = strategy.is_empty();
        let runs: Vec<(Vec<f64>, Option<SetStats>, Option<DelistHits>)> = (0..opt.random_runs)
            .into_par_iter()
            .map(|run| {
                let (chosen, unmatched) = match opt.random_match {
                    RandomMatch::Uniform => (uniform_guesser(run, &held, nc, alive.as_ref()), 0),
                    RandomMatch::Volatility => volatility_guesser(run, &held, nc, &deciles, alive.as_ref()),
                };
                let stats = first.then(|| SetStats::of(&chosen, &deciles, unmatched));
                let hits = (first && delist.is_some()).then(|| delist_hits(&chosen));
                (path(&|k| chosen[k].clone(), c), stats, hits)
            })
            .collect();
        let mut finals: Vec<f64> = runs
            .into_iter()
            .map(|(p, stats, hits)| {
                run_stats.extend(stats);
                run_delist.extend(hits);
                p
            })
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
    let matching = Matching::new(opt.random_match, &SetStats::of(&held, &deciles, 0), &run_stats);
    let delisting = delist.map(|(rule, day, kind)| {
        let (first, last) = (days[0] as i32, *days.last().unwrap() as i32);
        let mut candidates_by_kind = [0usize; 3];
        let mut dead_before = 0;
        for &j in &candidates {
            if day[j] >= 0 && day[j] < first {
                dead_before += 1;
            } else if day[j] >= first && day[j] <= last && kind[j] > 0 {
                candidates_by_kind[kind[j] as usize - 1] += 1;
            }
        }
        let (held_by_kind, bought_dead) = delist_hits(&held);
        let runs = run_delist.len().max(1) as f64;
        DelistStats {
            rule,
            candidates: candidates_by_kind,
            dead_before,
            held: held_by_kind,
            random_held: std::array::from_fn(|i| run_delist.iter().map(|r| r.0[i]).sum::<usize>() as f64 / runs),
            bought_dead,
            random_bought_dead: run_delist.iter().map(|r| r.1).sum::<usize>() as f64 / runs,
        }
    });

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
            benchmarks.push(Outcome { label: label.into(), ticker: Some(ticker), final_value: v });
        }
    }
    let hold_all: f64 = candidates
        .iter()
        .map(|&j| days.iter().map(|&d| 1.0 + ret(j, d)).product::<f64>())
        .sum::<f64>()
        / nc.max(1) as f64;
    benchmarks.push(Outcome { label: format!("Buy and hold all {nc} candidates"), ticker: None, final_value: opt.cash * hold_all });

    // Daily curves for plotting: picks (first cost level), SPY, QQQ, random median.
    let first_cost = opt.costs_bps.first().copied().unwrap_or(0.0);
    let pick_path = path(&|k| held[k].clone(), first_cost);
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
    // rank IC of scores vs forward census-variable sums and vs forward returns, and a long-short
    // decile diagnostic. NaN scores (no score) are left out of the ICs and the deciles.
    let (mut hits, mut tries) = (0usize, 0usize);
    let mut ics = Vec::new();
    let mut ics_ret = Vec::new();
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
        let (a, b): (Vec<f64>, Vec<f64>) = (0..nc).filter(|&c| sc[c].is_finite()).map(|c| (sc[c], fwd_ret(candidates[c], d, rebalance))).unzip();
        if a.len() > 10 {
            ics_ret.push(corr(&rank(&a), &rank(&b)));
        }
        let mut order: Vec<usize> = (0..nc).filter(|&c| sc[c].is_finite()).collect();
        let m = order.len();
        if m < 2 {
            continue;
        }
        order.sort_by(|&x, &y| sc[y].total_cmp(&sc[x]));
        let q = (m / 10).max(1);
        let (long, short) = (order[..q].to_vec(), order[m - q..].to_vec());
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
    let ic_ret_mean = ics_ret.iter().sum::<f64>() / ics_ret.len().max(1) as f64;
    let ic_ret_sd = (ics_ret.iter().map(|x| (x - ic_ret_mean).powi(2)).sum::<f64>() / ics_ret.len().max(1) as f64).sqrt();
    let kept = opt.keep_scores.then(|| Kept { candidates: candidates.clone(), days: days.clone(), scores, picks, held });
    Ok(Report {
        first_day: var.days[h0],
        last_day: var.days[t - 1],
        days: days.len(),
        pairs: pairs_used,
        windows_used,
        candidates: nc,
        benchmarks,
        strategy,
        hit_rate: hits as f64 / tries.max(1) as f64,
        hits: tries,
        ic_mean,
        ic_t: ic_mean / ic_sd.max(1e-12) * (ics.len() as f64).sqrt(),
        ic_ret_mean,
        ic_ret_t: ic_ret_mean / ic_ret_sd.max(1e-12) * (ics_ret.len() as f64).sqrt(),
        // Per holding period: mean return as a daily equivalent, Sharpe annualized.
        ls_mean: ls_mean / rebalance as f64,
        ls_sharpe: ls_mean / ls_sd.max(1e-12) * (252.0 / rebalance as f64).sqrt(),
        ls_t: ls_mean / ls_sd.max(1e-12) * (ls.len() as f64).sqrt(),
        ls_turnover,
        curve,
        matching,
        kept,
        delisting,
    })
}

type Scores = (Vec<Option<Vec<f64>>>, usize, usize);
/// Names held over the day after their last close (buyout, failure, unknown), and names bought
/// after it.
type DelistHits = ([usize; 3], usize);

/// Under a delisting rule: which candidates the random accounts may buy on each account day.
struct Alive<'a> {
    /// Per candidate, the decision day (variable row) of its last close; i32::MAX if listed.
    last: Vec<i32>,
    /// The decision day of each account day.
    days: &'a [usize],
}

impl Alive<'_> {
    /// Whether candidate `c` still trades at the close of account day `k`'s decision day.
    fn at(&self, k: usize, c: usize) -> bool {
        self.last[c] >= self.days[k] as i32
    }
}

/// Stocks with >= 90% of days with a census value over the `window` days before `h0`, and median
/// price >= $5 and median dollar volume >= $5M over the last year before it.
/// (correlation2's `liquid_universe`, which is private there; copied unchanged, except that the
/// price test reads `price`: `var.close` as in correlation2, or as-traded closes.)
fn liquid_universe(var: &Variable, price: &[f32], h0: usize, window: usize) -> Vec<usize> {
    let vt = var.t;
    let w0 = h0.saturating_sub(window);
    let tail = (h0 - w0).min(variable::LOOKBACK);
    (0..var.n)
        .filter(|&j| {
            let v = &var.v[j * vt + w0..j * vt + h0];
            if h0 == w0 || v.iter().filter(|x| x.is_finite()).count() * 10 < (h0 - w0) * 9 {
                return false;
            }
            let mut pc: Vec<f32> = price[j * vt + h0 - tail..j * vt + h0].iter().copied().filter(|x| x.is_finite()).collect();
            let mut pv: Vec<f32> = var.dollar_volume[j * vt + h0 - tail..j * vt + h0].iter().copied().filter(|x| x.is_finite()).collect();
            let med = |v: &mut Vec<f32>| {
                let mid = v.len() / 2;
                if v.is_empty() { 0.0 } else { *v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1 }
            };
            med(&mut pc) >= variable::MIN_PRICE && med(&mut pv) >= variable::MIN_DOLLAR_VOLUME
        })
        .collect()
}

/// Positive-control scores: minus each stock's own log return over the last `lookback` days.
fn reversal_scores(var: &Variable, candidates: &[usize], t: usize, days: &[usize], rebalance: usize, lookback: usize) -> Scores {
    let vt = var.t;
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
    (scores, candidates.len(), 0)
}

/// Scores-file family: the reversal family's candidates, each scored by `matrix` on rebalance
/// days. `pairs` = candidates with at least one finite score on those days.
fn file_scores(var: &Variable, candidates: &[usize], matrix: &ScoreMatrix, t: usize, days: &[usize], rebalance: usize) -> Result<Scores> {
    ensure!(matrix.n == var.n && matrix.t == var.t, "scores are [{}, {}] but the variable is [{}, {}]", matrix.n, matrix.t, var.n, var.t);
    let scores: Vec<Option<Vec<f64>>> = days
        .par_iter()
        .enumerate()
        .map(|(k, &d)| (k % rebalance == 0 && d < t).then(|| candidates.iter().map(|&j| matrix.get(j, d)).collect()))
        .collect();
    let pairs = (0..candidates.len()).filter(|&c| scores.iter().flatten().any(|s| s[c].is_finite())).count();
    Ok((scores, pairs, 0))
}

/// Uniform random guesser `run` (correlation2's, unchanged without `alive`): mirrors the size and
/// the number of replacements of `held` each day, drawing uniformly from the `nc` candidates.
/// With `alive` (a delisting rule) it sells its members past their last close first, as the
/// strategy can't keep a dead stock past a rebalance (no score), and buys only stocks still
/// trading that day (any, if too few are left).
fn uniform_guesser(run: usize, held: &[Vec<usize>], nc: usize, alive: Option<&Alive>) -> Vec<Vec<usize>> {
    let mut state = seed(run);
    let mut chosen: Vec<Vec<usize>> = Vec::with_capacity(held.len());
    let mut set: Vec<usize> = Vec::new();
    for k in 0..held.len() {
        let target = held[k].len().min(nc);
        let replaced = if k == 0 { target } else { held[k].iter().filter(|c| !held[k - 1].contains(c)).count() };
        let keep = target.saturating_sub(replaced).min(set.len());
        if let Some(a) = alive {
            while set.len() > keep {
                let Some(i) = set.iter().position(|&c| !a.at(k, c)) else { break };
                set.swap_remove(i);
            }
        }
        // Drop random members down to `keep`, then add random non-members up to `target`.
        while set.len() > keep {
            let i = (rng(&mut state) % set.len() as u64) as usize;
            set.swap_remove(i);
        }
        let strict = alive.is_some_and(|a| set.len() < target && (0..nc).filter(|&c| a.at(k, c) && !set.contains(&c)).count() >= target - set.len());
        while set.len() < target {
            let c = (rng(&mut state) % nc as u64) as usize;
            if !set.contains(&c) && (!strict || alive.is_some_and(|a| a.at(k, c))) {
                set.push(c);
            }
        }
        chosen.push(set.clone());
    }
    chosen
}

/// Volatility-matched random guesser `run`. Each of the account's names shadows one of the
/// strategy's: when the strategy sells a name the account sells its shadow, and when it buys a
/// name the account buys a random non-member from that name's volatility decile on that day
/// (any non-member if the decile is exhausted or the name has no decile). Names sold that day
/// are not bought back, so turnover equals the strategy's exactly. With `alive` (a delisting
/// rule) it buys only stocks still trading that day: the deciles leave the others out, and so
/// does the fallback while it can. Returns the sets and the number of unmatched buys.
fn volatility_guesser(run: usize, held: &[Vec<usize>], nc: usize, deciles: &Deciles, alive: Option<&Alive>) -> (Vec<Vec<usize>>, usize) {
    let mut state = seed(run);
    let mut chosen: Vec<Vec<usize>> = Vec::with_capacity(held.len());
    // (strategy's name, the account's name in that slot)
    let mut slots: Vec<(usize, usize)> = Vec::new();
    let mut unmatched = 0;
    let none = Vec::new();
    for k in 0..held.len() {
        let (now, prev) = (&held[k], if k == 0 { &none } else { &held[k - 1] });
        let mut sold = Vec::new();
        slots.retain(|&(s, mine)| {
            let keep = now.contains(&s);
            if !keep {
                sold.push(mine);
            }
            keep
        });
        for &bought in now.iter().filter(|c| !prev.contains(c)) {
            let member = |c: usize, slots: &[(usize, usize)]| slots.iter().any(|&(_, m)| m == c);
            let q = deciles.dec[k][bought];
            let matched = if q == NO_DECILE { None } else { draw(&mut state, &deciles.members[k][q as usize], |c| !member(c, &slots) && !sold.contains(&c)) };
            let mine = match matched {
                Some(c) => c,
                None => {
                    unmatched += 1;
                    let trades = |c: usize| alive.is_none_or(|a| a.at(k, c));
                    draw_any(&mut state, nc, |c| !member(c, &slots) && !sold.contains(&c) && trades(c))
                        .or_else(|| draw_any(&mut state, nc, |c| !member(c, &slots) && trades(c)))
                        .or_else(|| draw_any(&mut state, nc, |c| !member(c, &slots)))
                        .expect("at least as many candidates as picks")
                }
            };
            slots.push((bought, mine));
        }
        chosen.push(slots.iter().map(|&(_, m)| m).collect());
    }
    (chosen, unmatched)
}

/// A uniformly random element of `pool` satisfying `ok` (by rejection), or None if none does.
fn draw(state: &mut u64, pool: &[usize], ok: impl Fn(usize) -> bool) -> Option<usize> {
    if !pool.iter().any(|&c| ok(c)) {
        return None;
    }
    loop {
        let c = pool[(rng(state) % pool.len() as u64) as usize];
        if ok(c) {
            return Some(c);
        }
    }
}

/// A uniformly random candidate in `0..nc` satisfying `ok`, or None if none does.
fn draw_any(state: &mut u64, nc: usize, ok: impl Fn(usize) -> bool) -> Option<usize> {
    if !(0..nc).any(&ok) {
        return None;
    }
    loop {
        let c = (rng(state) % nc as u64) as usize;
        if ok(c) {
            return Some(c);
        }
    }
}

const NO_DECILE: u8 = u8::MAX;
/// Trailing window of daily returns for realized volatility.
pub const VOL_DAYS: usize = 63;
/// Fewest valid returns in the window for a volatility estimate.
pub const VOL_MIN_RETURNS: usize = 21;

/// Volatility deciles of the candidates on each account day.
struct Deciles {
    /// dec[k][c]: decile 0..=9 of candidate c on account day k (decision day days[k]), or NO_DECILE.
    dec: Vec<Vec<u8>>,
    /// members[k][q]: the candidates in decile q on account day k, ascending.
    members: Vec<[Vec<usize>; 10]>,
}

impl Deciles {
    /// With `alive`, the candidates past their last close on a day are in no decile that day.
    fn new(var: &Variable, candidates: &[usize], days: &[usize], alive: Option<&Alive>) -> Deciles {
        let nc = candidates.len();
        let (dec, members): (Vec<Vec<u8>>, Vec<[Vec<usize>; 10]>) = days
            .par_iter()
            .enumerate()
            .map(|(k, &d)| {
                let vol: Vec<f64> = candidates
                    .iter()
                    .enumerate()
                    .map(|(c, &j)| if alive.is_none_or(|a| a.at(k, c)) { realized_vol(var, j, d) } else { f64::NAN })
                    .collect();
                let mut order: Vec<usize> = (0..nc).filter(|&c| vol[c].is_finite()).collect();
                order.sort_by(|&a, &b| vol[a].total_cmp(&vol[b]));
                let m = order.len();
                let mut dec = vec![NO_DECILE; nc];
                for (r, &c) in order.iter().enumerate() {
                    dec[c] = (10 * r / m) as u8;
                }
                let mut members: [Vec<usize>; 10] = std::array::from_fn(|_| Vec::new());
                for (c, &q) in dec.iter().enumerate() {
                    if q != NO_DECILE {
                        members[q as usize].push(c);
                    }
                }
                (dec, members)
            })
            .unzip();
        Deciles { dec, members }
    }
}

/// Sample standard deviation of the daily log returns over the `VOL_DAYS` returns ending at day
/// `d` (closes d-63 ..= d): data through the close of day d only. NaN with fewer than
/// `VOL_MIN_RETURNS` valid returns.
pub fn realized_vol(var: &Variable, j: usize, d: usize) -> f64 {
    let c = &var.close[j * var.t..(j + 1) * var.t];
    let lo = d.saturating_sub(VOL_DAYS);
    let (mut s, mut ss, mut k) = (0f64, 0f64, 0usize);
    for i in lo + 1..=d {
        let (a, b) = (c[i - 1], c[i]);
        if a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 {
            let r = (b as f64 / a as f64).ln();
            s += r;
            ss += r * r;
            k += 1;
        }
    }
    if k < VOL_MIN_RETURNS {
        return f64::NAN;
    }
    let kf = k as f64;
    ((ss - s * s / kf) / (kf - 1.0)).max(0.0).sqrt()
}

/// Trading statistics of one sequence of daily sets.
#[derive(Clone, Copy, Debug, Default)]
struct SetStats {
    buys: usize,
    buy_decile_sum: f64,
    buy_decile_n: usize,
    held_decile_sum: f64,
    held_decile_n: usize,
    unmatched: usize,
}

impl SetStats {
    fn of(sets: &[Vec<usize>], deciles: &Deciles, unmatched: usize) -> SetStats {
        let mut st = SetStats { unmatched, ..SetStats::default() };
        let none = Vec::new();
        for (k, now) in sets.iter().enumerate() {
            let prev = if k == 0 { &none } else { &sets[k - 1] };
            for &c in now {
                let q = deciles.dec[k][c];
                if q != NO_DECILE {
                    st.held_decile_sum += q as f64;
                    st.held_decile_n += 1;
                }
                if !prev.contains(&c) {
                    st.buys += 1;
                    if q != NO_DECILE {
                        st.buy_decile_sum += q as f64;
                        st.buy_decile_n += 1;
                    }
                }
            }
        }
        st
    }
}

impl Matching {
    fn new(mode: RandomMatch, strategy: &SetStats, runs: &[SetStats]) -> Matching {
        let ratio = |s: f64, n: usize| if n > 0 { s / n as f64 } else { f64::NAN };
        let sum = |f: fn(&SetStats) -> f64| runs.iter().map(f).sum::<f64>();
        let count = |f: fn(&SetStats) -> usize| runs.iter().map(f).sum::<usize>();
        Matching {
            mode,
            strategy_buys: strategy.buys,
            random_buys_mean: ratio(count(|s| s.buys) as f64, runs.len()),
            random_buys_min: runs.iter().map(|s| s.buys).min().unwrap_or(0),
            random_buys_max: runs.iter().map(|s| s.buys).max().unwrap_or(0),
            strategy_buy_decile: ratio(strategy.buy_decile_sum, strategy.buy_decile_n),
            random_buy_decile: ratio(sum(|s| s.buy_decile_sum), count(|s| s.buy_decile_n)),
            strategy_held_decile: ratio(strategy.held_decile_sum, strategy.held_decile_n),
            random_held_decile: ratio(sum(|s| s.held_decile_sum), count(|s| s.held_decile_n)),
            unmatched: count(|s| s.unmatched),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use c2_data::store::{Series, SymbolInfo};
    use c2_data::{Kind, days_from_civil};

    struct Gen(u64);
    impl Gen {
        fn u(&mut self) -> f64 {
            (rng(&mut self.0) >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Roughly standard normal (sum of 12 uniforms).
        fn z(&mut self) -> f64 {
            (0..12).map(|_| self.u()).sum::<f64>() - 6.0
        }
    }

    fn weekdays(from: i32, to: i32) -> Vec<i32> {
        // 1970-01-01 was a Thursday: (day + 3) % 7 == 0 is a Monday.
        (from..=to).filter(|d| (d + 3).rem_euclid(7) < 5).collect()
    }

    /// A synthetic variable with short-term reversal, missing prices, some illiquid and cheap
    /// stocks, and volatility that differs by stock; plus a store holding SPY and QQQ.
    fn fixture(n: usize, seed: u64) -> (Plan, Store) {
        let days = weekdays(days_from_civil(2003, 1, 2), days_from_civil(2009, 12, 31));
        let t = days.len();
        let mut g = Gen(0x1234_5678_9ABC_DEF0 ^ seed);
        let market: Vec<f64> = (0..t).map(|_| 0.0003 + 0.01 * g.z()).collect();
        let (mut v, mut close, mut dollar_volume) = (vec![f32::NAN; n * t], vec![f32::NAN; n * t], vec![0f32; n * t]);
        for j in 0..n {
            let sd = 0.008 + 0.03 * g.u();
            let base = if j % 10 == 8 { 2.0 } else { 15.0 + 40.0 * g.u() };
            let volume = if j % 10 == 9 { 1.0e4 } else { 1.0e6 };
            let (mut logp, mut prev) = (0.0f64, 0.0f64);
            for d in 0..t {
                let e = sd * g.z();
                let r = e - 0.2 * prev + market[d];
                prev = e;
                logp += r;
                let missing = g.u() < 0.01;
                if !missing {
                    close[j * t + d] = (base * logp.exp()) as f32;
                    dollar_volume[j * t + d] = close[j * t + d] * volume;
                    if d >= 252 {
                        v[j * t + d] = ((e / sd) as f32).clamp(-5.0, 5.0);
                    }
                } else {
                    dollar_volume[j * t + d] = f32::NAN;
                }
            }
        }
        let var = Variable { days: days.clone(), ids: (0..n).collect(), t, n, v, close, dollar_volume, market: market.clone() };
        let plan = Plan { cfg: c2_census::grid::Config::full(), var, windows: vec![], universes: vec![], tasks: vec![], cost: vec![] };

        let dir = std::env::temp_dir().join(format!("stockai-judge-store-{}-{seed}", std::process::id()));
        let series: Vec<Series> = ["QQQ", "SPY"]
            .iter()
            .enumerate()
            .map(|(i, ticker)| {
                let mut p = 100.0f64;
                let close: Vec<f32> = market
                    .iter()
                    .map(|m| {
                        p *= (m * (1.0 + 0.3 * i as f64)).exp();
                        p as f32
                    })
                    .collect();
                Series {
                    info: SymbolInfo { ticker: ticker.to_string(), exchange: "nyse".into(), kind: Kind::Etf, first: days[0], last: days[t - 1], rows: t, source: String::new() },
                    date: days.clone(),
                    open: close.clone(),
                    high: close.clone(),
                    low: close.clone(),
                    volume: vec![1e6; t],
                    close,
                }
            })
            .collect();
        c2_data::store::write(&dir, &series).unwrap();
        let store = Store::open(&dir).unwrap();
        (plan, store)
    }

    fn opts<'a>(family: Family<'a>, rebalance: usize, costs: &[f64], random_runs: usize) -> Options<'a> {
        Options {
            family,
            rebalance,
            start: days_from_civil(2007, 6, 28),
            end: days_from_civil(2008, 6, 28),
            picks: 5,
            costs_bps: costs.to_vec(),
            random_runs,
            cash: 10_000.0,
            delay: 0,
            random_match: RandomMatch::Uniform,
            keep_scores: true,
            extras: None,
            delist: None,
            price_test: PriceTest::Adjusted,
        }
    }

    fn bits(x: f64) -> u64 {
        if x.is_nan() { u64::MAX } else { x.to_bits() }
    }

    fn same_as_c2(ours: &Report, c2: &c2_census::holdout::Report) {
        assert_eq!((ours.first_day, ours.last_day, ours.days), (c2.first_day, c2.last_day, c2.days));
        assert_eq!((ours.pairs, ours.windows_used, ours.candidates, ours.hits), (c2.pairs, c2.windows_used, c2.candidates, c2.hits));
        assert_eq!(ours.benchmarks.len(), c2.benchmarks.len());
        for (a, b) in ours.benchmarks.iter().zip(&c2.benchmarks) {
            assert_eq!((a.label.as_str(), bits(a.final_value)), (b.label.as_str(), bits(b.final_value)));
        }
        assert_eq!(ours.strategy.len(), c2.strategy.len());
        for (a, b) in ours.strategy.iter().zip(&c2.strategy) {
            assert_eq!([bits(a.0), bits(a.1), bits(a.2), bits(a.3)], [bits(b.0), bits(b.1), bits(b.2), bits(b.3)], "strategy row");
        }
        for (name, a, b) in [
            ("hit_rate", ours.hit_rate, c2.hit_rate),
            ("ic_mean", ours.ic_mean, c2.ic_mean),
            ("ic_t", ours.ic_t, c2.ic_t),
            ("ls_mean", ours.ls_mean, c2.ls_mean),
            ("ls_sharpe", ours.ls_sharpe, c2.ls_sharpe),
            ("ls_t", ours.ls_t, c2.ls_t),
            ("ls_turnover", ours.ls_turnover, c2.ls_turnover),
        ] {
            assert_eq!(bits(a), bits(b), "{name}: {a} vs {b}");
        }
        assert_eq!(ours.curve.len(), c2.curve.len());
        for (a, b) in ours.curve.iter().zip(&c2.curve) {
            assert_eq!((a.0, bits(a.1), bits(a.2), bits(a.3), bits(a.4)), (b.0, bits(b.1), bits(b.2), bits(b.3), bits(b.4)));
        }
    }

    fn same_report(a: &Report, b: &Report) {
        let c2_like = |r: &Report| c2_census::holdout::Report {
            first_day: r.first_day,
            last_day: r.last_day,
            days: r.days,
            pairs: r.pairs,
            windows_used: r.windows_used,
            candidates: r.candidates,
            benchmarks: r.benchmarks.iter().map(|o| c2_census::holdout::Outcome { label: o.label.clone(), final_value: o.final_value }).collect(),
            strategy: r.strategy.clone(),
            hit_rate: r.hit_rate,
            hits: r.hits,
            ic_mean: r.ic_mean,
            ic_t: r.ic_t,
            ls_mean: r.ls_mean,
            ls_sharpe: r.ls_sharpe,
            ls_t: r.ls_t,
            ls_turnover: r.ls_turnover,
            curve: r.curve.clone(),
        };
        same_as_c2(a, &c2_like(b));
        assert_eq!((bits(a.ic_ret_mean), bits(a.ic_ret_t)), (bits(b.ic_ret_mean), bits(b.ic_ret_t)));
        let (ma, mb) = (&a.matching, &b.matching);
        assert_eq!((ma.strategy_buys, ma.random_buys_min, ma.random_buys_max, ma.unmatched), (mb.strategy_buys, mb.random_buys_min, mb.random_buys_max, mb.unmatched));
        assert_eq!(
            [bits(ma.random_buys_mean), bits(ma.strategy_buy_decile), bits(ma.random_buy_decile), bits(ma.strategy_held_decile), bits(ma.random_held_decile)],
            [bits(mb.random_buys_mean), bits(mb.strategy_buy_decile), bits(mb.random_buy_decile), bits(mb.strategy_held_decile), bits(mb.random_held_decile)]
        );
    }

    /// The scores the reversal family computed, as a [N, T] f64 matrix with NaN elsewhere.
    fn dump(plan: &Plan, r: &Report) -> Vec<f64> {
        let (n, t) = (plan.var.n, plan.var.t);
        let mut m = vec![f64::NAN; n * t];
        let k = r.kept.as_ref().unwrap();
        for (i, s) in k.scores.iter().enumerate() {
            if let Some(s) = s {
                for (c, &x) in s.iter().enumerate() {
                    m[k.candidates[c] * t + k.days[i]] = x;
                }
            }
        }
        m
    }

    #[test]
    fn reversal_and_its_scores_file_match_correlation2_bit_for_bit() {
        let (plan, store) = fixture(90, 1);
        let dir = std::env::temp_dir();
        for (lookback, rebalance) in [(5, 5), (21, 21), (5, 1), (3, 2)] {
            let family = Family::Reversal { lookback, window: 756 };
            let ours = holdout(&store, &plan, &opts(family, rebalance, &[0.0, 5.0, 20.0], 64)).unwrap();
            let c2_opt = c2_census::holdout::Options {
                family: c2_census::holdout::Family::Reversal { lookback, window: 756 },
                rebalance,
                start: days_from_civil(2007, 6, 28),
                end: days_from_civil(2008, 6, 28),
                picks: 5,
                costs_bps: vec![0.0, 5.0, 20.0],
                random_runs: 64,
                cash: 10_000.0,
                all_windows: false,
            };
            let theirs = c2_census::holdout::holdout(&store, &plan, &dir, &c2_opt).unwrap();
            assert!(ours.candidates >= 50 && ours.candidates < 90, "fixture universe: {}", ours.candidates);
            same_as_c2(&ours, &theirs);

            // The reversal scores, graded as a scores file (f64), give the same report.
            let matrix = ScoreMatrix::from_f64(plan.var.n, plan.var.t, dump(&plan, &ours)).unwrap();
            let graded = holdout(&store, &plan, &opts(Family::Scores { matrix: &matrix, window: 756 }, rebalance, &[0.0, 5.0, 20.0], 64)).unwrap();
            same_as_c2(&graded, &theirs);
            same_report(&graded, &ours);
        }
    }

    #[test]
    fn nan_scores_are_never_picked_and_left_out_of_ic_and_deciles() {
        let (plan, store) = fixture(90, 2);
        let (n, t) = (plan.var.n, plan.var.t);
        let base = holdout(&store, &plan, &opts(Family::Reversal { lookback: 5, window: 756 }, 5, &[0.0], 16)).unwrap();
        let kept = base.kept.as_ref().unwrap();
        // Knock out a third of the candidates entirely and 30% of the remaining (stock, day) scores.
        let mut m = dump(&plan, &base);
        let mut g = Gen(99);
        let gone: Vec<usize> = kept.candidates.iter().copied().filter(|j| j % 3 == 0).collect();
        for j in 0..n {
            for d in 0..t {
                if gone.contains(&j) || g.u() < 0.3 {
                    m[j * t + d] = f64::NAN;
                }
            }
        }
        let matrix = ScoreMatrix::from_f64(n, t, m.clone()).unwrap();
        let r = holdout(&store, &plan, &opts(Family::Scores { matrix: &matrix, window: 756 }, 5, &[0.0], 16)).unwrap();
        let k = r.kept.as_ref().unwrap();
        assert_eq!(r.candidates, base.candidates);
        assert_eq!(r.pairs, base.candidates - gone.len());
        for (i, s) in k.scores.iter().enumerate() {
            if let Some(s) = s {
                for &c in &k.picks[i] {
                    assert!(s[c].is_finite() && s[c] > 0.0, "picked a stock without a positive score");
                }
            }
            for &c in &k.picks[i] {
                assert!(!gone.contains(&k.candidates[c]), "picked a stock that never had a score");
            }
        }

        // One rebalance day: the ICs and the long-short book use only the finite scores.
        let one = k.days.len();
        let r1 = holdout(&store, &plan, &opts(Family::Scores { matrix: &matrix, window: 756 }, one, &[0.0], 16)).unwrap();
        let k1 = r1.kept.as_ref().unwrap();
        let (d, sc) = (k1.days[0], k1.scores[0].as_ref().unwrap());
        let last = k1.days[one - 1] + 1;
        let e = (d + one).min(last);
        let var = &plan.var;
        let resid = |j: usize| {
            let v: Vec<f32> = (d + 1..=e).map(|r| var.v[j * t + r]).filter(|x| x.is_finite()).collect();
            if v.is_empty() { f64::NAN } else { v.iter().sum::<f32>() as f64 }
        };
        let fret = |j: usize| {
            let (a, b) = (var.close[j * t + d], var.close[j * t + e]);
            if a.is_finite() && b.is_finite() && a > 0.0 { (b / a - 1.0) as f64 } else { 0.0 }
        };
        let finite: Vec<usize> = (0..k1.candidates.len()).filter(|&c| sc[c].is_finite()).collect();
        assert!(finite.len() > 20 && finite.len() < k1.candidates.len());
        let spearman = |x: &[f64], y: &[f64]| {
            let rk = |v: &[f64]| {
                let mut i: Vec<usize> = (0..v.len()).collect();
                i.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
                let mut r = vec![0.0; v.len()];
                for (p, &q) in i.iter().enumerate() {
                    r[q] = p as f64;
                }
                r
            };
            let (a, b) = (rk(x), rk(y));
            let m = (a.len() as f64 - 1.0) / 2.0;
            let (sab, saa, sbb) = a.iter().zip(&b).fold((0.0, 0.0, 0.0), |(s, p, q), (u, v)| (s + (u - m) * (v - m), p + (u - m).powi(2), q + (v - m).powi(2)));
            sab / (saa * sbb).sqrt()
        };
        let ok: Vec<usize> = finite.iter().copied().filter(|&c| resid(k1.candidates[c]).is_finite()).collect();
        let want_ic = spearman(&ok.iter().map(|&c| sc[c]).collect::<Vec<_>>(), &ok.iter().map(|&c| resid(k1.candidates[c])).collect::<Vec<_>>());
        assert!((r1.ic_mean - want_ic).abs() < 1e-12, "{} vs {want_ic}", r1.ic_mean);
        let want_ic_ret = spearman(&finite.iter().map(|&c| sc[c]).collect::<Vec<_>>(), &finite.iter().map(|&c| fret(k1.candidates[c])).collect::<Vec<_>>());
        assert!((r1.ic_ret_mean - want_ic_ret).abs() < 1e-12, "{} vs {want_ic_ret}", r1.ic_ret_mean);
        let mut order = finite.clone();
        order.sort_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap());
        let q = (order.len() / 10).max(1);
        let mean = |s: &[usize]| s.iter().map(|&c| fret(k1.candidates[c])).sum::<f64>() / s.len() as f64;
        let want_ls = (mean(&order[..q]) - mean(&order[order.len() - q..])) / one as f64;
        assert!((r1.ls_mean - want_ls).abs() < 1e-15, "{} vs {want_ls}", r1.ls_mean);

        // Giving the missing stocks extreme finite scores changes all of these.
        let mut loud = m.clone();
        for &j in &gone {
            for d in 0..t {
                loud[j * t + d] = if j % 2 == 0 { 1e6 } else { -1e6 };
            }
        }
        let loud = ScoreMatrix::from_f64(n, t, loud).unwrap();
        let r2 = holdout(&store, &plan, &opts(Family::Scores { matrix: &loud, window: 756 }, one, &[0.0], 16)).unwrap();
        assert_ne!(bits(r2.ls_mean), bits(r1.ls_mean));
        assert_ne!(bits(r2.ic_ret_mean), bits(r1.ic_ret_mean));
    }

    #[test]
    fn no_finite_scores_is_an_error_and_infinite_scores_are_refused() {
        let (plan, store) = fixture(60, 3);
        let (n, t) = (plan.var.n, plan.var.t);
        let matrix = ScoreMatrix::from_f64(n, t, vec![f64::NAN; n * t]).unwrap();
        let e = holdout(&store, &plan, &opts(Family::Scores { matrix: &matrix, window: 756 }, 1, &[0.0], 4)).err().expect("all-NaN scores must fail");
        assert!(e.to_string().contains("no candidate has a finite score"), "{e}");
        let mut v = vec![0.5f32; n * t];
        v[7 * t + 100] = f32::INFINITY;
        let e = ScoreMatrix::from_f32(n, t, v).err().expect("inf must fail").to_string();
        assert!(e.contains("infinite score at stock column 7, day index 100"), "{e}");
        assert!(ScoreMatrix::from_f64(n, t + 1, vec![0.0; n * t]).is_err());
    }

    #[test]
    fn scores_files_load_with_checks() {
        let (n, t) = (4, 6);
        let dir = std::env::temp_dir().join(format!("stockai-judge-scores-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let v: Vec<f32> = (0..n * t).map(|k| if k % 5 == 0 { f32::NAN } else { k as f32 - 10.0 }).collect();
        let p = dir.join("s.npy");
        npy::write(&p, &[n, t], &v).unwrap();
        let m = ScoreMatrix::load(&p, n, t).unwrap();
        assert_eq!(m.dtype, "<f4");
        assert!(m.get(0, 0).is_nan());
        assert_eq!(m.get(1, 2), (6 + 2) as f64 - 10.0);
        assert_eq!(m.finite(), n * t - (n * t).div_ceil(5));
        let e = format!("{:#}", ScoreMatrix::load(&p, n, t + 1).err().unwrap());
        assert!(e.contains("shape [4, 6]") && e.contains("[4, 7]"), "{e}");
        let pi = dir.join("i.npy");
        npy::write(&pi, &[n, t], &vec![1i32; n * t]).unwrap();
        let e = format!("{:#}", ScoreMatrix::load(&pi, n, t).err().unwrap());
        assert!(e.contains("float32 or float64"), "{e}");
        let p8 = dir.join("d.npy");
        npy::write(&p8, &[n, t], &v.iter().map(|&x| x as f64).collect::<Vec<_>>()).unwrap();
        let m8 = ScoreMatrix::load(&p8, n, t).unwrap();
        assert_eq!(m8.dtype, "<f8");
        assert_eq!(m8.get(3, 5).to_bits(), m.get(3, 5).to_bits());
    }

    #[test]
    fn delay_holds_the_picks_decided_earlier_and_nothing_before() {
        let (plan, store) = fixture(90, 4);
        let family = Family::Reversal { lookback: 5, window: 756 };
        let r0 = holdout(&store, &plan, &opts(family, 5, &[0.0, 20.0], 32)).unwrap();
        for delay in [1, 2, 7] {
            let mut o = opts(family, 5, &[0.0, 20.0], 32);
            o.delay = delay;
            let r = holdout(&store, &plan, &o).unwrap();
            let (k0, k) = (r0.kept.as_ref().unwrap(), r.kept.as_ref().unwrap());
            assert_eq!(k.picks, k0.picks, "decisions do not depend on the delay");
            for i in 0..k.held.len() {
                if i < delay {
                    assert!(k.held[i].is_empty());
                    assert_eq!(r.curve[i].1, 10_000.0, "nothing held, nothing traded");
                } else {
                    assert_eq!(k.held[i], k0.picks[i - delay]);
                }
            }
            // Decision-day measures are unchanged; the account is not.
            assert_eq!((bits(r.ic_mean), bits(r.ic_ret_mean), bits(r.ls_mean), bits(r.hit_rate)), (bits(r0.ic_mean), bits(r0.ic_ret_mean), bits(r0.ls_mean), bits(r0.hit_rate)));
            assert_ne!(bits(r.strategy[0].1), bits(r0.strategy[0].1));
            // Uniform guessers can buy back a name they just dropped, so they buy at most as often.
            assert!(r.matching.random_buys_max <= r.matching.strategy_buys, "{:?}", r.matching);
        }
    }

    #[test]
    fn volatility_matched_guessers_match_turnover_and_deciles() {
        let (plan, store) = fixture(300, 5);
        for (rebalance, picks) in [(5, 5), (1, 5), (21, 10)] {
            let mut o = opts(Family::Reversal { lookback: 5, window: 756 }, rebalance, &[0.0, 5.0], 40);
            o.picks = picks;
            o.random_match = RandomMatch::Volatility;
            let r = holdout(&store, &plan, &o).unwrap();
            let m = &r.matching;
            assert_eq!((m.random_buys_min, m.random_buys_max), (m.strategy_buys, m.strategy_buys), "turnover matched exactly: {m:?}");
            // Unmatched buys (decile exhausted) are rare; every matched buy has the strategy's decile.
            let share = m.unmatched as f64 / (40 * m.strategy_buys) as f64;
            assert!(share < 0.02, "unmatched share {share}: {m:?}");
            if m.unmatched == 0 {
                assert_eq!(m.random_buy_decile, m.strategy_buy_decile);
            }
            assert!((m.random_buy_decile - m.strategy_buy_decile).abs() <= 9.0 * share + 1e-12, "{m:?}");
            assert!((m.random_held_decile - m.strategy_held_decile).abs() < 1.0, "{m:?}");
            // Same strategy as the uniform run; only the random accounts differ. Deterministic.
            let mut u = opts(Family::Reversal { lookback: 5, window: 756 }, rebalance, &[0.0, 5.0], 40);
            u.picks = picks;
            let ru = holdout(&store, &plan, &u).unwrap();
            assert_eq!(bits(r.strategy[1].1), bits(ru.strategy[1].1));
            assert_eq!((bits(r.ic_t), bits(r.ls_t)), (bits(ru.ic_t), bits(ru.ls_t)));
            let again = holdout(&store, &plan, &o).unwrap();
            same_report(&r, &again);
            // Each guesser's sets: no duplicates, same size as the strategy's.
            let deciles = Deciles::new(&plan.var, &r.kept.as_ref().unwrap().candidates.clone(), &r.kept.as_ref().unwrap().days, None);
            let held = &r.kept.as_ref().unwrap().held;
            for run in 0..5 {
                let (sets, _) = volatility_guesser(run, held, r.candidates, &deciles, None);
                for (s, h) in sets.iter().zip(held) {
                    let mut x = s.clone();
                    x.sort_unstable();
                    x.dedup();
                    assert_eq!((x.len(), s.len()), (h.len(), h.len()));
                }
            }
        }
    }

    /// Extras for the fixture: as-traded closes equal to the adjusted ones, nothing delisted.
    fn plain_extras(plan: &Plan) -> Extras {
        Extras {
            close_unadj: plan.var.close.clone(),
            delist_day: vec![-1; plan.var.n],
            delist_kind: vec![0; plan.var.n],
            source: serde_json::json!({}),
            delisted: [0; 3],
            moved: 0,
            moved_max_days: 0,
            without_close: 0,
        }
    }

    #[test]
    fn delisting_rules_in_daily_and_forward_returns() {
        let (mut plan, store) = fixture(90, 7);
        let t = plan.var.t;
        let family = Family::Reversal { lookback: 5, window: 756 };
        // The first pick, held over the holdout's first 21 days, stops trading after decision day 5.
        let base = holdout(&store, &plan, &opts(family, 21, &[0.0], 8)).unwrap();
        let k0 = base.kept.as_ref().unwrap();
        let xc = k0.held[0][0];
        let x = k0.candidates[xc];
        let l = k0.days[5];
        for d in l + 1..t {
            plan.var.close[x * t + d] = f32::NAN;
            plan.var.v[x * t + d] = f32::NAN;
        }
        let mut extras = plain_extras(&plan);
        extras.delist_day[x] = l as i32;
        let run = |extras: &Extras, rule: Option<DelistRule>, rebalance: usize| {
            let mut o = opts(family, rebalance, &[0.0, 20.0], 64);
            o.extras = Some(extras);
            o.delist = rule;
            holdout(&store, &plan, &o).unwrap()
        };
        let strategy_bits = |r: &Report| r.strategy.iter().map(|s| [bits(s.1), bits(s.2), bits(s.3)]).collect::<Vec<_>>();

        let none = run(&extras, None, 21);
        assert!(none.delisting.is_none());
        let k = none.kept.as_ref().unwrap();
        assert_eq!((k.candidates[xc], k.days[5]), (x, l));
        assert!((0..21).all(|i| k.held[i].contains(&xc)), "held through day 20");
        assert!((21..k.held.len()).all(|i| !k.held[i].contains(&xc)), "not bought again once its prices stop");
        for kind in [DelistKind::Buyout, DelistKind::Failure, DelistKind::Unknown] {
            extras.delist_kind[x] = kind.code();
            let lp = run(&extras, Some(DelistRule::LastPrice), 21);
            let main = run(&extras, Some(DelistRule::Main), 21);
            let tl = run(&extras, Some(DelistRule::TotalLoss), 21);
            // Last price: what a missing price already did (the position earns 0 after its last close).
            // (The random accounts differ from those without a rule: they no longer buy x after it.)
            assert_eq!(lp.strategy.iter().map(|s| bits(s.1)).collect::<Vec<_>>(), none.strategy.iter().map(|s| bits(s.1)).collect::<Vec<_>>(), "{kind:?}");
            // Main: a buyout at its last price, a failure or an unknown delisting as a total loss.
            let want = if kind == DelistKind::Buyout { &lp } else { &tl };
            assert_eq!(strategy_bits(&main), strategy_bits(want), "{kind:?}");
            // Total loss: over day L+1 the held name returns -1 instead of 0; every other day is the same.
            let w = 1.0 / 5.0;
            let ret = |j: usize, d: usize| {
                let (a, b) = (plan.var.close[j * t + d], plan.var.close[j * t + d + 1]);
                if a.is_finite() && b.is_finite() && a > 0.0 { (b / a - 1.0) as f64 } else { 0.0 }
            };
            let g: f64 = k.held[5].iter().filter(|&&c| c != xc).map(|&c| w * ret(k.candidates[c], l)).sum();
            for (a, b) in tl.strategy.iter().zip(&lp.strategy) {
                let want = b.1 * (1.0 + g - w) / (1.0 + g);
                assert!((a.1 / want - 1.0).abs() < 1e-12, "{} bps: {} vs {want}", a.0, a.1);
            }
            let d = tl.delisting.as_ref().unwrap();
            let mut one = [0usize; 3];
            one[kind.code() as usize - 1] = 1;
            assert_eq!((d.rule, d.candidates, d.held, d.dead_before, d.bought_dead), (DelistRule::TotalLoss, one, one, 0, 0));
            assert_eq!(d.random_bought_dead, 0.0);
        }

        // The random accounts follow the same rule: with every candidate failing on day L, every account (and
        // buy-and-hold of all candidates) is worth nothing after it.
        let mut all = plain_extras(&plan);
        for &j in &k.candidates {
            all.delist_day[j] = l as i32;
            all.delist_kind[j] = DelistKind::Failure.code();
        }
        let tl = run(&all, Some(DelistRule::TotalLoss), 21);
        let lp = run(&all, Some(DelistRule::LastPrice), 21);
        assert!(tl.strategy.iter().all(|s| s.1 == 0.0 && s.2 == 0.0 && s.3 == 1.0), "{:?}", tl.strategy);
        assert!(lp.strategy.iter().all(|s| s.1 > 0.0 && s.2 > 0.0));
        assert_eq!(tl.benchmarks.iter().find(|b| b.ticker.is_none()).unwrap().final_value, 0.0);
        let d = tl.delisting.as_ref().unwrap();
        assert_eq!((d.held, d.random_held), ([0, 5, 0], [0.0, 5.0, 0.0]));
        assert_eq!(d.candidates, [0, k.candidates.len(), 0]);

        // Forward returns over a holding period that contains the last close: one rebalance for the whole holdout,
        // so the long-short check's long book (the top tenth of scores, which holds the first pick) moves by
        // x's outcome: 0 without a rule, close[L] / close[d] - 1 at the last price, -1 as a total loss.
        let one = k.days.len();
        extras.delist_kind[x] = DelistKind::Failure.code();
        let r_none = run(&extras, None, one);
        let r_lp = run(&extras, Some(DelistRule::LastPrice), one);
        let r_tl = run(&extras, Some(DelistRule::TotalLoss), one);
        let r_main = run(&extras, Some(DelistRule::Main), one);
        let q = (r_none.candidates / 10).max(1) as f64;
        let d0 = k.days[0];
        let at_last = (plan.var.close[x * t + l] / plan.var.close[x * t + d0] - 1.0) as f64;
        assert!(((r_lp.ls_mean - r_none.ls_mean) * one as f64 * q - at_last).abs() < 1e-9, "{} {} {at_last}", r_lp.ls_mean, r_none.ls_mean);
        assert!(((r_tl.ls_mean - r_none.ls_mean) * one as f64 * q + 1.0).abs() < 1e-9);
        assert_eq!(bits(r_main.ls_mean), bits(r_tl.ls_mean));
        assert_ne!(bits(r_lp.ic_ret_mean), bits(r_tl.ic_ret_mean));
        // The census-variable measures do not depend on the rule.
        assert_eq!((bits(r_tl.ic_mean), bits(r_tl.hit_rate)), (bits(r_none.ic_mean), bits(r_none.hit_rate)));
    }

    #[test]
    fn random_accounts_never_buy_a_stock_after_its_last_close() {
        let (mut plan, store) = fixture(150, 9);
        let t = plan.var.t;
        let family = Family::Reversal { lookback: 5, window: 756 };
        let base = holdout(&store, &plan, &opts(family, 5, &[0.0], 4)).unwrap();
        let k0 = base.kept.as_ref().unwrap();
        // A third of the candidates die: every third of those before the first decision day (still in the
        // universe from the year before), the others on days spread over the holdout.
        let mut extras = plain_extras(&plan);
        let mut g = Gen(77);
        for (i, &j) in k0.candidates.iter().enumerate().filter(|(i, _)| i % 3 == 0) {
            let l = if i % 9 == 0 { k0.days[0] - 1 - (g.u() * 40.0) as usize } else { k0.days[(g.u() * (k0.days.len() - 1) as f64) as usize] };
            for d in l + 1..t {
                plan.var.close[j * t + d] = f32::NAN;
                plan.var.v[j * t + d] = f32::NAN;
            }
            extras.delist_day[j] = l as i32;
            extras.delist_kind[j] = [1, 2, 3][(i / 3) % 3];
        }
        let run = |m: RandomMatch| {
            let mut o = opts(family, 5, &[0.0, 20.0], 64);
            o.random_match = m;
            o.extras = Some(&extras);
            o.delist = Some(DelistRule::Main);
            holdout(&store, &plan, &o).unwrap()
        };
        for m in [RandomMatch::Uniform, RandomMatch::Volatility] {
            let r = run(m);
            let d = r.delisting.as_ref().unwrap();
            assert!(d.dead_before > 0 && d.candidates.iter().sum::<usize>() > 0, "dead and dying candidates: {d:?}");
            assert_eq!((d.bought_dead, d.random_bought_dead), (0, 0.0), "{m:?}");
        }

        // The guessers themselves, on that run's candidates and holdings, against what they did without a rule.
        let r = run(RandomMatch::Uniform);
        let k = r.kept.as_ref().unwrap();
        let nc = k.candidates.len();
        let alive = Alive { last: k.candidates.iter().map(|&j| if extras.delist_day[j] >= 0 { extras.delist_day[j] } else { i32::MAX }).collect(), days: &k.days };
        let everyone = Alive { last: vec![i32::MAX; nc], days: &k.days };
        let dead = |kk: usize, c: usize| !alive.at(kk, c);
        let dead_buys = |sets: &[Vec<usize>]| -> usize {
            (0..sets.len()).map(|kk| sets[kk].iter().filter(|&&c| dead(kk, c) && (kk == 0 || !sets[kk - 1].contains(&c))).count()).sum()
        };
        let d_none = Deciles::new(&plan.var, &k.candidates, &k.days, None);
        let d_alive = Deciles::new(&plan.var, &k.candidates, &k.days, Some(&alive));
        let d_everyone = Deciles::new(&plan.var, &k.candidates, &k.days, Some(&everyone));
        assert_eq!(d_everyone.dec, d_none.dec);
        for kk in 0..k.days.len() {
            for c in (0..nc).filter(|&c| dead(kk, c)) {
                assert_eq!(d_alive.dec[kk][c], NO_DECILE, "a dead stock in a volatility group");
            }
        }
        let (mut before_uniform, mut before_volatility) = (0, 0);
        for run in 0..32 {
            // Nobody dead: the same draws as without a rule (so a Stooq run can't change).
            assert_eq!(uniform_guesser(run, &k.held, nc, Some(&everyone)), uniform_guesser(run, &k.held, nc, None));
            assert_eq!(volatility_guesser(run, &k.held, nc, &d_everyone, Some(&everyone)).0, volatility_guesser(run, &k.held, nc, &d_none, None).0);
            before_uniform += dead_buys(&uniform_guesser(run, &k.held, nc, None));
            before_volatility += dead_buys(&volatility_guesser(run, &k.held, nc, &d_none, None).0);

            let u = uniform_guesser(run, &k.held, nc, Some(&alive));
            assert_eq!(dead_buys(&u), 0, "uniform run {run}");
            // Its members past their last close are the first it sells.
            for kk in 1..u.len() {
                let target = k.held[kk].len().min(nc);
                let replaced = k.held[kk].iter().filter(|c| !k.held[kk - 1].contains(c)).count();
                let sells = u[kk - 1].len() - target.saturating_sub(replaced).min(u[kk - 1].len());
                let dead_before_sells = u[kk - 1].iter().filter(|&&c| dead(kk, c)).count();
                let dead_after = u[kk].iter().filter(|&&c| dead(kk, c)).count();
                assert_eq!(dead_after, dead_before_sells.saturating_sub(sells), "uniform run {run}, day {kk}");
            }
            let (v, _) = volatility_guesser(run, &k.held, nc, &d_alive, Some(&alive));
            assert_eq!(dead_buys(&v), 0, "volatility run {run}");
        }
        assert!(before_uniform > 0 && before_volatility > 0, "without a rule they bought dead stocks: {before_uniform}, {before_volatility}");
    }

    #[test]
    fn traded_price_test_reads_as_traded_closes() {
        let (plan, store) = fixture(90, 8);
        let t = plan.var.t;
        let family = Family::Reversal { lookback: 5, window: 756 };
        let base = holdout(&store, &plan, &opts(family, 5, &[0.0, 20.0], 32)).unwrap();
        let cands = base.kept.as_ref().unwrap().candidates.clone();
        let mut extras = plain_extras(&plan);
        let run = |extras: &Extras, price_test: PriceTest| {
            let mut o = opts(family, 5, &[0.0, 20.0], 32);
            o.extras = Some(extras);
            o.price_test = price_test;
            holdout(&store, &plan, &o).unwrap()
        };
        // With extras but neither a rule nor the traded test, and with as-traded closes equal to the adjusted ones,
        // nothing changes.
        same_report(&run(&extras, PriceTest::Adjusted), &base);
        same_report(&run(&extras, PriceTest::Traded), &base);
        // A stock that traded at a hundredth of its adjusted price (a later 1-for-100 reverse split) leaves the
        // universe.
        let j = cands[3];
        for d in 0..t {
            extras.close_unadj[j * t + d] /= 100.0;
        }
        let traded = run(&extras, PriceTest::Traded);
        let tc = &traded.kept.as_ref().unwrap().candidates;
        assert_eq!(tc.len(), cands.len() - 1);
        assert!(!tc.contains(&j) && tc.iter().all(|c| cands.contains(c)));
        same_report(&run(&extras, PriceTest::Adjusted), &base);
        // The price test and the rules need extras.
        let mut o = opts(family, 5, &[0.0], 4);
        o.price_test = PriceTest::Traded;
        assert!(holdout(&store, &plan, &o).is_err());
        let mut o = opts(family, 5, &[0.0], 4);
        o.delist = Some(DelistRule::Main);
        assert!(holdout(&store, &plan, &o).is_err());
    }

    #[test]
    fn realized_vol_uses_data_through_the_day_only() {
        let (mut plan, _) = fixture(20, 6);
        let t = plan.var.t;
        let (j, d) = (3, 800);
        let before = realized_vol(&plan.var, j, d);
        assert!(before.is_finite() && before > 0.0);
        for k in d + 1..t {
            plan.var.close[j * t + k] *= 3.0;
        }
        assert_eq!(realized_vol(&plan.var, j, d).to_bits(), before.to_bits());
        plan.var.close[j * t + d] *= 3.0;
        assert_ne!(realized_vol(&plan.var, j, d).to_bits(), before.to_bits());
    }
}
