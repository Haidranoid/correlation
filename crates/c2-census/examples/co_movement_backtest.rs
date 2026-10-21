//! Out-of-sample test of the same-day co-movement tools (`c2_census::together`) against random
//! choices: does each tool's advice, made with data up to a date, beat a random choice over the
//! following year? The design was fixed before any outcome was computed; it is written out in
//! `experiments/+5.1-co-movement-tools.md` ("The test"). In short:
//!
//! * Dates: October 17 (or the trading day before) of 2008..2025. The tools see the two years up
//!   to the date (`Universe::build(store, as_of, 2)`, as the CLI); the following year runs to the
//!   next date, and for 2025-10-17 to the last day of data (2026-10-19).
//! * Test portfolios per date: 200 random equal-weight 5-stock portfolios from the 500 most-traded
//!   liquid stocks, and 100 concentrated ones (5 random members of a random group of that date's
//!   groups map, among groups with 5 or more members). Seeds fixed per date.
//! * Best diversifiers: `diversifiers(u, p, 0.10, 0.75, 15)` (the CLI's defaults); the first fund
//!   row and the first single-stock row, each added at 10% (holdings at 90%). Next-year volatility
//!   of the 6 holdings (days where all six have a return, sd x sqrt(252)) against the expectation
//!   over every candidate of the same kind the tool's own filters keep (uniform random), the 20
//!   such candidates nearest the pick in past volatility (volatility-matched), and SPY.
//! * Hedge finder: each of the 500 most-traded liquid stocks; `hedges(u, ticker, all)`. The top
//!   tracker (and the top-tracking fund) at its past ratio; next-year risk left = sd(stock - ratio
//!   x hedge) / sd(stock), against every scanned fund at its own past ratio, and SPY at its beta.
//! * Portfolio check: predicted independent bets against next year's, same portfolios.
//! * Groups map: `groups(u, 20, 1000, 0)`; next year's within-group market-relative (and raw)
//!   correlation against random groups of the same sizes (200 relabelings).
//!
//! Run (CPU only; 8 threads):
//!   cargo build --release -p c2-census --example co_movement_backtest
//!   target/release/examples/co_movement_backtest [--out DIR] [--smoke] [--check] [--clean]
//!
//! `--clean` (a robustness run, decided after the smoke run showed an unadjusted reverse split
//! of +957% in MSTU's next year, and before the full run): next-year daily moves of more than 2x
//! up or down are treated as missing for every series alike.
//!
//! It reads the default price store (never writes it) and writes TSVs under `--out` (default
//! `~/.local/share/correlation2/experiments/co_movement_tools/`): `diversify.tsv` (one row per
//! portfolio and arm), `portfolio.tsv`, `hedge.tsv`, `groups.tsv`, `groups_detail.tsv`, `checks.tsv`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use c2_census::together::{self, Universe};
use c2_data::{Calendar, Store};
use rayon::prelude::*;

const ADD: f64 = 0.10;
const MAX_VOL: f64 = 0.75;
const SHOW: usize = 15;
const RANDOM_PORTFOLIOS: usize = 200;
const CONCENTRATED: usize = 100;
const HOLDINGS: usize = 5;
const POOL: usize = 500;
const MATCHED: usize = 20;
const MIN_FUTURE: usize = 126;
const GROUPS: usize = 20;
const GROUP_STOCKS: usize = 1000;
const SHUFFLES: usize = 200;
const YEAR: usize = together::YEAR;

// ---------------------------------------------------------------------------------------------
// Copies of the private helpers in `together.rs`, verbatim in effect.

#[derive(Clone, Copy, Debug)]
struct Pair {
    n: usize,
    corr: f64,
    cov: f64,
    var_a: f64,
    var_b: f64,
}

fn pair(a: impl Iterator<Item = f64>, b: impl Iterator<Item = f64>) -> Pair {
    let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0usize, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (x, y) in a.zip(b) {
        if x.is_finite() && y.is_finite() {
            n += 1;
            sa += x;
            sb += y;
            saa += x * x;
            sbb += y * y;
            sab += x * y;
        }
    }
    if n < 3 {
        return Pair { n, corr: f64::NAN, cov: f64::NAN, var_a: f64::NAN, var_b: f64::NAN };
    }
    let m = n as f64;
    let cov = (sab - sa * sb / m) / (m - 1.0);
    let (var_a, var_b) = ((saa - sa * sa / m) / (m - 1.0), (sbb - sb * sb / m) / (m - 1.0));
    let corr = if var_a > 0.0 && var_b > 0.0 { cov / (var_a * var_b).sqrt() } else { f64::NAN };
    Pair { n, corr, cov, var_a, var_b }
}

fn annual(var: f64) -> f64 {
    (var * YEAR as f64).sqrt()
}

fn round2(r: f64) -> f64 {
    (r * 100.0).round() / 100.0
}

/// Sample variance of the finite values.
fn var_of(x: &[f64]) -> f64 {
    pair(x.iter().copied(), x.iter().copied()).var_a
}

// ---------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// `k` distinct elements of `pool`, in draw order.
    fn choose(&mut self, pool: &[usize], k: usize) -> Vec<usize> {
        let mut v = pool.to_vec();
        for i in 0..k.min(v.len()) {
            let j = i + self.below(v.len() - i);
            v.swap(i, j);
        }
        v.truncate(k);
        v
    }
}

/// Next-year daily returns of every column of the tool's universe `u`, looked up by ticker in a
/// universe `f` built at the end of the following year: rows with days in (after, until].
struct Future {
    t: usize,
    ret: Vec<f32>,
    first: i32,
    last: i32,
}

impl Future {
    /// With `clean`, a next-year daily move of more than 2x up or down is treated as missing (a
    /// data error such as an unadjusted reverse split; the tool's own repair only catches 2x moves
    /// that undo themselves the next day). Returns the struct and how many returns were blanked.
    fn build(u: &Universe, f: &Universe, after: i32, until: i32, clean: bool) -> (Future, usize) {
        let rows: Vec<usize> = (0..f.t).filter(|&r| f.days[r] > after && f.days[r] <= until).collect();
        let t = rows.len();
        let mut ret = vec![f32::NAN; u.n * t];
        let blanked: usize = ret
            .par_chunks_mut(t.max(1))
            .enumerate()
            .map(|(j, out)| {
                let mut b = 0;
                if let Some(k) = f.find(&u.tickers[j]) {
                    let c = f.col(k);
                    for (o, &r) in out.iter_mut().zip(&rows) {
                        *o = c[r];
                        if clean && o.is_finite() && !(*o >= -0.5 && *o <= 1.0) {
                            *o = f32::NAN;
                            b += 1;
                        }
                    }
                }
                b
            })
            .sum();
        (Future { t, ret, first: f.days[rows[0]], last: f.days[*rows.last().unwrap()] }, blanked)
    }
    fn col(&self, j: usize) -> &[f32] {
        &self.ret[j * self.t..(j + 1) * self.t]
    }
}

/// Next-year volatility and independent bets of weights `w` on columns `cols` (the tool's
/// formulas: days where every column has a return; bets = (sum w vol / vol)^2), and the days used.
fn realized(fu: &Future, cols: &[usize], w: &[f64]) -> (f64, f64, usize) {
    let k = cols.len();
    let rows: Vec<usize> = (0..fu.t).filter(|&d| cols.iter().all(|&j| fu.col(j)[d].is_finite())).collect();
    if rows.len() < 3 {
        return (f64::NAN, f64::NAN, rows.len());
    }
    let port: Vec<f64> = rows.iter().map(|&d| (0..k).map(|a| w[a] * fu.col(cols[a])[d] as f64).sum()).collect();
    let vol = annual(var_of(&port));
    let avg: f64 = (0..k)
        .map(|a| {
            let x: Vec<f64> = rows.iter().map(|&d| fu.col(cols[a])[d] as f64).collect();
            w[a] * annual(var_of(&x))
        })
        .sum();
    (vol, (avg / vol).powi(2), rows.len())
}

fn f(x: f64) -> String {
    if x.is_finite() { format!("{x:.6}") } else { "nan".into() }
}

struct Eligible {
    j: usize,
    etf: bool,
    corr: f64,
    vol: f64,
}

/// The candidates `diversifiers` ranks before it collapses look-alikes and cuts to `show`: liquid,
/// not held, a return on 90% of the portfolio's days, finite correlation, 3% to `MAX_VOL`
/// volatility, and not in the hedge partition.
fn eligible(u: &Universe, cols: &[usize], rows: &[usize], port: &[f64]) -> Vec<Eligible> {
    let spy = u.find("SPY");
    let year = u.year(0).unwrap();
    let mut v: Vec<Eligible> = (0..u.n)
        .into_par_iter()
        .filter(|&j| u.liquid[j] && !cols.contains(&j))
        .filter_map(|j| {
            let c = u.col(j);
            let s = pair(rows.iter().map(|&d| c[d] as f64), port.iter().copied());
            if s.n * 10 < rows.len() * 9 || !s.corr.is_finite() {
                return None;
            }
            let vol = annual(s.var_a);
            if !(0.03..=MAX_VOL).contains(&vol) {
                return None;
            }
            let spy_corr = spy.map(|s| pair(year.clone().map(|d| c[d] as f64), year.clone().map(|d| u.col(s)[d] as f64)).corr).unwrap_or(0.0);
            if round2(s.corr) <= -0.3 || round2(spy_corr) <= -0.5 {
                return None;
            }
            Some(Eligible { j, etf: u.etf[j], corr: s.corr, vol })
        })
        .collect();
    v.sort_by(|a, b| a.corr.total_cmp(&b.corr).then(u.tickers[a.j].cmp(&u.tickers[b.j])));
    v
}

struct PortfolioOut {
    portfolio_row: String,
    arm_rows: Vec<String>,
    check: Vec<(String, String)>,
}

/// One test portfolio: the tool's advice at the date and what happened over the next year.
fn test_portfolio(u: &Universe, fu: &Future, label: &str, cols: &[usize]) -> Result<PortfolioOut> {
    let k = cols.len();
    let h: Vec<(String, f64)> = cols.iter().map(|&j| (u.tickers[j].clone(), 1.0)).collect();
    let p = together::portfolio(u, &h)?;
    let d = together::diversifiers(u, &p, ADD, MAX_VOL, SHOW);
    // The rows and portfolio returns `diversifiers` used (private in the report): recomputed the
    // same way.
    let w: Vec<f64> = vec![1.0 / k as f64; k];
    let year = u.year(0).unwrap();
    let rows: Vec<usize> = year.filter(|&d| cols.iter().all(|&j| u.col(j)[d].is_finite())).collect();
    let port: Vec<f64> = rows.iter().map(|&d| (0..k).map(|a| w[a] * u.col(cols[a])[d] as f64).sum()).collect();
    let el = eligible(u, cols, &rows, &port);
    let tick = cols.iter().map(|&j| u.tickers[j].as_str()).collect::<Vec<_>>().join(",");
    let (fvol, fbets, fn_) = realized(fu, cols, &w);
    let head = format!("{label}\t{tick}\t{}\t{}\t{}", f(p.vol), f(p.bets), p.days.2);
    let portfolio_row = format!("{head}\t{fn_}\t{}\t{}", f(fvol), f(fbets));
    let mut arm_rows = Vec::new();
    let mut check = Vec::new();
    if fn_ < MIN_FUTURE {
        return Ok(PortfolioOut { portfolio_row, arm_rows, check: vec![("portfolio".into(), "dropped: under 126 next-year days".into())] });
    }
    // Next-year volatility and bets with candidate `j` at 10% (holdings at 90%), if it has a return
    // on 90% of the portfolio's next-year days.
    let mut cw: Vec<f64> = w.iter().map(|x| x * (1.0 - ADD)).collect();
    cw.push(ADD);
    let with = |j: usize| -> Option<(f64, f64)> {
        let mut c = cols.to_vec();
        c.push(j);
        let (v, b, n) = realized(fu, &c, &cw);
        (n * 10 >= fn_ * 9 && v.is_finite()).then_some((v, b))
    };
    let spy = u.find("SPY").and_then(&with);
    for (arm, list) in [("fund", &d.funds), ("stock", &d.stocks)] {
        let Some(pick) = list.first() else {
            check.push((arm.into(), "no pick".into()));
            continue;
        };
        let pj = u.find(&pick.ticker).unwrap();
        let kind: Vec<&Eligible> = el.iter().filter(|e| e.etf == (arm == "fund")).collect();
        // The pick must be one of the eligible candidates: the lowest-correlation one, or a less
        // volatile look-alike that took a row's place.
        let status = match kind.iter().position(|e| e.j == pj) {
            Some(0) => "first".to_string(),
            Some(i) => {
                let alike = pair(rows.iter().map(|&d| u.col(kind[0].j)[d] as f64), rows.iter().map(|&d| u.col(pj)[d] as f64)).corr >= 0.95;
                format!("{}{i}", if alike { "alike-of-first:" } else { "row-reordered:" })
            }
            None => "NOT-ELIGIBLE".to_string(),
        };
        check.push((arm.into(), status.split(':').next().unwrap().to_string()));
        let Some((pv, pb)) = with(pj) else {
            check.push((format!("{arm}-future"), "pick dropped: no next-year data".into()));
            continue;
        };
        // Every eligible candidate of the kind, next year.
        let outs: Vec<(f64, f64, f64)> = kind.par_iter().filter_map(|e| with(e.j).map(|(v, b)| (v, b, e.vol))).collect();
        let mean = |x: &[(f64, f64, f64)], i: usize| x.iter().map(|t| [t.0, t.1, t.2][i]).sum::<f64>() / x.len() as f64;
        let rank = outs.iter().filter(|t| t.0 < pv).count() as f64 / outs.len() as f64;
        let mut vols: Vec<f64> = outs.iter().map(|t| t.0).collect();
        vols.sort_by(f64::total_cmp);
        let median = if vols.is_empty() { f64::NAN } else if vols.len() % 2 == 1 { vols[vols.len() / 2] } else { 0.5 * (vols[vols.len() / 2 - 1] + vols[vols.len() / 2]) };
        // Volatility-matched: the 20 eligible candidates of the kind nearest the pick in past
        // volatility, the pick excluded.
        let mut near: Vec<(f64, usize)> = kind.iter().filter(|e| e.j != pj).map(|e| ((e.vol / pick.vol).ln().abs(), e.j)).collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut matched: Vec<(f64, f64, f64)> = Vec::new();
        for &(_, j) in &near {
            if matched.len() == MATCHED {
                break;
            }
            if let Some((v, b)) = with(j) {
                matched.push((v, b, kind.iter().find(|e| e.j == j).unwrap().vol));
            }
        }
        arm_rows.push(format!(
            "{head}\t{fn_}\t{}\t{}\t{arm}\t{}\t{status}\t{}\t{}\t{}\t{pv:.6}\t{pb:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{rank:.6}\t{}",
            f(fvol),
            f(fbets),
            pick.ticker,
            f(pick.corr),
            f(pick.vol),
            f(pick.vol_after),
            kind.len(),
            outs.len(),
            f(mean(&outs, 0)),
            f(mean(&outs, 1)),
            f(mean(&outs, 2)),
            f(mean(&matched, 0)),
            f(mean(&matched, 1)),
            f(mean(&matched, 2)),
            spy.map(|s| f(s.0)).unwrap_or("nan".into()),
            spy.map(|s| f(s.1)).unwrap_or("nan".into()),
            f(median),
        ));
    }
    Ok(PortfolioOut { portfolio_row, arm_rows, check })
}

/// Next-year risk left of a hedge at ratio `ratio`: sd(x - ratio h) / sd(x) over the days both
/// have a return, if that is 90% of x's next-year days.
fn risk_left(fu: &Future, x: usize, h: usize, ratio: f64, nx: usize) -> Option<f64> {
    let (a, b) = (fu.col(x), fu.col(h));
    let both: Vec<usize> = (0..fu.t).filter(|&d| a[d].is_finite() && b[d].is_finite()).collect();
    if both.len() * 10 < nx * 9 || both.len() < 3 {
        return None;
    }
    let xs: Vec<f64> = both.iter().map(|&d| a[d] as f64).collect();
    let hs: Vec<f64> = both.iter().map(|&d| a[d] as f64 - ratio * b[d] as f64).collect();
    let r = (var_of(&hs) / var_of(&xs)).sqrt();
    r.is_finite().then_some(r)
}

fn test_hedge(u: &Universe, fu: &Future, ticker: &str) -> Result<Option<String>> {
    let r = together::hedges(u, ticker, usize::MAX)?;
    let x = u.find(ticker).unwrap();
    let nx = fu.col(x).iter().filter(|v| v.is_finite()).count();
    if nx < MIN_FUTURE || r.closest.is_empty() {
        return Ok(None);
    }
    let xs: Vec<f64> = fu.col(x).iter().map(|&v| v as f64).collect();
    let fut_vol = annual(var_of(&xs));
    let real = |h: &together::HedgeRow| u.find(&h.ticker).and_then(|j| risk_left(fu, x, j, h.ratio, nx));
    let top = &r.closest[0];
    let top_real = real(top);
    let fund = r.closest.iter().find(|h| h.etf);
    let fund_real = fund.and_then(real);
    let funds: Vec<(f64, f64)> = r.closest.iter().filter(|h| h.etf).filter_map(|h| real(h).map(|v| (h.risk_left, v))).collect();
    let n = funds.len() as f64;
    let rand_pred = funds.iter().map(|t| t.0).sum::<f64>() / n;
    let rand_real = funds.iter().map(|t| t.1).sum::<f64>() / n;
    let mut reals: Vec<f64> = funds.iter().map(|t| t.1).collect();
    reals.sort_by(f64::total_cmp);
    let rand_median = if reals.is_empty() { f64::NAN } else if reals.len() % 2 == 1 { reals[reals.len() / 2] } else { 0.5 * (reals[reals.len() / 2 - 1] + reals[reals.len() / 2]) };
    let fund_rank = fund_real.map(|v| funds.iter().filter(|t| t.1 < v).count() as f64 / n).unwrap_or(f64::NAN);
    let spy = r.closest.iter().find(|h| h.ticker == "SPY");
    let spy_real = spy.and_then(real);
    let opt = |v: Option<f64>| v.map(f).unwrap_or("nan".into());
    Ok(Some(format!(
        "{ticker}\t{nx}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        f(fut_vol),
        top.ticker,
        together::kind(top.etf),
        f(top.corr),
        f(top.ratio),
        f(top.risk_left),
        opt(top_real),
        fund.map(|h| h.ticker.clone()).unwrap_or("-".into()),
        opt(fund.map(|h| h.corr)),
        opt(fund.map(|h| h.ratio)),
        opt(fund.map(|h| h.risk_left)),
        opt(fund_real),
        funds.len(),
        f(rand_pred),
        f(rand_real),
        f(fund_rank),
        opt(spy.map(|h| h.corr)),
        opt(spy.map(|h| h.ratio)),
        opt(spy.map(|h| h.risk_left)),
        opt(spy_real),
        r.scanned,
        f(rand_median),
    )))
}

/// Market-relative (or raw) standardized columns, the tool's `standardized`: `x` is `n` columns
/// of `t` rows; the market is the equal-weight mean of the finite values each day; each column
/// minus its beta times the market, centred, unit sum of squares, missing = 0.
fn standardized(x: &[f32], n: usize, t: usize, relative: bool) -> Vec<f64> {
    let market: Vec<f64> = (0..t)
        .map(|d| {
            let (s, c) = (0..n).map(|j| x[j * t + d]).filter(|v| v.is_finite()).fold((0.0, 0usize), |(s, c), v| (s + v as f64, c + 1));
            if c > 0 { s / c as f64 } else { 0.0 }
        })
        .collect();
    let mut z = vec![0f64; n * t];
    z.par_chunks_mut(t).enumerate().for_each(|(j, out)| {
        let col: Vec<f64> = x[j * t..(j + 1) * t].iter().map(|&v| v as f64).collect();
        let resid: Vec<f64> = if relative {
            let p = pair(col.iter().copied(), market.iter().copied());
            let beta = if p.var_b > 0.0 { p.cov / p.var_b } else { 0.0 };
            col.iter().zip(&market).map(|(v, m)| v - beta * m).collect()
        } else {
            col
        };
        let fin: Vec<f64> = resid.iter().copied().filter(|v| v.is_finite()).collect();
        let mean = fin.iter().sum::<f64>() / fin.len().max(1) as f64;
        let norm = fin.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>().sqrt();
        for (o, v) in out.iter_mut().zip(&resid) {
            *o = if v.is_finite() && norm > 0.0 { (v - mean) / norm } else { 0.0 };
        }
    });
    z
}

fn corr_of(z: &[f64], n: usize, t: usize) -> Vec<f64> {
    let mut c = vec![0f64; n * n];
    c.par_chunks_mut(n).enumerate().for_each(|(a, row)| {
        let za = &z[a * t..(a + 1) * t];
        for (b, o) in row.iter_mut().enumerate() {
            *o = za.iter().zip(&z[b * t..(b + 1) * t]).map(|(p, q)| p * q).sum();
        }
    });
    c
}

/// Pair-weighted mean of the within-group correlations, and the equal-weighted mean of the
/// groups' own means.
fn within(c: &[f64], n: usize, label: &[usize], k: usize) -> (f64, f64) {
    let mut s = vec![0.0; k];
    let mut m = vec![0usize; k];
    for a in 0..n {
        for b in a + 1..n {
            if label[a] == label[b] {
                s[label[a]] += c[a * n + b];
                m[label[a]] += 1;
            }
        }
    }
    let pooled = s.iter().sum::<f64>() / m.iter().sum::<usize>() as f64;
    let groups: Vec<f64> = (0..k).filter(|&g| m[g] > 0).map(|g| s[g] / m[g] as f64).collect();
    (pooled, groups.iter().sum::<f64>() / groups.len() as f64)
}

fn all_pairs(c: &[f64], n: usize) -> f64 {
    let mut s = 0.0;
    for a in 0..n {
        for b in a + 1..n {
            s += c[a * n + b];
        }
    }
    s / (n * (n - 1) / 2) as f64
}

struct GroupsOut {
    row: String,
    detail: Vec<String>,
    max_diff: f64,
}

fn test_groups(u: &Universe, fu: &Future, m: &together::GroupsMap, seed: u64) -> GroupsOut {
    let n = m.stocks.len();
    let k = m.groups.len();
    let year = u.year(0).unwrap();
    let past: Vec<f32> = m.stocks.iter().flat_map(|&j| u.col(j)[year.clone()].to_vec()).collect();
    let fut: Vec<f32> = m.stocks.iter().flat_map(|&j| fu.col(j).to_vec()).collect();
    let (tp, tf) = (year.len(), fu.t);
    let past_rel = corr_of(&standardized(&past, n, tp, true), n, tp);
    let past_raw = corr_of(&standardized(&past, n, tp, false), n, tp);
    let fut_rel = corr_of(&standardized(&fut, n, tf, true), n, tf);
    let fut_raw = corr_of(&standardized(&fut, n, tf, false), n, tf);
    // This re-implementation against the tool's own matrix (the market there is the mean of the
    // same stocks; only the column order differs, and the tool's product is f32).
    let max_diff = (0..n * n).map(|i| (past_rel[i] - m.corr[i] as f64).abs()).fold(0.0, f64::max);
    let label = &m.group_of;
    let mut rng = Rng(seed);
    let shuffled = |c: &[f64], rng: &mut Rng| -> (f64, f64) {
        let v: Vec<f64> = (0..SHUFFLES)
            .map(|_| {
                let idx: Vec<usize> = (0..n).collect();
                let perm = rng.choose(&idx, n);
                let lab: Vec<usize> = perm.iter().map(|&i| label[i]).collect();
                within(c, n, &lab, k).0
            })
            .collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (mean, (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt())
    };
    let mut cols = Vec::new();
    for c in [&past_rel, &fut_rel, &past_raw, &fut_raw] {
        let (w, eq) = within(c, n, label, k);
        let all = all_pairs(c, n);
        let (sm, ssd) = shuffled(c, &mut rng);
        cols.push(format!("{}\t{}\t{}\t{}\t{}", f(w), f(eq), f(all), f(sm), f(ssd)));
    }
    let nfut = m.stocks.iter().filter(|&&j| fu.col(j).iter().filter(|v| v.is_finite()).count() >= MIN_FUTURE).count();
    let row = format!("{n}\t{k}\t{nfut}\t{}", cols.join("\t"));
    let detail = m
        .groups
        .iter()
        .enumerate()
        .map(|(g, grp)| {
            let mean_of = |c: &[f64]| {
                let mem = &grp.members;
                let (mut s, mut cnt) = (0.0, 0usize);
                for a in 0..mem.len() {
                    for b in a + 1..mem.len() {
                        s += c[mem[a] * n + mem[b]];
                        cnt += 1;
                    }
                }
                s / cnt.max(1) as f64
            };
            format!(
                "{g}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                grp.members.len(),
                grp.tracks.as_ref().map(|t| t.0.clone()).unwrap_or("-".into()),
                f(grp.corr_relative),
                f(mean_of(&fut_rel)),
                f(grp.corr_raw),
                f(mean_of(&fut_raw)),
                grp.members.iter().take(8).map(|&i| m.tickers[i].as_str()).collect::<Vec<_>>().join(","),
                f(mean_of(&past_rel)),
            )
        })
        .collect();
    GroupsOut { row, detail, max_diff }
}

fn main() -> Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(8).build_global().ok();
    let args: Vec<String> = std::env::args().collect();
    let flag = |s: &str| args.iter().any(|a| a == s);
    let out_dir = args
        .iter()
        .position(|a| a == "--out")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| c2_data::data_dir().join("experiments/co_movement_tools"));
    let smoke = flag("--smoke");
    let clean = flag("--clean");
    let store = Store::open(&c2_data::store_dir())?;
    eprintln!("store {}", store.dir().display());

    if flag("--check") {
        // The CLI's `c2 diversify AAPL MSFT NVDA --as-of 2022-06-30 --show 5`, through the same
        // calls, plus this file's eligible set.
        let u = Universe::build(&store, Some(c2_data::parse_date("2022-06-30")?), 2)?;
        let h = together::parse_holdings("AAPL MSFT NVDA")?;
        let p = together::portfolio(&u, &h)?;
        let d = together::diversifiers(&u, &p, 0.10, 0.75, 5);
        println!("bets {:.2} vol {:.4}", p.bets, p.vol);
        for (t, l) in [("funds", &d.funds), ("stocks", &d.stocks), ("hedges", &d.hedges)] {
            println!("{t}: {}", l.iter().map(|c| format!("{} {:+.2} {:.1}%", c.ticker, c.corr, 100.0 * c.vol_after)).collect::<Vec<_>>().join(", "));
        }
        let cols: Vec<usize> = h.iter().map(|(t, _)| u.find(t).unwrap()).collect();
        let rows: Vec<usize> = u.year(0).unwrap().filter(|&d| cols.iter().all(|&j| u.col(j)[d].is_finite())).collect();
        let port: Vec<f64> = rows.iter().map(|&d| (0..3).map(|a| (1.0 / 3.0) * u.col(cols[a])[d] as f64).sum()).collect();
        let el = eligible(&u, &cols, &rows, &port);
        for etf in [true, false] {
            println!("eligible {}: {} first {}", if etf { "funds" } else { "stocks" }, el.iter().filter(|e| e.etf == etf).count(), el.iter().filter(|e| e.etf == etf).take(8).map(|e| format!("{} {:+.2}", u.tickers[e.j], e.corr)).collect::<Vec<_>>().join(", "));
        }
        println!("scanned {} cash-like {} too volatile {}", d.scanned, d.cash_like, d.too_volatile);
        let r = together::hedges(&u, "XOM", 3)?;
        println!("hedge XOM: {}", r.closest.iter().map(|h| format!("{} {:+.2} ratio {:.2}", h.ticker, h.corr, h.ratio)).collect::<Vec<_>>().join(", "));
        return Ok(());
    }

    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let open = |name: &str, header: &str| -> Result<BufWriter<File>> {
        let mut w = BufWriter::new(File::create(out_dir.join(name))?);
        writeln!(w, "{header}")?;
        Ok(w)
    };
    let base = "as_of\tnext_from\tnext_to\tset\tid\tholdings\tpast_vol\tpast_bets\tpast_days\tnext_days\tnext_vol\tnext_bets";
    let mut w_port = open("portfolio.tsv", base)?;
    let mut w_div = open(
        "diversify.tsv",
        &format!("{base}\tarm\tpick\tstatus\tpick_corr\tpick_vol\tpred_vol_after\tnext_vol_pick\tnext_bets_pick\tn_eligible\tn_eligible_next\tnext_vol_uniform\tnext_bets_uniform\tpast_vol_uniform\tnext_vol_matched\tnext_bets_matched\tpast_vol_matched\tnext_vol_spy\tnext_bets_spy\trank_pick\tnext_vol_uniform_median"),
    )?;
    let mut w_hedge = open(
        "hedge.tsv",
        "as_of\tnext_from\tnext_to\tticker\tnext_days\tnext_vol\ttop\ttop_kind\ttop_corr\ttop_ratio\ttop_pred\ttop_next\tfund\tfund_corr\tfund_ratio\tfund_pred\tfund_next\tn_funds\tfunds_pred\tfunds_next\tfund_rank\tspy_corr\tspy_ratio\tspy_pred\tspy_next\tscanned\tfunds_next_median",
    )?;
    let g_cols = ["past_rel", "next_rel", "past_raw", "next_raw"].iter().flat_map(|c| ["within", "within_eq", "all", "shuffle_mean", "shuffle_sd"].map(|s| format!("{c}_{s}"))).collect::<Vec<_>>().join("\t");
    let mut w_groups = open("groups.tsv", &format!("as_of\tnext_from\tnext_to\tstocks\tgroups\tstocks_with_next\t{g_cols}\tmax_diff_vs_tool"))?;
    let mut w_gd = open("groups_detail.tsv", "as_of\tgroup\tsize\ttracks\tpast_rel\tnext_rel\tpast_raw\tnext_raw\tmembers\tpast_rel_check")?;
    let mut w_check = open("checks.tsv", "as_of\tset\tid\twhat\tstatus")?;

    let cal = Calendar::trading_days(&store)?;
    let last = *cal.days.last().unwrap();
    let years: Vec<i32> = (2008..=2025).collect();
    let mut dates: Vec<i32> = years.iter().map(|&y| cal.days[cal.days.partition_point(|&x| x <= c2_data::days_from_civil(y, 10, 17)) - 1]).collect();
    dates.push(last);
    let t0 = std::time::Instant::now();
    let todo: Vec<usize> = if smoke { vec![0, dates.len() - 2] } else { (0..dates.len() - 1).collect() };
    for i in todo {
        let (d, next) = (dates[i], dates[i + 1]);
        let u = Universe::build(&store, Some(d), 2)?;
        let fu_u = Universe::build(&store, Some(next), 2)?;
        let (fu, blanked) = Future::build(&u, &fu_u, u.as_of(), next, clean);
        drop(fu_u);
        let ds = c2_data::date_string(u.as_of());
        let span = format!("{ds}\t{}\t{}", c2_data::date_string(fu.first), c2_data::date_string(fu.last));
        let seed = 0xC0_40E5 ^ (d as u64).wrapping_mul(0x9E37_79B9);
        let mut rng = Rng(seed);
        // The 500 most-traded liquid stocks.
        let mut pool: Vec<usize> = (0..u.n).filter(|&j| !u.etf[j] && u.liquid[j] && u.dollar_volume[0][j].is_finite()).collect();
        pool.sort_by(|&a, &b| u.dollar_volume[0][b].total_cmp(&u.dollar_volume[0][a]).then(a.cmp(&b)));
        pool.truncate(POOL);
        let m = together::groups(&u, GROUPS, GROUP_STOCKS, 0, &|_| {})?;
        let big: Vec<usize> = (0..m.groups.len()).filter(|&g| m.groups[g].members.len() >= HOLDINGS).collect();
        let (n_r, n_c) = if smoke { (6, 4) } else { (RANDOM_PORTFOLIOS, CONCENTRATED) };
        let mut tests: Vec<(String, Vec<usize>)> = (0..n_r).map(|p| (format!("random\t{p}"), rng.choose(&pool, HOLDINGS))).collect();
        for p in 0..n_c {
            let g = big[rng.below(big.len())];
            let members: Vec<usize> = m.groups[g].members.iter().map(|&i| m.stocks[i]).collect();
            tests.push((format!("concentrated\t{p}"), rng.choose(&members, HOLDINGS)));
        }
        let results: Vec<(String, Result<PortfolioOut>)> = tests.par_iter().map(|(label, cols)| (label.clone(), test_portfolio(&u, &fu, label, cols))).collect();
        let (mut arms, mut bad) = (0, 0);
        for (label, r) in results {
            match r {
                Ok(o) => {
                    writeln!(w_port, "{span}\t{}", o.portfolio_row)?;
                    for a in &o.arm_rows {
                        writeln!(w_div, "{span}\t{a}")?;
                        arms += 1;
                    }
                    for (what, status) in o.check {
                        bad += usize::from(status.contains("NOT-ELIGIBLE"));
                        writeln!(w_check, "{ds}\t{label}\t{what}\t{status}")?;
                    }
                }
                Err(e) => writeln!(w_check, "{ds}\t{label}\tportfolio\terror: {e}")?,
            }
        }
        let hedge_stocks: Vec<usize> = if smoke { pool[..10].to_vec() } else { pool.clone() };
        let hedges: Vec<Result<Option<String>>> = hedge_stocks.par_iter().map(|&j| test_hedge(&u, &fu, &u.tickers[j])).collect();
        let mut nh = 0;
        for r in hedges {
            match r {
                Ok(Some(row)) => {
                    writeln!(w_hedge, "{span}\t{row}")?;
                    nh += 1;
                }
                Ok(None) => {}
                Err(e) => writeln!(w_check, "{ds}\thedge\t-\thedge\terror: {e}")?,
            }
        }
        let g = test_groups(&u, &fu, &m, seed ^ 0x6E0);
        writeln!(w_groups, "{span}\t{}\t{:.6}", g.row, g.max_diff)?;
        for row in &g.detail {
            writeln!(w_gd, "{ds}\t{row}")?;
        }
        eprintln!(
            "{ds} -> {}: {} portfolios, {arms} arms ({bad} picks not in the eligible set), {nh} hedges, groups check max diff {:.2e}, {blanked} next-year returns blanked [{:.0}s]",
            c2_data::date_string(fu.last),
            tests.len(),
            g.max_diff,
            t0.elapsed().as_secs_f64()
        );
        for w in [&mut w_port, &mut w_div, &mut w_hedge, &mut w_groups, &mut w_gd, &mut w_check] {
            w.flush()?;
        }
    }
    Ok(())
}
