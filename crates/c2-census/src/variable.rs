//! The census variable: market-residual daily log returns in units of each stock's own trailing
//! volatility, clipped at +-5. Everything at day t uses only data through day t-1 (betas,
//! volatility, liquidity), so values never change when later data arrives. That fixed unit is
//! what lets running sums be extended as new days come in.

use anyhow::Result;
use c2_data::{Calendar, Kind, Panel, Store};
use rayon::prelude::*;

pub const LOOKBACK: usize = 252;
pub const MIN_OBS: usize = 126;
const LIQUIDITY_EVERY: usize = 21;
pub const MIN_PRICE: f32 = 5.0;
pub const MIN_DOLLAR_VOLUME: f32 = 5e6;
pub const CLIP: f32 = 5.0;

/// Column-major `t x n` matrices over calendar rows, one column per stock.
pub struct Variable {
    pub days: Vec<i32>,
    pub ids: Vec<usize>,
    pub t: usize,
    pub n: usize,
    /// Census variable; NaN where unavailable (missing price, or not enough history yet).
    pub v: Vec<f32>,
    pub close: Vec<f32>,
    pub dollar_volume: Vec<f32>,
    /// Equal-weight mean daily log return of the liquid stocks, per row (kept for the explainer).
    pub market: Vec<f64>,
}

impl Variable {
    pub fn col<'a>(&self, m: &'a [f32], j: usize) -> &'a [f32] {
        &m[j * self.t..(j + 1) * self.t]
    }
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let mid = v.len() / 2;
    *v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
}

/// Builds the production census variable for every common stock with data on or after `from`.
/// Rows start ~3 years earlier so it is valid from `from`: the market needs a year of liquidity
/// history, then beta and volatility each need at least half a year more.
pub fn production(store: &Store, cal: &Calendar, from: i32) -> Result<Variable> {
    let ids: Vec<usize> = (0..store.symbols.len())
        .filter(|&i| store.symbols[i].kind == Kind::Stock && store.symbols[i].last >= from)
        .collect();
    let rows = cal.range(from - 1100, i32::MAX);
    let mut panel = Panel::build(store, cal, rows, &ids);
    panel.repair_bad_prints();
    let (t, n) = (panel.t, panel.n);

    // Daily log returns.
    let mut ret = vec![f32::NAN; t * n];
    ret.par_chunks_mut(t).zip(panel.close.par_chunks(t)).for_each(|(r, c)| {
        for d in 1..t {
            r[d] = (c[d] / c[d - 1]).ln();
        }
    });
    let dollar_volume: Vec<f32> = panel.close.iter().zip(&panel.volume).map(|(c, v)| c * v).collect();

    // Liquidity flag, re-evaluated every 21 rows from the previous 252 rows (point-in-time).
    let mut liquid = vec![false; t * n];
    liquid.par_chunks_mut(t).enumerate().for_each(|(j, flag)| {
        let (c, dv) = (&panel.close[j * t..(j + 1) * t], &dollar_volume[j * t..(j + 1) * t]);
        let mut d = LOOKBACK;
        while d < t {
            let (mut pc, mut pv): (Vec<f32>, Vec<f32>) = (d - LOOKBACK..d).filter(|&k| c[k].is_finite()).map(|k| (c[k], dv[k])).unzip();
            let ok = pc.len() >= LOOKBACK / 2 && median(&mut pc) >= MIN_PRICE && median(&mut pv) >= MIN_DOLLAR_VOLUME;
            for f in flag.iter_mut().skip(d).take(LIQUIDITY_EVERY) {
                *f = ok;
            }
            d += LIQUIDITY_EVERY;
        }
    });

    // Equal-weight market return over liquid stocks.
    let chunk = n.div_ceil(rayon::current_num_threads() * 4).max(1);
    let (sum, cnt) = (0..n)
        .collect::<Vec<_>>()
        .par_chunks(chunk)
        .map(|js| {
            let (mut s, mut c) = (vec![0f64; t], vec![0u32; t]);
            for &j in js {
                for d in 0..t {
                    let r = ret[j * t + d];
                    if liquid[j * t + d] && r.is_finite() {
                        s[d] += r as f64;
                        c[d] += 1;
                    }
                }
            }
            (s, c)
        })
        .reduce(
            || (vec![0f64; t], vec![0u32; t]),
            |(mut s1, mut c1), (s2, c2)| {
                for d in 0..t {
                    s1[d] += s2[d];
                    c1[d] += c2[d];
                }
                (s1, c1)
            },
        );
    let market: Vec<f64> = sum.iter().zip(&cnt).map(|(s, &c)| if c > 0 { s / c as f64 } else { f64::NAN }).collect();

    // Trailing beta (rows d-252..d-1), residual, trailing residual volatility, census variable.
    let mut v = vec![f32::NAN; t * n];
    v.par_chunks_mut(t).enumerate().for_each(|(j, out)| {
        let r = &ret[j * t..(j + 1) * t];
        let mut resid = vec![f64::NAN; t];
        // Sliding sums over valid (r, m) pairs.
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
                    let beta = (srm - sr * sm / kf) / var_m;
                    resid[d] = r[d] as f64 - beta * market[d];
                }
            }
        }
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
                    out[d] = ((resid[d] - s / k as f64) / var.sqrt()).clamp(-CLIP as f64, CLIP as f64) as f32;
                }
            }
        }
    });

    Ok(Variable { days: panel.days, ids, t, n, v, close: panel.close, dollar_volume, market })
}

/// The same standardization as `research/lag_spectrum.py` and `census_oracle.py`, for
/// validation only: betas from window rows against the equal-weight mean of `ids`, residuals
/// standardized with window mean and sd, clipped at +-5, missing as 0. Returns `w x n` f64
/// (window rows only) for symbols `ids`, in that order.
pub fn python_style(store: &Store, cal: &Calendar, ids: &[usize], load_from: i32, window: (i32, i32)) -> Vec<f64> {
    let mut panel = Panel::build(store, cal, cal.range(load_from, i32::MAX), ids);
    panel.repair_bad_prints();
    let (t, n) = (panel.t, panel.n);
    let w0 = panel.days.partition_point(|&d| d < window.0);
    let w1 = panel.days.partition_point(|&d| d <= window.1);
    let ret: Vec<f64> = (0..t * n)
        .map(|k| if k % t == 0 { f64::NAN } else { (panel.close[k] as f64 / panel.close[k - 1] as f64).ln() })
        .collect();
    let nanmean = |it: &mut dyn Iterator<Item = f64>| {
        let (s, c) = it.filter(|v| v.is_finite()).fold((0.0, 0usize), |(s, c), v| (s + v, c + 1));
        if c > 0 { s / c as f64 } else { f64::NAN }
    };
    let mkt: Vec<f64> = (0..t).map(|d| nanmean(&mut (0..n).map(|j| ret[j * t + d]))).collect();
    let mkt_mean = nanmean(&mut (w0..w1).map(|d| mkt[d]));
    let mm: f64 = (w0..w1).map(|d| mkt[d] - mkt_mean).filter(|v| v.is_finite()).map(|v| v * v).sum();
    let mut out = vec![0f64; (w1 - w0) * n];
    out.par_chunks_mut(w1 - w0).enumerate().for_each(|(j, o)| {
        let r = &ret[j * t..(j + 1) * t];
        let rm = nanmean(&mut (w0..w1).map(|d| r[d]));
        let num: f64 = (w0..w1).map(|d| (r[d] - rm) * (mkt[d] - mkt_mean)).filter(|v| v.is_finite()).sum();
        let beta = num / mm;
        let e: Vec<f64> = (w0..w1).map(|d| r[d] - beta * mkt[d]).collect();
        let em = nanmean(&mut e.iter().copied());
        let valid: Vec<f64> = e.iter().copied().filter(|v| v.is_finite()).collect();
        let sd = (valid.iter().map(|v| (v - em).powi(2)).sum::<f64>() / valid.len().max(1) as f64).sqrt();
        let sd = if sd > 0.0 { sd } else { 1.0 };
        for (k, v) in e.iter().enumerate() {
            o[k] = if v.is_finite() { ((v - em) / sd).clamp(-5.0, 5.0) } else { 0.0 };
        }
    });
    out
}
