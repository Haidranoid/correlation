//! Family B: cluster lead-lag, after Bennett, Cucuringu & Reinert (2022), "Lead-lag detection and
//! network clustering for multivariate time series with an application to the US equity market"
//! (Machine Learning; arXiv 2201.08283), sections 3 and 6.
//!
//! Every `refit` trading days, on the `window` days up to and including the decision day:
//! 1. Universe: the `stocks` stocks with the largest median dollar volume among those with a
//!    close on at least 90% of the days and on the decision day (NYSE only with `nyse_only`, as
//!    in the paper). There is no price floor, so the look-ahead of the census's $5 rule on
//!    split-adjusted closes doesn't reach this family.
//! 2. Lead-lag matrix, the paper's ccf-auc: I(i, j) = sum over offsets l = 1..=max_lag of
//!    |corr(Y_i(t - l), Y_j(t))|, with distance correlation (the paper's choice) or Pearson, and
//!    S(i, j) = sign(I(i, j) - I(j, i)) * max(I(i, j), I(j, i)) / (I(i, j) + I(j, i)).
//!    S(i, j) > 0: i moves first and j follows.
//! 3. Hermitian clustering of i·S into `groups` groups (`embed`, `kmeans`).
//! 4. Meta-flow F(a, b), the mean of S over a's members against b's; a link a -> b where F(a, b)
//!    is above the `link_quantile` quantile of the off-diagonal F.
//! 5. For every pair of groups, OLS without intercept of group b's mean return on day t on the
//!    exponentially weighted (alpha) sum of group a's mean returns before day t.
//!
//! Each day until the next refit, group b's forecast for the next day is the sum, over the links
//! a -> b, of theta(a, b) times a's weighted sum through the decision day. The paper's signal
//! holds b's stocks long when the forecast is positive and short when it is negative.

use std::ops::Range;

use anyhow::{Result, anyhow};
use c2_data::Store;
use rayon::prelude::*;

use crate::variable::Variable;

/// The correlation in the lead-lag measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Distance correlation (Székely et al. 2007), which also sees non-linear dependence; the
    /// paper's choice for US stocks.
    Distance,
    Pearson,
}

/// The daily series the groups are found and forecast on. Trading results always use the
/// stocks' actual returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Returns {
    /// Daily log returns of closes carried forward over missing days, as in the paper.
    Raw,
    /// The census variable: market-relative, in units of the stock's own volatility.
    MarketRelative,
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub stocks: usize,
    pub groups: usize,
    pub max_lag: usize,
    /// Trading days each fit looks back over.
    pub window: usize,
    /// Trading days between fits.
    pub refit: usize,
    pub alpha: f64,
    pub link_quantile: f64,
    pub metric: Metric,
    pub returns: Returns,
    pub nyse_only: bool,
}

impl Settings {
    /// The paper's settings: 500 stocks by dollar volume, 10 groups, offsets 1 to 5 days with
    /// distance correlation on raw returns, a year's look-back refit every 2 months, alpha 0.4,
    /// links above the 90% quantile, NYSE only.
    pub fn paper() -> Settings {
        Settings {
            stocks: 500,
            groups: 10,
            max_lag: 5,
            window: 252,
            refit: 42,
            alpha: 0.4,
            link_quantile: 0.9,
            metric: Metric::Distance,
            returns: Returns::Raw,
            nyse_only: true,
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "top {} {}stocks by dollar volume, {} groups, offsets 1-{} days, {} correlation of {}, {}-day look-back refit every {} days, alpha {}, links above the {:.0}% quantile",
            self.stocks,
            if self.nyse_only { "NYSE " } else { "" },
            self.groups,
            self.max_lag,
            match self.metric { Metric::Distance => "distance", Metric::Pearson => "Pearson" },
            match self.returns { Returns::Raw => "raw daily returns", Returns::MarketRelative => "market-relative returns (the census variable)" },
            self.window,
            self.refit,
            self.alpha,
            100.0 * self.link_quantile
        )
    }
}

/// Rows before a fit's window that warm up the exponentially weighted sums (0.6^30 < 1e-6).
const WARM: usize = 30;

/// Stock `j`'s daily returns on `rows`, in the units of `kind`. Raw returns carry the last close
/// forward over missing days (0 on a day without a close; the first close after a gap gets the
/// whole move), as the paper forward-fills prices.
fn returns(var: &Variable, j: usize, rows: Range<usize>, kind: Returns) -> Vec<f64> {
    let vt = var.t;
    match kind {
        Returns::MarketRelative => rows.map(|d| var.v[j * vt + d]).map(|v| if v.is_finite() { v as f64 } else { 0.0 }).collect(),
        Returns::Raw => {
            let c = &var.close[j * vt..(j + 1) * vt];
            let ok = |d: usize| c[d].is_finite() && c[d] > 0.0;
            let mut last = (rows.start.saturating_sub(WARM)..rows.start).rev().find(|&d| ok(d)).map(|d| c[d] as f64);
            rows.map(|d| {
                if !ok(d) {
                    return 0.0;
                }
                let now = c[d] as f64;
                let r = last.map(|p| (now / p).ln()).unwrap_or(0.0);
                last = Some(now);
                r
            })
            .collect()
        }
    }
}

/// Simple return of stock `j` from the close of row `d` to the close of row `d + 1` (0 without
/// both closes): what a position held over day d + 1 earns.
pub fn next_day_return(var: &Variable, j: usize, d: usize) -> f64 {
    let (a, b) = (var.close[j * var.t + d], var.close[j * var.t + d + 1]);
    if a.is_finite() && b.is_finite() && a > 0.0 { (b / a - 1.0) as f64 } else { 0.0 }
}

/// The fit's universe at decision row `r` (see the module comment), ascending columns.
fn universe(store: &Store, var: &Variable, r: usize, s: &Settings) -> Vec<usize> {
    let vt = var.t;
    let w0 = r + 1 - s.window;
    let mut ranked: Vec<(usize, f32)> = (0..var.n)
        .into_par_iter()
        .filter(|&j| !s.nyse_only || store.symbols[var.ids[j]].exchange == "nyse")
        .filter_map(|j| {
            let c = &var.close[j * vt + w0..=j * vt + r];
            if !c[s.window - 1].is_finite() || c.iter().filter(|x| x.is_finite()).count() * 10 < s.window * 9 {
                return None;
            }
            if s.returns == Returns::MarketRelative && var.v[j * vt + w0..=j * vt + r].iter().filter(|x| x.is_finite()).count() * 10 < s.window * 9 {
                return None;
            }
            let mut dv: Vec<f32> = var.dollar_volume[j * vt + w0..=j * vt + r].iter().copied().filter(|x| x.is_finite()).collect();
            let mid = dv.len() / 2;
            Some((j, *dv.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1))
        })
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(s.stocks);
    let mut cols: Vec<usize> = ranked.into_iter().map(|(j, _)| j).collect();
    cols.sort_unstable();
    cols
}

/// Doubly centred distance matrix of `a`, upper triangle with the off-diagonal entries scaled by
/// sqrt 2, so the dot product of two such vectors is the sum over the full matrices.
fn centred_distances(a: &[f64], out: &mut [f32]) {
    let n = a.len();
    let mut row = vec![0.0; n];
    for s in 0..n {
        row[s] = a.iter().map(|&x| (a[s] - x).abs()).sum::<f64>() / n as f64;
    }
    let grand = row.iter().sum::<f64>() / n as f64;
    let mut k = 0;
    for s in 0..n {
        for t in s..n {
            let v = (a[s] - a[t]).abs() - row[s] - row[t] + grand;
            out[k] = (if s == t { v } else { v * std::f64::consts::SQRT_2 }) as f32;
            k += 1;
        }
    }
}

/// The ccf-auc lead-lag matrix S (row-major p x p) of the `w x p` column-major panel `y`.
fn lead_lag(y: &[f64], w: usize, p: usize, s: &Settings) -> Vec<f64> {
    // i_sum[i * p + j] = I(i, j) = sum over offsets of |corr(Y_i(t - l), Y_j(t))|.
    let mut i_sum = vec![0.0f64; p * p];
    for l in 1..=s.max_lag {
        let n = w - l;
        let mut c = vec![0f32; p * p];
        match s.metric {
            Metric::Pearson => {
                // Leaders' first n days against followers' last n days, standardized to unit norm.
                let standardized = |part: &[f64]| -> Vec<f32> {
                    let mean = part.iter().sum::<f64>() / n as f64;
                    let norm = part.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>().sqrt();
                    part.iter().map(|x| if norm > 0.0 { ((x - mean) / norm) as f32 } else { 0.0 }).collect()
                };
                let head: Vec<f32> = (0..p).into_par_iter().flat_map_iter(|j| standardized(&y[j * w..j * w + n])).collect();
                let tail: Vec<f32> = (0..p).into_par_iter().flat_map_iter(|j| standardized(&y[j * w + l..(j + 1) * w])).collect();
                c2_engine::cpu::xprod_f32(&head, &tail, n, p, n, &mut c);
            }
            Metric::Distance => {
                let m = n * (n + 1) / 2;
                let (mut head, mut tail) = (vec![0f32; p * m], vec![0f32; p * m]);
                head.par_chunks_mut(m).enumerate().for_each(|(j, out)| centred_distances(&y[j * w..j * w + n], out));
                tail.par_chunks_mut(m).enumerate().for_each(|(j, out)| centred_distances(&y[j * w + l..(j + 1) * w], out));
                let dvar = |v: &[f32]| -> Vec<f64> { v.par_chunks(m).map(|x| x.iter().map(|&a| a as f64 * a as f64).sum()).collect() };
                let (vh, vt) = (dvar(&head), dvar(&tail));
                c2_engine::cpu::xprod_f32(&head, &tail, m, p, m, &mut c);
                c.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
                    for (j, x) in row.iter_mut().enumerate() {
                        let den = (vh[i] * vt[j]).sqrt();
                        *x = if den > 0.0 { (*x as f64 / den).max(0.0).sqrt() as f32 } else { 0.0 };
                    }
                });
            }
        }
        i_sum.par_iter_mut().zip(c.par_iter()).for_each(|(a, &b)| *a += (b as f64).abs());
    }
    let mut sm = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            let (a, b) = (i_sum[i * p + j], i_sum[j * p + i]);
            if i != j && a + b > 0.0 && a != b {
                sm[i * p + j] = (a - b).signum() * a.max(b) / (a + b);
            }
        }
    }
    sm
}

/// Hermitian clustering's embedding (Cucuringu et al. 2020) of the skew-symmetric `sm`: row i of
/// the returned p x k matrix is stock i's coordinates. i·B, with B = D^-1/2 S D^-1/2 and D the
/// degrees sum_j |S(i, j)|, is Hermitian; its eigenvalues come in pairs +mu, -mu with conjugate
/// eigenvectors g and conj(g), and the real span of Re g and Im g is the eigenspace of the real
/// symmetric B^T B for mu^2. So the projector onto i·B's top k eigenvectors by |mu| (k/2 pairs,
/// the paper's l = k) is the projector onto B^T B's top k eigenvectors, and k-means on the rows
/// of that projector, the paper's step, gives the same distances as k-means on the rows of any
/// orthonormal basis of the eigenspace. The paper's random-walk normalization is taken in this
/// symmetric form, which keeps B skew-symmetric; ccf-auc entries are all 0.5..1 in size, so the
/// degrees hardly differ anyway.
fn embed(sm: &[f64], p: usize, k: usize) -> Result<Vec<f64>> {
    let deg: Vec<f64> = (0..p).map(|i| sm[i * p..(i + 1) * p].iter().map(|x| x.abs()).sum::<f64>()).collect();
    let inv: Vec<f64> = deg.iter().map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect();
    // Columns of B, contiguous: bc[i * p + l] = B(l, i).
    let mut bc = vec![0.0; p * p];
    for l in 0..p {
        for i in 0..p {
            bc[i * p + l] = sm[l * p + i] * inv[l] * inv[i];
        }
    }
    let m: Vec<f64> = (0..p * p)
        .into_par_iter()
        .map(|ij| {
            let (i, j) = (ij / p, ij % p);
            bc[i * p..(i + 1) * p].iter().zip(&bc[j * p..(j + 1) * p]).map(|(a, b)| a * b).sum()
        })
        .collect();
    let evd = faer::Mat::<f64>::from_fn(p, p, |i, j| m[i * p + j])
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| anyhow!("eigendecomposition failed: {e:?}"))?;
    let u = evd.U();
    // Eigenvalues ascend: the top k are the last k columns.
    Ok((0..p).flat_map(|i| (0..k).map(move |c| u[(i, p - 1 - c)])).collect())
}

fn rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn uniform(state: &mut u64) -> f64 {
    (rng(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// k-means on the rows of the p x dim matrix `x`: k-means++ starts, Lloyd iterations, the
/// lowest within-group sum of squares of `restarts` seeded tries. An emptied group takes the
/// point farthest from its centre.
fn kmeans(x: &[f64], p: usize, dim: usize, k: usize, seed: u64, restarts: usize) -> Vec<usize> {
    let dist = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(u, v)| (u - v) * (u - v)).sum::<f64>();
    let row = |i: usize| &x[i * dim..(i + 1) * dim];
    let mut best: (f64, Vec<usize>) = (f64::INFINITY, Vec::new());
    for run in 0..restarts {
        let mut state = seed ^ (run as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        rng(&mut state);
        let mut centres: Vec<f64> = row((rng(&mut state) % p as u64) as usize).to_vec();
        let mut near: Vec<f64> = (0..p).map(|i| dist(row(i), &centres[..dim])).collect();
        while centres.len() < k * dim {
            let total: f64 = near.iter().sum();
            let mut pick = uniform(&mut state) * total;
            let mut chosen = p - 1;
            for (i, &d) in near.iter().enumerate() {
                if pick < d {
                    chosen = i;
                    break;
                }
                pick -= d;
            }
            let c = row(chosen).to_vec();
            for (i, n) in near.iter_mut().enumerate() {
                *n = n.min(dist(row(i), &c));
            }
            centres.extend(c);
        }
        let mut label = vec![usize::MAX; p];
        for _ in 0..300 {
            let new: Vec<usize> = (0..p)
                .map(|i| (0..k).min_by(|&a, &b| dist(row(i), &centres[a * dim..(a + 1) * dim]).total_cmp(&dist(row(i), &centres[b * dim..(b + 1) * dim]))).unwrap())
                .collect();
            if new == label {
                break;
            }
            label = new;
            centres.iter_mut().for_each(|c| *c = 0.0);
            let mut size = vec![0usize; k];
            for i in 0..p {
                size[label[i]] += 1;
                for (c, v) in centres[label[i] * dim..(label[i] + 1) * dim].iter_mut().zip(row(i)) {
                    *c += v;
                }
            }
            for g in 0..k {
                if size[g] == 0 {
                    let far = (0..p).max_by(|&a, &b| dist(row(a), &centres[label[a] * dim..(label[a] + 1) * dim]).total_cmp(&dist(row(b), &centres[label[b] * dim..(label[b] + 1) * dim]))).unwrap();
                    centres[g * dim..(g + 1) * dim].copy_from_slice(row(far));
                    label[far] = g;
                    size[g] = 1;
                } else {
                    for c in &mut centres[g * dim..(g + 1) * dim] {
                        *c /= size[g] as f64;
                    }
                }
            }
        }
        let inertia: f64 = (0..p).map(|i| dist(row(i), &centres[label[i] * dim..(label[i] + 1) * dim])).sum();
        if inertia < best.0 {
            best = (inertia, label);
        }
    }
    best.1
}

/// Adjusted Rand index of two labellings of the same stocks.
pub fn adjusted_rand(a: &[usize], b: &[usize]) -> f64 {
    let (ka, kb) = (a.iter().max().map_or(0, |m| m + 1), b.iter().max().map_or(0, |m| m + 1));
    let mut table = vec![0f64; ka * kb];
    for (&x, &y) in a.iter().zip(b) {
        table[x * kb + y] += 1.0;
    }
    let pairs = |n: f64| n * (n - 1.0) / 2.0;
    let index: f64 = table.iter().map(|&n| pairs(n)).sum();
    let rows: f64 = (0..ka).map(|x| pairs(table[x * kb..(x + 1) * kb].iter().sum())).sum();
    let cols: f64 = (0..kb).map(|y| pairs((0..ka).map(|x| table[x * kb + y]).sum())).sum();
    let expected = rows * cols / pairs(a.len() as f64);
    let max = (rows + cols) / 2.0;
    if max - expected == 0.0 { 0.0 } else { (index - expected) / (max - expected) }
}

/// The linked pairs of groups and their regressions, for one labelling.
#[derive(Clone)]
pub struct Model {
    /// Meta-flow F, k x k row-major: F(a, b) > 0 when a moves first.
    pub flow: Vec<f64>,
    /// a -> b is used: F(a, b) above the quantile.
    pub links: Vec<bool>,
    /// theta(a, b): group b's next-day mean return per unit of a's weighted sum.
    pub theta: Vec<f64>,
}

/// One refit: the universe, its lead-lag matrix and groups, and the members' returns the
/// regressions and forecasts need.
pub struct Fit {
    /// Decision row of the fit; it uses rows up to and including it.
    pub row: usize,
    /// Variable columns of the universe, ascending.
    pub members: Vec<usize>,
    /// Group of each member, numbered from the group that moves first most (0) to the one that
    /// follows most (the paper's leadingness order).
    pub group: Vec<usize>,
    /// Each member's row sum of S: positive when it tends to move first.
    pub lead: Vec<f64>,
    pub model: Model,
    sm: Vec<f64>,
    /// Members' returns on rows `row + 1 - window - WARM ..= row`, `rows` per member.
    y: Vec<f64>,
    rows: usize,
}

impl Fit {
    pub fn new(store: &Store, var: &Variable, row: usize, s: &Settings) -> Result<Fit> {
        anyhow::ensure!(row + 1 >= s.window + WARM, "not enough history before {} for a {}-day fit", c2_data::date_string(var.days[row]), s.window);
        let members = universe(store, var, row, s);
        let p = members.len();
        anyhow::ensure!(p >= 4 * s.groups, "only {p} stocks qualify on {}", c2_data::date_string(var.days[row]));
        let from = row + 1 - s.window - WARM;
        let rows = s.window + WARM;
        let y: Vec<f64> = members.par_iter().flat_map_iter(|&j| returns(var, j, from..row + 1, s.returns)).collect();
        // The lead-lag matrix uses the window only, not the warm-up rows.
        let window: Vec<f64> = (0..p).flat_map(|i| y[i * rows + WARM..(i + 1) * rows].iter().copied()).collect();
        let sm = lead_lag(&window, s.window, p, s);
        let lead: Vec<f64> = (0..p).map(|i| sm[i * p..(i + 1) * p].iter().sum()).collect();
        let x = embed(&sm, p, s.groups)?;
        let raw = kmeans(&x, p, s.groups, s.groups, 0x5EED ^ row as u64, 10);
        // Number the groups by leadingness: mean row sum of their members, highest first.
        let mut score = vec![(0.0, 0usize); s.groups];
        for (i, &g) in raw.iter().enumerate() {
            score[g].0 += lead[i];
            score[g].1 += 1;
        }
        let mut order: Vec<usize> = (0..s.groups).collect();
        order.sort_by(|&a, &b| (score[b].0 / score[b].1.max(1) as f64).total_cmp(&(score[a].0 / score[a].1.max(1) as f64)));
        let mut rank = vec![0; s.groups];
        for (k, &g) in order.iter().enumerate() {
            rank[g] = k;
        }
        let group: Vec<usize> = raw.iter().map(|&g| rank[g]).collect();
        let mut fit = Fit { row, members, group, lead, model: Model { flow: vec![], links: vec![], theta: vec![] }, sm, y, rows };
        fit.model = fit.model_for(&fit.group, s);
        Ok(fit)
    }

    /// Group mean returns, `rows` per group, for a labelling.
    fn group_means(&self, label: &[usize], k: usize) -> Vec<f64> {
        let mut g = vec![0.0; k * self.rows];
        let mut size = vec![0usize; k];
        for (i, &a) in label.iter().enumerate() {
            size[a] += 1;
            for (o, v) in g[a * self.rows..(a + 1) * self.rows].iter_mut().zip(&self.y[i * self.rows..(i + 1) * self.rows]) {
                *o += v;
            }
        }
        for a in 0..k {
            for o in &mut g[a * self.rows..(a + 1) * self.rows] {
                *o /= size[a].max(1) as f64;
            }
        }
        g
    }

    /// Meta-flow, links and regressions for a labelling (the fit's own groups, or a shuffled one
    /// for the paper's null).
    pub fn model_for(&self, label: &[usize], s: &Settings) -> Model {
        let (p, k) = (self.members.len(), s.groups);
        let mut flow = vec![0.0; k * k];
        let mut size = vec![0.0; k];
        for i in 0..p {
            size[label[i]] += 1.0;
            for j in 0..p {
                flow[label[i] * k + label[j]] += self.sm[i * p + j];
            }
        }
        for a in 0..k {
            for b in 0..k {
                flow[a * k + b] = if a == b || size[a] == 0.0 || size[b] == 0.0 { 0.0 } else { flow[a * k + b] / (size[a] * size[b]) };
            }
        }
        let mut off: Vec<f64> = (0..k * k).filter(|&ab| ab / k != ab % k).map(|ab| flow[ab]).collect();
        off.sort_by(|a, b| a.total_cmp(b));
        let cut = quantile(&off, s.link_quantile);
        let links: Vec<bool> = (0..k * k).map(|ab| ab / k != ab % k && flow[ab] > cut).collect();
        // Regressions over the window rows; the weighted sums start in the warm-up rows.
        let g = self.group_means(label, k);
        let x = weighted_sums(&g, self.rows, k, s.alpha);
        let mut theta = vec![0.0; k * k];
        for a in 0..k {
            let xa = &x[a * self.rows + WARM..(a + 1) * self.rows];
            let xx: f64 = xa.iter().map(|v| v * v).sum();
            for b in 0..k {
                let gb = &g[b * self.rows + WARM..(b + 1) * self.rows];
                theta[a * k + b] = if xx > 0.0 { xa.iter().zip(gb).map(|(u, v)| u * v).sum::<f64>() / xx } else { 0.0 };
            }
        }
        Model { flow, links, theta }
    }

    /// Group forecasts for days `row + 1 ..= row + days`: entry [i * k + b] is made at the close
    /// of row `row + i` from data through that row. `ahead` holds the members' returns on rows
    /// `row + 1 .. row + days` (column-major, `days - 1` per member; the last day's return is
    /// never needed).
    pub fn forecasts(&self, label: &[usize], model: &Model, ahead: &[f64], days: usize, s: &Settings) -> Vec<f64> {
        let (p, k) = (self.members.len(), s.groups);
        let g = self.group_means(label, k);
        let x = weighted_sums(&g, self.rows, k, s.alpha);
        // State: each group's weighted sum for the next day, from returns through the current row.
        let mut next: Vec<f64> = (0..k).map(|a| g[a * self.rows + self.rows - 1] + (1.0 - s.alpha) * x[a * self.rows + self.rows - 1]).collect();
        let mut size = vec![0f64; k];
        for &a in label {
            size[a] += 1.0;
        }
        let mut out = vec![0.0; days * k];
        let span = days.saturating_sub(1);
        for i in 0..days {
            for b in 0..k {
                out[i * k + b] = (0..k).filter(|&a| model.links[a * k + b]).map(|a| model.theta[a * k + b] * next[a]).sum();
            }
            if i + 1 < days {
                // Row row + i + 1's returns arrive.
                let mut mean = vec![0.0; k];
                for m in 0..p {
                    mean[label[m]] += ahead[m * span + i];
                }
                for a in 0..k {
                    next[a] = mean[a] / size[a].max(1.0) + (1.0 - s.alpha) * next[a];
                }
            }
        }
        out
    }

    /// The members' returns on rows `row + 1 .. row + days` for `forecasts`.
    pub fn ahead(&self, var: &Variable, days: usize, s: &Settings) -> Vec<f64> {
        let end = self.row + days.max(1);
        self.members.par_iter().flat_map_iter(|&j| returns(var, j, self.row + 1..end, s.returns)).collect()
    }
}

/// x(t) = sum over l >= 1 of (1 - alpha)^(l - 1) g(t - l), per group (`rows` per group).
fn weighted_sums(g: &[f64], rows: usize, k: usize, alpha: f64) -> Vec<f64> {
    let mut x = vec![0.0; k * rows];
    for a in 0..k {
        for t in 1..rows {
            x[a * rows + t] = g[a * rows + t - 1] + (1.0 - alpha) * x[a * rows + t - 1];
        }
    }
    x
}

/// Linear-interpolation quantile of sorted values (numpy's default).
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Scores for the holdout harness on decision rows `days` (consecutive), refitting every
/// `s.refit` of them: each member's group forecast, with members of a group ordered by how much
/// they follow (lowest row sum first) through a tiny factor that keeps the sign. Stocks outside
/// the current fit's universe get NaN. Returns (candidates, per-day scores on rebalance days,
/// links used summed over fits, fits).
pub fn scores(store: &Store, var: &Variable, days: &[usize], rebalance: usize, s: &Settings) -> Result<(Vec<usize>, Vec<Option<Vec<f64>>>, usize, usize)> {
    let mut blocks = Vec::new();
    for start in (0..days.len()).step_by(s.refit.max(1)) {
        let n = s.refit.min(days.len() - start);
        let fit = Fit::new(store, var, days[start], s)?;
        let ahead = fit.ahead(var, n, s);
        let f = fit.forecasts(&fit.group, &fit.model, &ahead, n, s);
        blocks.push((start, n, fit, f));
    }
    let mut candidates: Vec<usize> = blocks.iter().flat_map(|(_, _, fit, _)| fit.members.iter().copied()).collect();
    candidates.sort_unstable();
    candidates.dedup();
    let pos = |j: usize| candidates.binary_search(&j).unwrap();
    let mut out: Vec<Option<Vec<f64>>> = vec![None; days.len()];
    let mut links = 0;
    for (start, n, fit, f) in &blocks {
        links += fit.model.links.iter().filter(|&&l| l).count();
        let p = fit.members.len();
        // Follow rank: 0 for the member that moves first most.
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&a, &b| fit.lead[b].total_cmp(&fit.lead[a]));
        let mut follow = vec![0.0; p];
        for (r, &m) in order.iter().enumerate() {
            follow[m] = r as f64 / p as f64;
        }
        for i in 0..*n {
            if (start + i) % rebalance != 0 {
                continue;
            }
            let mut sc = vec![f64::NAN; candidates.len()];
            for (m, &j) in fit.members.iter().enumerate() {
                sc[pos(j)] = f[i * s.groups + fit.group[m]] * (1.0 + 1e-9 * follow[m]);
            }
            out[start + i] = Some(sc);
        }
    }
    Ok((candidates, out, links, blocks.len()))
}

/// The paper's own test over decision rows `first ..= last` (holding each next day): daily
/// returns of the sign portfolio, and of the same pipeline with shuffled group labels.
pub struct Backtest {
    /// Day held (the date of row d + 1).
    pub day: Vec<i32>,
    /// Equal weight over the universe: +1/p per member of a group forecast up, -1/p down.
    pub ret: Vec<f64>,
    /// Sum of |change in weight| from the day before (trading per day, as a fraction of capital).
    pub turnover: Vec<f64>,
    /// Gross exposure: share of the universe held long or short.
    pub exposure: Vec<f64>,
    /// The market's daily log return (equal weight of liquid stocks), for correlation.
    pub market: Vec<f64>,
    /// Shuffled-label pipelines' daily returns, one series per draw.
    pub null: Vec<Vec<f64>>,
    pub fits: Vec<FitSummary>,
}

pub struct FitSummary {
    pub day: i32,
    pub stocks: usize,
    pub sizes: Vec<usize>,
    pub links: Vec<(usize, usize)>,
    /// Adjusted Rand index against the previous fit's groups, on the stocks in both.
    pub ari_prev: f64,
    pub seconds: f64,
}

pub fn backtest(store: &Store, var: &Variable, first: usize, last: usize, s: &Settings, draws: usize, progress: &dyn Fn(&str)) -> Result<Backtest> {
    anyhow::ensure!(first <= last && last + 1 < var.t, "backtest outside the data");
    let k = s.groups;
    let mut bt = Backtest { day: vec![], ret: vec![], turnover: vec![], exposure: vec![], market: vec![], null: vec![Vec::new(); draws], fits: vec![] };
    let mut prev_weights: Vec<(usize, f64)> = Vec::new();
    let mut prev_fit: Option<(Vec<usize>, Vec<usize>)> = None;
    let mut start = first;
    while start <= last {
        let n = s.refit.min(last + 1 - start);
        let t0 = std::time::Instant::now();
        let fit = Fit::new(store, var, start, s)?;
        let p = fit.members.len();
        let ahead = fit.ahead(var, n, s);
        let f = fit.forecasts(&fit.group, &fit.model, &ahead, n, s);
        // Next-day returns of every member on every day of the block.
        let next: Vec<f64> = fit.members.par_iter().flat_map_iter(|&j| (0..n).map(move |i| next_day_return(var, j, start + i))).collect();
        let book = |label: &[usize], f: &[f64], i: usize| -> Vec<(usize, f64)> {
            fit.members.iter().enumerate().filter_map(|(m, &j)| {
                let sign = f[i * k + label[m]].signum();
                (f[i * k + label[m]] != 0.0).then_some((j, sign / p as f64))
            }).collect()
        };
        let pnl = |w: &[(usize, f64)], i: usize| -> f64 {
            w.iter().map(|&(j, x)| x * next[fit.members.binary_search(&j).unwrap() * n + i]).sum()
        };
        let turn = |a: &[(usize, f64)], b: &[(usize, f64)]| -> f64 {
            let mut all: Vec<(usize, f64)> = a.iter().copied().chain(b.iter().map(|&(j, x)| (j, -x))).collect();
            all.sort_by_key(|e| e.0);
            let mut total = 0.0;
            let mut i = 0;
            while i < all.len() {
                let mut v = all[i].1;
                while i + 1 < all.len() && all[i + 1].0 == all[i].0 {
                    i += 1;
                    v += all[i].1;
                }
                total += v.abs();
                i += 1;
            }
            total
        };
        for i in 0..n {
            let w = book(&fit.group, &f, i);
            bt.day.push(var.days[start + i + 1]);
            bt.ret.push(pnl(&w, i));
            bt.turnover.push(turn(&w, &prev_weights));
            bt.exposure.push(w.len() as f64 / p as f64);
            bt.market.push(var.market[start + i + 1]);
            prev_weights = w;
        }
        // The paper's null: the same pipeline with the group labels shuffled (sizes kept).
        // Each refit draws its own shuffle, as the paper draws a random clustering per refit.
        let null: Vec<Vec<f64>> = (0..draws)
            .into_par_iter()
            .map(|q| {
                let mut state = 0xC1u64 ^ (start as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (q as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                let mut label = fit.group.clone();
                for i in (1..label.len()).rev() {
                    let j = (rng(&mut state) % (i as u64 + 1)) as usize;
                    label.swap(i, j);
                }
                let model = fit.model_for(&label, s);
                let f = fit.forecasts(&label, &model, &ahead, n, s);
                (0..n).map(|i| pnl(&book(&label, &f, i), i)).collect()
            })
            .collect();
        for (q, series) in null.into_iter().enumerate() {
            bt.null[q].extend(series);
        }
        let ari_prev = match &prev_fit {
            Some((members, group)) => {
                let (mut a, mut b) = (Vec::new(), Vec::new());
                for (m, &j) in fit.members.iter().enumerate() {
                    if let Ok(i) = members.binary_search(&j) {
                        a.push(group[i]);
                        b.push(fit.group[m]);
                    }
                }
                adjusted_rand(&a, &b)
            }
            None => f64::NAN,
        };
        let mut sizes = vec![0; k];
        for &g in &fit.group {
            sizes[g] += 1;
        }
        let links = (0..k * k).filter(|&ab| fit.model.links[ab]).map(|ab| (ab / k, ab % k)).collect();
        bt.fits.push(FitSummary { day: var.days[start], stocks: p, sizes, links, ari_prev, seconds: t0.elapsed().as_secs_f64() });
        progress(&format!("fit {} ({} stocks, {:.1}s)", c2_data::date_string(var.days[start]), p, t0.elapsed().as_secs_f64()));
        prev_fit = Some((fit.members.clone(), fit.group.clone()));
        start += n;
    }
    Ok(bt)
}

/// Annualized Sharpe ratio of daily returns.
pub fn sharpe(r: &[f64]) -> f64 {
    let n = r.len() as f64;
    let mean = r.iter().sum::<f64>() / n;
    let sd = (r.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)).sqrt();
    if sd > 0.0 { mean / sd * 252f64.sqrt() } else { 0.0 }
}

/// The paper's scaling: each day's return times 10% a year over the standard deviation of the
/// previous 21 days' returns (days with fewer than 21 before them are left out). Returns the
/// scaled series and each day's scale (index-aligned to `r[21..]`).
pub fn vol_target(r: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let target = 0.10 / 252f64.sqrt();
    let mut out = Vec::new();
    let mut scale = Vec::new();
    for d in 21..r.len() {
        let w = &r[d - 21..d];
        let mean = w.iter().sum::<f64>() / 21.0;
        let sd = (w.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / 20.0).sqrt();
        let k = if sd > 0.0 { target / sd } else { 0.0 };
        out.push(k * r[d]);
        scale.push(k);
    }
    (out, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_correlation_matches_the_direct_formula() {
        // dCor of y = x^2 (non-linear, Pearson ~0) against the O(n^2) definition.
        let a: Vec<f64> = (0..40).map(|i| (i as f64 - 19.5) / 10.0).collect();
        let b: Vec<f64> = a.iter().map(|x| x * x + 0.1 * (x * 7.0).sin()).collect();
        let n = a.len();
        let m = n * (n + 1) / 2;
        let (mut ca, mut cb) = (vec![0f32; m], vec![0f32; m]);
        centred_distances(&a, &mut ca);
        centred_distances(&b, &mut cb);
        let dot = |u: &[f32], v: &[f32]| u.iter().zip(v).map(|(x, y)| *x as f64 * *y as f64).sum::<f64>();
        let fast = (dot(&ca, &cb) / (dot(&ca, &ca) * dot(&cb, &cb)).sqrt()).sqrt();
        let centred = |v: &[f64]| -> Vec<f64> {
            let d: Vec<f64> = (0..n * n).map(|st| (v[st / n] - v[st % n]).abs()).collect();
            let row: Vec<f64> = (0..n).map(|s| d[s * n..(s + 1) * n].iter().sum::<f64>() / n as f64).collect();
            let g = row.iter().sum::<f64>() / n as f64;
            (0..n * n).map(|st| d[st] - row[st / n] - row[st % n] + g).collect()
        };
        let (da, db) = (centred(&a), centred(&b));
        let dd = |u: &[f64], v: &[f64]| u.iter().zip(v).map(|(x, y)| x * y).sum::<f64>();
        let direct = (dd(&da, &db) / (dd(&da, &da) * dd(&db, &db)).sqrt()).sqrt();
        assert!((fast - direct).abs() < 1e-5, "{fast} vs {direct}");
        assert!(direct > 0.3);
    }

    #[test]
    fn planted_leaders_and_followers_are_found() {
        // Four groups of 15: group 0 leads group 1 by one day, group 2 leads group 3 by two days,
        // each pair on its own factor. With k = 4 the embedding is the two top eigenvector pairs
        // (the paper's l = k; an odd k would take half of a pair).
        let (w, p, size) = (400, 60, 15);
        let mut state = 7u64;
        let mut noise = || {
            let (u, v) = (uniform(&mut state).max(1e-12), uniform(&mut state));
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        };
        let f1: Vec<f64> = (0..w).map(|_| noise()).collect();
        let f2: Vec<f64> = (0..w).map(|_| noise()).collect();
        let mut y = vec![0.0; w * p];
        for j in 0..p {
            for t in 0..w {
                y[j * w + t] = match j / size {
                    0 => f1[t],
                    1 if t >= 1 => 0.8 * f1[t - 1],
                    2 => f2[t],
                    3 if t >= 2 => 0.8 * f2[t - 2],
                    _ => 0.0,
                } + noise();
            }
        }
        let s = Settings { stocks: p, groups: 4, max_lag: 3, window: w, refit: 1, alpha: 0.4, link_quantile: 0.5, metric: Metric::Pearson, returns: Returns::Raw, nyse_only: false };
        for metric in [Metric::Pearson, Metric::Distance] {
            let sm = lead_lag(&y, w, p, &Settings { metric, ..s });
            // Leading groups' mean row sums are positive, following groups' negative (single
            // stocks' row sums are noisy: ccf-auc gives unrelated pairs a size of 0.5 or more).
            for g in 0..4 {
                let mean = (g * size..(g + 1) * size).map(|i| sm[i * p..(i + 1) * p].iter().sum::<f64>()).sum::<f64>() / size as f64;
                assert!(if g % 2 == 0 { mean > 0.0 } else { mean < 0.0 }, "{metric:?}: group {g} mean row sum {mean}");
            }
            let x = embed(&sm, p, 4).unwrap();
            let label = kmeans(&x, p, 4, 4, 1, 10);
            let truth: Vec<usize> = (0..p).map(|j| j / size).collect();
            assert!((adjusted_rand(&label, &truth) - 1.0).abs() < 1e-12, "{metric:?}: {label:?}");
        }
    }

    #[test]
    fn forecasts_use_only_returns_through_the_decision_day() {
        // Two groups of 10: group 1 repeats group 0's return a day later.
        let (p, k, window) = (20, 2, 100);
        let rows = window + WARM;
        let mut state = 11u64;
        let lead: Vec<f64> = (0..rows + 10).map(|_| uniform(&mut state) - 0.5).collect();
        let series = |m: usize, t: usize| if m < 10 { lead[t] } else if t > 0 { lead[t - 1] } else { 0.0 };
        let y: Vec<f64> = (0..p).flat_map(|m| (0..rows).map(move |t| series(m, t))).collect();
        let mut sm = vec![0.0; p * p];
        for i in 0..10 {
            for j in 10..20 {
                sm[i * p + j] = 1.0;
                sm[j * p + i] = -1.0;
            }
        }
        let group: Vec<usize> = (0..p).map(|m| m / 10).collect();
        let s = Settings { stocks: p, groups: k, max_lag: 1, window, refit: 8, alpha: 0.4, link_quantile: 0.0, metric: Metric::Pearson, returns: Returns::Raw, nyse_only: false };
        let mut fit = Fit { row: rows - 1, members: (0..p).collect(), group: group.clone(), lead: vec![0.0; p], model: Model { flow: vec![], links: vec![], theta: vec![] }, sm, y, rows };
        fit.model = fit.model_for(&group, &s);
        // Group 0 -> 1 is the only link, and the regression finds the one-day echo.
        assert_eq!(fit.model.links, vec![false, true, false, false]);
        assert!(fit.model.theta[1] > 0.3, "theta {}", fit.model.theta[1]);
        let days = 8;
        let ahead: Vec<f64> = (0..p).flat_map(|m| (rows..rows + days - 1).map(move |t| series(m, t))).collect();
        let f = fit.forecasts(&group, &fit.model, &ahead, days, &s);
        // The forecast made at day i must not change when a later return (row + i + 1 on) does.
        for i in 0..days - 1 {
            let mut changed = ahead.clone();
            for m in 0..p {
                changed[m * (days - 1) + i] += 5.0;
            }
            let g = fit.forecasts(&group, &fit.model, &changed, days, &s);
            assert_eq!(&f[..(i + 1) * k], &g[..(i + 1) * k], "day {i}");
            assert_ne!(f[(i + 1) * k + 1], g[(i + 1) * k + 1]);
        }
    }

    #[test]
    fn quantile_interpolates() {
        assert_eq!(quantile(&[0.0, 1.0, 2.0, 3.0], 0.5), 1.5);
        assert_eq!(quantile(&[0.0, 10.0], 0.9), 9.0);
    }
}
