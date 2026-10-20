//! Same-day co-movement: which stocks and funds move together on the same day. Unlike lead-lag,
//! the census found this real and lasting, which makes it useful for risk. Four tools, all on
//! daily returns of the last year (and the year before it, to show whether a relationship held):
//! - `portfolio`: how much a portfolio's holdings move together, how many independent bets they
//!   amount to, each holding's share of the risk, and near-duplicate pairs;
//! - `diversifiers`: the stocks and funds that would add the most independent bets to it;
//! - `hedges`: what tracks one stock most closely and what moves most opposite to it;
//! - `groups`: groups of stocks that move together once the market is taken out, labelled by
//!   the fund they track, with how stable each group has been.
//!
//! Unlike the census, funds (ETFs) are included: bond, gold and sector funds are natural
//! diversifiers and hedges. Returns are simple daily returns of split- and dividend-adjusted
//! closes; a day without a close is missing (never filled with 0 for a holding).

use std::collections::HashMap;
use std::ops::Range;

use anyhow::{Result, anyhow, bail};
use c2_data::{Calendar, Kind, Panel, Store};
use rayon::prelude::*;

/// Trading days in a year.
pub const YEAR: usize = 252;
/// The liquidity screen for candidates (holdings and the hedged stock are exempt).
const MIN_PRICE: f32 = 5.0;
const MIN_DOLLAR_VOLUME: f32 = 5e6;
/// Fewest days with data in the year before for a correlation "a year earlier".
const MIN_BEFORE: usize = YEAR / 2;

/// Plain words for a correlation, the same everywhere the tools show one. They go by the value
/// rounded to the 2 decimals shown, so "+0.90" always reads "almost always together".
pub fn words(r: f64) -> &'static str {
    let r = (r * 100.0).round() / 100.0;
    if !r.is_finite() {
        "unknown"
    } else if r >= 0.9 {
        "almost always together"
    } else if r >= 0.6 {
        "usually together"
    } else if r >= 0.3 {
        "often together"
    } else if r > -0.3 {
        "mostly independent"
    } else if r > -0.6 {
        "often opposite"
    } else if r > -0.9 {
        "usually opposite"
    } else {
        "almost always opposite"
    }
}

/// Daily returns of every stock and fund still trading, over `years` years up to a day.
pub struct Universe {
    /// Day of each return row, oldest first; the last is the as-of day.
    pub days: Vec<i32>,
    pub t: usize,
    pub n: usize,
    pub tickers: Vec<String>,
    pub etf: Vec<bool>,
    /// Passes the candidate screen over the last year: a close on 90% of days and on the as-of
    /// day, last close >= $5, median dollar volume >= $5M.
    pub liquid: Vec<bool>,
    /// Median daily dollar volume in each year back (index 0 = the last year), NaN without data.
    pub dollar_volume: Vec<Vec<f32>>,
    /// Simple daily returns, column-major `t x n`; NaN without a close that day and the day
    /// before.
    pub ret: Vec<f32>,
    index: HashMap<String, usize>,
}

impl Universe {
    /// Every stock and fund with a close in the last year before `as_of` (default: the last day
    /// of data), `years` years of returns.
    pub fn build(store: &Store, as_of: Option<i32>, years: usize) -> Result<Universe> {
        let cal = Calendar::trading_days(store)?;
        let end = match as_of {
            Some(d) => cal.days.partition_point(|&x| x <= d),
            None => cal.days.len(),
        };
        anyhow::ensure!(end > YEAR + 1, "less than a year of prices before {}", as_of.map(c2_data::date_string).unwrap_or_default());
        let rows = end - (years * YEAR + 1).min(end)..end;
        let (first_needed, last_day) = (cal.days[end - YEAR], cal.days[end - 1]);
        let ids: Vec<usize> = (0..store.symbols.len())
            .filter(|&i| {
                let s = &store.symbols[i];
                matches!(s.kind, Kind::Stock | Kind::Etf) && s.last >= first_needed && s.first <= last_day
            })
            .collect();
        let mut panel = Panel::build(store, &cal, rows, &ids);
        panel.repair_bad_prints();
        let (pt, n) = (panel.t, panel.n);
        let t = pt - 1;
        let years_have = t / YEAR;
        let cols: Vec<(Vec<f32>, Vec<f32>, bool)> = (0..n)
            .into_par_iter()
            .map(|j| {
                let c = &panel.close[j * pt..(j + 1) * pt];
                let v = &panel.volume[j * pt..(j + 1) * pt];
                let ret: Vec<f32> = (1..pt).map(|d| if c[d].is_finite() && c[d - 1].is_finite() && c[d - 1] > 0.0 { c[d] / c[d - 1] - 1.0 } else { f32::NAN }).collect();
                let dv: Vec<f32> = (0..years_have)
                    .map(|y| {
                        let mut x: Vec<f32> = (pt - YEAR * (y + 1)..pt - YEAR * y).filter(|&d| c[d].is_finite()).map(|d| c[d] * v[d]).collect();
                        if x.is_empty() {
                            return f32::NAN;
                        }
                        let mid = x.len() / 2;
                        *x.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
                    })
                    .collect();
                let covered = ret[t - YEAR..].iter().filter(|x| x.is_finite()).count();
                let liquid = covered * 10 >= YEAR * 9 && c[pt - 1].is_finite() && c[pt - 1] >= MIN_PRICE && dv.first().is_some_and(|&d| d >= MIN_DOLLAR_VOLUME);
                (ret, dv, liquid)
            })
            .collect();
        let tickers: Vec<String> = ids.iter().map(|&i| store.symbols[i].ticker.clone()).collect();
        let etf: Vec<bool> = ids.iter().map(|&i| store.symbols[i].kind == Kind::Etf).collect();
        let index = tickers.iter().enumerate().map(|(j, s)| (s.clone(), j)).collect();
        let mut ret = Vec::with_capacity(t * n);
        let mut dollar_volume = vec![vec![f32::NAN; n]; years_have];
        let mut liquid = Vec::with_capacity(n);
        for (j, (r, dv, l)) in cols.into_iter().enumerate() {
            ret.extend(r);
            for (y, d) in dv.into_iter().enumerate() {
                dollar_volume[y][j] = d;
            }
            liquid.push(l);
        }
        Ok(Universe { days: panel.days[1..].to_vec(), t, n, tickers, etf, liquid, dollar_volume, ret, index })
    }

    pub fn col(&self, j: usize) -> &[f32] {
        &self.ret[j * self.t..(j + 1) * self.t]
    }

    pub fn find(&self, ticker: &str) -> Option<usize> {
        self.index.get(&ticker.trim().to_uppercase()).copied()
    }

    pub fn as_of(&self) -> i32 {
        *self.days.last().unwrap_or(&0)
    }

    /// Rows of the year `back` years before the last one (0 = the last year), if the data
    /// reaches that far.
    pub fn year(&self, back: usize) -> Option<Range<usize>> {
        (self.t >= YEAR * (back + 1)).then(|| self.t - YEAR * (back + 1)..self.t - YEAR * back)
    }
}

/// Pairwise statistics of two daily series over the rows where both have values.
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

/// Annualized volatility from a daily variance.
fn annual(var: f64) -> f64 {
    (var * YEAR as f64).sqrt()
}

/// Holdings as typed: tickers, each optionally followed by an amount (dollars or percent; only
/// proportions matter), separated by spaces, commas, semicolons or new lines; `AAPL:30` and
/// `AAPL=30` work too. Every holding has an amount or none does (equal weights).
pub fn parse_holdings(text: &str) -> Result<Vec<(String, f64)>> {
    let mut out: Vec<(String, Option<f64>)> = Vec::new();
    for token in text.split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | ':' | '=')).filter(|s| !s.is_empty()) {
        let number = token.trim_start_matches('$').trim_end_matches('%');
        if let Ok(v) = number.parse::<f64>() {
            match out.last_mut() {
                Some((t, a @ None)) => {
                    anyhow::ensure!(v.is_finite() && v > 0.0, "the amount for {t} must be above 0 (got {token})");
                    *a = Some(v);
                }
                Some((t, Some(_))) => bail!("{token} follows {t}, which already has an amount"),
                None => bail!("{token} comes before any ticker"),
            }
        } else {
            let t = token.to_uppercase();
            anyhow::ensure!(t.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-'), "{token} is neither a ticker nor an amount");
            anyhow::ensure!(!out.iter().any(|(x, _)| *x == t), "{t} is listed twice");
            out.push((t, None));
        }
    }
    anyhow::ensure!(!out.is_empty(), "no holdings given");
    let missing: Vec<&str> = out.iter().filter(|(_, a)| a.is_none()).map(|(t, _)| t.as_str()).collect();
    if !missing.is_empty() && missing.len() < out.len() {
        bail!("give an amount for every holding or for none (then they are equal); no amount for {}", missing.join(", "));
    }
    Ok(out.into_iter().map(|(t, a)| (t, a.unwrap_or(1.0))).collect())
}

pub struct Holding {
    pub ticker: String,
    pub etf: bool,
    pub weight: f64,
    /// Annualized volatility.
    pub vol: f64,
    /// Beta to SPY (NaN without SPY).
    pub beta: f64,
    /// Share of the portfolio's variance (the shares sum to 1).
    pub risk_share: f64,
    /// Correlation with the rest of the portfolio (the other holdings at their weights).
    pub corr_rest: f64,
}

pub struct PortfolioReport {
    pub holdings: Vec<Holding>,
    /// First and last day of the rows used, and how many.
    pub days: (i32, i32, usize),
    /// Why fewer than a year of days were used, if they were.
    pub shortened: Option<String>,
    /// The year before: first and last day.
    pub before_days: Option<(i32, i32)>,
    /// Annualized volatility of the portfolio, and the weighted mean of the holdings'.
    pub vol: f64,
    pub avg_vol: f64,
    /// (sum of w_i vol_i / vol)^2: how many independent bets the holdings behave like.
    pub bets: f64,
    pub beta: f64,
    pub spy_corr: f64,
    /// Correlations, k x k row-major: over the rows used, and in the year before (None without
    /// half a year of data for the pair then).
    pub corr: Vec<f64>,
    pub corr_before: Vec<Option<f64>>,
    // What `diversifiers` needs: the universe columns, weights, rows used and the portfolio's
    // daily returns on them, and the same for the year before.
    cols: Vec<usize>,
    rows: Vec<usize>,
    port: Vec<f64>,
    before: Option<(Vec<usize>, Vec<f64>)>,
}

impl PortfolioReport {
    /// Pairs of holdings at or above `min` correlation, highest first: (a, b, corr, before).
    pub fn duplicates(&self, min: f64) -> Vec<(usize, usize, f64, Option<f64>)> {
        let k = self.holdings.len();
        let mut v: Vec<_> = (0..k).flat_map(|a| (a + 1..k).map(move |b| (a, b))).filter(|&(a, b)| self.corr[a * k + b] >= min).map(|(a, b)| (a, b, self.corr[a * k + b], self.corr_before[a * k + b])).collect();
        v.sort_by(|x, y| y.2.total_cmp(&x.2));
        v
    }
}

/// The portfolio check over the last year (shortened to the days every holding has a return).
pub fn portfolio(u: &Universe, holdings: &[(String, f64)]) -> Result<PortfolioReport> {
    let unknown: Vec<&str> = holdings.iter().filter(|(t, _)| u.find(t).is_none()).map(|(t, _)| t.as_str()).collect();
    if !unknown.is_empty() {
        bail!("no prices in the last year before {} for {}", c2_data::date_string(u.as_of()), unknown.join(", "));
    }
    let cols: Vec<usize> = holdings.iter().map(|(t, _)| u.find(t).unwrap()).collect();
    let total: f64 = holdings.iter().map(|(_, a)| a).sum();
    let w: Vec<f64> = holdings.iter().map(|(_, a)| a / total).collect();
    let k = cols.len();
    let year = u.year(0).ok_or_else(|| anyhow!("less than a year of data"))?;
    let all_have = |d: usize| cols.iter().all(|&j| u.col(j)[d].is_finite());
    let rows: Vec<usize> = year.clone().filter(|&d| all_have(d)).collect();
    anyhow::ensure!(rows.len() >= 20, "the holdings have returns on the same days only {} times in the last year", rows.len());
    let shortened = (rows.len() * 10 < YEAR * 9).then(|| {
        let late: Vec<String> = cols
            .iter()
            .filter_map(|&j| {
                let first = year.clone().find(|&d| u.col(j)[d].is_finite())?;
                (first > year.start + 5).then(|| format!("{} has returns only from {}", u.tickers[j], c2_data::date_string(u.days[first])))
            })
            .collect();
        format!(
            "{} of the last year's {YEAR} days have a return for every holding{}",
            rows.len(),
            if late.is_empty() { String::new() } else { format!(" ({})", late.join("; ")) }
        )
    });
    let r = |j: usize, d: usize| u.col(cols[j])[d] as f64;
    let port: Vec<f64> = rows.iter().map(|&d| (0..k).map(|j| w[j] * r(j, d)).sum()).collect();
    // Covariances over the rows used.
    let m = rows.len() as f64;
    let mean: Vec<f64> = (0..k).map(|j| rows.iter().map(|&d| r(j, d)).sum::<f64>() / m).collect();
    let cov: Vec<f64> = (0..k * k)
        .map(|ab| {
            let (a, b) = (ab / k, ab % k);
            rows.iter().map(|&d| (r(a, d) - mean[a]) * (r(b, d) - mean[b])).sum::<f64>() / (m - 1.0)
        })
        .collect();
    let sw: Vec<f64> = (0..k).map(|a| (0..k).map(|b| cov[a * k + b] * w[b]).sum()).collect();
    let var_p: f64 = (0..k).map(|a| w[a] * sw[a]).sum();
    let corr: Vec<f64> = (0..k * k).map(|ab| cov[ab] / (cov[(ab / k) * k + ab / k] * cov[(ab % k) * k + ab % k]).sqrt()).collect();
    let spy = u.find("SPY");
    let spy_stats = |x: &dyn Fn(usize) -> f64| -> (f64, f64) {
        match spy {
            Some(s) => {
                let p = pair(rows.iter().map(|&d| x(d)), rows.iter().map(|&d| u.col(s)[d] as f64));
                (p.cov / p.var_b, p.corr)
            }
            None => (f64::NAN, f64::NAN),
        }
    };
    let holdings_out: Vec<Holding> = (0..k)
        .map(|a| {
            // The rest of the portfolio: the other holdings at their weights, rescaled.
            let rest_w: f64 = 1.0 - w[a];
            let corr_rest = if rest_w > 0.0 {
                let rest = |d: usize| (0..k).filter(|&b| b != a).map(|b| w[b] * r(b, d)).sum::<f64>() / rest_w;
                pair(rows.iter().map(|&d| r(a, d)), rows.iter().map(|&d| rest(d))).corr
            } else {
                f64::NAN
            };
            Holding {
                ticker: u.tickers[cols[a]].clone(),
                etf: u.etf[cols[a]],
                weight: w[a],
                vol: annual(cov[a * k + a]),
                beta: spy_stats(&|d| r(a, d)).0,
                risk_share: w[a] * sw[a] / var_p,
                corr_rest,
            }
        })
        .collect();
    let avg_vol: f64 = holdings_out.iter().map(|h| h.weight * h.vol).sum();
    let vol = annual(var_p);
    let port_at: HashMap<usize, f64> = rows.iter().copied().zip(port.iter().copied()).collect();
    let (beta, spy_corr) = spy_stats(&|d| port_at[&d]);
    // The year before: pairwise for the table; all holdings together for the portfolio.
    let before_range = u.year(1);
    let corr_before: Vec<Option<f64>> = (0..k * k)
        .map(|ab| {
            let range = before_range.clone()?;
            let p = pair(range.clone().map(|d| r(ab / k, d)), range.map(|d| r(ab % k, d)));
            (p.n >= MIN_BEFORE).then_some(p.corr)
        })
        .collect();
    let before = before_range.as_ref().and_then(|range| {
        let rows: Vec<usize> = range.clone().filter(|&d| all_have(d)).collect();
        (rows.len() >= MIN_BEFORE).then(|| {
            let p = rows.iter().map(|&d| (0..k).map(|j| w[j] * r(j, d)).sum()).collect();
            (rows, p)
        })
    });
    Ok(PortfolioReport {
        holdings: holdings_out,
        days: (u.days[rows[0]], u.days[*rows.last().unwrap()], rows.len()),
        shortened,
        before_days: before_range.map(|r| (u.days[r.start], u.days[r.end - 1])),
        vol,
        avg_vol,
        bets: (avg_vol / vol).powi(2),
        beta,
        spy_corr,
        corr,
        corr_before,
        cols,
        rows,
        port,
        before,
    })
}

pub struct Candidate {
    pub ticker: String,
    pub etf: bool,
    pub vol: f64,
    /// Correlation with the portfolio over its rows, and in the year before (None: no data then).
    pub corr: f64,
    pub corr_before: Option<f64>,
    /// The portfolio's volatility with this added at the chosen share of the money.
    pub vol_after: f64,
    /// Candidates that move almost exactly like this one (correlation 0.95 or more over the
    /// portfolio's days), such as other funds on the same index or its leveraged versions; this
    /// one is the least volatile of them.
    pub alike: Vec<String>,
}

pub struct Diversifiers {
    /// Share of the money the candidate is added at (the holdings scale down to make room).
    pub add: f64,
    pub funds: Vec<Candidate>,
    pub stocks: Vec<Candidate>,
    /// Candidates that move opposite: correlation -0.3 or lower with the portfolio, or -0.5 or
    /// lower with SPY (inverse funds). Hedges rather than diversifiers.
    pub hedges: Vec<Candidate>,
    /// Liquid candidates looked at, and those set aside: cash-like (under 3% a year volatility)
    /// and very volatile (above `max_vol` a year, mostly leveraged funds).
    pub scanned: usize,
    pub cash_like: usize,
    pub too_volatile: usize,
    pub max_vol: f64,
}

/// The candidates that move least with the portfolio. Correlation is the ranking because it
/// doesn't depend on size or leverage: a 3x fund has its 1x version's correlations. Near-identical
/// candidates are shown once (`Candidate::alike`).
pub fn diversifiers(u: &Universe, p: &PortfolioReport, add: f64, max_vol: f64, show: usize) -> Diversifiers {
    let a = add.clamp(0.0, 0.95);
    let vol_p = p.vol;
    let spy = u.find("SPY");
    let year = u.year(0).unwrap_or(0..u.t);
    let all: Vec<(usize, Candidate, f64)> = (0..u.n)
        .into_par_iter()
        .filter(|&j| u.liquid[j] && !p.cols.contains(&j))
        .filter_map(|j| {
            let c = u.col(j);
            let s = pair(p.rows.iter().map(|&d| c[d] as f64), p.port.iter().copied());
            if s.n * 10 < p.rows.len() * 9 || !s.corr.is_finite() {
                return None;
            }
            let vol = annual(s.var_a);
            let vol_after = ((1.0 - a).powi(2) * vol_p * vol_p + a * a * vol * vol + 2.0 * a * (1.0 - a) * s.corr * vol * vol_p).max(0.0).sqrt();
            let corr_before = p.before.as_ref().and_then(|(rows, port)| {
                let s = pair(rows.iter().map(|&d| c[d] as f64), port.iter().copied());
                (s.n >= MIN_BEFORE).then_some(s.corr)
            });
            let spy_corr = spy.map(|s| pair(year.clone().map(|d| c[d] as f64), year.clone().map(|d| u.col(s)[d] as f64)).corr).unwrap_or(0.0);
            Some((j, Candidate { ticker: u.tickers[j].clone(), etf: u.etf[j], vol, corr: s.corr, corr_before, vol_after, alike: Vec::new() }, spy_corr))
        })
        .collect();
    let scanned = all.len();
    let cash_like = all.iter().filter(|(_, c, _)| c.vol < 0.03).count();
    let too_volatile = all.iter().filter(|(_, c, _)| c.vol > max_vol).count();
    // The split follows the words: by the correlation rounded as shown.
    let round = |r: f64| (r * 100.0).round() / 100.0;
    let (mut hedges, mut rest): (Vec<_>, Vec<_>) =
        all.into_iter().filter(|(_, c, _)| c.vol >= 0.03 && c.vol <= max_vol).partition(|(_, c, spy_corr)| round(c.corr) <= -0.3 || round(*spy_corr) <= -0.5);
    let by_corr = |x: &(usize, Candidate, f64), y: &(usize, Candidate, f64)| x.1.corr.total_cmp(&y.1.corr).then(x.1.ticker.cmp(&y.1.ticker));
    hedges.sort_by(by_corr);
    rest.sort_by(by_corr);
    // Near-identical candidates (0.95 or more over the portfolio's days) become one row, shown
    // under the least volatile of them.
    let similar = |a: usize, b: usize| pair(p.rows.iter().map(|&d| u.col(a)[d] as f64), p.rows.iter().map(|&d| u.col(b)[d] as f64)).corr >= 0.95;
    let pick = |list: Vec<(usize, Candidate, f64)>| -> Vec<Candidate> {
        let mut out: Vec<(usize, Candidate)> = Vec::new();
        for (j, c, _) in list {
            match out.iter().position(|(k, _)| similar(*k, j)) {
                Some(i) => {
                    let (k, rep) = &mut out[i];
                    if c.vol < rep.vol {
                        let mut alike = std::mem::take(&mut rep.alike);
                        alike.push(rep.ticker.clone());
                        *k = j;
                        *rep = Candidate { alike, ..c };
                    } else {
                        rep.alike.push(c.ticker);
                    }
                }
                None if out.len() < show => out.push((j, c)),
                None => {}
            }
        }
        // A less volatile look-alike may have taken a row's place: keep the rows in order.
        let mut out: Vec<Candidate> = out.into_iter().map(|(_, c)| c).collect();
        out.sort_by(|x, y| x.corr.total_cmp(&y.corr).then(x.ticker.cmp(&y.ticker)));
        out
    };
    let (funds, stocks): (Vec<_>, Vec<_>) = rest.into_iter().partition(|x| x.1.etf);
    Diversifiers { add: a, funds: pick(funds), stocks: pick(stocks), hedges: pick(hedges), scanned, cash_like, too_volatile, max_vol }
}

#[derive(Clone)]
pub struct HedgeRow {
    pub ticker: String,
    pub etf: bool,
    pub vol: f64,
    pub corr: f64,
    pub corr_before: Option<f64>,
    /// Dollars of this per dollar of the stock that best offsets it (short it when positive).
    pub ratio: f64,
    /// Share of the stock's volatility left after that hedge: sqrt(1 - corr^2).
    pub risk_left: f64,
}

pub struct HedgeReport {
    pub ticker: String,
    pub etf: bool,
    pub vol: f64,
    pub days: (i32, i32, usize),
    pub before_days: Option<(i32, i32)>,
    pub closest: Vec<HedgeRow>,
    pub opposite: Vec<HedgeRow>,
    pub scanned: usize,
}

/// What tracks `ticker` most closely and what moves most opposite to it over the last year.
pub fn hedges(u: &Universe, ticker: &str, show: usize) -> Result<HedgeReport> {
    let target = u.find(ticker).ok_or_else(|| anyhow!("no prices in the last year before {} for {}", c2_data::date_string(u.as_of()), ticker.trim().to_uppercase()))?;
    let year = u.year(0).ok_or_else(|| anyhow!("less than a year of data"))?;
    let x = u.col(target);
    let rows: Vec<usize> = year.clone().filter(|&d| x[d].is_finite()).collect();
    anyhow::ensure!(rows.len() >= 60, "{} has returns on only {} days of the last year", u.tickers[target], rows.len());
    let before_rows: Option<Vec<usize>> = u.year(1).map(|r| r.filter(|&d| x[d].is_finite()).collect());
    let var_x = pair(rows.iter().map(|&d| x[d] as f64), rows.iter().map(|&d| x[d] as f64)).var_a;
    let mut all: Vec<HedgeRow> = (0..u.n)
        .into_par_iter()
        .filter(|&j| u.liquid[j] && j != target)
        .filter_map(|j| {
            let c = u.col(j);
            let s = pair(rows.iter().map(|&d| x[d] as f64), rows.iter().map(|&d| c[d] as f64));
            if s.n * 10 < rows.len() * 9 || !s.corr.is_finite() {
                return None;
            }
            let corr_before = before_rows.as_ref().and_then(|rows| {
                let s = pair(rows.iter().map(|&d| x[d] as f64), rows.iter().map(|&d| c[d] as f64));
                (s.n >= MIN_BEFORE).then_some(s.corr)
            });
            Some(HedgeRow { ticker: u.tickers[j].clone(), etf: u.etf[j], vol: annual(s.var_b), corr: s.corr, corr_before, ratio: s.cov / s.var_b, risk_left: (1.0 - s.corr * s.corr).max(0.0).sqrt() })
        })
        .collect();
    let scanned = all.len();
    all.sort_by(|a, b| b.corr.total_cmp(&a.corr).then(a.ticker.cmp(&b.ticker)));
    // Both ends of the same ranking (with fewer than 2 x `show` candidates they overlap).
    let opposite: Vec<HedgeRow> = all.iter().rev().take(show).cloned().collect();
    all.truncate(show);
    let closest = all;
    Ok(HedgeReport {
        ticker: u.tickers[target].clone(),
        etf: u.etf[target],
        vol: annual(var_x),
        days: (u.days[rows[0]], u.days[*rows.last().unwrap()], rows.len()),
        before_days: u.year(1).map(|r| (u.days[r.start], u.days[r.end - 1])),
        closest,
        opposite,
        scanned,
    })
}

pub struct Group {
    /// Indices into `GroupsMap::stocks`, by dollar volume, largest first.
    pub members: Vec<usize>,
    /// The fund whose daily returns follow the group's average most closely, and the correlation.
    pub tracks: Option<(String, f64)>,
    /// Mean correlation between members with the market taken out, and including it.
    pub corr_relative: f64,
    pub corr_raw: f64,
    /// The group's equal-weight return over the year.
    pub year_return: f64,
    /// Per earlier year (1, 2, ... back): of the members' pairs present then (at least 30), the
    /// share that were grouped together that year too, and what chance alone would give.
    pub kept: Vec<Option<(f64, f64)>>,
}

pub struct GroupsMap {
    pub days: (i32, i32, usize),
    /// The stocks grouped: universe columns and tickers, by group then dollar volume.
    pub stocks: Vec<usize>,
    pub tickers: Vec<String>,
    pub groups: Vec<Group>,
    /// Group of each stock in `stocks` order (groups in `groups` order).
    pub group_of: Vec<usize>,
    /// Market-relative correlations in `stocks` order, row-major, for the heatmap.
    pub corr: Vec<f32>,
    /// The earlier years compared: first and last day.
    pub earlier: Vec<(i32, i32)>,
}

/// The top `stocks` stocks by median dollar volume in the year `back` years before the last, with
/// a return on 95% of its days (not only the candidate screen: earlier years need their own).
fn group_universe(u: &Universe, back: usize, stocks: usize) -> Vec<usize> {
    let Some(range) = u.year(back) else { return Vec::new() };
    let mut v: Vec<(usize, f32)> = (0..u.n)
        .filter(|&j| !u.etf[j])
        .filter(|&j| u.dollar_volume[back][j] >= MIN_DOLLAR_VOLUME)
        .filter(|&j| u.col(j)[range.clone()].iter().filter(|x| x.is_finite()).count() * 20 >= YEAR * 19)
        .map(|j| (j, u.dollar_volume[back][j]))
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(stocks);
    v.into_iter().map(|(j, _)| j).collect()
}

/// Column-major `rows x n` panel of `cols` over `range`, standardized to mean 0 and unit sum of
/// squares (missing = 0 after centring), so Z^T Z is the correlation matrix. With `relative`,
/// each stock's beta times the equal-weight mean of `cols` is taken out first.
fn standardized(u: &Universe, cols: &[usize], range: Range<usize>, relative: bool) -> Vec<f32> {
    let rows = range.len();
    let market: Vec<f64> = range
        .clone()
        .map(|d| {
            let (s, c) = cols.iter().map(|&j| u.col(j)[d]).filter(|x| x.is_finite()).fold((0.0, 0usize), |(s, c), x| (s + x as f64, c + 1));
            if c > 0 { s / c as f64 } else { 0.0 }
        })
        .collect();
    let mut z = vec![0f32; rows * cols.len()];
    z.par_chunks_mut(rows).zip(cols.par_iter()).for_each(|(out, &j)| {
        let x: Vec<f64> = range.clone().map(|d| u.col(j)[d] as f64).collect();
        let resid: Vec<f64> = if relative {
            let p = pair(x.iter().copied(), market.iter().copied());
            let beta = if p.var_b > 0.0 { p.cov / p.var_b } else { 0.0 };
            x.iter().zip(&market).map(|(v, m)| v - beta * m).collect()
        } else {
            x
        };
        let fin: Vec<f64> = resid.iter().copied().filter(|v| v.is_finite()).collect();
        let mean = fin.iter().sum::<f64>() / fin.len().max(1) as f64;
        let norm = fin.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>().sqrt();
        for (o, v) in out.iter_mut().zip(&resid) {
            *o = if v.is_finite() && norm > 0.0 { ((v - mean) / norm) as f32 } else { 0.0 };
        }
    });
    z
}

/// Spectral clustering (Ng, Jordan & Weiss) of the positive part of a correlation matrix into
/// `k` groups: top k eigenvectors of D^-1/2 A D^-1/2, rows scaled to unit length, k-means.
fn spectral(corr: &[f32], n: usize, k: usize, seed: u64) -> Result<Vec<usize>> {
    let a = |i: usize, j: usize| if i == j { 0.0 } else { (corr[i * n + j] as f64).max(0.0) };
    let deg: Vec<f64> = (0..n).map(|i| (0..n).map(|j| a(i, j)).sum::<f64>()).collect();
    let inv: Vec<f64> = deg.iter().map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect();
    let evd = faer::Mat::<f64>::from_fn(n, n, |i, j| a(i, j) * inv[i] * inv[j])
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| anyhow!("eigendecomposition failed: {e:?}"))?;
    let vecs = evd.U();
    let mut x: Vec<f64> = (0..n).flat_map(|i| (0..k).map(move |c| vecs[(i, n - 1 - c)])).collect();
    for row in x.chunks_mut(k) {
        let len = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if len > 0.0 {
            row.iter_mut().for_each(|v| *v /= len);
        }
    }
    Ok(crate::cluster::kmeans(&x, n, k, k, seed, 10))
}

fn corr_matrix(z: &[f32], rows: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0f32; n * n];
    c2_engine::cpu::xprod_f32(z, z, rows, n, rows, &mut c);
    c
}

/// Groups of the top `stocks` stocks by dollar volume that move together once the market is
/// taken out, from the last year; each group's stability is checked against the same grouping
/// made independently on each of up to `years_back` earlier years.
pub fn groups(u: &Universe, k: usize, stocks: usize, years_back: usize, progress: &(dyn Fn(&str) + Sync)) -> Result<GroupsMap> {
    let year = u.year(0).ok_or_else(|| anyhow!("less than a year of data"))?;
    let cols = group_universe(u, 0, stocks);
    let n = cols.len();
    anyhow::ensure!(n >= 4 * k, "only {n} stocks qualify for {k} groups");
    progress(&format!("grouping {n} stocks"));
    let z = standardized(u, &cols, year.clone(), true);
    let corr = corr_matrix(&z, YEAR, n);
    let label = spectral(&corr, n, k, 0x6E0)?;
    let raw = corr_matrix(&standardized(u, &cols, year.clone(), false), YEAR, n);
    // Groups by size, members by dollar volume.
    let mut size = vec![0usize; k];
    for &g in &label {
        size[g] += 1;
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| size[b].cmp(&size[a]).then(a.cmp(&b)));
    let mut rank = vec![0; k];
    for (r, &g) in order.iter().enumerate() {
        rank[g] = r;
    }
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &g) in label.iter().enumerate() {
        members[rank[g]].push(i);
    }
    for m in &mut members {
        m.sort_by(|&a, &b| u.dollar_volume[0][cols[b]].total_cmp(&u.dollar_volume[0][cols[a]]));
    }
    // Earlier years, each grouped on its own.
    let mut earlier_labels: Vec<HashMap<usize, usize>> = Vec::new();
    let mut earlier = Vec::new();
    for back in 1..=years_back {
        let Some(range) = u.year(back) else { break };
        let c = group_universe(u, back, stocks);
        if c.len() < 4 * k {
            break;
        }
        progress(&format!("grouping {} stocks of {}", c.len(), c2_data::civil_from_days(u.days[range.end - 1]).0));
        let lab = spectral(&corr_matrix(&standardized(u, &c, range.clone(), true), YEAR, c.len()), c.len(), k, 0x6E0 + back as u64)?;
        earlier.push((u.days[range.start], u.days[range.end - 1]));
        // Universe column -> its group that year.
        earlier_labels.push(c.iter().copied().zip(lab).collect());
    }
    let chance: Vec<f64> = earlier_labels
        .iter()
        .map(|m| {
            let mut s = HashMap::new();
            for &g in m.values() {
                *s.entry(g).or_insert(0usize) += 1;
            }
            let total = m.len() as f64;
            s.values().map(|&c| (c as f64 / total).powi(2)).sum()
        })
        .collect();
    let funds: Vec<usize> = (0..u.n).filter(|&j| u.etf[j] && u.liquid[j]).collect();
    let groups: Vec<Group> = members
        .par_iter()
        .map(|m| {
            let pairs: Vec<(usize, usize)> = (0..m.len()).flat_map(|a| (a + 1..m.len()).map(move |b| (m[a], m[b]))).collect();
            let mean_of = |c: &[f32]| pairs.iter().map(|&(a, b)| c[a * n + b] as f64).sum::<f64>() / pairs.len().max(1) as f64;
            let avg: Vec<f64> = year
                .clone()
                .map(|d| {
                    let (s, c) = m.iter().map(|&i| u.col(cols[i])[d]).filter(|x| x.is_finite()).fold((0.0, 0usize), |(s, c), x| (s + x as f64, c + 1));
                    if c > 0 { s / c as f64 } else { f64::NAN }
                })
                .collect();
            // The best-matching fund. Correlation is the same at any leverage, so among funds
            // within 0.02 of the best, leveraged ones (over 1.5x the calmest one's volatility)
            // are passed over and the most traded of the rest names the group.
            let fits: Vec<(usize, f64, f64)> = funds
                .iter()
                .map(|&f| {
                    let p = pair(avg.iter().copied(), year.clone().map(|d| u.col(f)[d] as f64));
                    (f, p.corr, p.var_b)
                })
                .filter(|x| x.1.is_finite())
                .collect();
            let best = fits.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
            let near: Vec<&(usize, f64, f64)> = fits.iter().filter(|x| x.1 >= best - 0.02).collect();
            let calm = near.iter().map(|x| x.2).fold(f64::INFINITY, f64::min);
            let tracks = near
                .into_iter()
                .filter(|x| x.2 <= calm * 2.25)
                .max_by(|a, b| u.dollar_volume[0][a.0].total_cmp(&u.dollar_volume[0][b.0]))
                .map(|&(f, r, _)| (u.tickers[f].clone(), r));
            let kept = earlier_labels
                .iter()
                .zip(&chance)
                .map(|(lab, &ch)| {
                    let (mut same, mut both) = (0usize, 0usize);
                    for &(a, b) in &pairs {
                        if let (Some(ga), Some(gb)) = (lab.get(&cols[a]), lab.get(&cols[b])) {
                            both += 1;
                            same += usize::from(ga == gb);
                        }
                    }
                    (both >= 30).then(|| (same as f64 / both as f64, ch))
                })
                .collect();
            Group {
                members: m.clone(),
                tracks,
                corr_relative: mean_of(&corr),
                corr_raw: mean_of(&raw),
                year_return: avg.iter().filter(|x| x.is_finite()).map(|x| 1.0 + x).product::<f64>() - 1.0,
                kept,
            }
        })
        .collect();
    // Stocks in group order for the heatmap; members re-indexed to that order.
    let flat: Vec<usize> = members.iter().flatten().copied().collect();
    let pos: HashMap<usize, usize> = flat.iter().enumerate().map(|(p, &i)| (i, p)).collect();
    let group_of: Vec<usize> = members.iter().enumerate().flat_map(|(g, m)| std::iter::repeat_n(g, m.len())).collect();
    let corr_sorted: Vec<f32> = flat.iter().flat_map(|&a| flat.iter().map(move |&b| (a, b))).map(|(a, b)| corr[a * n + b]).collect();
    let groups = groups.into_iter().map(|mut g| {
        g.members = g.members.iter().map(|i| pos[i]).collect();
        g
    }).collect();
    Ok(GroupsMap {
        days: (u.days[year.start], u.days[year.end - 1], year.len()),
        stocks: flat.iter().map(|&i| cols[i]).collect(),
        tickers: flat.iter().map(|&i| u.tickers[cols[i]].clone()).collect(),
        groups,
        group_of,
        corr: corr_sorted,
        earlier,
    })
}

impl GroupsMap {
    /// The group a ticker is in, if it was grouped.
    pub fn group_of_ticker(&self, ticker: &str) -> Option<usize> {
        let t = ticker.trim().to_uppercase();
        self.tickers.iter().position(|x| *x == t).map(|i| self.group_of[i])
    }
}

/// Kind words for tables: "stock" or "fund".
pub fn kind(etf: bool) -> &'static str {
    if etf { "fund" } else { "stock" }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A universe from given return columns (t x n), all liquid.
    fn universe(cols: Vec<(&str, bool, Vec<f32>)>) -> Universe {
        let t = cols[0].2.len();
        let n = cols.len();
        let tickers: Vec<String> = cols.iter().map(|c| c.0.to_string()).collect();
        let index = tickers.iter().enumerate().map(|(j, s)| (s.clone(), j)).collect();
        Universe {
            days: (0..t as i32).map(|d| 20_000 + d).collect(),
            t,
            n,
            etf: cols.iter().map(|c| c.1).collect(),
            liquid: vec![true; n],
            dollar_volume: vec![(0..n).map(|j| 1e9 - j as f32).collect(); t / YEAR],
            ret: cols.iter().flat_map(|c| c.2.iter().copied()).collect(),
            tickers,
            index,
        }
    }

    fn noise(state: &mut u64, t: usize, scale: f32) -> Vec<f32> {
        (0..t)
            .map(|_| {
                *state ^= *state << 13;
                *state ^= *state >> 7;
                *state ^= *state << 17;
                let (u1, u2) = (((*state >> 11) as f64 / (1u64 << 53) as f64).max(1e-12), ((state.rotate_left(17) >> 11) as f64 / (1u64 << 53) as f64));
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32 * scale
            })
            .collect()
    }

    #[test]
    fn holdings_parse_with_or_without_amounts() {
        assert_eq!(parse_holdings("aapl msft, spy").unwrap(), vec![("AAPL".into(), 1.0), ("MSFT".into(), 1.0), ("SPY".into(), 1.0)]);
        assert_eq!(parse_holdings("AAPL:30\nMSFT $20; BRK.B 50%").unwrap(), vec![("AAPL".into(), 30.0), ("MSFT".into(), 20.0), ("BRK.B".into(), 50.0)]);
        assert!(parse_holdings("AAPL 30 MSFT").unwrap_err().to_string().contains("no amount for MSFT"));
        assert!(parse_holdings("AAPL -5").is_err());
        assert!(parse_holdings("30 AAPL").is_err());
        assert!(parse_holdings("AAPL aapl").unwrap_err().to_string().contains("twice"));
        assert!(parse_holdings("AAPL 10 20").is_err());
    }

    #[test]
    fn bets_count_independent_holdings_and_risk_shares_sum_to_one() {
        let mut s = 3u64;
        let t = 2 * YEAR;
        let f = noise(&mut s, t, 0.01);
        let a = noise(&mut s, t, 0.01);
        let b = noise(&mut s, t, 0.01);
        let c = noise(&mut s, t, 0.01);
        // Three copies of one fund behave like one bet; three independent funds like three.
        let copies = universe(vec![("X", true, f.clone()), ("Y", true, f.clone()), ("Z", true, f.clone())]);
        let p = portfolio(&copies, &parse_holdings("X Y Z").unwrap()).unwrap();
        assert!((p.bets - 1.0).abs() < 1e-6, "{}", p.bets);
        let indep = universe(vec![("A", false, a), ("B", false, b), ("C", false, c)]);
        let p = portfolio(&indep, &parse_holdings("A 1 B 2 C 1").unwrap()).unwrap();
        assert!((p.holdings.iter().map(|h| h.risk_share).sum::<f64>() - 1.0).abs() < 1e-9);
        // Unequal weights: fewer than 3 bets, more than 2.
        assert!(p.bets > 2.3 && p.bets < 3.0, "{}", p.bets);
        assert!(p.corr_before.iter().all(|c| c.is_some()));
        assert!(portfolio(&indep, &parse_holdings("A NOPE ZZZ").unwrap()).err().unwrap().to_string().contains("NOPE, ZZZ"));
    }

    #[test]
    fn a_late_holding_shortens_the_window_instead_of_counting_zeros() {
        let mut s = 5u64;
        let t = 2 * YEAR;
        let a = noise(&mut s, t, 0.01);
        let mut b = a.clone();
        for v in &mut b[..t - 100] {
            *v = f32::NAN;
        }
        let u = universe(vec![("A", false, a), ("B", false, b)]);
        let p = portfolio(&u, &parse_holdings("A B").unwrap()).unwrap();
        assert_eq!(p.days.2, 100);
        assert!(p.shortened.as_deref().unwrap().contains("B has returns only from"));
        // Over the shared days they are the same series.
        assert!((p.corr[1] - 1.0).abs() < 1e-9);
        // B has no data the year before: "new", not a correlation.
        assert_eq!(p.corr_before[1], None);
    }

    #[test]
    fn hedge_ratio_and_risk_left_follow_a_planted_beta() {
        let mut s = 9u64;
        let t = 2 * YEAR;
        let h = noise(&mut s, t, 0.01);
        let e = noise(&mut s, t, 0.005);
        let target: Vec<f32> = h.iter().zip(&e).map(|(x, y)| 0.5 * x + y).collect();
        let inverse: Vec<f32> = h.iter().map(|x| -x).collect();
        let other = noise(&mut s, t, 0.01);
        let u = universe(vec![("T", false, target), ("H", true, h), ("INV", true, inverse), ("O", false, other)]);
        let r = hedges(&u, "t", 5).unwrap();
        assert_eq!(r.closest[0].ticker, "H");
        assert!((r.closest[0].ratio - 0.5).abs() < 0.06, "{}", r.closest[0].ratio);
        // corr = 0.5 / sqrt(0.25 + 0.25) = 0.71, so about 71% of the risk is left.
        assert!((r.closest[0].corr - 0.707).abs() < 0.05 && (r.closest[0].risk_left - 0.707).abs() < 0.05);
        assert_eq!(r.opposite[0].ticker, "INV");
        assert!((r.opposite[0].ratio + 0.5).abs() < 0.06);
    }

    #[test]
    fn diversifiers_prefer_what_moves_on_its_own() {
        let mut s = 21u64;
        let t = 2 * YEAR;
        let m = noise(&mut s, t, 0.01);
        let col = |s: &mut u64, load: f32| -> Vec<f32> { m.iter().zip(noise(s, t, 0.01)).map(|(x, e)| load * x + e).collect() };
        let a = col(&mut s, 1.0);
        let b = col(&mut s, 1.0);
        let same = col(&mut s, 1.5);
        let own = noise(&mut s, t, 0.01);
        let inverse: Vec<f32> = m.iter().map(|x| -x).collect();
        let cash = noise(&mut s, t, 0.0001);
        let own3: Vec<f32> = own.iter().map(|x| 3.0 * x).collect();
        let u = universe(vec![("SPY", true, m.clone()), ("A", false, a), ("B", false, b), ("SAME", false, same), ("OWN3", true, own3), ("OWN", true, own), ("SH", true, inverse), ("CASH", true, cash)]);
        let p = portfolio(&u, &parse_holdings("A B").unwrap()).unwrap();
        let d = diversifiers(&u, &p, 0.1, 0.75, 5);
        // OWN moves on its own; its 3x version is the same fund, shown under it.
        assert_eq!(d.funds[0].ticker, "OWN");
        assert_eq!(d.funds[0].alike, ["OWN3"]);
        assert!(d.funds[0].vol_after < p.vol);
        assert!(d.hedges.iter().any(|c| c.ticker == "SH"));
        assert!(!d.funds.iter().chain(&d.stocks).any(|c| c.ticker == "SH" || c.ticker == "CASH"));
        assert_eq!(d.cash_like, 1);
    }

    #[test]
    fn groups_find_planted_sectors_and_their_fund() {
        // Three sectors of 30 stocks on their own factors plus a market; one fund per sector.
        let mut s = 33u64;
        let t = 3 * YEAR;
        let market = noise(&mut s, t, 0.01);
        let sectors: Vec<Vec<f32>> = (0..3).map(|_| noise(&mut s, t, 0.01)).collect();
        let mut cols: Vec<(String, bool, Vec<f32>)> = Vec::new();
        for (g, f) in sectors.iter().enumerate() {
            for i in 0..30 {
                let e = noise(&mut s, t, 0.01);
                cols.push((format!("S{g}X{i}"), false, (0..t).map(|d| market[d] + f[d] + e[d]).collect()));
            }
            cols.push((format!("FUND{g}"), true, (0..t).map(|d| market[d] + f[d]).collect()));
        }
        let u = universe(cols.iter().map(|(a, b, c)| (a.as_str(), *b, c.clone())).collect());
        let map = groups(&u, 3, 1000, 2, &|_| {}).unwrap();
        assert_eq!(map.stocks.len(), 90);
        for g in &map.groups {
            let sector = &map.tickers[g.members[0]][1..2];
            assert!(g.members.iter().all(|&i| &map.tickers[i][1..2] == sector), "mixed group");
            assert_eq!(g.tracks.as_ref().unwrap().0, format!("FUND{sector}"));
            // The same sectors a year and two years earlier: all pairs kept, chance a third.
            assert!(g.kept.iter().all(|k| k.is_some_and(|(kept, chance)| kept > 0.99 && (chance - 1.0 / 3.0).abs() < 0.01)));
        }
    }

    #[test]
    fn words_cover_the_scale() {
        assert_eq!(words(0.95), "almost always together");
        assert_eq!(words(0.7), "usually together");
        assert_eq!(words(0.0), "mostly independent");
        assert_eq!(words(-0.7), "usually opposite");
        assert_eq!(words(f64::NAN), "unknown");
    }
}
