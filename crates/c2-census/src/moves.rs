//! Moves in price terms shared by the forecast, the stock timeline and the Explainer.
//!
//! A move is a sum of daily log returns over its move length. A market-relative move sums the
//! census's market residuals (`Variable::market_relative`): each day's return minus beta x the
//! market's return, before the census divides by volatility. That is the part of a stock's move
//! the census correlates, in price terms; if the market is flat it is the stock's own move.

use std::ops::Range;

use crate::grid;

/// Fewest observations with both moves known for a size to be given.
pub const MIN_PAIRS: usize = 10;

/// Sum of `daily` over rows end+1-h..=end: the h-day move ending at row `end`. NaN if any of
/// those days is missing, or the rows fall outside `daily`.
pub fn move_sum(daily: &[f64], end: usize, h: usize) -> f64 {
    if h == 0 || end + 1 < h || end >= daily.len() {
        return f64::NAN;
    }
    daily[end + 1 - h..=end].iter().sum()
}

/// One relationship's size in price terms.
#[derive(Clone, Copy, Debug)]
pub struct MoveSize {
    /// r x sd(B's moves) / sd(A's moves): B's move relative to the market per 1% move of A
    /// relative to the market (the "% per 1%" the timeline and the Explainer show, and the
    /// multiplier the forecast applies). Same sign as r.
    pub per_pct: f64,
    /// Standard deviations of A's h1-day and B's h2-day market-relative moves over the pairs
    /// (log changes; 0.02 is about 2%).
    pub sd_a: f64,
    pub sd_b: f64,
    /// Observations with both moves known.
    pub pairs: usize,
}

/// The size of relationship "A moves first over h1 days, B follows over h2 days ending `lag`
/// days after A's move ends", with census correlation `r`, in the correlation period on rows
/// `rows` (`[start, end)`). `a` and `b` are A's and B's market-relative daily log returns on the
/// same rows (`Variable::market_relative`). Measured on the census's own observation grid for
/// that period: A's moves end at row start + max(h1, h2) - 1 + q * max(h1, h2) for
/// q < `grid::observations(rows.len(), h1, h2, lag)`, each paired with B's move ending `lag`
/// rows later. Pairs with a missing move are skipped; None if fewer than `MIN_PAIRS` remain or
/// either move never varies.
pub fn move_size(a: &[f64], b: &[f64], r: f64, h1: usize, h2: usize, lag: usize, rows: Range<usize>) -> Option<MoveSize> {
    if h1 == 0 || h2 == 0 {
        return None;
    }
    let step = h1.max(h2);
    let pairs: Vec<(f64, f64)> = (0..grid::observations(rows.len(), h1, h2, lag))
        .map(|q| {
            let e = rows.start + step - 1 + q * step;
            (move_sum(a, e, h1), move_sum(b, e + lag, h2))
        })
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    if pairs.len() < MIN_PAIRS {
        return None;
    }
    let n = pairs.len() as f64;
    let (mx, my) = (pairs.iter().map(|p| p.0).sum::<f64>() / n, pairs.iter().map(|p| p.1).sum::<f64>() / n);
    let sxx: f64 = pairs.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    let syy: f64 = pairs.iter().map(|p| (p.1 - my) * (p.1 - my)).sum();
    if !(sxx > 0.0 && syy > 0.0) {
        return None;
    }
    let (sd_a, sd_b) = ((sxx / n).sqrt(), (syy / n).sqrt());
    let per_pct = r * sd_b / sd_a;
    per_pct.is_finite().then_some(MoveSize { per_pct, sd_a, sd_b, pairs: pairs.len() })
}

/// `move_size(...).per_pct`: B's move relative to the market per 1% move of A relative to the
/// market.
pub fn move_per_pct(a: &[f64], b: &[f64], r: f64, h1: usize, h2: usize, lag: usize, rows: Range<usize>) -> Option<f64> {
    move_size(a, b, r, h1, h2, lag, rows).map(|s| s.per_pct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_sum_needs_every_day() {
        let d = [0.01, 0.02, f64::NAN, 0.03];
        assert!((move_sum(&d, 1, 2) - 0.03).abs() < 1e-15);
        assert!(move_sum(&d, 2, 2).is_nan());
        assert!(move_sum(&d, 0, 2).is_nan());
        assert!(move_sum(&d, 4, 1).is_nan());
    }

    #[test]
    fn size_is_r_times_the_ratio_of_spreads_on_the_census_grid() {
        // B's daily residual is 3x A's from `lag` rows earlier: with h1 = h2 its moves on the
        // grid are exactly 3x A's, so sd_b / sd_a = 3.
        let (t, lag) = (400, 4);
        let a: Vec<f64> = (0..t).map(|k| ((k * 7919 % 101) as f64 - 50.0) / 1000.0).collect();
        let b: Vec<f64> = (0..t).map(|k| if k >= lag { 3.0 * a[k - lag] } else { f64::NAN }).collect();
        let s = move_size(&a, &b, 0.5, 2, 2, lag, 10..300).unwrap();
        assert!((s.sd_b / s.sd_a - 3.0).abs() < 1e-12);
        assert!((s.per_pct - 1.5).abs() < 1e-12);
        assert_eq!(s.pairs, grid::observations(290, 2, 2, lag));
        // Too few observations.
        assert!(move_size(&a, &b, 0.5, 2, 2, lag, 10..30).is_none());
    }
}
