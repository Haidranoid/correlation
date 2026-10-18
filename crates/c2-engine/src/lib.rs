//! Correlation engine: all-pairs lagged cross-correlation on the CPU and the GPU.
//!
//! Panels are stored column-major: `t` rows (time) by `n` columns (stocks), so each stock's
//! series is contiguous. The cross-product of a leader panel X and a target panel Y over the same
//! rows is `C = X^T Y` (n x n); Pearson r follows from C and per-column sums (`pearson`).

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod synth;
pub mod vnni;

/// Volatility-normalized returns are clipped at +-5 sd and stored as INT8 with this many steps per sd.
pub const INT8_SCALE: f32 = 25.0;

/// Per-column sums over a row range, enough to turn cross-products into exact Pearson r.
#[derive(Clone, Debug)]
pub struct ColumnStats {
    pub sum: Vec<f64>,
    pub sum_sq: Vec<f64>,
}

impl ColumnStats {
    /// Stats of rows `rows` of a column-major `t x n` panel.
    pub fn of<T: Copy + Into<f64>>(panel: &[T], t: usize, n: usize, rows: std::ops::Range<usize>) -> Self {
        let (mut sum, mut sum_sq) = (vec![0.0; n], vec![0.0; n]);
        for j in 0..n {
            for &v in &panel[j * t + rows.start..j * t + rows.end] {
                let v: f64 = v.into();
                sum[j] += v;
                sum_sq[j] += v * v;
            }
        }
        Self { sum, sum_sq }
    }
}

/// Exact Pearson r from a cross-product and the two columns' sums over `n` shared rows.
#[inline]
pub fn pearson(sxy: f64, sx: f64, sy: f64, sxx: f64, syy: f64, n: f64) -> f64 {
    let cov = sxy - sx * sy / n;
    let vx = sxx - sx * sx / n;
    let vy = syy - sy * sy / n;
    if vx <= 0.0 || vy <= 0.0 { 0.0 } else { cov / (vx * vy).sqrt() }
}

/// Reference r for leader column `i` at rows `s` and target column `j` at rows `s + lag`, in f64.
pub fn reference_r(panel: &[f32], t: usize, i: usize, j: usize, lag: usize) -> f64 {
    let rows = t - lag;
    let (x, y) = (&panel[i * t..i * t + rows], &panel[j * t + lag..j * t + lag + rows]);
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&a, &b) in x.iter().zip(y) {
        let (a, b) = (a as f64, b as f64);
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
    }
    pearson(sxy, sx, sy, sxx, syy, rows as f64)
}

/// Quantize a volatility-normalized value (in sd units) to INT8.
#[inline]
pub fn quantize(v: f32) -> i8 {
    (v * INT8_SCALE).round().clamp(-127.0, 127.0) as i8
}
