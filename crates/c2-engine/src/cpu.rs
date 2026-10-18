//! CPU backend: f32 cross-products with the `gemm` crate (AVX2/FMA, all cores via rayon).

/// `out[i * n + j] = sum_s x[i][s] * y[j][s]` for column-major `k x n` panels `x` and `y` whose
/// columns start `ld` elements apart. `out` is row-major `n x n`.
///
/// `gemm::gemm` takes each matrix as (pointer, column stride, row stride) and computes
/// `dst = alpha * dst + beta * lhs * rhs`. Here lhs = X^T (row i is stock i's series) and
/// rhs = Y (column j is stock j's series).
pub fn xprod_f32(x: &[f32], y: &[f32], k: usize, n: usize, ld: usize, out: &mut [f32]) {
    assert!(out.len() >= n * n && x.len() >= (n - 1) * ld + k && y.len() >= (n - 1) * ld + k);
    unsafe {
        gemm::gemm(
            n, n, k,
            out.as_mut_ptr(), 1, n as isize,
            false,
            x.as_ptr(), 1, ld as isize,
            y.as_ptr(), ld as isize, 1,
            0.0f32, 1.0f32,
            false, false, false,
            gemm::Parallelism::Rayon(0),
        );
    }
}
