//! How fast is the all-pairs lead-lag search on the GPU?
//!
//! For a T x N matrix Z of standardized returns (one column per stock), the correlation of
//! every stock on day t with every stock on day t+k is one matrix product:
//! C = Z[0..T-k]^T * Z[k..T] / (T-k). This times that product with cuBLAS at the sizes that
//! matter and checks a sample of entries against a CPU calculation.
//!
//!     LD_LIBRARY_PATH=$HOME/model-lab/qwen3.8-flash-next/cuda-env/lib cargo run --release

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaStream};

fn random_panel(n: usize, t: usize) -> Vec<f32> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n * t)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

/// Seconds per lag-1 all-pairs product, and the largest error against the CPU on sampled entries.
fn bench(stream: &Arc<CudaStream>, blas: &CudaBlas, n: usize, t: usize, reps: usize) -> Result<(f64, f32), Box<dyn Error>> {
    let lag = 1;
    let host = random_panel(n, t);
    let z = stream.clone_htod(&host)?;
    let mut c = stream.alloc_zeros::<f32>(n * n)?;
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: n as i32,
        k: (t - lag) as i32,
        alpha: 1.0 / (t - lag) as f32,
        lda: t as i32,
        ldb: t as i32,
        beta: 0.0,
        ldc: n as i32,
    };
    let a = z.slice(0..);
    let b = z.slice(lag..);
    unsafe { blas.gemm(cfg, &a, &b, &mut c)? };
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..reps {
        unsafe { blas.gemm(cfg, &a, &b, &mut c)? };
    }
    stream.synchronize()?;
    let secs = start.elapsed().as_secs_f64() / reps as f64;

    // Spot-check: C[i, j] = sum_s Z[s, i] * Z[s + lag, j] / (t - lag), column-major (row i, column j).
    let gpu = stream.clone_dtoh(&c)?;
    let mut worst = 0f32;
    for (i, j) in [(0, 1), (n / 2, n / 3), (n - 1, 0), (7 % n, n - 2)] {
        let mut sum = 0f64;
        for s in 0..t - lag {
            sum += host[i * t + s] as f64 * host[j * t + s + lag] as f64;
        }
        let cpu = (sum / (t - lag) as f64) as f32;
        worst = worst.max((gpu[j * n + i] - cpu).abs());
    }
    Ok((secs, worst))
}

fn main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    println!("GPU: {}", ctx.name()?);
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;

    println!("{:>8} {:>7} {:>12} {:>10} {:>12}", "stocks", "days", "ms per lag", "TFLOP/s", "max error");
    for (n, t, reps) in [(2_155, 755, 50), (5_000, 755, 20), (11_431, 755, 10), (11_431, 5_000, 5)] {
        let (secs, err) = bench(&stream, &blas, n, t, reps)?;
        let tflops = 2.0 * n as f64 * n as f64 * (t - 1) as f64 / secs / 1e12;
        println!("{:>8} {:>7} {:>12.2} {:>10.1} {:>12.2e}", n, t, secs * 1e3, tflops, err);
    }
    Ok(())
}
