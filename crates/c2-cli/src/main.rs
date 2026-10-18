//! `c2`: correlation2 command line.

use std::time::Instant;

use anyhow::Result;
use c2_engine::{ColumnStats, cpu, pearson, reference_r, synth};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "c2", about = "correlation2: stock lead-lag census and holdout testing")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Time the all-pairs cross-product in every precision on the GPU and CPU, checking each
    /// against an exact f64 reference.
    Bench {
        /// Number of stocks.
        #[arg(long, default_value_t = 5000)]
        n: usize,
        /// Rows (trading days) per product, comma-separated. 64 = one census block.
        #[arg(long, value_delimiter = ',', default_value = "64,256,768,1280")]
        rows: Vec<usize>,
        /// Skip the CPU measurements.
        #[arg(long)]
        no_cpu: bool,
    },
}

fn main() -> Result<()> {
    #[cfg(feature = "cuda")]
    c2_engine::gpu::reexec_with_cuda_libraries();
    match Cli::parse().cmd {
        Cmd::Bench { n, rows, no_cpu } => bench(n, &rows, !no_cpu),
    }
}

/// Lag-1 leader and target panels (`kp x n`, zero-padded to a multiple of 32 rows) cut from a
/// `k + 1` day synthetic panel, plus that panel for the reference.
fn lag1_panels(n: usize, k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let t = k + 1;
    let full = synth::returns_panel(t, n, 42 + k as u64);
    let kp = k.div_ceil(32) * 32;
    let (mut x, mut y) = (vec![0f32; kp * n], vec![0f32; kp * n]);
    for j in 0..n {
        x[j * kp..j * kp + k].copy_from_slice(&full[j * t..j * t + k]);
        y[j * kp..j * kp + k].copy_from_slice(&full[j * t + 1..j * t + 1 + k]);
    }
    (x, y, full, kp)
}

/// Pairs to check: random ones plus planted sector leads (leader i < 20, target in its sector).
fn sample_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut rng = synth::Rng::new(7);
    let mut pairs: Vec<(usize, usize)> = (0..1500)
        .map(|_| ((rng.next_u64() % n as u64) as usize, (rng.next_u64() % n as u64) as usize))
        .collect();
    pairs.extend((0..500).map(|q| (q % 20, 20 + (q * 20 + q % 20) % (n - 20))));
    pairs
}

/// Largest |r - reference| over `pairs`, given a cross-product lookup and the panels it came from.
fn max_error(cross: impl Fn(usize, usize) -> f64, xs: &[f32], ys: &[f32], kp: usize, k: usize, n: usize, full: &[f32], pairs: &[(usize, usize)]) -> f64 {
    let (sx, sy) = (ColumnStats::of(xs, kp, n, 0..k), ColumnStats::of(ys, kp, n, 0..k));
    pairs
        .iter()
        .map(|&(i, j)| {
            let r = pearson(cross(i, j), sx.sum[i], sy.sum[j], sx.sum_sq[i], sy.sum_sq[j], k as f64);
            (r - reference_r(full, k + 1, i, j, 1)).abs()
        })
        .fold(0.0, f64::max)
}

fn bench(n: usize, rows: &[usize], with_cpu: bool) -> Result<()> {
    let pairs = sample_pairs(n);
    println!("{n} stocks; times are per all-pairs product (one lag); error = max |r - f64 reference| over {} pairs\n", pairs.len());

    #[cfg(feature = "cuda")]
    if let Err(e) = bench_gpu(n, rows, &pairs) {
        println!("GPU unavailable, CPU only: {e:#}");
    }

    if with_cpu {
        println!("\nCPU ({} threads, gemm f32)", rayon::current_num_threads());
        println!("{:>6} {:>10} {:>9} {:>10}", "rows", "ms", "TOPS", "error");
        for &k in rows {
            let (x, y, full, kp) = lag1_panels(n, k);
            let mut out = vec![0f32; n * n];
            cpu::xprod_f32(&x, &y, kp, n, kp, &mut out);
            let reps = if k <= 256 { 5 } else { 2 };
            let start = Instant::now();
            for _ in 0..reps {
                cpu::xprod_f32(&x, &y, kp, n, kp, &mut out);
            }
            let secs = start.elapsed().as_secs_f64() / reps as f64;
            let err = max_error(|i, j| out[i * n + j] as f64, &x, &y, kp, k, n, &full, &pairs);
            println!("{:>6} {:>10.2} {:>9.2} {:>10.1e}", k, secs * 1e3, 2.0 * (n * n * kp) as f64 / secs / 1e12, err);
        }

        println!("\nCPU ({} threads, INT8 AVX-VNNI assembly micro-kernel)", rayon::current_num_threads());
        println!("{:>6} {:>10} {:>9} {:>10} {:>10} {:>12}", "rows", "ms", "TOPS", "error", "pack ms", "no-store ms");
        for &k in rows {
            let (x, y, full, kp) = lag1_panels(n, k);
            let (xq, yq): (Vec<i8>, Vec<i8>) = (x.iter().map(|&v| c2_engine::quantize(v)).collect(), y.iter().map(|&v| c2_engine::quantize(v)).collect());
            let start = Instant::now();
            let packed = c2_engine::vnni::Packed::new(&xq, &yq, kp, n, kp);
            let pack_secs = start.elapsed().as_secs_f64();
            let mut out = vec![0i32; n * n];
            packed.xprod(&mut out);
            let reps = if k <= 256 { 5 } else { 2 };
            let start = Instant::now();
            for _ in 0..reps {
                packed.xprod(&mut out);
            }
            let secs = start.elapsed().as_secs_f64() / reps as f64;
            // Compute only (tiles visited, nothing written): the cost a fused census epilogue starts from.
            let start = Instant::now();
            for _ in 0..reps {
                packed.for_each_tile(|_, _, t| {
                    std::hint::black_box(t);
                });
            }
            let compute_secs = start.elapsed().as_secs_f64() / reps as f64;
            let (xs, ys): (Vec<f32>, Vec<f32>) = (xq.iter().map(|&v| v as f32).collect(), yq.iter().map(|&v| v as f32).collect());
            let err = max_error(|i, j| out[i * n + j] as f64, &xs, &ys, kp, k, n, &full, &pairs);
            println!("{:>6} {:>10.2} {:>9.2} {:>10.1e} {:>10.2} {:>12.2}", k, secs * 1e3, 2.0 * (n * n * kp) as f64 / secs / 1e12, err, pack_secs * 1e3, compute_secs * 1e3);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn bench_gpu(n: usize, rows: &[usize], pairs: &[(usize, usize)]) -> Result<()> {
    use c2_engine::gpu::{DevProduct, Gpu, HIST_BINS, Precision};

    let gpu = Gpu::new()?;
    let (free, total) = gpu.memory()?;
    println!("GPU: {} ({:.1} of {:.1} GB free)", gpu.name()?, free as f64 / 1e9, total as f64 / 1e9);
    println!("{:>6} {:>6} {:>10} {:>9} {:>10}", "rows", "mode", "ms", "TOPS", "error");
    for &k in rows {
        let (x, y, full, kp) = lag1_panels(n, k);
        for p in Precision::ALL {
            let (dx, dy) = (gpu.upload(&x, p)?, gpu.upload(&y, p)?);
            let mut c = gpu.alloc_product(p, n)?;
            let plan = if p == Precision::Fp32 { None } else { Some(gpu.plan(p, n, kp)?) };
            gpu.xprod(p, plan.as_ref(), &dx, &dy, &mut c, n, kp)?;
            gpu.stream.synchronize()?;
            let reps = if k <= 256 { 50 } else { 20 };
            let start = Instant::now();
            for _ in 0..reps {
                gpu.xprod(p, plan.as_ref(), &dx, &dy, &mut c, n, kp)?;
            }
            gpu.stream.synchronize()?;
            let secs = start.elapsed().as_secs_f64() / reps as f64;

            // Error is measured on the values the GPU actually saw (rounded or quantized).
            let seen = |v: &[f32]| -> Vec<f32> {
                match p {
                    Precision::Fp16 => v.iter().map(|&a| half::f16::from_f32(a).to_f32()).collect(),
                    Precision::Bf16 => v.iter().map(|&a| half::bf16::from_f32(a).to_f32()).collect(),
                    Precision::Int8 => v.iter().map(|&a| c2_engine::quantize(a) as f32).collect(),
                    _ => v.to_vec(),
                }
            };
            let (xs, ys) = (seen(&x), seen(&y));
            let err = match &c {
                DevProduct::F32(d) => {
                    let h = gpu.stream.clone_dtoh(d)?;
                    max_error(|i, j| h[i + j * n] as f64, &xs, &ys, kp, k, n, &full, pairs)
                }
                DevProduct::I32(d) => {
                    let h = gpu.stream.clone_dtoh(d)?;
                    max_error(|i, j| h[i + j * n] as f64, &xs, &ys, kp, k, n, &full, pairs)
                }
            };
            println!("{:>6} {:>6} {:>10.3} {:>9.1} {:>10.1e}", k, p.name(), secs * 1e3, 2.0 * (n * n * kp) as f64 / secs / 1e12, err);

            // The census pass that follows each product: r, |z| histogram, compaction.
            if k == rows[0] && matches!(p, Precision::Fp32 | Precision::Int8) {
                let (sxh, syh) = (ColumnStats::of(&xs, kp, n, 0..k), ColumnStats::of(&ys, kp, n, 0..k));
                let factors = gpu.pass_factors(&sxh, &syh, k as f64)?;
                let s = &gpu.stream;
                let cap = 1 << 20;
                let mut hist = s.alloc_zeros::<u32>(HIST_BINS)?;
                let mut count = s.alloc_zeros::<u32>(1)?;
                let (mut oi, mut or) = (s.alloc_zeros::<i64>(cap)?, s.alloc_zeros::<f32>(cap)?);
                let run = |hist: &mut _, count: &mut _, oi: &mut _, or: &mut _| {
                    gpu.census_pass(&c, n, k as f64, &factors, 16.0, 4.0, cap, hist, count, oi, or)
                };
                run(&mut hist, &mut count, &mut oi, &mut or)?;
                s.synchronize()?;
                let start = Instant::now();
                for _ in 0..20 {
                    s.memset_zeros(&mut hist)?;
                    s.memset_zeros(&mut count)?;
                    run(&mut hist, &mut count, &mut oi, &mut or)?;
                }
                s.synchronize()?;
                let secs = start.elapsed().as_secs_f64() / 20.0;
                let (h, cnt) = (s.clone_dtoh(&hist)?, s.clone_dtoh(&count)?);
                let total: u64 = h.iter().map(|&v| v as u64).sum();
                println!("{:>6} {:>6} {:>10.3}   census pass ({:.0} GB/s read; {} of {} entries histogrammed; {} with |z| >= 4)",
                    "", "", secs * 1e3, (n * n * 4) as f64 / secs / 1e9, total, n * n, cnt[0]);
            }
        }
    }
    Ok(())
}
