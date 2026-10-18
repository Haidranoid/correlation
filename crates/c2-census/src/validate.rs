//! Correctness check against `research/census_oracle.json` (exact r from Python for one window).
//!
//! Recomputes the oracle's pairs three ways: exactly in f64 from the same standardized data, on
//! the GPU census path (prefix sums -> INT8 panels -> cuBLASLt -> census pass), and with the
//! CPU AVX-VNNI kernel on bit-identical INT8 panels.

use std::path::Path;

use anyhow::{Context, Result};
use c2_data::{Calendar, Store};
use c2_engine::{pearson, quantize_scaled, vnni, INT8_SCALE};

pub struct Report {
    pub universe: usize,
    pub missing: Vec<String>,
    pub days: usize,
    pub pairs: usize,
    pub max_err_exact: f64,
    pub max_err_gpu_int8: f64,
    pub max_err_cpu_int8: f64,
    pub max_cpu_gpu_diff: f64,
    pub top50_found: usize,
    pub top50_total: usize,
}

struct Pair {
    lag: usize,
    i: usize,
    j: usize,
    r: f64,
}

/// Per-column prefix sums of a `w x n` column-major panel: `(w + 1) x n`, `p[0] = 0`.
fn prefix_sums(d: &[f64], w: usize, n: usize) -> Vec<f64> {
    let mut p = vec![0f64; (w + 1) * n];
    for j in 0..n {
        for r in 0..w {
            p[j * (w + 1) + r + 1] = p[j * (w + 1) + r] + d[j * w + r];
        }
    }
    p
}

/// INT8 panel exactly as the GPU `build_panel` kernel computes it (h = 1, step 1).
fn host_panel(p: &[f64], w: usize, n: usize, offset: usize, m: usize, kp: usize) -> Vec<i8> {
    let mut out = vec![0i8; kp * n];
    for j in 0..n {
        let col = &p[j * (w + 1)..(j + 1) * (w + 1)];
        for q in 0..m {
            let e = q + offset;
            out[j * kp + q] = quantize_scaled((col[e + 1] - col[e]) as f32 * INT8_SCALE);
        }
    }
    out
}

fn column_stats(panel: &[i8], kp: usize, rows: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    (0..n)
        .map(|j| panel[j * kp..j * kp + rows].iter().fold((0.0, 0.0), |(s, q), &v| (s + v as f64, q + (v as f64) * (v as f64))))
        .unzip()
}

pub fn validate(store: &Store, oracle: &Path) -> Result<Report> {
    let text = std::fs::read_to_string(oracle).with_context(|| format!("read {}", oracle.display()))?;
    let o: serde_json::Value = serde_json::from_str(&text)?;
    let window = (
        c2_data::parse_date(o["window"][0].as_str().context("window")?)?,
        c2_data::parse_date(o["window"][1].as_str().context("window")?)?,
    );
    let tickers: Vec<String> = o["universe"].as_array().context("universe")?.iter().filter_map(|v| v.as_str().map(String::from)).collect();
    let mut ids = Vec::new();
    let mut col_of = std::collections::HashMap::new();
    let mut missing = Vec::new();
    for t in &tickers {
        match store.id(t) {
            Some(id) => {
                col_of.insert(t.clone(), ids.len());
                ids.push(id);
            }
            None => missing.push(t.clone()),
        }
    }
    let pairs: Vec<Pair> = o["pairs"]
        .as_array()
        .context("pairs")?
        .iter()
        .filter_map(|p| {
            Some(Pair {
                lag: p["lag"].as_u64()? as usize,
                i: *col_of.get(p["leader"].as_str()?)?,
                j: *col_of.get(p["target"].as_str()?)?,
                r: p["r"].as_f64()?,
            })
        })
        .collect();

    let cal = Calendar::from_symbol(store, "SPY")?;
    let d = crate::variable::python_style(store, &cal, &ids, c2_data::parse_date("2016-05-01")?, window);
    let n = ids.len();
    let w = d.len() / n;

    // Exact, f64.
    let exact = |p: &Pair| {
        let (x, y) = (&d[p.i * w..p.i * w + w - p.lag], &d[p.j * w + p.lag..p.j * w + w]);
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (&a, &b) in x.iter().zip(y) {
            sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b;
        }
        pearson(sxy, sx, sy, sxx, syy, (w - p.lag) as f64)
    };
    let max_err_exact = pairs.iter().map(|p| (exact(p) - p.r).abs()).fold(0.0, f64::max);

    // INT8 panels (bit-identical on CPU and GPU), CPU VNNI products.
    let pre = prefix_sums(&d, w, n);
    let kp = w.div_ceil(32) * 32;
    let x = host_panel(&pre, w, n, 0, w, kp);
    let mut cpu_r = std::collections::HashMap::new();
    for lag in [0usize, 1] {
        let m = w - lag;
        let y = host_panel(&pre, w, n, lag, m, kp);
        let (xs, xq) = column_stats(&x, kp, m, n);
        let (ys, yq) = column_stats(&y, kp, m, n);
        let mut c = vec![0i32; n * n];
        // Leader rows beyond m multiply zero target rows, so the full-kp product is exact.
        vnni::Packed::new(&x, &y, kp, n, kp).xprod(&mut c);
        for p in pairs.iter().filter(|p| p.lag == lag) {
            let r = pearson(c[p.i * n + p.j] as f64, xs[p.i], ys[p.j], xq[p.i], yq[p.j], m as f64);
            cpu_r.insert((lag, p.i, p.j), r);
        }
    }
    let max_err_cpu_int8 = pairs.iter().map(|p| (cpu_r[&(p.lag, p.i, p.j)] - p.r).abs()).fold(0.0, f64::max);

    let top50: Vec<&Pair> = pairs.iter().filter(|p| p.lag == 1).skip(2000.min(pairs.iter().filter(|p| p.lag == 1).count().saturating_sub(50))).collect();
    let (max_err_gpu_int8, max_cpu_gpu_diff, top50_found) = gpu_check(&pre, w, n, kp, &pairs, &cpu_r, &top50)?;
    Ok(Report {
        universe: tickers.len(),
        missing,
        days: w,
        pairs: pairs.len(),
        max_err_exact,
        max_err_gpu_int8,
        max_err_cpu_int8,
        max_cpu_gpu_diff,
        top50_found,
        top50_total: top50.len(),
    })
}

#[cfg(feature = "cuda")]
fn gpu_check(
    pre: &[f64], w: usize, n: usize, kp: usize, pairs: &[Pair],
    cpu_r: &std::collections::HashMap<(usize, usize, usize), f64>, top50: &[&Pair],
) -> Result<(f64, f64, usize)> {
    use c2_engine::gpu::{Gpu, HIST_BINS, Precision};
    let gpu = Gpu::new()?;
    let s = gpu.stream.clone();
    let np = crate::run::pad_width(n);
    let dpre = s.clone_htod(pre)?;
    let mut x = s.alloc_zeros::<i8>(kp * np)?;
    let mut y = s.alloc_zeros::<i8>(kp * np)?;
    let (mut xs, mut xq, mut ys, mut yq) = (s.alloc_zeros::<f32>(np)?, s.alloc_zeros::<f32>(np)?, s.alloc_zeros::<f32>(np)?, s.alloc_zeros::<f32>(np)?);
    let mut c = s.alloc_zeros::<i32>(np * np)?;
    let mut f = gpu.alloc_factors(np)?;
    let plan = gpu.plan(Precision::Int8, np, kp)?;
    gpu.build_panel(&dpre, w + 1, n, 0, 1, 0, 1, w, 0, 0, kp, &mut x, &mut xs, &mut xq)?;
    let (mut max_err, mut max_diff, mut found) = (0f64, 0f64, 0usize);
    for lag in [0usize, 1] {
        let m = w - lag;
        gpu.build_panel(&dpre, w + 1, n, 0, 1, lag, 1, m, 0, 0, kp, &mut y, &mut ys, &mut yq)?;
        gpu.col_stats(&x, kp, m, np, &mut xs, &mut xq)?;
        gpu.xprod_i8(&plan, &x, &y, &mut c)?;
        let (ch, xsh, xqh, ysh, yqh) = (s.clone_dtoh(&c)?, s.clone_dtoh(&xs)?, s.clone_dtoh(&xq)?, s.clone_dtoh(&ys)?, s.clone_dtoh(&yq)?);
        for p in pairs.iter().filter(|p| p.lag == lag) {
            // cuBLAS column-major: C[i + j * np].
            let r = pearson(ch[p.i + p.j * np] as f64, xsh[p.i] as f64, ysh[p.j] as f64, xqh[p.i] as f64, yqh[p.j] as f64, m as f64);
            max_err = max_err.max((r - p.r).abs());
            max_diff = max_diff.max((r - cpu_r[&(lag, p.i, p.j)]).abs());
        }
        if lag == 1 {
            gpu.factors(&xs, &xq, &ys, &yq, np, m, &mut f)?;
            let cap = 1 << 20;
            let mut hist = s.alloc_zeros::<u32>(HIST_BINS)?;
            let mut count = s.alloc_zeros::<u32>(1)?;
            let (mut oi, mut or) = (s.alloc_zeros::<i64>(cap)?, s.alloc_zeros::<f32>(cap)?);
            gpu.census_pass_i32(&c, np, m as f64, &f, 16.0, 3.5, cap, &mut hist, &mut count, &mut oi, &mut or)?;
            let cnt = (s.clone_dtoh(&count)?[0] as usize).min(cap);
            let idx: std::collections::HashSet<i64> = s.clone_dtoh(&oi)?[..cnt].iter().copied().collect();
            found = top50.iter().filter(|p| idx.contains(&((p.j * np + p.i) as i64))).count();
        }
    }
    Ok((max_err, max_diff, found))
}

#[cfg(not(feature = "cuda"))]
fn gpu_check(
    _: &[f64], _: usize, _: usize, _: usize, _: &[Pair],
    _: &std::collections::HashMap<(usize, usize, usize), f64>, _: &[&Pair],
) -> Result<(f64, f64, usize)> {
    anyhow::bail!("built without CUDA")
}
