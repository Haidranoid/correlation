//! Night and day, Part B (`experiments/-6.0-night-and-day.md`): the cluster lead-lag
//! strategy's own daily book (`c2_census::cluster`, all-exchanges settings), written out per
//! stock so it can be split into the overnight and intraday legs in Python.
//!
//! It replays `c2 census clusters --all-exchanges` exactly: decision rows from the first row with a
//! fit's look-back and warm-up (`window + 30`) to the second-to-last row, a fit every `refit` rows
//! from the first, and each day's book is +1/p for the members of groups forecast to rise, -1/p
//! for those forecast to fall (p = the fit's universe), as `cluster::backtest` holds them. The
//! variable is read from a judge export (the same columns as `load_export`), so the output is
//! aligned to it; the store is only opened because `Fit::new` takes one (exchange names).
//!
//! Output (float32 `[N, T]` .npy, column = decision row t, held over t + 1; NaN outside the fit's
//! universe): `cluster_w_<tag>.npy` (the book's weights) and `cluster_f_<tag>.npy` (the member's
//! group forecast); and float64 `cluster_s_<tag>.npy`, the holdout harness's score exactly as
//! `cluster::scores` makes it (the group forecast times 1 + 1e-9 x the member's follow rank).
//!
//! Build (outside the memory cap) and run (under it):
//!   cargo build --release -p c2-census --example night_and_day_cluster
//!   target/release/examples/night_and_day_cluster --export DIR --store DIR --out DIR --tag old

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use c2_census::cluster::{Fit, Settings};
use c2_census::variable::Variable;
use c2_data::Store;

fn read_npy(path: &Path) -> Result<(String, Vec<usize>, Vec<u8>)> {
    let mut buf = Vec::new();
    File::open(path).with_context(|| format!("open {}", path.display()))?.read_to_end(&mut buf)?;
    ensure!(buf.len() > 10 && &buf[..6] == b"\x93NUMPY", "{} is not an npy file", path.display());
    let (hlen, start) = match buf[6] {
        1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize, 12),
        v => bail!("npy version {v}"),
    };
    let header = String::from_utf8_lossy(&buf[start..start + hlen]).to_string();
    ensure!(header.contains("'fortran_order': False"), "{}: Fortran order", path.display());
    let descr = header.split("'descr': '").nth(1).and_then(|s| s.split('\'').next()).context("descr")?.to_string();
    let shape_s = header.split("'shape': (").nth(1).and_then(|s| s.split(')').next()).context("shape")?;
    let shape: Vec<usize> = shape_s.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
    Ok((descr, shape, buf[start + hlen..].to_vec()))
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn write_npy_f32(path: &Path, n: usize, t: usize, data: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_npy(path, "<f4", n, t, &bytes)
}

fn write_npy_f64(path: &Path, n: usize, t: usize, data: &[f64]) -> Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_npy(path, "<f8", n, t, &bytes)
}

fn write_npy(path: &Path, descr: &str, n: usize, t: usize, bytes: &[u8]) -> Result<()> {
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n}, {t}), }}");
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut f = std::io::BufWriter::new(File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    f.write_all(bytes)?;
    Ok(())
}

fn load_variable(dir: &Path) -> Result<Variable> {
    let (d, shape, b) = read_npy(&dir.join("days.npy"))?;
    ensure!(d == "<i4", "days dtype {d}");
    let days: Vec<i32> = b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let t = shape[0];
    let (d, _, b) = read_npy(&dir.join("market.npy"))?;
    ensure!(d == "<f8", "market dtype {d}");
    let market: Vec<f64> = b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let arr = |name: &str| -> Result<(usize, Vec<f32>)> {
        let (d, shape, b) = read_npy(&dir.join(format!("{name}.npy")))?;
        ensure!(d == "<f4" && shape.len() == 2 && shape[1] == t, "{name}: {d} {shape:?}");
        Ok((shape[0], f32s(&b)))
    };
    let (n, close) = arr("close")?;
    let (_, dollar_volume) = arr("dollar_volume")?;
    let (_, v) = arr("v")?;
    let mut ids = Vec::with_capacity(n);
    let tsv = std::fs::read_to_string(dir.join("symbols.tsv"))?;
    for line in tsv.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        ids.push(f[1].parse::<usize>()?);
    }
    ensure!(ids.len() == n, "symbols.tsv has {} rows, arrays {n}", ids.len());
    Ok(Variable { days, ids, t, n, v, close, dollar_volume, market })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| -> Option<String> { args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned() };
    let export = PathBuf::from(get("--export").context("--export DIR")?);
    let store_dir = PathBuf::from(get("--store").context("--store DIR")?);
    let out = PathBuf::from(get("--out").context("--out DIR")?);
    let tag = get("--tag").unwrap_or_else(|| "old".into());
    let store = Store::open(&store_dir)?;
    let var = load_variable(&export)?;
    for &id in &var.ids {
        ensure!(id < store.symbols.len(), "store id {id} outside the store");
    }
    let s = Settings { nyse_only: false, ..Settings::paper() };
    eprintln!("settings: {}", s.describe());
    let (n, t) = (var.n, var.t);
    let first = s.window + 30;
    let last = t - 2;
    let mut w = vec![f32::NAN; n * t];
    let mut fc = vec![f32::NAN; n * t];
    let mut sc = vec![f64::NAN; n * t];
    let mut start = first;
    let mut fits = 0;
    let t0 = std::time::Instant::now();
    while start <= last {
        let m = s.refit.min(last + 1 - start);
        let fit = Fit::new(&store, &var, start, &s)?;
        let p = fit.members.len();
        let ahead = fit.ahead(&var, m, &s);
        let f = fit.forecasts(&fit.group, &fit.model, &ahead, m, &s);
        // Follow rank as in `cluster::scores`: 0 for the member that moves first most.
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&a, &b| fit.lead[b].total_cmp(&fit.lead[a]));
        let mut follow = vec![0.0; p];
        for (r, &k) in order.iter().enumerate() {
            follow[k] = r as f64 / p as f64;
        }
        for i in 0..m {
            let row = start + i;
            for (k, &j) in fit.members.iter().enumerate() {
                let g = f[i * s.groups + fit.group[k]];
                fc[j * t + row] = g as f32;
                sc[j * t + row] = g * (1.0 + 1e-9 * follow[k]);
                w[j * t + row] = if g == 0.0 { 0.0 } else { (g.signum() / p as f64) as f32 };
            }
        }
        fits += 1;
        if fits % 20 == 0 {
            eprintln!("  {fits} fits, at {} ({:.0}s)", c2_data::date_string(var.days[start]), t0.elapsed().as_secs_f64());
        }
        start += m;
    }
    std::fs::create_dir_all(&out)?;
    write_npy_f32(&out.join(format!("cluster_w_{tag}.npy")), n, t, &w)?;
    write_npy_f32(&out.join(format!("cluster_f_{tag}.npy")), n, t, &fc)?;
    write_npy_f64(&out.join(format!("cluster_s_{tag}.npy")), n, t, &sc)?;
    eprintln!("{fits} fits; decision rows {}..{} ({}..{}); wrote {}", first, last, c2_data::date_string(var.days[first]), c2_data::date_string(var.days[last]), out.display());
    Ok(())
}
