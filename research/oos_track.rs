//! Forecast track record before and after a split day. The same calls as `c2 census forecast`
//! (Plan::for_run, forecast::prepare, forecast::forecast_from), market-relative throughout.
//!
//! - In sample: each stock's track record as of the split day (what the Forecast tab shows when
//!   the data ends there), pooled over the stocks.
//! - Out of sample: a 21-day forecast made as of the split day and then every 21 trading days,
//!   while the 21 days after it are in the data, compared with what happened.
//!
//! C2_DATA_DIR=... cargo run --release -p c2-census --example oos_track -- RUN SPLIT PAIRS_OUT TICKER...

use anyhow::{Context, Result};
use c2_census::{forecast, meta::RunMeta, run};
use std::io::Write;

fn corr(p: &[(f64, f64)]) -> f64 {
    let n = p.len() as f64;
    let (mx, my) = (p.iter().map(|q| q.0).sum::<f64>() / n, p.iter().map(|q| q.1).sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in p {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    if sxx > 0.0 && syy > 0.0 { sxy / (sxx * syy).sqrt() } else { 0.0 }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(args.len() >= 4, "usage: RUN SPLIT PAIRS_OUT TICKER...");
    let (name, split, out) = (&args[0], c2_data::parse_date(&args[1])?, &args[2]);
    let tickers: Vec<String> = args[3..].iter().map(|t| t.trim().to_uppercase()).collect();
    let store = c2_data::Store::open(&c2_data::data_dir().join("store"))?;
    let dir = run::run_dir(name);
    let meta = RunMeta::load(&dir)?;
    let plan = run::Plan::for_run(&store, &meta)?;
    let days = &plan.var.days;
    let r1 = days.partition_point(|&d| d <= split).checked_sub(1).context("split day before the data")?;
    let last = days.len() - 1;
    let h = 21;
    let opt = forecast::Options { significant_only: false, horizon: h, max_age: None };
    println!(
        "run {name}; data {} .. {}; split {}; {} tickers; out-of-sample forecasts as of {} .. (every {h} trading days)",
        c2_data::date_string(days[0]),
        c2_data::date_string(days[last]),
        c2_data::date_string(days[r1]),
        tickers.len(),
        c2_data::date_string(days[r1])
    );
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    writeln!(file, "ticker as_of predicted_log actual_log relationships")?;
    let (mut in_hits, mut in_dir, mut in_corrs) = (0usize, 0usize, Vec::new());
    let mut oos: Vec<(usize, usize, f64, f64)> = Vec::new(); // (ticker index, row, predicted, actual)
    let mut per_ticker: Vec<(String, usize, usize, f64)> = Vec::new();
    for (ti, t) in tickers.iter().enumerate() {
        let data = match forecast::prepare(&store, &plan.var, &dir, t) {
            Ok(d) => d,
            Err(e) => {
                println!("{t}: left out ({e:#})");
                continue;
            }
        };
        match forecast::forecast_from(&data, days[r1], &opt).ok().and_then(|f| f.skill) {
            Some(s) => {
                in_hits += (s.hit_rate * s.directional as f64).round() as usize;
                in_dir += s.directional;
                in_corrs.push(s.corr);
            }
            None => println!("{t}: no in-sample track record"),
        }
        let mut mine = Vec::new();
        let mut k = r1;
        while k + h <= last {
            if let Ok(f) = forecast::forecast_from(&data, days[k], &opt) {
                let actual = f.actual_pct.last().copied().flatten();
                if let (Some(&p), Some(a), true) = (f.predicted_pct.last(), actual, f.covering.iter().any(|&n| n > 0)) {
                    let (pl, al) = ((p / 100.0).ln_1p(), (a / 100.0).ln_1p());
                    writeln!(file, "{t} {} {pl:e} {al:e} {}", c2_data::date_string(days[k]), f.relationships)?;
                    oos.push((ti, k, pl, al));
                    mine.push((pl, al));
                }
            }
            k += h;
        }
        let d: Vec<&(f64, f64)> = mine.iter().filter(|(p, a)| *p != 0.0 && *a != 0.0).collect();
        let hits = d.iter().filter(|(p, a)| (*p > 0.0) == (*a > 0.0)).count();
        per_ticker.push((t.clone(), hits, d.len(), corr(&mine)));
    }
    file.flush()?;

    let pos = in_corrs.iter().filter(|&&c| c > 0.0).count();
    println!(
        "\nIn sample (track record as of {}): {} stocks, direction right {:.1}% of {in_dir}, mean correlation {:+.3} ({pos} of {} positive)",
        c2_data::date_string(days[r1]),
        in_corrs.len(),
        100.0 * in_hits as f64 / in_dir.max(1) as f64,
        in_corrs.iter().sum::<f64>() / in_corrs.len().max(1) as f64,
        in_corrs.len()
    );

    let pairs: Vec<(f64, f64)> = oos.iter().map(|q| (q.2, q.3)).collect();
    let dir_pairs: Vec<&(usize, usize, f64, f64)> = oos.iter().filter(|q| q.2 != 0.0 && q.3 != 0.0).collect();
    let hits = dir_pairs.iter().filter(|q| (q.2 > 0.0) == (q.3 > 0.0)).count();
    let n = dir_pairs.len() as f64;
    let hit = hits as f64 / n;
    let pp = dir_pairs.iter().filter(|q| q.2 > 0.0).count() as f64 / n;
    let pa = dir_pairs.iter().filter(|q| q.3 > 0.0).count() as f64 / n;
    let base = pp * pa + (1.0 - pp) * (1.0 - pa);
    // Forecasts made on the same day move together, so the honest error bar treats each
    // forecast day as one observation: the spread of the per-day hit rates.
    let mut rows: Vec<usize> = dir_pairs.iter().map(|q| q.1).collect();
    rows.sort();
    rows.dedup();
    let by_day: Vec<f64> = rows
        .iter()
        .map(|&r| {
            let on: Vec<_> = dir_pairs.iter().filter(|q| q.1 == r).collect();
            on.iter().filter(|q| (q.2 > 0.0) == (q.3 > 0.0)).count() as f64 / on.len() as f64
        })
        .collect();
    let m = by_day.iter().sum::<f64>() / by_day.len() as f64;
    let sd = (by_day.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (by_day.len() as f64 - 1.0)).sqrt();
    let se = sd / (by_day.len() as f64).sqrt();
    let tc: Vec<f64> = per_ticker.iter().filter(|p| p.2 >= 5).map(|p| p.3).collect();
    println!(
        "Out of sample: {} forecasts ({} with a direction) on {} forecast days; direction right {:.1}% ({hits} of {}), {:.1}% expected from the up/down rates alone",
        oos.len(),
        dir_pairs.len(),
        rows.len(),
        100.0 * hit,
        dir_pairs.len(),
        100.0 * base
    );
    println!(
        "  per forecast day: mean {:.1}%, standard error {:.1} points (so about {:.1}% to {:.1}% at 95%)",
        100.0 * m,
        100.0 * se,
        100.0 * (m - 1.96 * se),
        100.0 * (m + 1.96 * se)
    );
    println!(
        "  correlation of predicted with actual, pooled: {:+.3}; mean per stock {:+.3} ({} of {} positive)",
        corr(&pairs),
        tc.iter().sum::<f64>() / tc.len().max(1) as f64,
        tc.iter().filter(|&&c| c > 0.0).count(),
        tc.len()
    );
    println!(
        "  mean |predicted| {:.2}%, mean |actual| {:.2}%",
        100.0 * pairs.iter().map(|q| q.0.abs()).sum::<f64>() / pairs.len() as f64,
        100.0 * pairs.iter().map(|q| q.1.abs()).sum::<f64>() / pairs.len() as f64
    );
    println!("\nPer stock, out of sample (direction right / with a direction, correlation):");
    for (t, hits, n, c) in &per_ticker {
        println!("  {t:<6} {hits:>2}/{n:<2} {c:+.2}");
    }
    Ok(())
}
