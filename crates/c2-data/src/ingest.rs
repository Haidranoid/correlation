//! Stooq bulk-zip ingest (`d_us_txt.zip` and friends), parsed in parallel straight from the zip.
//!
//! Entry paths look like `data/daily/us/nasdaq stocks/1/aapl.us.txt`. The folder (`nasdaq
//! stocks`, `nyse etfs`, ...) gives the exchange and whether it is a stock or an ETF; v1 threw
//! this away. Files are `<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`.

use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;

use crate::store::{Kind, Series, SymbolInfo};

/// Exchange test tickers (ZVZZT, ZWZZT, ... and *TEST* variants). They sit in Stooq's stock
/// folders and trade like stocks, but their prices are meaningless (ZVZZT moves 3,000x a day).
pub fn is_test_ticker(t: &str) -> bool {
    let b = t.as_bytes();
    (b.len() == 5 && b[0] == b'Z' && &b[2..] == b"ZZT") || t.contains("TEST") || t.starts_with("ZXYZ")
}

#[derive(Debug, Default)]
pub struct IngestReport {
    pub files: usize,
    pub empty: usize,
    pub rows: usize,
    /// Files per Stooq folder (e.g. "us/nasdaq stocks").
    pub by_folder: BTreeMap<String, usize>,
    pub seconds: f64,
}

fn parse_series(text: &str) -> (Vec<i32>, [Vec<f32>; 5]) {
    let mut date = Vec::new();
    let mut cols: [Vec<f32>; 5] = Default::default();
    for line in text.lines().skip(1) {
        let mut f = line.split(',');
        let (Some(_), Some(_), Some(d), Some(_)) = (f.next(), f.next(), f.next(), f.next()) else { continue };
        if d.len() != 8 {
            continue;
        }
        let (Ok(y), Ok(m), Ok(dd)) = (d[..4].parse::<i32>(), d[4..6].parse::<u32>(), d[6..].parse::<u32>()) else { continue };
        let vals: Vec<f32> = f.take(5).map(|v| v.parse::<f32>().unwrap_or(f32::NAN)).collect();
        if vals.len() < 4 {
            continue;
        }
        date.push(crate::days_from_civil(y, m, dd));
        for (c, col) in cols.iter_mut().enumerate() {
            col.push(*vals.get(c).unwrap_or(&0.0));
        }
    }
    // Stooq files are date-sorted; make sure, and drop duplicate dates (keep the last).
    if date.windows(2).any(|w| w[0] >= w[1]) {
        let mut idx: Vec<usize> = (0..date.len()).collect();
        idx.sort_by_key(|&i| (date[i], i));
        idx.dedup_by(|b, a| date[*a] == date[*b] && { *a = *b; true });
        date = idx.iter().map(|&i| date[i]).collect();
        for col in cols.iter_mut() {
            *col = idx.iter().map(|&i| col[i]).collect();
        }
    }
    (date, cols)
}

/// Parses every `data/daily/<market>/...` text file in the given zips and writes the store.
/// `markets` limits which top-level markets are kept (e.g. `["us"]`).
pub fn ingest_stooq(zips: &[&Path], markets: &[&str], out_dir: &Path) -> Result<IngestReport> {
    let start = Instant::now();
    let mut report = IngestReport::default();
    let mut all: Vec<Series> = Vec::new();
    for zip_path in zips {
        let file = std::fs::File::open(zip_path).with_context(|| format!("open {}", zip_path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let names: Vec<(usize, String)> = {
            let archive = zip::ZipArchive::new(Cursor::new(&mmap[..]))?;
            (0..archive.len()).filter_map(|i| archive.name_for_index(i)?.ok().map(|n| (i, n.to_string()))).collect()
        };
        let wanted: Vec<(usize, String, String, String)> = names
            .into_iter()
            .filter_map(|(i, name)| {
                let parts: Vec<&str> = name.split('/').collect();
                if parts.len() < 5 || parts[0] != "data" || parts[1] != "daily" || !name.ends_with(".txt") {
                    return None;
                }
                let market = parts[2];
                if !markets.contains(&market) {
                    return None;
                }
                let stem = parts.last()?.trim_end_matches(".txt");
                let ticker = stem.strip_suffix(&format!(".{market}")).unwrap_or(stem).to_uppercase();
                Some((i, name.clone(), format!("{market}/{}", parts[3]), ticker))
            })
            .collect();
        for (_, _, folder, _) in &wanted {
            *report.by_folder.entry(folder.clone()).or_default() += 1;
        }
        report.files += wanted.len();

        let chunk = wanted.len().div_ceil(rayon::current_num_threads() * 4).max(1);
        let parsed: Vec<Series> = wanted
            .par_chunks(chunk)
            .map(|batch| -> Result<Vec<Series>> {
                let mut archive = zip::ZipArchive::new(Cursor::new(&mmap[..]))?;
                let mut out = Vec::with_capacity(batch.len());
                let mut text = String::new();
                for (i, name, folder, ticker) in batch {
                    text.clear();
                    archive.by_index(*i)?.read_to_string(&mut text)?;
                    let (date, [open, high, low, close, volume]) = parse_series(&text);
                    let words = folder.split('/').nth(1).unwrap_or("");
                    let kind = if is_test_ticker(ticker) {
                        Kind::Test
                    } else if words.contains("etf") {
                        Kind::Etf
                    } else if words.contains("stock") {
                        Kind::Stock
                    } else {
                        Kind::Other
                    };
                    let info = SymbolInfo {
                        ticker: ticker.clone(),
                        exchange: words.split(' ').next().unwrap_or("").to_string(),
                        kind,
                        first: *date.first().unwrap_or(&0),
                        last: *date.last().unwrap_or(&0),
                        rows: date.len(),
                        source: name.clone(),
                    };
                    out.push(Series { info, date, open, high, low, close, volume });
                }
                Ok(out)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        all.extend(parsed);
    }
    report.empty = all.iter().filter(|s| s.date.is_empty()).count();
    all.retain(|s| !s.date.is_empty());
    all.sort_by(|a, b| (&a.info.ticker, &a.info.exchange).cmp(&(&b.info.ticker, &b.info.exchange)));
    report.rows = all.iter().map(|s| s.date.len()).sum();
    crate::store::write(out_dir, &all)?;
    report.seconds = start.elapsed().as_secs_f64();
    Ok(report)
}
