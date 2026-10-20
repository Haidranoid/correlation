//! The price store: one memory-mapped binary file of per-symbol columns plus a symbol table.
//!
//! `prices.bin` layout (little-endian):
//!   magic `C2PRICE1`, u32 version, u32 symbol count, u64 row count,
//!   u64 row offset per symbol (+1 sentinel), then columns, each starting on a 64-byte boundary:
//!   date i32, open f32, high f32, low f32, close f32, volume f32 (all `rows` long).
//! Rows of one symbol are contiguous and sorted by date. Prices are as the source gives them
//! (Stooq: split- and dividend-adjusted); cleaning happens when panels are built.
//! `symbols.tsv`: id, ticker, exchange, kind, first date, last date, rows, source path.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

const MAGIC: &[u8; 8] = b"C2PRICE1";
const VERSION: u32 = 1;
const COLUMNS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Stock,
    Etf,
    /// Exchange test symbols such as ZVZZT: never real securities.
    Test,
    Other,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Stock => "stock",
            Kind::Etf => "etf",
            Kind::Test => "test",
            Kind::Other => "other",
        }
    }

    fn parse(s: &str) -> Kind {
        match s {
            "stock" => Kind::Stock,
            "etf" => Kind::Etf,
            "test" => Kind::Test,
            _ => Kind::Other,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SymbolInfo {
    pub ticker: String,
    pub exchange: String,
    pub kind: Kind,
    pub first: i32,
    pub last: i32,
    pub rows: usize,
    pub source: String,
}

/// One symbol's price series, ready to be written.
pub struct Series {
    pub info: SymbolInfo,
    pub date: Vec<i32>,
    pub open: Vec<f32>,
    pub high: Vec<f32>,
    pub low: Vec<f32>,
    pub close: Vec<f32>,
    pub volume: Vec<f32>,
}

fn pad64(w: &mut impl Write, pos: &mut u64) -> Result<()> {
    let pad = (64 - (*pos % 64)) % 64;
    w.write_all(&vec![0u8; pad as usize])?;
    *pos += pad;
    Ok(())
}

/// Writes `prices.bin` and `symbols.tsv` into `dir` (via temporary files, renamed at the end).
pub fn write(dir: &Path, series: &[Series]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let rows: u64 = series.iter().map(|s| s.date.len() as u64).sum();
    let tmp = dir.join("prices.bin.tmp");
    {
        let mut w = BufWriter::with_capacity(8 << 20, File::create(&tmp)?);
        let mut pos = 0u64;
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(series.len() as u32).to_le_bytes())?;
        w.write_all(&rows.to_le_bytes())?;
        pos += 24;
        let mut off = 0u64;
        for s in series {
            w.write_all(&off.to_le_bytes())?;
            off += s.date.len() as u64;
        }
        w.write_all(&off.to_le_bytes())?;
        pos += 8 * (series.len() as u64 + 1);
        type Col = fn(&Series) -> &[u8];
        let cols: [Col; COLUMNS] = [
            |s| bytes_of(&s.date),
            |s| bytes_of(&s.open),
            |s| bytes_of(&s.high),
            |s| bytes_of(&s.low),
            |s| bytes_of(&s.close),
            |s| bytes_of(&s.volume),
        ];
        for col in cols {
            pad64(&mut w, &mut pos)?;
            for s in series {
                let b = col(s);
                w.write_all(b)?;
                pos += b.len() as u64;
            }
        }
        w.flush()?;
    }
    std::fs::rename(&tmp, dir.join("prices.bin"))?;

    let tmp = dir.join("symbols.tsv.tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        writeln!(w, "id\tticker\texchange\tkind\tfirst\tlast\trows\tsource")?;
        for (id, s) in series.iter().enumerate() {
            let i = &s.info;
            writeln!(w, "{id}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", i.ticker, i.exchange, i.kind.as_str(),
                     crate::date_string(i.first), crate::date_string(i.last), i.rows, i.source)?;
        }
        w.flush()?;
    }
    std::fs::rename(&tmp, dir.join("symbols.tsv"))?;
    Ok(())
}

/// Parses a `symbols.tsv` (the store's, or a census run's copy of it). Line k after the header
/// describes store id k.
pub fn parse_symbols(tsv: &str) -> Result<Vec<SymbolInfo>> {
    tsv.lines()
        .skip(1)
        .enumerate()
        .map(|(k, line)| {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 7 || f[0].parse::<usize>().ok() != Some(k) {
                bail!("symbols.tsv line {} is not symbol {k}: {line:?}", k + 2);
            }
            Ok(SymbolInfo {
                ticker: f[1].to_string(),
                exchange: f[2].to_string(),
                kind: Kind::parse(f[3]),
                first: crate::parse_date(f[4])?,
                last: crate::parse_date(f[5])?,
                rows: f[6].parse()?,
                source: f.get(7).unwrap_or(&"").to_string(),
            })
        })
        .collect()
}

fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Read-only view of a written store.
pub struct Store {
    mmap: Mmap,
    dir: PathBuf,
    pub symbols: Vec<SymbolInfo>,
    offsets: Vec<u64>,
    col_start: [usize; COLUMNS],
}

impl Store {
    pub fn open(dir: &Path) -> Result<Store> {
        let file = File::open(dir.join("prices.bin")).with_context(|| format!("no price store in {} (run `c2 ingest`)", dir.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 24 || &mmap[..8] != MAGIC {
            bail!("{} is not a correlation2 price store", dir.join("prices.bin").display());
        }
        let u32_at = |p: usize| u32::from_le_bytes(mmap[p..p + 4].try_into().unwrap());
        let u64_at = |p: usize| u64::from_le_bytes(mmap[p..p + 8].try_into().unwrap());
        if u32_at(8) != VERSION {
            bail!("price store version {} (expected {VERSION}); re-run `c2 ingest`", u32_at(8));
        }
        let (n, rows) = (u32_at(12) as usize, u64_at(16) as usize);
        let offsets: Vec<u64> = (0..=n).map(|i| u64_at(24 + 8 * i)).collect();
        let mut pos = 24 + 8 * (n + 1);
        let mut col_start = [0usize; COLUMNS];
        for c in col_start.iter_mut() {
            pos = pos.div_ceil(64) * 64;
            *c = pos;
            pos += rows * 4;
        }
        if pos > mmap.len() {
            bail!("price store is truncated");
        }

        let symbols = parse_symbols(&std::fs::read_to_string(dir.join("symbols.tsv"))?)?;
        if symbols.len() != n {
            bail!("symbols.tsv has {} symbols, prices.bin {n}", symbols.len());
        }
        Ok(Store { mmap, dir: dir.to_path_buf(), symbols, offsets, col_start })
    }

    /// The folder the store was opened from.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn rows(&self) -> usize {
        *self.offsets.last().unwrap() as usize
    }

    fn col<T>(&self, c: usize, id: usize) -> &[T] {
        let (a, b) = (self.offsets[id] as usize, self.offsets[id + 1] as usize);
        let base = self.col_start[c] + a * 4;
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr().add(base) as *const T, b - a) }
    }

    pub fn dates(&self, id: usize) -> &[i32] {
        self.col(0, id)
    }
    pub fn open_(&self, id: usize) -> &[f32] {
        self.col(1, id)
    }
    pub fn high(&self, id: usize) -> &[f32] {
        self.col(2, id)
    }
    pub fn low(&self, id: usize) -> &[f32] {
        self.col(3, id)
    }
    pub fn close(&self, id: usize) -> &[f32] {
        self.col(4, id)
    }
    pub fn volume(&self, id: usize) -> &[f32] {
        self.col(5, id)
    }

    /// Id of the first symbol with this ticker.
    pub fn id(&self, ticker: &str) -> Option<usize> {
        self.symbols.iter().position(|s| s.ticker == ticker)
    }
}
