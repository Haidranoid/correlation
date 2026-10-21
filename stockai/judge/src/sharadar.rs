//! `stockai-judge import-sharadar`: Sharadar's bulk tables (stocks, funds, tickers, actions) into a correlation2
//! `C2PRICE1` store, with the sidecars of docs/sharadar.md: `close_unadj.f32`, `delist.tsv` and `source.json`.
//! `check-sharadar` (check.rs) reads the same files through the same code.
//!
//! The rules (docs/sharadar.md, "The store"):
//! - Files: one per table in the raw directory, found by name: the table's name (or its Nasdaq Data Link code, SEP
//!   or SFP) as a word of the file name, ending in .csv, .csv.gz or .zip (holding one .csv). Columns are found by
//!   their header names. A missing column, a malformed date (strict YYYY-MM-DD) or a malformed number stops the
//!   import; an empty number (or N/A, NA, NaN, null, none) is missing.
//! - Symbols: every ticker of `stocks`, and SPY and QQQ of `funds` (Kind::Etf). A stock is Kind::Stock when its
//!   `tickers` row (table = stocks) has a category containing "Common Stock", else Kind::Other; one without a
//!   `tickers` row is Kind::Other (listed in the log). A ticker that two permatickers share in `tickers` is an error.
//! - Rows: sorted by date. Of two rows with the same date the one with the later `lastupdated` is kept (then the
//!   later line). Rows without a positive closeadj are dropped. close = closeadj; open, high and low are scaled by
//!   closeadj / close; volume is the split-adjusted volume (0 where missing). `close_unadj.f32` holds closeunadj
//!   (NaN where missing or not positive), one value per store row.
//! - Delistings: a Kind::Stock with isdelisted = Y gets a line in `delist.tsv`: the date of its last row, a kind
//!   from its own actions dated within 30 calendar days of that date (failure if any failure code is among them,
//!   else buyout if any buyout code is, else unknown), and those actions' distinct codes other than dividend.
//! - `--survivors-only DATE` leaves out every `stocks` ticker whose last row is dated before DATE.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use c2_data::store::{Series, SymbolInfo};
use c2_data::{Kind, civil_from_days, date_string, days_from_civil};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::npy;

/// The import rules' version, recorded in source.json (the binary's version is in the export's meta.json).
pub const IMPORTER: &str = "stockai-judge import-sharadar 1";
pub const SOURCE_FORMAT: &str = "stockai-sharadar-1";

/// Codes that make a delisting a buyout or a failure (docs/sharadar.md, "The store").
pub const BUYOUT_CODES: [&str; 3] = ["acquisitionby", "mergerto", "mergerfrom"];
pub const FAILURE_CODES: [&str; 2] = ["bankruptcyliquidation", "regulatorydelisting"];
/// Every action code the importer knows of; others are counted and listed as unknown.
pub const KNOWN_CODES: [&str; 18] = [
    "acquisitionby",
    "acquisitionof",
    "adrratiosplit",
    "bankruptcyliquidation",
    "delisted",
    "dividend",
    "listed",
    "mergerfrom",
    "mergerto",
    "regulatorydelisting",
    "relation",
    "spinoff",
    "spinoffdividend",
    "split",
    "spunofffrom",
    "tickerchangefrom",
    "tickerchangeto",
    "voluntarydelisting",
];
/// Actions this many calendar days before or after a delisted stock's last row decide its kind.
pub const WINDOW_DAYS: i32 = 30;
/// The funds kept, as benchmarks.
pub const BENCHMARKS: [&str; 2] = ["SPY", "QQQ"];
/// correlation2's calendar stocks (`Calendar::trading_days` needs at least 5 of them).
pub const CALENDAR: [&str; 8] = ["IBM", "GE", "KO", "XOM", "JNJ", "PG", "MMM", "AAPL"];
/// Store sidecars, in the order they are written; `source.json` last marks a complete import.
pub const SIDECARS: [&str; 3] = ["close_unadj.f32", "delist.tsv", "source.json"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Table {
    Stocks,
    Funds,
    Tickers,
    Actions,
}

impl Table {
    pub const ALL: [Table; 4] = [Table::Stocks, Table::Funds, Table::Tickers, Table::Actions];

    pub fn name(self) -> &'static str {
        match self {
            Table::Stocks => "stocks",
            Table::Funds => "funds",
            Table::Tickers => "tickers",
            Table::Actions => "actions",
        }
    }

    /// Words that name the table in a file name (Nasdaq Data Link called stocks and funds SEP and SFP).
    fn aliases(self) -> &'static [&'static str] {
        match self {
            Table::Stocks => &["stocks", "sep"],
            Table::Funds => &["funds", "sfp"],
            Table::Tickers => &["tickers"],
            Table::Actions => &["actions"],
        }
    }
}

/// How a delisted stock left (delist.tsv's `kind`, delist_kind.npy's codes 1..3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DelistKind {
    Buyout,
    Failure,
    Unknown,
}

impl DelistKind {
    pub const ALL: [DelistKind; 3] = [DelistKind::Buyout, DelistKind::Failure, DelistKind::Unknown];

    pub fn name(self) -> &'static str {
        match self {
            DelistKind::Buyout => "buyout",
            DelistKind::Failure => "failure",
            DelistKind::Unknown => "unknown",
        }
    }

    /// delist_kind.npy's code (0 is "listed").
    pub fn code(self) -> i8 {
        match self {
            DelistKind::Buyout => 1,
            DelistKind::Failure => 2,
            DelistKind::Unknown => 3,
        }
    }

    pub fn parse(s: &str) -> Option<DelistKind> {
        DelistKind::ALL.into_iter().find(|k| k.name() == s)
    }

    /// The kind given by the codes of a stock's actions near its last row.
    pub fn of<'a>(codes: impl IntoIterator<Item = &'a str> + Clone) -> DelistKind {
        if codes.clone().into_iter().any(|c| FAILURE_CODES.contains(&c)) {
            DelistKind::Failure
        } else if codes.into_iter().any(|c| BUYOUT_CODES.contains(&c)) {
            DelistKind::Buyout
        } else {
            DelistKind::Unknown
        }
    }
}

/// A raw file as read.
#[derive(Clone, Debug)]
pub struct FileInfo {
    pub table: Table,
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    /// Data rows in the file (all of them, also those not kept).
    pub rows: usize,
    pub columns: Vec<String>,
    pub lastupdated_max: Option<i32>,
}

/// One raw file per table in `dir` (see the module notes). Hidden files (partial downloads) are skipped.
pub fn find_files(dir: &Path) -> Result<Vec<(Table, PathBuf)>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading the raw directory {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    paths.sort();
    let mut out = Vec::new();
    for table in Table::ALL {
        let hits: Vec<&PathBuf> = paths.iter().filter(|p| names_table(p, table)).collect();
        match hits.as_slice() {
            [one] => out.push((table, (*one).clone())),
            [] => bail!(
                "{}: no file for the {} table (a name such as {t}.csv, {t}.csv.gz or {t}.zip)",
                dir.display(),
                table.name(),
                t = table.name()
            ),
            many => bail!(
                "{}: {} files could be the {} table ({}); keep one",
                dir.display(),
                many.len(),
                table.name(),
                many.iter().map(|p| p.file_name().unwrap_or_default().to_string_lossy()).collect::<Vec<_>>().join(", ")
            ),
        }
    }
    Ok(out)
}

fn names_table(path: &Path, table: Table) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else { return false };
    let lower = name.to_ascii_lowercase();
    if lower.starts_with('.') {
        return false;
    }
    let Some(stem) = [".csv.gz", ".csv", ".zip"].iter().find_map(|ext| lower.strip_suffix(ext)) else { return false };
    stem.split(|c: char| !c.is_ascii_alphanumeric()).any(|w| table.aliases().contains(&w))
}

/// Calls `f` with the CSV text of `path`: plain, gzip (.gz) or the one .csv entry of a zip.
fn with_csv<T>(path: &Path, f: impl FnOnce(&mut dyn Read) -> Result<T>) -> Result<T> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let lower = path.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    if lower.ends_with(".zip") {
        let mut archive = zip::ZipArchive::new(BufReader::new(file)).with_context(|| format!("{}: not a readable zip", path.display()))?;
        let names: Vec<String> = (0..archive.len()).filter_map(|i| archive.name_for_index(i).and_then(|n| n.ok()).map(|n| n.into_owned())).collect();
        let csvs: Vec<usize> = (0..names.len()).filter(|&i| names[i].to_ascii_lowercase().ends_with(".csv")).collect();
        ensure!(csvs.len() == 1, "{}: expected one .csv inside, found {} ({})", path.display(), csvs.len(), names.join(", "));
        let mut entry = archive.by_index(csvs[0])?;
        f(&mut entry)
    } else if lower.ends_with(".gz") {
        f(&mut flate2::read::MultiGzDecoder::new(BufReader::with_capacity(1 << 20, file)))
    } else {
        f(&mut BufReader::with_capacity(1 << 20, file))
    }
}

/// Size and SHA-256 (hex) of a file.
pub fn sha256_file(path: &Path) -> Result<(u64, String)> {
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut bytes = 0u64;
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        bytes += k as u64;
    }
    Ok((bytes, format!("{:x}", h.finalize())))
}

/// Days since 1970-01-01 for a strict `YYYY-MM-DD` that is a real date, else None.
pub fn strict_date(b: &[u8]) -> Option<i32> {
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let num = |s: &[u8]| s.iter().try_fold(0u32, |a, &c| c.is_ascii_digit().then(|| a * 10 + u32::from(c - b'0')));
    let (y, m, d) = (num(&b[..4])? as i32, num(&b[5..7])?, num(&b[8..])?);
    if !(1..=12).contains(&m) || d == 0 {
        return None;
    }
    let z = days_from_civil(y, m, d);
    (civil_from_days(z) == (y, m, d)).then_some(z)
}

fn is_missing(b: &[u8]) -> bool {
    let t = b.trim_ascii();
    t.is_empty() || ["n/a", "na", "nan", "null", "none"].iter().any(|m| t.eq_ignore_ascii_case(m.as_bytes()))
}

/// A number field: Some(None) when missing, None when malformed (or not finite).
fn number(b: &[u8]) -> Option<Option<f32>> {
    if is_missing(b) {
        return Some(None);
    }
    let x: f32 = std::str::from_utf8(b).ok()?.trim().parse().ok()?;
    x.is_finite().then_some(Some(x))
}

/// Text of a field (lossy UTF-8, trimmed).
fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).trim().to_string()
}

/// A CSV with its columns found by name.
struct Csv<'a> {
    table: Table,
    path: &'a Path,
    names: Vec<String>,
}

impl Csv<'_> {
    fn new<'a, R: Read>(table: Table, path: &'a Path, rdr: &mut csv::Reader<R>) -> Result<Csv<'a>> {
        let h = rdr.byte_headers().with_context(|| format!("{}: reading the header", path.display()))?;
        let names = h
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let s = String::from_utf8_lossy(b);
                let s = if i == 0 { s.trim_start_matches('\u{feff}') } else { &s };
                s.trim().to_ascii_lowercase()
            })
            .collect();
        Ok(Csv { table, path, names })
    }

    fn col(&self, want: &str) -> Result<usize> {
        self.names.iter().position(|n| n == want).with_context(|| {
            format!("{} table ({}): no column {want:?} (columns: {})", self.table.name(), self.path.display(), self.names.join(", "))
        })
    }

    fn cols(&self, want: &[&str]) -> Result<Vec<usize>> {
        want.iter().map(|w| self.col(w)).collect()
    }

    fn bad(&self, rec: &csv::ByteRecord, col: usize, what: &str) -> anyhow::Error {
        anyhow::anyhow!(
            "{} table ({}), line {}: {what} {:?} in column {:?}",
            self.table.name(),
            self.path.display(),
            rec.position().map_or(0, |p| p.line()),
            String::from_utf8_lossy(rec.get(col).unwrap_or_default()),
            self.names[col]
        )
    }

    fn date(&self, rec: &csv::ByteRecord, col: usize) -> Result<i32> {
        strict_date(rec[col].trim_ascii()).ok_or_else(|| self.bad(rec, col, "malformed date"))
    }

    /// A date that may be missing (empty or N/A).
    fn date_or_none(&self, rec: &csv::ByteRecord, col: usize) -> Result<Option<i32>> {
        if is_missing(&rec[col]) { Ok(None) } else { self.date(rec, col).map(Some) }
    }
}

fn reader<R: Read>(r: R) -> csv::Reader<R> {
    csv::ReaderBuilder::new().has_headers(true).flexible(false).buffer_capacity(1 << 20).from_reader(r)
}

fn read_error(table: Table, path: &Path, e: csv::Error) -> anyhow::Error {
    anyhow::Error::new(e).context(format!("{} table ({}): unreadable CSV", table.name(), path.display()))
}

/// One price row of `stocks` or `funds`.
#[derive(Clone, Copy, Debug)]
struct PriceRow {
    date: i32,
    updated: i32,
    open: f32,
    high: f32,
    low: f32,
    close: f32,
    volume: f32,
    closeadj: f32,
    closeunadj: f32,
}

/// A ticker's rows as read (any order).
struct RawSeries {
    ticker: String,
    fund: bool,
    rows: Vec<PriceRow>,
}

const PRICE_COLUMNS: [&str; 10] = ["ticker", "date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj", "lastupdated"];
/// The numeric price columns, in `PriceRow` order (for counting missing values).
const PRICE_NUMBERS: [&str; 7] = ["open", "high", "low", "close", "volume", "closeadj", "closeunadj"];

struct Prices {
    series: Vec<RawSeries>,
    rows: usize,
    skipped: usize,
    missing: [usize; 7],
    lastupdated_missing: usize,
    lastupdated_max: Option<i32>,
    columns: Vec<String>,
}

/// Reads `stocks` (every ticker) or `funds` (the benchmarks only).
fn read_prices(table: Table, path: &Path) -> Result<Prices> {
    let fund = table == Table::Funds;
    with_csv(path, |r| {
        let mut rdr = reader(r);
        let csv = Csv::new(table, path, &mut rdr)?;
        let ix = csv.cols(&PRICE_COLUMNS)?;
        let mut p = Prices { series: Vec::new(), rows: 0, skipped: 0, missing: [0; 7], lastupdated_missing: 0, lastupdated_max: None, columns: csv.names.clone() };
        let mut index: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut last: Option<(Vec<u8>, usize)> = None;
        let mut rec = csv::ByteRecord::new();
        while rdr.read_byte_record(&mut rec).map_err(|e| read_error(table, path, e))? {
            p.rows += 1;
            let t = rec[ix[0]].trim_ascii();
            if fund && !BENCHMARKS.iter().any(|b| b.as_bytes() == t) {
                p.skipped += 1;
                continue;
            }
            let date = csv.date(&rec, ix[1])?;
            let mut v = [f32::NAN; 7];
            for (k, &c) in ix[2..9].iter().enumerate() {
                match number(&rec[c]) {
                    Some(Some(x)) => v[k] = x,
                    Some(None) => p.missing[k] += 1,
                    None => return Err(csv.bad(&rec, c, "malformed number")),
                }
            }
            let updated = match csv.date_or_none(&rec, ix[9])? {
                Some(u) => {
                    p.lastupdated_max = p.lastupdated_max.max(Some(u));
                    u
                }
                None => {
                    p.lastupdated_missing += 1;
                    i32::MIN
                }
            };
            let id = match &last {
                Some((name, id)) if name.as_slice() == t => *id,
                _ => {
                    let id = match index.get(t) {
                        Some(&id) => id,
                        None => {
                            let ticker = std::str::from_utf8(t).ok().filter(|s| !s.is_empty() && !s.contains(['\t', '\n', '\r']));
                            let Some(ticker) = ticker else { return Err(csv.bad(&rec, ix[0], "unusable ticker")) };
                            index.insert(t.to_vec(), p.series.len());
                            p.series.push(RawSeries { ticker: ticker.to_string(), fund, rows: Vec::new() });
                            p.series.len() - 1
                        }
                    };
                    last = Some((t.to_vec(), id));
                    id
                }
            };
            p.series[id].rows.push(PriceRow {
                date,
                updated,
                open: v[0],
                high: v[1],
                low: v[2],
                close: v[3],
                volume: v[4],
                closeadj: v[5],
                closeunadj: v[6],
            });
        }
        Ok(p)
    })
}

/// A `tickers` row of table stocks or funds.
#[derive(Clone, Debug)]
pub struct TickerInfo {
    /// "stocks" or "funds" (SEP and SFP are read as these).
    pub table: String,
    pub permaticker: String,
    pub ticker: String,
    pub name: String,
    pub exchange: String,
    pub delisted: bool,
    pub category: String,
    pub first_price: Option<i32>,
    pub last_price: Option<i32>,
}

const TICKER_COLUMNS: [&str; 9] = ["table", "permaticker", "ticker", "name", "exchange", "isdelisted", "category", "firstpricedate", "lastpricedate"];

struct Tickers {
    rows: usize,
    kept: Vec<TickerInfo>,
    other_tables: BTreeMap<String, usize>,
    lastupdated_max: Option<i32>,
    columns: Vec<String>,
}

fn read_tickers(path: &Path) -> Result<Tickers> {
    with_csv(path, |r| {
        let mut rdr = reader(r);
        let csv = Csv::new(Table::Tickers, path, &mut rdr)?;
        let ix = csv.cols(&TICKER_COLUMNS)?;
        let updated = csv.names.iter().position(|n| n == "lastupdated");
        let mut out = Tickers { rows: 0, kept: Vec::new(), other_tables: BTreeMap::new(), lastupdated_max: None, columns: csv.names.clone() };
        let mut rec = csv::ByteRecord::new();
        while rdr.read_byte_record(&mut rec).map_err(|e| read_error(Table::Tickers, path, e))? {
            out.rows += 1;
            let table = match text(&rec[ix[0]]).to_ascii_lowercase().as_str() {
                "stocks" | "sep" => "stocks",
                "funds" | "sfp" => "funds",
                other => {
                    *out.other_tables.entry(other.to_string()).or_default() += 1;
                    continue;
                }
            };
            let delisted = match rec[ix[5]].trim_ascii() {
                b"Y" | b"y" => true,
                b"N" | b"n" => false,
                _ => return Err(csv.bad(&rec, ix[5], "isdelisted is not Y or N:")),
            };
            if let Some(c) = updated {
                out.lastupdated_max = out.lastupdated_max.max(csv.date_or_none(&rec, c)?);
            }
            out.kept.push(TickerInfo {
                table: table.to_string(),
                permaticker: text(&rec[ix[1]]),
                ticker: text(&rec[ix[2]]),
                name: text(&rec[ix[3]]),
                exchange: text(&rec[ix[4]]),
                delisted,
                category: text(&rec[ix[6]]),
                first_price: csv.date_or_none(&rec, ix[7])?,
                last_price: csv.date_or_none(&rec, ix[8])?,
            });
        }
        Ok(out)
    })
}

/// An `actions` row (dividends are only counted).
#[derive(Clone, Debug)]
pub struct ActionRow {
    pub date: i32,
    pub code: String,
    /// None when empty or N/A.
    pub contraticker: Option<String>,
}

const ACTION_COLUMNS: [&str; 4] = ["date", "action", "ticker", "contraticker"];

struct Actions {
    rows: usize,
    by_ticker: HashMap<String, Vec<ActionRow>>,
    codes: BTreeMap<String, usize>,
    without_ticker: usize,
    columns: Vec<String>,
}

fn read_actions(path: &Path) -> Result<Actions> {
    with_csv(path, |r| {
        let mut rdr = reader(r);
        let csv = Csv::new(Table::Actions, path, &mut rdr)?;
        let ix = csv.cols(&ACTION_COLUMNS)?;
        let mut out = Actions { rows: 0, by_ticker: HashMap::new(), codes: BTreeMap::new(), without_ticker: 0, columns: csv.names.clone() };
        let mut rec = csv::ByteRecord::new();
        while rdr.read_byte_record(&mut rec).map_err(|e| read_error(Table::Actions, path, e))? {
            out.rows += 1;
            let date = csv.date(&rec, ix[0])?;
            let code = text(&rec[ix[1]]).to_ascii_lowercase();
            if code.is_empty() {
                return Err(csv.bad(&rec, ix[1], "empty action code"));
            }
            *out.codes.entry(code.clone()).or_default() += 1;
            if code == "dividend" {
                continue;
            }
            let ticker = text(&rec[ix[2]]);
            if is_missing(ticker.as_bytes()) {
                out.without_ticker += 1;
                continue;
            }
            let contra = (!is_missing(&rec[ix[3]])).then(|| text(&rec[ix[3]]));
            out.by_ticker.entry(ticker).or_default().push(ActionRow { date, code, contraticker: contra });
        }
        for v in out.by_ticker.values_mut() {
            v.sort_by(|a, b| (a.date, &a.code).cmp(&(b.date, &b.code)));
        }
        Ok(out)
    })
}

/// Everything read from a raw directory.
pub struct Raw {
    pub dir: PathBuf,
    pub files: Vec<FileInfo>,
    series: Vec<RawSeries>,
    pub tickers: Vec<TickerInfo>,
    /// Non-dividend actions by ticker, sorted by date.
    pub actions: HashMap<String, Vec<ActionRow>>,
    /// Rows per action code, dividends included.
    pub codes: BTreeMap<String, usize>,
    /// Missing values by "table.column".
    pub missing: BTreeMap<String, usize>,
    /// Rows read but not used, by reason.
    pub skipped: BTreeMap<String, usize>,
}

/// Reads the four tables of `dir` (hashing each file while it is parsed).
pub fn read_raw(dir: &Path) -> Result<Raw> {
    let files = find_files(dir)?;
    let mut raw = Raw {
        dir: dir.to_path_buf(),
        files: Vec::new(),
        series: Vec::new(),
        tickers: Vec::new(),
        actions: HashMap::new(),
        codes: BTreeMap::new(),
        missing: BTreeMap::new(),
        skipped: BTreeMap::new(),
    };
    for (table, path) in files {
        let info = |rows: usize, columns: Vec<String>, lastupdated_max: Option<i32>, (bytes, sha256): (u64, String)| FileInfo {
            table,
            path: path.clone(),
            bytes,
            sha256,
            rows,
            columns,
            lastupdated_max,
        };
        match table {
            Table::Stocks | Table::Funds => {
                let (hash, p) = rayon::join(|| sha256_file(&path), || read_prices(table, &path));
                let p = p?;
                for (k, c) in PRICE_NUMBERS.iter().enumerate() {
                    if p.missing[k] > 0 {
                        raw.missing.insert(format!("{}.{c}", table.name()), p.missing[k]);
                    }
                }
                if p.lastupdated_missing > 0 {
                    raw.missing.insert(format!("{}.lastupdated", table.name()), p.lastupdated_missing);
                }
                if p.skipped > 0 {
                    raw.skipped.insert("funds rows of other funds".into(), p.skipped);
                }
                raw.files.push(info(p.rows, p.columns, p.lastupdated_max, hash?));
                raw.series.extend(p.series);
            }
            Table::Tickers => {
                let (hash, t) = rayon::join(|| sha256_file(&path), || read_tickers(&path));
                let t = t?;
                for (name, n) in &t.other_tables {
                    raw.skipped.insert(format!("tickers rows of table {name}"), *n);
                }
                raw.files.push(info(t.rows, t.columns, t.lastupdated_max, hash?));
                raw.tickers = t.kept;
            }
            Table::Actions => {
                let (hash, a) = rayon::join(|| sha256_file(&path), || read_actions(&path));
                let a = a?;
                if a.without_ticker > 0 {
                    raw.skipped.insert("actions rows without a ticker".into(), a.without_ticker);
                }
                raw.files.push(info(a.rows, a.columns, None, hash?));
                raw.actions = a.by_ticker;
                raw.codes = a.codes;
            }
        }
    }
    Ok(raw)
}

/// A delisted stock's last row and how it left.
#[derive(Clone, Debug)]
pub struct Delisting {
    pub last: i32,
    pub kind: DelistKind,
    /// Distinct codes (other than dividend) of its actions within WINDOW_DAYS of `last`, sorted.
    pub codes: Vec<String>,
    /// Both a buyout and a failure code (counted as a failure).
    pub conflict: bool,
}

/// A store symbol and where it came from.
#[derive(Clone, Debug)]
pub struct Symbol {
    pub ticker: String,
    pub kind: Kind,
    pub fund: bool,
    /// Index into `Raw::tickers` of its row (table stocks or funds), if any.
    pub meta: Option<usize>,
    pub delisting: Option<Delisting>,
}

/// The store to write: series sorted by ticker, their as-traded closes and their symbols.
pub struct Built {
    pub series: Vec<Series>,
    pub close_unadj: Vec<Vec<f32>>,
    pub symbols: Vec<Symbol>,
    /// Tickers with price rows that several permatickers share in `tickers` (table stocks): (ticker, permatickers).
    pub shared: Vec<(String, Vec<String>)>,
    pub survivors_only: Option<i32>,
    pub stats: BuildStats,
}

#[derive(Clone, Debug, Default)]
pub struct BuildStats {
    pub rows_without_closeadj: usize,
    pub duplicate_dates: usize,
    /// Up to 20 "TICKER YYYY-MM-DD" examples of duplicate dates.
    pub duplicate_examples: Vec<String>,
    /// Tickers whose rows all lacked a positive closeadj.
    pub tickers_without_rows: Vec<String>,
    /// Stock tickers without a `tickers` row (Kind::Other).
    pub without_metadata: Vec<String>,
    /// Stock tickers whose `tickers.lastpricedate` differs from their last kept row: (ticker, lastpricedate, last row).
    pub last_price_disagrees: Vec<(String, Option<i32>, i32)>,
    /// Symbols left out by --survivors-only, by kind (stock, other).
    pub survivors_dropped: [usize; 2],
    /// Delisted stocks (isdelisted = Y) of Kind::Other, which get no delist.tsv line.
    pub delisted_other: usize,
    /// Volumes set to 0 because they were missing (rows kept).
    pub volume_zeroed: usize,
}

/// Builds the store's series and symbols from what was read (moving the rows out of `raw`).
pub fn build(raw: &mut Raw, survivors_only: Option<i32>) -> Result<Built> {
    let mut by_ticker: HashMap<(&str, &str), Vec<usize>> = HashMap::new();
    for (i, t) in raw.tickers.iter().enumerate() {
        by_ticker.entry((t.table.as_str(), t.ticker.as_str())).or_default().push(i);
    }
    let mut st = BuildStats::default();
    let mut shared = Vec::new();
    let mut out: Vec<(Series, Vec<f32>, Symbol)> = Vec::with_capacity(raw.series.len());
    for rs in std::mem::take(&mut raw.series) {
        let RawSeries { ticker, fund, mut rows } = rs;
        // By date; of rows with one date, the latest lastupdated (then the latest line) is kept.
        rows.sort_by_key(|r| (r.date, r.updated));
        let mut kept: Vec<PriceRow> = Vec::with_capacity(rows.len());
        for r in rows {
            if let Some(prev) = kept.last_mut()
                && prev.date == r.date
            {
                st.duplicate_dates += 1;
                if st.duplicate_examples.len() < 20 {
                    st.duplicate_examples.push(format!("{ticker} {}", date_string(r.date)));
                }
                *prev = r;
                continue;
            }
            kept.push(r);
        }
        let before = kept.len();
        kept.retain(|r| r.closeadj.is_finite() && r.closeadj > 0.0);
        st.rows_without_closeadj += before - kept.len();
        if kept.is_empty() {
            st.tickers_without_rows.push(ticker);
            continue;
        }
        let (first, last) = (kept[0].date, kept[kept.len() - 1].date);

        let table = if fund { "funds" } else { "stocks" };
        let rows_meta = by_ticker.get(&(table, ticker.as_str())).cloned().unwrap_or_default();
        let permas: BTreeSet<&str> = rows_meta.iter().map(|&i| raw.tickers[i].permaticker.as_str()).collect();
        if !fund && permas.len() > 1 {
            shared.push((ticker.clone(), permas.iter().map(|s| s.to_string()).collect()));
        }
        let meta = rows_meta.first().copied();
        let info = meta.map(|i| &raw.tickers[i]);
        let kind = if fund {
            Kind::Etf
        } else if info.is_some_and(|m| m.category.to_ascii_lowercase().contains("common stock")) {
            Kind::Stock
        } else {
            Kind::Other
        };
        if !fund && info.is_none() {
            st.without_metadata.push(ticker.clone());
        }
        if !fund && let Some(m) = info
            && m.last_price != Some(last)
        {
            st.last_price_disagrees.push((ticker.clone(), m.last_price, last));
        }
        if !fund && survivors_only.is_some_and(|s| last < s) {
            st.survivors_dropped[usize::from(kind != Kind::Stock)] += 1;
            continue;
        }
        let delisting = match info {
            Some(m) if m.delisted && kind == Kind::Stock => {
                let codes: BTreeSet<&str> = raw
                    .actions
                    .get(&ticker)
                    .map(|v| v.iter().filter(|a| (a.date - last).abs() <= WINDOW_DAYS).map(|a| a.code.as_str()).collect())
                    .unwrap_or_default();
                let conflict = codes.iter().any(|c| FAILURE_CODES.contains(c)) && codes.iter().any(|c| BUYOUT_CODES.contains(c));
                Some(Delisting { last, kind: DelistKind::of(codes.iter().copied()), codes: codes.iter().map(|c| c.to_string()).collect(), conflict })
            }
            Some(m) if m.delisted && kind == Kind::Other => {
                st.delisted_other += 1;
                None
            }
            _ => None,
        };

        let n = kept.len();
        let mut s = Series {
            info: SymbolInfo {
                ticker: ticker.clone(),
                exchange: info.map(|m| m.exchange.to_ascii_lowercase()).unwrap_or_default(),
                kind,
                first,
                last,
                rows: n,
                source: format!("sharadar:{table}:{}", info.map_or("?", |m| m.permaticker.as_str())),
            },
            date: Vec::with_capacity(n),
            open: Vec::with_capacity(n),
            high: Vec::with_capacity(n),
            low: Vec::with_capacity(n),
            close: Vec::with_capacity(n),
            volume: Vec::with_capacity(n),
        };
        let mut unadj = Vec::with_capacity(n);
        for r in &kept {
            // open, high and low on closeadj's scale; NaN where close is missing.
            let scale = if r.close.is_finite() && r.close > 0.0 { r.closeadj / r.close } else { f32::NAN };
            s.date.push(r.date);
            s.open.push(r.open * scale);
            s.high.push(r.high * scale);
            s.low.push(r.low * scale);
            s.close.push(r.closeadj);
            if r.volume.is_finite() {
                s.volume.push(r.volume);
            } else {
                st.volume_zeroed += 1;
                s.volume.push(0.0);
            }
            unadj.push(if r.closeunadj > 0.0 { r.closeunadj } else { f32::NAN });
        }
        out.push((s, unadj, Symbol { ticker, kind, fund, meta, delisting }));
    }
    out.sort_by(|a, b| a.2.ticker.cmp(&b.2.ticker));
    if let Some(w) = out.windows(2).find(|w| w[0].2.ticker == w[1].2.ticker) {
        bail!("ticker {} is in both stocks and funds", w[0].2.ticker);
    }
    shared.sort();
    let (mut series, mut close_unadj, mut symbols) = (Vec::with_capacity(out.len()), Vec::with_capacity(out.len()), Vec::with_capacity(out.len()));
    for (s, u, m) in out {
        series.push(s);
        close_unadj.push(u);
        symbols.push(m);
    }
    Ok(Built { series, close_unadj, symbols, shared, survivors_only, stats: st })
}

/// Action codes that are not in KNOWN_CODES, with their row counts.
pub fn unknown_codes(raw: &Raw) -> BTreeMap<String, usize> {
    raw.codes.iter().filter(|(c, _)| !KNOWN_CODES.contains(&c.as_str())).map(|(c, n)| (c.clone(), *n)).collect()
}

/// Delisted stocks by kind, and those with both buyout and failure codes.
pub fn delist_counts(b: &Built) -> ([usize; 3], usize) {
    let mut n = [0usize; 3];
    let mut conflicts = 0;
    for d in b.symbols.iter().filter_map(|s| s.delisting.as_ref()) {
        n[d.kind.code() as usize - 1] += 1;
        conflicts += usize::from(d.conflict);
    }
    (n, conflicts)
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn opt_date(d: Option<i32>) -> String {
    d.map(date_string).unwrap_or_else(|| "-".into())
}

/// The import log: what was read, kept, dropped and decided.
pub fn log(raw: &Raw, b: &Built) -> Vec<String> {
    let mut out = Vec::new();
    for f in &raw.files {
        out.push(format!(
            "read {} ({}): {} bytes, sha256 {}, {} rows{}",
            f.table.name(),
            file_name(&f.path),
            f.bytes,
            f.sha256,
            f.rows,
            f.lastupdated_max.map(|d| format!(", lastupdated up to {}", date_string(d))).unwrap_or_default()
        ));
    }
    for (why, n) in &raw.skipped {
        out.push(format!("  not used: {n} {why}"));
    }
    if !raw.missing.is_empty() {
        out.push(format!("missing values: {}", raw.missing.iter().map(|(c, n)| format!("{c} {n}")).collect::<Vec<_>>().join(", ")));
    }
    let count = |k: Kind| b.symbols.iter().filter(|s| s.kind == k).count();
    let rows: usize = b.series.iter().map(|s| s.date.len()).sum();
    let (first, last) = (b.series.iter().map(|s| s.info.first).min(), b.series.iter().map(|s| s.info.last).max());
    out.push(format!(
        "symbols: {} common stocks (Kind::Stock), {} other (preferred, warrants, units, rights, no tickers row ...), {} funds ({}); {rows} rows, {} .. {}",
        count(Kind::Stock),
        count(Kind::Other),
        count(Kind::Etf),
        b.symbols.iter().filter(|s| s.fund).map(|s| s.ticker.as_str()).collect::<Vec<_>>().join(", "),
        opt_date(first),
        opt_date(last)
    ));
    let missing_bench: Vec<&str> = BENCHMARKS.iter().copied().filter(|t| !b.symbols.iter().any(|s| s.fund && s.ticker == *t)).collect();
    if !missing_bench.is_empty() {
        out.push(format!("WARNING: no {} in funds: the grader will have no such benchmark", missing_bench.join(" or ")));
    }
    let cal: Vec<&str> = CALENDAR.iter().copied().filter(|t| b.symbols.iter().any(|s| !s.fund && s.ticker == *t)).collect();
    out.push(format!("calendar stocks: {} of 8 ({}){}", cal.len(), cal.join(", "), if cal.len() < 5 { "; WARNING: correlation2 needs at least 5" } else { "" }));
    let st = &b.stats;
    out.push(format!(
        "rows dropped: {} without a positive closeadj, {} duplicate dates (the latest lastupdated kept){}; {} missing volumes set to 0",
        st.rows_without_closeadj,
        st.duplicate_dates,
        if st.duplicate_examples.is_empty() { String::new() } else { format!(" e.g. {}", st.duplicate_examples.iter().take(5).cloned().collect::<Vec<_>>().join(", ")) },
        st.volume_zeroed
    ));
    if !st.tickers_without_rows.is_empty() {
        out.push(format!("tickers with no usable row (left out): {} ({})", st.tickers_without_rows.len(), head(&st.tickers_without_rows, 20)));
    }
    if !st.without_metadata.is_empty() {
        out.push(format!("stocks tickers without a tickers row (Kind::Other): {} ({})", st.without_metadata.len(), head(&st.without_metadata, 20)));
    }
    if !st.last_price_disagrees.is_empty() {
        out.push(format!(
            "tickers.lastpricedate differs from the last stocks row for {} tickers (e.g. {})",
            st.last_price_disagrees.len(),
            st.last_price_disagrees.iter().take(5).map(|(t, a, b)| format!("{t} {} vs {}", opt_date(*a), date_string(*b))).collect::<Vec<_>>().join(", ")
        ));
    }
    if let Some(s) = b.survivors_only {
        out.push(format!(
            "survivors only (last row on or after {}): left out {} common stocks and {} other symbols",
            date_string(s),
            st.survivors_dropped[0],
            st.survivors_dropped[1]
        ));
    }
    let (n, conflicts) = delist_counts(b);
    out.push(format!(
        "delisted common stocks: {} buyout, {} failure, {} unknown ({conflicts} with both buyout and failure codes, counted as failures); {} delisted other symbols get no line",
        n[0], n[1], n[2], st.delisted_other
    ));
    out.push(format!("action codes: {}", raw.codes.iter().map(|(c, n)| format!("{c} {n}")).collect::<Vec<_>>().join(", ")));
    let unknown = unknown_codes(raw);
    out.push(if unknown.is_empty() {
        "unknown action codes: none".into()
    } else {
        format!("unknown action codes: {}", unknown.iter().map(|(c, n)| format!("{c} {n}")).collect::<Vec<_>>().join(", "))
    });
    if !b.shared.is_empty() {
        out.push(format!(
            "ERROR: {} tickers are shared by two or more permatickers in tickers (table stocks): {}",
            b.shared.len(),
            b.shared.iter().take(20).map(|(t, p)| format!("{t} ({})", p.join(", "))).collect::<Vec<_>>().join("; ")
        ));
    }
    out
}

fn head(v: &[String], k: usize) -> String {
    let s = v.iter().take(k).cloned().collect::<Vec<_>>().join(", ");
    if v.len() > k { format!("{s}, ...") } else { s }
}

/// What an import wrote.
pub struct Imported {
    pub log: Vec<String>,
    /// What source.json holds. (Read by the tests.)
    #[cfg_attr(not(test), allow(dead_code))]
    pub source: Value,
}

/// Reads `raw_dir`, builds the store and writes it with its sidecars into `out`. A directory holding a price store
/// without any of the sidecars (correlation2's own) is refused; an earlier or interrupted import is replaced.
pub fn import(raw_dir: &Path, out: &Path, survivors_only: Option<i32>) -> Result<Imported> {
    if out.join("prices.bin").exists() && !SIDECARS.iter().any(|f| out.join(f).exists()) {
        bail!(
            "{} holds a price store this importer did not write (no {}; correlation2's own store?). Use another --out",
            out.display(),
            SIDECARS.join(", ")
        );
    }
    let mut raw = read_raw(raw_dir)?;
    let built = build(&mut raw, survivors_only)?;
    let mut log = log(&raw, &built);
    if !built.shared.is_empty() {
        bail!("{}\nthe import stops here: a ticker shared by two permatickers is reported, not guessed around (docs/sharadar.md)", log.join("\n"));
    }
    let source = write_store(out, &raw, &built)?;
    log.push(format!(
        "wrote {}: {} symbols, {} rows; {} delist.tsv lines",
        out.display(),
        built.series.len(),
        built.series.iter().map(|s| s.date.len()).sum::<usize>(),
        built.symbols.iter().filter(|s| s.delisting.is_some()).count()
    ));
    Ok(Imported { log, source })
}

/// Writes the store and its sidecars. `source.json` is removed first and written last, so a store without it is an
/// incomplete import (the export and the grader refuse it).
fn write_store(out: &Path, raw: &Raw, b: &Built) -> Result<Value> {
    std::fs::create_dir_all(out)?;
    let source_path = out.join("source.json");
    if source_path.exists() {
        std::fs::remove_file(&source_path)?;
    }
    npy::write_atomic(&out.join("close_unadj.f32"), |w| {
        let mut buf = Vec::with_capacity(1 << 20);
        for u in &b.close_unadj {
            for chunk in u.chunks(1 << 18) {
                buf.clear();
                for x in chunk {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
                w.write_all(&buf)?;
            }
        }
        Ok(())
    })?;
    npy::write_atomic(&out.join("delist.tsv"), |w| {
        writeln!(w, "ticker\tlast_date\tkind\tactions")?;
        for s in &b.symbols {
            if let Some(d) = &s.delisting {
                writeln!(w, "{}\t{}\t{}\t{}", s.ticker, date_string(d.last), d.kind.name(), d.codes.join(","))?;
            }
        }
        Ok(())
    })?;
    c2_data::store::write(out, &b.series)?;

    let symbols_tsv = std::fs::read(out.join("symbols.tsv"))?;
    let rows: usize = b.series.iter().map(|s| s.date.len()).sum();
    let count = |k: Kind| b.symbols.iter().filter(|s| s.kind == k).count();
    let (n, conflicts) = delist_counts(b);
    let st = &b.stats;
    let source = json!({
        "format": SOURCE_FORMAT,
        "importer": IMPORTER,
        "raw_dir": raw.dir.display().to_string(),
        "files": raw.files.iter().map(|f| json!({
            "table": f.table.name(),
            "file": file_name(&f.path),
            "bytes": f.bytes,
            "sha256": f.sha256,
            "rows": f.rows,
            "columns": f.columns,
            "lastupdated_max": f.lastupdated_max.map(date_string),
        })).collect::<Vec<_>>(),
        "lastupdated_max": raw.files.iter().filter_map(|f| f.lastupdated_max).max().map(date_string),
        "first_date": b.series.iter().map(|s| s.info.first).min().map(date_string),
        "last_date": b.series.iter().map(|s| s.info.last).max().map(date_string),
        "survivors_only": b.survivors_only.map(date_string),
        "store": {
            "symbols": b.series.len(),
            "rows": rows,
            "stocks": count(Kind::Stock),
            "other": count(Kind::Other),
            "funds": b.symbols.iter().filter(|s| s.fund).map(|s| s.ticker.clone()).collect::<Vec<_>>(),
            "symbols_tsv_fnv64": format!("{:016x}", crate::export::fnv(&symbols_tsv)),
        },
        "delisted": {"buyout": n[0], "failure": n[1], "unknown": n[2], "with_buyout_and_failure_codes": conflicts, "delisted_other_symbols": st.delisted_other},
        "dropped": {
            "rows_without_closeadj": st.rows_without_closeadj,
            "duplicate_dates": st.duplicate_dates,
            "duplicate_examples": st.duplicate_examples,
            "tickers_without_rows": st.tickers_without_rows.len(),
            "survivors_only": {"stocks": st.survivors_dropped[0], "other": st.survivors_dropped[1]},
        },
        "missing_values": raw.missing,
        "volumes_set_to_0": st.volume_zeroed,
        "not_used": raw.skipped,
        "action_codes": raw.codes,
        "unknown_action_codes": unknown_codes(raw),
        "tickers_without_metadata": {"count": st.without_metadata.len(), "examples": st.without_metadata.iter().take(50).collect::<Vec<_>>()},
        "lastpricedate_disagrees": {
            "count": st.last_price_disagrees.len(),
            "examples": st.last_price_disagrees.iter().take(50).map(|(t, a, b)| json!([t, a.map(date_string), date_string(*b)])).collect::<Vec<_>>(),
        },
        "calendar_stocks": CALENDAR.iter().filter(|t| b.symbols.iter().any(|s| !s.fund && s.ticker == **t)).collect::<Vec<_>>(),
        "rules": "docs/sharadar.md: close = closeadj, open/high/low scaled by closeadj/close, volume split-adjusted (missing: 0); \
Kind::Stock = tickers.category contains 'Common Stock'; funds: SPY and QQQ; duplicate dates: latest lastupdated; \
delisting kind from the stock's actions within 30 calendar days of its last row: failure > buyout > unknown",
        "created_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),
    });
    npy::write_atomic(&source_path, |w| {
        w.write_all(serde_json::to_string_pretty(&source)?.as_bytes())?;
        w.write_all(b"\n")?;
        Ok(())
    })?;
    Ok(source)
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Weekdays (Monday to Friday) from `from` to `to`, inclusive.
    pub fn weekdays(from: i32, to: i32) -> Vec<i32> {
        // 1970-01-01 was a Thursday: (day + 3) % 7 == 0 is a Monday.
        (from..=to).filter(|d| (d + 3).rem_euclid(7) < 5).collect()
    }

    /// A synthetic stock for the raw CSVs.
    pub struct Fake {
        pub ticker: &'static str,
        pub category: &'static str,
        pub delisted: bool,
        pub first: i32,
        pub last: i32,
        /// as-traded price = adjusted price x this (a later reverse split makes it < 1).
        pub unadj: f32,
        pub base: f64,
        pub volume: f64,
    }

    fn day(s: &str) -> i32 {
        c2_data::parse_date(s).unwrap()
    }

    pub fn fake(ticker: &'static str, category: &'static str, delisted: bool, first: &str, last: &str) -> Fake {
        Fake { ticker, category, delisted, first: day(first), last: day(last), unadj: 1.0, base: 40.0, volume: 1e6 }
    }

    /// Deterministic prices: a slow wave per stock, positive, with dividends making closeadj a bit lower than close.
    pub fn price(f: &Fake, d: i32) -> (f64, f64) {
        let k = f.ticker.bytes().map(u64::from).sum::<u64>() as f64;
        let close = f.base * (1.0 + 0.3 * ((d as f64) / (20.0 + k % 13.0)).sin()) * (1.0 + 0.0002 * (d - f.first) as f64);
        (close, close * 0.97)
    }

    /// The default synthetic universe: five calendar stocks over 2003..2009 and the special cases of the tests.
    pub fn universe() -> Vec<Fake> {
        let mut v: Vec<Fake> = ["IBM", "KO", "JNJ", "PG", "MMM"].iter().map(|t| fake(t, "Domestic Common Stock", false, "2003-01-02", "2009-12-31")).collect();
        v.push(fake("BUYCO", "Domestic Common Stock", true, "2003-01-02", "2007-03-15"));
        v.push(fake("BANKCO", "Domestic Common Stock Primary Class", true, "2003-01-02", "2008-09-12"));
        v.push(fake("GONECO", "Domestic Common Stock", true, "2003-01-02", "2006-05-01"));
        v.push(fake("BOTHCO", "Domestic Common Stock", true, "2003-01-02", "2008-01-15"));
        v.push(fake("OLDCO", "Domestic Common Stock", true, "2003-01-02", "2007-06-29"));
        v.push(fake("ADRCO", "ADR Common Stock", false, "2003-01-02", "2009-12-31"));
        v.push(fake("PREF", "Domestic Preferred Stock", false, "2003-01-02", "2009-12-31"));
        v.push(fake("WARR", "Domestic Warrant", true, "2004-01-02", "2006-12-29"));
        v.push(Fake { unadj: 0.1, ..fake("RVS", "Domestic Common Stock", false, "2003-01-02", "2009-12-31") });
        v.push(fake("NOMETA", "", false, "2003-01-02", "2009-12-31"));
        v
    }

    /// Writes stocks.csv, funds.csv, tickers.csv and actions.csv for `stocks` into `dir`, rows in a scrambled order,
    /// with two duplicate dates (IBM's later lastupdated has another close; KO's later line has no closeadj).
    pub fn write_raw(dir: &Path, stocks: &[Fake]) {
        std::fs::create_dir_all(dir).unwrap();
        let mut lines = Vec::new();
        for f in stocks {
            for d in weekdays(f.first, f.last) {
                let (c, adj) = price(f, d);
                let u = c * f.unadj as f64;
                lines.push(format!(
                    "{},{},{:.4},{:.4},{:.4},{:.4},{},{:.4},{:.4},2026-09-01",
                    f.ticker,
                    date_string(d),
                    c * 0.99,
                    c * 1.01,
                    c * 0.98,
                    c,
                    f.volume,
                    adj,
                    u
                ));
            }
        }
        // A scrambled order (by a hash of the line), as a bulk file need not be sorted.
        let mut mixed = lines;
        mixed.sort_by_cached_key(|l| crate::export::fnv(l.as_bytes()));
        // IBM 2005-03-01 again, updated later, with another close; and KO 2005-03-02 again (same lastupdated, so the
        // later line wins) without closeadj, which drops the date.
        mixed.push("IBM,2005-03-01,1,1,1,77.7,1000000,75.0,77.7,2026-10-01".into());
        mixed.push("KO,2005-03-02,1,1,1,1,1000000,,1,2026-09-01".into());
        let mut s = String::from("ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated\n");
        s += &mixed.join("\n");
        std::fs::write(dir.join("stocks.csv"), s).unwrap();

        let spy = Fake { base: 120.0, ..fake("SPY", "ETF", false, "2003-01-02", "2009-12-31") };
        let iwm = fake("IWM", "ETF", false, "2003-01-02", "2009-12-31");
        let mut s = String::from("ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated\n");
        for f in [&spy, &iwm] {
            for d in weekdays(f.first, f.last) {
                let (c, adj) = price(f, d);
                s += &format!("{},{},{c},{c},{c},{c},5000000,{adj},{c},2026-09-01\n", f.ticker, date_string(d));
            }
        }
        std::fs::write(dir.join("funds.csv"), s).unwrap();

        let mut s = String::from(
            "table,permaticker,ticker,name,exchange,isdelisted,category,cusips,siccode,lastupdated,firstadded,firstpricedate,lastpricedate\n",
        );
        for (i, f) in stocks.iter().enumerate() {
            if f.ticker == "NOMETA" {
                continue;
            }
            let row = |table: &str| {
                format!(
                    "{table},{},{},\"{} INC, THE\",NYSE,{},{},123 456,3571,2026-09-01,2014-09-24,{},{}\n",
                    100 + i,
                    f.ticker,
                    f.ticker,
                    if f.delisted { "Y" } else { "N" },
                    f.category,
                    date_string(f.first),
                    date_string(f.last)
                )
            };
            s += &row("stocks");
            s += &row("fundamentals");
        }
        s += "funds,900,SPY,SPDR S&P 500 ETF TRUST,NYSEARCA,N,ETF,,,2026-09-01,2018-06-05,2003-01-02,2009-12-31\n";
        std::fs::write(dir.join("tickers.csv"), s).unwrap();

        let actions = "date,action,ticker,name,value,contraticker,contraname\n\
2007-03-16,acquisitionby,BUYCO,BUYCO INC,1.0,IBM,INTERNATIONAL BUSINESS MACHINES\n\
2007-03-16,delisted,BUYCO,BUYCO INC,,N/A,N/A\n\
2008-09-20,bankruptcyliquidation,BANKCO,BANKCO INC,,N/A,N/A\n\
2008-08-01,dividend,BANKCO,BANKCO INC,0.1,N/A,N/A\n\
2006-05-02,voluntarydelisting,GONECO,GONECO INC,,N/A,N/A\n\
2006-05-03,weirdcode,GONECO,GONECO INC,,N/A,N/A\n\
2008-01-20,acquisitionby,BOTHCO,BOTHCO INC,,KO,COCA COLA CO\n\
2008-01-10,regulatorydelisting,BOTHCO,BOTHCO INC,,N/A,N/A\n\
2007-04-30,mergerto,OLDCO,OLDCO INC,,PG,PROCTER & GAMBLE\n\
2005-06-01,split,RVS,RVS INC,0.1,N/A,N/A\n\
2006-12-29,delisted,WARR,WARR INC,,N/A,N/A\n";
        std::fs::write(dir.join("actions.csv"), actions).unwrap();
    }

    pub fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("stockai-judge-sharadar-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        std::fs::read(path).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    }

    #[test]
    fn strict_dates_and_numbers() {
        assert_eq!(strict_date(b"2024-06-28"), Some(days_from_civil(2024, 6, 28)));
        assert_eq!(strict_date(b"2000-02-29"), Some(days_from_civil(2000, 2, 29)));
        for bad in ["2023-02-29", "2024-13-01", "2024-06-31", "2024-6-28", "20240628", "2024-06-28T00:00", "", "2024/06/28", "2024-0a-28"] {
            assert_eq!(strict_date(bad.as_bytes()), None, "{bad}");
        }
        assert_eq!(number(b"22.304"), Some(Some(22.304)));
        assert_eq!(number(b" 1e3 "), Some(Some(1000.0)));
        for m in ["", "N/A", "na", "NaN", "null", "None", "  "] {
            assert_eq!(number(m.as_bytes()), Some(None), "{m:?}");
        }
        for bad in ["abc", "1.2.3", "inf", "12,5"] {
            assert_eq!(number(bad.as_bytes()), None, "{bad}");
        }
    }

    #[test]
    fn finds_files_by_table_name() {
        let dir = scratch("find");
        for f in ["stocks.zip", "SHARADAR_SFP_2026.csv.gz", "tickers.csv", "actions.csv", "stocks.status.json", ".funds.part", "manifest.tsv"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let found = find_files(&dir).unwrap();
        let names: Vec<(Table, String)> = found.iter().map(|(t, p)| (*t, file_name(p))).collect();
        assert_eq!(
            names,
            vec![
                (Table::Stocks, "stocks.zip".into()),
                (Table::Funds, "SHARADAR_SFP_2026.csv.gz".into()),
                (Table::Tickers, "tickers.csv".into()),
                (Table::Actions, "actions.csv".into())
            ]
        );
        std::fs::write(dir.join("stocks.csv"), b"x").unwrap();
        let e = find_files(&dir).unwrap_err().to_string();
        assert!(e.contains("2 files could be the stocks table"), "{e}");
        std::fs::remove_file(dir.join("stocks.csv")).unwrap();
        std::fs::remove_file(dir.join("actions.csv")).unwrap();
        let e = find_files(&dir).unwrap_err().to_string();
        assert!(e.contains("no file for the actions table"), "{e}");
    }

    #[test]
    fn imports_kinds_delistings_and_sidecars() {
        let dir = scratch("import");
        let stocks = universe();
        write_raw(&dir.join("raw"), &stocks);
        let store_dir = dir.join("store");
        let imported = import(&dir.join("raw"), &store_dir, None).unwrap();
        let store = c2_data::Store::open(&store_dir).unwrap();

        // Symbols: sorted by ticker; kinds from the category; SPY from funds, IWM not kept.
        let tickers: Vec<&str> = store.symbols.iter().map(|s| s.ticker.as_str()).collect();
        let mut want: Vec<&str> = stocks.iter().map(|f| f.ticker).chain(["SPY"]).collect();
        want.sort();
        assert_eq!(tickers, want);
        let kind = |t: &str| store.symbols[store.id(t).unwrap()].kind;
        for t in ["IBM", "BUYCO", "BANKCO", "ADRCO", "RVS"] {
            assert_eq!(kind(t), Kind::Stock, "{t}");
        }
        for t in ["PREF", "WARR", "NOMETA"] {
            assert_eq!(kind(t), Kind::Other, "{t}");
        }
        assert_eq!(kind("SPY"), Kind::Etf);
        assert_eq!(store.symbols[store.id("IBM").unwrap()].source, "sharadar:stocks:100");
        assert_eq!(store.symbols[store.id("SPY").unwrap()].source, "sharadar:funds:900");

        // Rows sorted by date without duplicates; the duplicate kept the later lastupdated; KO's row without closeadj
        // is gone; close = closeadj, open scaled by closeadj / close.
        for id in 0..store.symbols.len() {
            assert!(store.dates(id).windows(2).all(|w| w[0] < w[1]), "{}", store.symbols[id].ticker);
        }
        let ibm = store.id("IBM").unwrap();
        let k = store.dates(ibm).binary_search(&day("2005-03-01")).unwrap();
        assert_eq!(store.close(ibm)[k], 75.0);
        assert!((store.open_(ibm)[k] - 75.0 / 77.7).abs() < 1e-6);
        let ko = store.id("KO").unwrap();
        assert!(store.dates(ko).binary_search(&day("2005-03-02")).is_err());
        let f = &stocks.iter().find(|f| f.ticker == "KO").unwrap();
        let k = store.dates(ko).binary_search(&day("2005-03-03")).unwrap();
        let (_, adj) = price(f, day("2005-03-03"));
        assert!((store.close(ko)[k] as f64 - adj).abs() < 1e-3 && (store.open_(ko)[k] as f64 - 0.99 * adj).abs() < 1e-3);
        assert_eq!(store.volume(ko)[k], 1e6);

        // close_unadj.f32: one value per store row, in prices.bin row order.
        let unadj = read_f32(&store_dir.join("close_unadj.f32"));
        assert_eq!(unadj.len(), store.rows());
        let mut off = 0;
        for id in 0..store.symbols.len() {
            let t = store.symbols[id].ticker.as_str();
            let f = stocks.iter().find(|f| f.ticker == t);
            for (k, &d) in store.dates(id).iter().enumerate() {
                let want = match f {
                    Some(f) => {
                        if t == "IBM" && d == day("2005-03-01") {
                            77.7
                        } else {
                            let (c, _) = price(f, d);
                            (c * f.unadj as f64) as f32
                        }
                    }
                    None => price(&Fake { base: 120.0, ..fake("SPY", "ETF", false, "2003-01-02", "2009-12-31") }, d).0 as f32,
                };
                assert!((unadj[off + k] - want).abs() <= 1e-3 * want.abs().max(1.0), "{t} {}: {} vs {want}", date_string(d), unadj[off + k]);
            }
            off += store.dates(id).len();
        }
        let rvs = store.id("RVS").unwrap();
        let rvs_off: usize = (0..rvs).map(|i| store.dates(i).len()).sum();
        assert!(unadj[rvs_off] < 0.11 * store.close(rvs)[0] / 0.97 + 1e-3, "a reverse split: as traded far below adjusted");

        // delist.tsv: delisted common stocks only, kinds from actions within 30 days of the last row.
        let tsv = std::fs::read_to_string(store_dir.join("delist.tsv")).unwrap();
        let lines: Vec<&str> = tsv.lines().collect();
        assert_eq!(lines[0], "ticker\tlast_date\tkind\tactions");
        assert_eq!(
            lines[1..],
            [
                "BANKCO\t2008-09-12\tfailure\tbankruptcyliquidation",
                "BOTHCO\t2008-01-15\tfailure\tacquisitionby,regulatorydelisting",
                "BUYCO\t2007-03-15\tbuyout\tacquisitionby,delisted",
                "GONECO\t2006-05-01\tunknown\tvoluntarydelisting,weirdcode",
                "OLDCO\t2007-06-29\tunknown\t",
            ]
        );

        // source.json: files with sizes and hashes, the latest lastupdated, unknown codes, counts.
        let src: Value = serde_json::from_str(&std::fs::read_to_string(store_dir.join("source.json")).unwrap()).unwrap();
        assert_eq!(src, imported.source);
        assert_eq!(src["importer"], IMPORTER);
        assert_eq!(src["files"].as_array().unwrap().len(), 4);
        let (bytes, sha) = sha256_file(&dir.join("raw/stocks.csv")).unwrap();
        assert_eq!(src["files"][0]["sha256"], sha.as_str());
        assert_eq!(src["files"][0]["bytes"], bytes);
        assert_eq!(src["lastupdated_max"], "2026-10-01");
        assert_eq!(src["unknown_action_codes"], json!({"weirdcode": 1}));
        assert_eq!(src["delisted"], json!({"buyout": 1, "failure": 2, "unknown": 2, "with_buyout_and_failure_codes": 1, "delisted_other_symbols": 1}));
        assert_eq!(src["dropped"]["duplicate_dates"], 2);
        assert_eq!(src["dropped"]["rows_without_closeadj"], 1);
        assert_eq!(src["tickers_without_metadata"]["examples"], json!(["NOMETA"]));
        assert_eq!(src["store"]["rows"], store.rows());
        assert_eq!(src["not_used"]["funds rows of other funds"], weekdays(day("2003-01-02"), day("2009-12-31")).len());
        let log = imported.log.join("\n");
        assert!(log.contains("unknown action codes: weirdcode 1"), "{log}");
        assert!(log.contains("calendar stocks: 5 of 8"), "{log}");

        // A second import over the same store is fine, and so is one over an interrupted import (no source.json);
        // correlation2's own kind of store (no sidecars at all) is refused.
        import(&dir.join("raw"), &store_dir, None).unwrap();
        std::fs::remove_file(store_dir.join("source.json")).unwrap();
        import(&dir.join("raw"), &store_dir, None).unwrap();
        for f in SIDECARS {
            std::fs::remove_file(store_dir.join(f)).unwrap();
        }
        let e = import(&dir.join("raw"), &store_dir, None).err().unwrap().to_string();
        assert!(e.contains("did not write"), "{e}");
    }

    #[test]
    fn survivors_only_keeps_symbols_alive_at_the_cutoff() {
        let dir = scratch("survivors");
        let stocks = universe();
        write_raw(&dir.join("raw"), &stocks);
        let imported = import(&dir.join("raw"), &dir.join("store"), Some(day("2008-01-15"))).unwrap();
        let store = c2_data::Store::open(&dir.join("store")).unwrap();
        let tickers: Vec<&str> = store.symbols.iter().map(|s| s.ticker.as_str()).collect();
        // BOTHCO's last row is the cutoff itself (kept); BUYCO, GONECO, OLDCO and WARR end before it.
        for gone in ["BUYCO", "GONECO", "OLDCO", "WARR"] {
            assert!(!tickers.contains(&gone), "{gone}");
        }
        for kept in ["BOTHCO", "BANKCO", "IBM", "SPY", "PREF"] {
            assert!(tickers.contains(&kept), "{kept}");
        }
        assert_eq!(imported.source["survivors_only"], "2008-01-15");
        assert_eq!(imported.source["dropped"]["survivors_only"], json!({"stocks": 3, "other": 1}));
        let tsv = std::fs::read_to_string(dir.join("store/delist.tsv")).unwrap();
        assert_eq!(tsv.lines().skip(1).map(|l| l.split('\t').next().unwrap()).collect::<Vec<_>>(), ["BANKCO", "BOTHCO"]);
        let unadj = read_f32(&dir.join("store/close_unadj.f32"));
        assert_eq!(unadj.len(), store.rows());
    }

    #[test]
    fn shared_tickers_missing_columns_and_bad_values_stop_the_import() {
        let dir = scratch("errors");
        let stocks = universe();
        let raw = dir.join("raw");
        write_raw(&raw, &stocks);
        let tickers = std::fs::read_to_string(raw.join("tickers.csv")).unwrap();

        // Two permatickers for IBM in table stocks.
        let extra = "stocks,555,IBM,OTHER IBM,NYSE,Y,Domestic Common Stock,,,2026-09-01,2014-09-24,1990-01-02,1995-01-02\n";
        std::fs::write(raw.join("tickers.csv"), format!("{tickers}{extra}")).unwrap();
        let e = import(&raw, &dir.join("store"), None).err().unwrap().to_string();
        assert!(e.contains("shared by two or more permatickers") && e.contains("IBM (100, 555)"), "{e}");
        assert!(!dir.join("store/prices.bin").exists());
        // The same ticker under another table (fundamentals) is not a problem.
        let extra = "fundamentals,555,IBM,OTHER IBM,NYSE,Y,Domestic Common Stock,,,2026-09-01,2014-09-24,1990-01-02,1995-01-02\n";
        std::fs::write(raw.join("tickers.csv"), format!("{tickers}{extra}")).unwrap();
        import(&raw, &dir.join("store"), None).unwrap();
        std::fs::write(raw.join("tickers.csv"), &tickers).unwrap();

        let stocks_csv = std::fs::read_to_string(raw.join("stocks.csv")).unwrap();
        let cases = [
            (stocks_csv.replacen("closeunadj", "close_unadj", 1), "no column \"closeunadj\""),
            (format!("{stocks_csv}\nIBM,2005-13-01,1,1,1,1,1,1,1,2026-09-01"), "malformed date \"2005-13-01\""),
            (format!("{stocks_csv}\nIBM,2005-03-04,1,1,1,abc,1,1,1,2026-09-01"), "malformed number \"abc\" in column \"close\""),
            (format!("{stocks_csv}\nIBM,2005-03-04,1,1,1"), "unreadable CSV"),
        ];
        for (text, want) in cases {
            std::fs::write(raw.join("stocks.csv"), text).unwrap();
            let e = format!("{:#}", import(&raw, &dir.join("store2"), None).err().unwrap());
            assert!(e.contains(want), "{want}: {e}");
        }
        std::fs::write(raw.join("stocks.csv"), &stocks_csv).unwrap();
        let actions = std::fs::read_to_string(raw.join("actions.csv")).unwrap();
        std::fs::write(raw.join("actions.csv"), actions.replace("2008-09-20", "2008-09-2")).unwrap();
        let e = import(&raw, &dir.join("store2"), None).err().unwrap().to_string();
        assert!(e.contains("actions table") && e.contains("malformed date \"2008-09-2\""), "{e}");
    }

    #[test]
    fn gzip_and_zip_files_import_the_same_store() {
        let dir = scratch("compressed");
        write_raw(&dir.join("plain"), &universe());
        import(&dir.join("plain"), &dir.join("store_plain"), None).unwrap();
        let packed = dir.join("packed");
        std::fs::create_dir_all(&packed).unwrap();
        for t in ["stocks", "funds"] {
            let mut gz = flate2::write::GzEncoder::new(File::create(packed.join(format!("{t}.csv.gz"))).unwrap(), flate2::Compression::fast());
            gz.write_all(&std::fs::read(dir.join(format!("plain/{t}.csv"))).unwrap()).unwrap();
            gz.finish().unwrap();
        }
        for t in ["tickers", "actions"] {
            let mut z = zip::ZipWriter::new(File::create(packed.join(format!("SHARADAR_{}.zip", t.to_uppercase()))).unwrap());
            z.start_file(format!("SHARADAR_{t}_2026.csv"), zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated))
                .unwrap();
            z.write_all(&std::fs::read(dir.join(format!("plain/{t}.csv"))).unwrap()).unwrap();
            z.finish().unwrap();
        }
        let imported = import(&packed, &dir.join("store_packed"), None).unwrap();
        for f in ["prices.bin", "symbols.tsv", "close_unadj.f32", "delist.tsv"] {
            assert_eq!(std::fs::read(dir.join("store_plain").join(f)).unwrap(), std::fs::read(dir.join("store_packed").join(f)).unwrap(), "{f}");
        }
        assert_eq!(imported.source["files"][0]["file"], "stocks.csv.gz");

        // A zip with two CSVs is refused.
        let mut z = zip::ZipWriter::new(File::create(packed.join("SHARADAR_ACTIONS.zip")).unwrap());
        for name in ["a.csv", "b.csv"] {
            z.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
            z.write_all(b"date,action,ticker,contraticker\n").unwrap();
        }
        z.finish().unwrap();
        let e = import(&packed, &dir.join("store_packed"), None).err().unwrap().to_string();
        assert!(e.contains("expected one .csv inside, found 2"), "{e}");
    }
}
