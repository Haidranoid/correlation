//! correlation2's external series (weather, politics, news, rates, markets, migration, prosperity,
//! predictions, ... listed in `<external>/catalog_*.csv`, values in `series/<id>.csv`) as census-style
//! daily variables on the export's trading days: `stockai-judge export-external` (docs/external.md).
//!
//! Provenance: everything between the two "copy" markers below is copied verbatim from correlation2's
//! `crates/c2-census/examples/external_test.rs` at commit 310ea71 (310ea710c2486ae62d7074a755e4dcdb0aa03054;
//! the file last changed in fd75089): its lines 76-83 (the category names), 243-293 (`parse_csv`) and
//! 299-765 (`Transform`, `Meta`, `Series`, `ExtVar`, `live_gap`, `bridged`, `delay_rows`,
//! `days_in_month`, `weekday`, `anomalies`, `standardize`, `build_ext`, `read_series`,
//! `single_contract`, `is_annual`, `read_catalogs`). Only the imports differ. The tests port that
//! file's timing self-checks (`selftests`), with its `Rng` and `syn_meta` (lines 771-813). Change the
//! copy only together with correlation2's original, so stockai times every series as correlation2 does.
//!
//! The rules, from the source's header: an observation dated d with `known_after_days` k counts from
//! the first trading day on or after d + k (calendar days). `logdiff`/`diff`: the move on trading day
//! t is the change in the latest known level since day t-1 (release days carry the whole move, other
//! days 0; a change across a gap longer than the series' normal spacing is no move). `anomaly`/
//! `anomaly_log1p`: the value (or ln(1 + value)) minus the mean of the 15-day spans centred on the
//! same date in each of the previous 10 years (at least 3 years); the anomalies of the calendar days
//! that land on one trading day are averaged. `level`: the value minus its mean over the previous 252
//! trading days. Then, exactly like the census's stock variable: minus the trailing 252-day mean and
//! divided by the trailing 252-day sd (both through t-1, at least 126 values), clipped at +-5. A series
//! is live while its latest observation is recent for its frequency; elsewhere it is missing.

// The copy keeps what only correlation2's census test uses (`delay_rows`, `Meta::units`,
// `ExtVar::gap`), and indexes its tables by row, as there.
#![allow(dead_code, clippy::needless_range_loop)]

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use c2_census::variable;
use c2_data::{date_string, days_from_civil};
use rayon::prelude::*;
use serde_json::{Value, json};

use crate::npy;

// ---------------------------------------------------------------------------------------------
// Copy of correlation2's crates/c2-census/examples/external_test.rs at 310ea71: lines 76-83.
const SYN: &str = "synthetic";
/// Category for finite-life single-contract series (flagged in the catalog's notes).
const SINGLE: &str = "single_contract";
/// Pure-noise calibration series (`--noise N`): the baseline every ratio is compared with.
const NOISE: &str = "noise_test";
/// Annual series: one release a year, often all on the same date, so they act as one calendar
/// pulse that standardization pushes to +-5. Reported apart unless `--include-annual`.
const ANNUAL: &str = "annual";

// Lines 243-293.
// ---------------------------------------------------------------------------------------------
// CSV (RFC 4180: quotes, doubled quotes, CRLF, BOM)

fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let (mut rows, mut row, mut field) = (Vec::new(), Vec::new(), String::new());
    let (mut in_q, mut any) = (false, false);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_q {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_q = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' => {
                in_q = true;
                any = true;
            }
            ',' => {
                row.push(std::mem::take(&mut field));
                any = true;
            }
            '\r' => {}
            '\n' => {
                if any || !field.is_empty() {
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                }
                any = false;
            }
            _ => {
                field.push(c);
                any = true;
            }
        }
    }
    if any || !field.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

// Lines 299-765.
// ---------------------------------------------------------------------------------------------
// External series

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transform {
    LogDiff,
    Diff,
    Anomaly,
    AnomalyLog1p,
    Level,
}

impl Transform {
    fn parse(s: &str) -> Option<Transform> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "logdiff" => Transform::LogDiff,
            "diff" => Transform::Diff,
            "anomaly" => Transform::Anomaly,
            "anomaly_log1p" => Transform::AnomalyLog1p,
            "level" => Transform::Level,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Transform::LogDiff => "logdiff",
            Transform::Diff => "diff",
            Transform::Anomaly => "anomaly",
            Transform::AnomalyLog1p => "anomaly_log1p",
            Transform::Level => "level",
        }
    }
}

#[derive(Clone, Debug)]
struct Meta {
    id: String,
    name: String,
    category: String,
    subcategory: String,
    region: String,
    frequency: String,
    units: String,
    catalog: String,
    transform: Transform,
    known_after: i32,
}

struct Series {
    meta: Meta,
    obs: Vec<(i32, f64)>,
}

/// One external series on the census variable's rows.
struct ExtVar {
    /// Census-style variable; NaN where unavailable.
    v: Vec<f32>,
    /// Unstandardized move per row: log change (logdiff), change (diff), mean anomaly
    /// (anomaly kinds) or mean value (level); 0 on live rows without a new observation, NaN
    /// where the series is not live.
    raw: Vec<f64>,
    /// Trading rows a series may go without a new observation and still count as live.
    gap: usize,
}

/// Rows of staleness allowed by frequency: the larger of the catalog's and the one the median
/// spacing of the first 100 observations implies (so later data can't change earlier days).
fn live_gap(freq: &str, obs: &[(i32, f64)]) -> usize {
    let named = match freq.trim().to_ascii_lowercase().as_str() {
        "daily" => 10,
        "weekly" => 15,
        "monthly" => 50,
        "quarterly" => 140,
        "annual" | "yearly" => 300,
        _ => 0,
    };
    let mut gaps: Vec<i32> = obs.windows(2).take(100).map(|w| w[1].0 - w[0].0).collect();
    let inferred = if gaps.is_empty() {
        300
    } else {
        let mid = gaps.len() / 2;
        match *gaps.select_nth_unstable(mid).1 {
            ..=4 => 10,
            5..=10 => 15,
            11..=45 => 50,
            46..=120 => 140,
            _ => 300,
        }
    };
    named.max(inferred)
}

/// For each observation, whether the level change since the previous one counts as a move:
/// the gap may be at most the 90th-percentile spacing of the previous 250 gaps plus 2 days
/// (weekends and holidays for weekday series, a late month for monthly ones), but exactly 1 day
/// for series observed every day, so a missing day (a prediction market's contract roll, an
/// outage) is never bridged. Causal: it looks only at earlier gaps.
fn bridged(obs: &[(i32, f64)]) -> Vec<bool> {
    let mut out = vec![true; obs.len()];
    let mut win: VecDeque<i32> = VecDeque::new();
    let mut counts: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
    for i in 1..obs.len() {
        let gap = obs[i].0 - obs[i - 1].0;
        if win.len() >= 20 {
            let target = (win.len() * 9 / 10).min(win.len() - 1);
            let (mut seen, mut p90) = (0usize, 0i32);
            for (&g, &c) in &counts {
                seen += c;
                if seen > target {
                    p90 = g;
                    break;
                }
            }
            out[i] = gap <= if p90 <= 1 { 1 } else { p90 + 2 };
        }
        win.push_back(gap);
        *counts.entry(gap).or_default() += 1;
        if win.len() > 250 {
            let old = win.pop_front().unwrap();
            let c = counts.get_mut(&old).unwrap();
            *c -= 1;
            if *c == 0 {
                counts.remove(&old);
            }
        }
    }
    out
}

/// Trading rows by which an external's value can trail the days it describes: its
/// known_after_days (calendar days, an upper bound in rows) plus its measurement span.
fn delay_rows(m: &Meta) -> usize {
    let span = match m.frequency.as_str() {
        "weekly" => 5,
        "monthly" => 21,
        "quarterly" => 63,
        "annual" | "yearly" => 252,
        _ => 0,
    };
    m.known_after.max(0) as usize + span
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 { 29 } else { 28 }
        }
    }
}

/// Monday = 0 .. Sunday = 6.
fn weekday(d: i32) -> i32 {
    (d + 3).rem_euclid(7)
}

/// Per-observation anomalies: x - mean of x over the 15 days centred on the same date in each
/// of the previous 10 years (x = value, or ln(1 + value)); needs at least 3 such years.
fn anomalies(obs: &[(i32, f64)], log1p: bool) -> Vec<(i32, f64)> {
    let xs: Vec<(i32, f64)> = obs
        .iter()
        .filter_map(|&(d, v)| {
            let x = if log1p {
                if v > -1.0 { v.ln_1p() } else { return None }
            } else {
                v
            };
            x.is_finite().then_some((d, x))
        })
        .collect();
    let (Some(&(d0, _)), Some(&(d1, _))) = (xs.first(), xs.last()) else { return Vec::new() };
    let mut dense = vec![f64::NAN; (d1 - d0 + 1) as usize];
    for &(d, x) in &xs {
        dense[(d - d0) as usize] = x;
    }
    let get = |d: i32| if d < d0 || d > d1 { f64::NAN } else { dense[(d - d0) as usize] };
    xs.iter()
        .filter_map(|&(d, x)| {
            let (y, m, dd) = c2_data::civil_from_days(d);
            let (mut s, mut c, mut years) = (0.0, 0usize, 0usize);
            for yy in y - 10..y {
                let centre = days_from_civil(yy, m, dd.min(days_in_month(yy, m)));
                let mut any = false;
                for o in -7..=7 {
                    let v = get(centre + o);
                    if v.is_finite() {
                        s += v;
                        c += 1;
                        any = true;
                    }
                }
                years += usize::from(any);
            }
            (years >= 3).then(|| (d, x - s / c as f64))
        })
        .collect()
}

/// The census's standardization (variable.rs, second pass), on any daily series: minus the
/// trailing mean, divided by the trailing sd (rows d-252..d-1, at least 126 values), clipped at
/// +-5. The trailing sums are recomputed for every row (no running-sum drift), and a trailing
/// stretch without a single non-zero value gives no value at all (sparse indicators can't
/// divide by a zero or round-off sd).
fn standardize(x: &[f64]) -> Vec<f32> {
    let (lb, min_obs, clip) = (variable::LOOKBACK, variable::MIN_OBS, variable::CLIP as f64);
    let mut out = vec![f32::NAN; x.len()];
    out.par_iter_mut().enumerate().for_each(|(d, o)| {
        if !x[d].is_finite() {
            return;
        }
        let (mut s, mut ss, mut k, mut nz) = (0f64, 0f64, 0usize, 0usize);
        for &v in &x[d.saturating_sub(lb)..d] {
            if v.is_finite() {
                s += v;
                ss += v * v;
                k += 1;
                nz += usize::from(v != 0.0);
            }
        }
        if k < min_obs || nz == 0 {
            return;
        }
        let kf = k as f64;
        let var = (ss - s * s / kf) / kf;
        if var > 1e-12 * (ss / kf) && var > 0.0 {
            *o = ((x[d] - s / kf) / var.sqrt()).clamp(-clip, clip) as f32;
        }
    });
    out
}

fn build_ext(days: &[i32], s: &Series) -> ExtVar {
    let t = days.len();
    let tf = s.meta.transform;
    let gap = live_gap(&s.meta.frequency, &s.obs);
    let obs: Vec<(i32, f64)> = match tf {
        Transform::LogDiff => s.obs.iter().filter(|o| o.1 > 0.0).map(|&(d, v)| (d, v.ln())).collect(),
        Transform::Diff | Transform::Level => s.obs.clone(),
        Transform::Anomaly => anomalies(&s.obs, false),
        Transform::AnomalyLog1p => anomalies(&s.obs, true),
    };
    // Row at which each observation becomes usable: the first trading day >= date + k.
    let avail: Vec<usize> = obs.iter().map(|o| days.partition_point(|&d| d < o.0.saturating_add(s.meta.known_after))).collect();
    let bridge = bridged(&obs);
    let (mut x, mut raw) = (vec![f64::NAN; t], vec![f64::NAN; t]);
    let mut i = 0;
    let mut level: Option<f64> = None;
    let mut last_date: Option<i32> = None;
    let mut last: Option<usize> = None;
    let mut trail: VecDeque<(usize, f64)> = VecDeque::new();
    let mut tsum = 0f64;
    for row in 0..t {
        let (mut sum, mut cnt, mut newest, mut broken) = (0f64, 0usize, f64::NAN, false);
        while i < obs.len() && avail[i] <= row {
            // A level change across a gap longer than the series' normal spacing (a contract
            // roll, an outage) is not a move.
            broken |= last_date.is_some() && !bridge[i];
            last_date = Some(obs[i].0);
            sum += obs[i].1;
            cnt += 1;
            newest = obs[i].1;
            i += 1;
        }
        if cnt > 0 {
            last = Some(row);
        }
        let live = row > 0 && last.is_some_and(|l| row - l <= gap);
        while trail.front().is_some_and(|&(r, _)| r + variable::LOOKBACK < row) {
            tsum -= trail.pop_front().unwrap().1;
        }
        match tf {
            Transform::LogDiff | Transform::Diff => {
                if cnt > 0 {
                    if let (Some(l), true, false) = (level, row > 0, broken) {
                        x[row] = newest - l;
                    }
                    level = Some(newest);
                } else if live && level.is_some() {
                    x[row] = 0.0;
                }
                raw[row] = x[row];
            }
            Transform::Anomaly | Transform::AnomalyLog1p => {
                if cnt > 0 && row > 0 {
                    x[row] = sum / cnt as f64;
                } else if live {
                    x[row] = 0.0;
                }
                raw[row] = x[row];
            }
            Transform::Level => {
                let enough = trail.len() >= 10;
                if cnt > 0 {
                    let val = sum / cnt as f64;
                    if row > 0 && enough {
                        x[row] = val - tsum / trail.len() as f64;
                        raw[row] = val;
                    }
                    trail.push_back((row, val));
                    tsum += val;
                } else if live && enough {
                    x[row] = 0.0;
                    raw[row] = 0.0;
                }
            }
        }
    }
    ExtVar { v: standardize(&x), raw, gap }
}

fn read_series(path: &Path) -> Result<(Vec<(i32, f64)>, usize)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let (mut obs, mut bad) = (Vec::new(), 0usize);
    for (n, line) in text.lines().enumerate() {
        let line = line.trim().trim_start_matches('\u{feff}');
        if line.is_empty() || (n == 0 && line.to_ascii_lowercase().starts_with("date")) {
            continue;
        }
        let mut f = line.split(',');
        let (Some(d), Some(v)) = (f.next(), f.next()) else {
            bad += 1;
            continue;
        };
        match (c2_data::parse_date(d.trim().trim_matches('"')), v.trim().trim_matches('"').parse::<f64>()) {
            (Ok(d), Ok(v)) if v.is_finite() => obs.push((d, v)),
            _ => bad += 1,
        }
    }
    obs.sort_by_key(|o| o.0);
    let before = obs.len();
    obs.dedup_by_key(|o| o.0);
    let dups = before - obs.len();
    Ok((obs, bad + dups))
}

/// A catalog note flagging a finite-life single-contract series. The predictions catalog marks
/// them with an upper-case "FINITE-LIFE" ("FINITE-LIFE single contract", "FINITE-LIFE, FINAL 90
/// DAYS ONLY", "FINITE-LIFE PLAY-MONEY FORECAST"); its continuous series built by stitching
/// contracts say "ROLLING (finite-life contracts stitched)" and are not flagged. Lower-case
/// mentions count unless the note describes a rolling or stitched series or negates them.
fn single_contract(notes: &str) -> bool {
    if notes.contains("FINITE-LIFE") {
        return true;
    }
    let n = notes.to_ascii_lowercase();
    if n.contains("rolling") || n.contains("stitched") {
        return false;
    }
    ["finite-life", "finite life", "single-contract", "single contract"].iter().any(|p| {
        n.match_indices(p).any(|(i, _)| {
            let before = n[..i].trim_end();
            !["not a", "not", "no", "non-", "non"].iter().any(|neg| before.ends_with(neg))
        })
    })
}

/// One release a year: by the catalog's frequency, or a median spacing of 300 days or more.
fn is_annual(m: &Meta, obs: &[(i32, f64)]) -> bool {
    if matches!(m.frequency.as_str(), "annual" | "yearly") {
        return true;
    }
    let mut gaps: Vec<i32> = obs.windows(2).map(|w| w[1].0 - w[0].0).collect();
    if gaps.is_empty() {
        return false;
    }
    let mid = gaps.len() / 2;
    *gaps.select_nth_unstable(mid).1 >= 300
}

/// Per catalog: (name, series listed, series loaded).
type CatalogCounts = Vec<(String, usize, usize)>;

fn read_catalogs(dir: &Path, only: &[String], warn: &mut Vec<String>) -> Result<(Vec<Series>, CatalogCounts)> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("no external data folder {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("catalog_") && n.ends_with(".csv")))
        .collect();
    files.sort();
    let (mut out, mut summary, mut seen) = (Vec::new(), Vec::new(), HashSet::new());
    for f in files {
        let cname = f.file_stem().and_then(|s| s.to_str()).unwrap_or("").trim_start_matches("catalog_").to_string();
        if !only.is_empty() && !only.contains(&cname) {
            warn.push(format!("catalog {cname} present but not included in this run (--catalogs)"));
            continue;
        }
        let rows = parse_csv(&std::fs::read_to_string(&f)?);
        let Some(header) = rows.first() else { continue };
        let col = |name: &str| header.iter().position(|h| h.trim().eq_ignore_ascii_case(name));
        let (Some(ci_id), Some(ci_tf), Some(ci_k)) = (col("id"), col("transform"), col("known_after_days")) else {
            warn.push(format!("{cname}: catalog lacks id/transform/known_after_days columns; skipped"));
            continue;
        };
        let cols = [col("name"), col("category"), col("subcategory"), col("region"), col("frequency"), col("units"), col("notes")];
        let (mut listed, mut used) = (0, 0);
        for r in &rows[1..] {
            let get = |i: Option<usize>| i.and_then(|i| r.get(i)).map(|s| s.trim().to_string()).unwrap_or_default();
            let id = get(Some(ci_id));
            if id.is_empty() {
                continue;
            }
            listed += 1;
            if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.') {
                warn.push(format!("{cname}: id {id:?} has unexpected characters; skipped"));
                continue;
            }
            if !seen.insert(id.clone()) {
                warn.push(format!("{cname}: duplicate id {id}; skipped"));
                continue;
            }
            let Some(tf) = Transform::parse(&get(Some(ci_tf))) else {
                warn.push(format!("{cname}: {id}: unknown transform {:?}; skipped", get(Some(ci_tf))));
                continue;
            };
            let k = match get(Some(ci_k)).parse::<f64>() {
                Ok(k) if k.is_finite() && k >= 0.0 => k.ceil() as i32,
                _ => {
                    warn.push(format!("{cname}: {id}: bad known_after_days {:?}; skipped", get(Some(ci_k))));
                    continue;
                }
            };
            let (obs, bad) = match read_series(&dir.join("series").join(format!("{id}.csv"))) {
                Ok(o) => o,
                Err(e) => {
                    warn.push(format!("{cname}: {id}: {e:#}; skipped"));
                    continue;
                }
            };
            if bad > 0 {
                warn.push(format!("{cname}: {id}: {bad} unreadable or duplicate rows dropped"));
            }
            if obs.len() < 2 {
                warn.push(format!("{cname}: {id}: only {} observations; skipped", obs.len()));
                continue;
            }
            let mut category = get(cols[1]).to_ascii_lowercase();
            if category.is_empty() || category == SYN || category == SINGLE || category == NOISE || category == ANNUAL {
                category = "other".into();
            }
            let mut subcategory = get(cols[2]);
            // Finite-life single-contract series (prediction markets) are reported apart from the
            // headline numbers.
            if single_contract(&get(cols[6])) {
                subcategory = format!("{category}/{subcategory}");
                category = SINGLE.into();
            }
            let meta = Meta {
                id: id.clone(),
                name: get(cols[0]),
                category,
                subcategory,
                region: get(cols[3]),
                frequency: get(cols[4]).to_ascii_lowercase(),
                units: get(cols[5]),
                catalog: cname.clone(),
                transform: tf,
                known_after: k,
            };
            out.push(Series { meta, obs });
            used += 1;
        }
        summary.push((cname, listed, used));
    }
    Ok((out, summary))
}

// ---------------------------------------------------------------------------------------------
// End of the copy. stockai's export: every catalog series on the export's trading days.

/// The correlation2 commit and file the copy above comes from.
pub const C2_COMMIT: &str = "310ea710c2486ae62d7074a755e4dcdb0aa03054";
pub const C2_SOURCE: &str = "crates/c2-census/examples/external_test.rs";
pub const FORMAT: &str = "stockai-external-1";
pub const HEADER: &str = "row\tid\tname\tcategory\tsubcategory\tregion\tcatalog\ttransform\tfrequency\tknown_after_days\tstatus\tfirst\tlast\tobservations";
/// The files and what they hold (meta.json's `files`).
pub const FILES: [(&str, &str); 3] = [
    ("ext_v.npy", "float32 [K, T]: row k = line k of ext_meta.tsv, column d = day d of the export (days.npy); the census-style standardized daily move, NaN where unavailable"),
    ("ext_raw.npy", "float32 [K, T]: the move before standardization (log change, change, mean anomaly or mean level), 0 on live days without a new observation, NaN where the series is not live"),
    ("ext_meta.tsv", "one line per row: the catalog's id, name, category (single_contract for finite-life contracts, as correlation2 reads them), subcategory, region, catalog, transform, frequency, known_after_days (rounded up); status; first and last day with a value in ext_v.npy and the days with a non-zero move (as correlation2's externals.csv)"),
];

/// correlation2's classes: a finite-life single contract (from the catalog's notes; `read_catalogs`
/// makes its category single_contract), annual (`is_annual`), its noise calibration category, and
/// every other series (the headline numbers).
fn status(s: &Series) -> &'static str {
    if s.meta.category == SINGLE {
        "single_contract"
    } else if is_annual(&s.meta, &s.obs) {
        "annual"
    } else if s.meta.category == NOISE {
        "noise"
    } else {
        "headline"
    }
}

/// What an export wrote.
pub struct Summary {
    pub out: PathBuf,
    pub k: usize,
    pub t: usize,
    pub statuses: BTreeMap<String, usize>,
    pub categories: BTreeMap<String, usize>,
    pub warnings: Vec<String>,
    /// Per year through 2023: the mean number of series with a value in ext_v.npy per trading day,
    /// all and headline only.
    pub live: Vec<(i32, f64, f64)>,
    pub bytes: u64,
}

/// A text field for the TSV (tabs and line breaks become spaces).
fn tsv(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

/// Writes ext_v.npy, ext_raw.npy, ext_meta.tsv and meta.json into `out` for every series of the
/// catalogs in `external` (read only: the catalog_*.csv files and series/), on `days`: the export's
/// trading days, whose export `fingerprint` meta.json records. Every file goes through a temporary
/// file and a rename; meta.json is removed first and written last.
pub fn export(days: &[i32], fingerprint: &str, external: &Path, out: &Path) -> Result<Summary> {
    let t = days.len();
    ensure!(t > 0, "no trading days");
    let mut warnings = Vec::new();
    let (series, catalogs) = read_catalogs(external, &[], &mut warnings)?;
    ensure!(!series.is_empty(), "{}: no series in its catalog_*.csv files", external.display());
    let vars: Vec<ExtVar> = series.par_iter().map(|s| build_ext(days, s)).collect();
    let k = series.len();
    let statuses: Vec<&str> = series.iter().map(status).collect();

    std::fs::create_dir_all(out)?;
    let meta_path = out.join("meta.json");
    if meta_path.exists() {
        std::fs::remove_file(&meta_path)?;
    }
    let v: Vec<f32> = vars.iter().flat_map(|x| x.v.iter().copied()).collect();
    npy::write(&out.join("ext_v.npy"), &[k, t], &v)?;
    let raw: Vec<f32> = vars.iter().flat_map(|x| x.raw.iter().map(|&r| r as f32)).collect();
    npy::write(&out.join("ext_raw.npy"), &[k, t], &raw)?;
    npy::write_atomic(&out.join("ext_meta.tsv"), |w| {
        writeln!(w, "{HEADER}")?;
        for (row, ((s, x), st)) in series.iter().zip(&vars).zip(&statuses).enumerate() {
            let m = &s.meta;
            let first = x.v.iter().position(|v| v.is_finite()).map(|d| date_string(days[d])).unwrap_or_default();
            let last = x.v.iter().rposition(|v| v.is_finite()).map(|d| date_string(days[d])).unwrap_or_default();
            let moves = x.raw.iter().filter(|r| r.is_finite() && **r != 0.0).count();
            writeln!(
                w,
                "{row}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{st}\t{first}\t{last}\t{moves}",
                tsv(&m.id),
                tsv(&m.name),
                tsv(&m.category),
                tsv(&m.subcategory),
                tsv(&m.region),
                tsv(&m.catalog),
                m.transform.name(),
                tsv(&m.frequency),
                m.known_after
            )?;
        }
        Ok(())
    })?;

    // Provenance: every catalog file's SHA-256, and one over the series files' (in row order).
    let mut files: Vec<PathBuf> = std::fs::read_dir(external)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("catalog_") && n.ends_with(".csv")))
        .collect();
    files.sort();
    let mut catalog_json = Vec::new();
    for f in &files {
        let (bytes, sha) = crate::sharadar::sha256_file(f)?;
        let name = f.file_stem().and_then(|s| s.to_str()).unwrap_or("").trim_start_matches("catalog_").to_string();
        let counts = catalogs.iter().find(|c| c.0 == name);
        catalog_json.push(json!({
            "file": f.file_name().map(|n| n.to_string_lossy().into_owned()),
            "bytes": bytes,
            "sha256": sha,
            "listed": counts.map(|c| c.1),
            "loaded": counts.map(|c| c.2),
        }));
    }
    let hashes: Vec<String> = series
        .par_iter()
        .map(|s| crate::sharadar::sha256_file(&external.join("series").join(format!("{}.csv", s.meta.id))).map(|(_, h)| format!("{}\t{h}\n", s.meta.id)))
        .collect::<Result<_>>()?;
    let series_sha = {
        use sha2::Digest;
        format!("{:x}", sha2::Sha256::digest(hashes.concat().as_bytes()))
    };

    let mut by_status: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_category: BTreeMap<String, usize> = BTreeMap::new();
    let mut cross: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for (s, st) in series.iter().zip(&statuses) {
        *by_status.entry(st.to_string()).or_default() += 1;
        *by_category.entry(s.meta.category.clone()).or_default() += 1;
        *cross.entry(s.meta.category.clone()).or_default().entry(st.to_string()).or_default() += 1;
    }
    // Series with a value per trading day, by year (through 2023: nothing from the locked year on).
    let (y0, y1) = (c2_data::civil_from_days(days[0]).0, c2_data::civil_from_days(days[t - 1]).0.min(2023));
    let headline: Vec<bool> = statuses.iter().map(|s| *s == "headline").collect();
    let live: Vec<(i32, f64, f64)> = (y0..=y1)
        .filter_map(|y| {
            let rows: Vec<usize> = (0..t).filter(|&d| c2_data::civil_from_days(days[d]).0 == y).collect();
            if rows.is_empty() {
                return None;
            }
            let (mut all, mut head) = (0usize, 0usize);
            for (x, &h) in vars.iter().zip(&headline) {
                let n = rows.iter().filter(|&&d| x.v[d].is_finite()).count();
                all += n;
                head += if h { n } else { 0 };
            }
            Some((y, all as f64 / rows.len() as f64, head as f64 / rows.len() as f64))
        })
        .collect();
    let meta = json!({
        "format": FORMAT,
        "k": k,
        "t": t,
        "first_day": date_string(days[0]),
        "last_day": date_string(days[t - 1]),
        "fingerprint": fingerprint,
        "days": "the trading days of stockai-judge export of the same store (its days.npy and fingerprint)",
        "correlation2_commit": C2_COMMIT,
        "correlation2_source": format!("{C2_SOURCE} at {}", &C2_COMMIT[..7]),
        "external_dir": external.display().to_string(),
        "catalogs": catalog_json,
        "series_files_sha256": series_sha,
        "counts": {"status": by_status, "category": by_category, "category_by_status": cross},
        "warnings": warnings,
        "live_per_day": live.iter().map(|&(y, a, h)| (y.to_string(), json!({"all": (a * 10.0).round() / 10.0, "headline": (h * 10.0).round() / 10.0}))).collect::<serde_json::Map<String, Value>>(),
        "files": FILES.iter().map(|(f, d)| (f.to_string(), Value::from(*d))).collect::<serde_json::Map<String, Value>>(),
        "writer": concat!("stockai-judge ", env!("CARGO_PKG_VERSION")),
        "created_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),
    });
    npy::write_atomic(&meta_path, |w| {
        w.write_all(serde_json::to_string_pretty(&meta)?.as_bytes())?;
        w.write_all(b"\n")?;
        Ok(())
    })?;
    let bytes = ["ext_v.npy", "ext_raw.npy", "ext_meta.tsv", "meta.json"].iter().map(|f| std::fs::metadata(out.join(f)).map(|m| m.len()).unwrap_or(0)).sum();
    Ok(Summary { out: out.to_path_buf(), k, t, statuses: by_status, categories: by_category, warnings, live, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------------------------
    // Copy of correlation2's external_test.rs at 310ea71: lines 771-813 (the self-checks' Rng and metadata).
    struct Rng(u64);

    impl Rng {
        fn u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self) -> f64 {
            ((self.u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (a, b) = (self.uniform(), self.uniform());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
        fn poisson(&mut self, lambda: f64) -> f64 {
            let (l, mut k, mut p) = ((-lambda).exp(), 0.0, 1.0);
            loop {
                p *= self.uniform();
                if p <= l {
                    return k;
                }
                k += 1.0;
            }
        }
    }

    fn syn_meta(id: &str, name: String, sub: &str, freq: &str, tf: Transform, k: i32) -> Meta {
        Meta {
            id: id.into(),
            name,
            category: SYN.into(),
            subcategory: sub.into(),
            region: "-".into(),
            frequency: freq.into(),
            units: "-".into(),
            catalog: "synthetic".into(),
            transform: tf,
            known_after: k,
        }
    }
    // End of the copy.

    /// A trading calendar for the self-checks (correlation2 runs them on the store's): weekdays from
    /// 1999 to 2025-09, less New Year's Day, Independence Day and Christmas.
    fn calendar() -> Vec<i32> {
        (days_from_civil(1999, 1, 4)..=days_from_civil(2025, 9, 1))
            .filter(|&d| {
                let (_, m, dd) = c2_data::civil_from_days(d);
                weekday(d) < 5 && !matches!((m, dd), (1, 1) | (7, 4) | (12, 25))
            })
            .collect()
    }

    fn meta(tf: Transform, k: i32, freq: &str) -> Meta {
        syn_meta("t", "test".into(), "test", freq, tf, k)
    }

    fn all() -> Vec<i32> {
        (days_from_civil(1999, 1, 1)..=days_from_civil(2025, 9, 1)).collect()
    }

    /// correlation2's self-check setup: a Friday in the middle of the data whose next trading day is the
    /// following Monday, and its row and Monday's.
    fn friday(days: &[i32]) -> (i32, usize, usize) {
        let row = |d: i32| days.binary_search(&d).ok();
        let fri = days.iter().copied().find(|&d| d > days_from_civil(2012, 3, 1) && weekday(d) == 4 && row(d + 3).is_some()).unwrap();
        (fri, row(fri).unwrap(), row(fri + 3).unwrap())
    }

    #[test]
    fn csv_parser_and_finite_life_flags() {
        let rows = parse_csv("\u{feff}a,\"b,c\",d\r\n\"x\"\"y\",,z\n\nlast");
        assert_eq!(rows, vec![vec!["a", "b,c", "d"], vec!["x\"y", "", "z"], vec!["last"]], "CSV parser: quotes, doubled quotes, CRLF, BOM, blank lines");
        let flags = [
            ("FINITE-LIFE single contract (crypto): \"Will Bitcoin hit $100k in 2024?\"", true),
            ("FINITE-LIFE PLAY-MONEY FORECAST, not a real-money market", true),
            ("FINITE-LIFE, FINAL 90 DAYS ONLY: \"Which party will win the 2020 election?\"", true),
            ("ROLLING (finite-life contracts stitched): Kalshi KXFED strike ladder", false),
            ("ROLLING by election cycle: House control after 2024, then 2026", false),
            ("a continuous index, not a single contract", false),
            ("one finite-life contract", true),
            ("finite-life single contract RECSSNBER-23 (open 2022-11-07, close 2024-01-25, settled: no)", true),
            ("finite-life single contract; play-money forecast, not a real-money market", true),
            ("ROLLING continuous series (successive contracts stitched): Kalshi KXFED/FED", false),
        ];
        for (n, want) in flags {
            assert_eq!(single_contract(n), want, "{n}");
        }
    }

    #[test]
    fn a_level_change_dated_friday_lands_by_known_after_days() {
        let days = calendar();
        let (fri, fr, mo) = friday(&days);
        assert_eq!((date_string(fri), date_string(fri + 3)), ("2012-03-02".to_string(), "2012-03-05".to_string()));
        let jump: Vec<(i32, f64)> = all().iter().map(|&d| (d, if d >= fri { 110.0 } else { 100.0 })).collect();
        let v0 = build_ext(&days, &Series { meta: meta(Transform::LogDiff, 0, "daily"), obs: jump.clone() });
        let v1 = build_ext(&days, &Series { meta: meta(Transform::LogDiff, 1, "daily"), obs: jump });
        let j = 1.1f64.ln();
        assert!((v0.raw[fr] - j).abs() < 1e-12 && v0.raw[mo] == 0.0, "known_after_days 0: that Friday");
        assert!(v1.raw[fr] == 0.0 && (v1.raw[mo] - j).abs() < 1e-12, "known_after_days 1: the Monday");
    }

    #[test]
    fn a_weekly_value_dated_by_week_start_lands_after_the_week() {
        let days = calendar();
        let (fri, _, _) = friday(&days);
        let sun0 = fri + 2 - 14;
        let weekly: Vec<(i32, f64)> = all().iter().copied().filter(|&d| weekday(d) == 6).map(|d| (d, if d >= sun0 { 60.0 } else { 50.0 })).collect();
        let vw = build_ext(&days, &Series { meta: meta(Transform::Diff, 7, "weekly"), obs: weekly });
        let land = days.partition_point(|&d| d < sun0 + 7);
        let quiet = (land.saturating_sub(30)..land).all(|r| vw.raw[r] == 0.0);
        assert!((vw.raw[land] - 10.0).abs() < 1e-12 && quiet && days[land] > sun0 + 6);
        assert_eq!((date_string(sun0), date_string(days[land])), ("2012-02-19".to_string(), "2012-02-27".to_string()));
    }

    #[test]
    fn weekend_values_known_the_next_day_are_averaged_onto_monday() {
        let days = calendar();
        let (fri, _, mo) = friday(&days);
        let lv: Vec<(i32, f64)> = all().iter().map(|&d| (d, if d == fri { 1.0 } else if d == fri + 1 { 2.0 } else if d == fri + 2 { 3.0 } else if d == fri + 3 { 10.0 } else { 0.0 })).collect();
        let vl = build_ext(&days, &Series { meta: meta(Transform::Level, 1, "daily"), obs: lv });
        assert!((vl.raw[mo] - 2.0).abs() < 1e-12, "Friday, Saturday and Sunday onto Monday: {}", vl.raw[mo]);
        assert!((vl.raw[mo + 1] - 10.0).abs() < 1e-12, "Monday's onto Tuesday: {}", vl.raw[mo + 1]);
    }

    #[test]
    fn a_seven_day_series_missing_a_day_doesnt_count_the_jump() {
        let days = calendar();
        let (fri, _, _) = friday(&days);
        let row = |d: i32| days.binary_search(&d).ok();
        let skip = fri + 3; // Monday
        let daily: Vec<(i32, f64)> = all().iter().copied().filter(|&d| d != skip).map(|d| (d, if d > skip { 120.0 } else { 100.0 })).collect();
        let vd = build_ext(&days, &Series { meta: meta(Transform::LogDiff, 0, "daily"), obs: daily });
        let wkd: Vec<(i32, f64)> = all().iter().copied().filter(|&d| weekday(d) < 5 && d != skip).map(|d| (d, if d > skip { 120.0 } else { 100.0 })).collect();
        let vk = build_ext(&days, &Series { meta: meta(Transform::LogDiff, 0, "daily"), obs: wkd });
        let tue = row(skip + 1).unwrap();
        assert!(vd.raw[tue].is_nan() && vd.raw[tue + 1] == 0.0, "a 7-day series missing a day: no move across the gap");
        assert!((vk.raw[tue] - 1.2f64.ln()).abs() < 1e-12, "a weekday series bridges a 1-day holiday");
    }

    #[test]
    fn no_look_ahead_for_every_transform() {
        let days = calendar();
        let all = all();
        let mut rng = Rng(99);
        let cut = days_from_civil(2013, 6, 14);
        for (tf, k, freq) in [(Transform::LogDiff, 0, "daily"), (Transform::Diff, 5, "weekly"), (Transform::Anomaly, 1, "daily"), (Transform::AnomalyLog1p, 1, "daily"), (Transform::Level, 7, "weekly"), (Transform::LogDiff, 30, "monthly")] {
            let ds: Vec<i32> = match freq {
                "weekly" => all.iter().copied().filter(|&d| weekday(d) == 6).collect(),
                "monthly" => all.iter().copied().filter(|&d| c2_data::civil_from_days(d).2 == 1).collect(),
                _ => all.iter().copied().filter(|_| rng.uniform() > 0.03).collect(),
            };
            let draw = |rng: &mut Rng, d: i32| -> f64 {
                let doy = (d % 365) as f64;
                match tf {
                    Transform::LogDiff => 100.0 * (0.2 * rng.normal()).exp(),
                    Transform::AnomalyLog1p => if rng.uniform() < 0.6 { 0.0 } else { 3.0 * rng.uniform() },
                    _ => 10.0 * (doy / 58.0).sin() + rng.normal(),
                }
            };
            let base: Vec<(i32, f64)> = ds.iter().map(|&d| (d, draw(&mut rng, d))).collect();
            let mut pert: Vec<(i32, f64)> = Vec::new();
            for &(d, v) in &base {
                if d < cut {
                    pert.push((d, v));
                } else if rng.uniform() > 0.2 {
                    pert.push((d, draw(&mut rng, d)));
                }
            }
            let a = build_ext(&days, &Series { meta: meta(tf, k, freq), obs: base });
            let b = build_ext(&days, &Series { meta: meta(tf, k, freq), obs: pert });
            let first = days.partition_point(|&d| d < cut + k);
            let same = (0..first).all(|r| a.v[r].to_bits() == b.v[r].to_bits() && a.raw[r].to_bits() == b.raw[r].to_bits());
            let finite = (0..first).filter(|&r| a.v[r].is_finite()).count();
            let differs = (first..days.len()).any(|r| a.v[r].to_bits() != b.v[r].to_bits());
            assert!(same && differs && finite > 1000, "{} k={k}: earlier rows identical {same}, {finite} finite, later rows differ {differs}", tf.name());
        }
    }

    #[test]
    fn anomalies_remove_the_seasonal_cycle() {
        let all = all();
        let mut rng = Rng(5);
        let noise: Vec<f64> = all.iter().map(|_| rng.normal()).collect();
        let doy = |d: i32| (d - days_from_civil(c2_data::civil_from_days(d).0, 1, 1)) as f64;
        let obs: Vec<(i32, f64)> = all.iter().zip(&noise).map(|(&d, &e)| (d, 15.0 + 12.0 * (std::f64::consts::TAU * doy(d) / 365.25).sin() + e)).collect();
        let an = anomalies(&obs, false);
        let nmap: std::collections::HashMap<i32, f64> = all.iter().copied().zip(noise.iter().copied()).collect();
        let pairs: Vec<(f64, f64)> = an.iter().filter(|o| o.0 >= days_from_civil(2004, 1, 1)).map(|o| (o.1, nmap[&o.0])).collect();
        // c2_engine::pearson, inline (stockai doesn't depend on c2-engine).
        let corr = {
            let n = pairs.len() as f64;
            let (sx, sy) = pairs.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
            let (sxx, syy, sxy) = pairs.iter().fold((0.0, 0.0, 0.0), |a, p| (a.0 + p.0 * p.0, a.1 + p.1 * p.1, a.2 + p.0 * p.1));
            let (cov, vx, vy) = (sxy - sx * sy / n, sxx - sx * sx / n, syy - sy * sy / n);
            if vx <= 0.0 || vy <= 0.0 { 0.0 } else { cov / (vx * vy).sqrt() }
        };
        let first_an = an.first().map(|o| o.0).unwrap_or(0);
        assert!(corr > 0.98 && first_an >= days_from_civil(2002, 1, 1) - 7, "correlation with the injected noise {corr:.3}; first anomaly {}", date_string(first_an));
    }

    #[test]
    fn sparse_indicators_stay_bounded() {
        let days = calendar();
        let row = |d: i32| days.binary_search(&d).ok();
        let mut rng = Rng(8);
        let sparse: Vec<(i32, f64)> = all().iter().copied().filter(|&d| weekday(d) < 5).map(|d| (d, if d < days_from_civil(2010, 1, 1) && rng.uniform() < 0.01 { 1.0 } else { 0.0 })).collect();
        let vs = build_ext(&days, &Series { meta: meta(Transform::Level, 0, "daily"), obs: sparse });
        let maxabs = vs.v.iter().filter(|v| v.is_finite()).fold(0f32, |a, v| a.max(v.abs()));
        let late = row(days_from_civil(2012, 6, 1)).unwrap_or_else(|| days.partition_point(|&d| d < days_from_civil(2012, 6, 1)));
        assert!(maxabs <= 5.0 && vs.v[late].is_nan() && vs.v.iter().all(|v| !v.is_infinite()), "max {maxabs}");
    }

    #[test]
    fn standardization_uses_rows_through_t_minus_1_only() {
        let mut rng = Rng(3);
        let x: Vec<f64> = (0..1200).map(|i| if i % 7 == 3 { f64::NAN } else { rng.normal() * (1.0 + i as f64 / 300.0) + 0.01 * i as f64 }).collect();
        let a = standardize(&x);
        // Row d: (x[d] - mean) / sd over the finite values of rows d-252 .. d-1 (at least 126), clipped.
        for d in [120usize, 150, 252, 253, 700, 1199] {
            let w: Vec<f64> = x[d.saturating_sub(variable::LOOKBACK)..d].iter().copied().filter(|v| v.is_finite()).collect();
            if !x[d].is_finite() || w.len() < variable::MIN_OBS {
                assert!(a[d].is_nan(), "row {d}");
                continue;
            }
            let n = w.len() as f64;
            let m = w.iter().sum::<f64>() / n;
            let sd = (w.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n).sqrt();
            let want = ((x[d] - m) / sd).clamp(-5.0, 5.0) as f32;
            assert!((a[d] - want).abs() < 1e-5, "row {d}: {} vs {want}", a[d]);
        }
        // Changing rows t and later changes nothing before t; rows before t-252 don't reach row t.
        let t = 600;
        let mut later = x.clone();
        for v in &mut later[t..] {
            *v = 50.0 * *v + 3.0;
        }
        let b = standardize(&later);
        assert!((0..t).all(|r| a[r].to_bits() == b[r].to_bits()));
        assert_ne!(a[t].to_bits(), b[t].to_bits());
        let mut early = x.clone();
        for v in &mut early[..t - variable::LOOKBACK] {
            *v *= -7.0;
        }
        let c = standardize(&early);
        assert!((t..x.len()).all(|r| a[r].to_bits() == c[r].to_bits()), "only rows t-252 .. t-1 enter row t");
    }

    #[test]
    fn export_writes_aligned_files() {
        let dir = std::env::temp_dir().join(format!("stockai-judge-external-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let ext = dir.join("external");
        std::fs::create_dir_all(ext.join("series")).unwrap();
        let days = calendar();
        let all = all();
        let write = |id: &str, obs: &[(i32, f64)]| {
            let mut s = String::from("date,value\n");
            for &(d, v) in obs {
                s += &format!("{},{v}\n", date_string(d));
            }
            std::fs::write(ext.join("series").join(format!("{id}.csv")), s).unwrap();
        };
        let mut rng = Rng(1);
        let mut lv = 0.0;
        let fx: Vec<(i32, f64)> = days.iter().map(|&d| { lv += 0.01 * rng.normal(); (d, 100.0 * f64::exp(lv)) }).collect();
        let wx: Vec<(i32, f64)> = all.iter().map(|&d| (d, 10.0 * ((d % 365) as f64 / 58.0).sin() + rng.normal())).collect();
        let annual: Vec<(i32, f64)> = (2000..2025).map(|y| (days_from_civil(y, 1, 1), y as f64 + rng.normal())).collect();
        let pm: Vec<(i32, f64)> = all.iter().copied().filter(|&d| d >= days_from_civil(2020, 1, 1)).map(|d| (d, 0.5 + 0.1 * rng.normal())).collect();
        write("fx_b", &fx);
        write("wx_a", &wx);
        write("ann_c", &annual);
        write("pm_d", &pm);
        write("one_e", &fx[..1]);
        std::fs::write(
            ext.join("catalog_test-a.csv"),
            "id,name,category,subcategory,region,source,url,frequency,units,transform,known_after_days,first,last,rows,notes\n\
             wx_a,\"Temperature, test\",Weather,temp,US,src,,daily,degF,anomaly,1.5,,,,\n\
             fx_b,FX test,fx,g10,US,src,,daily,USD,logdiff,0,,,,\n\
             ann_c,Annual test,macro,yearly,US,src,,annual,x,diff,90,,,,\n\
             bad_x,Bad transform,macro,x,US,src,,daily,x,ratio,0,,,,\n\
             one_e,One observation,fx,g10,US,src,,daily,x,logdiff,0,,,,\n",
        )
        .unwrap();
        std::fs::write(
            ext.join("catalog_test-b.csv"),
            "id,name,category,subcategory,region,source,url,frequency,units,transform,known_after_days,first,last,rows,notes\n\
             pm_d,Contract test,prediction_market,contract,US,src,,daily,prob,diff,0,,,,\"FINITE-LIFE single contract: test\"\n\
             fx_b,FX again,fx,g10,US,src,,daily,USD,logdiff,0,,,,\n",
        )
        .unwrap();
        let out = dir.join("out");
        let s = export(&days, "00ff", &ext, &out).unwrap();
        assert_eq!((s.k, s.t), (4, days.len()));
        assert_eq!(s.statuses, BTreeMap::from([("annual".into(), 1), ("headline".into(), 2), ("single_contract".into(), 1)]));
        assert_eq!(s.categories, BTreeMap::from([("fx".into(), 1), ("macro".into(), 1), ("single_contract".into(), 1), ("weather".into(), 1)]));
        assert_eq!(s.warnings.len(), 3, "{:?}", s.warnings);

        // Rows in catalog order, values exactly build_ext's on these days.
        let tsv = std::fs::read_to_string(out.join("ext_meta.tsv")).unwrap();
        let lines: Vec<Vec<&str>> = tsv.lines().map(|l| l.split('\t').collect()).collect();
        assert_eq!(lines[0].join("\t"), HEADER);
        let ids: Vec<&str> = lines[1..].iter().map(|l| l[1]).collect();
        assert_eq!(ids, ["wx_a", "fx_b", "ann_c", "pm_d"]);
        assert_eq!(lines[1][2..11], ["Temperature, test", "weather", "temp", "US", "test-a", "anomaly", "daily", "2", "headline"]);
        assert_eq!(lines[4][3..4], ["single_contract"]);
        assert_eq!(lines[3][10], "annual");
        let rows = read_catalogs(&ext, &[], &mut Vec::new()).unwrap().0;
        let a = npy::read(&out.join("ext_v.npy")).unwrap();
        assert_eq!(a.header.shape, [4, days.len()]);
        let npy::Data::F4(v) = a.data else { panic!("dtype") };
        let npy::Data::F4(raw) = npy::read(&out.join("ext_raw.npy")).unwrap().data else { panic!("dtype") };
        for (k, s) in rows.iter().enumerate() {
            let x = build_ext(&days, s);
            assert!(x.v.iter().zip(&v[k * days.len()..(k + 1) * days.len()]).all(|(a, b)| a.to_bits() == b.to_bits()), "{}", s.meta.id);
            assert!(x.raw.iter().zip(&raw[k * days.len()..(k + 1) * days.len()]).all(|(a, b)| (*a as f32).to_bits() == b.to_bits()), "{}", s.meta.id);
            let first = x.v.iter().position(|v| v.is_finite()).map(|d| date_string(days[d])).unwrap_or_default();
            assert_eq!(lines[k + 1][11], first);
        }
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(out.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["fingerprint"], "00ff");
        assert_eq!(meta["correlation2_commit"], C2_COMMIT);
        assert_eq!(meta["catalogs"].as_array().unwrap().len(), 2);
        assert_eq!(meta["catalogs"][0]["sha256"], crate::sharadar::sha256_file(&ext.join("catalog_test-a.csv")).unwrap().1.as_str());
        assert_eq!((meta["catalogs"][0]["listed"].as_u64(), meta["catalogs"][0]["loaded"].as_u64()), (Some(5), Some(3)));
        assert_eq!(meta["counts"]["category_by_status"]["fx"]["headline"], 1);
    }
}
