//! `stockai-judge check-sharadar`: the day-one checks of a Sharadar download (docs/sharadar.md;
//! test-results-30d.md, "Next steps"), on the raw files, read and decided by the importer's own code:
//! - SIVB, BBBY, FRC and TWTR: their price rows, `tickers` rows, actions near their last price and
//!   delisting kind (TWTR should be a buyout; SIVB, FRC and BBBY a failure or unknown).
//! - Delisted common stocks by year of their last price and kind; the action codes with counts; the
//!   `tickers` categories with the kind each gets (Kind::Stock when it contains "Common Stock").
//! - Daily returns from closeadj against the Stooq store's: AAPL, IBM and KO in detail (a check),
//!   then every common stock in both stores (a diagnostic of Stooq's returns): by year, the stocks
//!   whose returns differ on several days; the stocks whose Stooq closes equal the as-traded closes
//!   from 2023 on (no dividend adjustment); the biggest single-day gaps.
//!
//! Nothing in the locked year (2024-06-28 on) is shown: the price comparison stops at `until` and
//! later delistings are only counted together.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::Result;
use c2_data::{Kind, Store, civil_from_days, date_string, days_from_civil};
use rayon::prelude::*;

use crate::sharadar::{self, Built, DelistKind, KNOWN_CODES, Raw, WINDOW_DAYS};

/// The companies to look for: ticker, words of the company's name, the kinds that would be right.
const NAMED: [(&str, &str, &[DelistKind]); 4] = [
    ("SIVB", "SVB FINANCIAL", &[DelistKind::Failure, DelistKind::Unknown]),
    ("BBBY", "BED BATH", &[DelistKind::Failure, DelistKind::Unknown]),
    ("FRC", "FIRST REPUBLIC", &[DelistKind::Failure, DelistKind::Unknown]),
    ("TWTR", "TWITTER", &[DelistKind::Buyout]),
];
/// Stocks whose daily returns are compared with the Stooq store's.
const COMPARE: [&str; 3] = ["AAPL", "IBM", "KO"];
/// A day matches Stooq when the two daily returns differ by less than this (0.1 percentage point).
const RETURN_TOLERANCE: f64 = 1e-3;
/// The comparison passes when at least this share of days match.
const MATCH_SHARE: f64 = 0.95;
/// The diagnostic over every common stock: a day differs when the returns differ by more than this
/// (0.3 point), and a year counts when a stock has at least `MANY_DAYS` such days in it.
const GAP: f64 = 3e-3;
const MANY_DAYS: usize = 3;
/// Fewest common days for a match; fewer than this share of days within GAP: another security.
const MIN_DAYS: usize = 20;
const SAME_SECURITY: f64 = 0.5;
/// "Stooq equals as-traded": Stooq's close within this of the as-traded close on at least
/// `EQUAL_SHARE` of the days (and at least `MIN_EQUAL_DAYS` of them) from 2023 on (`compare_all`'s `as_traded_from`).
const EQUAL_TOLERANCE: f64 = 5e-4;
const EQUAL_SHARE: f64 = 0.95;
const MIN_EQUAL_DAYS: usize = 100;
/// Sharadar's adjustment (closeadj / closeunadj) moving by more than this over those days: dividends.
const DIVIDENDS_MOVE: f64 = 0.01;

/// Runs the checks and prints them; false if any failed.
pub fn run(raw_dir: &Path, stooq_dir: &Path, until: i32) -> Result<bool> {
    let mut raw = sharadar::read_raw(raw_dir)?;
    let built = sharadar::build(&mut raw, None)?;
    let mut problems: Vec<String> = Vec::new();

    println!("== What the import would do ({})", raw_dir.display());
    for line in sharadar::log(&raw, &built) {
        println!("  {line}");
    }
    if !built.shared.is_empty() {
        problems.push(format!("{} tickers are shared by two permatickers: the import would stop", built.shared.len()));
    }
    let calendar = sharadar::CALENDAR.iter().filter(|t| built.symbols.iter().any(|s| !s.fund && s.ticker == **t)).count();
    if calendar < 5 {
        problems.push(format!("only {calendar} of correlation2's 8 calendar stocks: the export would fail"));
    }

    println!("\n== SIVB, BBBY, FRC and TWTR");
    for (ticker, company, expect) in NAMED {
        named(&raw, &built, ticker, company, expect, &mut problems);
    }

    println!("\n== Delisted common stocks by year of their last price (isdelisted = Y; kind from actions within {WINDOW_DAYS} days)");
    delisted_by_year(&built, until);
    listing_status(&raw, &built);

    println!("\n== Categories of the tickers rows (table stocks) and the kind each gets");
    let mut cats: BTreeMap<&str, usize> = BTreeMap::new();
    for m in raw.tickers.iter().filter(|m| m.table == "stocks") {
        *cats.entry(m.category.as_str()).or_default() += 1;
    }
    for (c, n) in &cats {
        let kind = if c.to_ascii_lowercase().contains("common stock") { "stock" } else { "other" };
        println!("  {:<44} {n:>8}  -> {kind}", if c.is_empty() { "(empty)" } else { c });
    }

    println!("\n== Action codes (all rows)");
    for (code, n) in &raw.codes {
        let note = if sharadar::BUYOUT_CODES.contains(&code.as_str()) {
            "  -> buyout"
        } else if sharadar::FAILURE_CODES.contains(&code.as_str()) {
            "  -> failure"
        } else if !KNOWN_CODES.contains(&code.as_str()) {
            "  (unknown to the importer)"
        } else {
            ""
        };
        println!("  {code:<24} {n:>10}{note}");
    }

    println!("\n== Daily returns from closeadj vs the Stooq store ({}), through {}", stooq_dir.display(), date_string(until));
    if stooq_dir.join("prices.bin").exists() {
        let stooq = Store::open(stooq_dir)?;
        for t in COMPARE {
            compare(&built, &stooq, t, until, &mut problems);
        }
        // Stooq stopped adjusting some stocks' dividends around Oct-Nov 2022: equality is tested from 2023 on.
        let wide = compare_all(&built, &stooq, until, days_from_civil(2023, 1, 1));
        print_wide(&wide, until);
    } else {
        println!("  no store there; skipped");
    }

    println!();
    if problems.is_empty() {
        println!("All checks passed.");
    } else {
        println!("PROBLEMS:");
        for p in &problems {
            println!("  - {p}");
        }
    }
    Ok(problems.is_empty())
}

fn opt_date(d: Option<i32>) -> String {
    d.map(date_string).unwrap_or_else(|| "-".into())
}

/// One of the named companies: under its ticker, a renamed ticker (the ticker followed by digits or
/// a Q), or its name.
fn named(raw: &Raw, built: &Built, ticker: &str, company: &str, expect: &[DelistKind], problems: &mut Vec<String>) {
    let renamed = |t: &str| t.strip_prefix(ticker).is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit() || c == 'Q'));
    let by_name = |name: &str| name.to_ascii_uppercase().contains(company);
    let symbols: Vec<usize> = (0..built.symbols.len())
        .filter(|&i| {
            let s = &built.symbols[i];
            !s.fund && (s.ticker == ticker || renamed(&s.ticker) || s.meta.is_some_and(|m| by_name(&raw.tickers[m].name)))
        })
        .collect();
    let rows: Vec<usize> = (0..raw.tickers.len())
        .filter(|&i| {
            let m = &raw.tickers[i];
            m.table == "stocks" && (m.ticker == ticker || renamed(&m.ticker) || by_name(&m.name))
        })
        .collect();
    println!("  {ticker} ({company}):");
    if symbols.is_empty() && rows.is_empty() {
        println!("    not found in stocks or tickers");
        problems.push(format!("{ticker} is in neither stocks nor tickers"));
        return;
    }
    for &i in &rows {
        let m = &raw.tickers[i];
        println!(
            "    tickers: {} permaticker {}, {:?}, {}, isdelisted {}, {}, prices {} .. {}",
            m.ticker,
            m.permaticker,
            m.name,
            m.exchange,
            if m.delisted { "Y" } else { "N" },
            m.category,
            opt_date(m.first_price),
            opt_date(m.last_price)
        );
    }
    let mut verdict = None;
    for &i in &symbols {
        let (s, series, unadj) = (&built.symbols[i], &built.series[i], &built.close_unadj[i]);
        let n = series.date.len();
        println!(
            "    stocks: {} {} rows {} .. {}, last close {:.4} (closeadj), {:.4} as traded; {:?}",
            s.ticker,
            n,
            date_string(series.info.first),
            date_string(series.info.last),
            series.close[n - 1],
            unadj[n - 1],
            s.kind
        );
        let last = series.info.last;
        let near: Vec<String> = raw
            .actions
            .get(&s.ticker)
            .map(|v| {
                v.iter()
                    .filter(|a| (a.date - last).abs() <= WINDOW_DAYS)
                    .map(|a| format!("{} {}{}", date_string(a.date), a.code, a.contraticker.as_ref().map(|c| format!(" (contraticker {c})")).unwrap_or_default()))
                    .collect()
            })
            .unwrap_or_default();
        println!("    actions within {WINDOW_DAYS} days of {}: {}", date_string(last), if near.is_empty() { "none".into() } else { near.join("; ") });
        match &s.delisting {
            Some(d) => {
                let ok = expect.contains(&d.kind);
                println!(
                    "    delisting kind: {} ({}) -> {}",
                    d.kind.name(),
                    if d.codes.is_empty() { "no codes".into() } else { d.codes.join(",") },
                    if ok { "as expected" } else { "NOT the expected kind" }
                );
                if s.ticker == ticker || verdict.is_none() {
                    verdict = Some((s.ticker.clone(), d.kind, ok));
                }
            }
            None => println!("    not delisted in the store ({})", if s.kind == Kind::Stock { "isdelisted = N" } else { "not a common stock" }),
        }
    }
    match verdict {
        Some((_, _, true)) => {}
        Some((t, k, false)) => problems.push(format!(
            "{ticker}: {t} is delisted as {}, expected {}",
            k.name(),
            expect.iter().map(|k| k.name()).collect::<Vec<_>>().join(" or ")
        )),
        None => problems.push(format!("{ticker}: no delisted common stock found under that ticker or name")),
    }
}

/// Delisted common stocks by year of their last row and kind; later than `until`, one total.
fn delisted_by_year(built: &Built, until: i32) {
    let mut years: BTreeMap<i32, [usize; 3]> = BTreeMap::new();
    let mut later = 0;
    for d in built.symbols.iter().filter_map(|s| s.delisting.as_ref()) {
        if d.last > until {
            later += 1;
            continue;
        }
        years.entry(civil_from_days(d.last).0).or_default()[d.kind.code() as usize - 1] += 1;
    }
    println!("  {:<6} {:>8} {:>8} {:>8} {:>8}", "year", "buyout", "failure", "unknown", "all");
    let mut total = [0usize; 3];
    for (y, n) in &years {
        println!("  {y:<6} {:>8} {:>8} {:>8} {:>8}", n[0], n[1], n[2], n.iter().sum::<usize>());
        for k in 0..3 {
            total[k] += n[k];
        }
    }
    println!("  {:<6} {:>8} {:>8} {:>8} {:>8}", "all", total[0], total[1], total[2], total.iter().sum::<usize>());
    if later > 0 {
        println!("  and {later} delisted after {} (not broken down: the locked year is not looked at)", date_string(until));
    }
}

/// isdelisted against the price rows, and tickers that look renamed.
fn listing_status(raw: &Raw, built: &Built) {
    let end = built.series.iter().map(|s| s.info.last).max().unwrap_or(0);
    let stale = |s: &sharadar::Symbol, last: i32| {
        s.kind == Kind::Stock && s.delisting.is_none() && s.meta.is_some_and(|m| !raw.tickers[m].delisted) && last < end - 30
    };
    let n_stale = (0..built.symbols.len()).filter(|&i| stale(&built.symbols[i], built.series[i].info.last)).count();
    let n_dead_at_end = built.symbols.iter().filter_map(|s| s.delisting.as_ref()).filter(|d| d.last >= end - 3).count();
    let digits: Vec<&str> = built
        .symbols
        .iter()
        .filter(|s| !s.fund && s.ticker.chars().last().is_some_and(|c| c.is_ascii_digit()))
        .map(|s| s.ticker.as_str())
        .collect();
    println!("  last price date in the data: {}", date_string(end));
    println!("  common stocks with isdelisted = N whose prices stop more than 30 days before it: {n_stale}");
    println!("  delisted common stocks with prices within 3 days of it: {n_dead_at_end}");
    println!(
        "  stocks tickers ending in a digit (Sharadar renames reused tickers of delisted companies?): {}{}",
        digits.len(),
        if digits.is_empty() { String::new() } else { format!(", e.g. {}", digits.iter().take(10).copied().collect::<Vec<_>>().join(", ")) }
    );
    let disagree = &built.stats.last_price_disagrees;
    println!(
        "  tickers.lastpricedate differs from the last stocks row: {}{}",
        disagree.len(),
        if disagree.is_empty() {
            String::new()
        } else {
            format!(", e.g. {}", disagree.iter().take(5).map(|(t, a, b)| format!("{t} {} vs {}", opt_date(*a), date_string(*b))).collect::<Vec<_>>().join(", "))
        }
    );
}

/// Daily returns of `ticker` from closeadj against Stooq's closes, on pairs of consecutive days both have.
fn compare(built: &Built, stooq: &Store, ticker: &str, until: i32, problems: &mut Vec<String>) {
    let Some(i) = built.symbols.iter().position(|s| !s.fund && s.ticker == ticker) else {
        println!("  {ticker}: not in Sharadar's stocks");
        return;
    };
    let Some(id) = stooq.id(ticker) else {
        println!("  {ticker}: not in the Stooq store");
        return;
    };
    let (sd, sc) = (&built.series[i].date, &built.series[i].close);
    let (qd, qc) = (stooq.dates(id), stooq.close(id));
    let mut diffs: Vec<(f64, i32, f64, f64)> = Vec::new();
    let mut k = 0;
    for q in 1..qd.len() {
        if qd[q] > until {
            break;
        }
        while k < sd.len() && sd[k] < qd[q] {
            k += 1;
        }
        // Both have this day and the same previous day.
        if k == 0 || k >= sd.len() || sd[k] != qd[q] || sd[k - 1] != qd[q - 1] {
            continue;
        }
        let (a, b) = (sc[k] as f64 / sc[k - 1] as f64 - 1.0, qc[q] as f64 / qc[q - 1] as f64 - 1.0);
        if a.is_finite() && b.is_finite() {
            diffs.push(((a - b).abs(), qd[q], a, b));
        }
    }
    if diffs.is_empty() {
        println!("  {ticker}: no common days");
        problems.push(format!("{ticker}: no days in common with the Stooq store"));
        return;
    }
    let n = diffs.len();
    let mut sorted: Vec<f64> = diffs.iter().map(|d| d.0).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let within = |tol: f64| sorted.partition_point(|&d| d < tol) as f64 / n as f64;
    diffs.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!(
        "  {ticker}: {n} days ({} .. {}); |difference| < 0.1 point on {:.2}%, < 1 point on {:.2}%; median {:.2e}; worst: {}",
        date_string(diffs.iter().map(|d| d.1).min().unwrap()),
        date_string(diffs.iter().map(|d| d.1).max().unwrap()),
        100.0 * within(RETURN_TOLERANCE),
        100.0 * within(1e-2),
        sorted[n / 2],
        diffs.iter().take(3).map(|d| format!("{} Sharadar {:+.2}% Stooq {:+.2}%", date_string(d.1), 100.0 * d.2, 100.0 * d.3)).collect::<Vec<_>>().join("; ")
    );
    if within(RETURN_TOLERANCE) < MATCH_SHARE {
        problems.push(format!("{ticker}: only {:.1}% of daily returns match Stooq's within 0.1 point", 100.0 * within(RETURN_TOLERANCE)));
    }
}

/// One stock of both stores: its daily returns on the pairs of consecutive days both have
/// (through `until`), and its closes against the as-traded ones from `as_traded_from` on.
struct Pair {
    ticker: String,
    /// The Stooq stock's exchange (its folder: nyse, nasdaq, nysemkt ...).
    exchange: String,
    /// (day, Sharadar's return, Stooq's return)
    returns: Vec<(i32, f64, f64)>,
    /// Days from `as_traded_from` with both closes; of them, Stooq's close within EQUAL_TOLERANCE of
    /// the as-traded close; Sharadar's adjustment factor on the first and the last of them.
    traded_days: usize,
    traded_equal: usize,
    factor: Option<(f64, f64)>,
}

/// The diagnostic over every common stock of both stores.
pub struct Wide {
    pairs: Vec<Pair>,
    /// Tickers in both but several Stooq stocks with that ticker; with too few common days; with
    /// returns that mostly disagree (another company under the same ticker).
    ambiguous: usize,
    no_overlap: usize,
    other_security: usize,
    as_traded_from: i32,
}

/// Matches every Sharadar common stock with the Stooq common stock of the same ticker and compares
/// them (read-only; nothing after `until`).
pub fn compare_all(built: &Built, stooq: &Store, until: i32, as_traded_from: i32) -> Wide {
    let mut by_ticker: HashMap<&str, Vec<usize>> = HashMap::new();
    for (id, s) in stooq.symbols.iter().enumerate() {
        if s.kind == Kind::Stock {
            by_ticker.entry(s.ticker.as_str()).or_default().push(id);
        }
    }
    let mut ambiguous = 0;
    let mut jobs = Vec::new();
    for (i, s) in built.symbols.iter().enumerate() {
        if s.fund || s.kind != Kind::Stock {
            continue;
        }
        match by_ticker.get(s.ticker.as_str()).map(|v| v.as_slice()) {
            Some([id]) => jobs.push((i, *id)),
            Some(_) => ambiguous += 1,
            None => {}
        }
    }
    let found: Vec<Pair> = jobs
        .par_iter()
        .map(|&(i, id)| {
            let (sd, sc, su) = (&built.series[i].date, &built.series[i].close, &built.close_unadj[i]);
            let (qd, qc) = (stooq.dates(id), stooq.close(id));
            let mut p = Pair {
                ticker: built.symbols[i].ticker.clone(),
                exchange: stooq.symbols[id].exchange.clone(),
                returns: Vec::new(),
                traded_days: 0,
                traded_equal: 0,
                factor: None,
            };
            let mut k = 0;
            for q in 0..qd.len() {
                if qd[q] > until {
                    break;
                }
                while k < sd.len() && sd[k] < qd[q] {
                    k += 1;
                }
                if k >= sd.len() || sd[k] != qd[q] {
                    continue;
                }
                if q > 0 && k > 0 && sd[k - 1] == qd[q - 1] {
                    let (a, b) = (sc[k] as f64 / sc[k - 1] as f64 - 1.0, qc[q] as f64 / qc[q - 1] as f64 - 1.0);
                    if a.is_finite() && b.is_finite() {
                        p.returns.push((qd[q], a, b));
                    }
                }
                let u = su[k] as f64;
                if qd[q] >= as_traded_from && u > 0.0 && qc[q] > 0.0 {
                    p.traded_days += 1;
                    p.traded_equal += usize::from((qc[q] as f64 / u - 1.0).abs() < EQUAL_TOLERANCE);
                    let f = sc[k] as f64 / u;
                    p.factor = Some(p.factor.map_or((f, f), |(first, _)| (first, f)));
                }
            }
            p
        })
        .collect();
    let (mut pairs, mut no_overlap, mut other_security) = (Vec::new(), 0, 0);
    for p in found {
        let n = p.returns.len();
        if n < MIN_DAYS {
            no_overlap += 1;
        } else if (p.returns.iter().filter(|r| (r.1 - r.2).abs() <= GAP).count() as f64) < SAME_SECURITY * n as f64 {
            other_security += 1;
        } else {
            pairs.push(p);
        }
    }
    Wide { pairs, ambiguous, no_overlap, other_security, as_traded_from }
}

/// What the diagnostic reports.
struct WideSummary {
    /// (year, stocks compared, stocks with MANY_DAYS or more days differing by more than GAP)
    years: Vec<(i32, usize, usize)>,
    /// Stocks whose Stooq close equals the as-traded close from `as_traded_from` on.
    equal: usize,
    /// Of those, the ones that paid dividends then: (ticker, how far Sharadar's adjustment moved).
    unadjusted: Vec<(String, f64)>,
    /// Dividend payers then, by Stooq exchange: (left unadjusted, in all).
    payers: BTreeMap<String, (usize, usize)>,
    /// Each stock's biggest single-day gap from 2005 on: (ticker, day, Sharadar's, Stooq's return), largest first.
    gaps: Vec<(String, i32, f64, f64)>,
}

fn summarize(w: &Wide, until: i32) -> WideSummary {
    let years = (2005..=civil_from_days(until).0)
        .map(|y| {
            let (from, to) = (days_from_civil(y, 1, 1), days_from_civil(y + 1, 1, 1));
            let (mut compared, mut differing) = (0, 0);
            for p in &w.pairs {
                let days: Vec<&(i32, f64, f64)> = p.returns.iter().filter(|r| r.0 >= from && r.0 < to).collect();
                if !days.is_empty() {
                    compared += 1;
                    differing += usize::from(days.iter().filter(|r| (r.1 - r.2).abs() > GAP).count() >= MANY_DAYS);
                }
            }
            (y, compared, differing)
        })
        .collect();
    let moved = |p: &Pair| p.factor.map_or(0.0, |(first, last)| last / first - 1.0);
    let is_equal = |p: &Pair| p.traded_days >= MIN_EQUAL_DAYS && p.traded_equal as f64 >= EQUAL_SHARE * p.traded_days as f64;
    let paid = |p: &Pair| p.traded_days >= MIN_EQUAL_DAYS && moved(p) > DIVIDENDS_MOVE;
    let mut unadjusted: Vec<(String, f64)> = w.pairs.iter().filter(|p| is_equal(p) && paid(p)).map(|p| (p.ticker.clone(), moved(p))).collect();
    unadjusted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut payers: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for p in w.pairs.iter().filter(|p| paid(p)) {
        let e = payers.entry(p.exchange.clone()).or_default();
        e.0 += usize::from(is_equal(p));
        e.1 += 1;
    }
    let from = days_from_civil(2005, 1, 1);
    let mut gaps: Vec<(String, i32, f64, f64)> = w
        .pairs
        .iter()
        .filter_map(|p| {
            p.returns
                .iter()
                .filter(|r| r.0 >= from)
                .max_by(|a, b| (a.1 - a.2).abs().total_cmp(&(b.1 - b.2).abs()))
                .map(|r| (p.ticker.clone(), r.0, r.1, r.2))
        })
        .collect();
    gaps.sort_by(|a, b| (b.2 - b.3).abs().total_cmp(&(a.2 - a.3).abs()).then_with(|| a.0.cmp(&b.0)));
    WideSummary { years, equal: w.pairs.iter().filter(|p| is_equal(p)).count(), unadjusted, payers, gaps }
}

fn print_wide(w: &Wide, until: i32) {
    let s = summarize(w, until);
    println!(
        "  Every common stock in both stores: {} matched by ticker; left out {} with several Stooq stocks of that ticker, {} with fewer than {MIN_DAYS} common days, {} whose returns mostly disagree (another company under the ticker?)",
        w.pairs.len(),
        w.ambiguous,
        w.no_overlap,
        w.other_security
    );
    println!("  Stocks whose daily returns differ by more than {:.1} point on {MANY_DAYS} or more days of the year:", 100.0 * GAP);
    println!("  {:<6} {:>9} {:>9} {:>7}", "year", "compared", "differing", "share");
    let last = civil_from_days(until).0;
    for &(y, compared, differing) in &s.years {
        let label = if y == last { format!("{y}*") } else { y.to_string() };
        println!("  {label:<6} {compared:>9} {differing:>9} {:>6.1}%", 100.0 * differing as f64 / compared.max(1) as f64);
    }
    println!("  (* through {}: half a year, so about half the ex-dividend days)", date_string(until));
    println!(
        "  Stooq's close equals the as-traded close (within {:.2}% on {:.0}% of the days) from {} on: {} stocks; {} of them paid dividends then (Sharadar's adjustment moved by more than {:.0}%), so Stooq's returns leave those dividends out{}",
        100.0 * EQUAL_TOLERANCE,
        100.0 * EQUAL_SHARE,
        date_string(w.as_traded_from),
        s.equal,
        s.unadjusted.len(),
        100.0 * DIVIDENDS_MOVE,
        if s.unadjusted.is_empty() {
            String::new()
        } else {
            format!(". Largest: {}", s.unadjusted.iter().take(15).map(|(t, m)| format!("{t} {:.1}%", 100.0 * m)).collect::<Vec<_>>().join(", "))
        }
    );
    println!(
        "  Dividend payers then, left unadjusted by Stooq, by Stooq exchange: {}",
        s.payers.iter().map(|(e, (u, n))| format!("{e} {u} of {n}")).collect::<Vec<_>>().join("; ")
    );
    println!("  Biggest single-day gaps from 2005 on, one per stock (spin-offs, splits, bad prints):");
    println!("  {:<8} {:<10} {:>9} {:>9} {:>8}", "ticker", "day", "Sharadar", "Stooq", "gap");
    for (t, d, a, b) in s.gaps.iter().take(15) {
        println!("  {t:<8} {} {:>8.2}% {:>8.2}% {:>8.2}", date_string(*d), 100.0 * a, 100.0 * b, 100.0 * (a - b));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::sharadar::tests::{fake, scratch, universe, write_raw};

    fn rng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A Stooq store's series for `ticker` from Sharadar's rows: adjusted closes, or as-traded ones.
    fn stooq_series(built: &Built, ticker: &str, exchange: &str, as_traded: bool) -> c2_data::store::Series {
        let i = built.symbols.iter().position(|s| s.ticker == ticker).unwrap();
        let s = &built.series[i];
        let close = if as_traded { built.close_unadj[i].clone() } else { s.close.clone() };
        c2_data::store::Series {
            info: c2_data::store::SymbolInfo { exchange: exchange.into(), source: String::new(), ..s.info.clone() },
            date: s.date.clone(),
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            volume: s.volume.clone(),
            close,
        }
    }

    #[test]
    fn every_common_stock_against_stooq() {
        let dir = scratch("wide");
        write_raw(&dir.join("raw"), &universe());
        // JNJ pays a dividend every quarter: Sharadar's adjustment factor steps up 1% on each one.
        let csv = std::fs::read_to_string(dir.join("raw/stocks.csv")).unwrap();
        let first = days_from_civil(2003, 1, 2);
        let lines: Vec<String> = csv
            .lines()
            .map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                if f[0] != "JNJ" {
                    return l.to_string();
                }
                let quarters = (c2_data::parse_date(f[1]).unwrap() - first) / 91;
                let adj = f[5].parse::<f64>().unwrap() * 0.75 * 1.01f64.powi(quarters);
                format!("{},{},{},{},{},{},{},{adj:.4},{},{}", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[8], f[9])
            })
            .collect();
        std::fs::write(dir.join("raw/stocks.csv"), lines.join("\n")).unwrap();
        let mut raw = sharadar::read_raw(&dir.join("raw")).unwrap();
        let built = sharadar::build(&mut raw, None).unwrap();

        let mut ibm = stooq_series(&built, "IBM", "nyse", false);
        let bump = ibm.date.iter().position(|&d| d >= days_from_civil(2005, 9, 1)).unwrap();
        ibm.close[bump] *= 1.05;
        let mut other = stooq_series(&built, "MMM", "nyse", false);
        let mut g = 0x2545_F491_4F6C_DD1Du64;
        let mut p = 30.0f32;
        for c in other.close.iter_mut() {
            p *= 1.0 + 0.04 * ((rng(&mut g) >> 11) as f32 / (1u64 << 53) as f32 - 0.5);
            *c = p;
        }
        let mut short = stooq_series(&built, "ADRCO", "nyse", false);
        short.date.truncate(10);
        for c in [&mut short.open, &mut short.high, &mut short.low, &mut short.close, &mut short.volume] {
            c.truncate(10);
        }
        short.info.rows = 10;
        let series = vec![
            ibm,
            stooq_series(&built, "KO", "nyse", true),
            stooq_series(&built, "JNJ", "nyse", true),
            stooq_series(&built, "PG", "nyse", false),
            stooq_series(&built, "PG", "nasdaq", false),
            other,
            short,
        ];
        c2_data::store::write(&dir.join("stooq"), &series).unwrap();
        let stooq = Store::open(&dir.join("stooq")).unwrap();

        let until = days_from_civil(2009, 12, 31);
        let w = compare_all(&built, &stooq, until, days_from_civil(2006, 1, 1));
        let mut matched: Vec<&str> = w.pairs.iter().map(|p| p.ticker.as_str()).collect();
        matched.sort();
        assert_eq!(matched, ["IBM", "JNJ", "KO"]);
        assert_eq!((w.ambiguous, w.no_overlap, w.other_security), (1, 1, 1));
        let s = summarize(&w, until);
        // JNJ's unadjusted dividends differ on four days a year; IBM's one bad close only on two days.
        assert_eq!(s.years, (2005..=2009).map(|y| (y, 3, 1)).collect::<Vec<_>>());
        assert_eq!(s.equal, 2, "KO and JNJ equal their as-traded closes");
        assert_eq!(s.unadjusted.iter().map(|u| u.0.as_str()).collect::<Vec<_>>(), ["JNJ"]);
        assert!((s.unadjusted[0].1 - (1.01f64.powi(15) - 1.0)).abs() < 0.02, "{:?}", s.unadjusted);
        assert_eq!(s.payers, BTreeMap::from([("nyse".to_string(), (1, 1))]));
        assert_eq!((s.gaps[0].0.as_str(), s.gaps[0].1), ("IBM", ibm_day(&stooq, bump)));
        assert!((s.gaps[0].3 - s.gaps[0].2 - 0.05).abs() < 0.01, "{:?}", s.gaps[0]);
        print_wide(&w, until);
    }

    fn ibm_day(stooq: &Store, k: usize) -> i32 {
        stooq.dates(stooq.id("IBM").unwrap())[k]
    }

    #[test]
    fn checks_find_named_companies_and_compare_returns() {
        let dir = scratch("check");
        let mut stocks = universe();
        stocks.push(fake("TWTR", "Domestic Common Stock", true, "2003-01-02", "2007-10-26"));
        write_raw(&dir.join("raw"), &stocks);
        let mut actions = std::fs::read_to_string(dir.join("raw/actions.csv")).unwrap();
        actions += "2007-10-28,acquisitionby,TWTR,TWITTER INC,,N/A,N/A\n";
        std::fs::write(dir.join("raw/actions.csv"), actions).unwrap();
        // A Stooq-like store with IBM's adjusted closes, one day off by 5%.
        let mut raw = sharadar::read_raw(&dir.join("raw")).unwrap();
        let built = sharadar::build(&mut raw, None).unwrap();
        let i = built.symbols.iter().position(|s| s.ticker == "IBM").unwrap();
        let mut s = c2_data::store::Series {
            info: built.series[i].info.clone(),
            date: built.series[i].date.clone(),
            open: built.series[i].open.clone(),
            high: built.series[i].high.clone(),
            low: built.series[i].low.clone(),
            close: built.series[i].close.clone(),
            volume: built.series[i].volume.clone(),
        };
        s.close[100] *= 1.05;
        c2_data::store::write(&dir.join("stooq"), &[s]).unwrap();
        let ok = run(&dir.join("raw"), &dir.join("stooq"), days_from_civil(2008, 12, 31)).unwrap();
        // SIVB, BBBY and FRC are not in the synthetic data: the checks fail, and say so.
        assert!(!ok);

        let mut problems = Vec::new();
        named(&raw, &built, "TWTR", "TWITTER", &[DelistKind::Buyout], &mut problems);
        assert!(problems.is_empty(), "{problems:?}");
        named(&raw, &built, "BUYCO", "NOBODY", &[DelistKind::Failure], &mut problems);
        assert_eq!(problems, ["BUYCO: BUYCO is delisted as buyout, expected failure"]);
        problems.clear();
        compare(&built, &Store::open(&dir.join("stooq")).unwrap(), "IBM", days_from_civil(2008, 12, 31), &mut problems);
        assert!(problems.is_empty(), "two bad days of ~1500 still pass: {problems:?}");
    }
}
