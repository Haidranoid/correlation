//! A Sharadar store's sidecars (docs/sharadar.md: `close_unadj.f32`, `delist.tsv`, `source.json`), aligned to
//! `plan.var` for the export and the grader. A store without them (correlation2's Stooq store) has no extras, and
//! the export and the grader then do exactly what they did before.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use c2_census::variable::Variable;
use c2_data::Store;
use rayon::prelude::*;
use serde_json::Value;

use crate::sharadar::{DelistKind, SIDECARS, SOURCE_FORMAT, strict_date};

pub struct Extras {
    /// As-traded closes, `[N, T]` in `Variable.close`'s layout: the store's closeunadj on the days a stock has a
    /// row, NaN elsewhere (no bad-print repair).
    pub close_unadj: Vec<f32>,
    /// Per variable column: the day index of a delisted stock's last finite close in `Variable.close` (on or before
    /// delist.tsv's last_date), else -1.
    pub delist_day: Vec<i32>,
    /// Per variable column: 0 listed, else `DelistKind::code` (1 buyout, 2 failure, 3 unknown).
    pub delist_kind: Vec<i8>,
    /// source.json as the importer wrote it.
    pub source: Value,
    /// Delisted stocks among the variable's columns, by kind (buyout, failure, unknown).
    pub delisted: [usize; 3],
    /// Delisted columns whose last finite close falls on an earlier day than last_date (a last row off the calendar
    /// or a close blanked by the bad-print repair), and the largest such gap in calendar days.
    pub moved: usize,
    pub moved_max_days: i32,
    /// Delisted columns without any finite close (left as listed: delist_day -1, kind 0).
    pub without_close: usize,
}

impl Extras {
    /// The extras of the store in `dir`, aligned to `var`; None when the store has no sidecars. Some sidecars but
    /// not all (an interrupted import), or sidecars written for another store, are errors.
    pub fn load(store: &Store, dir: &Path, var: &Variable) -> Result<Option<Extras>> {
        let present: Vec<&str> = SIDECARS.iter().copied().filter(|f| dir.join(f).exists()).collect();
        if present.is_empty() {
            return Ok(None);
        }
        ensure!(
            present.len() == SIDECARS.len(),
            "{} has {} but not all of {}: an incomplete Sharadar import; run import-sharadar again",
            dir.display(),
            present.join(", "),
            SIDECARS.join(", ")
        );
        let source: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("source.json"))?).with_context(|| format!("reading {}", dir.join("source.json").display()))?;
        ensure!(source["format"] == SOURCE_FORMAT, "{}: format {}, expected {SOURCE_FORMAT}", dir.join("source.json").display(), source["format"]);
        ensure!(
            source["store"]["rows"].as_u64() == Some(store.rows() as u64),
            "{}: source.json describes {} rows, prices.bin has {}: the store was rewritten after the import",
            dir.display(),
            source["store"]["rows"],
            store.rows()
        );
        let tsv_hash = format!("{:016x}", crate::export::fnv(&std::fs::read(dir.join("symbols.tsv"))?));
        ensure!(
            source["store"]["symbols_tsv_fnv64"].as_str() == Some(tsv_hash.as_str()),
            "{}: symbols.tsv changed after the import (source.json has another fingerprint)",
            dir.display()
        );

        let bytes = std::fs::read(dir.join("close_unadj.f32"))?;
        ensure!(bytes.len() == 4 * store.rows(), "{}: {} bytes for {} rows", dir.join("close_unadj.f32").display(), bytes.len(), store.rows());
        let mut offsets = Vec::with_capacity(store.symbols.len());
        let mut o = 0usize;
        for id in 0..store.symbols.len() {
            offsets.push(o);
            o += store.dates(id).len();
        }
        let (n, t) = (var.n, var.t);
        let mut close_unadj = vec![f32::NAN; n * t];
        // The same walk as correlation2's Panel::build, over the store rows of each column's symbol.
        close_unadj.par_chunks_mut(t.max(1)).zip(var.ids.par_iter()).for_each(|(row, &id)| {
            let dates = store.dates(id);
            let mut k = dates.partition_point(|&x| x < *var.days.first().unwrap_or(&0));
            for (r, &day) in var.days.iter().enumerate() {
                while k < dates.len() && dates[k] < day {
                    k += 1;
                }
                if k < dates.len() && dates[k] == day {
                    let p = 4 * (offsets[id] + k);
                    let x = f32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
                    if x.is_finite() && x > 0.0 {
                        row[r] = x;
                    }
                }
            }
        });

        let path = dir.join("delist.tsv");
        let text = std::fs::read_to_string(&path)?;
        let mut lines = text.lines();
        ensure!(lines.next() == Some("ticker\tlast_date\tkind\tactions"), "{}: unexpected header", path.display());
        let mut by_ticker: HashMap<&str, (i32, DelistKind)> = HashMap::new();
        for (i, line) in lines.enumerate() {
            let f: Vec<&str> = line.split('\t').collect();
            ensure!(f.len() == 4, "{} line {}: {} fields, expected 4", path.display(), i + 2, f.len());
            let last = strict_date(f[1].as_bytes()).with_context(|| format!("{} line {}: bad date {:?}", path.display(), i + 2, f[1]))?;
            let kind = DelistKind::parse(f[2]).with_context(|| format!("{} line {}: unknown kind {:?}", path.display(), i + 2, f[2]))?;
            ensure!(store.id(f[0]).is_some(), "{} line {}: {} is not in the store", path.display(), i + 2, f[0]);
            ensure!(by_ticker.insert(f[0], (last, kind)).is_none(), "{} names {} twice", path.display(), f[0]);
        }
        let mut x = Extras {
            close_unadj,
            delist_day: vec![-1; n],
            delist_kind: vec![0; n],
            source,
            delisted: [0; 3],
            moved: 0,
            moved_max_days: 0,
            without_close: 0,
        };
        for (j, &id) in var.ids.iter().enumerate() {
            let Some(&(last, kind)) = by_ticker.get(store.symbols[id].ticker.as_str()) else { continue };
            // The last column on or before last_date with a finite close: the close the delisting rule uses.
            let end = var.days.partition_point(|&d| d <= last);
            let close = &var.close[j * t..(j + 1) * t];
            match (0..end).rev().find(|&d| close[d].is_finite()) {
                Some(l) => {
                    x.delist_day[j] = l as i32;
                    x.delist_kind[j] = kind.code();
                    x.delisted[kind.code() as usize - 1] += 1;
                    if var.days[l] != last {
                        x.moved += 1;
                        x.moved_max_days = x.moved_max_days.max(last - var.days[l]);
                    }
                }
                None => x.without_close += 1,
            }
        }
        Ok(Some(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npy::{self, Data};
    use crate::sharadar::tests::{Fake, price, scratch, universe, write_raw};
    use c2_census::grid::Config;
    use c2_census::run::Plan;

    fn read<T>(path: &Path, f: impl Fn(Data) -> Option<Vec<T>>) -> (Vec<usize>, Vec<T>) {
        let a = npy::read(path).unwrap();
        (a.header.shape.clone(), f(a.data).expect("dtype"))
    }

    /// The whole path: synthetic Sharadar CSVs -> import -> correlation2's Plan::new (calendar, Panel::build,
    /// bad-print repair) -> extras -> export, and the arrays checked against the CSV values.
    #[test]
    fn export_aligns_the_extras_through_the_real_plan() {
        let dir = scratch("export");
        let stocks = universe();
        write_raw(&dir.join("raw"), &stocks);
        let store_dir = dir.join("store");
        crate::sharadar::import(&dir.join("raw"), &store_dir, None).unwrap();
        let store = Store::open(&store_dir).unwrap();
        let plan = Plan::new(&store, Config::full()).unwrap();
        let var = &plan.var;
        let (n, t) = (var.n, var.t);
        let x = Extras::load(&store, &store_dir, var).unwrap().expect("a Sharadar store has extras");
        assert_eq!(x.delisted, [1, 2, 2]);
        assert_eq!((x.moved, x.without_close), (0, 0));
        let out = dir.join("export");
        crate::export::export(&store, &plan, &store_dir, &out, "test", Some(&x)).unwrap();

        let tickers: Vec<String> = std::fs::read_to_string(out.join("symbols.tsv")).unwrap().lines().skip(1).map(|l| l.split('\t').nth(2).unwrap().to_string()).collect();
        assert_eq!(tickers.len(), n);
        // Common stocks alive in 2005 or later; preferred, warrants and stocks without a tickers row are not columns.
        for t in ["IBM", "BUYCO", "GONECO", "ADRCO", "RVS"] {
            assert!(tickers.iter().any(|x| x == t), "{t}");
        }
        for t in ["PREF", "WARR", "NOMETA", "SPY"] {
            assert!(!tickers.iter().any(|x| x == t), "{t}");
        }
        let (shape, unadj) = read(&out.join("close_unadj.npy"), |d| if let Data::F4(v) = d { Some(v) } else { None });
        assert_eq!(shape, [n, t]);
        let (shape, close) = read(&out.join("close.npy"), |d| if let Data::F4(v) = d { Some(v) } else { None });
        assert_eq!(shape, [n, t]);
        let (shape, dday) = read(&out.join("delist_day.npy"), |d| if let Data::I4(v) = d { Some(v) } else { None });
        assert_eq!(shape, [n]);
        let (shape, dkind) = read(&out.join("delist_kind.npy"), |d| if let Data::I1(v) = d { Some(v) } else { None });
        assert_eq!(shape, [n]);
        let (_, days) = read(&out.join("days.npy"), |d| if let Data::I4(v) = d { Some(v) } else { None });

        let fake = |t: &str| stocks.iter().find(|f| f.ticker == t).unwrap();
        let last_of = |f: &Fake| f.last;
        for (j, ticker) in tickers.iter().enumerate() {
            let f = fake(ticker);
            for (d, &day) in days.iter().enumerate() {
                let u = unadj[j * t + d];
                let on = day >= f.first && day <= f.last && (day + 3).rem_euclid(7) < 5 && !(ticker == "KO" && day == c2_data::parse_date("2005-03-02").unwrap());
                if !on {
                    assert!(u.is_nan(), "{ticker} {}: {u}", c2_data::date_string(day));
                    continue;
                }
                let want = if ticker == "IBM" && day == c2_data::parse_date("2005-03-01").unwrap() { 77.7 } else { (price(f, day).0 * f.unadj as f64) as f32 };
                assert!((u - want).abs() <= 1e-3 * want.max(1.0), "{ticker} {}: {u} vs {want}", c2_data::date_string(day));
            }
            let (kind, last) = match ticker.as_str() {
                "BUYCO" => (1, Some(last_of(f))),
                "BANKCO" | "BOTHCO" => (2, Some(last_of(f))),
                "GONECO" | "OLDCO" => (3, Some(last_of(f))),
                _ => (0, None),
            };
            assert_eq!(dkind[j], kind, "{ticker}");
            match last {
                Some(l) => {
                    let d = dday[j] as usize;
                    assert_eq!(days[d], l, "{ticker}: delist_day is the column of the last price date");
                    assert!(close[j * t + d].is_finite(), "{ticker}: a finite close at delist_day");
                    assert!(close[j * t + d + 1..(j + 1) * t].iter().all(|c| c.is_nan()), "{ticker}: no close after it");
                }
                None => assert_eq!(dday[j], -1, "{ticker}"),
            }
        }
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(out.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["extras"].as_object().unwrap().keys().collect::<Vec<_>>(), ["close_unadj.npy", "delist_day.npy", "delist_kind.npy"]);
        assert_eq!(meta["source"], x.source);
        assert_eq!(meta["delisted"]["failure"], 2);
        assert!(meta["variable"].as_str().unwrap().contains("Sharadar closeadj"));

        // The same store without sidecars (as a Stooq store has none): no extras, the same base arrays, and an
        // earlier export's extra arrays are removed.
        let bare = dir.join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        for f in ["prices.bin", "symbols.tsv"] {
            std::fs::copy(store_dir.join(f), bare.join(f)).unwrap();
        }
        let bare_store = Store::open(&bare).unwrap();
        assert!(Extras::load(&bare_store, &bare, var).unwrap().is_none());
        let out2 = dir.join("export2");
        crate::export::export(&store, &plan, &store_dir, &out2, "test", Some(&x)).unwrap();
        crate::export::export(&bare_store, &plan, &bare, &out2, "test", None).unwrap();
        for f in ["days.npy", "symbols.tsv", "v.npy", "close.npy", "dollar_volume.npy", "market.npy"] {
            assert_eq!(std::fs::read(out.join(f)).unwrap(), std::fs::read(out2.join(f)).unwrap(), "{f}");
        }
        for (f, _) in crate::export::EXTRAS {
            assert!(!out2.join(f).exists(), "{f} left behind");
        }
        let meta2: Value = serde_json::from_str(&std::fs::read_to_string(out2.join("meta.json")).unwrap()).unwrap();
        for k in ["extras", "source", "delisted"] {
            assert!(meta2.get(k).is_none(), "{k}");
        }
        assert_eq!(meta2["variable"], crate::export::VARIABLE);
    }

    #[test]
    fn incomplete_or_foreign_sidecars_are_refused() {
        let dir = scratch("refuse");
        write_raw(&dir.join("raw"), &universe());
        let store_dir = dir.join("store");
        crate::sharadar::import(&dir.join("raw"), &store_dir, None).unwrap();
        let store = Store::open(&store_dir).unwrap();
        let plan = Plan::new(&store, Config::full()).unwrap();
        // Another import, with fewer rows, leaves sidecars that do not describe this prices.bin.
        let source = std::fs::read_to_string(store_dir.join("source.json")).unwrap();
        let v: Value = serde_json::from_str(&source).unwrap();
        let rows = v["store"]["rows"].as_u64().unwrap();
        std::fs::write(store_dir.join("source.json"), source.replace(&format!("\"rows\": {rows}"), &format!("\"rows\": {}", rows - 1))).unwrap();
        let e = Extras::load(&store, &store_dir, &plan.var).err().unwrap().to_string();
        assert!(e.contains("rewritten after the import"), "{e}");
        std::fs::write(store_dir.join("source.json"), &source).unwrap();
        assert!(Extras::load(&store, &store_dir, &plan.var).unwrap().is_some());
        std::fs::remove_file(store_dir.join("delist.tsv")).unwrap();
        let e = Extras::load(&store, &store_dir, &plan.var).err().unwrap().to_string();
        assert!(e.contains("incomplete Sharadar import"), "{e}");
    }
}
