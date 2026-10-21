//! `stockai-judge export`: writes correlation2's `plan.var` for Python (docs/c2-export.md).
//!
//! `Variable` matrices are column-major `t x n` (`v[j * t + d]`), which is the memory layout of a
//! C-order `[N, T]` array, so `v.npy`, `close.npy` and `dollar_volume.npy` are byte copies.
//!
//! A Sharadar store (docs/sharadar.md) adds `close_unadj.npy`, `delist_day.npy` and `delist_kind.npy`, listed in
//! meta.json under `extras` with the store's `source`. A store without sidecars (Stooq) writes exactly what it did
//! before.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use c2_census::run::Plan;
use c2_data::Store;

use crate::extras::Extras;
use crate::npy;

pub const FORMAT: &str = "c2-export-1";

/// What the census variable and the other arrays are.
pub const VARIABLE: &str = "v: census variable = (market-residual daily log return - its trailing 252-day mean) / \
trailing 252-day residual sd (beta, mean and sd from days d-252..d-1), clipped +-5, NaN where unavailable; so \
minus the trailing mean over sd is a part of v[d+1] known at d; close: Stooq split- and dividend-adjusted closes as \
of the download, after correlation2's bad-print repair (which looks at close[d+1]), NaN where missing; \
dollar_volume: close x volume; market: equal-weight mean daily log return of the liquid stocks";

/// VARIABLE for a Sharadar store: the closes are Sharadar's closeadj.
pub const VARIABLE_SHARADAR: &str = "v: census variable = (market-residual daily log return - its trailing 252-day mean) / \
trailing 252-day residual sd (beta, mean and sd from days d-252..d-1), clipped +-5, NaN where unavailable; so \
minus the trailing mean over sd is a part of v[d+1] known at d; close: Sharadar closeadj (split-, dividend- and \
spinoff-adjusted) as of the download, after correlation2's bad-print repair (which looks at close[d+1]), NaN where \
missing; dollar_volume: close x volume (split-adjusted volume); market: equal-weight mean daily log return of the \
liquid stocks (their $5 test on adjusted closes)";

/// The extra arrays of a Sharadar export, and what they hold (meta.json's `extras`).
pub const EXTRAS: [(&str, &str); 3] = [
    ("close_unadj.npy", "float32 [N, T]: as-traded closes (Sharadar closeunadj) on the days the stock has a row, NaN elsewhere; no bad-print repair"),
    ("delist_day.npy", "int32 [N]: for a delisted stock (delist.tsv), the column of its last finite close in close.npy (on or before its last price date), else -1"),
    ("delist_kind.npy", "int8 [N]: 0 listed, 1 buyout, 2 failure, 3 unknown (delist.tsv)"),
];

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01b3;

/// FNV-1a 64 (as correlation2's run.rs uses for manifests).
pub fn fnv(bytes: &[u8]) -> u64 {
    fnv_extend(FNV_OFFSET, bytes)
}

fn fnv_extend(h: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(FNV_PRIME))
}

/// The export's identity: FNV-1a 64 over the days (i32 LE) and then the store ids (u64 LE).
pub fn fingerprint(days: &[i32], ids: &[usize]) -> u64 {
    let h = days.iter().fold(FNV_OFFSET, |h, d| fnv_extend(h, &d.to_le_bytes()));
    ids.iter().fold(h, |h, &id| fnv_extend(h, &(id as u64).to_le_bytes()))
}

pub fn fingerprint_hex(days: &[i32], ids: &[usize]) -> String {
    format!("{:016x}", fingerprint(days, ids))
}

/// correlation2's config text (run.rs `config_text`, which is private), for the record.
pub fn config_text(cfg: &c2_census::grid::Config) -> String {
    format!(
        "horizons={:?} lengths={:?} full_history={} step={} start={} min_obs={} nulls={} keep_real={} keep_null={} floor_z={} fdr_q={}",
        cfg.horizons,
        cfg.lengths,
        cfg.full_history,
        cfg.step,
        c2_data::date_string(cfg.start),
        cfg.min_obs,
        cfg.nulls,
        cfg.keep_real,
        cfg.keep_null,
        cfg.floor_z,
        cfg.fdr_q
    )
}

pub struct Summary {
    pub dir: PathBuf,
    pub n: usize,
    pub t: usize,
    pub first_day: String,
    pub last_day: String,
    pub fingerprint: String,
    pub finite_v: usize,
    pub sum_v: f64,
    pub bytes: u64,
}

/// Writes the export into `out`. Every file goes through a temporary file and a rename;
/// `meta.json` is removed first and written last, so a reader never pairs a new meta with old
/// arrays, and an interrupted export has no `meta.json` (and fails to load). Without `extras`,
/// extra arrays left in `out` by an earlier Sharadar export are removed.
pub fn export(store: &Store, plan: &Plan, store_dir: &Path, out: &Path, git_commit: &str, extras: Option<&Extras>) -> Result<Summary> {
    let var = &plan.var;
    let (n, t) = (var.n, var.t);
    ensure!(var.days.len() == t && var.ids.len() == n && var.market.len() == t, "inconsistent variable sizes");
    ensure!(var.v.len() == n * t && var.close.len() == n * t && var.dollar_volume.len() == n * t, "inconsistent matrix sizes");
    ensure!(t > 0 && n > 0, "empty variable");
    std::fs::create_dir_all(out)?;
    let meta_path = out.join("meta.json");
    if meta_path.exists() {
        std::fs::remove_file(&meta_path)?;
    }

    npy::write(&out.join("days.npy"), &[t], &var.days)?;
    npy::write_atomic(&out.join("symbols.tsv"), |w| {
        writeln!(w, "col\tid\tticker")?;
        for (j, &id) in var.ids.iter().enumerate() {
            let ticker = &store.symbols[id].ticker;
            ensure!(!ticker.contains(['\t', '\n', '\r']), "ticker {ticker:?} (store id {id}) has a tab or newline");
            writeln!(w, "{j}\t{id}\t{ticker}")?;
        }
        Ok(())
    })?;
    npy::write(&out.join("v.npy"), &[n, t], &var.v)?;
    npy::write(&out.join("close.npy"), &[n, t], &var.close)?;
    npy::write(&out.join("dollar_volume.npy"), &[n, t], &var.dollar_volume)?;
    npy::write(&out.join("market.npy"), &[t], &var.market)?;
    match extras {
        Some(x) => {
            ensure!(x.close_unadj.len() == n * t && x.delist_day.len() == n && x.delist_kind.len() == n, "extras not aligned to the variable");
            npy::write(&out.join(EXTRAS[0].0), &[n, t], &x.close_unadj)?;
            npy::write(&out.join(EXTRAS[1].0), &[n], &x.delist_day)?;
            npy::write(&out.join(EXTRAS[2].0), &[n], &x.delist_kind)?;
        }
        None => {
            for (f, _) in EXTRAS {
                if out.join(f).exists() {
                    std::fs::remove_file(out.join(f))?;
                }
            }
        }
    }

    let symbols_tsv = std::fs::read(store_dir.join("symbols.tsv")).unwrap_or_default();
    let fp = fingerprint_hex(&var.days, &var.ids);
    let (first_day, last_day) = (c2_data::date_string(var.days[0]), c2_data::date_string(var.days[t - 1]));
    let mut meta = serde_json::json!({
        "format": FORMAT,
        "n": n,
        "t": t,
        "first_day": first_day,
        "last_day": last_day,
        "fingerprint": fp,
        "git_commit": git_commit,
        "variable": VARIABLE,
        "plan": "c2_census::run::Plan::new(&store, Config::full()).var",
        "config": config_text(&plan.cfg),
        "store_dir": store_dir.display().to_string(),
        "store_symbols": store.symbols.len(),
        "store_rows": store.rows(),
        "symbols_tsv_fnv64": format!("{:016x}", fnv(&symbols_tsv)),
        "writer": concat!("stockai-judge ", env!("CARGO_PKG_VERSION")),
        "created_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),
    });
    if let Some(x) = extras {
        meta["variable"] = VARIABLE_SHARADAR.into();
        meta["extras"] = EXTRAS.iter().map(|(f, what)| (f.to_string(), serde_json::Value::from(*what))).collect::<serde_json::Map<_, _>>().into();
        meta["source"] = x.source.clone();
        meta["delisted"] = serde_json::json!({
            "buyout": x.delisted[0],
            "failure": x.delisted[1],
            "unknown": x.delisted[2],
            "last_close_before_last_date": x.moved,
            "last_close_before_last_date_max_days": x.moved_max_days,
            "without_close_left_listed": x.without_close,
        });
    }
    npy::write_atomic(&meta_path, |w| {
        w.write_all(serde_json::to_string_pretty(&meta)?.as_bytes())?;
        w.write_all(b"\n")?;
        Ok(())
    })?;

    let bytes = ["days.npy", "symbols.tsv", "v.npy", "close.npy", "dollar_volume.npy", "market.npy", "meta.json"]
        .iter()
        .chain(EXTRAS.iter().map(|(f, _)| f))
        .map(|f| std::fs::metadata(out.join(f)).map(|m| m.len()).unwrap_or(0))
        .sum();
    // Checksums to compare with numpy's view of the files (row by row, in f64).
    let (finite_v, sum_v) = var.v.iter().filter(|x| x.is_finite()).fold((0usize, 0f64), |(c, s), &x| (c + 1, s + x as f64));
    Ok(Summary { dir: out.to_path_buf(), n, t, first_day, last_day, fingerprint: fp, finite_v, sum_v, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_matches_known_vectors() {
        // FNV-1a 64 test vectors (Fowler/Noll/Vo reference implementation).
        assert_eq!(fnv(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn fingerprint_is_days_then_ids() {
        let days = [11_684i32, 11_685, -3];
        let ids = [0usize, 7, 11_530];
        let mut bytes = Vec::new();
        for d in days {
            bytes.extend_from_slice(&d.to_le_bytes());
        }
        for id in ids {
            bytes.extend_from_slice(&(id as u64).to_le_bytes());
        }
        assert_eq!(fingerprint(&days, &ids), fnv(&bytes));
        assert_eq!(fingerprint_hex(&days, &ids).len(), 16);
        assert_ne!(fingerprint(&days, &ids), fingerprint(&days, &ids[..2]));
    }
}
