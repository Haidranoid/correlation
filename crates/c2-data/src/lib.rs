//! Price data: Stooq ingest, the memory-mapped price store, trading calendar, panels and
//! data-quality reports.
//!
//! Dates are `i32` days since 1970-01-01 throughout.

pub mod ingest;
pub mod panel;
pub mod quality;
pub mod store;

use std::path::PathBuf;

pub use panel::{Calendar, Panel};
pub use store::{Kind, Store, SymbolInfo};

/// Where correlation2 keeps its data: `C2_DATA_DIR`, else `~/.local/share/correlation2`.
pub fn data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("C2_DATA_DIR") {
        return dir.into();
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".local/share/correlation2")
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's `days_from_civil`).
pub fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = m as i32;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// (year, month, day) for days since 1970-01-01.
pub fn civil_from_days(z: i32) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i32::from(m <= 2), m, d)
}

/// `YYYY-MM-DD` for days since 1970-01-01.
pub fn date_string(days: i32) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Days since 1970-01-01 for `YYYY-MM-DD`.
pub fn parse_date(s: &str) -> anyhow::Result<i32> {
    let mut it = s.split('-');
    let (Some(y), Some(m), Some(d), None) = (it.next(), it.next(), it.next(), it.next()) else {
        anyhow::bail!("expected YYYY-MM-DD, got {s:?}");
    };
    Ok(days_from_civil(y.parse()?, m.parse()?, d.parse()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_round_trip() {
        for z in -800_000..800_000 {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(date_string(days_from_civil(2024, 6, 28)), "2024-06-28");
    }
}
