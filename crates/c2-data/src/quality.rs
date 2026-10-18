//! Data-quality summaries over the whole store.

use rayon::prelude::*;

use crate::store::Store;

/// One decade of the quality table (see ANALYSIS.md section 3).
#[derive(Clone, Debug, Default)]
pub struct DecadeRow {
    pub decade: i32,
    pub symbols: usize,
    pub rows: usize,
    /// Rows whose close equals the symbol's previous close.
    pub zero_return: usize,
    /// Rows that have a previous close (the denominator for `zero_return`).
    pub with_prev: usize,
    pub under_1: usize,
    /// Rows where open = high = low = close.
    pub close_only: usize,
}

pub fn decade_table(store: &Store) -> Vec<DecadeRow> {
    const FIRST: i32 = 1950;
    const SLOTS: usize = 10;
    let per_symbol: Vec<Vec<DecadeRow>> = (0..store.symbols.len())
        .into_par_iter()
        .map(|id| {
            let mut t = vec![DecadeRow::default(); SLOTS];
            let (d, o, h, l, c) = (store.dates(id), store.open_(id), store.high(id), store.low(id), store.close(id));
            for k in 0..d.len() {
                let year = crate::civil_from_days(d[k]).0;
                let slot = (((year - FIRST) / 10).max(0) as usize).min(SLOTS - 1);
                let row = &mut t[slot];
                row.rows += 1;
                row.under_1 += usize::from(c[k] < 1.0);
                row.close_only += usize::from(o[k] == c[k] && h[k] == c[k] && l[k] == c[k]);
                if k > 0 {
                    row.with_prev += 1;
                    row.zero_return += usize::from(c[k] == c[k - 1]);
                }
            }
            for row in t.iter_mut() {
                row.symbols = usize::from(row.rows > 0);
            }
            t
        })
        .collect();
    let mut total = vec![DecadeRow::default(); SLOTS];
    for t in per_symbol {
        for (acc, r) in total.iter_mut().zip(t) {
            acc.symbols += r.symbols;
            acc.rows += r.rows;
            acc.zero_return += r.zero_return;
            acc.with_prev += r.with_prev;
            acc.under_1 += r.under_1;
            acc.close_only += r.close_only;
        }
    }
    total
        .into_iter()
        .enumerate()
        .map(|(i, mut r)| {
            r.decade = FIRST + 10 * i as i32;
            r
        })
        .filter(|r| r.rows > 0)
        .collect()
}
