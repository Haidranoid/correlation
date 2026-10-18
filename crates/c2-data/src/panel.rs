//! Trading calendar and dense price panels.

use std::ops::Range;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::store::Store;

/// Trading days, sorted. Built from a symbol that traded every exchange day (SPY): inferring
/// days from how many symbols report drops real days in early, sparse years.
#[derive(Clone, Debug)]
pub struct Calendar {
    pub days: Vec<i32>,
}

impl Calendar {
    pub fn from_symbol(store: &Store, ticker: &str) -> Result<Calendar> {
        let id = store.id(ticker).with_context(|| format!("calendar symbol {ticker} not in the store"))?;
        Ok(Calendar { days: store.dates(id).to_vec() })
    }

    /// NYSE trading days from long-lived stocks: a day counts when at least 3 of 8 large caps
    /// (histories from 1962-1984) traded. SPY alone would do, but Stooq's SPY starts 2005-02-25,
    /// which leaves no warm-up year before a 2005 census start. Matches SPY's days exactly
    /// where both exist (checked by `c2 data-check`).
    pub fn trading_days(store: &Store) -> Result<Calendar> {
        const REFERENCE: [&str; 8] = ["IBM", "GE", "KO", "XOM", "JNJ", "PG", "MMM", "AAPL"];
        let mut count: std::collections::BTreeMap<i32, u32> = std::collections::BTreeMap::new();
        let mut found = 0;
        for t in REFERENCE {
            if let Some(id) = store.id(t) {
                found += 1;
                for &d in store.dates(id) {
                    *count.entry(d).or_default() += 1;
                }
            }
        }
        anyhow::ensure!(found >= 5, "only {found} of the calendar reference stocks are in the store");
        Ok(Calendar { days: count.into_iter().filter(|&(_, c)| c >= 3).map(|(d, _)| d).collect() })
    }

    /// Indices of the trading days in `[from, to]` (inclusive dates).
    pub fn range(&self, from: i32, to: i32) -> Range<usize> {
        self.days.partition_point(|&d| d < from)..self.days.partition_point(|&d| d <= to)
    }
}

/// Column-major `t x n` panel aligned to calendar days; NaN where a symbol has no price.
pub struct Panel {
    pub days: Vec<i32>,
    pub ids: Vec<usize>,
    pub t: usize,
    pub n: usize,
    pub close: Vec<f32>,
    pub volume: Vec<f32>,
}

impl Panel {
    /// Closes and volumes for calendar days `rows` and symbols `ids`, in parallel.
    pub fn build(store: &Store, cal: &Calendar, rows: Range<usize>, ids: &[usize]) -> Panel {
        let days = cal.days[rows].to_vec();
        let (t, n) = (days.len(), ids.len());
        let mut close = vec![f32::NAN; t * n];
        let mut volume = vec![0f32; t * n];
        close
            .par_chunks_mut(t.max(1))
            .zip(volume.par_chunks_mut(t.max(1)))
            .zip(ids.par_iter())
            .for_each(|((c, v), &id)| {
                let (d, pc, pv) = (store.dates(id), store.close(id), store.volume(id));
                let mut k = d.partition_point(|&x| x < *days.first().unwrap_or(&0));
                for (r, &day) in days.iter().enumerate() {
                    while k < d.len() && d[k] < day {
                        k += 1;
                    }
                    if k < d.len() && d[k] == day && pc[k] > 0.0 && pc[k].is_finite() {
                        c[r] = pc[k];
                        v[r] = pv[k];
                    }
                }
            });
        Panel { days, ids: ids.to_vec(), t, n, close, volume }
    }

    pub fn col(&self, j: usize) -> &[f32] {
        &self.close[j * self.t..(j + 1) * self.t]
    }

    /// Blanks one-day bad prints: a move of more than 2x that undoes itself (within 10%) the next
    /// day, judged on neighbouring calendar rows. Returns how many closes were blanked.
    /// Stooq prices are already split- and dividend-adjusted; these are data errors.
    pub fn repair_bad_prints(&mut self) -> usize {
        let t = self.t;
        let ln2 = std::f32::consts::LN_2;
        self.close
            .par_chunks_mut(t.max(1))
            .map(|c| {
                let bad: Vec<usize> = (1..t.saturating_sub(1))
                    .filter(|&r| {
                        let (a, b, d) = (c[r - 1], c[r], c[r + 1]);
                        let (jump, back) = ((b / a).ln(), (d / b).ln());
                        jump.abs() > ln2 && (jump + back).abs() < 0.1
                    })
                    .collect();
                for &r in &bad {
                    c[r] = f32::NAN;
                }
                bad.len()
            })
            .sum()
    }
}
