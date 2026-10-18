//! What the census enumerates: return horizons, lags, rolling windows and tasks.
//!
//! Lag convention: the leader's return covers trading days (g - h1, g] and the target's
//! (g + lag - h2, g + lag]. Lag 0 is the same-period reference. Lags 0 < lag < h2 would let the
//! two returns share days (v1's leak) and are never generated; predictive lags start at h2.
//! For h1 = h2 = 1, lag 1 is "next day" and lag 0 "same day", as in research/lag_spectrum.py.
//! Observations are non-overlapping: grid points g step by max(h1, h2).

#[derive(Clone, Debug)]
pub struct Config {
    pub horizons: Vec<usize>,
    /// Rolling window lengths in trading days.
    pub lengths: Vec<usize>,
    /// Also one window covering everything from `start`.
    pub full_history: bool,
    /// Rolling windows advance this many trading days.
    pub step: usize,
    pub start: i32,
    pub min_obs: usize,
    pub nulls: usize,
    pub keep_real: usize,
    pub keep_null: usize,
    /// Most false-discovery-selected entries stored per matrix beyond the top `keep_real`.
    pub fdr_cap: usize,
    /// INT8-screened |z| above which entries are pulled off the GPU.
    pub floor_z: f32,
    pub fdr_q: f64,
}

impl Config {
    /// The spec's full grid (SPEC.md section 6A).
    pub fn full() -> Config {
        Config {
            horizons: vec![1, 3, 5, 10, 21, 63],
            lengths: vec![252, 504, 756, 1260],
            full_history: true,
            step: 63,
            start: c2_data::days_from_civil(2005, 1, 3),
            min_obs: 30,
            nulls: 3,
            keep_real: 100,
            keep_null: 50,
            fdr_cap: 1000,
            floor_z: 3.5,
            fdr_q: 0.05,
        }
    }

    /// A reduced grid for a quick end-to-end run.
    pub fn smoke() -> Config {
        Config { horizons: vec![1, 5, 21], lengths: vec![252, 756], full_history: false, ..Config::full() }
    }

    /// (h1, h2) pairs: every leader horizon with every target horizon.
    pub fn horizon_pairs(&self) -> Vec<(usize, usize)> {
        self.horizons.iter().flat_map(|&h1| self.horizons.iter().map(move |&h2| (h1, h2))).collect()
    }
}

/// Lags for a target horizon: 0, then predictive lags from h2 in steps (about 25 per horizon;
/// every day for daily targets out to a quarter).
pub fn lags(h2: usize) -> Vec<usize> {
    let mut out = vec![0];
    match h2 {
        1 => out.extend(1..=63),
        5 => out.extend((5..=125).step_by(5)),
        10 => out.extend((10..=250).step_by(10)),
        21 => out.extend((21..=504).step_by(21)),
        63 => out.extend((63..=756).step_by(63)),
        h => out.extend((h..=h * 25).step_by(h)),
    }
    out
}

/// Number of observations for a window of `w` days at (h1, h2, lag).
pub fn observations(w: usize, h1: usize, h2: usize, lag: usize) -> usize {
    let (g0, step) = (h1.max(h2) - 1, h1.max(h2));
    if g0 + lag >= w { 0 } else { (w - 1 - lag - g0) / step + 1 }
}

#[derive(Clone, Debug)]
pub struct Window {
    pub id: usize,
    /// Rolling length in trading days, or 0 for the full-history window.
    pub length: usize,
    /// Calendar rows [start, end) of the census variable.
    pub start: usize,
    pub end: usize,
}

/// Rolling windows ending every `step` rows from the first full window, plus full history.
pub fn windows(cfg: &Config, first_row: usize, rows: usize) -> Vec<Window> {
    let mut out = Vec::new();
    for &len in &cfg.lengths {
        let mut end = first_row + len;
        while end <= rows {
            out.push(Window { id: out.len(), length: len, start: end - len, end });
            end += cfg.step;
        }
    }
    if cfg.full_history && rows > first_row {
        out.push(Window { id: out.len(), length: 0, start: first_row, end: rows });
    }
    out
}

#[derive(Clone, Debug)]
pub struct Task {
    pub id: usize,
    pub h1: usize,
    pub h2: usize,
    pub length: usize,
    /// Window ids in date order.
    pub windows: Vec<usize>,
}

pub fn tasks(cfg: &Config, windows: &[Window]) -> Vec<Task> {
    let mut lengths: Vec<usize> = cfg.lengths.clone();
    if cfg.full_history {
        lengths.push(0);
    }
    let mut out = Vec::new();
    for (h1, h2) in cfg.horizon_pairs() {
        for &len in &lengths {
            let ws: Vec<usize> = windows.iter().filter(|w| w.length == len).map(|w| w.id).collect();
            if !ws.is_empty() {
                out.push(Task { id: out.len(), h1, h2, length: len, windows: ws });
            }
        }
    }
    out
}
