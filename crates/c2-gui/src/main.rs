//! `c2-gui`: native desktop front end for correlation2.
//!
//! It drives the `c2` command (census runs happen in their own detached process, so closing the
//! window never interrupts one) and reads its output files. Tabs:
//!   Status   census progress and ETA; start, pause, resume, stop
//!   Strongest  stocks ranked by significant lead-lag relationships; click one for its timeline
//!   Stock    timeline of one stock's lead-lag partners on a days-ahead axis (which stocks' moves
//!            show up in it later, or where its own moves show up later, and by how much), with
//!            each relationship's history across census windows
//!   Forecast the stock's predicted price path from its leaders' recent moves, the actual path
//!            when the as-of date is in the past, and the forecast's track record
//!   Census   real vs chance by lag for each horizon pair; a ticker's strongest stored pairs
//!   Holdout  the $10,000 test for either signal family, with equity curves
//!   Data     price store summary and the decade quality table

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use c2_census::holdout::{Family, Options, Report};
use c2_census::run::{self, FLAG_NULL, Plan};
use c2_data::Store;
use eframe::egui;
use c2_census::analyze::ProfileRow;
use c2_census::forecast::Forecast;
use egui_plot::{Bar, BarChart, HLine, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoint, PlotPoints, Points, Text, VLine};

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Status,
    Strongest,
    Stock,
    Forecast,
    Census,
    Holdout,
    Data,
}

/// Real vs chance tail counts for one (h1, h2, lag), summed over windows.
#[derive(Default, Clone)]
struct LagRow {
    real_z4: f64,
    null_z4: f64,
    matrices: usize,
    fdr: usize,
}

struct PairRow {
    window: String,
    partner: String,
    role: &'static str,
    h1: u8,
    h2: u8,
    lag: u16,
    r: f32,
    z: f32,
    obs: u16,
}

struct App {
    tab: Tab,
    run_name: String,
    store: Option<Arc<Store>>,
    store_note: String,
    decades: Vec<c2_data::quality::DecadeRow>,
    status: Option<serde_json::Value>,
    status_err: String,
    pid: Option<u32>,
    control: String,
    last_poll: Instant,
    message: String,
    // Census tab.
    lag_table: BTreeMap<(usize, usize), BTreeMap<usize, LagRow>>,
    pair: (usize, usize),
    ticker: String,
    pairs_rx: Option<Receiver<Result<Vec<PairRow>, String>>>,
    pair_rows: Vec<PairRow>,
    // Holdout tab.
    family: usize,
    start: String,
    picks: usize,
    rebalance: usize,
    costs: String,
    random: usize,
    plan: Arc<Mutex<Option<Arc<Plan>>>>,
    holdout_rx: Option<Receiver<Result<Report, String>>>,
    holdout: Option<Report>,
    holdout_started: Option<Instant>,
    /// Self-check mode (`C2_GUI_SCREENSHOT=<path prefix>`, `C2_GUI_TAB=<tab>`): open a tab, wait
    /// for its data, save the window's pixels and exit.
    shot: Option<(String, u32, bool)>,
    // Stock timeline tab.
    tl: Timeline,
    // Forecast tab.
    fc: ForecastView,
    // Strongest tab.
    rank_rx: Option<Receiver<Result<Vec<c2_census::analyze::RankRow>, String>>>,
    rank: Vec<c2_census::analyze::RankRow>,
    rank_filter: String,
    /// 0 = count x strength (sum of |r|), 1 = significant (sum of |r| over significant ones),
    /// 2 = strongest single |r|, 3 = most relationships.
    rank_sort: usize,
    rank_error: String,
}

/// Key of one relationship regardless of window: (partner, stock leads?, lag, h1, h2).
type RelKey = (String, bool, u16, u8, u8);

#[derive(Default)]
struct Timeline {
    input: String,
    ticker: String,
    rx: Option<Receiver<Result<Vec<ProfileRow>, String>>>,
    rows: Vec<ProfileRow>,
    error: String,
    significant_only: bool,
    /// "all", "recent", "full", or a window label.
    window: String,
    /// (h1, h2) or None for all.
    horizons: Option<(u8, u8)>,
    max_shown: usize,
    show_r: bool,
    selected: Option<RelKey>,
    /// false: stocks whose moves show up in this one later (they move first); true: stocks this
    /// one moves first.
    stock_first: bool,
}

/// (ticker, as-of day or None for the latest, days ahead, significant only, max window age in
/// years or 0 for any).
type ForecastRequest = (String, Option<i32>, usize, bool, usize);

#[derive(Default)]
struct ForecastView {
    /// Ticker of the forecast (the ticker box is shared with the timeline tab).
    ticker: String,
    as_of: Option<i32>,
    as_of_text: String,
    horizon: usize,
    significant_only: bool,
    max_age_years: usize,
    /// Plot the value of this many dollars invested instead of the share price.
    amount: String,
    show_amount: bool,
    rx: Option<Receiver<Result<Forecast, String>>>,
    result: Option<Forecast>,
    error: String,
    requested: Option<ForecastRequest>,
}

fn key(r: &ProfileRow) -> RelKey {
    (r.partner.clone(), r.leads, r.lag, r.h1, r.h2)
}

/// Decimal year of the last date in a "first..last" window label.
fn window_end_year(w: &str) -> f64 {
    let last = w.split("..").nth(1).unwrap_or("");
    let p: Vec<f64> = last.split('-').filter_map(|v| v.parse().ok()).collect();
    if p.len() == 3 { p[0] + (p[1] - 1.0) / 12.0 + (p[2] - 1.0) / 365.0 } else { f64::NAN }
}

fn c2_binary() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("c2"))).filter(|p| p.exists()).unwrap_or_else(|| "c2".into())
}

fn running_pid(dir: &Path) -> Option<u32> {
    let pid: u32 = std::fs::read_to_string(dir.join("pid")).ok()?.trim().parse().ok()?;
    let exe = std::fs::read_link(format!("/proc/{pid}/exe")).ok()?;
    exe.file_name()?.to_str()?.starts_with("c2").then_some(pid)
}

fn pct(v: f64) -> String {
    format!("{:+.1}%", 100.0 * (v / 10_000.0 - 1.0))
}

impl App {
    fn new() -> App {
        let dir = c2_data::data_dir().join("store");
        let (store, store_note, decades) = match Store::open(&dir) {
            Ok(s) => {
                let d = c2_data::quality::decade_table(&s);
                let note = format!("{} symbols, {} daily rows, in {}", s.symbols.len(), s.rows(), dir.display());
                (Some(Arc::new(s)), note, d)
            }
            Err(e) => (None, format!("No price store yet ({e:#}). Run `c2 ingest`."), Vec::new()),
        };
        App {
            tab: Tab::Status,
            run_name: "main".into(),
            store,
            store_note,
            decades,
            status: None,
            status_err: String::new(),
            pid: None,
            control: "run".into(),
            last_poll: Instant::now() - Duration::from_secs(10),
            message: String::new(),
            lag_table: BTreeMap::new(),
            pair: (1, 1),
            ticker: "AAPL".into(),
            pairs_rx: None,
            pair_rows: Vec::new(),
            family: 0,
            start: "2024-06-28".into(),
            picks: 20,
            rebalance: 1,
            costs: "0,5,20".into(),
            random: 1000,
            plan: Arc::new(Mutex::new(None)),
            holdout_rx: None,
            holdout: None,
            holdout_started: None,
            shot: std::env::var("C2_GUI_SCREENSHOT").ok().map(|p| (p, 0, false)),
            tl: Timeline { input: "NAD".into(), window: "all".into(), max_shown: 40, ..Default::default() },
            fc: ForecastView { horizon: 21, amount: "10000".into(), ..Default::default() },
            rank_rx: None,
            rank: Vec::new(),
            rank_filter: String::new(),
            rank_sort: 0,
            rank_error: String::new(),
        }
    }

    fn strongest_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Strongest lead-lag relationships");
        if self.rank.is_empty() && self.rank_rx.is_none() && self.rank_error.is_empty() {
            if let Some(store) = self.store.clone() {
                let (tx, rx) = channel();
                self.rank_rx = Some(rx);
                let dir = self.dir();
                std::thread::spawn(move || {
                    let _ = tx.send(c2_census::analyze::stock_ranking(&store, &dir).map_err(|e| format!("{e:#}")));
                });
            }
        }
        if let Some(rx) = &self.rank_rx {
            if let Ok(res) = rx.try_recv() {
                match res {
                    Ok(r) => self.rank = r,
                    Err(e) => self.rank_error = e,
                }
                self.rank_rx = None;
            }
        }
        if self.rank_rx.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("reading the census (about a second)…");
            });
            return;
        }
        if !self.rank_error.is_empty() {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.rank_error);
            return;
        }
        let with_sig = self.rank.iter().filter(|r| r.significant > 0).count();
        const SORTS: [&str; 4] = [
            "number x strength (sum of |r|)",
            "significant relationships (sum of |r| over those)",
            "strongest single correlation",
            "most relationships",
        ];
        ui.horizontal(|ui| {
            ui.label("Find:");
            ui.add(egui::TextEdit::singleline(&mut self.rank_filter).desired_width(80.0));
            ui.label("Sort by:");
            egui::ComboBox::from_id_salt("rank_sort").selected_text(SORTS[self.rank_sort]).show_ui(ui, |ui| {
                for (k, s) in SORTS.iter().enumerate() {
                    ui.selectable_value(&mut self.rank_sort, k, *s);
                }
            });
        });
        ui.label(format!(
            "Stocks with both many stored lead-lag correlations and strong ones rank highest (score = sum of |r|). Most stored correlations are within \
             what chance produces when 176 billion pairs are tested; {with_sig} stocks have relationships that passed the significance test ('sig.'), and \
             'windows' counts how many separate census windows those come from (1 usually means one event, such as the 2020 crash). Click a stock for its timeline."
        ));
        let filter = self.rank_filter.trim().to_uppercase();
        let mut rows: Vec<&c2_census::analyze::RankRow> = self
            .rank
            .iter()
            .filter(|r| filter.is_empty() || r.ticker.contains(&filter))
            .filter(|r| self.rank_sort != 1 || r.significant > 0)
            .collect();
        match self.rank_sort {
            1 => rows.sort_by(|a, b| b.sig_sum_abs_r.total_cmp(&a.sig_sum_abs_r)),
            2 => rows.sort_by(|a, b| b.max_abs_r.total_cmp(&a.max_abs_r)),
            3 => rows.sort_by(|a, b| b.stored.cmp(&a.stored)),
            _ => rows.sort_by(|a, b| b.sum_abs_r.total_cmp(&a.sum_abs_r)),
        }
        let mut open: Option<(String, bool)> = None;
        egui::ScrollArea::vertical().id_salt("rank").show(ui, |ui| {
            egui::Grid::new("rank_grid").striped(true).show(ui, |ui| {
                for h in ["#", "stock", "score", "relationships", "mean |r|", "max |r|", "sig.", "windows", "strongest relationship", "z"] {
                    ui.strong(h);
                }
                ui.end_row();
                for (k, r) in rows.iter().take(1000).enumerate() {
                    ui.label(format!("{}", k + 1));
                    if ui.link(&r.ticker).clicked() {
                        open = Some((r.ticker.clone(), self.rank_sort == 1));
                    }
                    ui.label(format!("{:.0}", if self.rank_sort == 1 { r.sig_sum_abs_r } else { r.sum_abs_r }));
                    ui.label(r.stored.to_string());
                    ui.label(format!("{:.3}", r.mean_abs_r));
                    ui.label(format!("{:.3}", r.max_abs_r));
                    ui.label(if r.significant > 0 { format!("{} ({}↗ {}↙)", r.significant, r.leads, r.follows) } else { "0".into() });
                    ui.label(if r.significant > 0 { r.windows.to_string() } else { String::new() });
                    let (a, b) = if r.best_leads { (&r.ticker, &r.best_partner) } else { (&r.best_partner, &r.ticker) };
                    let rel = format!("{a} moves, {b} follows {} days later, r {:+.3}", r.best_lag, r.best_r);
                    ui.label(rel);
                    ui.label(format!("{:.1}", r.best_z));
                    ui.end_row();
                }
            });
        });
        if let Some((t, sig)) = open {
            self.tl.input = t;
            self.tl.significant_only = sig;
            self.tl.window = "all".into();
            self.tl.horizons = None;
            self.load_timeline();
            self.tab = Tab::Stock;
        }
    }

    fn load_timeline(&mut self) {
        let Some(store) = self.store.clone() else { return };
        let (tx, rx) = channel();
        self.tl.rx = Some(rx);
        self.tl.error.clear();
        let (dir, ticker) = (self.dir(), self.tl.input.trim().to_uppercase());
        self.tl.ticker = ticker.clone();
        self.fc.ticker = ticker.clone();
        self.tl.selected = None;
        std::thread::spawn(move || {
            let _ = tx.send(c2_census::analyze::stock_profile(&store, &dir, &ticker, false).map_err(|e| format!("{e:#}")));
        });
    }

    /// Rows passing the timeline's filters; with "all windows", the strongest window of each
    /// relationship.
    fn timeline_rows(&self) -> Vec<ProfileRow> {
        let tl = &self.tl;
        let last = tl.rows.iter().map(|r| window_end_year(&r.window)).fold(f64::MIN, f64::max);
        let longest = tl.rows.iter().map(|r| r.window.clone()).max_by_key(|w| {
            let y = |s: &str| s.split('-').next().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0);
            let mut it = w.split("..");
            y(it.nth(1).unwrap_or("")) - y(w.split("..").next().unwrap_or(""))
        });
        let mut out: Vec<ProfileRow> = tl
            .rows
            .iter()
            .filter(|r| r.leads == tl.stock_first)
            .filter(|r| !tl.significant_only || r.significant)
            .filter(|r| tl.horizons.is_none_or(|h| (r.h1, r.h2) == h))
            .filter(|r| match tl.window.as_str() {
                "all" => true,
                "recent" => (window_end_year(&r.window) - last).abs() < 1e-6,
                "full" => Some(&r.window) == longest.as_ref(),
                w => r.window == w,
            })
            .cloned()
            .collect();
        if tl.window == "all" {
            let mut best: std::collections::HashMap<RelKey, ProfileRow> = std::collections::HashMap::new();
            for r in out {
                let e = best.entry(key(&r)).or_insert_with(|| r.clone());
                if r.z.abs() > e.z.abs() {
                    *e = r;
                }
            }
            out = best.into_values().collect();
        }
        out.sort_by(|a, b| b.z.abs().total_cmp(&a.z.abs()));
        out.truncate(tl.max_shown);
        out
    }

    fn stock_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Stock timeline");
        ui.horizontal(|ui| {
            ui.label("Ticker:");
            let resp = ui.add(egui::TextEdit::singleline(&mut self.tl.input).desired_width(80.0));
            let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if (ui.add_enabled(self.tl.rx.is_none() && self.store.is_some(), egui::Button::new("Show")).clicked() || enter) && self.tl.rx.is_none() {
                self.load_timeline();
            }
            if self.tl.rx.is_some() {
                ui.spinner();
            }
        });
        if let Some(rx) = &self.tl.rx {
            if let Ok(res) = rx.try_recv() {
                match res {
                    Ok(rows) => self.tl.rows = rows,
                    Err(e) => {
                        self.tl.rows.clear();
                        self.tl.error = e;
                    }
                }
                self.tl.rx = None;
            }
        }
        if !self.tl.error.is_empty() {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.tl.error);
            return;
        }
        if self.tl.rows.is_empty() {
            ui.label("Enter a ticker to see which stocks tend to move before it and after it, at what delay, and by how much.");
            return;
        }
        let t = self.tl.ticker.clone();
        let total_sig = self.tl.rows.iter().filter(|r| r.significant && r.leads == self.tl.stock_first).count();
        // Filters.
        let mut windows: Vec<String> = self.tl.rows.iter().map(|r| r.window.clone()).collect();
        windows.sort_by(|a, b| window_end_year(a).total_cmp(&window_end_year(b)));
        windows.dedup();
        let mut horizons: Vec<(u8, u8)> = self.tl.rows.iter().map(|r| (r.h1, r.h2)).collect();
        horizons.sort();
        horizons.dedup();
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.tl.stock_first, false, format!("What moves {t} later (other stocks move first)"));
            ui.radio_value(&mut self.tl.stock_first, true, format!("What {t} moves later ({t} moves first)"));
            ui.separator();
            if ui.button(format!("Forecast {t}'s price ▶")).clicked() {
                self.fc.ticker = t.clone();
                self.tab = Tab::Forecast;
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.tl.significant_only, format!("significant only ({total_sig})"));
            let wl = |w: &str| match w {
                "all" => "all windows (strongest of each)".to_string(),
                "recent" => "most recent windows".to_string(),
                "full" => "full history 2005-2025".to_string(),
                w => w.to_string(),
            };
            ui.label("Window:");
            egui::ComboBox::from_id_salt("tl_window").selected_text(wl(&self.tl.window)).show_ui(ui, |ui| {
                for w in ["all", "recent", "full"].iter().map(|s| s.to_string()).chain(windows.iter().cloned()) {
                    let label = wl(&w);
                    ui.selectable_value(&mut self.tl.window, w, label);
                }
            });
            let hl = |h: Option<(u8, u8)>| h.map(|(a, b)| if a == 1 && b == 1 { "daily".into() } else { format!("{a}-day -> {b}-day") }).unwrap_or_else(|| "all horizons".into());
            ui.label("Returns:");
            egui::ComboBox::from_id_salt("tl_horizons").selected_text(hl(self.tl.horizons)).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.tl.horizons, None, hl(None));
                for h in &horizons {
                    ui.selectable_value(&mut self.tl.horizons, Some(*h), hl(Some(*h)));
                }
            });
            ui.add(egui::Slider::new(&mut self.tl.max_shown, 5..=200).text("shown"));
            ui.checkbox(&mut self.tl.show_r, "plot r instead of % move");
        });
        let rows = self.timeline_rows();
        ui.label(if self.tl.stock_first {
            format!("Day 0 = {t} moves today. A stock at +k days tends to move k trading days later; height = its typical move beyond the market \
                     per 1% move of {t}. Orange = passed the significance test; grey = within what chance produces.")
        } else {
            format!("Day 0 = each stock's move today. A stock at +k days moves {t} k trading days later; height = {t}'s typical move beyond the market \
                     per 1% move of that stock (these feed the Forecast tab). Orange = passed the significance test; grey = within what chance produces.")
        });
        let y_of = |r: &ProfileRow| if self.tl.show_r { r.r as f64 } else { r.move_per_pct.unwrap_or(r.r as f64) };
        let x_of = |r: &ProfileRow| r.lag as f64;
        let selected = self.tl.selected.clone();
        let stock_first = self.tl.stock_first;
        let orange = egui::Color32::from_rgb(255, 160, 40);
        let grey = egui::Color32::from_rgb(140, 150, 170);
        Plot::new("timeline")
            .height(330.0)
            .x_axis_label("trading days later")
            .y_axis_label(if self.tl.show_r { "r".to_string() } else { "% move per 1%".to_string() })
            .legend(Legend::default())
            .show(ui, |p| {
                p.vline(VLine::new("today", 0.0).color(egui::Color32::WHITE).width(2.0));
                p.hline(HLine::new("no move", 0.0).color(egui::Color32::DARK_GRAY));
                let pts = |sig: bool| -> Vec<[f64; 2]> { rows.iter().filter(|r| r.significant == sig).map(|r| [x_of(r), y_of(r)]).collect() };
                p.points(Points::new("within chance", PlotPoints::from(pts(false))).color(grey).radius(4.0).shape(MarkerShape::Circle).filled(true));
                p.points(Points::new("significant", PlotPoints::from(pts(true))).color(orange).radius(6.0).shape(MarkerShape::Diamond).filled(true));
                for (k, r) in rows.iter().enumerate() {
                    let is_sel = selected.as_ref() == Some(&key(r));
                    if k < 25 || r.significant || is_sel {
                        let c = if is_sel { egui::Color32::YELLOW } else if r.significant { orange } else { grey };
                        let anchor_v = if y_of(r) >= 0.0 { egui::Align2::CENTER_BOTTOM } else { egui::Align2::CENTER_TOP };
                        // Alternate sides so partners at the same lag don't print on top of each other.
                        let anchor = if k % 2 == 0 { egui::Align2::LEFT_CENTER } else { egui::Align2::RIGHT_CENTER };
                        let _ = anchor_v;
                        p.text(Text::new("", PlotPoint::new(x_of(r), y_of(r)), format!(" {} +{}d ", r.partner, r.lag)).color(c).anchor(anchor));
                    }
                }
                let today = if stock_first { format!("{t} moves") } else { "leader moves".to_string() };
                p.text(Text::new("", PlotPoint::new(0.0, 0.0), today).color(egui::Color32::WHITE).anchor(egui::Align2::LEFT_BOTTOM));
            });
        ui.columns(2, |cols| {
            let ui = &mut cols[0];
            ui.strong("Relationships (click one to see its history)");
            egui::ScrollArea::vertical().id_salt("tl_rows").max_height(260.0).show(ui, |ui| {
                egui::Grid::new("tl_grid").striped(true).show(ui, |ui| {
                    for h in ["", "relationship", "days later", "move/1%", "r", "z", "chance", "window"] {
                        ui.strong(h);
                    }
                    ui.end_row();
                    for r in &rows {
                        let k = key(r);
                        let sel = self.tl.selected.as_ref() == Some(&k);
                        if ui.selectable_label(sel, if r.significant { "★" } else { "·" }).clicked() {
                            self.tl.selected = Some(k.clone());
                        }
                        let arrow = if r.leads { format!("{t} -> {}", r.partner) } else { format!("{} -> {t}", r.partner) };
                        if ui.selectable_label(sel, arrow).clicked() {
                            self.tl.selected = Some(k);
                        }
                        ui.label(format!("+{}", r.lag));
                        ui.label(r.move_per_pct.map(|m| format!("{m:+.2}%")).unwrap_or_default());
                        ui.label(format!("{:+.3}", r.r));
                        ui.label(format!("{:+.1}", r.z));
                        ui.label(format!("{:.0} at |z|≥{}", r.chance, r.chance_z));
                        ui.label(&r.window);
                        ui.end_row();
                    }
                });
            });
            let ui = &mut cols[1];
            match &self.tl.selected {
                None => {
                    ui.label("Select a relationship to see it across every census window: a persistent relationship shows up again and again with the same sign; a one-off event shows up once.");
                }
                Some(k) => {
                    let (partner, leads, lag, h1, h2) = k.clone();
                    let hist: Vec<&ProfileRow> = self.tl.rows.iter().filter(|r| key(r) == *k).collect();
                    let arrow = if leads { format!("{t} -> {partner}") } else { format!("{partner} -> {t}") };
                    ui.strong(format!("{arrow}, {lag} days later ({h1}-day -> {h2}-day returns): r in each census window where it was among the strongest"));
                    let same_sign = hist.iter().filter(|r| r.r.signum() == hist[0].r.signum()).count();
                    ui.label(format!("stored in {} window(s); same sign in {same_sign}. Windows where it is absent: it was not among that window's strongest pairs.", hist.len()));
                    let mut pts: Vec<[f64; 2]> = hist.iter().map(|r| [window_end_year(&r.window), r.r as f64]).collect();
                    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
                    Plot::new("history").height(220.0).x_axis_label("census window end (year)").y_axis_label("r").show(ui, |p| {
                        p.hline(HLine::new("0", 0.0).color(egui::Color32::DARK_GRAY));
                        p.points(Points::new("r by window", PlotPoints::from(pts)).radius(5.0).color(egui::Color32::YELLOW).filled(true));
                    });
                }
            }
        });
    }

    fn forecast_request(&self) -> ForecastRequest {
        (self.fc.ticker.clone(), self.fc.as_of, self.fc.horizon, self.fc.significant_only, self.fc.max_age_years)
    }

    fn load_forecast(&mut self) {
        let Some(store) = self.store.clone() else { return };
        let req = self.forecast_request();
        let (tx, rx) = channel();
        self.fc.rx = Some(rx);
        self.fc.error.clear();
        self.fc.requested = Some(req.clone());
        let dir = self.dir();
        let (ticker, as_of, horizon, significant_only, age) = req;
        std::thread::spawn(move || {
            let opt = c2_census::forecast::Options { significant_only, horizon, max_age: (age > 0).then_some(age * 252) };
            let _ = tx.send(c2_census::forecast::forecast(&store, &dir, &ticker, as_of.unwrap_or(i32::MAX), &opt).map_err(|e| format!("{e:#}")));
        });
    }

    fn forecast_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Forecast");
        if let Some(rx) = &self.fc.rx {
            if let Ok(res) = rx.try_recv() {
                match res {
                    Ok(f) => {
                        self.fc.as_of_text = c2_data::date_string(f.as_of);
                        self.fc.result = Some(f);
                    }
                    Err(e) => {
                        self.fc.result = None;
                        self.fc.error = e;
                    }
                }
                self.fc.rx = None;
            }
        }
        if self.fc.ticker.is_empty() {
            self.fc.ticker = if self.tl.ticker.is_empty() { self.tl.input.trim().to_uppercase() } else { self.tl.ticker.clone() };
        }
        let current = self.fc.result.as_ref().map(|f| f.as_of);
        ui.horizontal_wrapped(|ui| {
            ui.label("Ticker:");
            let resp = ui.add(egui::TextEdit::singleline(&mut self.tl.input).desired_width(80.0));
            let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if (ui.add_enabled(self.tl.rx.is_none(), egui::Button::new("Forecast")).clicked() || enter) && self.tl.rx.is_none() {
                // Keeps the timeline tab on the same stock.
                self.load_timeline();
            }
            ui.separator();
            ui.label("As of:");
            let resp = ui.add(egui::TextEdit::singleline(&mut self.fc.as_of_text).desired_width(90.0));
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                match c2_data::parse_date(self.fc.as_of_text.trim()) {
                    Ok(d) => self.fc.as_of = Some(d),
                    Err(_) => self.fc.error = "dates look like 2024-06-28".into(),
                }
            }
            for (label, step) in [("◀ year", -365), ("◀ month", -30), ("month ▶", 30), ("year ▶", 365)] {
                if ui.add_enabled(current.is_some(), egui::Button::new(label)).clicked() {
                    self.fc.as_of = current.map(|d| d + step);
                }
            }
            if ui.button("latest").clicked() {
                self.fc.as_of = None;
            }
            if self.fc.rx.is_some() {
                ui.spinner();
            }
        });
        ui.horizontal_wrapped(|ui| {
            let hl = |h: usize| format!("{h} ({})", match h { 5 => "a week", 10 => "two weeks", 21 => "a month", 63 => "a quarter", 126 => "half a year", _ => "a year" });
            ui.label("Days ahead:");
            egui::ComboBox::from_id_salt("fc_h").selected_text(hl(self.fc.horizon)).show_ui(ui, |ui| {
                for h in [5, 10, 21, 63, 126, 252] {
                    ui.selectable_value(&mut self.fc.horizon, h, hl(h));
                }
            });
            ui.checkbox(&mut self.fc.significant_only, "significant relationships only");
            let al = |a: usize| if a == 0 { "any census window".to_string() } else { format!("windows ending in the last {a} year{}", if a == 1 { "" } else { "s" }) };
            ui.label("Use:");
            egui::ComboBox::from_id_salt("fc_age").selected_text(al(self.fc.max_age_years)).show_ui(ui, |ui| {
                for a in [0, 1, 3, 5] {
                    ui.selectable_value(&mut self.fc.max_age_years, a, al(a));
                }
            });
            ui.separator();
            ui.radio_value(&mut self.fc.show_amount, false, "share price");
            ui.radio_value(&mut self.fc.show_amount, true, "value of $");
            ui.add_enabled(self.fc.show_amount, egui::TextEdit::singleline(&mut self.fc.amount).desired_width(70.0));
        });
        if self.fc.rx.is_none() && !self.fc.ticker.is_empty() && self.fc.requested.as_ref() != Some(&self.forecast_request()) {
            self.load_forecast();
        }
        if !self.fc.error.is_empty() {
            ui.colored_label(egui::Color32::LIGHT_RED, &self.fc.error);
        }
        let Some(f) = &self.fc.result else {
            if self.fc.error.is_empty() {
                ui.label("Reading the census (about a second)…");
            }
            return;
        };
        let t = &f.ticker;
        let h = f.predicted_pct.len();
        let amount: f64 = self.fc.amount.trim().replace([',', '$'], "").parse().unwrap_or(10_000.0);
        let base = if self.fc.show_amount { amount } else { f.last_price };
        let val = |pct: f64| base * (1.0 + pct / 100.0);
        let day = c2_data::date_string(f.as_of);
        let end_pct = f.predicted_pct[h - 1];
        let typical = 100.0 * f.daily_sd * (h as f64).sqrt();
        let what = if self.fc.show_amount { format!("${amount:.0} in {t}") } else { format!("{t}") };
        ui.label(format!(
            "{what}: ${base:.2} on {day}. Forecast {h} trading days later: ${:.2} ({end_pct:+.2}%). It is the average of what {} relationships imply \
             (another stock moved first and {t} tends to follow some days later; each leader's actual move so far is scaled by the relationship). \
             For scale, {t}'s typical {h}-day move is ±{typical:.1}%.",
            val(end_pct),
            f.relationships
        ));
        if let Some(Some(a)) = f.actual_pct.last() {
            ui.label(format!("What happened: ${:.2} ({a:+.2}%) after {h} trading days.", val(*a)));
        } else if f.actual_pct.iter().any(|a| a.is_some()) {
            ui.label("What happened so far is drawn in blue; the data ends before the full horizon.");
        }
        match &f.skill {
            Some(s) => {
                let n = s.forecasts as f64;
                let z_hit = (s.hit_rate - 0.5) / (0.25 / n).sqrt();
                let z_corr = s.corr * n.sqrt();
                let lucky = z_hit.abs() < 2.0 && z_corr.abs() < 2.0;
                let text = format!(
                    "Track record for {t}: the same forecast made every 21 trading days in the past, each using only relationships known at the time: {} forecasts \
                     {}-day ahead, direction right {:.0}% of the time (a coin flip gets 50%), predicted vs actual correlation {:+.2}; the forecasts averaged {:.2}% \
                     while {t} actually moved {:.2}% on average. {}",
                    s.forecasts, s.horizon, 100.0 * s.hit_rate, s.corr, s.mean_abs_predicted_pct, s.mean_abs_actual_pct,
                    if lucky {
                        "That is within what luck produces, so treat the forecast line as noise."
                    } else {
                        "That is outside what luck usually produces for one stock, but with thousands of stocks some look this good by chance; check the holdout before trusting it."
                    }
                );
                ui.colored_label(if lucky { egui::Color32::from_rgb(230, 200, 90) } else { egui::Color32::from_rgb(120, 220, 120) }, text);
            }
            None => {
                ui.colored_label(egui::Color32::from_rgb(230, 200, 90), "Track record: too few past forecasts to judge this one.");
            }
        }
        let orange = egui::Color32::from_rgb(255, 160, 40);
        let blue = egui::Color32::from_rgb(90, 170, 255);
        let grey = egui::Color32::from_rgb(140, 150, 170);
        let pred: Vec<[f64; 2]> = std::iter::once([0.0, base]).chain(f.predicted_pct.iter().enumerate().map(|(j, v)| [(j + 1) as f64, val(*v)])).collect();
        let actual: Vec<[f64; 2]> = std::iter::once([0.0, base])
            .chain(f.actual_pct.iter().enumerate().filter_map(|(j, a)| a.map(|a| [(j + 1) as f64, val(a)])))
            .collect();
        let band = |sign: f64| -> Vec<[f64; 2]> { (0..=h).map(|j| [j as f64, base * (sign * f.daily_sd * (j as f64).sqrt()).exp()]).collect() };
        Plot::new("forecast_plot")
            .height(300.0)
            .x_axis_label(format!("trading days after {day}"))
            .y_axis_label(if self.fc.show_amount { "value ($)" } else { "share price ($)" })
            .legend(Legend::default())
            .show(ui, |p| {
                p.hline(HLine::new("", base).color(egui::Color32::DARK_GRAY));
                p.line(Line::new("typical range (±1 sd of past moves)", PlotPoints::from(band(1.0))).color(grey).style(LineStyle::dashed_loose()));
                p.line(Line::new("typical range (±1 sd of past moves)", PlotPoints::from(band(-1.0))).color(grey).style(LineStyle::dashed_loose()));
                if actual.len() > 1 {
                    p.line(Line::new("actual", PlotPoints::from(actual)).color(blue).width(2.0));
                }
                p.line(Line::new("forecast", PlotPoints::from(pred)).color(orange).width(2.5));
            });
        ui.label(format!("When the predicted moves land: the forecast's change on each day ahead (in %), from the relationships whose delay reaches that day."));
        let bars: Vec<Bar> = (0..h)
            .map(|j| {
                let prev = if j == 0 { 0.0 } else { f.predicted_pct[j - 1] };
                let d = (1.0 + f.predicted_pct[j] / 100.0) / (1.0 + prev / 100.0) * 100.0 - 100.0;
                let date = f.future_days[j].map(c2_data::date_string).unwrap_or_default();
                Bar::new((j + 1) as f64, d)
                    .width(0.8)
                    .fill(if d >= 0.0 { egui::Color32::from_rgb(110, 200, 120) } else { egui::Color32::from_rgb(230, 100, 90) })
                    .name(format!("day {} {date}: {d:+.3}% from {} relationships", j + 1, f.covering[j]))
            })
            .collect();
        Plot::new("forecast_days").height(110.0).x_axis_label("trading days ahead").y_axis_label("% that day").show(ui, |p| {
            p.bar_chart(BarChart::new("daily forecast", bars));
        });
        ui.strong(format!("Largest shares of the forecast at day {h} ({} relationships say something about these days)", f.contributions.len()));
        egui::ScrollArea::vertical().id_salt("fc_rows").show(ui, |ui| {
            egui::Grid::new("fc_grid").striped(true).show(ui, |ui| {
                for c in ["leader", "moves first by", "returns", "r", "", "leader's move", "alone implies", "share of forecast", "census window"] {
                    ui.strong(c);
                }
                ui.end_row();
                for c in f.contributions.iter().take(200) {
                    ui.label(&c.leader);
                    ui.label(format!("{} day{}", c.lag, if c.lag == 1 { "" } else { "s" }));
                    ui.label(if c.h1 == 1 && c.h2 == 1 { "daily".to_string() } else { format!("{}d -> {}d", c.h1, c.h2) });
                    ui.label(format!("{:+.3}", c.r));
                    ui.label(if c.significant { "★" } else { "" });
                    ui.label(format!("{:+.2}%", c.leader_move_pct));
                    ui.label(format!("{:+.2}%", c.implied_pct));
                    ui.label(format!("{:+.3} pts", c.share_pct));
                    ui.label(&c.window);
                    ui.end_row();
                }
            });
        });
    }

    fn dir(&self) -> PathBuf {
        run::run_dir(&self.run_name)
    }

    fn poll(&mut self) {
        if self.last_poll.elapsed() < Duration::from_secs(2) {
            return;
        }
        self.last_poll = Instant::now();
        let dir = self.dir();
        self.pid = running_pid(&dir);
        self.control = run::control_state(&dir);
        match std::fs::read_to_string(dir.join("status.json")).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string())) {
            Ok(v) => {
                self.status = Some(v);
                self.status_err.clear();
            }
            Err(e) => {
                self.status = None;
                self.status_err = format!("no status for census '{}' yet ({e})", self.run_name);
            }
        }
    }

    fn status_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Correlation census");
        ui.horizontal(|ui| {
            ui.label("Run name:");
            ui.text_edit_singleline(&mut self.run_name);
        });
        ui.separator();
        match &self.status {
            Some(s) => {
                let frac = s["fraction_done"].as_f64().unwrap_or(0.0) as f32;
                let state = s["state"].as_str().unwrap_or("?").to_string();
                ui.label(format!(
                    "State: {state}  ·  process: {}  ·  control: {}",
                    self.pid.map(|p| format!("running (pid {p})")).unwrap_or_else(|| "not running".into()),
                    self.control
                ));
                ui.add(egui::ProgressBar::new(frac).show_percentage());
                ui.label(format!(
                    "{} of {} windows · ETA {}",
                    s["units_done"], s["units_total"], run::hms(s["eta_seconds"].as_f64().unwrap_or(-1.0))
                ));
            }
            None => {
                ui.label(&self.status_err);
            }
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let dir = self.dir();
            if ui.add_enabled(self.pid.is_none(), egui::Button::new("Start / resume census")).clicked() {
                let _ = std::fs::create_dir_all(&dir);
                let log = std::fs::File::create(dir.join("run.out"));
                let mut cmd = std::process::Command::new(c2_binary());
                cmd.args(["census", "run", "--name", &self.run_name]);
                if let Ok(f) = log {
                    if let Ok(f2) = f.try_clone() {
                        cmd.stdout(f).stderr(f2);
                    }
                }
                use std::os::unix::process::CommandExt;
                cmd.process_group(0);
                self.message = match cmd.spawn() {
                    Ok(c) => format!("started census '{}' (pid {})", self.run_name, c.id()),
                    Err(e) => format!("could not start c2: {e}"),
                };
                self.last_poll = Instant::now() - Duration::from_secs(10);
            }
            if ui.button("Pause (frees the GPU)").clicked() {
                let _ = run::set_control(&dir, "pause");
                self.message = "pausing after the current window".into();
            }
            if ui.button("Resume").clicked() {
                let _ = run::set_control(&dir, "run");
                self.message = "resuming".into();
            }
            if ui.add_enabled(self.pid.is_some(), egui::Button::new("Stop")).clicked() {
                if let Some(pid) = self.pid {
                    let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
                    self.message = format!("asked pid {pid} to stop after committing the current window");
                }
            }
        });
        if !self.message.is_empty() {
            ui.label(&self.message);
        }
        ui.add_space(8.0);
        ui.label("Stopping or pausing loses at most the window in progress; starting again resumes where it left off.");
    }

    fn load_lag_table(&mut self) {
        self.lag_table.clear();
        let Ok(entries) = std::fs::read_dir(self.dir().join("tasks")) else {
            self.message = "no census output yet".into();
            return;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().and_then(|x| x.to_str()) != Some("csv") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let mut lines = text.lines();
            let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
            let col = |name: &str| header.iter().position(|h| *h == name);
            let (Some(ch1), Some(ch2), Some(clag), Some(cr4), Some(cn4), Some(cfdr)) =
                (col("h1"), col("h2"), col("lag"), col("real_z4"), col("null_z4"), col("fdr_r"))
            else {
                continue;
            };
            for line in lines {
                let f: Vec<&str> = line.split(',').collect();
                let num = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
                let key = (num(ch1) as usize, num(ch2) as usize);
                let row = self.lag_table.entry(key).or_default().entry(num(clag) as usize).or_default();
                row.real_z4 += num(cr4);
                row.null_z4 += num(cn4);
                row.matrices += 1;
                row.fdr += usize::from(f.get(cfdr).is_some_and(|v| !v.is_empty()));
            }
        }
        if !self.lag_table.contains_key(&self.pair) {
            if let Some(k) = self.lag_table.keys().next() {
                self.pair = *k;
            }
        }
    }

    fn census_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Real vs chance, by lag");
        ui.horizontal(|ui| {
            if ui.button(if self.lag_table.is_empty() { "Load census summaries" } else { "Reload" }).clicked() {
                self.load_lag_table();
            }
            let pairs: Vec<(usize, usize)> = self.lag_table.keys().copied().collect();
            egui::ComboBox::from_label("leader / target horizon (days)")
                .selected_text(format!("{} / {}", self.pair.0, self.pair.1))
                .show_ui(ui, |ui| {
                    for p in pairs {
                        ui.selectable_value(&mut self.pair, p, format!("{} / {}", p.0, p.1));
                    }
                });
        });
        if let Some(rows) = self.lag_table.get(&self.pair).cloned() {
            ui.label("Entries with |z| ≥ 4, summed over all windows. Real / chance near 1 means no lead-lag beyond chance (lag 0 is same-day co-movement).");
            let points: Vec<[f64; 2]> = rows.iter().filter(|(k, _)| **k > 0).map(|(k, r)| [*k as f64, r.real_z4 / r.null_z4.max(1.0)]).collect();
            Plot::new("ratio").height(220.0).legend(Legend::default()).show(ui, |p| {
                p.line(Line::new("real / chance (lags ≥ 1)", PlotPoints::from(points)));
                p.line(Line::new("chance level", PlotPoints::from(vec![[0.0, 1.0], [rows.keys().last().copied().unwrap_or(1) as f64, 1.0]])));
            });
            egui::ScrollArea::vertical().max_height(220.0).id_salt("lags").show(ui, |ui| {
                egui::Grid::new("lag_grid").striped(true).show(ui, |ui| {
                    for h in ["lag", "real", "chance", "real/chance", "matrices", "FDR cut-off"] {
                        ui.strong(h);
                    }
                    ui.end_row();
                    for (k, r) in &rows {
                        ui.label(k.to_string());
                        ui.label(format!("{:.0}", r.real_z4));
                        ui.label(format!("{:.0}", r.null_z4));
                        ui.label(format!("{:.2}", r.real_z4 / r.null_z4.max(1.0)));
                        ui.label(r.matrices.to_string());
                        ui.label(format!("{:.1}%", 100.0 * r.fdr as f64 / r.matrices.max(1) as f64));
                        ui.end_row();
                    }
                });
            });
        }
        ui.separator();
        ui.heading("A stock's strongest stored pairs");
        ui.horizontal(|ui| {
            ui.label("Ticker:");
            ui.text_edit_singleline(&mut self.ticker);
            if ui.add_enabled(self.pairs_rx.is_none() && self.store.is_some(), egui::Button::new("Find")).clicked() {
                let (tx, rx) = channel();
                self.pairs_rx = Some(rx);
                let (store, dir, ticker) = (self.store.clone().unwrap(), self.dir(), self.ticker.trim().to_uppercase());
                std::thread::spawn(move || {
                    let _ = tx.send(find_pairs(&store, &dir, &ticker).map_err(|e| format!("{e:#}")));
                });
            }
            if self.pairs_rx.is_some() {
                ui.spinner();
            }
        });
        if let Some(rx) = &self.pairs_rx {
            if let Ok(res) = rx.try_recv() {
                match res {
                    Ok(rows) => self.pair_rows = rows,
                    Err(e) => self.message = e,
                }
                self.pairs_rx = None;
            }
        }
        egui::ScrollArea::vertical().id_salt("pairs").show(ui, |ui| {
            egui::Grid::new("pair_grid").striped(true).show(ui, |ui| {
                for h in ["window", "partner", "role", "h1", "h2", "lag", "r", "z", "obs"] {
                    ui.strong(h);
                }
                ui.end_row();
                for r in &self.pair_rows {
                    ui.label(&r.window);
                    ui.label(&r.partner);
                    ui.label(r.role);
                    ui.label(r.h1.to_string());
                    ui.label(r.h2.to_string());
                    ui.label(r.lag.to_string());
                    ui.label(format!("{:+.3}", r.r));
                    ui.label(format!("{:+.1}", r.z));
                    ui.label(r.obs.to_string());
                    ui.end_row();
                }
            });
        });
    }

    fn holdout_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("$10,000 holdout test");
        ui.horizontal(|ui| {
            ui.label("Signal:");
            ui.radio_value(&mut self.family, 0, "census lead-lag pairs");
            ui.radio_value(&mut self.family, 1, "peer momentum");
        });
        egui::Grid::new("holdout_opts").min_col_width(120.0).show(ui, |ui| {
            ui.label("First day treated as the future");
            ui.add(egui::TextEdit::singleline(&mut self.start).desired_width(110.0));
            ui.end_row();
            ui.label("Stocks held");
            ui.add(egui::DragValue::new(&mut self.picks).range(2..=200));
            ui.end_row();
            ui.label("Rebalance every N trading days");
            ui.add(egui::DragValue::new(&mut self.rebalance).range(1..=63));
            ui.end_row();
            ui.label("Costs (bps per trade, comma-separated)");
            ui.add(egui::TextEdit::singleline(&mut self.costs).desired_width(110.0));
            ui.end_row();
            ui.label("Random-guess accounts");
            ui.add(egui::DragValue::new(&mut self.random).range(100..=10000));
            ui.end_row();
        });
        ui.horizontal(|ui| {
            let can_run = self.holdout_rx.is_none() && self.store.is_some();
            if ui.add_enabled(can_run, egui::Button::new("Run")).clicked() {
                self.start_holdout();
            }
            if let Some(t) = self.holdout_started {
                if self.holdout_rx.is_some() {
                    ui.spinner();
                    ui.label(format!("running… {:.0}s (the first run loads 20 years of data)", t.elapsed().as_secs_f64()));
                }
            }
        });
        if let Some(rx) = &self.holdout_rx {
            if let Ok(res) = rx.try_recv() {
                match res {
                    Ok(r) => self.holdout = Some(r),
                    Err(e) => self.message = e,
                }
                self.holdout_rx = None;
            }
        }
        if !self.message.is_empty() && self.holdout.is_none() {
            ui.label(&self.message);
        }
        let Some(r) = &self.holdout else { return };
        ui.separator();
        ui.label(format!(
            "{} .. {} ({} trading days) · {} stocks could be picked · picks beat the market {:.1}% of the time · rank IC t = {:+.2} · long-short t = {:+.2}",
            c2_data::date_string(r.first_day), c2_data::date_string(r.last_day), r.days, r.candidates, 100.0 * r.hit_rate, r.ic_t, r.ls_t
        ));
        egui::Grid::new("holdout_results").striped(true).show(ui, |ui| {
            for h in ["", "final value", "return", "random guessers (median)", "share of random at least as good"] {
                ui.strong(h);
            }
            ui.end_row();
            for b in &r.benchmarks {
                ui.label(&b.label);
                ui.label(format!("${:.0}", b.final_value));
                ui.label(pct(b.final_value));
                ui.label("");
                ui.label("");
                ui.end_row();
            }
            for (c, s, med, share) in &r.strategy {
                ui.label(format!("Picks, {c} bps per trade"));
                ui.label(format!("${s:.0}"));
                ui.label(pct(*s));
                ui.label(format!("${med:.0}"));
                ui.label(format!("{:.1}%  (p = {share:.3})", 100.0 * share));
                ui.end_row();
            }
        });
        let series = |f: fn(&(i32, f64, f64, f64, f64)) -> f64| -> PlotPoints {
            PlotPoints::from(r.curve.iter().enumerate().filter(|(_, c)| f(c).is_finite()).map(|(k, c)| [k as f64, f(c)]).collect::<Vec<_>>())
        };
        Plot::new("equity").legend(Legend::default()).show(ui, |p| {
            p.line(Line::new("picks (first cost level)", series(|c| c.1)));
            p.line(Line::new("SPY", series(|c| c.2)));
            p.line(Line::new("QQQ", series(|c| c.3)));
            p.line(Line::new("random guessers (median)", series(|c| c.4)));
        });
    }

    fn start_holdout(&mut self) {
        let Some(store) = self.store.clone() else { return };
        let (tx, rx) = channel();
        self.holdout_rx = Some(rx);
        self.holdout_started = Some(Instant::now());
        self.message.clear();
        let dir = self.dir();
        let plan_cache = self.plan.clone();
        let family = if self.family == 0 {
            Family::CensusPairs
        } else {
            Family::PeerMomentum { peers: 20, lookback: 21, window: 756 }
        };
        let start = self.start.clone();
        let (picks, rebalance, random) = (self.picks, self.rebalance, self.random);
        let costs: Vec<f64> = self.costs.split(',').filter_map(|c| c.trim().parse().ok()).collect();
        std::thread::spawn(move || {
            let res = (|| -> anyhow::Result<Report> {
                let plan = {
                    let mut guard = plan_cache.lock().unwrap();
                    if guard.is_none() {
                        let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
                        let full = m["config"].as_str().unwrap_or("").contains("full_history=true");
                        let cfg = if full { c2_census::grid::Config::full() } else { c2_census::grid::Config::smoke() };
                        *guard = Some(Arc::new(Plan::new(&store, cfg)?));
                    }
                    guard.clone().unwrap()
                };
                let opt = Options {
                    family,
                    rebalance,
                    start: c2_data::parse_date(&start)?,
                    end: i32::MAX,
                    picks,
                    costs_bps: if costs.is_empty() { vec![0.0] } else { costs },
                    random_runs: random,
                    cash: 10_000.0,
                    all_windows: false,
                };
                c2_census::holdout::holdout(&store, &plan, &dir, &opt)
            })();
            let _ = tx.send(res.map_err(|e| format!("{e:#}")));
        });
    }

    fn data_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Price store");
        ui.label(&self.store_note);
        ui.label("Stooq prices are split- and dividend-adjusted; test tickers are excluded and one-day bad prints repaired when panels are built.");
        ui.separator();
        egui::Grid::new("decades").striped(true).show(ui, |ui| {
            for h in ["decade", "symbols", "rows", "0% days", "under $1", "close only"] {
                ui.strong(h);
            }
            ui.end_row();
            for r in &self.decades {
                let p = |a: usize, b: usize| format!("{:.1}%", 100.0 * a as f64 / b.max(1) as f64);
                ui.label(format!("{}s", r.decade));
                ui.label(r.symbols.to_string());
                ui.label(r.rows.to_string());
                ui.label(p(r.zero_return, r.with_prev));
                ui.label(p(r.under_1, r.rows));
                ui.label(p(r.close_only, r.rows));
                ui.end_row();
            }
        });
    }
}

/// The strongest stored real pairs involving `ticker`, as leader or target, across all windows.
fn find_pairs(store: &Store, dir: &Path, ticker: &str) -> anyhow::Result<Vec<PairRow>> {
    let id = store.id(ticker).ok_or_else(|| anyhow::anyhow!("{ticker} is not in the price store"))? as u32;
    let windows: Vec<String> = std::fs::read_to_string(dir.join("windows.csv"))?
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            format!("{}..{}", f.get(2).unwrap_or(&""), f.get(3).unwrap_or(&""))
        })
        .collect();
    let mut rows = Vec::new();
    for e in std::fs::read_dir(dir.join("tasks"))?.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("rec") {
            continue;
        }
        for r in c2_census::analyze::read_records(&path)? {
            if r.flags & FLAG_NULL != 0 || r.lag == 0 || (r.leader != id && r.target != id) {
                continue;
            }
            let (partner, role) = if r.leader == id { (r.target, "leads") } else { (r.leader, "follows") };
            rows.push(PairRow {
                window: windows.get(r.window as usize).cloned().unwrap_or_default(),
                partner: store.symbols.get(partner as usize).map(|s| s.ticker.clone()).unwrap_or_default(),
                role,
                h1: r.h1,
                h2: r.h2,
                lag: r.lag,
                r: r.r,
                z: r.z,
                obs: r.obs,
            });
        }
    }
    rows.sort_by(|a, b| b.z.abs().total_cmp(&a.z.abs()));
    rows.truncate(200);
    Ok(rows)
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        if let Some((prefix, frames, requested)) = self.shot.clone() {
            if frames == 0 {
                match std::env::var("C2_GUI_TAB").as_deref() {
                    Ok("census") => {
                        self.tab = Tab::Census;
                        self.load_lag_table();
                        if let (Ok(t), Some(store)) = (std::env::var("C2_GUI_TICKER"), self.store.clone()) {
                            self.ticker = t.clone();
                            let (tx, rx) = channel();
                            self.pairs_rx = Some(rx);
                            let dir = self.dir();
                            std::thread::spawn(move || {
                                let _ = tx.send(find_pairs(&store, &dir, &t).map_err(|e| format!("{e:#}")));
                            });
                        }
                    }
                    Ok("holdout") => {
                        self.tab = Tab::Holdout;
                        self.start_holdout();
                    }
                    Ok("data") => self.tab = Tab::Data,
                    Ok("strongest") => self.tab = Tab::Strongest,
                    Ok("stock") => {
                        self.tab = Tab::Stock;
                        if let Ok(t) = std::env::var("C2_GUI_TICKER") {
                            self.tl.input = t;
                        }
                        if let Ok(w) = std::env::var("C2_GUI_WINDOW") {
                            self.tl.window = w;
                        }
                        self.tl.significant_only = std::env::var("C2_GUI_SIGNIFICANT").is_ok();
                        self.load_timeline();
                    }
                    Ok("forecast") => {
                        self.tab = Tab::Forecast;
                        self.fc.ticker = std::env::var("C2_GUI_TICKER").unwrap_or_else(|_| "AAPL".into());
                        self.tl.input = self.fc.ticker.clone();
                        self.fc.as_of = std::env::var("C2_GUI_AS_OF").ok().and_then(|d| c2_data::parse_date(&d).ok());
                        self.fc.horizon = std::env::var("C2_GUI_DAYS").ok().and_then(|d| d.parse().ok()).unwrap_or(21);
                        self.fc.significant_only = std::env::var("C2_GUI_SIGNIFICANT").is_ok();
                        self.load_forecast();
                    }
                    _ => self.tab = Tab::Status,
                }
            }
            let ready = self.holdout_rx.is_none() && self.pairs_rx.is_none() && self.tl.rx.is_none() && self.rank_rx.is_none() && self.fc.rx.is_none()
                && !(self.tab == Tab::Strongest && self.rank.is_empty());
            if ready && frames > 30 && !requested {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
                self.shot = Some((prefix.clone(), frames + 1, true));
            } else {
                self.shot = Some((prefix.clone(), frames + 1, requested));
            }
            let image = ctx.input(|i| {
                i.events.iter().find_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(img) = image {
                let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                let _ = std::fs::write(format!("{prefix}.rgba"), bytes);
                let _ = std::fs::write(format!("{prefix}.txt"), format!("{} {}", img.size[0], img.size[1]));
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Status, "Status");
                ui.selectable_value(&mut self.tab, Tab::Strongest, "Strongest");
                ui.selectable_value(&mut self.tab, Tab::Stock, "Stock timeline");
                ui.selectable_value(&mut self.tab, Tab::Forecast, "Forecast");
                ui.selectable_value(&mut self.tab, Tab::Census, "Census");
                ui.selectable_value(&mut self.tab, Tab::Holdout, "Holdout");
                ui.selectable_value(&mut self.tab, Tab::Data, "Data");
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::Status => self.status_tab(ui),
            Tab::Strongest => self.strongest_tab(ui),
            Tab::Stock => self.stock_tab(ui),
            Tab::Forecast => self.forecast_tab(ui),
            Tab::Census => self.census_tab(ui),
            Tab::Holdout => self.holdout_tab(ui),
            Tab::Data => self.data_tab(ui),
        });
        ctx.request_repaint_after(Duration::from_millis(500));
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 780.0]).with_title("correlation2"),
        ..Default::default()
    };
    eframe::run_native("correlation2", options, Box::new(|_cc| Ok(Box::new(App::new()))))
}
