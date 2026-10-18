//! `c2-gui`: native desktop front end for correlation2.
//!
//! It drives the `c2` command (census runs happen in their own detached process, so closing the
//! window never interrupts one) and reads its output files. Tabs:
//!   Status   census progress and ETA; start, pause, resume, stop
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
use egui_plot::{Legend, Line, Plot, PlotPoints};

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Status,
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
        }
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
                    _ => self.tab = Tab::Status,
                }
            }
            let ready = self.holdout_rx.is_none() && self.pairs_rx.is_none();
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
                ui.selectable_value(&mut self.tab, Tab::Census, "Census");
                ui.selectable_value(&mut self.tab, Tab::Holdout, "Holdout");
                ui.selectable_value(&mut self.tab, Tab::Data, "Data");
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::Status => self.status_tab(ui),
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
