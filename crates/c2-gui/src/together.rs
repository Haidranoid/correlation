//! The Together tab: same-day co-movement tools (`c2_census::together`), the same four as the
//! `c2 portfolio`, `c2 diversify`, `c2 hedge` and `c2 groups` commands. They don't read the
//! census, so switching runs keeps their results; a reopened price store drops them.

use std::sync::{Arc, Mutex};

use c2_census::together::{self, Candidate, Diversifiers, GroupsMap, HedgeReport, HedgeRow, PortfolioReport, Universe, words};
use c2_data::Store;
use eframe::egui;
use egui::RichText;

use super::{App, BLUE, GREY, Job, ORANGE, YELLOW_NOTE};

/// The return panel every tool reads, with the store generation it was built from.
type UniverseSlot = Arc<Mutex<Option<(u64, Arc<Universe>)>>>;

/// Years of returns: the last year, the year before it, and the groups map's earlier years.
const YEARS: usize = 6;

#[derive(Clone, Copy, PartialEq, Default)]
pub(crate) enum Mode {
    #[default]
    Portfolio,
    Diversify,
    Hedge,
    Groups,
}

/// (store generation, holdings as typed, added share in 1/10 %, volatility limit in 1/10 %).
type PortfolioKey = (u64, String, i64, i64);
/// (store generation, ticker).
type HedgeKey = (u64, String);
/// (store generation, groups, stocks).
type GroupsKey = (u64, usize, usize);

pub(crate) struct TogetherView {
    pub mode: Mode,
    pub holdings: String,
    pub add_pct: f64,
    pub max_vol_pct: f64,
    /// Show the correlation table of the year before instead of the last year's.
    pub before: bool,
    pub hedge_input: String,
    pub groups: usize,
    pub stocks: usize,
    slot: UniverseSlot,
    pub portfolio: Job<PortfolioKey, (PortfolioReport, Diversifiers)>,
    pub hedge: Job<HedgeKey, HedgeReport>,
    pub map: Job<GroupsKey, GroupsMap>,
    /// The inputs last asked for in each tool (None: not yet, so the first visit asks).
    asked: (Option<PortfolioKey>, Option<HedgeKey>, Option<GroupsKey>),
    /// The groups map's heatmap, made once per result (keyed by the job's generation).
    heat: Option<(u64, egui::TextureHandle)>,
    pub selected: Option<usize>,
}

impl TogetherView {
    pub fn new() -> TogetherView {
        TogetherView {
            mode: Mode::Portfolio,
            holdings: "AAPL MSFT NVDA GOOGL AMZN META".into(),
            add_pct: 10.0,
            max_vol_pct: 75.0,
            before: false,
            hedge_input: "XOM".into(),
            groups: 20,
            stocks: 1000,
            slot: Arc::new(Mutex::new(None)),
            portfolio: Job::default(),
            hedge: Job::default(),
            map: Job::default(),
            asked: (None, None, None),
            heat: None,
            selected: None,
        }
    }

    /// Drops every result and the return panel (the price store changed).
    pub fn clear(&mut self) {
        self.portfolio.clear();
        self.hedge.clear();
        self.map.clear();
        self.asked = (None, None, None);
        self.heat = None;
        if let Ok(mut guard) = self.slot.try_lock() {
            *guard = None;
        }
    }

    pub fn poll(&mut self, report: &dyn Fn(&str, bool, std::time::Duration) -> bool) {
        report("portfolio check", self.portfolio.poll(), self.portfolio.took);
        report("hedge finder", self.hedge.poll(), self.hedge.took);
        if report("groups map", self.map.poll(), self.map.took) {
            self.heat = None;
        }
    }

    pub fn is_running(&self) -> bool {
        self.portfolio.is_running() || self.hedge.is_running() || self.map.is_running()
    }

    /// Whether the shown tool has what it last asked for, for self-checks.
    pub fn ready(&self) -> bool {
        match self.mode {
            Mode::Portfolio | Mode::Diversify => self.asked.0.as_ref().is_some_and(|k| self.portfolio.get(k).is_some()),
            Mode::Hedge => self.asked.1.as_ref().is_some_and(|k| self.hedge.get(k).is_some()),
            Mode::Groups => self.asked.2.as_ref().is_some_and(|k| self.map.get(k).is_some()),
        }
    }
}

/// The shared return panel for this store generation, built on first use.
fn universe(slot: &UniverseSlot, store: &Store, generation: u64) -> Result<Arc<Universe>, String> {
    let mut guard = slot.lock().map_err(|_| "the return panel's lock was poisoned".to_string())?;
    if let Some((g, u)) = guard.as_ref()
        && *g == generation
    {
        return Ok(u.clone());
    }
    let u = Arc::new(Universe::build(store, None, YEARS).map_err(|e| format!("{e:#}"))?);
    *guard = Some((generation, u.clone()));
    Ok(u)
}

fn round2(r: f64) -> f64 {
    (r * 100.0).round() / 100.0
}

fn corr_color(r: f64) -> egui::Color32 {
    let r = round2(r);
    if r >= 0.6 {
        ORANGE
    } else if r >= 0.3 {
        YELLOW_NOTE
    } else if r > -0.3 {
        GREY
    } else {
        BLUE
    }
}

fn corr_text(r: f64) -> RichText {
    RichText::new(format!("{r:+.2} {}", words(r))).color(corr_color(r))
}

fn before_text(r: Option<f64>) -> RichText {
    match r {
        Some(r) => RichText::new(format!("{r:+.2}")).color(corr_color(r)),
        None => RichText::new("new").color(GREY),
    }
}

fn pct(v: f64) -> String {
    format!("{:.1}%", 100.0 * v)
}

fn span(d: (i32, i32, usize)) -> String {
    format!("{} .. {} ({} days)", c2_data::date_string(d.0), c2_data::date_string(d.1), d.2)
}

fn span2(d: Option<(i32, i32)>) -> String {
    d.map(|d| format!("{} .. {}", c2_data::date_string(d.0), c2_data::date_string(d.1))).unwrap_or_else(|| "not in the data".into())
}

const SCALE: &str = "Scale: 0.9 or more almost always together, 0.6 usually, 0.3 often; -0.3 to 0.3 mostly independent; below -0.3 often opposite, -0.6 usually, -0.9 almost always.";

/// Heatmap colour of a market-relative correlation: dark at 0, orange to yellow as it rises,
/// blue below 0.
fn heat_color(r: f32) -> [u8; 4] {
    let lerp = |a: [f32; 3], b: [f32; 3], t: f32| -> [u8; 4] {
        let t = t.clamp(0.0, 1.0);
        [(a[0] + (b[0] - a[0]) * t) as u8, (a[1] + (b[1] - a[1]) * t) as u8, (a[2] + (b[2] - a[2]) * t) as u8, 255]
    };
    let dark = [24.0, 26.0, 34.0];
    if r >= 0.5 {
        lerp([255.0, 160.0, 40.0], [255.0, 240.0, 150.0], (r - 0.5) / 0.5)
    } else if r >= 0.0 {
        lerp(dark, [255.0, 160.0, 40.0], r / 0.5)
    } else {
        lerp(dark, [90.0, 170.0, 255.0], -r / 0.3)
    }
}

impl App {
    fn portfolio_key(&self) -> PortfolioKey {
        (self.store_gen, self.tg.holdings.trim().to_string(), (self.tg.add_pct * 10.0).round() as i64, (self.tg.max_vol_pct * 10.0).round() as i64)
    }

    pub(crate) fn start_portfolio(&mut self) {
        let Some(store) = self.store.clone() else { return };
        let key = self.portfolio_key();
        self.tg.asked.0 = Some(key.clone());
        let (slot, generation, text, add, max_vol) = (self.tg.slot.clone(), self.store_gen, key.1.clone(), self.tg.add_pct / 100.0, self.tg.max_vol_pct / 100.0);
        self.tg.portfolio.start(key, move || {
            let h = together::parse_holdings(&text).map_err(|e| format!("{e:#}"))?;
            let u = universe(&slot, &store, generation)?;
            let p = together::portfolio(&u, &h).map_err(|e| format!("{e:#}"))?;
            let d = together::diversifiers(&u, &p, add, max_vol, 15);
            Ok((p, d))
        });
    }

    pub(crate) fn start_hedge(&mut self) {
        let Some(store) = self.store.clone() else { return };
        let key: HedgeKey = (self.store_gen, self.tg.hedge_input.trim().to_uppercase());
        self.tg.asked.1 = Some(key.clone());
        let (slot, generation, ticker) = (self.tg.slot.clone(), self.store_gen, key.1.clone());
        self.tg.hedge.start(key, move || {
            let u = universe(&slot, &store, generation)?;
            together::hedges(&u, &ticker, 15).map_err(|e| format!("{e:#}"))
        });
    }

    pub(crate) fn start_groups(&mut self) {
        let Some(store) = self.store.clone() else { return };
        let key: GroupsKey = (self.store_gen, self.tg.groups, self.tg.stocks);
        self.tg.asked.2 = Some(key);
        self.tg.selected = None;
        let (slot, generation, k, n) = (self.tg.slot.clone(), self.store_gen, self.tg.groups, self.tg.stocks);
        self.tg.map.start(key, move || {
            let u = universe(&slot, &store, generation)?;
            together::groups(&u, k, n, YEARS - 1, &|_| {}).map_err(|e| format!("{e:#}"))
        });
    }

    pub(crate) fn together_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Together: what moves with what on the same day");
        ui.label(
            "Unlike lead-lag, same-day co-movement is real and lasting, which makes it useful for risk. Stocks and funds (ETFs) on daily returns of the last year, \
             with the year before to show whether a relationship held.",
        );
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tg.mode, Mode::Portfolio, "Portfolio check");
            ui.selectable_value(&mut self.tg.mode, Mode::Diversify, "Best diversifiers");
            ui.selectable_value(&mut self.tg.mode, Mode::Hedge, "Hedge finder");
            ui.selectable_value(&mut self.tg.mode, Mode::Groups, "Groups map");
        });
        ui.separator();
        if self.store.is_none() {
            ui.colored_label(super::RED_NOTE, &self.store_note);
            return;
        }
        match self.tg.mode {
            Mode::Portfolio | Mode::Diversify => self.portfolio_view(ui),
            Mode::Hedge => self.hedge_view(ui),
            Mode::Groups => self.groups_view(ui),
        }
    }

    fn holdings_box(&mut self, ui: &mut egui::Ui) {
        ui.label("Holdings: tickers, each with an amount (dollars or percent; only the proportions matter) or none for equal weights. Example: AAPL 30 MSFT 20 SPY 50");
        let mut go = false;
        ui.horizontal(|ui| {
            let r = ui.add(egui::TextEdit::multiline(&mut self.tg.holdings).desired_rows(2).desired_width(560.0));
            if r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.ctrl) {
                go = true;
            }
            ui.vertical(|ui| {
                if ui.add_enabled(!self.tg.portfolio.is_running(), egui::Button::new("Check")).clicked() {
                    go = true;
                }
                ui.small("or Ctrl+Enter");
            });
        });
        if self.tg.mode == Mode::Diversify {
            ui.horizontal(|ui| {
                ui.label("Volatility column: with");
                ui.add(egui::DragValue::new(&mut self.tg.add_pct).range(1.0..=50.0).suffix("%"));
                ui.label("of the money moved into each.  Leave out candidates more volatile than");
                ui.add(egui::DragValue::new(&mut self.tg.max_vol_pct).range(10.0..=500.0).suffix("%"));
                ui.label("a year (mostly leveraged funds).");
                if ui.button("Apply").clicked() {
                    go = true;
                }
            });
        }
        if go || self.tg.asked.0.is_none() {
            self.start_portfolio();
        }
    }

    fn portfolio_view(&mut self, ui: &mut egui::Ui) {
        self.holdings_box(ui);
        if self.tg.portfolio.is_running() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("computing (the first time reads six years of prices for every stock and fund, a few seconds)…");
            });
        }
        let (p, d) = match &self.tg.portfolio.done {
            Some((k, Ok(r))) if k.0 == self.store_gen => r,
            Some((k, Err(e))) if k.0 == self.store_gen => {
                ui.colored_label(super::RED_NOTE, e);
                return;
            }
            _ => return,
        };
        ui.separator();
        let n = p.holdings.len();
        ui.label(format!("Daily returns {}; a year earlier: {}.", span(p.days), span2(p.before_days)));
        if let Some(why) = &p.shortened {
            ui.colored_label(YELLOW_NOTE, format!("{why}, so only those days are used."));
        }
        ui.label(
            RichText::new(format!("Your {n} holding{} behave{} like {:.1} independent bet{}.", if n == 1 { "" } else { "s" }, if n == 1 { "s" } else { "" }, p.bets, if (p.bets - 1.0).abs() < 0.05 { "" } else { "s" }))
                .strong()
                .size(16.0),
        );
        ui.horizontal(|ui| {
            ui.label(format!("Volatility {} a year for the portfolio, against {} for its holdings on average (by weight). Beta to SPY {:.2}; correlation with SPY", pct(p.vol), pct(p.avg_vol), p.beta));
            ui.label(corr_text(p.spy_corr));
        });
        if self.tg.mode == Mode::Diversify {
            diversifier_tables(ui, p, d);
            return;
        }
        egui::ScrollArea::vertical().id_salt("portfolio").show(ui, |ui| {
            ui.add_space(4.0);
            egui::Grid::new("holdings").striped(true).min_col_width(60.0).show(ui, |ui| {
                for h in ["holding", "kind", "weight", "volatility", "beta to SPY", "share of risk", "with the rest of the portfolio"] {
                    ui.strong(h);
                }
                ui.end_row();
                for h in &p.holdings {
                    ui.label(&h.ticker);
                    ui.label(together::kind(h.etf));
                    ui.label(pct(h.weight));
                    ui.label(pct(h.vol));
                    ui.label(format!("{:.2}", h.beta));
                    ui.label(pct(h.risk_share));
                    ui.label(corr_text(h.corr_rest));
                    ui.end_row();
                }
            });
            ui.add_space(6.0);
            let dups = p.duplicates(0.8);
            ui.strong("Near-duplicates (correlation 0.8 or more)");
            if dups.is_empty() {
                ui.label("none");
            }
            for (a, b, r, was) in dups {
                ui.horizontal(|ui| {
                    ui.label(format!("{} and {}:", p.holdings[a].ticker, p.holdings[b].ticker));
                    ui.label(corr_text(r));
                    ui.label("a year earlier");
                    ui.label(before_text(was));
                });
            }
            if n >= 2 {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.strong("Correlations");
                    ui.radio_value(&mut self.tg.before, false, format!("last year ({})", span(p.days)));
                    ui.radio_value(&mut self.tg.before, true, format!("a year earlier ({})", span2(p.before_days)));
                });
                egui::ScrollArea::horizontal().id_salt("corr").show(ui, |ui| {
                    egui::Grid::new("corr_table").striped(true).show(ui, |ui| {
                        ui.label("");
                        for h in &p.holdings {
                            ui.strong(&h.ticker);
                        }
                        ui.end_row();
                        for a in 0..n {
                            ui.strong(&p.holdings[a].ticker);
                            for b in 0..n {
                                if a == b {
                                    ui.label("");
                                } else if self.tg.before {
                                    ui.label(before_text(p.corr_before[a * n + b]));
                                } else {
                                    let r = p.corr[a * n + b];
                                    ui.label(RichText::new(format!("{r:+.2}")).color(corr_color(r))).on_hover_text(words(r));
                                }
                            }
                            ui.end_row();
                        }
                    });
                });
            }
            ui.add_space(6.0);
            ui.small(SCALE);
            ui.small("Independent bets: (weighted average volatility / portfolio volatility) squared, which is the number of holdings when they move independently and 1 when they move as one. Share of risk: each holding's part of the portfolio's variance.");
        });
    }

    fn hedge_view(&mut self, ui: &mut egui::Ui) {
        let mut go = self.tg.asked.1.is_none();
        ui.horizontal(|ui| {
            ui.label("Stock or fund:");
            let r = ui.add(egui::TextEdit::singleline(&mut self.tg.hedge_input).desired_width(90.0));
            if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                go = true;
            }
            if ui.add_enabled(!self.tg.hedge.is_running(), egui::Button::new("Find")).clicked() {
                go = true;
            }
            if self.tg.hedge.is_running() {
                ui.spinner();
            }
        });
        if go {
            self.start_hedge();
        }
        let r = match &self.tg.hedge.done {
            Some((k, Ok(r))) if k.0 == self.store_gen => r,
            Some((k, Err(e))) if k.0 == self.store_gen => {
                ui.colored_label(super::RED_NOTE, e);
                return;
            }
            _ => return,
        };
        ui.separator();
        ui.label(RichText::new(format!("{} ({}, {} a year volatility)", r.ticker, together::kind(r.etf), pct(r.vol))).strong().size(16.0));
        ui.label(format!("Daily returns {}; a year earlier: {}.", span(r.days), span2(r.before_days)));
        ui.label(format!(
            "Hedge ratio: dollars of it per $1 of {} that offset the most risk (sell it when the ratio is positive, buy it when negative). Risk left: the share of {}'s volatility that remains after that hedge.",
            r.ticker, r.ticker
        ));
        egui::ScrollArea::vertical().id_salt("hedge").show(ui, |ui| {
            hedge_grid(ui, "closest", &format!("Tracks {} most closely", r.ticker), &r.closest);
            ui.add_space(8.0);
            hedge_grid(ui, "opposite", &format!("Moves most opposite to {}", r.ticker), &r.opposite);
            ui.add_space(6.0);
            ui.small(format!("Looked at {} liquid stocks and funds. Leveraged and inverse funds are included; their volatility shows which they are. {SCALE}", r.scanned));
        });
    }

    fn groups_view(&mut self, ui: &mut egui::Ui) {
        let mut go = self.tg.asked.2.is_none();
        ui.horizontal(|ui| {
            ui.label("The");
            ui.add(egui::DragValue::new(&mut self.tg.stocks).range(100..=3000));
            ui.label("most-traded stocks in");
            ui.add(egui::DragValue::new(&mut self.tg.groups).range(2..=60));
            ui.label("groups");
            if ui.add_enabled(!self.tg.map.is_running(), egui::Button::new("Build")).clicked() {
                go = true;
            }
            if self.tg.map.is_running() {
                ui.spinner();
                ui.label("grouping this year and each of 5 earlier years…");
            }
        });
        if go {
            self.start_groups();
        }
        let generation = self.tg.map.generation;
        let m = match &self.tg.map.done {
            Some((k, Ok(m))) if k.0 == self.store_gen => m,
            Some((k, Err(e))) if k.0 == self.store_gen => {
                ui.colored_label(super::RED_NOTE, e);
                return;
            }
            _ => return,
        };
        let n = m.stocks.len();
        ui.label(format!(
            "Daily returns {}. Groups of stocks that move together once the market (the equal-weight average of these {n} stocks, times each stock's beta) is taken out; \
             each is named by the fund that follows its average most closely. Kept together: of a group's pairs, the share grouped together again when the same \
             grouping is made on an earlier year alone, against what chance gives.",
            span(m.days)
        ));
        if self.tg.heat.as_ref().is_none_or(|(g, _)| *g != generation) {
            let rgba: Vec<u8> = m.corr.iter().enumerate().flat_map(|(ij, &r)| if ij / n == ij % n { [255, 255, 255, 255] } else { heat_color(r) }).collect();
            let tex = ui.ctx().load_texture("groups-heatmap", egui::ColorImage::from_rgba_unmultiplied([n, n], &rgba), egui::TextureOptions::NEAREST);
            self.tg.heat = Some((generation, tex));
        }
        let mut select = None;
        let height = ui.available_height();
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(480.0);
                egui::ScrollArea::vertical().id_salt("groups").max_height(height).auto_shrink([false, false]).show(ui, |ui| {
                for (g, grp) in m.groups.iter().enumerate() {
                    let tracks = grp.tracks.as_ref().map(|(t, r)| format!("{t} {r:+.2}")).unwrap_or_else(|| "no fund".into());
                    let kept = grp.kept.first().copied().flatten().map(|(s, c)| format!("; a year back {:.0}% kept (chance {:.0}%)", 100.0 * s, 100.0 * c)).unwrap_or_default();
                    let head = format!("{}. {tracks} · {} stocks · together {:+.2}{kept}", g + 1, grp.members.len(), grp.corr_relative);
                    if ui.selectable_label(self.tg.selected == Some(g), RichText::new(head).strong()).clicked() {
                        select = Some(if self.tg.selected == Some(g) { None } else { Some(g) });
                    }
                    let names: Vec<&str> = grp.members.iter().take(if self.tg.selected == Some(g) { usize::MAX } else { 8 }).map(|&i| m.tickers[i].as_str()).collect();
                    let more = if self.tg.selected != Some(g) && grp.members.len() > 8 { format!(" … {} more", grp.members.len() - 8) } else { String::new() };
                    ui.label(format!("{}{more}", names.join(" ")));
                    if self.tg.selected == Some(g) {
                        ui.label(format!(
                            "Together {:+.2} without the market, {:+.2} with it; the group's return over the year {}.",
                            grp.corr_relative,
                            grp.corr_raw,
                            pct(grp.year_return)
                        ));
                        let years: Vec<String> = grp
                            .kept
                            .iter()
                            .zip(&m.earlier)
                            .map(|(k, &(_, last))| {
                                let y = c2_data::civil_from_days(last).0;
                                k.map(|(s, c)| format!("{y}: {:.0}% (chance {:.0}%)", 100.0 * s, 100.0 * c)).unwrap_or_else(|| format!("{y}: too few of its stocks then"))
                            })
                            .collect();
                        ui.label(format!("Kept together, year by year back: {}", years.join(", ")));
                    }
                    ui.add_space(4.0);
                }
                });
            });
            if let Some((_, tex)) = &self.tg.heat {
                let side = ui.available_width().min(height).max(200.0);
                let resp = ui.add(egui::Image::from_texture(tex).fit_to_exact_size(egui::vec2(side, side)).sense(egui::Sense::click()));
                let rect = resp.rect;
                let painter = ui.painter_at(rect);
                let at = |i: usize| i as f32 / n as f32 * side;
                let mut start = 0;
                for (g, grp) in m.groups.iter().enumerate() {
                    let end = start + grp.members.len();
                    let r = egui::Rect::from_min_max(rect.min + egui::vec2(at(start), at(start)), rect.min + egui::vec2(at(end), at(end)));
                    let stroke = if self.tg.selected == Some(g) { egui::Stroke::new(2.5, egui::Color32::WHITE) } else { egui::Stroke::new(0.8, egui::Color32::from_gray(150)) };
                    painter.rect_stroke(r, 0.0, stroke, egui::StrokeKind::Middle);
                    start = end;
                }
                if let Some(pos) = resp.hover_pos() {
                    let (j, i) = (((pos.x - rect.min.x) / side * n as f32) as usize, ((pos.y - rect.min.y) / side * n as f32) as usize);
                    if i < n && j < n {
                        let r = m.corr[i * n + j] as f64;
                        resp.clone().on_hover_text(format!("{} and {}: {:+.2} {} (market taken out); groups {} and {}", m.tickers[i], m.tickers[j], r, words(r), m.group_of[i] + 1, m.group_of[j] + 1));
                        if resp.clicked() {
                            select = Some(Some(m.group_of[i]));
                        }
                    }
                }
            }
        });
        if let Some(s) = select {
            self.tg.selected = s;
        }
    }
}

fn diversifier_tables(ui: &mut egui::Ui, p: &PortfolioReport, d: &Diversifiers) {
    ui.label(
        "The stocks and funds that move least with your portfolio on the same day, lowest correlation first. Correlation doesn't change with size or leverage, \
         so a 3x fund ranks with its 1x version; funds that move almost exactly alike are shown once, under the least volatile.",
    );
    egui::ScrollArea::vertical().id_salt("diversify").show(ui, |ui| {
        let table = |ui: &mut egui::Ui, id: &str, title: &str, rows: &[Candidate]| {
            ui.add_space(6.0);
            ui.strong(title);
            if rows.is_empty() {
                ui.label("none");
                return;
            }
            egui::Grid::new(id).striped(true).min_col_width(50.0).show(ui, |ui| {
                for h in ["", "kind", "volatility", "with the portfolio", "a year earlier", &format!("portfolio volatility with {:.0}% in", 100.0 * d.add), "moves almost exactly like it"] {
                    ui.strong(h);
                }
                ui.end_row();
                for c in rows {
                    ui.label(&c.ticker);
                    ui.label(together::kind(c.etf));
                    ui.label(pct(c.vol));
                    ui.label(corr_text(c.corr));
                    ui.label(before_text(c.corr_before));
                    ui.label(format!("{} ({:+.1})", pct(c.vol_after), 100.0 * (c.vol_after - p.vol)));
                    let alike = match c.alike.len() {
                        0 => String::new(),
                        k if k <= 4 => c.alike.join(", "),
                        k => format!("{} and {} more", c.alike[..3].join(", "), k - 3),
                    };
                    ui.label(alike).on_hover_text(c.alike.join(", "));
                    ui.end_row();
                }
            });
        };
        table(ui, "funds", "Funds", &d.funds);
        table(ui, "stocks", "Single stocks", &d.stocks);
        table(ui, "hedges", "Moving opposite: hedges rather than diversifiers (correlation -0.3 or lower with the portfolio, or -0.5 or lower with SPY, such as inverse funds)", &d.hedges);
        ui.add_space(6.0);
        ui.small(format!(
            "Looked at {} liquid stocks and funds (a close on 90% of days, last close $5 or more, median dollar volume $5M or more). Left out: {} cash-like ones (under 3% a year volatility) and {} more volatile than {:.0}% a year. {SCALE}",
            d.scanned,
            d.cash_like,
            d.too_volatile,
            100.0 * d.max_vol
        ));
    });
}

fn hedge_grid(ui: &mut egui::Ui, id: &str, title: &str, rows: &[HedgeRow]) {
    ui.strong(title);
    egui::Grid::new(id).striped(true).min_col_width(50.0).show(ui, |ui| {
        for h in ["", "kind", "volatility", "same day", "a year earlier", "hedge ratio", "risk left"] {
            ui.strong(h);
        }
        ui.end_row();
        for r in rows {
            ui.label(&r.ticker);
            ui.label(together::kind(r.etf));
            ui.label(pct(r.vol));
            ui.label(corr_text(r.corr));
            ui.label(before_text(r.corr_before));
            ui.label(format!("{:.2}", r.ratio));
            ui.label(pct(r.risk_left));
            ui.end_row();
        }
    });
}
