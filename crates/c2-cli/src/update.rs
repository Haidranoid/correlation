//! Scheduled data refresh and the forward prediction log (SPEC.md section 5).
//!
//! `c2 update` ingests any Stooq zip dropped into `<data dir>/inbox/` (Stooq downloads now need
//! a CAPTCHA, so fetching is manual for the time being) and archives it under
//! `raw/stooq/<date>/`. Then, whenever the default census (`census/LATEST`) is not a finished
//! full census of the price data as it now is, it runs one under a dated run name and makes it
//! the default (a full census takes ~12 minutes, so "refine as new data arrives" currently means
//! rerun; per-pair running sums are stored for a future live tracker). That check looks at the
//! census itself, not at whether this update ingested anything, so data ingested by `c2 ingest`,
//! `--smoke` or `--no-census`, or a census that failed or was stopped, is caught up on by the
//! next plain `c2 update`. Last, once the default census is the full census of the current data,
//! it appends today's picks for each family from it to an append-only forward log (one row per
//! data day and family, written before the outcomes exist); while it isn't, it logs nothing, so
//! an older census never takes a day's row. `c2 forward-report` scores past entries once their
//! holding period has passed.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use c2_census::grid::Config;
use c2_census::holdout::Family;
use c2_census::meta::{self, RunMeta};
use c2_census::run;
use c2_data::{Calendar, Store};

pub struct UpdateOptions {
    pub zips: Vec<PathBuf>,
    pub smoke: bool,
    pub no_census: bool,
}

fn update_log() -> PathBuf {
    c2_data::data_dir().join("update.log")
}

fn log(msg: &str) {
    let line = format!("[{}] {msg}", super::hms_now());
    println!("{line}");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(update_log()) {
        let _ = writeln!(f, "{line}");
    }
}

/// (rows, last trading day) of the store, if there is one.
fn fingerprint(dir: &Path) -> Option<(usize, i32)> {
    let s = Store::open(dir).ok()?;
    let last = Calendar::trading_days(&s).ok()?.days.last().copied()?;
    Some((s.rows(), last))
}

pub fn update(opt: UpdateOptions) -> Result<()> {
    let data = c2_data::data_dir();
    let inbox = data.join("inbox");
    std::fs::create_dir_all(&inbox)?;
    let mut zips = opt.zips.clone();
    if zips.is_empty() {
        for e in std::fs::read_dir(&inbox)?.flatten() {
            if e.path().extension().and_then(|x| x.to_str()) == Some("zip") {
                zips.push(e.path());
            }
        }
    }
    let store_dir = super::store_dir();
    if zips.is_empty() {
        log(&format!("no new zips in {}", inbox.display()));
    } else {
        // `--smoke` ingests and archives the inbox too: the store it builds is the one a plain
        // update would build, and the census step below goes by whether the default census is
        // out of date with the store, so the next plain `c2 update` still runs the full census.
        let before = fingerprint(&store_dir);
        // Census records hold store ids, which a rebuilt store renumbers: first give every run
        // without one a copy of the symbol table its ids refer to.
        let saved = meta::snapshot_symbols_for_all_runs().context("could not save the stock lists of the existing census runs, so the store was not rebuilt")?;
        let names = |want: meta::Snapshot| saved.iter().filter(|(_, s)| *s == want).map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ");
        let (written, lost) = (names(meta::Snapshot::Written), names(meta::Snapshot::Unrecoverable));
        if !written.is_empty() {
            log(&format!("saved the stock list of census runs {written}"));
        }
        if !lost.is_empty() {
            log(&format!("census runs {lost} have no copy of their stock list and the store already changed since they ran; their results can't be matched to stocks"));
        }
        let paths: Vec<&Path> = zips.iter().map(|p| p.as_path()).collect();
        let r = c2_data::ingest::ingest_stooq(&paths, &["us"], &store_dir)?;
        log(&format!("ingested {} files, {} rows from {} zip(s)", r.files, r.rows, zips.len()));
        let after = fingerprint(&store_dir);
        let changed = after != before;
        let stamp = after.map(|(_, d)| c2_data::date_string(d)).unwrap_or_else(|| "unknown".into());
        let raw = data.join("raw/stooq").join(&stamp);
        std::fs::create_dir_all(&raw)?;
        for z in &zips {
            if z.starts_with(&inbox) {
                let dest = raw.join(z.file_name().context("zip name")?);
                if std::fs::rename(z, &dest).is_err() {
                    std::fs::copy(z, &dest)?;
                    std::fs::remove_file(z)?;
                }
                log(&format!("archived {} -> {}", z.display(), dest.display()));
            }
        }
        log(&format!(
            "store {}: {:?} -> {:?} (rows, last day)",
            if changed { "changed" } else { "unchanged" },
            before.map(|(r, d)| (r, c2_data::date_string(d))),
            after.map(|(r, d)| (r, c2_data::date_string(d)))
        ));
    }

    let store = Store::open(&store_dir)?;
    // Whether to run a census depends on the default census and today's store, not on whether
    // this update ingested anything.
    let today = Today::of(&store)?;
    match latest_out_of_date(&today) {
        None => log(&format!("census {} is a finished full census of the current price data; no census to run", run::latest_run())),
        Some(why) if opt.no_census => log(&format!("no census run (--no-census), although {why}; the next plain `c2 update` runs it")),
        Some(why) => census(&store, &today, opt.smoke, &why)?,
    }
    // A day's picks are logged only from the full census of that day's data. A row from an older
    // census would hold the day's place for good (rows are written once per day and family), so
    // until the default census is current (after --smoke, --no-census, a stopped census or one
    // running elsewhere) nothing is logged; the update that makes it current logs the day.
    match latest_out_of_date(&today) {
        None => forward_log(&store),
        Some(why) => {
            log(&format!("forward log: skipped, {why}; today's picks are logged after the full census finishes"));
            Ok(())
        }
    }
}

/// What the manifest of a census made on the price store as it is now records about the store.
struct Today {
    rows: u64,
    symbols_fnv64: String,
    last_day: String,
}

impl Today {
    fn of(store: &Store) -> Result<Today> {
        let symbols = std::fs::read(super::store_dir().join("symbols.tsv")).context("read the store's symbols.tsv")?;
        let last = *Calendar::trading_days(store)?.days.last().context("the price store has no trading days")?;
        Ok(Today { rows: store.rows() as u64, symbols_fnv64: format!("{:016x}", meta::fnv64(&symbols)), last_day: c2_data::date_string(last) })
    }

    /// Why the run whose manifest is `m` is not a census with settings `cfg` of today's store
    /// (None if it is one). The store fields are the ones `run::run` writes and checks before
    /// resuming: rows, the fingerprint of its symbol table (which also changes with any
    /// symbol's first or last day or row count) and the census variable's last day.
    fn differs(&self, m: &serde_json::Value, cfg: &Config) -> Option<String> {
        let last = m["variable_days"][1].as_str();
        if m["config"].as_str() != Some(cfg.to_text().as_str()) {
            Some(format!("it was made with other census settings ({})", m["config"].as_str().unwrap_or("unrecorded")))
        } else if m["store_rows"].as_u64() != Some(self.rows) || m["symbols_tsv_fnv64"].as_str() != Some(self.symbols_fnv64.as_str()) || last != Some(self.last_day.as_str()) {
            Some(format!("it was made on other price data ({} rows to {} then, {} rows to {} now)", m["store_rows"], last.unwrap_or("?"), self.rows, self.last_day))
        } else {
            None
        }
    }
}

/// The `state` in a run's status.json ("finished", "stopped", "running"), or "" without one.
fn run_state(dir: &Path) -> String {
    let status: serde_json::Value = std::fs::read_to_string(dir.join("status.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    status["state"].as_str().unwrap_or("").to_string()
}

/// The default census named in census/LATEST, if there is one (`run::latest_run` falls back to
/// "main").
fn latest_name() -> Option<String> {
    std::fs::read_to_string(c2_data::data_dir().join("census/LATEST")).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn default_stays() -> String {
    match latest_name() {
        Some(name) => format!("the default census stays {name}"),
        None => "there is still no default census".into(),
    }
}

/// Why the default census (census/LATEST) is not a finished full census of today's store, or
/// None if it is one (then there is nothing to compute).
fn latest_out_of_date(today: &Today) -> Option<String> {
    let Some(name) = latest_name() else { return Some("there is no default census yet (census/LATEST)".into()) };
    let meta = match RunMeta::load(&run::run_dir(&name)) {
        Ok(meta) => meta,
        Err(e) => return Some(format!("the default census {name} can't be read ({e:#})")),
    };
    if let Some(why) = today.differs(&meta.manifest, &Config::full()) {
        return Some(format!("the default census {name} is out of date: {why}"));
    }
    match run_state(&meta.dir).as_str() {
        "finished" => None,
        "" => Some(format!("the default census {name} is not finished (it has no status)")),
        state => Some(format!("the default census {name} is not finished ({state})")),
    }
}

/// Where today's census goes: `<prefix><last data day>`, or that name with `-2`, `-3`, ... after
/// it while a run of the name holds a census that `plan` can't resume or find complete
/// (`run::resume_check`: other settings, price data, task list, correlation periods or their
/// stocks; e.g. a same-day run made by older code), which is left as it is. A run that passes is
/// used again: `run::run` resumes it if it was interrupted and reports it complete if it is.
fn census_name(prefix: &str, today: &Today, plan: &run::Plan, store: &Store) -> String {
    // A reason already in update.log (from an earlier update) is referred to, not repeated.
    let logged = std::fs::read_to_string(update_log()).unwrap_or_default();
    (1..)
        .map(|k| if k == 1 { format!("{prefix}{}", today.last_day) } else { format!("{prefix}{}-{k}", today.last_day) })
        .find(|name| match run::resume_check(&run::run_dir(name), plan, store) {
            Ok(()) => true,
            Err(e) => {
                let full = format!("census {name} can't be continued with today's plan, so it is left as it is: {e:#}");
                if logged.contains(&full) {
                    log(&format!("census {name} is left as it is, as before (it can't be continued with today's plan; the reason is earlier in update.log)"));
                } else {
                    log(&full);
                }
                false
            }
        })
        .expect("a free run name")
}

/// Runs today's census because the default census is out of date (`why`): the full grid, which
/// then becomes the default, or with `smoke` the reduced grid, which never does. Its plan and
/// progress go to the run's own run.log as with `c2 census run` (so the Status tab's history
/// shows its computing time), and to update.log. Ctrl-C or `c2 census stop` stops it cleanly
/// after the correlation period in progress.
fn census(store: &Store, today: &Today, smoke: bool, why: &str) -> Result<()> {
    let cfg = if smoke { Config::smoke() } else { Config::full() };
    // The plan comes first: it doesn't depend on the run's name, and choosing the name needs it.
    let t0 = std::time::Instant::now();
    let plan = run::Plan::new(store, cfg)?;
    let built = t0.elapsed().as_secs_f64();
    let name = census_name(if smoke { "smoke-" } else { "main-" }, today, &plan, store);
    let dir = run::run_dir(&name);
    let leave = |pid: u32| {
        log(&format!("census {name} is already running (pid {pid}); this update leaves it to finish{}", if smoke { "" } else { ", and the next `c2 update` makes it the default" }));
    };
    if let Some(pid) = run::running_pid(&dir) {
        leave(pid);
        return Ok(());
    }
    // The writer lock decides between this update and a start made at the same moment, before
    // anything is logged for the run.
    let lock = match run::lock_run(&dir) {
        Ok(lock) => lock,
        Err(e) => match run::running_pid(&dir) {
            Some(pid) => {
                leave(pid);
                return Ok(());
            }
            None => return Err(e),
        },
    };
    run::resume_check(&dir, &plan, store)?;
    // `analyze::run_history` recognizes update-started runs by this line ("running census NAME ").
    log(&format!("running census {name} because {why}"));
    let census_log = super::census_log(&dir, Some(update_log()));
    super::log_plan(&plan, built, &name, &dir, &census_log)?;
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    ctrlc::set_handler(move || {
        eprintln!("stopping after the correlation period in progress...");
        flag.store(true, Ordering::Relaxed);
    })?;
    run::run_locked(store, &plan, &dir, lock, &stop, &census_log)?;
    // `run::run` also returns normally when stopped; only a finished run may become the default.
    if run_state(&dir) != "finished" {
        log(&format!(
            "census {name} stopped before it finished, so {}; the next `c2 update{}` or `c2 census run --name {name}` resumes it",
            default_stays(), if smoke { " --smoke" } else { "" }
        ));
        return Ok(());
    }
    if smoke {
        log(&format!("smoke test: census {name} is not made the default; {}", default_stays()));
    } else {
        run::set_latest(&name)?;
        log(&format!("census {name} is now the default census"));
    }
    Ok(())
}

/// Appends today's picks for each family (once per data day and family) from the latest finished
/// census, which `update` makes sure is the full census of today's data. Rows are never rewritten.
fn forward_log(store: &Store) -> Result<()> {
    let name = run::latest_run();
    let dir = run::run_dir(&name);
    let status: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("status.json")).unwrap_or_default()).unwrap_or_default();
    if status["state"].as_str() != Some("finished") {
        log(&format!("forward log: census {name} is not finished; nothing logged"));
        return Ok(());
    }
    // The run's own settings, correlation periods, tasks and stock numbering, on today's data:
    // `current_picks` translates the run's stock ids to today's through the plan (and refuses a
    // plan that doesn't fit the run), so a store rebuilt since the run is fine.
    let meta = RunMeta::load(&dir)?;
    let plan = run::Plan::for_run(store, &meta).with_context(|| format!("forward log from census {name}"))?;
    let day = c2_data::date_string(*plan.var.days.last().unwrap());
    let path = c2_data::data_dir().join("forward_log.jsonl");
    // The check for an existing row and the append happen under an exclusive lock on the log, so
    // updates running at the same time never write a day and family twice.
    let mut f = std::fs::OpenOptions::new().create(true).read(true).append(true).open(&path)?;
    f.lock().with_context(|| format!("lock {}", path.display()))?;
    let mut existing = String::new();
    f.read_to_string(&mut existing)?;
    for (family, label, rebalance) in [
        (Family::CensusPairs, "census", 1usize),
        (Family::PeerMomentum { peers: 20, lookback: 21, window: 756 }, "peers", 21),
    ] {
        let key = format!("\"data_last_day\":\"{day}\",\"family\":\"{label}\"");
        if existing.contains(&key) {
            log(&format!("forward log: {label} picks for {day} already logged"));
            continue;
        }
        let picks = c2_census::holdout::current_picks(store, &plan, &dir, family, rebalance, 20)?;
        let row = serde_json::json!({
            "data_last_day": day,
            "family": label,
            "rebalance_days": rebalance,
            "census_run": name,
            "written_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),
            "picks": picks.iter().map(|(t, s)| serde_json::json!({"ticker": t, "score": s})).collect::<Vec<_>>(),
        });
        // Keep the key order stable so the duplicate check above works on the raw text.
        let line = format!(
            "{{\"data_last_day\":\"{day}\",\"family\":\"{label}\",\"rebalance_days\":{rebalance},\"census_run\":\"{name}\",\"written_unix\":{},\"picks\":{}}}",
            row["written_unix"], row["picks"]
        );
        writeln!(f, "{line}")?;
        f.sync_data()?;
        log(&format!("forward log: {} {label} picks for the period after {day}: {}", picks.len(),
            picks.iter().take(5).map(|(t, _)| t.as_str()).collect::<Vec<_>>().join(", ")));
    }
    Ok(())
}

/// Scores logged picks whose holding period has passed: equal-weight return vs SPY.
pub fn forward_report() -> Result<()> {
    let store = Store::open(&super::store_dir())?;
    let cal = Calendar::trading_days(&store)?;
    let text = std::fs::read_to_string(c2_data::data_dir().join("forward_log.jsonl")).context("no forward log yet (run `c2 update`)")?;
    let price = |ticker: &str, day: i32| -> Option<f64> {
        let id = store.id(ticker)?;
        let k = store.dates(id).binary_search(&day).ok()?;
        Some(store.close(id)[k] as f64)
    };
    println!("{:<12} {:<8} {:>6} {:>10} {:>10} {:>10}", "decided", "family", "picks", "picks", "SPY", "status");
    for line in text.lines() {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let day = c2_data::parse_date(v["data_last_day"].as_str().unwrap_or(""))?;
        let h = v["rebalance_days"].as_u64().unwrap_or(1) as usize;
        let k = cal.days.partition_point(|&d| d < day);
        let tickers: Vec<String> = v["picks"].as_array().map(|a| a.iter().filter_map(|p| p["ticker"].as_str().map(String::from)).collect()).unwrap_or_default();
        let (family, n) = (v["family"].as_str().unwrap_or("?"), tickers.len());
        match cal.days.get(k + h) {
            Some(&end) if cal.days.get(k) == Some(&day) => {
                let rets: Vec<f64> = tickers.iter().filter_map(|t| Some(price(t, end)? / price(t, day)? - 1.0)).collect();
                let pick = rets.iter().sum::<f64>() / rets.len().max(1) as f64;
                let spy = price("SPY", end).zip(price("SPY", day)).map(|(a, b)| a / b - 1.0).unwrap_or(f64::NAN);
                println!("{:<12} {family:<8} {n:>6} {:>9.2}% {:>9.2}% {:>10}", c2_data::date_string(day), 100.0 * pick, 100.0 * spy, "scored");
            }
            _ => println!("{:<12} {family:<8} {n:>6} {:>10} {:>10} {:>10}", c2_data::date_string(day), "", "", "pending"),
        }
    }
    Ok(())
}

/// systemd user units that run `c2 update` on weekday evenings (and catch up after downtime).
pub fn install_timer(enable: bool) -> Result<()> {
    let exe = std::env::current_exe()?;
    let service = format!(
        "[Unit]\nDescription=correlation2 data update and forward log\n\n[Service]\nType=oneshot\nExecStart={} update\nNice=10\n",
        exe.display()
    );
    let timer = "[Unit]\nDescription=Run correlation2 update on weekday evenings\n\n[Timer]\nOnCalendar=Mon..Fri 18:30 America/New_York\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n";
    if !enable {
        println!("# ~/.config/systemd/user/c2-update.service\n{service}\n# ~/.config/systemd/user/c2-update.timer\n{timer}");
        println!("# Install and start with: c2 install-timer --enable");
        return Ok(());
    }
    let dir = PathBuf::from(std::env::var_os("HOME").context("HOME")?).join(".config/systemd/user");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("c2-update.service"), service)?;
    std::fs::write(dir.join("c2-update.timer"), timer)?;
    for args in [vec!["--user", "daemon-reload"], vec!["--user", "enable", "--now", "c2-update.timer"]] {
        let ok = std::process::Command::new("systemctl").args(&args).status()?.success();
        anyhow::ensure!(ok, "systemctl {} failed", args.join(" "));
    }
    println!("installed and started c2-update.timer (weekdays 18:30 New York time); `systemctl --user list-timers c2-update.timer` shows the next run");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_census_is_current_only_for_the_same_settings_and_store() {
        let today = Today { rows: 100, symbols_fnv64: "00000000000000aa".into(), last_day: "2025-06-27".into() };
        let manifest = |config: String, rows: u64, fnv: &str, last: &str| {
            serde_json::json!({ "config": config, "store_rows": rows, "symbols_tsv_fnv64": fnv, "variable_days": ["2001-12-31", last] })
        };
        let full = Config::full();
        let same = manifest(full.to_text(), 100, "00000000000000aa", "2025-06-27");
        assert_eq!(today.differs(&same, &full), None);
        // A smoke census (as `update --smoke` used to make the default) is no full census of the data.
        assert!(today.differs(&manifest(Config::smoke().to_text(), 100, "00000000000000aa", "2025-06-27"), &full).is_some());
        assert_eq!(today.differs(&manifest(Config::smoke().to_text(), 100, "00000000000000aa", "2025-06-27"), &Config::smoke()), None);
        // Any change to the store: rows, its symbol table, or the last day of data.
        assert!(today.differs(&manifest(full.to_text(), 99, "00000000000000aa", "2025-06-27"), &full).is_some());
        assert!(today.differs(&manifest(full.to_text(), 100, "00000000000000ab", "2025-06-27"), &full).is_some());
        assert!(today.differs(&manifest(full.to_text(), 100, "00000000000000aa", "2025-06-26"), &full).is_some());
        // A manifest that records none of it never counts as current.
        assert!(today.differs(&serde_json::json!({}), &full).is_some());
    }
}
