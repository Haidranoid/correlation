//! Scheduled data refresh and the forward prediction log (SPEC.md section 5).
//!
//! `c2 update` ingests any Stooq zip dropped into `<data dir>/inbox/` (Stooq downloads now need
//! a CAPTCHA, so fetching is manual for the time being), archives it under `raw/stooq/<date>/`,
//! reruns the census under a dated run name if the data changed (a full census takes ~6 minutes,
//! so "refine as new data arrives" currently means rerun; per-pair running sums are stored for a
//! future live tracker), and appends today's picks for each family to an append-only forward log,
//! written before the outcomes exist. `c2 forward-report` scores past entries once their holding
//! period has passed.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use c2_census::holdout::Family;
use c2_census::meta::{self, RunMeta};
use c2_census::run;
use c2_data::{Calendar, Store};

pub struct UpdateOptions {
    pub zips: Vec<PathBuf>,
    pub smoke: bool,
    pub no_census: bool,
}

fn log(msg: &str) {
    let line = format!("[{}] {msg}", super::hms_now());
    println!("{line}");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(c2_data::data_dir().join("update.log")) {
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
    let before = fingerprint(&store_dir);
    let mut changed = false;
    if zips.is_empty() {
        log(&format!("no new zips in {}", inbox.display()));
    } else {
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
        changed = after != before;
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
    if changed && !opt.no_census {
        let last = fingerprint(&store_dir).map(|(_, d)| c2_data::date_string(d)).unwrap_or_default();
        let name = format!("{}{last}", if opt.smoke { "smoke-" } else { "main-" });
        let cfg = if opt.smoke { c2_census::grid::Config::smoke() } else { c2_census::grid::Config::full() };
        log(&format!("running census {name} on the new data"));
        // A new run: today's plan is its plan.
        let plan = run::Plan::new(&store, cfg)?;
        let dir = run::run_dir(&name);
        std::fs::create_dir_all(&dir)?;
        let stop = std::sync::atomic::AtomicBool::new(false);
        run::run(&store, &plan, &dir, &stop, &|m: &str| log(m))?;
        if opt.smoke {
            log(&format!("smoke test: census {name} is not made the default; the default stays {}", run::latest_run()));
        } else {
            run::set_latest(&name)?;
        }
    }
    forward_log(&store)
}

/// Appends today's picks for each family (once per data day and family) from the latest finished
/// census. Rows are never rewritten.
fn forward_log(store: &Store) -> Result<()> {
    let name = run::latest_run();
    let dir = run::run_dir(&name);
    let status: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("status.json")).unwrap_or_default()).unwrap_or_default();
    if status["state"].as_str() != Some("finished") {
        log(&format!("forward log: census {name} is not finished; nothing logged"));
        return Ok(());
    }
    // The run's own settings, correlation periods and tasks, on today's data.
    let meta = RunMeta::load(&dir)?;
    let plan = run::Plan::for_run(store, &meta).with_context(|| format!("forward log from census {name}"))?;
    // holdout::current_picks still reads record ids as today's store ids; until it translates
    // them through the run's stock list, log nothing when the two numberings differ.
    if !meta.id_map(store)?.is_identity() {
        log(&format!("forward log: census {name} numbers its stocks differently from today's store (it was made before the store was rebuilt); nothing logged"));
        return Ok(());
    }
    let day = c2_data::date_string(*plan.var.days.last().unwrap());
    let path = c2_data::data_dir().join("forward_log.jsonl");
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
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
