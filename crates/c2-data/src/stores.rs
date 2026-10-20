//! Where the price stores live. Each ingest builds a store of its own in
//! `<data dir>/stores/<name>/` (`stooq-<last trading day>` unless it is given a name) and keeps
//! the stores already there, so census results and the outside data, which stay in the data
//! directory, can be read against any of them. `stores/CURRENT` names the default store;
//! `--store` or `C2_STORE` picks another for one command (a name in `stores/`, or a path). A data
//! directory from before stores were split out has no `stores/CURRENT` and keeps its single
//! `store/`, which is then the one used.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, bail, ensure};

use crate::ingest::IngestReport;
use crate::{Calendar, Store};

/// The folder that holds the price stores, one folder each.
pub fn stores_dir() -> PathBuf {
    crate::data_dir().join("stores")
}

static CHOSEN: OnceLock<String> = OnceLock::new();

/// Makes this process use `choice` (a store name in `stores/`, or a path) in place of the default
/// store: the CLI's `--store`. Call it before anything opens a store.
pub fn choose(choice: &str) -> Result<()> {
    ensure!(!choice.trim().is_empty(), "--store needs a store name or a path");
    let dir = find(choice)?;
    CHOSEN.set(dir.to_str().context("the store's path is not UTF-8")?.to_string()).map_err(|_| anyhow::anyhow!("the store was already chosen"))
}

/// What picked the store in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// `--store`.
    Flag,
    /// The `C2_STORE` environment variable.
    Env,
    /// `stores/CURRENT`: the default store.
    Default,
    /// No `stores/CURRENT`: the data directory's own `store/`, as before stores were split out.
    Legacy,
}

impl Choice {
    /// How messages say what picked the store.
    pub fn describe(self) -> &'static str {
        match self {
            Choice::Flag => "chosen with --store",
            Choice::Env => "chosen by C2_STORE",
            Choice::Default => "the default store",
            Choice::Legacy => "the data directory's only store",
        }
    }
}

/// The price store in use: the one `--store` chose, else `C2_STORE`, else the default
/// (`stores/CURRENT`), else the data directory's own `store/`.
pub fn store_dir() -> PathBuf {
    selected().0
}

/// `store_dir`, and what picked it.
pub fn selected() -> (PathBuf, Choice) {
    let env = std::env::var("C2_STORE").ok();
    resolve(&crate::data_dir(), CHOSEN.get().map(String::as_str), env.as_deref())
}

fn resolve(data: &Path, flag: Option<&str>, env: Option<&str>) -> (PathBuf, Choice) {
    let stores = data.join("stores");
    let at = |s: &str| if s.contains('/') { PathBuf::from(s) } else { stores.join(s) };
    let given = |s: Option<&str>| s.map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
    if let Some(s) = given(flag) {
        return (at(&s), Choice::Flag);
    }
    if let Some(s) = given(env) {
        return (at(&s), Choice::Env);
    }
    match given(std::fs::read_to_string(stores.join("CURRENT")).ok().as_deref()) {
        Some(s) => (at(&s), Choice::Default),
        None => (data.join("store"), Choice::Legacy),
    }
}

/// A store's name in messages: its folder's name.
pub fn name(dir: &Path) -> String {
    dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| dir.display().to_string())
}

/// The default store as `stores/CURRENT` names it, if it names one.
pub fn default_name() -> Option<String> {
    std::fs::read_to_string(stores_dir().join("CURRENT")).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// The folder of the store `choice` names (a name in `stores/`, or a path), once it opens as a
/// price store.
pub fn find(choice: &str) -> Result<PathBuf> {
    find_in(&stores_dir(), choice)
}

fn find_in(stores: &Path, choice: &str) -> Result<PathBuf> {
    let choice = choice.trim();
    let dir = if choice.contains('/') {
        std::fs::canonicalize(choice).with_context(|| format!("there is no {choice}"))?
    } else {
        check_name(choice)?;
        let dir = stores.join(choice);
        if !dir.join("prices.bin").is_file() {
            let names: Vec<String> = list_in(stores).into_iter().map(|s| s.name).collect();
            bail!("there is no store {choice} in {} (stores there: {})", stores.display(), if names.is_empty() { "none".to_string() } else { names.join(", ") });
        }
        dir
    };
    Store::open(&dir).with_context(|| format!("{} is not a price store", dir.display()))?;
    Ok(dir)
}

/// Makes `choice` (a store name in `stores/`, or a path to a store) the default store, once it
/// opens as one. Returns its folder.
pub fn set_default(choice: &str) -> Result<PathBuf> {
    set_default_in(&stores_dir(), choice)
}

fn set_default_in(stores: &Path, choice: &str) -> Result<PathBuf> {
    let dir = find_in(stores, choice)?;
    // A store in stores/ is recorded by name, any other by its full path.
    let text = if choice.contains('/') { dir.to_str().context("the store's path is not UTF-8")?.to_string() } else { choice.trim().to_string() };
    std::fs::create_dir_all(stores)?;
    write_durable(&stores.join("CURRENT"), text.as_bytes())?;
    Ok(dir)
}

/// A store in `stores/`.
pub struct Listed {
    pub name: String,
    pub dir: PathBuf,
}

/// The stores in `stores/` by name: every folder there with a `prices.bin`, except hidden ones (a
/// build in progress).
pub fn list() -> Vec<Listed> {
    list_in(&stores_dir())
}

fn list_in(stores: &Path) -> Vec<Listed> {
    let mut out: Vec<Listed> = std::fs::read_dir(stores)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            (!name.starts_with('.') && e.path().join("prices.bin").is_file()).then(|| Listed { name, dir: e.path() })
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// A store's last trading day (`Calendar::trading_days`, which census run names also go by), or
/// for a store without the calendar's reference stocks, the last day any symbol has a price.
pub fn last_day(store: &Store) -> i32 {
    Calendar::trading_days(store)
        .ok()
        .and_then(|c| c.days.last().copied())
        .unwrap_or_else(|| store.symbols.iter().map(|s| s.last).max().unwrap_or(0))
}

/// What `build_stooq` made.
pub struct Built {
    pub report: IngestReport,
    pub name: String,
    pub dir: PathBuf,
    pub last_day: i32,
    /// A store of the same name was there already, and its files were replaced.
    pub replaced: bool,
}

/// Builds a price store from Stooq zips (`ingest::ingest_stooq_until`) in a hidden folder of
/// `stores/`, names it `stooq-<last trading day>` unless given `name`, and makes it the default.
/// The stores already there stay as they are, except one of the same name: its two files are
/// replaced, each by a rename (as the single store's always were), so a reader that has the old
/// one open keeps a whole copy.
pub fn build_stooq(zips: &[&Path], markets: &[&str], until: Option<i32>, name: Option<&str>) -> Result<Built> {
    build_stooq_in(&stores_dir(), zips, markets, until, name)
}

fn build_stooq_in(stores: &Path, zips: &[&Path], markets: &[&str], until: Option<i32>, name: Option<&str>) -> Result<Built> {
    if let Some(name) = name {
        check_name(name)?;
    }
    std::fs::create_dir_all(stores)?;
    remove_stale_builds(stores);
    let tmp = stores.join(format!(".building-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let built = place(stores, &tmp, zips, markets, until, name);
    // On success the folder was renamed into place or emptied; on failure this drops the half-built store.
    let _ = std::fs::remove_dir_all(&tmp);
    built
}

fn place(stores: &Path, tmp: &Path, zips: &[&Path], markets: &[&str], until: Option<i32>, name: Option<&str>) -> Result<Built> {
    let report = crate::ingest::ingest_stooq_until(zips, markets, tmp, until)?;
    let last_day = last_day(&Store::open(tmp)?);
    let name = name.map(str::to_string).unwrap_or_else(|| format!("stooq-{}", crate::date_string(last_day)));
    let dir = stores.join(&name);
    let replaced = dir.exists();
    if replaced {
        for f in ["prices.bin", "symbols.tsv"] {
            std::fs::rename(tmp.join(f), dir.join(f)).with_context(|| format!("replace {}", dir.join(f).display()))?;
        }
    } else {
        std::fs::rename(tmp, &dir).with_context(|| format!("rename {} to {}", tmp.display(), dir.display()))?;
    }
    set_default_in(stores, &name)?;
    Ok(Built { report, name, dir, last_day, replaced })
}

fn check_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty() && name.trim() == name && !name.contains('/') && !name.starts_with('.') && name != "CURRENT",
        "{name:?} can't name a store: it must be a folder name in stores/ that doesn't start with a dot"
    );
    Ok(())
}

/// Removes the hidden folders of builds whose process is gone (an ingest that was killed).
fn remove_stale_builds(stores: &Path) {
    for e in std::fs::read_dir(stores).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(pid) = name.strip_prefix(".building-").and_then(|p| p.parse::<u32>().ok()) else { continue };
        if !Path::new("/proc").join(pid.to_string()).exists() {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// Writes `path` through a flushed temporary file renamed over it, then flushes the directory, so
/// the change survives a power loss once this returns.
fn write_durable(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().context("a file in no directory")?;
    let tmp = dir.join(format!(".{}.{}.tmp", path.file_name().and_then(|n| n.to_str()).unwrap_or("file"), std::process::id()));
    {
        let mut f = File::create(&tmp).with_context(|| format!("write {}", tmp.display()))?;
        f.write_all(bytes).with_context(|| format!("write {}", tmp.display()))?;
        f.sync_all().with_context(|| format!("flush {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("rename {} to {}", tmp.display(), path.display()))?;
    File::open(dir).and_then(|d| d.sync_all()).with_context(|| format!("flush the directory {}", dir.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("c2-stores-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A Stooq-style bulk zip: for each (ticker, days), daily rows from 2024-01-01 on.
    fn stooq_zip(path: &Path, stocks: &[(&str, u32)]) {
        let mut z = zip::ZipWriter::new(File::create(path).unwrap());
        let opt = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for (ticker, days) in stocks {
            z.start_file(format!("data/daily/us/nasdaq stocks/1/{ticker}.us.txt"), opt).unwrap();
            writeln!(z, "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>").unwrap();
            for d in 1..=*days {
                writeln!(z, "{}.US,D,202401{d:02},000000,10,11,9,10.5,1000,0", ticker.to_uppercase()).unwrap();
            }
        }
        z.finish().unwrap();
    }

    #[test]
    fn the_store_in_use_is_the_flag_then_the_environment_then_current_then_store() {
        let data = scratch("resolve");
        assert_eq!(resolve(&data, None, None), (data.join("store"), Choice::Legacy));
        std::fs::create_dir_all(data.join("stores")).unwrap();
        std::fs::write(data.join("stores/CURRENT"), "stooq-2025-06-27\n").unwrap();
        assert_eq!(resolve(&data, None, None), (data.join("stores/stooq-2025-06-27"), Choice::Default));
        assert_eq!(resolve(&data, None, Some("stooq-2026-10-19")), (data.join("stores/stooq-2026-10-19"), Choice::Env));
        assert_eq!(resolve(&data, Some("/elsewhere/s"), Some("stooq-2026-10-19")), (PathBuf::from("/elsewhere/s"), Choice::Flag));
        // Blank choices count as none.
        assert_eq!(resolve(&data, Some(" "), Some("")), (data.join("stores/stooq-2025-06-27"), Choice::Default));
        std::fs::write(data.join("stores/CURRENT"), "/some/where\n").unwrap();
        assert_eq!(resolve(&data, None, None), (PathBuf::from("/some/where"), Choice::Default));
        std::fs::write(data.join("stores/CURRENT"), " \n").unwrap();
        assert_eq!(resolve(&data, None, None), (data.join("store"), Choice::Legacy));
        std::fs::remove_dir_all(&data).unwrap();
    }

    #[test]
    fn each_build_is_a_store_of_its_own_named_by_its_last_day() {
        let data = scratch("build");
        let stores = data.join("stores");
        let zip = data.join("d_us_txt.zip");
        stooq_zip(&zip, &[("aaa", 5), ("bbb", 3)]);
        let current = || std::fs::read_to_string(stores.join("CURRENT")).unwrap();
        let names = || list_in(&stores).into_iter().map(|s| s.name).collect::<Vec<_>>();

        // No reference stocks for the calendar, so the name goes by the last price: 2024-01-05.
        let a = build_stooq_in(&stores, &[&zip], &["us"], None, None).unwrap();
        assert_eq!((a.name.as_str(), a.replaced, a.report.rows), ("stooq-2024-01-05", false, 8));
        assert_eq!(current(), "stooq-2024-01-05");

        // A build that ends earlier is a second store, made the default; the first stays.
        std::fs::create_dir_all(stores.join(".building-4000000000")).unwrap();
        let b = build_stooq_in(&stores, &[&zip], &["us"], Some(crate::days_from_civil(2024, 1, 2)), None).unwrap();
        assert_eq!((b.name.as_str(), b.replaced, b.report.rows), ("stooq-2024-01-02", false, 4));
        assert_eq!(current(), "stooq-2024-01-02");
        assert_eq!(names(), ["stooq-2024-01-02", "stooq-2024-01-05"]);
        assert_eq!(Store::open(&a.dir).unwrap().rows(), 8);

        // The same data again replaces the store of that name; no hidden build folder is left,
        // including the one from a process that is gone.
        let c = build_stooq_in(&stores, &[&zip], &["us"], None, None).unwrap();
        assert!(c.replaced && c.dir == a.dir);
        assert_eq!(current(), "stooq-2024-01-05");
        let hidden: Vec<_> = std::fs::read_dir(&stores).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(".building")).collect();
        assert!(hidden.is_empty(), "{:?}", hidden.iter().map(|e| e.path()).collect::<Vec<_>>());

        // A given name, and names that can't be stores.
        assert_eq!(build_stooq_in(&stores, &[&zip], &["us"], None, Some("mine")).unwrap().dir, stores.join("mine"));
        for bad in ["", ".hidden", "a/b", "CURRENT", " x"] {
            assert!(build_stooq_in(&stores, &[&zip], &["us"], None, Some(bad)).is_err(), "{bad:?}");
        }

        // Only a store can be the default, and a refused one leaves the default as it was.
        let e = set_default_in(&stores, "missing").unwrap_err().to_string();
        assert!(e.contains("there is no store missing") && e.contains("stores there: mine, stooq-2024-01-02, stooq-2024-01-05"), "{e}");
        assert_eq!(current(), "mine");
        assert_eq!(set_default_in(&stores, "stooq-2024-01-02").unwrap(), stores.join("stooq-2024-01-02"));
        assert_eq!(current(), "stooq-2024-01-02");
        let path = stores.join("stooq-2024-01-05");
        set_default_in(&stores, path.to_str().unwrap()).unwrap();
        assert_eq!(resolve(&data, None, None), (path.canonicalize().unwrap(), Choice::Default));
        std::fs::remove_dir_all(&data).unwrap();
    }
}
