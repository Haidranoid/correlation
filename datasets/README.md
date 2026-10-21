# Datasets

**TL;DR:** The raw data every experiment here starts from. It holds daily prices for every US stock and fund on Stooq, in two snapshots (2025-06-27 and 2026-10-19), and 1,753 outside series such as weather, news counts and prediction markets. Nothing derived is stored. After cloning, run `datasets/unpack.sh` once to rebuild and check the price stores and link everything where the tools look.

## What's here

| Folder | What | Size in git |
|---|---|---|
| `stocks/stooq-2025-06-27/` | 11,531 stocks and funds, 26.7 million daily rows (open, high, low, close, volume) through 2025-06-27. The walk-forward tests use this snapshot | 329 MB, in 4 parts |
| `stocks/stooq-2026-10-19/` | 13,349 stocks and funds, 28.1 million rows through 2026-10-19. A later download; the 16 unseen months come from it | 344 MB, in 4 parts |
| `external/series/` | The 1,753 outside series, one CSV each | 111 MB |
| `external/catalog_*.csv` | What each series is, where it came from, and under what terms, one catalog per category | under 2 MB |
| `external/SOURCES_*.md`, `qc_*.txt` | Source notes and quality checks per category | under 1 MB |
| `external/derived/` | Series derived from the downloads (migration flows) | 17 MB |
| `external/scripts*/` | The Python that built the series from the original downloads | under 1 MB |

**Price store format.** It is correlation2's memory-mapped format; `crates/c2-data` reads and writes it.
- **`prices.bin`:** the rows.
- **`symbols.tsv`:** ticker, exchange, kind (stock or fund) and row range for each symbol.
- **Parts:** in git, `prices.bin` is stored as `prices.bin.zst.00`–`.03`, a zstd stream split under GitHub's 100 MB file limit. `SHA256SUMS` checks the rebuilt file.

## Setup

```
datasets/unpack.sh
```

- **Needs:** `zstd`.
- **Where it links:** `$C2_DATA_DIR`, by default `~/.local/share/correlation2`.
- **What it does:** rebuilds and verifies each `prices.bin`, links the stores into `stores/` (marking the latest as the default if none is set), and links the outside data into `external/`.
- **Safe to run again;** it never overwrites anything already there.
- **Then:** see "Running it" in the [root README](../README.md#running-it) for the census, the experiments and the GUI.

## Not included

- **The original downloads:**
  - Stooq's zip (it needs a browser and a CAPTCHA);
  - 8.3 GB of raw outside-data files (news archives, event data, government releases).
  The scripts in `external/scripts*/` show how the series were built from them.
- **Everything derived:**
  - census runs: about 17 minutes on a GPU with `c2 census`;
  - exports for the experiments' grader;
  - the experiments' large outputs: each `research/<experiment>/` folder rebuilds its own.

## Terms

- **Prices:** from [Stooq](https://stooq.com). Check Stooq's terms of use before redistributing them.
- **Outside series:** each comes from the source named in its category's catalog and `SOURCES_*.md`, under that source's terms.

**Rebuilding the outside series.** The scripts in `external/scripts*/` send a contact email with their requests, as BLS and Wikipedia ask of scripts. Set yours first: `export C2_CONTACT=you@example.com`.
