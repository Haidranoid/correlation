# External series: correlation2's outside data as model inputs

`stockai-judge export-external` writes correlation2's external series as census-style daily variables on the
export's trading days. The series cover weather, climate indices, politics, news, attention, rates, FX, world equity,
commodities, crypto, macro, migration, prosperity, prediction markets and space weather: 1,753 of them in 9 catalogs
as of 2026-10-19. Each is timed exactly as correlation2's lead-lag test times it. The method and its caveats are in
correlation2's summary, `~/.local/share/correlation2/external/results/ext-all-2/summary.txt`.

```
stockai-judge export-external [--external DIR] [--out DIR] [--export DIR]
```

- `--external`: correlation2's external data, default `~/.local/share/correlation2/external`. Only `catalog_*.csv`
  and `series/<id>.csv` are read (not `raw/`), and nothing there is written.
- `--out`: default `~/.local/share/stockai/external/`. Refused inside `~/.local/share/correlation2`.
- The days are the export's: `plan.var.days` of the store found through `C2_DATA_DIR`, as `stockai-judge export`
  writes them to `days.npy`. If an export exists (`--export`, default `$C2_DATA_DIR/export/full`), its `days.npy` and
  fingerprint must equal them. For a Sharadar store, set `C2_DATA_DIR` and give `--out` (e.g.
  `~/.local/share/stockai/sharadar/external`): its days are different.

## The files

| File | Type, shape | Meaning |
|---|---|---|
| `ext_v.npy` | float32 `[K, T]` | row k = series k (line k of `ext_meta.tsv`), column d = day d of the export: the standardized daily move, NaN where unavailable. C order, like the export's arrays |
| `ext_raw.npy` | float32 `[K, T]` | the move before standardization (log change, change, mean anomaly or mean level); 0 on live days without a new observation; NaN where the series is not live. For diagnostics |
| `ext_meta.tsv` | text | header `row id name category subcategory region catalog transform frequency known_after_days status first last observations` (tab-separated; tabs and line breaks in names become spaces) |
| `meta.json` | JSON | `format` (stockai-external-1), `k`, `t`, `first_day`, `last_day`, the export's `fingerprint`, `correlation2_commit`, every catalog file's size and SHA-256 with its series listed and loaded, a SHA-256 over the series files, counts by status and category, the loader's warnings, the series with a value per day by year (through 2023), and the tool version |

All files are written through temporary files and renamed. `meta.json` is removed first and written last, so a
directory without it is an interrupted export.

`ext_meta.tsv` columns:
- `category`: the catalog's, in lower case. A finite-life prediction-market contract gets `single_contract` (and
  `subcategory` becomes `category/subcategory`), as correlation2 reads it.
- `known_after_days`: the catalog's, rounded up.
- `status`: correlation2's classes, and the only thing in this file that tells annual series apart:
  - `single_contract`: notes flagged FINITE-LIFE;
  - `annual`: one release a year, by frequency or a median spacing of 300 days or more; often a calendar pulse that
    standardization pushes to ±5;
  - `noise`: correlation2's noise_test category. The catalogs have none; correlation2 makes its noise series in
    memory;
  - `headline`: everything else.
  correlation2 keeps the first three out of its headline numbers.
- `first`, `last`: the first and last export day with a value in `ext_v.npy`.
- `observations`: days with a non-zero move. These three are correlation2's `externals.csv` columns of the same names.

## The timing

These are correlation2's rules, from `crates/c2-census/examples/external_test.rs` at commit 310ea71. The code in
`judge/src/external.rs` is a verbatim copy of it.
- **When a value counts:** an observation dated d with `known_after_days` k counts from the first trading day on or
  after d + k (calendar days). Weekly series dated by week start with k = 7 land on the first trading day after the
  week ends. Calendar-day values landing on one trading day are averaged (a Friday's, Saturday's and Sunday's value
  known the next day all land on Monday).
- **logdiff and diff:** the move on trading day t is the change in the latest known level since day t − 1. Release
  days carry the whole move; other live days are 0. A change across a gap longer than the series' normal spacing is
  no move: the normal gap is the 90th percentile of its previous 250 gaps plus 2 days, but exactly 1 day for series
  observed every day, so a prediction market's roll or an outage is never bridged.
- **anomaly and anomaly_log1p:** the value (or ln(1 + value)) minus the mean of the 15-day spans centred on the same
  date in each of the previous 10 years, needing at least 3 years.
- **level:** the value minus its mean over the previous 252 trading days.
- **Standardization,** as the census does for stocks: minus the trailing 252-day mean, divided by the trailing 252-day
  sd (both over rows t − 252 .. t − 1, at least 126 values), clipped at ±5. No value when the trailing year has no
  non-zero move.
- **Live:** a series has values while its latest observation is recent for its frequency: 10 trading days for daily,
  15 weekly, 50 monthly, 140 quarterly, 300 annual. This is fixed from the spacing of its first 100 observations, so
  later data can't change it.

So `ext_v[k, d]` uses only observations that count by day d, which is the same point-in-time rule as the census
variable `v`. The tests in `judge/src/external.rs` port correlation2's self-checks:
- a level change dated Friday 2012-03-02 lands that Friday with k = 0 and on Monday 2012-03-05 with k = 1;
- a weekly value for the week starting Sunday 2012-02-19, with k = 7, lands on 2012-02-27;
- Friday, Saturday and Sunday values known the next day are averaged onto Monday;
- a 7-day series missing a day doesn't count the jump across the gap;
- changing every value dated on or after a cut date leaves every earlier row of `ext_v` and `ext_raw` bit-identical,
  for each transform;
- anomalies remove a seasonal cycle, and sparse 0/1 indicators stay within ±5.

Two more tests: the standardization reads only rows t − 252 .. t − 1 besides row t itself, and the export's files
equal `build_ext` row by row.

Caveats, from correlation2's summary:
- **Timing bounds:** `known_after_days` are the data agents' bounds.
- **Revisions:** population, macro, flu and other series are stored as their latest revisions, and the anomaly base
  periods were chosen after the fact.
- **Calendar pulses:** releases on fixed dates produce alignments that the sign-flip null doesn't model.

## Checked against correlation2 (2026-10-19)

On the Stooq store's 5,911 days (2001-12-31 .. 2025-06-27, equal to `export/full/days.npy`, fingerprint
8b5ac076267c83ab), the export has K = 1,753. For every series, `first`, `last` and `observations` equal those in
correlation2's `results/ext-all-2/externals.csv`, and so do transform, `known_after_days`, frequency, name, region and
catalog. The statuses equal its categories (1,316 headline, 346 annual, 91 single_contract), and the rows are in the
same order, so row k is correlation2's external id 1,000,000 + k in that run. The only difference is that
correlation2 rewrites an annual series' subcategory as `category/subcategory`.

Series with a value in `ext_v`, mean per trading day (all / headline): 2010: 1,373 / 1,028; 2016: 1,497 / 1,153;
2023: 1,523 / 1,174. `ext_v.npy` and `ext_raw.npy` are 41 MB each; the export takes about 1.5 s.
