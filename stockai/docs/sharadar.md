# Sharadar: import, delistings and as-traded prices

The contract between the Sharadar importer and export (`judge/`), the grader (`judge/src/holdout.rs`) and the Python
side (`stockai/data.py`, `stockai/h30/`, `stockai/gpt/`). Written 2026-10-19, before any Sharadar data was downloaded
or any result on it was seen. Why: test-results-30d.md, "Next steps".

## The product

Sharadar "Prices", full history: $39 for one month or $299 a year, personal use (sharadar.com/subscribe). Tables:
`tickers`, `actions`, `sp500`, `stocks`, `funds`, `metrics`. API: `https://api.sharadar.com/v1.0/data/<table>?api_key=KEY`
with `format=csv`; `years=full` answers with a redirect to a time-limited compressed CSV of the whole table (bulk).
Schemas: `https://api.sharadar.com/v1.0/schema/<table>?format=postgres`. The public key `test-api-key` serves a few
tickers (AAPL, MSFT, IBM, KO, JNJ, PG, MMM, NVDA, AMZN, SPY seen to work; GE, XOM, QQQ and delisted tickers refused).

Columns (from the schema endpoint, 2026-08-18):
- `stocks` and `funds`: ticker, date, open, high, low, close, volume, closeadj, closeunadj, lastupdated. open, high, low,
  close and volume are split-adjusted; closeadj is adjusted for splits, dividends and spinoffs; closeunadj is as traded.
- `tickers`: table, permaticker, ticker, name, exchange, isdelisted (Y/N), category, cusips, siccode, sicsector,
  sicindustry, figi, famaindustry, sector, industry, scalemarketcap, scalerevenue, relatedtickers, currency, location,
  lastupdated, firstadded, firstpricedate, lastpricedate, firstquarter, lastquarter, secfilings, companysite. One row
  per (table, permaticker, ticker); use the rows with `table` = `stocks` (and `funds` for SPY and QQQ).
- `actions`: date, action, ticker, name, value, contraticker, contraname. A company delisted because it was acquired
  names its acquirer in contraticker. The action codes are not documented on the site; the codes below come from
  other projects' use and must be confirmed on the real file.

Where the key lives: `~/.config/sharadar/api_key` (one line), or `SHARADAR_API_KEY`. Scripts never print it.
Raw downloads: `~/.local/share/stockai/sharadar/raw/<download date>/`. Each store gets its own data root, because the
grader finds the store through `C2_DATA_DIR` (`$C2_DATA_DIR/store`) and `scripts/30d.sh` reads `$C2_DATA_DIR/export/full`:
- full: `~/.local/share/stockai/sharadar/store/` and `~/.local/share/stockai/sharadar/export/full/`;
- survivors only: `~/.local/share/stockai/sharadar-survivors/store/` and `.../sharadar-survivors/export/full/`.

## The store

`stockai-judge import-sharadar --raw DIR --out STORE [--survivors-only YYYY-MM-DD]` (judge/src/sharadar.rs) writes
the same `C2PRICE1` store correlation2 reads (`c2_data::store::write`), so `stockai-judge export` and `holdout` work
on it unchanged:
- **Files:** one per table in DIR, found by name: the table's name (or Nasdaq Data Link's SEP for stocks, SFP for
  funds) as a word of the file name, ending in `.csv`, `.csv.gz` or `.zip` (one `.csv` inside). No file or two
  candidates for a table stop the import. Columns are found by their header names, so their order doesn't matter.
- **Stocks:** every `stocks` ticker whose `tickers.category` contains "Common Stock" (domestic, ADR and Canadian, any
  share class) is `Kind::Stock`. Preferred stock, warrants, units, rights and the like are `Kind::Other`, and so is
  a `stocks` ticker without a `tickers` row (listed in the log).
- **Funds:** SPY and QQQ from `funds`, `Kind::Etf` (benchmarks only).
- **Prices:** close = closeadj (Stooq's closes are meant to be split- and dividend-adjusted, but see "What the test
  key showed"), open, high and low scaled by closeadj / close, volume = the split-adjusted volume (0 where missing).
  So dollar volume (close × volume) carries later dividends, as on Stooq. Rows are sorted by date; of two rows with
  one date the later `lastupdated` is kept (then the later line); rows without a positive closeadj are dropped.
  Empty, N/A, NA, NaN, null and none are missing values; any other unreadable number, like a malformed date
  (strict YYYY-MM-DD), stops the import.
- **Tickers:** the `stocks` table's ticker, which is unique in that table. A ticker shared by two permatickers in
  `tickers` is an error to report, not to guess around: the import lists them and stops. Symbols are in ticker
  order; `symbols.tsv`'s source column is `sharadar:<table>:<permaticker>`, its exchange `tickers.exchange` in
  lower case.
- **Calendar:** correlation2 builds trading days from IBM, GE, KO, XOM, JNJ, PG, MMM and AAPL (at least 5 present).

Sidecars written next to `prices.bin`:
- `close_unadj.f32`: little-endian f32, one value per store row, in `prices.bin` row order (closeunadj; NaN where
  missing or not positive).
- `delist.tsv`: header `ticker	last_date	kind	actions`; one line per common stock (`Kind::Stock`) with `isdelisted`
  = Y, in ticker order. `last_date` is the date of its last row (after the cleaning above); `kind` is `buyout`,
  `failure` or `unknown`; `actions` lists the distinct codes other than `dividend` of its actions in the window
  below, sorted, comma-separated.
- `source.json`: files read with sizes and SHA-256, row counts, the latest `lastupdated`, the importer's version
  (`stockai-judge import-sharadar 1`), the store's row count and a fingerprint of its `symbols.tsv`, counts of
  everything dropped or missing, and the action codes with counts. It is written last: a store with some sidecars but
  no `source.json` is an interrupted import, and export and holdout refuse it, as they refuse sidecars whose row
  count or fingerprint don't match the store. The importer refuses an `--out` holding a store it didn't write.

Delisting kinds, from the stock's own `actions` rows dated within 30 calendar days of its last price date:
- `buyout`: acquisitionby, mergerto, mergerfrom (bought or merged away: holders got the deal's price or shares).
- `failure`: bankruptcyliquidation, regulatorydelisting. A failure code wins over a buyout code in the same window
  (the log counts such stocks).
- `unknown`: anything else, including voluntarydelisting and a bare `delisted`. Unknown action codes are counted and
  listed in the import log; a missing column or a malformed date stops the import.

A survivors-only store (the Stooq-like control) keeps only the `stocks` symbols whose last row is dated on or after
2025-06-27, Stooq's last day (funds are kept): `import-sharadar --survivors-only 2025-06-27`. Its `delist.tsv` still
lists the stocks delisted after that day.

`stockai-judge check-sharadar --raw DIR` (`scripts/sharadar_check.sh`, which first checks the files' SHA-256) prints
the day-one checks: SIVB, BBBY, FRC and TWTR (price rows, `tickers` rows, actions near the last price, the delisting
kind; TWTR should be a buyout, the others a failure or unknown), delisted common stocks by year and kind, every action
code with its count, every `tickers` category with the kind it gets, and daily returns from closeadj against the
Stooq store through 2024-06-27: AAPL, IBM and KO in detail (a check: at least 95% of days within 0.1 point), then
every common stock in both stores, matched by ticker (left out: tickers with several Stooq stocks, fewer than 20 common
days, or returns that mostly disagree, i.e. another company). For those it prints, by year 2005–2024, how many stocks
have 3 or more days whose returns differ by more than 0.3 point; how many have Stooq closes equal to the as-traded
closes from 2023 on (within 0.05% on 95% of the days), with the dividend payers among them (Sharadar's adjustment
moved by more than 1%), by Stooq exchange; and each stock's biggest single-day gap, top 15 (spin-offs, splits, bad
prints). It exits with status 1 when a check fails; the Stooq diagnostic is not a check.

What the test key showed (2026-10-19; `scripts/sharadar_download.sh --sample`): the CSVs have the columns above in
that order, dates YYYY-MM-DD, `N/A` for an empty contraticker or contraname, no quoted fields and no final newline.
`tickers` has rows for the tables stocks, fundamentals, insiders and funds. The key serves prices from about five
years back only (2021-10-20 on; 2001–2007 came back empty), actions only for AAPL (`403 Exceeds free tier` for the
others), and no bulk file (`401 A valid API key is required for bulk status`). Only `dividend` was seen, so the other
action codes are still unconfirmed. Against Stooq (2021-10-20 .. 2024-06-27), 99–100% of AAPL's, IBM's and KO's daily
returns agree within 0.1 point. The rest are IBM's Kyndryl spin-off (2021-11-04: Sharadar −0.53%, Stooq −4.94%) and
IBM's and KO's ex-dividend days. On the nine sample stocks, Stooq's closes equal the as-traded closes from 2023 on for
IBM, KO, JNJ, PG and MMM (NYSE: 5 of 5 dividend payers), so none of their dividends since about Nov 2022 is in Stooq's
returns, while AAPL and MSFT (Nasdaq: 0 of 2) are adjusted. In 2023, 5 of the 9 stocks have 3 or more days that
differ by more than 0.3 point; in 2021 and 2022, none. Stooq doesn't adjust spin-offs either: MMM on 2024-04-01
(Solventum) is +5.47% in Sharadar and −11.36% in Stooq.

## The export

`stockai-judge export` adds, when the sidecars are present, arrays aligned to the export's `[N, T]` layout:
- `close_unadj.npy`: float32 `[N, T]`, as-traded closes on the days the stock has a row, NaN elsewhere (no bad-print
  repair).
- `delist_day.npy`: int32 `[N]`, for a stock in `delist.tsv` the column of its last finite close in `close.npy` on or
  before its `last_date` (the `last_date` column itself unless that row is off the calendar or the bad-print repair
  blanked it; the export prints how many differ), else −1. A delisted stock without any close is left as listed.
- `delist_kind.npy`: int8 `[N]` (numpy `|i1`): 0 listed, 1 buyout, 2 failure, 3 unknown.
`meta.json` lists them under `extras` (file name: description), copies `source.json` under `source`, counts the
delisted columns by kind under `delisted`, and its `variable` says the closes are Sharadar's closeadj. With no
sidecars (Stooq) the export writes exactly what it did before: byte-identical arrays and `meta.json` but for
`created_unix`, checked against `~/.local/share/correlation2/export/full` on 2026-10-19. An export into a directory
that held a Sharadar export removes the old extra arrays. `stockai.data.load` returns them as optional attributes
(None when absent), so Stooq exports load as before.

## The delisting rule

A position decided at the close of day d and held for h days, in a stock whose last close L is before d + h:
- **main** (pre-registered): a buyout counts at its last price, close[L] / close[d] − 1; a failure or an unknown
  delisting counts as a total loss, −1. How many unknowns there are is reported.
- **last-price** (optimistic bound): every delisting at close[L] / close[d] − 1.
- **total-loss** (harsh bound): every delisting at −1.

A stock that is not delisted keeps today's rule: the end close is close[d + h], or the last close within
min(5, h − 1) days before it; otherwise no outcome. Python: `stockai.h30.targets.forward_returns(..., delist_day,
delist_kind, rule)`, used for the model's training target and for grading. Grader: `holdout --delist
main|last-price|total-loss` (default main when the store has `delist.tsv`; Stooq stores have none, so nothing changes
there): on day L + 1 a held failure or unknown loses everything under main and total-loss; after that its return is 0.
The random accounts use the same rule.

In the grader (judge/src/holdout.rs) the rule sits in the daily return, so it applies to the account, every random
account and buy-and-hold of all candidates alike; under last-price nothing changes there (a missing price was
already a 0 return). The forward return behind the IC in returns and the long-short check counts a stock whose last
close L falls in [d, d + h) at close[L] / close[d] − 1 or −1, as above. The census-variable measures (hit rate, IC)
don't change. `--delist` on a store without `delist.tsv` is an error. The JSON then has `options.delist`,
`options.price_test` and `options.store`, and in each year `delisting`: candidates whose last close falls inside the
year, by kind; candidates already dead at its start; the picks' and the random accounts' names (mean per account)
held over a delisting, by kind; and the names bought after their last close (picks; random accounts, mean per
account, 0 by construction). Only Sharadar runs have these keys: a Stooq run's JSON is byte-identical to before
(checked against `results/30d/judge/lgbm_uniform.json` and `lgbm_volatility.json` on 2026-10-19).

Dead stocks and the random accounts: candidates are the universe of the year before, so a stock that dies before or
during the test year stays a candidate after its last close. The picks can't buy it then (no features, no score), so
under a delisting rule the random accounts can't either: on each decision day d a candidate whose last close is
before d is not drawn, by the uniform accounts or the volatility-matched ones, and it is in no volatility group that
day. When a uniform account sells (as many names as the strategy replaced), it sells its dead members first, since
the picks can't keep a dead stock past a rebalance either. A stock that dies while held stays held until the account
sells it, and the delisting rule prices it. A volatility-matched account still keeps a dead stock while the strategy
keeps the name it shadows. Before this rule, the random accounts spent 2.8% of their held name-days in dead stocks
in a synthetic test in which a fifth of the stocks die.

## The $5 test

`--price-test traded|adjusted`. With `traded` (default when as-traded closes exist), the $5 median-price tests use
as-traded closes: the grader's universe (`holdout.rs liquid_universe`, `targets.judge_universe`) and the model's
liquidity filter. correlation2's market average inside the census variable still uses adjusted closes (correlation2
is not changed). Dollar-volume tests are unchanged. In the grader `traded` takes the upper median of the finite
as-traded closes over the same 252 days before h0, exactly `targets.judge_universe(..., price_close=close_unadj)`;
`--price-test traded` on a store without `close_unadj.f32` is an error.

## The runs (test-results-30d.md, steps 4 and 5)

1. Survivors-only store, `--price-test adjusted`: should roughly reproduce the Stooq result.
2. Full store, `--price-test traded`, delisting rule `main`, bounds alongside: the test.
3. Full store, `--price-test adjusted`: separates the effect of the $5 rule from that of the delistings.
