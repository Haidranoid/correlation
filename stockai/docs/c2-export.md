# The correlation2 interface

stockai never computes prices, the census variable or grades itself. correlation2's code does both ends, through `judge/target/release/stockai-judge` (the `judge/` crate: correlation2's holdout at commit 9335126, forked into stockai so correlation2 itself is not modified, with its data crates as pinned git dependencies):

- `stockai-judge export` writes the census variable and prices for Python.
- `stockai-judge holdout --family scores` grades a file of model scores with the same $10,000 test as every other correlation2 family.

## `stockai-judge export`

```
stockai-judge export [--out <dir>]        default dir: $C2_DATA_DIR/export/full  (~/.local/share/correlation2/export/full)
```

It builds `Plan::new(&store, Config::full())`, exactly as `c2 census holdout` does for a full census run, and writes `plan.var`. So an export and the holdout always see the same days, stocks and values.

| File | Type, shape | Meaning |
|---|---|---|
| `meta.json` | JSON | `format` ("c2-export-1"), `n`, `t`, `first_day`, `last_day` (YYYY-MM-DD), `fingerprint` (FNV-1a 64 of `days` and store ids, hex), `git_commit`, `variable` (a description of each array, including the trailing-mean subtraction and the two exceptions under "Point in time") |
| `days.npy` | int32 `[T]` | trading days, as days since 1970-01-01 (`Variable.days`) |
| `symbols.tsv` | text | header `col\tid\tticker`: column j, store id, ticker, for j = 0..N-1 (`Variable.ids` order) |
| `v.npy` | float32 `[N, T]` | census variable: (market-residual daily log return − its trailing 252-day mean) ÷ trailing 252-day residual sd, clipped ±5; NaN where unavailable (`Variable.v`). The mean and sd are over the residuals of days d−252 .. d−1. Because the trailing mean is subtracted, part of v[d+1] is known at day d (minus the past year's mean residual, in sd units): a score that leans against the past year's performance gets census-unit IC that is not in actual returns (test-results.md) |
| `close.npy` | float32 `[N, T]` | Stooq's split- and dividend-adjusted closes as of the download (data through 2025-06-27), after correlation2's bad-print repair; NaN where missing (`Variable.close`). Returns from them are point in time; price levels are not (below) |
| `dollar_volume.npy` | float32 `[N, T]` | close × volume (`Variable.dollar_volume`). Volume is split-adjusted too, so splits cancel, but the dividend adjustment of the close scales past values down (AAPL in 2010: by about 0.84) |
| `market.npy` | float64 `[T]` | equal-weight mean daily log return of the liquid stocks (`Variable.market`) |

Every `[N, T]` array is C-order with row j = stock j and column d = day d. That is the same memory layout as Rust's column-major `t × n` `Variable` (`v[j * t + d]`), so the files are byte copies.

**Point in time, with two exceptions.** v[d] uses betas and the residual mean and sd through day d−1, plus day d's own return; the market's liquid set is re-evaluated every 21 rows from the 252 rows before. Two inputs reach past d:
1. **The bad-print repair** blanks a close that moves more than 2× (up or down) and comes back within about 10% the next day, so whether close[d] (and v[d], v[d+1]) exists depends on close[d+1]. It touches 72 cells before 2024-06-27, 3 of them on graded universe days.
2. **Closes are adjusted as of the download** (but see the dividend gap below), so anything that uses price levels sees later splits, including splits in the locked year. The $5 median-price test (the market's liquid set and the judge's universe) therefore leaves out stocks that later split heavily: NVDA, trading $100M–$2.2B a day, is out of the universe in 2010–2017 and 2019 at an adjusted median below $5; AVGO and LRCX (10:1 splits on 2024-07-15 and 2024-10-03) in 2010–2014; CMG in 2010–2011. Dollar volume carries later dividends (above).

Returns, and so v, are ratios and are unaffected, except through which stocks enter the market average. Both exceptions affect every model and the random accounts alike.

**The dividend gap (found 2026-10-19 against Sharadar's free sample).** For some stocks Stooq's "adjusted" closes stop being dividend-adjusted in late 2022: IBM, KO, JNJ, PG and MMM equal their as-traded prices from about November 2022 on, so each later ex-dividend day shows the dividend as a loss (IBM on 2023-02-09: −1.64% against −0.43%). AAPL, MSFT and NVDA stay adjusted. Spin-offs aren't adjusted (IBM on 2021-11-04, the Kyndryl spin-off: −4.94% against −0.53%). Returns and v from those closes are wrong by those amounts; how many stocks are affected is a day-one check on the Sharadar data (`stockai-judge check-sharadar`).

## Scores files

```
stockai-judge holdout --family scores --scores <file.npy> --export <dir> [--rebalance N] [--delay D] [--random-match uniform|volatility] [--walk-forward] [--start YYYY-MM-DD] [--costs 0,5,20] [--json <out.json>] ...
```

- `.npy`, dtype `<f4` or `<f8`, shape `[N, T]`, C-order, aligned to the export of the same store (same N, T, column order and days).
- `scores[j, d]` is the score for stock j **decided at the close of day d using only data through day d**. With `--delay 0` the account buys at that close and holds over day d+1; with `--delay D` it holds the picks decided at d from the close of d+D, earning d+D to d+D+1. Higher means more attractive.
- NaN means no score: the stock can't be picked and is left out of rank IC and the long-short check that day.
- The judge picks from its own liquid universe (median price ≥ $5, median dollar volume ≥ $5M over the past year, data on ≥ 90% of the 756 days before the holdout), as for `--family reversal`. Scores for other stocks are ignored.
- The judge reads scores only on rebalance days.
- **Picks:** the 20 highest positive scores (0 and NaN are never picked). The sort is stable, so equal scores go to the lower column; the rank IC also orders ties by column. With fewer than 20 positive scores, the empty slots are cash.
- **Decision-day measures:** hit rate, both rank ICs and the long-short check compare the scores at d with the move from d to d + rebalance, whatever `--delay` is, so they are the same at every delay and in both `--random-match` modes. At `--delay 1` they describe a move the delayed account doesn't earn; only the account columns change with the delay.
- **The account** (correlation2's model, unchanged): $10,000 at a fixed weight of 1/20 per name, re-equal-weighted every day at no cost (weekly and monthly books too). A cost of c bps per side is charged on each name bought or sold, as V × (1 + gross − cost) in the same step as that day's return. The first day's purchases are charged; there is no closing cost; empty slots are cash earning 0. The random accounts follow the same rules, so comparisons are fair, but absolute returns at nonzero cost are slightly optimistic.
- **Random accounts:** `--random N` of them (1,000 in the POC), drawn from the same universe, holding as many names and replacing as many at each rebalance as the strategy. A uniform account can buy back a name it sold the same day, so it trades about 1% less than the strategy, which makes its p slightly conservative at 5 and 20 bps. `--random-match volatility` draws each buy from the bought name's trailing 63-day-volatility decile that day: exact on the buy day, but coarse within the decile (raw volatility within about 5% in the POC), and held names' deciles drift (by up to +0.25 daily and +0.36 weekly in the POC).

## Walk-forward years

`--walk-forward --start <Y>-06-28` grades every one-year period [y-06-28, (y+1)-06-28) for y = 2006 .. Y. Years with no finite scores are skipped.

| Period | Test years | Use |
|---|---|---|
| Development | 2010 .. 2023 (`--start 2023-06-28`) | everything during development |
| Locked year | 2024-06-28 .. 2025-06-27 | one run at the end (PLAN.md §4.4); never during development |
| Sealed exam | after 2025-06-27 | needs fresh data |

`stockai/splits.py` turns a test year into the exact training, validation and test day ranges, so every model uses the same days and gaps.
