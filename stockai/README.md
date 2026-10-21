# stockai

Trains a small GPT-style model on stock price history to predict each stock's next move, the way an LLM predicts the next word. It then tests whether the predictions beat random guessing after trading costs, using correlation2's $10,000 holdout test.

- Start with [PLAN.md](PLAN.md).
- The research behind it: [the report](<reports/AI models for stock price prediction.md>), with notes in `research_notes/`.
- Status (2026-10-19): the proof of concept has run end to end. stockGPT v1 passed gate G1: a real next-day signal, rank IC +0.0124 against next-day returns (z = 5.85 over 14 walk-forward years), nothing on the sign-flip null, no leakage tripwire. But its daily $10,000 account loses money at 5 and 20 bps per trade, and it is no better than turnover-matched random accounts at any cost level (it beats them at p < 0.1 in 4 / 4 / 3 of 14 years; combined over the years, z ≈ 0.8 at 0 bps and 0.5 at 20 bps). The 5-day reversal rule and LightGBM do better on that test. A LightGBM trained on actual next-day returns, added during verification, is stronger still in IC (z = 9.15; its daily picks beat random in 10 of 14 years, as the daily 5-day rule's do), but its daily edge is mostly gone a day late, and nothing in the POC compounds to more than SPY after 20 bps costs. Results and caveats: [test-results.md](test-results.md).
- stockai-30d (2026-10-19): a second test asks whether a model can pick the 10 stocks that will gain the most over the next calendar month. In this data LightGBM's monthly picks beat chance, the average stock (+3.2% a month against +1.0%) and comparable stocks, and a monthly $10,000 account ended well above SPY after costs. But a third of the picks are collapsed stocks, and this data holds only companies still listed in 2025, so none of them were delisted before then. When the model may not pick collapsed stocks, little of the edge is left, and half the extra return came in 5 of 162 months. It needs a rerun on survivorship-free data before it means much. Results and caveats: [test-results-30d.md](test-results-30d.md).
- stockai-1w (2026-10-19): the same test over one week. The weekly picks rank stocks better than chance, with less dependence on a few lucky periods or collapsed stocks, and the signal holds among the 500 largest stocks. But it is small (about 0.3 points a week) and weekly trading costs more than it earns: $19,052 after 14 years at 20 bps against SPY's $65,036. Results and caveats: [test-results-1w.md](test-results-1w.md).
- stockai-2m (2026-10-19): the same test over two months (42 trading days). No edge: the picks don't beat comparable stocks, and $10,000 ends at $31,176 at 20 bps against SPY's $65,036, even though trading costs only about 2 points a year. Results: [test-results-2m.md](test-results-2m.md). The three horizons side by side, and what each is good for: [test-results-horizons.md](test-results-horizons.md).
- stockGPT on the same tests, at one week, one month and two months (`scripts/gpt_horizon.sh`): no edge at any. It mostly learned to pick calm, low-volatility stocks, which rarely make the period's biggest gainers; its top 10 earned about what the average stock did. Details in both results files.
- correlation2's outside data (2026-10-19: 1,753 series of weather, politics, news, rates, currencies, commodities, search trends and more) as extra LightGBM inputs at all three horizons: no improvement. The one pass (one week) disappeared when revised official statistics were left out; at one and two months it made the picks worse, and a control given the same data shifted four years out of line did as well. This agrees with correlation2's own scan of the same series. Results: [test-results-external.md](test-results-external.md).
- Survivorship-free rerun: the code for Sharadar data (delisted stocks counted, the $5 test on prices as traded) is in place, checked against the Stooq results and run end to end on fake Sharadar-shaped data; the data hasn't been bought yet. See [docs/sharadar.md](docs/sharadar.md).

## How to run stockai-30d, stockai-1w and stockai-2m

```
scripts/30d.sh --quick               # test years 2022-2023 with a small LightGBM: a few minutes
scripts/30d.sh                       # the monthly test, 2010-2023: results/30d/, runs/30d/
scripts/30d.sh --horizon 5           # the weekly test, 2010-2023: results/1w/, runs/1w/
scripts/30d.sh --horizon 42          # the two-month test, 2010-2023: results/2m/, runs/2m/
```

## Survivorship-free rerun (Sharadar)

The plan is in `test-results-30d.md` ("Next steps") and the details in `docs/sharadar.md`. Once Sharadar "Prices", full history, is bought (sharadar.com/subscribe, $39 for one month), save the API key in a terminal of your own; this asks for it without echoing it:

```
mkdir -p ~/.config/sharadar && read -rsp 'Sharadar API key: ' KEY && printf '%s\n' "$KEY" > ~/.config/sharadar/api_key && chmod 600 ~/.config/sharadar/api_key && unset KEY
```

Then download, check, import, export and run (the runs take an hour or more and about 30 GB under `runs/`):

```
scripts/sharadar_download.sh
scripts/sharadar_check.sh
RAW=$(ls -d ~/.local/share/stockai/sharadar/raw/*/ | sort | tail -n 1)
judge/target/release/stockai-judge import-sharadar --raw "$RAW" --out ~/.local/share/stockai/sharadar/store
judge/target/release/stockai-judge import-sharadar --raw "$RAW" --out ~/.local/share/stockai/sharadar-survivors/store --survivors-only 2025-06-27
C2_DATA_DIR=~/.local/share/stockai/sharadar judge/target/release/stockai-judge export
C2_DATA_DIR=~/.local/share/stockai/sharadar-survivors judge/target/release/stockai-judge export
scripts/sharadar_runs.sh
```

`uv run python scripts/sharadar_smoke.py` runs the same chain on fake Sharadar-shaped data made from the Stooq prices, in about 30 seconds.

## Where results land

Results land in `results/30d/summary.md` (or `results/30d-quick/`, `results/1w/`), with the survivorship subsets in `large500/` and `no_collapse/` beside it. `test-results-30d.md` and `test-results-1w.md` explain the full runs. A full run takes about 5 minutes on an otherwise idle machine; LightGBM slows down more than tenfold when other jobs compete for the CPU (33 minutes measured once). Each run also writes about 4 GB of tie-broken scores to its `runs/.../jitter/`. With `C2_DATA_DIR` pointing at a Sharadar data folder, the same commands run on Sharadar data (docs/sharadar.md).

## How to run the POC

One command reproduces the proof of concept: the export, the G0 pipeline check, the stockGPT walk-forward, the sign-flip null, the baselines, grading by the judge, the controls, and the summary tables.

```
scripts/poc.sh --quick    # test years 2022-2023, one seed, the null on 2022: about 7 minutes (401 s measured on a free GPU; 633 s with the GPU shared)
scripts/poc.sh            # test years 2010-2023, seeds 0 1 2, the null on 2012/2017/2022: about an hour (61 minutes measured with the GPU partly shared, before the LightGBM extra added about 2 minutes)
```

Results land in `results/poc/summary.md` (or `results/poc-quick/summary.md`): gate G1, every table, the controls and the ledger of graded configurations. `test-results.md` explains the full run.

What it needs:
- correlation2's price store at `~/.local/share/correlation2/store` (only read; set `C2_DATA_DIR` for another location).
- The judge, `judge/target/release/stockai-judge`: the script builds it with `cargo build --release` if it is missing.
- uv and the project environment (`uv sync`), and an NVIDIA GPU for stockGPT.
- Optional: `C2_REF=<a c2 binary built at correlation2 commit 9335126>` adds the G0 check against correlation2's own reversal table.

What it writes:
- `results/poc/` (or `results/poc-quick/`): `summary.md` with every table, the judge's walk-forward results as `<model>__r<rebalance>_d<delay>_<random match>.json`, the printed tables in `tables/`, `g0.json`, the tripwire and diagnostics results, and `poc_log.txt`.
- `runs/poc/` (or `runs/poc-quick/`, not in git): scores files, checkpoints and training logs.

Options: `--steps export,g0,gpt,null,baselines,grade,controls,summary` runs a subset (for example `--steps grade,controls,summary` regrades existing scores), `--results DIR` and `--runs DIR` change the folders, and `--force` lets the stockGPT steps replace an earlier run (without it, a second run into the same `runs/` folder stops at the stockGPT step). The verification's own quick run is kept in `results/poc-quick-new/` and `runs/poc-quick-new/`, to compare a fresh `--quick` run (which writes `results/poc-quick/`) against.

The pieces can also be run on their own:

| Command | What it does |
|---|---|
| `judge/target/release/stockai-judge export` | Write correlation2's census variable and prices for Python (`docs/c2-export.md`) |
| `judge/target/release/stockai-judge holdout --family scores --scores F.npy --export ~/.local/share/correlation2/export/full --rebalance 5 --walk-forward --start 2023-06-28 --json OUT.json` | Grade a scores file, year by year, 2006-2023 (years without scores are skipped) |
| `uv run python -m stockai.gpt.walkforward --out DIR --years 2010-2023 --seeds 0 1 2 [--null sign-flip]` | stockGPT walk-forward: one model per test year and seed |
| `uv run python -m stockai.baselines.lgbm --out DIR --years 2010-2023 --threads 8 [--target ret]` | LightGBM walk-forward, trained on next-day v (the default) or, with `--target ret`, on next-day actual returns |
| `uv run python -m stockai.baselines.{reversal,momentum,vreversal} --out F.npy` | The rule baselines |
| `uv run python -m stockai.poc.controls {shift,permute,tripwire,lowvol,percentile} ...` | Score canaries, the large-cap leakage tripwire, the static volatility reference, and daily percentile ranks |
| `uv run pytest -q` | The tests |

The locked year (2024-06-28 to 2025-06-27) is never graded, trained on or read during development: the judge refuses it without `--locked`, `stockai/splits.py` refuses it, and every scores file is NaN from its first decision day.
