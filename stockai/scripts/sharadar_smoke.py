"""End-to-end smoke test of the Sharadar path on fake data, before any is bought: Stooq prices dressed as Sharadar's
bulk tables, with made-up delistings (failures, buyouts, unknowns) and reverse splits, through `import-sharadar`,
`export` and `scripts/30d.sh --quick`, for the full store and the survivors-only one.

    uv run python scripts/sharadar_smoke.py [--out runs/sharadar-smoke] [--stocks 400] [--keep]

Nothing here is a result. It shows that the pipeline runs on Sharadar-shaped files, that delisted stocks reach the
grades by the delisting rule, and that the random accounts never buy a dead stock. Only Stooq rows before the locked
year (through 2024-06-26) are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from stockai import data  # noqa: E402

REFERENCE = ["IBM", "GE", "KO", "XOM", "JNJ", "PG", "MMM", "AAPL"]  # correlation2's calendar stocks
LAST = "2024-06-26"  # the last column development code may read
SURVIVORS_FROM = "2024-06-01"  # survivors-only cut for the fake data (Sharadar's real one is 2025-06-27)
JUDGE = ROOT / "judge/target/release/stockai-judge"


def fake_tables(raw: Path, n_stocks: int, seed: int = 0) -> dict:
    ex = data.load()
    rng = np.random.default_rng(seed)
    last = ex.index_of("2024-06-27") - 1
    days = [data.day_to_str(int(d)) for d in ex.days[: last + 1]]
    close = np.asarray(ex.close[:, : last + 1], dtype=np.float64)
    dv = np.asarray(ex.dollar_volume[:, : last + 1], dtype=np.float64)
    finite = np.isfinite(close)
    recent = slice(ex.index_of("2019-01-01"), ex.index_of("2022-06-28"))
    with np.errstate(invalid="ignore"):
        liquid = (np.nanmedian(np.where(finite[:, recent], close[:, recent], np.nan), axis=1) >= 5) & \
                 (np.nanmedian(np.where(finite[:, recent], dv[:, recent], np.nan), axis=1) >= 5e6)
    pool = np.flatnonzero(liquid & (finite.sum(axis=1) > 3000))
    ref = [ex.column(t) for t in REFERENCE]
    others = rng.choice(np.setdiff1d(pool, ref), size=n_stocks - len(ref), replace=False)
    cols = np.r_[ref, np.sort(others)]
    # Delist 15% of the others: half failures, 30% buyouts, 20% unknown; a third of them during 2022-07 .. 2023-12, so the
    # quick run's test years (2022 and 2023) meet some.
    victims = rng.choice(others, size=int(0.15 * others.size), replace=False)
    kinds = rng.choice(["failure", "buyout", "unknown"], size=victims.size, p=[0.5, 0.3, 0.2])
    late = rng.random(victims.size) < 1 / 3
    lo_late, hi_late = ex.index_of("2022-07-01"), ex.index_of("2023-12-29")
    lo_early, hi_early = ex.index_of("2009-01-02"), ex.index_of("2022-06-01")
    cut = np.where(late, rng.integers(lo_late, hi_late, victims.size), rng.integers(lo_early, hi_early, victims.size))
    delist = {int(j): (int(c), k) for j, c, k in zip(victims, cut, kinds)}
    # Reverse splits for 5%: as traded they were a twentieth of the adjusted price until a day in 2015-2020.
    splitters = set(rng.choice(np.setdiff1d(others, victims), size=max(1, int(0.05 * others.size)), replace=False).tolist())
    split_day = {j: int(rng.integers(ex.index_of("2015-01-02"), ex.index_of("2020-12-31"))) for j in splitters}

    raw.mkdir(parents=True, exist_ok=True)
    expect = {"failure": 0, "buyout": 0, "unknown": 0}
    last_dates = {}
    with open(raw / "stocks.csv", "w", newline="") as fs, open(raw / "tickers.csv", "w", newline="") as ft, \
            open(raw / "actions.csv", "w", newline="") as fa:
        ws, wt, wa = csv.writer(fs), csv.writer(ft), csv.writer(fa)
        ws.writerow(["ticker", "date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj", "lastupdated"])
        wt.writerow(["table", "permaticker", "ticker", "name", "exchange", "isdelisted", "category", "cusips", "siccode",
                     "sicsector", "sicindustry", "figi", "famaindustry", "sector", "industry", "scalemarketcap",
                     "scalerevenue", "relatedtickers", "currency", "location", "lastupdated", "firstadded",
                     "firstpricedate", "lastpricedate", "firstquarter", "lastquarter", "secfilings", "companysite"])
        wa.writerow(["date", "action", "ticker", "name", "value", "contraticker", "contraname"])
        for k, j in enumerate(cols):
            j = int(j)
            ticker = ex.tickers[j]
            stop = delist[j][0] if j in delist else last
            rows = np.flatnonzero(finite[j, : stop + 1])
            if rows.size == 0:
                continue
            for d in rows:
                c = close[j, d]
                unadj = c / 20 if j in split_day and d < split_day[j] else c
                vol = dv[j, d] / c if np.isfinite(dv[j, d]) and c > 0 else ""
                ws.writerow([ticker, days[d], f"{c:.4f}", f"{c:.4f}", f"{c:.4f}", f"{c:.4f}", vol and f"{vol:.0f}",
                             f"{c:.4f}", f"{unadj:.4f}", "2026-10-19"])
            gone = j in delist
            last_dates[ticker] = days[rows[-1]]
            category = "ADR Common Stock" if k % 17 == 5 else "Domestic Common Stock"
            wt.writerow(["stocks", 100000 + k, ticker, f"FAKE {ticker}", "NYSE", "Y" if gone else "N", category] + [""] * 13 +
                        ["2026-10-19", "2014-09-24", days[rows[0]], days[rows[-1]], "", "", "", ""])
            if gone:
                kind = delist[j][1]
                expect[kind] += 1
                when = days[rows[-1]]
                if kind == "failure":
                    wa.writerow([when, "bankruptcyliquidation", ticker, f"FAKE {ticker}", "", "N/A", "N/A"])
                elif kind == "buyout":
                    wa.writerow([when, "acquisitionby", ticker, f"FAKE {ticker}", "1000.0", "AAPL", "APPLE INC"])
                else:
                    wa.writerow([when, "delisted", ticker, f"FAKE {ticker}", "", "N/A", "N/A"])
        wt.writerow(["funds", 999999, "SPY", "FAKE SPY", "NYSEARCA", "N", "ETF"] + [""] * 13 +
                    ["2026-10-19", "2014-09-24", days[0], days[-1], "", "", "", ""])
    market = np.nan_to_num(np.asarray(ex.market[: last + 1], dtype=np.float64))
    spy = 100 * np.exp(np.cumsum(market))  # a stand-in benchmark: the equal-weight market
    with open(raw / "funds.csv", "w", newline="") as ff:
        wf = csv.writer(ff)
        wf.writerow(["ticker", "date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj", "lastupdated"])
        for d, p in enumerate(spy):
            wf.writerow(["SPY", days[d], f"{p:.4f}", f"{p:.4f}", f"{p:.4f}", f"{p:.4f}", "1000000", f"{p:.4f}", f"{p:.4f}",
                         "2026-10-19"])
    return {"stocks": len(last_dates), "delisted": expect, "late_delistings": int(late.sum()),
            "reverse_splits": len(split_day), "survivors": sum(d >= SURVIVORS_FROM for d in last_dates.values())}


def run(cmd: list[str], env: dict | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, **(env or {})})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "runs/sharadar-smoke"))
    ap.add_argument("--stocks", type=int, default=400)
    ap.add_argument("--keep", action="store_true", help="keep the output folder (default: removed when all checks pass)")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    facts = fake_tables(out / "raw", a.stocks)
    print("fake tables:", facts, flush=True)
    full, surv = out / "full", out / "survivors"
    run([str(JUDGE), "import-sharadar", "--raw", str(out / "raw"), "--out", str(full / "store")])
    run([str(JUDGE), "import-sharadar", "--raw", str(out / "raw"), "--out", str(surv / "store"), "--survivors-only", SURVIVORS_FROM])
    for root in (full, surv):
        run([str(JUDGE), "export"], env={"C2_DATA_DIR": str(root)})
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"{'ok  ' if cond else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    ex = data.load(full / "export/full")
    kinds = {k: int((ex.delist_kind == c).sum()) for c, k in ((1, "buyout"), (2, "failure"), (3, "unknown"))}
    check("export carries every fake delisting by kind", kinds == facts["delisted"], f"{kinds} vs {facts['delisted']}")
    check("export carries as-traded closes", ex.close_unadj is not None and ex.close_unadj.shape == ex.close.shape)
    sv = data.load(surv / "export/full")
    check("export holds every fake stock", ex.n == facts["stocks"], f"{ex.n} vs {facts['stocks']}")
    check("survivors-only export keeps exactly the stocks still trading at the cut", sv.n == facts["survivors"],
          f"{sv.n} vs {facts['survivors']}")

    res_full, res_surv = out / "results-full", out / "results-survivors"
    run(["scripts/30d.sh", "--quick", "--results", str(res_full), "--runs", str(out / "runs-full")],
        env={"C2_DATA_DIR": str(full)})
    run(["scripts/30d.sh", "--quick", "--price-test", "adjusted", "--results", str(res_surv), "--runs", str(out / "runs-survivors")],
        env={"C2_DATA_DIR": str(surv)})
    g = json.loads((res_full / "grades.json").read_text())
    check("the grades use the delisting rule and the as-traded $5 test",
          g["meta"]["delist"] == "main" and g["meta"]["price_test"] == "traded", f"{g['meta']['delist']}, {g['meta']['price_test']}")
    met = g["meta"]["delisted"]
    check("graded windows meet delisted universe stocks", met and met["universe"] > 0, json.dumps(met))
    for name in ("lgbm", "oracle"):
        s = g["scores"][name]
        lo, mid, hi = (s["delist_bounds"]["total-loss"]["mean_pick_ret"], s["overall"]["mean_pick_ret"],
                       s["delist_bounds"]["last-price"]["mean_pick_ret"])
        check(f"{name}: total-loss <= main <= last-price", lo <= mid + 1e-12 <= hi + 2e-12, f"{lo:+.4%} {mid:+.4%} {hi:+.4%}")
    j = json.loads((res_full / "judge/lgbm_uniform.json").read_text())
    years = [y for y in j["years"] if y.get("delisting")]
    check("the grader reports delistings per year", bool(years) and j["options"].get("delist") == "main",
          f"{len(years)} years, rule {j['options'].get('delist')}")
    bought = [y["delisting"].get("random_bought_dead", y["delisting"].get("random_bought_after_last_close_mean")) for y in years]
    check("random accounts never buy a dead stock", all(b == 0 for b in bought), str(bought))
    summary = (res_full / "summary.md").read_text()
    check("summary shows the delisting table and the fixed checks",
          "Delisted stocks: the rule and its bounds" in summary and "The checks fixed before the Sharadar run" in summary)
    gs = json.loads((res_surv / "grades.json").read_text())
    check("survivors-only run grades with the adjusted $5 test", gs["meta"]["price_test"] == "adjusted")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", flush=True)
    if ok and not a.keep:
        shutil.rmtree(out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
