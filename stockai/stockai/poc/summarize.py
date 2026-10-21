"""Tables for test-results.md from the judge's walk-forward JSON files and the training logs.

    python -m stockai.poc.summarize --results results/poc --runs runs/poc [--years 2010-2023]

Reads every graded file RESULTS/<name>__r<R>_d<D>_<match>.json (stockai-judge holdout --walk-forward --json),
the tripwire files RESULTS/tripwire_<name>.json, the diagnostics RESULTS/diagnostics_<name>.json, RESULTS/g0.json,
RESULTS/poc_log.txt and the train logs under RUNS. Writes RESULTS/summary.md (the tables) and RESULTS/summary.json
(the same numbers, with the G1 gate, the canary readings and the timings), and prints the headline.

LightGBM trained on the next-day actual return (`lgbm_ret`, RUNS/baselines/lgbm_ret, added after the POC run) is
an extra: it gets rows and tables next to LightGBM's wherever its files exist, and nothing changes without them.

Every number is recomputed over the comparison years only (default 2010-2023, the GPT's test years), so models with
scores in more years (reversal and momentum from 2007) are compared on the same years:
- IC (census units) and IC in actual returns: mean over years of the judge's mean daily rank IC; combined
  z = sum of the yearly t / sqrt(years) (Stouffer), as the judge's own summary line. "Census units" is the judge's
  IC against forward sums of correlation2's census variable v; "actual returns" is against close-to-close returns
  over the same holding period;
- long-short top/bottom 10%: mean of the yearly bps/day, combined z the same way;
- beating random: years with p < 0.1 at each cost; returns: the mean of the yearly returns (and compounded).

Gate G1 (PLAN.md section 7), computed on the stockGPT ensemble graded daily, delay 0, uniform random accounts:
- gross signal: IC in actual returns, combined z >= 2 (the census-unit IC is shown next to it; it is not used for
  the gate because v subtracts each stock's trailing 252-day mean residual, so part of v[d+1] is known at d and
  is not in returns: 12-1 momentum and LightGBM show it, test-results.md explains it);
- nothing on the null: the sign-flip null model's combined z is between -2 and 2 for both ICs;
- no tripwire: the large-cap daily rank IC (stockai.poc.controls tripwire) is not above 0.05.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from ..baselines.common import parse_years

NAME = re.compile(r"^(?P<name>.+)__r(?P<r>\d+)_d(?P<d>\d+)_(?P<m>uniform|volatility)\.json$")
Z_GATE = 2.0

LABELS = {
    "gpt_ensemble": "stockGPT, ensemble of the seeds",
    "gpt_ensemble_pct": "stockGPT ensemble as daily percentile ranks (always 20 names), extra",
    "gpt_seed0": "stockGPT, seed 0",
    "gpt_seed1": "stockGPT, seed 1",
    "gpt_seed2": "stockGPT, seed 2",
    "gpt_null": "stockGPT sign-flip null (seed 0)",
    "lgbm": "LightGBM",
    "lgbm_ret": "LightGBM, actual-return target (added after the POC run), extra",
    "reversal5": "5-day reversal",
    "momentum_12_1": "12-1 momentum",
    "vreversal1": "1-day census reversal (-v[d]), extra",
    "canary_shift20": "canary: GPT ensemble, scores 20 days stale",
    "canary_permuted": "canary: GPT ensemble, dates permuted within year",
    "canary_shift250": "canary: GPT ensemble, scores a year (250 days) stale",
    "canary_vreversal1_shift20": "reference canary: 1-day reversal, 20 days stale",
    "canary_vreversal1_permuted": "reference canary: 1-day reversal, dates permuted",
    "lowvol": "static reference: -log volatility (252 days)",
    "decay_shift1": "GPT ensemble, scores used 1 day late",
}


def stouffer(ts: list[float]) -> float:
    ts = [t for t in ts if t is not None and math.isfinite(t)]
    return sum(ts) / math.sqrt(len(ts)) if ts else float("nan")


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def compound(pcts: list[float]) -> float:
    xs = [x for x in pcts if x is not None and math.isfinite(x)]
    return 100.0 * (float(np.prod([1 + x / 100.0 for x in xs])) - 1.0) if xs else float("nan")


def year_of(y: dict) -> int:
    return int(y["start"][:4])


def summarize_run(path: Path, years: set[int]) -> dict:
    j = json.loads(path.read_text())
    m = NAME.match(path.name)
    o = j["options"]
    rows = [y for y in j["years"] if year_of(y) in years]
    costs = [c["cost_bps"] for c in rows[0]["costs"]] if rows else []
    out = {
        "file": path.name,
        "name": m["name"],
        "label": LABELS.get(m["name"], m["name"]),
        "rebalance": o["rebalance"],
        "delay": o["delay"],
        "random_match": o["random_match"],
        "random": o["random"],
        "scores": o.get("scores"),
        "fingerprint": j["variable"]["fingerprint"],
        "years": len(rows),
        "first_year": min((year_of(y) for y in rows), default=None),
        "last_year": max((year_of(y) for y in rows), default=None),
        "all_graded_years": len(j["years"]),
        "judge_summary_all_years": j.get("summary"),
        "costs": costs,
        "ic_mean": mean([y["ic_mean"] for y in rows]),
        "ic_z": stouffer([y["ic_t"] for y in rows]),
        "ic_pos": sum(1 for y in rows if y["ic_t"] > 0),
        "ic_ret_mean": mean([y["ic_ret_mean"] for y in rows]),
        "ic_ret_z": stouffer([y["ic_ret_t"] for y in rows]),
        "ic_ret_pos": sum(1 for y in rows if y["ic_ret_t"] > 0),
        "ls_bps": mean([y["ls_bps_per_day"] for y in rows]),
        "ls_z": stouffer([y["ls_t"] for y in rows]),
        "ls_pos": sum(1 for y in rows if y["ls_bps_per_day"] > 0),
        "ls_turnover": mean([y["ls_turnover"] for y in rows]),
        "hit_rate": mean([y["hit_rate"] for y in rows]),
        "beat": [sum(1 for y in rows if y["costs"][k]["p"] < 0.1) for k in range(len(costs))],
        "ret_mean": [mean([y["costs"][k]["final_pct"] for y in rows]) for k in range(len(costs))],
        "ret_compound": [compound([y["costs"][k]["final_pct"] for y in rows]) for k in range(len(costs))],
        "random_median_mean": [mean([y["costs"][k]["random_median_pct"] for y in rows]) for k in range(len(costs))],
        "spy_mean": mean([y["spy_pct"] for y in rows]),
        "qqq_mean": mean([y["qqq_pct"] for y in rows]),
        "spy_compound": compound([y["spy_pct"] for y in rows]),
        "qqq_compound": compound([y["qqq_pct"] for y in rows]),
        "hold_all_mean": mean([y["hold_all_pct"] for y in rows]),
        "buy_decile_strategy": mean([y["random_check"]["strategy_buy_decile"] for y in rows]),
        "buy_decile_random": mean([y["random_check"]["random_buy_decile"] for y in rows]),
        "held_decile_strategy": mean([y["random_check"]["strategy_held_decile"] for y in rows]),
        "held_decile_random": mean([y["random_check"]["random_held_decile"] for y in rows]),
        "strategy_buys_per_year": mean([y["random_check"]["strategy_buys"] for y in rows]),
        "picks_per_day": mean([picks_per_day(y, o["rebalance"]) for y in rows]),
        "cash_years": sum(1 for y in rows if picks_per_day(y, o["rebalance"]) < 1.0),
        "per_year": rows,
    }
    return out


def picks_per_day(y: dict, rebalance: int) -> float:
    """Mean number of names picked per rebalance day (the judge's hit-rate count over its rebalance days); 20 when
    the account is full, 0 when no candidate had a positive score."""
    return y["hits"] / max(1, math.ceil(y["days"] / max(1, rebalance)))


def train_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def lgbm_val_ics(log: list[dict]) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    """A LightGBM train log's validation ICs by year: against actual returns, and against v."""
    return ({x["year"]: [x["val_ic_ret"]["mean"]] for x in log}, {x["year"]: [x["val_ic_v"]["mean"]] for x in log})


def lgbm_training_table(log: list[dict], title: str) -> list[str]:
    md = ["", f"### {title}", "",
          "| Test year | Train rows | Best iteration | Val L2 / zero forecast | Val IC v | Val IC actual | "
          "Largest gain shares |", "|---|---|---|---|---|---|---|"]
    for x in log:
        gs = sorted(x.get("gain_share", {}).items(), key=lambda kv: -kv[1])[:3]
        md.append(f"| {x['year']} | {x['train_rows']:,} | {x['best_iteration']} | {x['val_l2'] / x['val_l2_zero']:.4f} | "
                  f"{f(x['val_ic_v']['mean'])} | {f(x['val_ic_ret']['mean'])} | "
                  + ", ".join(f"{k} {v:.2f}" for k, v in gs) + " |")
    return md


def f(x, fmt="{:+.4f}") -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def pct(x) -> str:
    return f(x, "{:+.1f}%")


def horizon(r: int) -> str:
    return {1: "daily", 5: "weekly", 21: "monthly"}.get(r, f"{r}-day")


def variant(r: dict) -> str:
    return f"{horizon(r['rebalance'])}, delay {r['delay']}, {r['random_match']}"


def main_table(rows: list[dict]) -> list[str]:
    out = ["| Model | Rebalance, delay, random match | Years | IC census (z) | IC actual returns (z; years > 0) | "
           "Long-short bps/day (z) | Years beating random, p < 0.1 (0 / 5 / 20 bps) | Picks, mean yearly return "
           "(0 / 5 / 20 bps) | Random median (0 / 5 / 20 bps) | SPY | Picks' vol decile (random) | Names picked |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['label']} | {variant(r)} | {r['years']} | {f(r['ic_mean'])} ({f(r['ic_z'], '{:.2f}')}) | "
                   f"{f(r['ic_ret_mean'])} ({f(r['ic_ret_z'], '{:.2f}')}; {r['ic_ret_pos']}) | "
                   f"{f(r['ls_bps'], '{:.2f}')} ({f(r['ls_z'], '{:.2f}')}) | {' / '.join(str(b) for b in r['beat'])} | "
                   f"{' / '.join(pct(x) for x in r['ret_mean'])} | {' / '.join(pct(x) for x in r['random_median_mean'])} | "
                   f"{pct(r['spy_mean'])} | {f(r['held_decile_strategy'], '{:.1f}')} ({f(r['held_decile_random'], '{:.1f}')}) | "
                   f"{f(r['picks_per_day'], '{:.1f}')} |")
    return out


def per_year_table(r: dict, val: dict[int, list[float]] | None = None, val_v: dict[int, list[float]] | None = None,
                   val_label: str = "Val IC actual (mean of seeds)") -> list[str]:
    head = "| Test year | "
    sep = "|---|"
    if val is not None:
        head += f"{val_label} | Val IC v | "
        sep += "---|---|"
    head += ("Test IC census | Test IC actual | Long-short bps/day | Picks 0 bps | p | Picks 5 bps | p | Picks 20 bps | p | "
             "Random median (0 bps) | SPY | QQQ | Picks' vol decile | Names picked |")
    sep += "---|" * 14
    out = [head, sep]
    for y in r["per_year"]:
        yr = year_of(y)
        c = y["costs"]
        row = f"| {y['first_day']}..{y['last_day']} | "
        if val is not None:
            row += f"{f(mean(val.get(yr, [])))} | {f(mean(val_v.get(yr, [])))} | "
        row += (f"{f(y['ic_mean'])} (t {f(y['ic_t'], '{:.1f}')}) | {f(y['ic_ret_mean'])} (t {f(y['ic_ret_t'], '{:.1f}')}) | "
                f"{f(y['ls_bps_per_day'], '{:.2f}')} | "
                + " | ".join(f"{pct(c[k]['final_pct'])} | {f(c[k]['p'], '{:.3f}')}" for k in range(len(c)))
                + f" | {pct(c[0]['random_median_pct'])} | {pct(y['spy_pct'])} | {pct(y['qqq_pct'])} | "
                f"{f(y['random_check']['strategy_held_decile'], '{:.1f}')} | {picks_per_day(y, r['rebalance']):.1f} |")
        out.append(row)
    return out


def g1_gate(by: dict, trip: dict) -> dict:
    """Gate G1 on the stockGPT ensemble (daily, delay 0, uniform), the null and the tripwire."""
    ens = by.get(("gpt_ensemble", 1, 0, "uniform"))
    null = by.get(("gpt_null", 1, 0, "uniform"))
    tw = trip.get("gpt_ensemble")
    checks = []
    if ens:
        checks.append({"check": "gross signal: rank IC in actual returns, combined z >= 2",
                       "value": f"z = {ens['ic_ret_z']:.2f} (mean daily IC {ens['ic_ret_mean']:+.4f}, t > 0 in "
                                f"{ens['ic_ret_pos']} of {ens['years']} years); census units z = {ens['ic_z']:.2f} "
                                f"(IC {ens['ic_mean']:+.4f})",
                       "passed": bool(ens["ic_ret_z"] >= Z_GATE)})
    if null:
        checks.append({"check": "nothing on the sign-flip null: combined z between -2 and 2 for both ICs",
                       "value": f"actual returns z = {null['ic_ret_z']:.2f} (IC {null['ic_ret_mean']:+.4f}); census "
                                f"units z = {null['ic_z']:.2f} (IC {null['ic_mean']:+.4f}); test years: {null['years']}",
                       "passed": bool(abs(null["ic_ret_z"]) < Z_GATE and abs(null["ic_z"]) < Z_GATE)})
    if tw:
        o = tw["overall"]
        checks.append({"check": "no tripwire: large-cap daily rank IC vs next-day return <= 0.05",
                       "value": f"top {tw['top']} by dollar volume: mean {o['mean']:+.4f} (t {o['t']:.2f}), largest "
                                f"year {tw['max_year_mean']:+.4f}",
                       "passed": not tw["tripped"]})
    complete = len(checks) == 3
    return {"checks": checks, "complete": complete, "passed": complete and all(c["passed"] for c in checks)}


def canary_readings(by: dict) -> list[dict]:
    """Each canary's IC next to the unshifted scores' and the static -log volatility reference's."""
    base = by.get(("gpt_ensemble", 1, 0, "uniform"))
    lowvol = by.get(("lowvol", 1, 0, "uniform"))
    out = []
    for name in ("canary_shift20", "canary_permuted", "canary_shift250", "canary_vreversal1_shift20",
                 "canary_vreversal1_permuted"):
        r = by.get((name, 1, 0, "uniform"))
        if not r:
            continue
        census_ok = abs(r["ic_z"]) < Z_GATE
        actual_ok = abs(r["ic_ret_z"]) < Z_GATE
        text = (f"census units IC {r['ic_mean']:+.4f} (z {r['ic_z']:.2f}): "
                + ("about 0" if census_ok else "NOT about 0")
                + f"; actual returns IC {r['ic_ret_mean']:+.4f} (z {r['ic_ret_z']:.2f}): "
                + ("about 0" if actual_ok else "NOT about 0"))
        if not actual_ok and lowvol and name.startswith("canary_") and "vreversal1" not in name:
            text += (f" (the static -log volatility score alone: actual returns IC {lowvol['ic_ret_mean']:+.4f}, "
                     f"z {lowvol['ic_ret_z']:.2f}; census units z {lowvol['ic_z']:.2f})")
        out.append({"name": name, "label": r["label"], "ic_mean": r["ic_mean"], "ic_z": r["ic_z"],
                    "ic_ret_mean": r["ic_ret_mean"], "ic_ret_z": r["ic_ret_z"], "census_about_0": census_ok,
                    "actual_about_0": actual_ok, "reading": text,
                    "unshifted_ic_z": base["ic_z"] if base else None,
                    "unshifted_ic_ret_z": base["ic_ret_z"] if base else None})
    return out


def model_entry(r: dict, notes: str = "") -> dict:
    """The fields test-results.md's headline table (and the workflow's structured output) report per model."""
    return {
        "name": r["label"],
        "variant": f"rebalance {r['rebalance']} ({horizon(r['rebalance'])}) / delay {r['delay']} / {r['random_match']}",
        "years": r["years"],
        "ic_resid_mean": r["ic_mean"],
        "ic_resid_combined_z": r["ic_z"],
        "ic_ret_mean": r["ic_ret_mean"],
        "ic_ret_combined_z": r["ic_ret_z"],
        "ic_ret_years_positive": r["ic_ret_pos"],
        "ls_bps_per_day": r["ls_bps"],
        "ls_combined_z": r["ls_z"],
        "beat_random_years": " / ".join(str(b) for b in r["beat"]),
        "mean_return_pct": (f"picks {' / '.join(pct(x) for x in r['ret_mean'])}; random median "
                            f"{' / '.join(pct(x) for x in r['random_median_mean'])}; SPY {pct(r['spy_mean'])}"),
        "notes": notes,
    }


def step_timings(log_path: Path) -> dict[str, int]:
    """The last 'X took N s' line of each step in poc_log.txt (steps rerun later show their latest time)."""
    out: dict[str, int] = {}
    if not log_path.exists():
        return out
    for line in log_path.read_text().splitlines():
        m = re.match(r"^\s+(.+) took (\d+) s$", line)
        if m:
            out[m[1]] = int(m[2])
    return out


def gpt_timings(log: list[dict]) -> dict:
    by_seed: dict[int, float] = {}
    for x in log:
        by_seed[x["seed"]] = by_seed.get(x["seed"], 0.0) + x.get("total_seconds", 0.0)
    by_year: dict[int, list[dict]] = {}
    for x in log:
        by_year.setdefault(x["year"], []).append(x)
    return {
        "per_seed_seconds": by_seed,
        "per_year": {y: {"train_seconds": mean([x["train_seconds"] for x in xs]),
                         "score_seconds": mean([x["score_seconds"] for x in xs]),
                         "steps": mean([x["steps"] for x in xs]), "max_steps": mean([x["max_steps"] for x in xs]),
                         "best_step": mean([x["best_step"] for x in xs]),
                         "train_positions": xs[0]["train_positions"]} for y, xs in sorted(by_year.items())},
        "total_seconds": sum(by_seed.values()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.poc.summarize", description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--years", default="2010-2023", help="comparison years (default 2010-2023)")
    a = ap.parse_args(argv)
    res, runs = Path(a.results), Path(a.runs)
    years = set(parse_years(a.years))
    graded = [summarize_run(p, years) for p in sorted(res.glob("*__r*_d*_*.json")) if NAME.match(p.name)]
    by = {(r["name"], r["rebalance"], r["delay"], r["random_match"]): r for r in graded}
    fps = sorted({r["fingerprint"] for r in graded})

    gpt_log = train_log(runs / "gpt" / "train_log.jsonl")
    val, val_v = {}, {}
    for x in gpt_log:
        val.setdefault(x["year"], []).append(x["val_ic_actual"])
        val_v.setdefault(x["year"], []).append(x["val_ic_v"])
    null_log = train_log(runs / "gpt_null" / "train_log.jsonl")
    lgbm_all = train_log(runs / "baselines" / "lgbm" / "train_log.jsonl")
    lgbm_log = [x for x in lgbm_all if x.get("kind") == "year"]
    lgbm_val, lgbm_val_v = lgbm_val_ics(lgbm_log)
    lgbm_ret_all = train_log(runs / "baselines" / "lgbm_ret" / "train_log.jsonl")
    lgbm_ret_log = [x for x in lgbm_ret_all if x.get("kind") == "year"]
    lgbm_ret_val, lgbm_ret_val_v = lgbm_val_ics(lgbm_ret_log)
    trip = {p.stem.removeprefix("tripwire_"): json.loads(p.read_text()) for p in sorted(res.glob("tripwire_*.json"))}
    g0 = json.loads((res / "g0.json").read_text()) if (res / "g0.json").exists() else {}

    def pick(*keys):
        return [by[k] for k in keys if k in by]

    headline = pick(("gpt_ensemble", 1, 0, "uniform"), ("gpt_ensemble", 5, 0, "uniform"),
                    ("gpt_ensemble_pct", 1, 0, "uniform"), ("lgbm", 1, 0, "uniform"),
                    ("lgbm", 5, 0, "uniform"), ("lgbm_ret", 1, 0, "uniform"), ("lgbm_ret", 5, 0, "uniform"),
                    ("reversal5", 5, 0, "uniform"), ("reversal5", 1, 0, "uniform"),
                    ("vreversal1", 1, 0, "uniform"), ("momentum_12_1", 5, 0, "uniform"),
                    ("momentum_12_1", 21, 0, "uniform"))
    robust = [r for r in graded
              if r["name"] in ("gpt_ensemble", "gpt_ensemble_pct", "lgbm", "lgbm_ret", "reversal5", "vreversal1")]
    robust.sort(key=lambda r: (r["name"], r["rebalance"], r["delay"], r["random_match"]))
    seeds = [r for r in graded if r["name"].startswith("gpt_seed")]
    controls = pick(("gpt_ensemble", 1, 0, "uniform"), ("decay_shift1", 1, 0, "uniform"), ("canary_shift20", 1, 0, "uniform"),
                    ("canary_permuted", 1, 0, "uniform"), ("canary_shift250", 1, 0, "uniform"),
                    ("lowvol", 1, 0, "uniform"),
                    ("vreversal1", 1, 0, "uniform"), ("canary_vreversal1_shift20", 1, 0, "uniform"),
                    ("canary_vreversal1_permuted", 1, 0, "uniform"), ("gpt_null", 1, 0, "uniform"))
    gate = g1_gate(by, trip)
    canaries = canary_readings(by)
    pct_check = None
    base, pctv = by.get(("gpt_ensemble", 1, 0, "uniform")), by.get(("gpt_ensemble_pct", 1, 0, "uniform"))
    if base and pctv:
        keys = ("ic_mean", "ic_t", "ic_ret_mean", "ic_ret_t", "ls_bps_per_day", "ls_t")
        same = len(base["per_year"]) == len(pctv["per_year"]) and all(
            ya[k] == yb[k] for ya, yb in zip(base["per_year"], pctv["per_year"]) for k in keys)
        pct_check = {"same_rank_measures_every_year": same, "fields": list(keys),
                     "cash_years_original": base["cash_years"], "picks_per_day_original": base["picks_per_day"],
                     "picks_per_day_percentile": pctv["picks_per_day"]}

    run_json = runs / "gpt" / "run.json"
    gpt_seeds = json.loads(run_json.read_text()).get("seeds") if run_json.exists() else None
    md = [f"Comparison years: {min(years)}-{max(years)} (only those a model has scores for). stockGPT seeds in the "
          f"ensemble: {gpt_seeds}. Fingerprints graded: {', '.join(fps)}. IC z and long-short z: Stouffer over the "
          "years of the judge's yearly t. Returns: the $10,000 account's yearly return, averaged over the years. "
          "Picks' vol decile: mean trailing 63-day volatility decile of the names held (0 = calmest tenth of the "
          "day's candidates, 9 = most volatile), random accounts in brackets.", ""]
    md += ["### Gate G1 (stockGPT ensemble, daily, delay 0, uniform random accounts)", "",
           "| Check | Value | Passed |", "|---|---|---|"]
    for c in gate["checks"]:
        md.append(f"| {c['check']} | {c['value']} | {'yes' if c['passed'] else 'NO'} |")
    md.append(f"| **G1** | {'all three checks present' if gate['complete'] else 'checks missing'} | "
              f"**{'PASSED' if gate['passed'] else 'NOT PASSED'}** |")
    md += ["", "### Headline", ""]
    md += main_table(headline)
    if pct_check:
        md += ["", f"Percentile variant: rank ICs and long-short identical to the original scores in every year: "
               f"{'yes' if pct_check['same_rank_measures_every_year'] else 'NO'}; names picked per day "
               f"{pct_check['picks_per_day_original']:.1f} (original) against {pct_check['picks_per_day_percentile']:.1f}; "
               f"years in which the original account held cash (under 1 name a day on average): "
               f"{pct_check['cash_years_original']}."]
    md += ["", "### Every variant of the GPT ensemble, LightGBM and the reversal rules", ""]
    md += main_table(robust)
    md += ["", "### Seeds", ""]
    md += main_table(seeds)
    md += ["", "### Controls (judge, daily, delay 0, uniform)", ""]
    md += main_table(controls)
    if canaries:
        md += ["", "Canary readings (|combined z| < 2 counts as about 0):", ""]
        md += [f"- {c['label']}: {c['reading']}" for c in canaries]
    if trip:
        md += ["", "### Large-cap tripwire (top 500 by 63-day dollar volume, daily rank IC vs next-day return)", "",
               "| Scores | Mean daily IC | t | Days | Largest yearly mean | Tripped (> 0.05) |", "|---|---|---|---|---|---|"]
        for k, t in trip.items():
            o = t["overall"]
            md.append(f"| {k} | {f(o['mean'])} | {f(o['t'], '{:.2f}')} | {o['days']} | {f(t['max_year_mean'])} | "
                      f"{'yes: ' + str(t['flagged']) if t['tripped'] else 'no'} |")
        if "gpt_ensemble" in trip:
            md += ["", "stockGPT ensemble by year: " + "; ".join(
                f"{b['year']} {f(b['mean'])}" for b in trip["gpt_ensemble"]["years"])]
    diag = {p.stem.removeprefix("diagnostics_"): json.loads(p.read_text()) for p in sorted(res.glob("diagnostics_*.json"))}
    for k, dg in diag.items():
        refs = list(dg["overall"])
        md += ["", f"### What {k}'s scores resemble (inputs only: mean daily rank correlation over test days)", "",
               "| Test year | " + " | ".join(refs) + " |", "|---|" + "---|" * len(refs)]
        for row in dg["years"]:
            md.append(f"| {row['year']} | " + " | ".join(f(row[r], "{:+.3f}") for r in refs) + " |")
        md.append("| all | " + " | ".join(f(dg["overall"][r], "{:+.3f}") for r in refs) + " |")
    if ("gpt_ensemble", 1, 0, "uniform") in by:
        md += ["", "### stockGPT ensemble by test year (daily, delay 0, uniform random accounts)", ""]
        md += per_year_table(by[("gpt_ensemble", 1, 0, "uniform")], val, val_v)
    if ("gpt_ensemble", 1, 0, "volatility") in by:
        md += ["", "### stockGPT ensemble by test year (daily, delay 0, volatility-matched random accounts)", ""]
        md += per_year_table(by[("gpt_ensemble", 1, 0, "volatility")])
    if ("gpt_null", 1, 0, "uniform") in by:
        md += ["", "### Null by year", "", "| Test year | Val IC actual | Val IC vs own (flipped) target | Test IC census | "
               "Test IC actual | Long-short bps/day | Picks 0 bps | p |", "|---|---|---|---|---|---|---|---|"]
        nv = {x["year"]: x for x in null_log}
        for y in by[("gpt_null", 1, 0, "uniform")]["per_year"]:
            x = nv.get(year_of(y), {})
            md.append(f"| {y['first_day']}..{y['last_day']} | {f(x.get('val_ic_actual'))} | {f(x.get('val_ic_v'))} | "
                      f"{f(y['ic_mean'])} (t {f(y['ic_t'], '{:.1f}')}) | {f(y['ic_ret_mean'])} (t {f(y['ic_ret_t'], '{:.1f}')}) | "
                      f"{f(y['ls_bps_per_day'], '{:.2f}')} | {pct(y['costs'][0]['final_pct'])} | {f(y['costs'][0]['p'], '{:.3f}')} |")
    if ("lgbm", 1, 0, "uniform") in by:
        md += ["", "### LightGBM by test year (daily, delay 0, uniform random accounts)", ""]
        md += per_year_table(by[("lgbm", 1, 0, "uniform")], lgbm_val, lgbm_val_v, val_label="Val IC actual")
    if ("lgbm_ret", 1, 0, "uniform") in by:
        md += ["", f"### {LABELS['lgbm_ret']}: by test year (daily, delay 0, uniform random accounts)", ""]
        md += per_year_table(by[("lgbm_ret", 1, 0, "uniform")], lgbm_ret_val, lgbm_ret_val_v, val_label="Val IC actual")
    if ("reversal5", 5, 0, "uniform") in by:
        md += ["", "### 5-day reversal by test year (weekly, delay 0, uniform random accounts)", ""]
        md += per_year_table(by[("reversal5", 5, 0, "uniform")])
    if lgbm_log:
        md += lgbm_training_table(lgbm_log, "LightGBM training by year (validation only)")
    if lgbm_ret_log:
        md += lgbm_training_table(lgbm_ret_log, f"{LABELS['lgbm_ret']}: training by year (validation only; the L2 is "
                                                "in units of its daily rank-gauss target)")

    timings = {"steps": step_timings(res / "poc_log.txt"), "gpt": gpt_timings(gpt_log) if gpt_log else None,
               "null": gpt_timings(null_log) if null_log else None,
               "lgbm_per_year_seconds": {x["year"]: x["seconds"].get("total") for x in lgbm_log
                                         if isinstance(x.get("seconds"), dict)},
               "lgbm_total_seconds": next((x.get("seconds") for x in lgbm_all if x.get("kind") == "summary"), None)}
    if lgbm_ret_log:
        timings["lgbm_ret_per_year_seconds"] = {x["year"]: x["seconds"].get("total") for x in lgbm_ret_log
                                                if isinstance(x.get("seconds"), dict)}
        timings["lgbm_ret_total_seconds"] = next((x.get("seconds") for x in lgbm_ret_all if x.get("kind") == "summary"),
                                                 None)
    md += ["", "### Timings", ""]
    if timings["steps"]:
        md += ["| Step (poc_log.txt) | Seconds |", "|---|---|"]
        md += [f"| {k} | {v:,} |" for k, v in timings["steps"].items()]
    if timings["gpt"]:
        g = timings["gpt"]
        md += ["", "stockGPT, per seed (train + score, all years): "
               + ", ".join(f"seed {s} {v / 60:.1f} min" for s, v in g["per_seed_seconds"].items()), "",
               "| Test year | Train positions | Steps run / budget (mean of seeds) | Best step | Train s | Score s |",
               "|---|---|---|---|---|---|"]
        for y, t in g["per_year"].items():
            md.append(f"| {y} | {t['train_positions']:,} | {t['steps']:.0f} / {t['max_steps']:.0f} | {t['best_step']:.0f} | "
                      f"{t['train_seconds']:.0f} | {t['score_seconds']:.0f} |")
    if timings["null"]:
        md += ["", "Null, per year (train + score): " + ", ".join(
            f"{y} {t['train_seconds'] + t['score_seconds']:.0f} s" for y, t in timings["null"]["per_year"].items())]
    if timings["lgbm_per_year_seconds"]:
        md += ["", f"LightGBM: {f(timings['lgbm_total_seconds'], '{:.0f}')} s in all; per year (rows + fit + predict): "
               + ", ".join(f"{y} {s:.1f} s" for y, s in timings["lgbm_per_year_seconds"].items() if s is not None)]
    if timings.get("lgbm_ret_per_year_seconds"):
        md += ["", f"LightGBM, actual-return target (extra): {f(timings['lgbm_ret_total_seconds'], '{:.0f}')} s in all; "
               "per year (rows + fit + predict): " + ", ".join(
                   f"{y} {s:.1f} s" for y, s in timings["lgbm_ret_per_year_seconds"].items() if s is not None)]

    md += ["", "### Ledger: every graded configuration", "", "| File | Model | Variant | Years in comparison | All graded years |",
           "|---|---|---|---|---|"]
    for r in graded:
        md.append(f"| {r['file']} | {r['label']} | {variant(r)} | {r['years']} | {r['all_graded_years']} |")
    (res / "summary.md").write_text("\n".join(md) + "\n")

    def slim(r):
        return {k: v for k, v in r.items() if k not in ("per_year", "judge_summary_all_years")}

    out = {"years": sorted(years), "fingerprints": fps, "g1": gate, "canaries": canaries, "percentile_check": pct_check,
           "models": [model_entry(r) for r in headline], "graded": [slim(r) for r in graded],
           "tripwire": {k: {kk: t[kk] for kk in ("overall", "max_year_mean", "tripped", "flagged")} for k, t in trip.items()},
           "g0": {k: v.get("passed") for k, v in g0.items()},
           "diagnostics": {k: dg["overall"] for k, dg in diag.items()},
           "gpt_val_ic_actual_by_year": {y: mean(v) for y, v in sorted(val.items())},
           "gpt_runs": len(gpt_log), "null_runs": len(null_log), "timings": timings}
    (res / "summary.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    for r in headline:
        print(f"{r['label']:<44} {variant(r):<28} {r['years']:>2} yrs  IC {f(r['ic_mean'])} z {f(r['ic_z'], '{:.2f}'):>6}  "
              f"ICret {f(r['ic_ret_mean'])} z {f(r['ic_ret_z'], '{:.2f}'):>6}  LS {f(r['ls_bps'], '{:.2f}'):>6} bps/d "
              f"z {f(r['ls_z'], '{:.2f}'):>6}  beat {'/'.join(str(b) for b in r['beat'])}")
    for c in gate["checks"]:
        print(f"G1 {'pass' if c['passed'] else 'FAIL'}: {c['check']}: {c['value']}")
    print(f"G1: {'PASSED' if gate['passed'] else 'NOT PASSED'}" + ("" if gate["complete"] else " (checks missing)"))
    for c in canaries:
        print(f"{c['label']}: {c['reading']}")
    print(f"wrote {res / 'summary.md'} and {res / 'summary.json'} ({len(graded)} graded files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
