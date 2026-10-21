"""Grades stockai-30d's scores files: how often each one's top 10 were the period's top 10 gainers (stockai.h30.metrics),
year by year over the development test years, plus the $10,000 accounts the grader ran on them (judge JSON files).
The period is the horizon: 21 trading days (a month, the default) or 5 (a week).

    uv run python -m stockai.h30.evaluate --scores NAME=PATH [NAME=PATH ...] [--judge NAME=JSON ...] --out DIR
        [--horizon H] [--subset large500|no_collapse] [--price-test traded|adjusted] [--delist main|...]

Writes DIR/grades.json (every graded day, per year and overall) and DIR/summary.md (the tables test-results-30d.md
quotes). Reads the development columns only: windows reaching the locked year are never graded. On a Sharadar export
(docs/sharadar.md) stocks delisted inside a window count by the delisting rule (main, with both bounds reported) and the
universe's $5 test uses as-traded closes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .. import data
from ..baselines.common import load_dev, parse_years
from .metrics import COMPARABLE, TOP, account, grade, grade_day, past_year_ratio, trailing_vol
from .targets import DELIST_RULES, H, forward_returns, non_overlapping, year_spec

COSTS = (0, 5, 20, 50)


def load_scores(path: str, cutoff: int) -> np.ndarray:
    s = np.load(path, mmap_mode="r")
    if np.isfinite(np.asarray(s[:, cutoff:], dtype=np.float64)).any():
        raise ValueError(f"{path}: finite scores at or after the development cutoff")
    return np.asarray(s[:, :cutoff], dtype=np.float32)


SUBSETS = ("large500", "no_collapse")


def subset_mask(name: str, close: np.ndarray, dollar_volume: np.ndarray) -> np.ndarray:
    """[N, C] masks known at each day d:
    - large500: the 500 stocks with the highest average dollar volume over days d-62 .. d (missing days skipped);
    - no_collapse: stocks not down more than 50% over the past year (close[d] / close[d-252] > 0.5), and with that known.
    """
    c = np.asarray(close, dtype=np.float64)
    if name == "large500":
        x = np.asarray(dollar_volume, dtype=np.float64)
        ok = np.isfinite(x)
        cs = np.zeros((x.shape[0], x.shape[1] + 1))
        cn = np.zeros_like(cs)
        np.cumsum(np.where(ok, x, 0.0), axis=1, out=cs[:, 1:])
        np.cumsum(ok, axis=1, out=cn[:, 1:])
        hi = np.arange(1, x.shape[1] + 1)
        lo = np.maximum(hi - 63, 0)
        n = cn[:, hi] - cn[:, lo]
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = np.where(n >= 21, (cs[:, hi] - cs[:, lo]) / n, np.nan)
        out = np.zeros(x.shape, dtype=bool)
        for d in range(x.shape[1]):
            col = avg[:, d]
            fin = np.flatnonzero(np.isfinite(col))
            if fin.size:
                out[fin[np.argsort(-col[fin], kind="stable")[:500]], d] = True
        return out
    if name == "no_collapse":
        out = np.zeros(c.shape, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, 252:] = c[:, 252:] / c[:, :-252] > 0.5
        return out
    raise ValueError(name)


def judge_summary(path: str) -> dict:
    """The grader's walk-forward JSON for 2010-2023: mean yearly return at each cost, years beating random, compounding."""
    d = json.loads(Path(path).read_text())
    ys = [y for y in d["years"] if "2010-06-28" <= y["first_day"] < "2024-06-28"]
    out = {"years": len(ys), "match": d["options"]["random_match"], "picks": d["options"]["picks"],
           "rebalance": d["options"]["rebalance"], "costs": [int(c["cost_bps"]) for c in ys[0]["costs"]]}
    for ci, cost in enumerate(y["cost_bps"] for y in ys[0]["costs"]):
        rets = [y["costs"][ci]["final_pct"] / 100 for y in ys]
        out[f"mean_{int(cost)}"] = float(np.mean(rets))
        out[f"end_{int(cost)}"] = float(10000 * np.prod([1 + r for r in rets]))
        out[f"beat_{int(cost)}"] = sum(y["costs"][ci]["p"] < 0.1 for y in ys)
        out[f"worse_{int(cost)}"] = sum(y["costs"][ci]["p"] > 0.9 for y in ys)
        out[f"rand_{int(cost)}"] = float(np.mean([y["costs"][ci]["random_median_pct"] / 100 for y in ys]))
    spy = [y["spy_pct"] / 100 for y in ys]
    out["spy_mean"], out["spy_end"] = float(np.mean(spy)), float(10000 * np.prod([1 + r for r in spy]))
    out["ic_ret_z"] = float(sum(y["ic_ret_t"] for y in ys) / math.sqrt(len(ys)))
    return out


def fmt(x, nd=2, pct=False, sign=False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    v = 100 * x if pct else x
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s + ("%" if pct else "")


def period_words(h: int) -> tuple[str, str, str]:
    """(name, noun, adjective) for the horizon: ("stockai-30d", "month", "monthly") at 21 days."""
    if h == 21:
        return "stockai-30d", "month", "monthly"
    if h == 5:
        return "stockai-1w", "week", "weekly"
    if h == 42:
        return "stockai-2m", "2-month period", "2-month"
    return f"stockai-{h}d", f"{h}-day period", f"{h}-day"


def account_end(judges: dict, name: str, cost: int = 20) -> tuple[float | None, str]:
    """The grader's $10,000 after 14 years for `name`: the median over its random tie-breaks (scripts/30d.sh jitter
    step) when there are any, else its own run."""
    jit = [j[f"end_{cost}"] for k, j in judges.items()
           if k.startswith(f"{name}, ties at random, seed") and j["match"] == "uniform" and f"end_{cost}" in j]
    if jit:
        return float(np.median(jit)), f"median of {len(jit)} random tie-breaks"
    if name in judges and f"end_{cost}" in judges[name]:
        return judges[name][f"end_{cost}"], "ties by ticker order"
    return None, ""


def write_summary(results: dict, judges: dict, years: list, out: Path, h: int, cutoff: int,
                  subset: str | None = None, info: dict | None = None) -> str:
    info = info or {}
    title, per, adj = period_words(h)
    Per = per[0].upper() + per[1:]
    dropped = [int(d) for y in years for d in y.dropped]
    locked = sum(d + h > cutoff - 1 for d in dropped)  # the rest were dropped by targets.non_overlapping
    lines = [f"# {title} results (horizon {h} trading days, top {TOP}{', subset ' + subset if subset else ''})", "",
             f"Test years {years[0].year}-{years[-1].year}, one graded decision every {h} trading days "
             f"({sum(len(y.decisions) for y in years)} in all). Rebalance days left ungraded: {len(dropped) - locked} "
             f"whose {per} would overlap the next graded {per}, {locked} whose {per} reaches the locked year.", ""]
    if info.get("source_note"):
        lines += [info["source_note"], ""]
    lines += [f"## Top-10 hits on the {per}'s 10 biggest gainers", "",
              "| Scores | Graded days | Hits per day | Chance (uniform) | Chance (same volatility) | z vs uniform | "
              "z vs same volatility | Days tied at the cut | Hits among the 10 biggest losers per day | Chance of a loser hit "
              f"(same volatility) | Tail hits that were gainers | Picks in the top / bottom decile | Picks' mean {per} "
              "return vs all | Rank IC (t; days > 0) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        o = r["overall"]
        lines.append(
            f"| {name} | {o['days']} | {fmt(o['hits_gain_per_day'], 3)} | {fmt(o['exp_uniform_per_day'], 3)} | "
            f"{fmt(o['exp_vol_per_day'], 3)} | {fmt(o['z_vs_uniform'], 2, sign=True)} | {fmt(o['z_vs_vol'], 2, sign=True)} | "
            f"{o['days_tied_at_cut']} | {fmt(o['hits_loss'] / o['days'], 3)} | {fmt(o['exp_vol_loss_per_day'], 3)} | "
            f"{fmt(o['gain_share_of_tail_hits'], 0, pct=True)} | "
            f"{fmt(o['top_decile_share'], 1, pct=True)} / {fmt(o['bottom_decile_share'], 1, pct=True)} | "
            f"{fmt(o['mean_pick_ret'], 2, pct=True, sign=True)} vs {fmt(o['mean_all_ret'], 2, pct=True, sign=True)} "
            f"(t {fmt(o['excess_t'], 1)}) | {fmt(o['ic_mean'], 4, sign=True)} ({fmt(o['ic_t'], 1)}; {o['ic_positive']}) |")
    lines += ["", "Chance (uniform): hits expected from 10 random picks among the same stocks. Chance (same volatility): "
              "each pick swapped for a random stock from its own fiftieth of the day's stocks by trailing 63-day "
              "volatility. Tail hits that were gainers: of the picks that landed among the 10 biggest gainers or the 10 "
              "biggest losers, the share that were gainers (picking volatile stocks with no sense of direction lands "
              f"in both). The top decile is the tenth of stocks that rose most that {per}; a random pick lands there "
              "10% of the time.", ""]
    for name, r in results.items():
        lines += [f"### {name} by test year", "",
                  "| Year | Days | Hits | Chance (uniform) | Chance (same volatility) | Loser hits | Top / bottom decile | "
                  f"Picks' {per} return vs all | Rank IC |", "|---|---|---|---|---|---|---|---|---|"]
        for y in r["years"]:
            if not y.get("days"):
                continue
            lines.append(f"| {y['year']} | {y['days']} | {fmt(y['hits_gain'], 1)} | {fmt(y['exp_uniform_per_day'] * y['days'], 2)} | "
                         f"{fmt(y['exp_vol_per_day'] * y['days'], 2)} | {fmt(y['hits_loss'], 1)} | "
                         f"{fmt(y['top_decile_share'], 0, pct=True)} / {fmt(y['bottom_decile_share'], 0, pct=True)} | "
                         f"{fmt(y['mean_pick_ret'], 2, pct=True, sign=True)} vs {fmt(y['mean_all_ret'], 2, pct=True, sign=True)} | "
                         f"{fmt(y['ic_mean'], 3, sign=True)} |")
        lines.append("")
    lines += [f"## {Per}-level view and the collapsed picks", "",
              f"| Scores | Hits vs same-volatility chance, {per}-level t | {Per}s ahead of the average stock | Median {per}'s "
              f"excess | Share of the total excess from the 5 best {per}s | Excess without those 5 (t) | Pick weight on stocks "
              f"down >50% over the past year | Their return (pooled; averaged by {per}) | The other picks' return |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        o, cs = r["overall"], r.get("collapse_split", {})
        lines.append(f"| {name} | {fmt(o['hits_t_month'], 2, sign=True)} | {o['months_ahead']} of {o['days']} | "
                     f"{fmt(o['median_excess'], 2, pct=True, sign=True)} | {fmt(o['top5_share_of_excess'], 0, pct=True)} | "
                     f"{fmt(o['excess_without_top5'], 2, pct=True, sign=True)} ({fmt(o['excess_without_top5_t'], 2)}) | "
                     f"{fmt(cs.get('weight_share'), 1, pct=True)} | {fmt(cs.get('collapsed_ret'), 2, pct=True, sign=True)}; "
                     f"{fmt(cs.get('collapsed_ret_month_avg'), 2, pct=True, sign=True)} | {fmt(cs.get('other_ret'), 2, pct=True, sign=True)} |")
    lines.append("")
    lines += ["## Against comparable stocks", "",
              f"Each pick against the {COMPARABLE} eligible stocks it didn't pick that are closest to it that day in percentile "
              "rank of trailing volatility and of past-year fall (stockai.h30.metrics.comparable_excess). Stocks that "
              f"swung or fell alike as a group cancel out. The 5 best {per}s are those with the highest raw excess.", "",
              f"| Scores | Excess over comparable stocks per {per} (t) | {Per}s ahead | Without the 5 best {per}s (t) |",
              "|---|---|---|---|"]
    for name, r in results.items():
        o = r["overall"]
        lines.append(f"| {name} | {fmt(o.get('comparable_excess'), 2, pct=True, sign=True)} ({fmt(o.get('comparable_t'), 2)}) | "
                     f"{o.get('comparable_ahead', 0)} of {o.get('comparable_days', 0)} | "
                     f"{fmt(o.get('comparable_without_top5'), 2, pct=True, sign=True)} ({fmt(o.get('comparable_without_top5_t'), 2)}) |")
    lines.append("")
    if any("delist_bounds" in r for r in results.values()):
        lines += ["## Delisted stocks: the rule and its bounds", "",
                  f"A stock delisted inside its {per}: under main (the tables above) a buyout counts at its last price "
                  "and a failure or unknown delisting as a total loss; last-price and total-loss are the bounds.", ""]
        if info.get("delisted"):
            dl = info["delisted"]
            lines += [f"Universe stock-{per}s delisted inside the {per}: {dl['universe']} (buyout {dl['universe_kind'][1]}, "
                      f"failure {dl['universe_kind'][2]}, unknown {dl['universe_kind'][3]}).", ""]
        lines += [f"| Scores | Rule | Picks' {per} return vs all (t) | Excess over comparable stocks (t) | Hits vs "
                  f"same-volatility chance, {per}-level t | Pick weight delisted inside the {per} |", "|---|---|---|---|---|---|"]
        for name, r in results.items():
            for rule, o in [("main", r["overall"])] + list(r.get("delist_bounds", {}).items()):
                lines.append(f"| {name} | {rule} | {fmt(o['mean_pick_ret'], 2, pct=True, sign=True)} vs "
                             f"{fmt(o['mean_all_ret'], 2, pct=True, sign=True)} ({fmt(o['excess_t'], 1)}) | "
                             f"{fmt(o.get('comparable_excess'), 2, pct=True, sign=True)} ({fmt(o.get('comparable_t'), 2)}) | "
                             f"{fmt(o['hits_t_month'], 2, sign=True)} | {fmt(r.get('delisted_pick_share'), 2, pct=True)} |")
        lines.append("")
    costs = [c for c in COSTS if all(f"{c}bps" in (r.get("account") or {}) for r in results.values())]
    lines += [f"## A simple {adj} account in the picks (equal weight, from one graded {per} to the next)", "",
              "| Scores | $10,000 at 0 bps | " + " | ".join(f"at {c} bps" for c in costs[1:]) +
              " | Every eligible stock, equal weight (no costs) |", "|---" * (len(costs) + 2) + "|"]
    for name, r in results.items():
        acc = r.get("account")
        if acc:
            lines.append(f"| {name} | " + " | ".join(f"${acc[f'{c}bps']['end']:,.0f}" for c in costs) +
                         f" | ${acc['0bps']['all_end']:,.0f} |")
    lines.append("")
    if judges:
        jc = [c for c in COSTS if all(f"mean_{c}" in j for j in judges.values())]
        cs = " / ".join(str(c) for c in jc)
        lines += [f"## The $10,000 account: top 10 positive scores, rebalanced every {per} (the grader)", "",
                  f"| Scores | Random match | Years beating random, p < 0.1 ({cs} bps) | Years worse than random, "
                  f"p > 0.9 (0 bps) | Mean yearly return ({cs} bps) | Random median (0 / 20 bps) | $10,000 after "
                  "14 years at 20 bps | SPY: mean year, $10,000 after 14 years | Rank IC z (holding-period returns) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, j in judges.items():
            lines.append(f"| {name} | {j['match']} | " + " / ".join(str(j[f'beat_{c}']) for c in jc) + f" | {j['worse_0']} | "
                         + " / ".join(fmt(j[f'mean_{c}'], 1, pct=True, sign=True) for c in jc) +
                         f" | {fmt(j['rand_0'], 1, pct=True, sign=True)} / "
                         f"{fmt(j['rand_20'], 1, pct=True, sign=True)} | ${j['end_20']:,.0f} | "
                         f"{fmt(j['spy_mean'], 1, pct=True, sign=True)}, ${j['spy_end']:,.0f} | {fmt(j['ic_ret_z'], 2)} |")
        lines.append("")
        pairs = [n for n in results if f"{n}_null" in results]
        if pairs:
            lines += ["## The checks fixed before the Sharadar run", "",
                      "Written down in test-results-30d.md (Next steps) and docs/sharadar.md before any Sharadar result: "
                      "under the main delisting rule the picks beat comparable stocks with t >= 2, the sign-flip null "
                      "doesn't, and the grader's account beats SPY at 20 bps. "
                      + ("This is a Sharadar run: these are the checks." if info.get("source", "").startswith("Sharadar")
                         else "On Stooq data they are shown for reference only."), "",
                      "| Scores | Comparable-stock t >= 2 | Its null's comparable-stock t < 2 | Account at 20 bps vs SPY | All three |",
                      "|---|---|---|---|---|"]
            for n in pairs:
                t1, t0 = results[n]["overall"].get("comparable_t"), results[f"{n}_null"]["overall"].get("comparable_t")
                end, how = account_end(judges, n)
                spy = next(iter(judges.values()))["spy_end"]
                c1 = t1 is not None and math.isfinite(t1) and t1 >= 2
                c2 = t0 is not None and math.isfinite(t0) and t0 < 2
                c3 = end is not None and end > spy
                lines.append(f"| {n} | {fmt(t1, 2)}: {'yes' if c1 else 'no'} | {fmt(t0, 2)}: {'yes' if c2 else 'no'} | "
                             f"{'$' + format(end, ',.0f') if end is not None else 'n/a'} ({how}) vs ${spy:,.0f}: "
                             f"{'yes' if c3 else 'no'} | {'yes' if c1 and c2 and c3 else 'no'} |")
            lines.append("")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    return text


def delisted_counts(dev, years: list, h: int) -> dict | None:
    """Universe stock-periods on graded days whose stock was delisted inside the window, by kind."""
    if dev.delist_day is None:
        return None
    dd, dk = dev.delist_day, dev.delist_kind
    by_kind = {1: 0, 2: 0, 3: 0}
    for ys in years:
        last = dd[ys.universe]
        for d in ys.decisions:
            hit = (last >= d) & (last < d + h)
            for k in by_kind:
                by_kind[k] += int((hit & (dk[ys.universe] == k)).sum())
    return {"universe": sum(by_kind.values()), "universe_kind": by_kind}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.h30.evaluate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=None)
    ap.add_argument("--scores", nargs="+", required=True, help="NAME=PATH of [N, T] scores files")
    ap.add_argument("--judge", nargs="*", default=[], help="NAME=PATH of the grader's walk-forward JSON files")
    ap.add_argument("--years", default=None)
    ap.add_argument("--horizon", type=int, default=H)
    ap.add_argument("--subset", choices=SUBSETS, default=None, help="restrict every day to a subset known at d")
    ap.add_argument("--price-test", choices=("traded", "adjusted"), default=None,
                    help="closes for the universe's $5 test (default: traded when the export has them, else adjusted)")
    ap.add_argument("--delist", choices=DELIST_RULES, default="main",
                    help="outcome of a stock delisted inside its window (the other two rules are reported as bounds)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    dev = load_dev(a.export)
    price_test = a.price_test or dev.default_price_test()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    close = np.asarray(dev.close, dtype=np.float32)
    rule_fwd = lambda rule: forward_returns(close, a.horizon, delist_day=dev.delist_day,  # noqa: E731
                                            delist_kind=dev.delist_kind, rule=rule)
    fwd = rule_fwd(a.delist)
    vol = trailing_vol(close)
    ratio = past_year_ratio(close)
    v = np.asarray(dev.v, dtype=np.float32)
    dvol = np.asarray(dev.dollar_volume, dtype=np.float32)
    price_close = np.asarray(dev.price_close(price_test), dtype=np.float32) if price_test == "traded" else None
    years = non_overlapping([year_spec(dev.days, dev.cutoff, v, close, dvol, y, a.horizon, price_close)
                             for y in parse_years(a.years)], a.horizon)
    subset = subset_mask(a.subset, close, dvol) if a.subset else None
    bounds = {r: rule_fwd(r) for r in DELIST_RULES if r != a.delist} if dev.delist_day is not None else {}
    results, accounts = {}, {}
    for spec in a.scores:
        name, path = spec.split("=", 1)
        s = load_scores(path, dev.cutoff)
        results[name] = grade(s, fwd, vol, years, subset=subset, ratio=ratio)
        days = [r for ys in years for d in ys.decisions
                if (r := grade_day(s, fwd, vol, ys.universe, int(d), TOP, subset)) is not None]
        accounts[name] = {f"{c}bps": account(days, c) for c in COSTS}
        results[name]["account"] = accounts[name]
        if bounds:
            results[name]["delist_bounds"] = {r: grade(s, f, vol, years, subset=subset, ratio=ratio)["overall"]
                                              for r, f in bounds.items()}
            dd = dev.delist_day
            w_del = sum(w for day in results[name]["days"] for j, w in zip(day["picks"], day["weights"])
                        if day["day"] <= dd[j] < day["day"] + a.horizon)
            results[name]["delisted_pick_share"] = w_del / max(1, results[name]["overall"]["days"] * TOP)
        o = results[name]["overall"]
        print(f"{name}: {o['days']} days, {o['hits_gain_per_day']:.3f} hits a day against {o['exp_uniform_per_day']:.3f} "
              f"(uniform) and {o['exp_vol_per_day']:.3f} (same volatility); z {o['z_vs_uniform']:+.2f} / "
              f"{o['z_vs_vol']:+.2f}; IC {o['ic_mean']:+.4f} (t {o['ic_t']:.1f}); comparable-stock excess "
              f"{100 * o['comparable_excess']:+.2f} (t {o['comparable_t']:.2f})", flush=True)
    # How much of each score's edge comes from stocks that had collapsed (down more than 50% over the past year at d):
    # pooled over all picks, and period by period (each period's collapsed picks against its other picks).
    fell = ratio - 1.0
    for name, r in results.items():
        w_c = r_c = w_o = r_o = 0.0
        monthly = []
        for day in r["days"]:
            d = day["day"]
            mc = mo = wc_d = wo_d = 0.0
            for j, w in zip(day["picks"], day["weights"]):
                ret = fwd[j, d]
                if fell[j, d] < -0.5:
                    w_c += w; r_c += w * ret; wc_d += w; mc += w * ret
                else:
                    w_o += w; r_o += w * ret; wo_d += w; mo += w * ret
            if wc_d > 0:
                monthly.append(mc / wc_d)
        r["collapse_split"] = {"weight_share": w_c / (w_c + w_o) if w_c + w_o else float("nan"),
                               "collapsed_ret": r_c / w_c if w_c else float("nan"), "other_ret": r_o / w_o if w_o else float("nan"),
                               "collapsed_ret_month_avg": float(np.mean(monthly)) if monthly else float("nan"),
                               "months_with_collapsed": len(monthly)}
    judges = {}
    for spec in a.judge:
        name, path = spec.split("=", 1)
        judges[name] = judge_summary(path)
    src_meta = dev.export.meta.get("source")  # Sharadar exports copy the importer's source.json here
    if isinstance(src_meta, dict):
        src = (f"Sharadar ({src_meta.get('importer', 'import-sharadar')}, prices {src_meta.get('first_date')} .. "
               f"{src_meta.get('last_date')}, updated {src_meta.get('lastupdated_max')})")
    else:
        src = src_meta or ("Sharadar" if dev.delist_day is not None else "Stooq")
    info = {"price_test": price_test, "delist": a.delist if dev.delist_day is not None else None,
            "delisted": delisted_counts(dev, years, a.horizon), "source": src}
    if dev.delist_day is not None or price_test == "traded":
        info["source_note"] = (f"Data: {src} export {dev.export.path}. $5 test on {price_test} closes; delisting rule "
                               f"{info['delist'] or 'none (no delisting data)'}.")
    meta = {"horizon": a.horizon, "subset": a.subset, **{k: v for k, v in info.items() if k != "source_note"},
            "years": [{"year": y.year, "h0": data_day(dev, y.h0), "universe": int(y.universe.size),
                       "graded": [data_day(dev, int(d)) for d in y.decisions],
                       "dropped": [data_day(dev, int(d)) for d in y.dropped]} for y in years],
            "seconds": time.time() - t0}
    (out / "grades.json").write_text(json.dumps({"meta": meta, "scores": results, "judge": judges}))
    print(write_summary(results, judges, years, out, a.horizon, dev.cutoff, a.subset, info))
    return 0


def data_day(dev, idx: int) -> str:
    return data.day_to_str(int(dev.export.days[idx]))


if __name__ == "__main__":
    raise SystemExit(main())
