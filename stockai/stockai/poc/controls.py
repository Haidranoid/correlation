"""Controls for a scores file (PLAN.md §4.3): two score canaries for the judge, and the large-cap leakage tripwire.

    python -m stockai.poc.controls shift --scores IN.npy --out OUT.npy [--days 20] [--years 2010-2023]
    python -m stockai.poc.controls permute --scores IN.npy --out OUT.npy [--seed 0] [--years 2010-2023]
    python -m stockai.poc.controls tripwire --scores IN.npy [--top 500] [--years 2010-2023] [--json OUT.json]
    python -m stockai.poc.controls lowvol --out OUT.npy [--years 2010-2023]
    python -m stockai.poc.controls percentile --scores IN.npy --out OUT.npy

- shift: stale scores. out[:, d] = in[:, d - days] for every test decision day d of the chosen years: the score
  decided at d is used `days` trading days later. Still point in time, but a next-day signal should be gone after
  20 days, so the judge's rank IC should fall to about 0.
- permute: within each chosen year, the test decision days' score columns are shuffled across those days (seeded).
  Each day is ranked with another day's scores, so the rank IC should fall to about 0.
  Both canaries keep only the chosen years' test decision columns (NaN elsewhere), so they grade the same years.
- tripwire: on each test decision day d, the `top` stocks with the largest trailing dollar volume (mean over days
  d-62 .. d, at least 32 present: point in time) among those with a finite score; the mean daily Spearman rank IC
  of the score against the next day's actual return close[d+1] / close[d] - 1, over those of them with a next-day
  return. A daily rank IC above 0.05 on large caps is treated as a bug until proven otherwise (PLAN.md §4.3).
- lowvol: a static, point-in-time reference for reading the canaries: score = minus the log sd of the stock's daily
  log returns over the 252 days through d (stockai.gpt.features.trailing_log_vol), on the chosen years' test
  decision days. Permuting dates within a year keeps each stock's slowly varying tilt (stockGPT leans away from
  volatile stocks), so whatever IC this tilt alone earns is what the permuted canary can keep without any leak.
- percentile: not a canary but a grading variant. Each day's finite scores become their percentile rank among that
  day's scored stocks, in (0, 1] (ties keep equal ranks). The order is unchanged, so the judge's rank ICs and
  long-short check come out identical; only the account changes. The judge buys only positive scores, and an
  expected-value score can be negative for every stock (stockGPT's level drifts by seed and year), which leaves
  the account in cash; with percentiles it always holds the 20 highest-ranked candidates. Same-day data only.

Nothing reads the locked year (2024-06-28 on): outputs are NaN from its first decision day (dev_cutoff), inputs
must have no finite score there, and next-day returns use closes before locked_cut (2024-06-28) only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .. import data, splits
from ..baselines.common import dev_cutoff, parse_years
from ..gpt.features import locked_cut

ADV_WINDOW = 63
ADV_MIN_OBS = 32


def load_scores(path: str | Path, export: data.Export) -> np.ndarray:
    s = np.load(path)
    if s.shape != (export.n, export.t):
        raise ValueError(f"{path}: shape {s.shape}, expected {(export.n, export.t)}")
    if s.dtype not in (np.float32, np.float64):
        raise ValueError(f"{path}: dtype {s.dtype}, expected float32 or float64")
    cut = dev_cutoff(export.days)
    if np.isfinite(s[:, cut:]).any():
        raise ValueError(f"{path}: finite scores on or after {export.date(cut)}, the locked year's first decision")
    return s


def test_columns(days: np.ndarray, years: list[int]) -> list[range]:
    return [splits.split(days, y).test_decisions for y in years]


def shift_scores(scores: np.ndarray, days: np.ndarray, years: list[int], lag: int) -> np.ndarray:
    """out[:, d] = scores[:, d - lag] for d in the chosen years' test decisions; NaN elsewhere."""
    if lag < 1:
        raise ValueError("the shift must be at least 1 day later")
    out = np.full(scores.shape, np.nan, dtype=scores.dtype)
    for r in test_columns(days, years):
        src = np.arange(r.start, r.stop) - lag
        ok = src >= 0
        out[:, np.arange(r.start, r.stop)[ok]] = scores[:, src[ok]]
    out[:, dev_cutoff(days):] = np.nan
    return out


def permute_scores(scores: np.ndarray, days: np.ndarray, years: list[int], seed: int) -> np.ndarray:
    """Within each chosen year, the test decision columns shuffled across that year's test decision days."""
    out = np.full(scores.shape, np.nan, dtype=scores.dtype)
    for y, r in zip(years, test_columns(days, years)):
        cols = np.arange(r.start, r.stop)
        perm = np.random.default_rng([int(seed), int(y)]).permutation(cols)
        out[:, cols] = scores[:, perm]
    out[:, dev_cutoff(days):] = np.nan
    return out


def lowvol_scores(export: data.Export, years: list[int]) -> np.ndarray:
    """float32 [N, T]: -log trailing 252-day volatility on the chosen years' test decision days, NaN elsewhere."""
    from ..gpt.features import trailing_log_vol

    cut = dev_cutoff(export.days)
    lv = trailing_log_vol(np.asarray(export.close[:, :cut]))
    out = np.full((export.n, export.t), np.nan, dtype=np.float32)
    for r in test_columns(export.days, years):
        out[:, r.start:r.stop] = -lv[:, r.start:r.stop]
    return out


def percentile_scores(scores: np.ndarray, days: np.ndarray) -> np.ndarray:
    """float32 [N, T]: each day's finite scores as percentile ranks in (0, 1] among that day's scored stocks."""
    out = np.full(scores.shape, np.nan, dtype=np.float32)
    cut = dev_cutoff(days)
    fin = np.isfinite(scores[:, :cut])
    for d in np.flatnonzero(fin.any(axis=0)):
        m = fin[:, d]
        r = rankdata(np.asarray(scores[m, d], dtype=np.float64), method="average")
        out[m, d] = (r / m.sum()).astype(np.float32)
    return out


def trailing_adv(dollar_volume: np.ndarray, window: int = ADV_WINDOW, min_obs: int = ADV_MIN_OBS) -> np.ndarray:
    """Mean dollar volume over days d-window+1 .. d (present values only), float64 [N, T]; NaN with < min_obs."""
    dv = np.asarray(dollar_volume, dtype=np.float64)
    ok = np.isfinite(dv)
    x = np.where(ok, dv, 0.0)
    n, t = x.shape
    cs = np.zeros((n, t + 1))
    np.cumsum(x, axis=1, out=cs[:, 1:])
    cc = np.zeros((n, t + 1))
    np.cumsum(ok, axis=1, out=cc[:, 1:])
    lo = np.maximum(np.arange(1, t + 1) - window, 0)
    s = cs[:, 1:] - cs[:, lo]
    k = cc[:, 1:] - cc[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / k
    out[k < min_obs] = np.nan
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def summary(ics: list[float]) -> dict:
    x = np.asarray([v for v in ics if math.isfinite(v)])
    if x.size < 2:
        return {"mean": None, "t": None, "days": int(x.size)}
    m, sd = float(x.mean()), float(x.std(ddof=1))
    return {"mean": m, "t": m / sd * math.sqrt(x.size) if sd > 0 else None, "days": int(x.size)}


def tripwire(scores: np.ndarray, export: data.Export, years: list[int], top: int = 500, threshold: float = 0.05,
             min_stocks: int = 50) -> dict:
    """Mean daily rank IC of the scores against next-day actual returns on the `top` largest stocks by trailing
    dollar volume, by year and over all the chosen years."""
    lc = locked_cut(export.days)
    close = np.asarray(export.close[:, :lc])
    ret = data.next_day_returns(close)  # [N, lc]; the last column (2024-06-27) is NaN
    adv = trailing_adv(np.asarray(export.dollar_volume[:, :lc]))
    by_year, all_ics = [], []
    for y, r in zip(years, test_columns(export.days, years)):
        ics = []
        for d in range(r.start, r.stop):
            if d + 1 >= lc:
                raise AssertionError("a tripwire outcome would come from the locked year")
            s, o, a = scores[:, d].astype(np.float64), ret[:, d], adv[:, d]
            # The large-cap set is chosen from what is known at d (a score and trailing dollar volume); stocks whose
            # next-day return is missing are then left out of that day's IC.
            known = np.flatnonzero(np.isfinite(s) & np.isfinite(a))
            big = known[np.argsort(-a[known], kind="stable")[:top]]
            big = big[np.isfinite(o[big])]
            if big.size < min_stocks:
                continue
            ics.append(spearman(s[big], o[big]))
        all_ics += ics
        by_year.append({"year": y, **summary(ics)})
    overall = summary(all_ics)
    worst = max((b["mean"] for b in by_year if b["mean"] is not None), default=None)
    flagged = [b["year"] for b in by_year if b["mean"] is not None and b["mean"] > threshold]
    if overall["mean"] is not None and overall["mean"] > threshold:
        flagged.append("overall")
    return {"top": top, "threshold": threshold, "adv_window": ADV_WINDOW, "overall": overall, "max_year_mean": worst,
            "flagged": flagged, "tripped": bool(flagged), "years": by_year}


def _write(path: str, scores: np.ndarray, export: data.Export, what: str) -> None:
    cut = dev_cutoff(export.days)
    if np.isfinite(scores[:, cut:]).any():
        raise AssertionError("a canary score landed on or after the locked year's first decision")
    p = data.write_scores(path, scores, export)
    cols = np.flatnonzero(np.isfinite(scores).any(axis=0))
    span = f"{export.date(cols[0])}..{export.date(cols[-1])}" if cols.size else "none"
    print(f"{what}: {p} ({int(np.isfinite(scores).sum()):,} finite scores, decision days {span})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.poc.controls", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("shift", "permute", "tripwire", "lowvol", "percentile"):
        p = sub.add_parser(name)
        p.add_argument("--export", default=None, help="c2 export directory (default: $C2_DATA_DIR/export/full)")
        if name != "lowvol":
            p.add_argument("--scores", required=True)
        p.add_argument("--years", default="2010-2023", help="test years (default 2010-2023)")
        if name == "shift":
            p.add_argument("--days", type=int, default=20, help="trading days later (default 20)")
        if name == "permute":
            p.add_argument("--seed", type=int, default=0)
        if name in ("shift", "permute", "lowvol", "percentile"):
            p.add_argument("--out", required=True)
        if name == "tripwire":
            p.add_argument("--top", type=int, default=500)
            p.add_argument("--threshold", type=float, default=0.05)
            p.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    export = data.load(a.export)
    years = parse_years(a.years)
    if a.cmd == "lowvol":
        _write(a.out, lowvol_scores(export, years), export, "-log trailing volatility (static reference)")
        return 0
    scores = load_scores(a.scores, export)
    if a.cmd == "shift":
        _write(a.out, shift_scores(scores, export.days, years, a.days), export, f"shifted {a.days} days later")
    elif a.cmd == "percentile":
        _write(a.out, percentile_scores(scores, export.days), export, "percentile ranks within each day")
    elif a.cmd == "permute":
        _write(a.out, permute_scores(scores, export.days, years, a.seed), export, f"permuted within years (seed {a.seed})")
    else:
        res = tripwire(scores, export, years, a.top, a.threshold)
        res["scores"] = str(a.scores)
        o = res["overall"]
        print(f"tripwire {a.scores}: top {a.top} by {ADV_WINDOW}-day dollar volume, mean daily rank IC vs next-day "
              f"return {o['mean']:+.4f} (t {o['t']:.2f}, {o['days']} days); largest year mean {res['max_year_mean']:+.4f}; "
              + (f"TRIPPED (> {a.threshold}): {res['flagged']}" if res["tripped"] else f"not tripped (<= {a.threshold})"))
        for b in res["years"]:
            print(f"  {b['year']}: {b['mean']:+.4f} (t {b['t']:.2f}, {b['days']} days)" if b["mean"] is not None
                  else f"  {b['year']}: no days")
        if a.json:
            Path(a.json).parent.mkdir(parents=True, exist_ok=True)
            Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
