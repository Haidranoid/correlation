"""What a scores file resembles, from inputs only (no outcomes): its mean daily cross-sectional rank correlation with
today's census move v[d], with the stock's trailing log volatility, and with other scores files.

    python -m stockai.poc.diagnostics --scores A.npy [--vs NAME=B.npy ...] [--years 2010-2023] [--json OUT.json]

Each test decision day of the chosen years, over the stocks where both sides are finite: Spearman correlation.
Reported as the mean over days, by year and overall. Nothing at or after the locked year's first decision is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .. import data
from ..baselines.common import dev_cutoff, parse_years
from ..gpt.features import trailing_log_vol
from .controls import load_scores, test_columns

MIN_STOCKS = 50


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def daily_rank_corr(x: np.ndarray, y: np.ndarray, cols: range) -> list[float]:
    out = []
    for d in cols:
        a, b = x[:, d].astype(np.float64), y[:, d].astype(np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= MIN_STOCKS:
            out.append(_spearman(a[ok], b[ok]))
    return out


def describe(scores: np.ndarray, others: dict[str, np.ndarray], export: data.Export, years: list[int]) -> dict:
    cut = dev_cutoff(export.days)
    v = np.asarray(export.v[:, :cut])
    logvol = trailing_log_vol(np.asarray(export.close[:, :cut]))
    refs = {"v[d]": v, "log volatility (252 days)": logvol, **{k: o[:, :cut] for k, o in others.items()}}
    s = scores[:, :cut]
    res = {"years": [], "overall": {}}
    pooled: dict[str, list[float]] = {k: [] for k in refs}
    for y, r in zip(years, test_columns(export.days, years)):
        row = {"year": y}
        for k, ref in refs.items():
            c = daily_rank_corr(s, ref, r)
            pooled[k] += c
            row[k] = float(np.mean(c)) if c else None
        res["years"].append(row)
    res["overall"] = {k: (float(np.mean(c)) if c else None) for k, c in pooled.items()}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.poc.diagnostics", description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=None)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--vs", nargs="*", default=[], help="NAME=FILE.npy: other scores to correlate with")
    ap.add_argument("--years", default="2010-2023")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    export = data.load(a.export)
    years = parse_years(a.years)
    scores = load_scores(a.scores, export)
    others = {}
    for spec in a.vs:
        name, _, path = spec.partition("=")
        others[name] = load_scores(path, export)
    res = describe(scores, others, export, years)
    res["scores"] = a.scores
    print(f"{a.scores}: mean daily rank correlation over {years[0]}-{years[-1]} test days")
    for k, val in res["overall"].items():
        print(f"  with {k}: {val:+.3f}" if val is not None else f"  with {k}: n/a")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
