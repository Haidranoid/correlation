"""Validation metrics: daily cross-sectional Spearman rank IC and cross-entropy against the bin histogram."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata

MIN_STOCKS = 10  # days with fewer scored stocks are skipped


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def daily_rank_ic(day: np.ndarray, score: np.ndarray, outcome: np.ndarray, min_stocks: int = MIN_STOCKS) -> np.ndarray:
    """Spearman correlation between score and outcome among the pairs of each day (finite values only)."""
    ok = np.isfinite(score) & np.isfinite(outcome)
    day, score, outcome = day[ok], score[ok], outcome[ok]
    order = np.argsort(day, kind="stable")
    day, score, outcome = day[order], score[order], outcome[order]
    bounds = np.flatnonzero(np.diff(day)) + 1
    out = []
    for s, o in zip(np.split(score, bounds), np.split(outcome, bounds)):
        out.append(spearman(s, o) if s.size >= min_stocks else float("nan"))
    return np.asarray(out, dtype=np.float64)


def ic_summary(ics: np.ndarray) -> dict:
    x = ics[np.isfinite(ics)]
    if x.size == 0:
        return {"mean": float("nan"), "t": float("nan"), "days": 0}
    sd = x.std(ddof=1) if x.size > 1 else float("nan")
    return {"mean": float(x.mean()), "t": float(x.mean() / (sd / math.sqrt(x.size))) if sd and sd > 0 else float("nan"),
            "days": int(x.size)}
