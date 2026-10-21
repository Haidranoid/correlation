"""correlation2's reversal-family score, reproduced exactly: the G0 cross-check of the whole pipeline (PLAN.md M0).

correlation2 (crates/c2-census/src/holdout.rs, `reversal_scores`, commit 9335126) scores stock j at decision day d as

    -(ln(close[d] as f64 / close[d - lookback] as f64))

when both float32 closes are finite and close[d - lookback] > 0, and 0.0 otherwise or when d < lookback. This module
gives the same float64 for every stock and every decision day before the locked year, bit for bit:

- both float32 closes are widened to float64 before dividing (IEEE division, as in Rust);
- the logarithm is `math.log`, which calls the C library's `log`: the function Rust's `f64::ln` calls. numpy's
  `np.log` uses its own SIMD kernel on this CPU, which is not guaranteed to round the same way;
- fallbacks are +0.0, and an unchanged price gives -0.0 (Rust negates ln(1) = 0). The sign matters: the judge orders
  scores with `total_cmp`, which puts -0.0 below +0.0, in its rank IC and long-short check.

A zero close at day d gives +inf and a negative one NaN, as in Rust. Repaired closes should never be <= 0, but the
judge refuses a file with an infinite score, so the CLI writes NaN ("no score") there instead and names each such cell.
That only matters for G0 if the stock is a candidate on a rebalance day. Columns from the locked year's first decision
on are NaN (`common.dev_cutoff`): they are never computed.

    uv run python -m stockai.baselines.reversal --export DIR --lookback 5 --out FILE.npy
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from .. import data
from .common import describe_scores, full_width, load_dev


def libm_log(x: np.ndarray) -> np.ndarray:
    """Natural log of every element through the C library's `log` (what Rust's `f64::ln` calls), as float64.

    log(0) = -inf and log(x < 0) = NaN, as the C function returns (Python's math.log raises instead).
    """
    a = np.asarray(x, dtype=np.float64)
    flat = a.ravel()
    out = np.full(flat.shape, np.nan)
    pos = np.flatnonzero(flat > 0)
    for k in range(0, pos.size, 1 << 20):  # bounded memory for the Python floats
        idx = pos[k : k + (1 << 20)]
        out[idx] = np.fromiter(map(math.log, flat[idx].tolist()), dtype=np.float64, count=idx.size)
    out[flat == 0] = -np.inf
    return out.reshape(a.shape)


def reversal_scores(close: np.ndarray, lookback: int) -> np.ndarray:
    """correlation2's reversal score for every row (stock) and column (decision day) of float32 `close`: float64.

    out[j, d] = -(ln(close[j, d] / close[j, d - lookback])) in float64 where both closes are finite and
    close[j, d - lookback] > 0; +0.0 otherwise and for d < lookback.
    """
    c = np.asarray(close)
    if c.dtype != np.float32:
        raise TypeError(f"close must be float32 like the export (correlation2 widens f32 closes), got {c.dtype}")
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    n, t = c.shape
    out = np.zeros((n, t), dtype=np.float64)  # +0.0 fallbacks, including every d < lookback
    if lookback < t:
        p0 = c[:, : t - lookback].astype(np.float64)  # close[d - lookback] for d = lookback .. t-1
        p1 = c[:, lookback:].astype(np.float64)  # close[d]
        ok = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0)
        tail = out[:, lookback:]  # a view: writing it writes out
        tail[ok] = -libm_log(p1[ok] / p0[ok])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.baselines.reversal", description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=None, help="c2 export directory (default: $C2_DATA_DIR/export/full)")
    ap.add_argument("--lookback", type=int, default=5, help="return lookback in trading days (default 5)")
    ap.add_argument("--out", required=True, help="scores file to write (.npy, float64 [N, T])")
    a = ap.parse_args(argv)
    t0 = time.time()
    dev = load_dev(a.export)
    ex = dev.export
    part = reversal_scores(np.asarray(dev.close), a.lookback)
    odd_j, odd_d = np.nonzero(~np.isfinite(part))  # only from closes <= 0
    for j, d in list(zip(odd_j, odd_d))[:10]:
        print(f"  warning: {ex.tickers[j]} {ex.date(d)}: score {part[j, d]} (close {dev.close[j, d]} after "
              f"{dev.close[j, d - a.lookback]}); correlation2 scores this too, but the judge refuses infinite scores, "
              "so it is written as NaN (no score)", file=sys.stderr)
    part[odd_j, odd_d] = np.nan
    scores = full_width(part, ex.t)
    path = data.write_scores(a.out, scores, ex)
    bits = part.view(np.int64)
    neg_zero = int((bits == np.float64(-0.0).view(np.int64)).sum())
    pos_zero = int((bits == 0).sum())
    print(f"reversal, lookback {a.lookback}: {path} ({ex.n} stocks x {ex.t} days, float64)")
    print(f"  {describe_scores(scores, ex.days)}; NaN from decision day {ex.date(dev.cutoff)} on "
          "(the locked year's first decision)" if dev.cutoff < ex.t else f"  {describe_scores(scores, ex.days)}")
    print(f"  +0.0 (no score, as correlation2): {pos_zero:,}; -0.0 (unchanged price): {neg_zero:,}; "
          f"closes <= 0 written as NaN: {odd_j.size:,}  [{time.time() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
