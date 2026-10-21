"""12-1 momentum: score at decision day d = ln(close[d - 21] / close[d - 252]).

The stock's log return from about a year ago to about a month ago, skipping the last month (where short-term reversal
lives). NaN when either close is missing or not positive, and for d < 252. Uses closes through day d - 21 only, so it
is point in time. Columns from the locked year's first decision on are NaN (`common.dev_cutoff`).

    uv run python -m stockai.baselines.momentum --export DIR --out FILE.npy [--skip 21] [--lookback 252]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .. import data
from .common import describe_scores, full_width, load_dev


def momentum_scores(close: np.ndarray, skip: int = 21, lookback: int = 252) -> np.ndarray:
    """out[j, d] = ln(close[j, d - skip] / close[j, d - lookback]) as float64; NaN where either close is missing or <= 0."""
    if not 0 <= skip < lookback:
        raise ValueError("need 0 <= skip < lookback")
    c = np.asarray(close, dtype=np.float64)
    n, t = c.shape
    out = np.full((n, t), np.nan)
    if lookback < t:
        old = c[:, : t - lookback]  # close[d - lookback] for d = lookback .. t-1
        new = c[:, lookback - skip : t - skip]  # close[d - skip]
        ok = np.isfinite(old) & np.isfinite(new) & (old > 0) & (new > 0)
        tail = out[:, lookback:]
        tail[ok] = np.log(new[ok] / old[ok])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.baselines.momentum", description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=None, help="c2 export directory (default: $C2_DATA_DIR/export/full)")
    ap.add_argument("--skip", type=int, default=21, help="most recent days skipped (default 21)")
    ap.add_argument("--lookback", type=int, default=252, help="start of the return window, days back (default 252)")
    ap.add_argument("--out", required=True, help="scores file to write (.npy, float64 [N, T])")
    a = ap.parse_args(argv)
    t0 = time.time()
    dev = load_dev(a.export)
    scores = full_width(momentum_scores(dev.close, a.skip, a.lookback), dev.export.t)
    path = data.write_scores(a.out, scores, dev.export)
    print(f"momentum {a.lookback}-{a.skip}: {path} ({dev.export.n} stocks x {dev.export.t} days, float64)")
    print(f"  {describe_scores(scores, dev.export.days)}  [{time.time() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
