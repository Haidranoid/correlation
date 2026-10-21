"""One-day census-variable reversal: score at decision day d = -v[d].

The simplest score a next-day model of the census variable can rediscover: minus today's market-residual,
volatility-scaled move. stockGPT v1's scores correlate about -0.75 with v[d] (stockai/gpt), so this is the bar for
"more than one-day reversal". NaN where v[d] is missing. Point in time: v[d] uses data through the close of day d.
Columns from the locked year's first decision on are NaN (`common.dev_cutoff`).

    uv run python -m stockai.baselines.vreversal --export DIR --out FILE.npy
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .. import data
from .common import describe_scores, full_width, load_dev


def vreversal_scores(v: np.ndarray) -> np.ndarray:
    """out[j, d] = -v[j, d] as float32; NaN where v is missing."""
    return -np.asarray(v, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.baselines.vreversal", description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=None, help="c2 export directory (default: $C2_DATA_DIR/export/full)")
    ap.add_argument("--out", required=True, help="scores file to write (.npy, float32 [N, T])")
    a = ap.parse_args(argv)
    t0 = time.time()
    dev = load_dev(a.export)
    scores = full_width(vreversal_scores(dev.v), dev.export.t)
    path = data.write_scores(a.out, scores, dev.export)
    print(f"one-day census reversal (-v[d]): {path} ({dev.export.n} stocks x {dev.export.t} days, float32)")
    print(f"  {describe_scores(scores, dev.export.days)}  [{time.time() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
