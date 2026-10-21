"""Simple rules stockai-30d's model must beat, written as scores files on the development columns.

- momentum: the return from 252 to 21 trading days ago (12-1 momentum): recent winners keep winning.
- reversal: minus the return over the last 21 days: last month's losers bounce back.
- high_vol / low_vol: plus or minus the trailing 63-day volatility. Not a forecast of direction: the month's biggest
  gainers (and losers) are mostly volatile stocks, so high_vol shows how far picking volatile names alone gets.
- oracle: a positive control that peeks. The day's rank of the actual forward return (rank-gauss) plus noise with sd 4,
  so its rank correlation with the outcome is about 0.24. It shows the evaluation finds skill when there is some; it
  is never a model.

Every score at day d uses closes through d only, except the oracle's, by design. Nothing is written at or after the
development cutoff.

    uv run python -m stockai.h30.rules --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import data
from ..baselines.common import full_width, load_dev
from ..baselines.lgbm import rank_gauss
from .metrics import trailing_vol
from .targets import H, forward_returns

ORACLE_NOISE = 4.0


def log_return(close: np.ndarray, a: int, b: int) -> np.ndarray:
    """float32 [N, C]: ln(close[d - b] / close[d - a]) for a > b >= 0; NaN where either close is missing."""
    c = np.asarray(close, dtype=np.float64)
    out = np.full(c.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        hi = c[:, a - b:c.shape[1] - b] if b else c[:, a:]
        out[:, a:] = np.log(hi / c[:, :c.shape[1] - a])
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def rules(close: np.ndarray, h: int = H, seed: int = 0, delist_day: np.ndarray | None = None,
          delist_kind: np.ndarray | None = None) -> dict[str, np.ndarray]:
    vol = trailing_vol(close).astype(np.float32)
    fwd = forward_returns(close, h, delist_day=delist_day, delist_kind=delist_kind)  # the oracle peeks at the graded outcome
    oracle = rank_gauss(fwd, np.isfinite(fwd))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x0AC1E]))
    oracle = (oracle + ORACLE_NOISE * rng.standard_normal(oracle.shape).astype(np.float32)).astype(np.float32)
    oracle[~np.isfinite(fwd)] = np.nan
    return {
        "momentum": log_return(close, 252, 21),
        "reversal": -log_return(close, h, 0),
        "high_vol": vol,
        "low_vol": -vol,
        "oracle": oracle,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.h30.rules", description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=H)
    a = ap.parse_args(argv)
    dev = load_dev(a.export)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, s in rules(dev.close, a.horizon, delist_day=dev.delist_day, delist_kind=dev.delist_kind).items():
        p = data.write_scores(out / f"{name}.npy", full_width(s, dev.export.t), dev.export)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
