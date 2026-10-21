"""Breaks exact ties in a scores file at random, for the grader (which otherwise breaks them by column, ticker order).

Multiplies every finite score by 1 + 1e-9 * u (u a seeded uniform draw; exact zeros get + 1e-300 * u instead), in
float64. Distinct float32 scores differ by at least about 6e-8 of their size, so only exactly equal scores change
order (checked column by column before writing). Grading several seeds shows how much a result owes to the order of
tied names.

    uv run python -m stockai.h30.jitter --scores IN.npy --out OUT.npy --seed K
"""

from __future__ import annotations

import argparse

import numpy as np

from .. import data

REL = 1e-9


def jitter(scores: np.ndarray, seed: int) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    u = np.random.default_rng(np.random.SeedSequence([int(seed), 0x71E5])).random(s.shape)
    out = np.where(s == 0, 1e-300 * u, s * (1 + REL * u))
    out[~np.isfinite(s)] = np.nan
    return out


def keeps_strict_order(s32: np.ndarray, j: np.ndarray) -> bool:
    """In every column, scores that were strictly ordered stay in that order."""
    for d in np.flatnonzero(np.isfinite(s32).any(axis=0)):
        fin = np.flatnonzero(np.isfinite(s32[:, d]))
        order = np.lexsort((j[fin, d], s32[fin, d]))
        a, b = s32[fin[order], d], j[fin[order], d]
        if np.any((np.diff(a) > 0) & (np.diff(b) <= 0)):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.h30.jitter", description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=None)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    ex = data.load(a.export)
    s = np.load(a.scores, mmap_mode="r")
    j = jitter(s, a.seed)
    f32 = np.asarray(s, dtype=np.float32)
    if not (np.array_equal(np.isfinite(j), np.isfinite(f32)) and keeps_strict_order(f32, j)):
        raise AssertionError("the jitter changed more than the order of tied scores")
    data.write_scores(a.out, j, ex)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
