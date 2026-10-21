"""Tokens for stockGPT (PLAN.md §3.2): one token per stock per day from correlation2's census variable.

v (market-residual move in units of the stock's trailing volatility, clipped at +-5) becomes
bin = clip(round(v / 0.1), -50, 50) + 50, so bins 0..100 have centers -5.0 .. 5.0 in steps of 0.1.
NaN becomes the MISSING token 101. The model reads all 102 tokens and predicts only the 101 bins. At a horizon above
one day the training target is a daily rank-gauss score binned the same way (features.py).
"""

from __future__ import annotations

import numpy as np

BIN_WIDTH = 0.1
HALF = 50
N_BINS = 2 * HALF + 1  # 101 value bins
MISSING = N_BINS  # 101
VOCAB = N_BINS + 1  # 102
CENTERS = ((np.arange(N_BINS) - HALF) * BIN_WIDTH).astype(np.float32)  # -5.0 .. 5.0


def encode(v: np.ndarray) -> np.ndarray:
    """uint8 tokens for an array of census values; NaN (or +-inf) becomes MISSING."""
    x = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(x)
    b = np.clip(np.round(np.where(ok, x, 0.0) / BIN_WIDTH), -HALF, HALF).astype(np.int64) + HALF
    return np.where(ok, b, MISSING).astype(np.uint8)


def decode(tokens: np.ndarray) -> np.ndarray:
    """Bin centers for tokens; MISSING becomes NaN."""
    t = np.asarray(tokens).astype(np.int64)
    if np.any((t < 0) | (t > MISSING)):
        raise ValueError("token out of range")
    out = np.full(t.shape, np.nan, dtype=np.float32)
    ok = t != MISSING
    out[ok] = CENTERS[t[ok]]
    return out
