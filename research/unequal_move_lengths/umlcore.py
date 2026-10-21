"""Core helpers for the unequal-move-lengths experiment (correlation2).

Census variable reproduction, moves, block sign-flip null, universes and correlation matrices.
See experiments/-1.9-unequal-move-lengths.md for the rules.
"""
import os
import sys
import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

LOOKBACK = 252
MIN_OBS = 126
CLIP = 5.0
OUT = c2eval.WORK / 'unequal_move_lengths'

# The 20 combinations fixed in the plan: (number, A's move length, B's move length).
COMBOS = [(1, 1, 5), (2, 1, 10), (3, 1, 21), (4, 1, 42), (5, 1, 63), (6, 5, 10), (7, 5, 21), (8, 5, 42), (9, 5, 63),
          (10, 5, 126), (11, 10, 21), (12, 10, 42), (13, 10, 63), (14, 21, 42), (15, 21, 63), (16, 21, 126),
          (17, 63, 126), (18, 21, 5), (19, 63, 21), (20, 63, 5)]


def offsets(b):
    """Offsets for B's move length b: B starts right after A ends, a one-week gap, a one-month gap."""
    return [b, b + 5, b + 21]


def trailing_sums(x, valid, w=LOOKBACK):
    """Sums of x over rows d-w..d-1 where valid (x is 0 where not valid), and counts. x: [N, T] float64."""
    xs = np.where(valid, x, 0.0)
    cs = np.zeros((x.shape[0], x.shape[1] + 1))
    np.cumsum(xs, axis=1, out=cs[:, 1:])
    cn = np.zeros((x.shape[0], x.shape[1] + 1))
    np.cumsum(valid, axis=1, out=cn[:, 1:])
    T = x.shape[1]
    d = np.arange(T)
    lo = np.maximum(d - w, 0)
    s = cs[:, d] - cs[:, lo]
    k = cn[:, d] - cn[:, lo]
    return s, k


def residuals(close, market):
    """Census market residual (crates/c2-census/src/variable.rs `residuals`): ret[d] - beta[d] * market[d], beta
    from the valid (ret, market) pairs in rows d-252..d-1 (at least 126). close [N, T] float32, market [T]."""
    c = np.asarray(close, dtype=np.float32)
    ret = np.full(c.shape, np.nan, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        ret[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    r = ret.astype(np.float64)
    m = np.broadcast_to(market[None, :], r.shape)
    valid = np.isfinite(r) & np.isfinite(m)
    rr = np.where(valid, r, 0.0)
    mm = np.where(valid, m, 0.0)
    sr, k = trailing_sums(rr, valid)
    sm, _ = trailing_sums(mm, valid)
    srm, _ = trailing_sums(rr * mm, valid)
    smm, _ = trailing_sums(mm * mm, valid)
    with np.errstate(divide='ignore', invalid='ignore'):
        var_m = smm - sm * sm / k
        beta = (srm - sr * sm / k) / var_m
        resid = r - beta * m
    ok = (k >= MIN_OBS) & valid & (var_m > 0)
    return np.where(ok, resid, np.nan)


def standardize(resid):
    """Census variable from residuals (variable.rs `standardize`): (resid[d] - mean) / sd over rows d-252..d-1
    (at least 126 finite), clipped at +-5. resid [N, T] float64 with NaN where unavailable."""
    valid = np.isfinite(resid)
    x = np.where(valid, resid, 0.0)
    s, k = trailing_sums(x, valid)
    ss, _ = trailing_sums(x * x, valid)
    with np.errstate(divide='ignore', invalid='ignore'):
        var = (ss - s * s / k) / k
        v = (resid - s / k) / np.sqrt(var)
    ok = (k >= MIN_OBS) & valid & (var > 1e-12)
    return np.where(ok, np.clip(v, -CLIP, CLIP), np.nan)


def c_ab(a, b):
    """Overlap factor: sum over lags of the two overlapping-move autocorrelations (1-|l|/a)(1-|l|/b) for iid daily
    values. Effective observations = n / c_ab."""
    m = min(a, b)
    l = np.arange(-(m - 1), m)
    return float(np.sum((1 - np.abs(l) / a) * (1 - np.abs(l) / b)))


def block_signs(T, L, rng):
    """Signs for T days in blocks of L days with a random phase; block signs balanced (half +, half -, shuffled)."""
    ph = int(rng.integers(L))
    nb = (T + ph) // L + 1
    s = np.ones(nb)
    s[rng.permutation(nb)[: nb // 2]] = -1.0
    return s[(np.arange(T) + ph) // L]
