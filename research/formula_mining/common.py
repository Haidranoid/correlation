"""Formula mining: shared constants and helpers. Rules: experiments/-2.4-formula-mining.md (Rules, and
"Decisions made while building it").

Panels are stored [T, N] float32 (day-major) under BIG/feat_<tag>/<name>.npy, so a decision day's cross-section is
one contiguous read. Every feature value at day d uses only data through the close of day d.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

BIG = Path.home() / '.local/share/correlation2/experiments/formula_mining'
HERE = Path(__file__).resolve().parent
SRC = {'old': (c2eval.EXPORT_OLD, c2eval.STORE_OLD), 'new': (c2eval.EXPORT_NEW, c2eval.STORE_NEW)}

# The 26 base features (the plan's "about 25").
FEATURES = ['ret1', 'ret5', 'ret21', 'ret63', 'ret126', 'ret252', 'mom12_1',
            'vol21', 'vol63',
            'vr5_60', 'vr21_252', 'dvol21',
            'hl5', 'hl21', 'clv5', 'gap5', 'intra5',
            'maxret21', 'skew63',
            'beta252', 'mcorr252',
            'twres5', 'twres21', 'twres63', 'twcorr',
            'dist_high252']
NF = len(FEATURES)
PAIRS = [(i, j) for i in range(NF) for j in range(i + 1, NF)]
PI = np.array([p[0] for p in PAIRS])
PJ = np.array([p[1] for p in PAIRS])
NAMES = FEATURES + [f'{FEATURES[i]}*{FEATURES[j]}' for i, j in PAIRS] + [f'{FEATURES[i]}-{FEATURES[j]}' for i, j in PAIRS]
NFORM = len(NAMES)  # 26 + 325 + 325 = 676 distinct formulas, each usable in either direction (1,352)

OPEN_FROM = '2008-01-01'                       # opens used only from 2008
UNIV_START = {'old': '2005-01-03', 'new': '2025-06-30'}
WEEK_START = {'old': '2005-01-03', 'new': '2025-06-30'}
TRAIN_END = '2014-12-31'                       # training: decision dates whose forward 5-day window ends by this day
HOLD_START = '2015-01-01'                      # holdout IC: decision dates from the first trading day of 2015
SCORE_START = {'old': '2015-06-28', 'new': '2025-06-30'}   # money test: scores NaN before this day
MIN_IC_STOCKS = 100                            # a week's IC counts if >= 100 stocks have both values
TOP = 20
MIN_COMPONENTS = 10                            # composite NaN if fewer than 10 of the 20 formulas have a value
SEED = 20261021


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def feat_dir(tag):
    d = BIG / f'feat_{tag}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def open_panel(tag, name, mode='r'):
    return np.load(feat_dir(tag) / f'{name}.npy', mmap_mode=mode)


# ------------------------------------------------------------------ rolling statistics on [n, T] float64 chunks

def roll_sum(x, w):
    """x finite [n, T]; out[:, d] = sum of x[:, d-w+1..d], NaN for d < w-1."""
    c = np.zeros((x.shape[0], x.shape[1] + 1))
    np.cumsum(x, 1, out=c[:, 1:])
    out = np.full(x.shape, np.nan)
    out[:, w - 1:] = c[:, w:] - c[:, :-w]
    return out


def roll_mean(x, w, minobs):
    f = np.isfinite(x)
    s = roll_sum(np.where(f, x, 0.0), w)
    n = roll_sum(f.astype(np.float64), w)
    with np.errstate(all='ignore'):
        out = s / n
    out[~(n >= minobs)] = np.nan
    return out


def roll_std(x, w, minobs):
    f = np.isfinite(x)
    x0 = np.where(f, x, 0.0)
    s, s2, n = roll_sum(x0, w), roll_sum(x0 * x0, w), roll_sum(f.astype(np.float64), w)
    with np.errstate(all='ignore'):
        var = (s2 - s * s / n) / (n - 1)
        out = np.sqrt(np.maximum(var, 0))
    out[~(n >= minobs)] = np.nan
    return out


def roll_skew(x, w, minobs):
    """Population skewness of the window's finite values."""
    f = np.isfinite(x)
    x0 = np.where(f, x, 0.0)
    n = roll_sum(f.astype(np.float64), w)
    s1, s2, s3 = roll_sum(x0, w), roll_sum(x0 ** 2, w), roll_sum(x0 ** 3, w)
    with np.errstate(all='ignore'):
        m1 = s1 / n
        c2 = s2 / n - m1 ** 2
        c3 = s3 / n - 3 * m1 * s2 / n + 2 * m1 ** 3
        out = c3 / c2 ** 1.5
    out[~(n >= minobs) | ~(c2 > 1e-12)] = np.nan
    return out


def roll_max(x, w, minobs):
    """Rolling max of finite values (sparse table: about log2(w) passes)."""
    f = np.isfinite(x)
    M = np.where(f, x, -np.inf)
    span = 1
    while span * 2 <= w:
        M[:, span:] = np.maximum(M[:, span:], M[:, :-span])
        span *= 2
    out = np.full(x.shape, np.nan)
    sh = w - span
    out[:, w - 1:] = np.maximum(M[:, w - 1:], M[:, w - 1 - sh:x.shape[1] - sh]) if sh > 0 else M[:, w - 1:]
    n = roll_sum(f.astype(np.float64), w)
    out[~(n >= minobs) | ~np.isfinite(out)] = np.nan
    return out


def roll_beta_corr(r, m, w, minobs):
    """Rolling beta and correlation of r [n, T] with m [T] over days where both are finite."""
    f = np.isfinite(r) & np.isfinite(m)[None, :]
    a = np.where(f, r, 0.0)
    b = np.where(f, np.nan_to_num(m)[None, :], 0.0)
    n = roll_sum(f.astype(np.float64), w)
    sa, sb = roll_sum(a, w), roll_sum(b, w)
    saa, sbb, sab = roll_sum(a * a, w), roll_sum(b * b, w), roll_sum(a * b, w)
    with np.errstate(all='ignore'):
        cov = sab / n - sa * sb / n ** 2
        va = saa / n - (sa / n) ** 2
        vb = sbb / n - (sb / n) ** 2
        beta = cov / vb
        corr = cov / np.sqrt(va * vb)
    bad = ~(n >= minobs) | ~(va > 0) | ~(vb > 0)
    beta[bad] = np.nan
    corr[bad] = np.nan
    return beta, corr


# ------------------------------------------------------------------ ranks and ICs

def pct_rank(x):
    """Average-tie percentile ranks (r + 0.5) / n of the finite entries of a 1-D array, in (0, 1); NaN elsewhere."""
    x = np.asarray(x, np.float64)
    out = np.full(x.shape, np.nan)
    m = np.isfinite(x)
    v = x[m]
    n = len(v)
    if n == 0:
        return out
    o = np.argsort(v, kind='stable')
    sv = v[o]
    first = np.concatenate([[True], sv[1:] != sv[:-1]])
    starts = np.nonzero(first)[0]
    counts = np.diff(np.append(starts, n))
    avg = starts + (counts - 1) / 2.0
    r = np.empty(n)
    r[o] = avg[np.cumsum(first) - 1]
    out[m] = (r + 0.5) / n
    return out


def rank_features(P):
    """P [NF, n] raw feature values -> [NF, n] percentile ranks (NaN where missing)."""
    return np.stack([pct_rank(p) for p in P])


def formula_values(B):
    """B [NF, n] base ranks -> X [NFORM, n]: the 26 ranks, 325 rank products, 325 rank differences."""
    return np.concatenate([B, B[PI] * B[PJ], B[PI] - B[PJ]], 0)


def spearman_rows(X, y, min_n=MIN_IC_STOCKS):
    """Exact Spearman (ordinal ranks) of each row of X [F, n] with y [n], over entries where both are finite.
    Returns ic [F] (NaN if fewer than min_n entries)."""
    F, n = X.shape
    V = np.isfinite(X) & np.isfinite(y)[None, :]
    Xm = np.where(V, X, np.inf)                      # invalid sorted last
    ox = np.argsort(Xm, axis=1, kind='stable')
    rx = np.empty((F, n), np.float64)
    np.put_along_axis(rx, ox, np.broadcast_to(np.arange(n, dtype=np.float64), (F, n)), axis=1)
    oy = np.argsort(np.where(np.isfinite(y), y, np.inf), kind='stable')
    cr = np.cumsum(V[:, oy], axis=1, dtype=np.float64) - 1.0
    ry = np.empty((F, n), np.float64)
    ry[:, oy] = cr
    d = np.where(V, rx - ry, 0.0)
    nv = V.sum(1).astype(np.float64)
    with np.errstate(all='ignore'):
        ic = 1.0 - 6.0 * (d * d).sum(1) / (nv * (nv * nv - 1.0))
    ic[nv < min_n] = np.nan
    return ic


def composite(B, sel, sign):
    """Mean of the selected formulas' oriented, centered percentile ranks; a missing component counts as 0 (neutral);
    NaN where fewer than MIN_COMPONENTS components exist."""
    X = formula_values(B)[sel]
    R = np.stack([pct_rank(x) for x in X]) - 0.5
    R *= sign[:, None]
    have = np.isfinite(R).sum(0)
    c = np.nansum(R, 0) / len(sel)
    c[have < MIN_COMPONENTS] = np.nan
    return c


def tstat(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return float('nan'), float('nan'), len(a)
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), float(a.mean()), len(a)


def week_dates(days, start, step=5):
    """Fixed calendar of decision day indices from `start`, every `step` trading days, with d + 5 inside the export."""
    d0 = c2eval.day_index(days, start)
    return np.arange(d0, len(days) - 5, step)


def load_weekly(tag):
    z = np.load(BIG / f'weekly_{tag}.npz')
    return {k: z[k] for k in z.files}


def week_slices(W):
    o = W['offs']
    return [slice(o[k], o[k + 1]) for k in range(len(o) - 1)]
