"""Shared settings and helpers for the group breadth-thrust experiment (experiments/+2.6-breadth-thrust.md).

The groups map's clustering below (standardize, market_relative, kmeans, spectral) is copied verbatim, read-only,
from research/phantom_portfolios/pp_common.py, the numpy port of crates/c2-census/src/together.rs (`groups`,
`standardized`, `spectral`) and cluster.rs (`kmeans`) that the phantom-portfolios experiment used. Only its
parameters change, to this plan's fixed rules: 30 groups of the 1,500 most-traded stocks (phantom portfolios used
together.rs's defaults, 20 groups of 1,000). cluster_ols is copied from the same file.

numpy only; system python3.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
BIG = Path.home() / '.local/share/correlation2/experiments/breadth_thrust'

# Fixed rules (experiments/+2.6-breadth-thrust.md, "Rules (fixed before looking at results)")
REFIT_EVERY = 21      # trading days between refits (the twins calendar: export day 252, 273, ...)
CORR_WIN = 252        # correlation period for the groups map
N_STOCKS = 1500       # most-traded stocks grouped
K_GROUPS = 30
COVER = 0.95          # a return on >= 95% of the correlation period's days (together.rs group_universe)
MIN_DV = 0.0          # decision: no dollar-volume floor; together.rs's $5M left ~1,070 stocks in 2008, the plan says 1,500
SMA = 50              # breadth: close above its own 50-day simple moving average
SMA_MIN = 40          # decision: an SMA needs >= 40 of the 50 closes
BREADTH_MIN_SHARE = 0.8   # decision: a group's breadth needs a value for >= 80% of its members
MIN_MEMBERS = 10      # decision: groups with fewer members never thrust
LOW, HIGH = 0.40, 0.615   # thrust: breadth <= 0.40 on some day of the last 10, >= 0.615 today
LOOKBACK = 10
COOLDOWN = 21         # no repeat for the same group within 21 trading days
HOLD = 21             # hold 21 trading days
MOM = 10              # momentum-matching variable: the group's 10-day market-relative return
SEED = 0x6E0          # pp_refits.py's groups-map seed, + the refit day

EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}


def big(tag):
    p = BIG / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- copied from research/phantom_portfolios/pp_common.py

def standardize(x):
    """[n, rows] panel -> rows standardized to mean 0 and unit sum of squares (missing = 0 after centring),
    so Z Z^T is the correlation matrix (together.rs `standardized`)."""
    f = np.isfinite(x)
    cnt = np.maximum(f.sum(1, keepdims=True), 1)
    mean = np.where(f, x, 0.0).sum(1, keepdims=True) / cnt
    c = np.where(f, x - mean, 0.0)
    norm = np.sqrt((c * c).sum(1, keepdims=True))
    return np.where(norm > 0, c / np.where(norm > 0, norm, 1.0), 0.0)


def market_relative(x):
    """together.rs `standardized(relative=true)` before scaling: each row minus its beta times the
    equal-weight mean of the rows."""
    f = np.isfinite(x)
    cnt = f.sum(0)
    m = np.where(cnt > 0, np.where(f, x, 0.0).sum(0) / np.maximum(cnt, 1), 0.0)
    ok = f
    mm = np.where(ok, m[None, :], np.nan)
    xm = np.nanmean(x, 1, keepdims=True)
    mmean = np.nanmean(mm, 1, keepdims=True)
    cov = np.nansum((x - xm) * (mm - mmean), 1, keepdims=True) / np.maximum(ok.sum(1, keepdims=True) - 1, 1)
    var = np.nansum((mm - mmean) ** 2, 1, keepdims=True) / np.maximum(ok.sum(1, keepdims=True) - 1, 1)
    beta = np.where(var > 0, cov / np.where(var > 0, var, 1.0), 0.0)
    return x - beta * m[None, :], m


def kmeans(x, k, seed, restarts=10, iters=300):
    """k-means with k-means++ seeding, best of `restarts` by inertia (cluster.rs `kmeans`)."""
    rng = np.random.default_rng(seed)
    p = x.shape[0]
    xx = (x * x).sum(1)
    best = (np.inf, None)
    for _ in range(restarts):
        centres = [x[rng.integers(p)]]
        near = ((x - centres[0]) ** 2).sum(1)
        while len(centres) < k:
            tot = near.sum()
            if tot <= 0:
                i = rng.integers(p)
            else:
                i = int(np.searchsorted(np.cumsum(near), rng.random() * tot))
                i = min(i, p - 1)
            centres.append(x[i])
            near = np.minimum(near, ((x - x[i]) ** 2).sum(1))
        c = np.array(centres)
        label = None
        for _ in range(iters):
            d = xx[:, None] - 2 * x @ c.T + (c * c).sum(1)[None, :]
            new = d.argmin(1)
            if label is not None and np.array_equal(new, label):
                break
            label = new
            size = np.bincount(label, minlength=k)
            c = np.zeros_like(c)
            np.add.at(c, label, x)
            empty = size == 0
            c[~empty] /= size[~empty, None]
            if empty.any():
                dist = ((x - c[label]) ** 2).sum(1)
                for g in np.flatnonzero(empty):
                    far = int(dist.argmax())
                    c[g] = x[far]
                    label[far] = g
                    dist[far] = -1
        inertia = ((x - c[label]) ** 2).sum()
        if inertia < best[0]:
            best = (inertia, label.copy())
    return best[1]


def spectral(corr, k, seed):
    """Ng-Jordan-Weiss spectral clustering of the positive part of a correlation matrix (together.rs `spectral`):
    top k eigenvectors of D^-1/2 A D^-1/2, rows scaled to unit length, k-means with 10 restarts."""
    a = np.maximum(corr, 0.0).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    deg = a.sum(1)
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)), 0.0)
    m = a * inv[:, None] * inv[None, :]
    w, v = np.linalg.eigh(m)
    x = v[:, ::-1][:, :k].copy()
    ln = np.sqrt((x * x).sum(1, keepdims=True))
    x = np.where(ln > 0, x / np.where(ln > 0, ln, 1.0), x)
    return kmeans(x, k, seed, restarts=10)


def cluster_ols(y, X, groups):
    """OLS with standard errors clustered by `groups` (CR1). Returns (beta, se, cov)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    xtx_inv = np.linalg.pinv(X.T @ X)
    b = xtx_inv @ X.T @ y
    u = y - X @ b
    _, gi = np.unique(groups, return_inverse=True)
    g = gi.max() + 1
    s = np.zeros((g, X.shape[1]))
    np.add.at(s, gi, X * u[:, None])
    meat = s.T @ s
    n, k = X.shape
    adj = g / max(g - 1, 1) * (n - 1) / max(n - k, 1)
    cov = adj * xtx_inv @ meat @ xtx_inv
    return b, np.sqrt(np.diag(cov)), cov


# ---------------------------------------------------------------- this experiment

def clustered_mean(y, clusters):
    """Mean of y with a CR1 standard error clustered by `clusters`. Returns (mean, se, t, n, n_clusters)."""
    y = np.asarray(y, np.float64)
    if len(y) < 2:
        return (float(y.mean()) if len(y) else np.nan), np.nan, np.nan, len(y), len(set(clusters))
    b, se, _ = cluster_ols(y, np.ones((len(y), 1)), np.asarray(clusters))
    return float(b[0]), float(se[0]), float(b[0] / se[0]) if se[0] > 0 else np.nan, len(y), len(np.unique(clusters))


def loo_corr(rel, lab, k):
    """Each stock's correlation, over the correlation period, between its market-relative return and the mean of
    the other members of its group (NaN-aware; days where either is missing are skipped)."""
    f = np.isfinite(rel)
    x0 = np.where(f, rel, 0.0)
    out = np.full(len(lab), np.nan)
    for g in range(k):
        idx = np.flatnonzero(lab == g)
        if len(idx) < 2:
            continue
        s, c = x0[idx].sum(0), f[idx].sum(0)
        for i in idx:
            den = c - f[i]
            ok = f[i] & (den >= 1)
            if ok.sum() < 20:
                continue
            other = (s[ok] - x0[i, ok]) / den[ok]
            out[i] = np.corrcoef(rel[i, ok], other)[0, 1]
    return out


def iso(day):
    return c2eval.iso(day)


def year_of(day):
    return int(iso(day)[:4])
