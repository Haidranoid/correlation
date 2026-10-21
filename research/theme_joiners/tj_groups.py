"""The groups map's clustering, copied verbatim, read-only, from research/breadth_thrust/bt_common.py (itself copied
from research/phantom_portfolios/pp_common.py, the numpy port of crates/c2-census/src/together.rs and cluster.rs).
Used only by tj_refits.py to rebuild the refits on the new export the same way breadth thrust built them."""
import numpy as np


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
