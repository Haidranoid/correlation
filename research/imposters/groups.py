"""The groups map, every month, re-implemented in numpy from crates/c2-census/src/together.rs (`groups`):
the 1,500 most-traded stocks (median dollar volume >= $5M over the correlation period, a return on 95% of its days),
market-relative daily returns over a 252-day correlation period (market = equal-weight mean of those stocks, times
each stock's beta), the positive part of their correlation matrix, spectral clustering (Ng, Jordan & Weiss: top-k
eigenvectors of D^-1/2 A D^-1/2, rows scaled to unit length, k-means with 10 restarts) into 30 groups.

For each month end it also stores every grouped stock's co-movement exposure to every group: the slope of its
market-relative return on the group's (equal-weight mean of the members' market-relative returns, without the stock
itself), over the same 252 days, plus the same slope from odd and even days only (for the noise check of test 1).

    python3 -I groups.py            # writes ~/.local/share/correlation2/experiments/imposters/months/YYYY-MM.npz
"""
import sys
import time

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import (K, MIN_DV, MONTHS_DIR, ODD, STOCKS, COVER, YEAR, load, month_ends, residuals, np)



def kmeans(x, k, seed, restarts=10, iters=300):
    """k-means with k-means++ starts, best of `restarts` by inertia (as crates/c2-census/src/cluster.rs kmeans)."""
    rng = np.random.default_rng(seed)
    p = len(x)
    best = (np.inf, None)
    for _ in range(restarts):
        c = [x[rng.integers(p)]]
        near = ((x - c[0]) ** 2).sum(1)
        while len(c) < k:
            i = rng.choice(p, p=near / near.sum()) if near.sum() > 0 else rng.integers(p)
            c.append(x[i])
            near = np.minimum(near, ((x - x[i]) ** 2).sum(1))
        c = np.array(c)
        lab = None
        for _ in range(iters):
            d = (x ** 2).sum(1)[:, None] - 2 * x @ c.T + (c ** 2).sum(1)[None, :]
            new = d.argmin(1)
            if lab is not None and (new == lab).all():
                break
            lab = new
            for g in range(k):
                m = lab == g
                if m.any():
                    c[g] = x[m].mean(0)
                else:  # empty group: take the point farthest from its centre
                    far = d[np.arange(p), lab].argmax()
                    c[g] = x[far]
                    lab[far] = g
        inertia = ((x - c[lab]) ** 2).sum()
        if inertia < best[0]:
            best = (inertia, lab.copy())
    return best[1]


def universe(ret, dv, d_end):
    win = slice(d_end - YEAR + 1, d_end + 1)
    cover = np.isfinite(ret[:, win]).sum(1)
    with np.errstate(all='ignore'):
        d = np.where(np.isfinite(dv[:, win]) & (dv[:, win] > 0), dv[:, win], np.nan)
        med = np.nanmedian(d, axis=1)
    ok = (cover >= COVER * YEAR) & (med >= MIN_DV)
    idx = np.nonzero(ok)[0]
    order = np.lexsort((idx, -med[idx]))
    return idx[order][:STOCKS], med


def standardize(x):
    ok = np.isfinite(x)
    n = np.maximum(ok.sum(1), 1)
    mean = np.where(ok, x, 0).sum(1) / n
    c = np.where(ok, x - mean[:, None], 0.0)
    norm = np.sqrt((c ** 2).sum(1))
    return np.where(norm[:, None] > 0, c / np.where(norm > 0, norm, 1)[:, None], 0.0)


def spectral(corr, k, seed):
    a = np.maximum(corr, 0.0)
    np.fill_diagonal(a, 0.0)
    deg = a.sum(1)
    inv = np.where(deg > 0, 1 / np.sqrt(np.where(deg > 0, deg, 1)), 0.0)
    m = a * inv[:, None] * inv[None, :]
    w, v = np.linalg.eigh(m)
    x = v[:, ::-1][:, :k]
    ln = np.linalg.norm(x, axis=1)
    x = np.where(ln[:, None] > 0, x / np.where(ln > 0, ln, 1)[:, None], x)
    lab = kmeans(x, k, seed)
    # groups by size (largest first), as together.rs orders them
    size = np.bincount(lab, minlength=k)
    order = sorted(range(k), key=lambda g: (-size[g], g))
    rank = np.empty(k, int)
    rank[order] = np.arange(k)
    return rank[lab]


def exposures(r, lab, k, cols=None):
    """Slope of each stock's market-relative return on each group's (leave-one-out for its own group).
    r [n, T] with NaN; lab [n]. cols: day mask to use (None = all). Returns beta [n, k]."""
    if cols is not None:
        r = r[:, cols]
    ok = np.isfinite(r)
    x = np.where(ok, r, 0.0)
    onehot = np.zeros((len(lab), k))
    onehot[np.arange(len(lab)), lab] = 1
    s = onehot.T @ x                      # [k, T] sums
    c = onehot.T @ ok.astype(float)       # [k, T] counts
    g = s / np.maximum(c, 1)              # group means
    xm = x - x.mean(1, keepdims=True)
    gm = g - g.mean(1, keepdims=True)
    beta = (xm @ gm.T) / (gm ** 2).sum(1)[None, :]
    # own group without the stock itself
    own_s = s[lab] - x
    own_c = c[lab] - ok
    own = own_s / np.maximum(own_c, 1)
    om = own - own.mean(1, keepdims=True)
    beta[np.arange(len(lab)), lab] = (xm * om).sum(1) / (om ** 2).sum(1)
    return beta


def correlations(r, lab, k, cols=None):
    """Correlation of each stock's market-relative return with each group's (leave-one-out for its own group)."""
    if cols is not None:
        r = r[:, cols]
    ok = np.isfinite(r)
    x = np.where(ok, r, 0.0)
    onehot = np.zeros((len(lab), k))
    onehot[np.arange(len(lab)), lab] = 1
    s = onehot.T @ x
    c = onehot.T @ ok.astype(float)
    g = s / np.maximum(c, 1)
    xm = x - x.mean(1, keepdims=True)
    gm = g - g.mean(1, keepdims=True)
    sx = np.sqrt((xm ** 2).sum(1))
    corr = (xm @ gm.T) / np.maximum(sx[:, None] * np.sqrt((gm ** 2).sum(1))[None, :], 1e-12)
    own = (s[lab] - x) / np.maximum(c[lab] - ok, 1)
    om = own - own.mean(1, keepdims=True)
    corr[np.arange(len(lab)), lab] = (xm * om).sum(1) / np.maximum(sx * np.sqrt((om ** 2).sum(1)), 1e-12)
    return corr


def main():
    MONTHS_DIR.mkdir(parents=True, exist_ok=True)
    e = load()
    ret, dv = e['ret'], np.asarray(e['dollar_volume'], dtype=np.float64)
    idx, ms = month_ends(e['days'])
    t0 = time.time()
    for d_end, m in zip(idx, ms):
        out = MONTHS_DIR / f'{m}.npz'
        if out.exists():
            continue
        u, med = universe(ret, dv, d_end)
        r, beta_mkt, market = residuals(ret, u, d_end)
        z = standardize(r[:, np.arange(YEAR) % 2 == 1] if ODD else r)
        corr = z @ z.T
        lab = spectral(corr, K, seed=0x6E0)
        b = exposures(r, lab, K)
        days = np.arange(YEAR)
        b_odd = exposures(r, lab, K, cols=days % 2 == 1)
        b_even = exposures(r, lab, K, cols=days % 2 == 0)
        np.savez_compressed(out, day=d_end, stocks=u, label=lab, beta=b.astype(np.float32),
                            beta_odd=b_odd.astype(np.float32), beta_even=b_even.astype(np.float32),
                            beta_mkt=beta_mkt.astype(np.float32), dv=med[u].astype(np.float32))
        print(f'{m}: {len(u)} stocks, group sizes {np.bincount(lab, minlength=K).min()}-'
              f'{np.bincount(lab, minlength=K).max()}, own-group exposure {np.median(b[np.arange(len(u)), lab]):.2f}, '
              f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
