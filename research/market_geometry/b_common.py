"""Hypothesis B of experiments/untested-luca-market-geometry.md: shared settings and the graph geometry.

Every rule is in research/market_geometry/results_B.md ("Decisions"). Everything here is point in time: month k's
values use closes through t0 = tw['rebal'][k] only.

    import sys; sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_geometry'); import b_common as B
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/home/bhoult/programming/correlation')
HERE = REPO / 'research/market_geometry'
WORK = Path.home() / '.local/share/correlation2/experiments/market_geometry/b'
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / 'research/lib'))
sys.path.insert(0, str(REPO / 'research/published_data'))
import c2eval  # noqa: E402
import pubdata  # noqa: E402

EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}
OLD_LAST = '2025-06-27'

WIN = 63            # curvature window, as states()
WIN_LONG = 252      # the stability check's long window
N_UNIV = 500
MIN_FRAC = 0.9
EDGE_FRAC = 0.05    # primary graph: the 5% most-correlated pairs
PUB_THRESH = 0.75   # published-style graph: MST + rho >= 0.75 (Samal et al.)
H = 21
SEED = 20261023
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:8.1f}s]', *a, flush=True)


def load_module(name, path):
    """Load a file as a module under a distinct name (two shared folders both call theirs common.py)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mt_common():
    return load_module('mt_common', REPO / 'research/market_timing/common.py')


def cp_common():
    return load_module('cp_common', REPO / 'research/correlation_portfolios/common.py')


def cp_estimators():
    return load_module('cp_estimators', REPO / 'research/correlation_portfolios/estimators.py')


def trailing_median(x, k, nmax=60, nmin=24):
    """research/market_timing/common.py's rule (copied): median of the previous finite x[k-60..k-1], >= 24, else NaN."""
    prev = x[max(0, k - nmax):k]
    prev = prev[np.isfinite(prev)]
    return float(np.median(prev)) if len(prev) >= nmin else float('nan')


def trailing_z(x, k, nmax=60, nmin=24):
    """(x[k] - mean) / sd (ddof 1) of the previous finite x[k-60..k-1], >= 24, else NaN."""
    prev = x[max(0, k - nmax):k]
    prev = prev[np.isfinite(prev)]
    if len(prev) < nmin or not np.isfinite(x[k]):
        return float('nan')
    s = prev.std(ddof=1)
    return float((x[k] - prev.mean()) / s) if s > 0 else float('nan')


# ---------------------------------------------------------------------------------------------------------------
# SPY (Yahoo adjusted close) and the T-bill on export days

def spy_yahoo(days):
    d, p = pubdata.etf_adjclose(['SPY'])
    return pubdata.on_days(d, p['SPY'], days)


def rf(days):
    return pubdata.rf_daily(np.asarray(days))


# ---------------------------------------------------------------------------------------------------------------
# The universe and the window (states() in research/market_timing/common.py, reimplemented to return the stocks)

def window(close, dv, excl_k, t0, win=WIN, n_univ=N_UNIV):
    """The states() universe at close t0 over `win` daily log returns (days t0-win+1..t0).
    Returns sel [n] (export rows, most traded first), R [n, win] raw log returns (NaN missing), X [n, win] demeaned
    over finite days with missing 0, sd [n] (nanstd of R)."""
    c = np.asarray(close[:, t0 - win:t0 + 1], np.float64)
    with np.errstate(all='ignore'):
        r = np.log(c[:, 1:] / c[:, :-1])
        r[~np.isfinite(r)] = np.nan
        mdv = np.nanmedian(np.asarray(dv[:, t0 - win + 1:t0 + 1], np.float64), 1)
        sd = np.nanstd(r, 1)
    ok = (np.isfinite(r).mean(1) >= MIN_FRAC) & (np.nan_to_num(mdv) > 0) & ~excl_k & (np.nan_to_num(sd) > 0)
    n_elig = int(ok.sum())
    order = np.argsort(-np.where(ok, mdv, -np.inf), kind='stable')
    sel = order[:min(n_univ, n_elig)]
    R = r[sel]
    m = np.nanmean(R, 1, keepdims=True)
    X = R - m
    X[~np.isfinite(X)] = 0.0
    return sel, R, X, sd[sel], n_elig


def standardize(X, sd=None):
    """states(): divide by sd and normalize each row (rows already demeaned, missing 0). Returns Z with Z Z' = C."""
    Z = X / (sd[:, None] if sd is not None else X.std(1, keepdims=True))
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / nrm


def standardize_subset(R):
    """Odd/even halves: demean each row over its finite values in the subset, missing 0, divide by nanstd, normalize."""
    with np.errstate(all='ignore'):
        m = np.nanmean(R, 1, keepdims=True)
        s = np.nanstd(R, 1, keepdims=True)
    X = R - m
    X[~np.isfinite(X)] = 0.0
    Z = X / s
    Z[~np.isfinite(Z)] = 0.0
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return Z / nrm


def avg_corr_absorption(X, Z):
    """states(): mean off-diagonal correlation and the top-5 eigenvalues' share of the 63 x 63 Gram spectrum."""
    n = Z.shape[0]
    s = Z.sum(0)
    avg = (s @ s - n) / (n * (n - 1))
    ev = np.linalg.eigvalsh(X.T @ X)
    ev = np.sort(np.clip(ev, 0, None))[::-1]
    return float(avg), float(ev[:5].sum() / ev.sum())


# ---------------------------------------------------------------------------------------------------------------
# Graphs

def top_edges(C, frac=EDGE_FRAC):
    """The floor(frac * n(n-1)/2) most-correlated pairs (stable sort on the upper triangle). Returns (i, j) arrays."""
    n = C.shape[0]
    iu, ju = np.triu_indices(n, 1)
    m = int(np.floor(frac * len(iu)))
    order = np.argsort(-C[iu, ju], kind='stable')[:m]
    return iu[order], ju[order]


def pub_edges(C, thresh=PUB_THRESH):
    """Published style: the minimum spanning tree on L = sqrt(2(1 - rho)) plus every pair with rho >= thresh."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    L = lengths(C)
    np.fill_diagonal(L, 0.0)
    T = minimum_spanning_tree(L).tocoo()
    n = C.shape[0]
    A = np.zeros((n, n), bool)
    A[T.row, T.col] = True
    A |= C >= thresh
    np.fill_diagonal(A, False)
    A = A | A.T
    i, j = np.nonzero(np.triu(A, 1))
    return i, j


def lengths(C):
    return np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))


def adjacency(n, i, j):
    A = np.zeros((n, n), bool)
    A[i, j] = True
    A[j, i] = True
    return A


def triangles_per_edge(A, i, j):
    Af = A.astype(np.float32)
    P = Af @ Af
    return P[i, j].astype(np.int64)


def forman_aug(deg, i, j, t):
    return 4.0 - deg[i] - deg[j] + 3.0 * t


def forman_samal(n, i, j, w):
    """Samal et al.'s Eq. 4 (Sreejith et al. 2016) with node weights 1 and edge weights w:
    F(e) = 2 - sum_{e' at x, e' != e} sqrt(w_e / w_e') - the same at y = 4 - sqrt(w_e) (S_x + S_y),
    S_v = sum over edges at v of 1 / sqrt(w)."""
    w = np.maximum(w, 1e-9)
    inv = 1.0 / np.sqrt(w)
    S = np.zeros(n)
    np.add.at(S, i, inv)
    np.add.at(S, j, inv)
    return 4.0 - np.sqrt(w) * (S[i] + S[j])


def plain_stats(n, i, j, A, t, rho):
    """Degree concentration sum d^2 / m, triangle density T / m, mean edge rho, largest component's share of n."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    m = len(i)
    deg = A.sum(1).astype(np.float64)
    T = t.sum() / 3.0
    ncomp, lab = connected_components(csr_matrix(A), directed=False)
    lcc = np.bincount(lab).max() / n
    return {'m': m, 'degconc': float((deg ** 2).sum() / m), 'tridens': float(T / m), 'rho_edge': float(rho.mean()),
            'lcc': float(lcc), 'T': float(T), 'deg_max': int(deg.max())}


def apsp(n, i, j, L):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    G = csr_matrix((np.concatenate([L, L]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(n, n))
    return shortest_path(G, method='D', directed=False)


def neighbours(n, i, j):
    nb = [[] for _ in range(n)]
    for a, b in zip(i.tolist(), j.tolist()):
        nb[a].append(b)
        nb[b].append(a)
    return [np.array(sorted(x), np.int64) for x in nb]


def ollivier(n, i, j, L, D=None):
    """Ollivier-Ricci per edge: 1 - W1(m_x, m_y) / d(x, y), m_x = 1/2 on x and 1/(2 deg x) on each neighbour, ground
    distance d = weighted shortest paths on the graph with edge lengths L (D, computed if None; on a real graph
    d(x, y) = L_xy because L is a Euclidean metric). W1 exact by POT's network simplex."""
    import ot
    if D is None:
        D = apsp(n, i, j, L)
    nb = neighbours(n, i, j)
    out = np.empty(len(i))
    worst = None
    for e, (x, y) in enumerate(zip(i.tolist(), j.tolist())):
        sx = np.concatenate([[x], nb[x]])
        sy = np.concatenate([[y], nb[y]])
        a = np.full(len(sx), 0.5 / len(nb[x]))
        a[0] = 0.5
        b = np.full(len(sy), 0.5 / len(nb[y]))
        b[0] = 0.5
        M = np.ascontiguousarray(D[np.ix_(sx, sy)])
        w, lg = ot.emd2(a, b, M, numItermax=50_000_000, log=True)
        if lg.get('warning'):
            worst = lg['warning']
        out[e] = 1.0 - w / D[x, y]
    if worst:
        raise RuntimeError(f'emd did not converge: {worst}')
    return out


def rewire(n, i, j, L, seed, max_rounds=200):
    """Double-edge swaps keeping degrees and a simple graph; lengths travel with the swap. Rounds of m attempts, at
    least 10, stop when |mean T(last 5) - mean T(5 before)| <= sd(5 before). Returns (i, j, L, info)."""
    rng = np.random.default_rng(seed)
    m = len(i)
    ea, eb, el = i.tolist(), j.tolist(), L.tolist()
    have = set(min(a, b) * n + max(a, b) for a, b in zip(ea, eb))
    Ts, acc = [], 0
    for rnd in range(max_rounds):
        p = rng.integers(0, m, size=(m, 2)).tolist()
        coin = (rng.random(m) < 0.5).tolist()
        for (u, v), cf in zip(p, coin):
            if u == v:
                continue
            a, b, c, d = ea[u], eb[u], ea[v], eb[v]
            if cf:
                n1, n2 = (a, d), (c, b)
            else:
                n1, n2 = (a, c), (b, d)
            if n1[0] == n1[1] or n2[0] == n2[1]:
                continue
            k1 = min(n1) * n + max(n1)
            k2 = min(n2) * n + max(n2)
            if k1 == k2 or k1 in have or k2 in have:
                continue
            have.discard(min(a, b) * n + max(a, b))
            have.discard(min(c, d) * n + max(c, d))
            have.add(k1)
            have.add(k2)
            l1, l2 = el[u], el[v]
            ea[u], eb[u], el[u] = n1[0], n1[1], l1
            ea[v], eb[v], el[v] = n2[0], n2[1], l2
            acc += 1
        ii, jj = np.array(ea), np.array(eb)
        A = adjacency(n, ii, jj)
        Ts.append(float(triangles_per_edge(A, ii, jj).sum() / 3.0))
        if rnd + 1 >= 10:
            last, before = np.array(Ts[-5:]), np.array(Ts[-10:-5])
            if abs(last.mean() - before.mean()) <= before.std(ddof=1):
                break
    ii, jj, ll = np.array(ea), np.array(eb), np.array(el)
    lo, hi = np.minimum(ii, jj), np.maximum(ii, jj)
    return lo, hi, ll, {'rounds': len(Ts), 'accepted': acc, 'T_path': Ts}


def graph_measures(n, i, j, C, L=None, do_ollivier=True, keep_edges=False):
    """Everything on one graph: plain statistics, augmented and Samal Forman, Ollivier (per-edge means)."""
    rho = C[i, j]
    if L is None:
        L = lengths(C)[i, j]
    A = adjacency(n, i, j)
    deg = A.sum(1).astype(np.float64)
    t = triangles_per_edge(A, i, j)
    st = plain_stats(n, i, j, A, t, rho)
    fa = forman_aug(deg, i, j, t)
    fs = forman_samal(n, i, j, L)
    st['forman'] = float(fa.mean())
    st['forman_closed'] = float(4.0 - (deg ** 2).sum() / len(i) + 9.0 * st['T'] / len(i))
    st['forman_samal'] = float(fs.mean())
    if do_ollivier:
        oc = ollivier(n, i, j, L)
        st['ollivier'] = float(oc.mean())
    else:
        oc = None
        st['ollivier'] = float('nan')
    if keep_edges:
        st['edges'] = {'i': i.astype(np.int16), 'j': j.astype(np.int16), 'rho': rho.astype(np.float32),
                       'forman': fa.astype(np.float32), 'ollivier': None if oc is None else oc.astype(np.float32)}
    return st


# ---------------------------------------------------------------------------------------------------------------
# Statistics shared by every test (the plan's "Statistics")

def nw_se(x, lags=2):
    """Newey-West (Bartlett, `lags` lags) standard error of the mean of x."""
    x = np.asarray(x, np.float64)
    T = len(x)
    u = x - x.mean()
    v = u @ u / T
    for l in range(1, lags + 1):
        v += 2.0 * (1.0 - l / (lags + 1.0)) * (u[l:] @ u[:-l]) / T
    return float(np.sqrt(max(v, 0.0) / T))


def norm_sf(z):
    from math import erfc, sqrt
    return 0.5 * erfc(z / sqrt(2.0))


def block_boot(x, block=6, n=10_000, seed=SEED):
    """Moving-block bootstrap of the mean (blocks of whole periods). Returns (one-sided p for mean > 0 from the
    centred draws, 5th and 95th percentiles of the resampled mean)."""
    x = np.asarray(x, np.float64)
    T = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, nb * block)[:, :T]
    mstar = x[idx].mean(1)
    m = x.mean()
    p = float(np.mean(mstar - m >= m))
    return p, float(np.percentile(mstar, 5)), float(np.percentile(mstar, 95))


def mean_test(x, lags=2, block=6, seed=SEED):
    """One-sided test of mean(x) > 0: mean, NW se, t, normal p, bootstrap p, 90% intervals (NW and bootstrap)."""
    x = np.asarray(x, np.float64)
    nan = float('nan')
    if len(x) < 3:
        return {'n': int(len(x)), 'mean': float(x.mean()) if len(x) else nan, 'se': nan, 't': nan, 'p': nan,
                'p_boot': nan, 'ci90_nw': (nan, nan), 'ci90_boot': (nan, nan)}
    m = float(x.mean())
    se = nw_se(x, lags)
    t = m / se if se > 0 else float('nan')
    pb, lo, hi = block_boot(x, block, seed=seed) if len(x) > block else (nan, nan, nan)
    return {'n': int(len(x)), 'mean': m, 'se': se, 't': t, 'p': norm_sf(t) if t == t else float('nan'), 'p_boot': pb,
            'ci90_nw': (m - 1.645 * se, m + 1.645 * se), 'ci90_boot': (lo, hi)}


def holm(pvals, alpha=0.05):
    """Holm step-down: adjusted p-values and pass flags, in the input order."""
    p = np.asarray(pvals, np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj, adj < alpha
