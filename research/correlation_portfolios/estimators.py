"""The estimators and the solver (numpy only; checked in tests.py).

  ledoit_wolf_cc(X)       Ledoit & Wolf (2004), "Honey, I shrunk the sample covariance matrix": shrinkage of the
                          sample covariance toward the constant-correlation matrix with the optimal intensity.
  minvar_capped(S, cap)   exact long-only minimum variance with sum(w) = 1 and 0 <= w <= cap: accelerated projected
                          gradient (FISTA) to a 1e-10 tolerance, then an active-set finish that solves the KKT system on
                          the identified free set and verifies the multipliers (primal active-set iterations if not).
  hrp(S)                  López de Prado (2016) hierarchical risk parity: correlation distance, Euclidean distance
                          between its columns, single linkage, quasi-diagonalization, recursive bisection with
                          inverse-variance allocation.
  twin_cov(...)           the twin-structure covariance: SPY plus one factor per twin fund.
"""
import numpy as np


# ---------------------------------------------------------------------------------------------------------------
# Ledoit-Wolf 2004, constant-correlation target

def ledoit_wolf_cc(X):
    """X: [N, T] returns (one row per stock). Returns (Sigma [N, N], delta, rbar, S).

    With y_it demeaned (each row's mean removed), S = Y'Y/T (the paper's 1/T normalisation), r_ij = s_ij/sqrt(s_ii s_jj),
    rbar = mean of r_ij over i != j, F_ii = s_ii, F_ij = rbar sqrt(s_ii s_jj). The intensity is
    delta* = max(0, min(kappa/T, 1)), kappa = (pi - rho)/gamma, with (paper, Appendix B)
      pi    = sum_ij pi_ij,   pi_ij = (1/T) sum_t ((y_it y_jt) - s_ij)^2
      rho   = sum_i pi_ii + sum_i sum_{j != i} (rbar/2) (sqrt(s_jj/s_ii) theta_ii,ij + sqrt(s_ii/s_jj) theta_jj,ij)
      theta_ii,ij = (1/T) sum_t (y_it^2 - s_ii)(y_it y_jt - s_ij)
      gamma = sum_ij (f_ij - s_ij)^2
    and Sigma = delta* F + (1 - delta*) S."""
    Y = np.asarray(X, np.float64)
    N, T = Y.shape
    Y = Y - Y.mean(1, keepdims=True)
    S = Y @ Y.T / T
    var = np.diag(S).copy()
    sd = np.sqrt(var)
    R = S / np.outer(sd, sd)
    rbar = (R.sum() - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)
    Y2 = Y * Y
    pimat = Y2 @ Y2.T / T - S * S
    pi = pimat.sum()
    # theta[i, j] = theta_ii,ij = (1/T) sum_t y_it^3 y_jt - s_ii s_ij
    theta = (Y2 * Y) @ Y.T / T - var[:, None] * S
    np.fill_diagonal(theta, 0.0)
    ratio = sd[None, :] / sd[:, None]          # [i, j] = sqrt(s_jj / s_ii)
    rho = np.trace(pimat) + rbar * (ratio * theta).sum()
    gamma = ((F - S) ** 2).sum()
    kappa = (pi - rho) / gamma
    delta = max(0.0, min(kappa / T, 1.0))
    Sigma = delta * F + (1 - delta) * S
    return Sigma, delta, rbar, S


# ---------------------------------------------------------------------------------------------------------------
# Capped long-only minimum variance

def proj_capped_simplex(v, cap):
    """Euclidean projection onto {0 <= w <= cap, sum w = 1}: w = clip(v - tau, 0, cap), tau by bisection."""
    lo, hi = v.min() - cap, v.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:      # no more floating-point progress
            break
        s = np.clip(v - mid, 0.0, cap).sum()
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    w = np.clip(v - 0.5 * (lo + hi), 0.0, cap)
    # remove the last rounding error on the free coordinates (at most ~1e-16)
    free = (w > 0) & (w < cap)
    if free.any():
        w[free] += (1.0 - w.sum()) / free.sum()
    return w


def _face(H, lo, up, cap):
    """Minimum of w'Hw/2 on the face where w_lo = 0, w_up = cap and sum w = 1 (lo, up boolean masks).
    Returns (w, nu) with H_FF w_F + H_FU w_U = nu * 1, or (None, None) if the free set is empty."""
    n = H.shape[0]
    F = ~(lo | up)
    w = np.zeros(n)
    w[up] = cap
    if not F.any():
        return None, None
    s = 1.0 - cap * up.sum()
    HF = H[np.ix_(F, F)]
    b = -H[np.ix_(F, up)] @ w[up] if up.any() else np.zeros(F.sum())
    sol = np.linalg.solve(HF, np.column_stack([np.ones(F.sum()), b]))
    a, bb = sol[:, 0], sol[:, 1]
    nu = (s - bb.sum()) / a.sum()
    w[F] = nu * a + bb
    return w, nu


def kkt_violation(H, w, cap, nu=None):
    """Largest KKT violation of w for min w'Hw/2, sum w = 1, 0 <= w <= cap, relative to the gradient scale.
    Returns (primal, dual): primal = max bound/sum violation; dual = max over the sign conditions of the bound
    multipliers and the free-gradient equality, divided by |nu|."""
    g = H @ w
    at0 = w <= 0.0
    atc = w >= cap
    F = ~(at0 | atc)
    if nu is None:
        nu = g[F].mean() if F.any() else g.mean()
    primal = max(float(-w.min()), float((w - cap).max()), abs(float(w.sum()) - 1.0), 0.0)
    d = [0.0]
    if F.any():
        d.append(float(np.abs(g[F] - nu).max()))
    if at0.any():
        d.append(float(max(0.0, (nu - g[at0]).max())))
    if atc.any():
        d.append(float(max(0.0, (g[atc] - nu).max())))
    return primal, max(d) / abs(nu)


def _primal_active_set(H, x, cap, maxit=20000, tol=1e-12):
    """Primal active-set method (Nocedal & Wright, Algorithm 16.3) from a feasible x, working set = coordinates at
    their bounds. Exact up to rounding; finite for a positive-definite H."""
    n = len(x)
    lo = x <= 0.0
    up = x >= cap
    x = x.copy()
    x[lo] = 0.0
    x[up] = cap
    for it in range(maxit):
        y, nu = _face(H, lo, up, cap)
        if y is None:
            # a vertex (every coordinate at a bound): any nu in [max g_up, min g_lo] certifies it
            g = H @ x
            a = g[up].max() if up.any() else -np.inf
            b = g[lo].min() if lo.any() else np.inf
            nu = 0.5 * (a + b) if np.isfinite(a) and np.isfinite(b) else (a if np.isfinite(a) else b)
        else:
            p = y - x
            F = ~(lo | up)
            idx = np.nonzero(F)[0]
            pf, xf = p[idx], x[idx]
            with np.errstate(divide='ignore', invalid='ignore'):
                a_lo = np.where(pf < 0, -xf / pf, np.inf)
                a_up = np.where(pf > 0, (cap - xf) / pf, np.inf)
            j_lo, j_up = int(np.argmin(a_lo)), int(np.argmin(a_up))
            alpha, block, to_up = 1.0, -1, False
            if a_lo[j_lo] < alpha:
                alpha, block, to_up = a_lo[j_lo], idx[j_lo], False
            if a_up[j_up] < alpha:
                alpha, block, to_up = a_up[j_up], idx[j_up], True
            if block >= 0:                      # blocked: move to the bound and add it to the working set
                x = x + alpha * p
                if to_up:
                    up[block] = True
                    x[block] = cap
                else:
                    lo[block] = True
                    x[block] = 0.0
                x[lo] = 0.0
                x[up] = cap
                continue
            x = y                               # the face's minimiser is feasible: check the multipliers
        g = H @ x
        mu_lo = np.where(lo, g - nu, np.inf)
        mu_up = np.where(up, nu - g, np.inf)
        j0, j1 = int(np.argmin(mu_lo)), int(np.argmin(mu_up))
        worst = min(mu_lo[j0], mu_up[j1])
        if worst >= -tol * abs(nu):
            return x, nu, it + 1
        if mu_lo[j0] <= mu_up[j1]:
            lo[j0] = False
        else:
            up[j1] = False
    raise RuntimeError('active set did not converge')


def minvar_capped(Sigma, cap, tol=1e-10, maxit=200000, w0=None):
    """Exact minimiser of w' Sigma w subject to sum w = 1, 0 <= w <= cap. Returns (w, info)."""
    n = Sigma.shape[0]
    assert n * cap >= 1.0 - 1e-12, (n, cap)
    scale = np.trace(Sigma) / n
    H = Sigma / scale
    L = float(np.linalg.eigvalsh(H)[-1])
    x = proj_capped_simplex(np.full(n, 1.0 / n) if w0 is None else np.asarray(w0, float), cap)
    y, t = x.copy(), 1.0
    it = 0
    res = np.inf
    for it in range(1, maxit + 1):
        g = H @ y
        xn = proj_capped_simplex(y - g / L, cap)
        if (y - xn) @ (xn - x) > 0:   # gradient-based adaptive restart (O'Donoghue & Candes 2015)
            t = 1.0
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = xn + ((t - 1) / tn) * (xn - x)
        x, t = xn, tn
        if it % 10 == 0:
            # projected-gradient residual at x (in weight units)
            res = float(np.abs(x - proj_capped_simplex(x - (H @ x) / L, cap)).max())
            if res <= tol:
                break
    pg_iters, pg_res = it, res
    # active-set finish: solve the KKT system on the identified face and verify it
    lo = x <= 0.0
    up = x >= cap
    w, nu = _face(H, lo, up, cap)
    method = 'pg+face'
    ok = False
    if w is not None:
        p, d = kkt_violation(H, w, cap, nu)
        ok = p <= 1e-12 and d <= 1e-9
    if not ok:
        w, nu, n_as = _primal_active_set(H, x, cap)
        method = f'pg+active-set({n_as})'
    p, d = kkt_violation(H, w, cap, nu)
    w = np.clip(w, 0.0, cap)
    return w, {'pg_iters': pg_iters, 'pg_res': pg_res, 'method': method, 'kkt_primal': p, 'kkt_dual': d,
               'n_free': int(((w > 0) & (w < cap)).sum()), 'n_cap': int((w >= cap).sum())}


# ---------------------------------------------------------------------------------------------------------------
# Hierarchical risk parity (López de Prado 2016)

def single_linkage(D):
    """Single-linkage agglomerative clustering of a condensed-free square distance matrix D [n, n], returned in
    scipy's linkage format Z [n-1, 4] (as the paper's code gets from scipy.cluster.hierarchy.linkage(..., 'single')):
    scipy builds single linkage from the minimum spanning tree (Prim), sorts its edges by distance (stable), and
    labels merges with union-find, the smaller cluster id first; new clusters get ids n, n+1, ..."""
    n = D.shape[0]
    in_tree = np.zeros(n, bool)
    dist = np.full(n, np.inf)
    edges = []
    cur = 0
    in_tree[0] = True
    # Prim's algorithm as scipy's mst_single_linkage runs it: grow from node 0, update, pick the nearest node
    for _ in range(n - 1):
        dist = np.minimum(dist, D[cur])
        dist_masked = np.where(in_tree, np.inf, dist)
        nxt = int(np.argmin(dist_masked))
        edges.append((cur_parent(dist, D, in_tree, nxt), nxt, dist_masked[nxt]))
        in_tree[nxt] = True
        cur = nxt
    Z = np.zeros((n - 1, 4))
    Z[:, :3] = np.array(edges, np.float64)
    Z = Z[np.argsort(Z[:, 2], kind='mergesort')]
    parent = np.arange(2 * n - 1)
    size = np.ones(2 * n - 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n - 1):
        a, b = find(int(Z[i, 0])), find(int(Z[i, 1]))
        if a > b:
            a, b = b, a
        Z[i, 0], Z[i, 1] = a, b
        new = n + i
        parent[a] = new
        parent[b] = new
        size[new] = size[a] + size[b]
        Z[i, 3] = size[new]
    return Z


def cur_parent(dist, D, in_tree, nxt):
    """The tree node closest to nxt (the MST edge's other end); only its cluster matters for labelling."""
    cand = np.nonzero(in_tree)[0]
    return int(cand[np.argmin(D[nxt, cand])])


def quasi_diag(Z):
    """getQuasiDiag (López de Prado 2016): leaf order of the linkage, each cluster replaced by (item 1, item 2)."""
    Z = Z.astype(int)
    n = Z[-1, 3]
    order = [Z[-1, 0], Z[-1, 1]]
    while max(order) >= n:
        out = []
        for c in order:
            if c >= n:
                out += [Z[c - n, 0], Z[c - n, 1]]
            else:
                out.append(c)
        order = out
    return order


def _cluster_var(cov, items):
    c = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(c)
    ivp /= ivp.sum()
    return float(ivp @ c @ ivp)


def rec_bipart(cov, sort_ix):
    """getRecBipart (López de Prado 2016), with Python 3 integer halves."""
    n = cov.shape[0]
    w = np.ones(n)
    c_items = [list(sort_ix)]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c0, c1 = c_items[i], c_items[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return w


def hrp(cov):
    """HRP weights for a covariance matrix (its correlation drives the tree; its variances the allocation)."""
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    corr = np.clip(corr, -1.0, 1.0)
    D = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))            # correlDist
    np.fill_diagonal(D, 0.0)
    # the paper's second step: Euclidean distance between the columns of D (what linkage does with a square input)
    sq = (D * D).sum(0)
    Dt = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * D.T @ D, 0.0))
    np.fill_diagonal(Dt, 0.0)
    Dt = 0.5 * (Dt + Dt.T)
    Z = single_linkage(Dt)
    order = quasi_diag(Z)
    w = rec_bipart(cov, order)
    return w / w.sum(), Z, order


# ---------------------------------------------------------------------------------------------------------------
# Twin-structure covariance

def twin_cov(X, m, G, twin_of):
    """X [N, T] stock returns, m [T] SPY returns, G [K, T] the twin funds' returns (all demeaned, missing = 0),
    twin_of [N] row in G (-1 = no twin). Each fund's factor is g_f = r_f - beta_f m (beta_f by OLS over the window);
    each stock's model is r_i = b_m m + b_g g_twin(i) + e_i (OLS); Sigma = B Omega B' + diag(var e), with Omega the
    sample covariance of [m, g_1..g_K]. Funds whose factor is identically zero (SPY itself) give market-only loading.
    Returns (Sigma, info)."""
    X = np.asarray(X, np.float64)
    N, T = X.shape
    m = np.asarray(m, np.float64)
    mm = m @ m
    if mm <= 0:     # no SPY return in the window at all (store SPY starts 2005-02-25): no market factor
        mm = np.inf
    G = np.asarray(G, np.float64)
    K = G.shape[0]
    beta_f = G @ m / mm
    gf = G - beta_f[:, None] * m[None, :]
    gg = (gf * gf).sum(1)
    dead = gg <= 1e-10 * (m @ m)            # the fund is the market (factor identically zero)
    b_m = X @ m / mm                        # = the multiple-regression coefficient, since g_f is orthogonal to m
    b_g = np.zeros(N)
    has = twin_of >= 0
    has[has] &= ~dead[twin_of[has]]
    idx = np.nonzero(has)[0]
    f = twin_of[idx]
    b_g[idx] = (X[idx] * gf[f]).sum(1) / gg[f]
    fit = b_m[:, None] * m[None, :]
    fit[idx] += b_g[idx, None] * gf[f]
    e = X - fit
    resid = (e * e).sum(1) / T
    used = np.unique(f)
    pos = {u: i + 1 for i, u in enumerate(used)}
    Fm = np.vstack([m[None, :], gf[used]])          # [1 + K_used, T]
    Om = Fm @ Fm.T / T
    B = np.zeros((N, 1 + len(used)))
    B[:, 0] = b_m
    B[idx, [pos[u] for u in f]] = b_g[idx]
    Sigma = B @ Om @ B.T
    Sigma[np.diag_indices(N)] += resid
    return Sigma, {'b_m': b_m, 'b_g': b_g, 'n_twin': int(has.sum()), 'n_funds': int(len(used)),
                   'n_dead': int((twin_of >= 0).sum() - has.sum()), 'resid': resid}
