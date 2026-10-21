"""Checks of the estimators and the solver on small cases (writes tests_out.txt).

  1. Ledoit-Wolf 2004: the vectorised estimator against a direct loop transcription of the paper's formulas
     (Ledoit & Wolf 2004, "Honey, I shrunk the sample covariance matrix", J. Portfolio Mgmt, Appendix B).
  2. The capped minimum-variance solver against brute force: every assignment of each weight to {0, cap, free},
     the KKT system solved on each face, the best feasible one kept.
  3. The solver at full size (N = 500) against a pure primal active-set run from equal weights (no projected gradient).
  4. HRP: single linkage against a naive agglomerative loop as the paper describes it; HRP = inverse-variance weights
     when the covariance is diagonal; a two-block toy.
  5. The twin covariance: its diagonal equals each stock's sample variance (OLS decomposition), and it is PD.

    python3 -I tests.py
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, np  # noqa: E402
import estimators as est  # noqa: E402

out = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


# ---------------------------------------------------------------------------------------------------------------
# 1. Ledoit-Wolf against the paper's formulas, element by element

def lw_loops(X):
    """Direct transcription of Ledoit & Wolf (2004), Appendix B (loops; y is T x N in the paper)."""
    N, T = X.shape
    y = X.T.astype(np.float64)
    ybar = y.mean(0)
    s = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            s[i, j] = sum((y[t, i] - ybar[i]) * (y[t, j] - ybar[j]) for t in range(T)) / T
    r = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            r[i, j] = s[i, j] / np.sqrt(s[i, i] * s[j, j])
    rbar = 2.0 / ((N - 1) * N) * sum(r[i, j] for i in range(N - 1) for j in range(i + 1, N))
    f = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            f[i, j] = s[i, i] if i == j else rbar * np.sqrt(s[i, i] * s[j, j])
    pi_ = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            pi_[i, j] = sum(((y[t, i] - ybar[i]) * (y[t, j] - ybar[j]) - s[i, j]) ** 2 for t in range(T)) / T
    pi_hat = pi_.sum()

    def theta(ii, i, j):   # theta_ii,ij
        return sum(((y[t, ii] - ybar[ii]) ** 2 - s[ii, ii]) * ((y[t, i] - ybar[i]) * (y[t, j] - ybar[j]) - s[i, j])
                   for t in range(T)) / T

    rho_hat = sum(pi_[i, i] for i in range(N))
    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            rho_hat += rbar / 2.0 * (np.sqrt(s[j, j] / s[i, i]) * theta(i, i, j) + np.sqrt(s[i, i] / s[j, j]) * theta(j, i, j))
    gamma_hat = sum((f[i, j] - s[i, j]) ** 2 for i in range(N) for j in range(N))
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = max(0.0, min(kappa / T, 1.0))
    return delta * f + (1 - delta) * s, delta


P('== 1. Ledoit-Wolf (constant correlation) vs the paper\'s formulas, element by element')
rng = np.random.default_rng(1)
worst = 0.0
for case, (N, T, load) in enumerate([(5, 30, 0.0), (5, 30, 1.0), (7, 25, 0.5), (8, 60, 2.0), (6, 12, 0.3)]):
    f = rng.standard_normal(T)
    X = load * f[None, :] * rng.uniform(0.5, 1.5, (N, 1)) + rng.standard_normal((N, T)) * rng.uniform(0.5, 2.0, (N, 1))
    X *= 0.01
    S1, d1, rb, _ = est.ledoit_wolf_cc(X)
    S2, d2 = lw_loops(X)
    err = np.abs(S1 - S2).max() / np.abs(S2).max()
    worst = max(worst, err, abs(d1 - d2))
    ev = np.linalg.eigvalsh(S1).min()
    P(f'  case {case}: N={N} T={T} delta vectorised {d1:.12f} loops {d2:.12f}; max rel diff {err:.2e}; rbar {rb:+.3f}; '
      f'min eigenvalue {ev:.2e}')
assert worst < 1e-10, worst
P(f'  PASS (worst difference {worst:.1e})')

# ---------------------------------------------------------------------------------------------------------------
# 2. Solver against brute force


def brute(H, cap):
    n = H.shape[0]
    best, bw = np.inf, None
    for pat in itertools.product((0, 1, 2), repeat=n):
        pat = np.array(pat)
        lo, up = pat == 0, pat == 1
        if cap * up.sum() > 1 + 1e-12:
            continue
        w, nu = est._face(H, lo, up, cap)
        if w is None:
            if abs(cap * up.sum() - 1) > 1e-12:
                continue
            w = np.where(up, cap, 0.0)
        F = ~(lo | up)
        if (w[F] < -1e-12).any() or (w[F] > cap + 1e-12).any():
            continue
        v = w @ H @ w
        if v < best:
            best, bw = v, w
    return best, bw


P('== 2. Capped minimum variance vs brute force (all 3^n patterns of {0, cap, free})')
worst_obj, worst_w, n_cases, n_bind = 0.0, 0.0, 0, 0
for trial in range(300):
    n = int(rng.integers(3, 9))
    kind = trial % 3
    if kind == 0:     # LW of random factor-model returns
        T = int(rng.integers(n + 2, 3 * n + 10))
        f = rng.standard_normal((2, T))
        X = (rng.standard_normal((n, 2)) @ f) * rng.uniform(0.2, 1.0) + rng.standard_normal((n, T)) * rng.uniform(0.3, 2, (n, 1))
        H = est.ledoit_wolf_cc(X)[0]
    elif kind == 1:   # random Wishart
        A = rng.standard_normal((n, n + 2))
        H = A @ A.T + 1e-3 * np.eye(n)
    else:             # strongly correlated with spread-out variances (caps bind hard)
        c = rng.uniform(0.3, 0.9)
        sd = np.exp(rng.normal(0, 0.7, n))
        R = c * np.ones((n, n)) + (1 - c) * np.eye(n)
        H = R * np.outer(sd, sd)
    cap = float(rng.choice([1.0 / n + 0.01, 1.0 / n + 0.05, 0.3, 0.5, 1.0]))
    cap = max(cap, 1.0 / n + 1e-9)
    w, info = est.minvar_capped(H, cap)
    bv, bw = brute(H, cap)
    v = w @ H @ w
    worst_obj = max(worst_obj, (v - bv) / bv)
    worst_w = max(worst_w, np.abs(w - bw).max())
    n_cases += 1
    n_bind += int(info['n_cap'] > 0)
    if trial < 6 or abs(v - bv) / bv > 1e-10:
        P(f'  trial {trial}: n={n} cap={cap:.3f} solver {v:.12e} brute {bv:.12e} max|dw| {np.abs(w - bw).max():.1e} '
          f'({info["method"]}, {info["n_cap"]} at cap, {info["n_free"]} free)')
assert worst_obj < 1e-10 and worst_w < 1e-7, (worst_obj, worst_w)
P(f'  PASS: {n_cases} random cases ({n_bind} with the cap binding); worst relative objective gap {worst_obj:.1e}, '
  f'worst weight difference {worst_w:.1e}')

# also a primal active-set run from equal weights alone (no projected gradient) on the same kind of cases
P('== 3. Full size: projected gradient + finish vs a pure primal active-set run from equal weights')
for trial in range(3):
    n, T = 500, 252
    f = rng.standard_normal((5, T)) * 0.01
    X = rng.standard_normal((n, 5)) @ f + rng.standard_normal((n, T)) * 0.01 * np.exp(rng.normal(0, 0.4, (n, 1)))
    H = est.ledoit_wolf_cc(X)[0]
    w, info = est.minvar_capped(H, 0.02)
    scale = np.trace(H) / n
    w2, nu2, its = est._primal_active_set(H / scale, np.full(n, 1.0 / n), 0.02)
    d = np.abs(w - w2).max()
    rel = (w @ H @ w - w2 @ H @ w2) / (w2 @ H @ w2)
    P(f'  trial {trial}: PG {info["pg_iters"]} iterations, residual {info["pg_res"]:.1e}, {info["method"]}; KKT primal '
      f'{info["kkt_primal"]:.1e} dual {info["kkt_dual"]:.1e}; vs pure active set ({its} iterations): max|dw| {d:.1e}, '
      f'objective gap {rel:.1e}; names {int((w > 1e-3).sum())}, at cap {info["n_cap"]}')
    assert d < 1e-9 and abs(rel) < 1e-12
P('  PASS')

# ---------------------------------------------------------------------------------------------------------------
# 4. HRP


def naive_single(D):
    """The paper's description: repeatedly merge the closest pair of clusters (single linkage: the nearest point),
    returning the merges as (members_a, members_b, distance)."""
    n = D.shape[0]
    clusters = {i: [i] for i in range(n)}
    nxt = n
    merges = []
    while len(clusters) > 1:
        keys = sorted(clusters)
        best = (np.inf, None, None)
        for a_i in range(len(keys)):
            for b_i in range(a_i + 1, len(keys)):
                a, b = keys[a_i], keys[b_i]
                d = D[np.ix_(clusters[a], clusters[b])].min()
                if d < best[0]:
                    best = (d, a, b)
        d, a, b = best
        merges.append((a, b, d, sorted(clusters[a] + clusters[b])))
        clusters[nxt] = clusters.pop(a) + clusters.pop(b)
        nxt += 1
    return merges


P('== 4. HRP')
for trial in range(20):
    n = int(rng.integers(4, 25))
    T = 60
    f = rng.standard_normal((3, T))
    X = rng.standard_normal((n, 3)) @ f * rng.uniform(0, 1.5) + rng.standard_normal((n, T))
    C = np.cov(X)
    sd = np.sqrt(np.diag(C))
    corr = np.clip(C / np.outer(sd, sd), -1, 1)
    D = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(D, 0)
    Dt = np.sqrt(((D[:, :, None] - D[:, None, :]) ** 2).sum(0))       # Euclidean distance between columns
    Z = est.single_linkage(Dt)
    M = naive_single(Dt)
    mine = []
    members = {i: [i] for i in range(n)}
    for i in range(n - 1):
        a, b = int(Z[i, 0]), int(Z[i, 1])
        members[n + i] = sorted(members[a] + members[b])
        mine.append((Z[i, 2], members[n + i]))
    for (d1, m1), (_, _, d2, m2) in zip(mine, M):
        assert abs(d1 - d2) < 1e-12 and m1 == m2, (d1, d2, m1, m2)
    w, Z2, order = est.hrp(C)
    assert sorted(order) == list(range(n)) and abs(w.sum() - 1) < 1e-12 and (w > 0).all()
P('  single linkage = the naive agglomerative loop (same merge distances and members) on 20 random cases: PASS')
sd = np.exp(rng.normal(0, 0.5, 12))
C = np.diag(sd ** 2)
w, _, _ = est.hrp(C)
ivp = (1 / sd ** 2) / (1 / sd ** 2).sum()
P(f'  diagonal covariance -> HRP = inverse variance: max diff {np.abs(w - ivp).max():.1e}')
assert np.abs(w - ivp).max() < 1e-12
# two blocks: 4 stocks correlated 0.8 among themselves, 4 others among themselves, 0 across, equal variances
R = np.zeros((8, 8))
R[:4, :4] = 0.8
R[4:, 4:] = 0.8
np.fill_diagonal(R, 1)
perm = np.array([0, 4, 1, 5, 2, 6, 3, 7])
Rp = R[np.ix_(perm, perm)]
w, Z, order = est.hrp(Rp)
blocks = [int(perm[o] >= 4) for o in order]
P(f'  two-block toy (interleaved): quasi-diagonal order puts each block together: {blocks}; weights {np.round(w, 4)}')
assert blocks in ([0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]) and np.abs(w - 0.125).max() < 1e-12
P('  PASS')

# ---------------------------------------------------------------------------------------------------------------
# 5. Twin covariance
P('== 5. Twin covariance')
T, N, K = 252, 40, 5
m = rng.standard_normal(T) * 0.01
G = 0.9 * m[None, :] + rng.standard_normal((K, T)) * 0.006
G[0] = m                      # one twin is the market itself: market-only loading
twin_of = rng.integers(-1, K, N)
X = 1.1 * m[None, :] + np.where(twin_of[:, None] >= 0, G[np.maximum(twin_of, 0)], 0) * 0.5 + rng.standard_normal((N, T)) * 0.015
X -= X.mean(1, keepdims=True)
m0 = m - m.mean()
G0 = G - G.mean(1, keepdims=True)
Sig, info = est.twin_cov(X, m0, G0, twin_of)
dd = np.abs(np.diag(Sig) - (X * X).sum(1) / T).max() / np.diag(Sig).max()
ev = np.linalg.eigvalsh(Sig).min()
P(f'  diag(Sigma) vs sample variances: max rel diff {dd:.1e}; min eigenvalue {ev:.2e}; twins used {info["n_twin"]}, '
  f'funds {info["n_funds"]}, market-only because the twin is the market {info["n_dead"]}')
assert dd < 1e-12 and ev > 0
P('  PASS')

(HERE / 'tests_out.txt').write_text('\n'.join(out) + '\n')
