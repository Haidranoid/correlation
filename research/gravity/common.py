"""Gravity (experiments/untested-brandon-gravity.md): shared constants, loaders and statistics.

Every script runs with this experiment's environment (numpy 2.2.6, numba 0.61.2, Python 3.12) under the memory cap:
    systemd-run --user --scope -p MemoryMax=5G env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        NUMBA_NUM_THREADS=4 NUMBA_CACHE_DIR=~/.local/share/correlation2/experiments/gravity/numba_cache \
        research/gravity/.venv/bin/python -I -B research/gravity/<script>.py <args>

Point in time throughout: a force on day t uses prices through the close of t; spread means and sds use t-63..t-1;
attractors come from the window through the refit day in force on t; returns are measured after the close of t.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.append('/home/bhoult/programming/correlation/research/lib')
sys.path.append('/home/bhoult/programming/correlation/research/published_data')
import numpy as np  # noqa: E402
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
BIG = c2eval.WORK / 'gravity'
SRC = {'old': (c2eval.EXPORT_OLD, c2eval.STORE_OLD), 'new': (c2eval.EXPORT_NEW, c2eval.STORE_NEW)}
if os.environ.get('C2_FWD_EXPORT'):   # forward.py: a later download (its export and its store)
    SRC['fwd'] = (Path(os.environ['C2_FWD_EXPORT']), Path(os.environ['C2_FWD_STORE']))
    c2eval.STORE_FOR[str(SRC['fwd'][0])] = SRC['fwd'][1]
    c2eval.SANDBOX.setdefault(str(SRC['fwd'][0]), Path(os.environ.get('C2_FWD_SANDBOX', '')))

FIRST = '2008-01-01'        # open, high and low usable from the first trading day of 2008
MAX_RANGE = 0.5             # bad print: (H - L) / C > 0.5
TOP, TOP_W, TOP_MIN = 500, 63, 57   # universe: 500 most-traded by 63-day median dollar volume (>= 57 of 63 days)
SIG_W, SIG_MIN = 63, 57     # sigma: Garman-Klass over 63 days (>= 57 valid OHLC days), else close-to-close sd (>= 57)
D0_SHARE = 0.9              # development starts on the first day GK sigma exists for >= 90% of the universe
K_ATT, MIN_ATT = 10, 5      # attractors per stock; a force needs >= 5 valid attractor terms
CORR_W, CORR_MIN = 252, 227  # attractor window: 252 daily returns through the refit day, >= 227 common
BETA_M_W, BETA_M_MIN = 252, 227   # beta to the market for z: 252 days through t - 1, >= 227
SPR_W, SPR_MIN = 63, 57     # spread mean and sd: t-63..t-1, >= 57 valid
Z_CLIP = 5.0
W_PRESENT = 0.9             # a kernel sum is valid when >= 90% of its weight has data (missing terms count 0)
SEED = 20261024
BOOT_DRAWS, BOOT_BLOCK, NW_LAGS = 10000, 21, 10
HOLM_FAMILY = 0.0027
T_BAR = 3.0
COSTS = (0.0, 5.0, 20.0)
FWD_START = os.environ.get('C2_FWD_START', '2026-10-20')

# ---------------------------------------------------------------------------------------------------------------
# The grid (fixed by the plan)

G1_PE = [(0, None)] + [(p, e) for p in (1, 2, 3) for e in (0.1, 0.3, 1.0)]       # 10 candidates
G2_NA = [(N, a) for N in (5, 10, 21) for a in (0.5, 1.0, 2.0, 3.0)]               # 12 candidates
G2_BENCH = [(N, np.inf) for N in (5, 10, 21)]                                     # the sharp offset
G2_FLAT = [(N, 0.0) for N in (5, 10, 21)]                                         # reported
G2_ALL = G2_NA + G2_BENCH + G2_FLAT                                               # 18 kernels
G3_EPS = 0.3
G3_NAP = [(N, a, p) for p in (1, 2) for N in (5, 21) for a in (1.0, 2.0)]         # 8 candidates
G3_KERN = [(5, 1.0), (5, 2.0), (21, 1.0), (21, 2.0)]
TAU_MAX = 41


def g1_name(p, e):
    return 'G1 p=0' if p == 0 else f'G1 p={p} eps={e:g}'


def g2_name(N, a):
    return f'G2 N={N} a={"inf" if np.isinf(a) else f"{a:g}"}'


def g3_name(N, a, p):
    return f'G3 N={N} a={a:g} p={p} eps={G3_EPS:g}'


def kernel_weights(N, a):
    """w(tau) = (|tau - N| + 1)^(-a) for tau = 1..2N-1, normalized to sum to 1; a = inf puts all weight on tau = N."""
    tau = np.arange(1, 2 * N)
    if np.isinf(a):
        w = (tau == N).astype(np.float64)
    else:
        w = (np.abs(tau - N) + 1.0) ** (-a)
    return w / w.sum()


def variants():
    """Every force panel forces.py writes: list of dicts (key, stage, name, kind, bench key or None, world)."""
    out = []
    for world in ('real', 'random'):
        out.append({'key': f'g1_spring_{world}', 'stage': 'G1', 'name': 'G1 spring (benchmark)', 'kind': 'bench', 'bench': None, 'world': world})
        for i, (p, e) in enumerate(G1_PE):
            for cl in (False, True):
                out.append({'key': f'g1_{i}{"_c" if cl else ""}_{world}', 'stage': 'G1', 'name': g1_name(p, e) + (' closes only' if cl else ''),
                            'kind': 'closes' if cl else 'cand', 'bench': f'g1_spring_{world}', 'world': world, 'p': p, 'eps': e})
        for i, (N, a) in enumerate(G2_ALL):
            kind = 'cand' if i < 12 else ('bench' if i < 15 else 'flat')
            out.append({'key': f'g2_{i}_{world}', 'stage': 'G2', 'name': g2_name(N, a) + (' (benchmark)' if kind == 'bench' else ''),
                        'kind': kind, 'bench': None if kind == 'bench' else f'g2_{12 + [5, 10, 21].index(N)}_{world}',
                        'world': world, 'N': N, 'a': float(a)})
        for i, (N, a, p) in enumerate(G3_NAP):
            for cl in (False, True):
                bi = G1_PE.index((p, G3_EPS))
                out.append({'key': f'g3_{i}{"_c" if cl else ""}_{world}', 'stage': 'G3', 'name': g3_name(N, a, p) + (' closes only' if cl else ''),
                            'kind': 'closes' if cl else 'cand', 'bench': f'g1_{bi}{"_c" if cl else ""}_{world}', 'world': world,
                            'N': N, 'a': a, 'p': p, 'eps': G3_EPS})
    return out


# Periods by decision day: [start, end)
DEV = ('2008-01-01', '2020-01-01')
EVAL = ('2020-01-01', '2025-06-28')
LATE = ('2023-01-01', '2025-06-28')
ALL_OLD = ('2008-01-01', '2025-06-28')
UNSEEN = ('2025-06-30', '2026-10-20')
PERIODS_OLD = {'2008-2019': DEV, '2020-2025-06': EVAL, '2023-01..2025-06': LATE, '2008-2025-06': ALL_OLD}


def log(msg, fh=None):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n')
        fh.flush()


def decision(msg):
    """Append a timestamped entry to out/decisions_log.txt."""
    OUT.mkdir(exist_ok=True)
    with open(OUT / 'decisions_log.txt', 'a') as fh:
        fh.write(f'\n{time.strftime("%H:%M")}  {msg}\n')


def iso(days, k):
    return c2eval.iso(days[k])


def day_of(days, s):
    return c2eval.day_index(days, s)


def jdump(obj, path):
    def conv(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=conv))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bad_print(O, H, L, C):
    """The plan's bad-print rule (as -6.2): missing when a price is absent or <= 0, (H - L) / C > 0.5,
    H < max(O, C), L > min(O, C), or H = L. Returns (bad, parts)."""
    with np.errstate(all='ignore'):
        miss = ~(np.isfinite(O) & np.isfinite(H) & np.isfinite(L) & np.isfinite(C) & (O > 0) & (H > 0) & (L > 0) & (C > 0))
        rng = ((H - L) / C > MAX_RANGE) & ~miss
        hi = (H < np.maximum(O, C)) & ~miss
        lo = (L > np.minimum(O, C)) & ~miss
        eq = (H == L) & ~miss
    return miss | rng | hi | lo | eq, {'no row': miss, '(H-L)/C>0.5': rng, 'H<max(O,C)': hi, 'L>min(O,C)': lo, 'H=L': eq}


def yahoo_spy_pct(first_day, last_day):
    """SPY's Yahoo total return in percent over a judge period (research/after_costs/group_a/spy.py)."""
    sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_a')
    import spy
    return spy.spy_pct(first_day, last_day)


# ---------------------------------------------------------------------------------------------------------------
# Statistics

def nw_se(x, lags=NW_LAGS):
    """Newey-West standard error of the mean of x (Bartlett weights), NaN dropped."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float('nan')
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        if L >= n:
            break
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(np.sqrt(max(s, 0) / n))


def boot_weights(n, draws=BOOT_DRAWS, block=BOOT_BLOCK, seed=SEED):
    """Circular block bootstrap of whole dates: [draws, n] float32 multiplicities (each row sums to n)."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    W = np.zeros((draws, n), np.float32)
    for a in range(0, draws, 500):
        b = min(draws, a + 500)
        starts = rng.integers(0, n, size=(b - a, nb))
        idx = ((starts[:, :, None] + np.arange(block)[None, None, :]).reshape(b - a, -1)[:, :n]) % n
        for i in range(b - a):
            W[a + i] = np.bincount(idx[i], minlength=n)
    return W


def boot_means(W, X):
    """Bootstrap means of the columns of X [n, k] (NaN = no observation that day) under date weights W [draws, n]."""
    X = np.asarray(X, np.float64)
    if X.ndim == 1:
        X = X[:, None]
    M = np.isfinite(X)
    X0 = np.where(M, X, 0.0)
    num = W.astype(np.float64) @ X0
    den = W.astype(np.float64) @ M.astype(np.float64)
    with np.errstate(all='ignore'):
        return num / den


def summarize(est, draws):
    """Point estimate, bootstrap sd, t, one-sided normal p of t, bootstrap p (null imposed by centring), 90% interval."""
    from math import erfc, sqrt
    draws = np.asarray(draws, np.float64)
    draws = draws[np.isfinite(draws)]
    sd = float(np.std(draws, ddof=1)) if len(draws) > 1 else float('nan')
    t = est / sd if sd > 0 else float('nan')
    p_norm = 0.5 * erfc(t / sqrt(2)) if np.isfinite(t) else float('nan')
    p_boot = float((1 + np.sum(draws - est >= est)) / (len(draws) + 1)) if len(draws) else float('nan')
    lo, hi = (np.percentile(draws, [5, 95]) if len(draws) else (np.nan, np.nan))
    return {'est': float(est), 'se': sd, 't': float(t), 'p': float(p_norm), 'p_boot': p_boot, 'ci90': [float(lo), float(hi)]}


def romano_wolf(est, bm):
    """Romano-Wolf step-down max-t (one-sided, positive). est [k], bm [draws, k] bootstrap means.
    t_k = est_k / se_k with se_k the bootstrap sd; centred draws t*_k = (bm_k - est_k) / se_k. Returns (adjusted p per
    candidate, the stage p = the best candidate's adjusted p, the best index, t)."""
    est = np.asarray(est, np.float64)
    bm = np.asarray(bm, np.float64)
    se = np.nanstd(bm, 0, ddof=1)
    t = est / se
    ts = (bm - est[None, :]) / se[None, :]
    order = np.argsort(-t)
    B = bm.shape[0]
    adj = np.empty(len(est))
    run = 0.0
    for j, k in enumerate(order):
        rest = order[j:]
        mx = np.nanmax(ts[:, rest], 1)
        p = (1 + np.sum(mx >= t[k])) / (B + 1)
        run = max(run, p)
        adj[k] = run
    return adj, float(adj[order[0]]), int(order[0]), t


def holm(pvals, alpha):
    """Holm's step-down at family level alpha. Returns (adjusted p, reject) in input order."""
    p = np.asarray(pvals, np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj.tolist(), (adj < alpha).tolist()


def max_drawdown(r):
    r = np.asarray(r, np.float64)
    r = r[np.isfinite(r)]
    v = np.r_[1.0, np.cumprod(1 + r)]
    peak = np.maximum.accumulate(v)
    return float((v / peak - 1).min())


def ols_nw(y, X, lags=NW_LAGS):
    """OLS of y on [1, X] with Newey-West standard errors. Returns coef, se, t (intercept first)."""
    y = np.asarray(y, np.float64)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, np.float64)])
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    n = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe / n
    for L in range(1, lags + 1):
        G = Xe[L:].T @ Xe[:-L] / n
        S += (1 - L / (lags + 1)) * (G + G.T)
    V = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return {'coef': b.tolist(), 'se': se.tolist(), 't': (b / se).tolist(), 'n': int(n)}
