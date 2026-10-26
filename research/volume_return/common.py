"""Volume and return (experiments/untested-volume-return.md): shared constants, panels and statistics.

Llorente, Michaely, Saar & Wang (2002): R(t+1) = C0 + C1 R(t) + C2 V(t) R(t), V = detrended log volume; C2 is higher
where informed trading matters more. R1 replicates the cross-sectional pattern (dollar-volume thirds, wide universe);
M1 asks whether the V*R term adds to a reversal book traded a day late (judge-like universe).

Every script runs with numpy 2.2.6 / numba 0.61.2 (research/gravity/.venv) under the memory cap:
    systemd-run --user --scope -p MemoryMax=5G env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        NUMBA_NUM_THREADS=4 research/gravity/.venv/bin/python -I -B research/volume_return/<script>.py <args>

Point in time: everything known at day t uses closes and volumes through the close of t. V(t) needs the whole day's
volume, so trades are at the close of t + 1 (the judge's delay=1). R1 is descriptive and uses R(t + 1).
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
import numba as nb  # noqa: E402
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
BIG = c2eval.WORK / 'volume_return'
SRC = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORE = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}
# forward.py: a later download, named by environment variables (its export and its store)
if os.environ.get('C2_FWD_EXPORT'):
    SRC['fwd'] = Path(os.environ['C2_FWD_EXPORT'])
    STORE['fwd'] = Path(os.environ['C2_FWD_STORE'])
    c2eval.STORE_FOR[str(SRC['fwd'])] = STORE['fwd']

# ---- the plan's fixed rules -----------------------------------------------------------------------------------------
R_BAD = 0.5                 # |simple daily return| > 0.5 is a bad print (missing)
V_WIN, V_MIN = 200, 160     # V(t) = ln sv(t) - mean ln sv over valid days in t-200..t-1, >= 160 valid
BROKEN = 0.01               # sv(t) < 1% of the median sv over t-200..t-1: broken print (missing, not used later)
V_CLIP = 5.0
CAL_W, CAL_FRAC = 252, 0.9  # universes: 252 days through the calendar day, >= 90% closes (and >= 90% volume for U_W)
FLOOR_PRICE, FLOOR_PRICE_LOW = 5.0, 1.0
FLOOR_DV_WIDE, FLOOR_DV_JUDGE = 250_000.0, 5e6
TOP, TOP_W = 500, 63
MIN_PAIRS, MIN_PAIRS_HALF, MIN_PAIRS_DRAW = 200, 100, 100
BLOCK = 21
DRAWS = 10000
SEED = 20261025
WIN_LO, WIN_HI = 1.0, 99.0  # winsorizing percentiles
M1_MIN_GROUP, M1_MIN_ORIGINS = 100, 252
COST = 5e-4                 # 5 bps a side
COSTS_BPS = (0.0, 5.0, 20.0)
NW_LAGS = 5
HOLM_FAMILY, T_BAR, T_HIGH = 0.0228, 2.0, 3.0   # the README's published-strategy bar (coordinator's edit of the plan)
SPIKE_V = 1.5
DEV = ('2008-01-01', '2019-12-31')
CHECK = ('2020-01-01', '2025-06-27')
SCORE_START = '2008-01-02'
UNSEEN_START = '2025-06-30'
# 20 known splits (an earlier test's list)
SPLITS = [('AAPL', '2014-06-09', 7), ('AAPL', '2020-08-31', 4), ('TSLA', '2020-08-31', 5), ('TSLA', '2022-08-25', 3),
          ('NVDA', '2021-07-20', 4), ('NVDA', '2024-06-10', 10), ('GOOGL', '2022-07-18', 20), ('AMZN', '2022-06-06', 20),
          ('NFLX', '2015-07-15', 7), ('V', '2015-03-19', 4), ('WMT', '2024-02-26', 3), ('CMG', '2024-06-26', 50),
          ('AVGO', '2024-07-15', 10), ('SHW', '2021-04-01', 3), ('ISRG', '2021-10-05', 3), ('DXCM', '2022-06-13', 4),
          ('PANW', '2022-09-14', 3), ('LRCX', '2024-10-03', 10), ('SMCI', '2024-10-01', 10), ('ORLY', '2025-06-10', 15)]


def log(path, *lines):
    txt = '\n'.join(str(x) for x in lines)
    print(txt, flush=True)
    with open(path, 'a') as f:
        f.write(txt + '\n')


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tagdir(tag):
    d = BIG / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def export(tag, arrays=('close', 'dollar_volume')):
    return c2eval.load_export(SRC[tag], arrays=arrays)


def market_simple(tag):
    """The export market's simple daily return: exp(equal-weight mean daily log return of the liquid stocks) - 1."""
    m = np.load(Path(SRC[tag]) / 'market.npy')
    return np.expm1(m)


# ---- panels ---------------------------------------------------------------------------------------------------------

def simple_returns(close):
    """[N, T] float32: C(t)/C(t-1) - 1; NaN if a close is missing or <= 0, or |R| > R_BAD."""
    N, T = close.shape
    R = np.full((N, T), np.nan, np.float32)
    for a in range(0, N, 1000):
        c = np.asarray(close[a:a + 1000], np.float64)
        with np.errstate(all='ignore'):
            r = c[:, 1:] / c[:, :-1] - 1.0
            r[~((c[:, 1:] > 0) & (c[:, :-1] > 0))] = np.nan
        r[~np.isfinite(r) | (np.abs(r) > R_BAD)] = np.nan
        R[a:a + 1000, 1:] = r
    return R


def share_volume(close, dv):
    """[N, T] float64 ln(share volume) = ln(dollar volume / close); NaN where either is missing or <= 0."""
    N, T = close.shape
    out = np.full((N, T), np.nan)
    for a in range(0, N, 1000):
        c = np.asarray(close[a:a + 1000], np.float64)
        d = np.asarray(dv[a:a + 1000], np.float64)
        with np.errstate(all='ignore'):
            x = np.log(d / c)
        x[~((c > 0) & (d > 0) & np.isfinite(c) & np.isfinite(d))] = np.nan
        out[a:a + 1000] = x
    return out


@nb.njit(parallel=True, cache=False)
def _v_kernel(lnsv, win, vmin, broken_ln, clip):
    N, T = lnsv.shape
    V = np.full((N, T), np.nan, np.float32)
    brk = np.zeros((N, T), np.bool_)
    for i in nb.prange(N):
        usable = np.zeros(T, np.bool_)
        buf = np.empty(win)
        for t in range(T):
            x = lnsv[i, t]
            if not np.isfinite(x):
                continue
            a = t - win if t - win > 0 else 0
            m = 0
            s = 0.0
            for u in range(a, t):
                if usable[u]:
                    buf[m] = lnsv[i, u]
                    s += lnsv[i, u]
                    m += 1
            if m >= vmin:
                med = np.median(buf[:m])
                if x < med + broken_ln:
                    brk[i, t] = True        # broken print: missing, and not used in later windows
                    continue
                v = x - s / m
                if v > clip:
                    v = clip
                elif v < -clip:
                    v = -clip
                V[i, t] = v
            usable[t] = True                # too little history to test: kept for later windows
    return V, brk


def detrended_volume(lnsv):
    """V [N, T] float32 and the broken-print flags [N, T] bool (the plan's rules)."""
    return _v_kernel(np.ascontiguousarray(lnsv), V_WIN, V_MIN, np.log(BROKEN), V_CLIP)


def pct_rank(x):
    """Percentile ranks (rank - 0.5) / n with ties averaged, for a 1-D finite array."""
    n = len(x)
    if n == 0:
        return np.zeros(0)
    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks - 0.5) / n


def thirds_of(g):
    """0 = low third (G < 1/3), 1 = middle, 2 = high (G >= 2/3); -1 where g is NaN."""
    out = np.full(g.shape, -1, np.int8)
    ok = np.isfinite(g)
    out[ok & (g < 1 / 3)] = 0
    out[ok & (g >= 1 / 3) & (g < 2 / 3)] = 1
    out[ok & (g >= 2 / 3)] = 2
    return out


def load_panels(tag):
    d = tagdir(tag)
    R = np.load(d / 'R.npy', mmap_mode='r')
    V = np.load(d / 'V.npy', mmap_mode='r')
    cal = dict(np.load(d / 'cal.npz'))
    return R, V, cal


def cal_index(rebal, t):
    """Index of the latest calendar day at or before export day t (-1 if none)."""
    return int(np.searchsorted(rebal, t, side='right')) - 1


def day_range(days, a_iso, b_iso):
    """Export day indices with dates in [a_iso, b_iso]."""
    a = c2eval.day_index(days, a_iso)
    b = int(np.searchsorted(days, (np.datetime64(b_iso) - np.datetime64('1970-01-01')).astype(int), side='right')) - 1
    return a, b


# ---- statistics -----------------------------------------------------------------------------------------------------

def nw_t(x, lags=NW_LAGS):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float('nan')
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float('nan')


def norm_sf(t):
    from math import erfc, sqrt
    return 0.5 * erfc(t / sqrt(2)) if np.isfinite(t) else float('nan')


def circular_block_means(x, draws=DRAWS, block=BLOCK, seed=SEED):
    """Means of circular block bootstrap resamples of the series x (finite values, in order)."""
    x = np.asarray(x, float)
    n = len(x)
    nb_ = max(1, n // block)
    xx = np.concatenate([x, x[:block]])
    cs = np.concatenate([[0.0], np.cumsum(xx)])
    bsum = cs[block:block + n] - cs[:n]          # sum of the block starting at each position (circular)
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for a in range(0, draws, 1000):
        k = min(1000, draws - a)
        st = rng.integers(0, n, size=(k, nb_))
        out[a:a + k] = bsum[st].sum(1) / (nb_ * block)
    return out


def boot_summary(est, boots):
    """t = est / bootstrap sd; one-sided normal p; null-centred percentile p; 90% interval."""
    sd = float(np.nanstd(boots, ddof=1))
    t = est / sd if sd > 0 else float('nan')
    b = boots[np.isfinite(boots)]
    return {'est': float(est), 'sd': sd, 't': float(t), 'p': norm_sf(t),
            'p_boot': float(np.mean((b - b.mean()) >= est)) if len(b) else float('nan'),
            'lo90': float(np.percentile(b, 5)) if len(b) else float('nan'),
            'hi90': float(np.percentile(b, 95)) if len(b) else float('nan')}


def holm(ps, family=HOLM_FAMILY):
    """Holm step-down: returns (adjusted p's, pass flags at the family level)."""
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    passed = np.zeros(m, bool)
    run = 0.0
    alive = True
    for i, j in enumerate(order):
        a = min(1.0, (m - i) * ps[j])
        run = max(run, a)
        adj[j] = run
        if alive and ps[j] < family / (m - i):
            passed[j] = True
        else:
            alive = False
    return adj, passed


def ols(X, y):
    """Least squares coefficients (X [n, p], y [n]); NaN if rank-deficient."""
    try:
        b, res, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan)
    if rank < X.shape[1]:
        return np.full(X.shape[1], np.nan)
    return b


def winsor(x, lo=WIN_LO, hi=WIN_HI):
    """Clip a 1-D array at its own lo/hi percentiles (finite values only define them)."""
    f = np.isfinite(x)
    if f.sum() < 3:
        return x
    a, b = np.percentile(x[f], [lo, hi])
    return np.clip(x, a, b)
