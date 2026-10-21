"""The 52-week high (experiments/-7.4-52-week-high.md): shared settings and helpers.
The rules are in the .md ("Rules" and "Decisions made while building it")."""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

WORK = c2eval.WORK / 'week52_high'
WORK.mkdir(parents=True, exist_ok=True)
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}

# signal
HIGH_W = 252          # PTH = close_t / max close over t-251..t
HIGH_MIN = 227        # at least 90% of the window's closes present
MOM_FAR, MOM_NEAR = 252, 21   # 12-1 momentum: close[t-21] / close[t-252] - 1
JT6 = 126             # GH's Table I JT strategy: past 6-month return (reported only)
SKIP = 5              # the one-week-skip variant: rank at t-5, hold from t
# book
FRAC = 0.30           # top / bottom 30%
HOLD = 6              # periods (6 x 21 = 126 trading days)
COST_BPS = 5.0        # per side, at formation and at exit
MIN_NAMES = 100       # a formation day needs at least this many ranked stocks
# universe (copied from research/night_and_day/common.py)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500

# test windows: a period belongs to a window by its end day (ISO, inclusive)
TEST1 = ('2006-01-01', '2019-12-31')
TEST3 = ('2020-01-01', '2025-06-27')
UNSEEN_FROM = '2025-06-30'

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def universe(E, tw):
    """[K, N] bool universe and [K, N] bool 500 largest on tw['rebal'] days. Copied from
    research/night_and_day/common.py (universe)."""
    close, dv = E['close'], E['dollar_volume']
    rebal = tw['rebal']
    N = close.shape[0]
    excl = c2eval.excluded(E, tw)
    uni = np.zeros((len(rebal), N), bool)
    big = np.zeros((len(rebal), N), bool)
    for k, t0 in enumerate(rebal):
        if t0 < DATA_W - 1:
            continue
        c = np.asarray(close[:, t0 - DATA_W + 1:t0 + 1], np.float64)
        fin = np.isfinite(c).mean(1) >= DATA_FRAC
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mc = np.nanmedian(c[:, -MED_W:], 1)
            v = np.asarray(dv[:, t0 - MED_W + 1:t0 + 1], np.float64)
            mv = np.nanmedian(v, 1)
            mv63 = np.nanmedian(v[:, -DV_LARGE_W:], 1)
        u = fin & (np.nan_to_num(mc) >= MIN_PRICE) & (np.nan_to_num(mv) >= MIN_DV) & ~excl[k]
        uni[k] = u
        cand = np.nonzero(u)[0]
        o = cand[np.argsort(-np.nan_to_num(mv63[cand], nan=-1.0), kind='stable')]
        big[k, o[:N_LARGE]] = True
    return uni, big


def rolling_max(close, w=HIGH_W, min_n=HIGH_MIN, chunk=1000):
    """[N, T] float32: max of the finite closes over t-w+1..t, NaN where fewer than min_n of them are finite or
    close_t itself is missing. Sparse-table trick: with 2^p <= w < 2^(p+1), the window is the union of the two
    2^p-day blocks ending at t and at t-w+2^p, so max = max(B_p[t], B_p[t-w+2^p]); no [N, T, w] intermediate."""
    N, T = close.shape
    p = w.bit_length() - 1
    s = 1 << p
    out = np.full((N, T), np.nan, np.float32)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        c = np.asarray(close[a:b], np.float32)
        fin = np.isfinite(c)
        B = np.where(fin, c, -np.inf).astype(np.float32)   # B_0[t] = x[t]
        for q in range(p):                                  # B_{q+1}[t] = max(B_q[t], B_q[t-2^q])
            h = 1 << q
            B[:, h:] = np.maximum(B[:, h:], B[:, :-h].copy())
        res = B.copy()
        off = w - s
        if off > 0:
            res[:, off:] = np.maximum(B[:, off:], B[:, :-off])
        cs = np.zeros((b - a, T + 1), np.int32)
        np.cumsum(fin, 1, out=cs[:, 1:])
        n = np.zeros((b - a, T), np.int32)
        n[:, w - 1:] = cs[:, w:] - cs[:, :-w]
        ok = (n >= min_n) & fin & np.isfinite(res)
        out[a:b] = np.where(ok, res, np.nan)
    return out


def ffill(close):
    """Forward-filled closes (leading NaNs stay NaN). Copied from research/betting_against_beta/common.py."""
    c = np.asarray(close, np.float64)
    ok = np.isfinite(c)
    idx = np.where(ok, np.arange(c.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    P = np.take_along_axis(c, idx, 1)
    P[~ok.any(1)] = np.nan
    first = ok.argmax(1)
    P[np.arange(c.shape[1])[None, :] < first[:, None]] = np.nan
    return P


def ols(y, X=None, nw_lags=None):
    """OLS of y on [1, X]. Plain standard errors, or Newey-West (Bartlett) with nw_lags. Returns coef, t, n, r2.
    Copied from research/betting_against_beta/common.py."""
    y = np.asarray(y, float)
    cols = [np.ones(len(y))]
    if X is not None:
        cols.append(np.asarray(X, float).reshape(len(y), -1))
    X = np.column_stack(cols)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    if nw_lags is None:
        V = XtXi * (e @ e / (n - k))
    else:
        u = X * e[:, None]
        S = u.T @ u
        for L in range(1, nw_lags + 1):
            w = 1 - L / (nw_lags + 1)
            G = u[L:].T @ u[:-L]
            S += w * (G + G.T)
        V = XtXi @ S @ XtXi * n / (n - k)
    se = np.sqrt(np.diag(V))
    sst = (y - y.mean()) @ (y - y.mean())
    r2 = 1 - (e @ e) / sst if sst > 0 else float('nan')
    return b, b / se, n, r2


def iso_index(days, iso):
    return c2eval.day_index(days, iso)
