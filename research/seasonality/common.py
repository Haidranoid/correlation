"""Return seasonality (experiments/-0.2-seasonality.md, Heston & Sadka 2008): shared settings and helpers.

Rules are in the .md ("Rules" and "Decisions made while building it"). Run every script under the memory cap:
    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- env OMP_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 -I -B <script>.py <old|new>
"""
import os
import sys
import warnings
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = c2eval.WORK / 'seasonality'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}

# judge-like liquid set (copied from research/night_and_day/common.py), on c2eval.twins' calendar 252 + 21k
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500

# the signal
YEARS = 10          # annual lags 12, 24, ..., 120 months
MIN_YEARS = 5       # at least 5 of the 10 same-month returns present
MIN_OTHER = 55      # contrast: at least 55 of the 110 other-month returns present (half, as 5 of 10)
START_MIN = 300     # the test starts at the first holding month with >= 300 universe stocks holding a signal
DECILE = 0.10
COST_BPS = 5.0      # per side, on dollars traded


def log(*a):
    print(*a, flush=True)


def universe(E, tw):
    """[K, N] bool universe and [K, N] bool 500 largest on tw['rebal'] days (copied from research/night_and_day)."""
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


def month_of(days):
    """Calendar month number (year * 12 + month - 1) of each export day."""
    d = np.datetime64('1970-01-01') + days.astype('timedelta64[D]')
    y = d.astype('datetime64[Y]').astype(int) + 1970
    m = d.astype('datetime64[M]').astype(int) % 12
    return y * 12 + m


def month_label(mn):
    return f'{mn // 12}-{mn % 12 + 1:02d}'


def ffill(close):
    """Forward-filled closes (leading NaNs stay NaN)."""
    c = np.asarray(close, np.float64)
    ok = np.isfinite(c)
    idx = np.where(ok, np.arange(c.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    P = np.take_along_axis(c, idx, 1)
    first = ok.argmax(1)
    P[np.arange(c.shape[1])[None, :] < first[:, None]] = np.nan
    P[~ok.any(1)] = np.nan
    return P


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def nw_t(x, lags=3):
    """Newey-West (Bartlett) t of the mean of a series (NaNs dropped)."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lags + 3:
        return float('nan')
    u = x - x.mean()
    v = (u @ u) / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / n
    return float(x.mean() / np.sqrt(v / n))


def ols(y, X):
    """OLS of y on [1, X] with plain standard errors. Returns coef, t, n, r2."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + ([np.asarray(X, float).reshape(len(y), -1)] if X is not None else []))
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    s2 = (e @ e) / (n - k)
    se = np.sqrt(np.diag(XtXi) * s2)
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())) if n > 1 else float('nan')
    return b, b / se, n, r2


def day_num(iso):
    return int((np.datetime64(iso) - np.datetime64('1970-01-01')).astype(int))
