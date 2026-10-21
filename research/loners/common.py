"""Loners (experiments/-4.3-loners.md): shared settings and helpers.

Rules are in the .md ("Rules" and "Decisions made while building it"). Run every script under the memory cap:
    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- env OMP_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 -I <script>.py <old|new>
"""
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/loners'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

# calendar: c2eval.twins' rebalance days, export indices 252 + 21k
STEP = 21
WIN = 252            # correlation window (daily returns d-251..d)
MIN_FRAC = 0.9       # returns needed in the window (stocks and funds), as c2eval.twins
# judge-like liquid set (stockai/judge/src/holdout.rs liquid_universe, made point in time)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
# eligible funds, c2eval.twins' fund screens
FUND_MIN_DV, FUND_MAX_VOL, FUND_MAX_ABS = 5e6, 0.75, 0.5
SINGLE_BEST, SINGLE_GAP = 0.6, 0.12
# volatility (betting against correlation's definition)
VOL_W, VOL_MIN = 252, 120
# BAC inputs (copied from research/betting_against_correlation/common.py)
COR_W, COR_MIN = 1260, 750
COST_BPS = 5.0
N_LARGE = 500
MIN_CELL = 9         # a dollar-volume x volatility cell is used only with >= 9 stocks (>= 3 per leg)
RF_CSV = Path.home() / '.local/share/correlation2/external/series/fred_dgs3mo.csv'
SEC_MAP = Path.home() / '.local/share/correlation2/experiments/imposters/map/sec_tickers.json'


def log(*a):
    print(*a, flush=True)


def calendar(T):
    return np.arange(WIN, T, STEP)


def rank(x):
    return np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort')


def upper_median(X):
    """Row-wise upper median of the finite values ((k // 2)-th of k sorted, as the judge takes it); 0 if none."""
    S = np.sort(X, axis=1)
    k = np.isfinite(X).sum(1)
    out = S[np.arange(len(S)), np.minimum(k // 2, X.shape[1] - 1)]
    out[k == 0] = 0.0
    return out


def log_returns(close, max_abs=None):
    c = np.asarray(close, np.float64)
    r = np.full(c.shape, np.nan)
    with np.errstate(all='ignore'):
        r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    r[~np.isfinite(r)] = np.nan
    if max_abs is not None:
        r[np.abs(r) > max_abs] = np.nan
    return r


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


def standardize(X):
    """c2eval.twins' standardization: (x - mean) / sd over the row's finite values, missing -> 0, unit norm."""
    with np.errstate(all='ignore'):
        m = np.nanmean(X, 1, keepdims=True)
        s = np.nanstd(X, 1, keepdims=True)
        Z = (X - m) / s
        Z[~np.isfinite(Z)] = 0.0
        nrm = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = Z / nrm
    Z[~np.isfinite(Z)] = 0.0
    return Z


def market_relative(R, m):
    """R [n, w] daily log returns, m [w] the export's market over the same days. Residual r - beta * m with beta the
    in-window OLS slope over the days where both are finite (as the groups map's standardized(relative=true))."""
    ok = np.isfinite(R) & np.isfinite(m)[None, :]
    k = ok.sum(1)
    with np.errstate(all='ignore'):
        Rz = np.where(ok, R, 0.0)
        Mz = np.where(ok, m[None, :], 0.0)
        mr = Rz.sum(1) / k
        mm = Mz.sum(1) / k
        cov = (Rz * Mz).sum(1) / k - mr * mm
        var = (Mz * Mz).sum(1) / k - mm * mm
        beta = cov / var
    beta[~np.isfinite(beta)] = 0.0
    out = R - beta[:, None] * m[None, :]
    out[~ok] = np.nan
    return out, beta


def rf_daily(days):
    """Daily risk-free return per export day: FRED DGS3MO % a year / 100 / 252, latest value at or before the day
    (copied from research/betting_against_correlation/common.py)."""
    rows = [l.strip().split(',') for l in open(RF_CSV)][1:]
    d = np.array([(np.datetime64(r[0]) - np.datetime64('1970-01-01')).astype(int) for r in rows if r[1] not in ('', '.')])
    v = np.array([float(r[1]) for r in rows if r[1] not in ('', '.')])
    pos = np.searchsorted(d, days, side='right') - 1
    out = v[np.maximum(pos, 0)] / 100.0 / 252.0
    out[pos < 0] = np.nan
    return out


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


def ols_nw(y, X, lags=5):
    """OLS of y on [1, X] with Newey-West (Bartlett) standard errors (copied from betting_against_correlation)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + ([np.asarray(X, float).reshape(len(y), -1)] if X is not None else []))
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, b / se, n, r2


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def sec_names():
    import json
    return {r['ticker'].upper(): r['name'] for r in json.loads(SEC_MAP.read_text())}


def day_num(iso):
    return int((np.datetime64(iso) - np.datetime64('1970-01-01')).astype(int))
