"""Risk-managed momentum (experiments/+3.3-risk-managed-momentum.md): shared settings and helpers.
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
import pubdata  # noqa: E402

WORK = c2eval.WORK / 'risk_managed_momentum'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}

# 12-1 momentum (stockai/stockai/baselines/momentum.py): score at d = ln(close[d - 21] / close[d - 252])
SKIP, LOOKBACK = 21, 252
DECILE = 0.10
MIN_RANKED = 100        # a month-end needs at least this many ranked stocks to form a book
# Barroso & Santa-Clara (2015) scaling
VOL_W = 126             # trading days of daily WML returns
TARGET = 0.12           # annualized volatility target
COST_BPS = 5.0          # per side, on turnover
# universe (copied from research/night_and_day/common.py)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500

# periods (return months, inclusive)
TEST1 = ('2006-01', '2019-12')
TEST3 = ('2020-01', '2025-06')
FULL = ('2006-01', '2025-06')
POST22 = ('2022-07', '2025-06')
UNSEEN = ('2025-07', '2026-10')

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


def universe_cached(tag, E):
    p = WORK / f'universe_{tag}.npz'
    if p.exists():
        z = np.load(p)
        return z['rebal'], z['uni'], z['big']
    tw = c2eval.twins(E)
    uni, big = universe(E, tw)
    np.savez(p, rebal=tw['rebal'], uni=uni, big=big)
    return tw['rebal'], uni, big


def month_ends(days):
    """Indices of the last trading day of each calendar month in `days`, and the months as 'YYYY-MM' strings.
    (The export's last day counts as its month's end.)"""
    ym = np.array([c2eval.iso(d)[:7] for d in days])
    last = np.nonzero(ym[:-1] != ym[1:])[0]
    last = np.r_[last, len(days) - 1]
    return last, ym[last]


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


def divfix_factor(tag):
    """exp(cumulative sum of pubdata.div_fix(tag)) [N, T] float64: multiply closes by it to get a total-return index
    (div_fix is 0 before 2022-07-01, so the factor is 1 there)."""
    fix = np.asarray(pubdata.div_fix(tag), np.float64)
    fix[~np.isfinite(fix)] = 0.0
    np.cumsum(fix, axis=1, out=fix)
    np.exp(fix, out=fix)
    return fix


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


def month_mask(months, a, b):
    """Bool mask of 'YYYY-MM' strings within [a, b] inclusive."""
    months = np.asarray(months)
    return (months >= a) & (months <= b)
