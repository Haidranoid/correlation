"""Lottery stocks, MAX (experiments/-3.6-max-lottery.md): shared settings and helpers.
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
import pubdata  # noqa: E402
import numpy as np  # noqa: E402

WORK = c2eval.WORK / 'max_lottery'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}

H = 21                 # holding period and the MAX / IVOL window (t-20..t), trading days
MIN_OBS = 15           # valid daily returns needed in the 21-day window (Bali et al.'s 15-observation rule)
MAX_LOG = 0.5          # |daily log return| above this is a bad print: missing
BETA_W, BETA_MIN = 252, 200
MOM_FROM, MOM_TO = 252, 21   # 12-1: P[t-21] / P[t-252] - 1
DECILE = 0.10
COST_BPS = 5.0         # per side
NW_LAGS = 4
WINSOR = (0.01, 0.99)
# universe (copied from research/night_and_day/common.py)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500
MIN_UNIV = 100

# periods by the first return day of a formation's holding window (inclusive ISO dates)
TEST1 = ('2006-01-01', '2019-12-31')
TEST3 = ('2020-01-01', '2025-06-27')
UNSEEN = ('2025-06-30', '2026-10-19')
PERIODS = [('2006-2009', '2006-01-01', '2009-12-31'), ('2010-2014', '2010-01-01', '2014-12-31'),
           ('2015-2019', '2015-01-01', '2019-12-31'), ('2020-2025-06', '2020-01-01', '2025-06-27'),
           ('2022-07..2025-06', '2022-07-01', '2025-06-27'), ('All 2006..2025-06', '2006-01-01', '2025-06-27')]
DIV_FROM = '2022-07-01'

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def universe(E, tw):
    """[K, N] bool universe and [K, N] bool 500 largest on tw['rebal'] days, plus [K, N] float32 63-day median dollar
    volume. Copied from research/night_and_day/common.py (universe), with the 63-day median kept."""
    close, dv = E['close'], E['dollar_volume']
    rebal = tw['rebal']
    N = close.shape[0]
    excl = c2eval.excluded(E, tw)
    uni = np.zeros((len(rebal), N), bool)
    big = np.zeros((len(rebal), N), bool)
    dv63 = np.full((len(rebal), N), np.nan, np.float32)
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
        dv63[k] = mv63
    return uni, big, dv63


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


def log_returns(close):
    """Daily log returns from closes (both closes present), |r| > MAX_LOG -> NaN. float64 [N, T]."""
    c = np.asarray(close, np.float64)
    r = np.full(c.shape, np.nan)
    with np.errstate(all='ignore'):
        r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    r[~np.isfinite(r) | (np.abs(r) > MAX_LOG)] = np.nan
    return r


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


def mean_t(x, nw_lags=NW_LAGS):
    """Mean, plain t, Newey-West t (Bartlett, nw_lags) and n of a series (NaNs dropped)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float('nan'), float('nan'), float('nan'), n
    m = x.mean()
    t = m / (x.std(ddof=1) / np.sqrt(n))
    u = x - m
    v = (u @ u) / n
    for L in range(1, min(nw_lags, n - 1) + 1):
        v += 2 * (1 - L / (nw_lags + 1)) * (u[L:] @ u[:-L]) / n
    tnw = m / np.sqrt(v / n) if v > 0 else float('nan')
    return float(m), float(t), float(tnw), n


def pct_rank(x):
    """Percentile ranks in [0, 1] of the finite values (average ranks for ties); NaN stays NaN."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    v = x[ok]
    n = len(v)
    if n < 2:
        return out
    o = np.argsort(v, kind='mergesort')
    vs = v[o]
    starts = np.r_[0, np.nonzero(np.diff(vs) != 0)[0] + 1]
    ends = np.r_[starts[1:], n]
    r = np.empty(n)
    r[o] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    out[ok] = r / (n - 1)
    return out


def winsor(x, q=WINSOR):
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return x
    lo, hi = np.quantile(x[ok], q)
    x[ok] = np.clip(x[ok], lo, hi)
    return x


def in_period(days_iso, a, b):
    days_iso = np.asarray(days_iso)
    return (days_iso >= a) & (days_iso <= b)
