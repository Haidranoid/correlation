"""Betting against beta (experiments/-4.4-betting-against-beta.md): shared settings and helpers.
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

WORK = c2eval.WORK / 'betting_against_beta'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

# beta (Frazzini & Pedersen 2014)
VOL_W, VOL_MIN = 252, 120      # sigma: one year of daily log returns, >= 120
COR_W, COR_MIN = 1260, 750     # rho: five years of overlapping 3-day log returns, >= 750
SHRINK_W, SHRINK_PRIOR = 0.6, 0.4
# universe (copied from research/night_and_day/common.py)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500
MIN_UNIV = 100                 # a month-end needs at least this many stocks with a beta to form a book
COST_BPS = 5.0                 # per side
RF_CSV = Path.home() / '.local/share/correlation2/external/series/fred_dgs3mo.csv'

# periods (return months, inclusive); the first beta exists at the close of 2008-02 (see the .md, Decisions)
TEST1 = ('2008-03', '2019-12')
TEST3 = ('2020-01', '2025-06')
UNSEEN = ('2025-07', '2026-10')

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def rf_daily(days):
    """Daily risk-free return per export day: DGS3MO % a year / 100 / 252, the latest value at or before the day
    (research/lazy_prices/common.py, copied)."""
    rows = [l.strip().split(',') for l in open(RF_CSV)][1:]
    d = np.array([(np.datetime64(r[0]) - np.datetime64('1970-01-01')).astype(int) for r in rows if r[1] not in ('', '.')])
    v = np.array([float(r[1]) for r in rows if r[1] not in ('', '.')])
    pos = np.searchsorted(d, days, side='right') - 1
    out = v[np.maximum(pos, 0)] / 100.0 / 252.0
    out[pos < 0] = np.nan
    return out


def spy_close(tag, days):
    """SPY's store close aligned to the export days (float64, NaN where missing)."""
    S = c2eval.load_store(STORES[tag])
    sid = [s['id'] for s in S['syms'] if s['ticker'] == 'SPY' and s['kind'] == 'etf']
    assert len(sid) == 1, sid
    return c2eval.store_panel(S, sid, days, 'close')[0].astype(np.float64)


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


def month_ends(days):
    """Indices of the last trading day of each calendar month in `days`, and the months as 'YYYY-MM' strings."""
    ym = np.array([c2eval.iso(d)[:7] for d in days])
    last = np.nonzero(ym[:-1] != ym[1:])[0]
    last = np.r_[last, len(days) - 1]
    return last, ym[last]


def ffill(close):
    """Forward-filled closes (leading NaNs stay NaN)."""
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
    """OLS of y on [1, X]. Plain standard errors, or Newey-West (Bartlett) with nw_lags. Returns coef, t, n, r2."""
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
