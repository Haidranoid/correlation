"""Frog in the pan (experiments/+2.0-frog-in-the-pan.md): shared settings and helpers.
The rules are in the .md ("Rules" and "Decisions made while building it").

Signals at the close of day t, from the export's closes (times the dividend correction's total-return factor in the
'divfix' variant):
  - P = forward-filled closes; r_d = log(P_d / P_{d-1}) for d = t-250 .. t-21 (closes t-251 .. t-21: 230 returns);
  - r_d is missing where the raw close of d is missing (a gap day; the move across the gap lands on the next traded
    day) or |r_d| > 0.5 (a bad print);
  - at least 207 (90% of 230) valid returns;
  - PRET = sum of the valid r_d (log 12-1 return); %pos, %neg = shares of the valid days with r_d > 0, r_d < 0
    (zero-return days count in neither, but in the denominator);
  - ID = sign(PRET) x (%neg - %pos).
"""
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

WORK = c2eval.WORK / 'frog_in_the_pan'
WORK.mkdir(parents=True, exist_ok=True)
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}

# signal window: returns of days t-FAR+1 .. t-NEAR (closes t-FAR .. t-NEAR)
FAR, NEAR = 251, 21
NRET = FAR - NEAR            # 230 daily returns
MIN_VALID = 207              # 90% of 230
MAX_LOG = 0.5                # |daily log return| above this is a bad print
# book
STEP = 21                    # the twins calendar: export days 252 + 21k
N_PRET, N_ID = 5, 3
HOLD6 = 6                    # 6-month overlapping hold: 6 cohorts of 21-day periods (126 days)
COST_BPS = 5.0               # per side, on turnover
MIN_NAMES = 100              # a formation needs at least this many eligible stocks
# universe (copied from research/night_and_day/common.py)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500

# test windows: a 21-day period belongs to a window by its END day (ISO, inclusive)
WINDOWS = {
    'test1 2006-2019': ('2006-01-01', '2019-12-31'),
    'test3 2020-2025-06': ('2020-01-01', '2025-06-27'),
    'post 2022-07 .. 2025-06': ('2022-07-01', '2025-06-27'),
    'all 2006-2025-06': ('2006-01-01', '2025-06-27'),
}
UNSEEN = ('2025-06-28', '2026-10-19')   # periods ending after the old export's last day (new export)

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


def divfix_cum(tag, rows=None):
    """Cumulative sum of pubdata.div_fix(tag) along days [N, T] float64 (0 before 2022-07-01); exp() of it is the
    total-return factor to multiply closes by."""
    fix = pubdata.div_fix(tag)
    fix = np.asarray(fix if rows is None else fix[rows], np.float64)
    fix[~np.isfinite(fix)] = 0.0
    np.cumsum(fix, axis=1, out=fix)
    return fix


def prices(tag, E, px, rows=None):
    """Forward-filled closes for holding returns (times the total-return factor for 'divfix'), and the raw-close
    presence mask. rows: optional slice of stocks."""
    C = E['close'] if rows is None else E['close'][rows]
    raw = np.asarray(C, np.float64)
    ok = np.isfinite(raw) & (raw > 0)
    P = ffill(np.where(ok, raw, np.nan))
    if px == 'divfix':
        P *= np.exp(divfix_cum(tag, rows))
    return P, ok


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


def sorts(pret, idv, elig):
    """Group codes for one formation: fifth * 3 + third (0..14), -1 for ineligible stocks. Fifths by PRET rank
    across the eligible stocks (equal counts: floor(5 rank / n)); thirds by ID rank within each fifth
    (floor(3 rank / m)); ties broken by row order (stable sort). Also returns the PRET percentile (rank + 1) / n."""
    N = len(pret)
    g = np.full(N, -1, np.int8)
    pct = np.full(N, np.nan, np.float32)
    m = np.nonzero(elig)[0]
    n = len(m)
    if n < MIN_NAMES:
        return g, pct
    o = m[np.argsort(pret[m], kind='stable')]
    rk = np.arange(n)
    fifth = (N_PRET * rk) // n
    pct[o] = (rk + 1) / n
    for f in range(N_PRET):
        mem = o[fifth == f]
        mem = np.sort(mem)                       # row order, so ties in ID break by row
        oo = mem[np.argsort(idv[mem], kind='stable')]
        third = (N_ID * np.arange(len(oo))) // len(oo)
        g[oo] = f * N_ID + third
    return g, pct


def window_mask(end_iso, a, b):
    end_iso = np.asarray(end_iso)
    return (end_iso >= a) & (end_iso <= b)


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)
