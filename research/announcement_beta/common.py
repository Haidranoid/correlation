"""Announcement-day beta (experiments/-0.6-announcement-beta.md): shared settings, the beta builder, the universe
and the calendar.

Rules (the .md's "Rules" and "Decisions made while building it"):
  - beta[:, d] is indexed by the day d it is USED on: OLS slope of the stock's daily close-to-close log return (export
    closes) on SPY's (store closes) over the return days d-251 .. d-1 (251 days, all known at the close of d-1), with
    at least 227 days where both returns exist. |log return| > 0.5 is treated as a bad print (missing).
  - Universe on c2eval.twins' calendar (export days 252 + 21k), point in time: closes on >= 90% of the trailing 756
    days, median close >= $5 and median dollar volume >= $5M over the trailing 252 days, minus c2eval.excluded().
    Copied from research/night_and_day/common.py. Day d uses the latest rebalance at or before d-1.
  - A days: calendar.tsv (scheduled FOMC statements, CPI releases, Employment Situation releases) that are trading
    days of the export; releases on non-trading days are dropped.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = c2eval.WORK / 'announcement_beta'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

L = 251            # return days in the beta window: d-251 .. d-1
MIN_OBS = 227
MAX_LOG = 0.5      # |daily log move| above this is a bad print (beta inputs and slope inputs)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DECILE = 0.10
COST_BPS = 5.0     # per side; a $1-long/$1-short round trip on an A day = 4 sides = 20 bps
CRASH_MOVE = 0.04  # crash days: trading days in 2008 and 2020 with |SPY close-to-close log return| >= 4%
CRASH_YEARS = (2008, 2020)
PERIODS = {'old': [('2006-2019', '2006-01-01', '2019-12-31'), ('2020-2025-06', '2020-01-01', '2025-06-27'),
                   ('2006-2025-06', '2006-01-01', '2025-06-27')],
           'new': [('unseen 2025-06-30..2026-10-19', '2025-06-30', '2026-10-19')]}

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def dnum(iso):
    return int((np.datetime64(iso) - np.datetime64('1970-01-01')).astype(int))


def spy_close(tag, days):
    """SPY's store close aligned to the export days (float64, NaN where missing)."""
    S = c2eval.load_store(STORES[tag])
    sid = [s['id'] for s in S['syms'] if s['ticker'] == 'SPY' and s['kind'] == 'etf']
    assert len(sid) == 1, sid
    return c2eval.store_panel(S, sid, days, 'close')[0].astype(np.float64)


def log_returns(c):
    """[n, T] float64 close-to-close log returns; NaN where a close is missing/non-positive or |r| > MAX_LOG."""
    c = np.asarray(c, np.float64)
    r = np.full(c.shape, np.nan)
    with np.errstate(all='ignore'):
        r[..., 1:] = np.log(c[..., 1:] / c[..., :-1])
    r[~np.isfinite(r) | (np.abs(r) > MAX_LOG)] = np.nan
    return r


def rolling_beta(close, m, chunk=500):
    """beta [N, T] float32, column d = OLS slope over return days d-L .. d-1 (NaN if fewer than MIN_OBS pairs)."""
    N, T = close.shape
    out = np.full((N, T), np.nan, np.float32)
    mf = np.isfinite(m)
    mz = np.where(mf, m, 0.0)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        r = log_returns(close[a:b])
        M = np.isfinite(r) & mf[None, :]
        x = np.where(M, mz[None, :], 0.0)
        y = np.where(M, r, 0.0)

        def win(X):
            cs = np.zeros((X.shape[0], T + 1))
            np.cumsum(X, 1, out=cs[:, 1:])
            # cs[:, d] = sum over days 0..d-1; window d-L..d-1 = cs[:, d] - cs[:, d-L], for d = L..T-1
            return cs[:, L:T] - cs[:, 0:T - L]

        n = win(M.astype(np.float64))
        sx, sy = win(x), win(y)
        with np.errstate(all='ignore'):
            cxx = win(x * x) - sx * sx / n
            cxy = win(x * y) - sx * sy / n
            g = cxy / cxx
        g[(n < MIN_OBS) | ~np.isfinite(g)] = np.nan
        out[a:b, L:] = g
    return out


def universe(E, tw):
    """[K, N] bool universe on tw['rebal'] days (copied from research/night_and_day/common.py, without the 500)."""
    close, dv = E['close'], E['dollar_volume']
    rebal = tw['rebal']
    N = close.shape[0]
    excl = c2eval.excluded(E, tw)
    uni = np.zeros((len(rebal), N), bool)
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
        uni[k] = fin & (np.nan_to_num(mc) >= MIN_PRICE) & (np.nan_to_num(mv) >= MIN_DV) & ~excl[k]
    return uni


def kidx(rebal, t):
    return int(np.searchsorted(rebal, t, side='right')) - 1


def calendar(days):
    """Flags on the export's trading days: dict fomc, cpi, jobs, A ([T] bool), plus the calendar rows that are not
    trading days of this export (within its span)."""
    rows = [l.rstrip('\n').split('\t') for l in open(HERE / 'calendar.tsv')][1:]
    T = len(days)
    out = {k: np.zeros(T, bool) for k in ('fomc', 'cpi', 'jobs')}
    pos = {int(d): i for i, d in enumerate(days)}
    off = []
    for d, f, c, j in rows:
        i = pos.get(dnum(d))
        if i is None:
            if days[0] <= dnum(d) <= days[-1]:
                off.append(d)
            continue
        out['fomc'][i] |= f == '1'
        out['cpi'][i] |= c == '1'
        out['jobs'][i] |= j == '1'
    out['A'] = out['fomc'] | out['cpi'] | out['jobs']
    out['off'] = off
    return out


def avg_rank(x):
    """Average ranks (ties share the mean rank), 0-based."""
    o = np.argsort(x, kind='stable')
    xs = x[o]
    starts = np.r_[0, np.nonzero(np.diff(xs) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(xs)]
    r = np.empty(len(x))
    r[o] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    return r


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def welch(a, b):
    """Mean of a minus mean of b and Welch's t."""
    a = np.asarray(a, np.float64); a = a[np.isfinite(a)]
    b = np.asarray(b, np.float64); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float('nan'), float('nan')
    d = a.mean() - b.mean()
    return float(d), float(d / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
