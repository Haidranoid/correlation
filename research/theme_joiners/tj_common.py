"""Shared settings and helpers for the theme-joiners experiment (experiments/-3.5-theme-joiners.md).

The groups map is breadth thrust's (research/breadth_thrust/bt_refits.py, its clustering copied from phantom
portfolios' pp_common.py): 30 groups of the 1,500 most-traded stocks, refit every 21 trading days on the twins
calendar, minus c2eval.excluded. Its cached refits are read, read-only, from
~/.local/share/correlation2/experiments/breadth_thrust/old/refits.pkl. The unseen months' refits are rebuilt here by
tj_refits.py (a copy of bt_refits.py's universe and grouping steps).

numpy only; system python3.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
BIG = Path.home() / '.local/share/correlation2/experiments/theme_joiners'
BT_REFITS_OLD = Path.home() / '.local/share/correlation2/experiments/breadth_thrust/old/refits.pkl'
BT_REFITS_NEW = Path.home() / '.local/share/correlation2/experiments/breadth_thrust/new/refits.pkl'
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}

# Fixed rules (experiments/-3.5-theme-joiners.md, "Rules (fixed before looking at results)")
K_GROUPS = 30
N_HOT = 3             # top 3 groups = hot, bottom 3 = cold
RECENT = 63           # t-62 .. t
OLDER_END = 63        # older window t-314 .. t-63
OLDER_START = 314
MIN_RECENT = 55       # returns needed in the recent window
MIN_OLDER = 200       # returns needed in the older window
HOLD = 21             # outcome close t -> t+21
MOM_LONG, MOM_SKIP = 252, 21   # 12-1 return: close t-252 -> close t-21
COST_BPS = 5.0        # per side, the book

# Decisions (see the .md, "Decisions made while building it")
STALE = 5             # a close may be up to 5 trading days stale (last finite close on or before the day)
H_MIN_SHARE = 0.5     # H / C need a return from >= 50% of the 3 groups' members on a day
MIN_OBS = 100         # cross-sections with fewer stocks are skipped (none expected)

P1 = ('2006-01-01', '2019-12-31')
P2 = ('2020-01-01', '2025-06-27')
UNSEEN = ('2025-06-30', '2026-10-19')

# Sanity-check dates for the hot/cold groups and the top 10 joiners, chosen before any outcome was computed:
# the first refit on or after each date.
REPORT_DATES_OLD = ['2007-01-02', '2009-03-02', '2014-01-02', '2020-04-01', '2021-01-04', '2025-06-13']
REPORT_DATES_NEW = ['2026-01-02', '2026-07-01']


def big(*parts):
    p = BIG.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def iso(day):
    return c2eval.iso(day)


def load_refits(tag):
    path = BT_REFITS_OLD if tag == 'old' else BIG / 'new' / 'refits.pkl'
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def masked_corr(X, y):
    """Row-wise Pearson correlation of X [n, W] (NaN = missing) with y [W] (NaN = missing), over the days where both
    are finite. Returns (corr [n], days used [n])."""
    f = np.isfinite(X) & np.isfinite(y)[None, :]
    n = f.sum(1).astype(np.float64)
    x0 = np.where(f, X, 0.0)
    y0 = np.where(f, y[None, :], 0.0)
    sx, sy = x0.sum(1), y0.sum(1)
    with np.errstate(all='ignore'):
        cxy = (x0 * y0).sum(1) - sx * sy / n
        vx = (x0 * x0).sum(1) - sx * sx / n
        vy = (y0 * y0).sum(1) - sy * sy / n
        r = cxy / np.sqrt(vx * vy)
    r[~np.isfinite(r)] = np.nan
    return r, f.sum(1)


def masked_partial_corr(X, y, z):
    """Row-wise partial correlation of X and y given z (each with an intercept), over the days where all three are
    finite. Returns (corr [n], days used [n])."""
    keep = np.isfinite(y) & np.isfinite(z)
    Xk = np.where(keep[None, :], X, np.nan)
    yk = np.where(keep, y, np.nan)
    zk = np.where(keep, z, np.nan)
    rxy, n = masked_corr(Xk, yk)
    rxz, _ = masked_corr(Xk, zk)
    # r(y, z) over each row's own days
    f = np.isfinite(Xk)
    n_ = f.sum(1).astype(np.float64)
    y0 = np.where(f, yk[None, :], 0.0)
    z0 = np.where(f, zk[None, :], 0.0)
    sy, sz = y0.sum(1), z0.sum(1)
    with np.errstate(all='ignore'):
        ryz = ((y0 * z0).sum(1) - sy * sz / n_) / np.sqrt(((y0 * y0).sum(1) - sy * sy / n_) * ((z0 * z0).sum(1) - sz * sz / n_))
        r = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    r[~np.isfinite(r)] = np.nan
    return r, n


def last_close(L, rows, d, lo):
    """Log close of `rows` at day d: the last finite close on days max(lo, d-STALE) .. d (NaN if none)."""
    a = max(lo, d - STALE)
    if d >= L.shape[1] or a > d:
        return np.full(len(rows), np.nan)
    blk = np.asarray(L[rows, a:d + 1], np.float64)
    f = np.isfinite(blk)
    has = f.any(1)
    last = blk.shape[1] - 1 - np.argmax(f[:, ::-1], 1)
    out = blk[np.arange(len(rows)), last]
    out[~has] = np.nan
    return out


def pct(x):
    """Cross-sectional percentile in (0, 1): (rank - 0.5) / n, rank 1 = smallest (x all finite)."""
    o = np.argsort(x, kind='stable')
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    return (r - 0.5) / len(x)


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, y - X @ b


def fm(coefs):
    """Fama-MacBeth mean, se, t, n over refits (refits are 21 days apart and outcomes 21 days long: no overlap)."""
    c = np.asarray(coefs, np.float64)
    c = c[np.isfinite(c)]
    if len(c) < 2:
        return (float(c.mean()) if len(c) else np.nan), np.nan, np.nan, len(c)
    se = c.std(ddof=1) / np.sqrt(len(c))
    return float(c.mean()), float(se), float(c.mean() / se), len(c)


def in_period(day, period):
    return period[0] <= iso(day) <= period[1]
