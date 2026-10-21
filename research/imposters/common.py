"""Shared settings and helpers for the imposters experiment (experiments/+0.1-imposters.md).
Settings here were fixed before any outcome data was looked at; see the .md, "Decisions made while building it"."""
import os
import sys
from pathlib import Path

for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '8')

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/imposters'
EDGAR = WORK / 'edgar_s1'
# ODD=1: the groups map made from odd days only (test 1's robustness check, added after results)
ODD = os.environ.get('C2_IMPOSTERS_ODD') == '1'
MONTHS_DIR = WORK / ('months_odd' if ODD else 'months')

YEAR = 252            # correlation period, trading days
K = 30                # groups per month (the plan allows 20-40; 30 is the middle)
STOCKS = 1500         # most-traded stocks grouped
MIN_DV = 5e6          # median dollar volume over the correlation period
COVER = 0.95          # share of days with a return, as in together.rs group_universe
FIRST_MONTH = '2003-01'
LAST_MONTH = '2022-05'   # the corpus's last fiscal year (2020) is the expected latest 10-K until 2022-05-31


def load():
    """Export (old, through 2025-06-27) with simple daily returns from close."""
    e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close', 'dollar_volume'))
    close = np.asarray(e['close'], dtype=np.float64)
    ret = np.full_like(close, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        ret[:, 1:] = close[:, 1:] / close[:, :-1] - 1.0
    ret[~np.isfinite(ret)] = np.nan
    e['ret'] = ret
    return e


def month_ends(days, first=FIRST_MONTH, last=LAST_MONTH):
    """Indices of the last trading day of each month in [first, last]."""
    dt = (np.datetime64('1970-01-01') + days.astype('timedelta64[D]')).astype('datetime64[M]')
    idx = np.nonzero(dt[:-1] != dt[1:])[0]  # the export's last (incomplete) month never counts
    ms = dt[idx]
    keep = (ms >= np.datetime64(first)) & (ms <= np.datetime64(last))
    return idx[keep], ms[keep]


def week_ends(days):
    """Indices of the last trading day of each calendar week (Monday-start)."""
    wk = (days + 3) // 7  # days since 1970-01-01 (Thursday); +3 makes weeks start on Monday
    idx = np.nonzero(wk[:-1] != wk[1:])[0]
    return idx


def residuals(ret, rows, d_end, univ=None, length=YEAR):
    """Market-relative returns over the correlation period ending at day d_end (inclusive) for the stocks in `rows`.
    The market is the equal-weight mean of the `univ` stocks' returns (default: rows). Each stock's beta to it comes
    from the same days. Returns (resid [len(rows), YEAR] with NaN where missing, beta [len(rows)], market [YEAR])."""
    win = slice(d_end - length + 1, d_end + 1)
    x = ret[rows, win]
    mu = ret[univ if univ is not None else rows, win]
    market = np.nanmean(mu, axis=0)
    market = np.where(np.isfinite(market), market, 0.0)
    ok = np.isfinite(x)
    m = np.broadcast_to(market, x.shape)
    n = ok.sum(1)
    mx = np.where(ok, x, 0).sum(1) / np.maximum(n, 1)
    mm = np.where(ok, m, 0).sum(1) / np.maximum(n, 1)
    cov = np.where(ok, (x - mx[:, None]) * (m - mm[:, None]), 0).sum(1)
    var = np.where(ok, (m - mm[:, None]) ** 2, 0).sum(1)
    beta = np.where(var > 0, cov / np.where(var > 0, var, 1), 0.0)
    return x - beta[:, None] * market[None, :], beta, market
