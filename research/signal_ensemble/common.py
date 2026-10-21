"""Signal ensemble (experiments/-3.2-signal-ensemble.md): shared settings and helpers.

Rules are fixed in the .md ("Rules" and "Decisions made while building it"). Every script takes a tag, old or new:
old = the judge's export through 2025-06-27, new = the export through 2026-10-19 (the unseen months, run once).

Run every script under the memory-capped scope:
    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- env OMP_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 -I <script>.py old
"""
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/signal_ensemble'
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
LGBM = Path('/home/bhoult/programming/correlation/stockai/runs/poc/baselines/lgbm_ret/scores.npy')
MOM_STOCKAI = Path('/home/bhoult/programming/correlation/stockai/runs/poc/baselines/momentum_12_1.npy')

# Universe: the judge's liquid rule (stockai/judge/src/holdout.rs liquid_universe, --peer-window 756), point in time
DATA_W, DATA_FRAC = 756, 0.9      # census variable v finite on >= 90% of the 756 days ending at d
MED_W = 252                       # upper medians of close and dollar volume over the 252 days ending at d
MIN_PRICE, MIN_DV = 5.0, 5e6
N_LARGE = 500

# Components (order fixed); lgbm only on the old export
COMPONENTS = ['rev5', 'mom12_1', 'twin5', 'lowcorr', 'lgbm']
LABEL = {'rev5': '5-day reversal', 'mom12_1': '12-1 momentum', 'twin5': 'Twin residual reversal',
         'lowcorr': 'Low correlation', 'lgbm': 'LightGBM (actual returns)'}
MIN_COMP = 3                      # a stock needs >= 3 finite component ranks for a composite (see the .md)

# Low correlation (copied from research/betting_against_correlation, -4.4)
VOL_W, VOL_MIN = 252, 120
COR_W, COR_MIN = 1260, 750

# Books
WEEK0, WEEK = 252, 5              # weekly calendar: export day indices 252 + 5k
TOP, KEEP = 0.10, 0.30            # long-short: enter in the top 10%, keep while in the top 30% (bottom for shorts)
LO_IN, LO_KEEP = 20, 60           # long-only: buy on entering the top 20, hold while in the top 60
COST_BPS = 5.0
WEEKS_PER_YEAR = 252 / 5

PERIODS = [('2008–2019 (tests 1–2)', '2008-01-01', '2019-12-31'),
           ('2008–09', '2008-01-01', '2009-12-31'),
           ('2010–14', '2010-01-01', '2014-12-31'),
           ('2015–19', '2015-01-01', '2019-12-31'),
           ('2020–2025-06 (test 3)', '2020-01-01', '2025-06-27')]
UNSEEN = ('Unseen months', '2025-06-30', '2026-10-19')


def wdir(tag):
    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def rank_pm1(x):
    """Ranks of a 1-D finite array mapped to [-1, +1] (lowest -1, highest +1); ties get their average rank."""
    n = len(x)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.zeros(1)
    order = np.argsort(x, kind='mergesort')
    s = x[order]
    flag = np.empty(n, bool)
    flag[0] = True
    flag[1:] = s[1:] != s[:-1]
    starts = np.nonzero(flag)[0]
    ends = np.append(starts[1:], n)
    avg = (starts + ends - 1) / 2.0
    r = np.empty(n)
    r[order] = avg[np.cumsum(flag) - 1]
    return 2.0 * r / (n - 1) - 1.0


def ordinal_rank(x):
    return np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort')


def spearman(a, b):
    """Rank correlation of two equal-length finite arrays (average ranks for ties)."""
    if len(a) < 3:
        return np.nan
    ra, rb = rank_pm1(a), rank_pm1(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def ffill(close):
    """Forward-filled closes [N, T] float64 (leading NaNs stay NaN)."""
    c = np.asarray(close, np.float64)
    ok = np.isfinite(c)
    idx = np.where(ok, np.arange(c.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    P = np.take_along_axis(c, idx, 1)
    P[~np.isfinite(P)] = np.nan
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


def ols(y, x):
    """OLS of y on [1, x], plain standard errors. Returns alpha, t_alpha, beta, n."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    X = np.column_stack([np.ones(len(y)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = (e @ e) / (len(y) - 2)
    V = s2 * np.linalg.inv(X.T @ X)
    return float(b[0]), float(b[0] / np.sqrt(V[0, 0])), float(b[1]), len(y)


def weekly_days(T, last_needed=WEEK):
    """Weekly decision days 252 + 5k whose forward week d+1..d+5 is inside the export."""
    return np.arange(WEEK0, T - last_needed, WEEK)


def in_period(days, d_idx, a, b):
    """Mask of decision-day indices whose date is in [a, b]."""
    lo = (np.datetime64(a) - np.datetime64('1970-01-01')).astype(int)
    hi = (np.datetime64(b) - np.datetime64('1970-01-01')).astype(int)
    dd = days[d_idx]
    return (dd >= lo) & (dd <= hi)
