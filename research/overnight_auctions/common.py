"""Overnight auctions (experiments/+15.7-overnight-auctions.md): shared settings and helpers.

The universe and the 500 largest are copied from research/night_and_day/common.py (read-only there): on c2eval.twins'
calendar (export days 252 + 21k), point in time, closes on >= 90% of the trailing 756 days, median close >= $5 and
median dollar volume >= $5M over the trailing 252 days, minus c2eval.excluded(); the 500 largest are the universe's
top 500 by median dollar volume over the trailing 63 days. This experiment also needs a twin at that rebalance.

Timing: the signal for day t is formed before the 15:50 closing-auction cutoff, so it may use opens through t and
closes through t-1 only. The universe, the twin and beta for day t come from the rebalance at or before t-1
(kidx(rebal, t - 1)), whose inputs end at that rebalance's close.
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
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
WORK = c2eval.WORK / 'overnight_auctions'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

FIRST = '2008-01-01'   # opens are used from the first trading day of 2008 (earlier overnights count as missing)
MAX_LOG = 0.5          # night and day's bad-print rule: |log move| > 0.5 is missing
H, H_MIN = 21, 17      # signal: mean of x over the 21 overnights t-20..t, at least 17 present
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500
DECILE = 0.10
MIN_NAMES = 50         # a day's book needs at least this many ranked stocks
STOCK_SIDE_BPS = 2.0   # auction fills, per side
TWIN_SIDE_BPS = 1.0    # netted twin (or SPY) legs, per side
PICKS = 20             # money test account
N_RANDOM = 1000
SEED = 20261021        # the money test's random accounts
PERIODS_OLD = [('2008-2019', '2008-01-01', '2019-12-31'), ('2020-2025-06', '2020-01-01', '2025-06-27')]
PERIODS_NEW = [('unseen 2025-06-30..2026-10-19', '2025-06-30', '2026-10-19')]

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def kidx(rebal, t):
    """Index of the latest rebalance at or before export day t (-1 if none)."""
    return int(np.searchsorted(rebal, t, side='right')) - 1


def universe(E, tw):
    """[K, N] bool universe and [K, N] bool 500 largest on tw['rebal'] days (copied from night_and_day/common.py)."""
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


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def nw_t(x, lags=5):
    """Newey-West (Bartlett) t of the mean of a daily series (NaNs dropped); copied from night_and_day."""
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


def iso_to_day(s):
    return int((np.datetime64(s) - np.datetime64('1970-01-01')).astype(int))
