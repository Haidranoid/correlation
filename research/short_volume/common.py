"""Short volume (experiments/+0.0-short-volume.md): shared settings and helpers.

Every script takes a tag: old = the judge's export through 2025-06-27, new = the export through 2026-10-19 (the
unseen months, used once at the end). Run under the memory-capped scope (8G for processing):
    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- env OMP_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python3 -I -B <script>.py old
"""
import os
import sys
sys.dont_write_bytecode = True
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np  # noqa: E402
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
WORK = c2eval.WORK / 'short_volume'
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}

# Universe (copied from research/night_and_day/common.py): the judge-like liquid set on the twins calendar
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500

# Signal
MIN_TOTAL = 10_000        # SVR missing when TotalVolume < 10,000 shares
RECENT = 5                # mean SVR over t-4..t
MIN_RECENT = 3            # at least 3 of the 5 days
BASE_LO, BASE_HI = 256, 5 # baseline over t-256..t-5 (252 days)
MIN_BASE = 150
MIN_PEERS = 3             # at least 3 other universe stocks sharing the twin, with a finite abnormal SVR

# Weekly calendar and periods
CAL0 = '2010-01-04'
H = 5
DECILE = 0.10
COST_BPS = 5.0
MIN_NAMES = 50
TRAIN = ('2010-01-04', '2015-12-31')
PERIODS = {'train': ('2010-01-04', '2015-12-31'), 'test1': ('2016-01-01', '2019-12-31'),
           'test3': ('2020-01-01', '2025-06-27'), 'unseen': ('2025-06-30', '2026-10-19')}
SCORE_FROM = '2016-01-01'

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def wdir(tag):
    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def universe(E, tw):
    """[K, N] bool universe and [K, N] bool 500 largest on tw['rebal'] days (night_and_day's rule), restricted here to
    stocks with a twin at that rebalance (fund_row >= 0)."""
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
        u = fin & (np.nan_to_num(mc) >= MIN_PRICE) & (np.nan_to_num(mv) >= MIN_DV) & ~excl[k] & (tw['fund_row'][k] >= 0)
        uni[k] = u
        cand = np.nonzero(u)[0]
        o = cand[np.argsort(-np.nan_to_num(mv63[cand], nan=-1.0), kind='stable')]
        big[k, o[:N_LARGE]] = True
    return uni, big


def kidx(rebal, t):
    return int(np.searchsorted(rebal, t, side='right')) - 1


def rolling_mean(X, lo, hi, min_n):
    """Mean of the finite values of X[:, d-lo .. d-hi] (inclusive) for each d; NaN with fewer than min_n."""
    N, T = X.shape
    ok = np.isfinite(X)
    cs = np.zeros((N, T + 1))
    cn = np.zeros((N, T + 1))
    np.cumsum(np.where(ok, X, 0.0), 1, out=cs[:, 1:])
    np.cumsum(ok, 1, out=cn[:, 1:])
    out = np.full((N, T), np.nan, np.float32)
    d = np.arange(lo, T)
    a, b = d - lo, d - hi + 1           # window = [a, b)
    s = cs[:, b] - cs[:, a]
    n = cn[:, b] - cn[:, a]
    with np.errstate(all='ignore'):
        m = s / n
    m[n < min_n] = np.nan
    out[:, lo:] = m
    return out


def avg_rank(x):
    o = np.argsort(x, kind='stable')
    xs = x[o]
    r = np.empty(len(x))
    starts = np.r_[0, np.nonzero(np.diff(xs) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(xs)]
    r[o] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    return r


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra, rb = avg_rank(a), avg_rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def decision_days(days, T):
    """Weekly decision days: export indices idx(2010-01-04) + 5k whose outcome close t+5 is inside the export."""
    d0 = c2eval.day_index(days, CAL0)
    return np.arange(d0, T - H, H)


def in_period(days, idx, name):
    a, b = PERIODS[name]
    lo = (np.datetime64(a) - np.datetime64('1970-01-01')).astype(int)
    hi = (np.datetime64(b) - np.datetime64('1970-01-01')).astype(int)
    dd = days[idx]
    return (dd >= lo) & (dd <= hi)

# Volume-match guard against reused tickers (decided from out/mapcheck_old.txt, before any return was computed):
# a stock-day's SVR is used only if the correlation of log FINRA TotalVolume with log export volume over the 63 days
# through d (>= 40 common days) is at least VC_GUARD.
VC_GUARD = 0.3
