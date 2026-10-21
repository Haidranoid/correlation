"""Correlation-built portfolios (experiments/-4.6-correlation-portfolios.md): shared settings and helpers.

Rules are in the .md ("Rules" and "Decisions made while building it"). Copied pieces:
  - universe(): research/night_and_day/common.py (the judge-like liquid set and its 500 largest), verbatim logic;
  - rf_daily(), spy_close(), RF_CSV: research/lazy_prices/common.py.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

WORK = c2eval.WORK / 'correlation_portfolios'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}
RF_CSV = Path.home() / '.local/share/correlation2/external/series/fred_dgs3mo.csv'

# universe (night_and_day / the brief)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500

# the plan's rules
WIN = 252                 # estimation window: daily log returns over t-251..t
MIN_RET = 227             # at least 90% of the 252 returns (ceil(0.9 * 252))
CAP = 0.02                # MV-LW and MV-TW: each weight in [0, 2%]
COST_BPS = 5.0            # per side, on turnover sum |w_new - w_drifted| at each rebalance
PORTS = ('EW', 'IV', 'MV-LW', 'MV-TW', 'HRP')

PERIODS = {'2006-2019': ('2006-01-01', '2019-12-31'),
           '2020-2025-06': ('2020-01-01', '2025-06-27'),
           '2006-2025-06': ('2006-01-01', '2025-06-27')}
UNSEEN = ('2025-06-30', '2026-10-19')

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def rf_daily(days):
    """Daily risk-free return per export day: DGS3MO % a year / 100 / 252, latest value at or before the day.
    (Copied from research/lazy_prices/common.py. The CSV ends 2026-10-15, so 10-16..10-19 carry its last value.)"""
    rows = [l.strip().split(',') for l in open(RF_CSV)][1:]
    d = np.array([(np.datetime64(r[0]) - np.datetime64('1970-01-01')).astype(int) for r in rows if r[1] not in ('', '.')])
    v = np.array([float(r[1]) for r in rows if r[1] not in ('', '.')])
    pos = np.searchsorted(d, days, side='right') - 1
    out = v[np.maximum(pos, 0)] / 100.0 / 252.0
    out[pos < 0] = np.nan
    return out


def rf_last_date():
    rows = [l.strip().split(',') for l in open(RF_CSV)][1:]
    return [r[0] for r in rows if r[1] not in ('', '.')][-1]


def spy_close(days, store=c2eval.STORE_OLD):
    """SPY's store close aligned to the export days (copied from research/lazy_prices/common.py)."""
    S = c2eval.load_store(store)
    sid = [s['id'] for s in S['syms'] if s['ticker'] == 'SPY' and s['kind'] == 'etf']
    assert len(sid) == 1, sid
    return c2eval.store_panel(S, sid, days, 'close')[0].astype(np.float64)


def universe(E, tw, ks=None):
    """[K, N] bool universe and [K, N] bool 500 largest on tw['rebal'] days (research/night_and_day/common.py,
    copied; ks limits the work to those rebalance indices, the others stay False)."""
    close, dv = E['close'], E['dollar_volume']
    rebal = tw['rebal']
    N = close.shape[0]
    excl = c2eval.excluded(E, tw)
    uni = np.zeros((len(rebal), N), bool)
    big = np.zeros((len(rebal), N), bool)
    for k, t0 in enumerate(rebal):
        if ks is not None and k not in ks:
            continue
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


def ffill_rows(X):
    """Forward-fill NaNs along axis 1 (leading NaNs stay NaN)."""
    X = np.asarray(X, np.float64)
    ok = np.isfinite(X)
    idx = np.where(ok, np.arange(X.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = X[np.arange(X.shape[0])[:, None], idx]
    out[~np.isfinite(out)] = np.nan
    first = np.where(ok.any(1), ok.argmax(1), X.shape[1])
    out[np.arange(X.shape[1])[None, :] < first[:, None]] = np.nan
    return out


def window_logret(c):
    """Daily log returns of consecutive closes (NaN where either close is missing). c: [n, W+1] -> [n, W]."""
    c = np.asarray(c, np.float64)
    with np.errstate(all='ignore'):
        r = np.log(c[:, 1:] / c[:, :-1])
    r[~np.isfinite(r)] = np.nan
    return r


def demean_fill(R):
    """The plan's rule: demean each row over its finite values, then set missing values to 0."""
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mu = np.nanmean(R, -1, keepdims=True)
    X = R - mu
    X[~np.isfinite(X)] = 0.0
    return X
