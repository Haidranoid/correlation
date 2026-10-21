"""Hidden text peers (experiments/+4.6-hidden-text-peers.md): shared settings and helpers.
Rules are fixed in the .md ("Rules" and "Decisions made while building it")."""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = c2eval.WORK / 'hidden_text_peers'
TEXT = WORK / 'text'                 # per fiscal year: Item 1 word ids of every corpus filing (text.py)
PEERS = WORK / 'peers'               # per fiscal year: text-peer pairs of mapped companies (peers.py)
EDGAR = c2eval.WORK / 'lazy_prices/edgar'                                   # read only
MAP = c2eval.WORK / 'imposters/map/cik_to_rows.json'                        # read only
for _p in (WORK, TEXT, PEERS):
    _p.mkdir(parents=True, exist_ok=True)

FY_FIRST, FY_LAST = 2005, 2020       # FY2005 is the current year in January 2007; FY2020 is the corpus's last
MIN_ITEM1_WORDS = 1000               # Item 1 tokens (lower-cased runs of a-z), before any filtering
MAX_DF, MIN_DF = 0.25, 3             # drop words in more than 25% of the year's filings or in fewer than 3
TOP_PAIRS = 0.02                     # a pair is a text pair when its similarity is in the year's top 2%
CAP = 50                             # at most the 50 most similar per firm

# returns, correlation and signal
CORR_W = 252                         # correlation over t-251..t (daily market-relative log returns)
MIN_OVERLAP = 227                    # days with both returns, of the 252 (90%)
MAX_LOG = 0.5                        # |daily log return| above this is a bad print (correlation input only)
SIG_W = 21                           # signal over t-20..t (21 daily returns)
MIN_HIDDEN = 3

# calendar, universe and book (copied from research/night_and_day/common.py and research/lazy_prices)
STEP = 21
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500
TENTH = 10
COST_BPS = 5.0                       # per side, on actual turnover

FORM = ('2007-01-01', '2022-05-31')  # all formations (FY2020 peers stop counting on 2022-05-31)
TEST1 = ('2007-01-01', '2019-12-31')
TEST3 = ('2020-01-01', '2022-05-31')

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def epoch_of(day):
    """Expected latest fiscal year at a date (days since 1970): last year's from June 1, else the year before.
    The imposters experiment's rule, copied from research/lazy_prices/common.py."""
    dt = np.datetime64('1970-01-01') + np.timedelta64(int(day), 'D')
    y = int(str(dt)[:4])
    return y - 1 if dt >= np.datetime64(f'{y}-06-01') else y - 2


def avail_day(fy, fye):
    """The imposters experiment's filing-date rule (research/imposters/text.py), copied from
    research/lazy_prices/common.py: June 1 of FY+1, or 120 days after a later fiscal-year end mentioned in the text,
    whichever is later. Days since 1970."""
    a = np.datetime64(f'{fy + 1}-06-01')
    if fye is not None and fye[0] > fy:
        try:
            a = max(a, np.datetime64(f'{fye[0]:04d}-{fye[1]:02d}-{min(fye[2], 28):02d}') + np.timedelta64(120, 'D'))
        except ValueError:
            pass
    return int((a - np.datetime64('1970-01-01')).astype(int))


def epoch_end(fy):
    """First day (days since 1970) on which fiscal year fy's text no longer counts: June 1 of fy+2."""
    return int((np.datetime64(f'{fy + 2}-06-01') - np.datetime64('1970-01-01')).astype(int))


def cik_rows():
    m = json.loads(MAP.read_text())
    return {int(c): [int(r) for r in rows] for c, rows in m.items()}


def universe(E, tw):
    """[K, N] bool universe and [K, N] bool 500 largest on tw['rebal'] days. Copied from
    research/night_and_day/common.py: closes on >= 90% of t0-755..t0, median close >= $5 and median dollar volume
    >= $5M over t0-251..t0, minus c2eval.excluded(); the 500 largest by median dollar volume over t0-62..t0."""
    key = WORK / f"universe_{E['meta']['fingerprint']}.npz"
    if key.exists():
        z = np.load(key)
        return z['uni'], z['big']
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
    np.savez(key, uni=uni, big=big)
    return uni, big


def ffill(X):
    """Forward-fill NaNs along axis 1 (leading NaNs stay NaN)."""
    X = np.asarray(X, np.float64)
    ok = np.isfinite(X)
    idx = np.where(ok, np.arange(X.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return X[np.arange(X.shape[0])[:, None], idx]


def ols(y, X):
    """OLS of y on [1, X] with plain standard errors. Returns coef, t, n."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    s2 = e @ e / (n - k)
    se = np.sqrt(np.diag(XtXi) * s2)
    return b, b / se, n


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(x.mean()) if len(x) else float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)
