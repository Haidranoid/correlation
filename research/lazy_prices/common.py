"""Lazy prices (experiments/-1.3-lazy-prices.md): shared settings and helpers.
Rules are fixed in the .md ("Rules" and "Decisions made while building it")."""
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/lazy_prices'
EDGAR = WORK / 'edgar'          # streamed sections (fetch_edgar.py)
VEC = WORK / 'vec'              # term-frequency vectors per fiscal year (vectors.py)
MAP = Path.home() / '.local/share/correlation2/experiments/imposters/map/cik_to_rows.json'   # read only
WORK.mkdir(parents=True, exist_ok=True)

SECTIONS = ('section_1', 'section_1A', 'section_7')
SEC_NAMES = {'section_1': 'Item 1', 'section_1A': 'Item 1A', 'section_7': 'Item 7'}
MIN_SECTION_WORDS = 100     # a section counts as present with at least this many words
MIN_GAP_MONTHS, MAX_GAP_MONTHS = 9, 15   # fiscal-year ends of the two filings (when both are known)

# calendar and universe
STEP = 21                   # the twins' calendar: export day indices 252 + 21k
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
N_LARGE = 500
FIFTH = 5
COST_BPS = 5.0              # per side

TEST1 = ('2007-01-01', '2021-12-31')     # formation days
TEST3 = ('2022-01-01', '2022-12-31')
RF_CSV = Path.home() / '.local/share/correlation2/external/series/fred_dgs3mo.csv'


def epoch_of(day):
    """Expected latest fiscal year at a date (days since 1970): last year's from June 1, else the year before.
    The imposters experiment's rule (research/imposters/classes.py), copied."""
    dt = np.datetime64('1970-01-01') + np.timedelta64(int(day), 'D')
    y = int(str(dt)[:4])
    return y - 1 if dt >= np.datetime64(f'{y}-06-01') else y - 2


def avail_day(fy, fye, rule='june'):
    """The imposters experiment's filing-date rule (research/imposters/text.py), copied: June 1 of FY+1, or 120 days
    after a later fiscal-year end mentioned in the text, whichever is later. Days since 1970.
    rule='april' is the sensitivity fixed before results: April 1 of FY+1, or 90 days after a later fiscal-year end."""
    mmdd, lag = ('06-01', 120) if rule == 'june' else ('04-01', 90)
    a = np.datetime64(f'{fy + 1}-{mmdd}')
    if fye is not None and fye[0] > fy:
        try:
            a = max(a, np.datetime64(f'{fye[0]:04d}-{fye[1]:02d}-{min(fye[2], 28):02d}') + np.timedelta64(lag, 'D'))
        except ValueError:
            pass
    return int((a - np.datetime64('1970-01-01')).astype(int))


def rf_daily(days):
    """Daily risk-free return per export day: DGS3MO % a year / 100 / 252, latest value at or before the day."""
    rows = [l.strip().split(',') for l in open(RF_CSV)][1:]
    d = np.array([(np.datetime64(r[0]) - np.datetime64('1970-01-01')).astype(int) for r in rows if r[1] not in ('', '.')])
    v = np.array([float(r[1]) for r in rows if r[1] not in ('', '.')])
    pos = np.searchsorted(d, days, side='right') - 1
    out = v[np.maximum(pos, 0)] / 100.0 / 252.0
    out[pos < 0] = np.nan
    return out


def spy_close(days, store=c2eval.STORE_OLD):
    S = c2eval.load_store(store)
    sid = [s['id'] for s in S['syms'] if s['ticker'] == 'SPY' and s['kind'] == 'etf']
    assert len(sid) == 1, sid
    return c2eval.store_panel(S, sid, days, 'close')[0].astype(np.float64)


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
