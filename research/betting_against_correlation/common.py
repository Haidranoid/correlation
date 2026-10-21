"""Betting against correlation (experiments/-5.5-betting-against-correlation.md): shared settings and helpers.
Rules are fixed in the .md ("Rules" and "Decisions made while building it")."""
import os
import sys
from pathlib import Path

for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = '8'

import numpy as np  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/betting_against_correlation'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

STEP = 21          # calendar spacing (trading days)
CAL0 = 252         # the twins' calendar: indices 252 + 21k
FIRST = 756        # first formation (2004-12-31)
VOL_W, VOL_MIN = 252, 120
COR_W, COR_MIN = 1260, 750
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
COST_BPS = 5.0
N_LARGE = 500
RF_CSV = Path.home() / '.local/share/correlation2/external/series/fred_dgs3mo.csv'


def twins_for(tag, E):
    """Fund-like flags etc. Old export: c2eval's cache (read only). New export: computed without c2eval's cache and
    saved here (this experiment may not write elsewhere)."""
    if tag == 'old':
        return c2eval.twins(E)
    p = WORK / 'twins_new.npz'
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    tw = c2eval.twins(E, cache=False)
    np.savez(p, **{k: v for k, v in tw.items() if k != 'F'})
    return tw


def rf_daily(days):
    """Daily risk-free return per export day: DGS3MO % a year / 100 / 252, latest value at or before the day."""
    rows = [l.strip().split(',') for l in open(RF_CSV)][1:]
    d = np.array([(np.datetime64(r[0]) - np.datetime64('1970-01-01')).astype(int) for r in rows if r[1] not in ('', '.')])
    v = np.array([float(r[1]) for r in rows if r[1] not in ('', '.')])
    pos = np.searchsorted(d, days, side='right') - 1
    out = v[np.maximum(pos, 0)] / 100.0 / 252.0
    out[pos < 0] = np.nan
    return out


def spy_returns(tag, days):
    """SPY simple daily returns from the store's adjusted closes (NaN where either close is missing)."""
    S = c2eval.load_store(STORES[tag])
    sid = [s['id'] for s in S['syms'] if s['ticker'] == 'SPY' and s['kind'] == 'etf']
    assert len(sid) == 1, sid
    c = c2eval.store_panel(S, sid, days, 'close')[0].astype(np.float64)
    r = np.full(len(days), np.nan)
    r[1:] = c[1:] / c[:-1] - 1
    return r


def ols_nw(y, X, lags=5):
    """OLS of y on [1, X] with Newey-West (Bartlett) standard errors. Returns coef, t, n, r2."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + ([np.asarray(X, float).reshape(len(y), -1)] if X is not None else []))
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, b / se, n, r2


SEC_MAP = Path.home() / '.local/share/correlation2/experiments/imposters/map/sec_tickers.json'
HAND_FUNDS = {'ASA', 'TYG', 'SARK', 'JEPQ', 'QTUM', 'ARTY', 'QQQE', 'WGMI',
              'FSCO', 'SYSB', 'COLO', 'GARP'}  # second line: found scanning the new export's universe, before its returns


def sec_names():
    import json
    return {r['ticker'].upper(): r['name'] for r in json.loads(SEC_MAP.read_text())}


def not_common(tickers):
    """[N] bool: funds, ETFs, preferreds, warrants and units filed as stocks that the fund-like flag misses
    (the .md, 'Decisions'): ticker with '_' or ending -WS/-U/-R/-RT; SEC name with the word FUND, BLACKROCK...TRUST,
    SPROTT PHYSICAL; the hand list."""
    import re
    nm = sec_names()
    out = np.zeros(len(tickers), bool)
    for j, t in enumerate(tickers):
        t = str(t).upper()
        key = t.replace('.', '-')
        name = nm.get(key, '').upper()
        if '_' in t or re.search(r'-(WS|U|R|RT)$', t) or key in HAND_FUNDS:
            out[j] = True
        elif re.search(r'\bFUND\b', name) or 'SPROTT PHYSICAL' in name or (name.startswith('BLACKROCK') and 'TRUST' in name):
            out[j] = True
    return out
