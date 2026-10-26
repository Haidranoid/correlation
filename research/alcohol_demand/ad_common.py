"""Alcohol demand (experiments/untested-luca-alcohol-demand.md): shared settings, loaders and statistics.

The rules are fixed in the plan ("Rules"); the choices it leaves open are in its "Decisions" section. Everything is
point in time: an input used at decision day t0 is public by the close of t0, and the basket is bought at the close
of t0 + 1 and held to the close of t0 + 22.

The day axis is Yahoo SPY's trading days from 1993-01-29 to the last day of the tag's export: 'old' 2025-06-27,
'new' 2026-10-19. From 2001-12-31 it equals the export's days (checked in inputs.py), so export day j is axis day
j + OFFSET, and -3.2's calendar (export days 252 + 21k) extends back in steps of 21 axis days.

This folder is held outside the repo (correlation-pending/). It reads the repo's research/lib/c2eval.py and the
outside-data catalog's Google Trends parser read-only and never edits them.

    import sys; sys.path.insert(0, '/home/bhoult/programming/correlation-pending/research/alcohol_demand')
    import ad_common as K
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from math import erf, sqrt  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/alcohol_demand'
WORK.mkdir(parents=True, exist_ok=True)
YF = Path.home() / '.local/share/correlation2/experiments/published_data/raw'
YF_DIRS = (YF / 'yahoo_full_stocks', YF / 'yahoo_full_funds')
FRED_DIR = WORK / 'fred'
TRENDS_DIR = WORK / 'trends'
GT_SCRIPTS = Path.home() / '.local/share/correlation2/external/scripts_data-trends'

EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
CUT = {'old': '2025-06-27', 'new': '2026-10-19'}

# ---- Fixed by the plan --------------------------------------------------------------------------------------------
ALC = ('STZ', 'BF-B', 'TAP', 'SAM', 'MGPI')
GLOBAL = ('DEO', 'BUD', 'CCU', 'ABEV')
CANNABIS = ('TLRY', 'CGC', 'ACB', 'CRON', 'SNDL')
FUNDS = ('SPY', 'XLP', 'XLY', 'QQQ')
TICKERS = ALC + GLOBAL + CANNABIS + FUNDS
H = 21                  # hold and calendar spacing (trading days)
MIN_FIT = 36            # fewest eligible decisions in a fit
NW_LAGS = 3
BLOCK = 12              # moving-block bootstrap, in decisions
N_BOOT = 10_000
N_RANDOM = 10_000
N_NOISE = 200
SEED = 20261026
COST = 5e-4             # 5 bps on every traded dollar
INPUTS = ('E1', 'E2', 'P', 'T1', 'T2', 'T3')
TREND_QUERY = {'T1': 'alcohol', 'T2': 'moderation', 'T3': 'new_markets'}
# Releases later than the plan's month rule assumes (checked against BLS notices; plan, "When each month is public").
LATE = {('UNRATENSA', '2013-09'): '2013-10-22', ('CPI', '2013-09'): '2013-10-30',
        ('UNRATENSA', '2025-09'): '2025-11-20'}

DEV = ('2008-01-01', '2019-12-31')
CHECK = ('2020-01-01', '2025-06-27')
UNSEEN = ('2025-06-30', '2026-10-19')

iso = c2eval.iso
EPOCH = np.datetime64('1970-01-01', 'D')


def to_days(dates):
    return (np.asarray(dates, dtype='datetime64[D]') - EPOCH).astype(np.int64)


def iso_days(days):
    return np.array([iso(d) for d in days])


# ---- Loaders ------------------------------------------------------------------------------------------------------

def yahoo_cache():
    """Adjusted closes of TICKERS from the Yahoo full histories (fetched 2026-10-25), cached once as
    WORK/yahoo.npz: '<T>_d' days since 1970, '<T>_p' adjusted close."""
    f = WORK / 'yahoo.npz'
    if f.exists():
        z = np.load(f)
        return {t: (z[f'{t}_d'], z[f'{t}_p']) for t in TICKERS}
    rows = {t: [] for t in TICKERS}
    for d in YF_DIRS:
        for fn in sorted(glob.glob(str(d / 'batch_*.csv'))):
            with open(fn) as fh:
                r = csv.reader(fh)
                head = next(r)
                j = head.index('adj_close')
                for row in r:
                    if row[0] in rows and row[j] not in ('', 'nan'):
                        rows[row[0]].append((row[1], float(row[j])))
    out, save = {}, {}
    for t, lst in rows.items():
        assert lst, t
        lst = sorted(set(lst))
        dd = to_days(np.array([a for a, _ in lst], dtype='datetime64[D]'))
        assert np.all(np.diff(dd) > 0), t
        pp = np.array([b for _, b in lst])
        out[t] = (dd, pp)
        save[f'{t}_d'], save[f'{t}_p'] = dd, pp
    np.savez(f, **save)
    return out


def axis_for(tag, Y=None):
    """Yahoo SPY's trading days up to the tag's cut, and the export offset (export day 0 = axis day OFFSET)."""
    Y = Y or yahoo_cache()
    d, _ = Y['SPY']
    axis = d[d <= to_days(np.array([CUT[tag]], dtype='datetime64[D]'))[0]]
    E = c2eval.load_export(EXPORTS[tag], arrays=())
    ed = E['days']
    off = int(np.searchsorted(axis, ed[0]))
    return axis, off, ed


def on_axis(d, v, axis):
    """Values on the axis days; a missing day after the first value is filled with the previous value (counted)."""
    pos = np.searchsorted(d, axis)
    ok = (pos < len(d)) & (d[np.minimum(pos, len(d) - 1)] == axis)
    x = np.full(len(axis), np.nan)
    x[ok] = v[pos[ok]]
    n_fill = 0
    first = int(np.argmax(np.isfinite(x))) if np.isfinite(x).any() else len(x)
    for t in range(first + 1, len(x)):
        if not np.isfinite(x[t]) and axis[t] <= d[-1]:
            x[t] = x[t - 1]
            n_fill += 1
    return x, n_fill


def decision_days(axis_len, off):
    """-3.2's calendar (export days 252 + 21k) on the axis, extended back in steps of 21 axis days; only decisions
    whose hold ends on the axis (t0 + 22 <= last day)."""
    first = off + 252
    back = np.arange(first, -1, -H)[::-1]
    fwd = np.arange(first, axis_len, H)
    D = np.unique(np.concatenate([back, fwd]))
    return D[D + 1 + H <= axis_len - 1]


def read_fred(name):
    """FRED CSV -> dict 'YYYY-MM' -> value (missing values absent)."""
    out = {}
    for r in list(csv.reader(open(FRED_DIR / f'{name}.csv')))[1:]:
        if r[1] not in ('', '.', 'NA'):
            out[r[0][:7]] = float(r[1])
    return out


def read_trends(name):
    """Google Trends monthly series -> dict 'YYYY-MM' -> value (the partial month dropped)."""
    sys.path.insert(0, str(GT_SCRIPTS))
    import gtfetch
    out = {}
    for r in gtfetch.parse_timeline(str(TRENDS_DIR / f'{name}.json')):
        if not r['partial']:
            out[r['date'][:7]] = float(r['values'][0])
    return out


def mkey(i):
    return f'{i // 12:04d}-{i % 12 + 1:02d}'


def mindex(s):
    return int(s[:4]) * 12 + int(s[5:7]) - 1


def yoy(series, log=True, floor=None):
    """12-month change of a monthly dict, as a dict month -> change (only where both months exist)."""
    out = {}
    for k, v in series.items():
        k0 = mkey(mindex(k) - 12)
        if k0 in series:
            a, b = v, series[k0]
            if floor is not None:
                a, b = max(a, floor), max(b, floor)
            out[k] = (np.log(a) - np.log(b)) if log else (a - b)
    return out


def monthly_inputs():
    """The six inputs (and C3's retail-sales share) as dicts month -> value."""
    U, S = read_fred('UNRATENSA'), read_fred('UMCSENT')
    A, C = read_fred('CUUR0000SAF116'), read_fred('CPIAUCNS')
    ya, yc = yoy(A), yoy(C)
    X = {'E1': yoy(U, log=False), 'E2': yoy(S),
         'P': {k: ya[k] - yc[k] for k in ya if k in yc}}
    for k, q in TREND_QUERY.items():
        X[k] = yoy(read_trends(q), floor=1.0)
    R4, RT = read_fred('MRTSSM4453USS'), read_fred('MRTSSM44X72USS')
    X['R'] = yoy({k: R4[k] / RT[k] for k in R4 if k in RT})
    return X


SOURCE = {'E1': 'UNRATENSA', 'E2': 'UMCSENT', 'P': 'CPI', 'R': 'MRTS', 'T1': 'trends', 'T2': 'trends',
          'T3': 'trends'}


def usable_month(name, date_iso, caution=0):
    """The latest month whose value of input `name` is public at the close of date_iso (the plan's rule), before
    skipping missing months."""
    mi = mindex(date_iso)
    day = int(date_iso[8:10])
    src = SOURCE[name]
    if src == 'trends':
        m = mi - 1 if day >= 8 else mi - 2
    elif src == 'MRTS':
        m = mi - 3
    else:
        m = mi - 2
    m -= caution
    if src in ('UNRATENSA', 'CPI'):
        while (src, mkey(m)) in LATE and LATE[(src, mkey(m))] > date_iso:
            m -= 1
    return m


def input_at(X, name, date_iso, caution=0, max_back=3):
    """Value of input `name` known at date_iso: the usable month, or the latest earlier month with a value."""
    m = usable_month(name, date_iso, caution)
    for b in range(max_back + 1):
        k = mkey(m - b)
        if k in X[name]:
            return X[name][k], k
    return np.nan, None


# ---- Prices and baskets -------------------------------------------------------------------------------------------

def price_panel(tag, Y=None):
    Y = Y or yahoo_cache()
    axis, off, ed = axis_for(tag, Y)
    P, fills = {}, {}
    for t in TICKERS:
        P[t], fills[t] = on_axis(*Y[t], axis)
    return axis, off, ed, P, fills


def basket_log(P, names, t0, t1, need_at=None):
    """log of the equal-weight buy-and-hold basket from the close of t0 to the close of t1, over the names with a
    price at t0, t1 and (if given) need_at; NaN if none. Returns (value, number of names)."""
    ok = [n for n in names if np.isfinite(P[n][t0]) and np.isfinite(P[n][t1])
          and (need_at is None or np.isfinite(P[n][need_at]))]
    if not ok:
        return np.nan, 0
    return float(np.log(np.mean([P[n][t1] / P[n][t0] for n in ok]))), len(ok)


def target(P, names, D, bench=None):
    """21-day log return of a basket (or one ticker) bought at the close of t0 + 1 and held to t0 + 22, minus the
    benchmark's over the same days; a name must also trade at t0 (known at the decision)."""
    y = np.full(len(D), np.nan)
    for i, t0 in enumerate(D):
        v, _ = basket_log(P, names, t0 + 1, t0 + 1 + H, need_at=t0)
        if bench is not None:
            b, _ = basket_log(P, (bench,), t0 + 1, t0 + 1 + H)
            v = v - b
        y[i] = v
    return y


# ---- Forecasts ----------------------------------------------------------------------------------------------------

def eligible(D, i, sample):
    """Decisions whose target has ended by decision i (s + 22 <= t0, the training-eligibility rule)."""
    el = np.nonzero(sample[:i] & (D[:i] + 1 + H <= D[i]))[0]
    if len(el):
        assert D[el[-1]] + 1 + H <= D[i]
    return el


def oos(y, X, D, min_fit=MIN_FIT, sample=None, sink=True):
    """Expanding-window forecasts at each decision: the prevailing mean (f0), each input alone (fk[:, k]), their
    average (fc), and one regression on all inputs (fs). Every model uses the same decisions: those where y and every
    column of X are finite (and `sample`). Slopes of the single-input fits are kept per decision."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, K = X.shape
    smp = np.isfinite(y) & np.isfinite(X).all(1)
    if sample is not None:
        smp &= sample
    f0, fc, fs = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    fk, bk = np.full((n, K), np.nan), np.full((n, K), np.nan)
    for i in range(n):
        if not np.isfinite(X[i]).all():
            continue
        el = eligible(D, i, smp)
        if len(el) < min_fit:
            continue
        yy = y[el]
        f0[i] = yy.mean()
        for k in range(K):
            A = np.column_stack([np.ones(len(el)), X[el, k]])
            b, *_ = np.linalg.lstsq(A, yy, rcond=None)
            fk[i, k] = b[0] + b[1] * X[i, k]
            bk[i, k] = b[1]
        fc[i] = fk[i].mean()
        if sink and K > 1:
            A = np.column_stack([np.ones(len(el)), X[el]])
            b, *_ = np.linalg.lstsq(A, yy, rcond=None)
            fs[i] = b[0] + X[i] @ b[1:]
    return {'f0': f0, 'fk': fk, 'fc': fc, 'fs': fs, 'bk': bk, 'sample': smp}


# ---- Statistics ---------------------------------------------------------------------------------------------------

def nw_se(x, lags):
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for l in range(1, min(lags, n - 1) + 1):
        v += 2.0 * (1.0 - l / (lags + 1.0)) * (e[l:] @ e[:-l]) / n
    return float(np.sqrt(max(v, 0.0) / n))


def p_upper(t):
    return 0.5 * (1 - erf(t / sqrt(2))) if np.isfinite(t) else float('nan')


def block_index(n, block, n_draws, seed):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, size=(n_draws, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n_draws, -1)[:, :n]
    return idx


def evaluate(y, f0, fa, mask, seed=SEED, boot=True):
    """Out-of-sample comparison of the prevailing mean f0 and a forecast fa on the decisions in mask: MSEs, the
    out-of-sample R^2 1 - MSE_a / MSE_0 (with a 95% block-bootstrap interval), Clark-West (Newey-West t, normal p,
    moving-block bootstrap p)."""
    m = mask & np.isfinite(y) & np.isfinite(f0) & np.isfinite(fa)
    n = int(m.sum())
    if n < 12:
        return {'n': n}
    yy, b, a = y[m], f0[m], fa[m]
    eb, ea = yy - b, yy - a
    mse_b, mse_a = float(np.mean(eb ** 2)), float(np.mean(ea ** 2))
    f = eb ** 2 - ea ** 2 + (b - a) ** 2
    se = nw_se(f, NW_LAGS)
    t = float(f.mean() / se) if se > 0 else float('nan')
    out = {'n': n, 'mse0': mse_b, 'mse': mse_a, 'r2os': 1 - mse_a / mse_b, 'cw_t': t, 'cw_p': p_upper(t)}
    if boot and n >= 2 * BLOCK:
        idx = block_index(n, BLOCK, N_BOOT, seed)
        fc = f - f.mean()
        dr = fc[idx].mean(1)
        out['cw_boot_p'] = float(np.mean(dr >= f.mean()))
        r2 = 1 - (ea ** 2)[idx].mean(1) / (eb ** 2)[idx].mean(1)
        out['r2_lo95'], out['r2_hi95'] = float(np.percentile(r2, 2.5)), float(np.percentile(r2, 97.5))
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[j]))
        adj[j] = run
    return adj


def sel(dates, lo, hi):
    return (dates >= lo) & (dates <= hi)


def fmt(x, d=3):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.{d}f}'


def dump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=conv))
