"""Credit-spread timing (experiments/untested-credit-spread-timing.md): shared settings, loaders and statistics.

The rules are fixed in the plan ("Rules"); the choices it leaves open are in its "Decisions" section. Everything is
point in time: a value used at decision day t is public after the close of t. BAA10Y (and AAA10Y, DGS10) enter only
through the previous trading day, t - 1 (the plan's publication rule); VIX for day t is allowed (16:15, the trade is at
t + 1).

The day axis is Yahoo SPY's trading days from 1993-01-29 to the last day of the tag's export: 'old' 2025-06-27,
'new' 2026-10-19, 'fwd' a later export (forward.py). From 2001-12-31 the axis equals the export's days exactly
(checked in inputs.py), so export day j is axis day j + OFFSET.

Reuses, read-only and never edited: research/lib/c2eval.py (exports, stores) and research/market_timing/engine.py
(-3.2's measures, random timing, rel_paths; imported by timing.py).

    import sys; sys.path.insert(0, '/home/bhoult/programming/correlation/research/credit_spread'); import cs_common as K
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import csv  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from math import erf, sqrt  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)   # all-NaN windows are expected and handled

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/credit_spread'
WORK.mkdir(parents=True, exist_ok=True)

# ---- Data locations (read only) ---------------------------------------------------------------------------------
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}
FRED_DIR = Path.home() / '.local/share/correlation2/external/series'           # fred_<id>.csv (date,value)
YAHOO_DIR = Path.home() / '.local/share/correlation2/experiments/published_data/etf'   # <T>.csv (pubdata's format)
FREDS = {'old': FRED_DIR, 'new': FRED_DIR}
YAHOOS = {'old': YAHOO_DIR, 'new': YAHOO_DIR}
TAGS = ('old', 'new', 'fwd')
# Forward runs (forward.py): a later download, named by environment variables and registered at run time.
if os.environ.get('C2_FWD_EXPORT') and os.environ.get('C2_FWD_STORE'):
    EXPORTS['fwd'] = Path(os.environ['C2_FWD_EXPORT'])
    STORES['fwd'] = Path(os.environ['C2_FWD_STORE'])
    FREDS['fwd'] = Path(os.environ.get('C2_FWD_FRED_DIR', str(FRED_DIR)))
    YAHOOS['fwd'] = Path(os.environ.get('C2_FWD_YAHOO_DIR', str(YAHOO_DIR)))

# ---- Fixed by the plan --------------------------------------------------------------------------------------------
H = 21                 # target horizon and -3.2's calendar spacing (trading days)
SPREAD_LAG = 1         # the spread enters through t - 1 (C5: 0 optimistic, 2 extra caution)
WEEK = 5               # the week's spread change: S(t - 1) - S(t - 6)
R_DAYS = 6             # R(t) = log P(t) - log P(t - 6)
FLOOR_R2 = 1e-8        # each squared daily log return floored here before a log
MIN_FIT = 1260         # fewest eligible origins in a fit (forecasts start about 1998)
MIN_FIT_XLF = 504      # XLF's fits (the store starts 2005-02-25)
PCT_MAX, PCT_MIN = 1260, 252   # trailing windows for S2's c and the firing percentile
Q_FIRE = 0.80          # S2: a widening in the top fifth of the previous 1,260 days
GRID_Q = (None, 0.80, 0.90)     # S2's reported grid: above 0 only / top fifth / top tenth
BLOCK_DAYS = 63        # P1's moving-block bootstrap
BLOCK_ALT = 21         # reported beside it (the brief's default)
N_BOOT = 10_000
N_RANDOM = 10_000
SEED = 20261025
COST = 5e-4            # 5 bps on every SPY trade
MAX_CARRY = 1          # a FRED value is carried over at most one trading day (bond holidays)

# ---- Periods (by origin / decision date) --------------------------------------------------------------------------
EARLY = ('1998-01-01', '2007-12-31')
DEV = ('2008-01-01', '2019-12-31')
CHECK = ('2020-01-01', '2025-06-27')
UNSEEN = ('2025-06-30', '2026-10-19')
FWD = ('2026-10-20', '2100-12-31')
OLD_LAST = '2025-06-27'
NEW_LAST = '2026-10-19'

iso = c2eval.iso
EPOCH = np.datetime64('1970-01-01', 'D')


def to_days(dates):
    return (np.asarray(dates, dtype='datetime64[D]') - EPOCH).astype(np.int64)


def iso_days(days):
    return np.array([iso(d) for d in days])


# ---- Loaders ------------------------------------------------------------------------------------------------------

def export_days(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=())
    return E['days']


def read_fred(name, tag, cut_iso):
    """FRED CSV (date,value) -> (days since 1970, values), rows dated after cut_iso dropped (the old span never sees
    the unseen months)."""
    rows = list(csv.reader(open(FREDS[tag] / f'fred_{name}.csv')))[1:]
    rows = [r for r in rows if r[0] <= cut_iso and r[1] not in ('', '.', 'NA')]
    d = to_days(np.array([r[0] for r in rows], dtype='datetime64[D]'))
    v = np.array([float(r[1]) for r in rows])
    assert np.all(np.diff(d) > 0)
    return d, v


def read_yahoo(ticker, tag, cut_iso):
    """Yahoo adjusted close (pubdata's CSV: date, close, adj_close, dividends, splits) -> (days, adj_close), rows
    after cut_iso dropped."""
    lines = [l for l in (YAHOOS[tag] / f'{ticker}.csv').read_text().splitlines() if l]
    head = lines[0].split(',')
    j = head.index('adj_close')
    rows = [l.split(',') for l in lines[1:] if l[:10] <= cut_iso]
    d = to_days(np.array([r[0] for r in rows], dtype='datetime64[D]'))
    v = np.array([float(r[j]) if r[j] != '' else np.nan for r in rows])
    return d, v


def exact(d, v, axis):
    """Values on axis days where the series has a value dated exactly that day; NaN elsewhere."""
    pos = np.searchsorted(d, axis)
    ok = (pos < len(d)) & (d[np.minimum(pos, len(d) - 1)] == axis)
    out = np.full(len(axis), np.nan)
    out[ok] = v[pos[ok]]
    return out


def carry(d, v, axis, max_carry=MAX_CARRY):
    """FRED value dated the axis day; on a day with none (a bond holiday) the previous axis day's value, at most
    max_carry days in a row, and never past the series' last date. Returns (values, number of carried days)."""
    x = exact(d, v, axis)
    out = x.copy()
    n_car, run = 0, 0
    for t in range(1, len(x)):
        if np.isfinite(x[t]):
            run = 0
            continue
        run += 1
        if run <= max_carry and np.isfinite(out[t - 1]) and axis[t] <= d[-1]:
            out[t] = out[t - 1]
            n_car += 1
    return out, n_car


def store_close(tag, ticker, axis):
    """A fund's adjusted close from the tag's store on the axis days, forward-filled across one-day gaps after its
    first close (the store has no rows on 2011-02-17, as -3.2 found for SPY); NaN before its first close."""
    S = c2eval.load_store(STORES[tag])
    sid = [s['id'] for s in S['syms'] if s['ticker'] == ticker and s['kind'] == 'etf']
    assert len(sid) == 1, (ticker, sid)
    c = c2eval.store_panel(S, sid, axis, 'close')[0].astype(np.float64)
    first = int(np.argmax(np.isfinite(c)))
    filled = []
    for t in range(first + 1, len(c)):
        if not np.isfinite(c[t]):
            c[t] = c[t - 1]
            filled.append(iso(axis[t]))
    return c, filled


def load_inputs(tag):
    z = np.load(WORK / f'inputs_{tag}.npz')
    return {k: z[k] for k in z.files}


# ---- Fitting --------------------------------------------------------------------------------------------------------

def refit_days(axis_len, offset):
    """-3.2's calendar (export days 252 + 21k) on the axis, extended back in steps of 21 axis days."""
    first = offset + 252
    back = np.arange(first, -1, -H)[::-1]
    fwd = np.arange(first, axis_len, H)
    return np.unique(np.concatenate([back, fwd]))


FIT_LOG = {'fits': 0, 'lstsq_checks': 0}


def run_model(X, y, sample, refits, h, min_fit):
    """Expanding-window least squares with an intercept, refit on each refit day tk with at least min_fit eligible
    origins; a fit at tk uses only origins s in `sample` with s + 1 + h <= tk (asserted), and forecasts days
    tk .. next refit - 1 where every regressor is finite. Normal equations from cumulative sums; every 25th fit is
    checked against lstsq on the explicit origins. Returns yhat, fit index per day, fit days, betas, origins per fit."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    T = len(y)
    A = np.column_stack([np.ones(T), X])
    xok = np.isfinite(X).all(1)
    smp = sample & xok & np.isfinite(y)
    A0 = np.where(smp[:, None], A, 0.0)
    A0 = np.nan_to_num(A0)
    y0 = np.where(smp, y, 0.0)
    k = A.shape[1]
    cXX = np.concatenate([np.zeros((1, k, k)), np.cumsum(A0[:, :, None] * A0[:, None, :], 0)])
    cXy = np.concatenate([np.zeros((1, k)), np.cumsum(A0 * y0[:, None], 0)])
    cn = np.concatenate([[0], np.cumsum(smp)])
    idx_s = np.nonzero(smp)[0]
    yhat = np.full(T, np.nan)
    fit_of = np.full(T, -1, int)
    fdays, betas, nobs = [], [], []
    for j, tk in enumerate(refits):
        tk = int(tk)
        last_s = tk - 1 - h                                  # s + 1 + h <= tk
        if last_s < 0:
            continue
        n = int(cn[last_s + 1])
        if n < min_fit:
            continue
        newest = idx_s[n - 1]
        assert newest + 1 + h <= tk                          # the training-eligibility rule
        beta = np.linalg.solve(cXX[last_s + 1], cXy[last_s + 1])
        if len(fdays) % 25 == 0:
            el = idx_s[:n]
            b2, *_ = np.linalg.lstsq(A[el], y[el], rcond=None)
            p1, p2 = A[el] @ beta, A[el] @ b2
            assert np.max(np.abs(p1 - p2)) < 1e-6 * (1 + np.max(np.abs(p2))), 'normal equations disagree with lstsq'
            FIT_LOG['lstsq_checks'] += 1
        FIT_LOG['fits'] += 1
        t_next = int(refits[j + 1]) if j + 1 < len(refits) else T
        rng_ = np.arange(tk, min(t_next, T))
        ok = rng_[xok[rng_]]
        yhat[ok] = A[ok] @ beta
        fit_of[rng_] = len(fdays)
        fdays.append(tk)
        betas.append(beta)
        nobs.append(n)
    return {'yhat': yhat, 'fit_of': fit_of, 'fit_days': np.array(fdays, int), 'betas': np.array(betas),
            'nobs': np.array(nobs, int)}


def compare(base_cols, alt_cols, y, h, refits, min_fit, extra_mask=None):
    """Fit the baseline and the alternative on the same origins (every regressor of both, and the target, finite);
    also the historical mean (intercept only) on those origins."""
    Xb, Xa = np.column_stack(base_cols), np.column_stack(alt_cols)
    common = np.isfinite(Xb).all(1) & np.isfinite(Xa).all(1) & np.isfinite(y)
    if extra_mask is not None:
        common &= extra_mask
    mb = run_model(Xb, y, common, refits, h, min_fit)
    ma = run_model(Xa, y, common, refits, h, min_fit)
    mh = run_model(np.zeros((len(y), 0)), y, common, refits, h, min_fit)
    # the alternative's forecasts are scored only where the baseline's exist too (same fits, same days)
    return mb, ma, mh


# ---- Statistics -----------------------------------------------------------------------------------------------------

def nw_se(x, lags):
    """Newey-West (Bartlett) standard error of the mean of x."""
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for l in range(1, min(lags, n - 1) + 1):
        v += 2.0 * (1.0 - l / (lags + 1.0)) * (e[l:] @ e[:-l]) / n
    return float(np.sqrt(max(v, 0.0) / n))


def nw_t(x, lags):
    x = np.asarray(x, float)
    if len(x) < 3:
        return float('nan')
    se = nw_se(x, lags)
    return float(x.mean() / se) if se > 0 else float('nan')


def p_upper(t):
    """One-sided normal p of a t-statistic (upper tail)."""
    return 0.5 * (1 - erf(t / sqrt(2))) if np.isfinite(t) else float('nan')


def _block_index(n, block, n_draws, seed):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, size=(n_draws, nb))
    return st, nb, n - (nb - 1) * block


def block_sums(x, st, nb, rem, block):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return (cs[st[:, :-1] + block] - cs[st[:, :-1]]).sum(1) + (cs[st[:, -1] + rem] - cs[st[:, -1]])


def block_mean_draws(x, block=BLOCK_DAYS, n_draws=N_BOOT, seed=SEED):
    """Moving-block bootstrap means of x (blocks of `block` consecutive values, the last block cut so each draw has
    n values). Copied from an earlier timing test's code."""
    x = np.asarray(x, float)
    st, nb, rem = _block_index(len(x), block, n_draws, seed)
    return block_sums(x, st, nb, rem, block) / len(x)


def block_ratio_draws(a, b, block=BLOCK_DAYS, n_draws=N_BOOT, seed=SEED + 5):
    """Paired moving-block bootstrap of 1 - mean(a) / mean(b) (same blocks for both)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    st, nb, rem = _block_index(len(a), block, n_draws, seed)
    return 1.0 - block_sums(a, st, nb, rem, block) / block_sums(b, st, nb, rem, block)


def ols_nw(X, y, lags):
    """Least squares with an intercept and Newey-West (Bartlett) standard errors: (beta, se, t)."""
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    g = X * u[:, None]
    S = g.T @ g
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = g[l:].T @ g[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return beta, se, beta / se


def evaluate(y, fb, fa, mask, lags, fh=None, boot=True, seed=SEED, block=BLOCK_DAYS):
    """Out-of-sample comparison of a baseline (fb) and a larger model (fa) on the origins in mask: raw MSEs, the
    improvement 1 - MSE_a / MSE_b, Clark-West (Newey-West t, normal p, moving-block bootstrap p at `block` and at
    21 days), and each model's out-of-sample R^2 against the historical mean fh."""
    m = mask & np.isfinite(y) & np.isfinite(fb) & np.isfinite(fa)
    if fh is not None:
        m &= np.isfinite(fh)
    n = int(m.sum())
    if n < 30:
        return {'n': n}
    yy, b, a = y[m], fb[m], fa[m]
    eb, ea = yy - b, yy - a
    mse_b, mse_a = float(np.mean(eb ** 2)), float(np.mean(ea ** 2))
    f = eb ** 2 - ea ** 2 + (b - a) ** 2
    se = nw_se(f, lags)
    t = float(f.mean() / se) if se > 0 else float('nan')
    out = {'n': n, 'mse_base': mse_b, 'mse_alt': mse_a, 'rel_impr': 1 - mse_a / mse_b, 'cw_mean': float(f.mean()),
           'cw_se': se, 'cw_t': t, 'cw_p_normal': p_upper(t)}
    if fh is not None:
        eh = yy - fh[m]
        mse_h = float(np.mean(eh ** 2))
        out['mse_histmean'] = mse_h
        out['r2os_base'] = 1 - mse_b / mse_h
        out['r2os_alt'] = 1 - mse_a / mse_h
    if boot and n >= 2 * block:
        dr = block_mean_draws(f - f.mean(), block=block, seed=seed)
        out['cw_boot_p'] = float(np.mean(dr >= f.mean()))
        if n >= 2 * BLOCK_ALT:
            dr21 = block_mean_draws(f - f.mean(), block=BLOCK_ALT, seed=seed + 2)
            out['cw_boot_p_21'] = float(np.mean(dr21 >= f.mean()))
        rr = block_ratio_draws(ea ** 2, eb ** 2, block=block, seed=seed + 1)
        out['rel_impr_lo95'], out['rel_impr_hi95'] = float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))
    return out


def sel(dates, lo, hi):
    return (dates >= lo) & (dates <= hi)


def fmt(x, d=3):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.{d}f}'
