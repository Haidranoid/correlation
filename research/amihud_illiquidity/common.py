"""Amihud illiquidity (experiments/untested-amihud-illiquidity.md): the plan's rules as constants and shared helpers.

The rules are in the .md ("Rules (proposed; fixed before any result is seen)"); choices the plan left open are in
its "Decisions". Read-only on the data. Copies from other experiments say where they come from.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import c2eval  # noqa: E402
import pubdata  # noqa: E402
import numpy as np  # noqa: E402

WORK = c2eval.WORK / 'amihud_illiquidity'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

# ---- the plan's rules ----
H = 21                    # holding period and grid spacing, trading days
DELAY = 1                 # trade at the close of t + 1; hold to the close of t + 1 + H
ILLIQ_W, ILLIQ_MIN = 252, 200
ILLIQ_MIN_77 = 63         # the unscreened variant's minimum (the reported variant)
BAD_LOG = 0.5             # |log return| above this: a bad print
BROKEN_FRAC, BROKEN_W, BROKEN_MIN = 0.01, 63, 50
DATA_W, DATA_FRAC, MED_W = 756, 0.9, 252
MIN_PRICE, MIN_DV = 5.0, 5e6
N_LARGE, DV_LARGE_W = 500, 63
TRIM = 0.01               # Amihud's 1% rule, each tail
BETA_W, BETA_MIN = 252, 200
MOM_FROM, MOM_TO = 252, 21
NW_LAGS = 4
BLOCK, N_BOOT, SEED = 3, 10_000, 20261025
COST_BPS = 5.0
FIFTH = 0.2
MIN_UNIV = 100
ALPHA = 0.5 * math.erfc(2.0 / math.sqrt(2.0))     # one-sided p at t = 2: 0.02275
L_HAIRCUT = -0.55
S1_MIN_DAYS = 15

DEV = ('2008-01-01', '2019-12-31')
CHECK = ('2020-01-01', '2025-06-27')
PRE = ('2006-01-01', '2007-12-31')
UNSEEN = ('2025-06-30', '2026-10-19')
DIV_FROM = '2022-07-01'
OLD_T = 5911

FORWARD_FROM = '2026-10-20'   # forward periods: first return day after the unseen months


def register_forward(export_dir, store_dir):
    """Point the 'fwd' tag at a later download (forward.py): its export and the store the twins read."""
    export_dir, store_dir = Path(export_dir), Path(store_dir)
    EXPORTS['fwd'] = export_dir
    STORES['fwd'] = store_dir
    c2eval.STORE_FOR[str(export_dir)] = store_dir


if os.environ.get('AMIHUD_FWD_EXPORT') and os.environ.get('AMIHUD_FWD_STORE'):
    register_forward(os.environ['AMIHUD_FWD_EXPORT'], os.environ['AMIHUD_FWD_STORE'])

SURV_MODEL = Path('/home/bhoult/programming/correlation/research/survivorship_stress/out/part1.json')
M77 = c2eval.WORK / 'amihud_reference'

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def norm_sf(t):
    """One-sided normal p for a t (upper tail)."""
    return 0.5 * math.erfc(t / math.sqrt(2.0)) if np.isfinite(t) else float('nan')


# ---- the measure ----
def amihud_panel_77(close, dv):
    """the unscreened 63-day Amihud panel, copied unchanged from an earlier test's code (amihud_panel): mean of
    |close[d]/close[d-1] - 1| / dollar_volume[d] over the 252 days ending at d, at least 63 valid days."""
    c = np.asarray(close, np.float64)
    v = np.asarray(dv, np.float64)
    N, T = c.shape
    x = np.full((N, T), np.nan)
    with np.errstate(all='ignore'):
        x[:, 1:] = np.abs(c[:, 1:] / c[:, :-1] - 1.0) / v[:, 1:]
    x[~np.isfinite(x) | ~(v > 0)] = np.nan
    ok = np.isfinite(x)
    cs = np.zeros((N, T + 1))
    np.cumsum(np.where(ok, x, 0.0), axis=1, out=cs[:, 1:])
    cn = np.zeros((N, T + 1), np.int32)
    np.cumsum(ok, axis=1, out=cn[:, 1:])
    del x, ok
    lo = np.maximum(np.arange(T) + 1 - 252, 0)
    s = cs[:, 1:] - cs[:, lo]
    n = cn[:, 1:] - cn[:, lo]
    with np.errstate(all='ignore'):
        A = np.where(n >= 63, s / n, np.nan)
    return A.astype(np.float32)


def rolling_max_prev(x, w):
    """[N, T] max of x over days d-w..d-1 (x >= 0, NaN treated as 0), by doubling; 0 where no earlier day."""
    N, T = x.shape
    a = np.nan_to_num(np.asarray(x, np.float32), nan=0.0)
    # m[p][:, d] = max over d-2^p+1 .. d
    pw, m = 1, a
    while pw * 2 <= w:
        nm = m.copy()
        nm[:, pw:] = np.maximum(m[:, pw:], m[:, :-pw])
        pw *= 2
        m = nm
    # window of length w ending at d-1 = [d-w, d-1]: two overlapping blocks of length pw
    out = np.zeros((N, T), np.float32)
    e1 = np.zeros((N, T), np.float32)   # block ending at d-1
    e1[:, 1:] = m[:, :-1]
    e2 = np.zeros((N, T), np.float32)   # block ending at d-w+pw-1
    sh = w - pw + 1
    e2[:, sh:] = m[:, :-sh]
    np.maximum(e1, e2, out=out)
    return out


def ratio_panel(close, dv, screens=True, counts=None):
    """[N, T] float64 daily |r| / DV on the days that count (NaN elsewhere).
    r = close[d]/close[d-1] - 1 (both present), DV = dollar_volume[d] > 0. With screens: |log r| <= BAD_LOG, and DV
    >= BROKEN_FRAC x the median positive DV over d-63..d-1 when that median has >= BROKEN_MIN days.
    counts: optional dict, filled with the screens' flag masks summary."""
    c = np.asarray(close, np.float64)
    v = np.asarray(dv, np.float64)
    N, T = c.shape
    x = np.full((N, T), np.nan)
    with np.errstate(all='ignore'):
        g = c[:, 1:] / c[:, :-1]
        x[:, 1:] = np.abs(g - 1.0) / v[:, 1:]
    del c
    base = np.isfinite(x) & (v > 0)
    x[~base] = np.nan
    if not screens:
        return x
    bad = np.zeros((N, T), bool)
    with np.errstate(all='ignore'):
        bad[:, 1:] = ~(np.abs(np.log(g)) <= BAD_LOG)
    del g
    bad &= base
    # broken volume: candidates by the rolling max (>= the median), then the exact median for the candidates
    vp = np.where(v > 0, v, np.nan).astype(np.float32)
    okp = np.isfinite(vp)
    cn = np.zeros((N, T + 1), np.int32)
    np.cumsum(okp, axis=1, out=cn[:, 1:])
    ar = np.arange(T)
    lo = np.maximum(ar - BROKEN_W, 0)
    nprev = cn[:, ar] - cn[:, lo]          # count over d-63..d-1
    mx = rolling_max_prev(vp, BROKEN_W)
    cand = base & ~bad & (nprev >= BROKEN_MIN) & (v < BROKEN_FRAC * mx)
    del mx, cn
    broken = np.zeros((N, T), bool)
    ii, dd = np.nonzero(cand)
    for i, d in zip(ii, dd):
        w = vp[i, d - BROKEN_W:d]
        w = w[np.isfinite(w)]
        if len(w) >= BROKEN_MIN and v[i, d] < BROKEN_FRAC * float(np.median(w.astype(np.float64))):
            broken[i, d] = True
    if counts is not None:
        counts['base_days'] = int(base.sum())
        counts['bad_print_days'] = int(bad.sum())
        counts['broken_candidates'] = int(len(ii))
        counts['broken_days'] = int(broken.sum())
        counts['broken_mask'] = broken
        counts['bad_mask'] = bad
    x[bad | broken] = np.nan
    return x


def window_mean(x, w, nmin):
    """[N, T] float32 mean of finite x over the w days ending at d, NaN unless >= nmin finite; and the counts."""
    N, T = x.shape
    ok = np.isfinite(x)
    cs = np.zeros((N, T + 1))
    np.cumsum(np.where(ok, x, 0.0), axis=1, out=cs[:, 1:])
    cn = np.zeros((N, T + 1), np.int32)
    np.cumsum(ok, axis=1, out=cn[:, 1:])
    lo = np.maximum(np.arange(T) + 1 - w, 0)
    s = cs[:, 1:] - cs[:, lo]
    n = cn[:, 1:] - cn[:, lo]
    with np.errstate(all='ignore'):
        A = np.where(n >= nmin, s / n, np.nan)
    return A.astype(np.float32), n.astype(np.int16)


# ---- universes (copied from research/max_lottery/common.py, itself from research/night_and_day) ----
def universe(E, tw):
    """[K, N] bool U_J (the judge-like liquid set), U_W (the same without the dollar-volume floor), the 500 largest
    of U_J by 63-day median dollar volume, and the [K, N] 63-day median dollar volume, on tw['rebal'] days."""
    close, dv = E['close'], E['dollar_volume']
    rebal = tw['rebal']
    N = close.shape[0]
    excl = c2eval.excluded(E, tw)
    uj = np.zeros((len(rebal), N), bool)
    uw = np.zeros((len(rebal), N), bool)
    big = np.zeros((len(rebal), N), bool)
    dv63 = np.full((len(rebal), N), np.nan, np.float32)
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
        w = fin & (np.nan_to_num(mc) >= MIN_PRICE) & ~excl[k]
        u = w & (np.nan_to_num(mv) >= MIN_DV)
        uw[k], uj[k] = w, u
        cand = np.nonzero(u)[0]
        o = cand[np.argsort(-np.nan_to_num(mv63[cand], nan=-1.0), kind='stable')]
        big[k, o[:N_LARGE]] = True
        dv63[k] = mv63
    return uj, uw, big, dv63


def ffill(close):
    """Forward-filled closes (leading NaNs stay NaN). Copied from research/betting_against_beta/common.py."""
    c = np.asarray(close, np.float64)
    ok = np.isfinite(c)
    idx = np.where(ok, np.arange(c.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    P = np.take_along_axis(c, idx, 1)
    P[~ok.any(1)] = np.nan
    first = ok.argmax(1)
    P[np.arange(c.shape[1])[None, :] < first[:, None]] = np.nan
    return P


def log_returns(close):
    """Daily log returns (both closes present), |r| > BAD_LOG -> NaN. float64 [N, T]."""
    c = np.asarray(close, np.float64)
    r = np.full(c.shape, np.nan)
    with np.errstate(all='ignore'):
        r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    r[~np.isfinite(r) | (np.abs(r) > BAD_LOG)] = np.nan
    return r


# ---- statistics (copied from research/max_lottery/common.py) ----
def ols(y, X=None, nw_lags=None):
    """OLS of y on [1, X]. Plain standard errors, or Newey-West (Bartlett) with nw_lags. Returns coef, t, n, r2."""
    y = np.asarray(y, float)
    cols = [np.ones(len(y))]
    if X is not None:
        cols.append(np.asarray(X, float).reshape(len(y), -1))
    X = np.column_stack(cols)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    if nw_lags is None:
        V = XtXi * (e @ e / (n - k))
    else:
        u = X * e[:, None]
        S = u.T @ u
        for L in range(1, nw_lags + 1):
            w = 1 - L / (nw_lags + 1)
            G = u[L:].T @ u[:-L]
            S += w * (G + G.T)
        V = XtXi @ S @ XtXi * n / (n - k)
    se = np.sqrt(np.diag(V))
    sst = (y - y.mean()) @ (y - y.mean())
    r2 = 1 - (e @ e) / sst if sst > 0 else float('nan')
    return b, b / se, n, r2


def mean_t(x, nw_lags=NW_LAGS):
    """Mean, plain t, Newey-West t (Bartlett, nw_lags) and n of a series (NaNs dropped)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float('nan'), float('nan'), float('nan'), n
    m = x.mean()
    t = m / (x.std(ddof=1) / np.sqrt(n))
    u = x - m
    v = (u @ u) / n
    for L in range(1, min(nw_lags, n - 1) + 1):
        v += 2 * (1 - L / (nw_lags + 1)) * (u[L:] @ u[:-L]) / n
    tnw = m / np.sqrt(v / n) if v > 0 else float('nan')
    return float(m), float(t), float(tnw), n


def pct_rank(x):
    """Percentile ranks in [0, 1] of the finite values (average ranks for ties); NaN stays NaN."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    v = x[ok]
    n = len(v)
    if n < 2:
        return out
    o = np.argsort(v, kind='mergesort')
    vs = v[o]
    starts = np.r_[0, np.nonzero(np.diff(vs) != 0)[0] + 1]
    ends = np.r_[starts[1:], n]
    r = np.empty(n)
    r[o] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    out[ok] = r / (n - 1)
    return out


def winsor(x, q=(0.01, 0.99)):
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return x
    lo, hi = np.quantile(x[ok], q)
    x[ok] = np.clip(x[ok], lo, hi)
    return x


def block_boot_p(x, block=BLOCK, n=N_BOOT, seed=SEED):
    """One-sided p of a series' mean by a moving-block bootstrap of the centred series: the share of resampled means
    at or above the observed mean. Also the 2.5/97.5 percentiles of the uncentred resampled means."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 2 * block:
        return float('nan'), (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]
    means = x[idx].mean(1)
    m = x.mean()
    p = float(((means - m) >= m).mean())
    return p, (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def holm(pvals, alpha=ALPHA):
    """Holm's step-down at family level alpha: dict name -> (p, threshold, pass)."""
    order = sorted(pvals, key=lambda k: (pvals[k] if np.isfinite(pvals[k]) else 9))
    m = len(order)
    out, still = {}, True
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        ok = still and np.isfinite(pvals[k]) and pvals[k] < thr
        if not ok:
            still = False
        out[k] = (pvals[k], thr, bool(ok))
    return out


def in_period(iso_arr, a, b):
    iso_arr = np.asarray(iso_arr)
    return (iso_arr >= a) & (iso_arr <= b)


def dumps(obj):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)
    return json.dumps(obj, indent=1, default=conv)


# ---- survivorship hazard model (copied from research/survivorship_stress/common.py; the model read from its JSON) ----
def surv_model():
    return json.loads(SURV_MODEL.read_text())['model']


def surv_features(E, rows, d):
    """features_col: ret12, down50, vol63 (sd ddof 1 of daily log returns over 63 days, >= 42), logp, logdv (63-day
    median dollar volume), at column d for the given export rows."""
    close = E['close']
    if d >= 252:
        a = np.asarray(close[rows, d], np.float64)
        b = np.asarray(close[rows, d - 252], np.float64)
        with np.errstate(all='ignore'):
            ratio = a / b
        ratio[~np.isfinite(ratio)] = np.nan
    else:
        ratio = np.full(len(rows), np.nan)
    lo = max(d - 63, 0)
    cc = np.asarray(close[rows, lo:d + 1], np.float64)
    with np.errstate(all='ignore'):
        r = np.log(cc[:, 1:] / cc[:, :-1])
    ok = np.isfinite(r)
    k = ok.sum(1)
    xr = np.where(ok, r, 0.0)
    s1, s2 = xr.sum(1), (xr * xr).sum(1)
    with np.errstate(all='ignore'):
        var = (s2 - s1 * s1 / k) / (k - 1)
    vol = np.sqrt(np.where(var > 0, var, np.nan))
    vol[k < 42] = np.nan
    c = np.asarray(close[rows, d], np.float64)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mdv = np.nanmedian(np.asarray(E['dollar_volume'][rows, d - 62:d + 1], np.float64), 1)
        return {'ret12': ratio - 1.0, 'down50': np.where(np.isfinite(ratio), (ratio < 0.5).astype(float), 0.0),
                'vol63': vol, 'logp': np.where(c > 0, np.log(c), np.nan), 'logdv': np.where(mdv > 0, np.log(mdv), np.nan)}


def surv_hazard(feat, model):
    """Monthly hazard: logistic p over the model's window, then 1 - (1 - p)^(1/m)."""
    cols = [np.ones(len(feat['ret12']))]
    for name in model['names']:
        if name == 'down50':
            cols.append(feat['down50'])
        else:
            z = (feat[name] - model['mean'][name]) / model['sd'][name]
            cols.append(np.where(np.isfinite(z), z, 0.0))
    X = np.column_stack(cols)
    p = 1.0 / (1.0 + np.exp(-(X @ np.asarray(model['beta']))))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1.0 - (1.0 - p) ** (1.0 / model['m'])
