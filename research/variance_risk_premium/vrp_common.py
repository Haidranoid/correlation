"""The variance risk premium (experiments/untested-variance-risk-premium.md): shared settings and helpers.

The rules are fixed in the plan ("Definitions" and "Rules"); choices the plan leaves open are in its "Decisions"
section. Everything here is point in time: a value dated day t uses data through the close of t only (VIX's 16:15
close of day t included, as an earlier timing test settled).

Reuses, read-only and never edited: research/lib/c2eval.py (exports and stores, for the calendar, QQQ and the SPY
check), research/published_data/pubdata.py (Yahoo fund closes, French's RF, DGS3MO) and
research/market_timing/engine.py (measures, random timing, the 12-month block bootstrap of Sharpe ratios).

    import sys; sys.path.insert(0, '/home/bhoult/programming/correlation/research/variance_risk_premium'); import vrp_common as V
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import csv  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
sys.path.insert(1, '/home/bhoult/programming/correlation/research/published_data')
sys.path.insert(2, '/home/bhoult/programming/correlation/research/market_timing')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402
import pubdata  # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/variance_risk_premium'
WORK.mkdir(parents=True, exist_ok=True)
SERIES = Path.home() / '.local/share/correlation2/external/series'
FRED = {'vix': SERIES / 'fred_vixcls.csv', 'vxn': SERIES / 'fred_vxncls.csv', 'rvx': SERIES / 'fred_rvxcls.csv',
        'dgs3mo': SERIES / 'fred_dgs3mo.csv'}
FF3_CSV = pubdata.DATA / 'ff3_daily.csv'
ETF_DIR = pubdata.DATA / 'etf'
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}
LAST = {'old': '2025-06-27', 'new': '2026-10-19'}
TAGS = ('old', 'new', 'fwd')
# Forward runs (forward.py): a later download, named by environment variables. VRP_FWD_DIR holds SPY.csv and IWM.csv
# (pubdata's etf/ format), fred_vixcls.csv, fred_vxncls.csv, fred_rvxcls.csv, fred_dgs3mo.csv and ff3_daily.csv;
# VRP_FWD_STORE (optional) is a later Stooq store, for QQQ.
if os.environ.get('VRP_FWD_DIR'):
    _F = Path(os.environ['VRP_FWD_DIR'])
    FWD_FILES = {'etf': _F, 'vix': _F / 'fred_vixcls.csv', 'vxn': _F / 'fred_vxncls.csv', 'rvx': _F / 'fred_rvxcls.csv',
                 'dgs3mo': _F / 'fred_dgs3mo.csv', 'ff3': _F / 'ff3_daily.csv'}
    if os.environ.get('VRP_FWD_STORE'):
        STORES['fwd'] = Path(os.environ['VRP_FWD_STORE'])
    LAST['fwd'] = '2100-12-31'
else:
    FWD_FILES = None

# ---- Fixed by the plan ------------------------------------------------------------------------------------------
H = 63                 # the primary horizon, trading days
HORIZONS = (21, 42, 126, 252)   # reported only
RV_WIN = 21            # realized variance window: days t-20 .. t
STEP = 21              # -3.2's calendar spacing
CAL_ANCHOR = '2002-12-31'   # export day 252 = the first date of -3.2's calendar
FIT_FROM = '2008-01-02'     # forecasts start here; the first fit is the last calendar date before it (2007-12-03)
TRAIN_FROM = '1993-03-02'   # first origin with 21 SPY returns
REPL_FROM = '2004-01-02'    # the replications: the first origin with 21 returns after this day
NW_LAGS = 2 * H        # 126
BLOCK = 2 * H          # 126-day blocks for the bootstrap of Clark-West's f
BLOCKS_SENS = (63, 189, 252)
N_BOOT = 10_000
BAR_T = 2.0            # t >= 2 (published strategies), one-sided normal p <= 0.023
BAR_P = 0.023
FILL_MAX = 3           # a missing VIX value takes the previous trading day's, at most 3 days back
C_MAX, C_MIN = 1260, 252    # VT's c: median of sigma^2 over the previous 1,260 trading days (>= 252)
COST = 5e-4            # 5 bps on every fund trade
MIN_FIT = 126          # fewest eligible origins for a fit (decision before any result)
SEED = 20261025

DEV = ('2008-01-02', '2019-12-31')
CONF = ('2020-01-02', '2025-06-27')
UNSEEN = ('2025-06-30', '2026-10-19')
FWD = ('2026-10-20', '2100-12-31')
NEW_FIRST_DECISION = '2025-07-16'

iso = c2eval.iso


# ---- Loading ----------------------------------------------------------------------------------------------------

def _read_fred(path):
    rows = list(csv.reader(open(path)))[1:]
    d = np.array([np.datetime64(r[0]) for r in rows if r[1] not in ('', '.')])
    v = np.array([float(r[1]) for r in rows if r[1] not in ('', '.')])
    return d, v


def _align(d, v, axis):
    """Values on exact days of the axis (datetime64[D]); NaN elsewhere."""
    pos = np.searchsorted(d, axis)
    ok = (pos < len(d)) & (d[np.minimum(pos, len(d) - 1)] == axis)
    out = np.full(len(axis), np.nan)
    out[ok] = v[pos[ok]]
    return out


def fill_back(x, axis_dates, src_last, max_fill=FILL_MAX):
    """Fill a missing value from the previous trading day's, at most max_fill days back, never past the source's
    last date. Returns (filled array, list of filled dates)."""
    out = x.copy()
    filled = []
    first = int(np.argmax(np.isfinite(x))) if np.isfinite(x).any() else len(x)
    last, age = np.nan, 99
    for t in range(first, len(x)):
        if np.isfinite(x[t]):
            last, age = x[t], 0
        else:
            age += 1
            if age <= max_fill and np.isfinite(last) and axis_dates[t] <= src_last:
                out[t] = last
                filled.append(str(axis_dates[t]))
    return out, filled


def _etf_adj(ticker, tag):
    path = (FWD_FILES['etf'] if tag == 'fwd' else ETF_DIR) / f'{ticker}.csv'
    d, c = pubdata._read(path)
    return d, c['adj_close']


def load(tag, verbose=True):
    """Every daily series on the day axis = Yahoo SPY's trading days through LAST[tag] (asserted). Returns a dict."""
    assert tag in TAGS, tag
    if tag == 'fwd':
        assert FWD_FILES is not None, 'set VRP_FWD_DIR'
    last = np.datetime64(LAST[tag])
    ds, ps = _etf_adj('SPY', tag)
    ok = np.isfinite(ps) & (ds <= last)
    axis = ds[ok]
    P = ps[ok]
    assert axis[-1] <= last
    T = len(axis)
    days = pubdata.to_days(axis)
    dates = axis.astype(str)
    out = {'tag': tag, 'axis': axis, 'days': days, 'dates': dates, 'T': T, 'spy': P}
    # the calendar: -3.2's export day 252 + 21k, carried onto the same days
    a = int(np.searchsorted(dates, CAL_ANCHOR))
    assert dates[a] == CAL_ANCHOR
    out['cal'] = np.arange(a % STEP, T, STEP)
    # the export (old/new): its days must equal the axis over its span, and its day 252 must be the anchor
    if tag in EXPORTS:
        E = c2eval.load_export(EXPORTS[tag], arrays=())
        ed = E['days']
        lo = int(np.searchsorted(days, ed[0]))
        assert np.array_equal(days[lo:lo + len(ed)], ed) and lo + len(ed) == T, 'export days differ from Yahoo SPY days'
        assert dates[lo + 252] == CAL_ANCHOR
        out['export_days'] = len(ed)
    # IWM (Yahoo)
    di, pi = _etf_adj('IWM', tag)
    out['iwm'] = _align(di, pi, axis)
    # the stores: QQQ and SPY closes (forward-filled after their first close, as -3.2's spy_close does)
    out['qqq'] = np.full(T, np.nan)
    out['spy_store'] = np.full(T, np.nan)
    out['store_fills'] = {}
    if tag in STORES:
        S = c2eval.load_store(STORES[tag])
        for tk in ('QQQ', 'SPY'):
            sid = [s['id'] for s in S['syms'] if s['ticker'] == tk and s['kind'] == 'etf']
            assert len(sid) == 1, (tk, sid)
            c = c2eval.store_panel(S, sid, days, 'close')[0].astype(np.float64)
            first = int(np.argmax(np.isfinite(c)))
            fills = []
            for t in range(first + 1, T):
                if not np.isfinite(c[t]):
                    c[t] = c[t - 1]
                    fills.append(dates[t])
            out['qqq' if tk == 'QQQ' else 'spy_store'] = c
            out['store_fills'][tk] = fills
    # FRED volatility indexes, filled at most 3 days back
    out['fills'] = {}
    for k in ('vix', 'vxn', 'rvx'):
        d, v = _read_fred(FWD_FILES[k] if tag == 'fwd' else FRED[k])
        x = _align(d, v, axis)
        x[axis > last] = np.nan
        out[k], out['fills'][k] = fill_back(x, axis, d[-1])
        out[f'{k}_last'] = str(d[-1])
    # risk-free: French's daily RF through its last day, then DGS3MO / 252 (last value carried forward)
    fp = FWD_FILES['ff3'] if tag == 'fwd' else FF3_CSV
    fd, ff = pubdata._read(fp)
    rf = _align(fd, ff['RF'], axis)
    gp = FWD_FILES['dgs3mo'] if tag == 'fwd' else FRED['dgs3mo']
    gd, gv = _read_fred(gp)
    pos = np.searchsorted(gd, axis, side='right') - 1
    dgs = np.where(pos >= 0, gv[np.maximum(pos, 0)] / 100.0 / 252.0, np.nan)
    miss = ~np.isfinite(rf)
    out['rf_french_last'] = str(fd[-1])
    out['rf_fill_after_french'] = int((miss & (axis > fd[-1])).sum())
    out['rf_fill_inside_french'] = [str(x) for x in axis[miss & (axis <= fd[-1])]]
    rf[miss] = dgs[miss]
    out['rf'] = rf
    if verbose:
        print(f"[{tag}] axis {dates[0]}..{dates[-1]} ({T} days); calendar {len(out['cal'])} dates; fills {out['fills']}; "
              f"RF from DGS3MO on {out['rf_fill_after_french']} days after {out['rf_french_last']} "
              f"and {len(out['rf_fill_inside_french'])} inside", flush=True)
    return out


# ---- Series -----------------------------------------------------------------------------------------------------

def logret(P):
    P = np.asarray(P, np.float64)
    r = np.full(len(P), np.nan)
    with np.errstate(all='ignore'):
        r[1:] = np.log(P[1:] / P[:-1])
    r[~np.isfinite(r)] = np.nan
    return r


def window_sum(x, lo, hi):
    """sum of x[lo..hi] inclusive, vectorized; NaN if any value missing or the window is out of range."""
    x = np.asarray(x, float)
    T = len(x)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    cn = np.concatenate([[0], np.cumsum(~np.isfinite(x))])
    lo, hi = np.asarray(lo), np.asarray(hi)
    ok = (lo >= 0) & (hi <= T - 1) & (hi >= lo)
    lo_c, hi_c = np.clip(lo, 0, T - 1), np.clip(hi, 0, T - 1)
    return np.where(ok & (cn[hi_c + 1] - cn[lo_c] == 0), cs[hi_c + 1] - cs[lo_c], np.nan)


def rv(r, win=RV_WIN):
    """RV_t = 10,000 x sum of r^2 over days t-win+1 .. t (monthly percent squared); NaN unless all exist."""
    t = np.arange(len(r))
    return 1e4 * window_sum(np.asarray(r) ** 2, t - win + 1, t)


def target(r, rf, h):
    """y_t = sum over days t + 2 .. t + 1 + h of (r_d - ln(1 + RF_d)): the log excess return from the close of t + 1
    to the close of t + 1 + h. Checked directly on a sample of origins."""
    ex = np.asarray(r) - np.log1p(np.asarray(rf))
    t = np.arange(len(r))
    y = window_sum(ex, t + 2, t + 1 + h)
    for s in range(30, len(r) - h - 2, 911):
        if np.isfinite(y[s]):
            assert abs(y[s] - ex[s + 2:s + 2 + h].sum()) < 1e-12
    return y


def sigma2(r, t0):
    """-3.2's variance: np.var (ddof 0) of the 21 daily log returns ending at t0; NaN unless all exist."""
    w = np.asarray(r)[t0 - RV_WIN + 1:t0 + 1]
    return float(np.var(w)) if len(w) == RV_WIN and np.isfinite(w).all() else float('nan')


def build(D, price_key='spy', vol_key='vix', start=None):
    """The premium and its pieces for one fund and one volatility index: r, RV, IV, VRP, log VIX, log RV and the
    targets. start: the first origin allowed (ISO), else every origin with data."""
    P = D[price_key]
    r = logret(P)
    RV = rv(r)
    VOL = D[vol_key]
    IV = VOL ** 2 / 12.0
    out = {'r': r, 'rv': RV, 'iv': IV, 'vrp': IV - RV, 'lvix': np.log(VOL), 'lrv': np.log(RV), 'vol': VOL}
    out['ok_from'] = np.ones(D['T'], bool) if start is None else (D['dates'] >= start)
    return out


# ---- Fitting ----------------------------------------------------------------------------------------------------

def ols(X, y):
    A = np.column_stack([np.ones(len(y))] + ([X] if X.shape[1] else []))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta, y - A @ beta


def predict(beta, X):
    return beta[0] + (X @ beta[1:] if X.shape[1] else 0.0)


FIT_COUNT = {'fits': 0}


def run_model(D, cols, y, h, sample, fit_from=FIT_FROM, min_fit=MIN_FIT, restrict=None):
    """Expanding-window least squares, refit on each calendar date t_k from the last one before fit_from; forecasts
    for days t_k .. t_{k+1} - 1. A fit at t_k uses only origins s in `sample` with s + 1 + h <= t_k (asserted).
    restrict: None, or 'ct' (Campbell-Thompson: a negative slope uses the mean, negative forecasts become 0)."""
    T = D['T']
    cal = D['cal']
    X = np.column_stack(cols) if cols else np.zeros((T, 0))
    xok = np.isfinite(X).all(1)
    t_idx = np.arange(T)
    yhat = np.full(T, np.nan)
    fit_of = np.full(T, -1, int)
    fd = int(np.searchsorted(D['dates'], fit_from))
    k0 = max(0, int(np.searchsorted(cal, fd, side='right')) - 1)
    fdays, betas, nobs = [], [], []
    for k in range(k0, len(cal)):
        tk = int(cal[k])
        t_next = int(cal[k + 1]) if k + 1 < len(cal) else T
        elig = np.nonzero(sample & (t_idx + 1 + h <= tk))[0]
        if len(elig) < min_fit:
            continue
        assert elig.max() + 1 + h <= tk                      # the training-eligibility rule
        assert np.isfinite(y[elig]).all() and xok[elig].all()
        FIT_COUNT['fits'] += 1
        beta, _ = ols(X[elig], y[elig])
        rng = np.arange(tk, t_next)
        ok = xok[rng]
        f = predict(beta, X[rng[ok]])
        if restrict == 'ct':
            if X.shape[1] and beta[1] < 0:
                f = np.full(ok.sum(), y[elig].mean())
            f = np.maximum(f, 0.0)
        yhat[rng[ok]] = f
        fit_of[rng] = len(fdays)
        fdays.append(tk)
        betas.append(beta)
        nobs.append(len(elig))
    return {'yhat': yhat, 'fit_of': fit_of, 'fit_days': np.array(fdays, int), 'betas': np.array(betas),
            'nobs': np.array(nobs), 'X': X}


def compare(D, models, y, h, start_mask=None, fit_from=FIT_FROM):
    """Fit several models on the same origins (every model's regressors and the target finite). models: dict name ->
    list of columns. Returns dict name -> run_model output, and the common sample."""
    T = D['T']
    common = np.isfinite(y)
    for cols in models.values():
        for c in cols:
            common &= np.isfinite(c)
    if start_mask is not None:
        common &= start_mask
    return {nm: run_model(D, cols, y, h, common, fit_from=fit_from) for nm, cols in models.items()}, common


# ---- Statistics -------------------------------------------------------------------------------------------------

def nw_se(x, lags):
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for l in range(1, min(lags, n - 1) + 1):
        v += 2.0 * (1.0 - l / (lags + 1.0)) * (e[l:] @ e[:-l]) / n
    return float(np.sqrt(max(v, 0.0) / n))


def norm_sf(t):
    from math import erf, sqrt
    return 0.5 * (1 - erf(t / sqrt(2))) if np.isfinite(t) else float('nan')


def _block_sums(x, st, nb, block):
    n = len(x)
    rem = n - (nb - 1) * block
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return (cs[st[:, :-1] + block] - cs[st[:, :-1]]).sum(1) + (cs[st[:, -1] + rem] - cs[st[:, -1]])


def block_starts(n, block, n_draws, seed):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    return rng.integers(0, n - block + 1, size=(n_draws, nb)), nb


def block_mean_draws(x, block=BLOCK, n_draws=N_BOOT, seed=SEED):
    """Moving-block bootstrap means of x (blocks of `block` consecutive values, the last block cut to length n)."""
    x = np.asarray(x, float)
    st, nb = block_starts(len(x), block, n_draws, seed)
    return _block_sums(x, st, nb, block) / len(x)


def block_ratio_draws(a, b, block=BLOCK, n_draws=N_BOOT, seed=SEED + 5):
    """Paired moving-block bootstrap of 1 - sum(a) / sum(b)."""
    st, nb = block_starts(len(a), block, n_draws, seed)
    return 1.0 - _block_sums(np.asarray(a, float), st, nb, block) / _block_sums(np.asarray(b, float), st, nb, block)


def evaluate(y, fb, fa, mask, lags=NW_LAGS, block=BLOCK, boot=True, seed=SEED, blocks_sens=()):
    """Out-of-sample comparison of a nested pair: base forecasts fb (the smaller model), alternative fa.
    R2_oos = 1 - SSE_alt / SSE_base; Clark-West f = e_b^2 - e_a^2 + (fb - fa)^2, its Newey-West t and the one-sided
    moving-block bootstrap p of its mean (f minus its mean resampled; share of draws >= the observed mean)."""
    m = mask & np.isfinite(y) & np.isfinite(fb) & np.isfinite(fa)
    n = int(m.sum())
    if n < 30:
        return {'n': n}
    yy, b, a = y[m], fb[m], fa[m]
    eb, ea = yy - b, yy - a
    f = eb ** 2 - ea ** 2 + (b - a) ** 2
    se = nw_se(f, lags)
    out = {'n': n, 'mse_base': float(np.mean(eb ** 2)), 'mse_alt': float(np.mean(ea ** 2)),
           'r2_oos': float(1 - np.sum(ea ** 2) / np.sum(eb ** 2)), 'cw_mean': float(f.mean()), 'cw_se': se,
           'cw_t': float(f.mean() / se) if se > 0 else float('nan')}
    out['cw_p_normal'] = norm_sf(out['cw_t'])
    if boot and n >= 2 * block:
        dr = block_mean_draws(f - f.mean(), block=block, seed=seed)
        out['cw_boot_p'] = float(np.mean(dr >= f.mean()))
        rr = block_ratio_draws(ea ** 2, eb ** 2, block=block, seed=seed + 1)
        out['r2_lo95'], out['r2_hi95'] = float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))
        for bl in blocks_sens:
            if n >= 2 * bl:
                out[f'cw_boot_p_block{bl}'] = float(np.mean(block_mean_draws(f - f.mean(), block=bl, seed=seed) >= f.mean()))
    return out


def plain_t(x):
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else float('nan')


def hodrick_1b(x, r1, k):
    """Slope and Hodrick (1992) 1B t-statistic for regressing the k-period sum of r1 on [1, x], with origins spaced one
    period apart. r1[m] = the one-period return following origin m; y[m] = r1[m] + ... + r1[m + k - 1]. Under the
    null, u = r1 - mean(r1); w_m = u_m x sum_{j=0}^{k-1} Z_{m-j}. Returns dict (beta, t, adj R2, n)."""
    x, r1 = np.asarray(x, float), np.asarray(r1, float)
    M = len(x)
    y = np.full(M, np.nan)
    for m in range(M - k + 1):
        y[m] = r1[m:m + k].sum()
    ok = np.isfinite(y) & np.isfinite(x)
    Z = np.column_stack([np.ones(M), x])
    beta, *_ = np.linalg.lstsq(Z[ok], y[ok], rcond=None)
    res = y[ok] - Z[ok] @ beta
    r2 = 1 - res.var() / y[ok].var()
    n = int(ok.sum())
    adj = 1 - (1 - r2) * (n - 1) / (n - 2)
    u = r1 - np.nanmean(r1)
    S = np.zeros((2, 2))
    cnt = 0
    for m in range(k - 1, M):
        if not (np.isfinite(u[m]) and np.isfinite(x[m - k + 1:m + 1]).all()):
            continue
        zs = Z[m - k + 1:m + 1].sum(0)
        w = u[m] * zs
        S += np.outer(w, w)
        cnt += 1
    S /= cnt
    Q = Z[ok].T @ Z[ok] / n
    Qi = np.linalg.inv(Q)
    Vb = Qi @ S @ Qi / n
    return {'beta': float(beta[1]), 'const': float(beta[0]), 't': float(beta[1] / np.sqrt(Vb[1, 1])), 'adj_r2': float(adj), 'n': n}


def period_mask(D, lo, hi, h):
    """Origins t with t >= lo and the whole target (days t + 2 .. t + 1 + h) on or before hi."""
    T = D['T']
    t = np.arange(T)
    end = t + 1 + h
    ok = end <= T - 1
    m = np.zeros(T, bool)
    m[ok] = (D['dates'][t[ok]] >= lo) & (D['dates'][end[ok]] <= hi)
    return m


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float('nan')


def fmt(x, d=3):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.{d}f}'
