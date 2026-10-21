"""Night and day (experiments/-6.0-night-and-day.md): shared settings and the panel builder.

The rules are in the .md ("Rules" and "Decisions made while building it"). In short, at the close of day t:
  - Dimson regression per stock over tau = t-251..t: r_i(tau) = a + b0 * m(tau) + b_lag * m(tau-1) + e, with r_i the
    stock's close-to-close log return (export closes; |r| > 0.5 treated as missing) and m SPY's close-to-close log
    return (store closes); at least 227 of the 252 observations.
  - s = b_lag * m(t).
  - The next trading day d = t+1 splits into overnight log(open_d / close_t) and intraday log(close_d / open_d), from
    the store's opens and closes.
  - Universe on c2eval.twins' calendar (export days 252 + 21k), point in time: closes on >= 90% of the trailing 756
    days, median close >= $5 and median dollar volume >= $5M over the trailing 252 days, minus c2eval.excluded().
    The 500 largest: the universe's top 500 by median dollar volume over the trailing 63 days.

    python3 -I build.py old|new      # writes the panels under WORK
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
WORK = c2eval.WORK / 'night_and_day'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

W = 252            # Dimson regression window (observations)
MIN_OBS = 227      # 90% of W
MAX_LOG = 0.5      # |daily log move| above this is treated as a bad print (stock returns in the regression, legs)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500
DECILE = 0.10
MIN_NAMES = 50     # a day needs at least this many ranked stocks

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def spy_store(tag, days):
    """SPY's store open and close aligned to the export days (float64, NaN where missing)."""
    S = c2eval.load_store(STORES[tag])
    sid = [s['id'] for s in S['syms'] if s['ticker'] == 'SPY' and s['kind'] == 'etf']
    assert len(sid) == 1, sid
    o = c2eval.store_panel(S, sid, days, 'open')[0].astype(np.float64)
    c = c2eval.store_panel(S, sid, days, 'close')[0].astype(np.float64)
    return o, c


def dimson(close, m, chunk=500):
    """Rolling Dimson regression of each stock's daily log return on m(tau) and m(tau-1) over the W days ending at
    each t. Returns b0, b_lag [N, T] float32 (NaN where fewer than MIN_OBS observations)."""
    N, T = close.shape
    m1 = np.full(T, np.nan)
    m1[1:] = m[:-1]
    mf = np.isfinite(m) & np.isfinite(m1)
    b0 = np.full((N, T), np.nan, np.float32)
    bl = np.full((N, T), np.nan, np.float32)
    m0z, m1z = np.where(mf, m, 0.0), np.where(mf, m1, 0.0)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        c = np.asarray(close[a:b], np.float64)
        r = np.full(c.shape, np.nan)
        with np.errstate(all='ignore'):
            r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
        r[~np.isfinite(r) | (np.abs(r) > MAX_LOG)] = np.nan
        M = np.isfinite(r) & mf[None, :]
        x0 = np.where(M, m0z[None, :], 0.0)
        x1 = np.where(M, m1z[None, :], 0.0)
        y = np.where(M, r, 0.0)

        def rs(X):
            cs = np.zeros((X.shape[0], T + 1))
            np.cumsum(X, 1, out=cs[:, 1:])
            return cs[:, W:] - cs[:, :-W]          # column i <-> window ending at t = i + W - 1

        n = rs(M.astype(np.float64))
        s0, s1, sy = rs(x0), rs(x1), rs(y)
        with np.errstate(all='ignore'):
            c00 = rs(x0 * x0) - s0 * s0 / n
            c11 = rs(x1 * x1) - s1 * s1 / n
            c01 = rs(x0 * x1) - s0 * s1 / n
            cy0 = rs(y * x0) - sy * s0 / n
            cy1 = rs(y * x1) - sy * s1 / n
            det = c00 * c11 - c01 * c01
            g0 = (c11 * cy0 - c01 * cy1) / det
            g1 = (c00 * cy1 - c01 * cy0) / det
        bad = (n < MIN_OBS) | ~np.isfinite(g0) | ~np.isfinite(g1)
        g0[bad] = np.nan
        g1[bad] = np.nan
        b0[a:b, W - 1:] = g0
        bl[a:b, W - 1:] = g1
    return b0, bl


def legs(o, c, h, lo):
    """Overnight and intraday log legs indexed by the day d they end: on[:, d] = log(o_d / c_{d-1}),
    intra[:, d] = log(c_d / o_d). NaN (both) where a price is missing or <= 0, where the day has no trading range
    (open = close = high = low: a stale print), or where either |leg| > MAX_LOG."""
    N, T = c.shape
    on = np.full((N, T), np.nan, np.float32)
    intra = np.full((N, T), np.nan, np.float32)
    with np.errstate(all='ignore'):
        ok = (np.isfinite(o[:, 1:]) & (o[:, 1:] > 0) & np.isfinite(c[:, 1:]) & (c[:, 1:] > 0)
              & np.isfinite(c[:, :-1]) & (c[:, :-1] > 0))
        stale = (o[:, 1:] == c[:, 1:]) & (h[:, 1:] == lo[:, 1:]) & (o[:, 1:] == h[:, 1:])
        a = np.log(o[:, 1:].astype(np.float64) / c[:, :-1])
        b = np.log(c[:, 1:].astype(np.float64) / o[:, 1:])
        ok &= ~stale & (np.abs(a) <= MAX_LOG) & (np.abs(b) <= MAX_LOG)
    on[:, 1:] = np.where(ok, a, np.nan)
    intra[:, 1:] = np.where(ok, b, np.nan)
    return on, intra


def universe(E, tw):
    """[K, N] bool universe and [K, N] bool 500 largest on tw['rebal'] days."""
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


def build(tag):
    log('build', tag)
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = E['close'].shape
    log('N', N, 'T', T)
    so, sc = spy_store(tag, days)
    m = np.full(T, np.nan)
    with np.errstate(all='ignore'):
        m[1:] = np.log(sc[1:] / sc[:-1])
    m[~np.isfinite(m) | (np.abs(m) > MAX_LOG)] = np.nan
    spy_on = np.full(T, np.nan)
    spy_in = np.full(T, np.nan)
    with np.errstate(all='ignore'):
        spy_on[1:] = np.log(so[1:] / sc[:-1])
        spy_in[1:] = np.log(sc[1:] / so[1:])
    log('SPY days with a return', int(np.isfinite(m).sum()))
    b0, bl = dimson(E['close'], m)
    log('dimson done; finite b_lag', int(np.isfinite(bl).sum()))
    s = (bl * m[None, :].astype(np.float32)).astype(np.float32)
    s[:, ~np.isfinite(m) | (m == 0)] = np.nan
    np.save(WORK / f's_{tag}.npy', s)
    np.save(WORK / f'b0_{tag}.npy', b0)
    np.save(WORK / f'blag_{tag}.npy', bl)
    del s, b0, bl
    log('store panels')
    S = c2eval.load_store(STORES[tag])
    ids = E['ids']
    o = c2eval.store_panel(S, ids, days, 'open')
    c = c2eval.store_panel(S, ids, days, 'close')
    h = c2eval.store_panel(S, ids, days, 'high')
    lo = c2eval.store_panel(S, ids, days, 'low')
    # store close vs export close (the legs use the store's; the signal uses the export's)
    ce = np.asarray(E['close'], np.float32)
    both = np.isfinite(c) & np.isfinite(ce)
    with np.errstate(all='ignore'):
        mis = both & (np.abs(np.log(c / ce)) > 0.01)
    log('store close vs export close: both finite', int(both.sum()), 'differ > 1%', int(mis.sum()))
    del ce
    on, intra = legs(o, c, h, lo)
    del o, c, h, lo
    np.save(WORK / f'on_{tag}.npy', on)
    np.save(WORK / f'intra_{tag}.npy', intra)
    log('legs: valid stock-days', int(np.isfinite(on).sum()))
    del on, intra
    tw = c2eval.twins(E)
    uni, big = universe(E, tw)
    log('universe sizes (k with any): median', int(np.median(uni.sum(1)[uni.sum(1) > 0])), 'max', int(uni.sum(1).max()))
    np.savez(WORK / f'meta_{tag}.npz', days=days, m=m, spy_on=spy_on, spy_in=spy_in, spy_open=so, spy_close=sc,
             rebal=tw['rebal'], uni=uni, big=big, mismatch=int(mis.sum()), both=int(both.sum()))
    log('saved')


def load(tag, mmap=True):
    z = np.load(WORK / f'meta_{tag}.npz')
    P = {k: z[k] for k in z.files}
    for a in ('s', 'b0', 'blag', 'on', 'intra'):
        P[a] = np.load(WORK / f'{a}_{tag}.npy', mmap_mode='r' if mmap else None)
    P['tag'] = tag
    return P


def kidx(rebal, t):
    return int(np.searchsorted(rebal, t, side='right')) - 1


def avg_rank(x):
    """Average ranks (ties share the mean rank), 0-based."""
    o = np.argsort(x, kind='stable')
    xs = x[o]
    r = np.empty(len(x))
    # tie groups
    starts = np.r_[0, np.nonzero(np.diff(xs) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(xs)]
    vals = (starts + ends - 1) / 2.0
    rr = np.repeat(vals, ends - starts)
    r[o] = rr
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


def nw_t(x, lags=5):
    """Newey-West (Bartlett) t of the mean of a daily series (NaNs dropped)."""
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


if __name__ == '__main__':
    build(sys.argv[1])
