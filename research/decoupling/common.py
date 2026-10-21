"""Decoupling (experiments/-10.2-decoupling.md): shared settings, the universe and the break-measure panels.

At the close of day t, with the twin fund f and beta b of the latest twin rebalance k = rebal_index(t) (and only where
the twin at k is the twin at k-3, i.e. the same fund at t-63 and t, since the calendar steps by 21 = 63 / 3):
  - e(d) = r_i(d) - b * r_f(d), daily log returns built as c2eval.twin_residual builds them (export closes for the
    stock; store closes for the fund with |log move| > 0.5 treated as missing);
  - rho_now  = Pearson corr(r_i, r_f) over t-62..t, leaving out the 3 days with the largest |e| (>= 50 days left);
  - rho_before = Pearson corr(r_i, r_f) over t-314..t-63, all days (>= 200 days with both returns);
  - S3 = sum of e over t-62..t without those 3 days; Sall = with them;
  - fwd = sum of e over t+1..t+63 with the same f and b (NaN if > 3 missing daily returns of the stock or the fund,
    or if the window runs past the export).
Every quantity of a stock-day uses the same fund and beta.

    python3 -I -B build.py old|new      # panels, universe and the event table under WORK
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
WORK = c2eval.WORK / 'decoupling'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}

# break measure
NOW_W = 63          # t-62..t
N_BIG = 3           # days with the largest |e| left out of rho_now and S3
MIN_NOW = 50        # non-missing days left in the now window
BEF_LAG = 63        # the before window ends at t-63
BEF_W = 252         # t-314..t-63
MIN_BEF = 200
FWD = 63            # outcome t+1..t+63
FWD_MAX_MISSING = 3 # max(1, round(0.05 * 63)), c2eval.twin_residual's default
TWIN_LAG_K = 3      # 63 days = 3 rebalances
# events
RHO_BEFORE_MIN = 0.5
DROP_MIN = 0.30
SUSTAIN = 21
QUIET = 126
NONBREAK_MAX = 0.10
N_MATCH = 20
SEED = 20261021
# universe (copied from research/night_and_day/common.py)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def log_returns(close, max_abs=None):
    """c2eval._log_returns (copied): daily log returns, NaN where missing; |r| > max_abs -> NaN."""
    c = np.asarray(close, np.float64)
    r = np.full(c.shape, np.nan)
    with np.errstate(all='ignore'):
        r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    r[~np.isfinite(r)] = np.nan
    if max_abs is not None:
        r[np.abs(r) > max_abs] = np.nan
    return r


def universe(E, tw):
    """[K, N] bool universe (judge-like liquid set, point in time, minus c2eval.excluded, with a twin at k) and
    [K, N] bool 500 largest of it. Copied from research/night_and_day/common.py, plus the twin requirement."""
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
        u = (fin & (np.nan_to_num(mc) >= MIN_PRICE) & (np.nan_to_num(mv) >= MIN_DV) & ~excl[k]
             & (tw['fund_row'][k] >= 0))
        uni[k] = u
        cand = np.nonzero(u)[0]
        o = cand[np.argsort(-np.nan_to_num(mv63[cand], nan=-1.0), kind='stable')]
        big[k, o[:N_LARGE]] = True
    return uni, big


def pearson_from_sums(n, sx, sy, sxx, syy, sxy):
    with np.errstate(all='ignore'):
        num = n * sxy - sx * sy
        den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
        return num / den


def block_measures(sr, fr, tw, k, T):
    """Break measures for the days of rebalance block k, for stocks with a twin at k equal to the twin at k-3.
    Returns rows, days, and dict of [n, nd] float64 arrays: rho_b, rho_n, S3, Sall, fwd (NaN where invalid)."""
    rebal = tw['rebal']
    t0 = int(rebal[k])
    t1 = int(rebal[k + 1]) if k + 1 < len(rebal) else T
    nd = t1 - t0
    if k < TWIN_LAG_K or t0 - (BEF_LAG + BEF_W - 1) < 0:
        return None
    fk = tw['fund_row'][k]
    rows = np.nonzero((fk >= 0) & (tw['fund_row'][k - TWIN_LAG_K] == fk))[0]
    if len(rows) == 0:
        return None
    f = fk[rows]
    b = tw['beta'][k, rows].astype(np.float64)
    out = {}
    # ---- now window t-62..t, for t = t0..t1-1
    A = sr[rows, t0 - NOW_W + 1:t1]                       # [n, 62 + nd]
    B = fr[f, t0 - NOW_W + 1:t1]
    WA = sliding_window_view(A, NOW_W, axis=1)            # [n, nd, 63]
    WB = sliding_window_view(B, NOW_W, axis=1)
    e = WA - b[:, None, None] * WB                        # NaN where either is missing
    fin = np.isfinite(e)
    ae = np.where(fin, np.abs(e), -np.inf)
    top = np.argpartition(-ae, N_BIG - 1, axis=2)[..., :N_BIG]
    big = np.zeros(e.shape, bool)
    np.put_along_axis(big, top, True, axis=2)
    keep = fin & ~big
    n = keep.sum(2).astype(np.float64)
    x = np.where(keep, WA, 0.0)
    y = np.where(keep, WB, 0.0)
    rho_n = pearson_from_sums(n, x.sum(2), y.sum(2), (x * x).sum(2), (y * y).sum(2), (x * y).sum(2))
    S3 = np.where(keep, e, 0.0).sum(2)
    Sall = np.where(fin, e, 0.0).sum(2)
    ok = n >= MIN_NOW
    out['rho_n'] = np.where(ok, rho_n, np.nan)
    out['S3'] = np.where(ok, S3, np.nan)
    out['Sall'] = np.where(ok, Sall, np.nan)
    del WA, WB, e, fin, ae, top, big, keep, x, y
    # ---- before window t-314..t-63
    a0 = t0 - BEF_LAG - BEF_W + 1
    A2 = sr[rows, a0:t1 - BEF_LAG]                        # [n, 251 + nd]
    B2 = fr[f, a0:t1 - BEF_LAG]
    m = np.isfinite(A2) & np.isfinite(B2)
    x = np.where(m, A2, 0.0)
    y = np.where(m, B2, 0.0)

    def roll(X):
        cs = np.zeros((X.shape[0], X.shape[1] + 1))
        np.cumsum(X, 1, out=cs[:, 1:])
        return cs[:, BEF_W:] - cs[:, :-BEF_W]             # [n, nd]

    nb = roll(m.astype(np.float64))
    rho_b = pearson_from_sums(nb, roll(x), roll(y), roll(x * x), roll(y * y), roll(x * y))
    out['rho_b'] = np.where(nb >= MIN_BEF, rho_b, np.nan)
    # ---- forward window t+1..t+63, same fund and beta
    hi = min(T, t1 + FWD)
    A3 = sr[rows, t0 + 1:hi]
    B3 = fr[f, t0 + 1:hi]

    def cums(X):
        cs = np.zeros((X.shape[0], X.shape[1] + 1))
        np.cumsum(X, 1, out=cs[:, 1:])
        return cs

    ca, cma = cums(np.nan_to_num(A3)), cums(~np.isfinite(A3))
    cb, cmb = cums(np.nan_to_num(B3)), cums(~np.isfinite(B3))
    fwd = np.full((len(rows), nd), np.nan)
    for i in range(nd):                                    # t = t0 + i; window columns i .. i+62 of A3
        if t0 + i + FWD > T - 1:
            continue
        sa = ca[:, i + FWD] - ca[:, i]
        sb = cb[:, i + FWD] - cb[:, i]
        ma = cma[:, i + FWD] - cma[:, i]
        mb = cmb[:, i + FWD] - cmb[:, i]
        v = sa - b * sb
        v[(ma > FWD_MAX_MISSING) | (mb > FWD_MAX_MISSING)] = np.nan
        fwd[:, i] = v
    out['fwd'] = fwd
    return rows, np.arange(t0, t1), out


PANELS = ('rho_b', 'rho_n', 'S3', 'Sall', 'fwd')


def build_panels(E, tw):
    N, T = E['close'].shape
    sr = log_returns(E['close'])
    fr = log_returns(tw['F']['close'], max_abs=0.5)
    P = {p: np.full((N, T), np.nan, np.float32) for p in PANELS}
    for k in range(len(tw['rebal'])):
        r = block_measures(sr, fr, tw, k, T)
        if r is None:
            continue
        rows, days, out = r
        for p in PANELS:
            P[p][rows, days[0]:days[-1] + 1] = out[p]
    return P


def kidx(rebal, t):
    return int(np.searchsorted(rebal, t, side='right')) - 1


def cr1_mean(x, groups):
    """Mean of x and its t with standard errors clustered by group (Liang-Zeger CR1 for a mean). NaNs dropped.
    Returns mean, t, n, number of clusters, iid t."""
    x = np.asarray(x, np.float64)
    g = np.asarray(groups)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    n = len(x)
    if n < 3:
        return float('nan'), float('nan'), n, 0, float('nan')
    mu = x.mean()
    u = x - mu
    _, inv = np.unique(g, return_inverse=True)
    G = int(inv.max()) + 1
    Ug = np.bincount(inv, weights=u, minlength=G)
    var = (G / (G - 1)) * (Ug @ Ug) / n ** 2 if G > 1 else float('nan')
    t = float(mu / np.sqrt(var)) if var > 0 else float('nan')
    t_iid = float(mu / (x.std(ddof=1) / np.sqrt(n)))
    return float(mu), t, n, G, t_iid


def month_of(days_since_1970):
    """Calendar month id (YYYY*12 + MM-1) of export days."""
    d = np.asarray(days_since_1970).astype('datetime64[D]')
    return d.astype('datetime64[M]').astype(np.int64)


def year_of(days_since_1970):
    d = np.asarray(days_since_1970).astype('datetime64[D]')
    return d.astype('datetime64[Y]').astype(np.int64) + 1970
