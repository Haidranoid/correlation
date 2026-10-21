"""Pairs trading (experiments/+0.8-pairs-trading.md): Gatev, Goetzmann & Rouwenhorst 2006's distance method.

Shared pieces: the universe, the two price sources, total-return log-price panels, the SSD formation and the vectorized
trading simulation. The rules are in the .md ("Rules" and "Decisions made while building it"). In short:
  - Formation on c2eval.twins' calendar: t0 = export day 252 + 21k; the formation year is closes t0-251..t0.
  - Universe at t0: the judge-like liquid set (night_and_day/common.py) minus c2eval.excluded, with a close in BOTH
    price sources on every day of the formation year (after forward-filling the market-wide gap days).
  - Each stock's total-return path over the formation year, normalized to 1 at t0-251; SSD between every pair; the 20
    smallest. sigma = sd of the pair's normalized difference over the formation year.
  - Trading days t0+1..t0+126, the paths continuing from the formation normalization (GGR's Figure 1): signal at a
    close where |difference| > 2 sigma, open at the next close ($1 long the laggard, $1 short the leader), close at
    the close after the paths cross; anything open closes at t0+126 (or a stock's last close, if its data ends).
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

OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
WORK = c2eval.WORK / 'pairs_trading'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

FORM = 252          # formation days (closes t0-251..t0)
H = 126             # trading days t0+1..t0+126
BLOCK = 21          # a "month": 21 trading days on the fixed calendar
NPORT = H // BLOCK  # 6 overlapping portfolios
NPAIRS = 20
TRIGGER = 2.0       # sigmas
SIDE_BPS = 5.0      # per side per stock trade
NRAND = 1000        # random books per formation
DIV_FROM = '2022-07-01'

# universe (copied from research/night_and_day/common.py)
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def universe(E, tw):
    """[K, N] bool universe and [K, N] bool 500 largest on tw['rebal'] days (night_and_day/common.py's rule)."""
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


def load_universe(tag):
    p = WORK / f'universe_{tag}.npz'
    if p.exists():
        z = np.load(p)
        return z['rebal'], z['uni'], z['big']
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    tw = c2eval.twins(E)
    uni, big = universe(E, tw)
    np.savez(p, rebal=tw['rebal'], uni=uni, big=big)
    return tw['rebal'], uni, big


def gap_days(close, frac=0.5, look=10):
    """Market-wide data gaps (copied from research/twin_reversal/common.py): days where fewer than half as many series
    have a close as the median of the previous 10 days. On the old export: 2002-08-15, 2002-09-16, 2011-02-17."""
    n = np.isfinite(np.asarray(close)).sum(0)
    return [d for d in range(look, len(n)) if n[d] < frac * np.median(n[d - look:d])]


def source_close(tag, source, E):
    """[N, T] float64 closes of one price source aligned to the export's rows and days: 'export' = the judge's export
    (bad-print repaired), 'store' = the store's raw closes. Non-positive closes are NaN."""
    if source == 'export':
        c = np.array(E['close'], np.float64)
    else:
        S = c2eval.load_store(STORES[tag])
        c = c2eval.store_panel(S, E['ids'], E['days'], 'close').astype(np.float64)
    c[~(c > 0)] = np.nan
    return c


def gapfill(c, gd):
    """Copy of c with each market-wide gap day's missing closes set to the previous day's close."""
    c = c.copy()
    for d in gd:
        miss = ~np.isfinite(c[:, d]) & np.isfinite(c[:, d - 1])
        c[miss, d] = c[miss, d - 1]
    return c


def log_tr(c, days, divfix=None):
    """Cumulative total-return log price L [N, T] float64 from closes c (gap days already filled): daily log return on
    forward-filled closes (0 on a missing day; NaN before a stock's first close), plus the dividend correction
    divfix [N, T] (added to daily log returns from DIV_FROM on), cumulated. Returns L and the [N] float of each
    stock's first day."""
    N, T = c.shape
    fin = np.isfinite(c)
    # forward fill
    idx = np.where(fin, np.arange(T)[None, :], -1)
    np.maximum.accumulate(idx, axis=1, out=idx)
    cf = np.take_along_axis(np.where(fin, c, 0.0), np.maximum(idx, 0), axis=1)
    cf[idx < 0] = np.nan
    r = np.zeros((N, T))
    with np.errstate(all='ignore'):
        r[:, 1:] = np.log(cf[:, 1:] / cf[:, :-1])
    r[~np.isfinite(r)] = 0.0
    if divfix is not None:
        d0 = c2eval.day_index(days, DIV_FROM)
        add = np.nan_to_num(np.asarray(divfix[:, d0:], np.float64))
        add[~fin[:, d0:]] = 0.0          # no correction on days without a real close
        r[:, d0:] += add
    L = np.cumsum(r, 1)
    L[idx < 0] = np.nan
    return L


def top_pairs(P, k=NPAIRS):
    """Gram-matrix SSD over all pairs of the rows of P [n, F] (float64): SSD(i, j) = |p_i|^2 + |p_j|^2 - 2 p_i.p_j.
    Returns (i, j, ssd) of the k smallest, i < j, ascending; pairs with SSD <= 1e-12 (identical series) skipped."""
    sq = np.einsum('ij,ij->i', P, P)
    S = sq[:, None] + sq[None, :] - 2.0 * (P @ P.T)
    n = len(P)
    ar = np.arange(n)
    S[ar[:, None] >= ar[None, :]] = np.inf
    S[S <= 1e-12] = np.inf
    flat = np.argpartition(S.ravel(), k)[:k]
    flat = flat[np.argsort(S.ravel()[flat], kind='stable')]
    i, j = np.divmod(flat, n)
    return i, j, S.ravel()[flat]


def brute_ssd(P, k=NPAIRS):
    n = len(P)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            d = P[i] - P[j]
            s = float(d @ d)
            if s > 1e-12:
                out.append((s, i, j))
    out.sort()
    out = out[:k]
    return np.array([o[1] for o in out]), np.array([o[2] for o in out]), np.array([o[0] for o in out])


def simulate(PA, PB, obsA, obsB, sigma, endp, bps=SIDE_BPS):
    """Trade P pairs over one trading period, vectorized across pairs.
    PA, PB [P, H+1]: normalized total-return paths, column 0 = t0 (the formation's last close), columns 1..H the
    trading days (forward-filled). obsA, obsB [P, H+1] bool: a real close that day. sigma [P]. endp [P] int: the last
    column the pair can trade (H, or earlier if a stock's data ends).
    Rules: at a close h where both have a real close, flat, h + 1 < endp and |D| > TRIGGER sigma, open at close h+1
    long the laggard / short the leader ($1 each); while open, at a close h where both have a real close and the paths
    have crossed (D on the other side of 0, or equal), close at close h+1; force-close at endp.
    Returns dict: pnl [P, H+1] gross daily P&L per $1 pair (indexed by the day it accrues), cost [P, H+1],
    n_open [P], n_conv [P] (closed by crossing), n_forced [P], days_open [P]."""
    P_, W = PA.shape
    D = PA - PB
    with np.errstate(all='ignore'):
        rA = np.zeros_like(PA)
        rB = np.zeros_like(PB)
        rA[:, 1:] = PA[:, 1:] / PA[:, :-1] - 1.0
        rB[:, 1:] = PB[:, 1:] / PB[:, :-1] - 1.0
    rA[~np.isfinite(rA)] = 0.0
    rB[~np.isfinite(rB)] = 0.0
    both = obsA & obsB
    thr = TRIGGER * sigma
    c = bps * 1e-4
    pos = np.zeros(P_)            # +1 long A short B, -1 short A long B
    vA = np.zeros(P_)
    vB = np.zeros(P_)
    pend_open = np.zeros(P_)      # direction to open at the next close
    pend_close = np.zeros(P_, bool)
    pnl = np.zeros((P_, W))
    cost = np.zeros((P_, W))
    n_open = np.zeros(P_, np.int32)
    n_conv = np.zeros(P_, np.int32)
    n_forced = np.zeros(P_, np.int32)
    days_open = np.zeros(P_, np.int32)
    for h in range(1, W):
        live = h <= endp
        held = (pos != 0) & live
        # 1. P&L of positions held from close h-1 to close h
        pnl[:, h] = np.where(held, pos * (vA * rA[:, h] - vB * rB[:, h]), 0.0)
        vA = np.where(held, vA * (1.0 + rA[:, h]), vA)
        vB = np.where(held, vB * (1.0 + rB[:, h]), vB)
        days_open += held
        # 2. execute what was signalled at close h-1
        cl = pend_close & live
        cost[:, h] += np.where(cl, c * (vA + vB), 0.0)
        n_conv += cl
        pos = np.where(cl, 0.0, pos)
        pend_close[:] = False
        op = (pend_open != 0) & live
        pos = np.where(op, pend_open, pos)
        vA = np.where(op, 1.0, vA)
        vB = np.where(op, 1.0, vB)
        cost[:, h] += np.where(op, 2.0 * c, 0.0)
        n_open += op
        pend_open[:] = 0.0
        # 3. force-close at the pair's last tradeable close
        fc = (pos != 0) & (h == endp)
        cost[:, h] += np.where(fc, c * (vA + vB), 0.0)
        n_forced += fc
        pos = np.where(fc, 0.0, pos)
        # 4. signals at close h
        sig = both[:, h] & (h < endp)
        Dh = D[:, h]
        crossed = ((pos > 0) & (Dh >= 0)) | ((pos < 0) & (Dh <= 0))
        pend_close = sig & (pos != 0) & crossed
        can_open = sig & (pos == 0) & (h + 1 < endp) & (np.abs(Dh) > thr)
        pend_open = np.where(can_open, -np.sign(Dh), 0.0)   # D > 0: A ahead -> short A (pos -1)
    return {'pnl': pnl, 'cost': cost, 'n_open': n_open, 'n_conv': n_conv, 'n_forced': n_forced,
            'days_open': days_open}


def month_sums(x, H_=H):
    """[P, H+1] daily values (column 0 unused) -> [P, 6] sums over trading days 1-21, 22-42, ..."""
    return x[:, 1:H_ + 1].reshape(x.shape[0], NPORT, BLOCK).sum(2)


def nw_t(x, lags=6):
    """Newey-West (Bartlett) t of the mean (NaNs dropped)."""
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


def plain_t(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float('nan')
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
