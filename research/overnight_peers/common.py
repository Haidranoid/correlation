"""Overnight peers (experiments/-1.6-overnight-peers.md): shared settings, returns, peers, signal and book.

The rules are in the .md ("Rules" and "Decisions made while building it"). In short:
  - Returns, from the store's opens and closes (research/night_and_day's legs, read-only, with its bad-print rule):
    overnight log(O_d / C_{d-1}), intraday log(C_d / O_d), whole day = their sum = log(C_d / C_{d-1}). Each is
    market-adjusted by subtracting that day's cross-sectional mean over the universe of the latest calendar day at or
    before d.
  - Universe: the judge-like liquid set minus c2eval.excluded on the twins calendar (export days 252 + 21k), as
    built by research/night_and_day (meta_<tag>.npz, read-only).
  - Peers at each calendar day t (from 2009-01): over d = t-251..t, each universe stock's 20 most-correlated other
    universe stocks (pairwise-complete Pearson on market-adjusted returns of one kind), with >= 200 common days and
    r >= 0.10; a stock needs >= 5 such peers.
  - Signal at day d (calendar day t = latest at or before d): the peers' mean market-adjusted whole-day log return
    over d-20..d, then the residual of a cross-sectional OLS (with intercept) on the stock's own market-adjusted
    whole-day log return over d-20..d.
  - Book at each calendar day t: long the top tenth, short the bottom tenth of the residual, equal weight, held close
    t -> close t+21 in market-relative simple returns from the export's closes; 5 bps per side on turnover.
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
WORK = c2eval.WORK / 'overnight_peers'
WORK.mkdir(parents=True, exist_ok=True)
ND = c2eval.WORK / 'night_and_day'          # read-only: legs and universe built by research/night_and_day
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}

WIN = 252            # peer window: d = t-251..t
MIN_COMMON = 200     # common days a peer needs
MIN_CORR = 0.10      # correlation floor for a peer
NPEER = 20           # peers kept per stock
MIN_PEERS = 5        # a stock needs at least this many peers
H = 21               # signal window (d-20..d) and holding period (t -> t+21)
MAX_MISS = 1         # a 21-day sum may miss at most 1 daily return (c2eval's max(1, round(0.05 * h)))
COST = 5e-4          # 5 bps per side
DECILE = 0.10
MIN_NAMES = 50       # a day needs at least this many stocks with a signal for the OLS
START = '2009-01-01'
TYPES = ('on', 'intra', 'whole')
TYPE_NAME = {'on': 'overnight', 'intra': 'intraday', 'whole': 'whole-day'}

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def kidx(rebal, t):
    return int(np.searchsorted(rebal, t, side='right')) - 1


def meta(tag):
    z = np.load(ND / f'meta_{tag}.npz')
    return {'days': z['days'], 'rebal': z['rebal'], 'uni': z['uni'], 'tag': tag}


def k_range(tag, M):
    """Calendar indices whose peers this run builds: from the first calendar day in 2009 (old), or from the latest
    calendar day at or before the last old day 2025-06-27 (new: the unseen months need it)."""
    days, rebal = M['days'], M['rebal']
    if tag == 'old':
        k0 = int(np.searchsorted(rebal, c2eval.day_index(days, START)))
    else:
        k0 = kidx(rebal, c2eval.day_index(days, '2025-06-30') - 1)
    return list(range(k0, len(rebal)))


def raw_returns(tag):
    """Overnight, intraday and whole-day log returns [N, T] float32 (NaN = no return under the bad-print rule)."""
    on = np.load(ND / f'on_{tag}.npy')
    intra = np.load(ND / f'intra_{tag}.npy')
    assert np.array_equal(np.isfinite(on), np.isfinite(intra))
    whole = on + intra
    return {'on': on, 'intra': intra, 'whole': whole}


def market_adjust(X, rebal, uni):
    """X minus each day's cross-sectional mean over the universe of the latest calendar day at or before that day
    (NaN before the first non-empty universe). Returns the adjusted panel (float32) and the daily mean."""
    N, T = X.shape
    mkt = np.full(T, np.nan)
    for k, t0 in enumerate(rebal):
        t1 = rebal[k + 1] if k + 1 < len(rebal) else T
        rows = np.nonzero(uni[k])[0]
        if len(rows) == 0:
            continue
        blk = X[rows, t0:t1].astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mkt[t0:t1] = np.nanmean(blk, 0)
    return (X - mkt[None, :].astype(np.float32)).astype(np.float32), mkt


def roll_sum(X, h=H, max_miss=MAX_MISS):
    """[N, T] float32: sum of X over the h days ending at d; NaN if more than max_miss values are missing."""
    N, T = X.shape
    f = np.isfinite(X)
    cs = np.zeros((N, T + 1))
    np.cumsum(np.where(f, X, 0.0), 1, out=cs[:, 1:])
    cn = np.zeros((N, T + 1), np.int32)
    np.cumsum(f, 1, out=cn[:, 1:])
    out = np.full((N, T), np.nan, np.float32)
    s = cs[:, h:] - cs[:, :-h]
    c = cn[:, h:] - cn[:, :-h]
    s[c < h - max_miss] = np.nan
    out[:, h - 1:] = s
    return out


def corr_matrix(Xw):
    """Pairwise-complete Pearson correlation [n, n] (float64) and common-day counts [n, n] of the rows of Xw [n, W]
    (NaN = missing), from float32 matrix products on returns standardized over each row's own finite days."""
    X = np.asarray(Xw, np.float64)
    M = np.isfinite(X)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mu = np.nanmean(X, 1)
        sd = np.nanstd(X, 1)
        Z = (X - mu[:, None]) / sd[:, None]
    Z[~M | ~np.isfinite(Z)] = 0.0
    Mf = M.astype(np.float32)
    Zf = Z.astype(np.float32)
    n = (Mf @ Mf.T).astype(np.float64)
    Sx = (Zf @ Mf.T).astype(np.float64)            # Sx[i, j] = sum of z_i over days both i and j have
    Sxx = ((Zf * Zf) @ Mf.T).astype(np.float64)
    Sxy = (Zf @ Zf.T).astype(np.float64)
    del Zf, Mf
    with np.errstate(all='ignore'):
        vi = n * Sxx - Sx * Sx
        num = n * Sxy - Sx * Sx.T
        C = num / np.sqrt(vi * vi.T)
    return C, n


def peers_at(Xadj, rows, t):
    """Peers of each universe row at calendar day t from one market-adjusted return panel. Returns peer [n, 20]
    (export row indices, -1 = none, best first) and corr [n, 20] (NaN = none); rows with fewer than MIN_PEERS
    qualifying peers get none."""
    C, n = corr_matrix(Xadj[rows, t - WIN + 1:t + 1])
    np.fill_diagonal(C, -np.inf)
    C[(n < MIN_COMMON) | ~(C >= MIN_CORR)] = -np.inf
    top = np.argpartition(-C, NPEER - 1, axis=1)[:, :NPEER]
    vals = np.take_along_axis(C, top, 1)
    o = np.argsort(-vals, 1, kind='stable')
    top = np.take_along_axis(top, o, 1)
    vals = np.take_along_axis(vals, o, 1)
    valid = np.isfinite(vals)
    ok = valid.sum(1) >= MIN_PEERS
    keep = valid & ok[:, None]
    peer = np.where(keep, rows[top], -1).astype(np.int32)
    corr = np.where(keep, vals, np.nan).astype(np.float32)
    return peer, corr


def save_peers(tag, ks, P):
    """P[type][k] = (rows, peer, corr). Saved compactly per type under WORK."""
    for ty in TYPES:
        rows = np.concatenate([P[ty][k][0] for k in ks])
        peer = np.concatenate([P[ty][k][1] for k in ks])
        corr = np.concatenate([P[ty][k][2] for k in ks])
        cnt = np.array([len(P[ty][k][0]) for k in ks])
        np.savez(WORK / f'peers_{ty}_{tag}.npz', ks=np.array(ks), cnt=cnt, rows=rows, peer=peer, corr=corr)


def load_peers(tag):
    """dict type -> dict k -> (rows, peer, corr)."""
    out = {}
    for ty in TYPES:
        z = np.load(WORK / f'peers_{ty}_{tag}.npz')
        off = np.r_[0, np.cumsum(z['cnt'])]
        rows, peer, corr = z['rows'], z['peer'], z['corr']
        out[ty] = {int(k): (rows[off[i]:off[i + 1]], peer[off[i]:off[i + 1]], corr[off[i]:off[i + 1]])
                   for i, k in enumerate(z['ks'])}
    return out


def avg_rank(x):
    """Average ranks (ties share the mean rank), 0-based."""
    o = np.argsort(x, kind='stable')
    xs = x[o]
    r = np.empty(len(x))
    starts = np.r_[0, np.nonzero(np.diff(xs) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(xs)]
    r[o] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    return r


def residual_block(S21, rows, peer, D):
    """Residual signal for the stocks `rows` (with their peers) on days D (array of export day indices).
    Returns resid [n, len(D)] float64 (NaN = no signal), plus the raw peer mean and own return."""
    SD = S21[:, D]                                   # [N, nd]
    own = SD[rows].astype(np.float64)                # [n, nd]
    pv = SD[np.maximum(peer, 0)].astype(np.float64)  # [n, 20, nd]
    pv[peer < 0] = np.nan
    cnt = np.isfinite(pv).sum(1)
    with np.errstate(all='ignore'):
        pm = np.nansum(pv, 1) / cnt
    pm[cnt < MIN_PEERS] = np.nan
    res = np.full(pm.shape, np.nan)
    for j in range(len(D)):
        ok = np.isfinite(pm[:, j]) & np.isfinite(own[:, j])
        if ok.sum() < MIN_NAMES:
            continue
        x, y = own[ok, j], pm[ok, j]
        xm, ym = x.mean(), y.mean()
        b = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
        res[ok, j] = y - ym - b * (x - xm)
    return res, pm, own


def tstat(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float('nan'), float('nan'), len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)
