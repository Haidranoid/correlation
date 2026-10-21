"""Co-movement market timing (experiments/-3.2-market-timing.md): shared settings and helpers.

The rules are fixed in the .md ("Rules" and "Decisions made while building it"). Everything here is point in time:
the state at calendar day t0 uses closes through t0 only.

    import common as C; E, tw = C.load('old'); st = C.states(E, tw); P = C.spy_close('old', E['days'])
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/market_timing'
WORK.mkdir(parents=True, exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}

H = 21            # holding month and calendar spacing, trading days
WIN = 63          # correlation window and dollar-volume window
N_UNIV = 500
MIN_FRAC = 0.9    # a stock needs a daily return on >= 90% of the 63 days
MED_MAX = 60      # previous month-end values in the median (1,260 days)
MED_MIN = 24      # fewest previous values before the median is used
TOP_EIG = 5       # absorption ratio: eigenvectors counted
N_RANDOM = 10_000
N_BOOT = 10_000
BLOCK = 12        # bootstrap block, months
SEED = 20261021
OLD_LAST = '2025-06-27'


def load(tag):
    """The export (close and dollar volume only) and its twins (c2eval's cache)."""
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    tw = c2eval.twins(E)
    return E, tw


def spy_close(tag, days):
    """SPY's adjusted close from the store, aligned to the export's days, forward-filled after its first close (the one
    gap is 2011-02-17); NaN before its first close (2005-02-25)."""
    S = c2eval.load_store(STORES[tag])
    sid = [s['id'] for s in S['syms'] if s['ticker'] == 'SPY' and s['kind'] == 'etf']
    assert len(sid) == 1, sid
    c = c2eval.store_panel(S, sid, days, 'close')[0].astype(np.float64)
    first = int(np.argmax(np.isfinite(c)))
    for d in range(first + 1, len(c)):
        if not np.isfinite(c[d]):
            c[d] = c[d - 1]
    return c


def states(E, tw, verbose=True):
    """Per twin-calendar day t0 (tw['rebal']): the 500 most-traded eligible stocks' average pairwise correlation and
    absorption ratio over the 63 daily log returns ending at t0, plus the universe size."""
    close, dv = E['close'], E['dollar_volume']
    excl = c2eval.excluded(E, tw)
    rebal = tw['rebal']
    K = len(rebal)
    avg = np.full(K, np.nan)
    ar = np.full(K, np.nan)
    n_sel = np.zeros(K, int)
    n_elig = np.zeros(K, int)
    mdv_cut = np.full(K, np.nan)
    for k, t0 in enumerate(rebal):
        t0 = int(t0)
        assert c2eval.rebal_index(tw, t0) == k
        c = np.asarray(close[:, t0 - WIN:t0 + 1], np.float64)          # 64 closes -> 63 returns, days t0-62..t0
        with np.errstate(all='ignore'):
            r = np.log(c[:, 1:] / c[:, :-1])
            r[~np.isfinite(r)] = np.nan
            mdv = np.nanmedian(np.asarray(dv[:, t0 - WIN + 1:t0 + 1], np.float64), 1)
            sd = np.nanstd(r, 1)
        ok = (np.isfinite(r).mean(1) >= MIN_FRAC) & (np.nan_to_num(mdv) > 0) & ~excl[k] & (np.nan_to_num(sd) > 0)
        n_elig[k] = int(ok.sum())
        order = np.argsort(-np.where(ok, mdv, -np.inf), kind='stable')
        sel = order[:min(N_UNIV, n_elig[k])]
        n = len(sel)
        n_sel[k] = n
        if n < 2:
            continue
        mdv_cut[k] = mdv[sel[-1]]
        R = r[sel]
        m = np.nanmean(R, 1, keepdims=True)
        X = R - m
        X[~np.isfinite(X)] = 0.0                                      # missing days 0 after demeaning
        Z = X / sd[sel][:, None]
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        s = Z.sum(0)
        avg[k] = (s @ s - n) / (n * (n - 1))
        ev = np.linalg.eigvalsh(X.T @ X)                              # 63 x 63 Gram: same nonzero spectrum as the covariance
        ev = np.sort(np.clip(ev, 0, None))[::-1]
        ar[k] = ev[:TOP_EIG].sum() / ev.sum()
        if verbose and k % 40 == 0:
            print(f'  {c2eval.iso(E["days"][t0])} n {n} avg corr {avg[k]:.3f} AR {ar[k]:.3f}', flush=True)
    return {'rebal': np.asarray(rebal), 'avg_corr': avg, 'absorption': ar, 'n_sel': n_sel, 'n_elig': n_elig,
            'mdv_cut': mdv_cut}


def trailing_median(x, k):
    """Median of the previous (not current) values x[k-60..k-1] that are finite, if at least 24; else NaN."""
    prev = x[max(0, k - MED_MAX):k]
    prev = prev[np.isfinite(prev)]
    return float(np.median(prev)) if len(prev) >= MED_MIN else float('nan')


def realized_var(P, t0):
    """Variance (ddof 0) of SPY's 21 daily log returns ending at close t0; NaN unless all 21 exist."""
    r = np.diff(np.log(P[t0 - H:t0 + 1]))
    return float(np.var(r)) if np.isfinite(r).all() else float('nan')
