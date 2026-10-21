"""Component ranks and the composite, every day.

    python3 -I composite.py old|new

At each day d, among the universe stocks U(d) (universe.py):
  - rev5, mom12_1, twin5, lgbm: ranked among the universe stocks where the value is finite, to [-1, +1] (ties averaged);
  - lowcorr: universe stocks with finite vol and corr split into volatility quintiles (floor(5 x ordinal rank / n), as
    research/betting_against_correlation/money.py), then minus corr ranked to [-1, +1] within each quintile;
  - composite: the plain mean of the stock's finite component ranks, if it has >= MIN_COMP of them (else NaN).
The 500-largest version: the 500 universe stocks with the highest 252-day median dollar volume, every rank and the
composite recomputed within them.

Writes:
  WORK/<tag>/composite.npy  [N, T] float32 daily composite (the judge's scores file, via c2eval.save_scores)
  WORK/<tag>/weekly.npz     on the weekly calendar (export indices 252 + 5k): W [n_w] day indices, R [n_w, C, N]
                            component ranks, comp [n_w, N], comp4 [n_w, N] (without LightGBM), R500, comp500,
                            comp4_500 (the 500 largest), nu [n_w] universe size
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import (c2eval, np, EXPORTS, wdir, log, rank_pm1, ordinal_rank, LGBM, COMPONENTS, MIN_COMP, N_LARGE,
                    WEEK0, WEEK)


def ranks_at(u, d, X):
    """[C, len(u)] component ranks for universe rows u at day d (NaN where the component is missing)."""
    out = np.full((len(COMPONENTS), len(u)), np.nan)
    for ci, c in enumerate(COMPONENTS):
        if c == 'lowcorr':
            v, r = X['vol'][d, u], X['corr'][d, u]
            m = np.nonzero(np.isfinite(v) & np.isfinite(r))[0]
            if len(m) < 5:
                continue
            q = np.floor(5 * ordinal_rank(v[m]) / len(m)).astype(int)
            for qq in range(5):
                mi = m[q == qq]
                if len(mi):
                    out[ci, mi] = rank_pm1(-r[mi].astype(np.float64))
        else:
            if c not in X:
                continue
            x = X[c][d, u]
            m = np.nonzero(np.isfinite(x))[0]
            if len(m):
                out[ci, m] = rank_pm1(x[m].astype(np.float64))
    return out


def combine(R, cols):
    """Mean of the finite ranks over components `cols`; NaN with fewer than MIN_COMP."""
    sub = R[cols]
    k = np.isfinite(sub).sum(0)
    with np.errstate(all='ignore'):
        m = np.nanmean(sub, 0) if sub.size else np.full(R.shape[1], np.nan)
    m[k < MIN_COMP] = np.nan
    return m


def main(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=())
    days = E['days']
    N, T = len(E['ids']), len(days)
    U = np.load(wdir(tag) / 'universe.npy')
    MD = np.load(wdir(tag) / 'meddv.npy', mmap_mode='r')
    X = {}
    for nm in ('rev5', 'mom12_1', 'twin5', 'vol', 'corr'):
        X[nm] = np.ascontiguousarray(np.load(wdir(tag) / f'{nm}.npy').T)
        log('loaded', nm)
    if tag == 'old':
        X['lgbm'] = np.ascontiguousarray(np.load(LGBM).T)
        log('loaded lgbm')
    C = len(COMPONENTS)
    all_c = list(range(C))
    no_lgbm = [i for i, c in enumerate(COMPONENTS) if c != 'lgbm']
    comp = np.full((T, N), np.nan, np.float32)
    W = np.arange(WEEK0, T, WEEK)
    wpos = {int(d): i for i, d in enumerate(W)}
    nW = len(W)
    R_w = np.full((nW, C, N), np.nan, np.float32)
    comp_w = np.full((nW, N), np.nan, np.float32)
    comp4_w = np.full((nW, N), np.nan, np.float32)
    R5_w = np.full((nW, C, N), np.nan, np.float32)
    comp5_w = np.full((nW, N), np.nan, np.float32)
    comp45_w = np.full((nW, N), np.nan, np.float32)
    nu = np.zeros(nW, int)
    first = int(np.argmax(U.any(1)))
    for d in range(first, T):
        u = np.nonzero(U[d])[0]
        if len(u) == 0:
            continue
        R = ranks_at(u, d, X)
        comp[d, u] = combine(R, all_c)
        if d in wpos:
            i = wpos[d]
            nu[i] = len(u)
            R_w[i][:, u] = R
            comp_w[i, u] = comp[d, u]
            comp4_w[i, u] = combine(R, no_lgbm)
            md = np.asarray(MD[d, u], np.float64)
            big = u[np.argsort(-np.nan_to_num(md, nan=-1.0), kind='mergesort')[:N_LARGE]]
            big = np.sort(big)
            R5 = ranks_at(big, d, X)
            R5_w[i][:, big] = R5
            comp5_w[i, big] = combine(R5, all_c)
            comp45_w[i, big] = combine(R5, no_lgbm)
        if d % 500 == 0:
            log(c2eval.iso(days[d]), 'universe', len(u), 'composite', int(np.isfinite(comp[d]).sum()))
    p = c2eval.save_scores(comp.T, f'signal_ensemble/{tag}/composite', export=EXPORTS[tag])
    log('saved', p)
    np.savez(wdir(tag) / 'weekly.npz', W=W, R=R_w, comp=comp_w, comp4=comp4_w, R500=R5_w, comp500=comp5_w,
             comp4_500=comp45_w, nu=nu)
    log('saved weekly.npz')


if __name__ == '__main__':
    main(_sys.argv[1])
