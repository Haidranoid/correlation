"""The daily universe: the judge-like liquid set minus c2eval.excluded, point in time.

    python3 -I universe.py old|new

Stock j is in the universe at day d when (the judge's liquid_universe rule, stockai/judge/src/holdout.rs, with its
default --peer-window 756, made point in time with windows ending at d):
  - the census variable v is finite on >= 90% of the 756 days d-755..d;
  - the upper median of the finite closes and of the finite dollar volumes over d-251..d are >= $5 and >= $5M;
  - close[d] is finite and > 0;
  - not c2eval.excluded(E, tw)[rebal_index(tw, d)] (the fund-like flag and the non-common-stock list).
Writes WORK/<tag>/universe.npy [T, N] bool and meddv.npy [T, N] float32 (the 252-day upper median dollar volume,
NaN outside the universe; used to pick the 500 largest).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import c2eval, np, EXPORTS, wdir, log, DATA_W, DATA_FRAC, MED_W, MIN_PRICE, MIN_DV


def upper_median(X):
    """Row-wise upper median of the finite values ((k // 2)-th of k sorted, as the judge takes it); 0 if none."""
    S = np.sort(X, axis=1)            # NaN sort last
    k = np.isfinite(X).sum(1)
    out = S[np.arange(len(S)), np.minimum(k // 2, X.shape[1] - 1)]
    out[k == 0] = 0.0
    return out


def main(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume', 'v'))
    tw = c2eval.twins(E)
    exc = c2eval.excluded(E, tw)
    N, T = E['close'].shape
    close = np.asarray(E['close'], np.float32)
    dv = np.asarray(E['dollar_volume'], np.float32)
    vfin = np.zeros((N, T + 1), np.int32)
    np.cumsum(np.isfinite(np.asarray(E['v'])), 1, out=vfin[:, 1:])
    U = np.zeros((T, N), bool)
    MD = np.full((T, N), np.nan, np.float32)
    for d in range(DATA_W - 1, T):
        cnt = vfin[:, d + 1] - vfin[:, d + 1 - DATA_W]
        cand = np.nonzero((cnt * 10 >= DATA_W * 9) & np.isfinite(close[:, d]) & (np.nan_to_num(close[:, d]) > 0))[0]
        if len(cand) == 0:
            continue
        k = c2eval.rebal_index(tw, d)
        if k >= 0:
            cand = cand[~exc[k, cand]]
        mp = upper_median(close[cand, d - MED_W + 1:d + 1])
        md = upper_median(dv[cand, d - MED_W + 1:d + 1])
        ok = (mp >= MIN_PRICE) & (md >= MIN_DV)
        U[d, cand[ok]] = True
        MD[d, cand[ok]] = md[ok]
        if d % 500 == 0:
            log(c2eval.iso(E['days'][d]), 'universe', int(ok.sum()))
    np.save(wdir(tag) / 'universe.npy', U)
    np.save(wdir(tag) / 'meddv.npy', MD)
    n = U.sum(1)
    first = int(np.argmax(n > 0))
    log(f'{tag}: universe from {c2eval.iso(E["days"][first])}; size median {int(np.median(n[first:]))}, '
        f'min {int(n[first:].min())}, max {int(n.max())}')
    for y in range(2005, 2027):
        i = c2eval.day_index(E['days'], f'{y}-01-02')
        if i < T:
            log(f'  {c2eval.iso(E["days"][i])}: {int(n[i])}')


if __name__ == '__main__':
    main(_sys.argv[1])
