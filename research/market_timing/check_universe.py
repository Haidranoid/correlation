"""Signal-only check (no returns after t0): before 2005-02-25 the export holds only ~320 stocks, so the median's early
history is on them. For the months 2005-06..2010-06, compare the average correlation and absorption ratio of the
500-stock universe with the same measures on the subset of those 500 that already had a close before 2005-02-25."""
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_timing')
import numpy as np  # noqa: E402

import common as C  # noqa: E402

E, tw = C.load('old')
early = np.isfinite(np.asarray(E['close'][:, 756]))      # had a close on 2004-12-31
close, dv = E['close'], E['dollar_volume']
excl = C.c2eval.excluded(E, tw)


def measures(R, sd):
    n = R.shape[0]
    X = R - np.nanmean(R, 1, keepdims=True)
    X[~np.isfinite(X)] = 0.0
    Z = X / sd[:, None]
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    s = Z.sum(0)
    ev = np.sort(np.clip(np.linalg.eigvalsh(X.T @ X), 0, None))[::-1]
    return (s @ s - n) / (n * (n - 1)), ev[:C.TOP_EIG].sum() / ev.sum()


rows = []
for k in range(29, 90):
    t0 = int(tw['rebal'][k])
    c = np.asarray(close[:, t0 - C.WIN:t0 + 1], np.float64)
    with np.errstate(all='ignore'):
        r = np.log(c[:, 1:] / c[:, :-1])
        r[~np.isfinite(r)] = np.nan
        mdv = np.nanmedian(np.asarray(dv[:, t0 - C.WIN + 1:t0 + 1], np.float64), 1)
        sd = np.nanstd(r, 1)
    ok = (np.isfinite(r).mean(1) >= C.MIN_FRAC) & (np.nan_to_num(mdv) > 0) & ~excl[k] & (np.nan_to_num(sd) > 0)
    sel = np.argsort(-np.where(ok, mdv, -np.inf), kind='stable')[:C.N_UNIV]
    sub = sel[early[sel]]
    a_all, ar_all = measures(r[sel], sd[sel])
    a_sub, ar_sub = measures(r[sub], sd[sub])
    rows.append((C.c2eval.iso(E['days'][t0]), len(sub), a_all, a_sub, ar_all, ar_sub))
    print(f'{rows[-1][0]}  subset {len(sub)}  avg corr 500 {a_all:.3f} subset {a_sub:.3f} diff {a_sub - a_all:+.3f}   '
          f'AR 500 {ar_all:.3f} subset {ar_sub:.3f} diff {ar_sub - ar_all:+.3f}')
d = np.array([r[3] - r[2] for r in rows])
e = np.array([r[5] - r[4] for r in rows])
print(f'\navg corr, subset minus 500: mean {d.mean():+.4f}, range {d.min():+.4f}..{d.max():+.4f}, n {len(d)}')
print(f'absorption, subset minus 500: mean {e.mean():+.4f}, range {e.min():+.4f}..{e.max():+.4f}')
print('subset size: min', min(r[1] for r in rows), 'max', max(r[1] for r in rows))
