"""Short volume: the daily signal panels (no returns are used here).

At the close of each day d, for every stock:
  SVR(d)   = ShortVolume / TotalVolume from the facility sum; NaN when TotalVolume < 10,000 or no record
             (and, if common.VC_GUARD is set, when the volume-match correlation at d is below it).
  A(d)     = mean SVR over d-4..d (>= 3 of 5 days) - mean SVR over d-256..d-5 (>= 150 days): own abnormal SVR.
  G(d)     = A(d) - mean A(d) of the other universe stocks sharing its twin at rebal_index(d) (>= 3 others with a
             finite A); only for universe stocks (judge-like liquid set with a twin, minus excluded).
  A_lag, G_lag: the same one day earlier (windows d-5..d-1 and d-257..d-6), for the posting-time check: FINRA posts
             day d's file by 6pm ET, after the close of d.

    python3 -I -B signal_build.py old|new
Writes WORK/short_volume/<tag>/{A,G,A_lag,G_lag}.npy ([N, T] float32).
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np, c2eval = C.np, C.c2eval


def group_relative(A, uni, rebal, fund_row):
    N, T = A.shape
    G = np.full((N, T), np.nan, np.float32)
    for d in range(int(rebal[0]), T):
        k = C.kidx(rebal, d)
        u = np.nonzero(uni[k])[0]
        if len(u) == 0:
            continue
        a = A[u, d].astype(np.float64)
        ok = np.isfinite(a)
        u, a = u[ok], a[ok]
        if len(u) == 0:
            continue
        f = fund_row[k, u]
        nf = int(f.max()) + 1
        s = np.bincount(f, weights=a, minlength=nf)
        n = np.bincount(f, minlength=nf)
        others = n[f] - 1
        with np.errstate(all='ignore'):
            g = a - (s[f] - a) / others
        g[others < C.MIN_PEERS] = np.nan
        G[u, d] = g
    return G


def main(tag):
    wd = C.wdir(tag)
    short = np.load(wd / 'short.npy')
    total = np.load(wd / 'total.npy')
    with np.errstate(all='ignore'):
        svr = (short.astype(np.float64) / total).astype(np.float32)
    svr[~(total >= C.MIN_TOTAL)] = np.nan
    del short
    if C.VC_GUARD is not None:
        vc = np.load(wd / 'vcorr.npy')
        drop = np.isfinite(svr) & ~(vc >= C.VC_GUARD)
        C.log('volume-match guard drops', int(drop.sum()), 'of', int(np.isfinite(svr).sum()), 'SVR stock-days')
        svr[drop] = np.nan
        del vc
    del total
    np.save(wd / 'svr.npy', svr)
    recent = C.rolling_mean(svr, C.RECENT - 1, 0, C.MIN_RECENT)
    base = C.rolling_mean(svr, C.BASE_LO, C.BASE_HI, C.MIN_BASE)
    A = (recent - base).astype(np.float32)
    del recent, base, svr
    A_lag = np.full_like(A, np.nan)
    A_lag[:, 1:] = A[:, :-1]
    z = np.load(wd / 'universe.npz')
    uni, rebal, fund_row = z['uni'], z['rebal'], z['fund_row']
    E = c2eval.load_export(C.EXPORTS[tag], arrays=())
    G = group_relative(A, uni, rebal, fund_row)
    G_lag = group_relative(A_lag, uni, rebal, fund_row)
    for name, X in (('A', A), ('G', G), ('A_lag', A_lag), ('G_lag', G_lag)):
        np.save(wd / f'{name}.npy', X)
    days = E['days']
    iso = [c2eval.iso(d) for d in days]
    for y in sorted(set(i[:4] for i in iso if i >= '2009')):
        m = np.array([i[:4] == y for i in iso])
        nu = sum(int(uni[C.kidx(rebal, d)].sum()) for d in np.nonzero(m)[0] if C.kidx(rebal, d) >= 0)
        C.log(y, 'universe stock-days', nu, 'finite A in universe',
              sum(int((np.isfinite(A[:, d]) & uni[C.kidx(rebal, d)]).sum()) for d in np.nonzero(m)[0] if C.kidx(rebal, d) >= 0),
              'finite G', int(np.isfinite(G[:, m]).sum()))


if __name__ == '__main__':
    main(sys.argv[1])
