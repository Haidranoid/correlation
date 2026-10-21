"""Overnight peers: returns check, peers at each calendar day for the three kinds of return, peer diagnostics
(overlap, mean correlation, example lists), and the daily residual signals for the three peer types.

    python3 -I build.py old      # calendar days 2009-01-02 .. 2025-06-13 on the old export
    python3 -I build.py new      # calendar days 2025-06-13 .. 2026-10-15 on the new export (the unseen months)

Writes peers_<type>_<tag>.npz and resid_<type>_<tag>.npy under WORK, and out/build_<tag>.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval
iso = c2eval.iso

tag = sys.argv[1]
EXAMPLES = ('AAPL', 'MSFT', 'JPM', 'XOM', 'KO', 'NVDA')
EX_DATE = '2019-12-31'      # examples at the latest calendar day on or before this date (old export)


def jaccard(A, B):
    ok = (A[:, 0] >= 0) & (B[:, 0] >= 0)
    a, b = A[ok], B[ok]
    eq = (a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)
    inter = eq.any(2).sum(1)
    na, nb = (a >= 0).sum(1), (b >= 0).sum(1)
    return inter / (na + nb - inter)


def main():
    M = C.meta(tag)
    days, rebal, uni = M['days'], M['rebal'], M['uni']
    T = len(days)
    E = c2eval.load_export(C.EXPORTS[tag], arrays=('close',))
    tickers = E['tickers']
    R = C.raw_returns(tag)
    N = R['on'].shape[0]
    C.log('N', N, 'T', T)
    # check: overnight + intraday = whole day = log(C_d / C_{d-1}) from the export's closes (= the store's)
    ce = np.asarray(E['close'], np.float64)
    cc = np.full(ce.shape, np.nan)
    with np.errstate(all='ignore'):
        cc[:, 1:] = np.log(ce[:, 1:] / ce[:, :-1])
    del ce
    f = np.isfinite(R['whole'])
    d08 = c2eval.day_index(days, '2008-01-01')
    both = f & np.isfinite(cc)
    diff = float(np.max(np.abs(R['whole'][both].astype(np.float64) - cc[both])))
    only_cc = np.isfinite(cc) & ~f
    only_cc[:, :d08] = False
    n_cc = int(np.isfinite(cc[:, d08:]).sum())
    check = {'max_abs_whole_minus_export_cc': diff, 'stock_days_whole': int(both.sum()),
             'whole_nan_but_export_return_2008+': int(only_cc.sum()), 'export_returns_2008+': n_cc,
             'whole_without_export_return': int((f & ~np.isfinite(cc)).sum())}
    C.log('returns check', check)
    del cc, only_cc, both, f
    A = {}
    mk = {}
    for ty in C.TYPES:
        A[ty], mk[ty] = C.market_adjust(R[ty], rebal, uni)
    del R
    resid_adj = float(np.nanmax(np.abs((A['on'] + A['intra'] - A['whole'])[:, d08:])))
    check['max_abs_adjusted_on_plus_intra_minus_whole'] = resid_adj
    C.log('adjusted legs add up: max diff', resid_adj)
    ks = C.k_range(tag, M)
    C.log('calendar days', len(ks), iso(days[rebal[ks[0]]]), '..', iso(days[rebal[ks[-1]]]))
    P = {ty: {} for ty in C.TYPES}
    recs = []
    for k in ks:
        t = int(rebal[k])
        rows = np.nonzero(uni[k])[0]
        rec = {'k': k, 't': t, 'date': iso(days[t]), 'n_uni': len(rows)}
        for ty in C.TYPES:
            peer, corr = C.peers_at(A[ty], rows, t)
            P[ty][k] = (rows, peer, corr)
            has = peer[:, 0] >= 0
            rec[f'n_{ty}'] = int(has.sum())
            rec[f'npeer_{ty}'] = float((peer[has] >= 0).sum(1).mean()) if has.any() else np.nan
            rec[f'corr_{ty}'] = float(np.nanmean(corr[has], 1).mean()) if has.any() else np.nan
            rec[f'corr1_{ty}'] = float(corr[has, 0].mean()) if has.any() else np.nan
            rec[f'full20_{ty}'] = int((peer[:, -1] >= 0).sum())
        for a, b in (('on', 'whole'), ('on', 'intra'), ('intra', 'whole')):
            j = jaccard(P[a][k][1], P[b][k][1])
            rec[f'jac_{a}_{b}'] = float(j.mean()) if len(j) else np.nan
            rec[f'njac_{a}_{b}'] = int(len(j))
        rec['jac_random'] = float(np.mean([10.0 / (len(rows) - 1)]))
        recs.append(rec)
        if k % 20 == 0:
            C.log(rec)
    C.save_peers(tag, ks, P)
    C.log('peers saved')
    # examples
    examples = {}
    if tag == 'old':
        d_ex = (np.datetime64(EX_DATE) - np.datetime64('1970-01-01')).astype(int)
        ke = C.kidx(rebal, int(np.searchsorted(days, d_ex, side='right')) - 1)
        examples['date'] = iso(days[rebal[ke]])
        for tk in EXAMPLES:
            j = np.nonzero(tickers == tk)[0]
            if len(j) == 0:
                continue
            j = int(j[0])
            examples[tk] = {}
            for ty in C.TYPES:
                rows, peer, corr = P[ty][ke]
                i = np.nonzero(rows == j)[0]
                if len(i) == 0:
                    examples[tk][ty] = 'not in universe'
                    continue
                i = int(i[0])
                examples[tk][ty] = [(str(tickers[p]), round(float(c), 3)) for p, c in zip(peer[i], corr[i]) if p >= 0]
        C.log('examples', json.dumps(examples)[:2000])
    # daily residual signals for the three peer types (peers and universe from the latest calendar day)
    S21 = C.roll_sum(A['whole'])
    del A
    counts = {ty: [] for ty in C.TYPES}
    for ty in C.TYPES:
        res = np.full((N, T), np.nan, np.float32)
        for k in ks:
            t0 = int(rebal[k])
            t1 = int(rebal[k + 1]) if k + 1 < len(rebal) else T
            D = np.arange(t0, t1)
            rows, peer, corr = P[ty][k]
            has = peer[:, 0] >= 0
            r, pm, own = C.residual_block(S21, rows[has], peer[has], D)
            res[np.ix_(rows[has], D)] = r
            counts[ty].append(int(np.isfinite(r[:, 0]).sum()))
        np.save(C.WORK / f'resid_{ty}_{tag}.npy', res)
        C.log('resid saved', ty, 'stocks with a signal per calendar day: median', int(np.median(counts[ty])))
        del res
    np.save(C.WORK / f'S21_{tag}.npy', S21)
    for i, rec in enumerate(recs):
        for ty in C.TYPES:
            rec[f'nsig_{ty}'] = counts[ty][i]
    out = {'tag': tag, 'check': check, 'calendar': recs, 'examples': examples}
    (C.OUT / f'build_{tag}.json').write_text(json.dumps(out, indent=1, default=float))
    C.log('done')


if __name__ == '__main__':
    main()
