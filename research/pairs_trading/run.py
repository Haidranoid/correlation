"""Run the distance-method book and the random-pair books for one export, price source and dividend setting, on
the full universe and on the 500 largest. Saves WORK/run_<tag>_<source>_<div>_<variant>.npz.

    python3 -I -B run.py old|new export|store div|nodiv [--check]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np, log  # noqa: E402

tag, source, div = sys.argv[1], sys.argv[2], sys.argv[3]
assert source in ('export', 'store') and div in ('div', 'nodiv')
CHECK = '--check' in sys.argv
DAILY = tag == 'new'

U = np.load(C.WORK / f'universe_{tag}.npz')
rebal, uni, big, complete, gd = U['rebal'], U['uni'], U['big'], U['complete'], U['gap_days']
E = c2eval.load_export(C.EXPORTS[tag], arrays=('close',))
days = E['days']
N, T = E['close'].shape
assert np.array_equal(days, U['days'])
c = C.gapfill(C.source_close(tag, source, E), list(gd))
fin = np.isfinite(c)
divfix = None
if div == 'div':
    sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
    import pubdata  # noqa: E402
    divfix = pubdata.div_fix(tag)
    assert divfix.shape == (N, T)
L = C.log_tr(c, days, divfix)
del c, divfix
log('panels ready', tag, source, div)

K = len(rebal)
TAGC = {'old': 1, 'new': 2}[tag]


def run_variant(variant):
    res = {
        'valid': np.zeros(K, bool), 'nuni': np.zeros(K, np.int32),
        'pairs': np.full((K, C.NPAIRS, 2), -1, np.int32), 'ssd': np.full((K, C.NPAIRS), np.nan),
        'sigma': np.full((K, C.NPAIRS), np.nan),
        'pnl': np.zeros((K, C.NPAIRS, C.NPORT)), 'cost': np.zeros((K, C.NPAIRS, C.NPORT)),
        'n_open': np.zeros((K, C.NPAIRS), np.int32), 'n_conv': np.zeros((K, C.NPAIRS), np.int32),
        'n_forced': np.zeros((K, C.NPAIRS), np.int32), 'days_open': np.zeros((K, C.NPAIRS), np.int32),
        'rpnl': np.zeros((K, C.NRAND, C.NPORT), np.float32), 'rcost': np.zeros((K, C.NRAND, C.NPORT), np.float32),
        'r_open_rate': np.full(K, np.nan), 'r_trips': np.full(K, np.nan), 'r_sigma': np.full(K, np.nan),
        'r_forced_share': np.full(K, np.nan),
    }
    if DAILY:   # daily P&L, for re-blocking the unseen months from their first day
        res['pnl_d'] = np.zeros((K, C.NPAIRS, C.H + 1), np.float32)
        res['cost_d'] = np.zeros((K, C.NPAIRS, C.H + 1), np.float32)
        res['rpnl_d'] = np.zeros((K, C.NRAND, C.H + 1), np.float32)
        res['rcost_d'] = np.zeros((K, C.NRAND, C.H + 1), np.float32)
    for k, t0 in enumerate(rebal):
        if t0 < C.DATA_W - 1:
            continue
        m = uni[k] & complete[k] & (big[k] if variant == 'big' else True)
        idx = np.nonzero(m)[0]
        if len(idx) < 50:
            continue
        a = t0 - C.FORM + 1
        # point in time: formation reads L[:, a..t0] only
        Lf = L[idx, a:t0 + 1]
        assert Lf.shape[1] == C.FORM and np.isfinite(Lf).all()
        P = np.exp(Lf - Lf[:, :1])
        i, j, s = C.top_pairs(P)
        if CHECK and k == 200 and variant == 'all':
            sub = np.arange(min(150, len(idx)))
            bi, bj, bs = C.brute_ssd(P[sub])
            gi, gj, gs = C.top_pairs(P[sub])
            log('brute-force check (150 stocks):', 'same pairs' if (np.array_equal(bi, gi) and np.array_equal(bj, gj))
                else 'DIFFERENT PAIRS', 'max |SSD diff|', float(np.abs(bs - gs).max()))
        # trading window: columns t0..t0+H (padded past the export's end with zero returns)
        hmax = min(C.H, T - 1 - t0)

        def paths(ii, jj):
            gi_, gj_ = idx[ii], idx[jj]
            base_i, base_j = L[gi_, a], L[gj_, a]
            W = C.H + 1
            PA = np.empty((len(ii), W))
            PB = np.empty((len(ii), W))
            PA[:, :hmax + 1] = np.exp(L[gi_, t0:t0 + hmax + 1] - base_i[:, None])
            PB[:, :hmax + 1] = np.exp(L[gj_, t0:t0 + hmax + 1] - base_j[:, None])
            PA[:, hmax + 1:] = PA[:, hmax:hmax + 1]
            PB[:, hmax + 1:] = PB[:, hmax:hmax + 1]
            oA = np.zeros((len(ii), W), bool)
            oB = np.zeros((len(ii), W), bool)
            oA[:, :hmax + 1] = fin[gi_, t0:t0 + hmax + 1]
            oB[:, :hmax + 1] = fin[gj_, t0:t0 + hmax + 1]
            # last real close in the window (data ending inside the period closes the pair there)
            la = np.where(oA[:, :hmax + 1], np.arange(hmax + 1)[None, :], 0).max(1)
            lb = np.where(oB[:, :hmax + 1], np.arange(hmax + 1)[None, :], 0).max(1)
            endp = np.minimum(np.minimum(la, lb), hmax)
            Fi, Fj = P[ii], P[jj]
            sig = (Fi - Fj).std(1)
            return PA, PB, oA, oB, sig, endp

        PA, PB, oA, oB, sig, endp = paths(i, j)
        r = C.simulate(PA, PB, oA, oB, sig, endp)
        res['valid'][k] = True
        res['nuni'][k] = len(idx)
        res['pairs'][k] = np.stack([idx[i], idx[j]], 1)
        res['ssd'][k] = s
        res['sigma'][k] = sig
        res['pnl'][k] = C.month_sums(r['pnl'])
        res['cost'][k] = C.month_sums(r['cost'])
        if DAILY:
            res['pnl_d'][k] = r['pnl']
            res['cost_d'][k] = r['cost']
        for f in ('n_open', 'n_conv', 'n_forced', 'days_open'):
            res[f][k] = r[f]
        # random books: NRAND x 20 pairs drawn uniformly from the same universe (same draws for every source/div)
        rng = np.random.default_rng([TAGC, k, 0 if variant == 'all' else 1])
        n = len(idx)
        ri = rng.integers(0, n, C.NRAND * C.NPAIRS)
        rj = rng.integers(0, n - 1, C.NRAND * C.NPAIRS)
        rj = rj + (rj >= ri)
        for b0 in range(0, C.NRAND * C.NPAIRS, 5000):
            sl = slice(b0, b0 + 5000)
            PA, PB, oA, oB, sig, endp = paths(ri[sl], rj[sl])
            rr = C.simulate(PA, PB, oA, oB, sig, endp)
            mp = C.month_sums(rr['pnl']).reshape(-1, C.NPAIRS, C.NPORT).sum(1)
            mc = C.month_sums(rr['cost']).reshape(-1, C.NPAIRS, C.NPORT).sum(1)
            res['rpnl'][k, b0 // C.NPAIRS:(b0 + 5000) // C.NPAIRS] = mp
            res['rcost'][k, b0 // C.NPAIRS:(b0 + 5000) // C.NPAIRS] = mc
            if DAILY:
                res['rpnl_d'][k, b0 // C.NPAIRS:(b0 + 5000) // C.NPAIRS] = rr['pnl'].reshape(-1, C.NPAIRS, C.H + 1).sum(1)
                res['rcost_d'][k, b0 // C.NPAIRS:(b0 + 5000) // C.NPAIRS] = rr['cost'].reshape(-1, C.NPAIRS, C.H + 1).sum(1)
            if b0 == 0:
                res['r_open_rate'][k] = float((rr['n_open'] > 0).mean())
                res['r_trips'][k] = float(rr['n_open'].mean())
                res['r_sigma'][k] = float(np.median(sig))
                res['r_forced_share'][k] = float(rr['n_forced'].sum() / max(1, rr['n_open'].sum()))
        if k % 20 == 0:
            log(variant, 'k', k, c2eval.iso(days[t0]), 'n', len(idx), 'top ssd', float(s[0]), float(s[-1]),
                *(('book gross 6m', float(res['pnl'][k].sum() / C.NPAIRS)) if tag == 'old' else ()))
    return res


for variant in ('all', 'big'):
    res = run_variant(variant)
    np.savez(C.WORK / f'run_{tag}_{source}_{div}_{variant}.npz', rebal=rebal, days=days, T=T, **res)
    log('saved', variant)
