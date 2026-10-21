"""Data check (no outcomes): find the 10 funds in both stores, coverage on the export days, bad prints (|daily simple
move| > 25%), missing closes, and whether the old and new exports' days agree."""
import sys
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval
import numpy as np

TICK = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
for tag, store_p, exp_p in (('old', c2eval.STORE_OLD, c2eval.EXPORT_OLD), ('new', c2eval.STORE_NEW, c2eval.EXPORT_NEW)):
    E = c2eval.load_export(exp_p, arrays=())
    S = c2eval.load_store(store_p)
    days = E['days']
    print(f'== {tag}: export {c2eval.iso(days[0])}..{c2eval.iso(days[-1])} T={len(days)}')
    ids = []
    for t in TICK:
        m = [s for s in S['syms'] if s['ticker'] == t]
        print(t, [(s['id'], s['exchange'], s['kind'], s['rows']) for s in m])
        m = [s for s in m if s['kind'] == 'etf']
        ids.append(m[0]['id'])
    C = c2eval.store_panel(S, ids, days, 'close').astype(np.float64)
    for k, t in enumerate(TICK):
        c = C[k]
        fin = np.isfinite(c)
        first = np.argmax(fin)
        miss = np.nonzero(~fin[first:])[0] + first
        r = c[1:] / c[:-1] - 1
        big = np.nonzero(np.abs(r) > 0.25)[0] + 1
        print(f'{t}: first {c2eval.iso(days[first])}, missing after first: {len(miss)} '
              f'{[c2eval.iso(days[i]) for i in miss[:12]]}; |move|>25%: {[(c2eval.iso(days[i]), round(float(r[i-1]), 3)) for i in big]}; '
              f'max |move| {np.nanmax(np.abs(r)):.3f}')
    # days where any fund is missing
    anym = np.nonzero(~np.isfinite(C).all(0))[0]
    print('days with any of the 10 missing:', len(anym), [c2eval.iso(days[i]) for i in anym[:30]])
    if tag == 'old':
        d_old = days
    else:
        print('old days == new days[:len(old)]:', bool(np.array_equal(d_old, days[:len(d_old)])))
