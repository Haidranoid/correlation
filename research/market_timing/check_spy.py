"""Data check: SPY from both stores, aligned to each export's days. Bad prints, missing days, first day."""
import sys
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np
import c2eval

for tag, exp, st in (('old', c2eval.EXPORT_OLD, c2eval.STORE_OLD), ('new', c2eval.EXPORT_NEW, c2eval.STORE_NEW)):
    E = c2eval.load_export(exp, arrays=('market',))
    S = c2eval.load_store(st)
    sid = [s['id'] for s in S['syms'] if s['ticker'] == 'SPY' and s['kind'] == 'etf']
    c = c2eval.store_panel(S, sid, E['days'], 'close')[0].astype(np.float64)
    v = c2eval.store_panel(S, sid, E['days'], 'volume')[0].astype(np.float64)
    days = E['days']
    first = int(np.argmax(np.isfinite(c)))
    print(tag, 'T', len(days), 'SPY first export index', first, c2eval.iso(days[first]), 'last', c2eval.iso(days[np.nonzero(np.isfinite(c))[0][-1]]))
    miss = [d for d in range(first, len(days)) if not np.isfinite(c[d])]
    print(' missing export days after first:', len(miss), [c2eval.iso(days[d]) for d in miss][:20])
    # store rows not on export days
    a, b = S['offs'][sid[0]], S['offs'][sid[0] + 1]
    sd = np.asarray(S['date'][a:b])
    extra = np.setdiff1d(sd, days)
    print(' store SPY dates not in export days:', len(extra), [c2eval.iso(x) for x in extra[:10]])
    r = np.full(len(c), np.nan)
    r[1:] = np.log(c[1:] / c[:-1])
    big = np.nonzero(np.abs(np.nan_to_num(r)) > 0.07)[0]
    m = np.asarray(E['market'])
    for d in big:
        print(f'  {c2eval.iso(days[d])} SPY log r {r[d]:+.4f}  close {c[d-1]:.2f}->{c[d]:.2f}  vol {v[d]:.3g}  market(eq-wt) {m[d]:+.4f}')
    # price continuity: ratio of consecutive closes against market; flag days where |r - market| > 0.04
    dev = np.abs(np.nan_to_num(r) - np.nan_to_num(m))
    odd = np.nonzero(dev > 0.04)[0]
    print(' days with |SPY - eq-wt market| > 4%:', [(c2eval.iso(days[d]), round(r[d], 4), round(float(m[d]), 4)) for d in odd])
    # index-based spacing check: the twins calendar
    for k in (756, 777, 798, 819, 840):
        print('  calendar', k, c2eval.iso(days[k]), 'SPY returns in 21 days ending there:', int(np.isfinite(r[k-20:k+1]).sum()))
