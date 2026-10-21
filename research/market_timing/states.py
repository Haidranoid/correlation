"""Compute the monthly states (average correlation, absorption ratio) for one export and save them.
No returns after t0 are touched here.

    python3 -I states.py old|new
"""
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_timing')
import numpy as np  # noqa: E402

import common as C  # noqa: E402

tag = sys.argv[1]
assert tag in ('old', 'new')
E, tw = C.load(tag)
if tag == 'new':
    Eo = C.c2eval.load_export(C.EXPORTS['old'], arrays=())
    assert np.array_equal(E['days'][:len(Eo['days'])], Eo['days']), 'new export days do not continue the old ones'
    print('new export: first', len(Eo['days']), 'days equal the old export\'s')
print(tag, 'export', E['close'].shape, 'calendar points', len(tw['rebal']), C.c2eval.iso(E['days'][tw['rebal'][0]]),
      '..', C.c2eval.iso(E['days'][tw['rebal'][-1]]), flush=True)
st = C.states(E, tw)
st['days'] = np.asarray(E['days'])
np.savez(C.WORK / f'states_{tag}.npz', **st)
with open(C.HERE / f'states_{tag}.tsv', 'w') as f:
    f.write('k\tt0\tdate\tn_eligible\tn_universe\tmedian_dv_cutoff\tavg_corr\tabsorption\n')
    for k, t0 in enumerate(st['rebal']):
        f.write(f"{k}\t{t0}\t{C.c2eval.iso(E['days'][t0])}\t{st['n_elig'][k]}\t{st['n_sel'][k]}\t{st['mdv_cut'][k]:.4g}\t"
                f"{st['avg_corr'][k]:.4f}\t{st['absorption'][k]:.4f}\n")
a = st['avg_corr']
print('avg corr: min', np.nanmin(a).round(3), 'median', np.nanmedian(a).round(3), 'max', np.nanmax(a).round(3),
      'at', C.c2eval.iso(E['days'][st['rebal'][np.nanargmax(a)]]))
b = st['absorption']
print('absorption: min', np.nanmin(b).round(3), 'median', np.nanmedian(b).round(3), 'max', np.nanmax(b).round(3))
print('universe size: min', st['n_sel'].min(), 'eligible min', st['n_elig'].min())
print('corr(avg corr, absorption) =', np.corrcoef(a[np.isfinite(a)], b[np.isfinite(a)])[0, 1].round(3))
