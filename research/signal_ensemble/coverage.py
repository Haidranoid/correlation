"""Coverage of each component inside the universe (no returns are read): decides the minimum-components rule.

    python3 -I coverage.py old    -> coverage_old.txt
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import c2eval, np, EXPORTS, wdir, HERE, LGBM, weekly_days

tag = _sys.argv[1]
E = c2eval.load_export(EXPORTS[tag], arrays=())
days = E['days']
U = np.load(wdir(tag) / 'universe.npy', mmap_mode='r')
names = ['rev5', 'mom12_1', 'twin5', 'corr', 'vol']
P = {n: np.load(wdir(tag) / f'{n}.npy', mmap_mode='r') for n in names}
if tag == 'old':
    P['lgbm'] = np.load(LGBM, mmap_mode='r')
T = len(days)
W = weekly_days(T)
lines = []
for y in range(2006, 2027):
    ws = [d for d in W if c2eval.iso(days[d]).startswith(str(y))]
    if not ws:
        continue
    tot = {n: 0 for n in P}
    nu = 0
    allc = 0
    ge3 = 0
    for d in ws:
        u = np.nonzero(U[d])[0]
        nu += len(u)
        f = {n: np.isfinite(P[n][u, d]) for n in P}
        for n in P:
            tot[n] += int(f[n].sum())
        comps = [f['rev5'], f['mom12_1'], f['twin5'], f['corr'] & f['vol']]
        if 'lgbm' in P and f['lgbm'].mean() > 0.5:
            comps.append(f['lgbm'])
        k = np.sum(comps, 0)
        allc += int((k == len(comps)).sum())
        ge3 += int((k >= 3).sum())
    lines.append(f'{y}: universe {nu / len(ws):7.0f}/week; ' + ', '.join(f'{n} {tot[n] / nu:.3f}' for n in P)
                 + f'; all of the day\'s components {allc / nu:.3f}; >= 3 {ge3 / nu:.3f}')
txt = '\n'.join(lines)
print(txt)
(HERE / f'coverage_{tag}.txt').write_text(txt + '\n')
