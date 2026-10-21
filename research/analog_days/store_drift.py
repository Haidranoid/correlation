"""Where the new store's pre-2025-06-27 fund closes differ from the old store's (daily log returns)."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C
from common import c2eval, np
Po, Pn = C.build('old'), C.build('new')
L = C.LAST_OLD + 1
ro = np.diff(np.log(Po['c'][:, :L]), axis=1)
rn = np.diff(np.log(Pn['c'][:, :L]), axis=1)
d = rn - ro
days = Po['days']
for i, t in enumerate(C.TICK):
    x = np.abs(d[i]); ok = np.isfinite(x)
    big = np.nonzero(x > 1e-3)[0]
    print(f'{t}: days differing by >1e-4: {(x[ok] > 1e-4).sum()}, >1e-3: {len(big)}, max {np.nanmax(x):.2e}; largest at ' +
          ', '.join(f'{c2eval.iso(days[j + 1])} ({d[i, j]:+.4f})' for j in big[np.argsort(-x[big])][:4]))
# weekly feature drift: z-state differences
dz = np.abs(Pn['Z'][:L] - Po['Z'][:L])
print('state z differences: median', np.nanmedian(dz), '99th pct', np.nanpercentile(dz, 99), 'max', np.nanmax(dz))
# nearest-neighbour distance gaps: how close is the 50th analog to the 51st candidate typically
# analog-set and book agreement before / after the first >1e-4 difference
first_diff = int(np.nonzero(np.abs(np.nan_to_num(d)).max(0) > 1e-4)[0].min()) + 1
print('first daily return differing by >1e-4:', c2eval.iso(days[first_diff]))
tiny = np.abs(np.nan_to_num(d)).max(0)
print('days before it with any nonzero difference:', int((tiny[:first_diff - 1] > 0).sum()), 'max', float(tiny[:first_diff - 1].max()))
ho = Po['idx'][:L, 0] >= 0
same = (Po['idx'][:L] == Pn['idx'][:L]).all(1)
sameset = np.array([set(a) == set(b) for a, b in zip(Po['idx'][:L], Pn['idx'][:L])])
pre = np.arange(L) < first_diff
print(f'analog sets identical (same order) before {c2eval.iso(days[first_diff])}: {100 * same[ho & pre].mean():.2f}% of forecast days; after: {100 * same[ho & ~pre].mean():.2f}%')
print(f'analog sets identical (as sets): before {100 * sameset[ho & pre].mean():.2f}%, after {100 * sameset[ho & ~pre].mean():.2f}%')
fo = np.array([f for f in range(C.first_formation(ho, C.LAST_OLD), L) if f % C.STEP == 0 and f + C.H <= C.LAST_OLD])
bo = np.array([C.legs(Po['F'][f]) for f in fo]); bn = np.array([C.legs(Pn['F'][f]) for f in fo])
sb = (bo == bn).all(1)
pf = fo < first_diff
print(f'weekly book identical: before {100 * sb[pf].mean():.1f}% of {pf.sum()} weeks, after {100 * sb[~pf].mean():.1f}% of {(~pf).sum()} weeks')
# Test 1 and Test 3 with the new store's pre-2025-06 closes (diagnostic only; the tests use the old store)
yr = np.array([C.year_of(days, f) for f in fo])
for lab, P in (('old store', Po), ('new store', Pn)):
    r = C.weekly(P, fo, P['F'])
    a1 = C.tstat(r['net'][yr <= 2019]); a3 = C.tstat(r['net'][yr >= 2020])
    print(f'{lab}: Test 1 net {1e4 * a1[0]:+.2f} bps/wk (t {a1[1]:+.2f}); Test 3 net {1e4 * a3[0]:+.2f} bps/wk (t {a3[1]:+.2f})')
