"""Universe per formation for one export: the judge-like set (point in time, from the export) and the 500 largest,
plus 'complete' = a close in BOTH price sources (export and store, market-wide gap days forward-filled) on every day
of the formation year t0-251..t0. Saves WORK/universe_<tag>.npz.

    python3 -I -B prep.py old|new
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np, log  # noqa: E402

tag = sys.argv[1]
E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'dollar_volume'))
N, T = E['close'].shape
log('export', tag, N, T)
tw = c2eval.twins(E)
rebal = tw['rebal']
uni, big = C.universe(E, tw)
log('universe done; sizes median', int(np.median(uni.sum(1)[uni.sum(1) > 0])), 'max', int(uni.sum(1).max()))
ce = C.source_close(tag, 'export', E)
gd = C.gap_days(ce)
log('gap days', [c2eval.iso(E['days'][d]) for d in gd])
ce = C.gapfill(ce, gd)
cs = C.source_close(tag, 'store', E)
both = np.isfinite(ce) & np.isfinite(cs)
with np.errstate(all='ignore'):
    mis = both & (np.abs(np.log(cs / ce)) > 0.01)
log('store vs export closes: both finite', int(both.sum()), 'differ > 1%', int(mis.sum()),
    f'({mis.sum() / both.sum():.4%})')
gds = C.gap_days(cs)
log('store gap days', [c2eval.iso(E['days'][d]) for d in gds])
cs = C.gapfill(cs, sorted(set(gd) | set(gds)))
fin = np.isfinite(ce) & np.isfinite(cs)
del ce, cs
# rolling count of finite closes over the formation window
cnt = np.zeros((N, T + 1), np.int32)
np.cumsum(fin, 1, out=cnt[:, 1:])
complete = np.zeros_like(uni)
for k, t0 in enumerate(rebal):
    if t0 < C.FORM - 1:
        continue
    complete[k] = (cnt[:, t0 + 1] - cnt[:, t0 + 1 - C.FORM]) == C.FORM
nu = (uni & complete).sum(1)
log('universe with complete formation year: median', int(np.median(nu[nu > 0])), 'min', int(nu[nu > 0].min()),
    'max', int(nu.max()), '; dropped by completeness: median', int(np.median((uni & ~complete).sum(1)[nu > 0])))
np.savez(C.WORK / f'universe_{tag}.npz', rebal=rebal, uni=uni, big=big, complete=complete, days=E['days'],
         gap_days=np.array(sorted(set(gd) | set(gds))), mismatch=int(mis.sum()), both=int(both.sum()))
log('saved')
