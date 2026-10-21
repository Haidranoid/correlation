"""Hand checks of the plumbing (no test statistics):
  - PTH for a few stock-days recomputed with a plain loop over the export's closes;
  - one cohort's leg returns over one period recomputed from the raw closes with a plain loop;
  - the period breakpoints of the new export around 2025-06-27, and the grid being the same on both exports.

    python3 -I -B check.py   -> out/check.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, WORK, OUT, EXPORTS, FRAC  # noqa: E402
import portfolios as PF  # noqa: E402

lines = []
E = c2eval.load_export(EXPORTS['old'], arrays=('close',))
C = E['close']
pth = np.load(WORK / 'pth_old.npy', mmap_mode='r')
z = np.load(WORK / 'panels_old.npz')
rebal = z['rebal']
rng = np.random.default_rng(7)
bad = 0
for _ in range(2000):
    j, t = int(rng.integers(0, C.shape[0])), int(rng.integers(300, C.shape[1]))
    w = np.asarray(C[j, t - 251:t + 1], np.float64)
    n = np.isfinite(w).sum()
    ref = (C[j, t] / np.nanmax(w)) if (n >= 227 and np.isfinite(C[j, t])) else np.nan
    got = pth[j, t]
    if not ((np.isnan(ref) and np.isnan(got)) or abs(ref - got) < 1e-6):
        bad += 1
lines.append(f'PTH by hand, 2000 random stock-days: {bad} mismatches')

# one cohort, one period, by hand
k = int(np.searchsorted(rebal, c2eval.day_index(E['days'], '2012-06-01')))
t0 = rebal[k]
base = z['uni'][k] & np.isfinite(z['pth'][k]) & np.isfinite(z['mom'][k])
L, S, A = PF.legs_for(z['pth'][k], base, seed=1000 + k)
lines.append(f'cohort formed {c2eval.iso(E["days"][t0])}: eligible {len(A)}, legs {len(L)} / {len(S)} '
             f'(30% = {FRAC * len(A):.1f}); long PTH min {np.min(z["pth"][k][L]):.3f}, short PTH max {np.max(z["pth"][k][S]):.3f}')
a, b = rebal[k + 2], rebal[k + 3]        # its third period


def leg_by_hand(members):
    vals_a, vals_b = [], []
    for j in members:
        row = np.asarray(C[j], np.float64)
        p0 = row[t0]

        def last(t):
            x = row[:t + 1]
            f = np.nonzero(np.isfinite(x))[0]
            return x[f[-1]]
        vals_a.append(last(a) / p0)
        vals_b.append(last(b) / p0)
    return np.mean(vals_b) / np.mean(vals_a) - 1


s = np.load(WORK / 'series_old.npz')
P = PF.ffill(C)
vl = PF.leg_path(P, L, t0, t0 + 126)
vs = PF.leg_path(P, S, t0, t0 + 126)
lines.append(f'period {c2eval.iso(E["days"][a])}..{c2eval.iso(E["days"][b])}: long by hand {leg_by_hand(L):+.6f}, '
             f'code {vl[b - t0] / vl[a - t0] - 1:+.6f}; short by hand {leg_by_hand(S):+.6f}, code {vs[b - t0] / vs[a - t0] - 1:+.6f}')
jj = int(np.nonzero((s['pa'] == a) & (s['pb'] == b))[0][0])
lines.append(f'  that period in the book: {int(s["pth_ncoh"][jj])} cohorts active')

# grids and breakpoints
zn = np.load(WORK / 'panels_new.npz') if (WORK / 'panels_new.npz').exists() else None
if zn is not None:
    En = c2eval.load_export(EXPORTS['new'], arrays=())
    same = np.array_equal(zn['rebal'][:len(rebal)], rebal) and np.array_equal(En['days'][:len(E['days'])], E['days'])
    lines.append(f'grid identical on the shared days: {same}')
    pa, pb = PF.periods_for('new', En['days'], zn['rebal'], len(En['days']))
    for x, y in zip(pa, pb):
        if y >= c2eval.day_index(En['days'], '2025-05-01'):
            lines.append(f'  new period {c2eval.iso(En["days"][x])} -> {c2eval.iso(En["days"][y])} ({y - x} days)')
(OUT / 'check.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
