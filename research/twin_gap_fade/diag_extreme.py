"""Post hoc: how often the twin book's entry (the open) is the day's extreme -- the low for a long, the high for a
short -- against the whole ranked set. A print that is often the day's extreme cannot be had just after the open.

    python3 -I diag_extreme.py     # appends to out/diag_old.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import books as B  # noqa: E402
from books import c2eval, np  # noqa: E402

R = B.load_rows('old')
days = R['days']
E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
S = c2eval.load_store(c2eval.STORE_OLD)
keep = ~R['idbad']
t, i, z = R['t'][keep], R['i'][keep], R['z'][keep]
o = c2eval.store_panel(S, E['ids'], days, 'open')
lo_eq = np.zeros(len(t), bool); hi_eq = np.zeros(len(t), bool)
l = c2eval.store_panel(S, E['ids'], days, 'low')
lo_eq = o[i, t] == l[i, t]
del l
h = c2eval.store_panel(S, E['ids'], days, 'high')
hi_eq = o[i, t] == h[i, t]
del h
u, st, en = B.day_slices(t)
L = np.zeros(len(t), bool); Sh = np.zeros(len(t), bool)
for a, b in zip(st, en):
    n = b - a
    if n < B.MIN_N:
        continue
    nl = int(round(0.1 * n))
    oo = np.argsort(z[a:b], kind='stable')
    L[a + oo[:nl]] = True
    Sh[a + oo[n - nl:]] = True
out = []
for pn, a, b in (('2008-2019', '2008-01-01', '2020-01-01'), ('2020-2025-06', '2020-01-01', None)):
    m = B.period_mask(days, t, a, b)
    out.append(f'{pn}: open == day low: longs {lo_eq[m & L].mean():.1%} vs all ranked {lo_eq[m].mean():.1%}; '
               f'open == day high: shorts {hi_eq[m & Sh].mean():.1%} vs all ranked {hi_eq[m].mean():.1%}')
txt = '\nopen as the day\'s extreme (post hoc):\n' + '\n'.join('  ' + x for x in out)
print(txt)
with open(B.OUT / 'diag_old.txt', 'a') as fh:
    fh.write(txt + '\n')
