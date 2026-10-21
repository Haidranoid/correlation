"""Part B plumbing check on the new export (days through 2025-06-27 only, so no unseen-month outcome is read): the
sign book's full-day return from the new export's closes against research/cluster_leadlag_all-exchanges_daily.tsv,
which was made on the new store.

    python3 -I partb_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval

E = c2eval.load_export(c2eval.EXPORT_NEW, arrays=('close',))
days = E['days']
ce = np.asarray(E['close'], np.float64)
Wt = np.load(C.WORK / 'cluster_w_new.npy')
cut = c2eval.day_index(days, '2025-06-28')
rows = [l.rstrip('\n').split('\t') for l in open('/home/bhoult/programming/correlation/research/cluster_leadlag_all-exchanges_daily.tsv')][1:]
pub = {r[0]: float(r[1]) for r in rows}
diffs = []
for d in range(283, cut):
    t = d - 1
    w = np.nan_to_num(Wt[:, t].astype(np.float64))
    with np.errstate(all='ignore'):
        nd = ce[:, d] / ce[:, t] - 1
    nd[~np.isfinite(nd) | ~(ce[:, t] > 0)] = 0.0
    k = c2eval.iso(days[d])
    if k in pub:
        diffs.append(abs(float(w @ nd) - pub[k]))
diffs = np.array(diffs)
line = (f'new export, days through 2025-06-27: {len(diffs)} days compared, max |diff| {diffs.max():.2e}, '
        f'mean |diff| {diffs.mean():.2e}, days with |diff| > 1e-6: {(diffs > 1e-6).sum()}')
print(line)
(C.OUT / 'partb_check.txt').write_text(line + '\n')
