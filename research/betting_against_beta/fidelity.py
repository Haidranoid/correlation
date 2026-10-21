"""Fidelity check, run before any test is read: the monthly correlation of this BAB (gross) with AQR's US BAB over the
overlapping months of the old export (return months 2008-03..2025-06). Must be >= 0.5. Prints correlations only.

    python3 -I -B fidelity.py      -> fidelity.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import np, WORK, HERE  # noqa: E402
import pubdata  # noqa: E402

z = np.load(WORK / 'series_old.npz')
months = z['months']
ad, ac = pubdata.aqr_bab_us_monthly()
amon = np.array([str(d)[:7] for d in ad])
aqr = dict(zip(amon, ac['BAB_US']))
lines = []
for s in ('bab', 'bab500', 'bab_unlev', 'nmv_ew', 'nmv_dv'):
    x = z[f'{s}__gross']
    y = np.array([aqr.get(m, np.nan) for m in months])
    for lab, a, b in (('all', '2008-03', '2025-06'), ('2008-03..2019', '2008-03', '2019-12'), ('2020..2025-06', '2020-01', '2025-06')):
        ok = np.isfinite(x) & np.isfinite(y) & (months >= a) & (months <= b)
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        lines.append(f'{s:10s} gross vs AQR BAB_US, {lab:14s}: monthly correlation {r:.3f} over {int(ok.sum())} months '
                     f'({months[ok][0]}..{months[ok][-1]})')
lines.append('PASS fidelity (bab, all months, >= 0.5): ' + str(float(lines[0].split('correlation ')[1].split()[0]) >= 0.5))
print('\n'.join(lines))
(HERE / 'fidelity.txt').write_text('\n'.join(lines) + '\n')
