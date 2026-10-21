"""FIP, DIP, FIP - DIP and MOM net returns summed by calendar year of the period's end (liquid set, divfix; old export
through 2025-06, then the new export's unseen periods). Reported, not a test.

    python3 -I -B by_year.py      -> out/by_year.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import np, WORK, OUT, UNSEEN  # noqa: E402

lines = ['| Year (period end) | Periods | FIP net | DIP net | FIP − DIP | MOM net | FIP net, 500 largest |', '|---|---|---|---|---|---|---|']
for tag, lo, hi in (('old', '2006-01-01', '2025-06-27'), ('new', *UNSEEN)):
    z = np.load(WORK / f'books_{tag}_divfix.npz')
    end = z['end_iso']
    m = (end >= lo) & (end <= hi)
    yrs = sorted({e[:4] for e in end[m]})
    for y in yrs:
        mm = m & np.array([e[:4] == y for e in end])
        f = np.nansum(z['all__FIP__net21'][mm]) * 100
        d = np.nansum(z['all__DIP__net21'][mm]) * 100
        mo = np.nansum(z['all__MOM__net21'][mm]) * 100
        fb = np.nansum(z['big__FIP__net21'][mm]) * 100
        lab = y + (' (unseen)' if tag == 'new' else '')
        lines.append(f'| {lab} | {int(mm.sum())} | {f:+.1f}% | {d:+.1f}% | {f - d:+.1f}% | {mo:+.1f}% | {fb:+.1f}% |')
(OUT / 'by_year.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
