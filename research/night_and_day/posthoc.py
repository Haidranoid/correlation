"""Post-hoc splits (after the results were seen; descriptive only): the intraday and overnight books by sub-period,
from the saved daily series of tests.py.

    python3 -I posthoc.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval

P = np.load(C.WORK / 'meta_old.npz')
days = P['days']
out = []
for which in ('all', 'big'):
    R = np.load(C.WORK / f'series_old_{which}.npz')
    da = days[R['d']]
    for nm, a, b in (('2008-2009', '2008-01-01', '2009-12-31'), ('2010-2019', '2010-01-01', '2019-12-31'),
                     ('2013-2019', '2013-01-01', '2019-12-31'), ('2020-2025-06', '2020-01-01', '2025-06-27')):
        lo = (np.datetime64(a) - np.datetime64('1970-01-01')).astype(int)
        hi = (np.datetime64(b) - np.datetime64('1970-01-01')).astype(int)
        s = (da >= lo) & (da <= hi)
        cells = []
        for lg in ('on', 'in', 'full'):
            mu, t, n = C.tstat(R[f'ls_{lg}'][s])
            cells.append(f'{lg} {mu * 1e4:+.2f} (t {t:+.2f})')
        line = f'{which} {nm} days {int(s.sum())}: ' + ', '.join(cells)
        out.append(line)
        print(line)
(C.OUT / 'posthoc_old.txt').write_text('\n'.join(out) + '\n')
