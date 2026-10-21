"""Post hoc: the books within the 100 most-traded stocks (63-day median dollar volume), where spreads are 1-3 bps,
so bid-ask bounce in the open print can explain only a few bps a day.

    python3 -I diag_top100.py     # appends to out/diag_old.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import books as B  # noqa: E402
from books import np  # noqa: E402


def fmt(x):
    m, t, n = B.tstat(x)
    return f'{m * 1e4:+.2f} ({t:+.2f})'


R = B.load_rows('old')
days = R['days']
Bk = B.books(R, '100')
out = ['\nthe 100 most traded (legs of 10), bps/day (t):']
for pn, a, b in (('2008-2019', '2008-01-01', '2020-01-01'), ('2020-2025-06', '2020-01-01', None),
                 ('2015-2019', '2015-01-01', '2020-01-01'), ('2023-2025-06', '2023-01-01', None)):
    m = B.period_mask(days, Bk['t'], a, b)
    out.append(f'  {pn}: twin gross {fmt(Bk["twin_gross"][m])}, head {fmt(B.net(Bk, "twin")[m])}, raw gross '
               f'{fmt(Bk["raw_gross"][m])}, break-even p {np.mean(B.net(Bk, "twin", p=0)[m]) / 2 * 1e4:+.1f} bps')
txt = '\n'.join(out)
print(txt)
with open(B.OUT / 'diag_old.txt', 'a') as fh:
    fh.write(txt + '\n')
