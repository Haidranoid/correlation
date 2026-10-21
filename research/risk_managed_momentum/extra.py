"""Added after seeing results (reported, not a test): is the Sharpe gain only the 2009 crash? Sharpe RMOM - WML (net,
daily x sqrt(252)) over 2010-2019, and over 2006-2019 without 2009, with the same 21-day block bootstrap.

    python3 -I -B extra.py    -> extra.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import np, WORK, HERE  # noqa: E402
sys.argv = [sys.argv[0], 'extra']   # report.py reads a tag at import; main() doesn't run
import report as R  # noqa: E402  (report.py's helpers; its main() only runs as a script)

lines = []
z = dict(np.load(WORK / 'series_old_divfix.npz'))
dm = np.array([str(np.datetime64(int(d), 'D'))[:7] for d in z['days']])
for v in ('all', 'big'):
    for lab, mask in (('2010-2019', (dm >= '2010-01') & (dm <= '2019-12')),
                      ('2006-2019 without 2009', (dm >= '2006-01') & (dm <= '2019-12') & ((dm < '2009-01') | (dm > '2009-12'))),
                      ('2006-2008', (dm >= '2006-01') & (dm <= '2008-12'))):
        rm, wm = z[f'{v}__rmom__net_d'], z[f'{v}__wml__net_d']
        ok = mask & np.isfinite(rm) & np.isfinite(wm)
        d = R.sr_d(rm[ok]) - R.sr_d(wm[ok])
        bs = R.boot_dsharpe(rm[ok], wm[ok])
        s = (f'{v:3s} {lab:24s}: Sharpe WML {R.sr_d(wm[ok]):+.2f}, RMOM {R.sr_d(rm[ok]):+.2f}, difference {d:+.3f} '
             f'(95% {np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}), one-sided p {R.p_greater(bs, d):.3f}, {int(ok.sum())} days')
        print(s)
        lines.append(s)
(HERE / 'extra.txt').write_text('\n'.join(lines) + '\n')
