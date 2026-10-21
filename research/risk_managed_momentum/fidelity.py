"""Fidelity check, run before any test is read: correlation of our daily WML with French's daily UMD over
2006-01-01..2025-06-27 (old export) must be >= 0.5 (the .md, Rules). Prints only correlations.

    python3 -I -B fidelity.py      -> fidelity.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import np, WORK, HERE, pubdata  # noqa: E402

lines = []


def P(s=''):
    print(s, flush=True)
    lines.append(s)


ud, uc = pubdata.umd_daily()
for px in ('plain', 'divfix'):
    p = WORK / f'series_old_{px}.npz'
    if not p.exists():
        continue
    z = np.load(p)
    days = z['days']
    umd = pubdata.on_days(ud, uc['UMD'], days)
    for (a, b) in (('2006-01-01', '2025-06-27'), ('2006-01-01', '2019-12-31'), ('2020-01-01', '2025-06-27')):
        D = (days >= pubdata.to_days(np.datetime64(a))) & (days <= pubdata.to_days(np.datetime64(b)))
        for v in ('all', 'big'):
            w = z[f'{v}__wml_d']
            ok = D & np.isfinite(w) & np.isfinite(umd)
            r = np.corrcoef(w[ok], umd[ok])[0, 1]
            tag = '  <- the check (needs >= 0.5)' if (px == 'plain' and v == 'all' and a == '2006-01-01' and b == '2025-06-27') else ''
            P(f'{px:6s} {v:3s} {a}..{b}: daily corr(WML, French UMD) = {r:.3f} over {int(ok.sum())} days{tag}')
    # monthly: our monthly WML vs French's monthly UMD
    md, mc = pubdata.umd_monthly()
    fm = {str(d)[:7]: x for d, x in zip(md, mc['UMD'])}
    mo = z['months']
    for v in ('all', 'big'):
        y = z[f'{v}__wml__gross_m']
        u = np.array([fm.get(m, np.nan) for m in mo])
        ok = (mo >= '2006-01') & (mo <= '2025-06') & np.isfinite(y) & np.isfinite(u)
        P(f'{px:6s} {v:3s} monthly 2006-01..2025-06: corr(WML, French UMD) = {np.corrcoef(y[ok], u[ok])[0, 1]:.3f} over {int(ok.sum())} months')
(HERE / 'fidelity.txt').write_text('\n'.join(lines) + '\n')
