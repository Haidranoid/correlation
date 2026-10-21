"""Fidelity check, run before any test is read: monthly correlation of the portfolio's gross (0 bps) excess return
with AQR's all-asset TSMOM factor (Moskowitz-Ooi-Pedersen 2012 rule on futures). Pass: >= 0.5 over the primary window
2008-01..2025-06 (the plan's test months before the unseen ones). Joined on calendar (year, month): AQR's dates are
normalized to calendar month-ends by pubdata, ours are trading-day month-ends. Writes fidelity.txt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import common as C  # noqa: E402

P = C.build(C.END_MAIN)
ad, av = C.pubdata.aqr_tsmom_monthly()
aqr = C.calendar_month_join(P['label'], ad, av['TSMOM'])
ours = P['overlay_gross']
out = []
for name, a, b in (('primary 2008-01..2025-06', '2008-01', '2025-06'),
                   ('2006-01..2025-06', '2006-01', '2025-06'),
                   ('all overlapping months before the unseen ones, 1994-02..2025-06', '1994-02', '2025-06'),
                   ('2008-01..2019-12', '2008-01', '2019-12'),
                   ('2020-01..2025-06', '2020-01', '2025-06')):
    m = C.window(P, a, b) & np.isfinite(aqr) & (P['n_active'] > 0)
    r = np.corrcoef(ours[m], aqr[m])[0, 1]
    out.append(f'{name}: {m.sum()} months, correlation {r:+.3f}')
m = C.window(P, '2008-01', '2025-06')
ok = np.corrcoef(ours[m], aqr[m])[0, 1] >= 0.5
out.append(f'fidelity check (>= 0.5 over the primary window): {"PASS" if ok else "FAIL"}')
txt = '\n'.join(out)
print(txt)
(C.HERE / 'fidelity.txt').write_text(txt + '\n')
