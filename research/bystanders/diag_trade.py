"""Diagnostic (after results): the plan's trade (against the move, 5-day hold, market-relative) on the matched quiet
days, i.e. plain low-volume reversal, next to the bystander trades. Old export."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common, analyze  # noqa: E401,E402
iso = common.c2eval.iso
ev = analyze.load_events('old')
P = common.load_panels('old', names=('U', 'vr', 'x', 'spike', 'close', 'open'))
P = {k: (np.asarray(v) if hasattr(v, 'shape') and v.ndim == 2 else v) for k, v in P.items()}
days = ev['days']
ok, byst, part, insym = analyze.classify(ev)
mis = ev['xB'] - ev['resp'] * ev['xA']
lines = []
for a, b in (('2005-01-01', '2019-12-31'), ('2020-01-01', '2025-06-27')):
    lo = common.c2eval.day_index(days, a); hi = common.c2eval.day_index(days, b) + 1
    inper = (ev['d'] >= lo) & (ev['d'] < hi) & ~ev['expiry']
    rb = analyze.collapse(ev, byst & inper)
    rt = rb[np.isfinite(ev['resp'][rb])]
    me, md = analyze.matched_days(ev, rb, P)
    ym = analyze.fwd_matched(P, ev['B'][me], md, 5)
    xm = P['x'][ev['B'][me], md]
    f = np.isfinite(ym)
    def bps(sig, y):
        g = np.isfinite(sig) & np.isfinite(y) & (sig != 0)
        v = -np.sign(sig[g]) * y[g]
        return f'{v.mean() * 1e4:+.1f} (se {v.std(ddof=1) / np.sqrt(g.sum()) * 1e4:.1f}, n {g.sum():,})'
    lines.append(f'{a[:4]}-{b[:4]}: bystanders vs sympathy {bps(ev["xB"][rb], ev["fB5"][rb])}; twist events vs mismatch '
                 f'{bps(mis[rt], ev["fB5"][rt])}; twist events vs sympathy {bps(ev["xB"][rt], ev["fB5"][rt])}; '
                 f'matched quiet days vs own move {bps(xm[f], ym[f])}')
print('\n'.join(lines))
(analyze.OUT / 'diag_trade_old.txt').write_text('\n'.join(lines) + '\n')
