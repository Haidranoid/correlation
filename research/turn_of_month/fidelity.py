"""Fidelity check (added after the walk-forward results; reported only): McConnell & Xu's Table 1 on French's daily
CRSP value-weighted market (Mkt-RF + RF), with this folder's TOM labels and Welch test, 1926-2005 and its
subperiods; then the same market over this experiment's periods (through 2025-06 only; the unseen months are left
to unseen.py). Writes fidelity.txt and fidelity.json."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import np, pubdata  # noqa: E402

out = []
RES = {}


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


dates, f = pubdata.ff3_daily()
days = pubdata.to_days(dates)
raw = f['Mkt-RF'] + f['RF']
exc = f['Mkt-RF']
lab = C.tom_label(days, days)
tom = lab != 0
say(f"French daily {dates[0]}..{dates[-1]}, {len(days)} days; months without a label (calendar's open ends) "
    'are the first and last')
PER = {'M&X Panel A 1987-2005': ('1987-01-01', '2005-12-31'), 'M&X Panel B 1926-1986': ('1926-01-01', '1986-12-31'),
       'M&X Panel C 1926-2005': ('1926-01-01', '2005-12-31'), 'ours 2006-2019': ('2006-01-01', '2019-12-31'),
       'ours 2020-2025-06': ('2020-01-01', '2025-06-30'), 'ours 2006-2025-06': ('2006-01-01', '2025-06-30')}
for name, (lo, hi) in PER.items():
    mm = C.span_mask(days, lo, hi)
    t, o = mm & tom, mm & ~tom
    d, se, tt = C.welch(raw[t], raw[o])
    dx, sex, ttx = C.welch(exc[t], exc[o])
    pos = {p: float(raw[mm & (lab == p)].mean()) for p in C.POSITIONS}
    RES[name] = {'tom_raw': float(raw[t].mean()), 'other_raw': float(raw[o].mean()), 'diff_raw': d, 't_raw': tt,
                 'tom_ex': float(exc[t].mean()), 'other_ex': float(exc[o].mean()), 'diff_ex': dx, 't_ex': ttx,
                 'positions_raw': pos, 'n_tom': int(t.sum()), 'n_other': int(o.sum())}
    say(f'{name}: raw TOM {100 * raw[t].mean():.3f}% vs other {100 * raw[o].mean():.3f}%, diff {100 * d:.3f}% '
        f'(Welch t {tt:.2f}); excess diff {100 * dx:.3f}% (t {ttx:.2f}); positions raw ' +
        ', '.join(f'{p:+d}: {100 * v:.3f}%' for p, v in pos.items()) + f'; n TOM {int(t.sum())}, other {int(o.sum())}')
(C.HERE / 'fidelity.txt').write_text('\n'.join(out) + '\n')
(C.HERE / 'fidelity.json').write_text(json.dumps(RES, indent=1))
