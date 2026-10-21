"""Reported only (added after the unseen months were seen): how often a 16-month window inside the walk-forward span
2006-01..2025-06 shows a SPY TOM-minus-other excess gap, or a rule-minus-median-random gap, as large as the unseen
months' (+0.175 % a day, Welch t +1.69). Rolling start months, so the windows overlap. Writes context16.txt."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np, pubdata  # noqa: E402

Eo = c2eval.load_export(c2eval.EXPORT_OLD, arrays=())
En = c2eval.load_export(c2eval.EXPORT_NEW, arrays=())
do, dn = Eo['days'], En['days']
sp = C.span_mask(do, *C.SPAN)
days = do[sp]
lab = C.tom_label(do, dn)[sp]
r = C.spy_total(do)[0][sp]
ex = r - pubdata.rf_daily(do)[sp]
y, m = C.ymd(days)
mk = y * 12 + m
months = np.unique(mk)
diffs, ts = [], []
for i in range(len(months) - 15):
    w = (mk >= months[i]) & (mk <= months[i + 15])
    t, o = w & (lab != 0), w & (lab == 0)
    d, se, tt = C.welch(ex[t], ex[o])
    diffs.append(d)
    ts.append(tt)
diffs, ts = np.array(diffs), np.array(ts)
lines = [f'{len(diffs)} rolling 16-month windows, 2006-01..2025-06',
         f'diff: median {100 * np.median(diffs):+.3f}%, 5th..95th pct {100 * np.percentile(diffs, 5):+.3f}%..'
         f'{100 * np.percentile(diffs, 95):+.3f}%, max {100 * diffs.max():+.3f}%',
         f'share with diff >= +0.175%: {np.mean(diffs >= 0.00175):.3f}; share with Welch t >= 1.69: {np.mean(ts >= 1.69):.3f}',
         f'share with t <= -1.69: {np.mean(ts <= -1.69):.3f}']
print('\n'.join(lines))
(C.HERE / 'context16.txt').write_text('\n'.join(lines) + '\n')
