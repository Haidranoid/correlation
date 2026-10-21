"""Sanity check of the judge wiring: 5-day reversal graded with rebalance=5 should give +11.8 pts/yr, 8 of 18 years at
p < 0.1 (experiments/+5.4-reversal-control.md). Copied from research/twin_momentum/sanity_reversal5.py.

    python3 -I -B sanity_reversal5.py      # writes out/sanity_reversal5.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
c = np.asarray(e['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, np.float32)
with np.errstate(all='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
del c
(c2eval.WORK / 'decoupling').mkdir(parents=True, exist_ok=True)
p = c2eval.save_scores(s, 'decoupling/sanity_reversal5')
txt, js = c2eval.judge(p, rebalance=5)
lines = [c2eval.summary_line(js), str(c2eval.change_over_random(js))]
print('\n'.join(lines))
(Path(__file__).resolve().parent / 'out' / 'sanity_reversal5.txt').write_text('\n'.join(lines) + '\n')
