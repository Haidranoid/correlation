"""Sanity check of the judge wiring through this experiment's own path: 5-day reversal graded with rebalance=5 should
give +11.8 pts/yr, 8 of 18 years beating random at p < 0.1 (experiments/+5.4-reversal-control.md).

    python3 -I -B sanity_reversal5.py      # writes out/sanity_reversal5.txt
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

(c2eval.WORK / 'factor_momentum').mkdir(parents=True, exist_ok=True)
e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
c = np.asarray(e['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, np.float32)
with np.errstate(all='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
del c
p = c2eval.save_scores(s, 'factor_momentum/sanity_reversal5')
del s
txt, js = c2eval.judge(p, rebalance=5)
lines = [str(p), c2eval.summary_line(js), str(c2eval.change_over_random(js))]
print('\n'.join(lines))
(Path(__file__).resolve().parent / 'out' / 'sanity_reversal5.txt').write_text('\n'.join(lines) + '\n')
