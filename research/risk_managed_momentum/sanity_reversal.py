"""Plumbing check: 5-day reversal through this folder's save_scores + judge path should give +11.8 pts/yr, 8 of 18
years beating random at p < 0.1 (experiments/+5.4-reversal-control.md). Copied from research/betting_against_beta.

    python3 -I -B sanity_reversal.py   -> sanity_reversal.txt
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
c = np.asarray(e['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, np.float32)
with np.errstate(all='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
del c
p = c2eval.save_scores(s, 'risk_managed_momentum/sanity_reversal5')
del s
txt, js = c2eval.judge(p, rebalance=5)
line = c2eval.summary_line(js)
cor = c2eval.change_over_random(js)
print(line)
print(cor)
(HERE / 'sanity_reversal.txt').write_text(line + '\n' + str(cor) + '\n')
