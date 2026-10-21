"""Plumbing check: 5-day reversal through this folder's save_scores + judge path should give +11.8 pts/yr, 8 of 18
years (experiments/+5.4-reversal-control.md)."""
import sys
from pathlib import Path
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
(c2eval.WORK / 'volume_clock').mkdir(parents=True, exist_ok=True)
e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
c = np.asarray(e['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, np.float32)
with np.errstate(all='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
p = c2eval.save_scores(s, 'volume_clock/sanity_reversal5')
txt, js = c2eval.judge(p, rebalance=5)
line = c2eval.summary_line(js)
print(line)
print(c2eval.change_over_random(js))
(HERE / 'sanity_reversal.txt').write_text(line + '\n' + str(c2eval.change_over_random(js)) + '\n')
