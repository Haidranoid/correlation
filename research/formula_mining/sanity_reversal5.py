"""Sanity check of the judge wiring: 5-day reversal graded with rebalance=5 should give +11.8 pts/yr, 8 of 18 years."""
import sys
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
c = np.asarray(e['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, np.float32)
with np.errstate(all='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
del c
p = c2eval.save_scores(s, 'formula_mining/sanity_reversal5')
txt, js = c2eval.judge(p, rebalance=5)
print(c2eval.summary_line(js))
print(c2eval.change_over_random(js))
