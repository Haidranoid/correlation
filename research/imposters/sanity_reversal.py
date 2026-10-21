"""Sanity check that the judge is used right: 5-day reversal should give +11.8 pts/yr, 8 of 18 years, long-short
4.39 bps/day, z 2.93 (experiments/+5.4-reversal-control.md)."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import c2eval, np

e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
c = np.asarray(e['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, np.float32)
with np.errstate(all='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
p = c2eval.save_scores(s, 'imposters/sanity_reversal5')
txt, js = c2eval.judge(p, rebalance=5)
print(c2eval.summary_line(js))
print(c2eval.change_over_random(js))
