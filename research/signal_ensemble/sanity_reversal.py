"""Plumbing check: 5-day reversal through this folder's save_scores + judge path should give +11.8 pts/yr, 8 of 18
years (experiments/+5.4-reversal-control.md). Copied from research/betting_against_correlation/sanity_reversal.py.

    python3 -I sanity_reversal.py    -> sanity_reversal.txt
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import c2eval, np, HERE, wdir

wdir('old')
e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
c = np.asarray(e['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, np.float32)
with np.errstate(all='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
del c
p = c2eval.save_scores(s, 'signal_ensemble/old/sanity_reversal5')
txt, js = c2eval.judge(p, rebalance=5)
line = c2eval.summary_line(js)
cor = c2eval.change_over_random(js)
print(line)
print(cor)
(HERE / 'sanity_reversal.txt').write_text(line + '\n' + str(cor) + '\n')
