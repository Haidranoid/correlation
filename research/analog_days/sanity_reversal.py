"""Plumbing check: 5-day reversal through this folder's save_scores name + judge (rebalance=5) should give
+11.8 pts/yr, 8 of 18 years (experiments/+5.4-reversal-control.md)."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C
from common import c2eval, np

E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
c = np.asarray(E['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, np.float32)
with np.errstate(all='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
del c
p = c2eval.save_scores(s, 'analog_days/sanity_reversal5')
del s
txt, js = c2eval.judge(p, rebalance=5)
line = c2eval.summary_line(js)
cor = c2eval.change_over_random(js)
print(line)
print(cor)
(C.OUT / 'sanity_reversal.txt').write_text(line + '\n' + str(cor) + '\n')
p.unlink()      # 200 MB; the result is in out/sanity_reversal.txt
