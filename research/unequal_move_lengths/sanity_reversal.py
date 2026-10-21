"""Sanity check of the grading path: 5-day reversal through c2eval.judge (expected +11.8 pts/yr, 8 of 18 years,
long-short 4.39 bps/day, z 2.93)."""
import os
import sys

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_k, '8')
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umlcore as U  # noqa: E402
from umlcore import c2eval  # noqa: E402

e = c2eval.load_export()
c = np.asarray(e['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, dtype=np.float32)
with np.errstate(divide='ignore', invalid='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
s[~np.isfinite(s)] = np.nan
U.OUT.mkdir(parents=True, exist_ok=True)
p = c2eval.save_scores(s, 'unequal_move_lengths/sanity_reversal5')
txt, js = c2eval.judge(p, rebalance=5)
print(txt.splitlines()[-6:])
print(c2eval.summary_line(js))
print(c2eval.change_over_random(js))
