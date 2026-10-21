"""Pipeline check: 5-day reversal written as a scores file and graded by the shared judge (expect +11.8 pts/yr)."""
import os, sys, time
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(v, '8')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np
import c2eval

ex = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
c = np.asarray(ex['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, np.float32)
with np.errstate(divide='ignore', invalid='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
s[~np.isfinite(s)] = np.nan
p = c2eval.save_scores(s, 'v1_pair_finder/sanity_reversal5')
t = time.time()
txt, js = c2eval.judge(p, rebalance=5)
print(c2eval.summary_line(js))
print(f'judge {time.time() - t:.0f} s')
