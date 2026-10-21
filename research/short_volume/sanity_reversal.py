"""Short volume: the plumbing check. 5-day reversal through this folder's path, rebalance=5.
Must give +11.8 pts/yr, ahead in 8 of 18 years (experiments/+5.4-reversal-control.md).

    python3 -I -B sanity_reversal.py   # writes out/sanity_reversal5.txt
"""
import sys
sys.dont_write_bytecode = True
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np  # noqa: E402
import c2eval  # noqa: E402

(c2eval.WORK / 'short_volume').mkdir(parents=True, exist_ok=True)
e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
c = np.asarray(e['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, np.float32)
with np.errstate(all='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
del c
p = c2eval.save_scores(s, 'short_volume/sanity_reversal5')
txt, js = c2eval.judge(p, rebalance=5)
out = c2eval.summary_line(js) + '\n' + json.dumps(c2eval.change_over_random(js)) + '\n'
(HERE / 'out' / 'sanity_reversal5.txt').write_text(out)
print(out)
