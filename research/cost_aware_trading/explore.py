"""Orientation only: shapes, coverage and ties of the score files (no outcomes read)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np  # noqa: E402
import c2eval  # noqa: E402

SA = Path('/home/bhoult/programming/correlation/stockai')
E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
days = E['days']
print('export', E['close'].shape, c2eval.iso(days[0]), c2eval.iso(days[-1]))
for name in ['runs/1w/lgbm/scores.npy', 'runs/1w/jitter/lgbm_s4.npy', 'runs/1w/rules/reversal.npy',
             'runs/poc/baselines/lgbm_ret/scores.npy']:
    s = np.load(SA / name, mmap_mode='r')
    fin = np.isfinite(s).any(axis=0)
    cols = np.flatnonzero(fin)
    print(name, s.dtype, s.shape, 'first', c2eval.iso(days[cols[0]]), 'last', c2eval.iso(days[cols[-1]]),
          'ncols', len(cols))
    # per-day finite counts at a few days, ties in the top 30
    for d in cols[[0, len(cols) // 3, 2 * len(cols) // 3, -1]]:
        col = np.asarray(s[:, d], dtype=np.float64)
        f = col[np.isfinite(col)]
        top = np.sort(f)[::-1][:30]
        print('   ', c2eval.iso(days[d]), 'finite', f.size, 'pos', (f > 0).sum(), 'distinct in top30',
              len(np.unique(top)), 'top10 min', top[9])
