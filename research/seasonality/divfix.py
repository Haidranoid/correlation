"""Monthly dividend correction: cf[j, i] = sum of pubdata.div_fix(tag)[j, d] over the export days of holding month i
(after e[i-1] through e[i]); 0 before 2022-07 (the shared panel is 0 there). Added to the monthly log returns by
book.py fix and money.py ... fix.

    python3 -I -B divfix.py old|new      -> WORK/divfix_<tag>.npy
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import np, WORK, log, month_label  # noqa: E402
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
import pubdata  # noqa: E402

tag = sys.argv[1]
z = np.load(WORK / f'panel_{tag}.npz')
e, months = z['e'], z['months']
D = pubdata.div_fix(tag)
N, T = D.shape
assert N == z['r'].shape[0] and T == len(z['days']), (D.shape, z['r'].shape, len(z['days']))
cs = np.zeros((N, T + 1))
np.cumsum(np.nan_to_num(np.asarray(D, np.float64)), axis=1, out=cs[:, 1:])
M = len(months)
cf = np.zeros((N, M))
for i in range(1, M):
    cf[:, i] = cs[:, e[i] + 1] - cs[:, e[i - 1] + 1]
np.save(WORK / f'divfix_{tag}.npy', cf.astype(np.float32))
nz = cf != 0
first = np.nonzero(nz.any(0))[0]
log(f'{tag}: stock-months with a correction {int(nz.sum())}, first month {month_label(months[first[0]])}; '
    f'mean correction where nonzero {100 * cf[nz].mean():.3f}%; per month (post-2022-07) stocks corrected: median '
    f'{int(np.median(nz[:, first[0]:].sum(0)))}')
