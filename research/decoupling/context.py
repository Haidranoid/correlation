"""Decoupling, a diagnostic added after the results (no outcomes): do the events track market-wide falls in
correlation? Per calendar month, the median rho_before, rho_now and Drop over the universe stock-days with a valid
measure, against the number of events that month.

    python3 -I -B context.py old      # writes out/context_{tag}.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import np  # noqa: E402


def main(tag):
    U = dict(np.load(C.WORK / f'univ_{tag}.npz'))
    ev = dict(np.load(C.WORK / f'events_{tag}.npz'))
    P = {p: np.load(C.WORK / f'{p}_{tag}.npy', mmap_mode='r') for p in ('rho_b', 'rho_n', 'S3')}
    days = U['days']
    blk = np.searchsorted(U['rebal'], np.arange(len(days)), side='right') - 1
    mon = C.month_of(days)
    evm = C.month_of(ev['day'])
    rows = []
    for m in np.unique(mon[blk >= 0]):
        dd = np.nonzero((mon == m) & (blk >= 0))[0]
        Um = U['uni'][blk[dd]].T
        rb = np.asarray(P['rho_b'][:, dd], np.float64)
        rn = np.asarray(P['rho_n'][:, dd], np.float64)
        ok = Um & np.isfinite(rb) & np.isfinite(rn)
        if ok.sum() < 1000:
            continue
        rows.append((int(m), float(np.median(rb[ok])), float(np.median(rn[ok])), float(np.median((rb - rn)[ok])),
                     int((evm == m).sum())))
    a = np.array(rows)
    lines = [f'# Decoupling context, {tag} export: monthly medians over universe stock-days with a valid measure', '']
    c = np.corrcoef(a[:, 3], a[:, 4])[0, 1]
    lines.append(f'months {len(a)}; correlation across months of the median Drop with the number of events: {c:+.2f}')
    q = np.quantile(a[:, 3], [0.5, 0.9])
    top = a[:, 3] >= q[1]
    lines.append(f'months in the top 10% of median Drop (>= {q[1]:+.3f}): {int(top.sum())}, holding '
                 f'{int(a[top, 4].sum())} of {int(a[:, 4].sum())} events ({a[top, 4].sum() / a[:, 4].sum():.0%})')
    lines.append('')
    lines.append('month    med rho_before  med rho_now  med Drop  events')
    for r in a[np.argsort(-a[:, 4])][:15]:
        lines.append(f'{str(np.datetime64(int(r[0]), "M"))}   {r[1]:+.2f}          {r[2]:+.2f}        {r[3]:+.3f}    {int(r[4])}')
    lines.append('...')
    lines.append(f'all months: median of the monthly median Drop {np.median(a[:, 3]):+.3f}; of the monthly median '
                 f'rho_before {np.median(a[:, 1]):+.2f} (10th / 90th percentile {np.quantile(a[:, 1], 0.1):+.2f} / '
                 f'{np.quantile(a[:, 1], 0.9):+.2f}); of the monthly median rho_now {np.median(a[:, 2]):+.2f}')
    print('\n'.join(lines))
    (C.OUT / f'context_{tag}.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
