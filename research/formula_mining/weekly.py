"""Weekly cross-sections: on each decision day of the fixed 5-trading-day calendar, the universe's stocks, their 26
base-feature percentile ranks, and the outcome (next 5-day log return minus the export's market over the same days).

    systemd-run ... python3 -I weekly.py old|new

Output: BIG/weekly_<tag>.npz with dates [W] (export day indices), offs [W+1], idx [sum n] (stock rows),
B [NF, sum n] float32 ranks, y [sum n] float32 outcomes.
"""
import sys

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/formula_mining')
from common import BIG, FEATURES, SRC, WEEK_START, c2eval, log, open_panel, rank_features, week_dates  # noqa: E402


def main(tag):
    exp, _ = SRC[tag]
    E = c2eval.load_export(exp, arrays=('close', 'market'))
    days = E['days']
    close = np.asarray(E['close'], np.float64)
    mkt = np.asarray(E['market'], np.float64)
    U = np.load(BIG / f'universe_{tag}.npy', mmap_mode='r')
    panels = [open_panel(tag, f) for f in FEATURES]
    dates = week_dates(days, WEEK_START[tag])
    offs, idxs, Bs, ys = [0], [], [], []
    for k, d in enumerate(dates):
        idx = np.nonzero(U[d])[0]
        P = np.stack([np.asarray(p[d, idx], np.float64) for p in panels])
        B = rank_features(P).astype(np.float32)
        with np.errstate(all='ignore'):
            y = np.log(close[idx, d + 5] / close[idx, d]) - mkt[d + 1:d + 6].sum()
        y[~np.isfinite(y)] = np.nan
        idxs.append(idx.astype(np.int32)); Bs.append(B); ys.append(y.astype(np.float32))
        offs.append(offs[-1] + len(idx))
        if k % 100 == 0:
            cov = ' '.join(f'{f}:{np.isfinite(B[i]).mean():.2f}' for i, f in enumerate(FEATURES))
            log(c2eval.iso(days[d]), 'n', len(idx), 'y finite', f'{np.isfinite(y).mean():.3f}', cov)
    np.savez(BIG / f'weekly_{tag}.npz', dates=dates, offs=np.array(offs), idx=np.concatenate(idxs),
             B=np.concatenate(Bs, 1), y=np.concatenate(ys))
    log('weeks', len(dates), 'first', c2eval.iso(days[dates[0]]), 'last', c2eval.iso(days[dates[-1]]),
        'stocks per week median', int(np.median(np.diff(offs))))


if __name__ == '__main__':
    main(sys.argv[1])
