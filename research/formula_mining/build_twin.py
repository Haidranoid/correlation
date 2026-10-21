"""Twin base features as [T, N] float32 panels: residuals against the twin fund over 5, 21 and 63 days
(c2eval.twin_residual, scaled, max_missing = max(1, round(0.05 h))) and the correlation with the twin (point in time,
from the latest twins rebalance at or before each day).

    systemd-run ... python3 -I build_twin.py old|new
"""
import gc
import sys

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/formula_mining')
from common import SRC, c2eval, feat_dir, log


def main(tag):
    exp, _ = SRC[tag]
    E = c2eval.load_export(exp, arrays=('close',))
    days = E['days']
    N, T = len(E['ids']), len(days)
    tw = c2eval.twins(E)
    log(tag, 'twins rebalances', len(tw['rebal']))
    for h in (5, 21, 63):
        mm = max(1, round(0.05 * h))
        x = c2eval.twin_residual(E, tw, h, scaled=True, max_missing=mm)
        x[~np.isfinite(x)] = np.nan
        np.save(feat_dir(tag) / f'twres{h}.npy', np.ascontiguousarray(x.T))
        log(f'twres{h} (max_missing {mm}): finite share on the last day {np.isfinite(x[:, -1]).mean():.2f}')
        del x
        gc.collect()
    kd = np.searchsorted(tw['rebal'], np.arange(T), side='right') - 1
    x = np.full((T, N), np.nan, np.float32)
    ok = kd >= 0
    x[ok] = tw['corr'][kd[ok]]
    x[~np.isfinite(x)] = np.nan
    np.save(feat_dir(tag) / 'twcorr.npy', x)
    log('twcorr done')


if __name__ == '__main__':
    main(sys.argv[1])
