"""The universe per day, [T, N] bool (False before UNIV_START): the judge-like liquid set minus c2eval.excluded.
Liquid at day d: a finite close on d, >= 90% finite closes over d-251..d, median close >= $5 and median dollar volume
>= $5M over those days (the judge's liquid rule, point in time, as research/free_timing and silent_peer_drift).

    systemd-run ... python3 -I build_universe.py old|new
"""
import sys
import warnings

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/formula_mining')
from common import BIG, SRC, UNIV_START, c2eval, log


def main(tag):
    exp, _ = SRC[tag]
    E = c2eval.load_export(exp, arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = len(E['ids']), len(days)
    tw = c2eval.twins(E)
    exc = c2eval.excluded(E, tw)
    close = np.asarray(E['close'], np.float32)
    dv = np.asarray(E['dollar_volume'], np.float32)
    U = np.zeros((T, N), bool)
    d0 = c2eval.day_index(days, UNIV_START[tag])
    for d in range(d0, T):
        c = close[:, d - 251:d + 1]
        fin = np.isfinite(c).sum(1)
        cand = np.nonzero((fin * 10 >= 252 * 9) & np.isfinite(close[:, d]))[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mp = np.nanmedian(c[cand], 1)
            mv = np.nanmedian(dv[cand, d - 251:d + 1], 1)
        ok = cand[(np.nan_to_num(mp) >= 5.0) & (np.nan_to_num(mv) >= 5e6)]
        k = c2eval.rebal_index(tw, d)
        if k >= 0:
            ok = ok[~exc[k, ok]]
        U[d, ok] = True
        if (d - d0) % 500 == 0:
            log(c2eval.iso(days[d]), 'universe', len(ok))
    np.save(BIG / f'universe_{tag}.npy', U)
    n = U[d0:].sum(1)
    log('universe size: median', int(np.median(n)), 'min', int(n.min()), 'max', int(n.max()))


if __name__ == '__main__':
    main(sys.argv[1])
