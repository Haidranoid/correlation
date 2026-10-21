"""The groups map refit every 21 trading days, built exactly as research/breadth_thrust/bt_refits.py builds it (steps 1,
2 and 4 of that file: universe, groups, labels; the breadth tables are not needed here).

    python3 -I -B tj_refits.py check            # rebuild a few old-export refits; compare with breadth thrust's cache
    python3 -I -B tj_refits.py new 2025-06-01   # the unseen months: refits on the new export from that date (once)

At refit day t0 (c2eval.twins's rebalance days, so excluded(E, tw)[k] belongs to refit k):
  universe: stocks not excluded at k, with a return on >= 95% of days t0-251..t0, ranked by median dollar volume over
  those days; the top 1,500 (breadth thrust's decision: no $5M floor); refits with fewer than 1,500 are skipped;
  groups: simple daily returns over the 252 days, market-relative, correlation, spectral clustering into 30 groups,
  seed 0x6E0 + t0; label = the most common twin fund among the members (reporting only).
Writes BIG/new/refits.pkl (keys as breadth thrust's, minus its breadth tables).
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from tj_common import BT_REFITS_NEW, BT_REFITS_OLD, EXPORTS, K_GROUPS, big, c2eval, iso  # noqa: E402
from tj_groups import market_relative, spectral, standardize  # noqa: E402

CORR_WIN, N_STOCKS, COVER, MIN_DV, SEED = 252, 1500, 0.95, 0.0, 0x6E0   # bt_common.py's values


def build(tag, ks_wanted=None, first=None):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    tw = c2eval.twins(E)
    excl = c2eval.excluded(E, tw)
    fund_ticker = np.asarray(tw['fund_ticker'])
    del tw['F']
    close = np.asarray(E['close'], np.float64)
    N, T = close.shape
    ret = np.full((N, T), np.nan, np.float32)
    with np.errstate(all='ignore'):
        ok = np.isfinite(close[:, 1:]) & np.isfinite(close[:, :-1]) & (close[:, :-1] > 0)
        ret[:, 1:] = np.where(ok, close[:, 1:] / close[:, :-1] - 1, np.nan)
    del ok, close
    dv = E['dollar_volume']
    rebal = tw['rebal']
    ks = list(range(len(rebal)))
    if first is not None:
        d0 = c2eval.day_index(days, first)
        ks = [k for k in ks if rebal[k] >= d0]
    if ks_wanted is not None:
        ks = [k for k in ks if k in ks_wanted]
    out = []
    for k in ks:
        t0 = int(rebal[k])
        t1 = int(rebal[k + 1]) if k + 1 < len(rebal) else T
        a, b = t0 - CORR_WIN + 1, t0 + 1
        R = ret[:, a:b].astype(np.float64)
        cover = np.isfinite(R).mean(1)
        with np.errstate(all='ignore'):
            mdv = np.nanmedian(np.asarray(dv[:, a:b], np.float64), 1)
        cand = np.flatnonzero(~excl[k] & (cover >= COVER) & np.isfinite(mdv) & (mdv > MIN_DV))
        U = cand[np.argsort(-mdv[cand], kind='stable')][:N_STOCKS]
        if len(U) < N_STOCKS:
            print(f'  k={k} {iso(days[t0])} universe {len(U)} < {N_STOCKS}: skipped', flush=True)
            continue
        rel, _ = market_relative(R[U])
        z = standardize(rel)
        lab = spectral(z @ z.T, K_GROUPS, SEED + t0)
        sizes = np.bincount(lab, minlength=K_GROUPS)
        fr = tw['fund_row'][k, U]
        glabel = []
        for g in range(K_GROUPS):
            f = fr[(lab == g) & (fr >= 0)]
            glabel.append(str(fund_ticker[np.bincount(f).argmax()]) if len(f) else '-')
        out.append(dict(k=k, t0=t0, t1=t1, U=U, lab=lab, sizes=sizes, glabel=glabel))
        print(f'  k={k} {iso(days[t0])} universe {len(U)} sizes {sizes.min()}-{sizes.max()}', flush=True)
    return dict(days=days, T=T, N=N, refits=out)


def same(a, b):
    return (a['t0'] == b['t0'] and a['t1'] == b['t1'] and np.array_equal(a['U'], b['U'])
            and np.array_equal(a['lab'], b['lab']) and list(a['glabel']) == list(b['glabel']))


def compare(mine, cached_path):
    with open(cached_path, 'rb') as fh:
        C = pickle.load(fh)
    ck = {r['k']: r for r in C['refits']}
    n_same = sum(same(r, ck[r['k']]) for r in mine['refits'] if r['k'] in ck)
    n_both = sum(r['k'] in ck for r in mine['refits'])
    print(f'identical to breadth thrust\'s cache ({cached_path}): {n_same} of {n_both} refits in both', flush=True)


if __name__ == '__main__':
    t_start = time.time()
    if sys.argv[1] == 'check':
        mine = build('old', ks_wanted={38, 100, 200, 269})
        compare(mine, BT_REFITS_OLD)
    else:
        tag, first = sys.argv[1], sys.argv[2]
        mine = build(tag, first=first)
        with open(big(tag) / 'refits.pkl', 'wb') as fh:
            pickle.dump(mine, fh)
        compare(mine, BT_REFITS_NEW)
    print(f'done {time.time() - t_start:.0f}s')
