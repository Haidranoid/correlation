"""B's curvature series: every calendar month, the primary 5% graph and the published-style graph, both curvatures,
the plain statistics, group curvature, and the four checks (shift null, factor null, rewiring, odd/even and 252-day).
No return after t0 is touched here.

    python -I -B curvature.py old|new
Writes WORK/curv_<tag>.npz (per-month series), WORK/edges_<tag>.pkl (per-edge primary-graph curvatures),
WORK/groups_<tag>.pkl (group curvature), curvature_<tag>.txt here (checks and summaries).
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'          # 3 worker processes x 1 BLAS thread = the 3-core budget
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_geometry')
import numpy as np  # noqa: E402

import b_common as B  # noqa: E402

c2eval = B.c2eval
TAG = sys.argv[1]
assert TAG in ('old', 'new')
NPROC = 3
VARIANTS = ('main', 'pub', 'shift', 'factor', 'rewire', 'odd', 'even', 'long')
G = {}


def setup():
    E = c2eval.load_export(B.EXPORTS[TAG], arrays=('close', 'dollar_volume', 'market'))
    tw = c2eval.twins(E)
    G['E'] = E
    G['tw'] = tw
    G['excl'] = c2eval.excluded(E, tw)
    G['market'] = np.asarray(E['market'], np.float64)
    G['fclose'] = tw['F']['close']


def fund_window(rows, t0):
    c = np.asarray(G['fclose'][rows, t0 - B.WIN:t0 + 1], np.float64)
    with np.errstate(all='ignore'):
        r = np.log(c[:, 1:] / c[:, :-1])
    r[~np.isfinite(r) | (np.abs(r) > 0.5)] = np.nan
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = np.nanmean(r, 1, keepdims=True)
    X = r - m
    X[~np.isfinite(X)] = 0.0
    return X


def rownorm(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm


def month(k):
    warnings.simplefilter('ignore')
    E, tw = G['E'], G['tw']
    t0 = int(tw['rebal'][k])
    sel, R, X, sd, n_elig = B.window(E['close'], E['dollar_volume'], G['excl'][k], t0)
    n = len(sel)
    res = {'k': k, 't0': t0, 'n': n, 'n_elig': n_elig}
    Z = B.standardize(X, sd)
    C = Z @ Z.T
    res['avg_corr'], res['absorption'] = B.avg_corr_absorption(X, Z)
    i, j = B.top_edges(C)
    main = B.graph_measures(n, i, j, C, keep_edges=True)
    edges = main.pop('edges')
    res['main'] = main
    pi, pj = B.pub_edges(C)
    res['pub'] = B.graph_measures(n, pi, pj, C)
    # shift null: each stock's window returns circularly shifted by its own 1..62 days
    rng = np.random.default_rng(B.SEED + 1000 * k)
    s = rng.integers(1, B.WIN, size=n)
    Xs = np.stack([np.roll(X[a], s[a]) for a in range(n)])
    Zs = rownorm(Xs)
    Cs = Zs @ Zs.T
    si, sj = B.top_edges(Cs)
    res['shift'] = B.graph_measures(n, si, sj, Cs)
    # factor null: market and twin-fund fit + the residual shifted by 1..62 days
    mk = G['market'][t0 - B.WIN + 1:t0 + 1]
    if np.isfinite(mk).all():
        rng = np.random.default_rng(B.SEED + 1000 * k + 1)
        s = rng.integers(1, B.WIN, size=n)
        md = mk - mk.mean()
        fr = tw['fund_row'][k, sel]
        has = fr >= 0
        Fx = np.zeros((n, B.WIN))
        if has.any():
            Fx[has] = fund_window(fr[has], t0)
        Xf = np.empty_like(X)
        for a in range(n):
            D = md[:, None] if not has[a] else np.column_stack([md, Fx[a]])
            beta = np.linalg.lstsq(D, X[a], rcond=None)[0]
            fit = D @ beta
            Xf[a] = fit + np.roll(X[a] - fit, s[a])
        Zf = rownorm(Xf)
        Cf = Zf @ Zf.T
        fi, fj = B.top_edges(Cf)
        res['factor'] = B.graph_measures(n, fi, fj, Cf)
        res['factor']['n_twin'] = int(has.sum())
    else:
        res['factor'] = None
    # rewiring: degree-preserving double-edge swaps of the real graph, lengths travel with the swap
    Lij = B.lengths(C)[i, j]
    ri, rj, rl, info = B.rewire(n, i, j, Lij, seed=B.SEED + 1000 * k + 2)
    A = B.adjacency(n, ri, rj)
    deg = A.sum(1).astype(np.float64)
    t = B.triangles_per_edge(A, ri, rj)
    rw = B.plain_stats(n, ri, rj, A, t, C[ri, rj])
    rw['forman'] = float(B.forman_aug(deg, ri, rj, t).mean())
    rw['forman_closed'] = float(4.0 - (deg ** 2).sum() / len(ri) + 9.0 * rw['T'] / len(ri))
    rw['forman_samal'] = float(B.forman_samal(n, ri, rj, rl).mean())
    rw['ollivier'] = float(B.ollivier(n, ri, rj, rl).mean())
    rw['rounds'] = info['rounds']
    rw['accepted'] = info['accepted']
    rw['deg_equal'] = bool(np.array_equal(np.sort(deg), np.sort(B.adjacency(n, i, j).sum(1).astype(np.float64))) and
                           np.array_equal(deg, B.adjacency(n, i, j).sum(1).astype(np.float64)))
    res['rewire'] = rw
    # odd / even days of the window
    for nm, cols in (('even', slice(0, None, 2)), ('odd', slice(1, None, 2))):
        Zh = B.standardize_subset(R[:, cols])
        Ch = Zh @ Zh.T
        hi, hj = B.top_edges(Ch)
        res[nm] = B.graph_measures(n, hi, hj, Ch)
    # 252-day window, same selection rule
    sel2, R2, X2, sd2, _ = B.window(E['close'], E['dollar_volume'], G['excl'][k], t0, win=B.WIN_LONG)
    Z2 = B.standardize(X2, sd2)
    C2 = Z2 @ Z2.T
    li, lj = B.top_edges(C2)
    res['long'] = B.graph_measures(len(sel2), li, lj, C2)
    # groups: stocks of the month sharing a twin fund; internal edges of the primary graph
    fr = tw['fund_row'][k, sel]
    ge = np.where((fr[i] == fr[j]) & (fr[i] >= 0), fr[i], -1)
    groups = {}
    for f in np.unique(ge[ge >= 0]):
        e = ge == f
        ne = int(e.sum())
        if ne < 10:
            continue
        mem = np.nonzero(fr == f)[0]
        sz = len(mem)
        sg = Z[mem].sum(0)
        groups[int(f)] = {'n_edges': ne, 'size': sz, 'rho_pairs': float((sg @ sg - sz) / (sz * (sz - 1))),
                          'ollivier': float(edges['ollivier'][e].mean()), 'forman': float(edges['forman'][e].mean())}
    edges['sel'] = sel.astype(np.int32)
    return res, edges, groups


def main():
    setup()
    E, tw = G['E'], G['tw']
    days = E['days']
    rebal = tw['rebal']
    K = len(rebal)
    B.log(TAG, 'export', E['close'].shape, 'months', K, c2eval.iso(days[rebal[0]]), '..', c2eval.iso(days[rebal[-1]]))
    out = [None] * K
    edges = [None] * K
    groups = [None] * K
    ctx = mp.get_context('fork')
    with ctx.Pool(NPROC) as pool:
        for res, ed, gr in pool.imap(month, range(K), chunksize=1):
            k = res['k']
            out[k], edges[k], groups[k] = res, ed, gr
            if k % 20 == 0:
                B.log(f"{c2eval.iso(days[res['t0']])} n {res['n']} rewire rounds {res['rewire']['rounds']} "
                      f"pub m {res['pub']['m']} groups {len(gr)}")
    # flatten to arrays
    arr = {'rebal': np.asarray(rebal), 'days': np.asarray(days), 'n': np.array([o['n'] for o in out]),
           'n_elig': np.array([o['n_elig'] for o in out]),
           'avg_corr': np.array([o['avg_corr'] for o in out]), 'absorption': np.array([o['absorption'] for o in out])}
    keys = ('m', 'degconc', 'tridens', 'rho_edge', 'lcc', 'T', 'deg_max', 'forman', 'forman_closed', 'forman_samal',
            'ollivier')
    for v in VARIANTS:
        for key in keys:
            arr[f'{v}_{key}'] = np.array([np.nan if o[v] is None else float(o[v][key]) for o in out])
    arr['rewire_rounds'] = np.array([o['rewire']['rounds'] for o in out])
    arr['rewire_accepted'] = np.array([o['rewire']['accepted'] for o in out])
    arr['rewire_deg_equal'] = np.array([o['rewire']['deg_equal'] for o in out])
    np.savez(B.WORK / f'curv_{TAG}.npz', **arr)
    with open(B.WORK / f'edges_{TAG}.pkl', 'wb') as f:
        pickle.dump(edges, f, protocol=4)
    with open(B.WORK / f'groups_{TAG}.pkl', 'wb') as f:
        pickle.dump(groups, f, protocol=4)
    B.log('saved')
    report(arr, groups)


def report(a, groups):
    L = []
    p = L.append
    days, rebal = a['days'], a['rebal']
    st = np.load(B.mt_common().WORK / f'states_{TAG}.npz')
    d1 = np.nanmax(np.abs(st['avg_corr'] - a['avg_corr']))
    d2 = np.nanmax(np.abs(st['absorption'] - a['absorption']))
    p(f'Universe check against research/market_timing states_{TAG}.npz: max |avg corr diff| {d1:.2e}, '
      f'max |absorption diff| {d2:.2e}, universe sizes equal: {bool(np.array_equal(st["n_sel"], a["n"]))}')
    for v in VARIANTS:
        dd = np.nanmax(np.abs(a[f'{v}_forman'] - a[f'{v}_forman_closed']))
        p(f'Forman closed form ({v}): max |edge mean - (4 - sum d^2/m + 9T/m)| = {dd:.2e}')
    p(f"Rewiring: rounds min {a['rewire_rounds'].min()} median {int(np.median(a['rewire_rounds']))} max "
      f"{a['rewire_rounds'].max()}; every degree sequence kept: {bool(a['rewire_deg_equal'].all())}")
    p(f"Months {len(rebal)}: {c2eval.iso(days[rebal[0]])} .. {c2eval.iso(days[rebal[-1]])}; n min {a['n'].min()}; "
      f"primary m {int(np.nanmin(a['main_m']))}..{int(np.nanmax(a['main_m']))}; published m "
      f"{int(np.nanmin(a['pub_m']))}..{int(np.nanmax(a['pub_m']))}")
    ng = np.array([len(g) for g in groups])
    p(f'Groups with >= 10 internal edges per month: min {ng.min()} median {int(np.median(ng))} max {ng.max()}')
    txt = '\n'.join(L)
    print(txt)
    (B.HERE / f'curvature_{TAG}.txt').write_text(txt + '\n')


if __name__ == '__main__':
    main()
