"""Pilot (the plan's gate): time one year of both curvatures, and B5's 66 candidate covariances a month, before
anything else. Prints timings and graph sizes only, no curvature values and no outcome.

    python -I -B pilot.py        (one process, BLAS threads from the environment)
Writes pilot.txt here.
"""
import sys
import time

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_geometry')
import numpy as np  # noqa: E402

import b_common as B  # noqa: E402

c2eval = B.c2eval
L = []


def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


E = c2eval.load_export(B.EXPORTS['old'], arrays=('close', 'dollar_volume'))
tw = c2eval.twins(E)
excl = c2eval.excluded(E, tw)
days = E['days']
rebal = tw['rebal']
dates = np.array([c2eval.iso(days[t]) for t in rebal])
ks = [k for k in range(len(rebal)) if dates[k].startswith('2008')]
out(f'pilot year 2008: {len(ks)} months ({dates[ks[0]]} .. {dates[ks[-1]]}); also one calm month 2017-06')
k_calm = [k for k in range(len(rebal)) if dates[k].startswith('2017-06')][0]

tot = {'prep': 0.0, 'forman': 0.0, 'oll_primary': 0.0, 'oll_pub': 0.0, 'rewire': 0.0, 'apsp': 0.0}
for k in ks + [k_calm]:
    t0 = int(rebal[k])
    a = time.time()
    sel, R, X, sd, n_elig = B.window(E['close'], E['dollar_volume'], excl[k], t0)
    Z = B.standardize(X, sd)
    C = Z @ Z.T
    n = len(sel)
    i, j = B.top_edges(C)
    b = time.time()
    st = B.graph_measures(n, i, j, C, do_ollivier=False)
    c = time.time()
    Lij = B.lengths(C)[i, j]
    D = B.apsp(n, i, j, Lij)
    c2 = time.time()
    B.ollivier(n, i, j, Lij, D)
    d = time.time()
    pi, pj = B.pub_edges(C)
    Lp = B.lengths(C)[pi, pj]
    e0 = time.time()
    B.ollivier(n, pi, pj, Lp)
    e = time.time()
    ri, rj, rl, info = B.rewire(n, i, j, Lij, seed=B.SEED + 1000 * k + 2)
    f = time.time()
    if k != k_calm:
        tot['prep'] += b - a
        tot['forman'] += c - b
        tot['apsp'] += c2 - c
        tot['oll_primary'] += d - c
        tot['oll_pub'] += e - e0
        tot['rewire'] += f - e
    out(f'{dates[k]} n {n} m {len(i)} max deg {st["deg_max"]} | published graph m {len(pi)} | prep {b - a:.2f}s '
        f'forman+stats {c - b:.2f}s apsp {c2 - c:.2f}s ollivier primary {d - c:.1f}s published {e - e0:.1f}s '
        f'rewire {f - e:.1f}s ({info["rounds"]} rounds, {info["accepted"]} swaps)')

nm = len(ks)
out('\nper month in 2008 (one process):', ' '.join(f'{kk} {v / nm:.1f}s' for kk, v in tot.items()))
months = 270 + 16
oll_main = tot['oll_primary'] / nm
# Ollivier passes on the 5% graph: primary, shift null, factor null, rewired, odd, even, 252-day = 7
est_primary_all = 7 * oll_main * months
est_pub = tot['oll_pub'] / nm * months
out(f'extrapolated at the 2008 rate (worst case), {months} months: 7 Ollivier passes on 5% graphs '
    f'{est_primary_all / 3600:.2f} h serial, {est_primary_all / 3 / 3600:.2f} h on 3 processes; published graph '
    f'{est_pub / 3600:.2f} h serial, {est_pub / 3 / 3600:.2f} h on 3 processes')

# ---------------------------------------------------------------------------------------------------------------
# B5: one month's candidate covariances and their capped minimum-variance solves
cp = B.cp_common()
est = B.cp_estimators()
for kk in ([k for k in range(len(rebal)) if dates[k].startswith('2008-01')][0], k_calm):
    t0 = int(rebal[kk])
    a = time.time()
    uni, big = cp.universe(E, tw, {kk})
    sel = np.nonzero(big[kk])[0]
    R = cp.window_logret(np.asarray(E['close'][sel, t0 - 252:t0 + 1], np.float64))
    okr = np.isfinite(R).sum(1) >= cp.MIN_RET
    sel, R = sel[okr], R[okr]
    S = {}
    for w in (21, 63, 252):
        S[w] = est.ledoit_wolf_cc(cp.demean_fill(R[:, -w:]))[0]
    b = time.time()
    logs = {}
    for w in (21, 63, 252):
        ev, V = np.linalg.eigh(S[w])
        logs[w] = (V * np.log(ev)) @ V.T
    grid = [(a1 / 10, a2 / 10, (10 - a1 - a2) / 10) for a1 in range(11) for a2 in range(11 - a1)]
    c = time.time()
    mats_le = []
    for g in grid:
        Lm = g[0] * logs[21] + g[1] * logs[63] + g[2] * logs[252]
        ev, V = np.linalg.eigh(Lm)
        mats_le.append((V * np.exp(ev)) @ V.T)
    d = time.time()
    w_prev = None
    it_cold, t_cold = [], 0.0
    for g in grid[:6]:
        Sg = g[0] * S[21] + g[1] * S[63] + g[2] * S[252]
        x0 = time.time()
        w, info = est.minvar_capped(Sg, cp.CAP)
        t_cold += time.time() - x0
        it_cold.append(info['pg_iters'])
    x0 = time.time()
    it_warm = []
    for g in grid:
        Sg = g[0] * S[21] + g[1] * S[63] + g[2] * S[252]
        w, info = est.minvar_capped(Sg, cp.CAP, w0=w_prev)
        w_prev = w
        it_warm.append(info['pg_iters'])
    t_euc = time.time() - x0
    x0 = time.time()
    w_prev = None
    for Mx in mats_le:
        w, info = est.minvar_capped(Mx, cp.CAP, w0=w_prev)
        w_prev = w
    t_le = time.time() - x0
    out(f'\nB5 {dates[kk]}: n {len(sel)}; 3 LW fits {b - a:.1f}s (incl. universe); 3 logs {c - b:.2f}s; 66 exps {d - c:.1f}s; '
        f'6 cold solves {t_cold:.1f}s (pg iters {it_cold}); 66 Euclidean solves warm {t_euc:.1f}s '
        f'(median pg iters {int(np.median(it_warm))}); 66 log-Euclidean solves warm {t_le:.1f}s')
    per_month = (b - a) + (c - b) + (d - c) + t_euc + t_le
    out(f'B5 per month about {per_month:.0f}s; about {per_month * 262 / 3600:.1f} h serial for 262 months, '
        f'{per_month * 262 / 3 / 3600:.1f} h on 3 processes')
(B.HERE / 'pilot.txt').write_text('\n'.join(L) + '\n')
