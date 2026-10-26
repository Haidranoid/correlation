"""Gravity: the pilot gate. Times one year and 50 stocks of G3's heaviest candidates (N = 21, tau = 1..41) and of
G2's attractor selection, then extrapolates to the full run at the machine's load. It decides only how the run is done.

    python -I -B pilot.py

Rule (fixed before the pilot ran): every stage is computed in chunks of one refit (21 days) and never materializes the
lagged spreads. G3 runs as plain numpy if its extrapolated time for both worlds is under 30 minutes, else with the
numba kernel (kern.g3_accum). If the numba extrapolation exceeds 12 hours, G3 is reported as not run.
The numpy version is also compared with the numba kernel on the sample (they must agree to float32 precision).
Writes out/pilot.txt / .json.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import forces  # noqa: E402
import peers  # noqa: E402
np = K.np


def g3_numpy_pair(lnC, lnH, lnL, b, a, beta, m, ta, tb, Wk, eps, ps):
    """Plain numpy G3 for one attractor: returns acc [nd, nk, np, 2] (already times -m) and wsum [nd, nk]."""
    from numpy.lib.stride_tricks import sliding_window_view as swv
    nd = tb - ta
    nk = Wk.shape[0]
    acc = np.zeros((nd, nk, len(ps), 2))
    wsum = np.zeros((nd, nk))
    tt = np.arange(ta - 63, tb)
    for tau in range(1, Wk.shape[1]):
        if not (Wk[:, tau] > 0).any():
            continue
        x = lnC[b, tt] - beta * lnC[a, tt - tau]
        win = swv(x, 63)[:nd]
        n = np.isfinite(win).sum(1)
        with np.errstate(all='ignore'):
            mu = np.nanmean(win, 1)
            sd = np.nanstd(win, 1, ddof=1)
        d = np.arange(ta, tb)
        base = beta * lnC[a, d - tau] + mu
        with np.errstate(all='ignore'):
            gc = (lnC[b, d] - base) / sd
            gh = (lnH[b, d] - base) / sd
            gl = (lnL[b, d] - base) / sd
        ok = (n >= 57) & (sd > 0) & np.isfinite(gc) & np.isfinite(gh) & np.isfinite(gl)
        r = np.where((gl <= 0) & (gh >= 0), 0.0, np.minimum(np.abs(gl), np.abs(gh)))
        sg = np.sign(gc)
        for q in range(nk):
            w = Wk[q, tau]
            if w == 0:
                continue
            wsum[ok, q] += w
            for pi, p in enumerate(ps):
                acc[ok, q, pi, 0] += w * sg[ok] / (r[ok] + eps) ** p
                acc[ok, q, pi, 1] += w * sg[ok] / (np.abs(gc[ok]) + eps) ** p
    return -m * acc, wsum


def main():
    fh = open(K.OUT / 'pilot.txt', 'w')
    load0 = os.getloadavg()
    K.log(f'pilot: load average {load0[0]:.1f} / {load0[1]:.1f} / {load0[2]:.1f} on {os.cpu_count()} cores; 4 threads', fh)
    P, _ = forces.load_inputs('old') if (K.BIG / 'peers_old.npz').exists() else (None, None)
    if P is None:
        import data
        Pp = data.load_panel('old')
        P = {k: Pp[k] for k in ('days', 'refit', 'uni', 'inU', 'meta')}
        for k in ('lnC', 'lnH', 'lnL', 'z'):
            P[k] = np.ascontiguousarray(np.asarray(Pp[k], np.float64))
        P.update({'T': Pp['T'], 'd0': Pp['d0'], 'u0': Pp['u0'], 'f0': Pp['f0'], 'ids': np.array(Pp['meta']['ids'])})
    days = P['days']
    ref_all = forces.all_refits(P)
    y0 = K.day_of(days, '2012-01-01')
    year = [k for k in ref_all if K.day_of(days, '2012-01-01') <= int(P['refit'][k]) < K.day_of(days, '2013-01-01')]
    K.log(f'one year: {len(year)} refits ({K.iso(days, int(P["refit"][year[0]]))}..{K.iso(days, int(P["refit"][year[-1]]))}); '
          f'the full run has {len(ref_all)}', fh)
    res = {'load': load0, 'refits_year': len(year), 'refits_full': len(ref_all)}

    # G2's attractor selection (and G1's), the full universe for one year of refits
    t = time.time()
    PE = peers.compute(P, refits=year)
    tp = time.time() - t
    res['peers_year_s'] = tp
    res['peers_full_min'] = tp * len(ref_all) / len(year) / 60
    K.log(f'attractor selection (G1 + random + 18 G2 kernels, 500 stocks): {tp:.1f}s for the year -> full run '
          f'{res["peers_full_min"]:.1f} min', fh)

    # G3, 50 stocks, N = 21 (and N = 5, which shares the pass), real attractors
    rng = np.random.default_rng(K.SEED)
    u = P['uni'][year[0]]
    b50 = np.sort(rng.choice(u[u >= 0], 50, replace=False))
    t = time.time()
    forces.g3(P, PE, year[:1], brows=b50[:2], worlds=('real',))
    tc = time.time() - t
    t = time.time()
    F = forces.g3(P, PE, year, brows=b50, worlds=('real',))
    tn = time.time() - t
    # stocks of b50 that are in each refit's universe
    n_in = np.mean([np.isin(b50, P['uni'][k]).sum() for k in year])
    scale = (K.TOP / n_in) * (len(ref_all) / len(year)) * 2       # all stocks, all refits, two worlds
    res.update({'numba_compile_s': tc, 'numba_g3_sample_s': tn, 'stocks_in_universe_mean': float(n_in),
                'numba_g3_full_min': tn * scale / 60})
    K.log(f'G3 numba: compile+first refit {tc:.1f}s; one year x {n_in:.0f} stocks {tn:.1f}s -> full run (both worlds) '
          f'{res["numba_g3_full_min"]:.1f} min', fh)

    # the same with plain numpy, on 5 stocks x 2 refits, and the comparison with numba
    Wk = forces.g3_weights()
    ps = np.array([1.0, 2.0])
    t = time.time()
    worst = 0.0
    n_pairs = 0
    nchk = 0
    for k in year[:2]:
        ta, tb = forces._span(P, k)
        sel, b = forces._rows(P, k, b50[:5])
        att, mass, beta = forces._att(PE, 'real', k)
        for i, bb in zip(sel, b):
            accs = np.zeros((tb - ta, len(K.G3_KERN), 2, 2))
            cnt = np.zeros((tb - ta, len(K.G3_KERN)))
            for s in range(K.K_ATT):
                if att[i, s] < 0:
                    continue
                acc, ws = g3_numpy_pair(P['lnC'], P['lnH'], P['lnL'], bb, att[i, s], beta[i, s], mass[i, s], ta, tb, Wk, K.G3_EPS, ps)
                n_pairs += 1
                okq = ws >= K.W_PRESENT - 1e-12
                accs += np.where(okq[:, :, None, None], acc, 0.0)
                cnt += okq
            for j, (N, a, p) in enumerate(K.G3_NAP):
                q = K.G3_KERN.index((N, a))
                pi = [1, 2].index(p)
                ref = np.where(cnt[:, q] >= K.MIN_ATT, accs[:, q, pi, 0], np.nan)
                got = F[f'g3_{j}_real'][bb, ta:tb].astype(np.float64)
                both = np.isfinite(ref) & np.isfinite(got)
                assert (np.isfinite(ref) == np.isfinite(got)).all(), 'numpy and numba disagree on which days have a force'
                if both.any():
                    worst = max(worst, float(np.max(np.abs(ref[both] - got[both]) / np.maximum(1e-12, np.abs(ref[both])))))
                    nchk += int(both.sum())
    tnp = time.time() - t
    per_pair_refit = tnp / max(n_pairs, 1)
    full_pairs = K.TOP * K.K_ATT * len(ref_all) * 2
    res.update({'numpy_g3_sample_s': tnp, 'numpy_pairs': n_pairs, 'numpy_g3_full_min': per_pair_refit * full_pairs / 60,
                'numpy_vs_numba_max_rel_diff': worst, 'numpy_vs_numba_values': nchk})
    K.log(f'G3 plain numpy: {n_pairs} pair-refits in {tnp:.1f}s -> full run (both worlds) {res["numpy_g3_full_min"]:.1f} min; '
          f'numpy vs numba on {nchk} stock-day values: max relative difference {worst:.2e} (float32 output)', fh)
    if res['numpy_g3_full_min'] < 30:
        dec = 'plain numpy, in refit chunks'
    elif res['numba_g3_full_min'] < 720:
        dec = 'numba (kern.g3_accum), in refit chunks; no lagged spread is materialized'
    else:
        dec = 'G3 not run: the numba extrapolation exceeds 12 hours'
    res['decision'] = dec
    K.log(f'DECISION: {dec}', fh)
    load1 = os.getloadavg()
    res['load_end'] = load1
    K.log(f'load average at the end {load1[0]:.1f}', fh)
    K.jdump(res, K.OUT / 'pilot.json')
    fh.close()


if __name__ == '__main__':
    main()
