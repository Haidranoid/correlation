"""Gravity, step 3: every candidate's and benchmark's daily force, for real and random attractors.

    python -I -B forces.py old|new|fwd

G1 (price): F = -sum_A m sign(g_C) / (r + eps)^p over B's attractors (p = 0: -sum m sign(g_C)); closes only uses
  r = |g_C|; the spring benchmark F = -sum m g_C. G1's pair gaps (g_C and the closest approach r) are kept for shape.py.
G2 (time): F = sum_A m f_A(t), f_A(t) = sum_tau w(tau) z_A(t + 1 - tau), each kernel with its own attractors.
G3 (both): F = -sum_A m sum_tau w(tau) sign(g_C,tau) / (r_tau + eps)^p, tau = 1..2N-1, with G1's attractors.
A force needs >= 5 valid attractor terms (a G3 or G2 term is valid when >= 90% of its kernel weight has data), B in the
universe that day, and t >= D0. Writes BIG/forces_<tag>/<key>.npy (float32 [R, T]) and BIG/pairs_<tag>_<world>_{gC,r}.npy
(float32 [R, 10, T]); report out/forces_<tag>.txt.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import data  # noqa: E402
import kern  # noqa: E402
import peers  # noqa: E402
np = K.np


def _att(PE, world, k):
    if world == 'real':
        return PE['A1'][k], np.asarray(PE['M1'][k], np.float64), PE['B1'][k]
    return PE['AR'][k], np.maximum(np.asarray(PE['MR'][k], np.float64), 0.0), PE['BR'][k]   # mass = rho clipped at 0


def _span(P, k):
    refit = P['refit']
    t0 = int(refit[k])
    t1 = int(refit[k + 1]) if k + 1 < len(refit) else P['T']
    ta = max(t0, P['d0'], P['u0'])
    return ta, min(t1, P['T'])


def _rows(P, k, brows):
    u = np.asarray(P['uni'][k])
    sel = np.nonzero(u >= 0)[0]
    if brows is not None:
        sel = sel[np.isin(u[sel], brows)]
    return sel, u[sel]


def g1(P, PE, refits, brows=None, worlds=('real', 'random'), keep_pairs=False):
    R, T = P['lnC'].shape
    out, pairs = {}, {}
    for world in worlds:
        keys = [f'g1_spring_{world}'] + [f'g1_{i}{c}_{world}' for i in range(len(K.G1_PE)) for c in ('', '_c')]
        F = {key: np.full((R, T), np.nan, np.float32) for key in keys}
        if keep_pairs:
            pairs[world] = (np.full((R, K.K_ATT, T), np.nan, np.float32), np.full((R, K.K_ATT, T), np.nan, np.float32))
        for k in refits:
            ta, tb = _span(P, k)
            if tb <= ta:
                continue
            sel, b = _rows(P, k, brows)
            if len(b) == 0:
                continue
            att, mass, beta = _att(PE, world, k)
            att, mass, beta = att[sel], mass[sel], beta[sel]
            Bv = np.repeat(b, K.K_ATT).astype(np.int64)
            Av = att.reshape(-1).astype(np.int64)
            bv = beta.reshape(-1).astype(np.float64)
            nd = tb - ta
            gC = np.empty((len(Bv), nd))
            gR = np.empty((len(Bv), nd))
            kern.gaps(P['lnC'], P['lnH'], P['lnL'], Bv, Av, bv, 0, ta, tb, gC, gR)
            gC = gC.reshape(len(b), K.K_ATT, nd)
            gR = gR.reshape(len(b), K.K_ATT, nd)
            m = mass[:, :, None]
            valid = np.isfinite(gC) & np.isfinite(m) & (att[:, :, None] >= 0)
            cnt = valid.sum(1)
            g0 = np.where(valid, gC, 0.0)
            sg = np.sign(g0)
            r0 = np.where(valid, gR, 1.0)
            ag = np.abs(g0)
            mm = np.where(valid, m, 0.0)

            def put(key, val):
                val = np.where(cnt >= K.MIN_ATT, val, np.nan)
                F[key][b, ta:tb] = val

            put(f'g1_spring_{world}', -(mm * g0).sum(1))
            for i, (p, e) in enumerate(K.G1_PE):
                if p == 0:
                    v = -(mm * sg).sum(1)
                    put(f'g1_{i}_{world}', v)
                    put(f'g1_{i}_c_{world}', v)
                else:
                    put(f'g1_{i}_{world}', -(mm * sg / (r0 + e) ** p).sum(1))
                    put(f'g1_{i}_c_{world}', -(mm * sg / (ag + e) ** p).sum(1))
            if keep_pairs:
                pairs[world][0][b, :, ta:tb] = np.where(valid, gC, np.nan)
                pairs[world][1][b, :, ta:tb] = np.where(valid, gR, np.nan)
        out.update(F)
    return out, pairs


def g2(P, PE, refits, brows=None, worlds=('real', 'random')):
    R, T = P['z'].shape
    out = {}
    for q, (N, a) in enumerate(K.G2_ALL):
        f = peers.kernel_f(np.asarray(P['z']), N, a)
        for world in worlds:
            Fq = np.full((R, T), np.nan, np.float32)
            for k in refits:
                ta, tb = _span(P, k)
                if tb <= ta:
                    continue
                sel, b = _rows(P, k, brows)
                if len(b) == 0:
                    continue
                if world == 'real':
                    att, m = PE['A2'][q, k][sel], np.asarray(PE['M2'][q, k][sel], np.float64)
                else:
                    att, m = PE['AR'][k][sel], np.asarray(PE['M2R'][q, k][sel], np.float64)
                fa = f[np.maximum(att, 0)][:, :, ta:tb]           # [n, 10, nd]
                valid = np.isfinite(fa) & np.isfinite(m)[:, :, None] & (att >= 0)[:, :, None]
                cnt = valid.sum(1)
                v = (np.where(valid, fa, 0.0) * np.where(np.isfinite(m), m, 0.0)[:, :, None]).sum(1)
                Fq[b, ta:tb] = np.where(cnt >= K.MIN_ATT, v, np.nan)
            out[f'g2_{q}_{world}'] = Fq
        del f
    return out


def g3_weights():
    Wk = np.zeros((len(K.G3_KERN), K.TAU_MAX + 1))
    for q, (N, a) in enumerate(K.G3_KERN):
        Wk[q, 1:2 * N] = K.kernel_weights(N, a)
    return Wk


def g3(P, PE, refits, brows=None, worlds=('real', 'random'), log=None):
    R, T = P['lnC'].shape
    Wk = g3_weights()
    ps = np.array([1.0, 2.0])
    out = {}
    t_start = time.time()
    for world in worlds:
        F = {f'g3_{i}{c}_{world}': np.full((R, T), np.nan, np.float32) for i in range(len(K.G3_NAP)) for c in ('', '_c')}
        for n_done, k in enumerate(refits):
            ta, tb = _span(P, k)
            if tb <= ta:
                continue
            sel, b = _rows(P, k, brows)
            if len(b) == 0:
                continue
            att, mass, beta = _att(PE, world, k)
            nd = tb - ta
            Fa = np.zeros((len(b), nd, len(K.G3_KERN), 2, 2))
            cnt = np.zeros((len(b), nd, len(K.G3_KERN)), np.int64)
            kern.g3_accum(P['lnC'], P['lnH'], P['lnL'], b.astype(np.int64), att[sel].astype(np.int64),
                          np.where(att[sel] >= 0, mass[sel], np.nan), beta[sel].astype(np.float64), ta, tb, Wk,
                          K.G3_EPS, ps, K.W_PRESENT - 1e-12, Fa, cnt)
            for i, (N, a, p) in enumerate(K.G3_NAP):
                q = K.G3_KERN.index((N, a))
                pi = [1, 2].index(p)
                ok = cnt[:, :, q] >= K.MIN_ATT
                F[f'g3_{i}_{world}'][b, ta:tb] = np.where(ok, Fa[:, :, q, pi, 0], np.nan)
                F[f'g3_{i}_c_{world}'][b, ta:tb] = np.where(ok, Fa[:, :, q, pi, 1], np.nan)
            if log is not None and n_done % 24 == 0:
                log(f'  G3 {world} refit {k}: {time.time() - t_start:.0f}s')
        out.update(F)
    return out


def load_inputs(tag):
    P = data.load_panel(tag)
    Q = {k: P[k] for k in ('days', 'refit', 'uni', 'inU', 'meta')}
    for k in ('lnC', 'lnH', 'lnL', 'z'):
        Q[k] = np.ascontiguousarray(np.asarray(P[k], np.float64))
    Q.update({'T': P['T'], 'd0': P['d0'], 'u0': P['u0'], 'f0': P['f0'], 'ids': np.array(P['meta']['ids'])})
    PE = dict(np.load(K.BIG / f'peers_{tag}.npz'))
    return Q, PE


def all_refits(P):
    return [k for k in range(len(P['refit'])) if _span(P, k)[1] > _span(P, k)[0]]


def save(tag, F, fh):
    d = K.BIG / f'forces_{tag}'
    d.mkdir(parents=True, exist_ok=True)
    for key, v in F.items():
        np.save(d / f'{key}.npy', v)


def main(tag):
    fh = open(K.OUT / f'forces_{tag}.txt', 'w')
    P, PE = load_inputs(tag)
    ref = all_refits(P)
    inU = np.asarray(P['inU'])
    days = P['days']
    K.log(f'forces {tag}: {len(ref)} refits, days {K.iso(days, _span(P, ref[0])[0])}..{K.iso(days, P["T"] - 1)}', fh)
    uu = inU.copy()
    uu[:, :P['d0']] = False
    nU = uu.sum()

    def report(F):
        for key, v in F.items():
            K.log(f'  {key:>18}: force on {100 * np.isfinite(v[uu]).sum() / nU:.2f}% of universe stock-days from D0', fh)

    for world in ('real', 'random'):
        t = time.time()
        F, pairs = g1(P, PE, ref, worlds=(world,), keep_pairs=True)
        K.log(f'G1 {world} done in {time.time() - t:.0f}s', fh)
        save(tag, F, fh)
        report(F)
        gc, gr = pairs[world]
        np.save(K.BIG / f'pairs_{tag}_{world}_gC.npy', gc)
        np.save(K.BIG / f'pairs_{tag}_{world}_r.npy', gr)
        del F, pairs, gc, gr
    t = time.time()
    F = g2(P, PE, ref)
    K.log(f'G2 done in {time.time() - t:.0f}s', fh)
    save(tag, F, fh)
    report(F)
    del F
    t = time.time()
    F = g3(P, PE, ref, log=lambda s: K.log(s, fh))
    K.log(f'G3 done in {time.time() - t:.0f}s', fh)
    save(tag, F, fh)
    report(F)
    fh.close()


if __name__ == '__main__':
    main(sys.argv[1])
