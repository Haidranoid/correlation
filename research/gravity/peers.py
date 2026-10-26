"""Gravity, step 2: attractors, masses and beta at each refit, for G1/G3 and for each G2 kernel, plus random attractors.

    python -I -B peers.py old|new|fwd

At each refit t0 (the universe's calendar), for each stock B of the universe:
- G1/G3: its 10 attractors are the universe stocks A != B with the highest correlation rho of daily log close-to-close
  returns over the 252 days t0-251..t0 (>= 227 common valid days), with rho > 0; mass m = rho; beta = OLS slope of B's
  returns on A's over the same common days.
- G2, for each kernel (N, a): f_A(t) = sum_tau w(tau) z_A(t + 1 - tau) (missing z counts 0; valid when >= 90% of the
  weight has data); m = corr(f_A(t), z_B(t + 1)) over the 252 pairs with t + 1 in t0-251..t0 (>= 227); the 10
  attractors are the universe stocks with the largest |m|; m is signed.
- Random: 10 universe stocks drawn for each B, seeded by (refit date, B's store id) so a truncated build draws the same
  ones; they carry their own rho (clipped at 0 as the mass) and beta, and their own G2 m for each kernel.
Writes BIG/peers_<tag>.npz.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import data  # noqa: E402
np = K.np


def pw_corr(X, Y, min_n):
    """Pairwise-complete correlation and slope of Y's rows on X's rows: corr[a, b] = corr(X_a, Y_b) over the columns
    where both exist; beta[a, b] = cov / var(X_a) on those columns; NaN where fewer than min_n."""
    Mx, My = np.isfinite(X), np.isfinite(Y)
    X0, Y0 = np.where(Mx, X, 0.0), np.where(My, Y, 0.0)
    Mxf, Myf = Mx.astype(np.float64), My.astype(np.float64)
    n = Mxf @ Myf.T
    Sx = X0 @ Myf.T
    Sy = Mxf @ Y0.T
    Sxx = (X0 * X0) @ Myf.T
    Syy = Mxf @ (Y0 * Y0).T
    Sxy = X0 @ Y0.T
    with np.errstate(all='ignore'):
        cov = Sxy - Sx * Sy / n
        vx = Sxx - Sx * Sx / n
        vy = Syy - Sy * Sy / n
        corr = cov / np.sqrt(vx * vy)
        beta = cov / vx
    bad = (n < min_n) | ~np.isfinite(corr)
    corr[bad] = np.nan
    beta[bad] = np.nan
    return corr, beta, n


def kernel_f(z, N, a):
    """f[:, t] = sum_tau w(tau) z[:, t + 1 - tau] (missing counts 0), NaN unless >= 90% of the weight has data."""
    w = K.kernel_weights(N, a)
    R, T = z.shape
    v = np.isfinite(z)
    z0 = np.where(v, z, 0.0)
    f = np.zeros((R, T))
    pres = np.zeros((R, T))
    for i, wt in enumerate(w):
        if wt == 0:
            continue
        s = i            # tau = i + 1 -> z[t - i]
        f[:, s:] += wt * z0[:, :T - s]
        pres[:, s:] += wt * v[:, :T - s]
    f[pres < K.W_PRESENT - 1e-12] = np.nan
    return f


def random_draws(uni_k, ids_rows, t0_day):
    """[n, 10] panel rows of each B's random attractors (uni_k: panel rows of the universe)."""
    sid = ids_rows[uni_k]
    order = np.argsort(sid)
    srt = uni_k[order]
    out = np.full((len(uni_k), K.K_ATT), -1, np.int32)
    for i, b in enumerate(uni_k):
        cand = srt[srt != b]
        rng = np.random.default_rng([K.SEED, int(t0_day), int(ids_rows[b])])
        out[i] = cand[rng.choice(len(cand), K.K_ATT, replace=False)]
    return out


def top_k(score, exclude_self_idx, k, positive_only):
    """Indices of the k largest finite scores per column (self excluded); -1 padded."""
    S = score.copy()
    S[exclude_self_idx, np.arange(S.shape[1])] = np.nan
    if positive_only:
        S[~(S > 0)] = np.nan
    S = np.where(np.isfinite(S), S, -np.inf)
    idx = np.argsort(-S, axis=0, kind='stable')[:k].T          # [nB, k]
    ok = np.take_along_axis(S.T, idx, 1) > -np.inf
    return np.where(ok, idx, -1)


def compute(P, refits=None, log=None):
    """Attractors for refits (indices into P['refit']; default all in force from D0). Returns dict of arrays
    indexed by refit position."""
    lnC = np.asarray(P['lnC'])
    z = np.asarray(P['z'])
    days = P['days']
    T = P['T']
    ids_rows = np.asarray(P['ids'] if 'ids' in P else P['meta']['ids'])
    refit = np.asarray(P['refit'])
    uni = np.asarray(P['uni'])
    nK = len(refit)
    if refits is None:
        refits = range(nK)
    r = np.full(lnC.shape, np.nan)
    r[:, 1:] = lnC[:, 1:] - lnC[:, :-1]
    nq = len(K.G2_ALL)
    A1 = np.full((nK, K.TOP, K.K_ATT), -1, np.int32)
    M1 = np.full((nK, K.TOP, K.K_ATT), np.nan, np.float32)
    B1 = np.full((nK, K.TOP, K.K_ATT), np.nan, np.float64)
    AR = np.full((nK, K.TOP, K.K_ATT), -1, np.int32)
    MR = np.full((nK, K.TOP, K.K_ATT), np.nan, np.float32)      # random rho (unclipped; the mass clips at 0)
    BR = np.full((nK, K.TOP, K.K_ATT), np.nan, np.float64)
    A2 = np.full((nq, nK, K.TOP, K.K_ATT), -1, np.int32)
    M2 = np.full((nq, nK, K.TOP, K.K_ATT), np.nan, np.float32)
    M2R = np.full((nq, nK, K.TOP, K.K_ATT), np.nan, np.float32)
    fs = [kernel_f(z, N, a) for (N, a) in K.G2_ALL]
    t_start = time.time()
    for k in refits:
        t0 = int(refit[k])
        t1 = int(refit[k + 1]) if k + 1 < nK else T
        if t1 <= P['d0'] or t0 >= T:
            continue
        u = uni[k][uni[k] >= 0]
        n = len(u)
        Rw = r[u, t0 - K.CORR_W + 1:t0 + 1]
        corr, beta, _ = pw_corr(Rw, Rw, K.CORR_MIN)
        self_idx = np.arange(n)
        top = top_k(corr, self_idx, K.K_ATT, True)
        cols = np.arange(n)[:, None]
        A1[k, :n] = np.where(top >= 0, u[np.maximum(top, 0)], -1)
        M1[k, :n] = np.where(top >= 0, corr[np.maximum(top, 0), cols], np.nan)
        B1[k, :n] = np.where(top >= 0, beta[np.maximum(top, 0), cols], np.nan)
        rd = random_draws(u, ids_rows, days[t0])
        pos = {int(x): i for i, x in enumerate(u)}
        ri = np.vectorize(pos.get)(rd)
        AR[k, :n] = rd
        MR[k, :n] = corr[ri, cols]
        BR[k, :n] = beta[ri, cols]
        Y = z[u, t0 - K.CORR_W + 1:t0 + 1]
        for q in range(nq):
            X = fs[q][u, t0 - K.CORR_W:t0]
            m2, _, _ = pw_corr(X, Y, K.CORR_MIN)
            topq = top_k(np.abs(m2), self_idx, K.K_ATT, False)
            A2[q, k, :n] = np.where(topq >= 0, u[np.maximum(topq, 0)], -1)
            M2[q, k, :n] = np.where(topq >= 0, m2[np.maximum(topq, 0), cols], np.nan)
            M2R[q, k, :n] = m2[ri, cols]
        if log is not None and k % 24 == 0:
            log(f'  refit {k} ({K.iso(days, t0)}): {time.time() - t_start:.0f}s')
    return {'A1': A1, 'M1': M1, 'B1': B1, 'AR': AR, 'MR': MR, 'BR': BR, 'A2': A2, 'M2': M2, 'M2R': M2R}


def main(tag):
    fh = open(K.OUT / f'peers_{tag}.txt', 'w')
    P = data.load_panel(tag)
    K.log(f'peers {tag}: {len(P["refit"])} refits', fh)
    out = compute(P, log=lambda s: K.log(s, fh))
    np.savez(K.BIG / f'peers_{tag}.npz', **out)
    k = np.nonzero((out['A1'] >= 0).any((1, 2)))[0]
    nval = (out['A1'][k] >= 0).sum(2)
    K.log(f'G1: refits with attractors {len(k)}; attractors per stock: mean {nval.mean():.2f}, share with 10 '
          f'{np.mean(nval == 10):.4f}, share with < 5 {np.mean(nval < 5):.4f}', fh)
    m = out['M1'][k][out['A1'][k] >= 0]
    K.log(f'G1 masses (rho): median {np.median(m):.3f}, 10th pct {np.percentile(m, 10):.3f}, 90th {np.percentile(m, 90):.3f}', fh)
    mr = out['MR'][k][out['AR'][k] >= 0]
    K.log(f'random rho: median {np.nanmedian(mr):.3f}; share negative {np.mean(mr[np.isfinite(mr)] < 0):.3f}; '
          f'missing {np.mean(~np.isfinite(mr)):.4f}', fh)
    for q, (N, a) in enumerate(K.G2_ALL):
        mm = out['M2'][q][k][out['A2'][q][k] >= 0]
        K.log(f'{K.g2_name(N, a)}: |m| median {np.median(np.abs(mm)):.3f}, share m > 0 {np.mean(mm > 0):.3f}', fh)
    fh.close()


if __name__ == '__main__':
    main(sys.argv[1])
