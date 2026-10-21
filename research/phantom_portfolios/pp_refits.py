"""Step 2: one refit every 21 trading days (run after pp_panels.py), in parallel, one npz per refit under
BIG/<tag>/refits/.

    python3 -I pp_refits.py old|new [--workers 8] [--only R]

At refit row R (decision at the close of R-1; nothing from R on is used to fit):
  1. universe: stocks with returns on >= 95% of the 252 days before R, ranked by median dollar volume over those
     days; the top 1,500 are the phantom universe, the top 1,000 of them (median dollar volume >= $5M) the
     groups-map universe;
  2. groups map: together.rs `groups` in numpy (20 groups, market taken out with beta, spectral + k-means) on
     the 252 days before R; the other ~500 stocks join the group whose mean market-relative return they
     correlate with most;
  3. loadings: per stock, OLS over the 252 days before R of its return on [1, market, its group's return
     with the stock left out, 6 tercile style factors]; residual = return - loadings x factors (no intercept),
     for days R-252 .. R+40 with these fixed loadings (days >= R are out of sample);
  4. phantom clusters: residual correlations over the 126 days before R, spectral clustering into 50; each
     cluster's mean pairwise residual correlation against NULL_RUNS clusterings of the same panel with each
     stock's dates shifted in 21-day blocks (kept: >= 8 members and p < 0.01);
  5. daily tables for days R .. R+20: each cluster's equal-weight residual move, its z against the 126 days
     before, trailing 21-day residual, median member volume ratio, breadth, the largest share of the move
     from a member with its own 4x volume spike; member rows (outcomes over later days) for |z| >= 2;
     the same for the groups (returns before the group is removed), member rows for |z| >= 3.
"""
import os
import sys

if __name__ == '__main__':
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[v] = '1'

import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pp_common import (BLOCK, BREADTH, CORR_WIN, EXPORTS, HORIZONS, K_GROUPS, K_PHANTOM, LOAD_WIN,  # noqa: E402
                       MIN_DV, MIN_MEMBERS, N_GROUP_STOCKS, N_STOCKS, NULL_P, NULL_RUNS, POST, REFIT_EVERY, SPIKE,
                       big, c2eval, market_relative, mean_offdiag, spectral, standardize)

G = {}


def init(tag):
    out = big(tag)
    export, _ = EXPORTS[tag]
    E = c2eval.load_export(export, arrays=('dollar_volume',))
    G['dv'] = E['dollar_volume']
    G['days'] = E['days']
    for a in ('ret', 'intra', 'volratio', 'chars'):
        G[a] = np.load(out / f'{a}.npy', mmap_mode='r')
    f = np.load(out / 'factors.npz')
    for a in ('M', 'M_intra', 'F', 'F_intra', 'etf_ret', 'etf_tickers'):
        G[a] = f[a]
    G['out'] = out / 'refits'
    G['out'].mkdir(exist_ok=True)


def loo_group(x, lab, k):
    """Each row's group mean over the other members (NaN-aware) for an [S, L] panel and group labels."""
    f = np.isfinite(x)
    x0 = np.where(f, x, 0.0)
    s = np.zeros((k, x.shape[1]))
    c = np.zeros((k, x.shape[1]))
    np.add.at(s, lab, x0)
    np.add.at(c, lab, f.astype(np.float64))
    num = s[lab] - x0
    den = c[lab] - f
    with np.errstate(all='ignore'):
        return np.where(den >= 1, num / np.maximum(den, 1), np.nan)


def ols_batch(y, X, min_rows):
    """Per-row OLS: y [S, L], X [S, L, K] -> b [S, K] (NaN rows dropped), valid [S]."""
    w = np.isfinite(y) & np.isfinite(X).all(2)
    Xw = np.where(w[..., None], X, 0.0)
    yw = np.where(w, y, 0.0)
    xtx = np.einsum('slk,slm->skm', Xw, Xw)
    xty = np.einsum('slk,sl->sk', Xw, yw)
    kk = X.shape[2]
    xtx += 1e-12 * np.eye(kk)[None]
    valid = w.sum(1) >= min_rows
    b = np.full((len(y), kk), np.nan)
    if valid.any():
        b[valid] = np.linalg.solve(xtx[valid], xty[valid][..., None])[..., 0]
    return b, valid


def fsum(x, a, b):
    """Sum of x[:, a:b] over finite values; NaN if b exceeds the panel or all are missing."""
    if b > x.shape[1] or a >= b:
        return np.full(x.shape[0], np.nan)
    blk = x[:, a:b]
    f = np.isfinite(blk)
    return np.where(f.any(1), np.where(f, blk, 0).sum(1), np.nan)


def day_tables(kind, R, D0, members_list, resid, other, volr, M, extra):
    """Daily basket table for d in [R, R+21) and member rows for big days.
    resid: [S, L] residual used for the basket (clusters: full residual; groups: before the group is removed);
    other: dict of [S, L] panels for member outcomes."""
    L = resid.shape[1]
    rows, mem = [], []
    z_member = 2.0 if kind == 'cluster' else 3.0
    for c, idx in enumerate(members_list):
        if len(idx) < MIN_MEMBERS:
            continue
        x = resid[idx]
        f = np.isfinite(x)
        nv = f.sum(0)
        xc = np.where(nv >= max(0.8 * len(idx), 2), np.where(f, x, 0).sum(0) / np.maximum(nv, 1), np.nan)
        for dr in range(R - D0, min(R - D0 + REFIT_EVERY, L)):
            X = xc[dr]
            if not np.isfinite(X):
                continue
            hist = xc[dr - CORR_WIN:dr]
            hist = hist[np.isfinite(hist)]
            if len(hist) < 100:
                continue
            sig = hist.std(ddof=1)
            z = X / sig if sig > 0 else np.nan
            tr = xc[dr - 21:dr]
            trend = np.nansum(tr) if np.isfinite(tr).sum() >= 15 else np.nan
            e = x[:, dr]
            fe = np.isfinite(e)
            n = int(fe.sum())
            vr = volr[idx, dr]
            vmed = np.nanmedian(vr) if np.isfinite(vr).any() else np.nan
            down = float((e[fe] < 0).mean())
            up = float((e[fe] > 0).mean())
            sp = fe & np.isfinite(vr) & (vr >= SPIKE)
            spike = float((e[sp] / (n * X)).max()) if sp.any() and X != 0 else 0.0
            key = len(rows)
            rows.append((R, D0 + dr, c, len(idx), n, X, sig, z, trend, vmed, down, up, spike, M[D0 + dr],
                         extra[c] if extra is not None else 1))
            if np.isfinite(z) and abs(z) >= z_member:
                for i_loc, i in enumerate(idx):
                    if not fe[i_loc]:
                        continue
                    loo = (n * X - e[i_loc]) / (n - 1)
                    rec = [key, i, e[i_loc], loo, vr[i_loc]]
                    for name, (pan, mode) in other.items():
                        rec.append(pan(i, dr))
                    mem.append(rec)
    return rows, mem


def refit(args):
    tag, R = args
    t0 = time.time()
    ret, intra, volr, dv = G['ret'], G['intra'], G['volratio'], G['dv']
    M, Mi, F, Fi = G['M'], G['M_intra'], G['F'], G['F_intra']
    T = ret.shape[1]
    D0, D1 = R - LOAD_WIN, min(R + POST, T)
    L = D1 - D0
    W = LOAD_WIN
    # 1. universe
    rw = np.asarray(ret[:, D0:R])
    cov = np.isfinite(rw).mean(1)
    with np.errstate(all='ignore'):
        mdv = np.nanmedian(np.asarray(dv[:, D0:R]), 1)
    cand = np.flatnonzero((cov >= 0.95) & np.isfinite(mdv) & (mdv > 0))
    order = cand[np.argsort(-mdv[cand], kind='stable')]
    U = order[:N_STOCKS]
    gfit_mask = np.zeros(len(U), bool)
    gfit_mask[:N_GROUP_STOCKS] = True
    gfit_mask &= mdv[U] >= MIN_DV
    # 2. groups map on the 252 days before R
    RU = np.asarray(ret[U, D0:D1], np.float64)
    gi = np.flatnonzero(gfit_mask)
    rel, mkt = market_relative(RU[gi, :W])
    zg = standardize(rel)
    glab_fit = spectral(zg @ zg.T, K_GROUPS, 0x6E0 + R)
    glab = np.full(len(U), -1)
    glab[gi] = glab_fit
    gmean = np.array([np.nanmean(rel[glab_fit == g], 0) for g in range(K_GROUPS)])
    oi = np.flatnonzero(~gfit_mask)
    if len(oi):
        x = RU[oi, :W]
        mm = np.where(np.isfinite(x), mkt[None, :], np.nan)
        xm, mmn = np.nanmean(x, 1, keepdims=True), np.nanmean(mm, 1, keepdims=True)
        beta = np.nansum((x - xm) * (mm - mmn), 1, keepdims=True) / np.maximum(np.nansum((mm - mmn) ** 2, 1, keepdims=True), 1e-18)
        zo = standardize(x - beta * mkt[None, :])
        zm = standardize(gmean)
        glab[oi] = (zo @ zm.T).argmax(1)
    # 3. loadings and residuals
    IU = np.asarray(intra[U, D0:D1], np.float64)
    Gl = loo_group(RU, glab, K_GROUPS)
    Gli = loo_group(IU, glab, K_GROUPS)
    S = len(U)
    Xf = np.empty((S, L, 9))
    Xf[:, :, 0] = 1.0
    Xf[:, :, 1] = M[D0:D1][None, :]
    Xf[:, :, 2] = Gl
    Xf[:, :, 3:] = F[:, D0:D1].T[None, :, :]
    b, valid = ols_batch(RU[:, :W], Xf[:, :W], 200)
    c2, valid2 = ols_batch(RU[:, :W], Xf[:, :W, :3], 200)
    valid &= valid2
    pred = np.einsum('slk,sk->sl', Xf[:, :, 1:], b[:, 1:])
    e = RU - pred                                         # residual
    p = e + b[:, 2:3] * Gl                                # before the group is removed
    h = RU - c2[:, 1:2] * M[D0:D1][None, :] - c2[:, 2:3] * Gl  # hedged with group and market
    Xi = np.empty((S, L, 8))
    Xi[:, :, 0] = Mi[D0:D1][None, :]
    Xi[:, :, 1] = Gli
    Xi[:, :, 2:] = Fi[:, D0:D1].T[None, :, :]
    e_in = IU - np.einsum('slk,sk->sl', Xi, b[:, 1:])
    h_in = IU - c2[:, 1:2] * Mi[D0:D1][None, :] - c2[:, 2:3] * Gli
    e[~valid] = np.nan
    p[~valid] = np.nan
    # keep only stocks with loadings
    keep = np.flatnonzero(valid)
    U, glab, RU, e, p, h, e_in, h_in, b, c2 = (a[keep] for a in (U, glab, RU, e, p, h, e_in, h_in, b, c2))
    S = len(U)
    # 4. phantom clusters
    ew = e[:, W - CORR_WIN:W]
    z = standardize(ew)
    corr = z @ z.T
    clab = spectral(corr, K_PHANTOM, 0xC1A5 + R)
    sizes = np.bincount(clab, minlength=K_PHANTOM)
    mcorr = np.array([mean_offdiag(corr, np.flatnonzero(clab == c)) if sizes[c] >= 2 else np.nan for c in range(K_PHANTOM)])
    rng = np.random.default_rng(0x5EED + R)
    null_means, null_sizes = [], []
    nb = CORR_WIN // BLOCK
    for run in range(NULL_RUNS):
        starts = rng.integers(0, CORR_WIN, size=(S, nb))
        idx = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % CORR_WIN).reshape(S, nb * BLOCK)
        zs = standardize(np.take_along_axis(ew, idx, 1))
        cn = zs @ zs.T
        ln = spectral(cn, K_PHANTOM, 0x9001 + 1000 * R + run)
        sz = np.bincount(ln, minlength=K_PHANTOM)
        for c in range(K_PHANTOM):
            if sz[c] >= MIN_MEMBERS:
                null_means.append(mean_offdiag(cn, np.flatnonzero(ln == c)))
                null_sizes.append(sz[c])
    null_means = np.array(null_means)
    pval = np.array([(1 + (null_means >= mcorr[c]).sum()) / (1 + len(null_means)) if sizes[c] >= MIN_MEMBERS else np.nan
                     for c in range(K_PHANTOM)])
    kept = (sizes >= MIN_MEMBERS) & (pval < NULL_P)
    # cluster descriptions: modal group share, style characteristics (z across the universe at R-1), label ETFs
    purity = np.array([np.bincount(glab[clab == c], minlength=K_GROUPS).max() / max(sizes[c], 1) for c in range(K_PHANTOM)])
    modal = np.array([np.bincount(glab[clab == c], minlength=K_GROUPS).argmax() if sizes[c] else -1 for c in range(K_PHANTOM)])
    ch = np.asarray(G['chars'][:, U, R - 1], np.float64)
    chz = (ch - np.nanmean(ch, 1, keepdims=True)) / np.nanstd(ch, 1, keepdims=True)
    style = np.array([np.nanmean(chz[:, clab == c], 1) if sizes[c] else np.full(6, np.nan) for c in range(K_PHANTOM)])
    tick = list(G['etf_tickers'])
    er = G['etf_ret'][:, R - CORR_WIN:R].astype(np.float64)
    mw = M[R - CORR_WIN:R]
    lab_series, lab_names = [], []
    for a, bb in (('IWD', 'IWF'), ('IWN', 'IWO'), ('IWM', 'SPY'), ('MTUM', 'SPY'), ('QUAL', 'SPY'), ('USMV', 'SPY'), ('VLUE', 'SPY')):
        lab_series.append(er[tick.index(a)] - er[tick.index(bb)])
        lab_names.append(f'{a}-{bb}')
    for k, tk in enumerate(tick):
        if tk == 'SPY':
            continue
        x = er[k]
        ok = np.isfinite(x) & np.isfinite(mw)
        if ok.sum() >= 100:
            bta = np.cov(x[ok], mw[ok])[0, 1] / np.var(mw[ok], ddof=1)
            lab_series.append(x - bta * mw)
        else:
            lab_series.append(np.full(CORR_WIN, np.nan))
        lab_names.append(tk)
    lab_series = np.array(lab_series)
    lab_corr = np.full((K_PHANTOM, len(lab_names)), np.nan)
    for c in range(K_PHANTOM):
        if sizes[c] < MIN_MEMBERS:
            continue
        xc = np.nanmean(ew[clab == c], 0)
        for k in range(len(lab_names)):
            y = lab_series[k]
            ok = np.isfinite(xc) & np.isfinite(y)
            if ok.sum() >= 100:
                lab_corr[c, k] = np.corrcoef(xc[ok], y[ok])[0, 1]
    # 5. daily tables
    volU = np.asarray(volr[U, D0:D1])
    rawU = RU

    def fut_open(i, dr, hh, base, base_in):
        if dr + 1 + hh > L:
            return np.nan
        first = base_in[i, dr + 1]
        rest = fsum(base[i:i + 1], dr + 2, dr + 1 + hh)[0] if hh > 1 else 0.0
        return first + (rest if np.isfinite(rest) else 0.0)

    def raw_ret(i, dr, hh):
        if dr + 1 + hh > L:
            return np.nan
        x = rawU[i, dr + 1:dr + 1 + hh]
        return np.prod(1 + x[np.isfinite(x)]) - 1

    other = {}
    for hh in HORIZONS:
        other[f'e{hh}'] = ((lambda i, dr, hh=hh: fsum(e[i:i + 1], dr + 1, dr + 1 + hh)[0]), 0)
    for hh in HORIZONS:
        other[f'eo{hh}'] = ((lambda i, dr, hh=hh: fut_open(i, dr, hh, e, e_in)), 0)
    for hh in HORIZONS:
        other[f'p{hh}'] = ((lambda i, dr, hh=hh: fsum(p[i:i + 1], dr + 1, dr + 1 + hh)[0]), 0)
    other['p0'] = ((lambda i, dr: p[i, dr]), 0)
    other['e0'] = ((lambda i, dr: e[i, dr]), 0)
    other['h5'] = ((lambda i, dr: fsum(h[i:i + 1], dr + 1, dr + 6)[0]), 0)
    other['ho5'] = ((lambda i, dr: fut_open(i, dr, 5, h, h_in)), 0)
    other['raw5'] = ((lambda i, dr: raw_ret(i, dr, 5)), 0)
    other['hedge_m'] = ((lambda i, dr: c2[i, 1]), 0)
    other['hedge_g'] = ((lambda i, dr: c2[i, 2]), 0)
    other['j'] = ((lambda i, dr: U[i]), 0)
    member_cols = ['key', 'i', 'own', 'loo', 'vr'] + list(other.keys())

    cl_members = [np.flatnonzero(clab == c) for c in range(K_PHANTOM)]
    c_rows, c_mem = day_tables('cluster', R, D0, cl_members, e, other, volU, M, kept.astype(int))
    gr_members = [np.flatnonzero(glab == g) for g in range(K_GROUPS)]
    g_rows, g_mem = day_tables('group', R, D0, gr_members, p, other, volU, M, None)
    row_cols = ['R', 'd', 'c', 'size', 'n', 'X', 'sigma', 'z', 'trend', 'vmed', 'down', 'up', 'spike', 'M', 'kept']
    np.savez_compressed(
        G['out'] / f'R{R:05d}.npz', R=R, U=U, glab=glab, clab=clab, sizes=sizes, mcorr=mcorr, pval=pval, kept=kept,
        null_means=null_means, null_sizes=np.array(null_sizes), purity=purity, modal=modal, style=style,
        lab_corr=lab_corr, lab_names=np.array(lab_names), loadings=b, hedge=c2,
        c_rows=np.array(c_rows, np.float64).reshape(-1, len(row_cols)), c_mem=np.array(c_mem, np.float64).reshape(-1, len(member_cols)),
        g_rows=np.array(g_rows, np.float64).reshape(-1, len(row_cols)), g_mem=np.array(g_mem, np.float64).reshape(-1, len(member_cols)),
        row_cols=np.array(row_cols), member_cols=np.array(member_cols))
    return R, time.time() - t0, int(kept.sum()), int((sizes >= MIN_MEMBERS).sum())


def schedule(tag):
    out = big(tag)
    f = np.load(out / 'factors.npz')
    F, M = f['F'], f['M']
    ok = np.isfinite(F).all(0) & np.isfinite(M)
    first = int(np.flatnonzero(ok)[0])
    R0 = first + LOAD_WIN
    T = F.shape[1]
    return list(range(R0, T, REFIT_EVERY))


if __name__ == '__main__':
    tag = sys.argv[1]
    workers = 8
    only = None
    if '--workers' in sys.argv:
        workers = int(sys.argv[sys.argv.index('--workers') + 1])
    if '--only' in sys.argv:
        only = int(sys.argv[sys.argv.index('--only') + 1])
    Rs = schedule(tag)
    if only is not None:
        Rs = [Rs[only]]
    done = {int(p.stem[1:]) for p in (big(tag) / 'refits').glob('R*.npz')} if '--resume' in sys.argv else set()
    Rs = [r for r in Rs if r not in done]
    print(f'{tag}: {len(Rs)} refits, first R={Rs[0] if Rs else None}', flush=True)
    days = np.load(big(tag) / 'factors.npz')['days']
    t0 = time.time()
    with Pool(workers, initializer=init, initargs=(tag,)) as pool:
        for k, (R, sec, nk, n8) in enumerate(pool.imap_unordered(refit, [(tag, r) for r in Rs])):
            print(f'  R={R} {c2eval.iso(days[R])} '
                  f'{sec:.0f}s kept {nk}/{n8}  [{k + 1}/{len(Rs)} {time.time() - t0:.0f}s]', flush=True)
