"""Bystanders: build the event table (A's solo news days x A's co-movers) for one export.

    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 -I events.py old|new

Output: ~/.local/share/correlation2/experiments/bystanders/events_<tag>.npz, one row per (A, B, day) where day is a
solo news day of A and B is one of A's co-movers. Rules are those of experiments/-7.5-bystanders.md plus the
decisions recorded there.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CORR_DAYS, MIN_PAIR_OBS, TOP, MIN_CORR = 252, 200, 20, 0.3
SOLO_FRAC, PART_VR, BYST_VR = 0.2, 2.0, 1.5
MIN_NEWS, SHRINK = 8, 8.0
HS = (1, 5, 10)


def main(tag):
    P = common.load_panels(tag)
    days = P['days']; T = len(days)
    x = np.asarray(P['x']); r = np.asarray(P['r']); U = np.asarray(P['U'])
    sd60 = np.asarray(P['sd60']); vr = np.asarray(P['vr']); spike = np.asarray(P['spike'])
    c = np.asarray(P['close']); op = np.asarray(P['open'])
    msum, mcnt, m, moc = P['msum'], P['mcnt'], P['m'], P['moc']
    N = x.shape[0]
    # spike in the previous 5 days
    prev5 = np.zeros_like(spike)
    for k in range(1, 6):
        prev5[:, k:] |= spike[:, :-k]
    cand = spike & U & ~prev5
    news_rows = []   # (A, d, solo, n_co, n_part, xA_ex, mex)
    co_rows = []     # (A, B, d, corr, beta, n_pair)
    t_start = CORR_DAYS + 61
    for d in range(t_start, T):
        A_idx = np.where(cand[:, d])[0]
        if len(A_idx) == 0:
            continue
        pool = np.where(U[:, d])[0]
        W = x[pool, d - CORR_DAYS:d]
        Wa = x[A_idx, d - CORR_DAYS:d]
        Mp = np.isfinite(W).astype(np.float32); W0 = np.nan_to_num(W)
        Ma = np.isfinite(Wa).astype(np.float32); A0 = np.nan_to_num(Wa)
        n = Ma @ Mp.T
        sab = A0 @ W0.T
        sa = A0 @ Mp.T; sa2 = (A0 * A0) @ Mp.T
        sb = Ma @ W0.T; sb2 = Ma @ (W0 * W0).T
        with np.errstate(all='ignore'):
            cov = sab - sa * sb / n
            va = sa2 - sa * sa / n
            vb = sb2 - sb * sb / n
            cor = cov / np.sqrt(va * vb)
            beta = cov / va  # slope of B on A
        cor[~np.isfinite(cor) | (n < MIN_PAIR_OBS)] = -9
        for k, a in enumerate(A_idx):
            ck = cor[k].copy()
            ck[pool == a] = -9
            top = np.argsort(-ck)[:TOP]
            top = top[ck[top] >= MIN_CORR]
            Bs = pool[top]
            vrb = vr[Bs, d]
            n_part = int(np.sum(vrb >= PART_VR))
            n_co = len(Bs)
            solo = n_part < SOLO_FRAC * n_co if n_co > 0 else True
            mex = (msum[d] - r[a, d]) / (mcnt[d] - 1)
            news_rows.append((a, d, solo, n_co, n_part, r[a, d] - mex, mex))
            if solo:
                for t_, b in zip(top, Bs):
                    co_rows.append((a, b, d, cor[k, t_], beta[k, t_], n[k, t_]))
    news = np.array(news_rows, dtype=[('A', 'i4'), ('d', 'i4'), ('solo', '?'), ('n_co', 'i2'), ('n_part', 'i2'),
                                      ('xA', 'f8'), ('mex', 'f8')])
    co = np.array(co_rows, dtype=[('A', 'i4'), ('B', 'i4'), ('d', 'i4'), ('corr', 'f4'), ('beta', 'f4'), ('npair', 'i4')])
    print(f'news days {len(news)}, solo {news["solo"].sum()}, co-mover rows {len(co)}', flush=True)

    # per-A list of solo news days (sorted by day) with A's ex-A move and the ex-A market
    solo = news[news['solo']]
    order = np.lexsort((solo['d'], solo['A']))
    solo = solo[order]
    starts = np.searchsorted(solo['A'], np.arange(N + 1))

    # news-day response per co-mover row: slope (OLS with intercept) of B on A over A's earlier solo news days
    nn = np.zeros(len(co), np.int32); ns = np.full(len(co), np.nan)
    for i, (a, b, d) in enumerate(zip(co['A'], co['B'], co['d'])):
        s = solo[starts[a]:starts[a + 1]]
        s = s[s['d'] < d]
        if len(s) < MIN_NEWS:
            continue
        D = s['d']
        ok = U[b, D]
        if ok.sum() < MIN_NEWS:
            nn[i] = ok.sum(); continue
        xa = s['xA'][ok]; xb = r[b, D[ok]] - s['mex'][ok]
        xa_c = xa - xa.mean()
        nn[i] = ok.sum()
        ns[i] = float((xa_c * (xb - xb.mean())).sum() / (xa_c * xa_c).sum())
    w = nn / (nn + SHRINK)
    resp = np.where(nn >= MIN_NEWS, w * ns + (1 - w) * co['beta'], np.nan)

    # event-day quantities (ex-A market)
    nd = {(a, d): (xa, mex) for a, d, xa, mex in zip(news['A'], news['d'], news['xA'], news['mex'])}
    xA = np.array([nd[(a, d)][0] for a, d in zip(co['A'], co['d'])])
    mex = np.array([nd[(a, d)][1] for a, d in zip(co['A'], co['d'])])
    B, A, D = co['B'], co['A'], co['d']
    xB = r[B, D] - mex
    # forward market-relative returns, close to close from day d, and from the next open (store opens)
    mm = np.where(np.isfinite(m), m, 0.0)
    M = np.concatenate([[1.0], np.cumprod(1 + mm)])  # M[d+1] = index level at close of day d
    lvl = lambda d: M[d + 1]  # noqa: E731
    out = {}
    for h in HS:
        dh = D + h
        ok = dh < T
        dh_c = np.minimum(dh, T - 1)
        with np.errstate(all='ignore'):
            fb = c[B, dh_c] / c[B, D] - 1 - (lvl(dh_c) / lvl(D) - 1)
            fa = c[A, dh_c] / c[A, D] - 1 - (lvl(dh_c) / lvl(D) - 1)
            o1 = op[B, np.minimum(D + 1, T - 1)]
            fo = c[B, dh_c] / o1 - 1 - ((1 + moc[np.minimum(D + 1, T - 1)]) * lvl(dh_c) / lvl(np.minimum(D + 1, T - 1)) - 1)
        fb[~ok] = np.nan; fa[~ok] = np.nan; fo[~ok | ~(o1 > 0)] = np.nan
        out[f'fB{h}'] = fb.astype(np.float32); out[f'fA{h}'] = fa.astype(np.float32); out[f'foB{h}'] = fo.astype(np.float32)
    ex = np.zeros(T, bool); ex[common.expiry_days(days)] = True
    np.savez(common.BIG / f'events_{tag}.npz', A=A, B=B, d=D, corr=co['corr'], beta=co['beta'], npair=co['npair'],
             xA=xA, xB=xB, mex=mex, sdA=sd60[A, D], sdB=sd60[B, D], vrA=vr[A, D], vrB=vr[B, D],
             n_news=nn, slope_news=ns, resp=resp, expiry=ex[D], days=days,
             news_A=news['A'], news_d=news['d'], news_solo=news['solo'], news_nco=news['n_co'],
             news_npart=news['n_part'], news_xA=news['xA'], **out)
    print('saved', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'old')
