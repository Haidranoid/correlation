"""Step 1: per refit, the hot and cold groups, every stock's joiner score J, the controls and the 21-day outcome.

    python3 -I -B tj_build.py old     # breadth thrust's cached refits, 2006-03-03 .. 2025-06-13
    python3 -I -B tj_build.py new     # the unseen months (after tj_refits.py new), once, at the end

At refit t (= t0, export day index; groups from closes through t only):
  market-relative daily log return rel[j, d] = log(close[d] / close[d-1]) - market[d]  (the export's market: the
      equal-weight mean daily log return of the liquid stocks); NaN unless both closes exist;
  group score = mean over members of (log close t-63 -> t  minus  market over t-62..t);  top 3 = hot, bottom 3 = cold;
  H[d] = mean of rel[., d] over all members of the 3 hot groups (NaN if < 50% of them have a return); C likewise;
  J = [corr(rel_s, H) over t-62..t - corr(rel_s, H) over t-314..t-63] - [the same with C], needing >= 55 and >= 200
      days with both values; every stock of the 1,500 (members of hot and cold groups included);
  controls: own 63-day market-relative return (t-63 -> t), own 12-1 return (t-252 -> t-21), hot and cold dummies;
  outcome: y = log close t -> t+21 minus market over t+1..t+21; and the simple return close t -> t+21 for the book.
Reported only: Jb, the same J from partial correlations given the market (beta fitted in each window).
Writes BIG/<tag>/table.pkl.
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from tj_common import (H_MIN_SHARE, HOLD, K_GROUPS, MIN_OLDER, MIN_RECENT, MOM_LONG, MOM_SKIP, N_HOT,  # noqa: E402
                       OLDER_END, OLDER_START, RECENT, EXPORTS, big, c2eval, iso, last_close, load_refits,
                       masked_corr, masked_partial_corr)


def group_series(rel_w, members):
    """Equal-weight daily mean of rel over the members (rows of rel_w), NaN on days with < H_MIN_SHARE of them."""
    x = rel_w[members]
    f = np.isfinite(x)
    n = f.sum(0)
    with np.errstate(all='ignore'):
        s = np.where(f, x, 0.0).sum(0) / n
    return np.where(n >= H_MIN_SHARE * len(members), s, np.nan)


def main(tag):
    t_start = time.time()
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'market'))
    days = E['days']
    tickers = E['tickers']
    R = load_refits(tag)
    assert R['N'] == len(tickers) and R['T'] == len(days)
    with np.errstate(all='ignore'):
        L = np.log(np.asarray(E['close'], np.float64))
    L[~np.isfinite(L)] = np.nan
    N, T = L.shape
    m = np.asarray(E['market'], np.float64)
    rel = np.full((N, T), np.nan, np.float32)
    rel[:, 1:] = (L[:, 1:] - L[:, :-1]) - m[None, 1:]
    M = np.concatenate([[0.0], np.cumsum(np.nan_to_num(m))])   # M[d+1] - M[a] = sum market[a..d]
    Mn = np.concatenate([[0], np.cumsum(~np.isfinite(m))])

    def msum(a, b):  # market over days a..b inclusive (NaN if any missing)
        if b >= T or a < 0:
            return np.nan
        return M[b + 1] - M[a] if Mn[b + 1] - Mn[a] == 0 else np.nan

    print(f'{tag}: {N} x {T}, panels {time.time() - t_start:.0f}s', flush=True)
    out = []
    for rf in R['refits']:
        t, t1, U, lab = int(rf['t0']), int(rf['t1']), np.asarray(rf['U']), np.asarray(rf['lab'])
        a = t - OLDER_START
        assert a >= 1
        # own returns
        Lt = last_close(L, U, t, t - 5)
        own63 = Lt - last_close(L, U, t - RECENT, t - RECENT - 5) - msum(t - RECENT + 1, t)
        own121 = last_close(L, U, t - MOM_SKIP, t - MOM_SKIP - 5) - last_close(L, U, t - MOM_LONG, t - MOM_LONG - 5)
        # groups: score, hot, cold
        gscore = np.full(K_GROUPS, np.nan)
        for g in range(K_GROUPS):
            v = own63[lab == g]
            v = v[np.isfinite(v)]
            if len(v):
                gscore[g] = v.mean()
        order = np.argsort(-np.where(np.isfinite(gscore), gscore, -np.inf), kind='stable')
        hot_g = order[:N_HOT]
        fin = order[np.isfinite(gscore[order])]
        cold_g = fin[::-1][:N_HOT]
        hot = np.isin(lab, hot_g)
        cold = np.isin(lab, cold_g)
        # H and C, and the joiner score
        rel_w = np.asarray(rel[U, a:t + 1], np.float64)           # column c <-> day a + c
        H = group_series(rel_w, np.flatnonzero(hot))
        C = group_series(rel_w, np.flatnonzero(cold))
        mw = m[a:t + 1]
        split = OLDER_START - OLDER_END + 1                        # columns 0..split-1 = t-314..t-63
        old_w, rec_w = slice(0, split), slice(split, None)
        assert rel_w[:, rec_w].shape[1] == RECENT and rel_w[:, old_w].shape[1] == OLDER_START - OLDER_END + 1
        rHr, nHr = masked_corr(rel_w[:, rec_w], H[rec_w])
        rHo, nHo = masked_corr(rel_w[:, old_w], H[old_w])
        rCr, nCr = masked_corr(rel_w[:, rec_w], C[rec_w])
        rCo, nCo = masked_corr(rel_w[:, old_w], C[old_w])
        okJ = (nHr >= MIN_RECENT) & (nCr >= MIN_RECENT) & (nHo >= MIN_OLDER) & (nCo >= MIN_OLDER)
        dH = np.where(okJ, rHr - rHo, np.nan)
        dC = np.where(okJ, rCr - rCo, np.nan)
        J = dH - dC
        # reported only: beta-adjusted (partial correlation given the market)
        bHr, _ = masked_partial_corr(rel_w[:, rec_w], H[rec_w], mw[rec_w])
        bHo, _ = masked_partial_corr(rel_w[:, old_w], H[old_w], mw[old_w])
        bCr, _ = masked_partial_corr(rel_w[:, rec_w], C[rec_w], mw[rec_w])
        bCo, _ = masked_partial_corr(rel_w[:, old_w], C[old_w], mw[old_w])
        Jb = np.where(okJ, (bHr - bHo) - (bCr - bCo), np.nan)
        # outcome
        if t + HOLD <= T - 1:
            Lf = last_close(L, U, t + HOLD, t + 1)
            y = Lf - Lt - msum(t + 1, t + HOLD)
            ysimple = np.expm1(Lf - Lt)
        else:
            y = np.full(len(U), np.nan)
            ysimple = np.full(len(U), np.nan)
        out.append(dict(k=rf['k'], t0=t, t1=t1, U=U, lab=lab, glabel=rf['glabel'], gscore=gscore, hot_g=hot_g,
                        cold_g=cold_g, hot=hot, cold=cold, J=J, dH=dH, dC=dC, rHr=rHr, rHo=rHo, rCr=rCr, rCo=rCo,
                        Jb=Jb, own63=own63, own121=own121, y=y, ysimple=ysimple,
                        H_ok=int(np.isfinite(H).sum()), C_ok=int(np.isfinite(C).sum())))
        print(f'  k={rf["k"]} {iso(days[t])} J {np.isfinite(J).sum()} hot {hot.sum()} cold {cold.sum()} '
              f'H/C days {np.isfinite(H).sum()}/{np.isfinite(C).sum()} of {len(H)}', flush=True)
    with open(big(tag) / 'table.pkl', 'wb') as fh:
        pickle.dump(dict(days=days, tickers=tickers, T=T, N=N, refits=out), fh)
    print(f'done {len(out)} refits {time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
