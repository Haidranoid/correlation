"""Step 1: the groups map refit every 21 trading days, and each refit's daily group tables.

    python3 -I bt_refits.py old              # every refit, 2003 on (~15 min)
    python3 -I bt_refits.py new 2024-06-01   # the new export, refits from that date on (the unseen months)

Refit calendar: c2eval.twins's rebalance days (export day indices 252, 273, ...), so the shared exclusion mask
excluded(E, tw)[k] belongs to the same refit k. Point in time: refit k at day t0 uses closes through t0 only, and its
groups hold for decisions on days t0 .. t0+20 (until the next refit).

At refit day t0:
  1. universe: stocks not excluded at k, with a return on >= 95% of days t0-251..t0, ranked by median dollar volume
     over those days; the top 1,500 (no $5M floor: it left ~1,070 stocks in 2008);
  2. groups: simple daily returns over the 252 days, market-relative (beta to their equal-weight mean), correlation,
     spectral clustering into 30 groups (bt_common, copied from phantom portfolios' pp_common.py), seed 0x6E0 + t0;
  3. each member's leave-one-out correlation with its group's mean market-relative return (the judge's score);
  4. each group's label: the most common twin fund among its members (for reporting only);
  5. daily tables for this refit's members: breadth (share above their 50-day SMA) on days t0-11 .. t1-1, so the
     10-day lookback and the previous day use the same members; the group's 10-day market-relative log return and
     its forward 21-day market-relative log return (mean over members) on days t0 .. t1-1.
Writes BIG/<tag>/refits.pkl.
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_common import (BREADTH_MIN_SHARE, CORR_WIN, COVER, EXPORTS, HOLD, K_GROUPS, MIN_DV, MIN_MEMBERS, MOM,  # noqa: E402
                       N_STOCKS, SEED, SMA, SMA_MIN, LOOKBACK, big, c2eval, iso, loo_corr, market_relative, spectral,
                       standardize)


def above_sma(close):
    """[N, T] int8: 1 if close[d] > the mean of closes d-49..d, 0 if not, -1 if the close or the SMA is missing
    (the SMA needs >= SMA_MIN finite closes of the 50)."""
    N, T = close.shape
    out = np.full((N, T), -1, np.int8)
    for a in range(0, N, 1000):
        c = np.asarray(close[a:a + 1000], np.float64)
        f = np.isfinite(c)
        cs = np.concatenate([np.zeros((len(c), 1)), np.cumsum(np.where(f, c, 0.0), 1)], 1)
        cn = np.concatenate([np.zeros((len(c), 1), np.int32), np.cumsum(f, 1, dtype=np.int32)], 1)
        s = np.full(c.shape, np.nan)
        n = np.zeros(c.shape, np.int32)
        s[:, SMA - 1:] = cs[:, SMA:] - cs[:, :T - SMA + 1]
        n[:, SMA - 1:] = cn[:, SMA:] - cn[:, :T - SMA + 1]
        ok = f & (n >= SMA_MIN)
        with np.errstate(all='ignore'):
            sma = s / np.maximum(n, 1)
            out[a:a + 1000] = np.where(ok, (c > sma).astype(np.int8), np.int8(-1))
    return out


def main(tag, first=None, limit=None):
    t_start = time.time()
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume', 'market'))
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
    del ok
    above = above_sma(close)
    mkt = np.asarray(E['market'], np.float64)
    mc = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mkt))])        # mc[d+1] - mc[a+1] = sum market[a+1..d]
    mn = np.concatenate([[0], np.cumsum(~np.isfinite(mkt))])
    dv = E['dollar_volume']
    print(f'{tag}: {N} stocks x {T} days, panels {time.time() - t_start:.0f}s', flush=True)

    rebal = tw['rebal']
    ks = range(len(rebal))
    if first is not None:
        d0 = c2eval.day_index(days, first)
        ks = [k for k in ks if rebal[k] >= d0]
    if limit is not None:
        ks = list(ks)[:limit]
    out = []
    prev_t = time.time()
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
        if len(U) < N_STOCKS:   # decision: no groups until 1,500 stocks qualify (only refits before 2006)
            print(f'  k={k} {iso(days[t0])} universe {len(U)} < {N_STOCKS}: skipped', flush=True)
            continue
        rel, _ = market_relative(R[U])
        z = standardize(rel)
        lab = spectral(z @ z.T, K_GROUPS, SEED + t0)
        rho = loo_corr(rel, lab, K_GROUPS)
        sizes = np.bincount(lab, minlength=K_GROUPS)
        fr = tw['fund_row'][k, U]
        glabel = []
        for g in range(K_GROUPS):
            f = fr[(lab == g) & (fr >= 0)]
            glabel.append(str(fund_ticker[np.bincount(f).argmax()]) if len(f) else '-')
        # daily tables
        nb = t1 - (t0 - LOOKBACK - 1)
        nd = t1 - t0
        breadth = np.full((K_GROUPS, nb), np.nan)
        mom = np.full((K_GROUPS, nd), np.nan)
        y21 = np.full((K_GROUPS, nd), np.nan)
        ny = np.zeros((K_GROUPS, nd), np.int32)
        A = above[U, t0 - LOOKBACK - 1:t1]
        e = min(t1 + HOLD, T)
        with np.errstate(all='ignore'):
            Lc = np.log(close[U, t0 - MOM:e])          # column c <-> day t0 - MOM + c
        dd = np.arange(t0, t1)
        msum = lambda lo, hi: np.where(mn[hi + 1] - mn[lo + 1] == 0, mc[hi + 1] - mc[lo + 1], np.nan)  # noqa: E731
        m_mom = msum(dd - MOM, dd)                       # market over d-9..d
        has_fwd = dd + HOLD <= T - 1
        m_fwd = np.full(nd, np.nan)
        m_fwd[has_fwd] = msum(dd[has_fwd], dd[has_fwd] + HOLD)
        for g in range(K_GROUPS):
            idx = np.flatnonzero(lab == g)
            if len(idx) < MIN_MEMBERS:
                continue
            Ag = A[idx]
            nv = (Ag >= 0).sum(0)
            br = (Ag == 1).sum(0) / np.maximum(nv, 1)
            breadth[g] = np.where(nv >= BREADTH_MIN_SHARE * len(idx), br, np.nan)
            Lg = Lc[idx]
            c = dd - t0 + MOM
            x = Lg[:, c] - Lg[:, c - MOM]
            n = np.isfinite(x).sum(0)
            with np.errstate(all='ignore'):
                mom[g] = np.where(n >= 0.5 * len(idx), np.nanmean(x, 0), np.nan) - m_mom
            cf = c[has_fwd]
            xf = Lg[:, cf + HOLD] - Lg[:, cf]
            n = np.isfinite(xf).sum(0)
            ny[g, has_fwd] = n
            with np.errstate(all='ignore'):
                y21[g, has_fwd] = np.where(n >= 1, np.nanmean(xf, 0), np.nan) - m_fwd[has_fwd]
        out.append(dict(k=k, t0=t0, t1=t1, U=U, lab=lab, rho=rho, sizes=sizes, glabel=glabel, breadth=breadth,
                        mom=mom, y21=y21, ny=ny))
        now = time.time()
        print(f'  k={k} {iso(days[t0])} universe {len(U)} sizes {sizes.min()}-{sizes.max()} '
              f'(<{MIN_MEMBERS}: {(sizes < MIN_MEMBERS).sum()})  {now - prev_t:.1f}s', flush=True)
        prev_t = now
    with open(big(tag) / ('refits.pkl' if limit is None else 'refits_smoke.pkl'), 'wb') as fh:
        pickle.dump(dict(days=days, T=T, N=N, refits=out), fh)
    print(f'done {len(out)} refits {time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None,
         int(sys.argv[3]) if len(sys.argv) > 3 else None)
