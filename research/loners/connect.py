"""Connectedness c for one export (old or new): the universe and eligible funds on the calendar, and c every day.

    python3 -I connect.py old|new

At each calendar day t0 (c2eval.twins' rebalance days, export indices 252 + 21k):
  universe U[k]   the judge-like liquid set at t0 (v finite on >= 90% of t0-755..t0; upper median close >= $5 and
                  dollar volume >= $5M over t0-251..t0; a close at t0) minus c2eval.excluded(E, tw)[k]
  funds FE[k]     store ETFs passing c2eval.twins' fund screens over t0-251..t0 (>= 90% returns, |log| > 0.5 missing;
                  median dollar volume >= $5M; volatility <= 75% a year; not single-name by twins' rule), minus
                  c2eval.not_common(fund tickers)
  meddv, vol      252-day upper-median dollar volume; sd of daily log returns over t0-251..t0 (>= 120)
  BAC inputs      vol, correlation of overlapping 3-day log returns with the market over 1,260 days (>= 750), and
                  beta = 0.6 corr vol / sd_market + 0.4 (research/betting_against_correlation's definitions)
Every day d: c[j, d] = the highest correlation of stock j's market-relative daily log returns over d-251..d with any
other stock of U[k(d)] or fund of FE[k(d)] (k(d) = the latest calendar day <= d), both series with >= 90% returns in
the window. On calendar days also the best match.

Writes WORK/connect_<tag>.npz and WORK/c_daily_<tag>.npy ([N, T] float32).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import time
import warnings
from common import (c2eval, np, WORK, EXPORTS, log, calendar, upper_median, log_returns, standardize, market_relative,
                    WIN, MIN_FRAC, DATA_W, DATA_FRAC, MED_W, MIN_PRICE, MIN_DV, FUND_MIN_DV, FUND_MAX_VOL,
                    FUND_MAX_ABS, SINGLE_BEST, SINGLE_GAP, VOL_W, VOL_MIN, COR_W, COR_MIN)


def build(tag):
    t_start = time.time()
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume', 'v', 'market'))
    tw = c2eval.twins(E)
    F = tw['F']
    N, T = E['close'].shape
    cal = calendar(T)
    assert np.array_equal(cal, np.asarray(tw['rebal'])), 'calendar differs from c2eval.twins'
    K = len(cal)
    exc = c2eval.excluded(E, tw)
    nc_f = c2eval.not_common(F['tickers'])
    log(f'{tag}: N={N} T={T} K={K} funds={len(F["ids"])} (not_common funds {int(nc_f.sum())}) '
        f'{time.time() - t_start:.0f}s')

    close = np.asarray(E['close'], np.float32)
    dv = np.asarray(E['dollar_volume'], np.float32)
    mkt = np.asarray(E['market'], np.float64)
    sr = log_returns(close).astype(np.float32)                       # [N, T]
    fr = log_returns(F['close'], max_abs=FUND_MAX_ABS).astype(np.float32)
    fdv = np.asarray(F['dollar_volume'], np.float32)
    vfin = np.zeros((N, T + 1), np.int32)
    np.cumsum(np.isfinite(np.asarray(E['v'])), 1, out=vfin[:, 1:])
    log(f'returns ready {time.time() - t_start:.0f}s')

    U = np.zeros((K, N), bool)
    FE = np.zeros((K, len(F['ids'])), bool)
    meddv = np.full((K, N), np.nan, np.float32)
    vol = np.full((K, N), np.nan, np.float32)
    corr3 = np.full((K, N), np.nan, np.float32)
    beta_bac = np.full((K, N), np.nan, np.float32)
    n_single = np.zeros(K, int)
    n_fok = np.zeros(K, int)
    for k, t0 in enumerate(cal):
        a, b = t0 - WIN + 1, t0 + 1
        # universe
        if t0 - DATA_W + 1 >= 0:
            cnt = vfin[:, t0 + 1] - vfin[:, t0 + 1 - DATA_W]
            cand = np.nonzero((cnt * 10 >= DATA_W * 9) & np.isfinite(close[:, t0]) & (np.nan_to_num(close[:, t0]) > 0)
                              & ~exc[k])[0]
            if len(cand):
                mp = upper_median(close[cand, t0 - MED_W + 1:t0 + 1])
                md = upper_median(dv[cand, t0 - MED_W + 1:t0 + 1])
                ok = (mp >= MIN_PRICE) & (md >= MIN_DV)
                U[k, cand[ok]] = True
                meddv[k, cand[ok]] = md[ok]
        # volatility (BAC's: sd of daily log returns over t0-251..t0, >= 120 returns)
        W = sr[:, t0 - VOL_W + 1:t0 + 1].astype(np.float64)
        n = np.isfinite(W).sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v = np.nanstd(W, 1, ddof=1)
        v[n < VOL_MIN] = np.nan
        vol[k] = v
        # BAC inputs: 3-day log returns vs the market's 3-day sum over 1,260 days (>= 750 pairs)
        if t0 - COR_W - 3 >= 0:
            lo = t0 - COR_W + 1
            c_hi = close[:, lo:t0 + 1].astype(np.float64)
            c_lo = close[:, lo - 3:t0 - 2].astype(np.float64)
            with np.errstate(all='ignore'):
                x3 = np.log(c_hi / c_lo)
            m3 = mkt[lo:t0 + 1] + mkt[lo - 1:t0] + mkt[lo - 2:t0 - 1]
            ok3 = np.isfinite(x3) & np.isfinite(m3)[None, :]
            n3 = ok3.sum(1)
            X = np.where(ok3, x3, 0.0)
            M = np.where(ok3, m3[None, :], 0.0)
            with np.errstate(all='ignore'):
                sx, sm = X.sum(1), M.sum(1)
                cov = (X * M).sum(1) - sx * sm / n3
                vx = (X * X).sum(1) - sx * sx / n3
                vm = (M * M).sum(1) - sm * sm / n3
                r = cov / np.sqrt(vx * vm)
            r[(n3 < COR_MIN) | ~np.isfinite(r)] = np.nan
            corr3[k] = r
            mw = mkt[t0 - VOL_W + 1:t0 + 1]
            mw = mw[np.isfinite(mw)]
            sigm = mw.std(ddof=1) if len(mw) >= VOL_MIN else np.nan
            with np.errstate(all='ignore'):
                beta_bac[k] = 0.6 * r * vol[k] / sigm + 0.4
        # eligible funds, replicating c2eval.twins (raw returns)
        S = sr[:, a:b].astype(np.float64)
        Fw = fr[:, a:b].astype(np.float64)
        s_ok = np.isfinite(S).mean(1) >= MIN_FRAC
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f_ok = ((np.isfinite(Fw).mean(1) >= MIN_FRAC) & (np.nanmedian(fdv[:, a:b], 1) >= FUND_MIN_DV)
                    & (np.nanstd(Fw, 1) * np.sqrt(252) <= FUND_MAX_VOL))
        si, fi = np.nonzero(s_ok)[0], np.nonzero(f_ok)[0]
        n_fok[k] = len(fi)
        if len(si) and len(fi):
            Zs = standardize(S[si])
            Zf = standardize(Fw[fi])
            C = Zs @ Zf.T
            like = np.asarray(tw['fund_like'][k, si], bool)
            Creal = C[~like]
            single = np.zeros(len(fi), bool)
            if Creal.shape[0] >= 2:
                top2 = np.sort(Creal, 0)[-2:]
                single = (top2[1] >= SINGLE_BEST) & (top2[1] - top2[0] > SINGLE_GAP)
            n_single[k] = int(single.sum())
            FE[k, fi[~single]] = True
            FE[k] &= ~nc_f
            del C, Creal, Zs, Zf
        if k % 24 == 0:
            log(f'  cal {k}/{K} {c2eval.iso(E["days"][t0])}: universe {int(U[k].sum())}, funds ok {n_fok[k]}, '
                f'single {n_single[k]}, eligible {int(FE[k].sum())}  {time.time() - t_start:.0f}s')
    del vfin

    # daily c
    c_daily = np.full((N, T), np.nan, np.float32)
    best = np.full((K, N), -10 ** 9, np.int64)       # >= 0: stock row; < 0: -(fund row + 1)
    c_stock = np.full((K, N), np.nan, np.float32)    # best among stocks only (calendar days)
    c_fund = np.full((K, N), np.nan, np.float32)     # best among funds only
    n_scored = np.zeros(T, int)
    first_k = int(np.argmax(U.sum(1) > 0))
    for d in range(cal[first_k], T):
        k = int(np.searchsorted(cal, d, side='right')) - 1
        rows = np.nonzero(U[k])[0]
        frows = np.nonzero(FE[k])[0]
        a, b = d - WIN + 1, d + 1
        S = sr[rows, a:b].astype(np.float64)
        rows = rows[np.isfinite(S).mean(1) >= MIN_FRAC]
        S = sr[rows, a:b].astype(np.float64)
        Fw = fr[frows, a:b].astype(np.float64)
        frows = frows[np.isfinite(Fw).mean(1) >= MIN_FRAC]
        Fw = fr[frows, a:b].astype(np.float64)
        m = mkt[a:b]
        Rs, _ = market_relative(S, m)
        Rf, _ = market_relative(Fw, m)
        Zs = standardize(Rs).astype(np.float32)
        Zf = standardize(Rf).astype(np.float32)
        good = np.linalg.norm(Zs, axis=1) > 0.5
        ns = len(rows)
        Css = Zs @ Zs.T
        np.fill_diagonal(Css, -np.inf)
        Css[:, ~good] = -np.inf
        Csf = Zs @ Zf.T if len(frows) else np.full((ns, 0), -np.inf, np.float32)
        bs = Css.max(1) if ns > 1 else np.full(ns, -np.inf, np.float32)
        bf = Csf.max(1) if Csf.shape[1] else np.full(ns, -np.inf, np.float32)
        c = np.maximum(bs, bf)
        c[~good] = np.nan
        c_daily[rows, d] = c
        n_scored[d] = int(np.isfinite(c).sum())
        if cal[k] == d:
            js = Css.argmax(1) if ns > 1 else np.zeros(ns, int)
            jf = Csf.argmax(1) if Csf.shape[1] else np.zeros(ns, int)
            bb = np.where(bs >= bf, rows[js], -(frows[jf] + 1) if len(frows) else -10 ** 9)
            best[k, rows] = np.where(good, bb, -10 ** 9)
            c_stock[k, rows] = np.where(good, bs, np.nan)
            c_fund[k, rows] = np.where(good & np.isfinite(bf), bf, np.nan)
            if k % 12 == 0:
                cc = c[np.isfinite(c)]
                log(f'  {c2eval.iso(E["days"][d])}: scored {len(cc)}, funds {len(frows)}, c median {np.median(cc):.3f} '
                    f'p10 {np.quantile(cc, 0.1):.3f} p90 {np.quantile(cc, 0.9):.3f}  {time.time() - t_start:.0f}s')
        del Css, Csf
    np.save(WORK / f'c_daily_{tag}.npy', c_daily)
    np.savez(WORK / f'connect_{tag}.npz', cal=cal, U=U, FE=FE, meddv=meddv, vol=vol, corr3=corr3, beta_bac=beta_bac,
             best=best, c_stock=c_stock, c_fund=c_fund, n_scored=n_scored, n_single=n_single, n_fok=n_fok,
             fund_tickers=F['tickers'], fund_ids=F['ids'], nc_f=nc_f)
    log(f'{tag}: done in {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    build(_sys.argv[1])
