"""Daily inputs and the calendar universe for one export (old or new).

    python3 -I panels.py old|new

Writes WORK/panels_<tag>.npz:
  vol   [N, T] float32  sd of daily log returns over d-251..d (>= 120 returns)
  corr  [N, T] float32  correlation of overlapping 3-day log returns with the export's market, 3-day returns
                        ending d-1259..d (>= 750 of them)
  beta  [N, T] float32  0.6 * corr * vol / sigm + 0.4
  sigm  [T]             sd of the export's market over d-251..d
  cal   [K]             calendar days (export indices 252 + 21k)
  univ  [K, N] bool     universe at each calendar day (data, price, dollar volume, not fund-like, not another
                        non-common vehicle (common.not_common), valid inputs)
  univ_flag [K, N] bool the plan's universe with the fund-like flag only (sensitivity)
  meddv [K, N] float32  median dollar volume over t0-251..t0 (for the 500 largest)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import time
import warnings
from common import (c2eval, np, WORK, EXPORTS, twins_for, not_common, STEP, CAL0, VOL_W, VOL_MIN, COR_W, COR_MIN, DATA_W,
                    DATA_FRAC, MED_W, MIN_PRICE, MIN_DV)


def rolling(a, mask, w):
    """Window sums over d-w+1..d of a (where mask) and of mask, via cumsums along time. a, mask: [n, T]."""
    n, T = a.shape
    ca = np.zeros((n, T + 1)); ca[:, 1:] = np.cumsum(np.where(mask, a, 0.0), 1)
    cm = np.zeros((n, T + 1)); cm[:, 1:] = np.cumsum(mask, 1)
    lo = np.maximum(np.arange(T) - w + 1, 0)
    hi = np.arange(T) + 1
    return ca[:, hi] - ca[:, lo], cm[:, hi] - cm[:, lo]


def build(tag):
    t_start = time.time()
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume', 'market'))
    N, T = E['close'].shape
    mkt = np.asarray(E['market'], np.float64)
    # market sd over 252 days
    mm = np.isfinite(mkt)[None, :]
    s1, n1 = rolling(np.nan_to_num(mkt)[None, :], mm, VOL_W)
    s2, _ = rolling((np.nan_to_num(mkt) ** 2)[None, :], mm, VOL_W)
    with np.errstate(all='ignore'):
        sigm = np.sqrt((s2 - s1 ** 2 / n1) / (n1 - 1))[0]
    sigm[n1[0] < VOL_MIN] = np.nan
    m3 = np.full(T, np.nan); m3[2:] = mkt[2:] + mkt[1:-1] + mkt[:-2]

    vol = np.full((N, T), np.nan, np.float32)
    corr = np.full((N, T), np.nan, np.float32)
    CH = 1000
    for a in range(0, N, CH):
        b = min(N, a + CH)
        c = np.asarray(E['close'][a:b], np.float64)
        lr = np.full(c.shape, np.nan)
        with np.errstate(all='ignore'):
            lr[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
        ok = np.isfinite(lr)
        s1, n = rolling(lr, ok, VOL_W)
        s2, _ = rolling(lr * lr, ok, VOL_W)
        with np.errstate(all='ignore'):
            v = np.sqrt(np.maximum(s2 - s1 ** 2 / n, 0) / (n - 1))
        v[n < VOL_MIN] = np.nan
        vol[a:b] = v
        x3 = np.full(c.shape, np.nan)
        with np.errstate(all='ignore'):
            x3[:, 3:] = np.log(c[:, 3:] / c[:, :-3])
        ok3 = np.isfinite(x3) & np.isfinite(m3)[None, :]
        M3 = np.broadcast_to(np.nan_to_num(m3), c.shape)
        sx, n3 = rolling(x3, ok3, COR_W)
        sm, _ = rolling(M3, ok3, COR_W)
        sxx, _ = rolling(x3 * x3, ok3, COR_W)
        smm, _ = rolling(M3 * M3, ok3, COR_W)
        sxm, _ = rolling(x3 * M3, ok3, COR_W)
        with np.errstate(all='ignore'):
            cov = sxm - sx * sm / n3
            vx = sxx - sx * sx / n3
            vm = smm - sm * sm / n3
            r = cov / np.sqrt(vx * vm)
        r[(n3 < COR_MIN) | ~np.isfinite(r)] = np.nan
        corr[a:b] = r
        print(f'  rows {a}..{b} done {time.time() - t_start:.0f}s', flush=True)
    with np.errstate(all='ignore'):
        beta = (0.6 * corr * vol / sigm[None, :].astype(np.float32) + 0.4).astype(np.float32)

    # calendar universe
    tw = twins_for(tag, E)
    cal = np.arange(CAL0, T, STEP)
    K = len(cal)
    univ = np.zeros((K, N), bool)
    nc = not_common(E['tickers'])
    meddv = np.full((K, N), np.nan, np.float32)
    univ_flag = np.zeros((K, N), bool)
    rebal = np.asarray(tw['rebal'])
    for k, t0 in enumerate(cal):
        if t0 - DATA_W + 1 < 0:
            continue
        c = np.asarray(E['close'][:, t0 - DATA_W + 1:t0 + 1])
        frac = np.isfinite(c).mean(1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mp = np.nanmedian(c[:, -MED_W:], 1)
            md = np.nanmedian(np.asarray(E['dollar_volume'][:, t0 - MED_W + 1:t0 + 1]), 1)
        kt = np.searchsorted(rebal, t0, side='right') - 1
        assert kt >= 0 and rebal[kt] == t0, (t0, rebal[kt])
        fl = np.asarray(tw['fund_like'][kt], bool)
        u = (frac >= DATA_FRAC) & (np.nan_to_num(mp) >= MIN_PRICE) & (np.nan_to_num(md) >= MIN_DV) & ~fl
        u &= np.isfinite(vol[:, t0]) & np.isfinite(corr[:, t0]) & np.isfinite(beta[:, t0])
        univ[k] = u & ~nc
        univ_flag[k] = u
        meddv[k] = md
    out = WORK / f'panels_{tag}.npz'
    np.savez(out, vol=vol, corr=corr, beta=beta, sigm=sigm, cal=cal, univ=univ, univ_flag=univ_flag, meddv=meddv, not_common=nc)
    print(f'{tag}: wrote {out} in {time.time() - t_start:.0f}s; universe sizes from first nonempty:',
          [(c2eval.iso(E['days'][cal[k]]), int(univ[k].sum())) for k in range(0, K, 12) if univ[k].sum()][:30])


if __name__ == '__main__':
    build(_sys.argv[1])
