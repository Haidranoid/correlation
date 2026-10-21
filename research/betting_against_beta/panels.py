"""Daily betas (Frazzini & Pedersen 2014) and the calendar universe for one export (old or new).

    python3 -I -B panels.py old|new

Writes WORK/panels_<tag>.npz:
  vol   [N, T] float32  sigma_i: sd of daily log returns (export closes) over d-251..d, >= 120 returns
  corr  [N, T] float32  rho_i: correlation of overlapping 3-day log returns log(c_d / c_{d-3}) with SPY's over the
                        3-day returns ending d-1259..d, >= 750 pairs with both present
  beta  [N, T] float32  0.6 * rho_i * sigma_i / sigma_m + 0.4
  sigm  [T]             sigma_m: sd of SPY's daily log returns (store closes) over d-251..d, >= 120
  rebal [K]             twins calendar (export indices 252 + 21k)
  uni   [K, N] bool     the judge-like liquid set minus c2eval.excluded (research/night_and_day/common.py)
  big   [K, N] bool     its 500 largest by 63-day median dollar volume
  dv63  [K, N] float32  63-day median dollar volume
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, np, WORK, EXPORTS, VOL_W, VOL_MIN, COR_W, COR_MIN, SHRINK_W, SHRINK_PRIOR, log,  # noqa: E402
                    spy_close, universe)


def rolling(a, mask, w):
    """Window sums over d-w+1..d of a (where mask) and of mask, via cumsums along time. a, mask: [n, T]."""
    n, T = a.shape
    ca = np.zeros((n, T + 1))
    ca[:, 1:] = np.cumsum(np.where(mask, a, 0.0), 1)
    cm = np.zeros((n, T + 1))
    cm[:, 1:] = np.cumsum(mask, 1)
    lo = np.maximum(np.arange(T) - w + 1, 0)
    hi = np.arange(T) + 1
    return ca[:, hi] - ca[:, lo], cm[:, hi] - cm[:, lo]


def build(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = E['close'].shape
    log(tag, 'N', N, 'T', T)
    sc = spy_close(tag, days)
    log('SPY closes', int(np.isfinite(sc).sum()), 'first', c2eval.iso(days[np.isfinite(sc).argmax()]))
    m = np.full(T, np.nan)
    m3 = np.full(T, np.nan)
    with np.errstate(all='ignore'):
        m[1:] = np.log(sc[1:] / sc[:-1])
        m3[3:] = np.log(sc[3:] / sc[:-3])
    m[~np.isfinite(m)] = np.nan
    m3[~np.isfinite(m3)] = np.nan
    mm = np.isfinite(m)[None, :]
    s1, n1 = rolling(np.nan_to_num(m)[None, :], mm, VOL_W)
    s2, _ = rolling((np.nan_to_num(m) ** 2)[None, :], mm, VOL_W)
    with np.errstate(all='ignore'):
        sigm = np.sqrt(np.maximum(s2 - s1 ** 2 / n1, 0) / (n1 - 1))[0]
    sigm[n1[0] < VOL_MIN] = np.nan

    vol = np.full((N, T), np.nan, np.float32)
    corr = np.full((N, T), np.nan, np.float32)
    CH = 1000
    M3 = np.nan_to_num(m3)
    for a in range(0, N, CH):
        b = min(N, a + CH)
        c = np.asarray(E['close'][a:b], np.float64)
        lr = np.full(c.shape, np.nan)
        x3 = np.full(c.shape, np.nan)
        with np.errstate(all='ignore'):
            lr[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
            x3[:, 3:] = np.log(c[:, 3:] / c[:, :-3])
        lr[~np.isfinite(lr)] = np.nan
        x3[~np.isfinite(x3)] = np.nan
        ok = np.isfinite(lr)
        s1, n = rolling(lr, ok, VOL_W)
        s2, _ = rolling(lr * lr, ok, VOL_W)
        with np.errstate(all='ignore'):
            v = np.sqrt(np.maximum(s2 - s1 ** 2 / n, 0) / (n - 1))
        v[n < VOL_MIN] = np.nan
        vol[a:b] = v
        del lr, ok, s1, s2, n, v
        ok3 = np.isfinite(x3) & np.isfinite(m3)[None, :]
        Mb = np.broadcast_to(M3, c.shape)
        sx, n3 = rolling(x3, ok3, COR_W)
        sm, _ = rolling(Mb, ok3, COR_W)
        sxx, _ = rolling(x3 * x3, ok3, COR_W)
        smm, _ = rolling(Mb * Mb, ok3, COR_W)
        sxm, _ = rolling(x3 * Mb, ok3, COR_W)
        with np.errstate(all='ignore'):
            cov = sxm - sx * sm / n3
            vx = sxx - sx * sx / n3
            vm = smm - sm * sm / n3
            r = cov / np.sqrt(vx * vm)
        r[(n3 < COR_MIN) | ~np.isfinite(r)] = np.nan
        corr[a:b] = r
        del x3, ok3, sx, n3, sm, sxx, smm, sxm, cov, vx, vm, r, c
        log(f'rows {a}..{b}')
    with np.errstate(all='ignore'):
        beta = (SHRINK_W * corr * vol / sigm[None, :].astype(np.float32) + SHRINK_PRIOR).astype(np.float32)
    fb = np.isfinite(beta).any(0)
    log('first day with any beta', c2eval.iso(days[fb.argmax()]))

    tw = c2eval.twins(E)
    uni, big, dv63 = universe(E, tw)
    rebal = np.asarray(tw['rebal'])
    for k in range(0, len(rebal), 12):
        t0 = rebal[k]
        nb = int((uni[k] & np.isfinite(beta[:, t0])).sum())
        if uni[k].sum():
            log(f'  calendar {c2eval.iso(days[t0])}: universe {int(uni[k].sum())}, with beta {nb}')
    np.savez(WORK / f'panels_{tag}.npz', vol=vol, corr=corr, beta=beta, sigm=sigm, rebal=rebal, uni=uni, big=big,
             dv63=dv63, days=days)
    log('saved', WORK / f'panels_{tag}.npz')


if __name__ == '__main__':
    build(sys.argv[1])
