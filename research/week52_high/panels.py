"""Panels for the 52-week high: daily PTH, the universe and the 500 largest on the twins grid, and the formation
signals on grid days.

    python3 -I -B panels.py old|new   -> WORK/pth_<tag>.npy (daily PTH [N, T] float32), WORK/panels_<tag>.npz

PTH[j, t] = close[j, t] / max(close[j, t-251..t]) from the export's (split-adjusted) closes; NaN unless close_t is
present and at least 227 of the 252 closes are. Signals on grid day t_k (export index 252 + 21k):
  pth      PTH at t_k
  pth_skip PTH at t_k - 5 (the one-week-skip variant: ranked a week before the book is formed at t_k)
  mom      12-1 momentum: log(Cf[t_k - 21] / Cf[t_k - 252]), Cf = forward-filled export closes
  jt6      GH's Table I JT signal (reported only): log(Cf[t_k] / Cf[t_k - 126])
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, np, WORK, OUT, EXPORTS, log, universe, rolling_max, ffill, SKIP, MOM_FAR, MOM_NEAR,  # noqa: E402
                    JT6)


def build(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = E['close'].shape
    log(tag, 'N', N, 'T', T, c2eval.iso(days[0]), '..', c2eval.iso(days[-1]))
    hmax = rolling_max(E['close'])
    c = np.asarray(E['close'], np.float32)
    with np.errstate(all='ignore'):
        pth = (c / hmax).astype(np.float32)
    del hmax
    pth[~np.isfinite(pth)] = np.nan
    log('PTH finite stock-days', int(np.isfinite(pth).sum()), 'max', float(np.nanmax(pth)), 'min', float(np.nanmin(pth)))
    np.save(WORK / f'pth_{tag}.npy', pth)
    tw = c2eval.twins(E)
    rebal = tw['rebal']
    uni, big = universe(E, tw)
    K = len(rebal)
    Cf = ffill(E['close'])
    sig = {k: np.full((K, N), np.nan, np.float32) for k in ('pth', 'pth_skip', 'mom', 'jt6')}
    close_ok = np.zeros((K, N), bool)
    with np.errstate(all='ignore'):
        for k, t in enumerate(rebal):
            sig['pth'][k] = pth[:, t]
            if t - SKIP >= 0:
                sig['pth_skip'][k] = pth[:, t - SKIP]
            if t - MOM_FAR >= 0:
                sig['mom'][k] = np.log(Cf[:, t - MOM_NEAR] / Cf[:, t - MOM_FAR])
            if t - JT6 >= 0:
                sig['jt6'][k] = np.log(Cf[:, t] / Cf[:, t - JT6])
            close_ok[k] = np.isfinite(c[:, t])
    for v in sig.values():
        v[~np.isfinite(v)] = np.nan
    # data checks (no returns): universe size, ties at PTH = 1
    lines = []
    for k in range(0, K, 12):
        u = uni[k]
        if not u.any():
            continue
        p = sig['pth'][k][u]
        ok = np.isfinite(p)
        lines.append(f"{c2eval.iso(days[rebal[k]])} universe {int(u.sum())} with PTH {int(ok.sum())} "
                     f"PTH==1 {int((p[ok] == 1).sum())} median PTH {np.median(p[ok]) if ok.any() else np.nan:.3f} "
                     f"with mom {int(np.isfinite(sig['mom'][k][u]).sum())} big {int(big[k].sum())}")
    (OUT / f'panels_{tag}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    np.savez(WORK / f'panels_{tag}.npz', days=days, rebal=rebal, uni=uni, big=big, close_ok=close_ok, **sig)
    log('saved')


if __name__ == '__main__':
    build(sys.argv[1])
