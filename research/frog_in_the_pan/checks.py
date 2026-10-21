"""Construction checks, run before any test is read.

    python3 -I -B checks.py signals     # PRET and ID recomputed by a plain loop for 300 random stock-days (old, plain
                                        # and divfix); ID's distribution in the formation sorts
    python3 -I -B checks.py umd         # after books.py: our MOM (top - bottom fifth) vs French's UMD over the same
                                        # 21-day periods, 2006-2019 and 2020-2025-06
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, np, pubdata, WORK, OUT, EXPORTS, FAR, NEAR, MIN_VALID, MAX_LOG, WINDOWS,  # noqa: E402
                    log, window_mask, ols)


def hand(close, j, t):
    """PRET and ID by a plain loop over closes t-FAR .. t-NEAR (and earlier closes for the forward fill)."""
    c = [float(x) for x in close[j, :t + 1]]
    last = None
    P = []
    for d in range(t + 1):
        if math.isfinite(c[d]) and c[d] > 0:
            last = c[d]
            P.append((last, True))
        else:
            P.append((last, False))
    s, nv, npos, nneg = 0.0, 0, 0, 0
    for d in range(t - FAR + 1, t - NEAR + 1):
        p1, ok1 = P[d]
        p0, _ = P[d - 1]
        if not ok1 or p0 is None or p1 is None:
            continue
        r = math.log(p1 / p0)
        if abs(r) > MAX_LOG:
            continue
        nv += 1
        s += r
        npos += r > 0
        nneg += r < 0
    if nv < MIN_VALID:
        return float('nan'), float('nan')
    sg = 0 if abs(s) < 1e-9 else (s > 0) - (s < 0)
    return s, sg * (nneg - npos) / nv


def signals():
    lines = []
    E = c2eval.load_export(EXPORTS['old'], arrays=('close',))
    close = E['close']
    N, T = close.shape
    rng = np.random.default_rng(7)
    for px in ('plain', 'divfix'):
        PRET = np.load(WORK / f'pret_old_{px}.npy', mmap_mode='r')
        ID = np.load(WORK / f'id_old_{px}.npy', mmap_mode='r')
        fix = pubdata.div_fix('old') if px == 'divfix' else None
        worst_p = worst_i = 0.0
        nan_mismatch = 0
        checked = 0
        lo = c2eval.day_index(E['days'], '2022-08-01') if px == 'divfix' else FAR
        while checked < 300:
            j = int(rng.integers(N))
            t = int(rng.integers(lo, T))
            fc = None
            if fix is not None:
                fc = np.cumsum(np.nan_to_num(np.asarray(fix[j, :t + 1], np.float64)))
            hp, hi = hand(close, j, t) if fc is None else hand_fix(close, fc, j, t)
            p, i = float(PRET[j, t]), float(ID[j, t])
            if math.isnan(hp) != math.isnan(p):
                nan_mismatch += 1
                checked += 1
                continue
            if math.isnan(hp):
                continue
            worst_p = max(worst_p, abs(hp - p))
            worst_i = max(worst_i, abs(hi - i))
            if abs(hi - i) > 1e-6:
                lines.append(f'  {px} ID mismatch: {E["tickers"][j]} day {c2eval.iso(E["days"][t])}: loop PRET {hp:.3e} '
                             f'ID {hi:+.4f}; panel PRET {p:.3e} ID {i:+.4f}')
            checked += 1
        lines.append(f'{px}: 300 random stock-days (finite in the loop or the panel): NaN mismatches {nan_mismatch}; '
                     f'max |PRET diff| {worst_p:.2e}; max |ID diff| {worst_i:.2e}')
        log(lines[-1])
    # ID in the formation sorts (books must exist for this part; else skipped)
    p = WORK / 'books_old_plain.npz'
    if p.exists():
        z = np.load(p)
        st = z['start_iso']
        m = window_mask(st, '2005-12-01', '2019-12-31')
        mid = z['all__mean_id'][m]
        mp = z['all__mean_pret'][m]
        lines.append('mean ID by cell over formations 2005-12..2019-12 (rows = PRET fifth 1 low .. 5 high; cols = ID third '
                     'low, mid, high); mean PRET (log) in brackets:')
        for f in range(5):
            lines.append('  fifth %d: ' % (f + 1) + '  '.join(f'{np.nanmean(mid[:, f * 3 + c]):+.3f} ({np.nanmean(mp[:, f * 3 + c]):+.3f})'
                                                           for c in range(3)))
        cnt = z['all__cnt'][m]
        lines.append(f'names per cell: median {int(np.median(cnt))}, min {int(cnt[cnt > 0].min())}')
    (OUT / 'checks_signals.txt').write_text('\n'.join(lines) + '\n')


def hand_fix(close, fc, j, t):
    """hand() with the total-return factor exp(fc) applied to the closes."""
    c = np.asarray(close[j, :t + 1], np.float64)
    adj = c * np.exp(fc)
    return hand(adj[None, :], 0, t)


def umd():
    lines = []
    dates, F = pubdata.umd_daily()
    for px in ('plain', 'divfix'):
        z = np.load(WORK / f'books_old_{px}.npz')
        E = c2eval.load_export(EXPORTS['old'], arrays=())
        days = E['days']
        u = pubdata.on_days(dates, F['UMD'], days)
        rebal = z['rebal']
        K = len(rebal)
        umd21 = np.full(K, np.nan)
        for k in range(K - 1):
            x = u[rebal[k] + 1:rebal[k + 1] + 1]
            if np.isfinite(x).all():
                umd21[k] = np.prod(1 + x) - 1
        for name, (a, b) in WINDOWS.items():
            m = window_mask(z['end_iso'], a, b) & np.isfinite(umd21)
            for v in ('all', 'big'):
                y = z[f'{v}__MOM__gross21'][m]
                ok = np.isfinite(y)
                r = np.corrcoef(y[ok], umd21[m][ok])[0, 1]
                bb, tt, n, r2 = ols(y[ok], umd21[m][ok])
                lines.append(f'{px} {v:3s} {name:24s}: corr(MOM, UMD) {r:+.2f}, slope {bb[1]:.2f} (t {tt[1]:.1f}), '
                             f'n {n}; mean MOM {np.mean(y[ok]) * 100:+.2f}%, mean UMD {np.mean(umd21[m][ok]) * 100:+.2f}% per period')
                log(lines[-1])
    (OUT / 'checks_umd.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    {'signals': signals, 'umd': umd}[sys.argv[1]]()
