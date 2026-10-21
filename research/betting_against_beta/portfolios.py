"""BAB (Frazzini & Pedersen 2014), the same on the 500 largest, and Novy-Marx & Velikov's plain versions, as monthly
return series (the .md, 'Rules' and 'Decisions').

    python3 -I -B portfolios.py old|new [divfix]

Formation at the close of each month's last trading day t, held (buy and hold) to the close of the next month's last
trading day. Writes WORK/series_<tag>[_divfix].npz: months [M] ('YYYY-MM', the return month), rf [M], and for each
strategy S: S__gross, S__cost, S__net, S__RL, S__RH, S__bL, S__bH, S__levL, S__levH, S__nL, S__nH, S__turn [M]
(NaN where no book).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, WORK, EXPORTS, COST_BPS, MIN_UNIV, log, rf_daily, month_ends, ffill  # noqa: E402

STRATS = ('bab', 'bab500', 'bab_unlev', 'nmv_ew', 'nmv_dv')


def avg_rank(x):
    """Average ranks (ties share the mean rank), 1-based."""
    o = np.argsort(x, kind='stable')
    xs = x[o]
    starts = np.r_[0, np.nonzero(np.diff(xs) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(xs)]
    r = np.empty(len(x))
    r[o] = np.repeat((starts + ends - 1) / 2.0 + 1, ends - starts)
    return r


def rank_weights(b):
    """FP's rank weights: z = rank, k = 2 / sum|z - zbar|; high-beta weights k(z - zbar)+, low-beta k(z - zbar)-.
    Each leg sums to 1."""
    z = avg_rank(b)
    d = z - z.mean()
    k = 2.0 / np.abs(d).sum()
    return k * np.maximum(-d, 0), k * np.maximum(d, 0)


def thirds(b, w=None):
    """Low-beta third (rank < n/3) and high-beta third (rank >= 2n/3); equal weight, or proportional to w."""
    n = len(b)
    r = np.argsort(np.argsort(b, kind='mergesort'), kind='mergesort')
    lo, hi = r < n / 3, r >= 2 * n / 3
    wl = np.where(lo, 1.0 if w is None else w, 0.0)
    wh = np.where(hi, 1.0 if w is None else w, 0.0)
    return wl / wl.sum(), wh / wh.sum()


def prices(tag, E, divfix):
    P = ffill(E['close'])
    if divfix:
        import pubdata
        P = pubdata_fix(pubdata, tag, E, P)
    return P


def pubdata_fix(pubdata, tag, E, P):
    """Forward-filled closes with the shared dividend correction added to daily log returns (pubdata.div_fix is 0
    before 2022-07-01): P'_d = P_d * exp(sum of div_fix through d)."""
    fix = np.asarray(pubdata.div_fix(tag), np.float64)
    assert fix.shape == P.shape, (fix.shape, P.shape)
    fix[~np.isfinite(fix)] = 0.0
    np.cumsum(fix, axis=1, out=fix)
    log('div_fix: stock-days with a correction', int((np.diff(fix, axis=1) != 0).sum()))
    return P * np.exp(fix)


def run(tag, divfix=False):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
    z = np.load(WORK / f'panels_{tag}.npz')
    days = E['days']
    T = len(days)
    P = prices(tag, E, divfix)
    log('prices ready')
    rfd = rf_daily(days)
    beta, rebal, U, B, dv63 = z['beta'], z['rebal'], z['uni'], z['big'], z['dv63']
    me, months = month_ends(days)
    M = len(me) - 1                       # return months: months[1..]
    out = {'months': months[1:], 'form_day': days[me[:-1]], 'rf': np.full(M, np.nan)}
    cols = ('gross', 'cost', 'net', 'RL', 'RH', 'bL', 'bH', 'levL', 'levH', 'nL', 'nH', 'turn')
    for s in STRATS:
        for c in cols:
            out[f'{s}__{c}'] = np.full(M, np.nan)
    prev = {s: {} for s in STRATS}
    for i in range(M):
        t, t1 = me[i], me[i + 1]
        k = int(np.searchsorted(rebal, t, side='right')) - 1
        if k < 0:
            continue
        bt = beta[:, t]
        base = U[k] & np.isfinite(bt) & np.isfinite(P[:, t])
        if base.sum() < MIN_UNIV:
            continue
        rf = float(np.prod(1 + rfd[t + 1:t1 + 1]) - 1)
        out['rf'][i] = rf
        for s in STRATS:
            mem = base & B[k] if s == 'bab500' else base
            m = np.nonzero(mem)[0]
            if len(m) < MIN_UNIV:
                continue
            b = bt[m].astype(np.float64)
            if s in ('bab', 'bab500', 'bab_unlev'):
                wl, wh = rank_weights(b)
            elif s == 'nmv_ew':
                wl, wh = thirds(b)
            else:
                w = np.nan_to_num(dv63[k, m].astype(np.float64))
                wl, wh = thirds(b, w)
            bL, bH = float(wl @ b), float(wh @ b)
            levL, levH = (1 / bL, 1 / bH) if s in ('bab', 'bab500') else (1.0, 1.0)
            r = P[m, t1] / P[m, t] - 1
            assert np.isfinite(r).all(), (s, t)
            RL, RH = float(wl @ r), float(wh @ r)
            gross = levL * (RL - rf) - levH * (RH - rf)
            new = {}
            end = {}
            for j, a, bb, ri in zip(m, wl, wh, r):
                p = levL * a - levH * bb
                if p != 0:
                    new[j] = p
                    end[j] = levL * a * (1 + ri) / (1 + RL) - levH * bb * (1 + ri) / (1 + RH)
            pv = prev[s]
            turn = sum(abs(new.get(j, 0.0) - pv.get(j, 0.0)) for j in set(new) | set(pv))
            prev[s] = end
            cost = COST_BPS / 1e4 * turn
            for c, v in zip(cols, (gross, cost, gross - cost, RL, RH, bL, bH, levL, levH, int((wl > 0).sum()),
                                   int((wh > 0).sum()), turn)):
                out[f'{s}__{c}'][i] = v
    name = f'series_{tag}' + ('_divfix' if divfix else '')
    np.savez(WORK / f'{name}.npz', **out)
    for s in STRATS:
        ok = np.isfinite(out[f'{s}__gross'])
        log(f'{tag} {s}: {int(ok.sum())} months, first {out["months"][ok][0]}, last {out["months"][ok][-1]}; '
            f'mean names per leg {np.nanmean(out[f"{s}__nL"]):.0f}/{np.nanmean(out[f"{s}__nH"]):.0f}; '
            f'mean ex-ante leg betas {np.nanmean(out[f"{s}__bL"]):.2f}/{np.nanmean(out[f"{s}__bH"]):.2f}; '
            f'mean turnover {np.nanmean(out[f"{s}__turn"][ok][1:]):.2f}')
    log('saved', WORK / f'{name}.npz')


if __name__ == '__main__':
    run(sys.argv[1], divfix=len(sys.argv) > 2 and sys.argv[2] == 'divfix')
