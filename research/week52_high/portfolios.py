"""The overlapping-cohort books (George & Hwang's (6, 6) construction on the 21-day grid).

    python3 -I -B portfolios.py old|new [divfix]   -> WORK/series_<tag>[_divfix].npz

Cohort k is formed at the close of grid day t_k (export index 252 + 21k): among the eligible stocks, long the top 30%
and short the bottom 30% by the signal, equal weight, $1 per leg. Each leg is bought and held to the close of
t_k + 126 (forward-filled export closes, so positions drift with prices). Over a period (a, b] the cohort earns
r_L - r_S, each leg's own growth ratio V(b) / V(a) - 1 (Jegadeesh-Titman / George-Hwang convention), minus costs:
5 bps on each leg's notional at formation (10 bps in its first period) and 5 bps on each leg's drifted value at exit
(5 bps x (1 + r_L) + 5 bps x (1 + r_S) in its last period). No netting across cohorts. The book's period return is
the mean over the cohorts active in it (formed at or before a, exiting at or after b): six from the sixth period on.

Periods: (t_j, t_{j+1}] between grid days. Old export: full periods only (the 9-day stub after the last grid day is
dropped). New export: the period containing 2025-06-27 is split there, so the unseen window 2025-06-30..2026-10-19
is covered exactly, and the stub after the last grid day runs to the last day.

divfix: returns from the total-return index C x exp(cumsum(pubdata.div_fix(tag))) (0 before 2022-07); signals are
unchanged (from the raw export closes).

Variants (eligible = universe at t_k with finite PTH(t_k) and finite 12-1 momentum, the common sample):
  pth, mom, jt6 (GH's 6-month JT, reported), pth_skip (rank on PTH(t_k - 5); eligible: universe, close at t_k,
  finite PTH(t_k - 5), finite momentum), and the same on the 500 largest: pth_big, mom_big, pth_skip_big.
Also a market leg per variant: all eligible stocks, equal weight, held the same way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, np, WORK, EXPORTS, log, ffill, FRAC, HOLD, COST_BPS, MIN_NAMES, UNSEEN_FROM,  # noqa: E402
                    iso_index)
import pubdata  # noqa: E402

HOLD_DAYS = HOLD * 21


def legs_for(sig, elig, seed):
    """Indices of the long (top FRAC) and short (bottom FRAC) legs. Ties broken by a seeded random key."""
    idx = np.nonzero(elig & np.isfinite(sig))[0]
    n = len(idx)
    if n < MIN_NAMES:
        return None, None, idx
    rng = np.random.default_rng(seed)
    key = rng.random(n)
    o = np.lexsort((key, sig[idx]))         # ascending by signal, ties by key
    m = int(round(FRAC * n))
    return idx[o[-m:]], idx[o[:m]], idx


def leg_path(P, members, t0, t1):
    """Equal-weight buy-and-hold value path of a leg from t0 to t1 (inclusive), $1 at t0."""
    X = P[members, t0:t1 + 1]
    return np.nanmean(X / X[:, :1], 0)


def periods_for(tag, days, rebal, T):
    """Breakpoints (a, b] of the periods."""
    bp = list(rebal)
    if tag == 'new':
        cut = iso_index(days, UNSEEN_FROM) - 1           # 2025-06-27, the old export's last day
        bp = sorted(set(bp) | {cut})
        if bp[-1] < T - 1:
            bp.append(T - 1)
    return np.array(bp[:-1]), np.array(bp[1:])


def run(tag, divfix=False):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
    days = E['days']
    N, T = E['close'].shape
    z = np.load(WORK / f'panels_{tag}.npz')
    rebal, uni, big, close_ok = z['rebal'], z['uni'], z['big'], z['close_ok']
    P = ffill(E['close'])
    if divfix:
        D = pubdata.div_fix(tag)
        assert D.shape == (N, T), D.shape
        cs = np.cumsum(np.asarray(D, np.float64), 1)
        P = P * np.exp(cs)
        del cs
        log('divfix: total-return index built')
    pa, pb = periods_for(tag, days, rebal, T)
    J = len(pa)
    log(tag, 'divfix' if divfix else 'raw', 'periods', J, c2eval.iso(days[pa[0]]), '..', c2eval.iso(days[pb[-1]]))
    base = uni & np.isfinite(z['pth']) & np.isfinite(z['mom'])
    variants = {
        'pth': (z['pth'], base),
        'mom': (z['mom'], base),
        'jt6': (z['jt6'], base & np.isfinite(z['jt6'])),
        'pth_skip': (z['pth_skip'], uni & close_ok & np.isfinite(z['pth_skip']) & np.isfinite(z['mom'])),
    }
    for v in list(variants):
        if v == 'jt6':
            continue
        s, e = variants[v]
        variants[v + '_big'] = (s, e & big)
    c = COST_BPS / 1e4
    out = {'pa': pa, 'pb': pb, 'days': days, 'rebal': rebal}
    for name, (S, EL) in variants.items():
        net = np.zeros(J); gross = np.zeros(J); rl = np.zeros(J); rs = np.zeros(J); rm = np.zeros(J)
        ncoh = np.zeros(J, int)
        nleg = np.full(len(rebal), 0, int)
        for k, t0 in enumerate(rebal):
            L, Sh, A = legs_for(S[k], EL[k], seed=1000 + k)
            if L is None:
                continue
            nleg[k] = len(L)
            t1 = min(t0 + HOLD_DAYS, T - 1)
            vl, vs, vm = leg_path(P, L, t0, t1), leg_path(P, Sh, t0, t1), leg_path(P, A, t0, t1)
            js = np.nonzero((pa >= t0) & (pb <= t0 + HOLD_DAYS))[0]
            for j in js:
                a, b = pa[j] - t0, pb[j] - t0
                gl, gs, gm = vl[b] / vl[a] - 1, vs[b] / vs[a] - 1, vm[b] / vm[a] - 1
                cost = 0.0
                if pa[j] == t0:
                    cost += 2 * c
                if pb[j] == t0 + HOLD_DAYS:
                    cost += c * (1 + gl) + c * (1 + gs)
                gross[j] += gl - gs
                net[j] += gl - gs - cost
                rl[j] += gl
                rs[j] += gs
                rm[j] += gm
                ncoh[j] += 1
        with np.errstate(all='ignore'):
            for arr in (net, gross, rl, rs, rm):
                arr /= np.where(ncoh > 0, ncoh, np.nan)
        out[f'{name}_net'] = net
        out[f'{name}_gross'] = gross
        out[f'{name}_long'] = rl
        out[f'{name}_short'] = rs
        out[f'{name}_mkt'] = rm
        out[f'{name}_ncoh'] = ncoh
        out[f'{name}_nleg'] = nleg
        log(f'{name}: periods with 6 cohorts {int((ncoh == HOLD).sum())}, legs median {int(np.median(nleg[nleg > 0]))}')
    # French factors compounded over each period's days a+1..b (RF compounded separately; Mkt excess = Mkt - RF)
    fd, f5 = pubdata.ff5_daily()
    ud, um = pubdata.umd_daily()
    fac = {k: pubdata.on_days(fd, f5[k], days) for k in ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF')}
    fac['UMD'] = pubdata.on_days(ud, um['UMD'], days)
    F = {}
    for k in ('SMB', 'HML', 'RMW', 'CMA', 'UMD', 'RF'):
        F[k] = np.array([np.prod(1 + fac[k][a + 1:b + 1]) - 1 for a, b in zip(pa, pb)])
    mkt = fac['Mkt-RF'] + fac['RF']
    F['Mkt-RF'] = np.array([np.prod(1 + mkt[a + 1:b + 1]) - 1 for a, b in zip(pa, pb)]) - F['RF']
    for k, v in F.items():
        out[f'ff_{k}'] = v       # NaN where any day lacks French data (after 2026-08-31)
    p = WORK / f"series_{tag}{'_divfix' if divfix else ''}.npz"
    np.savez(p, **out)
    log('saved', p)


if __name__ == '__main__':
    run(sys.argv[1], divfix=len(sys.argv) > 2 and sys.argv[2] == 'divfix')
