"""WML (12-1 momentum deciles, equal weight) and RMOM (Barroso & Santa-Clara 2015 scaling) as daily and monthly
series, on the judge-like liquid set ('all') and its 500 largest ('big'). The rules are in the .md.

    python3 -I -B build.py old|new plain|divfix      -> WORK/series_<tag>_<px>.npz

At the close of each calendar month's last trading day t (formation):
  - score = ln(C[t-21] / C[t-252]) (stockai/stockai/baselines/momentum.py), export closes (times the dividend
    correction's total-return factor for 'divfix'); needs both closes and the close at t;
  - universe = the liquid set at the latest twins rebalance at or before t (minus c2eval.excluded);
  - long the top tenth, short the bottom tenth, equal weight, buy and hold to the next month-end t1.
Daily leg value V_d = mean_j P_j,d / P_j,t (forward-filled closes), leg return R_d = V_d / V_{d-1} - 1, daily
WML_d = R_W,d - R_L,d on days t+1..t1; monthly WML = V_W,t1 - V_L,t1.
Scale at t: sigma^2 = 252 * mean(WML_d^2) over the 126 trading days t-125..t (daily WML of the books held then; all
126 must exist), w = 0.12 / sigma, held for the month. RMOM = w * WML.
Costs: 5 bps per side x turnover, turnover = sum over names of |new position - drifted old position|, positions in
units of capital (x w for RMOM, so changes in the scale are traded); charged on the month's first holding day.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, np, WORK, EXPORTS, SKIP, LOOKBACK, DECILE, MIN_RANKED, VOL_W, TARGET, COST_BPS,  # noqa: E402
                    log, universe_cached, month_ends, ffill, divfix_factor)

VARIANTS = ('all', 'big')


def run(tag, px):
    assert px in ('plain', 'divfix')
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = E['close'].shape
    rebal, U, B = universe_cached(tag, E)
    log('universe ready; sizes median', int(np.median(U.sum(1)[U.sum(1) > 0])))
    me, ym = month_ends(days)
    M = len(me)
    # scores at month-ends [N, M]
    C = E['close']
    cols = sorted(set(me.tolist()) | {t - SKIP for t in me if t >= LOOKBACK} | {t - LOOKBACK for t in me if t >= LOOKBACK})
    F = divfix_factor(tag) if px == 'divfix' else None
    colv = {}
    for c in cols:
        x = np.asarray(C[:, c], np.float64)
        if F is not None:
            x = x * F[:, c]
        colv[c] = x
    score = np.full((N, M), np.nan)
    close_t = np.zeros((N, M), bool)
    for i, t in enumerate(me):
        close_t[:, i] = np.isfinite(colv[t]) & (colv[t] > 0)
        if t < LOOKBACK:
            continue
        a, b = colv[t - LOOKBACK], colv[t - SKIP]
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        score[ok, i] = np.log(b[ok] / a[ok])
    del colv
    P = ffill(C)
    if F is not None:
        P *= F
        del F
    log('prices ready', px)
    out = {'days': days, 'me': me, 'months': ym}
    for v in VARIANTS:
        wml = np.full(T, np.nan)
        RW = np.full(T, np.nan)
        RL = np.full(T, np.nan)
        m_wml = np.full(M, np.nan)       # monthly WML for the month ENDING at me[i] (book formed at me[i-1])
        m_RW = np.full(M, np.nan)
        m_RL = np.full(M, np.nan)
        nW = np.zeros(M, int)
        nL = np.zeros(M, int)
        nR = np.zeros(M, int)
        books = {}                       # i (return month index) -> (winners, losers, endW, endL)
        for i in range(M - 1):
            t, t1 = me[i], me[i + 1]
            k = int(np.searchsorted(rebal, t, side='right')) - 1
            if k < 0:
                continue
            base = U[k] & np.isfinite(score[:, i]) & close_t[:, i]
            if v == 'big':
                base &= B[k]
            m = np.nonzero(base)[0]
            if len(m) < MIN_RANKED:
                continue
            o = m[np.argsort(score[m, i], kind='stable')]
            nd = int(np.floor(len(m) * DECILE))
            los, win = o[:nd], o[-nd:]
            rel_w = P[win, t:t1 + 1] / P[win, t][:, None]
            rel_l = P[los, t:t1 + 1] / P[los, t][:, None]
            assert np.isfinite(rel_w).all() and np.isfinite(rel_l).all(), (v, i)
            VW, VL = rel_w.mean(0), rel_l.mean(0)
            rw, rl = VW[1:] / VW[:-1] - 1, VL[1:] / VL[:-1] - 1
            RW[t + 1:t1 + 1], RL[t + 1:t1 + 1] = rw, rl
            wml[t + 1:t1 + 1] = rw - rl
            j = i + 1
            m_RW[j], m_RL[j] = VW[-1] - 1, VL[-1] - 1
            m_wml[j] = VW[-1] - VL[-1]
            nW[j], nL[j], nR[j] = len(win), len(los), len(m)
            books[j] = (win, los, rel_w[:, -1] / VW[-1] / len(win), rel_l[:, -1] / VL[-1] / len(los))
        # scale at each formation me[i] for return month j = i + 1
        sigma = np.full(M, np.nan)
        for i in range(M - 1):
            t = me[i]
            if t - VOL_W + 1 < 0:
                continue
            x = wml[t - VOL_W + 1:t + 1]
            if np.isfinite(x).all():
                sigma[i + 1] = np.sqrt(252.0 * np.mean(x * x))
        scale = TARGET / sigma
        scale[~np.isfinite(m_wml)] = np.nan
        # turnover and costs
        res = {}
        for s_name, sc in (('wml', np.where(np.isfinite(m_wml), 1.0, np.nan)), ('rmom', scale)):
            turn = np.full(M, np.nan)
            prev = {}
            for j in range(M):
                if j not in books or not np.isfinite(sc[j]):
                    prev = {}
                    continue
                win, los, endW, endL = books[j]
                new = {}
                for a in win:
                    new[a] = new.get(a, 0.0) + sc[j] / len(win)
                for a in los:
                    new[a] = new.get(a, 0.0) - sc[j] / len(los)
                turn[j] = sum(abs(new.get(a, 0.0) - prev.get(a, 0.0)) for a in set(new) | set(prev))
                end = {}
                for a, e in zip(win, endW):
                    end[a] = end.get(a, 0.0) + sc[j] * e
                for a, e in zip(los, endL):
                    end[a] = end.get(a, 0.0) - sc[j] * e
                prev = end
            cost = COST_BPS / 1e4 * turn
            gross_m = sc * m_wml
            gross_d = np.full(T, np.nan)
            net_d = np.full(T, np.nan)
            for j in range(M):
                if not np.isfinite(gross_m[j]):
                    continue
                t, t1 = me[j - 1], me[j]
                gross_d[t + 1:t1 + 1] = sc[j] * wml[t + 1:t1 + 1]
                net_d[t + 1:t1 + 1] = gross_d[t + 1:t1 + 1]
                net_d[t + 1] -= cost[j]
            res[s_name] = dict(turn=turn, cost=cost, gross_m=gross_m, net_m=gross_m - cost, gross_d=gross_d, net_d=net_d)
        for s_name, d in res.items():
            for c, a in d.items():
                out[f'{v}__{s_name}__{c}'] = a
        out[f'{v}__wml_d'] = wml
        out[f'{v}__RW_d'], out[f'{v}__RL_d'] = RW, RL
        out[f'{v}__RW_m'], out[f'{v}__RL_m'] = m_RW, m_RL
        out[f'{v}__sigma'], out[f'{v}__scale'] = sigma, scale
        out[f'{v}__nW'], out[f'{v}__nL'], out[f'{v}__nR'] = nW, nL, nR
        okm = np.isfinite(m_wml)
        oks = np.isfinite(scale)
        log(f'{tag} {px} {v}: WML months {int(okm.sum())} first {ym[okm][0]} last {ym[okm][-1]}; RMOM first '
            f'{ym[oks][0]}; names per leg mean {nW[okm].mean():.0f} (min {nW[okm].min()}, max {nW[okm].max()}); '
            f'ranked mean {nR[okm].mean():.0f}; turnover WML {np.nanmean(res["wml"]["turn"][okm][1:]):.2f} '
            f'RMOM {np.nanmean(res["rmom"]["turn"][oks][1:]):.2f}')
    np.savez(WORK / f'series_{tag}_{px}.npz', **out)
    log('saved', WORK / f'series_{tag}_{px}.npz')


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
