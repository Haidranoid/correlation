"""The sorts and the books, as 21-day period series (rules in the .md).

    python3 -I -B books.py old|new plain|divfix      -> WORK/books_<tag>_<px>.npz

At each twins-calendar day t_k (export day 252 + 21k) with a universe:
  - eligible = the liquid set (minus c2eval.excluded) with finite PRET and ID at t_k and a close on t_k;
  - fifths by PRET, ID thirds within each fifth (common.sorts); group code = fifth * 3 + third (0..14);
  - 'all' sorts the whole eligible set; 'big' sorts the eligible stocks among the 500 largest.
Holding returns: equal weight at t_k, buy and hold (forward-filled closes, times the total-return factor for
'divfix').
  - 21-day hold (primary): period k = t_k -> t_{k+1}; a cell's return = mean_j P_j(t_{k+1}) / P_j(t_k) - 1.
  - 6-month overlapping hold (Jegadeesh-Titman calendar time): in period k, the average of the 6 cohorts formed at
    t_{k-5} .. t_k, each cohort's return over the period = V_c(t_{k+1}) / V_c(t_k) - 1 with V_c its buy-and-hold
    value since formation.
Books (long minus short, equal weight per leg):
  FIP = (top fifth, lowest-ID third) - (bottom fifth, lowest-ID third); DIP = the highest-ID thirds;
  MOM = top fifth - bottom fifth (all thirds pooled, equal weight per stock).
  - event time (the paper's view): each cohort's cumulative long-minus-short return over the 126 days after t_k
    (cum6), gross.
Costs: 5 bps per side x turnover. 21-day: turnover = sum_j |w_new - w_old drifted|, legs in units of capital (each
leg's drifted weights renormalized to 1); a book with no predecessor costs 2 (both legs bought). 6-month: 1/6 of the
turnover between the entering cohort and the leaving (drifted) cohort.
Market-relative cell returns: minus the equal-weight return of all eligible stocks over the same period.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, np, WORK, EXPORTS, STEP, N_PRET, N_ID, HOLD6, COST_BPS, MIN_NAMES, log,  # noqa: E402
                    universe_cached, prices, sorts)

NC = N_PRET * N_ID
BOOKS = {
    'FIP': ([4 * N_ID + 0], [0 * N_ID + 0]),
    'DIP': ([4 * N_ID + 2], [0 * N_ID + 2]),
    'MOM': ([4 * N_ID + 0, 4 * N_ID + 1, 4 * N_ID + 2], [0, 1, 2]),
}


def leg_weights(members, rel=None):
    """Equal weights over members (sum 1); with rel (price relatives since formation) the drifted weights."""
    if rel is None:
        return np.full(len(members), 1.0 / len(members))
    x = rel[members]
    return x / x.sum()


def turnover(new, old):
    """sum |new - old| over the union of names; new/old: dicts row -> signed weight."""
    keys = set(new) | set(old)
    return float(sum(abs(new.get(j, 0.0) - old.get(j, 0.0)) for j in keys))


def signed(L, wl, S, ws):
    d = {}
    for j, w in zip(L, wl):
        d[j] = d.get(j, 0.0) + w
    for j, w in zip(S, ws):
        d[j] = d.get(j, 0.0) - w
    return d


def run(tag, px):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = E['close'].shape
    rebal, U, B = universe_cached(tag, E)
    log('universe ready; sizes median', int(np.median(U.sum(1)[U.sum(1) > 0])))
    K = len(rebal)
    assert (np.diff(rebal) == STEP).all()
    PRET = np.load(WORK / f'pret_{tag}_{px}.npy', mmap_mode='r')
    ID = np.load(WORK / f'id_{tag}_{px}.npy', mmap_mode='r')
    # prices at the calendar days only (forward fill over the full history first)
    Pr = np.full((N, K), np.nan)
    okr = np.zeros((N, K), bool)
    for a in range(0, N, 1000):
        b = min(N, a + 1000)
        P, ok = prices(tag, E, px, slice(a, b))
        Pr[a:b] = P[:, rebal]
        okr[a:b] = ok[:, rebal]
    del P, ok
    log('prices ready', px)
    pret_r = np.asarray(PRET[:, rebal], np.float64)
    id_r = np.asarray(ID[:, rebal], np.float64)
    out = {'rebal': rebal, 'start_iso': np.array([c2eval.iso(days[t]) for t in rebal]),
           'end_iso': np.array([c2eval.iso(days[rebal[k + 1]]) if k + 1 < K else '' for k in range(K)])}
    for v in ('all', 'big'):
        G = np.full((K, N), -1, np.int8)
        for k in range(K):
            if not U[k].any():
                continue
            elig = U[k] & np.isfinite(pret_r[:, k]) & np.isfinite(id_r[:, k]) & okr[:, k]
            if v == 'big':
                elig &= B[k]
            G[k], _ = sorts(pret_r[:, k], id_r[:, k], elig)
        formed = (G >= 0).sum(1) >= MIN_NAMES
        cnt = np.zeros((K, NC), int)
        for c in range(NC):
            cnt[:, c] = (G == c).sum(1)
        # mean ID and PRET per cell (diagnostics)
        mean_id = np.full((K, NC), np.nan)
        mean_pret = np.full((K, NC), np.nan)
        # 21-day cell returns, period k = t_k -> t_{k+1}
        R21 = np.full((K, NC), np.nan)
        U21 = np.full(K, np.nan)
        R6 = np.full((K, NC), np.nan)
        U6 = np.full(K, np.nan)
        for k in range(K - 1):
            if formed[k]:
                rel = Pr[:, k + 1] / Pr[:, k]
                for c in range(NC):
                    m = G[k] == c
                    R21[k, c] = rel[m].mean() - 1
                    mean_id[k, c] = id_r[m, k].mean()
                    mean_pret[k, c] = pret_r[m, k].mean()
                U21[k] = rel[G[k] >= 0].mean() - 1
            # 6-month: cohorts formed at k-5..k
            coh = [k - a for a in range(HOLD6)]
            if min(coh) < 0 or not all(formed[c0] for c0 in coh):
                continue
            acc = np.zeros(NC)
            accu = 0.0
            for c0 in coh:
                rel1 = Pr[:, k + 1] / Pr[:, c0]
                rel0 = Pr[:, k] / Pr[:, c0]
                for c in range(NC):
                    m = G[c0] == c
                    acc[c] += rel1[m].mean() / rel0[m].mean() - 1
                m = G[c0] >= 0
                accu += rel1[m].mean() / rel0[m].mean() - 1
            R6[k] = acc / HOLD6
            U6[k] = accu / HOLD6
        out[f'{v}__R21'], out[f'{v}__U21'], out[f'{v}__R6'], out[f'{v}__U6'] = R21, U21, R6, U6
        out[f'{v}__cnt'], out[f'{v}__formed'] = cnt, formed
        out[f'{v}__mean_id'], out[f'{v}__mean_pret'] = mean_id, mean_pret
        out[f'{v}__G'] = G
        # books
        for name, (Lc, Sc) in BOOKS.items():
            gross21 = np.full(K, np.nan)
            cum6 = np.full(K, np.nan)     # event time: the cohort formed at t_k, cumulative over t_k .. t_{k+6}
            turn21 = np.full(K, np.nan)
            gross6 = np.full(K, np.nan)
            turn6 = np.full(K, np.nan)
            nL = np.zeros(K, int)
            nS = np.zeros(K, int)
            prev = None
            sets = {}
            for k in range(K):
                if not formed[k]:
                    prev = None
                    continue
                L = np.nonzero(np.isin(G[k], Lc))[0]
                S = np.nonzero(np.isin(G[k], Sc))[0]
                sets[k] = (L, S)
                nL[k], nS[k] = len(L), len(S)
                new = signed(L, leg_weights(L), S, leg_weights(S))
                if prev is None:
                    turn21[k] = 2.0
                else:
                    pL, pS = sets[k - 1]
                    rel = Pr[:, k] / Pr[:, k - 1]
                    turn21[k] = turnover(new, signed(pL, leg_weights(pL, rel), pS, leg_weights(pS, rel)))
                prev = new
                if k + 1 < K:
                    rel = Pr[:, k + 1] / Pr[:, k]
                    gross21[k] = rel[L].mean() - rel[S].mean()
                # 6-month turnover: entering cohort k vs leaving cohort k-6, drifted to t_k
                if k - HOLD6 in sets:
                    oL, oS = sets[k - HOLD6]
                    rel = Pr[:, k] / Pr[:, k - HOLD6]
                    turn6[k] = turnover(new, signed(oL, leg_weights(oL, rel), oS, leg_weights(oS, rel))) / HOLD6
                if k + 1 < K and all((k - a) in sets for a in range(HOLD6)):
                    acc = 0.0
                    for a in range(HOLD6):
                        c0 = k - a
                        cL, cS = sets[c0]
                        rel1 = Pr[:, k + 1] / Pr[:, c0]
                        rel0 = Pr[:, k] / Pr[:, c0]
                        acc += (rel1[cL].mean() / rel0[cL].mean()) - (rel1[cS].mean() / rel0[cS].mean())
                    gross6[k] = acc / HOLD6
                if k + HOLD6 < K:
                    rel = Pr[:, k + HOLD6] / Pr[:, k]
                    cum6[k] = rel[L].mean() - rel[S].mean()
            cost21 = COST_BPS / 1e4 * turn21
            cost6 = COST_BPS / 1e4 * turn6
            out[f'{v}__{name}__gross21'] = gross21
            out[f'{v}__{name}__net21'] = gross21 - cost21
            out[f'{v}__{name}__turn21'] = turn21
            out[f'{v}__{name}__gross6'] = gross6
            out[f'{v}__{name}__net6'] = gross6 - cost6
            out[f'{v}__{name}__turn6'] = turn6
            out[f'{v}__{name}__cum6'] = cum6
            out[f'{v}__{name}__nL'], out[f'{v}__{name}__nS'] = nL, nS
        ok = np.isfinite(out[f'{v}__FIP__gross21'])
        log(f'{tag} {px} {v}: formations {int(formed.sum())}, periods {int(ok.sum())} '
            f'({out["start_iso"][ok][0]} .. {out["end_iso"][ok][-1]}); FIP legs mean {out[f"{v}__FIP__nL"][ok].mean():.0f}/{out[f"{v}__FIP__nS"][ok].mean():.0f}; '
            f'turnover FIP {np.nanmean(out[f"{v}__FIP__turn21"][ok][1:]):.2f} MOM {np.nanmean(out[f"{v}__MOM__turn21"][ok][1:]):.2f}')
    np.savez(WORK / f'books_{tag}_{px}.npz', **out)
    log('saved', WORK / f'books_{tag}_{px}.npz')


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
