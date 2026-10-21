"""The loners book (full universe and the 500 largest) and the two factors (low volatility, betting against
correlation), as monthly and daily return series, for one export.

    python3 -I book.py old|new

Loners book at each calendar day t0 (formed at the close of t0, held to the close of the next calendar day):
  - stocks of the universe with c, volatility and dollar volume at t0;
  - independent terciles of 252-day upper-median dollar volume and of 252-day volatility, 9 cells;
  - in each cell with >= 9 stocks, long the third with the lowest c (rank < n/3), short the highest (rank >= 2n/3);
  - equal weight, $1 long and $1 short per cell, buy and hold; the book is the mean of the cells' long-shorts;
  - costs 5 bps x turnover, turnover = sum |new weight - drifted dollar weight| over every stock, on the first day.
Factors (research/betting_against_correlation's definitions, on this universe where the BAC inputs exist, gross):
  - BAC: volatility quintiles, thirds by 3-day correlation with the market, legs scaled to ex-ante beta 1, in
    excess of the risk-free rate, mean of the 5 quintiles;
  - low volatility: thirds of the universe by volatility, legs scaled to beta 1, in excess of the risk-free rate.

Writes WORK/book_<tag>.npz.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import time
from common import c2eval, np, WORK, EXPORTS, log, rank, ffill, rf_daily, COST_BPS, N_LARGE, MIN_CELL

MIN_UNIV_FACTOR = 100


def loner_legs(m, c, dvm, vol):
    """Legs (rows, sign, dollars) of the 9-cell book among rows m. Also per-cell info."""
    n = len(m)
    qd = np.floor(3 * rank(dvm[m]) / n).astype(int)
    qv = np.floor(3 * rank(vol[m]) / n).astype(int)
    cells = []
    for a in range(3):
        for b in range(3):
            mi = m[(qd == a) & (qv == b)]
            if len(mi) < MIN_CELL:
                continue
            rc = rank(c[mi])
            L, H = mi[rc < len(mi) / 3], mi[rc >= 2 * len(mi) / 3]
            cells.append((a, b, L, H))
    w = 1.0 / len(cells) if cells else 0.0
    legs = []
    for (a, b, L, H) in cells:
        legs += [(L, +1.0, w, (a, b)), (H, -1.0, w, (a, b))]
    return legs


def legs_bac(m, vol, corr, beta):
    """Copied from research/betting_against_correlation/portfolios.py (scale=True)."""
    n = len(m)
    q = np.floor(5 * rank(vol[m]) / n).astype(int)
    out = []
    for qq in range(5):
        mi = m[q == qq]
        nq = len(mi)
        rc = rank(corr[mi])
        L, H = mi[rc < nq / 3], mi[rc >= 2 * nq / 3]
        bL, bH = beta[L].mean(), beta[H].mean()
        out += [(L, +1.0, (1 / bL) / 5, qq), (H, -1.0, (1 / bH) / 5, qq)]
    return out


def legs_sort(m, key, beta):
    """Copied from research/betting_against_correlation/portfolios.py (scale=True)."""
    n = len(m)
    r = rank(key[m])
    L, H = m[r < n / 3], m[r >= 2 * n / 3]
    bL, bH = beta[L].mean(), beta[H].mean()
    return [(L, +1.0, 1 / bL, 0), (H, -1.0, 1 / bH, 0)]


def run(tag):
    t_start = time.time()
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'market'))
    z = np.load(WORK / f'connect_{tag}.npz')
    cd = np.load(WORK / f'c_daily_{tag}.npy', mmap_mode='r')
    days = E['days']
    T = len(days)
    cal = z['cal']
    K = len(cal)
    U, meddv, vol, corr3, beta_bac = z['U'], z['meddv'], z['vol'], z['corr3'], z['beta_bac']
    C = np.asarray(cd[:, cal]).T           # [K, N] c on calendar days
    P = ffill(E['close'])
    rf = rf_daily(days)
    mkt = np.expm1(np.asarray(E['market'], np.float64))
    log(f'{tag}: loaded {time.time() - t_start:.0f}s')

    out = {'days': days, 'cal': cal, 'rf': rf, 'mkt': mkt}
    for name in ('loners', 'loners500', 'bac', 'lowvol'):
        daily_g = np.full(T, np.nan)
        daily_c = np.zeros(T)
        M = {k: [] for k in ('k', 't0', 't1', 'complete', 'gross', 'cost', 'turn', 'nleg', 'ncell', 'mkt', 'rf',
                             'cL', 'cH', 'dvL', 'dvH', 'volL', 'volH', 'betaL', 'betaH', 'nuniv')}
        cellr = []
        prev = {}
        for k in range(K):
            t0 = int(cal[k])
            if t0 >= T - 1:
                break
            t1 = int(cal[k + 1]) if k + 1 < K and cal[k + 1] <= T - 1 else T - 1
            complete = k + 1 < K and cal[k + 1] <= T - 1
            c, dvm, v = C[k], meddv[k], vol[k]
            if name.startswith('loners'):
                m = np.nonzero(U[k] & np.isfinite(c) & np.isfinite(v) & np.isfinite(dvm))[0]
                if name == 'loners500':
                    m = np.sort(m[np.argsort(-dvm[m], kind='mergesort')[:N_LARGE]])
                if len(m) < 3 * 3 * MIN_CELL:
                    continue
                legs = loner_legs(m, c, dvm, v)
                if not legs:
                    continue
            else:
                b, r3 = beta_bac[k], corr3[k]
                m = np.nonzero(U[k] & np.isfinite(v) & np.isfinite(r3) & np.isfinite(b))[0]
                if len(m) < MIN_UNIV_FACTOR:
                    continue
                legs = legs_bac(m, v, r3, b) if name == 'bac' else legs_sort(m, v, b)
            g = np.zeros(t1 - t0)
            mret = 0.0
            new, end = {}, {}
            cells = {}
            for (S, sign, lev, cell) in legs:
                V = P[S, t0:t1 + 1] / P[S, t0][:, None]
                assert np.isfinite(V).all(), (name, t0)
                sv = V.mean(0)
                if name.startswith('loners'):
                    g += sign * lev * np.diff(sv)                       # dollar P&L of a buy-and-hold $lev leg
                    mret += sign * lev * (sv[-1] - 1)
                    cells[cell] = cells.get(cell, 0.0) + sign * (sv[-1] - 1)
                else:
                    g += sign * lev * (sv[1:] / sv[:-1] - 1 - rf[t0 + 1:t1 + 1])
                for i, j in enumerate(S):
                    new[j] = new.get(j, 0.0) + sign * lev / len(S)
                    end[j] = end.get(j, 0.0) + sign * lev * V[i, -1] / len(S)
            to = sum(abs(new.get(j, 0.0) - prev.get(j, 0.0)) for j in set(new) | set(prev))
            prev = end
            cost = COST_BPS / 1e4 * to
            daily_g[t0 + 1:t1 + 1] = g
            daily_c[t0 + 1] = cost
            if not name.startswith('loners'):
                mret = float(np.prod(1 + g) - 1)
            longs = np.concatenate([S for (S, s, _, _) in legs if s > 0])
            shorts = np.concatenate([S for (S, s, _, _) in legs if s < 0])
            for key, val in (('k', k), ('t0', t0), ('t1', t1), ('complete', complete), ('gross', mret), ('cost', cost),
                             ('turn', to), ('nleg', np.mean([len(S) for S, *_ in legs])),
                             ('ncell', len(legs) // 2 if name.startswith('loners') else 0),
                             ('mkt', float(np.prod(1 + np.nan_to_num(mkt[t0 + 1:t1 + 1])) - 1)),
                             ('rf', float(np.prod(1 + rf[t0 + 1:t1 + 1]) - 1)),
                             ('cL', float(np.nanmean(c[longs]))), ('cH', float(np.nanmean(c[shorts]))),
                             ('dvL', float(np.nanmedian(dvm[longs]))), ('dvH', float(np.nanmedian(dvm[shorts]))),
                             ('volL', float(np.nanmean(v[longs]))), ('volH', float(np.nanmean(v[shorts]))),
                             ('betaL', float(np.nanmean(beta_bac[k][longs]))), ('betaH', float(np.nanmean(beta_bac[k][shorts]))),
                             ('nuniv', len(m))):
                M[key].append(val)
            if name.startswith('loners'):
                cellr.append([cells.get((a, b), np.nan) for a in range(3) for b in range(3)])
        for key, val in M.items():
            out[f'{name}__m_{key}'] = np.array(val)
        out[f'{name}__m_net'] = out[f'{name}__m_gross'] - out[f'{name}__m_cost']
        out[f'{name}__d_gross'] = daily_g
        out[f'{name}__d_net'] = daily_g - np.where(np.isfinite(daily_g), daily_c, np.nan)
        if cellr:
            out[f'{name}__m_cells'] = np.array(cellr)
        log(f'{tag} {name}: {len(M["t0"])} formations (first {c2eval.iso(days[M["t0"][0]])}), '
            f'mean turnover {np.mean(M["turn"][1:]):.2f}  {time.time() - t_start:.0f}s')
    np.savez(WORK / f'book_{tag}.npz', **out)


if __name__ == '__main__':
    run(_sys.argv[1])
