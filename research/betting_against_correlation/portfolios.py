"""BAC, its controls and sensitivities as daily return series (the .md, 'Rules' and 'Decisions').

    python3 -I portfolios.py old|new

Writes WORK/series_<tag>.npz: for each strategy S, S__gross, S__cost, S__net [T] (NaN where not held), plus
S__form (formation day indices), S__turn (turnover at each formation), S__nleg (mean stocks per leg),
S__bL / S__bH (mean ex-ante leg betas), and days, rf, mkt (simple EW market), spy.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import time
from common import c2eval, np, WORK, EXPORTS, rf_daily, spy_returns, COST_BPS, N_LARGE

MIN_UNIV = 100   # formations are built from here; the graded periods start where the universe reaches 500


def ffill(close):
    c = np.asarray(close, np.float64)
    ok = np.isfinite(c)
    idx = np.where(ok, np.arange(c.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    P = np.take_along_axis(c, idx, 1)
    P[~np.isfinite(P)] = np.nan
    first = ok.argmax(1)
    P[np.arange(c.shape[1])[None, :] < first[:, None]] = np.nan
    P[~ok.any(1)] = np.nan
    return P


def rank(x):
    return np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort')


def legs_bac(m, vol, corr, beta, scale=True):
    n = len(m)
    q = np.floor(5 * rank(vol[m]) / n).astype(int)
    out = []
    for qq in range(5):
        mi = m[q == qq]
        nq = len(mi)
        rc = rank(corr[mi])
        L, H = mi[rc < nq / 3], mi[rc >= 2 * nq / 3]
        bL, bH = beta[L].mean(), beta[H].mean()
        out += [(L, +1.0, (1 / bL if scale else 1.0) / 5, bL), (H, -1.0, (1 / bH if scale else 1.0) / 5, bH)]
    return out


def legs_sort(m, key, beta, scale=True):
    n = len(m)
    r = rank(key[m])
    L, H = m[r < n / 3], m[r >= 2 * n / 3]
    bL, bH = beta[L].mean(), beta[H].mean()
    return [(L, +1.0, 1 / bL if scale else 1.0, bL), (H, -1.0, 1 / bH if scale else 1.0, bH)]


def run(tag):
    t_start = time.time()
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'market'))
    z = np.load(WORK / f'panels_{tag}.npz')
    days = E['days']
    T = len(days)
    P = ffill(E['close'])
    rf = rf_daily(days)
    assert np.isfinite(rf[756:]).all()
    mkt = np.expm1(np.asarray(E['market'], np.float64))
    spy = spy_returns(tag, days)
    cal, vol, corr, beta = z['cal'], z['vol'], z['corr'], z['beta']

    strategies = {
        'bac': ('univ', 'bac', True, False),
        'bac_flag': ('univ_flag', 'bac', True, False),
        'bac500': ('univ', 'bac', True, True),
        'bac_dollar': ('univ', 'bac', False, False),
        'bab': ('univ', 'beta', True, False),
        'lowvol': ('univ', 'vol', True, False),
        'bab500': ('univ', 'beta', True, True),
        'lowvol500': ('univ', 'vol', True, True),
    }
    out = {'days': days, 'rf': rf, 'mkt': mkt, 'spy': spy}
    for name, (uk, kind, scale, large) in strategies.items():
        U = z[uk]
        gross = np.full(T, np.nan)
        cost = np.zeros(T)
        forms, turns, nleg, bLs, bHs = [], [], [], [], []
        prev = {}
        ks = [k for k in range(len(cal)) if cal[k] < T - 1 and U[k].sum() >= MIN_UNIV]
        for k in ks:
            t0 = cal[k]
            t1 = min(cal[k + 1], T - 1) if k + 1 < len(cal) else T - 1
            m = np.nonzero(U[k])[0]
            if large:
                dv = z['meddv'][k, m]
                m = np.sort(m[np.argsort(-dv, kind='mergesort')[:N_LARGE]])
            v, c, b = vol[:, t0], corr[:, t0], beta[:, t0]
            if kind == 'bac':
                legs = legs_bac(m, v, c, b, scale)
            elif kind == 'beta':
                legs = legs_sort(m, b, b, scale)
            else:
                legs = legs_sort(m, v, b, scale)
            r = np.zeros(t1 - t0)
            new, end = {}, {}
            for (S, sign, lev, bl) in legs:
                V = P[S, t0:t1 + 1] / P[S, t0][:, None]
                assert np.isfinite(V).all(), (name, t0)
                sv = V.sum(0)
                r += sign * lev * (sv[1:] / sv[:-1] - 1 - rf[t0 + 1:t1 + 1])
                for i, j in enumerate(S):
                    new[j] = new.get(j, 0.0) + sign * lev / len(S)
                    end[j] = end.get(j, 0.0) + sign * lev * V[i, -1] / sv[-1]
            to = sum(abs(new.get(j, 0.0) - prev.get(j, 0.0)) for j in set(new) | set(prev))
            gross[t0 + 1:t1 + 1] = r
            cost[t0 + 1] = COST_BPS / 1e4 * to
            prev = end
            forms.append(t0); turns.append(to)
            nleg.append(np.mean([len(S) for S, *_ in legs]))
            bLs.append(np.mean([bl for (_, s, _, bl) in legs if s > 0])); bHs.append(np.mean([bl for (_, s, _, bl) in legs if s < 0]))
        out[f'{name}__gross'] = gross
        out[f'{name}__cost'] = np.where(np.isfinite(gross), cost, np.nan)
        out[f'{name}__net'] = gross - cost
        out[f'{name}__form'] = np.array(forms)
        out[f'{name}__turn'] = np.array(turns)
        out[f'{name}__nleg'] = np.array(nleg)
        out[f'{name}__bL'] = np.array(bLs)
        out[f'{name}__bH'] = np.array(bHs)
        print(f'{tag} {name}: {len(forms)} formations, {time.time() - t_start:.0f}s', flush=True)
    np.savez(WORK / f'series_{tag}.npz', **out)


if __name__ == '__main__':
    run(_sys.argv[1])
