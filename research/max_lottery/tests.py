"""The MAX tests: the decile long-short (lowest-MAX tenth minus highest), Fama-MacBeth with the controls, FF5 + UMD
alphas, the 500 largest, by period, with and without the dividend correction.

    python3 -I -B tests.py old   -> results_old.txt, series_old.npz (tests 1-3 on the old export)
    python3 -I -B tests.py new   -> results_new.txt, series_new.npz (the unseen months, once, at the end)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, pubdata, np, WORK, HERE, EXPORTS, H, DECILE, COST_BPS, NW_LAGS, MIN_UNIV,  # noqa: E402
                    TEST1, TEST3, UNSEEN, PERIODS, DIV_FROM, log, ols, mean_t, pct_rank, winsor, in_period)

REGS = ('max', 'ivol', 'rev', 'beta', 'size', 'mom')
SPECS = {'MAX alone': ('max',), 'IVOL alone': ('ivol',), 'MAX + IVOL': ('max', 'ivol'),
         'All controls (the test)': REGS, 'All controls, no MAX': ('ivol', 'rev', 'beta', 'size', 'mom')}
FACS = ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD')


def factor_periods(days, rebal, T):
    """[K, 6] compounded FF5 + UMD factor returns over each formation's holding days t+1..t+21 (NaN if any day missing)."""
    d5, f5 = pubdata.ff5_daily()
    du, fu = pubdata.umd_daily()
    cols = [pubdata.on_days(d5, f5[c], days) for c in FACS[:5]] + [pubdata.on_days(du, fu['UMD'], days)]
    out = np.full((len(rebal), 6), np.nan)
    for k, t in enumerate(rebal):
        if t + H > T - 1:
            continue
        for i, c in enumerate(cols):
            x = c[t + 1:t + H + 1]
            if np.isfinite(x).all():
                out[k, i] = np.prod(1 + x) - 1
    return out


def book(Z, sel, fwd, signal='max', within=None, n_within=5):
    """Decile long-short on `signal`: long the lowest tenth, short the highest, equal weight, held 21 days.
    sel [K, N] bool candidate stocks; within: optional [K, N] control variable -> sort first into n_within groups of it,
    take each group's lowest and highest tenth of the signal (pooled), as Bali et al.'s IVOL-controlled sort.
    Returns dict of [K] arrays: gross, net, long, short, mkt (EW mean of candidates), turnover, n_leg."""
    K, N = sel.shape
    out = {k: np.full(K, np.nan) for k in ('gross', 'net', 'long', 'short', 'mkt', 'turnover', 'n_leg')}
    wl_prev = np.zeros(N)
    ws_prev = np.zeros(N)
    prev_ok = False
    for k in range(K):
        s = Z[signal][k]
        f = fwd[k]
        m = sel[k] & np.isfinite(s) & np.isfinite(f)
        if within is not None:
            m &= np.isfinite(within[k])
        if m.sum() < MIN_UNIV or not Z['complete'][k]:
            prev_ok = False
            wl_prev[:] = 0
            ws_prev[:] = 0
            continue
        j = np.nonzero(m)[0]
        groups = [j]
        if within is not None:
            q = pct_rank(within[k][j])
            g = np.minimum((q * n_within).astype(int), n_within - 1)
            groups = [j[g == i] for i in range(n_within)]
        lo, hi = [], []
        for jj in groups:
            nd = int(round(DECILE * len(jj)))
            o = jj[np.argsort(s[jj], kind='mergesort')]
            lo.append(o[:nd])
            hi.append(o[len(o) - nd:])
        lo, hi = np.concatenate(lo), np.concatenate(hi)
        wl = np.zeros(N)
        ws = np.zeros(N)
        wl[lo] = 1.0 / len(lo)
        ws[hi] = 1.0 / len(hi)
        to = np.abs(wl - wl_prev).sum() + np.abs(ws - ws_prev).sum() if prev_ok else 2.0
        rl = float(f[lo].mean())
        rs = float(f[hi].mean())
        out['long'][k], out['short'][k], out['mkt'][k] = rl, rs, float(f[j].mean())
        out['gross'][k] = rl - rs
        out['turnover'][k] = to
        out['net'][k] = rl - rs - COST_BPS * 1e-4 * to
        out['n_leg'][k] = len(lo)
        # drift to the end of the holding period
        a = wl * (1 + np.nan_to_num(f))
        b = ws * (1 + np.nan_to_num(f))
        wl_prev = a / a.sum()
        ws_prev = b / b.sum()
        prev_ok = True
    return out


def fm(Z, sel, fwd, raw=False):
    """Fama-MacBeth: at each formation, OLS of the next-21-day market-relative return on each spec's regressors
    (percentile ranks within the regression sample, or winsorized raw values if raw). Same sample (all six regressors
    finite) for every spec. Returns {spec: [K, p] coefficients (excluding the intercept)} and [K] sample sizes."""
    K, N = sel.shape
    res = {s: np.full((K, len(r)), np.nan) for s, r in SPECS.items()}
    nobs = np.zeros(K, int)
    for k in range(K):
        if not Z['complete'][k]:
            continue
        m = sel[k] & np.isfinite(fwd[k])
        for r in REGS:
            m &= np.isfinite(Z[r][k])
        if m.sum() < MIN_UNIV:
            continue
        j = np.nonzero(m)[0]
        y = fwd[k][j].astype(float)
        y = y - y.mean()
        X = {r: (winsor(Z[r][k][j]) if raw else pct_rank(Z[r][k][j])) for r in REGS}
        nobs[k] = len(j)
        for s, regs in SPECS.items():
            A = np.column_stack([np.ones(len(j))] + [X[r] for r in regs])
            b, *_ = np.linalg.lstsq(A, y, rcond=None)
            res[s][k] = b[1:]
    return res, nobs


def deciles(Z, sel, fwd, mask_k):
    """Average over the formations in mask_k of each MAX decile's EW return, MAX, IVOL, beta, size proxy, count."""
    K = sel.shape[0]
    acc = {c: [[] for _ in range(10)] for c in ('ret', 'max', 'ivol', 'beta', 'dv', 'n', 'rev')}
    for k in np.nonzero(mask_k)[0]:
        m = sel[k] & np.isfinite(Z['max'][k]) & np.isfinite(fwd[k])
        j = np.nonzero(m)[0]
        if len(j) < MIN_UNIV:
            continue
        o = j[np.argsort(Z['max'][k][j], kind='mergesort')]
        parts = np.array_split(o, 10)
        for i, p in enumerate(parts):
            acc['ret'][i].append(np.mean(fwd[k][p]))
            acc['max'][i].append(np.nanmean(Z['max'][k][p]))
            acc['ivol'][i].append(np.nanmean(Z['ivol'][k][p]))
            acc['beta'][i].append(np.nanmean(Z['beta'][k][p]))
            acc['rev'][i].append(np.nanmean(Z['rev'][k][p]))
            acc['dv'][i].append(np.nanmedian(np.exp(Z['size'][k][p])) / 1e6)
            acc['n'][i].append(len(p))
    return {c: [float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for v in acc[c]] for c in acc}


def stats(x, Fk, mask):
    """Mean, plain t, NW t, n, and the FF5 + UMD alpha (coef, plain t, NW t, n_alpha) of a [K] series over mask."""
    m, t, tnw, n = mean_t(x[mask])
    y = x[mask]
    Xf = Fk[mask]
    ok = np.isfinite(y) & np.isfinite(Xf).all(1)
    if ok.sum() >= 10:
        b, tt, na, _ = ols(y[ok], Xf[ok])
        _, ttn, _, _ = ols(y[ok], Xf[ok], nw_lags=NW_LAGS)
        a, at, atn = b[0], tt[0], ttn[0]
        mk = b[1]
    else:
        a = at = atn = mk = float('nan')
        na = int(ok.sum())
    sd = np.nanstd(y, ddof=1)
    return dict(mean=m, t=t, tnw=tnw, n=n, alpha=a, at=at, atn=atn, na=na, mkt=mk,
                sharpe=m / sd * np.sqrt(12) if sd > 0 else np.nan)


def fmt_ls(name, st):
    return (f"| {name} | {st['n']} | {100 * st['mean']:+.3f} | {st['tnw']:+.2f} ({st['t']:+.2f}) | {st['sharpe']:+.2f} | "
            f"{100 * st['alpha']:+.3f} | {st['atn']:+.2f} ({st['at']:+.2f}), {st['na']} | {st['mkt']:+.2f} |")


def fm_stats(coef, mask):
    out = []
    for i in range(coef.shape[1]):
        m, t, tnw, n = mean_t(coef[mask, i])
        out.append((m, t, tnw, n))
    return out


def run(tag):
    Z = dict(np.load(WORK / f'form_{tag}.npz', allow_pickle=False))
    E = c2eval.load_export(EXPORTS[tag], arrays=())
    days = E['days']
    T = len(days)
    rebal = Z['rebal']
    K = len(rebal)
    first = Z['first_ret']
    Fk = factor_periods(days, rebal, T)
    uni, big = Z['uni'], Z['big']
    L = []
    P = L.append
    series = {}
    variants = {'raw': Z['fwd'], 'div': Z['fwdd']}
    B, FMr, FMw = {}, {}, {}
    for v, fwd in variants.items():
        B[('all', v)] = book(Z, uni, fwd)
        B[('big', v)] = book(Z, big, fwd)
        B[('all_ivolctl', v)] = book(Z, uni, fwd, within=Z['ivol'])
        FMr[('all', v)], n_all = fm(Z, uni, fwd)
        FMr[('big', v)], n_big = fm(Z, big, fwd)
        FMw[('all', v)], _ = fm(Z, uni, fwd, raw=True)
        for key, d in B.items():
            if key[1] == v:
                for c, x in d.items():
                    series[f'{key[0]}_{v}_{c}'] = x
        for key, d in FMr.items():
            if key[1] == v:
                for s, x in d.items():
                    series[f'fm_{key[0]}_{v}_{s}'] = x
    series['rebal'] = rebal
    series['first_ret'] = first
    series['factors'] = Fk
    series['fm_nobs_all'] = n_all
    series['fm_nobs_big'] = n_big
    np.savez(WORK / f'series_{tag}.npz', **series)

    if tag == 'old':
        periods = [('Test 1: 2006-2019', *TEST1)] + PERIODS
    else:
        periods = [('Unseen: first return day 2025-06-30..2026-10-19', *UNSEEN)]

    def pm(a, b):
        return in_period(first, a, b) & Z['complete']

    def div_variants(a, b):
        """Rows for a period: 'raw' alone if it ends before 2022-07; else 'div' (primary) and 'raw'."""
        return ['raw'] if b < DIV_FROM else ['div', 'raw']

    P(f'# MAX lottery tests ({tag} export), {c2eval.iso(days[0])}..{c2eval.iso(days[-1])}')
    P(f'Formations on the fixed calendar: {K}; complete 21-day holding periods: {int(Z["complete"].sum())}')
    P('Units: % per 21-day period ("a month"); t = Newey-West (4 lags) with plain t in brackets; alpha = intercept on '
      'compounded FF5 + UMD factors over the same 21 days (NW t, plain t, periods with factors).')
    P('')
    P('## Long-short: lowest-MAX tenth minus highest-MAX tenth, equal weight, net of 5 bps per side')
    for label, sel in (('Universe', 'all'), ('500 largest', 'big'), ('IVOL-controlled (MAX tenths within IVOL fifths)', 'all_ivolctl')):
        P(f'### {label}')
        P('| Period | Periods | Mean net, % | t NW (plain) | Sharpe | FF5+UMD alpha, % | alpha t NW (plain), n | Mkt loading |')
        P('|---|---|---|---|---|---|---|---|')
        for name, a, b in periods:
            for v in div_variants(a, b):
                st = stats(B[(sel, v)]['net'], Fk, pm(a, b))
                P(fmt_ls(name + (', dividends restored' if v == 'div' else (', no dividend fix' if b >= DIV_FROM else '')), st))
        P('')
    # gross, legs, turnover
    P('## Legs (market-relative = minus the EW mean of the candidates), gross, turnover; universe')
    P('| Period | Gross LS, % | Long - mkt, % | Short - mkt, % | Turnover (both legs, max 4) | Stocks per leg |')
    P('|---|---|---|---|---|---|')
    for name, a, b in periods:
        for v in div_variants(a, b):
            d = B[('all', v)]
            m = pm(a, b) & np.isfinite(d['gross'])
            P(f"| {name}{', div' if v == 'div' else ''} | {100 * np.mean(d['gross'][m]):+.3f} | "
              f"{100 * np.mean(d['long'][m] - d['mkt'][m]):+.3f} | {100 * np.mean(d['short'][m] - d['mkt'][m]):+.3f} | "
              f"{np.mean(d['turnover'][m]):.2f} | {np.mean(d['n_leg'][m]):.0f} |")
    P('')
    # Fama-MacBeth
    for sel, lab, store in (('all', 'Universe, percentile ranks (primary)', FMr), ('big', '500 largest, percentile ranks', FMr),
                            ('all', 'Universe, winsorized raw values (1/99%; secondary)', FMw)):
        P(f'## Fama-MacBeth: {lab}')
        if store is FMw:
            P('Coefficient x 100 = mean over formations of the cross-sectional slope on the winsorized raw variable: the '
              '21-day return change, %, per unit of the variable (MAX, IVOL, returns in decimals; beta; log $ volume).')
        else:
            P('Coefficient = mean over formations of the cross-sectional slope; ranks in [0,1], so a rank coefficient is the '
              '21-day return difference, %, between the highest and lowest stock on that variable, others held fixed.')
        P('Cells: coefficient (Newey-West t; plain t).')
        for name, a, b in periods:
            for v in div_variants(a, b):
                d = store[(sel, v)]
                m = pm(a, b)
                P(f"### {name}{', dividends restored' if v == 'div' else (', no dividend fix' if b >= DIV_FROM else '')}")
                P('| Spec | ' + ' | '.join(REGS) + ' | formations |')
                P('|---|' + '---|' * (len(REGS) + 1))
                for s, regs in SPECS.items():
                    st = fm_stats(d[s], m)
                    cells = []
                    for r in REGS:
                        if r in regs:
                            mm, t, tnw, n = st[regs.index(r)]
                            unit = 100.0
                            cells.append(f'{unit * mm:+.3f} ({tnw:+.2f}; {t:+.2f})')
                        else:
                            cells.append('')
                    P(f'| {s} | ' + ' | '.join(cells) + f' | {st[0][3]} |')
                P('')
    nn = series['fm_nobs_all']
    P(f'FM sample size (universe): median {int(np.median(nn[nn > 0]))}, min {int(nn[nn > 0].min())}, max {int(nn.max())}')
    P('')
    # decile characteristics
    for name, a, b in periods[:1] + ([p for p in periods if p[0].startswith('2020')] if tag == 'old' else []):
        for sel, lab in (('all', 'universe'), ('big', '500 largest')):
            v = 'raw' if b < DIV_FROM else 'div'
            D = deciles(Z, uni if sel == 'all' else big, variants[v], pm(a, b))
            P(f'## MAX deciles, {lab}, {name} (averages over formations)')
            P('| Decile | 21-day return, % | MAX, % | IVOL, % a day | Beta | 21-day return before, % | Median $ volume, $M | Stocks |')
            P('|---|---|---|---|---|---|---|---|')
            for i in range(10):
                P(f"| {i + 1} | {100 * D['ret'][i]:+.2f} | {100 * D['max'][i]:.2f} | {100 * D['ivol'][i]:.2f} | "
                  f"{D['beta'][i]:.2f} | {100 * D['rev'][i]:+.2f} | {D['dv'][i]:.0f} | {D['n'][i]:.0f} |")
            P('')
    # MAX-IVOL correlation
    cs = []
    for k in np.nonzero(pm(*TEST1 if tag == 'old' else UNSEEN))[0]:
        m = uni[k] & np.isfinite(Z['max'][k]) & np.isfinite(Z['ivol'][k])
        if m.sum() >= MIN_UNIV:
            cs.append((np.corrcoef(Z['max'][k][m], Z['ivol'][k][m])[0, 1],
                       np.corrcoef(pct_rank(Z['max'][k][m]), pct_rank(Z['ivol'][k][m]))[0, 1]))
    cs = np.array(cs)
    P(f'Cross-sectional correlation of MAX and IVOL, mean over formations: Pearson {cs[:, 0].mean():.3f}, '
      f'rank {cs[:, 1].mean():.3f} ({len(cs)} formations)')
    # Same-period checks: compounding totals for the unseen months
    if tag == 'new':
        for v in ('raw', 'div'):
            d = B[('all', v)]
            m = pm(*UNSEEN) & np.isfinite(d['net'])
            P(f"Unseen, universe, {v}: {int(m.sum())} periods, compounded net LS {100 * (np.prod(1 + d['net'][m]) - 1):+.2f}%, "
              f"first formation {Z['iso'][m][0]}, last {Z['iso'][m][-1]}")
            d = B[('big', v)]
            P(f"Unseen, 500 largest, {v}: compounded net LS {100 * (np.prod(1 + d['net'][m]) - 1):+.2f}%")
        # consistency with the old export on shared formations
        so = np.load(WORK / 'series_old.npz')
        ro = so['rebal']
        shared = [(i, int(np.nonzero(rebal == r)[0][0])) for i, r in enumerate(ro)
                  if np.isfinite(so['all_raw_gross'][i]) and r in rebal]
        a = np.array([so['all_raw_gross'][i] for i, _ in shared])
        b = np.array([series['all_raw_gross'][j] for _, j in shared])
        ok = np.isfinite(a) & np.isfinite(b)
        P(f'Rebuilt (new export) vs old export, gross LS on {int(ok.sum())} shared formations: corr '
          f'{np.corrcoef(a[ok], b[ok])[0, 1]:.4f}, mean old {100 * a[ok].mean():+.3f}% new {100 * b[ok].mean():+.3f}%')
    (HERE / f'results_{tag}.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    run(sys.argv[1])
