"""P1, P2, the variants, E1 and L on one export (the .md, "Rules").

    python3 -I -B run.py old     # development 2008..2019 and check 2020..2025-06
    python3 -I -B run.py new     # the unseen months (called once by unseen.py)

Writes p_<tag>.txt / .json and years_<tag>.tsv next to this script, and WORK/series_<tag>.npz (daily chained series).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eg_common as C  # noqa: E402

np = C.np
PORTS = {'R1d': ('d', 1), 'R1m': ('m', 1), 'R1q': ('q', 1), 'B1': ('B', 1), 'R0d': ('d', 0), 'R0m': ('m', 0),
         'B0': ('B', 0)}
RAW_PORTS = ('R1d', 'R1m', 'B1')
BANDS = ((1, 1), (2, 5), (6, 21), (22, 63), (64, 10 ** 9))


def decisions_for(D, tag, forward_start=None):
    if tag == 'old':
        dec = [C.year_end_index(D, y) for y in range(2007, 2025)]
        return dec, [str(y) for y in range(2008, 2026)]
    if tag == 'new':
        a = int(np.nonzero(D['iso'] == '2025-06-27')[0][0])
        return [a, C.year_end_index(D, 2025)], ['2025H2', '2026']
    # forward: a 500 chosen at the close of forward_start, then the year-end rule
    a = int(np.nonzero(D['iso'] == forward_start)[0][0])
    last_year = int(D['iso'][-1][:4])
    dec = [a] + [C.year_end_index(D, y) for y in range(int(forward_start[:4]), last_year)
                 if C.year_end_index(D, y) > a and C.year_end_index(D, y) < len(D['days']) - 1]
    return dec, [D['iso'][t + 1][:7] for t in dec]


def build_periods(D, tag, raw=None, nodiv=False, forward_start=None):
    """Per period: rows, holding data, window returns, gamma*, and every portfolio's results."""
    dec, labels = decisions_for(D, tag, forward_start)
    P = C.periods(D, dec)
    out = []
    for p, lab in zip(P, labels):
        rows, n_elig = C.select(D, p['t0'])
        H = C.holding(D, rows, p)
        drop = ~H['trade'][:, 0]
        if drop.any():
            rows = rows[~drop]
            H = C.holding(D, rows, p)
        X = C.window_logret(D, rows, p['t0'])
        Z = C.demean_fill(X)
        gam = C.gamma_star(X)
        rec = {'label': lab, 'p': p, 'rows': rows, 'n_elig': n_elig, 'n_drop': int(drop.sum()), 'H': H, 'Z': Z,
               'gamma': gam, 'days': np.arange(p['s'] + 1, p['end'] + 1), 'ports': {}}
        for name, (kind, delay) in PORTS.items():
            g, t, w0, we = C.run_period(H, kind, delay)
            rec['ports'][name] = {'gross': g[0], 'turn': t[0], 'w0': w0[0], 'wend': we[0]}
        if raw is not None:
            Hr = C.holding(D, rows, p, close=raw['close'], dv=raw['dv'], last=raw['last'])
            rec['n_raw_untradeable_purchase'] = int((~Hr['trade'][:, 0]).sum())
            Hr['trade'][:, 0] = True
            for name in RAW_PORTS:
                kind, delay = PORTS[name]
                g, t, w0, we = C.run_period(Hr, kind, delay)
                rec['ports'][name + '_raw'] = {'gross': g[0], 'turn': t[0], 'w0': w0[0], 'wend': we[0]}
        if nodiv:
            Hn = C.holding(D, rows, p, div=False)
            for name in ('R1d', 'B1'):
                kind, delay = PORTS[name]
                g, t, w0, we = C.run_period(Hn, kind, delay)
                rec['ports'][name + '_nodiv'] = {'gross': g[0], 'turn': t[0], 'w0': w0[0], 'wend': we[0]}
        C.log(f"{lab}: n {len(rows)} (eligible {n_elig}, dropped at purchase {rec['n_drop']}), days {H['n']}, "
              f"ends inside {H['n_end']}, untradeable stock-days {int((~H['trade'][:, 1:]).sum())}, "
              f"gamma* {252 * gam:.4f}/yr")
        out.append(rec)
    return out


def chained(per, name):
    pp = [{'rows': r['rows'], 'days': r['days'], **r['ports'][name]} for r in per]
    days, g, t, pid = C.chain(pp)
    return days, g, t, pid


def e1_period(r):
    """E1 for one period: realized EG (daily, monthly), J, gaps, and the random-walk draws."""
    H, Z, n = r['H'], r['Z'], r['H']['n']
    K = len(r['rows'])
    g = np.log1p(H['R'][:, 1:]).sum(1)                         # each stock's log return over the period
    mg = g.mean()
    lv = {k: np.log1p(r['ports'][k]['gross']).sum() for k in ('R0d', 'R0m', 'B0', 'R1d', 'R1m', 'B1')}
    res = {'pred': r['gamma'] * n, 'EG_d': lv['R0d'] - mg, 'EG_m': lv['R0m'] - mg, 'J': lv['B0'] - mg,
           'gap_d0': lv['R0d'] - lv['B0'], 'gap_m0': lv['R0m'] - lv['B0'], 'gap_d1': lv['R1d'] - lv['B1'],
           'gap_m1': lv['R1m'] - lv['B1'], 'n': n, 'K': K}
    # random walk: resample whole days of the window's demeaned log returns
    rng = np.random.default_rng(C.SEED + int(r['p']['s']))
    draws = rng.integers(0, Z.shape[1], size=(C.N_SIM, n))
    Rw = np.expm1(Z)
    counts = np.zeros((C.N_SIM, Z.shape[1]))
    for b in range(C.N_SIM):
        counts[b] = np.bincount(draws[b], minlength=Z.shape[1])
    gs = counts @ Z.T                                          # [N_SIM, K]
    w0 = np.full((C.N_SIM, K), 1.0 / K)
    sim = {}
    for name, freq in (('d', 'd'), ('m', 'm'), ('B', None)):
        gg, _, _ = C.simulate(n, lambda d: Rw[:, draws[:, d - 1]].T, w0, C.fills_for(freq, n, 0), 0)
        sim[name] = np.log1p(gg).sum(1)
    mgs = gs.mean(1)
    res['sim_EG_d'] = sim['d'] - mgs
    res['sim_EG_m'] = sim['m'] - mgs
    res['sim_J'] = sim['B'] - mgs
    res['sim_gap_d'] = sim['d'] - sim['B']
    res['sim_gap_m'] = sim['m'] - sim['B']
    # L: second-order gap by lag band (daily, delay 0)
    lr = np.log1p(H['R'][:, 1:])
    x = lr - lr.mean(0, keepdims=True)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(x, m, axis=1)
    ac = np.fft.irfft(F * np.conj(F), m, axis=1)[:, :n].mean(0)   # A_l, w = 1/K
    res['gap2'] = float(-ac[1:].sum())
    res['bands'] = [float(-ac[a:min(b, n - 1) + 1].sum()) for a, b in BANDS]
    res['G'] = 0.5 * ((x ** 2).sum(1) - x.sum(1) ** 2)          # per-stock second-order terms
    res['g'] = g
    return res


def subsets_period(r):
    """1,000 random 20-stock subsets of the period's stocks: gross and net log gaps, EG, J and gamma*."""
    H, Z, n = r['H'], r['Z'], r['H']['n']
    K = len(r['rows'])
    rng = np.random.default_rng(C.SEED + 7 * int(r['p']['s']))
    idx = np.stack([rng.choice(K, C.SUB_K, replace=False) for _ in range(C.N_SUB)])
    g = np.log1p(H['R'][:, 1:]).sum(1)
    out = {'gamma': C.gamma_star_batch(Z, idx) * n}
    lv, lv5 = {}, {}
    for name, (kind, delay) in PORTS.items():
        if name == 'R1q':
            continue
        gg, tt, _, _ = C.run_period(H, kind, delay, idx)
        tt = tt.copy()
        tt[:, 0] += 1.0                                        # purchase from cash
        tt[:, -1] += 1.0                                       # sale at the period's end
        lv[name] = np.log1p(gg).sum(1)
        lv5[name] = np.log1p(C.net(gg, tt, C.COST)).sum(1)
    mg = g[idx].mean(1)
    out.update({'gap_d1': lv['R1d'] - lv['B1'], 'gap_m1': lv['R1m'] - lv['B1'],
                'net_d1': lv5['R1d'] - lv5['B1'], 'net_m1': lv5['R1m'] - lv5['B1'],
                'EG_d': lv['R0d'] - mg, 'J': lv['B0'] - mg, 'gap_d0': lv['R0d'] - lv['B0']})
    return out


def h5(D):
    """Five-year holds: the 500 chosen at the ends of 2007, 2012 and 2017."""
    dec = [C.year_end_index(D, y) for y in (2007, 2012, 2017)]
    P = C.periods(D, dec, last_end=C.year_end_index(D, 2022) + 1)
    rows_out = []
    for p in P:
        rows, _ = C.select(D, p['t0'])
        H = C.holding(D, rows, p)
        drop = ~H['trade'][:, 0]
        if drop.any():
            rows = rows[~drop]
            H = C.holding(D, rows, p)
        gam = C.gamma_star(C.window_logret(D, rows, p['t0']))
        lv, lv5 = {}, {}
        for name in ('R1d', 'R1m', 'B1', 'R0d', 'B0'):
            kind, delay = PORTS[name]
            gg, tt, _, _ = C.run_period(H, kind, delay)
            tt = tt[0].copy()
            tt[0] += 1.0
            tt[-1] += 1.0
            lv[name] = float(np.log1p(gg[0]).sum())
            lv5[name] = float(np.log1p(C.net(gg[0], tt, C.COST)).sum())
        g = np.log1p(H['R'][:, 1:]).sum(1).mean()
        rows_out.append({'span': f"{D['iso'][p['s']]}..{D['iso'][p['end']]}", 'n_days': p['n'], 'K': len(rows),
                         'pred': gam * p['n'], 'EG_d0': lv['R0d'] - g, 'J0': lv['B0'] - g,
                         'gap_d0': lv['R0d'] - lv['B0'], 'gap_d1': lv['R1d'] - lv['B1'], 'gap_m1': lv['R1m'] - lv['B1'],
                         'net_d1': lv5['R1d'] - lv5['B1'], 'net_m1': lv5['R1m'] - lv5['B1']})
        C.log(f"H5 {rows_out[-1]['span']}: gap_d1 {rows_out[-1]['gap_d1']:+.4f}, pred {rows_out[-1]['pred']:.4f}")
    return rows_out


def load_raw(D, tag):
    S = C.c2eval.load_store(C.STORES[tag])
    close = C.c2eval.store_panel(S, D['E']['ids'], D['days'], 'close')
    vol = C.c2eval.store_panel(S, D['E']['ids'], D['days'], 'volume')
    with np.errstate(all='ignore'):
        dv = close * vol
    close[~(close > 0)] = np.nan
    fin = np.isfinite(close)
    T = fin.shape[1]
    last = np.where(fin.any(1), T - 1 - np.argmax(fin[:, ::-1], 1), -1)
    ex = np.asarray(D['E']['close'])
    both = np.isfinite(ex) & fin
    diff = int((both & (ex != close)).sum())
    only_raw = int((fin & ~np.isfinite(ex)).sum())
    return {'close': close, 'dv': dv, 'last': last, 'n_diff': diff, 'n_only_raw': only_raw}


def main(tag, forward=None):
    D = C.load(tag) if forward is None else C.load('fwd', export=forward['export'], store=forward['store'], div=False)
    C.log(tag, 'export', D['E']['close'].shape, D['iso'][0], '..', D['iso'][-1])
    raw = None
    if forward is None:
        raw = load_raw(D, tag)
        C.log(f"raw store: {raw['n_diff']} stock-days differ from the export where both exist; {raw['n_only_raw']} "
              f"stock-days have a raw close the export blanked")
    per = build_periods(D, tag, raw=raw, nodiv=forward is None,
                        forward_start=None if forward is None else forward['start'])
    labels = [r['label'] for r in per]
    names = list(per[0]['ports'])
    S = {}
    for nm in names:
        days, g, t, pid = chained(per, nm)
        S[nm] = (g, t)
    days, pid = days, pid
    iso = D['iso'][days]
    lab_day = np.array([labels[k] for k in pid])
    np.savez(C.WORK / f'series_{tag}.npz', days=days, iso=iso, pid=pid, labels=np.array(labels),
             **{f'{k}_gross': v[0] for k, v in S.items()}, **{f'{k}_turn': v[1] for k, v in S.items()})

    def dser(a, b, cost, mask):
        ra = C.net(*S[a], cost) if cost else S[a][0]
        rb = C.net(*S[b], cost) if cost else S[b][0]
        return (np.log1p(ra) - np.log1p(rb))[mask]

    if tag == 'old':
        spans = {'dev': np.isin(lab_day, [str(y) for y in range(2008, 2020)]),
                 'check': np.isin(lab_day, [str(y) for y in range(2020, 2026)]),
                 'dev_ex0809': np.isin(lab_day, [str(y) for y in range(2010, 2020)])}
    else:
        spans = {'unseen': np.ones(len(days), bool)}
    comps = {'P1': ('R1d', 'B1', C.COST), 'P2': ('R1m', 'B1', C.COST),
             'P1 gross': ('R1d', 'B1', 0), 'P2 gross': ('R1m', 'B1', 0),
             'P1 20 bps': ('R1d', 'B1', C.COST20), 'P2 20 bps': ('R1m', 'B1', C.COST20),
             'quarterly 5 bps': ('R1q', 'B1', C.COST), 'quarterly gross': ('R1q', 'B1', 0),
             'delay 0 daily 5 bps': ('R0d', 'B0', C.COST), 'delay 0 monthly 5 bps': ('R0m', 'B0', C.COST),
             'delay 0 daily gross': ('R0d', 'B0', 0), 'delay 0 monthly gross': ('R0m', 'B0', 0)}
    if raw is not None:
        comps.update({'P1 raw closes': ('R1d_raw', 'B1_raw', C.COST), 'P2 raw closes': ('R1m_raw', 'B1_raw', C.COST),
                      'P1 raw closes gross': ('R1d_raw', 'B1_raw', 0), 'P2 raw closes gross': ('R1m_raw', 'B1_raw', 0)})
    res = {'tag': tag, 'labels': labels, 'tests': {}, 'periods': [], 'turnover': {}}
    for sp, mask in spans.items():
        res['tests'][sp] = {}
        for cname, (a, b, cost) in comps.items():
            res['tests'][sp][cname] = C.test_series(dser(a, b, cost, mask))
        yrs = mask.sum() / 252.0
        res['turnover'][sp] = {k: float(S[k][1][mask].sum() / yrs) for k in names}
    # Holm on the primaries (development only)
    main_span = 'dev' if tag == 'old' else 'unseen'
    ps = [res['tests'][main_span]['P1']['p'], res['tests'][main_span]['P2']['p']]
    res['holm'] = {'p': ps, 'pass': C.holm(ps) if tag == 'old' else None,
                   't3': [p < C.P_T3 for p in ps]}
    # break-even cost per side over the main span: gross gap per year / extra turnover per year
    be = {}
    for cname, (a, b) in (('P1', ('R1d', 'B1')), ('P2', ('R1m', 'B1')), ('quarterly', ('R1q', 'B1'))):
        gg = res['tests'][main_span][f'{cname} gross' if cname != 'quarterly' else 'quarterly gross']['per_year']
        ex = res['turnover'][main_span][a] - res['turnover'][main_span][b]
        be[cname] = {'gross_per_year': gg, 'extra_turnover_per_year': ex, 'breakeven_bps': 1e4 * gg / ex if ex > 0 else None}
    res['breakeven'] = be

    # E1, L and the subsets, per period
    e1 = []
    subs = []
    for r in per:
        e = e1_period(r)
        s = subsets_period(r)
        e1.append(e)
        subs.append(s)
        res['periods'].append({
            'label': r['label'], 'span': f"{D['iso'][r['p']['s']]}..{D['iso'][r['p']['end']]}", 'K': len(r['rows']),
            'n_elig': r['n_elig'], 'n_drop': r['n_drop'], 'n_end': r['H']['n_end'],
            'untradeable': int((~r['H']['trade'][:, 1:]).sum()), 'n': e['n'],
            'gamma_day': r['gamma'], 'pred': e['pred'], 'EG_d': e['EG_d'], 'EG_m': e['EG_m'], 'J': e['J'],
            'gap_d0': e['gap_d0'], 'gap_m0': e['gap_m0'], 'gap_d1': e['gap_d1'], 'gap_m1': e['gap_m1'],
            'rw_EG_d': float(e['sim_EG_d'].mean()), 'rw_J': float(e['sim_J'].mean()),
            'rw_gap_d': float(e['sim_gap_d'].mean()), 'rw_gap_m': float(e['sim_gap_m'].mean()),
            'rw_gap_d_sd': float(e['sim_gap_d'].std(ddof=1)),
            'gap2': e['gap2'], 'bands': e['bands'], 'remainder': e['gap_d0'] - e['gap2'],
            'sub_gamma': float(s['gamma'].mean()), 'sub_EG_d': float(s['EG_d'].mean()), 'sub_J': float(s['J'].mean()),
            'sub_gap_d0': float(s['gap_d0'].mean()),
            'sub_gap_d1': float(s['gap_d1'].mean()), 'sub_gap_d1_sd': float(s['gap_d1'].std(ddof=1)),
            'sub_net_d1': float(s['net_d1'].mean()), 'sub_net_d1_sd': float(s['net_d1'].std(ddof=1)),
            'sub_net_d1_win': float((s['net_d1'] > 0).mean()),
            'sub_gap_m1': float(s['gap_m1'].mean()), 'sub_net_m1': float(s['net_m1'].mean()),
            'sub_net_m1_sd': float(s['net_m1'].std(ddof=1)), 'sub_net_m1_win': float((s['net_m1'] > 0).mean())})
        C.log(f"E1 {r['label']}: pred {e['pred']:.4f} EG_d {e['EG_d']:.4f} J {e['J']:.4f} gap_d0 {e['gap_d0']:+.4f} "
              f"RW gap {e['sim_gap_d'].mean():+.4f}")
    np.savez(C.WORK / f'e1_{tag}.npz', **{f"G_{r['label']}": e['G'] for r, e in zip(per, e1)},
             **{f"g_{r['label']}": e['g'] for r, e in zip(per, e1)},
             **{f"rows_{r['label']}": r['rows'] for r in per})

    # E1's readings over the development periods
    if tag == 'old':
        dv = [k for k, r in enumerate(per) if 2008 <= int(r['label']) <= 2019]
    else:
        dv = list(range(len(per)))
    pred = np.array([e1[k]['pred'] for k in dv])
    EGd = np.array([e1[k]['EG_d'] for k in dv])
    EGm = np.array([e1[k]['EG_m'] for k in dv])
    J = np.array([e1[k]['J'] for k in dv])
    gap = np.array([e1[k]['gap_d0'] for k in dv])
    simtot = np.sum([e1[k]['sim_gap_d'] for k in dv], 0)
    simtot_m = np.sum([e1[k]['sim_gap_m'] for k in dv], 0)
    gapm = np.array([e1[k]['gap_m0'] for k in dv])

    def slope(y, x):
        Xm = np.column_stack([np.ones_like(x), x])
        b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        e = y - Xm @ b
        if len(y) <= 2:
            return float(b[1]), float('nan'), float('nan')
        s2 = e @ e / (len(y) - 2)
        se = np.sqrt(s2 * np.linalg.inv(Xm.T @ Xm)[1, 1])
        tq = {10: 2.228, 1: 12.706}.get(len(y) - 2, 2.0)
        return float(b[1]), float(b[1] - tq * se), float(b[1] + tq * se)

    # e_t - 1/2 gamma* for the claim's reading
    gross_d0 = np.log1p(S['R0d'][0]) - np.log1p(S['B0'][0])
    gam_day = np.array([per[k]['gamma'] for k in pid])
    msk = spans[main_span]
    t_claim = C.nw_t((gross_d0 - 0.5 * gam_day)[msk], C.NW_LAGS)
    rd = {'ratio_d': float(EGd.sum() / pred.sum()), 'ratio_m': float(EGm.sum() / pred.sum()),
          'ratio_J': float(J.sum() / pred.sum()),
          'slope_d': slope(EGd, pred), 'slope_m': slope(EGm, pred),
          'sum_pred': float(pred.sum()), 'sum_EG_d': float(EGd.sum()), 'sum_EG_m': float(EGm.sum()),
          'sum_J': float(J.sum()), 'sum_gap_d0': float(gap.sum()), 'sum_gap_m0': float(gapm.sum()),
          't_claim': t_claim, 'claim_rejected': bool(t_claim <= -3),
          'rw_mean': float(simtot.mean()), 'rw_p2_5': float(np.percentile(simtot, 2.5)),
          'rw_p97_5': float(np.percentile(simtot, 97.5)), 'rw_pct': float((simtot < gap.sum()).mean() * 100),
          'rw_m_mean': float(simtot_m.mean()), 'rw_m_p2_5': float(np.percentile(simtot_m, 2.5)),
          'rw_m_p97_5': float(np.percentile(simtot_m, 97.5)), 'rw_m_pct': float((simtot_m < gapm.sum()).mean() * 100),
          'bands': np.sum([e1[k]['bands'] for k in dv], 0).tolist(), 'gap2': float(sum(e1[k]['gap2'] for k in dv))}
    rd['gamma_predicts'] = bool(0.8 <= rd['ratio_d'] <= 1.25 and rd['slope_d'][1] <= 1 <= rd['slope_d'][2])
    rd['rw_consistent'] = bool(rd['rw_p2_5'] <= gap.sum() <= rd['rw_p97_5'])
    sub = {'net_d1': float(np.mean([subs[k]['net_d1'].mean() for k in dv])),
           'net_m1': float(np.mean([subs[k]['net_m1'].mean() for k in dv])),
           'gap_d1': float(np.mean([subs[k]['gap_d1'].mean() for k in dv])),
           'win_d1': float(np.mean([(subs[k]['net_d1'] > 0).mean() for k in dv])),
           'win_m1': float(np.mean([(subs[k]['net_m1'] > 0).mean() for k in dv])),
           'sd_net_d1': float(np.mean([subs[k]['net_d1'].std(ddof=1) for k in dv])),
           'ratio_EG': float(np.sum([subs[k]['EG_d'].mean() for k in dv]) / np.sum([subs[k]['gamma'].mean() for k in dv])),
           'ratio_J': float(np.sum([subs[k]['J'].mean() for k in dv]) / np.sum([subs[k]['gamma'].mean() for k in dv])),
           'gap_d0': float(np.mean([subs[k]['gap_d0'].mean() for k in dv])),
           'pred': float(np.mean([subs[k]['gamma'].mean() for k in dv]))}
    res['E1'] = rd
    res['subsets'] = sub
    if tag == 'old':
        res['H5'] = h5(D)

    # calendar-year table (by return date)
    yr = np.array([s[:4] for s in iso])
    years = []
    for y in sorted(set(yr)):
        m = yr == y
        row = {'year': y, 'days': int(m.sum())}
        for k in ('R1d', 'R1m', 'B1'):
            row[f'{k}_gross'] = float(100 * (np.prod(1 + S[k][0][m]) - 1))
            row[f'{k}_net5'] = float(100 * (np.prod(1 + C.net(*S[k], C.COST)[m]) - 1))
            row[f'{k}_turn'] = float(S[k][1][m].sum())
        for cname in ('P1', 'P2'):
            a, b, cost = comps[cname]
            row[f'{cname}_net_log'] = float((np.log1p(C.net(*S[a], cost)) - np.log1p(C.net(*S[b], cost)))[m].sum())
            row[f'{cname}_gross_log'] = float((np.log1p(S[a][0]) - np.log1p(S[b][0]))[m].sum())
        years.append(row)
    res['years'] = years
    with open(C.HERE / f'years_{tag}.tsv', 'w') as f:
        keys = list(years[0])
        f.write('\t'.join(keys) + '\n')
        for row in years:
            f.write('\t'.join(str(round(v, 6)) if isinstance(v, float) else str(v) for v in row.values()) + '\n')
    report(res, tag)
    js = json.loads(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    (C.HERE / f'p_{tag}.json').write_text(json.dumps(js, indent=1))
    return res


def report(res, tag):
    L = []
    say = L.append
    say(f'Excess growth: rebalanced equal weight minus buy-and-hold of the same stocks ({tag} export)')
    say('Log points a year (mean daily log difference x 252). p = the larger of the bootstrap p and the NW-21 normal p.')
    for sp, tests in res['tests'].items():
        say(f'\n== {sp}')
        say(f"{'comparison':26s} {'days':>5s} {'per year':>9s} {'NW t21':>7s} {'NW t63':>7s} {'p boot':>7s} {'p':>7s} "
            f"{'boot 5..95%':>17s}")
        for cname, t in tests.items():
            say(f"{cname:26s} {t['days']:5d} {100 * t['per_year']:+8.3f}% {t['t_nw21']:+7.2f} {t['t_nw63']:+7.2f} "
                f"{t['p_boot']:7.4f} {t['p']:7.4f} {100 * t['boot5']:+7.3f}..{100 * t['boot95']:+7.3f}")
        say('turnover a year (sum |trade| / value, both sides): ' +
            ', '.join(f'{k} {v:.2f}' for k, v in res['turnover'][sp].items()))
    h = res['holm']
    say(f"\nHolm (family 0.0228; thresholds 0.0114, 0.0228): p P1 {h['p'][0]:.4f}, P2 {h['p'][1]:.4f} -> pass {h['pass']}; "
        f"t >= 3 (p < 0.00135): {h['t3']}")
    say('Break-even cost per side: ' + '; '.join(
        f"{k}: gross {100 * v['gross_per_year']:+.3f}%/yr, extra turnover {v['extra_turnover_per_year']:.2f}/yr -> "
        + (f"{v['breakeven_bps']:+.2f} bps" if v['breakeven_bps'] is not None else 'n/a') for k, v in res['breakeven'].items()))
    e = res['E1']
    say('\n== E1 (development periods; log points summed over the periods)')
    say(f"predicted sum gamma*.n {e['sum_pred']:.4f}; realized EG daily {e['sum_EG_d']:.4f} (ratio {e['ratio_d']:.3f}), "
        f"monthly {e['sum_EG_m']:.4f} (ratio {e['ratio_m']:.3f}); buy-and-hold J {e['sum_J']:.4f} (ratio {e['ratio_J']:.3f})")
    say(f"slope of EG on predicted across periods: daily {e['slope_d'][0]:.3f} (95% {e['slope_d'][1]:.3f}..{e['slope_d'][2]:.3f}), "
        f"monthly {e['slope_m'][0]:.3f} ({e['slope_m'][1]:.3f}..{e['slope_m'][2]:.3f}) -> gamma* predicts EG: {e['gamma_predicts']}")
    say(f"gap (daily, delay 0, gross) {e['sum_gap_d0']:+.4f}; claim's prediction {e['sum_pred']:.4f}; NW t of "
        f"(e_t - gamma*/2) {e['t_claim']:+.2f} -> claim rejected (t <= -3): {e['claim_rejected']}")
    say(f"random walk: simulated total mean {e['rw_mean']:+.4f}, 2.5..97.5% {e['rw_p2_5']:+.4f}..{e['rw_p97_5']:+.4f}; "
        f"realized at the {e['rw_pct']:.1f}th percentile -> consistent: {e['rw_consistent']}")
    say(f"monthly gap (delay 0, gross) {e['sum_gap_m0']:+.4f}; RW {e['rw_m_mean']:+.4f} ({e['rw_m_p2_5']:+.4f}..{e['rw_m_p97_5']:+.4f}), "
        f"percentile {e['rw_m_pct']:.1f}")
    say(f"L: second-order gap {e['gap2']:+.4f} by lag band 1 / 2-5 / 6-21 / 22-63 / 64+: " +
        ' / '.join(f'{b:+.4f}' for b in e['bands']) + f"; remainder {e['sum_gap_d0'] - e['gap2']:+.4f}")
    s = res['subsets']
    say(f"\n20-stock subsets (mean over periods of the mean over 1,000 subsets, log points per period): gross gap daily "
        f"{s['gap_d1']:+.4f}, net daily {s['net_d1']:+.4f} (sd across subsets {s['sd_net_d1']:.4f}, rebalancing ahead in "
        f"{100 * s['win_d1']:.1f}%), net monthly {s['net_m1']:+.4f} (ahead {100 * s['win_m1']:.1f}%); predicted gamma*.n "
        f"{s['pred']:.4f}, EG/pred {s['ratio_EG']:.3f}, J/pred {s['ratio_J']:.3f}, delay-0 gap {s['gap_d0']:+.4f}")
    say('\n== Periods (log points over the period)')
    say(f"{'period':7s} {'span':23s} {'K':>4s} {'ends':>4s} {'untr':>5s} {'pred':>7s} {'EG_d':>7s} {'EG_m':>7s} {'J':>7s} "
        f"{'gap_d0':>8s} {'gap_d1':>8s} {'gap_m1':>8s} {'RW gap':>8s} {'gap2':>8s} {'lag1':>8s} {'sub net d':>9s} {'sub win':>7s}")
    for p in res['periods']:
        say(f"{p['label']:7s} {p['span']:23s} {p['K']:4d} {p['n_end']:4d} {p['untradeable']:5d} {p['pred']:7.4f} "
            f"{p['EG_d']:7.4f} {p['EG_m']:7.4f} {p['J']:7.4f} {p['gap_d0']:+8.4f} {p['gap_d1']:+8.4f} {p['gap_m1']:+8.4f} "
            f"{p['rw_gap_d']:+8.4f} {p['gap2']:+8.4f} {p['bands'][0]:+8.4f} {p['sub_net_d1']:+9.4f} {100 * p['sub_net_d1_win']:6.1f}%")
    if 'H5' in res:
        say('\n== H5, five-year holds (log points over the hold)')
        for r in res['H5']:
            say(f"{r['span']}: K {r['K']}, pred {r['pred']:.4f}, EG {r['EG_d0']:.4f}, J {r['J0']:.4f}, gap delay 0 "
                f"{r['gap_d0']:+.4f}; delay 1 gross daily {r['gap_d1']:+.4f}, monthly {r['gap_m1']:+.4f}; net daily "
                f"{r['net_d1']:+.4f}, monthly {r['net_m1']:+.4f}")
    say('\n== Calendar years (percent, compounded by return date)')
    say(f"{'year':5s} {'days':>4s} {'R1d gross':>9s} {'B1 gross':>9s} {'R1d net5':>9s} {'R1m net5':>9s} {'B1 net5':>9s} "
        f"{'P1 net log':>10s} {'P2 net log':>10s} {'R1d turn':>8s}")
    for r in res['years']:
        say(f"{r['year']:5s} {r['days']:4d} {r['R1d_gross']:+9.2f} {r['B1_gross']:+9.2f} {r['R1d_net5']:+9.2f} "
            f"{r['R1m_net5']:+9.2f} {r['B1_net5']:+9.2f} {100 * r['P1_net_log']:+10.3f} {100 * r['P2_net_log']:+10.3f} "
            f"{r['R1d_turn']:8.2f}")
    txt = '\n'.join(L) + '\n'
    (C.HERE / f'p_{tag}.txt').write_text(txt)
    print(txt, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
