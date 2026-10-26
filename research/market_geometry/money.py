"""B3 money tests: the curvature market-allocation rule against its baselines and 10,000 random timings, and the
fund risk scaling. Rules in results_B.md, "Decisions" 28-34 and 49.

    python -I -B money.py old       (after curvature.py old and tests.py old)
Writes money_old.txt, money_old.json here. The functions take a tag so unseen.py can rerun them on the new export.
"""
import json
import pickle
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_geometry')
import numpy as np  # noqa: E402

import b_common as B  # noqa: E402

c2eval = B.c2eval
T_ = B.load_module('b_tests', B.HERE / 'tests.py')
H = 21
DPY = 252
N_RANDOM = 10_000
SEED_RANDOM = 20261021        # -3.2's seed
START = '2005-04-01'
PERIODS = {'2005-04..2025-06': ('2005-04-01', '2025-06-27'), '2005-04..2019-12': ('2005-04-01', '2019-12-31'),
           '2020-01..2025-06': ('2020-01-01', '2025-06-27')}
BLOCKS = {'2005-04..2009': ('2005-04-01', '2009-12-31'), '2010-2014': ('2010-01-01', '2014-12-31'),
          '2015-2019': ('2015-01-01', '2019-12-31'), '2020-2025-06': ('2020-01-01', '2025-06-27')}


def signal_weights(x, direction):
    """1 when x[k] is strictly above (+1) or strictly below (-1) the median of the previous 60 finite values (>= 24),
    else 0.5; NaN without a median (research/market_timing/engine.py's rule, copied)."""
    K = len(x)
    med = np.array([B.trailing_median(x, k) for k in range(K)])
    hi = x > med if direction > 0 else x < med
    w = np.where(hi, 1.0, 0.5)
    w[~np.isfinite(med) | ~np.isfinite(x)] = np.nan
    return w, med


def paths(t0s, P, rfd, lag):
    """Per month: SPY relative path P[e+j]/P[e] and the T-bill cash path, j = 0..21, e = t0 + lag."""
    e = np.asarray(t0s) + lag
    idx = e[:, None] + np.arange(H + 1)[None, :]
    rel = P[idx] / P[e][:, None]
    g = np.ones_like(rel)
    g[:, 1:] = np.cumprod(1.0 + rfd[idx[:, 1:]], 1)
    assert np.isfinite(rel).all() and np.isfinite(g).all()
    return rel, g, idx


def simulate(w, rel, g, cost, cash_rf=True):
    """Buy w of SPY at the month's first close (paying cost on |w - drifted weight|), hold 21 days. Returns daily
    returns [M, 21], monthly returns [M], turnover [M]."""
    M = len(w)
    cg = g if cash_rf else np.ones_like(g)
    dr = np.empty((M, H))
    mr = np.empty(M)
    turn = np.empty(M)
    s_prev = 0.0
    for m in range(M):
        tr = abs(w[m] - s_prev)
        turn[m] = tr
        v0 = 1.0 - tr * cost
        V = v0 * (w[m] * rel[m] + (1 - w[m]) * cg[m])
        Vfull = np.concatenate([[1.0], V[1:]])
        dr[m] = Vfull[1:] / Vfull[:-1] - 1.0
        mr[m] = V[-1] - 1.0
        s_prev = w[m] * rel[m, -1] / (w[m] * rel[m, -1] + (1 - w[m]) * cg[m, -1])
    return dr, mr, turn


def measures(dr, rf_days):
    f = dr.ravel()
    ex = f - rf_days.ravel()
    lg = np.log1p(f).sum()
    v = np.concatenate([[1.0], np.cumprod(1.0 + f)])
    return {'annual': float(np.exp(lg * DPY / len(f)) - 1.0), 'sharpe': float(ex.mean() / ex.std(ddof=1) * np.sqrt(DPY)),
            'vol': float(f.std(ddof=1) * np.sqrt(DPY)), 'mdd': float(np.max(1.0 - v / np.maximum.accumulate(v)))}


def vol_weights(var, target):
    """min(1, c / var) with c by bisection so the mean weight equals target (research/market_timing/engine.py)."""
    var = np.asarray(var, float)
    lo, hi = np.log(var.min() * 1e-6), np.log(var.max() * 1e6)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.minimum(1.0, np.exp(mid) / var).mean() < target:
            lo = mid
        else:
            hi = mid
    c = np.exp(0.5 * (lo + hi))
    return np.minimum(1.0, c / var), float(c)


def random_timing(w, rel, g, n=N_RANDOM, seed=SEED_RANDOM):
    """Gross compound annual returns of n random orderings of the rule's monthly weights (T-bill cash)."""
    rng = np.random.default_rng(seed)
    gross = lambda ww: ww[:, None] * rel[:, -1] + (1 - ww[:, None]) * g[:, -1]  # noqa: E731
    vals = np.unique(w)
    lg = {v: np.log(v * rel[:, -1] + (1 - v) * g[:, -1]) for v in vals}
    L = np.stack([lg[v] for v in vals])                     # [V, M]
    code = np.searchsorted(vals, w)
    out = np.empty(n)
    M = len(w)
    for a in range(n):
        cp = rng.permutation(code)
        out[a] = np.exp(L[cp, np.arange(M)].sum() * DPY / (M * H)) - 1.0
    del gross
    return out


def build(tag='old', lag=1):
    M, groups, days, P, rfd, lr = T_.load(tag)
    st = np.load(B.mt_common().WORK / f'states_{tag}.npz')
    assert np.array_equal(st['rebal'], M['t0'])
    T = len(days)
    var21 = np.full(M['K'], np.nan)
    for k, t0 in enumerate(M['t0']):
        r = lr[t0 - 20:t0 + 1]
        if np.isfinite(r).all():
            var21[k] = float(np.var(r))
    W = {}
    W['ollivier'], med_o = signal_weights(M['ollivier'], -1)       # calm = below median (Ollivier rises in crises)
    W['forman'], _ = signal_weights(M['forman'], +1)               # calm = above median (Forman falls in crises)
    W['vol_threshold'], _ = signal_weights(np.sqrt(var21), -1)
    W['avg_corr'], _ = signal_weights(st['avg_corr'], +1)
    W['absorption'], _ = signal_weights(st['absorption'], -1)
    ok = (M['date'] >= START) & (M['t0'] + 22 <= T - 1) & np.isfinite(var21)
    for v in W.values():
        ok &= np.isfinite(v)
    ks = np.nonzero(ok)[0]
    assert np.all(np.diff(M['t0'][ks]) == 21)
    return M, days, P, rfd, var21, W, ks, med_o


def evaluate(M, days, P, rfd, var21, W, ks, lag=1, cost=5e-4, cash_rf=True, random=True):
    rel, g, idx = paths(M['t0'][ks], P, rfd, lag)
    rf_days = rfd[idx[:, 1:]] if cash_rf else np.zeros((len(ks), H))
    out = {'n_months': int(len(ks)), 'first': M['date'][ks[0]], 'last': M['date'][ks[-1]]}
    w_rule = W['ollivier'][ks]
    strat = {'rule (Ollivier)': w_rule, 'Forman rule': W['forman'][ks], 'buy and hold': np.ones(len(ks)),
             'constant': np.full(len(ks), w_rule.mean()), 'volatility threshold': W['vol_threshold'][ks],
             'avg-correlation rule (-3.2)': W['avg_corr'][ks], 'absorption rule (-3.2)': W['absorption'][ks]}
    wv, c = vol_weights(var21[ks], w_rule.mean())
    strat['volatility-managed'] = wv
    res = {}
    mret = {}
    for nm, w in strat.items():
        dr0, mr0, tr0 = simulate(w, rel, g, 0.0, cash_rf)
        dr5, mr5, tr5 = simulate(w, rel, g, cost, cash_rf)
        dr20, mr20, _ = simulate(w, rel, g, 20e-4, cash_rf)
        res[nm] = {'gross': measures(dr0, rf_days), 'net5': measures(dr5, rf_days), 'net20': measures(dr20, rf_days),
                   'exposure': float(w.mean()), 'turnover_per_year': float(tr5.sum() / (len(ks) / 12.0)),
                   'switches': int(np.sum(np.abs(np.diff(w)) > 0))}
        mret[nm] = (mr0, mr5, dr5)
    res['volatility-managed']['c'] = c
    out['strategies'] = res
    if random:
        rd = random_timing(w_rule, rel, g if cash_rf else np.ones_like(g))
        ra = res['rule (Ollivier)']['gross']['annual']
        out['random'] = {'median': float(np.median(rd)), 'p': float((1 + np.sum(rd >= ra)) / (1 + len(rd))),
                         'number': round(100 * (ra - float(np.median(rd))), 1), 'p05': float(np.percentile(rd, 5)),
                         'p95': float(np.percentile(rd, 95))}
        out['random_draws_hist'] = np.histogram(rd, bins=40)[0].tolist(), np.histogram(rd, bins=40)[1].tolist()
    return out, mret, rel, g, idx


def paired(mret, a, b):
    d = mret[a][1] - mret[b][1]          # monthly net-of-5-bps returns
    return B.mean_test(d)


def after_costs(M, days, P, rfd, W, ks, lag=1, cash_rf=True):
    """Calendar-year return of the rule at 5 bps minus SPY's total return over the same days, averaged over years."""
    rel, g, idx = paths(M['t0'][ks], P, rfd, lag)
    dr, _, _ = simulate(W['ollivier'][ks], rel, g, 5e-4, cash_rf)
    d_idx = idx[:, 1:].ravel()
    r = dr.ravel()
    spy = P[d_idx] / P[d_idx - 1] - 1.0
    yr = np.array([c2eval.iso(days[d])[:4] for d in d_idx])
    rows = {}
    for y in sorted(set(yr)):
        m = yr == y
        a, s = 100 * (np.prod(1 + r[m]) - 1), 100 * (np.prod(1 + spy[m]) - 1)
        rows[y] = {'rule5': float(a), 'spy': float(s), 'diff': float(a - s), 'days': int(m.sum())}
    return {'years': rows, 'after_cost': float(np.mean([v['diff'] for v in rows.values()])),
            'first_day': c2eval.iso(days[d_idx[0]]), 'last_day': c2eval.iso(days[d_idx[-1]])}


# ---------------------------------------------------------------------------------------------------------------
# Fund risk scaling

def fund_scaling(tag, M, days, rfd, lo, hi):
    with open(B.WORK / f'fund_forecasts_{tag}.pkl', 'rb') as f:
        FF = pickle.load(f)
    E = c2eval.load_export(B.EXPORTS[tag], arrays=('close',))
    tw = c2eval.twins(E)
    cp = B.cp_common()
    Fc = np.asarray(tw['F']['close'], np.float64)
    T = len(days)
    k_all = FF['k']
    okr = np.isfinite(FF['m3']) & np.isfinite(FF['m2']) & np.isfinite(FF['rv63']) & (FF['rv63'] > 0)
    months = [k for k in np.unique(k_all[okr]) if lo <= M['date'][k] <= hi and M['t0'][k] + 22 <= T - 1]
    vers = {'with curvature (model 3)': lambda r: np.exp(FF['m3'][r]), 'without curvature (model 2)': lambda r: np.exp(FF['m2'][r]),
            'trailing 63-day volatility': lambda r: FF['rv63'][r]}
    out = {nm: {'dr': [], 'rv': [], 'turn': [], 'rf': []} for nm in vers}
    prev = {nm: {} for nm in vers}
    prev_k = None
    n_funds = []
    for k in months:
        t0 = int(M['t0'][k])
        rows = np.nonzero((k_all == k) & okr)[0]
        fr = FF['f'][rows]
        Pw = cp.ffill_rows(Fc[fr, t0 - 300:t0 + 23])[:, 301:]          # closes t0+1..t0+22
        with np.errstate(all='ignore'):
            Rd = Pw[:, 1:] / Pw[:, :-1] - 1.0
        bad = ~np.isfinite(Rd) | (np.abs(np.log1p(np.where(np.isfinite(Rd), Rd, 0.0))) > 0.5)
        Rd[bad] = 0.0
        V = np.cumprod(np.concatenate([np.ones((len(fr), 1)), 1.0 + Rd], 1), 1)
        n_funds.append(len(fr))
        for nm, fn in vers.items():
            s = fn(rows)
            w = (1.0 / s) / (1.0 / s).sum()
            contiguous = prev_k is not None and M['t0'][k] - M['t0'][prev_k] == 21
            old = prev[nm] if contiguous else {}
            new = dict(zip(fr.tolist(), w.tolist()))
            tr = sum(abs(new.get(x, 0.0) - old.get(x, 0.0)) for x in set(new) | set(old))
            val = (w @ V) * (1.0 - 5e-4 * tr)
            vf = np.concatenate([[1.0], val[1:]])
            d = vf[1:] / vf[:-1] - 1.0
            out[nm]['dr'].append(d)
            out[nm]['rv'].append(float(d @ d))
            out[nm]['turn'].append(tr)
            out[nm]['rf'].append(rfd[t0 + 2:t0 + 23])
            end = w * V[:, -1] / (w @ V[:, -1])
            prev[nm] = dict(zip(fr.tolist(), end.tolist()))
        prev_k = k
    if not months:
        return {'months': 0, 'first': None, 'last': None}
    res = {'months': len(months), 'first': M['date'][months[0]] if months else None,
           'last': M['date'][months[-1]] if months else None, 'funds_per_month_median': float(np.median(n_funds)) if n_funds else 0}
    for nm in vers:
        dr = np.concatenate(out[nm]['dr'])
        rfa = np.concatenate(out[nm]['rf'])
        ex = dr - rfa
        res[nm] = {'vol': float(dr.std(ddof=1) * np.sqrt(DPY)), 'sharpe': float(ex.mean() / ex.std(ddof=1) * np.sqrt(DPY)),
                   'annual': float(np.exp(np.log1p(dr).sum() * DPY / len(dr)) - 1), 'mean_monthly_rv': float(np.mean(out[nm]['rv'])),
                   'turnover_per_year': float(np.sum(out[nm]['turn']) / (len(months) / 12.0))}
    a = np.array(out['without curvature (model 2)']['rv'])
    b = np.array(out['with curvature (model 3)']['rv'])
    res['test_rv_without_minus_with'] = B.mean_test(a - b)
    c = np.array(out['trailing 63-day volatility']['rv'])
    res['rv_trailing_minus_with'] = B.mean_test(c - b)
    return res


def main():
    tag = 'old'
    M, days, P, rfd, var21, W, ks, med_o = build(tag)
    dates = M['date'][ks]
    R = {'periods': {}, 'blocks': {}}
    for nm, (lo, hi) in PERIODS.items():
        sub = ks[(dates >= lo) & (dates <= hi)]
        ev, mret, _, _, _ = evaluate(M, days, P, rfd, var21, W, sub)
        better = max(('avg-correlation rule (-3.2)', 'absorption rule (-3.2)'), key=lambda s: ev['strategies'][s]['net5']['annual'])
        ev['better_32_rule'] = better
        ev['paired'] = {b: paired(mret, 'rule (Ollivier)', b) for b in
                        ('constant', 'volatility threshold', better, 'volatility-managed', 'buy and hold',
                         'avg-correlation rule (-3.2)', 'absorption rule (-3.2)')}
        R['periods'][nm] = ev
    for nm, (lo, hi) in BLOCKS.items():
        sub = ks[(dates >= lo) & (dates <= hi)]
        ev, _, _, _, _ = evaluate(M, days, P, rfd, var21, W, sub)
        ev.pop('random_draws_hist', None)
        R['blocks'][nm] = ev
    # upper bound (trade at t0) and cash at 0, whole span
    R['at_t0'] = evaluate(M, days, P, rfd, var21, W, ks, lag=0)[0]
    R['at_t0'].pop('random_draws_hist', None)
    R['cash0'] = evaluate(M, days, P, rfd, var21, W, ks, cash_rf=False)[0]
    R['cash0'].pop('random_draws_hist', None)
    R['after_costs'] = after_costs(M, days, P, rfd, W, ks)
    R['after_costs_cash0'] = after_costs(M, days, P, rfd, W, ks, cash_rf=False)
    dev = R['periods']['2005-04..2019-12']
    trio = [dev['paired'][b] for b in ('constant', 'volatility threshold', dev['better_32_rule'])]
    R['b3_market_pass_raw'] = bool(all(x['mean'] > 0 for x in trio))
    R['b3_market_p'] = float(max(x['p'] for x in trio))
    R['number'] = R['periods']['2005-04..2025-06']['random']['number']
    R['fund_scaling'] = {'2008-2019': fund_scaling(tag, M, days, rfd, '2008-01-01', '2019-12-31'),
                         '2020-2025-06': fund_scaling(tag, M, days, rfd, '2020-01-01', '2025-06-27')}
    # Holm across B's secondary family (decision 49)
    tj = json.loads((B.HERE / 'tests_old.json').read_text())
    fam = {'SPY max drawdown, model 3 vs 2': tj['b2_spy']['y_mdd']['2008-2019']['m2->m3']['p'],
           'funds excess return, model 3 vs 2': (tj['b2_funds']['y_dir']['2008-2019']['m2->m3'] or {}).get('p', 1.0),
           'funds log RV, model 3 vs 2': (tj['b2_funds']['y_risk']['2008-2019']['m2->m3'] or {}).get('p', 1.0),
           'funds max drawdown, model 3 vs 2': (tj['b2_funds']['y_mdd']['2008-2019']['m2->m3'] or {}).get('p', 1.0),
           'B3 market rule (largest of its 3 p)': R['b3_market_p'],
           'B3 fund risk scaling': R['fund_scaling']['2008-2019']['test_rv_without_minus_with']['p']}
    adj, ok = B.holm(list(fam.values()))
    R['holm_secondary'] = {k: {'p': float(v), 'holm_p': float(a), 'pass': bool(o)} for (k, v), a, o in zip(fam.items(), adj, ok)}
    R['monthly'] = {'date': dates.tolist(), 'w_rule': W['ollivier'][ks].tolist(), 'ollivier': M['ollivier'][ks].tolist(),
                    'median': med_o[ks].tolist()}
    rel, g, idx = paths(M['t0'][ks], P, rfd, 1)
    for nm in ('rule (Ollivier)', 'buy and hold', 'constant', 'volatility threshold'):
        w = {'rule (Ollivier)': W['ollivier'][ks], 'buy and hold': np.ones(len(ks)),
             'constant': np.full(len(ks), W['ollivier'][ks].mean()), 'volatility threshold': W['vol_threshold'][ks]}[nm]
        dr, _, _ = simulate(w, rel, g, 5e-4)
        R['monthly'][f'value_{nm}'] = np.cumprod(1 + dr.ravel())[H - 1::H].tolist()
    (B.HERE / f'money_{tag}.json').write_text(json.dumps(R, indent=1, default=float))
    report(R)


def pct(v):
    return f'{100 * v:+.1f}%'


def report(R):
    L = []
    p = L.append
    for nm, ev in R['periods'].items():
        p(f"\n## {nm}: {ev['n_months']} months ({ev['first']} .. {ev['last']}), bought at t0+1, T-bill cash")
        p('| | Annual (gross) | Annual 5 bps | Annual 20 bps | Sharpe 5 bps | Vol | Max DD | Exposure | Turnover/yr |')
        p('|---|---|---|---|---|---|---|---|---|')
        for s, r in ev['strategies'].items():
            p(f"| {s} | {pct(r['gross']['annual'])} | {pct(r['net5']['annual'])} | {pct(r['net20']['annual'])} | {r['net5']['sharpe']:.2f} | "
              f"{100 * r['net5']['vol']:.1f}% | {pct(-r['net5']['mdd'])} | {r['exposure']:.0%} | {r['turnover_per_year']:.2f} |")
        rd = ev['random']
        p(f"random timing median {pct(rd['median'])} (5-95%: {pct(rd['p05'])}..{pct(rd['p95'])}); rule - median {rd['number']:+.1f} pts, p {rd['p']:.3f}")
        p(f"better -3.2 rule here: {ev['better_32_rule']}")
        for b, x in ev['paired'].items():
            p(f"  rule - {b}: {100 * x['mean']:+.3f}% a month net, t {x['t']:+.2f}, p {x['p']:.3f}, boot p {x['p_boot']:.3f}, "
              f"90% {100 * x['ci90_nw'][0]:+.3f}..{100 * x['ci90_nw'][1]:+.3f}")
    p('\n## By period (random timing within each)')
    for nm, ev in R['blocks'].items():
        r = ev['strategies']
        p(f"{nm}: rule {pct(r['rule (Ollivier)']['gross']['annual'])}, B&H {pct(r['buy and hold']['gross']['annual'])}, random median "
          f"{pct(ev['random']['median'])}, diff {ev['random']['number']:+.1f} (p {ev['random']['p']:.2f}); Sharpe rule "
          f"{r['rule (Ollivier)']['net5']['sharpe']:.2f}, B&H {r['buy and hold']['net5']['sharpe']:.2f}, vol thr {r['volatility threshold']['net5']['sharpe']:.2f}, "
          f"vol-man {r['volatility-managed']['net5']['sharpe']:.2f}; exposure {r['rule (Ollivier)']['exposure']:.0%}")
    for key, lab in (('at_t0', 'traded at t0 (upper bound)'), ('cash0', 'cash at 0')):
        ev = R[key]
        r = ev['strategies']['rule (Ollivier)']
        p(f"{lab}: rule {pct(r['gross']['annual'])} vs random median {pct(ev['random']['median'])} ({ev['random']['number']:+.1f}, p {ev['random']['p']:.3f})")
    ac = R['after_costs']
    p(f"\nAfter costs (5 bps, T-bill cash, {len(ac['years'])} calendar years {ac['first_day']}..{ac['last_day']}): {ac['after_cost']:+.2f} pts a year vs SPY; "
      f"cash at 0: {R['after_costs_cash0']['after_cost']:+.2f}")
    p(f"File-name number (rule - median random, 2005-04..2025-06, before costs): {R['number']:+.1f}")
    p(f"B3 market rule (2005-04..2019-12): all three differences positive: {R['b3_market_pass_raw']}; largest p {R['b3_market_p']:.3f}")
    for nm, fs in R['fund_scaling'].items():
        p(f"\n## Fund risk scaling {nm}: {fs['months']} months ({fs['first']}..{fs['last']}), median {fs['funds_per_month_median']:.0f} funds")
        for v in ('with curvature (model 3)', 'without curvature (model 2)', 'trailing 63-day volatility'):
            x = fs[v]
            p(f"  {v}: vol {100 * x['vol']:.2f}%, Sharpe {x['sharpe']:.2f}, annual {pct(x['annual'])}, turnover/yr {x['turnover_per_year']:.2f}")
        x = fs['test_rv_without_minus_with']
        p(f"  monthly RV without - with curvature: {x['mean']:+.3e}, t {x['t']:+.2f}, p {x['p']:.3f}, boot p {x['p_boot']:.3f}")
        x = fs['rv_trailing_minus_with']
        p(f"  monthly RV trailing - with curvature: {x['mean']:+.3e}, t {x['t']:+.2f}")
    p('\n## Holm across B\'s secondary family (6 tests; A2 joins at the merge)')
    for k, v in R['holm_secondary'].items():
        p(f"  {k}: p {v['p']:.4f}, Holm p {v['holm_p']:.4f}, {'pass' if v['pass'] else 'fail'}")
    txt = '\n'.join(L)
    print(txt)
    (B.HERE / 'money_old.txt').write_text(txt + '\n')


if __name__ == '__main__':
    main()
