"""Co-movement market timing on the old export (2005-04..2025-06): the rule, its baselines, the tests in order, the
absorption-ratio rule and the sensitivity with only 500-stock states in the median.

    python3 -I backtest.py      (after states.py old)
Writes results_old.txt, results_old.json and months_old.tsv here.
"""
import json
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_timing')
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import engine as G  # noqa: E402

iso = C.c2eval.iso


def month_table(tag):
    """Everything per calendar month for one export: states, SPY variance, which months are complete."""
    z = np.load(C.WORK / f'states_{tag}.npz')
    st = {k: z[k] for k in z.files}
    days = st['days']
    P = C.spy_close(tag, days)
    T = len(days)
    rebal = st['rebal']
    var = np.array([C.realized_var(P, int(t0)) if t0 + C.H < T else np.nan for t0 in rebal])
    complete = np.array([t0 + C.H <= T - 1 and np.isfinite(P[t0 - C.H:t0 + C.H + 1]).all() for t0 in rebal])
    return st, days, P, var, complete


def evaluate(name, k_sel, w_rule, w_ar, rel_all, var_all, c_fixed=None, boot=True, random=True):
    """All measures for one period. k_sel: positions (into the month arrays) of the period's months."""
    rel = rel_all[k_sel]
    wr, wa = w_rule[k_sel], w_ar[k_sel]
    out = {'n_months': int(len(k_sel))}
    out['rule'] = G.summarize(wr, rel)
    out['rule']['switches'] = G.switches(wr)
    out['bh'] = G.summarize(np.ones(len(k_sel)), rel)
    if c_fixed is None:
        wv, c = G.vol_weights(var_all[k_sel], wr.mean())
    else:
        c = c_fixed
        wv = np.minimum(1.0, c / var_all[k_sel])
    out['vol'] = G.summarize(wv, rel) | {'c': c}
    out['ar'] = G.summarize(wa, rel)
    out['ar']['switches'] = G.switches(wa)
    if random:
        rd = G.random_timing(wr, rel)
        out['random'] = {'annual': float(np.median(rd['annual'])), 'sharpe': float(np.median(rd['sharpe'])),
                         'mdd': float(np.median(rd['mdd'])), 'exposure': float(wr.mean()),
                         'p_annual': G.random_p(out['rule']['annual'], rd['annual']),
                         'p_sharpe': G.random_p(out['rule']['sharpe'], rd['sharpe'])}
        ra = G.random_timing(wa, rel, seed=C.SEED + 7)
        out['ar_random'] = {'annual': float(np.median(ra['annual'])), 'sharpe': float(np.median(ra['sharpe'])),
                            'p_annual': G.random_p(out['ar']['annual'], ra['annual']),
                            'p_sharpe': G.random_p(out['ar']['sharpe'], ra['sharpe'])}
    if boot:
        drr = G.daily_returns(wr, rel)
        out['boot_vs_vol'] = G.block_bootstrap_sharpe(drr, G.daily_returns(wv, rel))
        out['boot_vs_bh'] = G.block_bootstrap_sharpe(drr, G.daily_returns(np.ones(len(k_sel)), rel))
        out['ar_boot_vs_bh'] = G.block_bootstrap_sharpe(G.daily_returns(wa, rel), G.daily_returns(np.ones(len(k_sel)), rel))
    # diagnostics: SPY's next-month return after 100% and 50% months
    spy_m = rel[:, -1] - 1.0
    d, t = G.welch(spy_m[wr == 1.0], spy_m[wr == 0.5])
    out['diag'] = {'spy_after_high': float(spy_m[wr == 1.0].mean()) if (wr == 1.0).any() else float('nan'),
                   'spy_after_low': float(spy_m[wr == 0.5].mean()) if (wr == 0.5).any() else float('nan'),
                   'diff': d, 't': t, 'n_high': int((wr == 1.0).sum()), 'n_low': int((wr == 0.5).sum())}
    return out


def main():
    st, days, P, var, complete = month_table('old')
    rebal = st['rebal']
    K = len(rebal)
    x, ab = st['avg_corr'], st['absorption']
    valid_all = np.isfinite(x)
    valid_500 = valid_all & (st['n_sel'] == C.N_UNIV)
    ks = np.arange(K)
    w_rule, med_rule = G.signal_weights(x, ks, valid_all, +1)
    w_ar, med_ar = G.signal_weights(ab, ks, valid_all, -1)
    w_rule_b, med_rule_b = G.signal_weights(x, ks, valid_500, +1)
    w_ar_b, _ = G.signal_weights(ab, ks, valid_500, -1)
    ok = complete & np.isfinite(var) & np.isfinite(w_rule) & np.isfinite(w_ar)
    months = np.nonzero(ok)[0]
    print('months:', len(months), iso(days[rebal[months[0]]]), '..', iso(days[rebal[months[-1]]]),
          'last hold ends', iso(days[rebal[months[-1]] + C.H]), flush=True)
    rel_all = np.full((K, C.H + 1), np.nan)
    rel_all[months] = G.rel_paths(P, rebal[months])
    date = np.array([iso(days[t]) for t in rebal])

    def sel(lo, hi, base=months):
        return np.array([k for k in base if lo <= date[k] <= hi])

    full = sel('2005-01-01', '2025-06-27')
    res = {'first_month': date[full[0]], 'last_month': date[full[-1]], 'last_hold_end': iso(days[rebal[full[-1]] + C.H])}
    res['full'] = evaluate('full', full, w_rule, w_ar, rel_all, var)
    c_full = res['full']['vol']['c']
    res['test1'] = evaluate('test1', sel('2005-01-01', '2019-12-31'), w_rule, w_ar, rel_all, var)
    res['test3'] = evaluate('test3', sel('2020-01-01', '2025-06-27'), w_rule, w_ar, rel_all, var)
    res['blocks'] = {}
    for lo, hi, nm in (('2005-01-01', '2009-12-31', '2005-04..2009'), ('2010-01-01', '2014-12-31', '2010-2014'),
                       ('2015-01-01', '2019-12-31', '2015-2019'), ('2020-01-01', '2025-06-27', '2020-2025-06')):
        res['blocks'][nm] = evaluate(nm, sel(lo, hi), w_rule, w_ar, rel_all, var, c_fixed=c_full, boot=False)
    # sensitivity: only 500-stock states in the median (decisions once 24 of them exist)
    okb = complete & np.isfinite(var) & np.isfinite(w_rule_b) & np.isfinite(w_ar_b)
    mb = np.nonzero(okb)[0]
    rel_all[mb] = G.rel_paths(P, rebal[mb])
    res['sens500'] = {}
    for lo, hi, nm in (('2005-01-01', '2025-06-27', 'full'), ('2005-01-01', '2019-12-31', 'test1'),
                       ('2020-01-01', '2025-06-27', 'test3')):
        res['sens500'][nm] = evaluate(nm, sel(lo, hi, mb), w_rule_b, w_ar_b, rel_all, var)
        res['sens500'][nm]['first_month'] = date[sel(lo, hi, mb)[0]]
    # Pollet-Wilson style diagnostic over the full span: the state against SPY's next 21-day log return
    nxt = np.log(rel_all[full, -1])
    res['diag_full'] = {'corr_level': G.corr_t(x[full], nxt), 'corr_vs_median': G.corr_t(x[full] - med_rule[full], nxt),
                        'ar_corr_level': G.corr_t(ab[full], nxt)}
    # tests in order
    t1, t3 = res['test1'], res['test3']
    pass1 = (t1['random']['p_annual'] < 0.05) and (t1['rule']['sharpe'] >= t1['bh']['sharpe'])
    pass2 = (t1['rule']['sharpe'] > t1['vol']['sharpe']) and (t1['boot_vs_vol']['p'] < 0.10)
    pass3 = t3['rule']['annual'] > t3['random']['annual']
    res['tests'] = {'1': bool(pass1), '2': bool(pass2), '3': bool(pass3)}
    res['number'] = round(100 * (res['full']['rule']['annual'] - res['full']['random']['annual']), 1)
    yrs = len(full) / 12
    res['cost_5bps_per_year'] = res['full']['rule']['switches'] * 0.5 * 5e-4 / yrs
    res['c_full'] = c_full
    with open(C.HERE / 'results_old.json', 'w') as f:
        json.dump(res, f, indent=1)
    # per-month table
    with open(C.HERE / 'months_old.tsv', 'w') as f:
        f.write('date\tavg_corr\tmedian\tw_rule\tabsorption\tmedian_ar\tw_ar\tspy_var21\tw_vol_fullc\tspy_next21\tn_universe\n')
        for k in full:
            f.write(f"{date[k]}\t{x[k]:.4f}\t{med_rule[k]:.4f}\t{w_rule[k]:.1f}\t{ab[k]:.4f}\t{med_ar[k]:.4f}\t{w_ar[k]:.1f}\t"
                    f"{var[k]:.3e}\t{min(1.0, c_full / var[k]):.3f}\t{rel_all[k, -1] - 1:+.4f}\t{st['n_sel'][k]}\n")
    report(res)


def pct(v, d=1):
    return f'{100 * v:+.{d}f}%' if v == v else 'n/a'


def report(res):
    L = []
    p = L.append
    p(f"Months: {res['full']['n_months']} ({res['first_month']} .. {res['last_month']}, last hold ends {res['last_hold_end']})")
    for nm, lab in (('full', '2005-04..2025-06'), ('test1', '2005-04..2019-12 (tests 1-2)'), ('test3', '2020-01..2025-06 (test 3)')):
        r = res[nm]
        p(f'\n## {lab}: {r["n_months"]} months')
        p('| | Annual return | Sharpe | Max drawdown | Avg exposure |')
        p('|---|---|---|---|---|')
        p(f"| Rule (100% when avg corr above median) | {pct(r['rule']['annual'])} | {r['rule']['sharpe']:.2f} | {pct(-r['rule']['mdd'])} | {r['rule']['exposure']:.0%} |")
        p(f"| Buy and hold SPY | {pct(r['bh']['annual'])} | {r['bh']['sharpe']:.2f} | {pct(-r['bh']['mdd'])} | 100% |")
        p(f"| Random timing, median of 10,000 | {pct(r['random']['annual'])} | {r['random']['sharpe']:.2f} | {pct(-r['random']['mdd'])} | {r['random']['exposure']:.0%} |")
        p(f"| Volatility-managed SPY (c = {r['vol']['c']:.2e}) | {pct(r['vol']['annual'])} | {r['vol']['sharpe']:.2f} | {pct(-r['vol']['mdd'])} | {r['vol']['exposure']:.0%} |")
        p(f"| Absorption-ratio rule (reported only) | {pct(r['ar']['annual'])} | {r['ar']['sharpe']:.2f} | {pct(-r['ar']['mdd'])} | {r['ar']['exposure']:.0%} |")
        p(f"rule - median random: {100 * (r['rule']['annual'] - r['random']['annual']):+.2f} pts/yr; random p (annual) {r['random']['p_annual']:.4f}; "
          f"random p (Sharpe) {r['random']['p_sharpe']:.4f}")
        p(f"Sharpe rule - vol: {r['boot_vs_vol']['diff']:+.3f}, bootstrap p {r['boot_vs_vol']['p']:.4f} (5-95%: {r['boot_vs_vol']['lo5']:+.3f}..{r['boot_vs_vol']['hi95']:+.3f})")
        p(f"Sharpe rule - buy&hold: {r['boot_vs_bh']['diff']:+.3f}, bootstrap p {r['boot_vs_bh']['p']:.4f}")
        p(f"AR rule - its median random: {100 * (r['ar']['annual'] - r['ar_random']['annual']):+.2f} pts/yr, p {r['ar_random']['p_annual']:.4f}; "
          f"AR Sharpe - buy&hold {r['ar_boot_vs_bh']['diff']:+.3f}, bootstrap p {r['ar_boot_vs_bh']['p']:.4f}")
        dg = r['diag']
        p(f"SPY next-month return after 100% months {pct(dg['spy_after_high'], 2)} (n {dg['n_high']}), after 50% months "
          f"{pct(dg['spy_after_low'], 2)} (n {dg['n_low']}), diff t {dg['t']:+.2f}; rule switches {r['rule']['switches']}")
    p('\n## By period (random timing reordered within each; vol-managed uses the whole span\'s c)')
    p('| Period | Months | Rule | Buy and hold | Median random | Rule − random | p | Vol-managed | Rule Sharpe | B&H Sharpe | Vol Sharpe | Rule exposure |')
    p('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for nm, r in res['blocks'].items():
        p(f"| {nm} | {r['n_months']} | {pct(r['rule']['annual'])} | {pct(r['bh']['annual'])} | {pct(r['random']['annual'])} | "
          f"{100 * (r['rule']['annual'] - r['random']['annual']):+.1f} | {r['random']['p_annual']:.2f} | {pct(r['vol']['annual'])} | "
          f"{r['rule']['sharpe']:.2f} | {r['bh']['sharpe']:.2f} | {r['vol']['sharpe']:.2f} | {r['rule']['exposure']:.0%} |")
    p('\n## Absorption-ratio rule by period')
    p('| Period | AR rule | Median random | AR − random | p | AR Sharpe | Exposure |')
    p('|---|---|---|---|---|---|---|')
    for nm, r in [('2005-04..2025-06', res['full']), ('2005-04..2019-12', res['test1']), ('2020-01..2025-06', res['test3'])] + list(res['blocks'].items())[:3]:
        p(f"| {nm} | {pct(r['ar']['annual'])} | {pct(r['ar_random']['annual'])} | {100 * (r['ar']['annual'] - r['ar_random']['annual']):+.1f} | "
          f"{r['ar_random']['p_annual']:.2f} | {r['ar']['sharpe']:.2f} | {r['ar']['exposure']:.0%} |")
    p('\n## Sensitivity: only 500-stock states in the median')
    for nm, r in res['sens500'].items():
        p(f"{nm} from {r['first_month']} ({r['n_months']} months): rule {pct(r['rule']['annual'])} vs median random {pct(r['random']['annual'])} "
          f"({100 * (r['rule']['annual'] - r['random']['annual']):+.2f}, p {r['random']['p_annual']:.3f}); Sharpe rule {r['rule']['sharpe']:.2f}, "
          f"B&H {r['bh']['sharpe']:.2f}, vol {r['vol']['sharpe']:.2f} (boot p vs vol {r['boot_vs_vol']['p']:.3f}); exposure {r['rule']['exposure']:.0%}; "
          f"AR {pct(r['ar']['annual'])} vs {pct(r['ar_random']['annual'])} (p {r['ar_random']['p_annual']:.3f})")
    d = res['diag_full']
    p(f"\nDiagnostic, full span: corr(avg corr, next 21-day SPY log return) {d['corr_level'][0]:+.3f} (t {d['corr_level'][1]:+.2f}); "
      f"vs median {d['corr_vs_median'][0]:+.3f} (t {d['corr_vs_median'][1]:+.2f}); absorption {d['ar_corr_level'][0]:+.3f} (t {d['ar_corr_level'][1]:+.2f})")
    p(f"Cost at 5 bps on the half that moves, per year: {100 * res['cost_5bps_per_year']:.3f} pts")
    p(f"\nTests: 1 {'pass' if res['tests']['1'] else 'FAIL'}; 2 {'pass' if res['tests']['2'] else 'FAIL'}; 3 {'pass' if res['tests']['3'] else 'FAIL'}")
    p(f"File-name number (rule - median random, 2005-04..2025-06): {res['number']:+.1f}")
    txt = '\n'.join(L)
    print(txt)
    (C.HERE / 'results_old.txt').write_text(txt + '\n')


if __name__ == '__main__':
    main()
