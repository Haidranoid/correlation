"""Co-movement market timing on the unseen months (new export and store, 2025-06-30..2026-10-19), run once at the end.

    python3 -I states.py new && python3 -I unseen.py
Writes results_new.txt, results_new.json and months_new.tsv here.
"""
import json
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_timing')
import numpy as np  # noqa: E402

import backtest as B  # noqa: E402
import common as C  # noqa: E402
import engine as G  # noqa: E402

iso = C.c2eval.iso


def main():
    old = json.loads((C.HERE / 'results_old.json').read_text())
    st, days, P, var, complete = B.month_table('new')
    rebal = st['rebal']
    K = len(rebal)
    x, ab = st['avg_corr'], st['absorption']
    valid = np.isfinite(x)
    ks = np.arange(K)
    w_rule, med = G.signal_weights(x, ks, valid, +1)
    w_ar, med_ar = G.signal_weights(ab, ks, valid, -1)
    t_last_old = C.c2eval.day_index(days, C.OLD_LAST)
    assert iso(days[t_last_old]) == C.OLD_LAST
    ok = complete & np.isfinite(var) & np.isfinite(w_rule) & np.isfinite(w_ar) & (rebal >= t_last_old)
    m = np.nonzero(ok)[0]
    rel = G.rel_paths(P, rebal[m])
    wr, wa = w_rule[m], w_ar[m]
    res = {'n_months': int(len(m)), 'first_month': iso(days[rebal[m[0]]]), 'last_month': iso(days[rebal[m[-1]]]),
           'last_hold_end': iso(days[rebal[m[-1]] + C.H])}
    res['rule'] = G.summarize(wr, rel)
    res['bh'] = G.summarize(np.ones(len(m)), rel)
    c = old['c_full']
    wv = np.minimum(1.0, c / var[m])
    res['vol'] = G.summarize(wv, rel) | {'c': c}
    res['ar'] = G.summarize(wa, rel)
    rd = G.random_timing(wr, rel)
    res['random'] = {'annual': float(np.median(rd['annual'])), 'sharpe': float(np.median(rd['sharpe'])),
                     'mdd': float(np.median(rd['mdd'])), 'p_annual': G.random_p(res['rule']['annual'], rd['annual'])}
    ra = G.random_timing(wa, rel, seed=C.SEED + 7)
    res['ar_random'] = {'annual': float(np.median(ra['annual'])), 'p_annual': G.random_p(res['ar']['annual'], ra['annual'])}
    res['number'] = round(100 * (res['rule']['annual'] - res['random']['annual']), 1)
    # how the new export's states compare with the old export's on the old span
    zo = np.load(C.WORK / 'states_old.npz')
    n_old = len(zo['rebal'])
    assert np.array_equal(zo['rebal'], rebal[:n_old])
    xo = zo['avg_corr']
    both = np.isfinite(xo) & np.isfinite(x[:n_old])
    wo, _ = G.signal_weights(xo, np.arange(n_old), np.isfinite(xo), +1)
    same = np.isfinite(wo) & np.isfinite(w_rule[:n_old])
    res['consistency'] = {'corr_states': float(np.corrcoef(xo[both], x[:n_old][both])[0, 1]),
                          'max_abs_diff': float(np.max(np.abs(xo[both] - x[:n_old][both]))),
                          'weights_differ': int(np.sum(wo[same] != w_rule[:n_old][same])), 'months_compared': int(same.sum())}
    with open(C.HERE / 'results_new.json', 'w') as f:
        json.dump(res, f, indent=1)
    with open(C.HERE / 'months_new.tsv', 'w') as f:
        f.write('date\tavg_corr\tmedian\tw_rule\tabsorption\tmedian_ar\tw_ar\tspy_var21\tw_vol\tspy_next21\tn_universe\n')
        for j, k in enumerate(m):
            f.write(f"{iso(days[rebal[k]])}\t{x[k]:.4f}\t{med[k]:.4f}\t{w_rule[k]:.1f}\t{ab[k]:.4f}\t{med_ar[k]:.4f}\t{w_ar[k]:.1f}\t"
                    f"{var[k]:.3e}\t{wv[j]:.3f}\t{rel[j, -1] - 1:+.4f}\t{st['n_sel'][k]}\n")
    pct = B.pct
    L = [f"Unseen months: {res['n_months']} ({res['first_month']} .. {res['last_month']}, last hold ends {res['last_hold_end']})",
         '| | Annual return | Sharpe | Max drawdown | Avg exposure |', '|---|---|---|---|---|']
    for lab, r in (('Rule', res['rule']), ('Buy and hold SPY', res['bh'])):
        L.append(f"| {lab} | {pct(r['annual'])} | {r['sharpe']:.2f} | {pct(-r['mdd'])} | {r['exposure']:.0%} |")
    L.append(f"| Random timing, median of 10,000 | {pct(res['random']['annual'])} | {res['random']['sharpe']:.2f} | {pct(-res['random']['mdd'])} | {res['rule']['exposure']:.0%} |")
    L.append(f"| Volatility-managed SPY (whole old span's c) | {pct(res['vol']['annual'])} | {res['vol']['sharpe']:.2f} | {pct(-res['vol']['mdd'])} | {res['vol']['exposure']:.0%} |")
    L.append(f"| Absorption-ratio rule | {pct(res['ar']['annual'])} | {res['ar']['sharpe']:.2f} | {pct(-res['ar']['mdd'])} | {res['ar']['exposure']:.0%} |")
    L.append(f"rule - median random {100 * (res['rule']['annual'] - res['random']['annual']):+.2f} pts/yr, p {res['random']['p_annual']:.3f}; "
             f"AR - its median random {100 * (res['ar']['annual'] - res['ar_random']['annual']):+.2f}, p {res['ar_random']['p_annual']:.3f}")
    L.append(f"weights: rule {wr.tolist()}; AR {wa.tolist()}")
    cs = res['consistency']
    L.append(f"new vs old export states over the old span: corr {cs['corr_states']:.4f}, max |diff| {cs['max_abs_diff']:.4f}, "
             f"rule weights differ in {cs['weights_differ']} of {cs['months_compared']} months")
    txt = '\n'.join(L)
    print(txt)
    (C.HERE / 'results_new.txt').write_text(txt + '\n')


if __name__ == '__main__':
    main()
