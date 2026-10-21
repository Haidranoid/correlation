"""The money test (the judge) for the selected composite.

    systemd-run ... python3 -I money.py old      # walk-forward, scores NaN before 2015-06-28 (test years 2015-16..2024-25)
    systemd-run ... python3 -I money.py new      # the unseen months 2025-06-30..2026-10-19, once, at the end;
                                                 # also the composite's weekly IC there (test 3)

Scores on every trading day from SCORE_START: the composite of the 20 formulas fixed in selection_old.json, ranked
within that day's universe; NaN elsewhere. Graded with rebalance=5, uniform and volatility-matched random accounts.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/formula_mining')
from common import (BIG, FEATURES, HERE, SCORE_START, SRC, c2eval, composite, load_weekly, log, open_panel,  # noqa: E402
                    rank_features, spearman_rows, tstat, week_slices)


def build_scores(tag, sel, sign):
    exp, _ = SRC[tag]
    E = c2eval.load_export(exp, arrays=())
    days = E['days']
    N, T = len(E['ids']), len(days)
    U = np.load(BIG / f'universe_{tag}.npy', mmap_mode='r')
    panels = [open_panel(tag, f) for f in FEATURES]
    s = np.full((N, T), np.nan, np.float32)
    d0 = c2eval.day_index(days, SCORE_START[tag])
    for d in range(d0, T):
        idx = np.nonzero(U[d])[0]
        B = rank_features(np.stack([np.asarray(p[d, idx], np.float64) for p in panels]))
        s[idx, d] = composite(B, sel, sign)
    n = np.isfinite(s[:, d0:]).sum(0)
    log(tag, 'scored days', T - d0, 'from', c2eval.iso(days[d0]), 'stocks scored per day: median', int(np.median(n)),
        'min', int(n.min()))
    return c2eval.save_scores(s, f'formula_mining/composite_{tag}', export=exp), exp


def headline(js):
    s = js['summary']
    cor = c2eval.change_over_random(js)
    return {'change': cor['change'], 'se': cor['se'], 'ahead': cor['ahead'], 'years': cor['years'],
            'per_year': cor['per_year'],
            'change_5bps': c2eval.change_over_random(js, 1)['change'], 'change_20bps': c2eval.change_over_random(js, 2)['change'],
            'beat_random_p_lt_0_1': {int(b['cost_bps']): b['years'] for b in s['beat_random_p_lt_0_1']},
            'ic_ret_z': s['ic_ret_z'], 'ls_bps_per_day': s['ls_bps_per_day_mean'], 'ls_z': s['ls_z'],
            'ls_turnover': s['ls_turnover_mean'], 'ls_cost_per_bps': s['ls_cost_bps_per_day_per_bps'],
            'year_starts': [y['start'] for y in js['years']],
            'final_pct_0bps': [y['costs'][0]['final_pct'] for y in js['years']],
            'random_median_pct_0bps': [y['costs'][0]['random_median_pct'] for y in js['years']],
            'p_0bps': [y['costs'][0]['p'] for y in js['years']], 'summary_line': c2eval.summary_line(js)}


def single(js):
    rep = js['report']
    out = {'strategy': rep['strategy'], 'change': c2eval.change_over_random(js)['change']}
    for k in ('benchmarks', 'ic_ret_mean', 'ic_ret_t', 'ls_bps_per_day', 'ls_t', 'ls_turnover', 'hit_rate', 'candidates',
              'days', 'first_day', 'last_day'):
        if k in rep:
            out[k] = rep[k]
    return out


def unseen_ic(sel, sign):
    """Test 3: the composite's weekly IC on the new export's decision days from 2025-06-30 (and 5-day reversal's)."""
    W = load_weekly('new')
    sl = week_slices(W)
    r5 = FEATURES.index('ret5')
    comp, rev = [], []
    for k in range(len(sl)):
        B = W['B'][:, sl[k]].astype(np.float64)
        y = W['y'][sl[k]].astype(np.float64)
        comp.append(spearman_rows(composite(B, sel, sign)[None, :], y)[0])
        rev.append(spearman_rows(-B[r5][None, :], y)[0])
    comp, rev = np.array(comp), np.array(rev)
    E = c2eval.load_export(c2eval.EXPORT_NEW, arrays=())
    return {'composite': dict(zip(('t', 'mean', 'weeks'), tstat(comp))),
            'reversal5': dict(zip(('t', 'mean', 'weeks'), tstat(rev))),
            'weeks_from': c2eval.iso(E['days'][W['dates'][0]]), 'weeks_to': c2eval.iso(E['days'][W['dates'][-1]]),
            'composite_weekly': comp.tolist(), 'reversal_weekly': rev.tolist()}


def main(tag):
    S = json.load(open(HERE / 'selection_old.json'))
    sel, sign = np.array(S['sel_idx']), np.array(S['sign'], np.float64)
    path, exp = build_scores(tag, sel, sign)
    out = {}
    for rm in ('uniform', 'volatility'):
        if tag == 'old':
            txt, js = c2eval.judge(path, rebalance=5, random_match=rm)
            assert all(y['start'] >= '2015-06-28' for y in js['years']), 'a year before 2015-06-28 was graded'
            out[rm] = headline(js)
            log(rm, out[rm]['summary_line'])
        else:
            txt, js = c2eval.judge(path, export=exp, walk_forward=False, start='2025-06-30', rebalance=5, random_match=rm)
            out[rm] = single(js)
            log(rm, 'change over random', round(out[rm]['change'], 2))
        (HERE / f'judge_{tag}_{rm}.txt').write_text(txt)
    if tag == 'new':
        out['unseen_ic'] = unseen_ic(sel, sign)
        log('unseen weekly IC', out['unseen_ic']['composite'], 'reversal', out['unseen_ic']['reversal5'])
    json.dump(out, open(HERE / f'money_{tag}.json', 'w'), indent=1)
    log('done')


if __name__ == '__main__':
    main(sys.argv[1])
