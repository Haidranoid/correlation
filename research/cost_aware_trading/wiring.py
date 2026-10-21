"""Step 1, the wiring check: reproduce stockai's grader account for the weekly LightGBM (top 10 positive scores,
rebalanced every 5 trading days, one $10,000 account per test year, compounded over the 14 years) and compare with
stockai/results/1w/judge/*.json. Then the same top-10 baseline on the 700-graded-week calendar as one continuous
account, under stockai's accounting and under the plan's (no re-equalizing)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402

COSTS = [0.0, 5.0, 20.0, 50.0]


def grader_year(E, spec, scores, costs=COSTS, picks=10):
    """holdout.rs: decision days h0-1..t-2; on k % 5 == 0 the top `picks` candidates with score > 0 (ties by
    candidate order); daily value *= 1 + sum_i w*ret_i - w*(bought+sold)*cost; w = 1/picks."""
    h0, t, cand = spec['h0'], spec['t'], spec['universe']
    days = np.arange(h0 - 1, t - 1)
    close = E['close']
    vals = {c: 10000.0 for c in costs}
    prev = []
    held = []
    w = 1.0 / picks
    n_trades = 0
    for k, d in enumerate(days):
        if k % C.H == 0:
            s = np.asarray(scores[cand, d], dtype=np.float64)
            ok = np.flatnonzero(s > 0)
            order = ok[np.lexsort((ok, -s[ok]))][:picks]
            held = list(cand[order])
        r = C.day_returns(close, d)
        gross = sum(w * r[j] for j in held)
        kept = len(set(held) & set(prev))
        turnover = w * ((len(held) - kept) + (len(prev) - kept))
        n_trades += (len(held) - kept) + (len(prev) - kept)
        for c in costs:
            vals[c] *= 1.0 + gross - turnover * c / 1e4
        prev = held
    return vals, n_trades


def main():
    E = C.load()
    specs = C.year_specs(E)
    res = {}
    for name, jf in [('lgbm_s4', 'jitter_lgbm_s4_uniform'), ('lgbm', 'lgbm_uniform')]:
        J = json.load(open(C.SA / 'results/1w/judge' / f'{jf}.json'))
        jy = {y['start'][:4]: y for y in J['years']}
        sc = np.load(C.SCORES[name], mmap_mode='r')
        comp = {c: 1.0 for c in COSTS}
        jcomp = {c: 1.0 for c in COSTS}
        rows = []
        for spec in specs:
            vals, ntr = grader_year(E, spec, sc)
            jyy = jy[str(spec['year'])]
            assert jyy['candidates'] == len(spec['universe']), (spec['year'], jyy['candidates'], len(spec['universe']))
            row = {'year': spec['year'], 'candidates': len(spec['universe']), 'trades': ntr}
            for ci, c in enumerate(COSTS):
                jf_ = jyy['costs'][ci]['final']
                row[f'ours_{c:g}'] = vals[c]
                row[f'judge_{c:g}'] = jf_
                comp[c] *= vals[c] / 1e4
                jcomp[c] *= jf_ / 1e4
            rows.append(row)
            C.log(name, spec['year'], ' '.join(f"{c:g}bp {vals[c]:.2f}/{row[f'judge_{c:g}']:.2f}" for c in COSTS))
        res[name] = {'years': rows, 'ours_14y': {f'{c:g}': 1e4 * comp[c] for c in COSTS},
                     'judge_14y': {f'{c:g}': 1e4 * jcomp[c] for c in COSTS}}
        C.log(name, '14 years ours', {c: round(1e4 * comp[c]) for c in COSTS}, 'judge',
              {c: round(1e4 * jcomp[c]) for c in COSTS})
    # Calendar facts
    cal = []
    for s in specs:
        cal.append({'year': s['year'], 'first_day': C.c2eval.iso(E['days'][s['h0']]),
                    'last_day': C.c2eval.iso(E['days'][s['t'] - 1]),
                    'first_decision': C.c2eval.iso(E['days'][s['h0'] - 1]),
                    'rebalance_days': len(s['reb']), 'graded': len(s['decisions']),
                    'last_graded': C.c2eval.iso(E['days'][s['decisions'][-1]]), 'universe': len(s['universe'])})
    res['calendar'] = cal
    res['graded_total'] = int(sum(len(s['decisions']) for s in specs))
    C.log('graded weeks', res['graded_total'])
    C.save_json(res, 'wiring_grader.json')


if __name__ == '__main__':
    main()
