"""The top-10 baseline (K = 10) from stockai's grader accounting to the plan's, one change at a time:
  (i)   the grader: one account per test year on every rebalance day, weights reset to 1/10 every day (wiring.json);
  (ii)  the same daily 1/10 weights, on the graded-week calendar, one continuous account;
  (iii) weights reset to 1/10 at each decision only (buy and hold within the week), costs as the grader charges them;
  (iv)  the plan's account: no re-equalizing (engine.simulate).

    python3 -I -B bridge.py sel|test       (test only after the selection is written into the .md)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402
import engine as G  # noqa: E402

COSTS = (0.0, 5.0, 20.0)


def main():
    which = sys.argv[1]
    E = C.load()
    specs = C.year_specs(E)
    years = C.SEL_YEARS if which == 'sel' else C.TEST_YEARS
    dec, end = G.weekly_calendar(specs, years)
    R = C.Returns(E, int(dec[0]), end, div_fix=False)
    spy_r, spy_adj = C.spy_returns(E)
    scores = np.load(C.SCORES['lgbm_s4'])
    panel = G.Panel(E, specs, scores, dec, end)
    P, Pspy = G.period_products(R, panel, spy_r)
    res = G.simulate(panel, G.make_rule('buffer', {'K': 10}), P, Pspy, costs=COSTS)
    plan = res['V'][:, -1]
    slots = res['slots']
    ends = np.r_[dec[1:], end]
    daily = np.full(len(COSTS), 1e4); weekly = np.full(len(COSTS), 1e4)
    prev = set()
    cs = np.array(COSTS) / 1e4
    for k, (a, e) in enumerate(zip(dec, ends)):
        held = [int(j) for j in slots[k] if j >= 0]
        n_b = len(set(held) - prev); n_s = len(prev - set(held))
        turn = 0.1 * (n_b + n_s)
        daily *= 1 - turn * cs
        weekly *= 1 - turn * cs
        for d in range(int(a), int(e)):
            daily *= 1 + 0.1 * R(d)[held].sum()
        weekly *= 0.1 * P[k][held].sum() + 0.1 * (10 - len(held))
        prev = set(held)
    W = json.load(open(C.OUT / 'wiring_grader.json'))
    grader = {}
    for c in COSTS:
        v = 1e4
        for row in W['lgbm_s4']['years']:
            if row['year'] in years:
                v *= row[f'judge_{c:g}'] / 1e4
        grader[f'{c:g}'] = v
    out = {'period': which, 'grader_yearly_accounts': grader,
           'daily_reweighted_graded_calendar': dict(zip([f'{c:g}' for c in COSTS], daily.tolist())),
           'weekly_reweighted': dict(zip([f'{c:g}' for c in COSTS], weekly.tolist())),
           'plan_no_reweighting': dict(zip([f'{c:g}' for c in COSTS], plan.tolist()))}
    for k, v in out.items():
        C.log(k, v)
    C.save_json(out, f'bridge_{which}.json')


if __name__ == '__main__':
    main()
