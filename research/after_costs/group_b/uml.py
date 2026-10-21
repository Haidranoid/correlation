"""-1.9-unequal-move-lengths: the file-name number is the mean over the 20 combinations' judge change over random.
After-cost analogue: the mean over the 20 combinations of (judge account at 5 bps - SPY), per year then averaged."""
import sys, glob, json
import numpy as np
sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_b')
import ac
ps = sorted(glob.glob('/home/bhoult/.local/share/correlation2/experiments/unequal_move_lengths/money_walk_*__r*_d0_uniform.json'))
assert len(ps) == 20
res = [ac.judge_after_cost(p) for p in ps]
for p, r in zip(ps, res):
    print(f"{p.split('/')[-1]:45s} cor0 {r['cor0']:+6.2f} n={r['n']} strat5 {r['strat5']:+6.2f} spy {r['spy']:+6.2f} after {r['after_cost']:+6.2f}")
print('mean cor0', np.mean([r['cor0'] for r in res]))
print('mean strat5', np.mean([r['strat5'] for r in res]), 'mean spy', np.mean([r['spy'] for r in res]),
      'mean after', np.mean([r['after_cost'] for r in res]))
json.dump({p: {k: v for k, v in r.items()} for p, r in zip(ps, res)},
          open('/home/bhoult/.local/share/correlation2/experiments/after_costs/group_b_uml.json', 'w'), indent=1)
