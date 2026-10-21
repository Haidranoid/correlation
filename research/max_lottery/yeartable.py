"""Year-by-year table of the judge's walk-forward (uniform and volatility-matched) -> money_years_<tag>.txt."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, HERE  # noqa: E402

tag = sys.argv[1]
u = json.load(open(WORK / f'max_money_{tag}__r21_d0_uniform.json'))
v = json.load(open(WORK / f'max_money_{tag}__r21_d0_volatility.json'))
L = ['| Year | Picks | Median random, uniform | Median random, volatility-matched | SPY | p (uniform) | p (volatility) | Candidates |',
     '|---|---|---|---|---|---|---|---|']
if u.get('years'):
    for a, b in zip(u['years'], v['years']):
        ca, cb = a['costs'][0], b['costs'][0]
        L.append(f"| {a['start'][:4]}-{a['end'][2:4]} | {ca['final_pct']:+.1f}% | {ca['random_median_pct']:+.1f}% | "
                 f"{cb['random_median_pct']:+.1f}% | {a['spy_pct']:+.1f}% | {ca['p']:.2f} | {cb['p']:.2f} | {a['candidates']} |")
    import numpy as np
    pk = np.array([y['costs'][0]['final_pct'] for y in u['years']])
    L.append(f"mean picks {pk.mean():+.2f}%, mean median uniform {np.mean([y['costs'][0]['random_median_pct'] for y in u['years']]):+.2f}%, "
             f"mean median volatility {np.mean([y['costs'][0]['random_median_pct'] for y in v['years']]):+.2f}%")
else:
    for name, js in (('uniform', u), ('volatility', v)):
        r = js['report']
        L.append(f"{name}: {json.dumps({k: r[k] for k in r if k != 'strategy'})[:1500]}")
        L.append(f"{name} strategy: {r['strategy']}")
(HERE / f'money_years_{tag}.txt').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
