"""Queue the spec's overlapping single 5-year weekly windows for the given terms.
usage: plan_windows.py <prefix> term [term...]   -> queue/<prefix>_<nn>_<slug>.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terms import WEEKLY_WIN, slug, items
HERE = os.path.dirname(os.path.abspath(__file__))
prefix = sys.argv[1]
for i, kw in enumerate(sys.argv[2:]):
    tasks = [{'label': f'weekly_win {kw} {a}..{b}', 'out': f'weekly_win/{slug(kw)}/{a}_{b}.txt',
              'items': items(kw, [(a, b)])} for a, b in WEEKLY_WIN]
    p = os.path.join(HERE, 'queue', f'{prefix}_{i:02d}_{slug(kw)}.json')
    json.dump(tasks, open(p, 'w'), indent=1)
    print(p, len(tasks))
