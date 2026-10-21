"""Volume clock: the combined tables for the .md, from the saved outputs (no new numbers).

    python3 -I make_tables.py > tables.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests import HERE, bps, clustered, load  # noqa: E402


def year_table():
    Z = load('old')
    print('| Entry year | Windows | Hit rate | Raw-spike hit rate | Premium (bps, log) | t |')
    print('|---|---|---|---|---|---|')
    for y in range(2006, 2026):
        m = Z['ok'] & (Z['entry_year'] == y)
        kh = m & np.isfinite(Z['hit'])
        kr = m & np.isfinite(Z['ret'])
        s = clustered(Z['ret'][kr], Z['e'][kr])
        print(f"| {y} | {kh.sum()} | {Z['hit'][kh].mean():.1%} | {Z['hit_raw'][kh].mean():.1%} | {bps(s['mean'])} | {s['t']:+.2f} |")
    print()


def money_table():
    j = json.loads((HERE / 'money_old.json').read_text())
    print('| Test year | Candidates | Names held (avg) | Picks, 0 bps | Uniform random median | p | Volatility-matched median | p |')
    print('|---|---|---|---|---|---|---|---|')
    for k, (u, v) in enumerate(zip(j['uniform']['years'], j['volatility']['years'])):
        cu, cv = u['costs'][0], v['costs'][0]
        print(f"| {u['first'][:4]}–{u['last'][2:4]} | {u['candidates']} | {j['book'][k]:.1f} | {cu['final_pct']:+.1f}% | "
              f"{cu['random_median_pct']:+.1f}% | {cu['p']:.2f} | {cv['random_median_pct']:+.1f}% | {cv['p']:.2f} |")
    print()
    for rm in ('uniform', 'volatility'):
        print(rm, j[rm]['summary'])
        for i, c in enumerate(j[rm]['change']):
            print(f"  {(0, 5, 20)[i]} bps: {c['change']:+.2f} (se {c['se']:.2f}, ahead {c['ahead']}/{c['years']})")
    print('skipped', j['uniform'].get('skipped'))
    print('mean book', np.mean(j['book']))


if __name__ == '__main__':
    year_table()
    money_table()
