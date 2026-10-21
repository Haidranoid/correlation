"""Prints the markdown tables for the experiment file from the saved outputs."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umlcore as U  # noqa: E402
import engine as E  # noqa: E402
import cal_report as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {1: '1 day', 5: '1 week', 10: '2 weeks', 21: '1 month', 42: '2 months', 63: '1 quarter', 126: 'half a year'}

rolling = [f'Y{y}' for y in range(2006, 2026)]
tags = rolling + ['full']
d = C.load(tags)
print('| # | A → B | This null: noise A → stocks B | This null: stocks A → noise B | Old null (non-overlapping) | Overlapping, `v` flipped |')
print('|---|---|---|---|---|---|')
for ci, (num, a, b) in enumerate(U.COMBOS):
    cells = []
    for mi in range(4):
        r = C.ratio(C.per_series(d, tags, ci, mi, 1))
        ok = 0.90 <= r[2] <= 1.10 or abs(r[2] - 1) <= 2 * r[3]
        s = f'{r[2]:.2f} ± {r[3]:.2f}'
        cells.append(s if ok else f'**{s}**')
    bold = '**' if num == 7 else ''
    print(f'| {num} | {bold}{NAMES[a]} → {NAMES[b]}{bold} | ' + ' | '.join(cells) + ' |')
print()
rows = np.load(U.OUT / 'real_vs_chance_rows.npy', allow_pickle=True).item()
rep = np.load(U.OUT / 'replication.npz')
walk = json.load(open(os.path.join(HERE, 'money_walk.json')))
unseen = json.load(open(os.path.join(HERE, 'money_unseen.json')))
print('| # | A → B | Real/chance, rolling (z) | Noise ref | Same sign next period: real / null (z) | Change over random, 0 bps | 5 bps | p at 5 bps | Years ahead at 5 bps (2020+) | Unseen months, 0 bps | Fails |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for ci, (num, a, b) in enumerate(U.COMBOS):
    r = rows[num]
    same, tested = rep['same'][ci].sum(1), rep['tested'][ci].sum(1)
    rate = same / np.maximum(tested, 1)
    mu, sd = rate[1:].mean(), rate[1:].std(ddof=1)
    zr = (rate[0] - mu) / sd
    m, u = walk[str(num)], unseen[str(num)]
    fails = []
    if not r['excess']:
        fails.append('excess')
    if zr < 2:
        fails.append('replication')
    money = []
    if m['stouffer_p_5'] >= 0.0025:
        money.append('p')
    if not (3 * m['ahead_5'] >= 2 * m['years'] and m['ahead_5_from_2020'] >= 4):
        money.append('years')
    if not u['change_5'] > 0:
        money.append('unseen')
    if money:
        fails.append('money (' + ', '.join(money) + ')')
    bold = '**' if num == 7 else ''
    print(f'| {num} | {bold}{NAMES[a]} → {NAMES[b]}{bold} | {r["roll"][2]:.2f} ({r["roll"][3]:+.1f}) | {r["noise_ref"]:.2f} | '
          f'{100 * rate[0]:.1f}% / {100 * mu:.1f}% ({zr:+.1f}) | {m["change_0"]:+.1f} | {m["change_5"]:+.1f} | {m["stouffer_p_5"]:.3f} | '
          f'{m["ahead_5"]} of {m["years"]} ({m["ahead_5_from_2020"]} of {m["years_from_2020"]}) | {u["change_0"]:+.1f} | {"; ".join(fails)} |')
ch = np.array([walk[str(n)]['change_0'] for n, _, _ in U.COMBOS])
print(f'\nmean {ch.mean():+.3f}; median {np.median(ch):+.2f}; positive {int((ch > 0).sum())}; sd {ch.std(ddof=1):.2f}')
uc = np.array([unseen[str(n)]['change_0'] for n, _, _ in U.COMBOS])
print(f'unseen mean {uc.mean():+.2f}; ahead {int((uc > 0).sum())}')
