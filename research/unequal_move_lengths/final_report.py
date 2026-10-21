"""Combines calibration, real vs chance, replication and the money tests into one table per combination.

python3 -I final_report.py -> summary.txt here
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umlcore as U  # noqa: E402
import engine as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {1: '1 day', 5: '1 week', 10: '2 weeks', 21: '1 month', 42: '2 months', 63: '1 quarter', 126: 'half a year'}


def main():
    rows = np.load(U.OUT / 'real_vs_chance_rows.npy', allow_pickle=True).item()
    rep = np.load(U.OUT / 'replication.npz')
    walk = json.load(open(os.path.join(HERE, 'money_walk.json')))
    unseen_path = os.path.join(HERE, 'money_unseen.json')
    unseen = json.load(open(unseen_path)) if os.path.exists(unseen_path) else {}
    lines = []
    w = lines.append
    w(f'{"#":>3} {"A -> B":>24} {"real/chance":>11} {"repl real%":>10} {"null%":>6} {"z":>5} '
      f'{"chg 0bp":>8} {"chg 5bp":>8} {"Stouffer p 5bp":>14} {"ahead 5bp":>9} {"2020+":>5} {"unseen 0bp":>10} {"unseen 5bp":>10}  verdict')
    changes = []
    for ci, (num, a, b) in enumerate(U.COMBOS):
        r = rows[num]
        same, tested = rep['same'][ci].sum(1), rep['tested'][ci].sum(1)
        rate = same / np.maximum(tested, 1)
        mu, sd = rate[1:].mean(), rate[1:].std(ddof=1)
        zr = (rate[0] - mu) / sd if sd > 0 else float('nan')
        m = walk[str(num)]
        u = unseen.get(str(num))
        changes.append(m['change_0'])
        c1 = r['excess']
        c2 = zr >= 2
        c3a = m['stouffer_p_5'] < 0.0025
        c3b = 3 * m['ahead_5'] >= 2 * m['years'] and m['ahead_5_from_2020'] >= 4
        c3c = u is not None and u['change_5'] > 0
        failed = [n for n, ok in (('real vs chance', c1), ('replication', c2), ('money p', c3a), ('years ahead', c3b),
                                  ('unseen months', c3c)) if not ok]
        verdict = 'PASS' if not failed else 'fails: ' + ', '.join(failed)
        us0 = f'{u["change_0"]:+.1f}' if u else '-'
        us5 = f'{u["change_5"]:+.1f}' if u else '-'
        w(f'{num:>3} {NAMES[a] + " -> " + NAMES[b]:>24} {r["roll"][2]:>11.2f} {100 * rate[0]:>9.1f}% {100 * mu:>5.1f}% {zr:>5.1f} '
          f'{m["change_0"]:>+8.1f} {m["change_5"]:>+8.1f} {m["stouffer_p_5"]:>14.3f} {m["ahead_5"]:>5}/{m["years"]:<3} '
          f'{m["ahead_5_from_2020"]:>2}/{m["years_from_2020"]:<2} {us0:>10} {us5:>10}  {verdict}')
    ch = np.array(changes)
    w('')
    w(f'Mean change over random across the 20 combinations, 0 bps, walk-forward: {ch.mean():+.2f} points a year '
      f'(best {ch.max():+.1f} at #{U.COMBOS[int(ch.argmax())][0]}, worst {ch.min():+.1f} at #{U.COMBOS[int(ch.argmin())][0]}); '
      f'week -> month (#7): {walk["7"]["change_0"]:+.2f}')
    w(f'File name number: {ch.mean():+.1f}')
    out = '\n'.join(lines)
    open(os.path.join(HERE, 'summary.txt'), 'w').write(out + '\n')
    print(out)


if __name__ == '__main__':
    main()
