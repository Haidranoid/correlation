"""Post-hoc diagnostic (added after seeing the judge's results; reported only): how concentrated the judge's 20 picks
are. Each day, the 20 highest positive scores (approximately the judge's picks: its liquid-universe filter is not
applied here) are traced to the thrust that scored them, labelled by the group's most common twin fund.

    python3 -I bt_diag.py old|new
"""
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_common import HERE, HOLD, big, c2eval, iso  # noqa: E402
from bt_tests import find_thrusts  # noqa: E402


def main(tag):
    with open(big(tag) / 'refits.pkl', 'rb') as fh:
        R = pickle.load(fh)
    days, T = R['days'], R['T']
    ev, _ = find_thrusts(R)
    start = '2006-06-28' if tag == 'old' else '2025-06-30'
    a = c2eval.day_index(days, start)
    active = defaultdict(list)
    for i, e in enumerate(ev):
        for d in range(e['d'], min(e['d'] + HOLD, T)):
            active[d].append(i)
    out = []
    by_year = defaultdict(lambda: [0, 0, 0])   # days, sum of groups among picks, days with all picks in one group
    label_days = Counter()
    for d in range(a, T - 1):
        best = {}
        for i in active.get(d, []):
            e = ev[i]
            for j, r in zip(e['members'], e['rho']):
                if np.isfinite(r) and (j not in best or 1 + r > best[j][0]):
                    best[j] = (1 + r, i)
        if not best:
            continue
        top = sorted(best.items(), key=lambda kv: -kv[1][0])[:20]
        evs = Counter(v[1] for _, v in top)
        labs = Counter(ev[i]['label'] for i in evs.elements())
        y = iso(days[d])[:4] if tag == 'old' else 'unseen'
        by_year[y][0] += 1
        by_year[y][1] += len(evs)
        by_year[y][2] += len(evs) == 1
        for lb, c in labs.items():
            label_days[lb] += c / 20
    out.append(f'# Pick concentration, {tag} export, decision days from {start} (post-hoc, reported only)')
    tot = [sum(v[i] for v in by_year.values()) for i in range(3)]
    out.append(f'days with picks {tot[0]}; mean thrusting groups among the 20 picks {tot[1] / tot[0]:.2f}; '
               f'days with all 20 from one group {tot[2]} ({tot[2] / tot[0]:.0%})')
    for y in sorted(by_year):
        v = by_year[y]
        out.append(f'  {y}: {v[0]} days, {v[1] / v[0]:.2f} groups, one group {v[2] / v[0]:.0%}')
    out.append('share of pick-days by group label (top 12): ' + ', '.join(
        f'{lb} {c / tot[0]:.0%}' for lb, c in label_days.most_common(12)))
    txt = '\n'.join(out)
    print(txt)
    (HERE / f'diag_{tag}.txt').write_text(txt + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
