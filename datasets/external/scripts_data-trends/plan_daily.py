"""Adaptive daily request plan.

Grid: 261-day windows (daily resolution needs < 270 days), laid backwards from END with a 45-day
overlap; the earliest window is pinned to 2004-01-01. Consecutive grid windows are grouped into one
multirange request (joint scale, up to 5 items) only while grouping costs each member at most
LOSS x precision, judged from the weekly joint series (max over a window's weeks). Consecutive
requests share one 45-day overlap, used for chain stitching.

usage: plan_daily.py <prefix> <since YYYY-MM-DD> <until YYYY-MM-DD> term [term...]
writes queue/<prefix>_<slug>.json with tasks for windows whose start >= since and < until.
"""
import json
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terms import END, slug, items  # noqa: E402
from gtparse import parse_any  # noqa: E402

RAW = os.path.expanduser('~/.local/share/correlation2/external/raw/google_trends')
HERE = os.path.dirname(os.path.abspath(__file__))
LEN, OVL, LOSS, MAXN = 261, 45, 1.5, 5


def grid():
    e = date.fromisoformat(END)
    s0 = date(2004, 1, 1)
    ws = []
    while True:
        a = e - timedelta(days=LEN - 1)
        if a <= s0:
            a = s0
            ws.append((a, a + timedelta(days=LEN - 1)))
            break
        ws.append((a, e))
        e = a + timedelta(days=OVL - 1)
    return ws[::-1]  # chronological


def weekly_max(wk, a, b):
    vals = [v for d, v in wk.items() if d + timedelta(days=6) >= a and d <= b]
    return max(vals) if vals else 0.0


def plan(term, since, until):
    merged, _, _ = parse_any(os.path.join(RAW, 'weekly_joint', slug(term) + '.txt'))
    wk = {d: v[0] for d, v in merged.items()}
    g = [w for w in grid() if since <= w[0] < until]
    mx = [max(weekly_max(wk, a, b), 1.0) for a, b in g]
    # group from the most recent backwards (fetch order = most recent first)
    groups, cur = [], []
    for i in range(len(g) - 1, -1, -1):
        cand = cur + [i]
        gm = max(mx[j] for j in cand)
        if cur and (len(cand) > MAXN or any(gm / mx[j] > LOSS for j in cand)):
            groups.append(cur)
            cur = [i]
        else:
            cur = cand
    if cur:
        groups.append(cur)
    tasks = []
    for grp in groups:
        ws = [g[j] for j in sorted(grp)]
        rs = [(a.isoformat(), b.isoformat()) for a, b in ws]
        tasks.append({'label': f'daily {term} {rs[0][0]}..{rs[-1][1]} ({len(rs)})',
                      'out': f'daily_req/{slug(term)}/{rs[0][0]}_{rs[-1][1]}.txt',
                      'items': items(term, rs)})
    return tasks, len(g)


if __name__ == '__main__':
    prefix, since, until = sys.argv[1], date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3])
    for term in sys.argv[4:]:
        tasks, ng = plan(term, since, until)
        p = os.path.join(HERE, 'queue', f'{prefix}_{slug(term)}.json')
        json.dump(tasks, open(p, 'w'), indent=1)
        print(term, 'grid windows', ng, 'requests', len(tasks), [len(t['items']) for t in tasks])
