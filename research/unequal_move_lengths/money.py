"""Money test per combination: the 2,000 strongest real pairs from the 5-year period ending the day before each test
year; B's score on day d = sum of r x A's move over a days ending at d + b - offset (<= d). Graded by c2eval.judge
with rebalance = B's move length.

python3 -I money.py walk [nums]     walk-forward 2006-07..2024-25 on the old export
python3 -I money.py unseen [nums]   the 16 unseen months on the new export (run once, after the rules are final)
Writes money_<mode>.json here (small) and scores + judge JSON under the large-file folder.
"""
import json
import os
import sys
import time

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_k, '8')
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umlcore as U  # noqa: E402
import engine as E  # noqa: E402
from umlcore import c2eval  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def add_scores(scores, v, pairs, a, b, d0, d1):
    """Adds the pairs' contributions to scores[:, d0..d1] (inclusive). pairs [m, 5] = (z, A row, B row, offset, r)."""
    days = np.arange(d0, d1 + 1)
    arows = np.unique(pairs[:, 1].astype(int))
    lo = max(0, d0 - 200 - a)
    C = E.cums(v[arows, lo:d1 + 1])
    pos = {r: i for i, r in enumerate(arows)}
    for p in pairs:
        ia, ib, k, r = pos[int(p[1])], int(p[2]), int(p[3]), float(p[4])
        e = days + b - k - lo  # local row where A's move ends (<= d)
        mv = (C[ia, e + 1] - C[ia, e + 1 - a]) / np.sqrt(a)
        s = scores[ib, d0:d1 + 1]
        s[np.isnan(s)] = 0.0
        scores[ib, d0:d1 + 1] = s + r * mv


def stouffer_p(js, cost_index=1):
    from math import erfc, sqrt
    from statistics import NormalDist
    nd = NormalDist()
    zs = []
    for y in js['years']:
        p = min(max(y['costs'][cost_index]['p'], 0.0005), 0.9995)
        zs.append(nd.inv_cdf(1 - p))
    z = sum(zs) / sqrt(len(zs))
    return z, 0.5 * erfc(z / sqrt(2))


def summarize(js, walk):
    out = {}
    if not walk:
        # A single holdout keeps its costs under report.strategy (c2eval.change_over_random reads walk-forward JSON).
        for c in js['report']['strategy']:
            bps = int(c['cost_bps'])
            out[f'change_{bps}'] = c['final_pct'] - c['random_median_pct']
            out[f'p_{bps}'] = c['p']
            out[f'final_pct_{bps}'] = c['final_pct']
            out[f'random_median_pct_{bps}'] = c['random_median_pct']
    for ci_, bps in (((0, 0), (1, 5), (2, 20)) if walk else ()):
        cor = c2eval.change_over_random(js, ci_)
        out[f'change_{bps}'] = cor['change']
        out[f'se_{bps}'] = cor['se']
        out[f'ahead_{bps}'] = cor['ahead']
        out[f'per_year_{bps}'] = cor['per_year']
    if walk:
        out['years'] = len(js['years'])
        out['starts'] = [y['start'] for y in js['years']]
        z, p = stouffer_p(js, 1)
        out['stouffer_z_5'] = z
        out['stouffer_p_5'] = p
        late = [i for i, s in enumerate(out['starts']) if s >= '2020-06-01']
        out['ahead_5_from_2020'] = int(sum(1 for i in late if out['per_year_5'][i] > 0))
        out['years_from_2020'] = len(late)
        out['beat_p01'] = js['summary']['beat_random_p_lt_0_1']
        out['ic_ret_z'] = js['summary']['ic_ret_z']
        out['ls_bps'] = js['summary']['ls_bps_per_day_mean']
        out['ls_z'] = js['summary']['ls_z']
        out['ls_turnover'] = js['summary']['ls_turnover_mean']
        out['summary_line'] = c2eval.summary_line(js)
        out['skipped'] = js.get('skipped', [])
    else:
        rep = js['report']
        out['strategy'] = rep['strategy']
        out['benchmarks'] = rep['benchmarks']
        out['raw'] = {k: rep.get(k) for k in ('hit_rate', 'ic_ret_mean', 'ic_ret_t', 'ls_bps_per_day', 'ls_t', 'candidates', 'pairs', 'days')}
    return out


def walk(nums):
    data = E.Data()
    days = data.days
    T = data.T
    starts = [c2eval.day_index(days, f'{y}-06-28') for y in range(2006, 2025)] + [T + 1]
    tops = {y: np.load(U.OUT / 'real' / f'Y{y}.npz')['top'] for y in range(2006, 2025)}
    path_json = os.path.join(HERE, 'money_walk.json')
    results = json.load(open(path_json)) if os.path.exists(path_json) else {}
    for ci, (num, a, b) in enumerate(U.COMBOS):
        if nums and num not in nums:
            continue
        t0 = time.time()
        scores = np.full((data.N, T), np.nan, dtype=np.float32)
        npairs = []
        for yi, y in enumerate(range(2006, 2025)):
            pr = tops[y][ci, 0]
            pr = pr[np.isfinite(pr[:, 0])]
            npairs.append(len(pr))
            d0 = starts[yi] - 1
            d1 = min(starts[yi + 1] - 2, T - 1)
            add_scores(scores, data.v, pr, a, b, d0, d1)
        name = f'unequal_move_lengths/money_walk_{num:02d}_A{a}_B{b}'
        p = c2eval.save_scores(scores, name)
        del scores
        txt, js = c2eval.judge(p, rebalance=b)
        s = summarize(js, True)
        s.update({'num': num, 'a': a, 'b': b, 'pairs_per_year': npairs, 'judge_json': str(p).replace('.npy', f'__r{b}_d0_uniform.json')})
        results[str(num)] = s
        json.dump(results, open(path_json, 'w'), indent=1)
        open(os.path.join(HERE, 'logs', f'judge_walk_{num:02d}.txt'), 'w').write(txt)
        print(f'#{num} {a}->{b}: change {s["change_0"]:+.2f} (5 bps {s["change_5"]:+.2f}), Stouffer p(5 bps) {s["stouffer_p_5"]:.4f}, '
              f'ahead {s["ahead_5"]}/{s["years"]} (2020+: {s["ahead_5_from_2020"]}/{s["years_from_2020"]}) [{time.time() - t0:.0f}s]', flush=True)


def unseen(nums):
    data = E.Data(c2eval.EXPORT_NEW)
    S = c2eval.day_index(data.days, '2025-06-30')
    p0, p1 = S - E.PERIOD, S
    res = E.run_period(data, 99, 'unseen', p0, p1, 'real')
    np.savez(U.OUT / 'unseen_period.npz', **res)
    path_json = os.path.join(HERE, 'money_unseen.json')
    results = json.load(open(path_json)) if os.path.exists(path_json) else {}
    for ci, (num, a, b) in enumerate(U.COMBOS):
        if nums and num not in nums:
            continue
        pr = res['top'][ci, 0]
        pr = pr[np.isfinite(pr[:, 0])]
        scores = np.full((data.N, data.T), np.nan, dtype=np.float32)
        add_scores(scores, data.v, pr, a, b, S - 1, data.T - 1)
        p = c2eval.save_scores(scores, f'unequal_move_lengths/money_unseen_{num:02d}_A{a}_B{b}', export=c2eval.EXPORT_NEW)
        del scores
        txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=b)
        s = summarize(js, False)
        s.update({'num': num, 'a': a, 'b': b, 'pairs': len(pr)})
        results[str(num)] = s
        json.dump(results, open(path_json, 'w'), indent=1)
        open(os.path.join(HERE, 'logs', f'judge_unseen_{num:02d}.txt'), 'w').write(txt)
        print(f'#{num} {a}->{b}: unseen change {s["change_0"]:+.2f} (5 bps {s["change_5"]:+.2f}, p {s["p_5"]:.3f}); picks '
              f'{s["final_pct_0"]:+.1f}% vs random median {s["random_median_pct_0"]:+.1f}%', flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    nums = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else None
    {'walk': walk, 'unseen': unseen}[mode](nums)
