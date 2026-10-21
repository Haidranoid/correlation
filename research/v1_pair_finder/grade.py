"""Grade v1's pair-finder scores with the shared judge (walk-forward on the old export), and the universe control
(random scores on the same yearly 500-stock universes). Writes money_results.txt and money_results.json.

    python3 -I research/v1_pair_finder/grade.py
    python3 -I research/v1_pair_finder/grade.py --unseen      # the 16 unseen months, once
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np

import v1lib  # noqa: F401  (thread limits)
import c2eval

OUT = c2eval.WORK / 'v1_pair_finder'
REB = {7: 5, 14: 10, 21: 15, 30: 21, 40: 28, 60: 42, 90: 63, 180: 126}
CONFIGS = [('HEADLINE w=30, rebalance 21', 'aligned_w30', 21)] + \
          [('w=30, rebalance 30', 'aligned_w30', 30)] + \
          [(f'w={w}, rebalance {r}', f'aligned_w{w}', r) for w, r in REB.items() if w != 30] + \
          [('all 8 windows summed, rebalance 21', 'aligned_wall', 21),
           ('literal recipe w=30, rebalance 21', 'literal_w30', 21),
           ('null-picked relationships w=30, rebalance 21', 'aligned_null101_w30', 21)]
CONTROL_SEEDS = (1, 2, 3, 4, 5)


def stats(js):
    if 'report' in js and 'years' not in js:            # a single holdout: costs live in report.strategy
        rep = js['report']
        js = {**rep, 'costs': rep['strategy'],
              'spy_pct': next((b['return_pct'] for b in rep['benchmarks'] if b.get('ticker') == 'SPY'), None),
              'hold_all_pct': next((b['return_pct'] for b in rep['benchmarks'] if b.get('ticker') is None), None)}
    cor = c2eval.change_over_random(js)
    years = js.get('years') or [js]
    out = {'change': cor['change'], 'se': cor['se'], 'ahead': cor['ahead'], 'years': cor['years'], 'per_year': cor['per_year']}
    for ci, c in enumerate(years[0]['costs']):
        cc = c2eval.change_over_random(js, ci)
        out[f'change_{int(c["cost_bps"])}bps'] = cc['change']
        out[f'picks_mean_pct_{int(c["cost_bps"])}bps'] = float(np.mean([y['costs'][ci]['final_pct'] for y in years]))
        out[f'random_mean_pct_{int(c["cost_bps"])}bps'] = float(np.mean([y['costs'][ci]['random_median_pct'] for y in years]))
        out[f'p_{int(c["cost_bps"])}bps'] = [round(y['costs'][ci]['p'], 3) for y in years]
    if 'summary' in js:
        s = js['summary']
        out.update({k: s[k] for k in ('ic_ret_z', 'ls_bps_per_day_mean', 'ls_z', 'ls_turnover_mean', 'ls_cost_bps_per_day_per_bps') if k in s})
        out['beat_p_lt_0_1'] = s['beat_random_p_lt_0_1']
        out['line'] = c2eval.summary_line(js)
    else:
        out['line'] = (f"change over random {cor['change']:+.1f} pts; picks {years[0]['costs'][0]['final_pct']:.1f}% vs median random "
                       f"{years[0]['costs'][0]['random_median_pct']:.1f}%; IC actual t {js.get('ic_ret_t', float('nan')):+.2f}; "
                       f"long-short {js.get('ls_bps_per_day', float('nan')):.2f} bps/day t {js.get('ls_t', float('nan')):+.2f}; "
                       f"p (share of random at least as good) {years[0]['costs'][0]['p']:.3f}; SPY {js.get('spy_pct') or float('nan'):.1f}%, "
                       f"all candidates {js.get('hold_all_pct') or float('nan'):.1f}%")
        out.update({k: js.get(k) for k in ('ic_ret_t', 'ls_bps_per_day', 'ls_t', 'spy_pct', 'hold_all_pct', 'days', 'candidates')})
    return out


def control_scores(info_path, seed, export):
    info = json.loads(Path(info_path).read_text())
    ex = c2eval.load_export(export, arrays=('close',))
    days = ex['days']
    N, T = ex['close'].shape
    sc = np.full((N, T), np.nan, np.float32)
    rng = np.random.default_rng(seed)
    for name, p in info['periods'].items():
        u = np.array(p['universe'])
        a, b = p['decisions']
        alive = np.isfinite(np.asarray(ex['close'][u][:, a:b]))
        sc[np.ix_(u, np.arange(a, b))] = np.where(alive, rng.random((len(u), b - a)), np.nan).astype(np.float32)
    return c2eval.save_scores(sc, f'v1_pair_finder/control_universe_s{seed}', export=export)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unseen', action='store_true')
    a = ap.parse_args()
    res, lines = {}, []
    if a.unseen:
        export, prefix, kw = c2eval.EXPORT_NEW, 'unseen_', dict(export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30')
        info_path = OUT / 'unseen_aligned_info.json'
    else:
        export, prefix, kw = c2eval.EXPORT_OLD, '', {}
        info_path = OUT / 'aligned_info.json'
    for label, name, reb in CONFIGS:
        p = OUT / f'{prefix}{name}.npy'
        if not p.exists():
            continue
        txt, js = c2eval.judge(p, rebalance=reb, **kw)
        st = stats(js)
        res[label] = {'file': str(p), 'rebalance': reb, **st}
        lines.append(f'{label}: {st["line"]}')
        lines.append(f'    0 bps: picks {st["picks_mean_pct_0bps"]:.1f}%/yr vs median random {st["random_mean_pct_0bps"]:.1f}%; '
                     f'change at 5 bps {st.get("change_5bps", float("nan")):+.1f}, 20 bps {st.get("change_20bps", float("nan")):+.1f}; per year {st["per_year"]}')
        (HERE / f'judge_{prefix}{name}_r{reb}.txt').write_text(txt)
        print(lines[-2], flush=True)
    if info_path.exists():
        ch = []
        for sd in CONTROL_SEEDS:
            p = control_scores(info_path, sd, export)
            txt, js = c2eval.judge(p, rebalance=21, **kw)
            st = stats(js)
            ch.append(st['change'])
            res[f'universe control seed {sd}, rebalance 21'] = {'file': str(p), 'rebalance': 21, **st}
            lines.append(f'universe control (random scores on the same 500), seed {sd}: {st["line"]}')
            print(lines[-1], flush=True)
        res['universe control mean change'] = float(np.mean(ch))
        lines.append(f'universe control: mean change over random {np.mean(ch):+.1f} pts (seeds {", ".join(f"{c:+.1f}" for c in ch)})')
        print(lines[-1])
    (HERE / f'{prefix}money_results.txt').write_text('\n'.join(lines) + '\n')
    (HERE / f'{prefix}money_results.json').write_text(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
