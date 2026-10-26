"""The two README numbers (the .md, "The two README numbers").

    python3 -I -B after_costs.py old|new    # after run.py <tag>

After costs: P1's rebalanced portfolio (daily, delay 1, 5 bps a side), each calendar year's compounded daily return
minus SPY's Yahoo adjusted-close total return over the same days, averaged over the years. Buy-and-hold's and P2's
are reported beside it. Before costs, vs random: P1's rebalanced portfolio minus buy-and-hold, both gross, by
calendar year, averaged. Writes ac_<tag>.txt / .json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_b')
import eg_common as C  # noqa: E402
import ac  # noqa: E402

np = C.np


def main(tag):
    E = C.c2eval.load_export(C.EXPORTS[tag], arrays=())
    iso_all = np.array([C.c2eval.iso(d) for d in E['days']])
    S = np.load(C.WORK / f'series_{tag}.npz')
    days = S['days']
    iso = iso_all[days]
    yr = np.array([s[:4] for s in iso])
    ports = {k: (S[f'{k}_gross'], S[f'{k}_turn']) for k in ('R1d', 'R1m', 'B1')}
    rows = []
    for y in sorted(set(yr)):
        m = yr == y
        first, last = int(days[m][0]), int(days[m][-1])
        spy = 100 * ac.spy_ret_between(iso_all[first - 1], iso_all[last])
        row = {'year': y, 'from_close': iso_all[first - 1], 'to_close': iso_all[last], 'days': int(m.sum()), 'spy': spy}
        for k, (g, t) in ports.items():
            row[f'{k}_net5'] = 100 * (np.prod(1 + C.net(g, t, C.COST)[m]) - 1)
            row[f'{k}_gross'] = 100 * (np.prod(1 + g[m]) - 1)
            row[f'{k}_after'] = row[f'{k}_net5'] - spy
        row['before_vs_random'] = row['R1d_gross'] - row['B1_gross']
        rows.append(row)
    mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    out = {'tag': tag, 'rows': rows, 'n_years': len(rows),
           'after_costs': mean('R1d_after'), 'after_costs_B1': mean('B1_after'), 'after_costs_R1m': mean('R1m_after'),
           'before_vs_random': mean('before_vs_random'),
           'before_vs_random_se': float(np.std([r['before_vs_random'] for r in rows], ddof=1) / np.sqrt(len(rows)))
           if len(rows) > 1 else None,
           'ahead': int(sum(r['before_vs_random'] > 0 for r in rows)),
           'mean_R1d_net5': mean('R1d_net5'), 'mean_spy': mean('spy')}
    L = [f'After costs and before costs vs random ({tag}): calendar years by return date, percent']
    L.append(f"{'year':5s} {'span':24s} {'days':>4s} {'SPY':>7s} {'R1d net5':>8s} {'R1m net5':>8s} {'B1 net5':>8s} "
             f"{'R1d-SPY':>8s} {'B1-SPY':>8s} {'R1d-B1 gross':>12s}")
    for r in rows:
        L.append(f"{r['year']:5s} {r['from_close']}..{r['to_close']} {r['days']:4d} {r['spy']:+7.2f} {r['R1d_net5']:+8.2f} "
                 f"{r['R1m_net5']:+8.2f} {r['B1_net5']:+8.2f} {r['R1d_after']:+8.2f} {r['B1_after']:+8.2f} "
                 f"{r['before_vs_random']:+12.2f}")
    L.append(f"After costs (P1's rebalanced portfolio at 5 bps minus SPY, mean of {len(rows)} years): {out['after_costs']:+.2f}"
             f" (buy-and-hold {out['after_costs_B1']:+.2f}, monthly {out['after_costs_R1m']:+.2f}); portfolio "
             f"{out['mean_R1d_net5']:+.2f}, SPY {out['mean_spy']:+.2f}")
    L.append(f"Before costs, vs random (rebalanced minus buy-and-hold, gross): {out['before_vs_random']:+.2f} points a year"
             + (f" (se {out['before_vs_random_se']:.2f}; ahead in {out['ahead']} of {len(rows)} years)"
                if out['before_vs_random_se'] is not None else ''))
    txt = '\n'.join(L) + '\n'
    (C.HERE / f'ac_{tag}.txt').write_text(txt)
    (C.HERE / f'ac_{tag}.json').write_text(json.dumps(out, indent=1))
    print(txt)


if __name__ == '__main__':
    main(sys.argv[1])
