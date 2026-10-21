"""-16.2-twin-gap-fade: the daily $1-long/$1-short book (all stocks), from research/twin_gap_fade/out/daily_old.tsv.
Columns: gross (no costs), bound (gross - netted twin cost at 2 bps/side), head (bound - 2 x 10 bps entry penalty).
After-cost book: stocks 5 bps per side on every trade = in at the open and out at the close on both legs = 4 x 5 bps
= 20 bps a day (the same as the file's 10 bps entry penalty with free exits); twin funds 1 bp per side = half the
saved netted twin cost. Alternative (caller's literal reading): 5 bps entry penalty per side, free exits = 10 bps a day.
Calendar years 2008..2025 (2025 = 01-02..06-27), compounded within each year, then averaged."""
import json
import numpy as np
P = '/home/bhoult/programming/correlation/research/twin_gap_fade/out/daily_old.tsv'
rows = [l.rstrip('\n').split('\t') for l in open(P)]
h = rows[0]; rows = rows[1:]
col = {k: i for i, k in enumerate(h)}
date = np.array([r[0] for r in rows])
g = np.array([float(r[col['gross']]) for r in rows])
b = np.array([float(r[col['bound']]) for r in rows])
hd = np.array([float(r[col['head']]) for r in rows])
assert np.allclose(hd, b - 0.0020, atol=2e-6)
tc2 = g - b                      # netted twin cost at 2 bps a side
print('days', len(g), date[0], date[-1], 'head 252*mean %.1f%%' % (252 * hd.mean() * 100),
      'gross bps/day %.2f' % (1e4 * g.mean()), 'twin cost bps/day %.2f' % (1e4 * tc2.mean()))
variants = {
    'main_5bps_every_trade_20bps_day_fund1bp': g - 0.0020 - 0.5 * tc2,
    'alt_5bps_entry_only_10bps_day_fund1bp': g - 0.0010 - 0.5 * tc2,
    'file_headline_10bps_entry_fund2bp': hd,
}
years = np.array([d[:4] for d in date])
out = {}
for name, x in variants.items():
    yr = {}
    for y in sorted(set(years)):
        m = years == y
        yr[y] = {'compounded_pct': 100 * (np.prod(1 + x[m]) - 1), 'arith_252_pct': 100 * 252 * x[m].mean(),
                 'sum_pct': 100 * x[m].sum(), 'days': int(m.sum())}
    comp = np.mean([v['compounded_pct'] for v in yr.values()])
    ar = np.mean([v['sum_pct'] for v in yr.values()])
    print(f"{name:45s} mean bps/day {1e4 * x.mean():+6.2f}  mean of yearly compounded {comp:+10.1f}%  "
          f"mean of yearly sums {ar:+7.1f}%  252*mean {100 * 252 * x.mean():+7.1f}%")
    for y, v in yr.items():
        print(f"   {y} days {v['days']:3d} compounded {v['compounded_pct']:+10.1f}%  sum {v['sum_pct']:+7.1f}%")
    out[name] = {'mean_yearly_compounded_pct': comp, 'mean_yearly_sum_pct': ar, 'years': yr}
json.dump(out, open('/home/bhoult/.local/share/correlation2/experiments/after_costs/group_b_twin_gap_fade.json', 'w'), indent=1)
