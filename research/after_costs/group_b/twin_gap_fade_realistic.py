"""-16.2-twin-gap-fade, the tradable version: the headline all-stocks book entered at 9:35 (entry (c)) on the
58-day 5-minute window, 2026-07-29..10-19, from research/twin_gap_fade_check/out/daily_check.tsv (window 5m, book all,
entry 0935). Columns: gross (long minus short, each hedged with beta x twin from 9:35 to the store's close) and tc
(netted twin cost at 2 bps a side, books.fund_cost). Costs here: 5 bps on each of the 4 stock trades a day = 20 bps,
plus 1 bp a side on the netted fund legs = tc / 2."""
import numpy as np
P = '/home/bhoult/programming/correlation/research/twin_gap_fade_check/out/daily_check.tsv'
OUT = '/home/bhoult/programming/correlation/research/after_costs/group_b/twin_gap_fade_realistic.tsv'
rows = [l.rstrip('\n').split('\t') for l in open(P)]
h = rows[0]
sel = [r for r in rows[1:] if r[0] == '5m' and r[1] == 'all' and r[3] == '0935']
dates = [r[2] for r in sel]
g = np.array([float(r[h.index('gross')]) for r in sel])
tc = np.array([float(r[h.index('tc')]) for r in sel])
assert len(g) == 58 and dates[0] == '2026-07-29' and dates[-1] == '2026-10-19', (len(g), dates[0], dates[-1])

def t(x):
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))

plan = g - 0.0020 - tc                 # the file's headline costs (check: -10.26 bps, t -2.0)
net = g - 0.0020 - tc / 2              # brief's costs
print(f'gross {1e4 * g.mean():+.2f} bps/day t {t(g):+.2f}; plan-cost net {1e4 * plan.mean():+.2f} t {t(plan):+.2f}; '
      f'twin cost (2 bp) {1e4 * tc.mean():.2f} bps/day')
m = net.mean()
a = 100 * 252 * m
b = 100 * ((1 + m) ** 252 - 1)
c = t(net)
cum = 100 * (np.prod(1 + net) - 1)
print(f'net {1e4 * m:+.2f} bps/day: (a) 252 x mean {a:+.2f}%  (b) (1+mean)^252-1 {b:+.2f}%  (c) t {c:+.2f}; '
      f'compounded over the 58 days {cum:+.2f}%')
how = (f"The tradable version: the headline all-stocks book entered at 9:35 (entry (c) of the intraday check) instead of "
       f"the opening print, exit at the store's close, on the 5-minute window's common positions; costs 5 bps on each of "
       f"the 4 stock trades a day (20 bps) plus 1 bp a side on the netted twin legs (half the saved 2-bp tc). Net "
       f"{1e4 * m:+.2f} bps a day; after_cost = 252 x mean daily net ({a:+.1f}); (1 + mean)^252 - 1 = {b:+.1f}; t of the "
       f"daily mean {c:+.2f}; gross {1e4 * g.mean():+.2f} bps a day (t {t(g):+.2f}).")
with open(OUT, 'w') as fh:
    fh.write('file\tafter_cost\tkind\tyears\thow\tsources\n')
    fh.write('\t'.join(['-16.2-twin-gap-fade.md', f'{a:+.1f}', 'long-short net', '58 days 2026-07-29..10-19, entry at 9:35', how,
                        'research/twin_gap_fade_check/out/daily_check.tsv (window 5m, book all, entry 0935: gross, tc); '
                        'research/twin_gap_fade_check/analyze.py (daily()); research/twin_gap_fade/books.py (fund_cost, TWIN_SIDE); '
                        'experiments/-16.2-twin-gap-fade.md (Independent check with intraday bars)']) + '\n')
