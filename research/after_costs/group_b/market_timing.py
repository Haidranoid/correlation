"""-3.2-market-timing: the co-movement rule (100% or 50% SPY, rest cash at 0, bought at the close of t0 and held to
the close of t0+21), from research/market_timing/months_old.tsv (242 months, 2005-04-04..2025-06-13).
After cost: 5 bps on every SPY trade (the buy at the start and each month's trade back to the target weight from the
drifted one). Calendar years by each day's date (2005 = from the close of 2005-04-04; 2025 = to the close of
2025-06-13), compounded within each year from daily returns, minus SPY's total return over the same days.
SPY = Yahoo adjusted close (pubdata) for both the rule's holding and the benchmark; the store's SPY is a check."""
import sys, json
import numpy as np
sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_b')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_timing')
import ac
import c2eval
E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=())
days = E['days']
iso = np.array([c2eval.iso(d) for d in days])
rows = [l.rstrip('\n').split('\t') for l in open('/home/bhoult/programming/correlation/research/market_timing/months_old.tsv')]
h = rows[0]; rows = rows[1:]
t0 = np.array([int(np.searchsorted(iso, r[0])) for r in rows]); assert all(iso[t0] == [r[0] for r in rows])
w = np.array([float(r[h.index('w_rule')]) for r in rows])
assert np.all(np.diff(t0) == 21), 'months must be contiguous'
spy_next = np.array([float(r[h.index('spy_next21')]) for r in rows])
# SPY Yahoo adjclose on export days
k = np.searchsorted(ac.SPY_DATES, days.astype('datetime64[D]'))
ok = (k < len(ac.SPY_DATES)) & (ac.SPY_DATES[np.minimum(k, len(ac.SPY_DATES) - 1)] == days.astype('datetime64[D]'))
Py = np.where(ok, ac.SPY_ADJ[np.minimum(k, len(ac.SPY_DATES) - 1)], np.nan)
# the store's SPY (as the experiment used it), for the check
import common as MC
Ps = MC.spy_close('old', days)
end = t0[-1] + 21
span = np.arange(t0[0], end + 1)
assert np.isfinite(Py[span]).all(), 'Yahoo SPY missing on an export day in the span'

def run(P, cost):
    """daily returns of the rule over export days t0[0]+1..end, with cost on each SPY trade."""
    r = []
    wd = 0.0  # current SPY weight (fraction of account) before the month's trade
    prev = 1.0  # account value at the close of t0, before the trade there
    for m in range(len(t0)):
        tr = abs(w[m] - wd)
        v = prev * (1.0 - tr * cost)   # pay at the close of t0; charged in the first day's return of the month
        s0 = P[t0[m]]
        for j in range(1, 22):
            rel = P[t0[m] + j] / s0
            val = v * ((1 - w[m]) + w[m] * rel)
            r.append(val / prev - 1); prev = val
        rel_end = P[t0[m] + 21] / s0
        wd = w[m] * rel_end / ((1 - w[m]) + w[m] * rel_end)
    return np.array(r)

rd = days[span[1:]]
yr = np.array([c2eval.iso(d)[:4] for d in rd])
spy_d = Py[span[1:]] / Py[span[:-1]] - 1
check = {}
for name, P, cost in (('yahoo_5bps', Py, 5e-4), ('yahoo_0bps', Py, 0.0), ('store_0bps', Ps, 0.0), ('store_5bps', Ps, 5e-4)):
    x = run(P, cost)
    cagr = np.exp(np.log1p(x).sum() * 252 / len(x)) - 1
    check[name] = (x, cagr)
    print(f'{name}: CAGR {100 * cagr:+.2f}%')
# check: the store-SPY rule's month returns vs the tsv's spy_next21
mret = Ps[t0 + 21] / Ps[t0] - 1
print('max |store month return - tsv spy_next21|', np.abs(mret - spy_next).max())
x = check['yahoo_5bps'][0]
res = {}
for y in sorted(set(yr)):
    m = yr == y
    a = 100 * (np.prod(1 + x[m]) - 1); s = 100 * (np.prod(1 + spy_d[m]) - 1)
    a0 = 100 * (np.prod(1 + check['yahoo_0bps'][0][m]) - 1)
    res[y] = dict(rule5=a, rule0=a0, spy=s, diff=a - s, days=int(m.sum()))
    print(f"{y} days {m.sum():3d} rule@5 {a:+7.2f} rule@0 {a0:+7.2f} SPY {s:+7.2f} diff {a - s:+7.2f}")
r5 = np.mean([v['rule5'] for v in res.values()]); sp = np.mean([v['spy'] for v in res.values()])
print(f'mean over {len(res)} calendar years: rule@5 {r5:+.2f}  SPY {sp:+.2f}  after-cost {r5 - sp:+.2f}')
full = [y for y, v in res.items() if v['days'] > 240]
print(f'full years only ({full[0]}..{full[-1]}, {len(full)}): after-cost {np.mean([res[y]["diff"] for y in full]):+.2f}')
json.dump({'years': res, 'mean_rule5': r5, 'mean_spy': sp, 'after_cost': r5 - sp},
          open('/home/bhoult/.local/share/correlation2/experiments/after_costs/group_b_market_timing.json', 'w'), indent=1)
