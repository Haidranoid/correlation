"""Probe: 5-day reversal with scores NaN before 2015-06-29, graded walk-forward rebalance=5 (uniform and volatility).
Shows how the judge treats score-less years, and gives reversal's change over random on the test years 2015-16..2024-25."""
import sys
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
c = np.asarray(e['close'], dtype=np.float64)
s = np.full(c.shape, np.nan, np.float32)
with np.errstate(all='ignore'):
    s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
del c
d0 = c2eval.day_index(e['days'], '2015-06-28')
s[:, :d0] = np.nan
p = c2eval.save_scores(s, 'formula_mining/probe_reversal5_from2015')
for rm in ('uniform', 'volatility'):
    txt, js = c2eval.judge(p, rebalance=5, random_match=rm)
    print(rm, 'skipped:', js.get('skipped'))
    for y in js['years']:
        print(' ', y['start'], 'cand', y['candidates'], 'pairs', y['pairs'], 'days', y['days'], [round(c['final_pct'] - c['random_median_pct'], 2) for c in y['costs']])
