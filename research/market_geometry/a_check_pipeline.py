"""Mechanical checks of ladder.py's fitting and scoring, on synthetic signals only (no real forecaster is scored).

1. Perfect signal: x = the outcome itself. The expanding fit should find b = 1, a = 0, so R2 = 1 at every level.
2. Pure noise: x ~ N(0, 1), independent of everything. Out-of-sample R2 should be slightly negative (the cost of
   estimating a slope), and the market's (one pair a month) the most negative; a fit that leaked the scored month's
   own pairs would show positive R2 on average instead.
3. Leakage, directly: every fitted month k uses only pairs whose outcomes ended at or before t0 (asserted).

    python3 -I -B a_check_pipeline.py      -> a_check_pipeline.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder as L  # noqa: E402
import numpy as np  # noqa: E402

rows = dict(np.load(L.WORK / 'rows_old.npz'))
rebal = rows['rebal']
K = len(rebal)
mk = L.months_mask(rows, L.SCORE_FROM, L.SCORE_TO)
lines = []
say = lambda s: (print(s, flush=True), lines.append(s))  # noqa: E731

# 3. the lag rule
for h in L.HORIZONS:
    for off in (L.SKIP + h, h):
        lag = L._lag(rebal, off)
        ends = rebal[:-lag] + off
        assert np.all(ends <= rebal[lag:]), (h, off)
        assert np.any(rebal[:-(lag - 1)] + off > rebal[lag - 1:]) if lag > 1 else True
        say(f'lag check: outcome end offset {off} days -> fit uses decisions k-{lag} and earlier; '
            f'latest training outcome ends {int((rebal[lag:] - ends).min())} days before t0 at the closest')

# 1. perfect signal
X = rows['X'].copy()
rows_p = dict(rows, X=X)
X[:, L.FI['x_trend']] = X[:, L.FI['y21']]
for lvl in ('stock', 'b100', 'market'):
    r = L.score_return(rows_p, K, lvl, 0, 'x_trend', 'y21', L.SKIP + 21, rebal, ranks=False)
    say(f'perfect signal, {lvl}: R2 vs hist mean {1 - r["sse"][mk].sum() / r["sst"][mk].sum():.6f}; '
        f'b range {np.nanmin(r["b"][mk]):.4f}..{np.nanmax(r["b"][mk]):.4f}')

# 2. pure noise, 200 draws for the one-pair-a-month market and 20 for the pooled levels
rng = np.random.default_rng(7)
for lvl, draws in (('market', 200), ('stock', 20), ('b100', 20)):
    m = rows['lvl'] == L.LV[lvl]
    vals = []
    for _ in range(draws):
        X = rows['X'].copy()
        X[m, L.FI['x_trend']] = rng.standard_normal(m.sum()).astype(np.float32)
        r = L.score_return(dict(rows, X=X), K, lvl, 0, 'x_trend', 'y21', L.SKIP + 21, rebal, ranks=False)
        vals.append(1 - r['sse'][mk].sum() / r['sst'][mk].sum())
    vals = np.array(vals)
    say(f'pure noise, {lvl}: mean OOS R2 {vals.mean() * 100:+.3f}% (sd {vals.std() * 100:.3f}%, {draws} draws); '
        f'share of draws above 0: {(vals > 0).mean():.2f}')

(Path(__file__).resolve().parent / 'a_check_pipeline.txt').write_text('\n'.join(lines) + '\n')
