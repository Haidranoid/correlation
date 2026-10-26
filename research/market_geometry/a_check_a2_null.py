"""Reading aid for A2, added after results were seen (changes no rule and no verdict): the sign-flip null of the
groups-minus-matched R2 gap. Each of 200 draws flips every member's signal in month k by one random s_k = +-1, the same
flip for the groups and all 20 matched replicates. Every unit is refitted from scratch exactly as in ladder.py, and
the gap R2(groups) - R2(matched, pooled over the replicates) is rescored over 2008-01..2019-12. If groups' wider
spread of signals alone made their fitted line less noisy, the null gap would sit above 0.

    python3 -I -B a_check_a2_null.py      -> a_check_a2_null.txt (uses rows_old.npz and ladder.py, unchanged)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder as L  # noqa: E402
import numpy as np  # noqa: E402

N_DRAWS = 200
rows = dict(np.load(L.WORK / 'rows_old.npz'))
rebal = rows['rebal']
K = len(rebal)
mk = L.months_mask(rows, L.SCORE_FROM, L.SCORE_TO)
S = np.random.default_rng(L.SEED + 11).choice([-1.0, 1.0], (N_DRAWS, K))
real = json.loads((L.HERE / 'a_results_old.json').read_text())['periods']['test 2008-01..2019-12']['ret']


def unit_stats(lvl, rep, xcol, ycol):
    m = (rows['lvl'] == L.LV[lvl]) & (rows['rep'] == rep)
    x = rows['X'][m, L.FI[xcol]].astype(np.float64)
    y = rows['X'][m, L.FI[ycol]].astype(np.float64)
    k = rows['k'][m].astype(int)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, k = x[ok], y[ok], k[ok]
    bc = lambda w: np.bincount(k, weights=w, minlength=K)  # noqa: E731
    return bc(np.ones_like(x)), bc(x), bc(y), bc(x * x), bc(x * y), bc(y * y)


def sse_sst(st, s, lag):
    n, sx0, sy, sxx, sxy0, syy = st
    sx, sxy = sx0 * s, sxy0 * s
    cum = lambda v: np.concatenate([np.zeros(lag), np.cumsum(v)[:-lag]])  # noqa: E731
    Cn, Cx, Cy, Cxx, Cxy = cum(n), cum(sx), cum(sy), cum(sxx), cum(sxy)
    with np.errstate(all='ignore'):
        vx = Cxx - Cx * Cx / Cn
        b = np.where((Cn >= 3) & (vx > 0), (Cxy - Cx * Cy / Cn) / vx, np.nan)
        a = Cy / Cn - b * Cx / Cn
        ybar = Cy / Cn
    sse = syy - 2 * a * sy - 2 * b * sxy + n * a * a + 2 * a * b * sx + b * b * sxx
    sst = syy - 2 * ybar * sy + n * ybar * ybar
    good = mk & np.isfinite(sse) & np.isfinite(sst)
    return np.where(good, sse, 0).sum(), np.where(good, sst, 0).sum()


lines, out = [], {}
for sig, h in L.DISTINCT:
    xcol, ycol = L.sig_col(sig, h), f'y{h}'
    lag = L._lag(rebal, L.SKIP + h)
    g = unit_stats('group', 0, xcol, ycol)
    ms = [unit_stats('matched', r, xcol, ycol) for r in range(L.N_MATCH)]
    ones = np.ones(K)
    # sanity: with no flip this reproduces the real gap
    a, b = sse_sst(g, ones, lag)
    pm = [sse_sst(m, ones, lag) for m in ms]
    gap0 = (1 - a / b) - (1 - sum(p[0] for p in pm) / sum(p[1] for p in pm))
    gaps = np.empty(N_DRAWS)
    for d in range(N_DRAWS):
        a, b = sse_sst(g, S[d], lag)
        pm = [sse_sst(m, S[d], lag) for m in ms]
        gaps[d] = (1 - a / b) - (1 - sum(p[0] for p in pm) / sum(p[1] for p in pm))
    key = f'{sig}_h{h}'
    obs = real[key + '_skip']['a2_diff']
    out[key] = {'observed_gap_pct': obs * 100, 'recomputed_gap_pct': gap0 * 100, 'null_mean_pct': gaps.mean() * 100,
                'null_sd_pct': gaps.std(ddof=1) * 100, 'share_null_ge_observed': float((gaps >= obs).mean())}
    lines.append(f'{key}: observed groups - matched {obs * 100:+.3f} pts (recomputed {gap0 * 100:+.3f}); sign-flip null '
                 f'mean {gaps.mean() * 100:+.3f} sd {gaps.std(ddof=1) * 100:.3f} ({N_DRAWS} draws); share of null >= observed '
                 f'{(gaps >= obs).mean():.3f}')
    print(lines[-1], flush=True)
(L.HERE / 'a_check_a2_null.txt').write_text('\n'.join(lines) + '\n')
