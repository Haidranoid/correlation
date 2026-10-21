"""Fidelity check: the vectorized rolling-sum fits (common.fit_panel / fit_block) against a direct per-window fit with
np.linalg.lstsq, for chosen stock-days: random ones, first days of a twin block, windows with missing days, and a
window where a fund day is dropped by the |log| > 0.5 rule. No strategy returns are computed.

    python3 -I -B check_fit.py      # writes out/check_fit.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    C.log(s)


E, tw, uni, big = C.load('old')
days = E['days']
T = len(days)
rs = C.signal_returns(C.log_returns(np.asarray(E['close'], np.float64)))
rf = C.signal_returns(C.log_returns(np.asarray(tw['F']['close'], np.float64)))
rng = np.random.default_rng(7)
rebal = tw['rebal']
ks = np.nonzero(uni.sum(1))[0]
cases = []
# random stock-days
for _ in range(12):
    k = int(rng.choice(ks))
    t1 = int(rebal[k + 1]) if k + 1 < len(rebal) else T
    i = int(rng.choice(np.nonzero(uni[k])[0]))
    cases.append(('random', k, i, int(rng.integers(rebal[k], t1))))
# first day of a block
for _ in range(4):
    k = int(rng.choice(ks))
    i = int(rng.choice(np.nonzero(uni[k])[0]))
    cases.append(('block start', k, i, int(rebal[k])))
# windows with missing stock or fund days
found = 0
for k in rng.permutation(ks):
    t1 = int(rebal[k + 1]) if k + 1 < len(rebal) else T
    si = np.nonzero(uni[k])[0]
    fj = tw['fund_row'][k, si]
    for t in range(int(rebal[k]), t1):
        y = rs[si, t - C.W + 1:t + 1]
        x = rf[fj, t - C.W + 1:t + 1]
        nmiss = (~(np.isfinite(y) & np.isfinite(x))).sum(1)
        hit = np.nonzero((nmiss >= 1) & (nmiss <= 3))[0]
        if len(hit):
            cases.append(('missing days', int(k), int(si[hit[0]]), t))
            found += 1
            break
    if found >= 4:
        break
# a fund day dropped by the |log| > 0.5 rule
lrf = C.log_returns(np.asarray(tw['F']['close'], np.float64))
badf = np.argwhere(np.abs(np.nan_to_num(lrf)) > C.MAX_LOG)
done = False
for f, d in badf:
    for k in ks:
        t1 = int(rebal[k + 1]) if k + 1 < len(rebal) else T
        if not (rebal[k] <= d + 5 < t1):
            continue
        si = np.nonzero(uni[k] & (tw['fund_row'][k] == f))[0]
        if len(si):
            cases.append(('fund bad print', int(k), int(si[0]), int(d + 5)))
            done = True
            break
    if done:
        break
if not done:
    say('no universe stock had a twin with a |log| > 0.5 day inside a window')

keys = ('beta', 'alpha', 'a', 'b', 'var_zeta', 'm', 'sigeq')
worst = {k: 0.0 for k in keys}
say(f'{"case":15s} {"date":10s} {"ticker":7s} {"twin":6s} nv  ' + ' '.join(f'{k:>10s}' for k in keys) + '   max rel diff   X60')
for name, k, i, t in cases:
    t1 = int(rebal[k + 1]) if k + 1 < len(rebal) else T
    t0 = int(rebal[k])
    f = int(tw['fund_row'][k, i])
    fb = C.fit_block(rs[[i], t0 - C.W + 1:t1], rf[[f], t0 - C.W + 1:t1])
    j = t - t0
    vec = {kk: float(fb[kk][0, j]) for kk in keys}
    dr = C.fit_direct(rs[i, t - C.W + 1:t + 1], rf[f, t - C.W + 1:t + 1])
    rel = {kk: abs(vec[kk] - dr[kk]) / max(abs(dr[kk]), 1e-12) for kk in keys}
    for kk in keys:
        worst[kk] = max(worst[kk], rel[kk])
    say(f'{name:15s} {c2eval.iso(days[t])} {E["tickers"][i]:7s} {tw["fund_ticker"][f]:6s} {dr["nv"]:2d}  '
        + ' '.join(f'{dr[kk]:10.3g}' for kk in keys) + f'   {max(rel.values()):.1e}   {dr["X60"]:.1e}')
say('worst relative difference per quantity:', {k: f'{v:.1e}' for k, v in worst.items()})
# and the panel function itself (fit_panel) on one block equals fit_block
k = int(ks[len(ks) // 2])
si = np.nonzero(uni[k])[0][:50]
um = np.zeros_like(uni)
um[k, si] = True
fp = C.fit_panel(rs, rf, tw, um)
t0 = int(rebal[k])
t1 = int(rebal[k + 1])
fb = C.fit_block(rs[si, t0 - C.W + 1:t1], rf[tw['fund_row'][k, si], t0 - C.W + 1:t1])
d = np.nanmax(np.abs(fp['m'][si, t0:t1] - fb['m'].astype(np.float32)))
say(f'fit_panel vs fit_block on block {c2eval.iso(days[t0])} (50 stocks): max |m diff| {d:.1e}')
(C.OUT / 'check_fit.txt').write_text('\n'.join(LINES) + '\n')
