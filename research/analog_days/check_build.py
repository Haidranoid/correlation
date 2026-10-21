"""Plumbing checks on the built state (no returns): analog spacing, lag, count, z-score point-in-time, forecast."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C
from common import c2eval, np
P = C.build('old')
days, idx, Z, ok, O, F = P['days'], P['idx'], P['Z'], P['ok'], P['O'], P['F']
has = idx[:, 0] >= 0
ts = np.nonzero(has)[0]
for t in ts:
    a = np.sort(idx[t])
    assert len(set(a.tolist())) == C.K and (np.diff(a) >= C.GAP).all() and a.max() <= t - C.H and ok[a].all(), t
# nearest-first check on a few days: brute force
rng = np.random.default_rng(1)
for t in rng.choice(ts, 30, replace=False):
    cand = np.nonzero(ok[:t - C.H + 1])[0]
    d2 = ((Z[cand] - Z[t]) ** 2).sum(1)
    acc = []
    for a in cand[np.argsort(d2, kind='stable')]:
        if all(abs(a - b) >= C.GAP for b in acc):
            acc.append(a)
        if len(acc) == C.K: break
    assert acc == idx[t].tolist(), t
    assert np.allclose(F[t], O[acc].mean(0))
# z-score point in time: recompute one day from X
t = ts[len(ts) // 2]
w = P['X'][t - 251:t + 1]
assert np.allclose(Z[t], (P['X'][t] - w.mean(0)) / w.std(0, ddof=1))
print('first complete state', c2eval.iso(days[np.argmax(ok)]), 'first forecast', c2eval.iso(days[ts[0]]), 'n forecasts', len(ts),
      'gaps in forecasts after first:', int((~has[ts[0]:]).sum()))
print('notes:', list(P['notes']))
print('feature sd check (z over all ok days, should be ~1):', np.round(np.nanstd(Z[ok], 0), 2))
# portfolio plumbing: an oracle (next week's actual sector-minus-SPY return as the score) must earn a lot; the same
# score one week stale must not; a hand-computed week must match
has = idx[:, 0] >= 0
f0 = C.first_formation(has, C.LAST_OLD)
forms = np.array([f for f in range(f0, len(days)) if f % C.STEP == 0 and f + C.H <= C.LAST_OLD])
orc = C.weekly(P, forms, O)
stale = np.full_like(O, np.nan); stale[5:] = O[:-5]
st = C.weekly(P, forms, stale)
print('oracle: gross %+.1f bps/wk (t %+.1f); one week stale: %+.2f bps/wk (t %+.2f)' % (1e4 * orc['gross'].mean(), C.tstat(orc['gross'])[1],
      1e4 * st['gross'].mean(), C.tstat(st['gross'])[1]))
an = C.weekly(P, forms, F)
i = 100; f = forms[i]
order = np.argsort(-F[f]); L_, S_ = order[:3], order[-3:]
g = P['c'][1:, f + 5] / P['c'][1:, f] - 1
hand = g[L_].mean() - g[S_].mean()
prev_w = an['w'][i - 1] * (P['c'][1:, f] / P['c'][1:, forms[i - 1]])
hand_turn = np.abs(an['w'][i] - prev_w).sum()
print('hand-computed week', c2eval.iso(days[f]), 'gross', round(hand * 1e4, 3), 'code', round(an['gross'][i] * 1e4, 3),
      'turnover', round(hand_turn, 4), 'code', round(an['turnover'][i], 4))
assert abs(hand - an['gross'][i]) < 1e-12 and abs(hand_turn - an['turnover'][i]) < 1e-12
print('first week turnover (from cash):', an['turnover'][0])
