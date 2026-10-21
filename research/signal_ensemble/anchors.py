"""Sensitivity, added after seeing results (reported only, not a test): test 1 and test 2 on the other four weekly
calendars (export indices 252 + 5k + a, a = 1..4), with everything else unchanged. a = 0 is the fixed calendar.

    python3 -I anchors.py    -> anchors_old.txt
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json
from common import c2eval, np, EXPORTS, wdir, HERE, LGBM, COMPONENTS, LABEL, TOP, KEEP, ffill, tstat, in_period
from composite import ranks_at, combine
from tests import weekly_ic, paired_ic, book

tag = 'old'
E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
days = E['days']
N, T = E['close'].shape
P = ffill(E['close'])
U = np.load(wdir(tag) / 'universe.npy')
X = {nm: np.ascontiguousarray(np.load(wdir(tag) / f'{nm}.npy').T) for nm in ('rev5', 'mom12_1', 'twin5', 'vol', 'corr')}
X['lgbm'] = np.ascontiguousarray(np.load(LGBM).T)
lines, res = [], {}
for a in range(5):
    W = np.arange(252 + a, T - 5, 5)
    W = W[in_period(days, W, '2008-01-01', '2019-12-31')]
    nW = len(W)
    R = np.full((nW, len(COMPONENTS), N), np.nan, np.float32)
    comp = np.full((nW, N), np.nan, np.float32)
    for i, d in enumerate(W):
        u = np.nonzero(U[d])[0]
        Ri = ranks_at(u, d, X)
        R[i][:, u] = Ri
        comp[i, u] = combine(Ri, list(range(len(COMPONENTS))))
    G = (P[:, W + 5] / P[:, W]).T
    weeks = np.arange(nW)
    m, t, n = tstat(weekly_ic(comp, G, weeks))
    row = {'composite': (m, t)}
    beats = True
    for c in COMPONENTS:
        x, y = paired_ic(R[:, COMPONENTS.index(c)], comp, G, weeks)
        tc, tk = tstat(x)[1], tstat(y)[1]
        beats &= tk > tc
        row[c] = (tstat(x)[0], tc, tk)
    b = book(comp, G, weeks, TOP, KEEP)
    g, nt = tstat(b['gross']), tstat(b['net'])
    res[a] = {'ic': m, 'ic_t': t, 'beats': bool(beats), 'net': nt[0], 'net_t': nt[1], 'gross': g[0], 'gross_t': g[1],
              'components': {c: row[c] for c in COMPONENTS}}
    lines.append(f'anchor +{a}: composite IC {m:+.4f} (t {t:+.2f}, {n} wk); beats every component {beats}; '
                 + '; '.join(f'{LABEL[c]} t {row[c][1]:+.2f} (composite {row[c][2]:+.2f})' for c in COMPONENTS)
                 + f'; buffered long-short gross {g[0] * 1e4:+.2f} (t {g[1]:+.2f}), net {nt[0] * 1e4:+.2f} bps/wk (t {nt[1]:+.2f})')
    print(lines[-1], flush=True)
(HERE / 'anchors_old.txt').write_text('\n'.join(lines) + '\n')
(HERE / 'anchors_old.json').write_text(json.dumps(res, indent=1, default=float))
