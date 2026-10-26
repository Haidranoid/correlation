"""Diagnostic added after S2's results were seen (not part of the plan's tests): are the warning months worse than
random months?

    python3 -I -B check_random_warnings.py old|new      (after timing.py <tag>)

A rule that sits in cash for a share f of its months has a Sharpe ratio about sqrt(1 - f) times its baseline's even
when the cash months are random, because zero-return days lower the mean by (1 - f) and the standard deviation by
about sqrt(1 - f). So "Sharpe above rule B" penalises any cash overlay. This check compares each warning rule with
10,000 rules that put rule B in cash for the same number of randomly chosen months of the same period (rule B's
weights otherwise), at 5 bps. p = (1 + draws with a Sharpe at or above the rule's) / 10,001.
"""
import json
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/credit_spread')
import numpy as np  # noqa: E402

import cs_common as K  # noqa: E402

sys.path.insert(1, '/home/bhoult/programming/correlation/research/market_timing')
import engine as G  # noqa: E402

tag = sys.argv[1]
z = K.load_inputs(tag)
dates = K.iso_days(z['days'])
T = len(dates)
off = int(z['offset'])
ed = dates[off:]
rows = [l.split('\t') for l in (K.HERE / f'months_{tag}.tsv').read_text().splitlines()]
h = rows[0]
rows = rows[1:]
dd = np.array([r[0] for r in rows])
t0s = np.array([int(np.searchsorted(ed, d)) for d in dd])
assert all(ed[t0s] == dd) and np.all(np.diff(t0s) == K.H)
A0 = t0s + off
wB = np.array([float(r[h.index('w_B')]) for r in rows])
M = len(t0s)
P = z['P'][off:]


def path(t0, w, cost):
    out = np.empty((len(t0), K.H))
    wd, prev = 0.0, 1.0
    for m, t in enumerate(t0):
        s = t + 1
        v = prev * (1.0 - abs(w[m] - wd) * cost)
        rel = P[s + 1:s + K.H + 1] / P[s]
        val = v * ((1 - w[m]) + w[m] * rel)
        out[m] = val / np.concatenate([[prev], val[:-1]]) - 1.0
        prev = val[-1]
        wd = w[m] * rel[-1] / ((1 - w[m]) + w[m] * rel[-1])
    return out


def sharpe_many(t0, W, cost, chunk=1000):
    """Sharpe ratios (5 bps path, as path()) for many weight vectors W [D, M] at once."""
    out = np.empty(len(W))
    rel = np.stack([P[t + 2:t + K.H + 2] / P[t + 1] for t in t0])            # [M, 21]
    for a in range(0, len(W), chunk):
        Wc = W[a:a + chunk]
        D = len(Wc)
        wd, prev = np.zeros(D), np.ones(D)
        dr = np.empty((D, len(t0), K.H))
        for m in range(len(t0)):
            v = prev * (1.0 - np.abs(Wc[:, m] - wd) * cost)
            val = v[:, None] * ((1 - Wc[:, m])[:, None] + Wc[:, m][:, None] * rel[m][None, :])
            dr[:, m] = val / np.concatenate([prev[:, None], val[:, :-1]], 1) - 1.0
            prev = val[:, -1]
            wd = Wc[:, m] * rel[m, -1] / ((1 - Wc[:, m]) + Wc[:, m] * rel[m, -1])
        f = dr.reshape(D, -1)
        out[a:a + D] = f.mean(1) / f.std(1, ddof=1) * np.sqrt(G.DPY)
    return out


def trailing(x, t, fn):
    prev = x[max(0, t - K.PCT_MAX):t]
    prev = prev[np.isfinite(prev)]
    return float(fn(prev)) if len(prev) >= K.PCT_MIN else float('nan')


def fire(dS, q, stock=True):
    x = dS[A0]
    f = np.isfinite(x) & (x > 0)
    if q is not None:
        thr = np.array([trailing(dS, a, lambda v: np.percentile(v, 100 * q)) for a in A0])
        f &= np.isfinite(thr) & (x >= thr)
    if stock:
        f &= np.isfinite(z['R'][A0]) & (z['R'][A0] >= 0)
    return f


rules = {}
for q in K.GRID_Q:
    for stock in (True, False):
        nm = f"{'dS > 0' if q is None else f'dS >= {int(100 * q)}th pct'}{' and R >= 0' if stock else ', no stock condition'}"
        rules[nm] = fire(z['dS1'], q, stock)
rules['C2 default spread, top fifth and R >= 0'] = fire(z['dD1'], K.Q_FIRE)
main = rules['dS >= 80th pct and R >= 0']
assert np.array_equal(main.astype(int), np.array([int(r[h.index('warning')]) for r in rows]))
if tag == 'old':
    periods = {'2008-01..2019-12': K.sel(dd, *K.DEV), '2020-01..2025-06': K.sel(dd, *K.CHECK),
               '2010-2019': K.sel(dd, '2010-01-01', '2019-12-31')}
else:
    periods = {'unseen months': np.ones(M, bool)}
rng = np.random.default_rng(K.SEED + 77)
out = {}
for pn, m in periods.items():
    tt, wb = t0s[m], wB[m]
    sB = G.measures(path(tt, wb, K.COST))['sharpe']
    nxt = np.array([P[t + 1 + K.H] / P[t + 1] - 1 for t in tt])
    for nm, f in rules.items():
        ff = f[m]
        k = int(ff.sum())
        if k == 0:
            continue
        s = G.measures(path(tt, np.where(ff, 0.0, wb), K.COST))['sharpe']
        assert abs(sharpe_many(tt, np.where(ff, 0.0, wb)[None, :], K.COST)[0] - s) < 1e-10
        picks = np.argsort(rng.random((K.N_RANDOM, len(tt))), 1)[:, :k]
        Wr = np.tile(wb, (K.N_RANDOM, 1))
        np.put_along_axis(Wr, picks, 0.0, 1)
        draws = sharpe_many(tt, Wr, K.COST)
        out.setdefault(pn, {})[nm] = {'warning_months': k, 'sharpe': s, 'sharpe_B': sB, 'random_median': float(np.median(draws)),
                                      'p': float((1 + np.sum(draws >= s)) / (1 + K.N_RANDOM)),
                                      'spy_next21_warning': float(nxt[ff].mean()), 'spy_next21_other': float(nxt[~ff].mean())}
        print(pn, nm, out[pn][nm], flush=True)
with open(K.HERE / f'check_random_warnings_{tag}.json', 'w') as fo:
    json.dump(out, fo, indent=1)
L = ['Warning rules against rule B put in cash for the same number of random months (10,000 draws, 5 bps; after results)']
L.append('| Period | Rule | Warning months | Sharpe | Rule B | Median random-cash Sharpe | p | SPY next 21 d, warning / other months |')
L.append('|---|---|---|---|---|---|---|---|')
for pn, d in out.items():
    for nm, q in d.items():
        L.append(f"| {pn} | {nm} | {q['warning_months']} | {q['sharpe']:.3f} | {q['sharpe_B']:.3f} | {q['random_median']:.3f} | {q['p']:.4f} | "
                 f"{100 * q['spy_next21_warning']:+.2f}% / {100 * q['spy_next21_other']:+.2f}% |")
txt = '\n'.join(L)
print(txt)
(K.HERE / f'check_random_warnings_{tag}.txt').write_text(txt + '\n')
