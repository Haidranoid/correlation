"""C1 (reported): which market moves first? Daily origins, same-dated data (it describes, it doesn't trade, so no
publication lag), Newey-West standard errors.

    python3 -I -B leadlag.py old|new        (after inputs.py <tag>)

(a) the spread follows stocks:  S(t + k) - S(t)         on  R5(t) = log P(t) - log P(t - 5)  and  S(t) - S(t - 5)
(b) stocks follow the spread:   log P(t + k) - log P(t) on  the same two regressors
k = 5 (Newey-West 10 lags) and k = 21 (42 lags). S is BAA10Y; also the default spread BAA10Y - AAA10Y.
"""
import json
import sys
import time

sys.path.insert(0, '/home/bhoult/programming/correlation/research/credit_spread')
import numpy as np  # noqa: E402

import cs_common as K  # noqa: E402

t_start = time.time()
tag = sys.argv[1]
assert tag in ('old', 'new')
z = K.load_inputs(tag)
dates = K.iso_days(z['days'])
T = len(dates)
lp = np.log(z['P'])


def fwd(x, k):
    out = np.full(T, np.nan)
    out[:T - k] = x[k:] - x[:T - k]
    return out


def back(x, k):
    out = np.full(T, np.nan)
    out[k:] = x[k:] - x[:T - k]
    return out


if tag == 'old':
    periods = {'1993-2007': K.sel(dates, '1993-01-01', '2007-12-31'), '2008-2019': K.sel(dates, *K.DEV),
               '2020-2025-06': K.sel(dates, *K.CHECK)}
else:
    periods = {'unseen months': K.sel(dates, *K.UNSEEN)}
res = {'tag': tag, 'rows': []}
for sname, S in (('BAA10Y', z['S']), ('BAA10Y - AAA10Y', z['D'])):
    X = np.column_stack([back(lp, 5), back(S, 5)])
    for k, lags in ((5, 10), (21, 42)):
        for side, y in (('(a) spread follows stocks', fwd(S, k)), ('(b) stocks follow the spread', fwd(lp, k))):
            for pn, pm in periods.items():
                m = pm & np.isfinite(y) & np.isfinite(X).all(1)
                if m.sum() < 100:
                    continue
                beta, se, t = K.ols_nw(X[m], y[m], lags)
                yy = y[m]
                fit = np.column_stack([np.ones(m.sum()), X[m]]) @ beta
                r2 = 1 - np.sum((yy - fit) ** 2) / np.sum((yy - yy.mean()) ** 2)
                sdx = X[m].std(0)
                res['rows'].append({'spread': sname, 'k': k, 'side': side, 'period': pn, 'n': int(m.sum()),
                                    'coef_R5': float(beta[1]), 't_R5': float(t[1]), 'coef_dS5': float(beta[2]), 't_dS5': float(t[2]),
                                    'r2': float(r2), 'per_sd_R5': float(beta[1] * sdx[0]), 'per_sd_dS5': float(beta[2] * sdx[1])})
res['seconds'] = round(time.time() - t_start)
with open(K.HERE / f'c1_{tag}.json', 'w') as f:
    json.dump(res, f, indent=1)
L = [f'C1 ({tag}): which market moves first. Coefficients with Newey-West t; "per sd" = the coefficient times the '
     "regressor's standard deviation (in the target's units: percentage points of spread, or log return)."]
L.append('| Spread | k | Regression | Period | Origins | on R5: coef (t) | per sd | on dS5: coef (t) | per sd | R2 |')
L.append('|---|---|---|---|---|---|---|---|---|---|')
for r in res['rows']:
    L.append(f"| {r['spread']} | {r['k']} | {r['side']} | {r['period']} | {r['n']} | {r['coef_R5']:+.4f} ({r['t_R5']:+.2f}) | "
             f"{r['per_sd_R5']:+.4f} | {r['coef_dS5']:+.4f} ({r['t_dS5']:+.2f}) | {r['per_sd_dS5']:+.4f} | {r['r2']:.3f} |")
L.append(f"Runtime {res['seconds']} s")
txt = '\n'.join(L)
print(txt)
(K.HERE / f'c1_{tag}.txt').write_text(txt + '\n')
