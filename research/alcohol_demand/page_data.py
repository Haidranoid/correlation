"""Data for the explainer page (not part of the test): Google Trends series, the forecasts' running score against
the average, growth of $1, sales growth, and the tables, from the run's saved outputs and the 'new' inputs.

    python3 -I -B page_data.py     # -> artifact_data.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_common as K  # noqa: E402
import numpy as np  # noqa: E402
import timing  # noqa: E402


def r(x, d=4):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def main():
    out = {}
    tr = {}
    for q in ('alcohol', 'moderation', 'new_markets', 'wine', 'hard_seltzer', 'cannabis'):
        s = K.read_trends(q)
        tr[q] = [[k, v] for k, v in sorted(s.items())]
    out['trends'] = tr

    z = np.load(K.WORK / 'inputs_new.npz')
    D, dates, axis = z['D'], z['dates'], z['axis']
    y = z['y_ALC-SPY']
    X6 = np.column_stack([z[f'x_{n}_c0'] for n in K.INPUTS])
    o = K.oos(y, X6, D, sink=False)
    ok = np.isfinite(o['fc']) & np.isfinite(y)
    gain = np.cumsum(np.where(ok, (y - o['f0']) ** 2 - (y - o['fc']) ** 2, 0.0))
    out['score'] = [[dates[i], r(gain[i], 5), r(y[i]), r(o['fc'][i] - o['f0'][i], 5)] for i in np.nonzero(ok)[0]]

    use = ok & (dates <= K.UNSEEN[1])
    Du = D[use]
    H = timing.holds(z, Du)
    rule = (o['fc'][use] > 0).astype(int)
    acc = {}
    for nm, c in (('rule', rule), ('alc', np.ones(len(Du), int)), ('spy', np.zeros(len(Du), int))):
        dr, days, _, _ = timing.account(H, c, K.COST)
        acc[nm] = np.cumprod(1 + dr)
    keep = np.arange(0, len(days), 5)
    out['growth'] = {'dates': [K.iso(axis[d]) for d in days[keep]],
                     **{nm: [r(v, 3) for v in a[keep]] for nm, a in acc.items()},
                     'first_buy': K.iso(axis[Du[0] + 1])}

    S = K.read_fred('MRTSSM4453USS')
    g = K.yoy(S)
    out['sales_growth'] = [[k, r(v)] for k, v in sorted(g.items()) if k >= '2005-01']

    for f in ('p1_old', 's2_old', 'events_old', 'sales_old', 'news_old', 'unseen', 'codecheck_old'):
        out[f] = json.loads((K.HERE / f'{f}.json').read_text())
    (K.HERE / 'artifact_data.json').write_text(json.dumps(out, separators=(',', ':')))
    print(len(json.dumps(out)), 'bytes')


if __name__ == '__main__':
    main()
