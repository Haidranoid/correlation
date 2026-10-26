"""The unseen months (2025-06-30..2026-10-19), once, at the end: P1's forecasts for the decisions in them (fits on
every earlier eligible decision), each input alone, and S2's rule against always-ALC and SPY. Needs inputs.py new.
The old span's forecasts must come out unchanged in this run (asserted).

    python3 -I -B unseen.py     # -> unseen.txt, unseen.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_common as K  # noqa: E402
import numpy as np  # noqa: E402
import timing  # noqa: E402


def main():
    z = np.load(K.WORK / 'inputs_new.npz')
    zo = np.load(K.WORK / 'forecasts_old.npz')
    D, dates = z['D'], z['dates']
    y = z['y_ALC-SPY']
    X6 = np.column_stack([z[f'x_{n}_c0'] for n in K.INPUTS])
    o = K.oos(y, X6, D)
    n_old = len(zo['dates'])
    assert np.array_equal(dates[:n_old], zo['dates'])
    same = np.allclose(o['fc'][:n_old], zo['fc'], equal_nan=True) and np.allclose(o['f0'][:n_old], zo['f0'],
                                                                                   equal_nan=True)
    m = K.sel(dates, *K.UNSEEN)
    lines, res = [], {}

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    out(f'unseen decisions: {int(m.sum())}, {dates[m][0]}..{dates[m][-1]} (holds end by '
        f'{K.iso(z["axis"][D[m][-1] + 1 + K.H])}); old-span forecasts unchanged in this run: {same}')
    assert same
    ev = K.evaluate(y, o['f0'], o['fc'], m, boot=False)
    res['P1'] = ev
    out(f'P1: n {ev["n"]}, R2 {ev["r2os"]:+.4f}, Clark-West t {ev["cw_t"]:+.2f} (normal p {ev["cw_p"]:.3f})')
    res['C1'] = {}
    for k, n in enumerate(K.INPUTS):
        e = K.evaluate(y, o['f0'], o['fk'][:, k], m, boot=False)
        res['C1'][n] = e
    out('each input alone: ' + '; '.join(f'{n} R2 {e["r2os"]:+.3f} t {e["cw_t"]:+.2f}' for n, e in res['C1'].items()))
    # S2 over the unseen decisions
    Du = D[m]
    H = timing.holds(z, Du)
    rule = (o['fc'][m] > 0).astype(int)
    res['S2'] = {'share_alc': float(rule.mean())}
    accs = {}
    for nm, c in (('rule', rule), ('always ALC', np.ones(len(Du), int)), ('always SPY', np.zeros(len(Du), int))):
        dr, days, ml, paid = timing.account(H, c, K.COST)
        accs[nm] = dr
        res['S2'][nm] = {'total_return': float(np.prod(1 + dr) - 1), **timing.measures(dr)}
    for nm in ('rule', 'always ALC'):
        res['S2'][nm]['minus_spy'] = res['S2'][nm]['total_return'] - res['S2']['always SPY']['total_return']
    out(f'S2: the rule holds ALC in {rule.mean():.2f} of the unseen decisions; total return at 5 bps: rule '
        f'{res["S2"]["rule"]["total_return"]:+.3f}, always ALC {res["S2"]["always ALC"]["total_return"]:+.3f}, '
        f'SPY {res["S2"]["always SPY"]["total_return"]:+.3f}')
    (K.HERE / 'unseen.txt').write_text('\n'.join(lines) + '\n')
    K.dump(res, K.HERE / 'unseen.json')


if __name__ == '__main__':
    main()
