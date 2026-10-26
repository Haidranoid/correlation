"""P1 (primary) and the forecast checks C1, C2, C3, C7 and C8 (experiments/untested-luca-alcohol-demand.md).

    python3 -I -B forecast.py old    # -> p1_old.txt, p1_old.json, forecasts_old.npz (WORK)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_common as K  # noqa: E402
import numpy as np  # noqa: E402

C2_TARGETS = ('ALC-SPY', 'ALC', 'ALC-XLP', 'GLOBAL-SPY', 'XLP-SPY', 'XLY-SPY', 'QQQ-SPY', 'SPY')
RAW_ASSETS = ('ALC', 'GLOBAL', 'XLP', 'XLY', 'QQQ', 'SPY')


def periods(dates):
    return {'development': K.sel(dates, *K.DEV), 'check': K.sel(dates, *K.CHECK)}


def line(name, ev):
    if ev.get('n', 0) < 12:
        return f'{name}: n {ev.get("n", 0)}'
    s = (f'{name}: n {ev["n"]}, R2 {ev["r2os"]:+.4f}')
    if 'r2_lo95' in ev:
        s += f' [{ev["r2_lo95"]:+.3f}, {ev["r2_hi95"]:+.3f}]'
    s += f', Clark-West t {ev["cw_t"]:+.2f} (normal p {ev["cw_p"]:.3f}'
    if 'cw_boot_p' in ev:
        s += f', bootstrap p {ev["cw_boot_p"]:.4f}'
    return s + ')'


def main(tag):
    z = np.load(K.WORK / f'inputs_{tag}.npz')
    D, dates = z['D'], z['dates']
    X6 = np.column_stack([z[f'x_{n}_c0'] for n in K.INPUTS])
    y = z['y_ALC-SPY']
    per = periods(dates)
    res, lines = {}, []

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    # ---- P1 ----
    o = K.oos(y, X6, D)
    first = dates[np.argmax(np.isfinite(o['fc']))]
    out(f'P1: ALC - SPY, the six inputs combined against the prevailing mean. First forecast {first}.')
    res['P1'] = {}
    for pn, m in per.items():
        ev = K.evaluate(y, o['f0'], o['fc'], m)
        res['P1'][pn] = ev
        out('  ' + line(pn, ev))
    d = res['P1']['development']
    passed = d['cw_t'] >= 3 and d['cw_boot_p'] < 0.0013 and d['r2os'] > 0
    c = res['P1']['check']
    consistent = c['r2os'] > 0 and c['cw_p'] < 0.05
    res['P1']['pass'] = bool(passed)
    res['P1']['check_consistent'] = bool(consistent)
    out(f'  P1 passes (t >= 3, bootstrap p < 0.0013, R2 > 0 over development): {passed}; check consistent: '
        f'{consistent}')
    # by year, and without crisis years
    yrs = np.array([d_[:4] for d_ in dates])
    res['P1']['by_year'] = {}
    for yr in sorted(set(yrs[np.isfinite(o['fc'])])):
        ev = K.evaluate(y, o['f0'], o['fc'], yrs == yr, boot=False)
        if ev.get('n', 0) >= 12:
            res['P1']['by_year'][yr] = {'n': ev['n'], 'r2os': ev['r2os'], 'cw_t': ev['cw_t']}
    out('  by year (R2): ' + ', '.join(f'{k} {v["r2os"]:+.3f}' for k, v in res['P1']['by_year'].items()))
    dev = per['development']
    for nm, m in (('development without 2008-2009', dev & ~np.isin(yrs, ['2008', '2009'])),
                  ('check without 2020', per['check'] & (yrs != '2020'))):
        ev = K.evaluate(y, o['f0'], o['fc'], m)
        res['P1'][nm] = ev
        out('  ' + line(nm, ev))
    # the slopes over time
    fin = np.isfinite(o['bk']).all(1)
    res['P1']['slopes'] = {}
    for k, n in enumerate(K.INPUTS):
        b = o['bk'][fin, k]
        res['P1']['slopes'][n] = {'first': float(b[0]), 'median_dev': float(np.median(o['bk'][fin & dev, k])),
                                  'last': float(b[-1]), 'share_positive': float(np.mean(b > 0))}
    out('  slopes (first, median over development, last; share of fits positive): ' + '; '.join(
        f'{n} {v["first"]:+.4f} {v["median_dev"]:+.4f} {v["last"]:+.4f} {v["share_positive"]:.2f}'
        for n, v in res['P1']['slopes'].items()))
    np.savez(K.WORK / f'forecasts_{tag}.npz', dates=dates, D=D, y=y, f0=o['f0'], fc=o['fc'], fk=o['fk'],
             fs=o['fs'])

    # ---- C1: each input alone; the kitchen sink ----
    out()
    out('C1: each input alone against the prevailing mean (same decisions as P1); Holm across the six '
        '(normal p, development).')
    res['C1'] = {}
    ps = []
    for k, n in enumerate(K.INPUTS):
        res['C1'][n] = {pn: K.evaluate(y, o['f0'], o['fk'][:, k], m, seed=K.SEED + 10 + k) for pn, m in per.items()}
        ps.append(res['C1'][n]['development']['cw_p'])
    adj = K.holm(ps)
    for k, n in enumerate(K.INPUTS):
        res['C1'][n]['holm_dev'] = float(adj[k])
        out(f'  {n}: ' + line('development', res['C1'][n]['development']) + f', Holm {adj[k]:.3f}; '
            + line('check', res['C1'][n]['check']))
    res['C1']['sink'] = {pn: K.evaluate(y, o['f0'], o['fs'], m, seed=K.SEED + 20) for pn, m in per.items()}
    out('  kitchen sink (one regression on all six): ' + line('development', res['C1']['sink']['development'])
        + '; ' + line('check', res['C1']['sink']['check']))

    # ---- C2: the same pipeline on other targets; how speculative each is ----
    out()
    out('C2: the P1 pipeline on each target, with the six inputs and with the economy pair (E1, E2) only.')
    res['C2'] = {}
    XE = X6[:, :2]
    for tn in C2_TARGETS:
        yt = z[f'y_{tn}']
        o6 = K.oos(yt, X6, D, sink=False)
        oe = K.oos(yt, XE, D, sink=False)
        r = {'six': {pn: K.evaluate(yt, o6['f0'], o6['fc'], m) for pn, m in per.items()},
             'economy': {pn: K.evaluate(yt, oe['f0'], oe['fc'], m) for pn, m in per.items()}}
        res['C2'][tn] = r
        out(f'  {tn:10s} six: ' + line('development', r['six']['development']) + '; '
            + line('check', r['six']['check']))
        out(f'  {"":10s} economy: ' + line('development', r['economy']['development']) + '; '
            + line('check', r['economy']['check']))
    out('  how speculative each asset is (21-day log returns, decisions 2008-01..2025-06):')
    m = K.sel(dates, K.DEV[0], K.CHECK[1])
    spy = z['y_SPY'][m]
    res['C2']['describe'] = {}
    for a in RAW_ASSETS:
        r = z[f'y_{a}'][m]
        ok = np.isfinite(r) & np.isfinite(spy)
        rr, ss = r[ok], spy[ok]
        beta = float(np.cov(rr, ss)[0, 1] / np.var(ss, ddof=1))
        corr = float(np.corrcoef(rr, ss)[0, 1])
        vol = float(np.std(rr, ddof=1) * np.sqrt(252 / K.H))
        res['C2']['describe'][a] = {'n': int(ok.sum()), 'vol': vol, 'beta': beta, 'corr': corr, 'r2_spy': corr ** 2}
        out(f'    {a:7s} volatility {vol:.3f} a year, beta {beta:.2f}, correlation {corr:.2f}, '
            f'share of variance from SPY {corr ** 2:.2f}')

    # ---- C3: retail sales (revised; a look-ahead) ----
    out()
    out('C3: retail sales at beer, wine and liquor stores (share of all retail; final vintage, revised).')
    R = z['x_R_c0']
    X7 = np.column_stack([X6, R])
    o7 = K.oos(y, X7, D, sink=False)
    oR = K.oos(y, np.column_stack([R]), D, sink=False, sample=np.isfinite(X6).all(1))
    res['C3'] = {'seven': {pn: K.evaluate(y, o7['f0'], o7['fc'], m) for pn, m in per.items()},
                 'alone': {pn: K.evaluate(y, oR['f0'], oR['fc'], m) for pn, m in per.items()}}
    for nm in ('seven', 'alone'):
        out(f'  {nm}: ' + line('development', res['C3'][nm]['development']) + '; '
            + line('check', res['C3'][nm]['check']))

    # ---- C7: extra caution; C8: the wider basket ----
    out()
    X6c = np.column_stack([z[f'x_{n}_c1'] for n in K.INPUTS])
    oc = K.oos(y, X6c, D, sink=False)
    res['C7'] = {pn: K.evaluate(y, oc['f0'], oc['fc'], m) for pn, m in per.items()}
    out('C7: every input a month older: ' + line('development', res['C7']['development']) + '; '
        + line('check', res['C7']['check']))
    yw = z['y_WIDE-SPY']
    ow = K.oos(yw, X6, D, sink=False)
    res['C8'] = {pn: K.evaluate(yw, ow['f0'], ow['fc'], m) for pn, m in per.items()}
    out('C8: ALC + GLOBAL (nine names) - SPY: ' + line('development', res['C8']['development']) + '; '
        + line('check', res['C8']['check']))

    (K.HERE / f'p1_{tag}.txt').write_text('\n'.join(lines) + '\n')
    K.dump(res, K.HERE / f'p1_{tag}.json')


if __name__ == '__main__':
    main(sys.argv[1])
