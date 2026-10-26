"""C4 (other measures of the alcohol market), C6 (the news) and C8 (cannabis in Massachusetts): the same model,
unchanged. C4 and C6 on the development and check periods; C8 on all its months together (the plan). The two-lag
benchmark of Decisions 2 is reported beside the plan's for C4.

    python3 -I -B other.py     # -> other.txt, other.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mf_common as M  # noqa: E402  (puts ../alcohol_demand on the path)
import ad_common as K  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PERIODS = (('dev', M.DEV), ('check', M.CHECK))


def one(y, fb, fm, mask, boot=True):
    return M.evaluate([y], [fb], [fm], mask, boot=boot)


def main():
    lines, res = [], {}

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    X = M.common_inputs()

    # C4
    out('C4: other measures of the alcohol market, the model (nine inputs) against own history')
    measures = {'liquor-store sales (MRTS, NSA)': (M.fred_raw('MRTSSM4453USN'), K.read_fred('CUUR0000SAF116')),
                'TTB wine taxable withdrawals': (M.ttb('wine'), M.bls(M.PRICE['W'])),
                'TTB beer taxable removals': (M.ttb('beer'), M.bls(M.PRICE['B'])),
                'TTB spirits taxable withdrawals': (M.ttb('spirits'), M.bls(M.PRICE['S']))}
    res['C4'] = {'redacted': M.REDACTED}
    out(f'  months TTB withheld (missing): {M.REDACTED}')
    for nm, (S, price) in measures.items():
        res['C4'][nm] = {}
        for h in (12, 3):
            months, y, g, Xm = M.panel(X, S, price, h)
            for bn, G in (('one-lag', g), ('two-lag', M.two_lag(g))):
                o = M.oos(y, G, Xm, months, h, sink=False)
                fm = o['fk'].mean(1)
                for p, rng in PERIODS:
                    e = one(y, o['fb'], fm, M.period_mask(months, h, *rng))
                    eb = one(y, o['f0'], o['fb'], M.period_mask(months, h, *rng), boot=False)
                    res['C4'][nm][f'h{h} {bn} {p}'] = {'model': e, 'bench_vs_mean': eb}
                    out(f'  {nm}, h={h}, {bn}, {p}: {M.fmt_ev(e)}' +
                        (f'; own history vs the mean R2 {eb["r2os"]:+.3f}' if eb.get('n', 0) >= 12 else ''))

    # C6
    out()
    out('C6: the news (N1 headline share, N2 tone) as two more inputs, on the origins they cover (h = 12)')
    N = M.news_inputs()
    res['C6'] = {}
    ins = M.INPUTS + ('N1', 'N2')
    ys, fbs, f9, f11, fn1, fn2 = [], [], [], [], [], []
    for b, code in M.MARKETS.items():
        months, y, g, Xm = M.panel(X, M.bea(code), M.bls(M.PRICE[b]), 12, extra=N, inputs=ins)
        o = M.oos(y, g, Xm, months, 12, sink=False)
        ys.append(y), fbs.append(o['fb'])
        f9.append(o['fk'][:, :9].mean(1)), f11.append(o['fk'].mean(1))
        fn1.append(o['fk'][:, 9]), fn2.append(o['fk'][:, 10])
    have = np.isfinite(f11[0])
    out(f'  forecasts with the news exist for target ends {M.mkey(months[have][0] + 12)}..{M.mkey(months[have][-1] + 12)}')
    for p, rng in PERIODS:
        mask = M.period_mask(months, 12, *rng)
        r = {'eleven_vs_bench': M.evaluate(ys, fbs, f11, mask), 'eleven_vs_nine': M.evaluate(ys, f9, f11, mask),
             'N1': M.evaluate(ys, fbs, fn1, mask), 'N2': M.evaluate(ys, fbs, fn2, mask)}
        res['C6'][p] = r
        for k, e in r.items():
            out(f'  {p}, {k.replace("_", " ")}: {M.fmt_ev(e)}')

    # C8
    out()
    out('C8: cannabis, Massachusetts adult-use retail sales (dollars), inputs E1 E2 E3 P(price per gram) T3, '
        'at least 24 eligible origins per fit, all months together')
    S, price = M.ma_sales(), M.ma_price()
    res['C8'] = {}
    for h in (3, 12):
        months, y, g, Xm = M.panel(X, S, price, h, inputs=('E1', 'E2', 'E3', 'P', 'T3'))
        o = M.oos(y, g, Xm, months, h, min_fit=24, sink=False)
        fm = o['fk'].mean(1)
        mask = M.period_mask(months, h, '2018-12', '2026-09')
        e = one(y, o['fb'], fm, mask)
        eb = one(y, o['f0'], o['fb'], mask, boot=False)
        ok = mask & np.isfinite(fm) & np.isfinite(y)
        mae = {nm: float(100 * np.mean(np.abs(y[ok] - f[ok]))) for nm, f in
               (('mean', o['f0']), ('no change', g), ('own history', o['fb']), ('model', fm))}
        res['C8'][f'h{h}'] = {'model': e, 'bench_vs_mean': eb, 'mae': mae,
                              'span': [M.mkey(months[ok][0] + h), M.mkey(months[ok][-1] + h)] if ok.any() else None}
        out(f'  h={h}: targets ending {res["C8"][f"h{h}"]["span"]}: {M.fmt_ev(e)}')
        if eb.get('n', 0) >= 12:
            out(f'        own history against the mean: R2 {eb["r2os"]:+.3f}; MAE (pts) ' +
                ', '.join(f'{k} {v:.2f}' for k, v in mae.items()))

    (HERE / 'other.txt').write_text('\n'.join(lines) + '\n')
    M.dump(res, HERE / 'other.json')


if __name__ == '__main__':
    main()
