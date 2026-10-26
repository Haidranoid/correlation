"""C6, step 2: the news as inputs. Per source, the monthly share of alcohol headlines (N1) and their mean VADER tone
(N2), each standardized by the source's own previous 36 months (the share needs all 36; the tone at least 24 with a
value) and averaged over the sources that have the month. Month m is used at t0 if m <= month(t0) - 1. Each alone,
and both added to the six inputs (the eight-input combination against the six-input one), on the decisions they cover.

    python3 -I -B news.py old     # -> news_old.txt, news_old.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_common as K  # noqa: E402
import numpy as np  # noqa: E402


def zscores(mon, key, need):
    """month -> z of `key` against the previous 36 months of the same source."""
    ms = sorted(mon)
    out = {}
    for m in ms:
        v = mon[m][key]
        if v is None:
            continue
        prev = [mon.get(K.mkey(K.mindex(m) - j), {}).get(key) for j in range(1, 37)]
        prev = [p for p in prev if p is not None]
        if len(prev) < need:
            continue
        sd = np.std(prev, ddof=1)
        if sd > 0:
            out[m] = (v - np.mean(prev)) / sd
    return out


def main(tag):
    src = json.loads((K.WORK / 'news_monthly.json').read_text())
    for s in src.values():
        for v in s.values():
            v['share'] = v['n_alc'] / v['n'] if v['n'] else None
    Z = {'N1': [zscores(s, 'share', 36) for s in src.values()], 'N2': [zscores(s, 'vader', 24) for s in src.values()]}
    z = np.load(K.WORK / f'inputs_{tag}.npz')
    D, dates = z['D'], z['dates']
    y = z['y_ALC-SPY']
    X6 = np.column_stack([z[f'x_{n}_c0'] for n in K.INPUTS])
    N = {}
    for k, per_src in Z.items():
        v = np.full(len(D), np.nan)
        for i, d in enumerate(dates):
            mk = K.mkey(K.mindex(d) - 1)
            vals = [s[mk] for s in per_src if mk in s]
            if vals:
                v[i] = float(np.mean(vals))
        N[k] = v
    have = np.isfinite(N['N1']) & np.isfinite(N['N2']) & np.isfinite(X6).all(1)
    lines, res = [], {}

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    out(f'C6: decisions with both news inputs and the six: {int(have.sum())}, {dates[have][0]}..{dates[have][-1]}')
    per = {'development': K.sel(dates, *K.DEV), 'check': K.sel(dates, *K.CHECK)}
    for nm, X in (('N1 (share of alcohol headlines)', N['N1']), ('N2 (their tone)', N['N2'])):
        o = K.oos(y, X, D, sample=have, sink=False)
        res[nm] = {pn: K.evaluate(y, o['f0'], o['fc'], m) for pn, m in per.items()}
        out(f'  {nm} alone: ' + '; '.join(f'{pn} n {e.get("n")} R2 {e.get("r2os", np.nan):+.4f} Clark-West t '
                                         f'{e.get("cw_t", np.nan):+.2f}' for pn, e in res[nm].items()))
    o6 = K.oos(y, X6, D, sample=have, sink=False)
    o8 = K.oos(y, np.column_stack([X6, N['N1'], N['N2']]), D, sample=have, sink=False)
    res['eight vs mean'] = {pn: K.evaluate(y, o8['f0'], o8['fc'], m) for pn, m in per.items()}
    res['eight vs six'] = {pn: K.evaluate(y, o6['fc'], o8['fc'], m) for pn, m in per.items()}
    for nm in ('eight vs mean', 'eight vs six'):
        out(f'  {nm}: ' + '; '.join(f'{pn} n {e.get("n")} R2 {e.get("r2os", np.nan):+.4f} Clark-West t '
                                   f'{e.get("cw_t", np.nan):+.2f}' for pn, e in res[nm].items()))
    (K.HERE / f'news_{tag}.txt').write_text('\n'.join(lines) + '\n')
    K.dump(res, K.HERE / f'news_{tag}.json')


if __name__ == '__main__':
    main(sys.argv[1])
