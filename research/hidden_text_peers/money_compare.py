"""Added after the results (report only; decides nothing): the same money test for the visible-peer and all-peer
signals, to see whether the judge's number is about the hidden peers or about text-industry momentum in general.

    python3 -I money_compare.py     # writes money_compare.txt here
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import HERE, WORK, c2eval, log, np  # noqa: E402


def scores(t, row, h, N, T):
    s = np.full((N, T), np.nan, np.float32)
    bounds = np.r_[0, np.nonzero(np.diff(t))[0] + 1, len(t)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        v = h[a:b]
        ok = np.isfinite(v)
        n = int(ok.sum())
        if n < 2:
            continue
        r = np.empty(n)
        r[np.argsort(v[ok], kind='stable')] = np.arange(n)
        s[row[a:b][ok], t[a]] = r / (n - 1)
    return s


def main():
    E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=())
    N, T = len(E['tickers']), len(E['days'])
    z = np.load(WORK / 'panel.npz')
    lines = []
    for name in ('Vres', 'Ares'):
        s = scores(z['t'], z['row'], z[name], N, T)
        p = c2eval.save_scores(s, f'hidden_text_peers/scores/compare_{name}')
        del s
        for rm in ('uniform', 'volatility'):
            txt, js = c2eval.judge(p, rebalance=21, random_match=rm)
            ch = c2eval.change_over_random(js)
            lines.append(f'{name} {rm}: {c2eval.summary_line(js)}; per year {ch["per_year"]}')
            log(lines[-1])
    (HERE / 'money_compare.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
