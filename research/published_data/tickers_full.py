"""Tickers to fetch from Yahoo for the full-history dividend panel (div_panel_full.py): the store's closes have no
dividend adjustment before about February 2011 as well as after 2022, so the panel must cover every year.

Stocks: every stock in the judge-like liquid set (tickers.universe, copied from night_and_day) on any twins-calendar
day of either export, from the first one (2005) on. Funds: every fund chosen as some stock's twin on any calendar day
of either export, plus SPY, QQQ, IWM, DIA and the nine SPDR sector funds.

    python3 -I -B tickers_full.py   # -> out/tickers_full_stocks.txt, out/tickers_full_map.tsv,
                                    #    out/tickers_full_funds.txt, out/tickers_full_funds_map.tsv
"""
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

from tickers import EXPORTS, OUT, universe, yahoo_symbol  # noqa: E402

FIXED_FUNDS = ('SPY', 'QQQ', 'IWM', 'DIA', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY')


def write_map(name, rows):
    tick = sorted(rows)
    ysym = [yahoo_symbol(t) for t in tick]
    assert len(set(ysym)) == len(ysym)
    (OUT / f'{name}.txt').write_text('\n'.join(ysym) + '\n')
    with open(OUT / f'{name}_map.tsv', 'w') as f:
        f.write('export_ticker\tyahoo\tin\n')
        for t, y in zip(tick, ysym):
            f.write(f'{t}\t{y}\t{",".join(sorted(rows[t]))}\n')
    return tick


def main():
    stocks, funds = {}, {}
    for tag, path in EXPORTS.items():
        E = c2eval.load_export(path, arrays=('close', 'dollar_volume'))
        tw = c2eval.twins(E)
        uni = universe(E, tw, 0)
        for j in np.nonzero(uni.any(0))[0]:
            stocks.setdefault(str(E['tickers'][j]), set()).add(tag)
        F = tw['F']
        used = np.unique(tw['fund_row'][tw['fund_row'] >= 0])
        for r in used:
            funds.setdefault(str(F['tickers'][r]), set()).add(tag)
        ft = set(str(t) for t in F['tickers'])
        for t in FIXED_FUNDS:
            assert t in ft, (tag, t)
            funds.setdefault(t, set()).add(tag)
        sizes = uni.sum(1)
        print(f'{tag}: calendar days {len(sizes)}, liquid set min {sizes.min()} median {int(np.median(sizes))} max '
              f'{sizes.max()}; stocks ever {int(uni.any(0).sum())}; twin funds ever {len(used)}', flush=True)
        del E, tw
    ts = write_map('tickers_full_stocks', stocks)
    tf = write_map('tickers_full_funds', funds)
    # the map files keep the names tickers.py uses for its own lists: *_map.tsv next to *.txt
    print(f'stocks: {len(ts)}; funds: {len(tf)}')


if __name__ == '__main__':
    main()
