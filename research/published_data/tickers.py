"""Tickers to fetch from Yahoo for the dividend-correction panel: every stock in the judge-like liquid set on any
twins-calendar day (export index 252 + 21k) from 2022-01-01 on, in both exports (c2eval.EXPORT_OLD and EXPORT_NEW).

The liquid-set rule is copied from research/night_and_day/common.py (universe()): closes on >= 90% of the trailing
756 days, median close >= $5 and median dollar volume >= $5M over the trailing 252 days, minus c2eval.excluded().

    python3 -I -B tickers.py      # -> out/tickers_stocks.txt (Yahoo symbols), out/tickers_map.tsv
"""
import os
import sys
import warnings
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
FROM = '2022-01-01'
DATA_W, DATA_FRAC = 756, 0.9
MED_W = 252
MIN_PRICE, MIN_DV = 5.0, 5e6


def yahoo_symbol(t):
    """Store ticker -> Yahoo symbol: class shares use '-', any '.US'-style suffix stripped
    (research/twin_gap_fade_check/positions.py, copied)."""
    t = str(t).upper()
    if t.endswith('.US'):
        t = t[:-3]
    return t.replace('.', '-').replace('/', '-')


def universe(E, tw, k_from=0):
    """[K, N] bool liquid set on tw['rebal'] days (rows before k_from left False). Copied from night_and_day."""
    close, dv = E['close'], E['dollar_volume']
    rebal = tw['rebal']
    N = close.shape[0]
    excl = c2eval.excluded(E, tw)
    uni = np.zeros((len(rebal), N), bool)
    for k, t0 in enumerate(rebal):
        if t0 < DATA_W - 1 or k < k_from:
            continue
        c = np.asarray(close[:, t0 - DATA_W + 1:t0 + 1], np.float64)
        fin = np.isfinite(c).mean(1) >= DATA_FRAC
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mc = np.nanmedian(c[:, -MED_W:], 1)
            mv = np.nanmedian(np.asarray(dv[:, t0 - MED_W + 1:t0 + 1], np.float64), 1)
        uni[k] = fin & (np.nan_to_num(mc) >= MIN_PRICE) & (np.nan_to_num(mv) >= MIN_DV) & ~excl[k]
    return uni


def main():
    rows = {}
    for tag, path in EXPORTS.items():
        E = c2eval.load_export(path, arrays=('close', 'dollar_volume'))
        tw = c2eval.twins(E)
        days = E['days']
        k_from = int(np.searchsorted(tw['rebal'], c2eval.day_index(days, FROM)))   # first rebal day on/after FROM
        uni = universe(E, tw, k_from)
        ever = uni[k_from:].any(0)
        sizes = uni[k_from:].sum(1)
        print(f'{tag}: rebal days from {c2eval.iso(days[tw["rebal"][k_from]])}: {len(sizes)}; liquid set size '
              f'min {sizes.min()} median {int(np.median(sizes))} max {sizes.max()}; union {int(ever.sum())}', flush=True)
        for j in np.nonzero(ever)[0]:
            t = str(E['tickers'][j])
            rows.setdefault(t, set()).add(tag)
        del E, tw
    tick = sorted(rows)
    ysym = [yahoo_symbol(t) for t in tick]
    assert len(set(ysym)) == len(ysym)
    (OUT / 'tickers_stocks.txt').write_text('\n'.join(ysym) + '\n')
    with open(OUT / 'tickers_map.tsv', 'w') as f:
        f.write('export_ticker\tyahoo\tin\n')
        for t, y in zip(tick, ysym):
            f.write(f'{t}\t{y}\t{",".join(sorted(rows[t]))}\n')
    print(f'union over both exports: {len(tick)} tickers (old only {sum(rows[t] == {"old"} for t in tick)}, '
          f'new only {sum(rows[t] == {"new"} for t in tick)}, both {sum(len(rows[t]) == 2 for t in tick)})')


if __name__ == '__main__':
    main()
