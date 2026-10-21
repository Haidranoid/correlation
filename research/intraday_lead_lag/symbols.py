"""The universe of the intraday lead-lag test (experiments/-100.0-intraday-lead-lag.md): the 500 most-traded stocks
of the judge-like liquid set on the new export's last day, minus c2eval.excluded, plus the plan's 20 funds.
Writes out/symbols.txt (Yahoo symbols, one per line) and out/universe.tsv (export row, store ticker, Yahoo symbol,
median 63-day dollar volume, kind).

The liquid set is night_and_day/common.py's rule evaluated at t0 = the export's last day (not the last twins
rebalance day): closes on >= 90% of the trailing 756 days, median close >= $5 and median dollar volume >= $5M over
the trailing 252 days, minus excluded(E, tw)[rebal_index(tw, t0)]; the 500 largest by median dollar volume over the
trailing 63 days. Only the universe comes from the export; no return from it is used.

    python3 -I symbols.py
"""
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

FUNDS = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY',
         'SMH', 'KRE', 'XBI', 'GDX', 'XOP']
DATA_W, DATA_FRAC, MED_W = 756, 0.9, 252
MIN_PRICE, MIN_DV = 5.0, 5e6
DV_LARGE_W, N_LARGE = 63, 500


def yahoo_symbol(t):
    """Store ticker -> Yahoo symbol (as twin_gap_fade_check/positions.py): class shares use '-'."""
    t = str(t).upper()
    if t.endswith('.US'):
        t = t[:-3]
    return t.replace('.', '-').replace('/', '-')


def main():
    E = c2eval.load_export(c2eval.EXPORT_NEW, arrays=('close', 'dollar_volume'))
    tw = c2eval.twins(E)
    N, T = E['close'].shape
    t0 = T - 1
    k = c2eval.rebal_index(tw, t0)
    excl = c2eval.excluded(E, tw)[k]
    c = np.asarray(E['close'][:, t0 - DATA_W + 1:t0 + 1], np.float64)
    fin = np.isfinite(c).mean(1) >= DATA_FRAC
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mc = np.nanmedian(c[:, -MED_W:], 1)
        v = np.asarray(E['dollar_volume'][:, t0 - MED_W + 1:t0 + 1], np.float64)
        mv = np.nanmedian(v, 1)
        mv63 = np.nanmedian(v[:, -DV_LARGE_W:], 1)
    u = fin & (np.nan_to_num(mc) >= MIN_PRICE) & (np.nan_to_num(mv) >= MIN_DV) & ~excl
    cand = np.nonzero(u)[0]
    o = cand[np.argsort(-np.nan_to_num(mv63[cand], nan=-1.0), kind='stable')][:N_LARGE]
    stocks = [(int(j), str(E['tickers'][j]), yahoo_symbol(E['tickers'][j]), float(mv63[j])) for j in o]
    ssyms = {s[2] for s in stocks}
    clash = sorted(ssyms & set(FUNDS))
    with open(HERE / 'out' / 'universe.tsv', 'w') as f:
        f.write('row\tticker\tyahoo\tmedian_dv63\tkind\n')
        for j, t, y, d in stocks:
            f.write(f'{j}\t{t}\t{y}\t{d:.0f}\tstock\n')
        for y in FUNDS:
            f.write(f'-1\t{y}\t{y}\t\tfund\n')
    syms = [s[2] for s in stocks] + [y for y in FUNDS if y not in ssyms]
    (HERE / 'out' / 'symbols.txt').write_text('\n'.join(syms) + '\n')
    print(f'export last day {c2eval.iso(E["days"][t0])} (t0={t0}); twins rebalance {c2eval.iso(E["days"][tw["rebal"][k]])}; '
          f'liquid set {int(u.sum())}; 500 largest: smallest median dv63 ${stocks[-1][3] / 1e6:.1f}M; '
          f'fund clashes {clash}; symbols {len(syms)}')


if __name__ == '__main__':
    main()
