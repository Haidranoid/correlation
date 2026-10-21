"""Data-quality look at the store's opens (no returns of any book are computed here).

    python3 -I dataq.py      # writes out/dataq.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np
import c2eval

OUT = Path(__file__).resolve().parent / 'out'
LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close', 'dollar_volume'))
S = c2eval.load_store(c2eval.STORE_OLD)
days = E['days']
T = len(days)
t08 = c2eval.day_index(days, '2008-01-01')
say('export', len(E['ids']), 'x', T, 'from 2008 index', t08, c2eval.iso(days[t08]))
o = c2eval.store_panel(S, E['ids'], days, 'open')
c = c2eval.store_panel(S, E['ids'], days, 'close')
h = c2eval.store_panel(S, E['ids'], days, 'high')
l = c2eval.store_panel(S, E['ids'], days, 'low')
ce = np.asarray(E['close'])
dv = np.asarray(E['dollar_volume'])
# crude liquid mask: dollar volume >= 5M and close >= 5 on the day (only for this quality look)
liq = (np.nan_to_num(dv) >= 5e6) & (np.nan_to_num(ce) >= 5)
liq[:, :t08] = False
n = liq.sum()
say('stock-days with DV>=5M and close>=5 from 2008:', n)
fo, fc = np.isfinite(o), np.isfinite(c)
say('open missing (NaN):', (liq & ~fo).sum() / n)
say('open <= 0:', (liq & fo & (o <= 0)).sum() / n)
say('open == close:', (liq & (o == c)).sum() / n)
say('open == high == low == close:', (liq & (o == c) & (h == l) & (o == h)).sum() / n)
pc = np.full_like(c, np.nan); pc[:, 1:] = c[:, :-1]
say('open == previous close:', (liq & (o == pc)).sum() / n)
say('open outside [low, high] by > 0.1%:', (liq & ((o > h * 1.001) | (o < l * 0.999))).sum() / n)
with np.errstate(all='ignore'):
    rel = np.abs(c / ce - 1)
say('store close vs export close differ by > 0.5%:', (liq & (rel > 0.005)).sum() / n)
with np.errstate(all='ignore'):
    on = np.log(o / pc)
    idr = np.log(c / o)
say('|overnight log| > 0.5:', (liq & (np.abs(on) > 0.5)).sum(), ' |intraday log| > 0.5:', (liq & (np.abs(idr) > 0.5)).sum())
for y in (2008, 2012, 2016, 2020, 2024):
    a, b = c2eval.day_index(days, f'{y}-01-01'), c2eval.day_index(days, f'{y + 1}-01-01')
    m = liq[:, a:b]
    say(y, 'liquid stock-days', m.sum(), 'open==close', round(float((m & (o[:, a:b] == c[:, a:b])).sum() / m.sum()), 4),
        'open==prev close', round(float((m & (o[:, a:b] == pc[:, a:b])).sum() / m.sum()), 4),
        'open NaN', round(float((m & ~fo[:, a:b]).sum() / m.sum()), 4))
OUT.mkdir(exist_ok=True)
(OUT / 'dataq.txt').write_text('\n'.join(LINES) + '\n')
