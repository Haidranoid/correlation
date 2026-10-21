"""Data-quality look at the store's opens (no signal, no outcome-vs-signal relationship): how often the open is
missing, equal to the close, outside the day's high-low range, and how often the store's close disagrees with the
export's (repaired) close. Used to fix the leg filters before any test is run.

    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 -I dataqc.py old
"""
import sys
from pathlib import Path

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else 'old'
EXP = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}[tag]
STO = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}[tag]
E = c2eval.load_export(EXP, arrays=('close', 'dollar_volume'))
S = c2eval.load_store(STO)
days = E['days']
T = len(days)
d0 = c2eval.day_index(days, '2008-01-01')
spy = [s for s in S['syms'] if s['ticker'] == 'SPY']
print('SPY entries', spy)
ids = E['ids']
o = c2eval.store_panel(S, ids, days, 'open')[:, d0:]
c = c2eval.store_panel(S, ids, days, 'close')[:, d0:]
h = c2eval.store_panel(S, ids, days, 'high')[:, d0:]
lo = c2eval.store_panel(S, ids, days, 'low')[:, d0:]
ce = np.asarray(E['close'][:, d0:], np.float32)
dv = np.asarray(E['dollar_volume'][:, d0:], np.float32)
liq = np.nan_to_num(dv) >= 5e6          # rough liquid filter for the QC only
fc = np.isfinite(c) & (c > 0)
print('stock-days with store close (2008+):', int(fc.sum()))
print('  open missing or <= 0:', float((fc & ~(np.isfinite(o) & (o > 0))).sum() / fc.sum()))
eq = fc & (o == c)
print('  open == close:', float(eq.sum() / fc.sum()), ' liquid (dv>=5M):', float((eq & liq).sum() / (fc & liq).sum()))
eq3 = fc & (o == c) & (h == lo)
print('  open == close == high == low:', float(eq3.sum() / fc.sum()), ' liquid:', float((eq3 & liq).sum() / (fc & liq).sum()))
with np.errstate(all='ignore'):
    out = fc & ((o > h * 1.001) | (o < lo * 0.999))
print('  open outside [low, high] by > 0.1%:', float(out.sum() / fc.sum()), ' liquid:', float((out & liq).sum() / (fc & liq).sum()))
fe = fc & np.isfinite(ce)
with np.errstate(all='ignore'):
    mis = fe & (np.abs(np.log(c / ce)) > 0.01)
print('  store close vs export close differ > 1%:', float(mis.sum() / fe.sum()), ' liquid:', float((mis & liq).sum() / (fe & liq).sum()))
print('  export close finite but store close missing:', float((np.isfinite(ce) & ~fc).sum() / np.isfinite(ce).sum()))
# leg magnitudes (distribution only)
with np.errstate(all='ignore'):
    on = np.log(o[:, 1:] / c[:, :-1])
    intra = np.log(c[:, 1:] / o[:, 1:])
for nm, x in (('overnight', on), ('intraday', intra)):
    f = np.isfinite(x)
    ax = np.abs(x[f])
    print(f'  {nm}: n {f.sum()}, |x|>0.2 {float((ax > 0.2).mean()):.2e}, |x|>0.5 {float((ax > 0.5).mean()):.2e}, '
          f'p99 {np.quantile(ax, 0.99):.4f}')
# by year: fraction open == close among liquid stock-days
yrs = np.array([int(c2eval.iso(d)[:4]) for d in days[d0:]])
for y in np.unique(yrs):
    m = yrs == y
    a = (fc & liq)[:, m].sum()
    print(y, 'liquid stock-days', int(a), 'open==close', round(float((eq & liq)[:, m].sum() / max(a, 1)), 4),
          'outside range', round(float((out & liq)[:, m].sum() / max(a, 1)), 4),
          'open missing', round(float(((fc & liq) & ~(np.isfinite(o) & (o > 0)))[:, m].sum() / max(a, 1)), 4))
