"""Data checks before any book is run (no returns of any strategy here): the export's store ids map to the same
tickers in the matching store, how often the store's raw close disagrees with the export's repaired close, and the
universe's size.

    python3 -I -B dataqc.py old|new      # writes out/dataqc_<tag>.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else 'old'
LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    C.log(s)


E, tw, uni, big = C.load(tag)
days = E['days']
S = c2eval.load_store(C.STORES[tag])
by_id = {s['id']: s for s in S['syms']}
tick_store = np.array([by_id[i]['ticker'] if i in by_id else '?' for i in E['ids']])
kind_store = np.array([by_id[i]['kind'] if i in by_id else '?' for i in E['ids']])
same = tick_store == E['tickers']
say(f'{tag}: export rows {len(E["ids"])}; store ticker == export ticker for {int(same.sum())}; differ {int((~same).sum())}')
if (~same).any():
    say('  examples of differing rows:', list(zip(E['tickers'][~same][:10], tick_store[~same][:10])))
say('  store kinds of export rows:', dict(zip(*np.unique(kind_store, return_counts=True))))
ce = np.asarray(E['close'], np.float64)
cs = C.stock_close(E, tag, 'store')
dv = np.asarray(E['dollar_volume'])
d0 = c2eval.day_index(days, '2005-01-01')
U = C.by_day(uni, tw['rebal'], len(days))
both = U & np.isfinite(ce) & np.isfinite(cs)
both[:, :d0] = False
with np.errstate(all='ignore'):
    rel = np.abs(np.log(cs / ce))
say(f'universe stock-days from 2005 with both closes: {int(both.sum())}')
for thr in (0.001, 0.005, 0.01, 0.05):
    say(f'  store vs export close differ by > {thr:.1%}: {float((both & (rel > thr)).sum() / both.sum()):.5f}')
say(f'  universe stock-days with export close but no store close: {int((U & np.isfinite(ce) & ~np.isfinite(cs))[:, d0:].sum())}')
say(f'  universe stock-days with store close but no export close: {int((U & ~np.isfinite(ce) & np.isfinite(cs))[:, d0:].sum())}')
# daily log moves beyond 0.5 in each source, universe stock-days
for nm, c in (('export', ce), ('store', cs)):
    lr = C.log_returns(c)
    big_move = U & (np.abs(np.nan_to_num(lr)) > C.MAX_LOG)
    big_move[:, :d0] = False
    say(f'  {nm}: universe stock-days with |daily log move| > 0.5: {int(big_move.sum())}')
# relative disagreement by year
yrs = np.array([int(c2eval.iso(d)[:4]) for d in days])
row = []
for y in range(2005, yrs.max() + 1):
    m = yrs == y
    b = both[:, m]
    if b.sum():
        row.append(f'{y}: {float((b & (rel[:, m] > 0.005)).sum() / b.sum()):.4f}')
say('  share differing by > 0.5% by year:', ', '.join(row))
n_u = uni.sum(1)
n_b = big.sum(1)
ks = np.nonzero(n_u)[0]
say(f'universe: first block {c2eval.iso(days[tw["rebal"][ks[0]]])}; size median {int(np.median(n_u[ks]))}, '
    f'min {int(n_u[ks].min())}, max {int(n_u[ks].max())}; 500 largest median {int(np.median(n_b[ks]))}')
for k in ks[::24]:
    say(f'  {c2eval.iso(days[tw["rebal"][k]])}: universe {int(n_u[k])}, distinct twins {len(set(tw["fund_row"][k, uni[k]].tolist()))}')
(C.OUT / f'dataqc_{tag}.txt').write_text('\n'.join(LINES) + '\n')
