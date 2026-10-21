"""Post hoc diagnostics (written after seeing Tests 1-3): is the fade a property of the open prints?

1. Stale opens in the legs: the share of leg positions whose open equals the previous close (gap exactly 0) or the
   close (intraday exactly 0), and the book without them.
2. The return profile across the whole ranked set: mean hedged intraday return by z bin.
3. How much of the selected gap comes back: the legs' overnight, intraday and close-to-close hedged returns.
4. The most liquid funds: does SPY's (or QQQ's, XLF's ...) own open gap predict its own intraday return?
   In clean data the slope of intraday on overnight is near 0; noisy opens push it toward -1.
5. The same fade on the funds themselves: rank twin funds by their gap against SPY's.

    python3 -I diag.py      # writes out/diag_old.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import books as B  # noqa: E402
from books import c2eval, np  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def fmt(x):
    m, t, n = B.tstat(x)
    return f'{m * 1e4:+.2f} ({t:+.2f})'


R = B.load_rows('old')
days = R['days']
T1 = ('2008-01-01', '2020-01-01')
T3 = ('2020-01-01', None)

# 1. stale opens
stale_on = R['g'] == 0
stale_id = R['rs'] == 0
say(f'ranked set: open == previous close {stale_on.mean():.2%}, open == close {stale_id.mean():.2%}')
Bk = B.books(R, 'all')
# which rows are in the twin legs? recompute membership
u, st, en = B.day_slices(R['t'][~R['idbad']])
keep = ~R['idbad']
z = R['z'][keep]; g = R['g'][keep]; rs = R['rs'][keep]; in5 = R['in500'][keep]
inleg = np.zeros(keep.sum(), bool)
for a, b in zip(st, en):
    n = b - a
    if n < B.MIN_N:
        continue
    nl = int(round(0.1 * n))
    o = np.argsort(z[a:b], kind='stable')
    inleg[a + o[:nl]] = True
    inleg[a + o[n - nl:]] = True
say(f'twin legs: open == previous close {(g[inleg] == 0).mean():.2%}, open == close {(rs[inleg] == 0).mean():.2%}; '
    f'500 largest legs n/a (ranked separately)')
for name, mask in (('without open == previous close', ~stale_on), ('without open == close', ~stale_id),
                   ('without either', ~stale_on & ~stale_id)):
    Rm = {k: (v[mask] if k != 'days' else v) for k, v in R.items()}
    for sub in ('all', '500'):
        Bm = B.books(Rm, sub)
        line = []
        for pn, (a, b) in (('2008-2019', T1), ('2020-2025-06', T3)):
            m = B.period_mask(days, Bm['t'], a, b)
            line.append(f'{pn}: gross {fmt(Bm["twin_gross"][m])}, head {fmt(B.net(Bm, "twin")[m])}, raw head {fmt(B.net(Bm, "raw")[m])}')
        say(f'{name}, {sub}: ' + '; '.join(line))

# 2. profile across z (pooled over 2008-2019, by within-day percentile of z)
h = R['rs'] - R['beta'].astype(np.float64) * R['rf']
pct = np.zeros(len(R['t']))
uu, st2, en2 = B.day_slices(R['t'])
for a, b in zip(st2, en2):
    o = np.argsort(np.argsort(R['z'][a:b], kind='stable'), kind='stable')
    pct[a:b] = (o + 0.5) / (b - a)
m1 = B.period_mask(days, R['t'], *T1)
say('\nmean hedged intraday return (bps) by within-day percentile of z, 2008-2019 (bin: all stocks | 500 largest | '
    'median z | median raw gap bps):')
for k in range(20):
    sel = m1 & (pct >= k / 20) & (pct < (k + 1) / 20)
    s5 = sel & R['in500']
    say(f'  {k * 5:>2}-{k * 5 + 5:<3}%: {h[sel].mean() * 1e4:+7.2f} | {h[s5].mean() * 1e4:+7.2f} | {np.median(R["z"][sel]):+.2f} | '
        f'{np.median(R["g"][sel]) * 1e4:+.1f}')
for lo, hi in ((0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.90, 0.95), (0.95, 0.99), (0.99, 1.0)):
    sel = m1 & (pct >= lo) & (pct < hi)
    say(f'  tail {lo:.2f}-{hi:.2f}: {h[sel].mean() * 1e4:+8.2f} bps, median z {np.median(R["z"][sel]):+.2f}, n {sel.sum()}')

# 3. overnight vs intraday of the legs (hedged with the twin's own legs), 2008-2019
E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close', 'dollar_volume'))
tw = c2eval.twins(E)
F = tw.pop('F')
S = c2eval.load_store(c2eval.STORE_OLD)
say('\nlegs of the twin book, 2008-2019: mean raw gap g (bps) long / short:',
    f'{np.mean(g[inleg & (z < 0) & m1[keep]]) * 1e4:+.1f} / {np.mean(g[inleg & (z > 0) & m1[keep]]) * 1e4:+.1f}')

# 4. the most liquid funds: slope of intraday on overnight, and the share of days the open equals the previous close
say('\nfunds: OLS slope of intraday log return on overnight log return, by period (clean data: near 0; open noise: '
    'toward -1); share of days with open == previous close')
tick = list(tw['fund_ticker'])
ids = [tw['fund'][tick.index(x)] for x in ('SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLU', 'GLD', 'TLT', 'EEM')]
fo = c2eval.store_panel(S, ids, days, 'open').astype(np.float64)
fc = c2eval.store_panel(S, ids, days, 'close').astype(np.float64)
with np.errstate(all='ignore'):
    on = np.full(fo.shape, np.nan); on[:, 1:] = np.log(fo[:, 1:] / fc[:, :-1])
    idr = np.log(fc / fo)
for j, x in enumerate(('SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLU', 'GLD', 'TLT', 'EEM')):
    parts = []
    for pn, a, b in (('2008-19', '2008-01-01', '2020-01-01'), ('2020-25', '2020-01-01', None)):
        lo = c2eval.day_index(days, a); hi = c2eval.day_index(days, b) if b else len(days)
        o_, i_ = on[j, lo:hi], idr[j, lo:hi]
        ok = np.isfinite(o_) & np.isfinite(i_)
        sl = np.polyfit(o_[ok], i_[ok], 1)[0]
        cr = np.corrcoef(o_[ok], i_[ok])[0, 1]
        st_ = np.mean(fo[j, lo:hi][ok] == fc[j, lo - 1:hi - 1][ok])
        parts.append(f'{pn}: slope {sl:+.2f}, corr {cr:+.2f}, open==prev close {st_:.1%}')
    say(f'  {x}: ' + '; '.join(parts))

# 5. the large caps themselves: slope for the 20 most-traded stocks at the end of 2019
dv = np.asarray(E['dollar_volume'][:, c2eval.day_index(days, '2019-01-01'):c2eval.day_index(days, '2020-01-01')])
with np.errstate(all='ignore'):
    import warnings
    warnings.simplefilter('ignore')
    md = np.nan_to_num(np.nanmedian(dv, 1))
top = np.argsort(-md)[:20]
so = c2eval.store_panel(S, E['ids'][top], days, 'open').astype(np.float64)
sc = c2eval.store_panel(S, E['ids'][top], days, 'close').astype(np.float64)
with np.errstate(all='ignore'):
    on = np.full(so.shape, np.nan); on[:, 1:] = np.log(so[:, 1:] / sc[:, :-1])
    idr = np.log(sc / so)
say('\n20 most-traded stocks of 2019: slope of intraday on overnight, 2008-2019 / 2020-2025, and open==prev close share')
for j, i in enumerate(top):
    parts = []
    for a, b in (('2008-01-01', '2020-01-01'), ('2020-01-01', None)):
        lo = c2eval.day_index(days, a); hi = c2eval.day_index(days, b) if b else len(days)
        o_, i_ = on[j, lo:hi], idr[j, lo:hi]
        ok = np.isfinite(o_) & np.isfinite(i_) & (np.abs(o_) < .5) & (np.abs(i_) < .5)
        parts.append(f'{np.polyfit(o_[ok], i_[ok], 1)[0]:+.2f} (corr {np.corrcoef(o_[ok], i_[ok])[0, 1]:+.2f}, '
                     f'stale {np.mean(so[j, lo:hi][ok] == sc[j, lo - 1:hi - 1][ok]):.1%})')
    say(f'  {E["tickers"][i]}: ' + ' / '.join(parts))

# 6. by listing exchange: NYSE stocks' opening auctions often printed minutes after 9:30 before ~2016, while
# Nasdaq's opening cross and the funds open at 9:30. A market move in between sits in u and comes back in h.
exch = {sy['id']: sy['exchange'] for sy in S['syms']}
ex_row = np.array([exch[int(i)] for i in E['ids']])
say('\nby listing exchange (the book ranked within each exchange), bps/day (t):')
for ex in ('nyse', 'nasdaq', 'nysemkt'):
    mask = ex_row[R['i']] == ex
    Rm = {k: (v[mask] if k != 'days' else v) for k, v in R.items()}
    for sub in ('all', '500'):
        Bm = B.books(Rm, sub)
        if len(Bm['t']) == 0:
            continue
        parts = []
        for pn, a, b in (('2008-09', '2008-01-01', '2010-01-01'), ('2010-14', '2010-01-01', '2015-01-01'),
                         ('2015-19', '2015-01-01', '2020-01-01'), ('2020-25', '2020-01-01', None)):
            m = B.period_mask(days, Bm['t'], a, b)
            parts.append(f'{pn} gross {fmt(Bm["twin_gross"][m])} raw {fmt(Bm["raw_gross"][m])} (ranked {np.median(Bm["n"][m]):.0f})')
        say(f'  {ex} {sub}: ' + '; '.join(parts))
(B.OUT / 'diag_old.txt').write_text('\n'.join(LINES) + '\n')
