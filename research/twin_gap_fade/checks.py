"""Code checks of the book machinery (run before the tests):
- an oracle signal (minus today's own hedged intraday return) must earn a lot;
- a signal of pure noise must earn about zero gross;
- one day's twin book recomputed by hand from the store must match books().

    python3 -I checks.py      # writes out/checks.txt
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


R = B.load_rows('old')
days = R['days']
h = R['rs'] - R['beta'].astype(np.float64) * R['rf']
# oracle: fade the realized hedged return (so z = h -> long most negative... use -h to be long the best)
Ro = dict(R)
Ro['z'] = (-h).astype(np.float32)
Ro['g'] = np.random.default_rng(1).standard_normal(len(h)).astype(np.float32)
Bo = B.books(Ro, 'all')
m, t, n = B.tstat(Bo['twin_gross'])
say(f'oracle (long the best hedged intraday returns): gross {m * 1e4:+.1f} bps/day, t {t:+.1f}, days {n}')
m, t, n = B.tstat(Bo['raw_gross'])
say(f'noise signal: gross {m * 1e4:+.2f} bps/day, t {t:+.2f}')
# legs are exactly the deciles
say(f'leg size / ranked set: median {np.median(Bo["nl"] / Bo["n"]):.4f}, min n {Bo["n"].min()}, days {len(Bo["n"])}')

# hand check of one day from the raw store
E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
tw = c2eval.twins(E)
S = c2eval.load_store(c2eval.STORE_OLD)
day = c2eval.day_index(days, '2014-03-12')
sel = R['t'] == day
idx = np.nonzero(sel)[0]
ii, ff = R['i'][idx], R['f'][idx]
o = c2eval.store_panel(S, E['ids'][ii], days[day - 1:day + 1], 'open')
c = c2eval.store_panel(S, E['ids'][ii], days[day - 1:day + 1], 'close')
fo = c2eval.store_panel(S, tw['fund'][ff], days[day - 1:day + 1], 'open')
fc = c2eval.store_panel(S, tw['fund'][ff], days[day - 1:day + 1], 'close')
k = c2eval.rebal_index(tw, day - 1)
assert np.all(tw['fund_row'][k, ii] == ff), 'twin mismatch'
beta = tw['beta'][k, ii].astype(np.float64)
assert np.allclose(beta, R['beta'][idx]), 'beta mismatch'
g = np.log(o[:, 1].astype(np.float64) / c[:, 0])
assert np.allclose(g, R['g'][idx], atol=1e-6), 'gap mismatch'
rs = c[:, 1].astype(np.float64) / o[:, 1] - 1
rf = fc[:, 1].astype(np.float64) / fo[:, 1] - 1
hh = rs - beta * rf
z = R['z'][idx]
n = len(idx)
nl = int(round(0.1 * n))
oz = np.argsort(z, kind='stable')
gross = hh[oz[:nl]].mean() - hh[oz[-nl:]].mean()
B1 = B.books(R, 'all')
dpos = int(np.nonzero(B1['t'] == day)[0][0])
say(f'hand check {c2eval.iso(days[day])}: n {n}, legs {nl}; gross by hand {gross * 1e4:+.3f} bps, books() '
    f'{B1["twin_gross"][dpos] * 1e4:+.3f} bps')
assert abs(gross - B1['twin_gross'][dpos]) < 1e-6
# z by hand for one stock: u / sd over the 252 previous days with the current twin and beta
j = int(oz[0])
i0, f0, b0 = ii[j], ff[j], beta[j]
win = days[day - 252:day + 1]
so_ = c2eval.store_panel(S, [E['ids'][i0]], win, 'open')[0].astype(np.float64)
sc_ = c2eval.store_panel(S, [E['ids'][i0]], win, 'close')[0].astype(np.float64)
fo_ = c2eval.store_panel(S, [tw['fund'][f0]], win, 'open')[0].astype(np.float64)
fc_ = c2eval.store_panel(S, [tw['fund'][f0]], win, 'close')[0].astype(np.float64)
on_s = np.log(so_[1:] / sc_[:-1]); id_s = np.log(sc_[1:] / so_[1:])
on_f = np.log(fo_[1:] / fc_[:-1]); id_f = np.log(fc_[1:] / fo_[1:])
bad = (np.abs(on_s) > .5) | (np.abs(id_s) > .5) | (np.abs(on_f) > .5) | (np.abs(id_f) > .5)
u = on_s - b0 * on_f
u[bad] = np.nan
hist = u[:-1]  # days day-251 .. day-1 (the window's first day has no previous close in this slice)
# the window is t-252..t-1; the slice above starts one day late, so add day-252's u from a wider slice
w2 = days[day - 253:day - 251]
a_o = c2eval.store_panel(S, [E['ids'][i0]], w2, 'open')[0].astype(np.float64)
a_c = c2eval.store_panel(S, [E['ids'][i0]], w2, 'close')[0].astype(np.float64)
b_o = c2eval.store_panel(S, [tw['fund'][f0]], w2, 'open')[0].astype(np.float64)
b_c = c2eval.store_panel(S, [tw['fund'][f0]], w2, 'close')[0].astype(np.float64)
u0 = np.log(a_o[1] / a_c[0]) - b0 * np.log(b_o[1] / b_c[0])
bad0 = (abs(np.log(a_o[1] / a_c[0])) > .5) | (abs(np.log(a_c[1] / a_o[1])) > .5) | (abs(np.log(b_o[1] / b_c[0])) > .5) | (abs(np.log(b_c[1] / b_o[1])) > .5)
hist = np.concatenate([[np.nan if bad0 else u0], hist])
sd = np.nanstd(hist, ddof=1)
z_hand = u[-1] / sd
say(f'z by hand for {E["tickers"][i0]} (twin {tw["fund_ticker"][f0]}, beta {b0:.2f}): {z_hand:+.4f}; build.py {z[j]:+.4f}; '
    f'window values {np.isfinite(hist).sum()}')
assert abs(z_hand - z[j]) < 1e-3
(B.OUT / 'checks.txt').write_text('\n'.join(LINES) + '\n')
say('all checks passed')
