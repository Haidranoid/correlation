"""Full-history dividend-correction panels for the exports' stocks and for the store's funds.

Why: the store's closes carry no dividend adjustment before about February 2011 as well as after 2022 (found
2026-10-25: 24 large payers and 23 funds against Yahoo matched Yahoo's raw closes on every ex-date in 2005-2010,
its adjusted closes in 2011-2022, and its raw closes again from 2023, with exceptions). div_panel.py covers only
ex-dates from 2022-07-01; this covers every ex-date the exports hold.

Same rules as div_panel.py, imported unchanged (MIN_GAP, MAX_REL, ABS_TOL, BIG_Y, FRAME_TOL; 'raw', 'adj',
'raw_inferred', 'raw_suspect', 'unclassified', 'missing'; the value log(1 + D / C_d) on raw events, Yahoo's own yield
on a split-frame mismatch). Two panels per export tag:
  stocks: [N, T] aligned to the export's rows (tickers out/tickers_full_stocks_map.tsv),
  funds:  [F, T] aligned to c2eval.funds(E) rows, the store's unrepaired fund closes (out/tickers_full_funds_map.tsv).

    python3 -I -B div_panel_full.py
      -> DATA/div_fix_full_{old,new}.npy, DATA/div_events_full_{old,new}.csv,
         DATA/fund_div_fix_{old,new}.npy, DATA/fund_div_events_{old,new}.csv, out/div_full_classification.txt
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402
import numpy as np  # noqa: E402

from div_panel import ABS_TOL, BIG_Y, DATA, EXPORTS, FRAME_TOL, MAX_REL, MIN_GAP, OUT, dnum  # noqa: E402

RAW = {'stocks': DATA / 'raw' / 'yahoo_full_stocks', 'funds': DATA / 'raw' / 'yahoo_full_funds'}
MAPS = {'stocks': OUT / 'tickers_full_stocks_map.tsv', 'funds': OUT / 'tickers_full_funds_map.tsv'}
NAMES = {'stocks': ('div_fix_full', 'div_events_full'), 'funds': ('fund_div_fix', 'fund_div_events')}
LINES = []
COLS = ['ticker', 'row', 'date', 'col', 'dividend', 'yield', 'fix', 'r_store', 'r_raw', 'r_adj', 'dist_raw', 'dist_adj',
        'class', 'reason', 'frame_fallback']


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def load_yahoo(path):
    acc = defaultdict(list)
    for f in sorted(path.glob('batch_*.csv')):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                acc[r['sym']].append((r['date'], r['close'], r['adj_close'], r['dividends'], r['splits']))
    Y = {}
    for s, rows in acc.items():
        rows.sort()
        a = np.array(rows, dtype=object)
        fl = lambda c: np.array([float(x) if x not in ('', None) else np.nan for x in a[:, c]])  # noqa: E731
        Y[s] = {'day': np.array([dnum(x) for x in a[:, 0]]), 'close': fl(1), 'adj': fl(2), 'div': fl(3)}
    return Y


def read_map(path):
    m = {}
    for l in list(open(path))[1:]:
        f = l.rstrip('\n').split('\t')
        m[f[0]] = f[1]
    return m


def classify(close, days, tickers, Y, ymap):
    """div_panel.classify's per-event rules over every ex-date the closes cover. Returns (fix [N, T], events)."""
    N, T = close.shape
    fix = np.zeros((N, T), np.float32)
    ev = []
    for j in range(N):
        t = str(tickers[j])
        if t not in ymap or ymap[t] not in Y:
            continue
        y = Y[ymap[t]]
        c = np.asarray(close[j], np.float64)
        for k in np.nonzero((y['div'] > 0) & (y['day'] >= int(days[0])) & (y['day'] <= int(days[-1])))[0]:
            d = int(y['day'][k])
            D = float(y['div'][k])
            rec = {'ticker': t, 'row': j, 'date': c2eval.iso(d), 'dividend': D, 'col': -1, 'yield': np.nan, 'fix': 0.0,
                   'r_store': np.nan, 'r_raw': np.nan, 'r_adj': np.nan, 'dist_raw': np.nan, 'dist_adj': np.nan,
                   'class': 'missing', 'reason': '', 'frame_fallback': 0}
            ev.append(rec)
            p = int(np.searchsorted(days, d))
            if p >= T or days[p] != d or p == 0:
                rec['reason'] = 'ex-date not an export day'
                continue
            rec['col'] = p
            if k == 0 or y['day'][k - 1] != days[p - 1]:
                rec['reason'] = 'Yahoo day before ex-date is not the export day before'
                continue
            cp, cd = c[p - 1], c[p]
            if not (np.isfinite(cp) and np.isfinite(cd) and cp > 0 and cd > 0):
                rec['reason'] = 'export close missing at d or d-1'
                continue
            yc0, yc1, ya0, ya1 = y['close'][k - 1], y['close'][k], y['adj'][k - 1], y['adj'][k]
            if not all(np.isfinite([yc0, yc1, ya0, ya1])) or min(yc0, yc1, ya0, ya1) <= 0:
                rec['reason'] = 'Yahoo close missing'
                continue
            rs, rr, ra = np.log(cd / cp), np.log(yc1 / yc0), np.log(ya1 / ya0)
            g = ra - rr
            dr, da = abs(rs - rr), abs(rs - ra)
            if abs(np.log(cp / yc0)) > FRAME_TOL:
                yld, yfx, rec['frame_fallback'] = D / yc0, D / yc1, 1
            else:
                yld, yfx = D / cp, D / cd
            rec.update({'yield': yld, 'fix': float(np.log1p(yfx)), 'r_store': rs, 'r_raw': rr, 'r_adj': ra,
                        'dist_raw': dr, 'dist_adj': da})
            if g < MIN_GAP:
                rec['class'], rec['reason'] = 'unclassified', 'gap too small'
            elif min(dr, da) > min(MAX_REL * g, ABS_TOL):
                rec['class'], rec['reason'] = 'unclassified', 'store matches neither'
            else:
                rec['class'] = 'raw' if dr < da else 'adj'
    by_row = defaultdict(list)
    for i, r in enumerate(ev):
        by_row[r['row']].append(i)
    for idx in by_row.values():
        idx.sort(key=lambda i: ev[i]['date'])
        cls = [ev[i]['class'] for i in idx]
        for a, i in enumerate(idx):
            if cls[a] != 'unclassified':
                continue
            before = next((cls[b] for b in range(a - 1, -1, -1) if cls[b] in ('raw', 'adj')), None)
            after = next((cls[b] for b in range(a + 1, len(idx)) if cls[b] in ('raw', 'adj')), None)
            if before == 'raw' and after in ('raw', None):
                ev[i]['class'] = 'raw_inferred'
    for r in ev:
        if r['class'] in ('raw', 'raw_inferred') and r['yield'] > BIG_Y and r['r_store'] > -0.5 * np.log1p(r['yield']):
            r['class'], r['reason'] = 'raw_suspect', 'big yield not visible in the export return'
    for r in ev:
        if r['class'] in ('raw', 'raw_inferred'):
            fix[r['row'], r['col']] += np.float32(r['fix'])
    return fix, ev


def report(tag, kind, ev, fix, n_rows, n_fetched):
    say(f'\n=== {tag} {kind}: rows {n_rows}, with Yahoo data {n_fetched}')
    cl = np.array([r['class'] for r in ev])
    say(f'events {len(ev)}: ' + ', '.join(f'{c} {int((cl == c).sum())}' for c in
                                         ('raw', 'adj', 'raw_inferred', 'raw_suspect', 'unclassified', 'missing')))
    reasons = defaultdict(int)
    for r in ev:
        if r['reason']:
            reasons[r['reason']] += 1
    say('reasons: ' + '; '.join(f'{k} {v}' for k, v in sorted(reasons.items(), key=lambda x: -x[1])))
    q = np.array([min(r['dist_raw'], r['dist_adj']) / (r['r_adj'] - r['r_raw']) for r in ev if r['class'] in ('raw', 'adj')])
    if len(q):
        say(f'classified: nearer distance / gap quantiles 50% {np.quantile(q, .5):.3f} 90% {np.quantile(q, .9):.3f} '
            f'99% {np.quantile(q, .99):.3f}')
    sus = [r for r in ev if r['class'] == 'raw_suspect']
    say(f'raw_suspect (not corrected): {len(sus)} ' + ' '.join(f'{r["ticker"]}@{r["date"]}:{r["yield"]:.3f}' for r in sus[:40]))
    big = [r for r in ev if r['class'] in ('raw', 'raw_inferred') and r['yield'] > 0.05]
    say(f'corrected events with yield > 5%: {len(big)} ' + ' '.join(f'{r["ticker"]}@{r["date"]}:{r["yield"]:.3f}' for r in big[:40]))
    say(f'panel: nonzero cells {int((fix != 0).sum())}, sum of corrections {float(fix.sum(dtype=np.float64)):.3f}')
    say('year   events  raw% of classified   adj    raw  inferred suspect unclass missing   mean yearly fix per raw-paying row (bp)')
    yr = defaultdict(lambda: defaultdict(int))
    fx = defaultdict(lambda: defaultdict(float))
    for r in ev:
        yr[r['date'][:4]][r['class']] += 1
        if r['class'] in ('raw', 'raw_inferred'):
            fx[r['date'][:4]][r['row']] += r['fix']
    for y in sorted(yr):
        z = yr[y]
        n = sum(z.values())
        nc = z['raw'] + z['adj']
        per = 1e4 * np.mean(list(fx[y].values())) if fx[y] else 0.0
        say(f'{y}  {n:6d}  {100 * z["raw"] / nc if nc else float("nan"):6.1f}  {z["adj"]:6d} {z["raw"]:6d} '
            f'{z["raw_inferred"]:6d} {z["raw_suspect"]:6d} {z["unclassified"]:6d} {z["missing"]:6d}   {per:7.1f}')


def write(tag, kind, fix, ev):
    fname, ename = NAMES[kind]
    np.save(DATA / f'{fname}_{tag}.npy', fix)
    with open(DATA / f'{ename}_{tag}.csv', 'w') as fh:
        fh.write(','.join(COLS) + '\n')
        for r in ev:
            fh.write(','.join((f'{r[c]:.6g}' if isinstance(r[c], float) else str(r[c])) for c in COLS) + '\n')


def check_total_return(tag, kind, close, days, tickers, fix, Y, ymap, names):
    """(store log return + fix) minus Yahoo's adjusted-close log return, bp a year, by calendar year; without the fix
    in brackets. Days where both have a return and the calendars agree."""
    tk = [str(t) for t in tickers]
    say(f'check {tag} {kind}: (store + fix) - Yahoo adjusted, bp a year [store alone]')
    for nm in names:
        if nm not in tk or ymap.get(nm) not in Y:
            say(f'  {nm}: not available')
            continue
        j = tk.index(nm)
        y = Y[ymap[nm]]
        c = np.asarray(close[j], np.float64)
        pos = np.searchsorted(days, y['day'])
        ok = (pos < len(days)) & (days[np.minimum(pos, len(days) - 1)] == y['day'])
        yi = np.nonzero(ok)[0]
        p = pos[yi]
        cons = (p[1:] == p[:-1] + 1)
        a, b = yi[:-1][cons], yi[1:][cons]
        pa, pb = p[:-1][cons], p[1:][cons]
        with np.errstate(all='ignore'):
            rs = np.log(c[pb] / c[pa])
            ra = np.log(y['adj'][b] / y['adj'][a])
        f = fix[j, pb].astype(np.float64)
        good = np.isfinite(rs) & np.isfinite(ra) & (np.abs(rs - ra) < 0.2)
        yrs = np.array([c2eval.iso(int(days[q]))[:4] for q in pb])
        parts = []
        for yy in sorted(set(yrs[good])):
            m = good & (yrs == yy)
            parts.append(f'{yy[2:]}:{1e4 * (rs[m] + f[m] - ra[m]).sum():+.0f}[{1e4 * (rs[m] - ra[m]).sum():+.0f}]')
        say(f'  {nm}: ' + ' '.join(parts))


def main():
    Ys = {k: load_yahoo(p) for k, p in RAW.items()}
    maps = {k: read_map(p) for k, p in MAPS.items()}
    for k in Ys:
        say(f'Yahoo {k}: symbols with data {len(Ys[k])} of {len(set(maps[k].values()))} requested')
    for tag, path in EXPORTS.items():
        E = c2eval.load_export(path, arrays=('close',))
        days = E['days']
        fix, ev = classify(E['close'], days, E['tickers'], Ys['stocks'], maps['stocks'])
        write(tag, 'stocks', fix, ev)
        nf = sum(1 for t in E['tickers'] if maps['stocks'].get(str(t)) in Ys['stocks'])
        report(tag, 'stocks', ev, fix, len(E['tickers']), nf)
        check_total_return(tag, 'stocks', E['close'], days, E['tickers'], fix, Ys['stocks'], maps['stocks'],
                           ('XOM', 'JNJ', 'KO', 'T', 'MO', 'VZ', 'PG', 'AAPL'))
        # agreement with the existing post-2022-07 panel on its own events
        old = np.load(DATA / f'div_fix_{tag}.npy', mmap_mode='r')
        c0 = int(np.searchsorted(days, dnum('2022-07-01')))
        a, b = np.asarray(old[:, c0:], np.float64), fix[:, c0:].astype(np.float64)
        both = (a != 0) | (b != 0)
        say(f'agreement with div_fix_{tag} from 2022-07-01: cells either nonzero {int(both.sum())}, equal '
            f'{int((np.abs(a - b) < 1e-7)[both].sum())}, only in full {int(((a == 0) & (b != 0)).sum())}, '
            f'only in div_fix {int(((a != 0) & (b == 0)).sum())}')
        del old, a, b
        F = c2eval.funds(E)
        ffix, fev = classify(F['close'], days, F['tickers'], Ys['funds'], maps['funds'])
        write(tag, 'funds', ffix, fev)
        nf = sum(1 for t in F['tickers'] if maps['funds'].get(str(t)) in Ys['funds'])
        report(tag, 'funds', fev, ffix, len(F['tickers']), nf)
        check_total_return(tag, 'funds', F['close'], days, F['tickers'], ffix, Ys['funds'], maps['funds'],
                           ('SPY', 'QQQ', 'IWM', 'XLF', 'XLU', 'TLT'))
        del E, F
    (OUT / 'div_full_classification.txt').write_text('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
