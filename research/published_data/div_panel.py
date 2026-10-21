"""Dividend-correction panel for the exports' stocks (see pubdata.div_fix).

The store's closes (and so the export's) are back-adjusted for dividends until about Oct-Nov 2022 for most payers,
and raw after that, so from late 2022 a stock's return on an ex-dividend date includes the dividend drop. For every
Yahoo ex-dividend event (stock, ex-date d in 2022-07-01..2026-10-19) of the tickers in out/tickers_map.tsv:

  r_s   = log(C_d / C_{d-1}) from the export's closes (store closes after the export's bad-print repair)
  r_raw = Yahoo's Close return over the same two days (split-adjusted, includes the dividend drop)
  r_adj = Yahoo's Adj Close return (dividend-adjusted)
  g     = r_adj - r_raw (about D / C_{d-1}); dist_raw = |r_s - r_raw|, dist_adj = |r_s - r_adj|

Class (rules fixed before looking at the classification counts):
  'raw'  dist_raw < dist_adj, 'adj' otherwise, when g >= MIN_GAP and min(dist) <= min(MAX_REL * g, ABS_TOL);
  'unclassified' otherwise (dividend too small to tell, or the store matches neither), with a reason;
  'missing' when an input is missing (no export day for d, NaN close at d or d-1, Yahoo's day before d is not the
  export's day before d).
  'raw_inferred': an unclassified event whose stock's nearest classified event before is 'raw' and nearest after
  is 'raw' or absent.
  'raw_suspect': a raw / raw_inferred event with D / C_{d-1} > BIG_Y whose export return on d does not show at least
  half the drop (r_s > -0.5 * log(1 + D / C_{d-1})): Yahoo's dividend is then not visible in the prices (e.g. NFE's
  quarterly "5" dividends, 13-46% of the price). Not corrected.
ABS_TOL and the raw_suspect rule were added after the first pass (which showed CIB 2025-03-31 classified with an 8.9%
distance, and NFE's bogus dividends); the rest was fixed before looking.
Panel value on 'raw' and 'raw_inferred' events: log(1 + D / C_d), C_d the export's close on the ex-date (raw then),
so that log(C_d / C_{d-1}) + value = log((C_d + D) / C_{d-1}), the exact total return (CRSP's definition). The brief
said log(1 + D / C_{d-1}); that understates the total return by about (D / C)^2 per event (MO: -16 bp a year against
Yahoo's adjusted closes), so C_d is used instead (decided after the first check run). If the export's close differs
from Yahoo's raw close by more than 2% (split-frame mismatch), Yahoo's own D / Close is used and the event counted.
0 elsewhere, and 0 for every row whose ticker was not fetched.

    python3 -I -B div_panel.py       # -> DATA/div_fix_{old,new}.npy, DATA/div_events_{old,new}.csv, out/div_*.txt
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

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
DATA = c2eval.WORK / 'published_data'
RAW = DATA / 'raw' / 'yahoo_stocks'
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
EV_FROM, EV_TO = '2022-07-01', '2026-10-19'
MIN_GAP = 1e-4      # |r_adj - r_raw| below this: dividend too small to tell the two apart
MAX_REL = 0.5       # the nearer of the two must be within this fraction of the gap
ABS_TOL = 0.005     # ... and within this absolute log distance
BIG_Y = 0.10        # yields above this need the export's own return to show at least half the drop
FRAME_TOL = 0.02    # |log(C_{d-1} / Yahoo Close_{d-1})| above this: use Yahoo's own yield
EPOCH = np.datetime64('1970-01-01')
LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def dnum(s):
    return int((np.datetime64(s) - EPOCH).astype(int))


def load_yahoo():
    """sym -> dict of arrays: day (days since 1970), close, adj, div, split."""
    acc = defaultdict(list)
    for f in sorted(RAW.glob('batch_*.csv')):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                acc[r['sym']].append((r['date'], r['close'], r['adj_close'], r['dividends'], r['splits']))
    Y = {}
    for s, rows in acc.items():
        rows.sort()
        a = np.array(rows, dtype=object)
        fl = lambda c: np.array([float(x) if x not in ('', None) else np.nan for x in a[:, c]])  # noqa: E731
        Y[s] = {'day': np.array([dnum(x) for x in a[:, 0]]), 'close': fl(1), 'adj': fl(2), 'div': fl(3), 'split': fl(4)}
    return Y


def classify(tag, Y, ymap):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
    days, close = E['days'], E['close']
    N, T = close.shape
    lo, hi = dnum(EV_FROM), min(dnum(EV_TO), int(days[-1]))
    fix = np.zeros((N, T), np.float32)
    ev = []
    frame_med = []
    rows_fetched = rows_nodata = 0
    for j in range(N):
        t = str(E['tickers'][j])
        if t not in ymap:
            continue
        rows_fetched += 1
        y = Y.get(ymap[t])
        if y is None:
            rows_nodata += 1
            continue
        c = np.asarray(close[j], np.float64)
        # premise check: export close vs Yahoo raw close over 2024 (both raw then)
        pos = np.searchsorted(days, y['day'])
        ok = (pos < T) & (days[np.minimum(pos, T - 1)] == y['day'])
        m24 = ok & (y['day'] >= dnum('2024-01-01')) & (y['day'] <= dnum('2024-12-31'))
        if m24.sum() >= 50:
            with np.errstate(all='ignore'):
                lr = np.log(c[pos[m24]] / y['close'][m24])
            lr = lr[np.isfinite(lr)]
            if len(lr):
                frame_med.append(float(np.median(lr)))
        for k in np.nonzero((y['div'] > 0) & (y['day'] >= lo) & (y['day'] <= hi))[0]:
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
            rec.update({'yield': yld, 'fix': float(np.log1p(yfx)), 'r_store': rs, 'r_raw': rr, 'r_adj': ra, 'dist_raw': dr, 'dist_adj': da})
            if g < MIN_GAP:
                rec['class'], rec['reason'] = 'unclassified', 'gap too small'
            elif min(dr, da) > min(MAX_REL * g, ABS_TOL):
                rec['class'], rec['reason'] = 'unclassified', 'store matches neither'
            else:
                rec['class'] = 'raw' if dr < da else 'adj'
    # raw_inferred: unclassified events between two 'raw' events of the same stock (or raw before and none after)
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
    np.save(DATA / f'div_fix_{tag}.npy', fix)
    cols = ['ticker', 'row', 'date', 'col', 'dividend', 'yield', 'fix', 'r_store', 'r_raw', 'r_adj', 'dist_raw', 'dist_adj',
            'class', 'reason', 'frame_fallback']
    with open(DATA / f'div_events_{tag}.csv', 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in ev:
            fh.write(','.join((f'{r[c]:.6g}' if isinstance(r[c], float) else str(r[c])) for c in cols) + '\n')
    # report
    say(f'\n=== {tag}: export {E["path"]}  N {N} T {T}  last day {c2eval.iso(days[-1])}')
    say(f'rows with a fetched ticker {rows_fetched}; of which no Yahoo data {rows_nodata}')
    fm = np.abs(np.array(frame_med))
    say(f'premise check (2024, |median log(export close / Yahoo close)|): stocks {len(fm)}, median {np.median(fm):.5f}, '
        f'> 0.5% {int((fm > 0.005).sum())}, > 2% {int((fm > 0.02).sum())}')
    cl = np.array([r['class'] for r in ev])
    say(f'events {len(ev)}: ' + ', '.join(f'{c} {int((cl == c).sum())}' for c in
                                         ('raw', 'adj', 'raw_inferred', 'raw_suspect', 'unclassified', 'missing')))
    reasons = defaultdict(int)
    for r in ev:
        if r['reason']:
            reasons[r['reason']] += 1
    say('reasons: ' + '; '.join(f'{k} {v}' for k, v in sorted(reasons.items(), key=lambda x: -x[1])))
    q = np.array([min(r['dist_raw'], r['dist_adj']) / (r['r_adj'] - r['r_raw']) for r in ev if r['class'] in ('raw', 'adj')])
    say(f'classified events: nearer distance / gap quantiles 50% {np.quantile(q, .5):.3f} 90% {np.quantile(q, .9):.3f} '
        f'99% {np.quantile(q, .99):.3f}')
    ff = sum(r['frame_fallback'] for r in ev if r['class'] in ('raw', 'raw_inferred'))
    big = [r for r in ev if r['class'] in ('raw', 'raw_inferred') and r['yield'] > 0.05]
    say(f'raw events using Yahoo yield (frame mismatch > 2%): {ff}; corrected events with yield > 5%: {len(big)} '
        + ' '.join(f'{r["ticker"]}@{r["date"]}:{r["yield"]:.3f}' for r in big))
    sus = [r for r in ev if r['class'] == 'raw_suspect']
    say(f'raw_suspect (not corrected): {len(sus)} ' + ' '.join(f'{r["ticker"]}@{r["date"]}:{r["yield"]:.3f}/r={r["r_store"]:+.3f}' for r in sus))
    say(f'panel: nonzero cells {int((fix != 0).sum())}, sum of corrections {float(fix.sum(dtype=np.float64)):.3f}')
    say('month    events  raw%(of classified)  adj  raw  inferred  suspect  unclass  missing')
    mon = defaultdict(lambda: defaultdict(int))
    for r in ev:
        mon[r['date'][:7]][r['class']] += 1
    for m in sorted(mon):
        z = mon[m]
        n = sum(z.values())
        nc = z['raw'] + z['adj']
        say(f'{m}  {n:6d}  {100 * z["raw"] / nc if nc else float("nan"):6.1f}  {z["adj"]:5d} {z["raw"]:5d} '
            f'{z["raw_inferred"]:5d} {z["raw_suspect"]:5d} {z["unclassified"]:5d} {z["missing"]:5d}')
    return ev


def main():
    ymap = {}
    for l in list(open(OUT / 'tickers_map.tsv'))[1:]:
        f = l.rstrip('\n').split('\t')
        ymap[f[0]] = f[1]
    Y = load_yahoo()
    say(f'Yahoo symbols with data: {len(Y)} of {len(set(ymap.values()))} requested')
    for tag in ('old', 'new'):
        classify(tag, Y, ymap)
    (OUT / 'div_classification.txt').write_text('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
