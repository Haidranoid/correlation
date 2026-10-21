"""Optional check: did v1's own saved relationships (MSFT -> AAPL, read read-only from its Postgres into
v1_saved_relationships.tsv) hold in later data? Each saved (w, k) is recomputed on AAPL changes whose whole window lies
after 2025-06-27 (the 2026-10-19 store), paired with MSFT's change exactly k calendar days earlier. Base rate: every
offset 0..14,300 at the same windows, recomputed the same way, against its full-history sign. Writes check_saved.txt."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np

import v1lib
import c2eval

CUT = int((np.datetime64('2025-06-27') - np.datetime64('1970-01-01')).astype(int))


def series(st, tk):
    i = max((i for i, s in enumerate(st['syms']) if s['ticker'] == tk), key=lambda i: st['syms'][i]['rows'])
    a, b = st['offs'][i], st['offs'][i + 1]
    d, c = np.asarray(st['date'][a:b]).astype(np.int64), np.asarray(st['close'][a:b], dtype=np.float64)
    ok = np.isfinite(c) & (c > 0)
    return d[ok], c[ok]


def cal_changes(d, c, w, lo, hi):
    """Calendar-indexed array [lo, hi] of w-row % changes (NaN on non-trading days)."""
    out = np.full(hi - lo + 1, np.nan)
    ch = (c[w:] - c[:-w]) / c[:-w] * 100.0
    out[d[w:] - lo] = ch
    return out


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 2:
        return np.nan, n
    a, b = x[m] - x[m].mean(), y[m] - y[m].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return (float((a * b).sum() / den) if den > 0 else np.nan), n


def main():
    rows = [r for r in csv.DictReader(open(HERE / 'v1_saved_relationships.tsv'), delimiter='\t') if r['symbol1'] == 'MSFT']
    st = c2eval.load_store(c2eval.STORE_NEW)
    dA, cA = series(st, 'MSFT')
    dB, cB = series(st, 'AAPL')
    lo, hi = int(min(dA[0], dB[0])), int(max(dA[-1], dB[-1]))
    lines = [f'MSFT {c2eval.iso(dA[0])}..{c2eval.iso(dA[-1])}, AAPL {c2eval.iso(dB[0])}..{c2eval.iso(dB[-1])} (store 2026-10-19)']
    windows = sorted({int(r['window1_days']) for r in rows})
    agree_saved, agree_base, sg_saved, n_saved, out_rows = [], [], [], [], []
    local_base, clusters = [], []
    for w in windows:
        xa = cal_changes(dA, cA, w, lo, hi)
        xb = cal_changes(dB, cB, w, lo, hi)
        # B's change must lie wholly after the cut: its window starts at or after the last close <= CUT
        first_b = dB[np.searchsorted(dB, CUT, side='right') - 1 + w]
        later_b = np.arange(lo, hi + 1) >= first_b
        old_b = np.arange(lo, hi + 1) <= CUT
        saved = {int(r['offset_days']): float(r['correlation_coefficient']) for r in rows if int(r['window1_days']) == w}
        kmax = max(saved) if saved else 0
        kmax = max(kmax, 14300)
        base_same, base_n = 0, 0
        same_k = np.full(kmax + 1, np.nan)
        for k in range(0, kmax + 1):
            if k >= len(xa):
                break
            y = xb[k:]
            x = xa[:len(y)]
            r_full, n_full = corr(x, np.where(old_b[k:], y, np.nan))          # v1's own data (through 2025-06-27)
            r_new, n_new = corr(x, np.where(later_b[k:], y, np.nan))
            if not (np.isfinite(r_full) and np.isfinite(r_new)) or n_new < v1lib.N_MIN or n_full < v1lib.N_MIN:
                continue
            same = np.sign(r_full) == np.sign(r_new)
            same_k[k] = same
            base_same += same
            base_n += 1
            if k in saved:
                agree_saved.append(same)
                sg_saved.append(np.sign(saved[k]) * r_new)
                n_saved.append(n_new)
                out_rows.append((w, k, saved[k], r_full, r_new, n_new))
        for o in out_rows:
            if o[0] == w:
                loc = same_k[max(0, o[1] - 100):o[1] + 101]
                local_base.append(float(np.nanmean(loc)))
        ks = sorted(saved)
        clusters.append(1 + sum(1 for a_, b_ in zip(ks, ks[1:]) if b_ - a_ > 50) if ks else 0)
        nsw = sum(1 for o in out_rows if o[0] == w)
        sw = sum(np.sign(o[2]) == np.sign(o[4]) for o in out_rows if o[0] == w)
        lines.append(f'w={w}: {len(saved)} saved rows, {nsw} checkable; same sign later {sw}/{nsw} = {sw / max(1, nsw):.2f}; '
                     f'all offsets at this window {base_same}/{base_n} = {base_same / max(1, base_n):.2f}')
        agree_base.append((base_same, base_n))
    s = np.array(agree_saved)
    bs = sum(a for a, _ in agree_base) / max(1, sum(b for _, b in agree_base))
    lines.append(f'ALL saved MSFT->AAPL rows checkable: {len(s)}; same sign later {s.mean():.3f} (base, every offset at the same windows: {bs:.3f}); '
                 f'mean sign(saved r) x later r {np.mean(sg_saved):+.3f}; later points per row median {int(np.median(n_saved))} '
                 f'(min {min(n_saved)}, max {max(n_saved)})')
    lines.append(f'k-matched base (all offsets within +-100 days of each saved one, same window): {np.mean(local_base):.3f}; '
                 f'the saved rows fall in {sum(clusters)} clusters of nearby offsets (gaps > 50 days) across the {len(windows)} windows')
    full_diff = np.array([abs(o[2] - o[3]) for o in out_rows])
    lines.append(f'(full-history r on the 2026 store vs saved r: median |diff| {np.median(full_diff):.3f}; the 2026 store is re-adjusted for later dividends)')
    for band, lo_k, hi_k in (('0', 0, 0), ('1-30', 1, 30), ('31-365', 31, 365), ('>365', 366, 10 ** 9)):
        sel = [o for o in out_rows if lo_k <= o[1] <= hi_k]
        if sel:
            lines.append(f'  offsets {band}: {len(sel)} rows, same sign {np.mean([np.sign(o[2]) == np.sign(o[4]) for o in sel]):.2f}, '
                         f'mean saved r {np.mean([o[2] for o in sel]):+.2f}, mean later r {np.mean([o[4] for o in sel]):+.2f}')
    top = sorted(out_rows, key=lambda o: -abs(o[2]))[:5]
    for o in top:
        lines.append(f'  strongest saved: w={o[0]} k={o[1]} saved r {o[2]:+.3f} -> later r {o[4]:+.3f} on {o[5]} points')
    (HERE / 'check_saved.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
