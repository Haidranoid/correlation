"""Diagnostics added after results were seen (reported only; no rule depends on them):
  1. the stability checks and B1 on the months with the full 500 stocks only (2005-06 on), since 2002-12..2005-05
     has 314-320 stocks and a smaller edge count, which shifts every series' level;
  2. SPY drawdowns of >= 15% with the peak reset after each trough, which adds the 2010 and 2011 corrections that
     the pre-registered definition folds into the 2007-2012 underwater period.

    python -I -B b_extra.py
Writes extra_old.txt and extra_old.json here.
"""
import json
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/market_geometry')
import numpy as np  # noqa: E402

import b_common as B  # noqa: E402

T_ = B.load_module('b_tests', B.HERE / 'tests.py')
c2eval = B.c2eval


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main():
    M, groups, days, P, rfd, lr = T_.load('old')
    z = np.load(B.WORK / 'curv_old.npz')
    full = z['n'] == 500
    out = {'months_500': int(full.sum()), 'first_500': M['date'][full][0]}
    L = [f"Months with 500 stocks: {out['months_500']} (from {out['first_500']})"]
    for s in ('ollivier', 'forman'):
        a, b = np.where(full, M[f'odd_{s}'], np.nan), np.where(full, M[f'even_{s}'], np.nan)
        lg = np.where(full, M[f'long_{s}'], np.nan)
        re = np.where(full, M[s], np.nan)
        o = {'odd_even': corr(a, b), 'r63_252': corr(re, lg)}
        # odd/even as levels relative to the real series: does each half track the full-window series?
        o['odd_real'] = corr(a, re)
        o['even_real'] = corr(b, re)
        out[s] = o
        L.append(f"{s}, 500-stock months: odd vs even {o['odd_even']:+.3f}; 63 vs 252 days {o['r63_252']:+.3f}; "
                 f"odd vs full window {o['odd_real']:+.3f}, even vs full {o['even_real']:+.3f}")
    Xa = np.column_stack([M['avg_corr'], M['absorption'], M['lrv63']])
    Xb = np.column_stack([M[c] for c in T_.STATS])
    for s in ('ollivier', 'forman'):
        y = np.where(full, M[s], np.nan)
        ra, rb = T_.r2_resid(y, Xa), T_.r2_resid(y, Xb)
        out[f'b1_{s}_500'] = {'yardsticks': ra, 'stats': rb}
        L.append(f"B1 on 500-stock months, {s}: yardsticks R2 {ra['r2']:.3f}, plain statistics R2 {rb['r2']:.3f}")
    # drawdowns with the peak reset after each trough
    d0, d1 = c2eval.day_index(days, '2003-01-01'), c2eval.day_index(days, '2025-06-27')
    p = P[d0:d1 + 1]
    eps = []
    i = 0
    while i < len(p) - 1:
        pk = i
        j = i + 1
        while j < len(p):
            if p[j] > p[pk]:
                pk = j
            elif 1 - p[j] / p[pk] >= 0.15:
                break
            j += 1
        if j >= len(p):
            break
        # in a drawdown from pk: the trough is the low before the price regains 15% off it or a new high
        tr = j
        k = j
        while k < len(p) and p[k] < p[pk] and p[k] < p[tr] * 1.15:
            if p[k] < p[tr]:
                tr = k
            k += 1
        eps.append((d0 + pk, d0 + tr, float(1 - p[tr] / p[pk])))
        i = tr
    t0 = M['t0']
    zs = {s: np.array([B.trailing_z(M[s], k) for k in range(M['K'])]) for s in ('ollivier', 'forman')}
    rows = []
    L.append('\nDrawdowns >= 15% with the peak reset after each trough (a trough ends when SPY rises 15% off it or regains the peak):')
    for pk, trg, depth in eps:
        kp = int(np.searchsorted(t0, pk, side='right') - 1)
        kt = int(np.searchsorted(t0, trg, side='right') - 1)
        row = {'peak': c2eval.iso(days[pk]), 'trough': c2eval.iso(days[trg]), 'depth': depth}
        for s, sign in (('ollivier', +1), ('forman', -1)):
            zz = zs[s]
            first = None
            for k in range(max(0, kp - 12), kt + 1):
                if np.isfinite(zz[k]) and sign * zz[k] > 1:
                    first = k - kp
                    break
            row[s] = {'z_peak_m6': float(zz[kp - 6]), 'z_peak_m3': float(zz[kp - 3]), 'z_peak': float(zz[kp]),
                      'z_trough': float(zz[kt]), 'first_beyond_1_months_from_peak': first}
        rows.append(row)
        o, f = row['ollivier'], row['forman']
        L.append(f"{row['peak']} -> {row['trough']} ({100 * depth:.1f}%): Ollivier z -6m {o['z_peak_m6']:+.2f}, -3m {o['z_peak_m3']:+.2f}, "
                 f"peak {o['z_peak']:+.2f}, trough {o['z_trough']:+.2f}, first > +1 at {o['first_beyond_1_months_from_peak']}; "
                 f"Forman z {f['z_peak_m6']:+.2f}, {f['z_peak_m3']:+.2f}, {f['z_peak']:+.2f}, {f['z_trough']:+.2f}, first < -1 at "
                 f"{f['first_beyond_1_months_from_peak']}")
    out['drawdowns_reset'] = rows
    # base rate: how often the z-score is beyond 1 on the crisis side in any month, and in any 13-month span
    for s_, sign in (('ollivier', +1), ('forman', -1)):
        zz = zs[s_]
        ok = np.isfinite(zz)
        hit = (sign * zz > 1) & ok
        win = [hit[k:k + 13].any() for k in range(M['K'] - 13) if ok[k:k + 13].all()]
        out[f'base_rate_{s_}'] = {'share_months': float(hit[ok].mean()), 'share_13_month_spans_with_one': float(np.mean(win))}
        L.append(f"base rate, {s_}: beyond 1 on the crisis side in {100 * hit[ok].mean():.0f}% of months; at least once in "
                 f"{100 * np.mean(win):.0f}% of all 13-month spans (the span searched from peak - 12 months to the peak)")
    (B.HERE / 'extra_old.json').write_text(json.dumps(out, indent=1))
    txt = '\n'.join(L)
    print(txt)
    (B.HERE / 'extra_old.txt').write_text(txt + '\n')


if __name__ == '__main__':
    main()
