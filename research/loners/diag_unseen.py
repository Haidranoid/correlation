"""Checks written after the unseen months' result (they change nothing in the tests):
1. the book rebuilt on the new export against the old export's book over the shared months (plumbing);
2. the unseen months split into long and short legs, and the stocks that contributed most;
3. the same split for 2020-2025 on the old export.

    python3 -I diag_unseen.py    -> diag_unseen.txt
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from collections import defaultdict
from common import c2eval, np, WORK, HERE, EXPORTS, ffill, sec_names, day_num, tstat
from book import loner_legs

lines = []


def P(s=''):
    print(s)
    lines.append(s)


zo = np.load(WORK / 'book_old.npz')
zn = np.load(WORK / 'book_new.npz')
to, tn = zo['loners__m_t0'], zn['loners__m_t0']
co = zo['loners__m_complete']
common_t0 = [t for i, t in enumerate(to) if co[i] and t in set(tn)]
go = np.array([zo['loners__m_gross'][list(to).index(t)] for t in common_t0])
gn = np.array([zn['loners__m_gross'][list(tn).index(t)] for t in common_t0])
P(f'1. old vs new export, loners book, {len(common_t0)} shared complete months: corr {np.corrcoef(go, gn)[0, 1]:.4f}; '
  f'mean %/month old {100 * go.mean():+.3f}, new {100 * gn.mean():+.3f}')
P()


def split(tag, a, b):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
    tk = np.array([str(t) for t in E['tickers']])
    days = E['days']
    z = np.load(WORK / f'connect_{tag}.npz')
    cd = np.load(WORK / f'c_daily_{tag}.npy', mmap_mode='r')
    cal = z['cal']
    C = np.asarray(cd[:, cal]).T
    Pr = ffill(E['close'])
    T = len(days)
    lo, hi = day_num(a), day_num(b)
    legsum = {+1: 0.0, -1: 0.0}
    contrib = defaultdict(float)
    cellsum = defaultdict(float)
    nmon = 0
    for k, t0 in enumerate(cal):
        t1 = int(cal[k + 1]) if k + 1 < len(cal) and cal[k + 1] <= T - 1 else T - 1
        if t0 >= T - 1:
            break
        dd = np.arange(t0 + 1, t1 + 1)
        dd = dd[(days[dd] >= lo) & (days[dd] <= hi)]
        if len(dd) == 0:
            continue
        nmon += 1
        m = np.nonzero(z['U'][k] & np.isfinite(C[k]) & np.isfinite(z['vol'][k]) & np.isfinite(z['meddv'][k]))[0]
        legs = loner_legs(m, C[k], z['meddv'][k], z['vol'][k])
        e0, e1 = dd[0] - 1, dd[-1]
        for (S, sign, lev, cell) in legs:
            # dollar P&L over the window part of the hold: lev/len(S) per stock, value relative to t0
            pnl = (Pr[S, e1] - Pr[S, e0]) / Pr[S, t0]
            legsum[int(sign)] += sign * lev * pnl.mean()
            cellsum[cell] += sign * lev * pnl.mean()
            for j, p in zip(S, pnl):
                contrib[tk[j]] += sign * lev * p / len(S)
    P(f'{tag} {a}..{b}: {nmon} holding periods; long leg {100 * legsum[1]:+.2f}%, short leg {100 * legsum[-1]:+.2f}% '
      f'(sum {100 * (legsum[1] + legsum[-1]):+.2f}%, gross, % of the $1/$1 book)')
    P('   by cell (dollar-volume tercile, volatility tercile; 0 = low): ' +
      ', '.join(f'{c}: {100 * v:+.2f}%' for c, v in sorted(cellsum.items())))
    nm = sec_names()
    s = sorted(contrib.items(), key=lambda x: x[1])
    P('   most negative contributions: ' + ', '.join(f"{t} {100 * v:+.2f} ({nm.get(t.replace('.', '-'), '?')[:18]})" for t, v in s[:15]))
    P('   most positive contributions: ' + ', '.join(f"{t} {100 * v:+.2f} ({nm.get(t.replace('.', '-'), '?')[:18]})" for t, v in s[::-1][:15]))
    P()


P('2./3. legs and contributions (gross)')
split('new', '2025-06-30', '2026-10-19')
split('old', '2020-01-01', '2025-06-27')
split('old', '2006-01-01', '2019-12-31')
(HERE / 'diag_unseen.txt').write_text('\n'.join(lines) + '\n')
