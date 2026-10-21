"""Who the loners are (no returns used): the 30 loneliest stocks at a few dates, the twin funds of the long and short
legs, and a scan of the long leg for fund-like stocks the exclusion rule missed.

    python3 -I look.py old|new     -> loners_<tag>.txt
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import re
from collections import Counter
from common import c2eval, np, WORK, HERE, EXPORTS, sec_names, day_num
from book import loner_legs

tag = _sys.argv[1]
E = c2eval.load_export(EXPORTS[tag], arrays=())
tw = c2eval.twins(E)
z = np.load(WORK / f'connect_{tag}.npz')
cd = np.load(WORK / f'c_daily_{tag}.npy', mmap_mode='r')
cal = z['cal']
C = np.asarray(cd[:, cal]).T
U, meddv, vol, beta_bac, best = z['U'], z['meddv'], z['vol'], z['beta_bac'], z['best']
c_stock, c_fund = z['c_stock'], z['c_fund']
days = E['days']
tk = np.array([str(t) for t in E['tickers']])
ftk = np.array([str(t) for t in z['fund_tickers']])
twt = np.array([str(t) for t in tw['fund_ticker']])
nm = sec_names()
lines = []


def P(s=''):
    print(s)
    lines.append(s)


def name(t):
    return nm.get(t.upper().replace('.', '-'), '(not in SEC map)')


def match(k, j):
    b = best[k, j]
    if b >= 0:
        return tk[b]
    if b > -10 ** 8:
        return ftk[-b - 1] + ' (fund)'
    return '?'


def twin(k, j):
    f = tw['fund_row'][k, j]
    return twt[f] if f >= 0 else '-'


def book_set(k):
    c, dvm, v = C[k], meddv[k], vol[k]
    return np.nonzero(U[k] & np.isfinite(c) & np.isfinite(v) & np.isfinite(dvm))[0]


dates = ['2007-01-02', '2010-01-04', '2013-01-02', '2016-01-04', '2019-01-02', '2022-01-03', '2025-01-02'] if tag == 'old' \
    else ['2025-07-01', '2026-01-02', '2026-07-01']
for iso in dates:
    k = int(np.searchsorted(cal, c2eval.day_index(days, iso), side='left'))
    if k >= len(cal):
        continue
    m = book_set(k)
    o = m[np.argsort(C[k, m], kind='mergesort')[:30]]
    P(f'== {c2eval.iso(days[cal[k]])}: the 30 loneliest of {len(m)} (c median {np.median(C[k, m]):.3f}) ==')
    P(f"{'ticker':8s} {'c':>6s} {'stocks':>6s} {'funds':>6s} {'best match':16s} {'twin':6s} {'$M/day':>7s} {'vol%':>5s} {'beta':>5s}  SEC name")
    for j in o:
        P(f"{tk[j]:8s} {C[k, j]:6.3f} {c_stock[k, j]:6.3f} {c_fund[k, j]:6.3f} {match(k, j):16s} {twin(k, j):6s} "
          f"{meddv[k, j] / 1e6:7.1f} {100 * vol[k, j] * np.sqrt(252):5.0f} {beta_bac[k, j]:5.2f}  {name(tk[j])[:44]}")
    P()

# long and short legs over all formations (test-1 period for the old export): twin funds, best-match kind
if tag == 'old':
    ks = [k for k in range(len(cal)) if day_num('2006-01-01') <= days[cal[k]] <= day_num('2019-12-31')]
else:
    ks = [k for k in range(len(cal)) if days[cal[k]] >= day_num('2025-06-01')]
twL, twH, inL = Counter(), Counter(), Counter()
fundL, fundH, nL, nH = 0, 0, 0, 0
n_form = 0
for k in ks:
    m = book_set(k)
    if len(m) < 81:
        continue
    n_form += 1
    legs = loner_legs(m, C[k], meddv[k], vol[k])
    for (S, sign, _, _) in legs:
        for j in S:
            t = twin(k, j)
            isf = best[k, j] < 0
            if sign > 0:
                twL[t] += 1; nL += 1; fundL += isf; inL[tk[j]] += 1
            else:
                twH[t] += 1; nH += 1; fundH += isf
P(f'== Legs over {n_form} formations ==')
P(f'best match is a fund: long leg {100 * fundL / nL:.0f}%, short leg {100 * fundH / nH:.0f}%')
P('twin fund (c2eval.twins, the groups map labels), share of leg-months:')
P('  long : ' + ', '.join(f'{t} {100 * n / nL:.1f}%' for t, n in twL.most_common(20)))
P('  short: ' + ', '.join(f'{t} {100 * n / nH:.1f}%' for t, n in twH.most_common(20)))
P()
P('most frequent long-leg names (months in the long leg):')
P('  ' + ', '.join(f'{t} {n} ({name(t)[:22]})' for t, n in inL.most_common(80)))
P()
# scan for fund-like names the exclusion rule missed
pat = re.compile(r'\b(TRUST|INCOME|ETF|ISHARES|PROSHARES|DIREXION|ROYALTY|PORTFOLIO|MUNICIPAL|MUNI|BOND|OPPORTUNITIES|'
                 r'STRATEGIC|SECURITIES|CLOSED|INVESTORS|TOTAL RETURN|DIVIDEND|PREFERRED|CREDIT|ADVANTAGED|LTD DURATION)\b')
flag = [(t, n, name(t)) for t, n in inL.items() if pat.search(name(t).upper()) or name(t) == '(not in SEC map)']
P(f'long-leg names whose SEC name looks fund-like or is missing ({len(flag)} of {len(inL)}):')
for t, n, s in sorted(flag, key=lambda x: -x[1]):
    P(f'  {t:8s} {n:3d}  {s}')
(HERE / f'loners_{tag}.txt').write_text('\n'.join(lines) + '\n')
