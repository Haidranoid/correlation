"""Hand check for fund-like stocks the exclusion rule misses (no returns used): every stock that was ever in the
universe with an SEC name that could be an investment company, plus closed-end funds known not to have FUND in their
names, with months in the universe and in the long leg, and its median c.

    python3 -I diag_funds.py old|new    -> funds_check_<tag>.txt
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import re
from collections import Counter, defaultdict
from common import c2eval, np, WORK, HERE, EXPORTS, sec_names
from book import loner_legs

tag = _sys.argv[1]
E = c2eval.load_export(EXPORTS[tag], arrays=())
tk = np.array([str(t) for t in E['tickers']])
z = np.load(WORK / f'connect_{tag}.npz')
cd = np.load(WORK / f'c_daily_{tag}.npy', mmap_mode='r')
cal = z['cal']
C = np.asarray(cd[:, cal]).T
nm = sec_names()
KNOWN_CEF = {'OXLC', 'ECC', 'OCCI', 'GDV', 'GAB', 'GUT', 'GGN', 'GGT', 'XAI', 'RVT', 'RMT', 'TY', 'GAM', 'CET', 'HQH', 'HQL',
             'TEAF', 'SOR', 'BCV', 'NXDT', 'EIC', 'PFLT', 'SAR', 'OFS', 'PSEC', 'MAIN', 'ARCC', 'HTGC', 'BXSL', 'FSK',
             'GBDC', 'OBDC', 'TSLX', 'OXSQ', 'CCAP', 'TRIN', 'CSWC', 'GAIN', 'GLAD', 'MFIC', 'NMFC', 'BBDC', 'TCPC', 'CGBD',
             'SLRC', 'HRZN', 'TPVG', 'WHF', 'PNNT', 'FDUS', 'GSBD', 'MSDL', 'KBDC', 'PSBD', 'CION', 'TPZ', 'NMAI', 'SPE'}
pat = re.compile(r'\b(TRUST|INVESTORS|INCOME|SECURITIES|OPPORTUNITIES|CAPITAL CORP|CREDIT CO|INVESTMENT CORP|'
                 r'STRATEGIC|PREMIUM|DIVIDEND|EQUITY|PARTNERS|ETF|SHARES)\b')
inU, inL = Counter(), Counter()
cs = defaultdict(list)
for k in range(len(cal)):
    m = np.nonzero(z['U'][k] & np.isfinite(C[k]) & np.isfinite(z['vol'][k]) & np.isfinite(z['meddv'][k]))[0]
    if len(m) < 81:
        continue
    for j in m:
        inU[tk[j]] += 1
        cs[tk[j]].append(C[k, j])
    for (S, s, _, _) in loner_legs(m, C[k], z['meddv'][k], z['vol'][k]):
        if s > 0:
            for j in S:
                inL[tk[j]] += 1
rows = []
for t in inU:
    name = nm.get(t.upper().replace('.', '-'), '')
    if t in KNOWN_CEF or pat.search(name.upper()):
        rows.append((t, inU[t], inL[t], float(np.median(cs[t])), name))
rows.sort(key=lambda r: -r[2])
lines = [f'{tag}: universe stocks with an investment-company-like SEC name or on the closed-end fund/BDC list: {len(rows)}',
         f"{'ticker':8s} {'months':>6s} {'long':>5s} {'c med':>6s}  SEC name"]
lines += [f'{t:8s} {u:6d} {l:5d} {c:6.3f}  {n}' for (t, u, l, c, n) in rows]
print('\n'.join(lines))
(HERE / f'funds_check_{tag}.txt').write_text('\n'.join(lines) + '\n')
