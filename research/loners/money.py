"""The money test: daily scores = 1 - (c's percentile within the stock's dollar-volume tercile), graded by the judge
with rebalance=21, uniform and volatility-matched random accounts.

    python3 -I money.py scores old      -> WORK/loners_money_old.npy, picks_old.txt
    python3 -I money.py judge old       -> walk-forward 2006-07..2024-25 (money_old.txt / .json)
    python3 -I money.py scores new / judge new  -> the unseen months (once, at the end)

Day d: the stocks of the universe of the latest calendar day k(d) with c at d; dollar-volume terciles by rank of
the calendar day's 252-day median dollar volume among them; score = 1 - r / (n - 1), r the 0-based rank of c in the
tercile (the loneliest of each tercile scores 1). NaN elsewhere.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json
from collections import Counter
from common import c2eval, np, WORK, HERE, EXPORTS, rank, sec_names


def scores(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=())
    z = np.load(WORK / f'connect_{tag}.npz')
    cd = np.load(WORK / f'c_daily_{tag}.npy', mmap_mode='r')
    cal, U, meddv = z['cal'], z['U'], z['meddv']
    N, T = cd.shape
    S = np.full((N, T), np.nan, np.float32)
    first = int(cal[int(np.argmax(U.sum(1) > 0))])
    for d in range(first, T):
        k = int(np.searchsorted(cal, d, side='right')) - 1
        c = np.asarray(cd[:, d])
        m = np.nonzero(U[k] & np.isfinite(c) & np.isfinite(meddv[k]))[0]
        if len(m) < 3:
            continue
        q = np.floor(3 * rank(meddv[k, m]) / len(m)).astype(int)
        for qq in range(3):
            mi = m[q == qq]
            if len(mi) < 2:
                continue
            S[mi, d] = 1 - rank(c[mi]) / (len(mi) - 1)
    p = c2eval.save_scores(S, f'loners/loners_money_{tag}', export=EXPORTS[tag])
    return E, S, p


def look(tag, E, S):
    """Top-20 scores on the judge's first decision day of each test year, with SEC names, and the most frequent picks."""
    nm = sec_names()
    tk = [str(t) for t in E['tickers']]
    days = E['days']
    lines, cnt = [], Counter()
    years = range(2006, 2025) if tag == 'old' else [2025]
    for y in years:
        h0 = c2eval.day_index(days, f'{y}-06-28' if tag == 'old' else '2025-06-30')
        for d in range(h0 - 1, min(h0 + 251, len(days) - 1), 21):
            s = S[:, d]
            ok = np.isfinite(s)
            top = np.nonzero(ok)[0][np.argsort(-s[ok], kind='mergesort')[:20]]
            cnt.update(tk[j] for j in top)
            if d == h0 - 1:
                lines.append(f'{c2eval.iso(days[d])}: ' + ', '.join(f"{tk[j]} ({nm.get(tk[j].replace('.', '-'), '?')[:24]})" for j in top))
    lines.append('most frequent top-20 names on the judge decision days: ' +
                 ', '.join(f"{t} {c} ({nm.get(t.replace('.', '-'), '?')[:20]})" for t, c in cnt.most_common(60)))
    (HERE / f'picks_{tag}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[-3:]))


def grade(tag):
    p = WORK / f'loners_money_{tag}.npy'
    res = {}
    out = []
    for rm in ('uniform', 'volatility'):
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=21, random_match=rm)
        else:
            txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=21,
                                   random_match=rm)
        cor = c2eval.change_over_random(js)
        res[rm] = cor
        out.append(f'===== random_match={rm} =====\n{txt}')
        if tag == 'old':
            out.append('SUMMARY ' + c2eval.summary_line(js))
        out.append(f"CHANGE OVER RANDOM ({rm}, 0 bps): {cor['change']:+.2f} pts/yr; per year {cor['per_year']}; "
                   f"ahead {cor['ahead']}/{cor['years']}; se {cor['se']:.2f}")
        for ci in (1, 2):
            c2 = c2eval.change_over_random(js, cost_index=ci)
            out.append(f"  at cost level {ci}: {c2['change']:+.2f} pts/yr, ahead {c2['ahead']}/{c2['years']}")
    (HERE / f'money_{tag}.txt').write_text('\n'.join(out) + '\n')
    (HERE / f'money_{tag}.json').write_text(json.dumps(res, indent=1))
    print('\n'.join(l for l in out if l.startswith(('SUMMARY', 'CHANGE', '  at'))))


if __name__ == '__main__':
    what, tag = _sys.argv[1], _sys.argv[2]
    if what == 'scores':
        E, S, p = scores(tag)
        print('saved', p, 'finite per day (median over scored days):', int(np.median(np.isfinite(S).sum(0)[np.isfinite(S).any(0)])))
        look(tag, E, S)
    elif what == 'judge':
        grade(tag)
