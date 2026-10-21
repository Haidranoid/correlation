"""The money test: daily scores = 1 - (correlation percentile within the stock's volatility quintile), graded by the
judge with rebalance=21, uniform and volatility-matched random accounts.

    python3 -I money.py scores old      -> WORK/bac_money_old.npy and a look at the picks (picks_old.txt)
    python3 -I money.py judge old       -> walk-forward 2006-07..2024-25 (money_old.txt / .json)
    python3 -I money.py scores new / judge new  -> the unseen months (once, at the end)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json
from collections import Counter
from common import c2eval, np, WORK, HERE, EXPORTS, sec_names, FIRST


def rank(x):
    return np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort')


def scores(tag, univ_key='univ'):
    E = c2eval.load_export(EXPORTS[tag], arrays=())
    z = np.load(WORK / f'panels_{tag}.npz')
    cal, vol, corr, U = z['cal'], z['vol'], z['corr'], z[univ_key]
    N, T = vol.shape
    S = np.full((N, T), np.nan, np.float32)
    for d in range(FIRST, T):
        k = np.searchsorted(cal, d, side='right') - 1
        m = np.nonzero(U[k])[0]
        if len(m) == 0:
            continue
        m = m[np.isfinite(vol[m, d]) & np.isfinite(corr[m, d])]
        q = np.floor(5 * rank(vol[m, d]) / len(m)).astype(int)
        for qq in range(5):
            mi = m[q == qq]
            if len(mi) < 2:
                continue
            S[mi, d] = 1 - rank(corr[mi, d]) / (len(mi) - 1)
    name = f'betting_against_correlation/bac_money_{tag}' + ('' if univ_key == 'univ' else '_flag')
    p = c2eval.save_scores(S, name, export=EXPORTS[tag])
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
            ok = np.isfinite(s) & (s > 0)
            top = np.nonzero(ok)[0][np.argsort(-s[ok], kind='mergesort')[:20]]
            cnt.update(tk[j] for j in top)
            if d == h0 - 1:
                lines.append(f'{c2eval.iso(days[d])}: ' + ', '.join(f"{tk[j]} ({nm.get(tk[j].replace('.', '-'), '?')[:24]})" for j in top))
    lines.append('most frequent top-20 names on the judge decision days: ' +
                 ', '.join(f"{t} {c} ({nm.get(t.replace('.', '-'), '?')[:20]})" for t, c in cnt.most_common(60)))
    (HERE / f'picks_{tag}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def grade(tag, suffix=''):
    p = WORK / f'bac_money_{tag}{suffix}.npy'
    res = {}
    out = []
    for rm in ('uniform', 'volatility'):
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=21, random_match=rm)
        else:
            txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=21, random_match=rm)
        if 'years' not in js and 'report' in js:
            # c2eval.change_over_random reads walk-forward JSON; a single holdout keeps its costs in report.strategy
            js = {**js, 'costs': js['report']['strategy']}
        cor = c2eval.change_over_random(js)
        res[rm] = {'change_over_random': cor, 'judge': js}
        out.append(f'===== random_match={rm} =====\n{txt}')
        if tag == 'old':
            out.append('SUMMARY ' + c2eval.summary_line(js))
        out.append(f"CHANGE OVER RANDOM ({rm}, 0 bps): {cor['change']:+.2f} pts/yr; per year {cor['per_year']}; ahead {cor['ahead']}/{cor['years']}; se {cor['se']:.2f}")
        for ci in (1, 2):
            c2 = c2eval.change_over_random(js, cost_index=ci)
            out.append(f"  at cost level {ci}: {c2['change']:+.2f} pts/yr, ahead {c2['ahead']}/{c2['years']}")
    (HERE / f'money_{tag}{suffix}.txt').write_text('\n'.join(out) + '\n')
    (HERE / f'money_{tag}{suffix}.json').write_text(json.dumps({k: v['change_over_random'] for k, v in res.items()}, indent=1))
    print('\n'.join(l for l in out if l.startswith(('SUMMARY', 'CHANGE', '  at'))))


if __name__ == '__main__':
    what, tag = _sys.argv[1], _sys.argv[2]
    if what == 'scores':
        E, S, p = scores(tag)
        print('saved', p, 'finite per day (mean over scored days):', np.isfinite(S).sum(0)[FIRST:].mean())
        look(tag, E, S)
        if tag == 'old':
            scores(tag, 'univ_flag')
    elif what == 'judge':
        grade(tag)
    elif what == 'judge_flag':
        grade(tag, '_flag')
