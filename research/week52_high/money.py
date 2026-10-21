"""The money test: daily score = PTH's percentile within the universe (latest grid day on or before d, stocks with a
finite PTH at d), ties broken by a seeded pseudo-random key per stock and day. Graded with rebalance=21, uniform and
volatility-matched random accounts.

    python3 -I -B money.py scores old      -> WORK/pth_money_old.npy and a look at the picks (out/picks_old.txt)
    python3 -I -B money.py judge old       -> walk-forward (out/money_old.txt / .json)
    python3 -I -B money.py scores new / judge new  -> the unseen months (once, at the end)
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, WORK, OUT, EXPORTS, log  # noqa: E402


def sec_names():
    try:
        return {r['ticker'].upper(): r['name'] for r in json.loads(Path(c2eval.SEC_MAP).read_text())}
    except (OSError, ValueError, KeyError):
        return {}


def scores(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=())
    z = np.load(WORK / f'panels_{tag}.npz')
    rebal, U = z['rebal'], z['uni']
    pth = np.load(WORK / f'pth_{tag}.npy', mmap_mode='r')
    N, T = pth.shape
    S = np.full((N, T), np.nan, np.float32)
    ties = []
    first = None
    for d in range(T):
        k = int(np.searchsorted(rebal, d, side='right')) - 1
        if k < 0 or not U[k].any():
            continue
        col = np.asarray(pth[:, d])
        m = np.nonzero(U[k] & np.isfinite(col))[0]
        if len(m) < 2:
            continue
        first = d if first is None else first
        key = np.random.default_rng(52_000_000 + d).random(len(m))
        o = np.lexsort((key, col[m]))             # ascending PTH, ties by the random key
        r = np.empty(len(m))
        r[o] = np.arange(len(m))
        S[m, d] = r / (len(m) - 1)
        ties.append(int((col[m] == 1).sum()))
    p = c2eval.save_scores(S, f'week52_high/pth_money_{tag}', export=EXPORTS[tag])
    ties = np.array(ties)
    log('saved', p, 'first scored day', c2eval.iso(E['days'][first]), 'mean scored per day',
        float(np.isfinite(S[:, first:]).sum(0).mean()), 'stocks at PTH = 1 per day: median', float(np.median(ties)),
        'share of days with > 20', float((ties > 20).mean()))
    return E, S


def look(tag, E, S):
    """Top-20 scores on the judge's first decision day of each test year, with SEC names, and the most frequent picks."""
    nm = sec_names()
    tk = [str(t) for t in E['tickers']]
    days = E['days']
    lines, cnt = [], Counter()
    years = range(2007, 2025) if tag == 'old' else [2025]
    for y in years:
        h0 = c2eval.day_index(days, f'{y}-06-28' if tag == 'old' else '2025-06-30')
        for d in range(h0 - 1, min(h0 + 251, len(days) - 1), 21):
            s = S[:, d]
            ok = np.isfinite(s)
            if not ok.any():
                continue
            top = np.nonzero(ok)[0][np.argsort(-s[ok], kind='mergesort')[:20]]
            cnt.update(tk[j] for j in top)
            if d == h0 - 1:
                lines.append(f'{c2eval.iso(days[d])}: ' + ', '.join(
                    f"{tk[j]} ({nm.get(tk[j].replace('.', '-'), '?')[:24]})" for j in top))
    lines.append('most frequent top-20 names on the judge decision days: ' +
                 ', '.join(f"{t} {c} ({nm.get(t.replace('.', '-'), '?')[:20]})" for t, c in cnt.most_common(60)))
    (OUT / f'picks_{tag}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:3]))


def grade(tag):
    p = WORK / f'pth_money_{tag}.npy'
    res, out = {}, []
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
    (OUT / f'money_{tag}.txt').write_text('\n'.join(out) + '\n')
    (OUT / f'money_{tag}.json').write_text(json.dumps(res, indent=1))
    print('\n'.join(l for l in out if l.startswith(('SUMMARY', 'CHANGE', '  at'))))


if __name__ == '__main__':
    what, tag = sys.argv[1], sys.argv[2]
    if what == 'scores':
        E, S = scores(tag)
        look(tag, E, S)
    elif what == 'judge':
        grade(tag)
