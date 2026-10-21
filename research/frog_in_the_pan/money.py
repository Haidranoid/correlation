"""The money test: the judge's account on the plan's score.

    python3 -I -B money.py build old|new plain|divfix   -> WORK/score_<tag>_<px>.npy (via c2eval.save_scores)
    python3 -I -B money.py walk                         # walk-forward on the old export, rebalance=21
    python3 -I -B money.py build old divfix mom; python3 -I -B money.py walk_mom   # plain-momentum baseline
    python3 -I -B money.py unseen                       # the 16 unseen months on the new export (once, at the end)

Score at the close of day d: eligible = the liquid set at the latest twins rebalance at or before d (minus
c2eval.excluded) with finite PRET and ID at d (closes through d - 21) and a close on d; fifths by PRET, ID thirds
within each fifth (common.sorts, as in the books); score = PRET's percentile (rank + 1) / n among the eligible stocks
for the lowest-ID third of every fifth, NaN for all others. The judge buys the 20 highest positive scores: the
strongest continuous winners. Computed every day because the judge rebalances on its own 21-day schedule.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, WORK, OUT, EXPORTS, N_ID, log, universe_cached, sorts  # noqa: E402


def build(tag, px, which='fip'):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    N, T = E['close'].shape
    rebal, U, _ = universe_cached(tag, E)
    PRET = np.ascontiguousarray(np.load(WORK / f'pret_{tag}_{px}.npy').T)
    ID = np.ascontiguousarray(np.load(WORK / f'id_{tag}_{px}.npy').T)
    C = np.ascontiguousarray(np.asarray(E['close']).T)
    S = np.full((T, N), np.nan, np.float32)
    nd = 0
    for d in range(T):
        k = int(np.searchsorted(rebal, d, side='right')) - 1
        if k < 0 or not U[k].any():
            continue
        p, i, c = PRET[d], ID[d], C[d]
        elig = U[k] & np.isfinite(p) & np.isfinite(i) & np.isfinite(c) & (c > 0)
        g, pct = sorts(p.astype(np.float64), i.astype(np.float64), elig)
        low = (g >= 0) & (g % N_ID == 0) if which == 'fip' else (g >= 0)
        S[d, low] = pct[low]
        nd += 1
    del PRET, ID, C
    name = f'score_{tag}_{px}' if which == 'fip' else f'score_{which}_{tag}_{px}'
    path = c2eval.save_scores(S.T, f'frog_in_the_pan/{name}', export=EXPORTS[tag])
    cnt = np.isfinite(S).sum(1)
    log(f'{which} {tag} {px}: scored days {nd}, names scored per day median {int(np.median(cnt[cnt > 0]))}; saved {path}')


def grade(path, label, **kw):
    txt, js = c2eval.judge(path, **kw)
    cor = c2eval.change_over_random(js)
    head = c2eval.summary_line(js) if js.get('years') else ''
    with open(OUT / 'money.txt', 'a') as f:
        f.write(f'=== {label}\n{txt}\n{head}\nchange over random: {json.dumps(cor)}\n\n')
    log(label, head, cor)
    return js, cor


def walk():
    for px, rms in (('divfix', ('uniform', 'volatility')), ('plain', ('uniform', 'volatility'))):
        p = WORK / f'score_old_{px}.npy'
        for rm in rms:
            grade(p, f'walk-forward old export, {px} signals, rebalance=21, random={rm}', rebalance=21, random_match=rm)


def walk_mom():
    """Baseline added after the FIP results (the README's rule: beat the nearest known effect): plain 12-1 momentum
    through the same path, score = PRET's percentile for every eligible stock."""
    p = WORK / 'score_mom_old_divfix.npy'
    for rm in ('uniform', 'volatility'):
        grade(p, f'BASELINE plain momentum (PRET percentile, all eligible), walk-forward old export, divfix, rebalance=21, '
                 f'random={rm}', rebalance=21, random_match=rm)


def unseen():
    for px in ('divfix', 'plain'):
        p = WORK / f'score_new_{px}.npy'
        for rm in ('uniform', 'volatility'):
            grade(p, f'UNSEEN new export from 2025-06-30, {px} signals, rebalance=21, random={rm}',
                  export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=21, random_match=rm)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'build':
        build(a[1], a[2], *(a[3:4] or ['fip']))
    elif a[0] == 'walk_mom':
        walk_mom()
    elif a[0] == 'walk':
        walk()
    elif a[0] == 'unseen':
        unseen()
