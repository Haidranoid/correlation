"""The judge's money test: at the close of t, score = beta percentile if t+1 is an A day, else 1 - beta percentile.

    python3 -I -B scores.py old       # walk-forward, rebalance=1, uniform and volatility-matched random -> judge_old.txt
    python3 -I -B scores.py new       # the 16 unseen months, once                                    -> judge_new.txt

beta[:, t+1] uses returns through the close of t; the universe is the latest rebalance at or before t; the percentile
is (average rank + 0.5) / n among the universe stocks with a beta, so it lies strictly inside (0, 1) and every
universe stock gets a positive score on both kinds of day. Whether t+1 is an A day is known in advance (the calendar
is published). The last export day has no t+1 and gets no score.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, WORK, EXPORTS, avg_rank, c2eval, calendar, kidx, log, np  # noqa: E402


def build_scores(tag):
    z = np.load(WORK / f'meta_{tag}.npz')
    days, rebal, uni = z['days'], z['rebal'], z['uni']
    B = np.load(WORK / f'beta_{tag}.npy')
    N, T = B.shape
    A = calendar(days)['A']
    S = np.full((N, T), np.nan, np.float32)
    for t in range(T - 1):
        k = kidx(rebal, t)
        if k < 0:
            continue
        b = B[:, t + 1]
        sel = uni[k] & np.isfinite(b)
        n = int(sel.sum())
        if n < 50:
            continue
        idx = np.nonzero(sel)[0]
        pct = (avg_rank(b[idx].astype(np.float64)) + 0.5) / n
        S[idx, t] = pct if A[t + 1] else 1.0 - pct
    log(tag, 'scored stock-days', int(np.isfinite(S).sum()), 'days with scores', int(np.isfinite(S).any(0).sum()))
    return S


def main(tag):
    (WORK / 'scores').mkdir(parents=True, exist_ok=True)
    S = build_scores(tag)
    p = c2eval.save_scores(S, f'announcement_beta/scores/beta_timing_{tag}', export=EXPORTS[tag])
    del S
    out = []
    for rm in ('uniform', 'volatility'):
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=1, random_match=rm)
            line = c2eval.summary_line(js)
            cor = c2eval.change_over_random(js)
            out += [f'--- walk-forward, rebalance=1, random_match={rm}', line, str(cor), txt]
        else:
            txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=1,
                                   random_match=rm)
            cor = c2eval.change_over_random(js)
            out += [f'--- unseen months 2025-06-30..2026-10-19, rebalance=1, random_match={rm}', str(cor), txt]
        log(rm, 'change over random', cor['change'])
    (HERE / f'judge_{tag}.txt').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1])
