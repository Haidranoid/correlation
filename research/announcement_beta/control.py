"""Post-hoc control (added after the walk-forward judge results, not in the plan): the same account with the timing
removed. Score at the close of t = 1 - beta percentile on every day (always the lowest-beta stocks), built exactly
like scores.py otherwise. Shows how much of the money test is the timing and how much is a low-beta book.

    python3 -I -B control.py old|new     -> control_{tag}.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, WORK, EXPORTS, avg_rank, c2eval, kidx, log, np  # noqa: E402


def main(tag):
    z = np.load(WORK / f'meta_{tag}.npz')
    rebal, uni = z['rebal'], z['uni']
    B = np.load(WORK / f'beta_{tag}.npy')
    N, T = B.shape
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
        S[idx, t] = 1.0 - (avg_rank(b[idx].astype(np.float64)) + 0.5) / n
    del B
    p = c2eval.save_scores(S, f'announcement_beta/scores/low_beta_always_{tag}', export=EXPORTS[tag])
    del S
    out = []
    for rm in ('uniform', 'volatility'):
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=1, random_match=rm)
            out += [f'--- always low beta, walk-forward, rebalance=1, random_match={rm}', c2eval.summary_line(js),
                    str(c2eval.change_over_random(js)), txt]
        else:
            txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=1,
                                   random_match=rm)
            out += [f'--- always low beta, unseen months, rebalance=1, random_match={rm}', str(c2eval.change_over_random(js)), txt]
        log(rm, c2eval.change_over_random(js)['change'])
    (HERE / f'control_{tag}.txt').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1])
