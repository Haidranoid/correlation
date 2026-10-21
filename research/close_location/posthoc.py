"""Close location, post hoc (decided after seeing the money test; not part of the plan's pass/fail):
how much of the judge's result is plain reversal?

    python3 -I posthoc.py    # -> out/posthoc_old.txt, out/posthoc_old.json

On exactly the cells the money test scored (universe, from 2008-01-08), graded with rebalance=5, uniform and
volatility-matched:
  (a) plain 5-day reversal: -log(C_d / C_{d-5});
  (b) P's own part: each day, rank(P) regressed on ranks of the same-day and 5-day returns; score = sign x residual
      (the money-test version of Test 1).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval


def main():
    sign = json.loads((C.OUT / 'tests_old.json').read_text())['sign']
    E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
    P = np.load(C.BIG / 'P_old.npy')
    N, T = P.shape
    c0 = c2eval.day_index(E['days'], C.FIRST) - 6
    Cx = np.asarray(E['close'][:, c0:], np.float64)
    with np.errstate(all='ignore'):
        r1 = np.full(Cx.shape, np.nan)
        r5 = np.full(Cx.shape, np.nan)
        r1[:, 1:] = np.log(Cx[:, 1:] / Cx[:, :-1])
        r5[:, 5:] = np.log(Cx[:, 5:] / Cx[:, :-5])
    del Cx
    rev = np.full((N, T), np.nan, np.float32)
    own = np.full((N, T), np.nan, np.float32)
    for d in range(c0 + 6, T):
        j = d - c0
        ok = np.isfinite(P[:, d]) & np.isfinite(r1[:, j]) & np.isfinite(r5[:, j])
        si = np.nonzero(ok)[0]
        if len(si) < C.MIN_NAMES:
            continue
        rev[si, d] = -r5[si, j]
        X = np.c_[np.ones(len(si)), C.rank01(r1[si, j]), C.rank01(r5[si, j])]
        rp = C.rank01(P[si, d].astype(np.float64))
        b, *_ = np.linalg.lstsq(X, rp, rcond=None)
        own[si, d] = sign * (rp - X @ b)
    del r1, r5, P
    res, lines = {}, []
    for name, sc in (('reversal5_same_cells', rev), ('P_own_part', own)):
        p = c2eval.save_scores(sc, f'close_location/posthoc_{name}')
        for rm in ('uniform', 'volatility'):
            txt, js = c2eval.judge(p, rebalance=5, random_match=rm)
            cor = c2eval.change_over_random(js)
            line = c2eval.summary_line(js)
            res[f'{name}_{rm}'] = {'change': cor, 'summary_line': line}
            lines.append(f'{name} [{rm}]: {line}')
            print(lines[-1], flush=True)
    (C.OUT / 'posthoc_old.json').write_text(json.dumps(res, indent=1, default=float))
    (C.OUT / 'posthoc_old.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
