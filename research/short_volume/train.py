"""Short volume: the training period only (decision days 2010-01-04..2015-12-31). Sets the sign.

Sign = the sign of the mean weekly rank IC of the group-relative abnormal SVR (G) with the next week's
market-relative return over the training decision days. Also reported: the own abnormal SVR (A) and both one-day
lagged variants. No statistic after the training period's last outcome (the close of the week after the last
training decision day) is computed here.

    python3 -I -B train.py      # old export; writes out/train_old.json and out/train_old.txt
"""
import sys
sys.dont_write_bytecode = True
import json  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import weekly as Wk  # noqa: E402
np, c2eval = C.np, C.c2eval


def main():
    W = Wk.load('old')
    same, nxt = Wk.market_check(W['E'])
    mask = C.in_period(W['days'], W['dd'], 'train')
    last = W['dd'][mask][-1]
    res = {'market_check': {'corr_same_day': same, 'corr_next_day': nxt},
           'weeks_from': c2eval.iso(W['days'][W['dd'][mask][0]]), 'weeks_to': c2eval.iso(W['days'][last]),
           'last_outcome_day': c2eval.iso(W['days'][last + C.H])}
    lines = [f"market[d] vs median stock log return: corr same day {same:.3f}, next day {nxt:.3f}",
             f"training decision days {res['weeks_from']}..{res['weeks_to']} (last outcome close {res['last_outcome_day']})"]
    for s in Wk.SIGNALS:
        ic, nn = Wk.ics(W, s, 1.0, 'U', mask)
        mean, t, n = C.tstat(ic)
        sign = 1.0 if mean > 0 else -1.0
        res[s] = {'mean_ic_raw': mean, 't': t, 'weeks': n, 'sign': sign, 'median_names': float(np.median(nn[mask & (nn > 0)]))
                  if (mask & (nn > 0)).any() else 0, 'first_week_with_ic': c2eval.iso(W['days'][W['dd'][np.nonzero(np.isfinite(ic))[0][0]]])}
        lines.append(f"{s:6s} raw mean weekly IC {mean:+.4f} (t {t:+.2f}, {n} weeks, median names {res[s]['median_names']:.0f}, "
                     f"first {res[s]['first_week_with_ic']}) -> sign {sign:+.0f}")
    (C.OUT / 'train_old.json').write_text(json.dumps(res, indent=1))
    (C.OUT / 'train_old.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
