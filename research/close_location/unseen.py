"""Close location: the unseen months on the new export (once, at the end), the plan's regression and book.

    python3 -I build.py new && python3 -I unseen.py && python3 -I money.py new

Weeks formed on or after 2025-06-27 whose hold ends by 2026-10-19, direction fixed by Test 1 (out/tests_old.json).
The book is run from 2025-01 so that the first unseen week pays real turnover; only unseen weeks are graded.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import tests as TS  # noqa: E402
np = C.np
c2eval = C.c2eval

LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def main():
    old = json.loads((C.OUT / 'tests_old.json').read_text())
    sign = old['sign']
    R = C.load_rows('new')
    days = R['days']
    warm = R['t'] >= c2eval.day_index(days, '2025-01-01')
    W = TS.weekly(R, warm)
    W5 = TS.weekly(R, warm & R['in500'])
    a, b = C.OLD_END, '2026-10-20'
    m = TS.window(W, days, a, b)
    m5 = TS.window(W5, days, a, b)
    s = TS.stats(W, m, sign)
    s5 = TS.stats(W5, m5, sign)
    wk = W['d'][m]
    say(f"Unseen weeks: {s['weeks']} formed {c2eval.iso(days[wk[0]])}..{c2eval.iso(days[wk[-1]])}, last hold ends "
        f"{c2eval.iso(days[wk[-1] + C.WIN])}; median names {s['n_median']:.0f}; direction sign {sign:+d}")
    for name, x in (('all', s), ('500 largest', s5)):
        say(f"{name}: P (controls) {100 * x['b_P']:+.3f}% a week (t {x['t_P']:+.2f}); raw {100 * x['raw_P']:+.3f}% (t {x['t_raw']:+.2f}); "
            f"same-day {100 * x['b_r1']:+.3f} (t {x['t_r1']:+.2f}); 5-day {100 * x['b_r5']:+.3f} (t {x['t_r5']:+.2f}); "
            f"book gross {100 * x['gross']:+.3f}% (t {x['t_gross']:+.2f}), net {100 * x['net']:+.3f}% (t {x['t_net']:+.2f}); "
            f"sum of weeks gross {100 * x['gross'] * x['weeks']:+.1f}%, net {100 * x['net'] * x['weeks']:+.1f}%; traded {x['traded']:.2f}")
    same = bool(np.sign(s['b_P']) == np.sign(old['test1']['b_P']))
    say(f"Test 3 (unseen part): P's coefficient sign {np.sign(s['b_P']):+.0f} vs Test 1 {np.sign(old['test1']['b_P']):+.0f} -> "
        f"{'same sign (passes)' if same else 'opposite sign (fails)'}")
    say('deciles of P (unseen), % next week minus ranked mean: ' + '  '.join(f'{100 * x:+.3f}' for x in s['dec']))
    (C.OUT / 'unseen.json').write_text(json.dumps({'sign': sign, 'all': s, 'top500': s5, 'same_sign': same}, indent=1, default=float))
    (C.OUT / 'unseen.txt').write_text('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
