"""Hypothesis A on the 16 unseen months (2025-06-30 .. 2026-10-19), run once, at the end, with the frozen ladder.py.

The ladder is rebuilt on the new export (its own twins, universe and baskets; Yahoo SPY on its days), every
forecaster is refitted from 2002-12 on that export exactly as on the old one, and only decision dates from 2025-06-30
whose 21-day outcome ends by 2026-10-19 are scored. Those months have been read by about 65 earlier experiments, so
this is reported, never confirming.

Checks first: the new export's first 5,911 days equal the old export's (so the decision calendar continues), and the
new export's 2008-01..2019-12 test statistics are recomputed beside the old ones.

    python3 -I -B unseen_a.py     -> a_results_new.json, a_unseen.log (WORK)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder as L  # noqa: E402
import c2eval  # noqa: E402
import numpy as np  # noqa: E402


def main():
    t = time.time()
    d_old = np.load(Path(c2eval.EXPORT_OLD) / 'days.npy')
    d_new = np.load(Path(c2eval.EXPORT_NEW) / 'days.npy')
    assert np.array_equal(d_new[:len(d_old)], d_old), 'the new export does not extend the old calendar'
    if not (L.WORK / 'rows_new.npz').exists():
        _, meta = L.build('new')
        print(f'build new: {meta["seconds"]:.0f}s; market check {meta["market_check"]}; spy check {meta["spy_check"]}',
              flush=True)
    rows = dict(np.load(L.WORK / 'rows_new.npz'))
    res = L.score(rows)
    mk = L.months_mask(rows, L.UNSEEN_FROM, '2026-10-19')
    out = {'unseen': L.period_stats(rows, res, mk, L.SEED + 99),
           'test_on_new_export': L.period_stats(rows, res, L.months_mask(rows, L.SCORE_FROM, L.SCORE_TO), L.SEED),
           'months': [c2eval.iso(d) for d in rows['days'][rows['rebal'][mk]]],
           'calendar_check': {'old_days': int(len(d_old)), 'new_days': int(len(d_new)), 'prefix_equal': True}}
    out['A1_unseen'] = L.a1_decision(out['unseen'])
    out['A2_unseen'] = L.a2_decision(out['unseen'])
    L.dump(out, L.HERE / 'a_results_new.json')
    print(f'unseen months: {int(mk.sum())} decision dates, {out["months"][0]} .. {out["months"][-1]} '
          f'({time.time() - t:.0f}s)')


if __name__ == '__main__':
    main()
