"""Gravity: everything once on the 16 unseen months (2025-06-30 .. 2026-10-19), with the frozen code, at the end.

    python -I -B unseen.py

Runs data.py, peers.py, forces.py, tests.py, shape.py and judge.py on the new export and store (universe and attractors
from the refit in force on 2025-01-01, so the masses, kernels and spread windows are full on 2025-06-30; statistics from
2025-06-30), checks that the new export's refit calendar matches the old one's and that the first unseen day has its
forces, and writes out/unseen.json / .txt. Reported, never confirming: about 72 earlier experiments read these months.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import data  # noqa: E402
import peers  # noqa: E402
import forces  # noqa: E402
import tests  # noqa: E402
import shape  # noqa: E402
import judge  # noqa: E402
np = K.np


def main():
    data.main('new')
    peers.main('new')
    forces.main('new')
    tests.main('new')
    shape.main('new')
    judge.main('new')
    fh = open(K.OUT / 'unseen.txt', 'w')
    Po, Pn = data.load_panel('old'), data.load_panel('new')
    same_cal = bool(np.array_equal(np.asarray(Pn['days'])[:len(Po['days'])], np.asarray(Po['days'])))
    t_first = K.day_of(Pn['days'], K.UNSEEN[0])
    inU = np.asarray(Pn['inU'])[:, t_first]
    cover = {}
    for v in K.variants():
        F = np.load(K.BIG / 'forces_new' / f'{v["key"]}.npy', mmap_mode='r')
        cover[v['key']] = float(np.isfinite(F[inU, t_first]).mean())
    t = json.loads((K.OUT / 'tests_new.json').read_text())
    j = json.loads((K.OUT / 'judge_new.json').read_text())
    out = {'first': K.UNSEEN[0], 'last': K.c2eval.iso(Pn['days'][-1]), 'same_day_calendar_as_old': same_cal,
           'first_day_force_coverage_min': min(cover.values()), 'first_day_force_coverage': cover,
           'six': t['six'], 'judge_summary': j['summary']}
    K.jdump(out, K.OUT / 'unseen.json')
    K.log(f'unseen months {out["first"]}..{out["last"]}: new export days start with the old export\'s: {same_cal}; '
          f'force coverage of the universe on {K.UNSEEN[0]}: min {out["first_day_force_coverage_min"]:.3f}', fh)
    for s in t['six']:
        K.log(f'  {s["test"]}: best {s["best"]}: est {s["est"]:+.6f}, t {s["t"]:+.2f}, RW p {s["p"]:.4f}', fh)
    K.log(f'  judge: {j["summary"]}', fh)
    fh.close()


if __name__ == '__main__':
    main()
