"""Gravity: the forward test on closes after 2026-10-19, under the frozen code.

    python -I -B forward.py prepare          # record the sha256 of every script (out/forward_frozen.json)
    C2_FWD_EXPORT=<export dir> C2_FWD_STORE=<store dir> python -I -B forward.py log     # after each new download
    C2_FWD_EXPORT=... C2_FWD_STORE=... python -I -B forward.py score                     # after >= 250 trading days

C2_FWD_START (default 2026-10-20) is the first untouched day; C2_FWD_USTART (default 2025-10-01) the first universe day,
so the attractors, masses and windows are full by then. `log` refuses to run if any script differs from its frozen
hash; it rebuilds the panel, attractors and forces from the download and writes, for every day >= C2_FWD_START not
logged before, the forces of all 110 variants for that day's universe (BIG/forward/forces_<date>.npz) and a line in
BIG/forward/forces_log.jsonl with the file's sha256 and the time, before the day's outcome exists. `score` runs the
tests on the forward days and applies the plan's confirmation: each primary that passed in development must be positive
at one-sided p < 0.05 under Holm (family 0.05); with none passed, it can only confirm the failure.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
np = K.np

SCRIPTS = ['common.py', 'kern.py', 'data.py', 'peers.py', 'forces.py', 'pilot.py', 'checks.py', 'tests.py', 'shape.py',
           'judge.py', 'unseen.py', 'intraday.py', 'forward.py', 'artifact.py', 'run.sh']
FROZEN = K.OUT / 'forward_frozen.json'
LOGDIR = K.BIG / 'forward'


def hashes():
    return {s: K.sha256(K.HERE / s) for s in SCRIPTS}


def prepare():
    K.jdump({'scripts': hashes(), 'note': 'sha256 of the frozen scripts; forward.py log/score refuse to run if any differs'}, FROZEN)
    print(f'wrote {FROZEN}')


def verify():
    want = json.loads(FROZEN.read_text())['scripts']
    have = hashes()
    bad = [s for s in want if want[s] != have.get(s)]
    if bad:
        raise SystemExit(f'scripts changed since the freeze: {bad}')


def log():
    verify()
    assert 'fwd' in K.SRC, 'set C2_FWD_EXPORT and C2_FWD_STORE'
    import data
    import peers
    import forces
    data.main('fwd')
    peers.main('fwd')
    forces.main('fwd')
    P = data.load_panel('fwd')
    days = P['days']
    inU = np.asarray(P['inU'])
    f0 = K.day_of(days, K.FWD_START)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    lg = LOGDIR / 'forces_log.jsonl'
    seen = set()
    if lg.exists():
        seen = {json.loads(line)['date'] for line in lg.read_text().splitlines()}
    keys = [v['key'] for v in K.variants()]
    Fs = {k: np.load(K.BIG / 'forces_fwd' / f'{k}.npy', mmap_mode='r') for k in keys}
    n = 0
    with open(lg, 'a') as fh:
        for t in range(f0, len(days)):
            ds = K.iso(days, t)
            if ds in seen:
                continue
            rows = np.nonzero(inU[:, t])[0]
            path = LOGDIR / f'forces_{ds}.npz'
            np.savez_compressed(path, tickers=np.asarray(P['tickers'])[rows], keys=np.array(keys),
                                forces=np.stack([np.asarray(Fs[k][rows, t]) for k in keys]))
            fh.write(json.dumps({'date': ds, 'file': path.name, 'sha256': K.sha256(path), 'stocks': int(len(rows)),
                                 'logged_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'data_through': K.iso(days, len(days) - 1)}) + '\n')
            n += 1
    print(f'logged {n} new days to {lg}')


def score():
    verify()
    assert 'fwd' in K.SRC, 'set C2_FWD_EXPORT and C2_FWD_STORE'
    import data
    import peers
    import forces
    import tests
    data.main('fwd')
    peers.main('fwd')
    forces.main('fwd')
    tests.main('fwd')
    t = json.loads((K.OUT / 'tests_fwd.json').read_text())
    dev = json.loads((K.OUT / 'tests_old.json').read_text())
    passed = [s['test'] for s in dev['six'] if s['pass']]
    pr = list(t['periods'].values())[0]
    n = pr['days']
    six = t['six']
    adj, rej = K.holm([s['p'] for s in six], 0.05)
    out = {'forward_days': int(n), 'enough': bool(n >= 250), 'passed_in_development': passed, 'six': six, 'holm_p_0.05': adj,
           'confirmed': {s['test']: bool(n >= 250 and r and s['est'] > 0) for s, r in zip(six, rej) if s['test'] in passed}}
    K.jdump(out, K.OUT / 'forward_score.json')
    print(f'forward days {n}; passed in development: {passed or "none"}; '
          + '; '.join(f'{s["test"]} t {s["t"]:+.2f} p {s["p"]:.3g} Holm {a:.3g}' for s, a in zip(six, adj))
          + ('' if n >= 250 else ' (not enough days yet)'))


if __name__ == '__main__':
    {'prepare': prepare, 'log': log, 'score': score}[sys.argv[1]]()
