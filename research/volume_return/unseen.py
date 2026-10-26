"""The 16 unseen months (2025-06-30..2026-10-19), run once at the end with the frozen code (plan: "Periods").

    unseen.py -> out/unseen.txt, out/unseen.json, plus build_new.*, r1_new.*, m1_new.*, money_new.*

Checks the sha256 hashes in frozen_sha256.txt before and after; builds the new export's panels (with the overlap check
against the old export), then R1 on the unseen months as one period, M1's books on the unseen decisions, and the judge
as a single holdout from 2025-06-30 (delay 1). Reported, never confirming.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/volume_return')
import common as C  # noqa: E402

PY = '/home/bhoult/programming/correlation/research/gravity/.venv/bin/python'


def check_hashes(rep, when):
    bad = []
    for line in (C.HERE / 'frozen_sha256.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        h, name = line.split()
        if C.sha256(C.HERE / name) != h:
            bad.append(name)
    C.log(rep, f'hashes {when}: ' + ('all match' if not bad else 'MISMATCH ' + ', '.join(bad)))
    if bad:
        raise SystemExit('frozen scripts changed')


def main():
    rep = C.OUT / 'unseen.txt'
    rep.write_text('')
    check_hashes(rep, 'before')
    env = dict(os.environ)
    for script in ('build.py', 'r1.py', 'm1.py', 'money.py'):
        C.log(rep, f'running {script} new')
        r = subprocess.run([PY, '-I', '-B', str(C.HERE / script), 'new'], env=env, capture_output=True, text=True)
        (C.OUT / f'{script[:-3]}_new.log').write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise SystemExit(f'{script} failed:\n{r.stderr[-3000:]}')
    b = (C.OUT / 'build_new.txt').read_text()
    C.log(rep, '', '== Overlap ==', *[l for l in b.splitlines() if 'tickers in both' in l])
    r1 = json.load(open(C.OUT / 'r1_new.json'))
    m1 = json.load(open(C.OUT / 'm1_new.json'))
    mo = json.load(open(C.OUT / 'money_new.json'))
    key = [k for k in r1 if k.startswith('R1 on unseen')][0]
    s = r1[key]['stats']['C2_delta']
    C.log(rep, '', '== R1 on the unseen months ==', f"Delta {s['est']:+.4f}, bootstrap t {s.get('t', float('nan')):+.2f}, "
                                                    f"p {s.get('p', float('nan')):.4f}")
    g = m1['M1_unseen']['gross']
    n = m1['M1_unseen']['net5']
    C.log(rep, '', '== M1 on the unseen months ==', f"D gross {g['est'] * 1e4:+.2f} bps a day (t {g.get('t', float('nan')):+.2f}, "
                                                    f"p {g.get('p', float('nan')):.4f}); net {n['est'] * 1e4:+.2f}")
    p = mo['primary']
    C.log(rep, '', '== The judge, single holdout ==',
          f"P_full: {p['pct0']:+.1f}% vs random {p['rand0']:+.1f}% (p {p['p0']:.3f}), change {p['change0']:+.1f}; at 5 bps "
          f"{p['pct5']:+.1f}% vs SPY {p['spy']:+.1f}%: after costs {p['after_cost_period']:+.1f} over the period "
          f"({p['after_cost_annual']:+.1f} a year)")
    check_hashes(rep, 'after')
    json.dump({'r1': r1[key]['stats'], 'm1': m1['M1_unseen'], 'money': mo}, open(C.OUT / 'unseen.json', 'w'), indent=1,
              default=lambda o: None)


if __name__ == '__main__':
    main()
