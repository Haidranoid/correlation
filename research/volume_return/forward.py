"""Forward confirmation on closes after 2026-10-19, through the frozen code (plan: "Periods", Forward).

    C2_FWD_EXPORT=<export dir> C2_FWD_STORE=<store dir> [C2_FWD_START=2026-10-20] forward.py log
        Checks the frozen hashes, runs build.py and m1.py on the download (tag 'fwd'), and appends to the ledger
        (BIG/forward/ledger_*.npz) every forward day not yet logged: R, V, P_full, P_base, U_W, G (U_W), U_J, by
        ticker. Values already in the ledger are kept (first seen); revisions in later downloads are counted and
        logged, never applied. A day missing from a download is a gap, logged, never filled later.
    forward.py score
        From the ledger only: M1's paired books on forward decisions (delay 1) and R1 on the forward days as one
        period. At 250 or more forward days, each primary that passed on 2008-2019 must be positive at one-sided
        p < 0.05 under Holm (family 0.05). -> out/forward_score.txt
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/volume_return')
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import m1  # noqa: E402
import r1  # noqa: E402

c2eval = C.c2eval
PY = '/home/bhoult/programming/correlation/research/gravity/.venv/bin/python'
LEDGER = C.BIG / 'forward'
LOG = C.OUT / 'forward_log.txt'


def check_hashes():
    for line in (C.HERE / 'frozen_sha256.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            h, name = line.split()
            if C.sha256(C.HERE / name) != h:
                raise SystemExit(f'frozen script changed: {name}')


def ledgers():
    return sorted(LEDGER.glob('ledger_*.npz'))


def log_download():
    check_hashes()
    LEDGER.mkdir(parents=True, exist_ok=True)
    start = os.environ.get('C2_FWD_START', '2026-10-20')
    for script in ('build.py', 'm1.py'):
        r = subprocess.run([PY, '-I', '-B', str(C.HERE / script), 'fwd'], env=dict(os.environ), capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise SystemExit(f'{script} failed:\n{r.stderr[-3000:]}')
    E = C.export('fwd', arrays=())
    days, tickers = E['days'], E['tickers']
    R, V, cal = C.load_panels('fwd')
    d = C.tagdir('fwd')
    Pf = np.load(d / 'scores_full.npy', mmap_mode='r')
    Pb = np.load(d / 'scores_base.npy', mmap_mode='r')
    seen = {}
    for p in ledgers():
        z = np.load(p, allow_pickle=False)
        for j, ds in enumerate(z['dates']):
            seen[str(ds)] = (p, j)
    a = c2eval.day_index(days, start)
    new = [t for t in range(a, len(days)) if c2eval.iso(days[t]) not in seen]
    revised = 0
    row = {t: j for j, t in enumerate(tickers)}
    for ds, (p, j) in seen.items():          # revisions: count, never apply
        t = c2eval.day_index(days, ds)
        if t < len(days) and c2eval.iso(days[t]) == ds:
            z = np.load(p)
            rows = [row.get(str(tk), -1) for tk in z['tickers']]
            ok = np.array(rows) >= 0
            old = z['R'][ok, j]
            cur = np.asarray(R[np.array(rows)[ok], t])
            revised += int(np.sum(~((np.isnan(old) & np.isnan(cur)) | (np.abs(old - cur) <= 1e-6))))
    gap = ''
    if seen and new:
        last_seen = max(seen)
        expected = c2eval.day_index(days, last_seen) + 1
        if new[0] != expected:
            gap = f'; GAP: days {c2eval.iso(days[expected])}..{c2eval.iso(days[new[0] - 1])} missing'
    if new:
        cols = np.array(new)
        ks = [C.cal_index(cal['rebal'], t) for t in new]
        np.savez(LEDGER / f'ledger_{c2eval.iso(days[new[0]])}_{c2eval.iso(days[new[-1]])}.npz',
                 dates=np.array([c2eval.iso(days[t]) for t in new]), tickers=np.array([str(x) for x in tickers]),
                 R=np.asarray(R[:, cols]), V=np.asarray(V[:, cols]), Pf=np.asarray(Pf[:, cols]),
                 Pb=np.asarray(Pb[:, cols]), UW=np.stack([cal['UW'][k] for k in ks], 1),
                 GW=np.stack([cal['GW'][k] for k in ks], 1), UJ=np.stack([cal['UJ'][k] for k in ks], 1))
    C.log(LOG, f'download {os.environ["C2_FWD_EXPORT"]} (last day {c2eval.iso(days[-1])}): {len(new)} new forward '
               f'days logged; {revised} revised values in already-logged days ignored{gap}')


def score():
    rep = C.OUT / 'forward_score.txt'
    rep.write_text('')
    zs = [np.load(p) for p in ledgers()]
    if not zs:
        raise SystemExit('no ledger')
    tick = sorted(set().union(*[set(map(str, z['tickers'])) for z in zs]))
    ti = {t: i for i, t in enumerate(tick)}
    dates = np.concatenate([z['dates'] for z in zs])
    n = len(dates)
    P = {k: np.full((len(tick), n), np.nan, np.float32) for k in ('R', 'V', 'Pf', 'Pb', 'GW')}
    UW = np.zeros((len(tick), n), bool)
    c0 = 0
    for z in zs:
        rows = np.array([ti[str(t)] for t in z['tickers']])
        m = len(z['dates'])
        for k in P:
            P[k][rows, c0:c0 + m] = z[k]
        UW[rows, c0:c0 + m] = z['UW']
        c0 += m
    dec = np.arange(0, n - 2)
    bf = m1.book(P['Pf'], P['R'], dec, 1, 1)
    bb = m1.book(P['Pb'], P['R'], dec, 1, 1)
    D = bf['gross'] - bb['gross']
    Dn = m1.net(bf, C.COST) - m1.net(bb, C.COST)
    sM = m1.series_stats(D)
    cal = {'rebal': np.array([0]), 'UW': UW[:, 0][None], 'GW': P['GW'][:, 0][None]}
    res = r1.family(P['R'], P['V'], cal, [('forward', 0, n - 2, C.MIN_PAIRS)], 'UW', 'GW')
    sR = res['stats']['C2_delta']
    C.log(rep, f'forward days {dates[0]}..{dates[-1]} ({n})',
          f"M1: D gross {sM['est'] * 1e4:+.2f} bps a day (t {sM.get('t', float('nan')):+.2f}, p {sM.get('p', float('nan')):.4f}); "
          f"net {np.nanmean(Dn) * 1e4:+.2f}",
          f"R1: Delta {sR['est']:+.4f} (t {sR.get('t', float('nan')):+.2f}, p {sR.get('p', float('nan')):.4f})")
    if n >= 250:
        fam = json.load(open(C.OUT / 'family.json'))
        names = [k for k in fam if k.startswith('R1:') or k.startswith('M1:')]
        passed = [fam[k]['pass'] for k in names]
        ps = [C.norm_sf(sR.get('t', 0.0) / 1.11), sM.get('p', 1.0)]   # R1 scaled by the null's sd, as on 2008-2019
        adj, ok = C.holm([p for p, q in zip(ps, passed) if q] or [1.0], 0.05)
        C.log(rep, f'confirmation (each primary that passed on 2008-2019 positive at one-sided p < 0.05 under Holm): '
                   + ('no primary passed on 2008-2019, so the forward test can only confirm the failure'
                      if not any(passed) else ('confirmed' if all(ok) else 'not confirmed')))
    else:
        C.log(rep, f'{n} forward days: report only; the rule applies from 250')


if __name__ == '__main__':
    {'log': log_download, 'score': score}[sys.argv[1]]()
