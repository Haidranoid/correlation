"""The 16 unseen months (2025-06-30 .. 2026-10-19), once, at the end, with the frozen code: P1, G and T on Yahoo's and
FRED's files read through 2026-10-19 and the new export and store. Most of the project's experiments have read those
months, so this is reported, never confirming.

    python3 -I -B unseen.py        -> forecast_new.*, timing_new.*, months_new.tsv, unseen.txt

Checks frozen_sha256.txt before and after. forecast.py new stops if its forecasts over the old span differ from the
old run's; timing.py new stops if its exposures over the old decisions differ.
"""
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/variance_risk_premium')
import numpy as np  # noqa: E402

import vrp_common as V  # noqa: E402


def check_hashes():
    bad = []
    for line in (V.HERE / 'frozen_sha256.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        h, name = line.split()
        if hashlib.sha256((V.HERE / name).read_bytes()).hexdigest() != h:
            bad.append(name)
    if bad:
        raise SystemExit(f'frozen hashes differ: {bad}')


check_hashes()
for s in ('forecast.py', 'timing.py'):
    r = subprocess.run([sys.executable, '-I', '-B', str(V.HERE / s), 'new'], capture_output=True, text=True)
    (V.HERE / f'{s[:-3]}_new.log').write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f'{s} new failed:\n{r.stderr[-3000:]}')
    print(s, 'new: done', flush=True)
check_hashes()

fr = json.loads((V.HERE / 'forecast_new.json').read_text())
tm = json.loads((V.HERE / 'timing_new.json').read_text())
u = fr['periods']['unseen']
q, g = u['p1'], u['gate']
L = ['The 16 unseen months, run once with the frozen code (reported, not confirming); hashes checked before and after',
     f"Forecasts over the old span equal the old run's: {fr['previous_run_match']}",
     f"P1: {q['n']} origins {u['first']} .. {u['last']}; MSE mean {q['mse_base']:.5f}, VRP {q['mse_alt']:.5f}; R2_oos {100 * q['r2_oos']:+.2f}% "
     f"(95% {100 * q.get('r2_lo95', np.nan):+.2f}..{100 * q.get('r2_hi95', np.nan):+.2f}); CW t {q['cw_t']:+.2f}, normal p {q['cw_p_normal']:.3f}, "
     f"bootstrap p {q.get('cw_boot_p', float('nan')):.3f}",
     f"G: MSE x1e3 mean {1e3 * g['mse_mean']:.4f}, VRP {1e3 * g['mse_vrp']:.4f}, (a) {1e3 * g['mse_a']:.4f}, (b) {1e3 * g['mse_b']:.4f}, "
     f"(c) {1e3 * g['mse_c']:.4f}, (d) {1e3 * g['mse_d']:.4f}: {'pass' if g['pass'] else 'fail'}"]
pq = tm['periods']['unseen months']
b = pq['boot_vrp_vs_vt']
L.append(f"T: {tm['months']} decisions {tm['first_decision']} .. {tm['last_decision']} (hold ends {tm['last_hold_end']}); "
         f"VRP rule {100 * pq['vrp']['annual']:+.1f}% Sharpe {pq['vrp']['sharpe']:.3f} ({pq['vrp']['exposure']:.0%}); VT {100 * pq['vt']['annual']:+.1f}% "
         f"Sharpe {pq['vt']['sharpe']:.3f} ({pq['vt']['exposure']:.0%}); SPY {100 * pq['bh']['annual']:+.1f}% Sharpe {pq['bh']['sharpe']:.3f}; "
         f"Sharpe VRP - VT {b['diff']:+.3f}, bootstrap p {b['p']:.3f}; VRP over random {100 * pq['vrp']['over_random']:+.2f} (p {pq['vrp']['random_p_annual']:.3f})")
for nm in ('vrp', 'vt'):
    a = tm['after_cost'][nm]
    L.append(f"After costs, {nm}: mean of the partial calendar years' differences {a['number']:+.2f} ("
             + ', '.join(f"{y} {v['diff']:+.1f}" for y, v in a['years'].items()) + f"); over the whole span, annualized: {a['span_annualized_diff']:+.2f}")
L.append(f"Exposures over the old decisions equal the old run's: {tm['previous_run_match']}")
for nm, rr in fr['replications'].items():
    q2, g2 = rr['periods']['unseen']['p1'], rr['periods']['unseen']['gate']
    t2 = tm['replications'].get(nm, {}).get('unseen months')
    line = (f"Replication {nm}: R2_oos {100 * q2['r2_oos']:+.2f}%, CW t {q2['cw_t']:+.2f}; gate {'pass' if g2.get('pass') else 'fail'}")
    if t2:
        line += (f"; rule Sharpe {t2['vrp']['sharpe']:.3f} vs VT {t2['vt']['sharpe']:.3f} (p {t2['boot_vrp_vs_vt']['p']:.3f})")
    L.append(line)
L.append(f"Old-span QQQ/IWM forecasts (different store for QQQ): " + str({k: v for k, v in fr['previous_run_match'].items() if k.startswith(('qqq', 'iwm'))}))
txt = '\n'.join(L)
print(txt)
(V.HERE / 'unseen.txt').write_text(txt + '\n')
