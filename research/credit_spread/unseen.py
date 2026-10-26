"""The 16 unseen months (2025-06-30 .. 2026-10-19), once, at the end, with the frozen code: every test on the new
export and store. About 70 experiments have read those months, so this is reported, never confirming.

    python3 -I -B unseen.py      -> inputs_new.*, p1_new.*, s2_new.*, months_new.tsv, c1_new.*, unseen.txt
"""
import json
import subprocess
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/credit_spread')
import cs_common as K  # noqa: E402

for s in ('inputs.py', 'forecast.py', 'timing.py', 'leadlag.py'):
    r = subprocess.run([sys.executable, '-I', '-B', str(K.HERE / s), 'new'], capture_output=True, text=True)
    (K.HERE / f'{s[:-3]}_new.log').write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f'{s} new failed:\n{r.stderr[-3000:]}')
    print(s, 'new: done', flush=True)

inp = json.loads((K.HERE / 'inputs_new.json').read_text())
p1 = json.loads((K.HERE / 'p1_new.json').read_text())
s2 = json.loads((K.HERE / 's2_new.json').read_text())
c1 = json.loads((K.HERE / 'c1_new.json').read_text())
u = p1['primary']['unseen']
s1 = p1['S1']['unseen']
q = s2['periods']['unseen months']
L = ['The 16 unseen months, run once with the frozen code (reported, not confirming)',
     f"Data checks: the new export continues the old one: {inp['prev_export_overlap_equal']}; overlaps with the old run: "
     + '; '.join(f"{k[8:]} {v}" for k, v in inp.items() if k.startswith('overlap_')),
     f"P1: {u['n']} origins {u.get('first')} .. {u.get('last')}; 1 - MSE ratio {100 * u['rel_impr']:+.2f}% "
     f"(95% {100 * u.get('rel_impr_lo95', float('nan')):+.2f}..{100 * u.get('rel_impr_hi95', float('nan')):+.2f}); CW t {u['cw_t']:+.2f}, "
     f"normal p {u['cw_p_normal']:.3f}, bootstrap p {u.get('cw_boot_p', float('nan')):.3f}; median dS coefficient {u['coef']['median']:+.4f}",
     f"S1: 1 - MSE ratio {100 * s1['rel_impr']:+.2f}%, CW t {s1['cw_t']:+.2f}",
     f"P1 forecasts, old vs new run over the old span: base {p1.get('old_vs_new_yhat_base')}, spread {p1.get('old_vs_new_yhat_spread')}",
     f"S2: {q['months']} decisions {s2['first_decision']} .. {s2['last_decision']}, warning on {q['fires']} ({', '.join(s2['fire_dates']) or 'none'}); "
     f"Sharpe spread rule {q['spread rule']['sharpe']:.3f} vs rule B {q['rule B (variance timing)']['sharpe']:.3f} "
     f"(bootstrap p {q['boot_spread_vs_B']['p']:.3f}); annual {100 * q['spread rule']['annual']:+.1f}% vs {100 * q['rule B (variance timing)']['annual']:+.1f}%, "
     f"SPY {100 * q['buy and hold']['annual']:+.1f}%; over random {100 * q['spread rule']['over_random']:+.2f} (p {q['spread rule']['random_p_annual']:.3f}); "
     f"after costs minus SPY: mean of the calendar-year differences {s2['after_cost']['spread rule']['number']:+.1f}, whole span annualized "
     f"{s2['after_cost']['spread rule']['span_annualized_diff']:+.1f} (rule B {s2['after_cost']['rule B']['number']:+.1f} / "
     f"{s2['after_cost']['rule B']['span_annualized_diff']:+.1f})"]
for r in c1['rows']:
    if r['spread'] == 'BAA10Y':
        L.append(f"C1 {r['side']}, k {r['k']}: on R5 {r['coef_R5']:+.3f} (t {r['t_R5']:+.2f}), on dS5 {r['coef_dS5']:+.3f} (t {r['t_dS5']:+.2f}), n {r['n']}")
txt = '\n'.join(L)
print(txt)
(K.HERE / 'unseen.txt').write_text(txt + '\n')
