"""Holm across R1 and M1 (plan: "Statistics and the bar", as edited in review: one-sided family level 0.0228, each
primary also needing t >= 2; t >= 3 reported), and the check period's consistency.

    report.py -> out/family.txt, out/family.json
"""
import json
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/volume_return')
import numpy as np  # noqa: E402

import common as C  # noqa: E402


def main():
    rep = C.OUT / 'family.txt'
    rep.write_text('')
    r1 = json.load(open(C.OUT / 'r1_old.json'))
    m1 = json.load(open(C.OUT / 'm1_old.json'))
    R = r1['R1 (primary): U_W, 2008-2019']['stats']['C2_delta']
    M = m1['M1_dev']['gross']
    Mn = m1['M1_dev']['net5']
    # Decisions (before any R1 result): the shuffled-V null's t's had sd 1.11, so R1 also needs its t / 1.11 (and that
    # t's normal p) to clear the bar; the plan's unscaled rule is reported beside it.
    NULL_SD = 1.11
    tR_cal = R['t'] / NULL_SD
    pR_cal = C.norm_sf(tR_cal)
    adj_u, passed_u = C.holm([R['p'], M['p']], C.HOLM_FAMILY)
    ps = [pR_cal, M['p']]
    adj, passed = C.holm(ps, C.HOLM_FAMILY)
    ok = [passed[0] and tR_cal >= C.T_BAR, passed[1] and M['t'] >= C.T_BAR and Mn['est'] > 0]
    ok_unscaled_R1 = bool(passed_u[0] and R['t'] >= C.T_BAR)
    C.log(rep, f'R1 calibrated for the null: t {R["t"]:+.2f} / {NULL_SD} = {tR_cal:+.2f}, p {pR_cal:.4f}; '
               f'by the plan\'s unscaled rule R1 would {"pass" if ok_unscaled_R1 else "fail"} (Holm p {adj_u[0]:.4f})')
    C.log(rep, f'== Holm across R1 and M1, 2008-2019 (one-sided family level {C.HOLM_FAMILY}: the smaller p below '
               f'{C.HOLM_FAMILY / 2}, the other below {C.HOLM_FAMILY}; each needs t >= {C.T_BAR}) ==',
          'test | estimate (90% interval) | bootstrap t | p (one-sided) | percentile p | Holm p | t >= 3? | result')
    rows = [('R1: C2 low third - high third (U_W)', R, 1.0, ''), ('M1: D gross, bps a day (U_J)', M, 1e4, '')]
    out = {}
    for (name, s, sc, _), a, pz, okk in zip(rows, adj, passed, ok):
        extra = ''
        if name.startswith('M1'):
            extra = f'; net D at 5 bps {Mn["est"] * 1e4:+.2f} bps a day (t {Mn["t"]:+.2f})'
        C.log(rep, f"{name} | {s['est'] * sc:+.4f} ({s['lo90'] * sc:+.4f}..{s['hi90'] * sc:+.4f}) | {s['t']:+.2f} | "
                   f"{s['p']:.4f} | {s['p_boot']:.4f} | {a:.4f} | {'yes' if s['t'] >= C.T_HIGH else 'no'} | "
                   f"{'PASS' if okk else 'fail'}{extra}")
        out[name] = {'stat': s, 'holm_p': float(a), 'pass': bool(okk), 'clears_t3': bool(s['t'] >= C.T_HIGH)}
    out['R1_calibrated'] = {'t': float(tR_cal), 'p': float(pR_cal), 'unscaled_pass': ok_unscaled_R1}
    Rc = r1['R1 on the check period 2020-2025-06 (consistent if > 0 at one-sided p < 0.05)']['stats']['C2_delta']
    Mc = m1['M1_check']['gross']
    C.log(rep, '', '== The check period, 2020-01..2025-06 (consistent: same sign, one-sided p < 0.05) ==',
          f"R1: {Rc['est']:+.4f}, t {Rc['t']:+.2f}, p {Rc['p']:.4f}: "
          f"{'consistent' if (Rc['est'] > 0 and Rc['p'] < 0.05) else 'not consistent'}",
          f"M1: {Mc['est'] * 1e4:+.2f} bps a day, t {Mc['t']:+.2f}, p {Mc['p']:.4f}: "
          f"{'consistent' if (Mc['est'] > 0 and Mc['p'] < 0.05) else 'not consistent'}")
    out['check'] = {'R1': Rc, 'M1': Mc}
    json.dump(out, open(C.OUT / 'family.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
