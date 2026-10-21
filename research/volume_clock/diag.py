"""Volume clock: diagnostics added AFTER seeing tests 1-2 (they decide nothing; the verdict rests on tests.py).

    python3 -I diag.py old|new

1. Why the clock misses: hit rate at +-10 trading days, share of windows with a volume ratio >= 2 within +-3, and
   where the counted spike nearest to c falls.
2. Log against simple returns: the plan's premium is a mean of log returns, which subtracts about half the variance;
   the published premium is in simple returns. Simple market-relative return per event, and each event's return
   minus its own stock's mean outside its windows (both clustered by entry day).
Writes diag_<tag>.md and diag_<tag>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests import BIG, HERE, bps, clustered, in_period, load  # noqa: E402


def run(tag):
    Z = load(tag)
    out, J = [], {}
    P = list(Z['periods'])
    out.append(f'# Volume clock diagnostics (added after seeing tests 1–2), {tag} export\n')
    out.append('## Why the clock misses\n')
    out.append('| Period | Events | Hit ±3 (the rule) | Hit ±10 | Volume ≥ 2× within ±3 (chance) | Nearest spike before c−3 | c−3..c−1 | c | c+1..c+3 | after c+3 | none within ±10 |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|')
    J['clock'] = {}
    for name in P:
        m = in_period(Z, name) & np.isfinite(Z['hit10'])
        o = Z['near_off'][m].astype(int)
        sh = lambda k: float(np.mean(k))  # noqa: E731
        b = Z[f'base_{name}']
        w = np.bincount(Z['j'][in_period(Z, name)], minlength=b.shape[1]).astype(float)
        with np.errstate(all='ignore'):
            n2 = b[5] / b[2]
        k2 = (w > 0) & np.isfinite(n2)
        chance2 = float((w[k2] * n2[k2]).sum() / w[k2].sum())
        r = dict(chance_vol2=chance2, events=int(m.sum()), hit3=float(np.mean(Z['hit'][m])), hit10=float(np.mean(Z['hit10'][m])),
                 vol2=float(np.mean(Z['vol2'][m])), before=sh((o >= -10) & (o < -3)), early=sh((o >= -3) & (o < 0)),
                 at=sh(o == 0), late=sh((o > 0) & (o <= 3)), after=sh((o > 3) & (o <= 10)), none=sh(o == 99))
        J['clock'][name] = r
        out.append(f"| {name} | {r['events']} | {r['hit3']:.1%} | {r['hit10']:.1%} | {r['vol2']:.1%} ({r['chance_vol2']:.1%}) | {r['before']:.1%} | "
                   f"{r['early']:.1%} | {r['at']:.1%} | {r['late']:.1%} | {r['after']:.1%} | {r['none']:.1%} |")
    out.append('')

    out.append('## Log against simple returns (per event; SE clustered by entry day)\n')
    out.append('| Period | Events | Log mean (the rule) | Simple mean | t | Simple − own stock outside | t | Log − own stock outside | t | Simple: hits / misses |')
    out.append('|---|---|---|---|---|---|---|---|---|---|')
    J['returns'] = {}
    for name in P:
        b = Z[f'base_{name}']
        with np.errstate(all='ignore'):
            out_log = b[1] / b[0]
            out_s = b[4] / b[0]
        m = in_period(Z, name) & np.isfinite(Z['ret'])
        j, g = Z['j'][m], Z['e'][m]
        rl, rs = Z['ret'][m], Z['ret_simple'][m]
        s_l = clustered(rl, g)
        s_s = clustered(rs, g)
        x_s = clustered(rs - out_s[j], g)
        x_l = clustered(rl - out_log[j], g)
        h = Z['hit'][m]
        sh_ = clustered(rs[h == 1], g[h == 1])
        sm_ = clustered(rs[h == 0], g[h == 0])
        J['returns'][name] = dict(log=s_l, simple=s_s, simple_minus_outside=x_s, log_minus_outside=x_l,
                                  simple_hits=sh_, simple_misses=sm_, sd_log=float(np.std(rl)))
        out.append(f"| {name} | {s_l['n']} | {bps(s_l['mean'])} (t {s_l['t']:+.2f}) | {bps(s_s['mean'])} | {s_s['t']:+.2f} | "
                   f"{bps(x_s['mean'])} | {x_s['t']:+.2f} | {bps(x_l['mean'])} | {x_l['t']:+.2f} | "
                   f"{bps(sh_['mean'])} (t {sh_['t']:+.2f}) / {bps(sm_['mean'])} (t {sm_['t']:+.2f}) |")
    out.append('')
    out.append('Returns in bps over the 7 trading days from close c−6 to close c+1. "Own stock outside" is the same '
               'stock\'s mean 7-day return over its in-universe days outside its predicted windows in the same period.\n')
    txt = '\n'.join(out)
    (HERE / f'diag_{tag}.md').write_text(txt + '\n')
    (HERE / f'diag_{tag}.json').write_text(json.dumps(J, indent=1, default=float))
    print(txt)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'old')
