"""Volume clock: tests 1-4 from the event table (vc.py).

    python3 -I tests.py old [1]     # old export: tests 1-3 (pass "1" to print test 1 only)
    python3 -I tests.py new         # the unseen months: test 4 (and the hit rate there)

Writes results_<tag>.md (tables) and results_<tag>.json next to this file.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
BIG = c2eval.WORK / 'volume_clock'
COST = 0.0010  # 10 bps per round trip


def clustered(x, g):
    """Mean, clustered SE (clusters g), t, n, number of clusters."""
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    n = len(x)
    if n < 2:
        return dict(mean=float(np.mean(x)) if n else np.nan, se=np.nan, t=np.nan, n=n, clusters=n)
    m = x.mean()
    e = x - m
    u, inv = np.unique(g, return_inverse=True)
    S = np.bincount(inv, weights=e)
    G = len(u)
    var = G / (G - 1) * (S ** 2).sum() / n ** 2
    se = float(np.sqrt(var))
    return dict(mean=float(m), se=se, t=float(m / se), n=n, clusters=G)


def bps(v):
    return f'{v * 1e4:+.1f}'


def load(tag):
    z = np.load(BIG / f'events_{tag}.npz')
    Z = {k: z[k] for k in z.files}
    Z['ok'] = Z['univ'] & ~Z['fund_like']
    Z['periods'] = {str(n): tuple(b) for n, b in zip(Z['period_names'], Z['period_bounds'])}
    return Z


def in_period(Z, name):
    d0, d1 = Z['periods'][name]
    return Z['ok'] & (Z['e'] >= d0) & (Z['e'] < d1)


def chance(Z, name, m):
    """Event-weighted chance hit rate and outside 7-day mean for the stocks with events in mask m."""
    b = Z[f'base_{name}']  # [4, N]: n_out, s_out, n_u, s_near
    w = np.bincount(Z['j'][m], minlength=b.shape[1]).astype(float)
    with np.errstate(all='ignore'):
        near = b[3] / b[2]
        out = b[1] / b[0]
    okn = (w > 0) & np.isfinite(near)
    oko = (w > 0) & np.isfinite(out)
    return (float((w[okn] * near[okn]).sum() / w[okn].sum()), float((w[oko] * out[oko]).sum() / w[oko].sum()),
            int(oko.sum()), float(b[0][oko].sum()))


def hit_table(Z, m, years):
    rows = []
    for y in years:
        k = m & (Z['entry_year'] == y) & np.isfinite(Z['hit'])
        if k.sum() == 0:
            continue
        rows.append((y, int(k.sum()), float(np.mean(Z['hit'][k])), float(np.mean(Z['hit_raw'][k]))))
    return rows


def run(tag, only1=False):
    Z = load(tag)
    days = Z['days']
    out, J = [], {}
    P = list(Z['periods'])
    main = P[0]
    out.append(f'# Volume clock results, {tag} export ({c2eval.iso(days[0])}..{c2eval.iso(days[-1])})\n')
    out.append(f"Events (kept predictions) {len(Z['j'])}; in the universe and not fund-like {Z['ok'].sum()}.\n")

    # ---- Test 1: the clock's hit rate
    out.append('## Test 1: hit rate (a counted spike within ±3 trading days of c)\n')
    out.append('| Entry year | Events | Hit rate (counted spikes) | Hit rate (raw spikes) |')
    out.append('|---|---|---|---|')
    m_all = Z['ok'] if tag == 'old' else in_period(Z, 'unseen')  # new export: only the unseen months
    years = sorted(set(Z['entry_year'][m_all].tolist()))
    for y, n, h, hr in hit_table(Z, m_all, years):
        out.append(f'| {y} | {n} | {h:.1%} | {hr:.1%} |')
    out.append('')
    out.append('| Period | Events | Hit rate | Raw-spike hit rate | Chance for the same stocks | Late June hit rate (events) | Other weeks |')
    out.append('|---|---|---|---|---|---|---|')
    J['test1'] = {}
    for name in P:
        m = in_period(Z, name)
        k = m & np.isfinite(Z['hit'])
        ch, _, _, _ = chance(Z, name, m)
        lj = k & Z['late_june']
        nl = k & ~Z['late_june']
        h = float(Z['hit'][k].mean()) if k.any() else np.nan
        J['test1'][name] = dict(events=int(k.sum()), hit=h, hit_raw=float(Z['hit_raw'][k].mean()) if k.any() else np.nan,
                                chance=ch, late_june=float(Z['hit'][lj].mean()) if lj.any() else np.nan,
                                late_june_n=int(lj.sum()), other=float(Z['hit'][nl].mean()) if nl.any() else np.nan)
        t = J['test1'][name]
        out.append(f"| {name} | {t['events']} | {t['hit']:.1%} | {t['hit_raw']:.1%} | {ch:.1%} | "
                   f"{t['late_june']:.1%} ({t['late_june_n']}) | {t['other']:.1%} |")
    out.append('')
    if tag == 'old':
        h = J['test1'][main]['hit']
        J['test1']['pass'] = bool(h >= 0.40)
        out.append(f"**Test 1 ({main}):** hit rate {h:.1%} against the 40% bar: {'PASS' if h >= 0.40 else 'FAIL'}.\n")
    if only1:
        print('\n'.join(out))
        return

    # ---- Test 2 / 3 / 4: the premium
    ret, g = Z['ret'], Z['e']
    out.append('## The premium: market-relative log return, close c−6 to close c+1, per event (SE clustered by entry day)\n')
    out.append('| Period | Events | Mean (bps) | t | Net of 10 bps | Median (bps) | Same stocks outside their windows (bps per 7 days) | Mean − outside |')
    out.append('|---|---|---|---|---|---|---|---|')
    J['premium'] = {}
    for name in P:
        m = in_period(Z, name) & np.isfinite(ret)
        s = clustered(ret[m], g[m])
        _, outside, nst, nout = chance(Z, name, in_period(Z, name))
        s.update(net=s['mean'] - COST, median=float(np.median(ret[m])), outside=outside, outside_stocks=nst,
                 outside_stock_days=nout)
        J['premium'][name] = s
        out.append(f"| {name} | {s['n']} ({s['clusters']} entry days) | {bps(s['mean'])} | {s['t']:+.2f} | {bps(s['net'])} | "
                   f"{bps(s['median'])} | {bps(outside)} | {bps(s['mean'] - outside)} |")
    out.append('')

    out.append('### Hits against misses (the hit is known only after the sale; a diagnostic, not a trade)\n')
    out.append('| Period | Hits: events, mean (bps), t | Misses: events, mean (bps), t | Hits − misses (bps) |')
    out.append('|---|---|---|---|')
    J['hits_misses'] = {}
    for name in P:
        m = in_period(Z, name) & np.isfinite(ret) & np.isfinite(Z['hit'])
        hm = m & (Z['hit'] == 1)
        mm = m & (Z['hit'] == 0)
        sh, sm = clustered(ret[hm], g[hm]), clustered(ret[mm], g[mm])
        J['hits_misses'][name] = dict(hits=sh, misses=sm)
        out.append(f"| {name} | {sh['n']}, {bps(sh['mean'])}, {sh['t']:+.2f} | {sm['n']}, {bps(sm['mean'])}, {sm['t']:+.2f} | "
                   f"{bps(sh['mean'] - sm['mean'])} |")
    out.append('')

    out.append('### Last week of June (centre on June 22–30) against the rest\n')
    out.append('| Period | Late June: events, hit rate, mean (bps), t | Rest: events, hit rate, mean (bps), t |')
    out.append('|---|---|---|')
    J['late_june'] = {}
    for name in P:
        m = in_period(Z, name) & np.isfinite(ret)
        res = {}
        cells = []
        for lab, k in (('late_june', m & Z['late_june']), ('rest', m & ~Z['late_june'])):
            s = clustered(ret[k], g[k])
            kh = k & np.isfinite(Z['hit'])
            s['hit'] = float(Z['hit'][kh].mean()) if kh.any() else np.nan
            res[lab] = s
            cells.append(f"{s['n']}, {s['hit']:.1%}, {bps(s['mean'])}, {s['t']:+.2f}")
        J['late_june'][name] = res
        out.append(f'| {name} | {cells[0]} | {cells[1]} |')
    out.append('')

    out.append('### By entry year\n')
    out.append('| Entry year | Events | Hit rate | Mean (bps) | t |')
    out.append('|---|---|---|---|---|')
    J['by_year'] = {}
    for y in years:
        m = m_all & (Z['entry_year'] == y) & np.isfinite(ret)
        if m.sum() < 2:
            continue
        s = clustered(ret[m], g[m])
        kh = m & np.isfinite(Z['hit'])
        s['hit'] = float(Z['hit'][kh].mean()) if kh.any() else np.nan
        J['by_year'][int(y)] = s
        out.append(f"| {y} | {s['n']} | {s['hit']:.1%} | {bps(s['mean'])} | {s['t']:+.2f} |")
    out.append('')
    if tag == 'old':
        out.append('### By period\n')
        out.append('| Entry years | Events | Hit rate | Mean (bps) | t |')
        out.append('|---|---|---|---|---|')
        J['by_period'] = {}
        for lab, (y0, y1) in (('2006–2009', (2006, 2009)), ('2010–2014', (2010, 2014)), ('2015–2019', (2015, 2019)),
                              ('2020–2025-06', (2020, 2025))):
            m = Z['ok'] & (Z['entry_year'] >= y0) & (Z['entry_year'] <= y1) & np.isfinite(ret)
            s = clustered(ret[m], g[m])
            kh = m & np.isfinite(Z['hit'])
            s['hit'] = float(Z['hit'][kh].mean())
            J['by_period'][lab] = s
            out.append(f"| {lab} | {s['n']} | {s['hit']:.1%} | {bps(s['mean'])} | {s['t']:+.2f} |")
        out.append('')
        p2 = J['premium']['2006-2019']
        pass2 = (p2['mean'] >= 0.0020) and (p2['t'] >= 3) and (p2['net'] > 0)
        p3 = J['premium']['2020-2025']
        pass3 = bool(np.sign(p3['mean']) == np.sign(p2['mean']) and p3['t'] * np.sign(p2['mean']) >= 1.5)
        J['test2_pass'], J['test3_pass'] = bool(pass2), pass3
        out.append(f"**Test 2 (2006–2019):** mean {bps(p2['mean'])} bps, t {p2['t']:+.2f}, net {bps(p2['net'])} bps: "
                   f"{'PASS' if pass2 else 'FAIL'} (needs ≥ +20 bps, t ≥ 3, net > 0).\n")
        out.append(f"**Test 3 (2020–2025-06):** mean {bps(p3['mean'])} bps, t {p3['t']:+.2f}: {'PASS' if pass3 else 'FAIL'} "
                   f"(needs the same sign as test 2 and t ≥ 1.5).\n")
    else:
        p4 = J['premium']['unseen']
        out.append(f"**Test 4 (unseen months):** mean {bps(p4['mean'])} bps, t {p4['t']:+.2f}.\n")
    txt = '\n'.join(out)
    (HERE / f'results_{tag}.md').write_text(txt + '\n')
    (HERE / f'results_{tag}.json').write_text(json.dumps(J, indent=1, default=float))
    print(txt)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'old', only1=len(sys.argv) > 2 and sys.argv[2] == '1')
