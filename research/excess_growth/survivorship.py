"""S1 (deep losers) and S2 (synthetic failures) on the old export (the .md, "Survivorship").

    python3 -I -B survivorship.py old      # after run.py old (reads WORK/e1_old.npz for the per-stock terms)

Writes surv_old.txt / .json next to this script.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eg_common as C  # noqa: E402
import run as RUN  # noqa: E402

np = C.np


def period_data(D):
    dec, labels = RUN.decisions_for(D, 'old')
    P = C.periods(D, dec)
    out = []
    for p, lab in zip(P, labels):
        rows, _ = C.select(D, p['t0'])
        H = C.holding(D, rows, p)
        drop = ~H['trade'][:, 0]
        if drop.any():
            rows = rows[~drop]
            H = C.holding(D, rows, p)
        out.append({'label': lab, 'p': p, 'rows': rows, 'H': H, 'days': np.arange(p['s'] + 1, p['end'] + 1)})
    return out


def sub_H(H, keep):
    return {'R': H['R'][keep], 'trade': H['trade'][keep], 'ends': H['ends'][keep], 'price': H['price'][keep],
            'n': H['n'], 'n_end': int((H['ends'][keep] <= H['n']).sum())}


def with_failures(H, k, L):
    """H plus k synthetic failing stocks: each day's log return = the real stocks' equal-weight mean log return plus a
    constant that takes it to S2_END of its purchase price relative to the index by offset n - 1; return L on offset n."""
    if k == 0:
        return H
    n = H['n']
    ml = np.log1p(H['R']).mean(0)                              # offsets 0..n
    c = np.log(C.S2_END) / (n - 1)
    lr = ml + c
    lr[0] = ml[0]
    syn = np.expm1(lr)
    syn[n] = L
    Rs = np.repeat(syn[None, :], k, 0)
    return {'R': np.vstack([H['R'], Rs]), 'trade': np.vstack([H['trade'], np.ones((k, n + 1), bool)]),
            'ends': np.concatenate([H['ends'], np.full(k, n + 1)]), 'n': n, 'n_end': H['n_end']}


def chain_series(per, Hs, rows_list, names=('R1d', 'R1m', 'B1')):
    res = {}
    for nm in names:
        kind, delay = RUN.PORTS[nm]
        pp = []
        for r, H, rows in zip(per, Hs, rows_list):
            g, t, w0, we = C.run_period(H, kind, delay)
            pp.append({'rows': rows, 'days': r['days'], 'gross': g[0], 'turn': t[0], 'w0': w0[0], 'wend': we[0]})
        days, g, t, pid = C.chain(pp)
        res[nm] = (g, t)
    return res, pid


def diff_tests(S, pid, labels, years):
    lab = np.array([labels[k] for k in pid])
    m = np.isin(lab, [str(y) for y in years])
    out = {}
    for cname, a in (('P1', 'R1d'), ('P2', 'R1m')):
        for cost, tag in ((C.COST, ''), (0.0, ' gross')):
            ra = C.net(*S[a], cost)
            rb = C.net(*S['B1'], cost)
            out[cname + tag] = C.test_series((np.log1p(ra) - np.log1p(rb))[m])
    return out


def main(tag='old'):
    assert tag == 'old'
    D = C.load('old')
    per = period_data(D)
    labels = [r['label'] for r in per]
    e1 = np.load(C.WORK / 'e1_old.npz')
    L = []
    say = L.append
    out = {'S1': {'periods': []}, 'S2': {}}

    # ---- S1 -------------------------------------------------------------------------------------------------------
    keep_rows, keep_H = [], []
    agg = {'deep_down': 0.0, 'deep_recovered': 0.0, 'not_deep': 0.0, 'quintiles': np.zeros(5)}
    for r in per:
        H, n = r['H'], r['H']['n']
        P = H['price']
        rel = P / P[:, :1]
        deep = (rel[:, 1:].min(1) <= 1 - C.DEEP)
        lastv = np.minimum(H['ends'], n)
        end_rel = rel[np.arange(len(rel)), lastv]
        down = end_rel <= 1 - C.DEEP
        G = e1[f"G_{r['label']}"]
        g = e1[f"g_{r['label']}"]
        assert np.array_equal(e1[f"rows_{r['label']}"], r['rows'])
        K = len(G)
        q = np.argsort(np.argsort(g)) * 5 // K
        rec = {'label': r['label'], 'K': K, 'deep': int(deep.sum()), 'deep_down': int((deep & down).sum()),
               'deep_recovered': int((deep & ~down).sum()),
               'G_deep_down': float(G[deep & down].sum() / K), 'G_deep_recovered': float(G[deep & ~down].sum() / K),
               'G_not_deep': float(G[~deep].sum() / K), 'G_total': float(G.sum() / K),
               'G_quintiles': [float(G[q == j].sum() / K) for j in range(5)]}
        out['S1']['periods'].append(rec)
        if 2008 <= int(r['label']) <= 2019:
            agg['deep_down'] += rec['G_deep_down']
            agg['deep_recovered'] += rec['G_deep_recovered']
            agg['not_deep'] += rec['G_not_deep']
            agg['quintiles'] += np.array(rec['G_quintiles'])
        keep_rows.append(r['rows'][~deep])
        keep_H.append(sub_H(H, ~deep))
        C.log(f"S1 {r['label']}: deep losers {rec['deep']} (still down {rec['deep_down']}, recovered "
              f"{rec['deep_recovered']}); second-order gap {rec['G_total']:+.4f}: deep-down {rec['G_deep_down']:+.4f}, "
              f"deep-recovered {rec['G_deep_recovered']:+.4f}, rest {rec['G_not_deep']:+.4f}")
    agg['quintiles'] = agg['quintiles'].tolist()
    out['S1']['dev_second_order'] = agg
    S, pid = chain_series(per, keep_H, keep_rows)
    out['S1']['dev'] = diff_tests(S, pid, labels, range(2008, 2020))
    out['S1']['check'] = diff_tests(S, pid, labels, range(2020, 2026))
    S0, pid0 = chain_series(per, [r['H'] for r in per], [r['rows'] for r in per])
    out['S1']['dev_all'] = diff_tests(S0, pid0, labels, range(2008, 2020))

    # ---- S2 -------------------------------------------------------------------------------------------------------
    dev = [k for k, r in enumerate(per) if 2008 <= int(r['label']) <= 2019]
    per_d = [per[k] for k in dev]
    labels_d = [labels[k] for k in dev]
    for Lr in C.S2_L:
        rows_tab = []
        for k in C.S2_K:
            Hs, rl = [], []
            for j, r in enumerate(per_d):
                Hs.append(with_failures(r['H'], k, Lr))
                rl.append(np.concatenate([r['rows'], -(1 + 100 * j + np.arange(k))]))
            S, pid = chain_series(per_d, Hs, rl)
            t = diff_tests(S, pid, labels_d, range(2008, 2020))
            rows_tab.append({'k': k, 'P1': t['P1'], 'P2': t['P2']})
            C.log(f"S2 L {Lr:+.2f} k {k}: P1 net {100 * t['P1']['per_year']:+.3f}%/yr t {t['P1']['t_nw21']:+.2f}; "
                  f"P2 net {100 * t['P2']['per_year']:+.3f}%/yr t {t['P2']['t_nw21']:+.2f}")

        def breakeven(key, field, level):
            xs = [r[key][field] for r in rows_tab]
            if xs[0] <= level:
                return 0.0
            for i in range(1, len(xs)):
                if xs[i] <= level:
                    return float(i - 1 + (xs[i - 1] - level) / (xs[i - 1] - xs[i]))
            return None                                            # beyond the grid
        out['S2'][str(Lr)] = {'rows': rows_tab,
                              'be_gap': {c: breakeven(c, 'per_year', 0.0) for c in ('P1', 'P2')},
                              'be_t3': {c: breakeven(c, 't_nw21', 3.0) for c in ('P1', 'P2')},
                              'be_t2': {c: breakeven(c, 't_nw21', 2.0) for c in ('P1', 'P2')}}

    # ---- report ---------------------------------------------------------------------------------------------------
    say('S1, deep losers (a close at least 50% below the purchase price within the period)')
    say(f"{'period':7s} {'K':>4s} {'deep':>5s} {'down':>5s} {'recov':>5s} {'G total':>9s} {'G deep-down':>11s} "
        f"{'G deep-rec':>10s} {'G rest':>9s}  G by period-return quintile (lowest..highest)")
    for rec in out['S1']['periods']:
        say(f"{rec['label']:7s} {rec['K']:4d} {rec['deep']:5d} {rec['deep_down']:5d} {rec['deep_recovered']:5d} "
            f"{rec['G_total']:+9.4f} {rec['G_deep_down']:+11.4f} {rec['G_deep_recovered']:+10.4f} {rec['G_not_deep']:+9.4f}  "
            + ' '.join(f'{x:+.4f}' for x in rec['G_quintiles']))
    a = out['S1']['dev_second_order']
    say(f"development sum: deep-down {a['deep_down']:+.4f}, deep-recovered {a['deep_recovered']:+.4f}, rest "
        f"{a['not_deep']:+.4f}; quintiles " + ' '.join(f'{x:+.4f}' for x in a['quintiles']))
    for sp in ('dev_all', 'dev', 'check'):
        say(f"\nP1/P2 {'with all stocks' if sp == 'dev_all' else 'without deep losers'} ({'2008-2019' if sp != 'check' else '2020-2025-06'}):")
        for cname, t in out['S1'][sp].items():
            say(f"  {cname:9s} {100 * t['per_year']:+.3f}%/yr  NW t {t['t_nw21']:+.2f}  p {t['p']:.4f}")
    say('\nS2, synthetic failures added at equal weight (development 2008-2019, net at 5 bps a side)')
    for Lr, v in out['S2'].items():
        say(f"L = {float(Lr):+.0%}: " + '; '.join(f"k {r['k']}: P1 {100 * r['P1']['per_year']:+.3f} (t {r['P1']['t_nw21']:+.2f}), "
                                                  f"P2 {100 * r['P2']['per_year']:+.3f} (t {r['P2']['t_nw21']:+.2f})" for r in v['rows']))
        say(f"   break-even k (net gap reaches 0): P1 {v['be_gap']['P1']}, P2 {v['be_gap']['P2']}; t below 3: "
            f"P1 {v['be_t3']['P1']}, P2 {v['be_t3']['P2']}; t below 2: P1 {v['be_t2']['P1']}, P2 {v['be_t2']['P2']}")
    txt = '\n'.join(L) + '\n'
    (C.HERE / 'surv_old.txt').write_text(txt)
    print(txt, flush=True)
    js = json.loads(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    (C.HERE / 'surv_old.json').write_text(json.dumps(js, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'old')
