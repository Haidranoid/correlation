"""Short volume: the tests, after the sign is fixed (out/train_old.json) and written into the .md.

    python3 -I -B tests.py old    # training, 2016-2019 (Tests 1-2), 2020-2025-06 (Test 3), by year
    python3 -I -B tests.py new    # the unseen months (decision days from 2025-06-30), once, at the end

Signals, each oriented by its training sign: G (primary, group-relative abnormal SVR), A (own abnormal SVR);
G_lag and A_lag (one day later, the posting-time check) use G's and A's signs.
Sets: all universe stocks ('U') and the 500 largest ('B'; ranks and deciles re-formed inside them).
"""
import sys
sys.dont_write_bytecode = True
import json  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import weekly as Wk  # noqa: E402
np, c2eval = C.np, C.c2eval

SIGN_OF = {'G': 'G', 'A': 'A', 'G_lag': 'G', 'A_lag': 'A'}


def summarize(W, s, sign, members, mask):
    ic, nn = Wk.ics(W, s, sign, members, mask)
    bk = Wk.book(W, s, sign, members, mask)
    mi, ti, ni = C.tstat(ic)
    mg, tg, _ = C.tstat(bk['gross'])
    mn, tn, nb = C.tstat(bk['net'])
    to = np.nanmean(bk['turnover'])
    return {'ic_mean': mi, 'ic_t': ti, 'weeks': ni, 'ic_pos': float(np.nanmean(ic[np.isfinite(ic)] > 0)) if ni else float('nan'),
            'median_names': float(np.median(nn[nn > 0])) if (nn > 0).any() else 0.0,
            'gross_bps': mg * 1e4, 'gross_t': tg, 'net_bps': mn * 1e4, 'net_t': tn, 'book_weeks': nb,
            'turnover_per_side': to, 'breakeven_bps_per_side': (mg / (2 * to)) * 1e4 if to and np.isfinite(to) else float('nan'),
            'median_side': float(np.median(bk['m'][bk['m'] > 0])) if (bk['m'] > 0).any() else 0.0}, ic, bk


def fmt(r):
    return (f"IC {r['ic_mean']:+.4f} (t {r['ic_t']:+.2f}, {r['weeks']} wk, {r['ic_pos']:.0%} >0, {r['median_names']:.0f} names) | "
            f"book gross {r['gross_bps']:+.1f} bps/wk (t {r['gross_t']:+.2f}), net {r['net_bps']:+.1f} (t {r['net_t']:+.2f}), "
            f"turnover {r['turnover_per_side']:.0%}/side, break-even {r['breakeven_bps_per_side']:.1f} bps/side, {r['median_side']:.0f}/side")


def main(tag):
    tr = json.loads((C.OUT / 'train_old.json').read_text())
    W = Wk.load(tag)
    periods = ['train', 'test1', 'test3'] if tag == 'old' else ['unseen']
    res = {'signs': {s: tr[SIGN_OF[s]]['sign'] for s in Wk.SIGNALS}}
    lines = [f"signs from training: " + ', '.join(f'{s} {res["signs"][s]:+.0f}' for s in Wk.SIGNALS)]
    weekly = {}
    for p in periods:
        mask = C.in_period(W['days'], W['dd'], p)
        if tag == 'new':
            mask &= W['dd'] + C.H <= len(W['days']) - 1
        dsel = W['dd'][mask]
        lines.append(f"\n== {p}: decision days {c2eval.iso(W['days'][dsel[0]])}..{c2eval.iso(W['days'][dsel[-1]])} ({mask.sum()})")
        for members in ('U', 'B'):
            for s in Wk.SIGNALS:
                r, ic, bk = summarize(W, s, res['signs'][s], members, mask)
                res[f'{p}|{members}|{s}'] = r
                lines.append(f"{members} {s:6s} " + fmt(r))
                if s == 'G':
                    weekly[f'{p}|{members}'] = {'days': [c2eval.iso(W['days'][d]) for d in dsel],
                                                'ic': ic[mask].tolist(), 'gross': bk['gross'][mask].tolist(),
                                                'net': bk['net'][mask].tolist()}
    if tag == 'old':
        lines.append('\n== by year, G (primary), all universe stocks')
        icG, _ = Wk.ics(W, 'G', res['signs']['G'], 'U', None)
        bkG = Wk.book(W, 'G', res['signs']['G'], 'U', None)
        icB, _ = Wk.ics(W, 'G', res['signs']['G'], 'B', None)
        yrs = np.array([c2eval.iso(W['days'][d])[:4] for d in W['dd']])
        res['by_year'] = {}
        for y in sorted(set(yrs)):
            m = yrs == y
            a = C.tstat(icG[m]); g = C.tstat(bkG['gross'][m]); n = C.tstat(bkG['net'][m]); b = C.tstat(icB[m])
            res['by_year'][y] = {'ic': a, 'gross': g, 'net': n, 'ic_big': b}
            lines.append(f"{y}: IC {a[0]:+.4f} (t {a[1]:+.2f}, {a[2]} wk); gross {g[0] * 1e4:+.1f} bps/wk (t {g[1]:+.2f}); "
                         f"net {n[0] * 1e4:+.1f} (t {n[1]:+.2f}); 500 largest IC {b[0]:+.4f} (t {b[1]:+.2f})")
        t1 = res['test1|U|G']
        t2 = res['test1|B|G']
        t3 = res['test3|U|G']
        lines.append('\n== tests (G, training sign)')
        lines.append(f"Test 1 (2016-2019): IC t {t1['ic_t']:+.2f} (needs >= 2.5) and book net {t1['net_bps']:+.1f} bps/wk (needs > 0): "
                     f"{'PASS' if t1['ic_t'] >= 2.5 and t1['net_bps'] > 0 else 'FAIL'}")
        lines.append(f"Test 2 (500 largest, 2016-2019): IC {t2['ic_mean']:+.4f} (needs > 0): {'PASS' if t2['ic_mean'] > 0 else 'FAIL'}")
        lines.append(f"Test 3a (2020-2025-06): IC {t3['ic_mean']:+.4f} (needs > 0): {'PASS' if t3['ic_mean'] > 0 else 'FAIL'}")
    else:
        u = res['unseen|U|G']
        lines.append(f"\nTest 3b (unseen months): IC {u['ic_mean']:+.4f} (t {u['ic_t']:+.2f}) (needs > 0): {'PASS' if u['ic_mean'] > 0 else 'FAIL'}")
    (C.OUT / f'tests_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    (C.OUT / f'tests_{tag}_weekly.json').write_text(json.dumps(weekly, default=float))
    (C.OUT / f'tests_{tag}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1])
