"""Tests 1-3 and the reported tables for the twin gap fade, on the old export (or the unseen months with 'new').

    python3 -I tests.py old     # out/tests_old.txt, out/tests_old.json, out/daily_old.tsv
    python3 -I tests.py new     # the unseen months 2025-06-30..2026-10-19, once, at the end
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import books as B  # noqa: E402
from books import c2eval, np  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def fmt(x, scale=1e4, d=2):
    m, t, n = B.tstat(x)
    return f'{m * scale:+.{d}f} ({t:+.2f})'


PERIODS_OLD = [('2008..2019 (Test 1)', '2008-01-01', '2020-01-01'), ('2020..2025-06 (Test 3)', '2020-01-01', None),
               ('2008..2025-06 (all)', '2008-01-01', None),
               ('2008..2009', '2008-01-01', '2010-01-01'), ('2010..2014', '2010-01-01', '2015-01-01'),
               ('2015..2019', '2015-01-01', '2020-01-01'), ('2020..2022', '2020-01-01', '2023-01-01'),
               ('2023..2025-06', '2023-01-01', None)]
PERIODS_NEW = [('unseen 2025-06-30..2026-10-19', '2025-06-30', None)]


def series(Bk):
    s = {
        'gross': Bk['twin_gross'],
        'bound': B.net(Bk, 'twin', p=0),
        'head': B.net(Bk, 'twin'),
        'head_unnetted': B.net(Bk, 'twin', netted=False),
        'raw_head': B.net(Bk, 'raw'), 'raw_bound': B.net(Bk, 'raw', p=0), 'raw_gross': Bk['raw_gross'],
        'akbas_head': B.net(Bk, 'akbas'), 'akbas_bound': B.net(Bk, 'akbas', p=0), 'akbas_gross': Bk['akbas_gross'],
    }
    s['d_raw'] = s['head'] - s['raw_head']
    s['d_akbas'] = s['head'] - s['akbas_head']
    s['d_raw_gross'] = Bk['twin_gross'] - Bk['raw_gross']
    s['d_akbas_gross'] = Bk['twin_gross'] - Bk['akbas_gross']
    return s


def main(tag):
    R = B.load_rows(tag)
    days = R['days']
    periods = PERIODS_OLD if tag == 'old' else PERIODS_NEW
    res = {}
    daily = {}
    for sub in ('all', '500'):
        Bk = B.books(R, sub)
        s = series(Bk)
        daily[sub] = (Bk, s)
        res[sub] = {}
        say(f'\n===== universe: {sub} =====')
        say('bps/day (t). head = entry 10 bps worse than the open + netted twin costs; bound = entry at the open + '
            'twin costs; gross = no costs.')
        hdr = ('period', 'days', 'ranked/day', 'twin gross', 'twin bound', 'TWIN HEAD', 'head un-netted', 'raw (a) head',
               'akbas (b) head', 'twin-(a)', 'twin-(b)', 'twin-(a) gross', 'twin-(b) gross', 'break-even p bps', 'NW t head')
        say(' | '.join(hdr))
        for name, a, b in periods:
            m = B.period_mask(days, Bk['t'], a, b)
            if m.sum() < 3:
                continue
            row = {k: B.tstat(v[m]) for k, v in s.items()}
            be = row['bound'][0] / 2 * 1e4
            res[sub][name] = {k: dict(mean_bps=v[0] * 1e4, t=v[1], n=v[2]) for k, v in row.items()}
            res[sub][name]['break_even_bps'] = be
            res[sub][name]['nw_t_head'] = B.nw_t(s['head'][m])
            res[sub][name]['ranked_per_day'] = float(np.median(Bk['n'][m]))
            res[sub][name]['leg_size'] = float(np.median(Bk['nl'][m]))
            res[sub][name]['tc_bps'] = float(np.mean(Bk['twin_tc'][m]) * 1e4)
            res[sub][name]['tcu_bps'] = float(np.mean(Bk['twin_tcu'][m]) * 1e4)
            for sg in B.SIGNALS:
                res[sub][name][f'{sg}_long_bps'] = float(np.mean(Bk[f'{sg}_long'][m]) * 1e4)
                res[sub][name][f'{sg}_short_bps'] = float(np.mean(Bk[f'{sg}_short'][m]) * 1e4)
            say(' | '.join([name, str(int(m.sum())), f'{np.median(Bk["n"][m]):.0f}'] +
                           [fmt(s[k][m]) for k in ('gross', 'bound', 'head', 'head_unnetted', 'raw_head', 'akbas_head',
                                                   'd_raw', 'd_akbas', 'd_raw_gross', 'd_akbas_gross')] +
                           [f'{be:+.2f}', f'{res[sub][name]["nw_t_head"]:+.2f}']))
        say('\nlegs (mean hedged return of each leg, bps/day; a long-short = long - short) and twin costs:')
        for name, a, b in periods:
            if name not in res[sub]:
                continue
            r = res[sub][name]
            say(f'{name}: twin long {r["twin_long_bps"]:+.2f} short {r["twin_short_bps"]:+.2f}; raw long {r["raw_long_bps"]:+.2f} '
                f'short {r["raw_short_bps"]:+.2f}; akbas long {r["akbas_long_bps"]:+.2f} short {r["akbas_short_bps"]:+.2f}; '
                f'twin costs netted {r["tc_bps"]:.2f}, un-netted {r["tcu_bps"]:.2f} bps/day; legs {r["leg_size"]:.0f}')
        # volume split (post hoc)
        say('\nvolume split of the twin book (post hoc): day volume >= 2x its 60-day median vs below; gross long-short '
            'within each group, bps/day (t); the 10 bps penalty would subtract 20 more.')
        for name, a, b in periods:
            m = B.period_mask(days, Bk['t'], a, b)
            if m.sum() < 3:
                continue
            share = Bk['vs_hi_n'][m].sum() / (Bk['vs_hi_n'][m].sum() + Bk['vs_lo_n'][m].sum())
            vh, vl = B.tstat(Bk['vs_hi'][m]), B.tstat(Bk['vs_lo'][m])
            res[sub][name]['volume_split'] = dict(share_hi=float(share), hi=vh, lo=vl,
                                                  hi_long=float(np.nanmean(Bk['vs_hi_long'][m]) * 1e4), hi_short=float(np.nanmean(Bk['vs_hi_short'][m]) * 1e4),
                                                  lo_long=float(np.nanmean(Bk['vs_lo_long'][m]) * 1e4), lo_short=float(np.nanmean(Bk['vs_lo_short'][m]) * 1e4))
            say(f'{name}: positions with a volume spike {share:.0%}; spike {vh[0] * 1e4:+.2f} ({vh[1]:+.2f}, {vh[2]} days), '
                f'no spike {vl[0] * 1e4:+.2f} ({vl[1]:+.2f}, {vl[2]} days); legs: spike long {res[sub][name]["volume_split"]["hi_long"]:+.1f} '
                f'short {res[sub][name]["volume_split"]["hi_short"]:+.1f}, no spike long {res[sub][name]["volume_split"]["lo_long"]:+.1f} '
                f'short {res[sub][name]["volume_split"]["lo_short"]:+.1f}')
    # sensitivity: intraday bad prints kept, clipped
    Bc = B.books(R, 'all', clip=True)
    say('\nsensitivity: stock days with an intraday bad print kept with the return clipped at +-0.5 (all stocks)')
    for name, a, b in periods:
        m = B.period_mask(days, Bc['t'], a, b)
        if m.sum() < 3:
            continue
        say(f'{name}: head {fmt(B.net(Bc, "twin")[m])}; such positions in the twin legs: {int(Bc["twin_idbad"][m].sum())}, '
            f'raw legs {int(Bc["raw_idbad"][m].sum())}, akbas legs {int(Bc["akbas_idbad"][m].sum())}')
    say(f'rows with an intraday-only bad print: {int(R["idbad"].sum())} of {len(R["idbad"])}')
    # calendar years (all stocks)
    Bk, s = daily['all']
    yrs = np.array([int(c2eval.iso(days[t])[:4]) for t in Bk['t']])
    say('\ncalendar years, all stocks, bps/day (t): head, bound, raw (a) head, akbas (b) head, twin-(a), twin-(b)')
    res['years'] = {}
    for y in np.unique(yrs):
        m = (yrs == y) & B.period_mask(days, Bk['t'], periods[0][1], periods[-1][2] if tag == 'new' else None)
        if m.sum() < 3:
            continue
        res['years'][int(y)] = {k: B.tstat(s[k][m]) for k in ('head', 'bound', 'raw_head', 'akbas_head', 'd_raw', 'd_akbas')}
        say(y, int(m.sum()), *[fmt(s[k][m]) for k in ('head', 'bound', 'raw_head', 'akbas_head', 'd_raw', 'd_akbas')])
    # verdict (old only)
    if tag == 'old':
        r = res['all']
        t1 = r['2008..2019 (Test 1)']
        t3 = r['2020..2025-06 (Test 3)']
        p1 = t1['head']['t'] >= 3 and t1['head']['mean_bps'] > 0
        p2 = t1['d_raw']['t'] >= 2 and t1['d_akbas']['t'] >= 2
        p3 = t3['head']['mean_bps'] > 0
        say(f'\nTest 1 (2008-2019 headline mean, t >= 3 net): {t1["head"]["mean_bps"]:+.2f} bps/day, t {t1["head"]["t"]:+.2f} -> {"PASS" if p1 else "FAIL"}')
        say(f'Test 2 (paired vs (a) and (b), t >= 2 both): {t1["d_raw"]["mean_bps"]:+.2f} (t {t1["d_raw"]["t"]:+.2f}), '
            f'{t1["d_akbas"]["mean_bps"]:+.2f} (t {t1["d_akbas"]["t"]:+.2f}) -> {"PASS" if p2 else "FAIL"}')
        say(f'Test 3a (2020-2025-06 headline positive): {t3["head"]["mean_bps"]:+.2f} bps/day (t {t3["head"]["t"]:+.2f}) -> {"PASS" if p3 else "FAIL"}')
        res['verdict'] = dict(test1=p1, test2=p2, test3a=p3)
    (B.OUT / f'tests_{tag}.txt').write_text('\n'.join(LINES) + '\n')
    (B.OUT / f'tests_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    with open(B.OUT / f'daily_{tag}.tsv', 'w') as fh:
        Bk, s = daily['all']
        Bf, sf = daily['500']
        pos = {t: j for j, t in enumerate(Bf['t'])}
        keys = ('gross', 'bound', 'head', 'raw_head', 'akbas_head')
        fh.write('date\tn\t' + '\t'.join(keys) + '\t' + '\t'.join(f'top500_{k}' for k in keys) + '\n')
        for j, t in enumerate(Bk['t']):
            jj = pos.get(t)
            fh.write(c2eval.iso(days[t]) + f'\t{Bk["n"][j]}\t' + '\t'.join(f'{s[k][j]:.6f}' for k in keys) + '\t' +
                     '\t'.join(f'{sf[k][jj]:.6f}' if jj is not None else 'nan' for k in keys) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
