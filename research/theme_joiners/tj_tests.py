"""Step 2: the Fama-MacBeth regressions, the residualized book, the sanity-check tables, the scores file and the
judge (run after tj_build.py).

    python3 -I -B tj_tests.py old     # tests 1-3 on 2006-03 .. 2025-06, the walk-forward judge; results_old.txt
    python3 -I -B tj_tests.py new     # the unseen months, once, at the end; results_new.txt

Per refit, on the stocks of the 1,500 with J and both own returns (S):
  percentiles over S: pJ, p63 (own 63-day market-relative return), p121 (own 12-1 return); dummies hot, cold;
  regressions on S with an outcome: y ~ 1 + pJ  (no controls)  and  y ~ 1 + pJ + p63 + p121 + hot + cold;
  y = the next 21 days' market-relative log return (close t -> t+21 minus the export's market);
  book: pJ residualized on 1 + p63 + p121 + hot + cold over S; long the top decile, short the bottom decile, equal
  weight, simple returns close t -> t+21, cost 5 bps x the weight traded (both legs, against the previous refit's book);
  score for the judge: the residual's percentile over S on days t .. next refit - 1.
"""
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from tj_common import (BIG, COST_BPS, EXPORTS, HERE, MIN_OBS, P1, P2, REPORT_DATES_NEW, REPORT_DATES_OLD,  # noqa: E402
                       UNSEEN, big, c2eval, fm, iso, ols, pct)

os.environ.setdefault('RAYON_NUM_THREADS', '4')
OUT = []


def say(s=''):
    print(s, flush=True)
    OUT.append(s)


def cross_sections(D):
    rows = []
    prev_w = {}
    for rf in D['refits']:
        J, o63, o121 = rf['J'], rf['own63'], rf['own121']
        S = np.flatnonzero(np.isfinite(J) & np.isfinite(o63) & np.isfinite(o121))
        r = dict(k=rf['k'], t0=rf['t0'], t1=rf['t1'], n=len(S), S=S)
        if len(S) < MIN_OBS:
            rows.append(r)
            continue
        pJ, p63, p121 = pct(J[S]), pct(o63[S]), pct(o121[S])
        hot, cold = rf['hot'][S].astype(float), rf['cold'][S].astype(float)
        one = np.ones(len(S))
        Xc = np.column_stack([one, p63, p121, hot, cold])
        _, resid = ols(pJ, Xc)
        presid = pct(resid)
        r.update(pJ=pJ, presid=presid)
        # diagnostics (no outcome)
        r['rho_J_63'] = np.corrcoef(pJ, p63)[0, 1]
        r['rho_J_121'] = np.corrcoef(pJ, p121)[0, 1]
        r['J_hot'] = np.nanmean(J[S][hot > 0])
        r['J_cold'] = np.nanmean(J[S][cold > 0])
        r['J_other'] = np.nanmean(J[S][(hot == 0) & (cold == 0)])
        r['dH_mean'] = np.nanmean(rf['dH'][S])
        r['dC_mean'] = np.nanmean(rf['dC'][S])
        # the book's weights (formation, no outcome)
        nd = len(S) // 10
        o = np.argsort(presid, kind='stable')
        longs, shorts = S[o[-nd:]], S[o[:nd]]
        w = {int(j): 1.0 / nd for j in longs}
        for j in shorts:
            w[int(j) + 10 ** 7] = -1.0 / nd           # separate keys for the short leg
        U = rf['U']
        wkeys = {(int(U[j % 10 ** 7]), j >= 10 ** 7): v for j, v in w.items()}
        traded = sum(abs(wkeys.get(key, 0.0) - prev_w.get(key, 0.0)) for key in set(wkeys) | set(prev_w))
        prev_w = wkeys
        r['traded'] = traded
        r['longs'], r['shorts'] = longs, shorts
        # outcome regressions
        y = rf['y'][S]
        ok = np.isfinite(y)
        r['n_y'] = int(ok.sum())
        if ok.sum() >= MIN_OBS:
            b0, _ = ols(y[ok], np.column_stack([one, pJ])[ok])
            b1, _ = ols(y[ok], np.column_stack([one, pJ, p63, p121, hot, cold])[ok])
            r['b_nc'] = b0[1] * 1e4
            r['b_full'] = b1 * 1e4
            # reported only: the beta-adjusted J
            Jb = rf['Jb'][S]
            okb = ok & np.isfinite(Jb)
            pJb = np.full(len(S), np.nan)
            pJb[np.isfinite(Jb)] = pct(Jb[np.isfinite(Jb)])
            bb0, _ = ols(y[okb], np.column_stack([one, pJb])[okb])
            bb1, _ = ols(y[okb], np.column_stack([one, pJb, p63, p121, hot, cold])[okb])
            r['bb_nc'], r['bb_full'] = bb0[1] * 1e4, bb1[1] * 1e4
            ys = rf['ysimple']
            gl = np.nanmean(ys[longs])
            gs = np.nanmean(ys[shorts])
            r['gross'] = (gl - gs) * 1e4
            r['cost'] = COST_BPS * traded
            r['net'] = r['gross'] - r['cost']
            r['long'], r['short'] = gl * 1e4, gs * 1e4
        rows.append(r)
    return rows


def fmt_fm(vals, unit='bps'):
    m, se, t, n = fm(vals)
    return f'{m:+.1f} {unit} (se {se:.1f}, t {t:+.2f}, {n} refits)'


def period_rows(rows, days, period):
    return [r for r in rows if 'b_nc' in r and period[0] <= iso(days[r['t0']]) <= period[1]]


def summary(rows, days, period, name):
    rs = period_rows(rows, days, period)
    if not rs:
        say(f'{name}: no refits')
        return {}
    bn = [r['b_nc'] for r in rs]
    bf = [r['b_full'][1] for r in rs]
    net = [r['net'] for r in rs]
    gross = [r['gross'] for r in rs]
    out = dict(nc=fm(bn), full=fm(bf), net=fm(net), gross=fm(gross),
               bb_nc=fm([r['bb_nc'] for r in rs]), bb_full=fm([r['bb_full'] for r in rs]),
               controls=[fm([r['b_full'][i] for r in rs]) for i in range(6)],
               cost=float(np.mean([r['cost'] for r in rs])), n=len(rs),
               first=iso(days[rs[0]['t0']]), last=iso(days[rs[-1]['t0']]),
               pos_full=float(np.mean(np.array(bf) > 0)), pos_net=float(np.mean(np.array(net) > 0)),
               long=fm([r['long'] for r in rs]), short=fm([r['short'] for r in rs]))
    say(f'### {name}: {out["n"]} refits {out["first"]} .. {out["last"]}')
    say(f'  J coefficient, no controls:   {fmt_fm(bn)}')
    say(f'  J coefficient, with controls: {fmt_fm(bf)}  (positive in {out["pos_full"]:.0%} of refits)')
    nm = ['intercept', 'pJ', 'p63', 'p12-1', 'hot', 'cold']
    say('  full model, every coefficient (bps per 21 days; percentiles: bottom -> top): ' +
        '; '.join(f'{nm[i]} {c[0]:+.1f} (t {c[2]:+.2f})' for i, c in enumerate(out['controls'])))
    say(f'  reported only, beta-adjusted J: no controls {fmt_fm([r["bb_nc"] for r in rs])}; '
        f'with controls {fmt_fm([r["bb_full"] for r in rs])}')
    say(f'  book (residualized J, top minus bottom decile, 21 days): gross {fmt_fm(gross)}; '
        f'mean cost {out["cost"]:.1f} bps; net {fmt_fm(net)} (positive in {out["pos_net"]:.0%})')
    say(f'    legs (simple 21-day return): long {out["long"][0]:+.1f} bps, short {out["short"][0]:+.1f} bps; '
        f'net a year (x12) {out["net"][0] * 12:+.0f} bps')
    return out


def by_year(rows, days, period=('0000', '9999')):
    rows = [r for r in rows if period[0] <= iso(days[r['t0']]) <= period[1]]
    say('### By year (refit year): J coefficient no controls / with controls; book net (bps per 21 days)')
    say('| Year | Refits | No controls | t | With controls | t | Book net | t |')
    say('|---|---|---|---|---|---|---|---|')
    ys = sorted({iso(days[r['t0']])[:4] for r in rows if 'b_nc' in r})
    for y in ys:
        rs = [r for r in rows if 'b_nc' in r and iso(days[r['t0']])[:4] == y]
        a, b, c = fm([r['b_nc'] for r in rs]), fm([r['b_full'][1] for r in rs]), fm([r['net'] for r in rs])
        say(f'| {y} | {len(rs)} | {a[0]:+.1f} | {a[2]:+.2f} | {b[0]:+.1f} | {b[2]:+.2f} | {c[0]:+.1f} | {c[2]:+.2f} |')


def diagnostics(rows, days, period, name):
    rs = [r for r in rows if 'pJ' in r and period[0] <= iso(days[r['t0']]) <= period[1]]
    if not rs:
        return
    g = lambda key: np.nanmean([r[key] for r in rs])  # noqa: E731
    say(f'### What J looks like, {name} ({len(rs)} refits; no outcome used)')
    say(f'  stocks with J per refit: mean {np.mean([r["n"] for r in rs]):.0f}, min {min(r["n"] for r in rs)}')
    say(f'  mean cross-sectional correlation of pct(J) with pct(own 63-day return) {g("rho_J_63"):+.3f}, '
        f'with pct(own 12-1) {g("rho_J_121"):+.3f}')
    say(f'  mean J: hot-group members {g("J_hot"):+.3f}, cold-group members {g("J_cold"):+.3f}, others {g("J_other"):+.3f}')
    say(f'  mean change in corr with H {g("dH_mean"):+.3f}, with C {g("dC_mean"):+.3f}')


def group_name(U, lab, g, tickers, n=4):
    idx = np.flatnonzero(lab == g)               # U is sorted by median dollar volume, largest first
    return ', '.join(str(tickers[U[i]]) for i in idx[:n])


def report_dates(D, rows, dates):
    days, tickers = D['days'], D['tickers']
    byk = {r['k']: r for r in rows}
    for dt in dates:
        d0 = c2eval.day_index(days, dt)
        rf = next((x for x in D['refits'] if x['t0'] >= d0), None)
        if rf is None:
            continue
        U, lab = rf['U'], rf['lab']
        say(f'### {iso(days[rf["t0"]])} (refit k={rf["k"]})')
        say('| | Group (largest members) | Twin-fund label | Members | 63-day market-relative return |')
        say('|---|---|---|---|---|')
        for name, gs in (('Hot', rf['hot_g']), ('Cold', rf['cold_g'])):
            for g in gs:
                say(f'| {name} | {group_name(U, lab, g, tickers)} | {rf["glabel"][g]} | {(lab == g).sum()} | '
                    f'{rf["gscore"][g] * 100:+.1f}% |')
        J = rf['J']
        o = np.argsort(-np.where(np.isfinite(J), J, -np.inf))[:10]
        hl = {g: 'hot' for g in rf['hot_g']} | {g: 'cold' for g in rf['cold_g']}
        say('')
        say('Top 10 joiners:')
        say('| Stock | J | Change in corr with H | with C | Own group (largest members) | Own 63-day return |')
        say('|---|---|---|---|---|---|')
        for i in o:
            g = lab[i]
            tag = f' ({hl[g]})' if g in hl else ''
            say(f'| {tickers[U[i]]} | {J[i]:+.2f} | {rf["dH"][i]:+.2f} | {rf["dC"][i]:+.2f} | '
                f'{group_name(U, lab, g, tickers, 3)}{tag} | {rf["own63"][i] * 100:+.1f}% |')
        say('')


def scores_array(D, rows):
    s = np.full((D['N'], D['T']), np.nan, np.float32)
    for rf, r in zip(D['refits'], rows):
        if 'presid' not in r:
            continue
        rows_ = rf['U'][r['S']]
        s[rows_, rf['t0']:rf['t1']] = r['presid'][:, None]
    return s


def judge_report(path, tag, rebalance, **kw):
    res = {}
    for rm in ('uniform', 'volatility'):
        txt, js = c2eval.judge(path, rebalance=rebalance, random_match=rm, **kw)
        (HERE / f'judge_{tag}_{rm}.txt').write_text(txt)
        res[rm] = js
        cor = c2eval.change_over_random(js)
        if js.get('years'):
            say(f'  {rm}: {c2eval.summary_line(js)}')
            say(f'    per year: {cor["per_year"]}')
            for ci in (1, 2):
                c = c2eval.change_over_random(js, ci)
                say(f'    at cost level {ci}: change over random {c["change"]:+.1f} (ahead {c["ahead"]}/{c["years"]})')
        else:
            st = js['report']['strategy']
            say(f'  {rm}: ' + '; '.join(f"{int(c.get('cost_bps', 0))} bps: picks {c['final_pct']:+.1f}%, random median "
                                        f"{c['random_median_pct']:+.1f}%, p {c.get('p', float('nan')):.3f}" for c in st))
            say(f'    change over random (0 bps): {cor["change"]:+.1f}; ls {js["report"].get("ls_bps_per_day", float("nan")):.2f} '
                f'bps/day (t {js["report"].get("ls_t", float("nan")):+.2f}); IC ret t {js["report"].get("ic_ret_t", float("nan")):+.2f}')
            spy = [b for b in js['report'].get('benchmarks', []) if 'SPY' in str(b)]
            if spy:
                say(f'    benchmark: {spy[0]}')
    return res


def main(tag):
    D = pickle.load(open(BIG / tag / 'table.pkl', 'rb'))
    days = D['days']
    rows = cross_sections(D)
    say(f'# Theme joiners: {tag} export ({len(rows)} refits, {iso(days[rows[0]["t0"]])} .. {iso(days[rows[-1]["t0"]])})')
    say('Units: coefficients are bps of 21-day market-relative log return per bottom-to-top move of the percentile.')
    say('')
    res = {}
    if tag == 'old':
        diagnostics(rows, days, ('2006-01-01', '2025-06-27'), '2006-03 .. 2025-06')
        say('')
        say('## Test 1 (2006-03 .. 2019): J coefficient with controls, positive with t >= 3')
        res['p1'] = summary(rows, days, P1, '2006-03 .. 2019')
        say('')
        say('## Test 3 (2020 .. 2025-06): same sign')
        res['p2'] = summary(rows, days, P2, '2020 .. 2025-06')
        say('')
        say('## Reported only: 2006-03 .. 2025-06')
        res['all'] = summary(rows, days, ('2006-01-01', '2025-06-27'), '2006-03 .. 2025-06')
        say('')
        by_year(rows, days)
        say('')
        say('## Hot and cold groups, and the top 10 joiners, at dates fixed in advance')
        report_dates(D, rows, REPORT_DATES_OLD)
        say('## Money test: the judge, rebalance=21, walk-forward (score = residualized J percentile)')
        s = scores_array(D, rows)
        say(f'  scored stock-days {np.isfinite(s).sum():,}; days with a score {np.isfinite(s).any(0).sum()}')
        path = c2eval.save_scores(s, 'theme_joiners/scores_old')
        del s
        res['judge'] = judge_report(path, 'old', 21)
    else:
        diagnostics(rows, days, UNSEEN, 'the unseen months')
        say('')
        say('## Test 3, the unseen months (refits from 2025-06-30 with a full 21-day outcome): same sign')
        res['unseen'] = summary(rows, days, UNSEEN, 'the unseen months')
        say('')
        by_year(rows, days, UNSEEN)
        say('')
        say('## Hot and cold groups, and the top 10 joiners')
        report_dates(D, rows, REPORT_DATES_NEW)
        say('## The judge on the unseen months (holdout from 2025-06-30, rebalance=21)')
        s = scores_array(D, rows)
        path = c2eval.save_scores(s, 'theme_joiners/scores_new', export=c2eval.EXPORT_NEW)
        del s
        res['judge'] = judge_report(path, 'new', 21, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30')
    (HERE / f'results_{tag}.txt').write_text('\n'.join(OUT) + '\n')
    with open(BIG / tag / 'results.pkl', 'wb') as fh:
        pickle.dump(dict(rows=[{k: v for k, v in r.items() if k not in ('S', 'pJ', 'presid', 'longs', 'shorts')}
                               for r in rows], res=res), fh)


if __name__ == '__main__':
    main(sys.argv[1])
