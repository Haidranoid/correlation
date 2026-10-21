"""Silent-peer drift: the money test (the judge) and the plan's own long-short.

    PY="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -- env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 -I"
    $PY money.py old     # walk-forward 2006-07 .. 2024-25, uniform and volatility-matched random accounts
    $PY money.py new     # the unseen months 2025-06-30 .. 2026-10-19 (once, at the end)

Judge scores: for each silent event (A, t), score = sign(A's move) x |A's move in sd60 units| = xA / sd60, written on
days t+1 .. t+20 (so the account buys at the close of t+1 and holds through t+21). If a later silent event of the same
stock overlaps, the later event's score replaces the earlier one from its own t+1 on. NaN elsewhere.

The plan's long-short: long silent good-news events, short silent bad-news events, 20-day hold from the close of t+1 to
the close of t+21, net of 10 bps per round trip.
  - per trade: P&L = sign x (A's simple return - the equal-weight market's compounded simple return over the window);
  - calendar time: each day, the long leg is the equal-weight mean actual return of the active good-news positions and
    the short leg that of the bad-news ones; LS = long - short (a leg with no positions is replaced by the market).
    Costs: 5 bps x (entries + exits) / positions, per leg, per day.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

build = analyze.build
c2eval = analyze.c2eval
iso = c2eval.iso
OUT = analyze.OUT
HOLD = 20
RT_BPS = 10.0


def judge_years(days, first='2006-06-28'):
    out = []
    y = int(first[:4])
    while True:
        a = c2eval.day_index(days, f'{y}-06-28')
        b = c2eval.day_index(days, f'{y + 1}-06-28')
        if a >= len(days) - 1:
            break
        out.append((a, min(b, len(days))))
        y += 1
        if b >= len(days):
            break
    return out


def make_scores(ev, rows, N, T):
    sc = np.full((N, T), np.nan, np.float32)
    o = rows[np.argsort(ev['d'][rows], kind='stable')]
    for i in o:
        a, d = int(ev['A'][i]), int(ev['d'][i])
        z = ev['xA'][i] / ev['sdA'][i]
        sc[a, d + 1:min(d + 1 + HOLD, T)] = z
    return sc


def plan_long_short(tag, ev, rows, years):
    Ex = c2eval.load_export(analyze.EXPORT[tag], arrays=('close',))
    c = Ex['close']
    T = c.shape[1]
    U = np.load(build.BIG / f'univ_{tag}.npz')
    ms = np.nan_to_num(U['msimple'])
    Mlvl = np.concatenate([[1.0], np.cumprod(1 + ms)])  # Mlvl[d+1] = level at the close of d
    A, d = ev['A'][rows], ev['d'][rows]
    sgn = np.sign(ev['xA'][rows])
    s0, s1 = d + 1, d + 1 + HOLD
    ok = s1 <= T - 1
    s0c, s1c = np.minimum(s0, T - 1), np.minimum(s1, T - 1)
    with np.errstate(all='ignore'):
        rA = np.asarray(c[A, s1c], np.float64) / np.asarray(c[A, s0c], np.float64) - 1
        rM = Mlvl[s1c + 1] / Mlvl[s0c + 1] - 1
    pnl = sgn * (rA - rM)
    pnl[~ok | ~np.isfinite(pnl)] = np.nan
    f = np.isfinite(pnl)
    L = []
    W = L.append
    res = {'per_trade': {}, 'calendar': {}}
    W('**Per trade** (equal dollar, market-hedged, 20 days from the close of t+1; net = gross - 10 bps per round trip; '
      't clustered by event date)\n')
    W('| Year | Trades | Long (good news) | Short (bad news) | Gross bps per trade (t) | Net bps per trade |')
    W('|---|---|---|---|---|---|')
    pos = 0; ny = 0
    for a, b in years:
        s = f & (d >= a) & (d < b)
        if s.sum() < 3:
            continue
        ny += 1
        r = analyze.mean_t(np.where(s, pnl, np.nan), d)
        g = r['mean'] * 1e4
        pos += (g - RT_BPS) > 0
        W(f'| {iso(Ex["days"][a])[:4]}-{iso(Ex["days"][b - 1])[2:4]} | {s.sum():,} | {(s & (sgn > 0)).sum():,} | '
          f'{(s & (sgn < 0)).sum():,} | {g:+.1f} ({r["t"]:+.2f}) | {g - RT_BPS:+.1f} |')
        res['per_trade'][iso(Ex['days'][a])] = dict(n=int(s.sum()), gross_bps=g, t=r['t'])
    s = f & (d >= years[0][0]) & (d < years[-1][1])
    r = analyze.mean_t(np.where(s, pnl, np.nan), d, analyze.month_of(Ex['days'], d))
    rl = analyze.mean_t(np.where(s & (sgn > 0), pnl, np.nan), d)
    rs = analyze.mean_t(np.where(s & (sgn < 0), pnl, np.nan), d)
    W(f'| **All** | {s.sum():,} | {(s & (sgn > 0)).sum():,} | {(s & (sgn < 0)).sum():,} | **{r["mean"] * 1e4:+.1f} '
      f'({r["t"]:+.2f}; month t {r["t_month"]:+.2f})** | **{r["mean"] * 1e4 - RT_BPS:+.1f}** |')
    W(f'\nYears net-positive after 10 bps: {pos} of {ny}. Long side alone {rl["mean"] * 1e4:+.1f} bps (t {rl["t"]:+.2f}); '
      f'short side alone {rs["mean"] * 1e4:+.1f} bps (t {rs["t"]:+.2f}).\n')
    res['per_trade']['all'] = dict(n=r['n'], gross_bps=r['mean'] * 1e4, t=r['t'], t_month=r['t_month'],
                                   long_bps=rl['mean'] * 1e4, long_t=rl['t'], short_bps=rs['mean'] * 1e4, short_t=rs['t'],
                                   years_net_pos=int(pos), n_years=ny)
    # calendar time
    lo, hi = years[0][0], years[-1][1]
    nL = np.zeros(T); nS = np.zeros(T); sumL = np.zeros(T); sumS = np.zeros(T)
    tvL = np.zeros(T); tvS = np.zeros(T)
    for a_, d_, sg in zip(A, d, sgn):
        u0, u1 = d_ + 2, min(d_ + 1 + HOLD, T - 1)  # held over days t+2 .. t+21
        if u0 > T - 1:
            continue
        cc = np.asarray(c[a_, u0 - 1:u1 + 1], np.float64)
        with np.errstate(all='ignore'):
            rr = cc[1:] / cc[:-1] - 1
        rr[~np.isfinite(rr)] = 0.0
        uu = np.arange(u0, u1 + 1)
        if sg > 0:
            nL[uu] += 1; sumL[uu] += rr; tvL[u0] += 1; tvL[u1] += 1
        else:
            nS[uu] += 1; sumS[uu] += rr; tvS[u0] += 1; tvS[u1] += 1
    with np.errstate(all='ignore'):
        legL = np.where(nL > 0, sumL / nL, ms)
        legS = np.where(nS > 0, sumS / nS, ms)
        cost = 0.5 * RT_BPS * 1e-4 * (np.where(nL > 0, tvL / nL, 0) + np.where(nS > 0, tvS / nS, 0))
    active = (nL + nS) > 0
    ls = np.where(active, legL - legS, 0.0)
    W('**Calendar time** ($1 long the good-news leg, $1 short the bad-news leg, equal weight within each leg; days with '
      'no positions count as zero)\n')
    W('| Period | Days with positions | Mean positions (long / short) | Gross % a year | Net % a year | Sharpe (gross) | t (daily) |')
    W('|---|---|---|---|---|---|---|')
    for nm, (a, b) in [('all years', (lo, hi))] + [(f'{iso(Ex["days"][a])[:4]}-{iso(Ex["days"][b - 1])[2:4]}', (a, b)) for a, b in years]:
        sl = slice(a, b)
        g = ls[sl]; net = (ls - cost)[sl]
        if len(g) < 5:
            continue
        sh = g.mean() / g.std(ddof=1) * np.sqrt(252) if g.std() > 0 else np.nan
        t = g.mean() / g.std(ddof=1) * np.sqrt(len(g)) if g.std() > 0 else np.nan
        W(f'| {nm} | {int(active[sl].sum())} of {len(g)} | {nL[sl].mean():.1f} / {nS[sl].mean():.1f} | {100 * 252 * g.mean():+.1f} | '
          f'{100 * 252 * net.mean():+.1f} | {sh:+.2f} | {t:+.2f} |')
        res['calendar'][nm] = dict(gross_pct_yr=100 * 252 * g.mean(), net_pct_yr=100 * 252 * net.mean(), sharpe=sh, t=t)
    return '\n'.join(L), res


def main(tag):
    ev = analyze.load_events(tag)
    days = ev['days']
    cls, size, _ = analyze.classify(ev)
    T = len(days)
    U = np.load(build.BIG / f'univ_{tag}.npz')
    N = len(U['ids'])
    if tag == 'old':
        start = c2eval.day_index(days, '2006-01-01')
        years = judge_years(days)
    else:
        start = c2eval.day_index(days, '2025-06-30')
        years = [(start, T)]
    rows = np.where(cls == 0)[0]  # every silent event; the judge reads only its own test days
    sc = make_scores(ev, rows, N, T)
    print('scored stock-days', int(np.isfinite(sc).sum()), 'positive', int((sc > 0).sum()), flush=True)
    sub = build.BIG.name
    path = c2eval.save_scores(sc, f'{sub}/silent_scores_{tag}', export=analyze.EXPORT[tag])
    del sc
    L = []
    W = L.append
    res = {'judge': {}}
    W(f'## Money test ({tag} export)\n')
    W(f'Scored stock-days: silent events, score = xA / sd60 on t+1 .. t+20.\n')
    for match in ('uniform', 'volatility'):
        if tag == 'old':
            txt, js = c2eval.judge(path, rebalance=1, random_match=match)
        else:
            txt, js = c2eval.judge(path, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=1,
                                   random_match=match)
        (OUT / f'judge_{tag}_{match}.txt').write_text(txt)
        cor = c2eval.change_over_random(js)
        cor1 = c2eval.change_over_random(js, 1)
        cor2 = c2eval.change_over_random(js, 2)
        r = dict(change=cor['change'], se=cor['se'], ahead=cor['ahead'], years=cor['years'], per_year=cor['per_year'],
                 change_5=cor1['change'], change_20=cor2['change'])
        if tag == 'old':
            r['summary'] = c2eval.summary_line(js)
            r['summary_json'] = js['summary']
            r['year_rows'] = [dict(start=y['start'], picks=y['costs'][0]['final_pct'], rand=y['costs'][0]['random_median_pct'],
                                   p=y['costs'][0]['p'], ls=y['ls_bps_per_day'], ls_t=y['ls_t'], pairs=y['pairs'],
                                   candidates=y['candidates']) for y in js['years']]
            r['skipped'] = js.get('skipped')
            W(f'**Judge, {match} random accounts:** {r["summary"]}; at 5 bps {cor1["change"]:+.1f}, at 20 bps {cor2["change"]:+.1f}.\n')
        else:
            rep = js['report']
            r['report'] = dict(strategy=rep['strategy'], ls_bps_per_day=rep['ls_bps_per_day'], ls_t=rep['ls_t'],
                               ic_ret_t=rep['ic_ret_t'], benchmarks=rep['benchmarks'], pairs=rep['pairs'])
            st = rep['strategy']
            W(f'**Judge, {match} random accounts, one holdout:** ' + '; '.join(
                f'{int(s["cost_bps"])} bps: {s["final_pct"]:+.1f}% vs random median {s["random_median_pct"]:+.1f}% (p {s["p"]:.3f})'
                for s in st) + f'; change over random {cor["change"]:+.1f}; long-short {rep["ls_bps_per_day"]:+.2f} bps/day '
                f't {rep["ls_t"]:+.2f}; rank IC (actual returns) t {rep["ic_ret_t"]:+.2f}; scored stocks {rep["pairs"]}.\n')
        res['judge'][match] = r
        print(match, cor, flush=True)
    if tag == 'old':
        W('| Year | Picks, 0 bps | Random median (uniform) | p | Random median (volatility) | p | Long-short bps/day (t) | Scored stocks |')
        W('|---|---|---|---|---|---|---|---|')
        vol_rows = {y['start']: y for y in res['judge']['volatility']['year_rows']}
        for y in res['judge']['uniform']['year_rows']:
            v = vol_rows.get(y['start'], {})
            W(f'| {y["start"][:4]}-{str(int(y["start"][:4]) + 1)[2:]} | {y["picks"]:+.1f}% | {y["rand"]:+.1f}% | {y["p"]:.3f} | '
              f'{v.get("rand", np.nan):+.1f}% | {v.get("p", np.nan):.3f} | {y["ls"]:+.1f} ({y["ls_t"]:+.2f}) | {y["pairs"]} |')
        W('')
    W('### The plan\'s own long-short\n')
    txt, r = plan_long_short(tag, ev, np.where((cls == 0) & (ev['d'] >= years[0][0]) & (ev['d'] < years[-1][1]))[0], years)
    W(txt)
    res['long_short'] = r
    out = '\n'.join(L)
    (OUT / f'money_{tag}.md').write_text(out + '\n')
    (OUT / f'money_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    print(out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'old')
