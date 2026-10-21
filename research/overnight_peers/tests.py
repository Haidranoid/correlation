"""Overnight peers: Tests 1-3 of experiments/-1.6-overnight-peers.md, the monthly long-short books of the three
peer types, on one export.

    python3 -I tests.py old      # 2009-2019 and 2020-2025-06
    python3 -I tests.py new      # the unseen months 2025-06-30 .. 2026-10-19 (once, at the end)

At each calendar day t (export days 252 + 21k): rank the residual signal over universe stocks with one; long the top
round(10% x n), short the bottom round(10% x n), equal weight; hold close t -> close t+21. A member's return is its
export close at t+21 (or its last close in t+1..t+21) over its close at t, minus the export market's simple return
over t+1..t+21 (exp of the summed daily market log returns, minus 1). Gross = long mean - short mean. Costs: 5 bps
per side on turnover, sum |w_new - w_old| over both sides against the previous calendar day's book (the first book
pays its entries). Periods by the first held day t+1; a period needs t+21 inside the export.

Diagnostics (not tests): the same books on the raw peer mean (no OLS on own return), the long and short legs
against the universe's mean, and the monthly rank IC of the residual with the holding return.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval
iso = c2eval.iso

tag = sys.argv[1] if len(sys.argv) > 1 else 'old'
PERIODS = {'old': [('2009-2019', '2009-01-01', '2019-12-31'), ('2020-2025-06', '2020-01-01', '2025-06-27')],
           'new': [('unseen', '2025-06-30', '2026-10-19')]}[tag]


def dnum(s):
    return int((np.datetime64(s) - np.datetime64('1970-01-01')).astype(int))


def main():
    M = C.meta(tag)
    days, rebal, uni = M['days'], M['rebal'], M['uni']
    T = len(days)
    E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'market'))
    close = E['close']
    mkt = np.asarray(E['market'], np.float64)
    ks = C.k_range(tag, M)
    PE = C.load_peers(tag)
    S21 = np.load(C.WORK / f'S21_{tag}.npy', mmap_mode='r')
    RES = {ty: np.load(C.WORK / f'resid_{ty}_{tag}.npy', mmap_mode='r') for ty in C.TYPES}
    books = {f'{ty}{v}': [] for ty in C.TYPES for v in ('', '_raw')}
    prev = {b: (set(), set(), 0, 0) for b in books}
    months = []
    n_last_close = 0
    for k in ks:
        t = int(rebal[k])
        if t + C.H > T - 1:
            break
        c0 = np.asarray(close[:, t], np.float64)
        cw = np.asarray(close[:, t + 1:t + C.H + 1], np.float64)
        fin = np.isfinite(cw)
        last = np.where(fin.any(1), C.H - 1 - np.argmax(fin[:, ::-1], 1), -1)
        c1 = np.where(last >= 0, cw[np.arange(len(c0)), np.maximum(last, 0)], np.nan)
        m_simple = np.expm1(np.nansum(mkt[t + 1:t + C.H + 1]))
        with np.errstate(all='ignore'):
            rel = c1 / c0 - 1.0 - m_simple
        months.append({'k': k, 't': t, 'date': iso(days[t]), 'first_held': iso(days[t + 1]),
                       'end': iso(days[t + C.H]), 'market': float(m_simple)})
        for ty in C.TYPES:
            rows, peer, corr = PE[ty][k]
            has = peer[:, 0] >= 0
            rows_h, peer_h = rows[has], peer[has]
            res = np.asarray(RES[ty][rows_h, t], np.float64)
            # raw peer mean (diagnostic): the same peers, no OLS on own return
            _, pm, own = C.residual_block(S21, rows_h, peer_h, np.array([t]))
            for v, sig in (('', res), ('_raw', np.where(np.isfinite(res), pm[:, 0], np.nan))):
                b = f'{ty}{v}'
                ok = np.isfinite(sig)
                r_ok, s_ok = rows_h[ok], sig[ok]
                n = len(r_ok)
                rec = {'n': n}
                if n < C.MIN_NAMES:
                    books[b].append(rec | {'gross': np.nan, 'net': np.nan})
                    continue
                o = r_ok[np.argsort(-s_ok, kind='stable')]
                nd = int(round(C.DECILE * n))
                L, S = o[:nd], o[-nd:]
                rl, rs, ru = rel[L], rel[S], rel[r_ok]
                if ty == 'on' and v == '':
                    n_last_close += int(((last[L] >= 0) & (last[L] < C.H - 1)).sum() + ((last[S] >= 0) & (last[S] < C.H - 1)).sum())
                gl, gs, gu = np.nanmean(rl), np.nanmean(rs), np.nanmean(ru)
                gross = gl - gs
                pL, pS, nL, nS = prev[b]
                wl, ws = 1.0 / len(L), 1.0 / len(S)
                Ls, Ss = set(L.tolist()), set(S.tolist())
                turn = 0.0
                for new, old, wn, wo in ((Ls, pL, wl, 1.0 / nL if nL else 0.0), (Ss, pS, ws, 1.0 / nS if nS else 0.0)):
                    turn += sum(abs(wn - (wo if x in old else 0.0)) for x in new) + sum(wo for x in old if x not in new)
                prev[b] = (Ls, Ss, len(L), len(S))
                cost = C.COST * turn
                okr = np.isfinite(ru)
                ic = float(np.corrcoef(C.avg_rank(s_ok[okr]), C.avg_rank(ru[okr]))[0, 1]) if okr.sum() > 10 else np.nan
                rec.update({'nl': int(np.isfinite(rl).sum()), 'ns': int(np.isfinite(rs).sum()), 'gross': float(gross),
                            'turn': float(turn), 'cost': float(cost), 'net': float(gross - cost),
                            'long_x': float(gl - gu), 'short_x': float(gs - gu), 'ic': ic})
                books[b].append(rec)
        if k % 20 == 0:
            C.log(months[-1]['date'], {b: round(books[b][-1].get('net', np.nan) * 100, 2) for b in books})
    first_held = np.array([dnum(m['first_held']) for m in months])
    yrs = np.array([int(m['first_held'][:4]) for m in months])
    out = {'tag': tag, 'months': months, 'books': books, 'n_members_exited_early_on_book': n_last_close}

    def col(b, key):
        return np.array([r.get(key, np.nan) for r in books[b]], np.float64)

    summ = {}
    for name, a, z in PERIODS:
        sel = (first_held >= dnum(a)) & (first_held <= dnum(z))
        ps = {'months': int(sel.sum()), 'from': months[int(np.argmax(sel))]['date'],
              'to': months[len(sel) - 1 - int(np.argmax(sel[::-1]))]['end']}
        for b in books:
            g, n_, tu, ic = col(b, 'gross')[sel], col(b, 'net')[sel], col(b, 'turn')[sel], col(b, 'ic')[sel]
            lx, sx = col(b, 'long_x')[sel], col(b, 'short_x')[sel]
            mg, tg, _ = C.tstat(g)
            mn, tn, nn = C.tstat(n_)
            mi, ti, _ = C.tstat(ic)
            ml, tl, _ = C.tstat(lx)
            ms, tsx, _ = C.tstat(sx)
            ps[b] = {'gross': mg, 't_gross': tg, 'net': mn, 't_net': tn, 'n': nn, 'turn': float(np.nanmean(tu)),
                     'ic': mi, 't_ic': ti, 'long_x': ml, 't_long_x': tl, 'short_x': ms, 't_short_x': tsx,
                     'names': float(np.nanmean(col(b, 'n')[sel])), 'per_side': float(np.nanmean(col(b, 'nl')[sel]))}
        for a_, b_ in (('on', 'whole'), ('on', 'intra'), ('intra', 'whole'), ('on_raw', 'whole_raw')):
            d = col(a_, 'net')[sel] - col(b_, 'net')[sel]
            md, td, nd = C.tstat(d)
            ps[f'{a_}-{b_}'] = {'net': md, 't_net': td, 'n': nd}
        summ[name] = ps
    by_year = {}
    for y in np.unique(yrs):
        sel = yrs == y
        by_year[int(y)] = {b: C.tstat(col(b, 'net')[sel])[:2] for b in ('on', 'intra', 'whole')}
        by_year[int(y)]['on-whole'] = C.tstat(col('on', 'net')[sel] - col('whole', 'net')[sel])[:2]
        by_year[int(y)]['months'] = int(sel.sum())
    out['summary'] = summ
    out['by_year'] = by_year
    (C.OUT / f'tests_{tag}.json').write_text(json.dumps(out, indent=1, default=float))
    # readable tables
    L = []
    for name, ps in summ.items():
        L.append(f'## {name}: {ps["months"]} months, books formed {ps["from"]} .. last held {ps["to"]}')
        L.append('| Book | Names with a signal | Per side | Gross % a month (t) | Turnover | Net % a month (t) | Long vs universe (t) | Short vs universe (t) | Rank IC (t) |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for b in books:
            s = ps[b]
            L.append(f'| {b} | {s["names"]:.0f} | {s["per_side"]:.0f} | {100 * s["gross"]:+.3f} ({s["t_gross"]:.2f}) | {s["turn"]:.2f} | '
                     f'{100 * s["net"]:+.3f} ({s["t_net"]:.2f}) | {100 * s["long_x"]:+.3f} ({s["t_long_x"]:.2f}) | '
                     f'{100 * s["short_x"]:+.3f} ({s["t_short_x"]:.2f}) | {s["ic"]:+.4f} ({s["t_ic"]:.2f}) |')
        for a_ in ('on-whole', 'on-intra', 'intra-whole', 'on_raw-whole_raw'):
            s = ps[a_]
            L.append(f'- {a_}: net {100 * s["net"]:+.3f}% a month (t {s["t_net"]:.2f}, {s["n"]} months)')
        L.append('')
    L.append('## By year (net % a month, t)')
    L.append('| Year | Months | Overnight | Intraday | Whole day | Overnight - whole |')
    L.append('|---|---|---|---|---|---|')
    for y, r in by_year.items():
        L.append(f'| {y} | {r["months"]} | ' + ' | '.join(f'{100 * r[b][0]:+.2f} ({r[b][1]:.2f})' for b in ('on', 'intra', 'whole', 'on-whole')) + ' |')
    L.append(f'\nOvernight book members exited before t+21 (last close earlier): {n_last_close}')
    (C.OUT / f'tests_{tag}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
