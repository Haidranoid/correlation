"""Overnight auctions: post hoc diagnostics of the money test (added after seeing it; no rule or pass/fail changed).

The account's picks sit at the 84th volatility percentile of the ranked set and the mean change over random is
carried by 2008, 2009 and 2020, on survivor-only data. So, on the same days:
  1. volatility-matched random accounts (judge-style): for each pick, a distinct random stock from the same decile
     of trailing 63-day close-to-close volatility (through t-1) within that day's ranked set;
  2. the money test inside the 500 largest (picks and random draws from the 500's ranked set);
  3. robust aggregates of the yearly differences (median; mean without 2008, 2009 and 2020);
  4. the account's intraday (open -> close of t+1) and close-to-close legs on the same positions, against the ranked
     set's equal-weight mean;
  5. the share of picks with close_{t-1} < $5, and the account restricted to stocks with close_{t-1} >= $5 (picks
     and random draws from that set).
Random draws use seed common.SEED + 1 (vol-matched), + 2 (500 largest), + 3 (>= $5).

    python3 -I diag.py old|new     # -> out/diag_<tag>.md, .json
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import build as B  # noqa: E402
import money as M  # noqa: E402
np = C.np
c2eval = C.c2eval
iso = c2eval.iso

tag = sys.argv[1] if len(sys.argv) > 1 else 'old'


def uniform_draws(rng, n, k=C.PICKS):
    return np.argpartition(rng.random((C.N_RANDOM, n)), k - 1, axis=1)[:, :k] if n > k else np.tile(np.arange(n), (C.N_RANDOM, 1))


def main():
    P = B.load(tag)
    days = P['days']
    E = c2eval.load_export(C.EXPORTS[tag], arrays=('close',))
    close = E['close']
    if tag == 'old':
        d_lo = c2eval.day_index(days, C.FIRST)
        d_hi = int(np.searchsorted(days, C.iso_to_day('2025-06-27'), side='right')) - 1
    else:
        d_lo = c2eval.day_index(days, '2025-06-30')
        d_hi = int(np.searchsorted(days, C.iso_to_day('2026-10-19'), side='right')) - 1
    rv, rb, rp = (np.random.default_rng(C.SEED + i) for i in (1, 2, 3))
    c0, rebal = P['c0'], P['rebal']
    ST = P['S'].T
    onT, inT, okT = P['on_s'].T, P['in_s'].T, P['ok_s'].T
    rec = {k: [] for k in ('d', 'acc', 'accB', 'accP', 'acc_in', 'acc_full', 'uni_on', 'uni_in', 'uni_full', 'lt5', 'nB', 'nP')}
    RV, RB, RP = [], [], []
    for d in range(d_lo, d_hi + 1):
        t = d - 1
        j = d - c0
        k = C.kidx(rebal, t - 1)
        if j < 1 or k < 0:
            continue
        U = P['uni'][k] & (P['fund_row'][k] >= 0)
        s = np.asarray(ST[t], np.float64)
        rows = np.nonzero(U & np.isfinite(s))[0]
        n = len(rows)
        if n < C.MIN_NAMES:
            continue
        ok, on, inn = np.asarray(okT[j]), np.asarray(onT[j], np.float64), np.asarray(inT[j], np.float64)
        r = np.where(ok, np.expm1(on), np.nan)
        ri = np.where(ok, np.expm1(inn), np.nan)
        rf = np.where(ok, np.expm1(on + inn), np.nan)
        order = rows[np.argsort(-s[rows], kind='stable')]
        top = order[:C.PICKS]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            acc = np.nanmean(r[top]) if np.isfinite(r[top]).any() else 0.0
            # 1. volatility deciles within the ranked set (trailing 63 days through t-1)
            cw = np.asarray(close[rows, t - 63:t], np.float64)
            with np.errstate(all='ignore'):
                lr = np.log(cw[:, 1:] / cw[:, :-1])
            lr[~np.isfinite(lr) | (np.abs(lr) > C.MAX_LOG)] = np.nan
            vol = np.nanstd(lr, 1)
            vol = np.where(np.isfinite(vol), vol, np.nanmedian(vol))
            rank = np.argsort(np.argsort(vol, kind='stable'), kind='stable')
            dec = np.minimum(rank * 10 // n, 9)
            pos = np.searchsorted(rows, top)
            pdec = dec[pos]
            rr = r[rows]
            draws = np.empty((C.N_RANDOM, len(top)), np.int64)
            col = 0
            for q in np.unique(pdec):
                m = int((pdec == q).sum())
                members = np.nonzero(dec == q)[0]
                if m >= len(members):
                    draws[:, col:col + m] = members[:m][None, :]
                else:
                    draws[:, col:col + m] = members[np.argpartition(rv.random((C.N_RANDOM, len(members))), m - 1, axis=1)[:, :m]]
                col += m
            RV.append(np.nan_to_num(np.nanmean(rr[draws], 1)).astype(np.float32))
            # 2. the 500 largest
            rowsB = rows[P['big'][k][rows]]
            topB = rowsB[np.argsort(-s[rowsB], kind='stable')][:C.PICKS]
            accB = np.nanmean(r[topB]) if np.isfinite(r[topB]).any() else 0.0
            RB.append(np.nan_to_num(np.nanmean(r[rowsB][uniform_draws(rb, len(rowsB))], 1)).astype(np.float32))
            # 5. close_{t-1} >= $5
            pc = np.asarray(close[rows, t - 1], np.float64)   # the close of t-1 (known before t's cutoff)
            okp = np.nan_to_num(pc) >= 5.0
            rowsP = rows[okp]
            topP = rowsP[np.argsort(-s[rowsP], kind='stable')][:C.PICKS]
            accP = np.nanmean(r[topP]) if np.isfinite(r[topP]).any() else 0.0
            RP.append(np.nan_to_num(np.nanmean(r[rowsP][uniform_draws(rp, len(rowsP))], 1)).astype(np.float32))
            lt5 = float((~okp[pos]).mean())
            # 4. legs of the same positions
            rec['acc_in'].append(np.nanmean(ri[top]) if np.isfinite(ri[top]).any() else 0.0)
            rec['acc_full'].append(np.nanmean(rf[top]) if np.isfinite(rf[top]).any() else 0.0)
            rec['uni_on'].append(np.nanmean(r[rows])); rec['uni_in'].append(np.nanmean(ri[rows]))
            rec['uni_full'].append(np.nanmean(rf[rows]))
        rec['d'].append(d); rec['acc'].append(acc); rec['accB'].append(accB); rec['accP'].append(accP)
        rec['lt5'].append(lt5); rec['nB'].append(len(rowsB)); rec['nP'].append(len(rowsP))
    D = {k: np.array(v) for k, v in rec.items()}
    D['rnd_vol'], D['rndB'], D['rndP'] = np.array(RV), np.array(RB), np.array(RP)
    np.savez(C.WORK / f'diag_daily_{tag}.npz', **D)
    C.log('daily done')
    yrs = np.array([int(iso(x)[:4]) for x in days[D['d']]])
    if tag == 'old':
        groups = [(str(y), yrs == y) for y in np.unique(yrs)]
    else:
        groups = [('unseen 2025-06-30..2026-10-19', np.ones(len(yrs), bool))]
    res = {'tag': tag, 'seeds': {'vol': C.SEED + 1, 'big': C.SEED + 2, 'price5': C.SEED + 3}}
    for label, key, rkey in (('vol_matched', 'acc', 'rnd_vol'), ('500_largest', 'accB', 'rndB'), ('close_ge_5', 'accP', 'rndP')):
        for cl, cost in (('0bps', 0.0), ('2bps', 4e-4)):
            rows = M.yearly(D, days, key, rkey, cost, groups)
            res[f'{label}|{cl}'] = {'rows': rows}
            if tag == 'old':
                res[f'{label}|{cl}']['all'] = M.agg(rows)
                res[f'{label}|{cl}']['2008-2022'] = M.agg(rows, set(range(2008, 2023)))
    # 3. robust aggregates of the headline (uniform) and the variants
    if tag == 'old':
        head = json.loads((C.OUT / 'money_old.json').read_text())
        rob = {}
        for label, rows in (('uniform (headline)', head['0bps']['rows']), ('vol_matched', res['vol_matched|0bps']['rows']),
                            ('500_largest', res['500_largest|0bps']['rows']), ('close_ge_5', res['close_ge_5|0bps']['rows'])):
            dd = np.array([r['diff'] for r in rows])
            yy = np.array([int(r['period']) for r in rows])
            ex = ~np.isin(yy, [2008, 2009, 2020])
            rob[label] = {'mean': float(dd.mean()), 'median': float(np.median(dd)), 'mean_ex_2008_2009_2020': float(dd[ex].mean()),
                          'se_ex': float(dd[ex].std(ddof=1) / np.sqrt(ex.sum())), 'ahead_ex': int((dd[ex] > 0).sum()), 'years_ex': int(ex.sum())}
        res['robust'] = rob
    # 4. legs (bps a day; account minus ranked-set mean, t)
    legs = {}
    pers = [('2008-2019', '2008-01-01', '2019-12-31'), ('2020-2025-06', '2020-01-01', '2025-06-27')] if tag == 'old' else C.PERIODS_NEW
    da = days[D['d']]
    for nm, a, b in pers:
        sel = (da >= C.iso_to_day(a)) & (da <= C.iso_to_day(b))
        row = {}
        for lg, ak, uk in (('overnight', 'acc', 'uni_on'), ('intraday', 'acc_in', 'uni_in'), ('close_to_close', 'acc_full', 'uni_full')):
            x = D[ak][sel] - D[uk][sel]
            m, tt, _ = C.tstat(x)
            row[lg] = {'account_bps': float(np.mean(D[ak][sel])) * 1e4, 'ranked_mean_bps': float(np.mean(D[uk][sel])) * 1e4,
                       'excess_bps': m * 1e4, 't': tt}
        legs[nm] = row
    res['legs'] = legs
    # 5. share of picks under $5 by year
    res['lt5_by_year'] = {str(y): float(D['lt5'][yrs == y].mean()) for y in np.unique(yrs)}
    (C.OUT / f'diag_{tag}.json').write_text(json.dumps(res, indent=1))
    L = []
    W = L.append
    W(f'# Overnight auctions: post hoc diagnostics of the money test ({tag})\n')
    for label in ('vol_matched', '500_largest', 'close_ge_5'):
        for cl in ('0bps', '2bps'):
            r = res[f'{label}|{cl}']
            W(f'## {label}, {cl}\n')
            if tag == 'old':
                for nm in ('all', '2008-2022'):
                    a = r[nm]
                    W(f"- {nm}: change over random {a['change']:+.2f} pts/yr (se {a['se']:.2f}), ahead {a['ahead']}/{a['years']}, p<0.1 in {a['p_lt_0_1']}")
            W('\n| Period | Days | Account % | Median random % | Difference | p | Random 5-95% |')
            W('|---|---|---|---|---|---|---|')
            for x in r['rows']:
                W(f"| {x['period']} | {x['days']} | {x['account_pct']:+.1f} | {x['random_median_pct']:+.1f} | {x['diff']:+.1f} | {x['p']:.3f} | {x['random_p5']:+.1f}..{x['random_p95']:+.1f} |")
            W('')
    if tag == 'old':
        W('## Robust aggregates of the yearly differences (0 bps; chosen after seeing the by-year table)\n')
        W('| Comparison | Mean | Median | Mean without 2008, 2009, 2020 (se; ahead) |')
        W('|---|---|---|---|')
        for label, r in res['robust'].items():
            W(f"| {label} | {r['mean']:+.1f} | {r['median']:+.1f} | {r['mean_ex_2008_2009_2020']:+.1f} ({r['se_ex']:.1f}; {r['ahead_ex']}/{r['years_ex']}) |")
    W('\n## The account\'s legs on the same positions (bps a day; account, ranked-set mean, excess with t)\n')
    W('| Period | Overnight | Intraday | Close to close |')
    W('|---|---|---|---|')
    for nm, row in legs.items():
        W(f'| {nm} | ' + ' | '.join(f"{row[lg]['account_bps']:+.1f}, {row[lg]['ranked_mean_bps']:+.1f}, {row[lg]['excess_bps']:+.1f} ({row[lg]['t']:+.2f})"
                                     for lg in ('overnight', 'intraday', 'close_to_close')) + ' |')
    W('\n## Share of picks with close_{t-1} < $5, by year\n')
    W(' '.join(f'{y}: {v:.2f}' for y, v in res['lt5_by_year'].items()))
    txt = '\n'.join(L) + '\n'
    (C.OUT / f'diag_{tag}.md').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main()
