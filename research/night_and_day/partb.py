"""Night and day, Part B: the cluster lead-lag strategy's own forecasts (all-exchanges settings), split into the
overnight and intraday legs. Inputs from crates/c2-census/examples/night_and_day_cluster.rs.

    python3 -I partb.py old | new

Two books, each per decision day t, held over d = t+1:
  - the strategy's own sign book (cluster::backtest): +1/p for members of groups forecast up, -1/p down. Checked
    against research/cluster_leadlag_all-exchanges_daily.tsv on the full day. An intraday version opens and closes
    every position daily: cost 2 x c x exposure a day.
  - the decile long-short of the harness's scores (cluster::scores), as Part A's books and the judge's long-short:
    top 10% minus bottom 10% of the fit's members, equal weight; cost 4 x c a day.
Legs as in Part A (store opens and closes; members without valid legs count 0 in the sign book, as next_day_return
does, and are dropped from the decile means).
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
PERIODS = {'old': [('2008-2019', '2008-01-01', '2019-12-31'), ('2020-2025-06', '2020-01-01', '2025-06-27')],
           'new': [('unseen 2025-06-30..2026-10-19', '2025-06-30', '2026-10-19')]}[tag]
LEGS = ('on', 'in', 'full')


def main():
    P = C.load(tag, mmap=False)
    days = P['days']
    T = len(days)
    E = c2eval.load_export(C.EXPORTS[tag], arrays=('close',))
    ce = np.asarray(E['close'], np.float64)
    Wt = np.load(C.WORK / f'cluster_w_{tag}.npy')
    St = np.load(C.WORK / f'cluster_s_{tag}.npy')
    on = np.asarray(P['on'], np.float64)
    intra = np.asarray(P['intra'], np.float64)
    first = c2eval.day_index(days, PERIODS[0][1])
    end = (np.datetime64(PERIODS[-1][2]) - np.datetime64('1970-01-01')).astype(int)
    last = int(np.searchsorted(days, end, side='right')) - 1
    rec = {k: [] for k in ('d', 'sign_full_export', 'sign_on', 'sign_in', 'sign_full', 'exposure', 'turnover',
                           'ls_on', 'ls_in', 'ls_full', 'ic_on', 'ic_in', 'ic_full', 'n')}
    prev = None
    for d in range(max(first, 1), last + 1):
        t = d - 1
        w = Wt[:, t].astype(np.float64)
        mem = np.isfinite(w)
        if mem.sum() < C.MIN_NAMES:
            prev = None
            continue
        ww = np.where(mem, w, 0.0)
        # full day from export closes, as cluster::next_day_return (0 without both closes)
        with np.errstate(all='ignore'):
            nd = ce[:, d] / ce[:, t] - 1
        nd[~np.isfinite(nd) | ~(ce[:, t] > 0)] = 0.0
        lon, lin = on[:, d], intra[:, d]
        V = np.isfinite(lon)
        g_on = np.where(V, np.expm1(lon), 0.0)
        g_in = np.where(V, np.expm1(lin), 0.0)
        g_full = np.where(V, np.expm1(lon + lin), 0.0)
        rec['d'].append(d)
        rec['sign_full_export'].append(float(ww @ nd))
        rec['sign_on'].append(float(ww @ g_on))
        rec['sign_in'].append(float(ww @ g_in))
        rec['sign_full'].append(float(ww @ g_full))
        rec['exposure'].append(float(np.abs(ww).sum()))
        rec['turnover'].append(float(np.abs(ww - prev).sum()) if prev is not None else np.nan)
        prev = ww
        # decile long-short of the harness's scores
        s = St[:, t]
        rows = np.nonzero(np.isfinite(s))[0]
        n = len(rows)
        o = rows[np.argsort(-s[rows], kind='stable')]
        m = int(round(C.DECILE * n))
        L, S = o[:m], o[-m:]
        LV, SV, UV = L[V[L]], S[V[S]], rows[V[rows]]
        rec['n'].append(n)
        for lg, x in (('on', lon), ('in', lin), ('full', lon + lin)):
            g = np.expm1(x)
            rec[f'ls_{lg}'].append(g[LV].mean() - g[SV].mean() if len(LV) and len(SV) else np.nan)
            rec[f'ic_{lg}'].append(C.spearman(s[UV], x[UV]))
    R = {k: np.array(v) for k, v in rec.items()}
    np.savez(C.WORK / f'partb_series_{tag}.npz', **R)
    res = {}
    L_ = []
    Wl = L_.append
    # check against the published daily series
    if tag == 'old':
        tsv = Path('/home/bhoult/programming/correlation/research/cluster_leadlag_all-exchanges_daily.tsv')
        rows = [l.rstrip('\n').split('\t') for l in open(tsv)][1:]
        pub = {r[0]: float(r[1]) for r in rows}
        mine = {iso(days[d]): v for d, v in zip(R['d'], R['sign_full_export'])}
        common_days = sorted(set(pub) & set(mine))
        diff = np.array([abs(pub[k] - mine[k]) for k in common_days])
        res['check'] = {'days': len(common_days), 'max_abs_diff': float(diff.max()), 'mean_abs_diff': float(diff.mean())}
        Wl(f"Check: the sign book's full-day return from export closes against research/cluster_leadlag_all-exchanges_daily.tsv: "
           f"{len(common_days)} common days, max |diff| {diff.max():.2e}, mean |diff| {diff.mean():.2e}\n")
    da = days[R['d']]
    for nm, a, b in PERIODS:
        lo = (np.datetime64(a) - np.datetime64('1970-01-01')).astype(int)
        hi = (np.datetime64(b) - np.datetime64('1970-01-01')).astype(int)
        sel = (da >= lo) & (da <= hi)
        r = {'days': int(sel.sum()), 'exposure': float(np.nanmean(R['exposure'][sel])),
             'turnover': float(np.nanmean(R['turnover'][sel])), 'names': float(np.median(R['n'][sel]))}
        for lg in LEGS:
            mu, tt, _ = C.tstat(R[f'sign_{lg}'][sel])
            r[f'sign_{lg}'] = (mu * 1e4, tt)
            mu, tt, _ = C.tstat(R[f'ls_{lg}'][sel])
            x = R[f'ls_{lg}'][sel]
            r[f'ls_{lg}'] = (mu * 1e4, tt, C.nw_t(x))
            r[f'ls_{lg}_net2'] = C.tstat(x - 8e-4)[:2]
            r[f'ls_{lg}_net2'] = (r[f'ls_{lg}_net2'][0] * 1e4, r[f'ls_{lg}_net2'][1])
            r[f'ls_{lg}_net5'] = (C.tstat(x - 20e-4)[0] * 1e4, C.tstat(x - 20e-4)[1])
            r[f'ic_{lg}'] = C.tstat(R[f'ic_{lg}'][sel])[:2]
        mu, tt, _ = C.tstat(R['sign_full_export'][sel])
        r['sign_full_export'] = (mu * 1e4, tt)
        res[nm] = r
    Wl('**Sign book** (the strategy\'s own: +-1/p; gross bps a day, t). Intraday version costs 2 x c x exposure; the '
       'close-to-close book costs c x turnover.\n')
    Wl('| Period | Days | Exposure | Turnover (close to close) | Overnight | Intraday | Full day (store) | Full day (export, as published) | Break-even intraday bps/side | Break-even close-to-close bps/side |')
    Wl('|---|---|---|---|---|---|---|---|---|---|')
    for nm, _, _ in PERIODS:
        r = res[nm]
        Wl(f"| {nm} | {r['days']} | {r['exposure']:.2f} | {r['turnover']:.2f} | {r['sign_on'][0]:+.2f} ({r['sign_on'][1]:+.2f}) | "
           f"{r['sign_in'][0]:+.2f} ({r['sign_in'][1]:+.2f}) | {r['sign_full'][0]:+.2f} ({r['sign_full'][1]:+.2f}) | "
           f"{r['sign_full_export'][0]:+.2f} ({r['sign_full_export'][1]:+.2f}) | {r['sign_in'][0] / (2 * r['exposure']):+.2f} | "
           f"{r['sign_full_export'][0] / r['turnover']:+.2f} |")
    Wl('\n**Decile long-short of the harness\'s scores** (top 10% minus bottom 10% of the fit\'s members; gross bps a day '
       '(t; NW t); net at 2 and 5 bps per side = gross - 4c; break-even = gross / 4)\n')
    Wl('| Period | Median names | Book | Gross | Net 2 bps (t) | Net 5 bps (t) | Break-even bps/side | Mean IC (t) |')
    Wl('|---|---|---|---|---|---|---|---|')
    for nm, _, _ in PERIODS:
        r = res[nm]
        for lg in LEGS:
            g = r[f'ls_{lg}']
            Wl(f"| {nm} | {r['names']:.0f} | {lg} | {g[0]:+.2f} ({g[1]:+.2f}; {g[2]:+.2f}) | {r[f'ls_{lg}_net2'][0]:+.2f} ({r[f'ls_{lg}_net2'][1]:+.2f}) | "
               f"{r[f'ls_{lg}_net5'][0]:+.2f} ({r[f'ls_{lg}_net5'][1]:+.2f}) | {g[0] / 4:+.2f} | {r[f'ic_{lg}'][0]:+.4f} ({r[f'ic_{lg}'][1]:+.2f}) |")
    txt = '\n'.join(L_) + '\n'
    (C.OUT / f'partb_{tag}.md').write_text(txt)
    (C.OUT / f'partb_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    print(txt)


if __name__ == '__main__':
    main()
