"""Descriptive tables for the .md: peers per firm by year, hidden and visible peers' correlations, and example
text-peer lists for well-known firms at one date (a sanity check of the text peers).

    python3 -I report.py        # writes report.txt here
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import (CAP, CORR_W, HERE, MIN_OVERLAP, PEERS, WORK, c2eval, cik_rows, epoch_of, np, universe)  # noqa: E402
from build import returns  # noqa: E402

EXAMPLE_DATE = '2016-07-01'
EXAMPLES = ['AAPL', 'MSFT', 'INTC', 'JPM', 'XOM', 'CVX', 'PFE', 'MRK', 'UNH', 'HD', 'F', 'BA', 'NKE', 'DAL', 'KO',
            'WMT', 'SBUX', 'GS']


def main():
    E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close', 'dollar_volume', 'market'))
    days, tick = E['days'], E['tickers']
    tw = c2eval.twins(E)
    rebal = tw['rebal']
    uni, big = universe(E, tw)
    lines = []
    # 1. peers per firm by year, at the formation days (universe firms with a current-year filing)
    F = dict(np.load(WORK / 'formation_counts.npz'))
    z = np.load(WORK / 'panel.npz')
    P = {k: z[k] for k in z.files}
    form_days = np.unique(F['t'])
    inP = np.isin(P['t'], form_days)
    yrF = np.array([c2eval.iso(days[t])[:4] for t in F['t']])
    yrP = np.array([c2eval.iso(days[t])[:4] for t in P['t'][inP]])
    lines.append('Peers per firm at the formation days, by calendar year (universe firms with a current-year filing). '
                 'Columns: formations | firms | text peers (top 50 of all filings above the cutoff), mean / median | '
                 'mapped text peers with both filings dated | of them in the universe | with a correlation | '
                 'firms with >= 3 hidden (share) | among those: peers mean, hidden mean / median | '
                 'mean correlation, hidden / visible peers')
    rows_out = []
    for y in sorted(set(yrF)):
        m = yrF == y
        mp = yrP == y
        nf = len(np.unique(F['t'][m]))
        ok = F['n_hidden'][m] >= 3
        nP = P['n_peers'][inP][mp]
        hP = P['n_hidden'][inP][mp]
        ch = P['corr_h'][inP][mp]
        cv = P['corr_v'][inP][mp]
        r = (y, nf, m.sum() / nf, F['n_text'][m].mean(), np.median(F['n_text'][m]), F['n_mapped'][m].mean(),
             F['n_univ'][m].mean(), F['n_corr'][m].mean(), ok.sum() / nf, ok.mean(), nP.mean(), hP.mean(),
             np.median(hP), ch.mean(), cv.mean())
        rows_out.append(r)
        lines.append(f'{r[0]} | {r[1]} | {r[2]:.0f} | {r[3]:.1f} / {r[4]:.0f} | {r[5]:.1f} | {r[6]:.1f} | {r[7]:.1f} | '
                     f'{r[8]:.0f} ({r[9]:.0%}) | {r[10]:.1f}, {r[11]:.1f} / {r[12]:.0f} | {r[13]:.2f} / {r[14]:.2f}')
    m = np.ones(len(F['t']), bool)
    lines.append(f"all | {len(form_days)} | {len(F['t']) / len(form_days):.0f} | {F['n_text'].mean():.1f} / "
                 f"{np.median(F['n_text']):.0f} | {F['n_mapped'].mean():.1f} | {F['n_univ'].mean():.1f} | "
                 f"{F['n_corr'].mean():.1f} | {(F['n_hidden'] >= 3).sum() / len(form_days):.0f} "
                 f"({(F['n_hidden'] >= 3).mean():.0%}) | {P['n_peers'][inP].mean():.1f}, {P['n_hidden'][inP].mean():.1f} / "
                 f"{np.median(P['n_hidden'][inP]):.0f} | {P['corr_h'][inP].mean():.2f} / {P['corr_v'][inP].mean():.2f}")
    lines.append(f"every day (not just formations): {len(P['t'])} firm-days; mean correlation hidden {P['corr_h'].mean():.3f}, "
                 f"visible {P['corr_v'].mean():.3f}; firm-days where the hidden mean is below 0: "
                 f"{(P['corr_h'] < 0).mean():.1%}")
    # 2. example peer lists
    d = c2eval.day_index(days, EXAMPLE_DATE)
    fy = epoch_of(days[d])
    k = int(np.searchsorted(rebal, d, side='right')) - 1
    U = uni[k]
    mr, r21, cff = returns(E)
    del cff
    zz = np.load(PEERS / f'{fy}.npz')
    cik = zz['cik']
    pos = {int(c): i for i, c in enumerate(cik)}
    cr = cik_rows()
    row_cik = {}
    for c, rs in cr.items():
        for r in rs:
            row_cik[str(tick[r])] = c
    lines.append('')
    lines.append(f'Example text peers on {c2eval.iso(days[d])} (fiscal year {fy} text; top-2% cutoff '
                 f'{float(zz["cutoff"]):.3f}). Each peer: ticker, similarity, 252-day correlation of market-relative '
                 f'returns, H = hidden / V = visible. Peers outside the ticker map (delisted, merged, private) are counted '
                 f'but not listed; peers outside the liquid universe are marked (out).')
    for t in EXAMPLES:
        c = row_cik.get(t)
        i = pos.get(c) if c is not None else None
        if i is None:
            lines.append(f'  {t}: no FY{fy} filing with >= 1,000 Item 1 words in the corpus')
            continue
        me = cr[c][0]
        msk = (zz['pi'] == i) & (zz['rank_all'] < CAP)
        js, sims = zz['pj'][msk], zz['sim'][msk]
        n_text = len(js)
        n_unmapped = int((~zz['mapped'][js]).sum())
        items = []
        for j, s in zip(js, sims):
            if not zz['mapped'][j]:
                continue
            for r in cr[int(cik[j])]:
                if not U[r]:
                    items.append((str(tick[r]), float(s), np.nan, 'out'))
                    continue
                x = mr[me, d - CORR_W + 1:d + 1].astype(np.float64)
                y = mr[r, d - CORR_W + 1:d + 1].astype(np.float64)
                ok = np.isfinite(x) & np.isfinite(y)
                cc = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= MIN_OVERLAP else np.nan
                items.append((str(tick[r]), float(s), cc, ''))
        cs = np.array([x[2] for x in items if np.isfinite(x[2])])
        med = np.median(cs) if len(cs) else np.nan
        parts = []
        for tk, s, cc, tag in items:
            if tag == 'out':
                parts.append(f'{tk} {s:.2f} (out)')
            elif np.isfinite(cc):
                parts.append(f'{tk} {s:.2f} {cc:+.2f} {"H" if cc < med else "V"}')
            else:
                parts.append(f'{tk} {s:.2f} (no corr)')
        lines.append(f'  {t} ({"in" if U[me] else "NOT in"} universe; {n_text} text peers, {n_unmapped} unmapped): '
                     + '; '.join(parts))
    (HERE / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
