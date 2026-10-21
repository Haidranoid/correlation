"""The intraday-bar check of the twin gap fade (experiments/-16.2-twin-gap-fade.md, "Independent check with intraday
bars"), following the rules fixed there and out/decisions.txt.

    python -I analyze.py      # reads WORK/twin_gap_fade_check/{positions.npz, y5m.csv, y1m.csv, y1d.csv}
                              # -> out/check.txt, out/check.json, out/daily_check.tsv

Positions come from positions.py (the headline book as tested, reproduced exactly). Every Yahoo price is put on the
store's scale with k = median(store close / Yahoo daily close) per symbol and segment between Yahoo's dividend and
split ex-dates (the store holds Stooq's adjusted prices). Exit: the store's close.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/bhoult/programming/correlation/research/twin_gap_fade')
import books as B  # noqa: E402  (read-only: constants, fund_cost, tstat)

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
BIG = Path.home() / '.local/share/correlation2/experiments/twin_gap_fade_check'
LAST_STORE_DAY = '2026-10-19'
CLOSE_TOL = 1e-3
SCALE_TOL = 0.02  # D13 (decided after seeing results): intraday last-bar close within 2% of Yahoo's daily close
TIMES = {'5m': ['first', '0935', '0945', '1000'], '1m': ['first', '0931', '0935', '0945', '1000']}
LABEL = {'a': '(a) store open', 'first': "(b) Yahoo's first-bar open", '0931': '9:31', '0935': '(c) 9:35',
         '0945': '(d) 9:45', '1000': '(e) 10:00'}
LINES = []
RES = {}


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def iso(d):
    return str(np.datetime64('1970-01-01') + np.timedelta64(int(d), 'D'))


def fmt(x, d=2):
    m, t, n = B.tstat(x)
    return f'{m * 1e4:+.{d}f} ({t:+.2f})'


# ------------------------------------------------------------------------------------------------ inputs

def load():
    P = np.load(BIG / 'positions.npz')
    days = P['days']
    smap = dict(zip(P['stk_i'].tolist(), P['stk_sym'].tolist()))
    fmap = dict(zip(P['fnd_f'].tolist(), P['fnd_sym'].tolist()))
    rk = pd.DataFrame({k[3:]: P[k] for k in P.files if k.startswith('rk_')})
    for c in ('s_open', 's_close', 's_prevclose', 'f_open', 'f_close', 'f_prevclose', 'sd', 'gf', 'beta', 'g', 'z'):
        rk[c] = rk[c].astype(np.float64)
    rk['date'] = [iso(days[t]) for t in rk['t']]
    rk['ssym'] = rk['i'].map(smap)
    rk['fsym'] = rk['f'].map(fmap)
    books = {}
    for tag in ('all', 'b500'):
        d = pd.DataFrame({k[len(tag) + 1:]: P[k] for k in P.files if k.startswith(tag + '_')})
        d = d.merge(rk[['t', 'i', 'date', 'ssym', 'fsym', 's_open', 's_close', 'f_open', 'f_close', 'exchange']],
                    on=['t', 'i'], how='left', validate='one_to_one')
        assert d['ssym'].notna().all()
        books['all' if tag == 'all' else '500'] = d
    y5 = pd.read_csv(BIG / 'y5m.csv', dtype={'sym': str, 'date': str, 'offs': str})
    y1 = pd.read_csv(BIG / 'y1m.csv', dtype={'sym': str, 'date': str, 'offs': str}) if (BIG / 'y1m.csv').exists() else None
    yd = pd.read_csv(BIG / 'y1d.csv', dtype={'sym': str, 'date': str})
    return days, rk, books, y5, y1, yd


def scale_table(rk, yd):
    """Per symbol and date: k (store scale / Yahoo scale), usable (close agrees within CLOSE_TOL), event (dividend or
    split ex-date that day), k1 (segment with no event after the day through the store's last day)."""
    sc = pd.concat([rk[['ssym', 'date', 's_close']].rename(columns={'ssym': 'sym', 's_close': 'cs'}),
                    rk[['fsym', 'date', 'f_close']].rename(columns={'fsym': 'sym', 'f_close': 'cs'})]).drop_duplicates(['sym', 'date'])
    y = yd[yd['date'] <= LAST_STORE_DAY].copy()
    y['ev'] = (y['dividends'].fillna(0) > 0) | (y['stock_splits'].fillna(0) > 0)
    evd = y[y['ev']].groupby('sym')['date'].apply(lambda s: np.sort(s.values))
    df = y[['sym', 'date', 'open', 'close', 'ev', 'dividends', 'stock_splits']].merge(sc, on=['sym', 'date'], how='left')

    def seg(row_sym, dates):
        e = evd.get(row_sym)
        if e is None:
            return np.zeros(len(dates), int)
        return len(e) - np.searchsorted(e, dates, side='right')  # events strictly after the day

    df['seg'] = 0
    for s, idx in df.groupby('sym').groups.items():
        df.loc[idx, 'seg'] = seg(s, df.loc[idx, 'date'].values)
    with np.errstate(all='ignore'):
        df['ratio'] = df['cs'] / df['close']
    df['k'] = df.groupby(['sym', 'seg'])['ratio'].transform('median')
    df['usable'] = (np.abs(df['ratio'] / df['k'] - 1) <= CLOSE_TOL)
    df['k1'] = (df['seg'] == 0)
    return df.rename(columns={'open': 'yd_open', 'close': 'yd_close'})


def window_days(y, mode):
    s = y[y['sym'] == 'SPY']
    n = 78 if mode == '5m' else 390
    last = '15:55' if mode == '5m' else '15:59'
    ok = ((s['n_bars'] == n) & (s['first_time'] == '09:30') & (s['last_time'] == last) & (s['offs'] == '-04:00')
          & (s['tz'] == 'America/New_York'))
    return sorted(s.loc[ok, 'date']), s


# ------------------------------------------------------------------------------------------------ pricing

def price(pos, y, K, mode):
    """Hedged open-to-close returns of each position at every entry time of the window; flags for the common set."""
    times = TIMES[mode]
    ycols = ['first_open'] + [f'open_{t}' for t in times if t != 'first'] + ['last_close']
    ys = y[['sym', 'date'] + ycols]
    kk = K[['sym', 'date', 'k', 'usable', 'ev', 'k1', 'yd_open', 'yd_close']]
    d = pos.merge(ys.add_prefix('s_y_').rename(columns={'s_y_sym': 'ssym', 's_y_date': 'date'}), on=['ssym', 'date'], how='left')
    d = d.merge(ys.add_prefix('f_y_').rename(columns={'f_y_sym': 'fsym', 'f_y_date': 'date'}), on=['fsym', 'date'], how='left')
    d = d.merge(kk.add_prefix('s_').rename(columns={'s_sym': 'ssym', 's_date': 'date'}), on=['ssym', 'date'], how='left')
    d = d.merge(kk.add_prefix('f_').rename(columns={'f_sym': 'fsym', 'f_date': 'date'}), on=['fsym', 'date'], how='left')
    assert len(d) == len(pos)
    b = d['beta'].astype(float)
    d['h_a'] = (d['s_close'] / d['s_open'] - 1) - b * (d['f_close'] / d['f_open'] - 1)
    d['hx_a'] = (d['s_y_last_close'] * d['s_k'] / d['s_open'] - 1) - b * (d['f_y_last_close'] * d['f_k'] / d['f_open'] - 1)
    for t in times:
        c = 'first_open' if t == 'first' else f'open_{t}'
        ps, pf = d[f's_y_{c}'] * d['s_k'], d[f'f_y_{c}'] * d['f_k']
        d[f'h_{t}'] = (d['s_close'] / ps - 1) - b * (d['f_close'] / pf - 1)
        d[f'hx_{t}'] = (d['s_y_last_close'] / d[f's_y_{c}'] - 1) - b * (d['f_y_last_close'] / d[f'f_y_{c}'] - 1)
    d['has_bars'] = d['s_y_first_open'].notna() & d['f_y_first_open'].notna()
    with np.errstate(all='ignore'):
        d['scale_ok'] = ((np.abs(d['s_y_last_close'] / d['s_yd_close'] - 1) <= SCALE_TOL)
                         & (np.abs(d['f_y_last_close'] / d['f_yd_close'] - 1) <= SCALE_TOL))
    need = [f'h_{t}' for t in times] + [f'hx_{t}' for t in times] + ['hx_a']
    d['common'] = (d['has_bars'] & np.isfinite(d[need]).all(axis=1) & (d['s_usable'] == True) & (d['f_usable'] == True)  # noqa: E712
                   & (d['s_ev'] == False) & (d['f_ev'] == False) & d['scale_ok'])  # noqa: E712
    d['why_scale'] = d['has_bars'] & ~d['scale_ok']
    d['why_event'] = d['has_bars'] & ((d['s_ev'] == True) | (d['f_ev'] == True))  # noqa: E712
    d['why_close'] = d['has_bars'] & ~((d['s_usable'] == True) & (d['f_usable'] == True))  # noqa: E712
    return d


def daily(d, col, mask):
    """Per day: gross long-short of column col over positions in mask, and the netted twin cost. Days with an empty
    leg are skipped."""
    x = d[mask]
    out = {}
    for date, g in x.groupby('date'):
        L, S = g[g['side'] == 1], g[g['side'] == -1]
        if len(L) == 0 or len(S) == 0:
            continue
        gross = L[col].mean() - S[col].mean()
        fi = np.concatenate([L['f'].values, S['f'].values])
        w = np.concatenate([-L['beta'].values.astype(float) / len(L), S['beta'].values.astype(float) / len(S)])
        out[date] = (gross, B.fund_cost(fi, w), L[col].mean(), S[col].mean())
    return pd.DataFrame(out, index=['gross', 'tc', 'long', 'short']).T.sort_index()


def entry_table(d, mode, name, ref_a=None):
    """Gross, net, share of (a) kept, paired difference against (a), for every entry time, on the common set."""
    times = TIMES[mode]
    m = d['common']
    A = daily(d, 'h_a', m)
    if ref_a is not None:  # the variant: share measured against the main book's (a) on the same days
        A_ref = ref_a
    else:
        A_ref = A
    res = {}
    say(f'\n{name}: {len(A)} days, {int(m.sum())} positions in the common set')
    say('entry | gross bps/day (t) | net (t) | share of (a) kept | minus (a), gross (t) | exit at Yahoo last bar: gross (t) | long leg | short leg')
    rows = [('a', 'h_a', 'hx_a')] + [(t, f'h_{t}', f'hx_{t}') for t in times]
    for t, col, colx in rows:
        D = daily(d, col, m)
        X = daily(d, colx, m)
        common_days = D.index.intersection(A_ref.index)
        g = D.loc[common_days, 'gross']
        net = g - 2 * B.P_HEAD - D.loc[common_days, 'tc']
        share = g.mean() / A_ref.loc[common_days, 'gross'].mean()
        diff = g - A_ref.loc[common_days, 'gross']
        r = dict(days=len(common_days), gross=B.tstat(g.values), net=B.tstat(net.values), share=float(share),
                 diff=B.tstat(diff.values), exit_yahoo=B.tstat(X.loc[common_days, 'gross'].values),
                 long=float(D.loc[common_days, 'long'].mean()), short=float(D.loc[common_days, 'short'].mean()),
                 tc=float(D.loc[common_days, 'tc'].mean()))
        res[t] = r
        dtxt = '' if (t == 'a' and ref_a is None) else fmt(diff.values)
        say(f'{LABEL[t]} | {fmt(g.values)} | {fmt(net.values)} | {share:.2f} | {dtxt} | {fmt(X.loc[common_days, "gross"].values)} | '
            f'{r["long"] * 1e4:+.1f} | {r["short"] * 1e4:+.1f}')
    return res, A


def variant_positions(rk, y5, K, wd5):
    """D8: rank each day's ranked rows by z recomputed from Yahoo's 5-minute first-bar open; deciles as books.py."""
    r = rk[rk['date'].isin(wd5)].copy()
    y5f = y5[['sym', 'date', 'first_open', 'last_close']].merge(K[['sym', 'date', 'yd_close']], on=['sym', 'date'], how='left')
    with np.errstate(all='ignore'):
        bad = ~(np.abs(y5f['last_close'] / y5f['yd_close'] - 1) <= SCALE_TOL)  # D13
    y5f = y5f.assign(first_open=y5f['first_open'].where(~bad))[['sym', 'date', 'first_open']]
    r = r.merge(y5f.rename(columns={'sym': 'ssym', 'first_open': 's_p1'}), on=['ssym', 'date'], how='left')
    r = r.merge(y5f.rename(columns={'sym': 'fsym', 'first_open': 'f_p1'}), on=['fsym', 'date'], how='left')
    kk2 = K[['sym', 'date', 'k', 'usable', 'ev']]
    r = r.merge(kk2.rename(columns={'sym': 'ssym', 'k': 's_k', 'usable': 's_u', 'ev': 's_e'}), on=['ssym', 'date'], how='left')
    r = r.merge(kk2.rename(columns={'sym': 'fsym', 'k': 'f_k', 'usable': 'f_u', 'ev': 'f_e'}), on=['fsym', 'date'], how='left')
    with np.errstate(all='ignore'):
        gv = np.log(r['s_p1'] * r['s_k'] / r['s_prevclose'].astype(float))
        gfv = np.log(r['f_p1'] * r['f_k'] / r['f_prevclose'].astype(float))
        r['zv'] = (gv - r['beta'].astype(float) * gfv) / r['sd']
    okv = np.isfinite(r['zv']) & (r['s_u'] == True) & (r['f_u'] == True) & (r['s_e'] == False) & (r['f_e'] == False)  # noqa: E712
    nr, nok = len(r), int(okv.sum())
    r = r[okv].sort_values(['t', 'i'])
    vp = []
    for t, g in r.groupby('t'):
        n = len(g)
        if n < B.MIN_N:
            continue
        nl = int(round(B.FRAC * n))
        o = np.argsort(g['zv'].values, kind='stable')
        vp.append(g.iloc[o[:nl]].assign(side=1))
        vp.append(g.iloc[o[n - nl:]].assign(side=-1))
    return pd.concat(vp), nr, nok


# ------------------------------------------------------------------------------------------------ main

def main():
    days, rk, books, y5, y1, yd = load()
    K = scale_table(rk, yd)
    say('Intraday-bar check of the twin gap fade. bps a day of a $1 long / $1 short book (t).')
    # scale: does store/Yahoo close step at ex-dates?
    kk = K.dropna(subset=['ratio'])
    nseg = kk.groupby(['sym', 'seg'])['k'].first()
    say(f'\nprice scale: {kk["sym"].nunique()} symbols with store and Yahoo closes on {len(kk)} symbol-days; '
        f'symbols with a dividend or split ex-date in 2026-06-01..{LAST_STORE_DAY}: {K[K["ev"]]["sym"].nunique()}')
    say(f'  k on segments with no later event: median {nseg.xs(0, level="seg").median():.6f}, share with |k-1| > 1e-4: '
        f'{(np.abs(nseg.xs(0, level="seg") - 1) > 1e-4).mean():.3%}')
    if (nseg.index.get_level_values('seg') > 0).any():
        e = nseg[nseg.index.get_level_values('seg') > 0]
        say(f'  k on segments before an event: median {e.median():.6f}, share with |k-1| > 1e-4: {(np.abs(e - 1) > 1e-4).mean():.1%} '
            f'(the store is dividend-adjusted if this is large)')
    say(f'  symbol-days whose close disagrees with the segment k by more than {CLOSE_TOL:.1%}: {(~kk["usable"]).sum()} of {len(kk)}')
    RES['scale'] = dict(symbol_days=int(len(kk)), close_disagree=int((~kk['usable']).sum()))

    for mode, y in (('5m', y5), ('1m', y1)):
        if y is None:
            say(f'\n{mode}: no bars')
            continue
        wd, spy = window_days(y, mode)
        book_days = sorted(set(books['all']['date']))
        wd = [x for x in wd if x in set(book_days)]
        say(f'\n===== {mode} window: {len(wd)} days, {wd[0]}..{wd[-1]} =====')
        offs = y[y['date'].isin(wd)]['offs'].value_counts()
        say(f'UTC offsets of the bars on window days: {dict(offs)}; symbol-days with bars outside 09:30-16:00: '
            f'{int((y[y["date"].isin(wd)]["n_outside"] > 0).sum())}, with a 16:00 bar: {int(y[y["date"].isin(wd)]["has_1600"].sum())}')
        dropped = sorted(set(spy['date']) & set(book_days) - set(wd))
        say(f'store days with SPY bars that are not full sessions (dropped): {dropped}')
        RES[mode] = {'days': wd}
        for bk in ('all', '500'):
            pos = books[bk][books[bk]['date'].isin(wd)].copy()
            d = price(pos, y, K, mode)
            # (a) on all positions = the tested book; check against daily_new.tsv
            A_all = daily(d, 'h_a', np.ones(len(d), bool))
            A_bars = daily(d, 'h_a', d['has_bars'].values)
            n, nb, nc = len(d), int(d['has_bars'].sum()), int(d['common'].sum())
            say(f'\n--- {mode}, {bk} book ---')
            say(f'coverage: positions {n}; with bars for stock and twin {nb} ({nb / n:.1%}); in the common set {nc} ({nc / n:.1%}); '
                f'with bars but dropped for an ex-date {int(d["why_event"].sum())}, for a close disagreement {int(d["why_close"].sum())}, '
                f'for an intraday scale break (D13) {int((d["why_scale"] & ~d["why_event"] & ~d["why_close"]).sum())}, '
                f'for a missing entry bar {int((d["has_bars"] & ~d["common"] & ~d["why_event"] & ~d["why_close"] & ~d["why_scale"]).sum())}')
            ref = pd.read_csv('/home/bhoult/programming/correlation/research/twin_gap_fade/out/daily_new.tsv', sep='\t',
                              dtype={'date': str}).set_index('date')
            rc = ('gross', 'head') if bk == 'all' else ('top500_gross', 'top500_head')
            dg = np.abs(A_all['gross'] - ref.loc[A_all.index, rc[0]]).max()
            dh = np.abs(A_all['gross'] - 2 * B.P_HEAD - A_all['tc'] - ref.loc[A_all.index, rc[1]]).max()
            say(f'check: (a) on all positions vs daily_new.tsv on the {len(A_all)} window days: max |gross diff| {dg:.1e}, '
                f'max |headline diff| {dh:.1e} (the tsv has 6 decimals)')
            RES.setdefault('checks', {})[f'{mode}_{bk}'] = dict(days=len(A_all), max_gross_diff=float(dg), max_head_diff=float(dh))
            say(f'(a) on all positions (the book as tested): gross {fmt(A_all["gross"].values)}, headline '
                f'{fmt((A_all["gross"] - 2 * B.P_HEAD - A_all["tc"]).values)}; on positions with bars: gross {fmt(A_bars["gross"].values)}')
            res, A = entry_table(d, mode, f'{mode} {bk}: entry times on the common set')
            RES[mode][bk] = dict(coverage=dict(positions=n, with_bars=nb, common=nc, event=int(d['why_event'].sum()),
                                               close=int(d['why_close'].sum()), scale=int(d['why_scale'].sum())),
                                 a_all=dict(gross=B.tstat(A_all['gross'].values),
                                            head=B.tstat((A_all['gross'] - 2 * B.P_HEAD - A_all['tc']).values)),
                                 a_bars=B.tstat(A_bars['gross'].values), entries=res)
            # POST HOC (added after the first 5-minute results were seen; changes no rule): each entry time on its own
            # position set, needing only that time's bars (plus usable closes and no ex-date), paired with (a) on it
            ok0 = (d['has_bars'] & (d['s_usable'] == True) & (d['f_usable'] == True) & (d['s_ev'] == False) & (d['f_ev'] == False)  # noqa: E712
                   & d['scale_ok'])
            ph = {}
            say('post hoc: each entry time on its own position set (only its own bars required): entry | positions | gross (t) | share of (a) on that set')
            for t in TIMES[mode]:
                mt = ok0 & np.isfinite(d[f'h_{t}'])
                Dt, At = daily(d, f'h_{t}', mt), daily(d, 'h_a', mt)
                cd = Dt.index.intersection(At.index)
                ph[t] = dict(positions=int(mt.sum()), gross=B.tstat(Dt.loc[cd, 'gross'].values),
                             share=float(Dt.loc[cd, 'gross'].mean() / At.loc[cd, 'gross'].mean()))
                say(f'{LABEL[t]} | {int(mt.sum())} | {fmt(Dt.loc[cd, "gross"].values)} | {ph[t]["share"]:.2f}')
            RES[mode][bk]['post_hoc_own_sets'] = ph
            if bk == 'all':
                RES[mode]['_A_all_common'] = A
                RES[mode]['_d_all'] = d
                missing = d[~d['has_bars']]
                ms = sorted(set(missing.loc[missing['s_y_first_open'].isna(), 'ssym']))
                mf = sorted(set(missing.loc[missing['f_y_first_open'].isna(), 'fsym']))
                say(f'stocks without bars on some position day: {len(ms)} (e.g. {ms[:12]}); twins: {len(mf)} {mf[:12]}')
    # ---------------------------------------------------------------- the variant: signal from Yahoo's first-bar open
    say('\n===== variant: the signal recomputed from Yahoo\'s first-bar open (5-minute bars) =====')
    wd5 = RES['5m']['days']
    V, nr, nok = variant_positions(rk, y5, K, wd5)
    say(f'ranked rows on the 5-minute window {nr}; with a usable first-bar open for stock and twin {nok} ({nok / nr:.1%})')
    main_pos = books['all'][books['all']['date'].isin(wd5)]
    same = V.merge(main_pos[['t', 'i', 'side']], on=['t', 'i', 'side'], how='inner')
    say(f'variant positions {len(V)}; also in the main book on the same side {len(same)} ({len(same) / len(V):.1%})')
    RES['variant'] = dict(positions=int(len(V)), overlap=float(len(same) / len(V)))
    Vcols = ['t', 'i', 'f', 'beta', 'side', 'date', 'ssym', 'fsym', 's_open', 's_close', 'f_open', 'f_close', 'exchange', 'in500', 'in100']
    V = V[Vcols]
    for mode, y in (('5m', y5), ('1m', y1)):
        if y is None or mode not in RES:
            continue
        Vm = V[V['date'].isin(RES[mode]['days'])]
        d = price(Vm, y, K, mode)
        nc = int(d['common'].sum())
        say(f'\n--- variant on the {mode} window: positions {len(d)}, with bars {int(d["has_bars"].sum())}, common set {nc} ({nc / len(d):.1%}) ---')
        res, _ = entry_table(d, mode, f'{mode} variant: entry times (share and minus (a) against the MAIN book\'s (a), same days)',
                             ref_a=RES[mode]['_A_all_common'])
        RES['variant'][mode] = dict(coverage=dict(positions=int(len(d)), common=nc), entries=res)
    # ---------------------------------------------------------------- open agreement (5-minute window, all-stocks book positions)
    say('\n===== open agreement: store open vs Yahoo daily open vs Yahoo first-bar open (5-minute window, '
        'all-stocks book position stock-days, each once) =====')
    d = RES['5m']['_d_all']
    o = d.drop_duplicates(['t', 'i'])
    o = o[o['s_y_first_open'].notna() & o['s_yd_open'].notna() & o['s_k'].notna()].copy()
    o['so'] = o['s_open'].astype(float) / o['s_k']  # store open on Yahoo's scale
    o['yd'] = o['s_yd_open']
    o['yb'] = o['s_y_first_open']
    o['dv'] = np.where(o['in100'], '100 most traded', np.where(o['in500'], '101-500', 'rest'))
    pairs = [('store vs Yahoo daily', 'so', 'yd'), ('store vs first bar', 'so', 'yb'), ('Yahoo daily vs first bar', 'yd', 'yb')]

    def rates(x):
        out = {'n': int(len(x)), 'n_k1': int(x['s_k1'].sum())}
        for name, a, b in pairs:
            diff = np.abs(x[a] - x[b])
            rel = diff / x[b]
            k1 = x['s_k1'] == True  # noqa: E712
            out[name] = dict(cent=float((diff[k1] > 0.01 + 1e-6).mean()) if k1.any() else np.nan,
                             half_pct=float((rel > 0.005).mean()))
        return out

    groups = [('all', o)] + [(f'exchange {e}', g) for e, g in o.groupby('exchange')] + \
             [(f'dollar volume {k}', g) for k, g in o.groupby('dv')]
    say('group | stock-days (k = 1) | ' + ' | '.join(f'{n}: > 1 cent, > 0.5%' for n, _, _ in pairs))
    RES['open_agreement'] = {}
    for name, g in groups:
        rr = rates(g)
        RES['open_agreement'][name] = rr
        say(f'{name} | {rr["n"]} ({rr["n_k1"]}) | ' + ' | '.join(f'{rr[n]["cent"]:.1%}, {rr[n]["half_pct"]:.1%}' for n, _, _ in pairs))
    # legs: is the disagreement concentrated in the book's legs vs all ranked stock-days? (same 5-minute days)
    # ---------------------------------------------------------------- verdict
    e = RES['5m']['all']['entries']
    sh, (m935, t935, _) = e['0935']['share'], e['0935']['gross']
    if sh >= 0.5 and m935 > 0 and t935 >= 2:
        v = 'Real'
    elif sh < 0.25:
        v = 'Opening-print effect'
    else:
        v = 'Unresolved'
    say(f'\nVERDICT (5-minute window, all stocks, gross): entry at 9:35 keeps {sh:.2f} of entry (a)\'s spread; 9:35 gross '
        f'{m935 * 1e4:+.2f} bps/day, t {t935:+.2f} -> {v}')
    RES['verdict'] = dict(share_0935=sh, gross_0935_bps=m935 * 1e4, t_0935=t935, verdict=v)
    # ---------------------------------------------------------------- outputs
    with open(OUT / 'daily_check.tsv', 'w') as fh:
        fh.write('window\tbook\tdate\tentry\tgross\ttc\tlong\tshort\n')
        for mode in ('5m', '1m'):
            if mode not in RES or '_d_all' not in RES[mode]:
                continue
            d = RES[mode]['_d_all']
            for t in ['a'] + TIMES[mode]:
                D = daily(d, f'h_{t}', d['common'])
                for date, row in D.iterrows():
                    fh.write(f'{mode}\tall\t{date}\t{t}\t{row.gross:.6f}\t{row.tc:.6f}\t{row.long:.6f}\t{row.short:.6f}\n')
    for mode in ('5m', '1m'):
        if mode in RES:
            RES[mode].pop('_A_all_common', None)
            RES[mode].pop('_d_all', None)
    (OUT / 'check.txt').write_text('\n'.join(LINES) + '\n')
    (OUT / 'check.json').write_text(json.dumps(RES, indent=1, default=float))


def variant_symbols():
    """Write out/symbols_1m_variant.txt: stocks and twins of the variant's positions on 1-minute days that
    symbols_1m.txt lacks. Prints no returns."""
    days, rk, books, y5, y1, yd = load()
    K = scale_table(rk, yd)
    wd5, _ = window_days(y5, '5m')
    V, _, _ = variant_positions(rk, y5, K, wd5)
    V = V[V['date'] >= '2026-09-22']
    have = set((OUT / 'symbols_1m.txt').read_text().split())
    extra = sorted((set(V['ssym']) | set(V['fsym'])) - have)
    (OUT / 'symbols_1m_variant.txt').write_text('\n'.join(extra) + '\n')
    print(f'variant positions on 1-minute days {len(V)}; extra symbols {len(extra)}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'variant-symbols':
        variant_symbols()
    else:
        main()
