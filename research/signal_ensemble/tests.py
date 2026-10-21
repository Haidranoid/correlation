"""Tests 1-3 and every reported table, on the weekly calendar (the .md, "Rules" and "Decisions").

    python3 -I tests.py old    -> results_old.txt / .json, WORK/old/series.npz
    python3 -I tests.py new    -> results_new.txt / .json (the unseen months, once)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json
from common import (c2eval, np, EXPORTS, wdir, log, HERE, COMPONENTS, LABEL, PERIODS, UNSEEN, TOP, KEEP, LO_IN, LO_KEEP,
                    COST_BPS, WEEKS_PER_YEAR, spearman, ffill, tstat, ols, in_period)

STORES = {'old': c2eval.STORE_OLD, 'new': c2eval.STORE_NEW}
MIN_IC = 100
BOOK_START = {'old': '2008-01-01', 'new': '2025-06-30'}


def week_growth(tag, W):
    """[nW, N] close[d+5] / close[d] on forward-filled closes (NaN where no close at d or d+5 beyond the export);
    and SPY's [nW]."""
    E = c2eval.load_export(EXPORTS[tag], arrays=('close',))
    T = E['close'].shape[1]
    P = ffill(E['close'])
    ok = W + 5 <= T - 1
    G = np.full((len(W), P.shape[0]), np.nan)
    with np.errstate(all='ignore'):
        G[ok] = (P[:, W[ok] + 5] / P[:, W[ok]]).T
    del P
    S = c2eval.load_store(STORES[tag])
    sid = [s['id'] for s in S['syms'] if s['ticker'] == 'SPY' and s['kind'] == 'etf']
    assert len(sid) == 1, sid
    sp = ffill(c2eval.store_panel(S, sid, E['days'], 'close'))[0]
    gs = np.full(len(W), np.nan)
    gs[ok] = sp[W[ok] + 5] / sp[W[ok]]
    return E, G, gs


def weekly_ic(S, G, weeks):
    ic = np.full(len(S), np.nan)
    for i in weeks:
        m = np.isfinite(S[i]) & np.isfinite(G[i])
        if m.sum() >= MIN_IC:
            ic[i] = spearman(S[i, m].astype(np.float64), G[i, m])
    return ic


def paired_ic(Sc, Sk, G, weeks):
    """ICs of a component Sc and the composite Sk on the stocks where both (and the return) are finite."""
    a = np.full(len(Sc), np.nan)
    b = np.full(len(Sc), np.nan)
    for i in weeks:
        m = np.isfinite(Sc[i]) & np.isfinite(Sk[i]) & np.isfinite(G[i])
        if m.sum() >= MIN_IC:
            a[i] = spearman(Sc[i, m].astype(np.float64), G[i, m])
            b[i] = spearman(Sk[i, m].astype(np.float64), G[i, m])
    return a, b


def book(S, G, weeks, n_in, n_keep, short=True, frac=True):
    """Buffered weekly book. frac: n_in/n_keep are fractions of the ranked stocks (long-short), else counts (long-only).
    Returns dict of [nW] arrays: gross, turn (sum |dw|), net, nl, ns."""
    nW = len(S)
    out = {k: np.full(nW, np.nan) for k in ('gross', 'turn', 'net', 'nl', 'ns', 'long', 'short')}
    prevL, prevS = {}, {}
    for i in weeks:
        s = S[i]
        m = np.nonzero(np.isfinite(s) & np.isfinite(G[i]))[0]
        if len(m) < 50:
            continue
        order = m[np.argsort(-s[m].astype(np.float64), kind='mergesort')]
        n = len(m)
        ki = int(round(n_in * n)) if frac else n_in
        kk = int(round(n_keep * n)) if frac else n_keep
        L = set(order[:ki].tolist()) | (set(prevL) & set(order[:kk].tolist()))
        L = np.array(sorted(L))
        gl = G[i, L]
        wl = 1.0 / len(L)
        sl = set(L.tolist())
        turn = sum(abs((wl if j in sl else 0.0) - prevL.get(j, 0.0)) for j in sl | set(prevL))
        rl = gl.mean() - 1
        prevL = dict(zip(L.tolist(), (gl / gl.sum()).tolist()))
        rs = 0.0
        if short:
            rev = order[::-1]
            Sh = set(rev[:ki].tolist()) | (set(prevS) & set(rev[:kk].tolist()))
            Sh = np.array(sorted(Sh))
            gs = G[i, Sh]
            ws = 1.0 / len(Sh)
            ss = set(Sh.tolist())
            turn += sum(abs((ws if j in ss else 0.0) - prevS.get(j, 0.0)) for j in ss | set(prevS))
            rs = gs.mean() - 1
            prevS = dict(zip(Sh.tolist(), (gs / gs.sum()).tolist()))
            out['ns'][i] = len(Sh)
        out['long'][i] = rl
        out['short'][i] = rs
        out['gross'][i] = rl - rs
        out['turn'][i] = turn
        out['net'][i] = rl - rs - COST_BPS * 1e-4 * turn
        out['nl'][i] = len(L)
    return out


def fmt_t(m, t, scale=1.0, dec=2):
    return f'{m * scale:+.{dec}f} ({t:+.2f})' if np.isfinite(m) else 'n/a'


def main(tag):
    z = np.load(wdir(tag) / 'weekly.npz')
    W, R, comp, comp4 = z['W'], z['R'], z['comp'], z['comp4']
    R5, comp5, comp45 = z['R500'], z['comp500'], z['comp4_500']
    E, G, gspy = week_growth(tag, W)
    days = E['days']
    nW = len(W)
    has_ret = np.isfinite(G).any(1)
    start = in_period(days, W, BOOK_START[tag], '2100-01-01')
    weeks = np.nonzero(start & has_ret)[0]
    log(f'{tag}: {len(weeks)} book weeks, {c2eval.iso(days[W[weeks[0]]])}..{c2eval.iso(days[W[weeks[-1]]])}')
    comps = [c for c in COMPONENTS if np.isfinite(R[:, COMPONENTS.index(c)]).any()]
    periods = PERIODS if tag == 'old' else [UNSEEN]
    pmask = {p[0]: in_period(days, W, p[1], p[2]) for p in periods}
    res = {'tag': tag, 'weeks': [c2eval.iso(days[W[weeks[0]]]), c2eval.iso(days[W[weeks[-1]]])]}
    lines = []
    P = lines.append

    # ------------------------------------------------------------------ ICs
    ic = {'composite': weekly_ic(comp, G, weeks)}
    if tag == 'old':
        ic['composite (no LightGBM)'] = weekly_ic(comp4, G, weeks)
    for c in comps:
        ic[c] = weekly_ic(R[:, COMPONENTS.index(c)], G, weeks)
    ic500 = {'composite': weekly_ic(comp5, G, weeks)}
    if tag == 'old':
        ic500['composite (no LightGBM)'] = weekly_ic(comp45, G, weeks)
    for c in comps:
        ic500[c] = weekly_ic(R5[:, COMPONENTS.index(c)], G, weeks)

    # ------------------------------------------------------------------ books
    B = {'composite': book(comp, G, weeks, TOP, KEEP),
         'composite, no buffer': book(comp, G, weeks, TOP, TOP)}
    if tag == 'old':
        B['composite (no LightGBM)'] = book(comp4, G, weeks, TOP, KEEP)
    for c in comps:
        B[c] = book(R[:, COMPONENTS.index(c)], G, weeks, TOP, KEEP)
    B500 = {'composite': book(comp5, G, weeks, TOP, KEEP), 'composite, no buffer': book(comp5, G, weeks, TOP, TOP)}
    for c in comps:
        B500[c] = book(R5[:, COMPONENTS.index(c)], G, weeks, TOP, KEEP)
    LO = book(comp, G, weeks, LO_IN, LO_KEEP, short=False, frac=False)
    ew = np.full(nW, np.nan)
    for i in weeks:
        m = np.isfinite(comp[i]) & np.isfinite(G[i])
        ew[i] = G[i, m].mean() - 1
    spy = gspy - 1

    def ic_row(x, mask):
        mm, t, n = tstat(x[mask])
        return mm, t, n

    def ls_row(b, mask, key='net'):
        return tstat(b[key][mask])

    # ------------------------------------------------------------------ test 1
    main_p = periods[0][0]
    mk = pmask[main_p] & np.isin(np.arange(nW), weeks)
    P(f'==== {tag}: weeks {res["weeks"][0]}..{res["weeks"][1]} ({len(weeks)}) ====')
    P(f'\n## Rank IC, {main_p}: mean weekly IC (t), weeks; composite paired on the component\'s weeks and stocks')
    m_, t_, n_ = ic_row(ic['composite'], mk)
    res['ic'] = {'composite': {'mean': m_, 't': t_, 'n': n_}}
    P(f'composite: {m_:+.4f} (t {t_:+.2f}, {n_} weeks)')
    if tag == 'old':
        m4, t4, n4 = ic_row(ic['composite (no LightGBM)'], mk)
        res['ic']['composite4'] = {'mean': m4, 't': t4, 'n': n4}
        P(f'composite without LightGBM: {m4:+.4f} (t {t4:+.2f}, {n4} weeks)')
    beats = True
    for c in comps:
        a, b = paired_ic(R[:, COMPONENTS.index(c)], comp, G, weeks)
        ma, ta, na = tstat(a[mk])
        mb, tb, nb = tstat(b[mk])
        beats &= tb > ta
        res['ic'][c] = {'mean': ma, 't': ta, 'n': na, 'comp_mean': mb, 'comp_t': tb, 'first_week': c2eval.iso(days[W[np.nonzero(np.isfinite(a) & mk)[0][0]]])}
        P(f'{LABEL[c]:28s} {ma:+.4f} (t {ta:+.2f}, {na} weeks from {res["ic"][c]["first_week"]}); composite on the same: {mb:+.4f} (t {tb:+.2f})')
    if tag == 'old':
        t1 = (t_ >= 3) and beats
        res['test1'] = {'pass': bool(t1), 't_ge_3': bool(t_ >= 3), 'beats_every_component': bool(beats)}
        P(f'TEST 1: composite t >= 3: {t_ >= 3}; beats every component: {beats} -> {"PASS" if t1 else "FAIL"}')

    # ------------------------------------------------------------------ correlations
    if tag == 'old':
        P(f'\n## Component correlations, {main_p}: mean weekly cross-sectional correlation of ranks (upper), '
          'correlation of weekly IC series (lower)')
        names = comps + ['composite']
        mats = {}
        X = {c: R[:, COMPONENTS.index(c)] for c in comps}
        X['composite'] = comp
        C = np.full((len(names), len(names)), np.nan)
        Cic = np.full((len(names), len(names)), np.nan)
        for ia, a in enumerate(names):
            for ib, b in enumerate(names):
                if ib <= ia:
                    continue
                vals = []
                for i in np.nonzero(mk)[0]:
                    m = np.isfinite(X[a][i]) & np.isfinite(X[b][i])
                    if m.sum() >= MIN_IC:
                        vals.append(np.corrcoef(X[a][i, m], X[b][i, m])[0, 1])
                C[ia, ib] = np.mean(vals)
                x, y = ic[a if a != 'composite' else 'composite'][mk], ic[b][mk]
                ok = np.isfinite(x) & np.isfinite(y)
                Cic[ib, ia] = np.corrcoef(x[ok], y[ok])[0, 1]
        P(' ' * 28 + ''.join(f'{n[:10]:>11s}' for n in names))
        for ia, a in enumerate(names):
            row = ''.join(f'{(C[ia, ib] if ib > ia else Cic[ia, ib]):>+11.2f}' if ib != ia else f'{"":>11s}' for ib in range(len(names)))
            P(f'{(LABEL.get(a, a))[:28]:28s}{row}')
        res['corr'] = {'names': names, 'xs': C.tolist(), 'ic_series': Cic.tolist()}
        # Grinold-Kahn style expectation for the equal-weight mean of the components with all weeks (LightGBM weeks)
        lw = mk & np.isfinite(ic['lgbm']) if 'lgbm' in ic else mk
        ics = np.array([np.nanmean(ic[c][lw]) for c in comps])
        rho = np.eye(len(comps))
        for ia in range(len(comps)):
            for ib in range(ia + 1, len(comps)):
                vals = []
                for i in np.nonzero(lw)[0]:
                    m = np.isfinite(X[comps[ia]][i]) & np.isfinite(X[comps[ib]][i])
                    if m.sum() >= MIN_IC:
                        vals.append(np.corrcoef(X[comps[ia]][i, m], X[comps[ib]][i, m])[0, 1])
                rho[ia, ib] = rho[ib, ia] = np.mean(vals)
        expect = ics.sum() / np.sqrt(rho.sum())
        actual = np.nanmean(ic['composite'][lw])
        res['gk'] = {'expected': float(expect), 'actual': float(actual), 'component_ics': ics.tolist()}
        P(f'Expected IC of the equal-weight mean from the components\' ICs and correlations (weeks with LightGBM): '
          f'{expect:+.4f}; actual {actual:+.4f}; components {np.round(ics, 4).tolist()}')

    # ------------------------------------------------------------------ test 2 and books
    P(f'\n## Weekly books, {main_p}: mean weekly return in bps (t); turnover = share of the book traded a week')
    res['books'] = {}
    for k, b in B.items():
        g = ls_row(b, mk, 'gross'); nt = ls_row(b, mk, 'net')
        tv = np.nanmean(b['turn'][mk]) / 4
        res['books'][k] = {'gross': g[0], 'gross_t': g[1], 'net': nt[0], 'net_t': nt[1], 'n': nt[2], 'turnover': tv,
                           'nl': float(np.nanmean(b['nl'][mk])), 'ns': float(np.nanmean(b['ns'][mk])),
                           'long': float(np.nanmean(b['long'][mk])), 'short': float(np.nanmean(b['short'][mk]))}
        P(f'{LABEL.get(k, k):32s} gross {fmt_t(g[0], g[1], 1e4)}  net {fmt_t(nt[0], nt[1], 1e4)}  '
          f'= {nt[0] * WEEKS_PER_YEAR * 100:+.2f}% a year net; turnover {tv:.1%}; names/leg {np.nanmean(b["nl"][mk]):.0f}/{np.nanmean(b["ns"][mk]):.0f}; '
          f'long leg {np.nanmean(b["long"][mk]) * 1e4:+.1f}, short leg {np.nanmean(b["short"][mk]) * 1e4:+.1f} bps/wk')
    if tag == 'old':
        t2 = res['books']['composite']['net_t'] >= 2
        res['test2'] = {'pass': bool(t2)}
        P(f'TEST 2: buffered long-short net t {res["books"]["composite"]["net_t"]:+.2f} >= 2 -> {"PASS" if t2 else "FAIL"}')

    # ------------------------------------------------------------------ by period
    P('\n## By period: composite IC mean (t); buffered long-short gross and net bps/week (t); turnover; '
      'no-buffer net; long-only net minus equal-weight universe, % a year (t)')
    res['periods'] = {}
    for name, a, bb in periods:
        mm = pmask[name] & np.isin(np.arange(nW), weeks)
        if not mm.any():
            continue
        i_ = tstat(ic['composite'][mm]); g = ls_row(B['composite'], mm, 'gross'); nt = ls_row(B['composite'], mm, 'net')
        nb = ls_row(B['composite, no buffer'], mm, 'net')
        ex = tstat((LO['net'] - ew)[mm])
        tv = np.nanmean(B['composite']['turn'][mm]) / 4
        res['periods'][name] = {'ic': i_[0], 'ic_t': i_[1], 'weeks': i_[2], 'gross': g[0], 'gross_t': g[1], 'net': nt[0],
                                'net_t': nt[1], 'turnover': tv, 'nobuf_net': nb[0], 'nobuf_net_t': nb[1],
                                'lo_ex_ew': ex[0], 'lo_ex_ew_t': ex[1]}
        if tag == 'old':
            i4 = tstat(ic['composite (no LightGBM)'][mm]); n4 = ls_row(B['composite (no LightGBM)'], mm, 'net')
            res['periods'][name].update({'ic4': i4[0], 'ic4_t': i4[1], 'net4': n4[0], 'net4_t': n4[1]})
        P(f'{name:24s} IC {fmt_t(i_[0], i_[1], 1, 4)} [{i_[2]} wk]  gross {fmt_t(g[0], g[1], 1e4)}  net {fmt_t(nt[0], nt[1], 1e4)}  '
          f'turnover {tv:.1%}  no-buffer net {fmt_t(nb[0], nb[1], 1e4)}  long-only vs EW {fmt_t(ex[0], ex[1], WEEKS_PER_YEAR * 100)}'
          + (f'  | no LightGBM: IC {fmt_t(i4[0], i4[1], 1, 4)}, net {fmt_t(n4[0], n4[1], 1e4)}' if tag == 'old' else ''))
    if tag == 'old':
        p3 = res['periods']['2020–2025-06 (test 3)']
        t3 = (p3['ic'] > 0) and (p3['net'] > 0)
        res['test3_2020'] = {'pass': bool(t3)}
        P(f'TEST 3 (2020-2025-06): IC {p3["ic"]:+.4f}, net {p3["net"] * 1e4:+.2f} bps/wk -> {"same sign: PASS" if t3 else "FAIL"}')
    else:
        p3 = res['periods'][UNSEEN[0]]
        t3 = (p3['ic'] > 0) and (p3['net'] > 0)
        res['test3_unseen'] = {'pass': bool(t3)}
        P(f'TEST 3 (unseen months): IC {p3["ic"]:+.4f}, net {p3["net"] * 1e4:+.2f} bps/wk -> {"same sign: PASS" if t3 else "FAIL"}')

    # ------------------------------------------------------------------ per-component IC by period
    P('\n## Each component\'s IC by period: mean (t)')
    res['ic_periods'] = {}
    for name, a, bb in periods:
        mm = pmask[name] & np.isin(np.arange(nW), weeks)
        if not mm.any():
            continue
        row = {}
        for k in ic:
            row[k] = tstat(ic[k][mm])[:2]
        res['ic_periods'][name] = row
        P(f'{name:24s} ' + '; '.join(f'{LABEL.get(k, k)} {fmt_t(v[0], v[1], 1, 4)}' for k, v in row.items()))

    # ------------------------------------------------------------------ long-only account
    P('\n## Buffered long-only account (enter top 20, hold while top 60), net of 5 bps unless gross')
    res['long_only'] = {}
    for name, a, bb in periods:
        mm = pmask[name] & np.isin(np.arange(nW), weeks)
        if not mm.any():
            continue
        r = {}
        for key, ser in (('net', LO['net']), ('gross', LO['gross']), ('spy', spy), ('ew', ew)):
            x = ser[mm]
            x = x[np.isfinite(x)]
            r[key] = {'mean_yr': float(x.mean() * WEEKS_PER_YEAR), 'cagr': float(np.prod(1 + x) ** (WEEKS_PER_YEAR / len(x)) - 1),
                      'vol': float(x.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)), 'total': float(np.prod(1 + x) - 1)}
        for bn, bser in (('spy', spy), ('ew', ew)):
            d_ = tstat((LO['net'] - bser)[mm])
            al = ols(LO['net'][mm], bser[mm])
            r[f'ex_{bn}'] = {'mean_yr': d_[0] * WEEKS_PER_YEAR, 't': d_[1], 'alpha_yr': al[0] * WEEKS_PER_YEAR,
                             'alpha_t': al[1], 'beta': al[2]}
        r['turnover'] = float(np.nanmean(LO['turn'][mm]) / 2)
        r['names'] = float(np.nanmean(LO['nl'][mm]))
        res['long_only'][name] = r
        P(f'{name:24s} account {r["net"]["cagr"]:+.1%}/yr compounded (gross {r["gross"]["cagr"]:+.1%}, vol {r["net"]["vol"]:.1%}); '
          f'SPY {r["spy"]["cagr"]:+.1%}; EW universe {r["ew"]["cagr"]:+.1%}; excess vs SPY {r["ex_spy"]["mean_yr"]:+.1%} (t {r["ex_spy"]["t"]:+.2f}), '
          f'alpha {r["ex_spy"]["alpha_yr"]:+.1%} (t {r["ex_spy"]["alpha_t"]:+.2f}, beta {r["ex_spy"]["beta"]:.2f}); excess vs EW {r["ex_ew"]["mean_yr"]:+.1%} '
          f'(t {r["ex_ew"]["t"]:+.2f}), alpha {r["ex_ew"]["alpha_yr"]:+.1%} (t {r["ex_ew"]["alpha_t"]:+.2f}, beta {r["ex_ew"]["beta"]:.2f}); '
          f'turnover {r["turnover"]:.1%}/wk; {r["names"]:.0f} names')

    # ------------------------------------------------------------------ 500 largest
    P('\n## The 500 largest: IC mean (t); buffered long-short gross and net bps/week (t), turnover')
    res['large'] = {}
    for name, a, bb in periods:
        mm = pmask[name] & np.isin(np.arange(nW), weeks)
        if not mm.any():
            continue
        row = {}
        for k in ic500:
            row[k] = {'ic': tstat(ic500[k][mm])[:2]}
        for k, b in B500.items():
            row.setdefault(k, {})
            row[k].update({'gross': ls_row(b, mm, 'gross')[:2], 'net': ls_row(b, mm, 'net')[:2],
                           'turnover': float(np.nanmean(b['turn'][mm]) / 4)})
        res['large'][name] = row
        P(f'{name}:')
        for k, v in row.items():
            P(f'   {LABEL.get(k, k):32s} ' + (f'IC {fmt_t(*v["ic"], 1, 4)}  ' if 'ic' in v else '')
              + (f'gross {fmt_t(*v["gross"], 1e4)}  net {fmt_t(*v["net"], 1e4)}  turnover {v["turnover"]:.1%}' if 'net' in v else ''))

    # ------------------------------------------------------------------ yearly composite (diagnostic)
    P('\n## By calendar year: composite IC, buffered long-short net bps/week, long-only net minus EW (% a year)')
    yrs = sorted({c2eval.iso(days[W[i]])[:4] for i in weeks})
    res['years'] = {}
    for y in yrs:
        mm = in_period(days, W, f'{y}-01-01', f'{y}-12-31') & np.isin(np.arange(nW), weeks)
        i_ = tstat(ic['composite'][mm]); nt = tstat(B['composite']['net'][mm]); ex = tstat((LO['net'] - ew)[mm])
        res['years'][y] = {'ic': i_[0], 'net': nt[0], 'lo_ex_ew': ex[0]}
        P(f'{y}: IC {i_[0]:+.4f}  net {nt[0] * 1e4:+7.1f}  long-only vs EW {ex[0] * WEEKS_PER_YEAR:+.1%}')

    txt = '\n'.join(lines)
    print(txt)
    (HERE / f'results_{tag}.txt').write_text(txt + '\n')
    (HERE / f'results_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    np.savez(wdir(tag) / 'series.npz', W=W, weeks=weeks, ic=np.array([ic[k] for k in ic]), ic_names=np.array(list(ic)),
             ls_net=B['composite']['net'], ls_gross=B['composite']['gross'], ls_turn=B['composite']['turn'],
             lo_net=LO['net'], lo_gross=LO['gross'], ew=ew, spy=spy)


if __name__ == '__main__':
    main(_sys.argv[1])
