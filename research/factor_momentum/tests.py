"""Tests 1-3 (old export part) of experiments/-3.8-factor-momentum.md, the baselines, per-component results,
eigenvalue shares, stability, betas, exposures and turnover, periods and years.

    python3 -I -B tests.py      # writes out/tests_old.txt, out/tests_old.json, out/monthly_old.tsv,
                                #        out/components_old.tsv, out/eigen_old.tsv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402
import books  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def pct(x):
    return f'{100 * x:+.3f}'


def describe(H, key):
    """Stats of one return series over a list of hold records (net/gross key prefix '' for the book)."""
    g = np.array([h[key + 'gross'] for h in H])
    n_ = np.array([h[key + 'net'] for h in H])
    s = np.array([h['spy'] for h in H])
    ok = np.isfinite(g) & np.isfinite(s)
    to = np.array([h[key + 'turnover'] for h in H[1:]]) if len(H) > 1 else np.array([np.nan])
    b = C.ols(g[ok], [s[ok]])
    return dict(gross=C.tstat(g), net=C.tstat(n_), sd=float(np.nanstd(g, ddof=1)), sharpe_gross=C.sharpe(g[np.isfinite(g)]),
                sharpe_net=C.sharpe(n_[np.isfinite(n_)]), beta=b['b'][1], beta_t=b['t'][1],
                turnover=float(np.nanmean(to)), cost_bps=float(1e4 * np.nanmean(g - n_)),
                skew=float(((g - g.mean()) ** 3).mean() / g.std() ** 3) if np.isfinite(g).all() else float('nan'),
                n=int(np.isfinite(g).sum()))


def row(name, d):
    return (f'{name:<34} {pct(d["gross"][0]):>8} {d["gross"][1]:+6.2f} {pct(d["net"][0]):>8} {d["net"][1]:+6.2f} '
            f'{100 * d["sd"]:6.2f} {d["sharpe_gross"]:+6.2f} {d["sharpe_net"]:+6.2f} {d["beta"]:+6.2f} ({d["beta_t"]:+5.1f}) '
            f'{d["skew"]:+6.2f} {d["turnover"]:5.2f} {d["cost_bps"]:5.1f} {d["n"]:4d}')


HEAD = (f'{"":<34} {"gross":>8} {"t":>6} {"net":>8} {"t":>6} {"sd":>6} {"SR g":>6} {"SR n":>6} {"b_SPY (t)":>14} '
        f'{"skew":>6} {"turn":>5} {"cost":>5} {"n":>4}')


def main():
    D = C.load('old')
    k0 = C.first_k(D)
    ks = list(range(k0, D['K']))
    C.log(f'formations {C.iso(D, D["rebal"][ks[0]])} .. {C.iso(D, D["rebal"][ks[-1]])} ({len(ks)})')
    forms = books.run(D, ks)
    H = [f for f in forms if f['hold']]
    days = D['days']
    lo06, lo20 = c2eval.day_index(days, '2006-01-01'), c2eval.day_index(days, '2020-01-01')
    k500 = C.first_k500(D)
    t500 = int(D['rebal'][k500])
    periods = {
        'Test 1: 2006..2019': (lo06, lo20),
        '  2006-02..2007-11 (~300 stocks)': (lo06, t500),
        '  2007-12..2019 (500 stocks)': (t500, lo20),
        'Test 3: 2020..2025-05': (lo20, D['T']),
        'all 2006..2025-05': (lo06, D['T']),
    }

    def sel(lo, hi):
        return [h for h in H if lo <= h['t'] < hi]

    res = dict(first_formation=forms[0]['date'], last_formation=forms[-1]['date'], n_formations=len(forms),
               n_holds=len(H), last_hold=H[-1]['date'], first_500=C.iso(D, t500), periods={})
    say(f'formations {forms[0]["date"]} .. {forms[-1]["date"]} ({len(forms)}); complete holds {len(H)} (last formed '
        f'{H[-1]["date"]}, ends {C.iso(D, H[-1]["t"] + C.HOLD)}); 500-stock universe from {C.iso(D, t500)}')

    say('\n== Book and baselines: mean % per 21-day hold (t). Book = $2 gross; baselines $1 long + $1 short ==')
    names = {'book (12-1 sign)': '', 'diag: book on 1-month sign': 'm1_', '(a) plain 12-1 momentum': 'a_',
             '(b) twin-group momentum': 'b_'}
    for pname, (lo, hi) in periods.items():
        Hs = sel(lo, hi)
        say(f'\n-- {pname} ({len(Hs)} holds)')
        say(HEAD)
        res['periods'][pname] = {}
        for nm, key in names.items():
            Hk = [h for h in Hs if np.isfinite(h[key + 'gross'])]
            d = describe(Hk, key)
            res['periods'][pname][nm] = d
            say(row(nm, d))

    # exposures and ex-ante beta
    say('\n== Book exposures at formation (mean over Test 1 / Test 3 formations) ==')
    for pname in ('Test 1: 2006..2019', 'Test 3: 2020..2025-05'):
        lo, hi = periods[pname]
        Hs = sel(lo, hi)
        ex = {f: float(np.nanmean([h[f] for h in Hs])) for f in ('long_exp', 'short_exp', 'net_exp', 'beta_exante', 'n_long', 'n')}
        ex['net_exp_range'] = [float(np.min([h['net_exp'] for h in Hs])), float(np.max([h['net_exp'] for h in Hs]))]
        ex['beta_exante_range'] = [float(np.nanmin([h['beta_exante'] for h in Hs])), float(np.nanmax([h['beta_exante'] for h in Hs]))]
        res['periods'][pname]['exposure'] = ex
        say(f'{pname}: long ${ex["long_exp"]:.3f}, short ${ex["short_exp"]:.3f}, net {ex["net_exp"]:+.3f} '
            f'(range {ex["net_exp_range"][0]:+.3f}..{ex["net_exp_range"][1]:+.3f}); ex-ante beta {ex["beta_exante"]:+.3f} '
            f'(range {ex["beta_exante_range"][0]:+.3f}..{ex["beta_exante_range"][1]:+.3f}); long names {ex["n_long"]:.0f} of {ex["n"]:.0f}')

    # Test 2: regression on both baselines
    say('\n== Test 2: book hold returns on both baselines (OLS; Newey-West 3 lags as a check) ==')
    res['test2'] = {}
    for pname in ('Test 1: 2006..2019', '  2007-12..2019 (500 stocks)', 'Test 3: 2020..2025-05', 'all 2006..2025-05'):
        lo, hi = periods[pname]
        Hs = [h for h in sel(lo, hi) if h['b_ok']]
        for kind in ('net', 'gross'):
            y = [h[kind] for h in Hs]
            r = C.ols(y, [[h['a_' + kind] for h in Hs], [h['b_' + kind] for h in Hs]], nw_lags=3)
            res['test2'][f'{pname.strip()}|{kind}'] = r
            say(f'{pname.strip():<30} {kind:<5}: alpha {pct(r["b"][0])}% a hold (t {r["t"][0]:+.2f}, NW {r["t_nw"][0]:+.2f}); '
                f'on (a) {r["b"][1]:+.3f} (t {r["t"][1]:+.2f}); on (b) {r["b"][2]:+.3f} (t {r["t"][2]:+.2f}); '
                f'R2 {r["r2"]:.3f}; n {r["n"]}')
        g = np.array([h['gross'] for h in Hs])
        cr = dict(a=float(np.corrcoef(g, [h['a_gross'] for h in Hs])[0, 1]),
                  b=float(np.corrcoef(g, [h['b_gross'] for h in Hs])[0, 1]),
                  ab=float(np.corrcoef([h['a_gross'] for h in Hs], [h['b_gross'] for h in Hs])[0, 1]))
        res['test2'][f'{pname.strip()}|corr'] = cr
        say(f'{"":<30} correlations of gross hold returns: book-(a) {cr["a"]:+.2f}, book-(b) {cr["b"]:+.2f}, (a)-(b) {cr["ab"]:+.2f}')

    # per component
    say('\n== Per component, Test 1 holds (2006..2019): contribution to the book (% a hold, t), the signed gross-1 '
        'eigenportfolio alone (% a hold, t, share of holds > 0), mean |B| gross in the book, eigenvalue share ==')
    res['components'] = {}
    for pname in ('Test 1: 2006..2019', 'Test 3: 2020..2025-05', 'all 2006..2025-05'):
        lo, hi = periods[pname]
        Hs = sel(lo, hi)
        Cn = np.array([h['contrib'] for h in Hs])
        Sf = np.array([h['signed_fac'] for h in Hs])
        Gb = np.array([h['comp_gross'] for h in Hs])
        Sh = np.array([h['share'] for h in Hs])
        comp = []
        if pname.startswith('Test 1'):
            say(f'{"comp":>4} {"contrib":>8} {"t":>6} {"alone":>8} {"t":>6} {"hit":>5} {"|B|":>6} {"share":>6}')
        for j in range(C.NK):
            c_ = C.tstat(Cn[:, j])
            s_ = C.tstat(Sf[:, j])
            comp.append(dict(comp=j + C.FIRST, contrib=c_, alone=s_, hit=float((Sf[:, j] > 0).mean()),
                             gross_in_book=float(Gb[:, j].mean()), share=float(Sh[:, j + 1].mean())))
            if pname.startswith('Test 1'):
                say(f'{j + C.FIRST:4d} {pct(c_[0]):>8} {c_[1]:+6.2f} {pct(s_[0]):>8} {s_[1]:+6.2f} '
                    f'{comp[-1]["hit"]:5.2f} {comp[-1]["gross_in_book"]:6.3f} {comp[-1]["share"]:6.3f}')
        groups = {'2-5': slice(0, 4), '6-11': slice(4, 10), '12-21': slice(10, 20)}
        gsum = {gname: C.tstat(Cn[:, sl].sum(1)) for gname, sl in groups.items()}
        res['components'][pname] = dict(per=comp, groups=gsum,
                                        pooled_hit=float((Sf > 0).mean()), pooled_alone=C.tstat(Sf.mean(1)))
        say(f'{pname}: contribution by group ' + '; '.join(f'{g_} {pct(v[0])} (t {v[1]:+.2f})' for g_, v in gsum.items())
            + f'; signed eigenportfolios > 0 in {100 * (Sf > 0).mean():.1f}% of component-holds; '
              f'equal-weight average of the 20 signed eigenportfolios {pct(C.tstat(Sf.mean(1))[0])} (t {C.tstat(Sf.mean(1))[1]:+.2f})')

    # eigenvalue shares
    say('\n== Eigenvalue shares (share of variance = lambda / n), mean by year of formation ==')
    say('year  n    comp1   comp2   comp3   comp4   comp5   2-21 total')
    res['eigen_years'] = {}
    for y in range(2004, 2026):
        Fy = [f for f in forms if f['date'][:4] == str(y)]
        if not Fy:
            continue
        Sh = np.array([f['share'] for f in Fy])
        res['eigen_years'][y] = dict(n=float(np.mean([f['n'] for f in Fy])), c1=float(Sh[:, 0].mean()),
                                     c2_21=float(Sh[:, 1:].sum(1).mean()), top5=Sh[:, :5].mean(0).tolist())
        say(f'{y}  {np.mean([f["n"] for f in Fy]):4.0f}  ' + '  '.join(f'{x:.3f}' for x in Sh[:, :5].mean(0))
            + f'   {Sh[:, 1:].sum(1).mean():.3f}')
    allsh = np.array([f['share'] for f in forms])
    i1 = int(np.argmax(allsh[:, 0]))
    i0 = int(np.argmin(allsh[:, 0]))
    res['eigen'] = dict(c1_mean=float(allsh[:, 0].mean()), c1_max=(forms[i1]['date'], float(allsh[i1, 0])),
                        c1_min=(forms[i0]['date'], float(allsh[i0, 0])), c2_21_mean=float(allsh[:, 1:].sum(1).mean()),
                        mean_shares=allsh.mean(0).tolist())
    say(f'component 1: mean {allsh[:, 0].mean():.3f}, max {allsh[i1, 0]:.3f} ({forms[i1]["date"]}), min {allsh[i0, 0]:.3f} '
        f'({forms[i0]["date"]}); components 2-21 together: mean {allsh[:, 1:].sum(1).mean():.3f}')

    # stability
    say('\n== Stability between consecutive formations (eigenvectors of the correlation matrix, aligned on the union '
        'of stocks): mean best |cos| among the previous top 50, share matched to the same index, mean |cos| with the '
        'same index, mean retention in the previous 2-21 subspace ==')
    Fs = [f for f in forms if 'stab_best' in f and f['date'] >= '2006']
    Bst = np.array([f['stab_best'] for f in Fs])
    Arg = np.array([f['stab_arg'] for f in Fs])
    Sm = np.array([f['stab_same'] for f in Fs])
    Rt = np.array([f['stab_ret'] for f in Fs])
    res['stability'] = dict(per=[], c1=float(np.mean([f['stab_c1'] for f in Fs])))
    say(f'component 1: mean |cos| with the previous component 1 {res["stability"]["c1"]:.3f}')
    say(f'{"comp":>4} {"best":>6} {"p10":>6} {"same idx":>8} {"|cos| same":>10} {"retention":>9} {"|cos|>0.7":>9}')
    for j in range(C.NK):
        d = dict(comp=j + C.FIRST, best=float(Bst[:, j].mean()), p10=float(np.percentile(Bst[:, j], 10)),
                 same_idx=float((Arg[:, j] == j + C.FIRST).mean()), same=float(Sm[:, j].mean()),
                 retention=float(Rt[:, j].mean()), over07=float((Bst[:, j] > 0.7).mean()))
        res['stability']['per'].append(d)
        say(f'{d["comp"]:4d} {d["best"]:6.3f} {d["p10"]:6.3f} {d["same_idx"]:8.2f} {d["same"]:10.3f} {d["retention"]:9.3f} '
            f'{d["over07"]:9.2f}')
    say(f'all 20: mean best |cos| {Bst.mean():.3f}, retention {Rt.mean():.3f}; matched to the same index '
        f'{100 * (Arg == np.arange(C.FIRST, C.LAST + 1)[None, :]).mean():.1f}%')
    # sign persistence of the trade itself: does sign x weights stay? (cos between consecutive signed B columns
    # is not defined without matching; the book's turnover is the direct measure)

    # by period blocks and years
    say('\n== By period, % per hold (t): book net, book gross, (a) gross, (b) gross ==')
    blocks = [('2006..2009', '2006-01-01', '2010-01-01'), ('2010..2014', '2010-01-01', '2015-01-01'),
              ('2015..2019', '2015-01-01', '2020-01-01'), ('2020..2022', '2020-01-01', '2023-01-01'),
              ('2023..2025-05', '2023-01-01', None)]
    res['blocks'] = {}
    for lab, a_, b_ in blocks:
        lo = c2eval.day_index(days, a_)
        hi = c2eval.day_index(days, b_) if b_ else D['T']
        Hs = sel(lo, hi)
        r_ = dict(net=C.tstat([h['net'] for h in Hs]), gross=C.tstat([h['gross'] for h in Hs]),
                  a=C.tstat([h['a_gross'] for h in Hs]), b=C.tstat([h['b_gross'] for h in Hs]),
                  sharpe_net=C.sharpe([h['net'] for h in Hs]))
        res['blocks'][lab] = r_
        say(f'{lab:<14} ' + '  '.join(f'{k_} {pct(v[0])} ({v[1]:+.2f})' for k_, v in r_.items() if isinstance(v, tuple))
            + f'  SR net {r_["sharpe_net"]:+.2f}  n {r_["net"][2]}')
    say('\n== By calendar year of formation: mean % per hold (book net, gross; (a) gross; (b) gross; SPY) ==')
    res['years'] = {}
    for y in range(2005, 2026):
        Hs = [h for h in H if h['date'][:4] == str(y)]
        if not Hs:
            continue
        r_ = {k_: float(np.nanmean([h[k_] for h in Hs])) for k_ in ('net', 'gross', 'a_gross', 'b_gross', 'spy')}
        r_['n'] = len(Hs)
        res['years'][y] = r_
        say(f'{y}  net {100 * r_["net"]:+6.2f}  gross {100 * r_["gross"]:+6.2f}  (a) {100 * r_["a_gross"]:+6.2f}  '
            f'(b) {100 * r_["b_gross"]:+6.2f}  SPY {100 * r_["spy"]:+6.2f}  holds {len(Hs)}')

    say('\n== Worst and best holds of the book, 2006..2025-05 (net %) ==')
    Hs = sel(lo06, D['T'])
    g = np.array([h['net'] for h in Hs])
    o = np.argsort(g)
    res['worst'] = [(Hs[i]['date'], float(g[i]), Hs[i]['spy'], Hs[i]['a_gross']) for i in o[:5]]
    res['best'] = [(Hs[i]['date'], float(g[i]), Hs[i]['spy'], Hs[i]['a_gross']) for i in o[::-1][:5]]
    mdd, i, j = C.max_drawdown(g)
    res['max_dd'] = dict(size=mdd, peak=Hs[i]['date'] if i < len(Hs) else None, trough=Hs[j - 1]['date'])
    say('worst: ' + '; '.join(f'{d} {100 * x:+.2f} (SPY {100 * s:+.1f}, (a) {100 * a:+.1f})' for d, x, s, a in res['worst']))
    say('best:  ' + '; '.join(f'{d} {100 * x:+.2f} (SPY {100 * s:+.1f}, (a) {100 * a:+.1f})' for d, x, s, a in res['best']))
    say(f'largest drawdown (cumulative sum of net holds): {100 * mdd:.1f} points, from {res["max_dd"]["peak"]} to the hold '
        f'formed {res["max_dd"]["trough"]}')

    # the tests
    p1 = res['periods']['Test 1: 2006..2019']['book (12-1 sign)']
    p3 = res['periods']['Test 3: 2020..2025-05']['book (12-1 sign)']
    t2 = res['test2']['Test 1: 2006..2019|net']
    res['tests'] = {
        '1': dict(net=p1['net'], gross=p1['gross'], passed=bool(p1['net'][1] >= 2.5)),
        '2': dict(alpha=t2['b'][0], t=t2['t'][0], t_nw=t2['t_nw'][0], passed=bool(t2['b'][0] > 0 and t2['t'][0] >= 2)),
        '3_old': dict(net=p3['net'], gross=p3['gross'], passed=bool(p3['net'][0] > 0)),
    }
    T_ = res['tests']
    say('\n== Tests ==')
    say(f'Test 1 (2006..2019, book net): {pct(T_["1"]["net"][0])}% a hold, t {T_["1"]["net"][1]:+.2f} (n {T_["1"]["net"][2]}); '
        f'gross {pct(T_["1"]["gross"][0])} t {T_["1"]["gross"][1]:+.2f} -> {"PASS" if T_["1"]["passed"] else "FAIL"}')
    say(f'Test 2 (alpha on (a) and (b), net): {pct(T_["2"]["alpha"])}% a hold, t {T_["2"]["t"]:+.2f} (NW {T_["2"]["t_nw"]:+.2f}) '
        f'-> {"PASS" if T_["2"]["passed"] else "FAIL"}')
    say(f'Test 3, old part (2020..2025-05, net mean positive): {pct(T_["3_old"]["net"][0])} t {T_["3_old"]["net"][1]:+.2f} '
        f'-> {"PASS" if T_["3_old"]["passed"] else "FAIL"}')

    # files
    with open(C.OUT / 'monthly_old.tsv', 'w') as f:
        cols = ['date', 'n', 'gross', 'net', 'cost', 'turnover', 'long', 'short', 'long_exp', 'short_exp', 'net_exp',
                'beta_exante', 'spy', 'm1_gross', 'm1_net', 'a_gross', 'a_net', 'a_turnover', 'b_gross', 'b_net', 'b_turnover']
        f.write('\t'.join(cols) + '\n')
        for h in H:
            f.write('\t'.join(str(h[c]) if isinstance(h[c], (str, int)) else f'{h[c]:.6f}' for c in cols) + '\n')
    with open(C.OUT / 'components_old.tsv', 'w') as f:
        f.write('date\tcomp\tshare\tpast12\tpast1\tvol\tscale\tcontrib\tsigned_fac\tstab_best\tstab_arg\tstab_same\tstab_ret\n')
        for h in forms:
            for j in range(C.NK):
                vals = [h['date'], str(j + C.FIRST), f'{h["share"][j + 1]:.5f}', f'{h["past12"][j]:.5f}',
                        f'{h["past1"][j]:.5f}', f'{h["vol"][j]:.5f}', f'{h["scale"][j]:.4f}',
                        f'{h["contrib"][j]:.6f}' if h['hold'] else '', f'{h["signed_fac"][j]:.6f}' if h['hold'] else '',
                        f'{h["stab_best"][j]:.4f}' if 'stab_best' in h else '', str(h['stab_arg'][j]) if 'stab_arg' in h else '',
                        f'{h["stab_same"][j]:.4f}' if 'stab_same' in h else '', f'{h["stab_ret"][j]:.4f}' if 'stab_ret' in h else '']
                f.write('\t'.join(vals) + '\n')
    with open(C.OUT / 'eigen_old.tsv', 'w') as f:
        f.write('date\tn\t' + '\t'.join(f'share{j}' for j in range(1, C.LAST + 1)) + '\n')
        for h in forms:
            f.write(f'{h["date"]}\t{h["n"]}\t' + '\t'.join(f'{x:.5f}' for x in h['share']) + '\n')
    (C.OUT / 'tests_old.json').write_text(json.dumps(res, indent=1, default=float))
    (C.OUT / 'tests_old.txt').write_text('\n'.join(LINES) + '\n')
    C.log('done')


if __name__ == '__main__':
    main()
