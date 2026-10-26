"""The judge (plan: "The two numbers for the index", reported items 15-17). Judge accounts run one at a time.

    money.py sanity -> out/money_sanity.txt   (5-day reversal at rebalance 5, delay 0 must give +11.8)
    money.py old    -> out/money_old.txt, out/money_old.json   (walk-forward on the old export)
    money.py new    -> out/money_new.txt, out/money_new.json   (single holdout 2025-06-30..2026-10-19; unseen.py)
    money.py fwd    -> forward.py

File name ("after costs"): the judge's long-only top-20 account on P_full, rebalance 1, delay 1, at 5 bps a side,
its yearly return minus SPY's Yahoo adjusted-close total return, averaged over the graded judge years
(research/after_costs/group_b/ac.py). "Before costs, vs random": c2eval.change_over_random at 0 bps, uniform.
"""
import json
import sys
import time

sys.path.insert(0, '/home/bhoult/programming/correlation/research/volume_return')
sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_b')
import numpy as np  # noqa: E402

import common as C  # noqa: E402

c2eval = C.c2eval


def run(path, export, rep, label, walk_forward=True, start='2024-06-28', **kw):
    t0 = time.time()
    txt, js = c2eval.judge(path, export=export, walk_forward=walk_forward, start=start, **kw)
    out = {'label': label, 'path': str(path), 'kw': kw}
    if walk_forward:
        cor0 = c2eval.change_over_random(js, 0)
        cor5 = c2eval.change_over_random(js, 1)
        cor20 = c2eval.change_over_random(js, 2)
        import ac
        jpath = str(path).replace('.npy', f"__r{kw.get('rebalance', 1)}_d{kw.get('delay', 0)}_"
                                          f"{kw.get('random_match', 'uniform')}.json")
        a = ac.judge_after_cost(jpath, 5.0)
        s = js['summary']
        out.update({'cor0': cor0, 'cor5': cor5['change'], 'cor20': cor20['change'], 'after_cost': a['after_cost'],
                    'strat5': a['strat5'], 'spy': a['spy'], 'n_years': a['n'], 'first': a['first'], 'last': a['last'],
                    'rows': a['rows'], 'ic_ret_z': s.get('ic_ret_z'), 'ls_bps': s.get('ls_bps_per_day_mean'),
                    'ls_z': s.get('ls_z'), 'beat': s.get('beat_random_p_lt_0_1')})
        dev = [r for r in a['rows'] if r[0] < '2019-07-01']
        C.log(rep, f'{label}: change over random {cor0["change"]:+.1f} pts/yr at 0 bps (se {cor0["se"]:.1f}, ahead '
                   f'{cor0["ahead"]} of {cor0["years"]}), {cor5["change"]:+.1f} at 5 bps, {cor20["change"]:+.1f} at '
                   f'20 bps; after costs vs SPY {a["after_cost"]:+.1f} (account {a["strat5"]:+.1f}% a year at 5 bps, '
                   f'SPY {a["spy"]:+.1f}%), {a["n"]} judge years {a["first"]}..{a["last"]}; '
                   f'IC z {s.get("ic_ret_z"):+.2f}, long-short {s.get("ls_bps_per_day_mean"):.2f} bps/day '
                   f'(z {s.get("ls_z"):+.2f}) [{time.time() - t0:.0f} s]')
        if dev:
            C.log(rep, f'   judge years 2007-08..2018-19 only: after costs '
                       f'{np.mean([r[2] - r[4] for r in dev]):+.1f}; change over random '
                       f'{np.mean(cor0["per_year"][:len(dev)]):+.1f}')
    else:
        import ac
        r = js['report']
        st = {round(x['cost_bps']): x for x in r['strategy']}
        spy = 100 * ac.spy_ret(r['first_day'], r['last_day'])
        yrs = r['days'] / 252
        out.update({'first': r['first_day'], 'last': r['last_day'], 'days': r['days'],
                    'pct0': st[0]['final_pct'], 'rand0': st[0]['random_median_pct'], 'p0': st[0]['p'],
                    'pct5': st[5]['final_pct'], 'pct20': st[20]['final_pct'], 'spy': spy,
                    'change0': st[0]['final_pct'] - st[0]['random_median_pct'],
                    'after_cost_period': st[5]['final_pct'] - spy,
                    'after_cost_annual': 100 * ((1 + st[5]['final_pct'] / 100) ** (1 / yrs) - (1 + spy / 100) ** (1 / yrs)),
                    'ic_ret_t': r.get('ic_ret_t'), 'ls_bps': r.get('ls_bps_per_day'), 'ls_t': r.get('ls_t')})
        C.log(rep, f"{label}: {r['first_day']}..{r['last_day']} ({r['days']} days): account {st[0]['final_pct']:+.1f}% "
                   f"at 0 bps against random median {st[0]['random_median_pct']:+.1f}% (p {st[0]['p']:.3f}): change "
                   f"{out['change0']:+.1f} points; at 5 bps {st[5]['final_pct']:+.1f}%, at 20 bps "
                   f"{st[20]['final_pct']:+.1f}%; SPY {spy:+.1f}%; after costs {out['after_cost_period']:+.1f} points "
                   f"over the period ({out['after_cost_annual']:+.1f} a year); IC t {r.get('ic_ret_t'):+.2f}, "
                   f"long-short {r.get('ls_bps_per_day'):+.2f} bps/day (t {r.get('ls_t'):+.2f}) [{time.time() - t0:.0f} s]")
    return out


def main(mode):
    rep = C.OUT / f'money_{mode}.txt'
    rep.write_text('')
    J = {}
    if mode == 'sanity':
        E = C.export('old', arrays=('close',))
        c = np.asarray(E['close'], np.float64)
        S = np.full(c.shape, np.nan, np.float32)
        with np.errstate(all='ignore'):
            S[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
        S[~np.isfinite(S)] = np.nan
        p = C.tagdir('old') / 'sanity_rev5.npy'
        np.save(p, S)
        J['sanity'] = run(p, c2eval.EXPORT_OLD, rep, '5-day reversal, every stock, rebalance 5, delay 0 (expect +11.8)',
                          rebalance=5, delay=0)
    elif mode == 'old':
        d = C.tagdir('old')
        ex = c2eval.EXPORT_OLD
        J['primary'] = run(d / 'scores_full.npy', ex, rep, 'P_full, rebalance 1, delay 1, uniform (the file name)',
                           rebalance=1, delay=1)
        J['primary_vol'] = run(d / 'scores_full.npy', ex, rep, 'P_full, rebalance 1, delay 1, volatility-matched',
                               rebalance=1, delay=1, random_match='volatility')
        J['base'] = run(d / 'scores_base.npy', ex, rep, 'P_base, same cells, rebalance 1, delay 1', rebalance=1, delay=1)
        J['negR'] = run(d / 'scores_negR.npy', ex, rep, 'plain 1-day reversal -R, same cells, rebalance 1, delay 1',
                        rebalance=1, delay=1)
        J['full_d0'] = run(d / 'scores_full_d0.npy', ex, rep, "P_full at the paper's timing (fit on R(t+1)), delay 0",
                           rebalance=1, delay=0)
        J['week'] = run(d / 'scores_week.npy', ex, rep, 'weekly P_full, rebalance 5, delay 1', rebalance=5, delay=1)
        J['neg5'] = run(d / 'scores_neg5.npy', ex, rep, '5-day reversal on the weekly cells, rebalance 5, delay 1',
                        rebalance=5, delay=1)
        S = np.load(d / 'scores_full.npy', mmap_mode='r')
        pos = []
        for t in range(S.shape[1]):
            s = np.asarray(S[:, t])
            f = np.isfinite(s)
            if f.sum() >= 20:
                pos.append(int((np.sort(s[f])[-20:] > 0).sum()))
        C.log(rep, f'P_full: days on which all of the top 20 scores are positive: {np.mean(np.array(pos) == 20):.4f} '
                   f'of {len(pos)} (the judge buys only positive scores)')
    else:
        d = C.tagdir(mode)
        ex = C.SRC[mode]
        start = C.UNSEEN_START if mode == 'new' else C.os.environ.get('C2_FWD_START', '2026-10-20')
        kw = dict(walk_forward=False, start=start, rebalance=1, delay=1)
        J['primary'] = run(d / 'scores_full.npy', ex, rep, 'P_full, rebalance 1, delay 1, uniform', **kw)
        J['primary_vol'] = run(d / 'scores_full.npy', ex, rep, 'P_full, volatility-matched', random_match='volatility',
                               **kw)
        J['negR'] = run(d / 'scores_negR.npy', ex, rep, 'plain 1-day reversal -R, same cells', **kw)
        J['base'] = run(d / 'scores_base.npy', ex, rep, 'P_base, same cells', **kw)
    json.dump(J, open(C.OUT / f'money_{mode}.json', 'w'), indent=1, default=lambda o: None)


if __name__ == '__main__':
    main(sys.argv[1])
