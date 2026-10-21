"""Checks of the code and the data (not results), on the old export:
- coverage by year (universe, ranked set, twin quality, how often the signal's twin differs from the hedge's), and
  Test 1's start chosen from coverage alone;
- plain 12-1 momentum recomputed with stockai's function vs stockai/runs/poc/baselines/momentum_12_1.npy;
- the twin and market-residual signals recomputed by hand for one stock (window t-252 .. t-21, beta from k-1);
- an oracle signal (the next 21 days' return, look-ahead on purpose) must score hugely; one hedged leg by hand;
  the legs are the deciles.

    python3 -I checks.py      # writes out/checks.txt, out/coverage_old.json
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def start_index(P, cov):
    """Test 1's first formation (rule fixed before any returns): the first rebalance on or after 2006-01-01 with a
    full 500-stock universe, more than one distinct twin, SPY betas, and >= 450 universe stocks with every signal."""
    lo = c2eval.day_index(P.days, '2006-01-01')
    return next(c['k'] for c in cov if P.rebal[c['k']] >= lo and c['universe'] >= C.N_UNIV and c['distinct_twins'] > 1
                and c['spy_beta_ok'] and c['ranked'] >= C.MIN_RANKED)


def main():
    E, tw = C.load(c2eval.EXPORT_OLD)
    say(f'gap days filled: stocks {E["gap_days"]}, funds {E["fund_gap_days"]}')
    P = C.Panel(E, tw)
    masks = C.ranked_sets(P, [P.s_twin, P.s_plain, P.s_mkt, P.s_twin_raw, P.s_mkt_raw])
    cov = C.coverage(P, masks)
    k0 = start_index(P, cov)
    say(f'Test 1 start: rebalance {k0} = {c2eval.iso(P.days[P.rebal[k0]])}')
    for c in cov:
        if c['day'] >= '2005-06' and c['day'] <= '2006-06':
            say(f"  {c['day']} k {c['k']} eligible {c['eligible']} universe {c['universe']} ranked {c['ranked']} "
                f"distinct twins {c['distinct_twins']} spy {c['spy_beta_ok']}")
    say('\n== Coverage by year (mean over the year\'s rebalances) ==')
    say('year  eligible  universe  ranked  plain500-no-twin  plain500-excluded  median-corr  distinct-twins  broad  twin-changed(k vs k-1)')
    rows = {}
    for y in range(2003, 2026):
        cs = [c for c in cov if c['day'][:4] == str(y)]
        if not cs:
            continue
        r = {f: float(np.nanmean([c[f] for c in cs])) for f in
             ('eligible', 'universe', 'ranked', 'plain_no_twin', 'plain_excluded', 'median_corr', 'distinct_twins',
              'broad_share', 'twin_changed')}
        rows[y] = r
        say(f"{y}  {r['eligible']:7.0f}  {r['universe']:5.0f}  {r['ranked']:5.0f}  {r['plain_no_twin']:6.1f}  "
            f"{r['plain_excluded']:6.1f}  {r['median_corr']:.3f}  {r['distinct_twins']:5.0f}  {r['broad_share']:.2f}  "
            f"{r['twin_changed']:.3f}")
    (C.OUT / 'coverage_old.json').write_text(json.dumps({'start_k': k0, 'start': c2eval.iso(P.days[P.rebal[k0]]),
                                                         'years': rows, 'rebalances': cov}, indent=1, default=float))

    # plain momentum vs stockai's stored file (computed by stockai on raw closes; stops 2024-06)
    say('\n== Plain 12-1 momentum vs stockai/runs/poc/baselines/momentum_12_1.npy ==')
    stored = np.load('/home/bhoult/programming/correlation/stockai/runs/poc/baselines/momentum_12_1.npy', mmap_mode='r')
    raw = C.momentum_scores(c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))['close'], C.SKIP, C.LOOK)
    last = int(np.nonzero(np.isfinite(stored).any(0))[0][-1])
    say(f'stored finite through column {last} = {c2eval.iso(P.days[last])}')
    mx, n_both, n_dis = 0.0, 0, 0
    for a in range(0, last + 1, 500):
        b = min(a + 500, last + 1)
        s_, r_ = np.asarray(stored[:, a:b]), raw[:, a:b]
        both = np.isfinite(s_) & np.isfinite(r_)
        n_both += int(both.sum())
        n_dis += int((np.isfinite(s_) != np.isfinite(r_)).sum())
        if both.any():
            mx = max(mx, float(np.abs(s_[both] - r_[both]).max()))
    say(f'raw closes, columns 0..{last}: {n_both} cells finite in both, max abs diff {mx:.2e}; '
        f'{n_dis} cells finite in only one')
    fil = P.s_plain.astype(np.float64)
    both = np.isfinite(raw) & np.isfinite(fil)
    ch = int((np.abs(raw[both] - fil[both]) > 1e-6).sum()) + int((np.isfinite(raw) != np.isfinite(fil)).sum())
    on_reb = int(sum(((np.abs(raw[:, t] - fil[:, t]) > 1e-6) | (np.isfinite(raw[:, t]) != np.isfinite(fil[:, t]))).sum()
                     for t in P.rebal))
    say(f'gap-filled closes change {ch} of {int(np.isfinite(raw).sum())} finite cells (float32 rounding aside), '
        f'{on_reb} of them on rebalance days')
    del raw, fil, both

    # hand check of the two residual signals for one stock at one rebalance
    say('\n== Signals by hand ==')
    k = 150
    t = int(P.rebal[k])
    rows = np.nonzero(masks[k])[0]
    c = np.asarray(P.close, np.float64)
    fc = np.asarray(tw['F']['close'], np.float64)
    for j in rows[:400]:
        f = int(tw['fund_row'][k - 1, j])
        seg = c[j, t - C.LOOK:t - C.SKIP + 1]
        fseg = fc[f, t - C.LOOK:t - C.SKIP + 1]
        if np.isfinite(seg).all() and np.isfinite(fseg).all():
            break
    beta, rsd = float(tw['beta'][k - 1, j]), float(tw['resid_sd'][k - 1, j])
    rs = np.log(c[j, t - C.SKIP] / c[j, t - C.LOOK])
    rf = np.log(fc[f, t - C.SKIP] / fc[f, t - C.LOOK])
    hand = (rs - beta * rf) / (rsd * np.sqrt(C.WIN))
    say(f'twin signal {E["tickers"][j]} vs {tw["F"]["tickers"][f]} (twin at k-1), beta {beta:.3f}, decided '
        f'{c2eval.iso(P.days[t])}: code {P.s_twin[j, t]:+.6f} hand {hand:+.6f}; twin at k '
        f'{tw["F"]["tickers"][tw["fund_row"][k, j]]}')
    hand_raw = rs - beta * rf
    say(f'  unscaled: code {P.s_twin_raw[j, t]:+.6f} hand {hand_raw:+.6f}; plain: code {P.s_plain[j, t]:+.6f} hand {rs:+.6f}')
    mb, mr = float(P.mbeta[k - 1, j]), float(P.mresid_sd[k - 1, j])
    rspy = np.log(fc[P.spy, t - C.SKIP] / fc[P.spy, t - C.LOOK])
    hm = (rs - mb * rspy) / (mr * np.sqrt(C.WIN))
    say(f'market-residual signal: beta {mb:.3f}: code {P.s_mkt[j, t]:+.6f} hand {hm:+.6f}; unscaled code '
        f'{P.s_mkt_raw[j, t]:+.6f} hand {rs - mb * rspy:+.6f}')
    # correlation of the signals across the ranked sets
    cors = {}
    for nm, s in (('twin~plain', P.s_plain), ('twin~mkt', P.s_mkt), ('twin~twin_raw', P.s_twin_raw)):
        cc = []
        for kk in range(60, P.K):
            r_ = np.nonzero(masks[kk])[0]
            if len(r_) > 50:
                x, y = P.s_twin[r_, P.rebal[kk]], s[r_, P.rebal[kk]]
                cc.append(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
        cors[nm] = float(np.mean(cc))
    say('mean cross-sectional rank correlation with the twin signal: ' + ', '.join(f'{a} {b:.2f}' for a, b in cors.items()))

    # oracle and the book code
    say('\n== Book code ==')
    oracle = np.full((P.N, P.T), np.nan, np.float32)
    oracle[:, :-C.HOLD] = np.log(P.cff[:, C.HOLD:] / P.cff[:, :-C.HOLD])
    ks = list(range(k0, P.K - 2))
    for hedge in ('twin', None):
        ms = C.run_strategy(P, ks, oracle, hedge, masks)
        m, tt, n = C.tstat([x['gross'] for x in ms])
        say(f'oracle (next 21 days from close t), hedge {hedge}: {100 * m:+.2f}%/month t {tt:+.1f} n {n}')
    del oracle
    b = C.month_book(P, k, P.s_twin, 'twin', masks[k])
    jj = int(b['L'][0])
    f = int(tw['fund_row'][k, jj])
    beta = float(tw['beta'][k, jj])
    hand = (c[jj, t + 21] / c[jj, t] - 1) - beta * (fc[f, t + 21] / fc[f, t] - 1)
    say(f'hand leg {E["tickers"][jj]} vs {tw["F"]["tickers"][f]} beta {beta:.3f} formed {c2eval.iso(P.days[t])}: '
        f'code {b["legs"][0]:+.6f} hand {hand:+.6f}')
    s = P.s_twin[np.nonzero(masks[k])[0], t]
    say(f'deciles: n {b["n"]} legs {len(b["L"])}/{len(b["S"])}; min s in long {P.s_twin[b["L"], t].min():.3f} = 90th '
        f'pct {np.sort(s)[-len(b["L"])]:.3f}; max s in short {P.s_twin[b["S"], t].max():.3f} = 10th pct '
        f'{np.sort(s)[len(b["S"]) - 1]:.3f}')
    say(f'hedge: net fund weight {sum(b["fund_end"].values()):+.3f} (end of hold); long-leg mean beta '
        f'{np.mean(tw["beta"][k, b["L"]]):.2f}, short-leg {np.mean(tw["beta"][k, b["S"]]):.2f}')
    (C.OUT / 'checks.txt').write_text('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
