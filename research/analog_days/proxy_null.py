"""Diagnostic (added after seeing the judge's results): a random-sector null for the proxy account.

The judge's account holds the 20 highest scores, i.e. the 20 stocks most correlated with the top-ranked sector: a
one-sector bet. Its random accounts draw across all sectors, so they understate how far a one-sector account swings in
a single period. Here the judge's account is replayed (its candidates, rebalance days, daily equal weight, 0 bps), and
compared with 2,000 accounts that rank the 9 sectors at random on each rebalance day and otherwise follow the same rule.

    python3 -I proxy_null.py        # writes out/proxy_null.txt / .json
"""
import json
import sys as _sys, os as _os
import warnings

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

RUNS = 2000
LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def judge_candidates(E, h0, window=756):
    """The judge's liquid universe for a holdout starting at h0 (stockai judge holdout.rs liquid_universe)."""
    v = np.asarray(E['v'][:, max(h0 - window, 0):h0])
    ok = np.isfinite(v).sum(1) * 10 >= (h0 - max(h0 - window, 0)) * 9
    tail = min(h0 - max(h0 - window, 0), 252)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mp = np.nanmedian(np.asarray(E['close'][:, h0 - tail:h0]), 1)
        md = np.nanmedian(np.asarray(E['dollar_volume'][:, h0 - tail:h0]), 1)
    return np.nonzero(ok & (np.nan_to_num(mp) >= 5.0) & (np.nan_to_num(md) >= 5e6))[0]


def sector_of(tag, E):
    """[T-block] the mapping's sector per stock per rebalance, from money.mapping (same rules)."""
    import money
    P = C.build(tag)
    tw = money.twins_read_only(E)
    maps = money.mapping(E, tw, P)
    return P, tw, maps


def main():
    rng = np.random.default_rng(11)
    out = {}
    for tag in ('old', 'new'):
        E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'dollar_volume', 'v'))
        S = np.load(C.WORK / f'analog_proxy_{tag}.npy', mmap_mode='r')
        P, tw, maps = sector_of(tag, E)
        has = P['idx'][:, 0] >= 0
        R = np.full((len(P['days']), 9), np.nan)
        R[has] = np.argsort(np.argsort(P['F'][has], 1, kind='stable'), 1) + 1
        T = len(E['days'])
        if tag == 'old':
            periods = [(c2eval.day_index(E['days'], f'{y}-06-28'), c2eval.day_index(E['days'], f'{y + 1}-06-28')) for y in range(2007, 2025)]
            periods[-1] = (periods[-1][0], T)
        else:
            periods = [(c2eval.day_index(E['days'], '2025-06-30'), T)]
        js = json.loads((C.WORK / f'analog_proxy_{tag}__r5_d0_uniform.json').read_text())
        judge_final = [y['costs'][0]['final_pct'] for y in js['years']] if js.get('years') else [js['report']['strategy'][0]['final_pct']]
        res = []
        for (h0, t), jf in zip(periods, judge_final):
            rebal = tw['rebal']
            Sd = np.asarray(S[:, h0 - 1:t], np.float32)
            a, rr, nc = replay_blocks(E, Sd, h0, t, R, maps, rebal, rng)
            p = float((rr >= a).mean())
            res.append(dict(start=c2eval.iso(E['days'][h0]), analog_pct=a, judge_pct=jf, random_sector_median=float(np.median(rr)),
                            p=p, p5=float(np.percentile(rr, 5)), p95=float(np.percentile(rr, 95)), candidates=nc))
            say(f'{tag} {c2eval.iso(E["days"][h0])}: analog account {a:+7.1f}% (judge {jf:+7.1f}%), random-sector median '
                f'{np.median(rr):+7.1f}% (5-95%: {np.percentile(rr, 5):+.1f}..{np.percentile(rr, 95):+.1f}), p = {p:.3f}')
        out[tag] = res
        if tag == 'old':
            d = np.array([x['analog_pct'] - x['random_sector_median'] for x in res])
            say(f'old walk-forward: analog minus random-sector median {d.mean():+.2f} pts/yr (se {d.std(ddof=1) / np.sqrt(len(d)):.2f}), '
                f'ahead {(d > 0).sum()}/{len(d)}; p < 0.1 in {sum(x["p"] < 0.1 for x in res)} of {len(res)} years')
    (C.OUT / 'proxy_null.txt').write_text('\n'.join(LINES) + '\n')
    (C.OUT / 'proxy_null.json').write_text(json.dumps(out, indent=1, default=float))


def replay_blocks(E, Sd, h0, t, R, maps, rebal, rng):
    """The analog account and RUNS random-sector accounts over one period (decision days h0-1 .. t-2, every 5th), each
    rebalance day using the mapping (sector per stock) of the latest twins rebalance."""
    cand = judge_candidates(E, h0)
    c = np.asarray(E['close'][cand, h0 - 1:t], np.float64)
    with np.errstate(all='ignore'):
        r = c[:, 1:] / c[:, :-1] - 1
    r[~np.isfinite(r)] = 0.0
    an = 1.0
    rn = np.ones(RUNS)
    for d in np.arange(h0 - 1, t - 1, 5):
        hold = np.arange(d, min(d + 5, t - 1)) - (h0 - 1)
        s = Sd[cand, d - (h0 - 1)].astype(np.float64)
        fin = np.isfinite(s) & (s > 0)
        if not fin.any() or not np.isfinite(R[d]).all():
            continue
        k = int(np.searchsorted(rebal, d, side='right') - 1)
        secmap = np.full(E['close'].shape[0], -1)
        secmap[maps[k]['rows']] = maps[k]['sec']
        rows = np.nonzero(fin)[0]
        sec = secmap[cand[rows]]
        assert (sec >= 0).all()
        corr = s[rows] - R[d][sec]                                  # score = rank + corr
        assert (corr > -1e-4).all() and (corr < 1).all()

        def top20(order):
            rank = np.empty(9); rank[order] = np.arange(9, 0, -1)
            return rows[np.argsort(-(rank[sec] + corr), kind='stable')[:20]]

        def grow(sel):
            return float(np.prod(1 + r[sel][:, hold].sum(0) / 20.0)) if len(hold) else 1.0
        an *= grow(top20(np.argsort(-R[d], kind='stable')))
        cnt = np.bincount(sec, minlength=9)
        cache = {}
        for i in range(RUNS):
            o = rng.permutation(9)
            j = int(np.searchsorted(np.cumsum(cnt[o]), 20)) + 1     # the sectors that fill the top 20
            key = tuple(o[:j])
            if key not in cache:
                cache[key] = grow(top20(o))
            rn[i] *= cache[key]
    return 100 * (an - 1), 100 * (rn - 1), len(cand)


if __name__ == '__main__':
    main()
