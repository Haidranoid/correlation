"""Short volume: the money test (the judge).

Scores = the oriented G's percentile among universe stocks with a finite G, on every day from 2016-01-01 (NaN
before), graded with rebalance=5. Uniform and volatility-matched random accounts. Also, as the posting-time check,
the same scores with delay=1 (the account trades one day after each decision; FINRA posts day d's file after d's
close).

    python3 -I -B money.py old    # walk-forward; the years it grades start 2015-06-28 (half-empty) .. 2024-06-28
    python3 -I -B money.py new    # the unseen months 2025-06-30..2026-10-19 (once, at the end)
"""
import sys
sys.dont_write_bytecode = True
import json  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np, c2eval = C.np, C.c2eval


def scores(tag, sign):
    G = np.load(C.WORK / tag / 'G.npy')
    E = c2eval.load_export(C.EXPORTS[tag], arrays=())
    days = E['days']
    z = np.load(C.WORK / tag / 'universe.npz')
    uni, rebal = z['uni'], z['rebal']
    N, T = G.shape
    sc = np.full((N, T), np.nan, np.float32)
    d0 = c2eval.day_index(days, C.SCORE_FROM)
    names = []
    for d in range(d0, T):
        k = C.kidx(rebal, d)
        ok = uni[k] & np.isfinite(G[:, d])
        idx = np.nonzero(ok)[0]
        if len(idx) < C.MIN_NAMES:
            continue
        r = C.avg_rank(sign * G[idx, d].astype(np.float64))
        sc[idx, d] = ((r + 1) / len(idx)).astype(np.float32)
        names.append(len(idx))
    C.log(tag, 'scored days', len(names), 'median names', int(np.median(names)), 'first', c2eval.iso(days[d0]))
    return sc


def grade(js, walk):
    cor = c2eval.change_over_random(js)
    out = {'change': cor, 'at_cost': [c2eval.change_over_random(js, cost_index=i)['change'] for i in (0, 1, 2)]}
    if walk:
        ys = js['years']
        out['years'] = [{'start': y['start'], 'end': y.get('end'), 'picks': y['costs'][0]['final_pct'],
                         'random': y['costs'][0]['random_median_pct'], 'p': y['costs'][0].get('p'), 'spy': y.get('spy_pct'),
                         'ic_ret_t': y.get('ic_ret_t'), 'ls_bps_per_day': y.get('ls_bps_per_day'), 'ls_t': y.get('ls_t'),
                         'candidates': y.get('candidates'), 'first_day': y.get('first_day')} for y in ys]
        out['picks_mean'] = float(np.mean([y['costs'][0]['final_pct'] for y in ys]))
        out['random_mean'] = float(np.mean([y['costs'][0]['random_median_pct'] for y in ys]))
        out['summary_line'] = c2eval.summary_line(js)
        out['summary'] = js['summary']
        out['skipped'] = js.get('skipped')
    else:
        rep = js['report']
        out['strategy'] = rep['strategy']
        out['benchmarks'] = rep.get('benchmarks')
        out['report'] = {k: rep.get(k) for k in ('candidates', 'days', 'first_day', 'last_day', 'ic_ret_mean', 'ic_ret_t',
                                                  'ls_bps_per_day', 'ls_t', 'ls_turnover')}
    return out


def main(tag):
    sign = json.loads((C.OUT / 'train_old.json').read_text())['G']['sign']
    exp = C.EXPORTS[tag]
    sc = scores(tag, sign)
    (c2eval.WORK / 'short_volume').mkdir(parents=True, exist_ok=True)
    p = c2eval.save_scores(sc, f'short_volume/scores_G_{tag}', export=exp)
    del sc
    res, txts = {}, []
    for delay in (0, 1):
        for rm in ('uniform', 'volatility'):
            if tag == 'old':
                txt, js = c2eval.judge(p, rebalance=5, random_match=rm, delay=delay)
                r = grade(js, True)
            else:
                txt, js = c2eval.judge(p, export=exp, walk_forward=False, start='2025-06-30', rebalance=5, random_match=rm, delay=delay)
                r = grade(js, False)
            res[f'd{delay}|{rm}'] = r
            txts.append(f'===== delay={delay} random_match={rm} =====\n{txt}')
            C.log(f'delay {delay} {rm}: change {r["change"]["change"]:+.2f} (ahead {r["change"]["ahead"]}/{r["change"]["years"]}, '
                  f'se {r["change"]["se"]:.2f}); at 0/5/20 bps {np.round(r["at_cost"], 2).tolist()}')
            if tag == 'old':
                C.log('   ', r['summary_line'])
    (C.OUT / f'money_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    (C.OUT / f'judge_{tag}.txt').write_text('\n'.join(txts))


if __name__ == '__main__':
    main(sys.argv[1])
