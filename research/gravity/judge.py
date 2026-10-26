"""Gravity, step 6: the judge's accounts (improvement over random and the file-name number).

    python -I -B judge.py old|new

Each of the 30 gravity candidates and the benchmarks (the spring; the three sharp offsets; G3's benchmarks are G1
candidates) goes to c2eval.judge as scores: the force's centred rank among the day's universe stocks with a force
(NaN elsewhere), delay 1 (the next close), daily rebalance, costs 0, 5 and 20 bps, 1,000 uniform random accounts.
- old: the judge's walk-forward; its years 2008-09 .. 2024-25 (start >= 2008-06-28; 17 years) are graded, the
  2007-08 year it also runs is dropped. Per year: the account's return at 0/5/20 bps, the median random account's,
  SPY's Yahoo total return over the year's first..last day; improvement over random = account - random at 0 bps;
  the after-cost number = account at 5 bps - SPY. Averaged over the years, then over the 30 candidates.
- new: one holdout over the unseen months (2025-06-30 .. 2026-10-19), the new export, once at the end.
Writes BIG/judge_<tag>/ (scores, the judge's JSON) and out/judge_<tag>.json / .txt.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import data  # noqa: E402
import kern  # noqa: E402
np = K.np
c2eval = K.c2eval

VARS = K.variants()
JUDGED = [v for v in VARS if v['world'] == 'real' and v['kind'] in ('cand', 'bench')]   # 30 + 4
FIRST_YEAR = '2008-06-28'


def scores_for(tag, v, P, E_rows, T_exp):
    F = np.asarray(np.load(K.BIG / f'forces_{tag}' / f'{v["key"]}.npy', mmap_mode='r'), np.float64)
    M = np.asarray(P['inU']) & np.isfinite(F)
    rk = kern.rank_center(np.where(M, F, 0.0), M)
    S = np.full((E_rows, T_exp), np.nan, np.float32)
    S[np.asarray(P['rows'])] = rk[:, :T_exp].astype(np.float32)
    return S


def main(tag):
    os.environ.setdefault('RAYON_NUM_THREADS', '4')
    out_dir = K.BIG / f'judge_{tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(K.OUT / f'judge_{tag}.txt', 'w')
    exp = K.SRC[tag][0]
    E = c2eval.load_export(exp, arrays=())
    P = data.load_panel(tag)
    assert P['meta']['fingerprint'] == E['meta']['fingerprint']
    res = {'tag': tag, 'accounts': []}
    t0 = time.time()
    for v in JUDGED:
        jp = out_dir / f'{v["key"]}.json'
        sp = out_dir / f'{v["key"]}.npy'
        if not jp.exists():
            S = scores_for(tag, v, P, len(E['ids']), len(E['days']))
            np.save(sp, S)
            del S
            if tag == 'old':
                c2eval.judge(sp, export=exp, rebalance=1, delay=1, walk_forward=True, start='2024-06-28', costs='0,5,20',
                             json_out=jp)
            else:
                c2eval.judge(sp, export=exp, rebalance=1, delay=1, walk_forward=False, start=K.UNSEEN[0], costs='0,5,20',
                             json_out=jp)
            sp.unlink()       # the scores are rebuilt from the forces; keep the judge's JSON
        js = json.loads(jp.read_text())
        rec = {'key': v['key'], 'name': v['name'], 'stage': v['stage'], 'kind': v['kind']}
        if tag == 'old':
            yrs = [y for y in js['years'] if y['start'] >= FIRST_YEAR]
            rows = []
            for y in yrs:
                c = {int(x['cost_bps']): x for x in y['costs']}
                spy = K.yahoo_spy_pct(y['first_day'], y['last_day'])
                rows.append({'start': y['start'], 'first_day': y['first_day'], 'last_day': y['last_day'],
                             'acct0': c[0]['final_pct'], 'acct5': c[5]['final_pct'], 'acct20': c[20]['final_pct'],
                             'rand0': c[0]['random_median_pct'], 'rand5': c[5]['random_median_pct'], 'rand20': c[20]['random_median_pct'],
                             'p0': c[0]['p'], 'p5': c[5]['p'], 'spy': spy,
                             'over_random0': c[0]['final_pct'] - c[0]['random_median_pct'], 'after_cost': c[5]['final_pct'] - spy,
                             'ls_bps': y['ls_bps_per_day'], 'ls_t': y['ls_t'], 'ic_ret': y['ic_ret_mean'], 'ic_ret_t': y['ic_ret_t']})
            d = np.array([r['over_random0'] for r in rows])
            a = np.array([r['after_cost'] for r in rows])
            rec.update({'years': rows, 'n_years': len(rows), 'span': f'{rows[0]["first_day"]}..{rows[-1]["last_day"]}',
                        'over_random0': float(d.mean()), 'over_random0_se': float(d.std(ddof=1) / np.sqrt(len(d))),
                        'ahead': int((d > 0).sum()), 'after_cost': float(a.mean()), 'after_cost_se': float(a.std(ddof=1) / np.sqrt(len(a))),
                        'over_random5': float(np.mean([r['acct5'] - r['rand5'] for r in rows])),
                        'over_random20': float(np.mean([r['acct20'] - r['rand20'] for r in rows])),
                        'acct5_mean': float(np.mean([r['acct5'] for r in rows])), 'spy_mean': float(np.mean([r['spy'] for r in rows])),
                        'beat_p_lt_0_1_at0': int(sum(r['p0'] < 0.1 for r in rows)),
                        'skipped_or_dropped': [y['start'] for y in js['years'] if y['start'] < FIRST_YEAR] + list(js.get('skipped', []))})
            K.log(f'{v["name"]:>30}: {rec["n_years"]} years {rec["span"]}; over random (0 bps) {rec["over_random0"]:+.2f} '
                  f'(se {rec["over_random0_se"]:.2f}, ahead {rec["ahead"]}); account 5 bps - SPY {rec["after_cost"]:+.2f} '
                  f'[{time.time() - t0:.0f}s]', fh)
        else:
            rep = js['report']
            c = {int(x['cost_bps']): x for x in rep['strategy']}
            spy = K.yahoo_spy_pct(rep['first_day'], rep['last_day'])
            rec.update({'first_day': rep['first_day'], 'last_day': rep['last_day'], 'acct0': c[0]['final_pct'],
                        'acct5': c[5]['final_pct'], 'acct20': c[20]['final_pct'], 'rand0': c[0]['random_median_pct'],
                        'rand5': c[5]['random_median_pct'], 'p0': c[0]['p'], 'spy': spy,
                        'over_random0': c[0]['final_pct'] - c[0]['random_median_pct'], 'after_cost': c[5]['final_pct'] - spy})
            K.log(f'{v["name"]:>30}: {rec["first_day"]}..{rec["last_day"]}: account 5 bps {rec["acct5"]:+.2f}% vs SPY '
                  f'{spy:+.2f}%; over random (0 bps) {rec["over_random0"]:+.2f} [{time.time() - t0:.0f}s]', fh)
        res['accounts'].append(rec)
    acc = res['accounts']
    cand = [a for a in acc if a['kind'] == 'cand']
    res['summary'] = {'over_random0_mean_30': float(np.mean([a['over_random0'] for a in cand])),
                      'after_cost_mean_30': float(np.mean([a['after_cost'] for a in cand])), 'n_candidates': len(cand)}
    for st in ('G1', 'G2', 'G3'):
        cs = [a for a in cand if a['stage'] == st]
        res['summary'][st] = {'over_random0': float(np.mean([a['over_random0'] for a in cs])),
                              'after_cost': float(np.mean([a['after_cost'] for a in cs])), 'n': len(cs)}
    if tag == 'old':
        ys = np.array([[r['over_random0'] for r in a['years']] for a in cand])
        ya = np.array([[r['after_cost'] for r in a['years']] for a in cand])
        res['summary']['by_year'] = [{'start': r['start'], 'over_random0_mean': float(ys[:, i].mean()), 'after_cost_mean': float(ya[:, i].mean()),
                                      'spy': r['spy']} for i, r in enumerate(cand[0]['years'])]
        m = ys.mean(0)
        res['summary']['over_random0_se_years'] = float(m.std(ddof=1) / np.sqrt(len(m)))
        res['summary']['over_random0_ahead_years'] = int((m > 0).sum())
        a = ya.mean(0)
        res['summary']['after_cost_se_years'] = float(a.std(ddof=1) / np.sqrt(len(a)))
        res['summary']['after_cost_ahead_years'] = int((a > 0).sum())
        d08 = [i for i, r in enumerate(cand[0]['years']) if r['start'] < '2019-06-28']
        res['summary']['over_random0_2008_2019'] = float(ys[:, d08].mean())
        res['summary']['after_cost_2008_2019'] = float(ya[:, d08].mean())
    K.log(f'SUMMARY: {res["summary"]}', fh)
    K.jdump(res, K.OUT / f'judge_{tag}.json')
    fh.close()


if __name__ == '__main__':
    main(sys.argv[1])
