"""Test 6, the money test, weekly: in each group in the top third of groups by this week's market-relative return,
short its imposters and buy as many true members matched on exposure to the group; in the bottom third, the reverse.
Hold 4 weeks (the book is the average of the last 4 weekly cohorts). Against random guessers that hold the same
numbers of stocks drawn at random from the same groups' members (fixed through each classification month, so they
trade about as often). Also writes the judge's scores file (long side positive, short side negative).

    python3 -I money.py      # prints the tables; writes money_out.json next to this script and the scores .npy
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import HERE, K, c2eval, np  # noqa: E402
from tests import Blocks, CL, nw_t  # noqa: E402

RUNS = 1000
COST_BPS = (0.0, 5.0, 10.0)
HOLD = 4
FIRST_YEAR, LAST_YEAR = 2006, 2021   # judge test years 2006-07 .. 2021-22 (the years with text)


def build():
    bl = Blocks()
    days, ret = bl.days, bl.ret
    N, T = ret.shape
    months = bl.month_list
    cohorts = []          # per week: dict(tw, rows, w, slots, keys)
    subs_n = {}           # (m, g) -> number of imposter slots
    members = {}          # (m, g) -> export rows of the group's members
    scores = np.full((N, T), np.nan, np.float32)
    for mi, m in enumerate(months):
        blk = bl.month(m)
        c, a = blk['c'], blk['a']
        rows = c['rows']
        beta = c['beta'].astype(float)
        imp_i = np.nonzero(c['imp'])[0]
        true_i = np.nonzero(c['true'])[0]
        lab = c['label']
        for w in bl.weeks(blk):
            t0, tw = bl.we[w - 1], bl.we[w]
            if w + HOLD >= len(bl.we):
                continue
            te = bl.we[w + HOLD]
            R = np.array([(np.where(blk['C'][g] > 0, blk['S'][g] / np.maximum(blk['C'][g], 1), 0.0))
                          [t0 - a + 1:tw - a + 1].sum() for g in range(K)])
            order = np.argsort(R)
            weak, strong = set(order[:K // 3]), set(order[-(K // 3):])
            pos_rows, pos_sign, pos_slot, pos_str, pos_blk = [], [], [], [], []
            for g in range(K):
                if g not in weak and g not in strong:
                    continue
                gi = imp_i[c['gstar'][imp_i] == g]
                gt = true_i[c['gstar'][true_i] == g]
                if len(gi) == 0 or len(gt) == 0:
                    continue
                s = -1.0 if g in strong else 1.0     # sign of the imposter side
                used = set()
                for k, i in enumerate(gi[np.argsort(-beta[gi, g])]):
                    free = [j for j in gt if j not in used]
                    pool = np.array(free if free else list(gt))
                    j = pool[np.argmin(np.abs(beta[pool, g] - beta[i, g]))]
                    used.add(j)
                    pos_rows += [rows[i], rows[j]]
                    pos_blk += [i, j]
                    pos_sign += [s, -s]
                    pos_slot += [(m, g, 0, k), (m, g, 1, k)]
                    pos_str += [abs(R[g])] * 2
                subs_n[(m, g)] = len(gi)
                members[(m, g)] = rows[lab == g]
            if not pos_rows:
                cohorts.append(dict(tw=tw, te=te, rows=np.zeros(0, int), w=np.zeros(0), slots=[]))
                continue
            pos_rows, pos_sign = np.array(pos_rows), np.array(pos_sign)
            npairs = len(pos_rows) // 2
            wts = pos_sign / npairs           # $1 long and $1 short per cohort
            # market-relative return of the cohort over its hold (tw, te], from the month's block
            xs = blk['x'][np.array(pos_blk), tw - a + 1:te - a + 1].sum(1)
            cohorts.append(dict(tw=tw, te=te, rows=pos_rows, w=wts, slots=pos_slot, mr=float((wts * xs).sum())))
            for r_, sg, st in zip(pos_rows, pos_sign, pos_str):
                cur = scores[r_, tw:te]
                scores[r_, tw:te] = np.where(np.isnan(cur), 0, cur) + sg * st
    # one cohort per week-end (a week-end can appear only once: months don't overlap)
    return bl, cohorts, subs_n, members, scores


def evaluate(bl, cohorts, subs_n, members):
    days, ret = bl.days, bl.ret
    N, T = ret.shape
    r0 = np.where(np.isfinite(ret), ret, 0.0)
    C = np.zeros((N, T + 1))
    np.cumsum(r0, axis=1, out=C[:, 1:])       # C[:, t+1] = sum of returns through day t
    years = list(range(FIRST_YEAR, LAST_YEAR + 1))
    ys = [c2eval.day_index(days, f'{y}-06-28') for y in years] + [c2eval.day_index(days, f'{LAST_YEAR + 1}-06-28')]
    rng = np.random.default_rng(20261021)
    # random substitutes per (month, group): [RUNS, 2 * n] rows
    subs = {}
    for key, n in subs_n.items():
        mem = members[key]
        if len(mem) >= 2 * n:
            idx = np.argsort(rng.random((RUNS, len(mem))), axis=1)[:, :2 * n]
        else:
            idx = rng.integers(len(mem), size=(RUNS, 2 * n))
        subs[key] = mem[idx]
    gross = np.zeros(len(years))
    gross_r = np.zeros((len(years), RUNS))
    turn = np.zeros(len(years))
    turn_r = np.zeros((len(years), RUNS))
    weekly = []   # (tw, strategy contribution in the week's cohort, over its hold)

    def year_of(t):
        k = np.searchsorted(ys, t, side='right') - 1
        return k if 0 <= k < len(years) else None

    rand_rows = []
    for ci, co in enumerate(cohorts):
        if len(co['rows']):
            rr = np.stack([subs[(m, g)][:, side * subs_n[(m, g)] + k] for (m, g, side, k) in co['slots']], axis=1)
        else:
            rr = np.zeros((RUNS, 0), int)
        rand_rows.append(rr)
        # returns over the hold (tw, te], split by test year
        s, e = co['tw'], co['te']
        for yi in range(len(years)):
            lo, hi = max(s, ys[yi] - 1), min(e, ys[yi + 1] - 1)
            if hi <= lo or not len(co['rows']):
                continue
            gross[yi] += (co['w'] * (C[co['rows'], hi + 1] - C[co['rows'], lo + 1])).sum() / HOLD
            gross_r[yi] += ((C[rr, hi + 1] - C[rr, lo + 1]) * co['w'][None, :]).sum(1) / HOLD
        if len(co['rows']):
            weekly.append((s, (co['w'] * (C[co['rows'], e + 1] - C[co['rows'], s + 1])).sum()))
        # turnover at tw: the book changes by (cohort_w - cohort_{w-4}) / 4
        yi = year_of(s)
        if yi is None:
            continue
        old = cohorts[ci - HOLD] if ci >= HOLD else dict(rows=np.zeros(0, int), w=np.zeros(0))
        oldr = rand_rows[ci - HOLD] if ci >= HOLD else np.zeros((RUNS, 0), int)
        rows_s = np.concatenate([co['rows'], old['rows']])
        w_s = np.concatenate([co['w'], -old['w']])
        if len(rows_s):
            u, inv = np.unique(rows_s, return_inverse=True)
            turn[yi] += np.abs(np.bincount(inv, weights=w_s)).sum() / HOLD
            keys = np.concatenate([rr, oldr], axis=1) + (np.arange(RUNS) * N)[:, None]
            wk = np.broadcast_to(w_s, keys.shape)
            u, inv = np.unique(keys.ravel(), return_inverse=True)
            sums = np.abs(np.bincount(inv, weights=wk.ravel()))
            turn_r[yi] += np.bincount(u // N, weights=sums, minlength=RUNS) / HOLD
    out = {'years': years, 'per_year': {}, 'overall': {}}
    for cb in COST_BPS:
        net = 100 * (gross - cb * 1e-4 * turn)
        net_r = 100 * (gross_r - cb * 1e-4 * turn_r)
        med = np.median(net_r, axis=1)
        p_year = ((net_r >= net[:, None]).sum(1) + 1) / (RUNS + 1)
        tot, tot_r = net.mean(), net_r.mean(0)
        p_all = ((tot_r >= tot).sum() + 1) / (RUNS + 1)
        diff = net - med
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        out['overall'][cb] = dict(strategy_pct_yr=float(tot), random_median_pct_yr=float(np.median(tot_r)),
                                  change=float(diff.mean()), se=float(se), ahead=int((diff > 0).sum()),
                                  years=len(diff), p=float(p_all))
        out['per_year'][cb] = [dict(year=f'{y}-{(y + 1) % 100:02d}', strategy=float(a), random_median=float(b),
                                    p=float(p)) for y, a, b, p in zip(years, net, med, p_year)]
    out['turnover_per_year'] = [float(x) for x in turn]
    out['random_turnover_per_year'] = [float(x) for x in turn_r.mean(1)]
    wk = np.array([v for _, v in weekly])
    mean, se, t, n = nw_t(wk, HOLD - 1)
    out['cohort_4w_return'] = dict(mean_pct=100 * mean, t=t, cohorts=n)
    mr = np.array([co['mr'] for co in cohorts if len(co['rows']) and ys[0] <= co['tw'] < ys[-1]])
    mean, se, t, n = nw_t(mr, HOLD - 1)
    # a cohort is a quarter of the book for 4 weeks: yearly % = mean cohort return x weeks per year / 4 x 4
    out['market_relative'] = dict(cohort_4w_mean_pct=100 * mean, t=t, cohorts=n, pct_per_year=100 * mean * 52 / HOLD)
    return out


def main():
    bl, cohorts, subs_n, members, scores = build()
    out = evaluate(bl, cohorts, subs_n, members)
    # scores only inside the test years with text (2006-07 .. 2021-22); the last cohorts' holds would spill a few
    # weeks into 2022-23, which the judge would grade as a year of its own. The judge decides on day d for day d+1,
    # so the last day of 2021-22 (its decision day for 2022-23's first day) gets no score either.
    end = c2eval.day_index(bl.days, f'{LAST_YEAR + 1}-06-28')
    scores[:, end - 1:] = np.nan
    p = c2eval.save_scores(scores, 'imposters/imposters_money')
    out['scores'] = str(p)
    out['cohorts_with_trades'] = int(sum(len(c['rows']) > 0 for c in cohorts))
    out['positions_per_cohort'] = float(np.mean([len(c['rows']) for c in cohorts if len(c['rows'])]))
    for reb in (5, 1):
        txt, js = c2eval.judge(p, rebalance=reb)
        out[f'judge_r{reb}'] = dict(summary=c2eval.summary_line(js), change=c2eval.change_over_random(js),
                                    change_5bps=c2eval.change_over_random(js, 1), text=txt)
        print(f'judge rebalance={reb}:', c2eval.summary_line(js), flush=True)
    (HERE / 'money_out.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_year'}, indent=1))
    for cb in COST_BPS:
        print(f'{cb} bps:', ' '.join(f"{r['year']}:{r['strategy']:+.1f}/{r['random_median']:+.1f}(p{r['p']:.2f})"
                                     for r in out['per_year'][cb]))


if __name__ == '__main__':
    main()
