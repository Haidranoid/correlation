"""Step 0: reproduce the file's headline numbers before any haircut.

- evaluate.py on the raw LightGBM scores (tie-sharing): picks +3.21% vs +1.03% a month (t 2.8), +2.05 over comparable
  stocks (t 3.57), over 162 graded months;
- the grader's account on seed 2 (runs/30d/jitter/lgbm_s2.npy): yearly returns at 0/5/20/50 bps against the JSON,
  $230,343 at 20 bps; SPY $65,036 from the Stooq store, and SPY total return from pubdata;
- the same 162-month statistics for seed 2's picks as the grader made them (the haircut's starting point).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json  # noqa: E402

import numpy as np  # noqa: E402

import common as C  # noqa: E402


def evaluate_raw(E, specs, scores32):
    """evaluate.py / metrics.grade_day on float32 scores with tie-sharing weights."""
    close = E['close']
    pick, allr, comp = [], [], []
    for y, h0, t, uni, graded in specs:
        for d in graded:
            d = int(d)
            f = C.fwd_col(close, uni, d)
            s = np.asarray(scores32[uni, d], dtype=np.float32)
            ok = np.isfinite(f) & np.isfinite(s)
            u, f, s = uni[ok], f[ok], s[ok].astype(np.float64)
            if u.size < 5 * C.TOP:
                continue
            cut = np.sort(s)[::-1][C.TOP - 1]
            above, tied = s > cut, s == cut
            w = np.where(above, 1.0, np.where(tied, (C.TOP - above.sum()) / tied.sum(), 0.0))
            pick.append(float(np.dot(w, f) / C.TOP))
            allr.append(float(f.mean()))
            vv, rr = C.vol_col(close, u, d), C.ratio_col(close, u, d)
            parts = C.comparable_parts(f, w, vv, rr)
            comp.append(float('nan') if parts is None else
                        sum(wi * (f[i] - f[nb].mean()) for i, wi, nb in parts) / C.TOP)
    pick, allr, comp = map(np.array, (pick, allr, comp))
    ex = pick - allr
    return {'months': int(pick.size), 'pick': pick.mean(), 'all': allr.mean(), 'excess': ex.mean(),
            'excess_t': C.tstat(ex), 'comparable': np.nanmean(comp), 'comparable_t': C.tstat(comp)}


def main():
    E = C.load()
    cutoff = C.dev_cutoff(E['days'])
    specs = C.graded_days(E, cutoff)
    out = {'graded_months': int(sum(len(s[4]) for s in specs)),
           'universe_sizes': {s[0]: int(s[3].size) for s in specs}}
    print('graded months', out['graded_months'], flush=True)

    raw = np.load(C.RUNS / 'lgbm/scores.npy', mmap_mode='r')
    if np.isfinite(np.asarray(raw[:, cutoff:], dtype=np.float64)).any():
        raise SystemExit('finite raw scores at or after the cutoff')
    out['raw_lgbm'] = evaluate_raw(E, specs, raw)
    print('raw lgbm', out['raw_lgbm'], flush=True)

    s2 = np.load(C.RUNS / 'jitter/lgbm_s2.npy', mmap_mode='r')
    js = json.loads((C.JUDGE_DIR / 'jitter_lgbm_s2_uniform.json').read_text())
    jy = {y['first_day'][:4]: y for y in js['years']}
    rows, growth = [], {c: [] for c in C.COSTS}
    for k, (y, h0, t, uni, graded) in enumerate(specs):
        gy = C.grader_year(E, s2, y, cand=uni)
        ref = jy[str(y)]
        assert ref['candidates'] == uni.size, (y, ref['candidates'], uni.size)
        mine = {c: C.grader_value(gy, c) - 1.0 for c in C.COSTS}
        for c in C.COSTS:
            growth[c].append(1 + mine[c])
        theirs = {int(c['cost_bps']): c['final_pct'] / 100 for c in ref['costs']}
        rows.append({'year': y, 'blocks': len(gy['blocks']), 'candidates': int(uni.size),
                     **{f'mine_{c}': mine[c] for c in C.COSTS}, **{f'json_{c}': theirs[c] for c in C.COSTS},
                     'json_spy': ref['spy_pct'] / 100})
        print(y, len(gy['blocks']), {c: round(100 * mine[c], 3) for c in C.COSTS},
              {c: round(100 * theirs[c], 3) for c in C.COSTS}, flush=True)
    out['account_years'] = rows
    out['account_end'] = {c: float(10000 * np.prod(growth[c])) for c in C.COSTS}
    out['account_max_abs_diff_pct'] = max(abs(r[f'mine_{c}'] - r[f'json_{c}']) * 100 for r in rows for c in C.COSTS)
    dates, close = C.spy_store()
    spy = C.spy_years(E, dates, close)
    out['spy_store_years'] = spy.tolist()
    out['spy_store_end'] = float(10000 * np.prod(1 + spy))
    out['spy_store_max_diff_vs_json'] = float(max(abs(spy[k] - rows[k]['json_spy']) for k in range(len(rows))))
    tr = C.spy_tr_years(E)
    out['spy_tr_years'] = tr.tolist()
    out['spy_tr_end'] = float(10000 * np.prod(1 + tr))
    print('account end', out['account_end'], 'max diff (pct points)', out['account_max_abs_diff_pct'])
    print('SPY store', out['spy_store_end'], 'SPY total return', out['spy_tr_end'])
    C.save_json(out, 'reproduce.json')


if __name__ == '__main__':
    main()
