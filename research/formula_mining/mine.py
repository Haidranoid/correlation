"""Selection on 2005-2014, the shuffled-data luck calibration, and the 2015-2025 holdout ICs (old export).

    systemd-run ... python3 -I mine.py timing        # one shuffled selection, prints only its run time
    systemd-run ... python3 -I mine.py run R         # the real selection + R shuffled selections

Rules (experiments/-2.4-formula-mining.md): each of the 676 distinct formulas (26 base ranks, 325 rank products,
325 rank differences; each usable in either direction, 1,352) gets a weekly Spearman IC with the next 5-day
market-relative return; its training t = mean IC / (sd / sqrt(weeks)) over training weeks. The 20 with the highest
|t| are kept, oriented by the sign of t; composite = mean of their oriented ranks.
Shuffled runs: the training weeks' outcome vectors are permuted across weeks (whole weeks, matched by stock), the
whole selection is redone, and the run's best |t| and its composite's real holdout IC are recorded.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/home/bhoult/programming/correlation/research/formula_mining')
from common import (BIG, FEATURES, HERE, HOLD_START, NAMES, NFORM, SEED, TOP, TRAIN_END, c2eval, composite,  # noqa: E402
                    formula_values, load_weekly, log, spearman_rows, tstat, week_slices)


def setup():
    E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=())
    days = E['days']
    W = load_weekly('old')
    sl = week_slices(W)
    dates = W['dates']
    end = (np.datetime64(TRAIN_END) - np.datetime64('1970-01-01')).astype(int)
    hs = (np.datetime64(HOLD_START) - np.datetime64('1970-01-01')).astype(int)
    tr = np.nonzero(days[dates + 5] <= end)[0]
    ho = np.nonzero(days[dates] >= hs)[0]
    return E, days, W, sl, dates, tr, ho


def ic_matrix(W, sl, weeks, ys=None):
    out = np.full((NFORM, len(weeks)), np.nan)
    for i, k in enumerate(weeks):
        B = W['B'][:, sl[k]].astype(np.float64)
        y = W['y'][sl[k]] if ys is None else ys[i]
        out[:, i] = spearman_rows(formula_values(B), y.astype(np.float64))
    return out


def t_rows(ic):
    n = np.isfinite(ic).sum(1)
    m = np.nanmean(ic, 1)
    s = np.nanstd(ic, 1, ddof=1)
    with np.errstate(all='ignore'):
        return m / (s / np.sqrt(n)), m, n


def select(t):
    sel = np.argsort(-np.abs(np.nan_to_num(t)), kind='stable')[:TOP]
    return sel, np.sign(t[sel])


def composite_ic(W, sl, weeks, sel, sign, ys=None):
    out = np.full(len(weeks), np.nan)
    for i, k in enumerate(weeks):
        B = W['B'][:, sl[k]].astype(np.float64)
        y = W['y'][sl[k]] if ys is None else ys[i]
        c = composite(B, sel, sign)
        out[i] = spearman_rows(c[None, :], y.astype(np.float64))[0]
    return out


def shuffled_ys(W, sl, tr, rng):
    """Outcome vectors for the training weeks after permuting whole weeks: week tr[i] gets week tr[perm[i]]'s outcomes,
    matched by stock (NaN for stocks not in that week)."""
    N = int(W['idx'].max()) + 1
    perm = rng.permutation(len(tr))
    out = []
    for i, k in enumerate(tr):
        src = tr[perm[i]]
        g = np.full(N, np.nan, np.float32)
        g[W['idx'][sl[src]]] = W['y'][sl[src]]
        out.append(g[W['idx'][sl[k]]])
    return out, perm


def one_shuffle(W, sl, tr, ho, r):
    rng = np.random.default_rng(SEED + r)
    ys, perm = shuffled_ys(W, sl, tr, rng)
    ic = ic_matrix(W, sl, tr, ys)
    t, m, n = t_rows(ic)
    sel, sign = select(t)
    hic = composite_ic(W, sl, ho, sel, sign)
    ht, hm, hn = tstat(hic)
    b = int(np.nanargmax(np.abs(t)))
    return {'run': r, 'fixed_points': int((perm == np.arange(len(perm))).sum()), 'best_abs_t': float(abs(t[b])),
            'best_formula': NAMES[b], 'p95_abs_t_of_676': float(np.nanpercentile(np.abs(t), 95)),
            'selected': [NAMES[s] for s in sel], 'sign': sign.tolist(), 'sel_idx': sel.tolist(),
            'holdout_ic_mean': hm, 'holdout_ic_t': ht, 'holdout_weeks': hn}


def main():
    mode = sys.argv[1]
    E, days, W, sl, dates, tr, ho = setup()
    log('training weeks', len(tr), c2eval.iso(days[dates[tr[0]]]), '..', c2eval.iso(days[dates[tr[-1]]]),
        '| holdout weeks', len(ho), c2eval.iso(days[dates[ho[0]]]), '..', c2eval.iso(days[dates[ho[-1]]]))
    if mode == 'timing':
        t0 = time.time()
        one_shuffle(W, sl, tr, ho, 999)        # a throwaway seed; nothing about its result is printed
        log(f'one shuffled selection (training ICs of 676 formulas + holdout composite): {time.time() - t0:.0f} s')
        return
    R = int(sys.argv[2])
    t0 = time.time()
    ic = ic_matrix(W, sl, np.arange(len(dates)))
    np.save(BIG / 'ic_real_old.npy', ic.astype(np.float32))
    log(f'real ICs, all weeks: {time.time() - t0:.0f} s')
    t_tr, m_tr, n_tr = t_rows(ic[:, tr])
    t_ho, m_ho, n_ho = t_rows(ic[:, ho])
    sel, sign = select(t_tr)
    comp_tr = composite_ic(W, sl, tr, sel, sign)
    comp_ho = composite_ic(W, sl, ho, sel, sign)
    r5 = FEATURES.index('ret5')
    rev_ho = -ic[r5, ho]
    # how much of the composite is 5-day reversal: weekly Spearman of the composite with -ret5, holdout weeks
    overlap = np.full(len(ho), np.nan)
    for i, k in enumerate(ho):
        B = W['B'][:, sl[k]].astype(np.float64)
        c = composite(B, sel, sign)
        overlap[i] = spearman_rows(c[None, :], -B[r5])[0]
    real = {'selected': [{'formula': NAMES[s], 'sign': int(g), 'train_t': float(t_tr[s]), 'train_ic': float(m_tr[s]),
                          'train_weeks': int(n_tr[s]), 'holdout_t_oriented': float(g * t_ho[s]),
                          'holdout_ic_oriented': float(g * m_ho[s]), 'holdout_weeks': int(n_ho[s])}
                         for s, g in zip(sel, sign)],
            'sel_idx': sel.tolist(), 'sign': sign.tolist(),
            'best_abs_t': float(np.nanmax(np.abs(t_tr))), 'best_formula': NAMES[int(np.nanargmax(np.abs(t_tr)))],
            'p95_abs_t_of_676': float(np.nanpercentile(np.abs(t_tr), 95)),
            'composite_train': dict(zip(('t', 'mean', 'weeks'), tstat(comp_tr))),
            'composite_holdout': dict(zip(('t', 'mean', 'weeks'), tstat(comp_ho))),
            'reversal5_holdout': dict(zip(('t', 'mean', 'weeks'), tstat(rev_ho))),
            'composite_minus_reversal_holdout': dict(zip(('t', 'mean', 'weeks'), tstat(comp_ho - rev_ho))),
            'composite_vs_reversal_rank_corr_mean': float(np.nanmean(overlap)),
            'weeks': {'train': [c2eval.iso(days[dates[tr[0]]]), c2eval.iso(days[dates[tr[-1]]]), int(len(tr))],
                      'holdout': [c2eval.iso(days[dates[ho[0]]]), c2eval.iso(days[dates[ho[-1]]]), int(len(ho))]}}
    # holdout composite IC by calendar year (context)
    yrs = np.array([int(c2eval.iso(days[dates[k]])[:4]) for k in ho])
    real['composite_holdout_by_year'] = {int(y): [float(np.nanmean(comp_ho[yrs == y])), float(np.nanmean(rev_ho[yrs == y]))]
                                          for y in np.unique(yrs)}
    np.savez(BIG / 'weekly_series_old.npz', comp_tr=comp_tr, comp_ho=comp_ho, rev_ho=rev_ho, tr=tr, ho=ho,
             t_tr=t_tr, t_ho=t_ho)
    json.dump(real, open(HERE / 'selection_old.json', 'w'), indent=1)
    log('real selection saved')
    runs = []
    for r in range(R):
        t0 = time.time()
        runs.append(one_shuffle(W, sl, tr, ho, r))
        json.dump(runs, open(HERE / 'shuffled_runs.json', 'w'), indent=1)
        log(f'shuffled run {r} done in {time.time() - t0:.0f} s')
    log('done')


if __name__ == '__main__':
    main()
